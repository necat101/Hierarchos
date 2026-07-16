import importlib
from types import SimpleNamespace

import torch
import torch.nn as nn

from hierarchos import HierarchosCore
from hierarchos.evaluation.arc_agi import generate_text as generate_arc_text
from hierarchos.evaluation.lm_eval_wrapper import HierarchosLM
from hierarchos.inference.chat import (
    advance_chat_model_state,
    boundary_drift_seed,
    resolve_inference_prefill_chunk_size,
)
from hierarchos.models.core import WorkerLoop
from hierarchos.models.quantized import _uses_exact_refinement_policy
from hierarchos.utils.checkpoint import (
    load_full_model_with_config,
    sanitize_model_state_dict,
)
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks
from test_rwkv_v8_integrity import _tiny_config


def _full_sample_config():
    config = _tiny_config()
    config.full_sample_bptt = True
    config.training_chunk_size = 3
    config.compile_static_worker_loop = True
    config.memory_gate_warmup_steps = 0
    config.detach_every_n_steps = None
    config.max_h_steps = 5
    config.max_l_steps = 4
    config.h_halt_thresh = 0.4
    return config


def test_full_sample_checkpoint_defaults_to_one_chat_prefill_graph():
    exact = SimpleNamespace(full_sample_bptt=True, training_chunk_size=256)
    parity = SimpleNamespace(inference_logit_parity=True, training_chunk_size=256)
    legacy = SimpleNamespace(full_sample_bptt=False, training_chunk_size=256)

    assert resolve_inference_prefill_chunk_size(exact) == 0
    assert resolve_inference_prefill_chunk_size(parity) == 0
    assert resolve_inference_prefill_chunk_size(legacy) == 256
    assert resolve_inference_prefill_chunk_size(exact, requested=128) == 128
    drift = torch.ones(1, 4)
    assert boundary_drift_seed(drift, 128, 128, exact_full_sample=True) is None
    assert boundary_drift_seed(drift, 128, 128) is drift


def test_explicit_chat_prefill_segments_preserve_full_sample_drift_recurrence():
    torch.manual_seed(17)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        monolithic = model(input_ids, suppress_hebbian=True)["logits"]

    model.reset_memory()
    state = (None, None, None, None, None, None)
    segmented_logits = []
    segment_size = 3
    with torch.no_grad():
        for start in range(0, input_ids.shape[1], segment_size):
            end = min(start + segment_size, input_ids.shape[1])
            h_state, l_state, prev_context, target_context, drift_state, ltm_state = state
            outputs, state = advance_chat_model_state(
                model,
                input_ids[:, start:end],
                device=torch.device("cpu"),
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                drift_seed=boundary_drift_seed(
                    drift_state,
                    start,
                    segment_size,
                    exact_full_sample=True,
                ),
                ltm_state=ltm_state,
                global_pos_offset=start,
            )
            segmented_logits.append(outputs["logits"])

    torch.testing.assert_close(
        torch.cat(segmented_logits, dim=1),
        monolithic,
        rtol=1e-6,
        atol=5e-7,
    )


def test_lm_eval_explicit_segments_preserve_exact_drift_recurrence():
    torch.manual_seed(19)
    model = HierarchosCore(_full_sample_config()).eval()
    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        expected = model(input_ids, suppress_hebbian=True)["logits"]

    # Bypass the optional lm-eval package constructor and force explicit
    # segmentation so this directly exercises the wrapper's boundary policy.
    wrapper = object.__new__(HierarchosLM)
    wrapper.model = model
    wrapper.device = torch.device("cpu")
    wrapper._prefill_chunk_size = 3
    model.reset_memory()
    actual = wrapper._model_call(input_ids)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=5e-7)


class _SurfaceTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, _text, add_special_tokens=False, return_tensors=None):
        ids = [1, 2, 1, 2, 3, 1, 2, 4]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(str(int(token)) for token in tokens)


class _RecordingExactInferenceModel:
    def __init__(self):
        self.config = SimpleNamespace(
            full_sample_bptt=True,
            inference_logit_parity=True,
            training_chunk_size=3,
            context_dim=4,
        )
        self.calls = []
        self.suppress_hebbian = True

    def eval(self):
        return self

    def __call__(self, input_ids, **kwargs):
        self.calls.append((input_ids.detach().clone(), kwargs.get("drift_state")))
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        next_id = 1 if len(self.calls) == 1 else 0
        logits[..., next_id] = 10.0
        return {
            "logits": logits,
            "h_state": kwargs.get("h_state"),
            "l_state": kwargs.get("l_state"),
            "prev_context": kwargs.get("prev_context"),
            "target_context": kwargs.get("target_context"),
            "drift_state": torch.ones(batch, 4),
            "ltm_memory_state": kwargs.get("ltm_memory_state"),
        }


def test_arc_generation_ignores_cache_chunk_geometry_and_never_reseeds_exact_drift():
    model = _RecordingExactInferenceModel()
    generate_arc_text(
        model,
        _SurfaceTokenizer(),
        torch.device("cpu"),
        "grid prompt",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert len(model.calls) == 2
    assert model.calls[0][0].shape[1] == 8
    assert all(drift_seed is None for _ids, drift_seed in model.calls)


def test_gui_bridge_uses_exact_prefill_geometry_and_no_external_drift_seed(monkeypatch):
    bridge = importlib.import_module("hierarchos_bridge_server")
    model = _RecordingExactInferenceModel()
    emitted = []

    class _ImmediateThread:
        def __init__(self, *, target, daemon=None):
            self.target = target

        def start(self):
            self.target()

    monkeypatch.setattr(bridge, "_model", model)
    monkeypatch.setattr(bridge, "_tokenizer", _SurfaceTokenizer())
    monkeypatch.setattr(bridge, "_device", torch.device("cpu"))
    monkeypatch.setattr(bridge, "_config", dict(vars(model.config)))
    monkeypatch.setattr(bridge, "_h_state", None)
    monkeypatch.setattr(bridge, "_l_state", None)
    monkeypatch.setattr(bridge, "_prev_context", None)
    monkeypatch.setattr(bridge, "_target_context", None)
    monkeypatch.setattr(bridge, "_drift_state", torch.full((1, 4), 7.0))
    monkeypatch.setattr(bridge, "_ltm_state", None)
    monkeypatch.setattr(bridge, "_total_tokens_generated", 3)
    monkeypatch.setattr(bridge, "emit", lambda event, data=None: emitted.append((event, data)))
    monkeypatch.setattr(bridge.threading, "Thread", _ImmediateThread)

    bridge.handle_generate(
        {
            "message": "hello",
            "sampling": {"max_new_tokens": 0, "temperature": 0.0},
        }
    )

    assert not [event for event, _data in emitted if event == "error"]
    assert len(model.calls) == 1
    assert model.calls[0][0].shape[1] == 8
    assert model.calls[0][1] is None


def test_quantized_exact_checkpoint_uses_training_refinement_policy_flag():
    assert _uses_exact_refinement_policy(
        SimpleNamespace(full_sample_bptt=True, inference_logit_parity=False)
    )
    assert _uses_exact_refinement_policy(
        {"full_sample_bptt": False, "inference_logit_parity": True}
    )
    assert not _uses_exact_refinement_policy(
        SimpleNamespace(full_sample_bptt=False, inference_logit_parity=False)
    )


class _StateStableRNN(nn.Module):
    def forward(self, x, state, timestep=None, deepemb_vec=None):
        return x, state


def test_full_sample_worker_eval_uses_the_training_refinement_policy():
    config = SimpleNamespace(
        max_l_steps=3,
        l_conv_atol=0.01,
        commitment_threshold=0.1,
        recurrent_state_clamp=50.0,
        context_state_clamp=50.0,
        drift_state_clamp=5.0,
        drift_norm_clamp=0.0,
        drift_delta_scale=1.0,
        activation_clamp=100.0,
        compile_static_worker_loop=True,
        full_sample_bptt=True,
    )
    rnn = _StateStableRNN()
    input_projection = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        input_projection.weight.copy_(
            torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        )
    loop = WorkerLoop(
        config,
        rnn,
        input_projection,
        nn.Identity(),
        nn.Identity(),
    )
    inputs = (
        torch.tensor([[0.5, 0.25]]),
        torch.tensor([[0.1, 0.2]]),
        torch.zeros(1, 2, 3),
        torch.zeros(1, 2),
    )

    rnn.train()
    training_outputs = loop(*inputs)
    rnn.eval()
    inference_outputs = loop(*inputs)

    for inference_value, training_value in zip(inference_outputs, training_outputs):
        torch.testing.assert_close(inference_value, training_value, rtol=0, atol=0)


def test_cached_training_and_live_chat_rosa_have_exact_full_sample_logits():
    torch.manual_seed(3)
    config = _full_sample_config()
    model = HierarchosCore(config)
    with torch.no_grad():
        # This would activate the legacy inference-only manager early exit.
        model.h_halt_proj.weight.zero_()
        model.h_halt_proj.bias.zero_()

    input_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)
    cached_rosa_ids = torch.tensor(
        [
            precompute_rosa_ids_for_chunks(
                input_ids[0].tolist(),
                vocab_size=config.vocab_size,
                chunk_size=config.training_chunk_size,
                rosa_max_ctx=config.rosa_max_context,
            )
        ],
        dtype=torch.long,
    )

    model.train()
    model.reset_memory()
    training_logits = model(
        input_ids,
        labels=input_ids,
        rosa_ids=cached_rosa_ids,
        suppress_hebbian=True,
    )["logits"].detach()

    model.eval()
    model.reset_memory()
    with torch.no_grad():
        chat_logits = model(input_ids, suppress_hebbian=True)["logits"]

    torch.testing.assert_close(chat_logits, training_logits, rtol=0, atol=0)


def test_autoregressive_chat_state_matches_the_full_sequence_next_logit():
    torch.manual_seed(4)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    prompt_ids = torch.tensor([[1, 2, 1, 2, 3, 1, 2, 4]], dtype=torch.long)
    next_id = torch.tensor([[5]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        prefill = model(prompt_ids, suppress_hebbian=True)
        incremental, _ = advance_chat_model_state(
            model,
            next_id,
            device=torch.device("cpu"),
            h_state=prefill["h_state"],
            l_state=prefill["l_state"],
            prev_context=prefill["prev_context"],
            target_context=prefill["target_context"],
            drift_state=prefill["drift_state"],
            drift_seed=None,
            ltm_state=prefill["ltm_memory_state"],
            global_pos_offset=prompt_ids.shape[1],
        )

        model.reset_memory()
        monolithic = model(
            torch.cat([prompt_ids, next_id], dim=1),
            suppress_hebbian=True,
        )

    # Different GEMM shapes may round the last bit differently; the recurrence,
    # ROSA history, positions, and resulting next-token distribution must agree.
    torch.testing.assert_close(
        incremental["logits"][:, -1],
        monolithic["logits"][:, -1],
        rtol=1e-6,
        atol=5e-7,
    )


def test_inference_export_resets_transient_ltm_without_logit_drift(tmp_path):
    torch.manual_seed(9)
    config = _full_sample_config()
    model = HierarchosCore(config).eval()
    input_ids = torch.tensor([[7, 3, 7, 3, 9]], dtype=torch.long)

    model.reset_memory()
    with torch.no_grad():
        expected_logits = model(input_ids, suppress_hebbian=True)["logits"].clone()
        model.ltm.fast_vals.fill_(4.0)
        model.ltm._mom_vals.fill_(3.0)
        model.ltm.timestamps.fill_(2.0)
        model.ltm.sources.fill_(1)

    torch.save(
        {
            "model_state_dict": sanitize_model_state_dict(model),
            "config": dict(config),
            "training_complete": True,
        },
        tmp_path / "hierarchos.pt",
    )
    loaded, loaded_config = load_full_model_with_config(str(tmp_path), "cpu")

    assert loaded_config.full_sample_bptt is True
    assert torch.count_nonzero(loaded.ltm.fast_vals).item() == 0
    assert torch.count_nonzero(loaded.ltm._mom_vals).item() == 0
    with torch.no_grad():
        actual_logits = loaded(input_ids, suppress_hebbian=True)["logits"]
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
