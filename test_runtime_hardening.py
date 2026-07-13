import os
import random
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import hierarchos_cli
from hierarchos import HierarchosCore
from hierarchos.evaluation.lm_eval_wrapper import HierarchosLM
from hierarchos.inference.chat import (
    _checkpoint_has_trained_hebbian_writer,
    advance_chat_model_state,
    boundary_drift_seed,
    consolidate_ltm_state_for_save,
    ltm_replay_seed_state,
    prepare_online_ltm_gradients,
    reset_active_ltm_state,
    sample_next_token,
    tbptt_chunk_ranges,
    zero_ltm_momentum_state,
)
from hierarchos.models.ltm import LTMModule
from hierarchos.training.trainer import _prepare_ltm_update_gradients, mark_val_proj_trained
from hierarchos.utils import checkpoint as checkpoint_utils
from hierarchos.utils import rosa as rosa_utils
from hierarchos.utils.checkpoint import (
    load_full_model_with_config,
    sanitize_model_state_dict,
    save_checkpoint_safely,
)
from test_rwkv_v8_integrity import _tiny_config


def test_legacy_hebbian_writer_is_blocked_until_trained():
    config = _tiny_config()
    config.use_rosa = False
    config.use_deepembed = False
    model = HierarchosCore(config).eval()
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    before = model.ltm.fast_vals.detach().clone()

    with torch.no_grad():
        model(ids, allow_hebbian_update=True)
    assert torch.equal(model.ltm.fast_vals, before)
    assert not _checkpoint_has_trained_hebbian_writer(config)

    config.val_proj_trained = True
    with torch.no_grad():
        model(ids, allow_hebbian_update=True)
    assert not torch.equal(model.ltm.fast_vals, before)
    assert _checkpoint_has_trained_hebbian_writer(config)


def test_ltm_value_alignment_trains_existing_writer_without_layout_or_logit_drift():
    torch.manual_seed(123)
    config = _tiny_config()
    config.use_rosa = False
    config.use_deepembed = False
    model = HierarchosCore(config)
    before_layout = {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    }
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    labels = ids.clone()

    model.eval()
    with torch.no_grad():
        baseline = model(ids, labels=labels)["logits"]
        aligned = model(
            ids,
            labels=labels,
            compute_ltm_value_alignment=True,
        )
    assert torch.equal(baseline, aligned["logits"])
    assert torch.isfinite(aligned["ltm_value_alignment_cost"])

    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(
        ids,
        labels=labels,
        compute_ltm_value_alignment=True,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
    )
    outputs["ltm_value_alignment_cost"].backward()

    assert model.val_proj.weight.grad is not None
    assert torch.isfinite(model.val_proj.weight.grad).all()
    assert model.val_proj.weight.grad.norm().item() > 0.0
    other_grads = [
        name
        for name, parameter in model.named_parameters()
        if name != "val_proj.weight" and parameter.grad is not None
    ]
    assert other_grads == []
    assert before_layout == {
        name: tuple(tensor.shape) for name, tensor in model.state_dict().items()
    }


def test_hebbian_writer_readiness_requires_sustained_alignment_updates():
    config = {"ltm_value_alignment_min_updates": 3}
    model = SimpleNamespace(config=config)

    mark_val_proj_trained(model)
    mark_val_proj_trained(model)
    assert config["val_proj_alignment_updates"] == 2
    assert not config.get("val_proj_trained", False)

    mark_val_proj_trained(model)
    assert config["val_proj_alignment_updates"] == 3
    assert config["val_proj_trained"] is True


def test_rosa_first_build_matches_reference_and_preserves_state():
    rng = random.Random(1234)
    for length in (0, 1, 2, 17, 128, 512):
        tokens = [rng.randrange(23) for _ in range(length)]
        predictions, state = rosa_utils.rosa_single(tokens)

        assert predictions == rosa_utils.ROSA(tokens)
        assert state.tokens == tokens

        suffix = [rng.randrange(23) for _ in range(19)]
        suffix_predictions, state = rosa_utils.rosa_single(suffix, state)
        combined = rosa_utils.ROSA(tokens + suffix)
        assert suffix_predictions == combined[-len(suffix):]
        assert state.tokens == tokens + suffix


def test_rosa_automatic_worker_count_is_bounded(monkeypatch):
    class FakePool:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def shutdown(self, wait=False):
            return None

    monkeypatch.setattr(rosa_utils.os, "cpu_count", lambda: 256)
    monkeypatch.setattr(rosa_utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(rosa_utils, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(rosa_utils, "_ROSA_POOL", None)
    monkeypatch.setattr(rosa_utils, "_ROSA_POOL_PID", None)

    pool = rosa_utils._get_pool()
    assert pool.max_workers == rosa_utils._ROSA_AUTO_WORKER_LIMIT


def _save_model(tmpdir, model, config, state=None):
    path = os.path.join(tmpdir, "hierarchos.pt")
    torch.save(
        {
            "model_state_dict": state if state is not None else model.state_dict(),
            "config": dict(config),
            "training_complete": True,
        },
        path,
    )
    return path


def test_direct_checkpoint_tokenizer_source_uses_parent_and_vocab_is_validated(tmp_path):
    checkpoint_path = tmp_path / "epoch13.pt"
    checkpoint_path.touch()

    assert hierarchos_cli.resolve_tokenizer_source(None, str(checkpoint_path)) == str(tmp_path)
    assert hierarchos_cli.resolve_tokenizer_source("exact-tokenizer", str(checkpoint_path)) == "exact-tokenizer"

    tokenizer = type("Tokenizer", (), {"__len__": lambda self: 11})()
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        hierarchos_cli.validate_tokenizer_vocab(tokenizer, {"vocab_size": 12}, "epoch13")


def test_single_nested_download_directory_resolves_weights_and_tokenizer(tmp_path):
    outer = tmp_path / "chatHRM"
    inner = outer / "chatHRM"
    inner.mkdir(parents=True)
    (inner / "hierarchos.pt").touch()
    (inner / "tokenizer.json").write_text("{}", encoding="utf-8")
    (inner / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    weights_path, model_dir = checkpoint_utils._resolve_weights_path(str(outer))
    assert weights_path == str(inner / "hierarchos.pt")
    assert model_dir == str(inner)
    assert hierarchos_cli.resolve_tokenizer_source(None, str(outer)) == str(inner)


def test_export_tokenizer_uses_training_provenance_before_legacy_name():
    config = {
        "tokenizer_path": "organization/exact-training-tokenizer",
        "tokenizer_name": "legacy-fallback-tokenizer",
    }

    assert hierarchos_cli.resolve_export_tokenizer_source(None, config) == config["tokenizer_path"]
    assert hierarchos_cli.resolve_export_tokenizer_source("explicit-tokenizer", config) == "explicit-tokenizer"
    assert hierarchos_cli.resolve_export_tokenizer_source(None, {}) == "openai-community/gpt2"


def test_loader_rejects_missing_non_rwkv_weight():
    config = _tiny_config()
    model = HierarchosCore(config)
    state = dict(model.state_dict())
    state.pop("in_proj.weight")

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model(tmpdir, model, config, state)
        with pytest.raises(ValueError, match="randomly initialized"):
            load_full_model_with_config(tmpdir, "cpu")


def test_loader_allows_tied_alias_and_transient_omissions():
    config = _tiny_config()
    model = HierarchosCore(config)
    expected = model.tok_emb.weight.detach().clone()
    state = dict(model.state_dict())
    state.pop("lm_head.weight")
    state.pop("time_freqs")
    for key in ("ltm.fast_vals", "ltm._mom_vals", "ltm.timestamps", "ltm.sources"):
        state.pop(key)

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model(tmpdir, model, config, state)
        loaded, _ = load_full_model_with_config(tmpdir, "cpu")

    assert torch.equal(loaded.tok_emb.weight, expected)
    assert loaded.tok_emb.weight.data_ptr() == loaded.lm_head.weight.data_ptr()
    assert torch.count_nonzero(loaded.ltm.fast_vals).item() == 0


def test_loader_rejects_conflicting_tied_alias_and_nonfinite_weight():
    config = _tiny_config()
    model = HierarchosCore(config)
    tied_state = dict(model.state_dict())
    tied_state["lm_head.weight"] = tied_state["lm_head.weight"].clone()
    tied_state["lm_head.weight"][0, 0] += 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model(tmpdir, model, config, tied_state)
        with pytest.raises(ValueError, match="Tied embedding mismatch"):
            load_full_model_with_config(tmpdir, "cpu")

    nonfinite_state = dict(model.state_dict())
    nonfinite_state["persistent"] = nonfinite_state["persistent"].clone()
    nonfinite_state["persistent"][0] = float("nan")
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model(tmpdir, model, config, nonfinite_state)
        with pytest.raises(ValueError, match="Non-finite tensor 'persistent'"):
            load_full_model_with_config(tmpdir, "cpu")


def test_legacy_qproj_adaptation_is_zero_filled_and_deterministic():
    config = _tiny_config()
    model = HierarchosCore(config)
    state = dict(model.state_dict())
    old_qproj = state["qproj.weight"][:, :config.context_dim].clone()
    state["qproj.weight"] = old_qproj

    with tempfile.TemporaryDirectory() as tmpdir:
        _save_model(tmpdir, model, config, state)
        loaded, _ = load_full_model_with_config(tmpdir, "cpu")

    assert torch.equal(loaded.qproj.weight[:, :config.context_dim], old_qproj)
    assert torch.count_nonzero(loaded.qproj.weight[:, config.context_dim:]).item() == 0


def test_compile_prefix_collision_is_rejected():
    with pytest.raises(ValueError, match="collapse to the same name"):
        sanitize_model_state_dict(
            {
                "weight": torch.tensor([1.0]),
                "_orig_mod.weight": torch.tensor([2.0]),
            }
        )


def test_safe_checkpoint_save_restores_previous_file_on_install_failure(monkeypatch, tmp_path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"known-good-checkpoint")
    real_replace = os.replace
    replace_calls = 0

    def fail_new_checkpoint_install(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("simulated install failure")
        return real_replace(source, destination)

    monkeypatch.setattr(checkpoint_utils.os, "replace", fail_new_checkpoint_install)
    with pytest.raises(OSError, match="simulated install failure"):
        save_checkpoint_safely({"value": torch.tensor([1.0])}, str(path))

    assert path.read_bytes() == b"known-good-checkpoint"


class _LTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = LTMModule(n_slots=4, key_dim=2, val_dim=3)


def _active_ltm_state(model):
    return (
        torch.ones_like(model.ltm.fast_vals),
        torch.full_like(model.ltm._mom_vals, 2.0),
        torch.tensor([[1, 2, 3]]),
        ["rosa-state"],
        torch.full_like(model.ltm.timestamps, 7.0),
        torch.full_like(model.ltm.sources, LTMModule.SRC_CORRECTION),
    )


def test_reset_ltm_clears_active_state_but_preserves_rosa_context():
    model = _LTMModel()
    state = reset_active_ltm_state(model, _active_ltm_state(model), preserve_rosa=True)

    assert torch.count_nonzero(state[0]).item() == 0
    assert torch.count_nonzero(state[1]).item() == 0
    assert torch.equal(state[2], torch.tensor([[1, 2, 3]]))
    assert state[3] == ["rosa-state"]
    assert torch.count_nonzero(state[4]).item() == 0
    assert torch.count_nonzero(state[5]).item() == 0
    assert torch.count_nonzero(model.ltm.fast_vals).item() == 0


def test_ltm_replay_and_momentum_helpers_do_not_duplicate_context():
    model = _LTMModel()
    state = _active_ltm_state(model)
    replay = ltm_replay_seed_state(state)
    assert replay[0] is state[0]
    assert replay[2] is None
    assert replay[3] is None
    assert replay[4] is state[4]

    model.ltm._mom_vals.fill_(4.0)
    reset = zero_ltm_momentum_state(model, state)
    assert torch.count_nonzero(reset[1]).item() == 0
    assert torch.count_nonzero(model.ltm._mom_vals).item() == 0


def test_online_ltm_gradients_reject_nonfinite_and_clip_finite_norm():
    assert prepare_online_ltm_gradients(torch.tensor([1.0, float("nan")]), 0.75) is None
    clipped = prepare_online_ltm_gradients(torch.full((100,), 10.0), 0.75)
    assert clipped is not None
    assert torch.isfinite(clipped).all()
    assert clipped.norm().item() <= 0.75001
    assert clipped.abs().max().item() <= 0.75

    assert _prepare_ltm_update_gradients(torch.tensor([float("inf")]), 0.75) is None
    trainer_clipped = _prepare_ltm_update_gradients(torch.full((100,), 10.0), 0.75)
    assert trainer_clipped.norm().item() <= 0.75001


def test_ltm_inner_update_defensively_rejects_nonfinite_gradient():
    ltm = LTMModule(n_slots=4, key_dim=2, val_dim=3)
    with pytest.raises(ValueError, match="rejected non-finite gradients"):
        ltm.inner_update(
            torch.tensor([[[0]]]),
            torch.tensor([[[[float("nan"), 0.0, 0.0]]]]),
            current_lr=0.1,
            timestamp=0.0,
        )


def test_ltm_consolidation_preserves_effective_memory_and_clears_transients():
    model = _LTMModel()
    state = _active_ltm_state(model)
    expected = model.ltm.vals.detach().clone() + state[0]

    assert consolidate_ltm_state_for_save(model, state) is True
    assert torch.allclose(model.ltm.vals, expected)
    assert torch.count_nonzero(model.ltm.fast_vals).item() == 0
    assert torch.count_nonzero(model.ltm._mom_vals).item() == 0


def test_sampling_helper_honors_filters_without_mutating_logits():
    logits = torch.tensor([[5.0, 4.0, 1.0]])
    original = logits.clone()
    for _ in range(5):
        assert sample_next_token(logits, temperature=1.0, top_k=1, top_p=1.0).item() == 0
    assert torch.equal(logits, original)
    assert sample_next_token(logits, temperature=0.0, top_k=0, top_p=1.0).item() == 0
    with pytest.raises(RuntimeError, match="non-finite logits"):
        sample_next_token(torch.tensor([[float("nan"), 1.0]]))


def test_chat_state_step_uses_absolute_tbptt_boundaries_and_updates_once():
    assert tbptt_chunk_ranges(10, 4, global_offset=2) == [(0, 2), (2, 6), (6, 10)]
    drift = torch.tensor([[3.0]])
    assert boundary_drift_seed(drift, 4, 4) is drift
    assert boundary_drift_seed(drift, 5, 4) is None
    assert boundary_drift_seed(drift, 0, 4) is None

    class RecordingModel:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return {
                "logits": torch.zeros(1, kwargs["input_ids"].shape[1], 7),
                "h_state": kwargs["h_state"] + 1,
                "l_state": kwargs["l_state"] + 1,
                "prev_context": kwargs["prev_context"] + 1,
                "target_context": kwargs["target_context"] + 1,
                "drift_state": kwargs["drift_state"] + 1,
                "ltm_memory_state": ("updated",),
            }

    model = RecordingModel()
    one = torch.ones(1, 1)
    outputs, state = advance_chat_model_state(
        model,
        torch.tensor([[2]], dtype=torch.long),
        device=torch.device("cpu"),
        h_state=one,
        l_state=one,
        prev_context=one,
        target_context=one,
        drift_state=torch.zeros_like(one),
        drift_seed=drift,
        ltm_state=("old",),
        global_pos_offset=4,
        min_timestamp=2.0,
        source_filter=3,
    )

    assert outputs["logits"].shape == (1, 1, 7)
    assert len(model.calls) == 1
    assert model.calls[0]["global_pos_offset"] == 4
    assert model.calls[0]["drift_state"] is drift
    assert torch.equal(state[0], one + 1)
    assert state[-1] == ("updated",)


def test_ltm_mixed_filtered_batch_leaves_no_match_row_unchanged():
    torch.manual_seed(4)
    ltm = LTMModule(n_slots=5, key_dim=2, val_dim=3, momentum=0.7, forget_rate=0.2)
    topk_idx = torch.tensor([[[0, 1]], [[-1, -1]]])
    grads = torch.randn(2, 1, 2, 3)
    fast = torch.randn(2, 5, 3)
    mom = torch.randn(2, 5, 3)

    new_fast, new_mom = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.1,
        timestamp=1.0,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=torch.zeros(2, 5),
        sources=torch.zeros(2, 5, dtype=torch.long),
        inplace=True,
    )

    assert torch.equal(new_fast[1], fast[1])
    assert torch.equal(new_mom[1], mom[1])


@pytest.mark.parametrize("cpu_gather", [False, True])
def test_ltm_filtered_retrieval_keeps_valid_rows_and_score_gradients(cpu_gather):
    ltm = LTMModule(
        n_slots=4,
        key_dim=2,
        val_dim=2,
        cpu_gather_retrieval=cpu_gather,
    )
    with torch.no_grad():
        ltm.keys.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))
        ltm.vals.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0], [-2.0, 0.0], [0.0, -3.0]]))

    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
    timestamps = torch.tensor([[5.0, 5.0, 5.0, 5.0], [0.0, 0.0, 0.0, 0.0]])
    sources = torch.full((2, 4), LTMModule.SRC_TRAINING_DATA, dtype=torch.long)
    values, indices, _ = ltm.retrieve_topk(
        queries,
        topk=2,
        min_timestamp=1.0,
        source_filter=LTMModule.SRC_TRAINING_DATA,
        timestamps=timestamps,
        sources=sources,
    )

    assert torch.all(indices[0] >= 0)
    assert torch.any(values[0] != 0)
    assert torch.all(indices[1] == -1)
    assert torch.all(values[1] == 0)
    values[0].sum().backward()
    assert queries.grad is not None
    assert torch.any(queries.grad[0] != 0)


def test_ltm_inner_update_rejects_invalid_learning_rate():
    ltm = LTMModule(n_slots=3, key_dim=2, val_dim=2)
    indices = torch.zeros(1, 1, 1, dtype=torch.long)
    grads = torch.ones(1, 1, 1, 2)

    for learning_rate in (float("nan"), float("inf"), -1e-3):
        with pytest.raises(ValueError, match="finite nonnegative"):
            ltm.inner_update(indices, grads, learning_rate, timestamp=0.0)


def test_timestamp_encoding_supports_one_and_two_value_dimensions():
    for val_dim in (1, 2):
        config = _tiny_config()
        config.use_rosa = False
        config.ltm_val_dim = val_dim
        model = HierarchosCore(config).eval()
        with torch.no_grad():
            outputs = model(torch.tensor([[1, 2]], dtype=torch.long), suppress_hebbian=True)
        assert outputs["logits"].shape == (1, 2, config.vocab_size)
        assert torch.isfinite(outputs["logits"]).all()


def test_architecture_geometry_fails_fast_and_zero_detach_is_supported():
    invalid_width = _tiny_config()
    invalid_width.h_hidden = invalid_width.context_dim + 1
    with pytest.raises(ValueError, match="h_hidden == context_dim"):
        HierarchosCore(invalid_width)

    invalid_stride = _tiny_config()
    invalid_stride.h_stride = 0
    with pytest.raises(ValueError, match="h_stride.*positive integer"):
        HierarchosCore(invalid_stride)

    invalid_head = _tiny_config()
    invalid_head.rwkv_head_size = 5
    with pytest.raises(ValueError, match="does not divide"):
        HierarchosCore(invalid_head)

    no_detach = _tiny_config()
    no_detach.detach_every_n_steps = 0
    model = HierarchosCore(no_detach)
    assert model.config.detach_every_n_steps is None
    assert model.h_rnn.detach_every_n_steps is None
    model.train()
    outputs = model(torch.tensor([[1, 2]], dtype=torch.long), suppress_hebbian=True)
    assert torch.isfinite(outputs["logits"]).all()


class _BoundaryTokenizer:
    eos_token_id = 0
    eos_token = "<eos>"
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        mapping = {
            "": [],
            "A": [1],
            " B": [2],
            "A B": [1, 2],
            "B": [3],
            "C": [4],
        }
        return list(mapping.get(text, [5]))

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(str(token) for token in tokens)


class _CountingEvalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(max_length=8, training_chunk_size=0)
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, 8, device=input_ids.device)
        logits[..., 2] = 2.0
        return {
            "logits": logits,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
            "ltm_memory_state": None,
        }


class _Request:
    def __init__(self, *args):
        self.args = args


def test_lm_eval_uses_joint_bpe_empty_context_and_real_batches():
    model = _CountingEvalModel()
    wrapper = HierarchosLM(model, _BoundaryTokenizer(), torch.device("cpu"), batch_size=2)
    assert wrapper._encode_pair("A", " B") == ([1], [2])
    assert wrapper._encode_pair("", "B") == ([0], [3])

    results = wrapper.loglikelihood([
        _Request("A", " B"),
        _Request("", "B"),
    ])
    assert len(results) == 2
    assert model.calls == 1
    assert all(torch.isfinite(torch.tensor(score)) for score, _ in results)


def test_lm_eval_rolling_scores_first_token_and_returns_float():
    model = _CountingEvalModel()
    wrapper = HierarchosLM(model, _BoundaryTokenizer(), torch.device("cpu"), batch_size=1)
    result = wrapper.loglikelihood_rolling([_Request("A B")])
    assert len(result) == 1
    assert isinstance(result[0], float)
    assert result[0] < 0.0
