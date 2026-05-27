from types import SimpleNamespace
import tempfile

import torch
import torch.nn as nn

from hierarchos.inference.chat import (
    _chat_ltm_state_from_rosa_context,
    load_hierarchical_chat_state,
    save_hierarchical_chat_state,
)
from hierarchos.inference.chat_state import clear_ltm_working_memory
from hierarchos.utils.rosa import ROSA, ROSAState, precompute_rosa_ids_for_chunks, rosa_async_pipeline, rosa_single


class _FakeLTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.vals = nn.Parameter(torch.ones(4, 3))
        self.register_buffer("fast_vals", torch.ones(4, 3))
        self.register_buffer("_mom_vals", torch.full((4, 3), 2.0))
        self.register_buffer("timestamps", torch.arange(4, dtype=torch.float32))
        self.register_buffer("sources", torch.arange(4, dtype=torch.long))


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = _FakeLTM()


def _fake_config():
    return SimpleNamespace(
        context_dim=3,
        h_hidden=3,
        l_hidden=3,
        h_stride=1,
        max_h_steps=1,
        max_l_steps=1,
        vocab_size=100,
        rwkv_head_size=1,
    )


def test_chat_state_preserves_full_rosa_history_and_v8_state():
    model = _FakeModel()
    config = _fake_config()
    past_tokens = torch.arange(2048, dtype=torch.long).reshape(1, 2048)
    rosa_state = ROSAState.new()
    rosa_state.tokens = list(range(2048))
    ltm_state = (
        model.ltm.fast_vals,
        model.ltm._mom_vals,
        past_tokens,
        [rosa_state],
        model.ltm.timestamps,
        model.ltm.sources,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    tmp.close()
    path = tmp.name
    save_hierarchical_chat_state(
        path,
        config=config,
        model=model,
        model_path="model.pt",
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_state=ltm_state,
        total_tokens_generated=2048,
    )
    restored = load_hierarchical_chat_state(path, config=config, device=torch.device("cpu"), model=model)

    assert restored["rosa_past_tokens"].shape == (1, 2048)
    assert torch.equal(restored["rosa_past_tokens"], past_tokens)
    assert restored["rosa_states"][0].tokens == list(range(2048))

    rehydrated = _chat_ltm_state_from_rosa_context(
        model,
        restored["rosa_past_tokens"],
        restored["rosa_states"],
    )
    assert len(rehydrated) == 6
    assert torch.equal(rehydrated[2], past_tokens)
    assert rehydrated[3][0].tokens == list(range(2048))


def test_clear_ltm_working_memory_zeros_transient_inference_buffers():
    model = _FakeModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)

    assert clear_ltm_working_memory(model) is True
    assert torch.count_nonzero(model.ltm.fast_vals).item() == 0
    assert torch.count_nonzero(model.ltm._mom_vals).item() == 0
    assert torch.count_nonzero(model.ltm.timestamps).item() == 0
    assert torch.equal(model.ltm.sources, torch.zeros_like(model.ltm.sources))
    assert torch.count_nonzero(model.ltm.vals).item() > 0


def test_rosa_async_pipeline_returns_uncapped_past_tokens():
    past_tokens = torch.arange(20, dtype=torch.long).reshape(1, 20)
    input_ids = torch.tensor([[20, 21]], dtype=torch.long)
    expected = torch.cat([past_tokens, input_ids], dim=1)

    finalize = rosa_async_pipeline(
        input_ids,
        past_tokens,
        rosa_states=None,
        vocab_size=100,
        device=torch.device("cpu"),
        rosa_max_ctx=8,
    )
    rosa_ids, new_past_tokens, new_states = finalize()

    assert rosa_ids.shape == input_ids.shape
    assert torch.equal(new_past_tokens, expected)
    assert new_states[0].tokens == expected.squeeze(0).tolist()


def test_rosa_async_pipeline_extends_uncapped_state_incrementally():
    past_tokens = torch.arange(20, dtype=torch.long).reshape(1, 20)
    input_ids = torch.tensor([[20, 21]], dtype=torch.long)
    state = ROSAState.new()
    rosa_single(past_tokens.squeeze(0).tolist(), state)

    finalize = rosa_async_pipeline(
        input_ids,
        past_tokens,
        rosa_states=[state],
        vocab_size=100,
        device=torch.device("cpu"),
        rosa_max_ctx=8,
    )
    _, new_past_tokens, new_states = finalize()

    assert torch.equal(new_past_tokens, torch.cat([past_tokens, input_ids], dim=1))
    assert new_states[0].tokens == new_past_tokens.squeeze(0).tolist()


def test_rosa_precompute_matches_full_history_across_chunks():
    tokens = [1, 2, 3, 1, 2, 4, 1, 2, 3, 5]
    vocab_size = 100
    expected = [vocab_size if pred == -1 else pred for pred in ROSA(tokens)]

    actual = precompute_rosa_ids_for_chunks(
        tokens,
        vocab_size=vocab_size,
        chunk_size=3,
        rosa_max_ctx=2,
    )

    assert actual == expected
