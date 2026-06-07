import pytest
import torch

from hierarchos.utils.checkpoint import _reject_rwkv_load_mismatch, _reject_unsupported_rwkv_state_dict


def test_v8_rwkv_state_dict_is_accepted():
    state = {
        "h_rnn.x_r": torch.empty(1, 16),
        "h_rnn.r_k": torch.empty(1, 16),
        "l_rnn.x_r": torch.empty(1, 16),
    }

    _reject_unsupported_rwkv_state_dict(state, "v8.pt")


def test_scalar_rwkv_state_dict_is_rejected():
    state = {
        "h_rnn.time_decay": torch.empty(16),
        "h_rnn.time_mix_k": torch.empty(1, 1, 16),
        "l_rnn.time_decay": torch.empty(16),
    }

    with pytest.raises(ValueError, match="v8-only"):
        _reject_unsupported_rwkv_state_dict(state, "legacy.pt")


def test_partial_rwkv_load_mismatch_is_rejected():
    with pytest.raises(ValueError, match="partial recurrent-block loading"):
        _reject_rwkv_load_mismatch(
            missing_keys=["h_rnn.x_r", "lm_head.weight"],
            unexpected_keys=["l_rnn.time_decay"],
            source="bad-v8.pt",
        )
