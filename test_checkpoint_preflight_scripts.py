import subprocess
import sys

import torch


def _write_checkpoint(path, state_dict):
    torch.save(
        {
            "model_state_dict": state_dict,
            "config": {
                "context_dim": 8,
                "h_hidden": 8,
                "l_hidden": 8,
                "persistent_dim": 4,
                "ltm_key_dim": 4,
                "ltm_val_dim": 4,
            },
        },
        path,
    )


def _run_script(script_name, checkpoint_path):
    return subprocess.run(
        [sys.executable, script_name, str(checkpoint_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def test_verify_checkpoint_rejects_legacy_scalar_rwkv(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    _write_checkpoint(
        ckpt,
        {
            "h_rnn.time_decay": torch.zeros(8),
            "h_rnn.time_mix_k": torch.zeros(1, 1, 8),
            "l_rnn.time_decay": torch.zeros(8),
        },
    )

    result = _run_script("verify_checkpoint.py", ckpt)

    assert result.returncode == 1
    assert "Unsupported legacy scalar-RWKV checkpoint" in result.stdout


def test_check_config_rejects_legacy_scalar_rwkv(tmp_path):
    ckpt = tmp_path / "legacy.pt"
    _write_checkpoint(
        ckpt,
        {
            "h_rnn.time_decay": torch.zeros(8),
            "h_rnn.time_mix_k": torch.zeros(1, 1, 8),
            "l_rnn.time_decay": torch.zeros(8),
        },
    )

    result = _run_script("check_config.py", ckpt)

    assert result.returncode == 1
    assert "Unsupported legacy scalar-RWKV checkpoint" in result.stdout


def test_checkpoint_preflight_scripts_accept_v8_rwkv(tmp_path):
    ckpt = tmp_path / "v8.pt"
    _write_checkpoint(
        ckpt,
        {
            "h_rnn.x_r": torch.zeros(1, 8),
            "h_rnn.r_k": torch.zeros(1, 8),
            "l_rnn.x_r": torch.zeros(1, 8),
        },
    )

    verify = _run_script("verify_checkpoint.py", ckpt)
    config = _run_script("check_config.py", ckpt)

    assert verify.returncode == 0
    assert "RWKV architecture: v8 matrix-state OK" in verify.stdout
    assert config.returncode == 0
    assert "RWKV architecture: v8 matrix-state OK" in config.stdout
