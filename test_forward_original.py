import os
import importlib.util
import sys

import pytest
import torch

DEFAULT_CKPT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "rog_ally_model",
    "hierarchos_epoch_31.pt",
)


def test_original_forward_checkpoint():
    if not os.path.exists(DEFAULT_CKPT_PATH):
        pytest.skip(f"legacy comparison checkpoint not found: {DEFAULT_CKPT_PATH}")

    spec = importlib.util.spec_from_file_location("hierarchos_mono", "hierarchos.py")
    orig = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = orig
    spec.loader.exec_module(orig)

    ckpt = torch.load(DEFAULT_CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = {
        "vocab_size": 50257,
        "context_dim": 384,
        "h_hidden": 384,
        "l_hidden": 384,
        "ltm_slots": 1024,
        "ltm_key_dim": 128,
        "ltm_val_dim": 128,
        "ltm_topk": 4,
        "persistent_dim": 128,
        "max_h_steps": 5,
        "max_l_steps": 5,
        "h_stride": 4,
        "max_length": 1024,
        "compile": False,
        "commitment_threshold": 0.1,
    }
    cfg.update({k: v for k, v in ckpt.get("config", {}).items() if k not in cfg})

    model = orig.HierarchosCore(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    torch.manual_seed(42)
    x = torch.randint(0, 1000, (1, 10))
    labels = x.clone()
    labels[:, 0] = -100

    with torch.no_grad():
        out = model(x, labels=labels)

    loss = out["loss"]
    logits = out["logits"]
    print(f"Original Loss: {loss.item():.4f}")
    print(f"Original Logits shape: {logits.shape}")
    print(f"Original Logits sample (first 5): {logits[0, 0, :5]}")

    assert torch.isfinite(loss)
    assert logits.shape[:2] == x.shape


if __name__ == "__main__":
    test_original_forward_checkpoint()
