import sys
import importlib.util

# Directly load the monolithic hierarchos.py file
spec = importlib.util.spec_from_file_location("hierarchos_mono", "hierarchos.py")
orig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig)

import torch

# Load checkpoint
ckpt = torch.load('rog_ally_model/hierarchos_epoch_31.pt', map_location='cpu', weights_only=False)

# Build config as dict
cfg = {
    'vocab_size': 50257,
    'context_dim': 384,
    'h_hidden': 384,
    'l_hidden': 384,
    'ltm_slots': 1024,
    'ltm_key_dim': 128,
    'ltm_val_dim': 128,
    'ltm_topk': 4,
    'persistent_dim': 128,
    'max_h_steps': 5,
    'max_l_steps': 5,
    'h_stride': 4,
    'max_length': 1024,
    'compile': False,
    'commitment_threshold': 0.1,
}
# Merge checkpoint config
for k, v in ckpt.get('config', {}).items():
    if k not in cfg:
        cfg[k] = v

# Create and load original model
model = orig.HierarchosCore(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Test forward pass with same input
torch.manual_seed(42)
x = torch.randint(0, 1000, (1, 10))
labels = x.clone()
labels[:, 0] = -100

with torch.no_grad():
    out = model(x, labels=labels)

print(f'Original Loss: {out["loss"].item():.4f}')
print(f'Original Logits shape: {out["logits"].shape}')
print(f'Original Logits sample (first 5): {out["logits"][0, 0, :5]}')
