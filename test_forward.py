import sys
sys.path.insert(0, '.')
from hierarchos.training.trainer import AttrDict
from hierarchos.models.core import HierarchosCore
import torch

# Load checkpoint
ckpt = torch.load('rog_ally_model/hierarchos_epoch_31.pt', map_location='cpu', weights_only=False)
cfg = AttrDict(ckpt.get('config', {}))
cfg.vocab_size = 50257
cfg.context_dim = 384
cfg.h_hidden = 384
cfg.l_hidden = 384
cfg.ltm_slots = 1024
cfg.ltm_key_dim = 128
cfg.ltm_val_dim = 128
cfg.ltm_topk = 4
cfg.persistent_dim = 128
cfg.max_h_steps = 5
cfg.max_l_steps = 5
cfg.h_stride = 4

# Create and load model
model = HierarchosCore(cfg)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Test forward pass
torch.manual_seed(42)
x = torch.randint(0, 1000, (1, 10))
labels = x.clone()
labels[:, 0] = -100

with torch.no_grad():
    out = model(x, labels=labels)

print(f'Loss: {out["loss"].item():.4f}')
print(f'Logits shape: {out["logits"].shape}')
print(f'Logits sample (first 5): {out["logits"][0, 0, :5]}')
