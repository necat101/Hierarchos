import torch
import json

ckpt = torch.load(r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_1.pt', 
                  map_location='cpu', weights_only=False)

config = ckpt.get('config', {})

print("=" * 60)
print("CHECKPOINT CONFIGURATION")
print("=" * 60)
print(json.dumps(config, indent=2) if isinstance(config, dict) else str(config))
print("=" * 60)

# Also check model state dict keys to understand structure
print("\nModel State Dict Keys (first 30):")
state_keys = list(ckpt.get('model_state_dict', {}).keys())[:30]
for key in state_keys:
    print(f"  {key}")
