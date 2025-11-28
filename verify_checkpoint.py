import torch

ckpt = torch.load(r'C:\Users\User\Downloads\Hierarchos-main\Hierarchos-main\rog_ally_model\hierarchos_epoch_1.pt', 
                  map_location='cpu', weights_only=False)

state_dict = ckpt['model_state_dict']
config = ckpt['config']

print("=" * 70)
print("CHECKPOINT FILE ANALYSIS")
print("=" * 70)

# Count params in state dict
total_params_all = sum(p.numel() for p in state_dict.values())
total_params_no_deltas = sum(p.numel() for k, p in state_dict.items() if 'ltm_deltas' not in k)

print(f"\nTotal params in checkpoint state_dict: {total_params_all:,}")
print(f"(excluding ltm_deltas buffer): {total_params_no_deltas:,}")

print("\nConfig dimensions:")
print(f"  context_dim:    {config['context_dim']}")
print(f"  h_hidden:       {config['h_hidden']}")
print(f"  l_hidden:       {config['l_hidden']}")
print(f"  persistent_dim: {config['persistent_dim']}")
print(f"  ltm_key_dim:    {config['ltm_key_dim']}")
print(f"  ltm_val_dim:    {config['ltm_val_dim']}")

# Show largest layers
print("\nTop 10 largest layers:")
sorted_layers = sorted(state_dict.items(), key=lambda x: x[1].numel(), reverse=True)[:10]
for name, tensor in sorted_layers:
    print(f"  {name:50s} {tensor.shape!s:30s} {tensor.numel():>12,} params")

print("\n" + "=" * 70)
