import sys
import os
import torch
import importlib.util

# 1. Load Monolithic Version
spec = importlib.util.spec_from_file_location("hierarchos_mono", "hierarchos.py")
orig_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orig_mod)

# 2. Load Modular Version
sys.path.insert(0, '.')
from hierarchos.models.core import HierarchosCore as ModularHierarchos
from hierarchos.training.trainer import AttrDict

def compare_models(ckpt_path):
    print(f"--- Deep Parity Check: {ckpt_path} ---")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    
    # Standard config for both
    cfg_base = {
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
        'ltm_lr': 1e-2,
        'l_conv_atol': 1e-4, #monolith default
        'detach_every_n_steps': 32,
        'compile': False
    }
    
    # Merge checkpoint config
    for k, v in ckpt.get('config', {}).items():
        cfg_base[k] = v
        
    # --- Instantiate Models ---
    model_orig = orig_mod.HierarchosCore(cfg_base)
    model_mod = ModularHierarchos(AttrDict(cfg_base))
    
    # --- Exact Weight Sync ---
    model_orig.load_state_dict(ckpt['model_state_dict'], strict=True)
    model_mod.load_state_dict(ckpt['model_state_dict'], strict=True)
    
    model_orig.eval()
    model_mod.eval()
    
    # --- Test Inputs ---
    torch.manual_seed(42)
    x = torch.randint(0, 1000, (1, 16)) # Use 16 tokens to cross a stride boundary
    labels = x.clone()
    labels[:, :5] = -100 # Mask some labels
    
    print("\nRunning Forward Passes...")
    with torch.no_grad():
        out_orig = model_orig(x, labels=labels)
        out_mod = model_mod(x, labels=labels)
    
    # --- Comparison ---
    print("\n--- Numerical Comparison ---")
    
    # 1. Logits
    logits_close = torch.allclose(out_orig['logits'], out_mod['logits'], atol=1e-6)
    logits_max_diff = (out_orig['logits'] - out_mod['logits']).abs().max().item()
    print(f"Logits Match: {logits_close} (Max Diff: {logits_max_diff:.2e})")
    
    # 2. Loss
    loss_close = torch.allclose(out_orig['loss'], out_mod['loss'], atol=1e-7)
    print(f"Loss Match:   {loss_close} (Orig: {out_orig['loss'].item():.6f}, Mod: {out_mod['loss'].item():.6f})")
    
    # 3. Ponder Cost
    p_orig = out_orig['ponder_cost']
    p_mod = out_mod['ponder_cost']
    p_close = torch.allclose(p_orig, p_mod, atol=1e-7)
    print(f"Ponder Match: {p_close} (Orig: {p_orig.item():.6f}, Mod: {p_mod.item():.6f})")
    
    # 4. commitment Cost
    c_orig = out_orig['commitment_cost']
    c_mod = out_mod['commitment_cost']
    c_close = torch.allclose(c_orig, c_mod, atol=1e-7)
    print(f"Commitment Match: {c_close} (Orig: {c_orig.item():.6f}, Mod: {c_mod.item():.6f})")

    # 5. Hidden States (Final)
    h_close = torch.allclose(out_orig['h_state'], out_mod['h_state'], atol=1e-6)
    l_close = torch.allclose(out_orig['l_state'], out_mod['l_state'], atol=1e-6)
    print(f"H-State Match: {h_close}")
    print(f"L-State Match: {l_close}")

    if logits_close and loss_close and p_close and c_close:
        print("\n[SUCCESS] Modular and Monolithic versions are numerically IDENTICAL.")
    else:
        print("\n[FAILURE] Numerical discrepancy detected.")
        sys.exit(1)

if __name__ == "__main__":
    compare_models('rog_ally_model/hierarchos_epoch_31.pt')
