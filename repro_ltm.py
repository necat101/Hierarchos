
import torch
import torch.nn as nn
from hierarchos import HierarchosCore, AttrDict, LTMModule

def test_ltm_persistence():
    print("--- Testing LTM Persistence (In-Place Updates) ---")
    
    # 1. Setup Config
    config = AttrDict({
        'context_dim': 32,
        'persistent_dim': 16,
        'h_hidden': 32,
        'l_hidden': 32,
        'ltm_slots': 64,
        'ltm_key_dim': 16,
        'ltm_val_dim': 16,
        'ltm_topk': 2,
        'h_stride': 4,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'vocab_size': 100,
        'max_length': 20,
        'l_conv_atol': 1e-4,
        'ltm_lr': 0.1, # High LR to see changes
        'device': 'cpu'
    })
    
    torch.manual_seed(42)
    model = HierarchosCore(config)
    model.eval()
    
    # 2. Initial State Check
    initial_fast = model.ltm.fast_vals.clone()
    initial_mom = model.ltm._mom_vals.clone()
    
    print(f"Initial Fast Vals Sum: {initial_fast.sum().item():.6f}")
    
    # 3. Simulate an Update (Hebbian - Inference Style)
    print("\nPerforming Hebbian Update (Inference)...")
    topk_idx = torch.tensor([[0, 1]], dtype=torch.long)
    vals = torch.randn(1, 2, config.ltm_val_dim)
    
    # Call inner_update directly with inplace=True (as done in chat/inference)
    model.ltm.update_memory_hebbian(
        topk_idx, 
        None, 
        vals, 
        current_lr=config.ltm_lr, 
        timestamp=1.0, 
        tokens_covered=1, 
        inplace=True
    )
    
    # 4. Check Persistence
    current_fast = model.ltm.fast_vals.clone()
    diff = (current_fast - initial_fast).abs().sum().item()
    print(f"Fast Vals Diff after Hebbian: {diff:.6f}")
    
    if diff < 1e-6:
        print("FAIL: Hebbian update not persisted!")
    else:
        print("PASS: Hebbian update persisted.")
        
    # 5. Simulate Gradient Update (Training Style - but using inner_update directly)
    print("\nPerforming Gradient Update (Training)...")
    grads = torch.randn(1, 2, config.ltm_val_dim)
    
    model.ltm.inner_update(
        topk_idx,
        grads,
        current_lr=config.ltm_lr,
        timestamp=2.0,
        source=LTMModule.SRC_TRAINING_DATA,
        inplace=True
    )
    
    # 6. Check Persistence
    new_fast = model.ltm.fast_vals.clone()
    diff_grad = (new_fast - current_fast).abs().sum().item()
    print(f"Fast Vals Diff after Gradient: {diff_grad:.6f}")
    
    if diff_grad < 1e-6:
        print("FAIL: Gradient update not persisted!")
    else:
        print("PASS: Gradient update persisted.")

if __name__ == "__main__":
    test_ltm_persistence()
