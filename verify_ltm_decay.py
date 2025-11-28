import torch
import torch.nn as nn
from hierarchos import LTMModule

def test_ltm_decay_parity():
    print("Testing LTM Decay Parity...")
    
    # Setup - Disable WD and Momentum to isolate Decay
    ltm = LTMModule(n_slots=10, key_dim=4, val_dim=4, forget_rate=0.1, reference_chunk_len=10, wd=0.0, momentum=0.0)
    
    # Create dummy data
    topk_idx = torch.tensor([[0, 1]])
    vals = torch.ones(1, 2, 4) # Value of 1.0
    
    # 1. Test inner_update (Training Path)
    ltm.reset_working_memory()
    # Initial fast_vals is 0.
    # We need to set fast_vals to something non-zero first.
    ltm.fast_vals.fill_(1.0)
    
    # Pass ZERO grads to trigger update logic but with 0 signal
    grads_zero = torch.zeros(1, 2, 4)
    
    fast_vals_train, _ = ltm.inner_update(
        topk_idx, 
        grads_tensor=grads_zero, 
        current_lr=0.01,
        timestamp=0.0,
        tokens_covered=10,
        inplace=False
    )
    
    expected_decay = (1.0 - 0.1) ** (10 / 10) # 0.9
    print(f"Train (inner_update) Value: {fast_vals_train[0,0].item():.4f}, Expected: {expected_decay:.4f}")
    
    # 2. Test update_memory_hebbian (Inference Path)
    ltm.reset_working_memory()
    ltm.fast_vals.fill_(1.0)
    
    # We pass vals=0 to avoid adding new info, just want to see decay?
    # update_memory_hebbian: new_fast = curr_fast * retention_rate + slot_updates * current_lr
    # If we pass vals=0, slot_updates=0.
    vals_zero = torch.zeros_like(vals)
    
    fast_vals_infer = ltm.update_memory_hebbian(
        topk_idx, 
        keys=None, 
        vals=vals_zero, 
        current_lr=0.01, 
        timestamp=0.0,
        tokens_covered=10, # Simulate full chunk for parity check
        inplace=False
    )
    
    print(f"Infer (hebbian) Value: {fast_vals_infer[0,0].item():.4f}, Expected: {expected_decay:.4f}")
    
    assert torch.allclose(fast_vals_train, fast_vals_infer), "Mismatch between Training and Inference decay!"
    print("SUCCESS: Decay logic matches!")

if __name__ == "__main__":
    test_ltm_decay_parity()
