
import torch
from hierarchos import LTMModule

def test_ltm_update_bug():
    print("Testing LTM Update Bug...")
    ltm = LTMModule(n_slots=10, val_dim=4)
    
    # Initial state should be zero
    print(f"Initial fast_vals sum: {ltm.fast_vals.sum().item()}")
    
    # Simulate update logic from chat/train (calling inner_update directly)
    idx = torch.tensor([0, 1])
    grads = torch.randn(2, 4)
    
    # This is how it's called in chat/train (ignoring return values)
    # ltm.inner_update(idx, grads, current_lr=0.1, timestamp=1.0)
    
    # <<< FIX: Capture and persist LTM updates (Optimized In-Place) >>>
    new_fast, new_mom = ltm.inner_update(idx, grads, current_lr=0.1, timestamp=1.0, inplace=True)
    # ltm.fast_vals.copy_(new_fast) # No longer needed
    # ltm._mom_vals.copy_(new_mom) # No longer needed
    
    # Check if state changed
    current_sum = ltm.fast_vals.sum().item()
    print(f"Post-update fast_vals sum: {current_sum}")
    
    if current_sum == 0:
        print("[FAIL] fast_vals was NOT updated! (Bug confirmed)")
    else:
        print("[PASS] fast_vals WAS updated.")

if __name__ == "__main__":
    test_ltm_update_bug()
