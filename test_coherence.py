import torch
import numpy as np
import random
from hierarchos import HierarchosCore, LTMModule

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def test_deterministic_memory():
    print("\n=== Test: Deterministic Memory (Token Timestamps) ===")
    set_seed(42)
    
    config = {
        'vocab_size': 100,
        'context_dim': 32,
        'persistent_dim': 8,
        'ltm_slots': 32,
        'ltm_key_dim': 16,
        'ltm_val_dim': 16,
        'ltm_lr': 0.1,
        'ltm_topk': 2,
        'h_hidden': 32,
        'l_hidden': 32,
        'max_h_steps': 1,
        'max_l_steps': 1,
        'l_conv_atol': 1e-4,
        'h_stride': 2,
        'commitment_threshold': 0.05,
        'max_length': 64
    }
    
    model = HierarchosCore(config)
    model.eval()
    
    # Create dummy data
    input_ids = torch.randint(0, 100, (1, 10))
    
    # Run 1
    model.reset_memory()
    outputs1 = model(input_ids)
    
    # Simulate an update with a specific timestamp
    # We need to manually call update_memory because the forward pass doesn't update LTM in eval mode/inference
    # unless we are in the training loop.
    # Let's simulate what the training loop does.
    
    topk_idx = outputs1['topk_idx']
    # Create fake grads with shape [B, T, K, val_dim]
    # topk_idx shape is [B, T, K]
    val_dim = model.ltm.vals.shape[-1]
    fake_grads = torch.randn(topk_idx.shape[0], topk_idx.shape[1], topk_idx.shape[2], val_dim)
    
    # Update with timestamp 10.0
    model.update_memory(topk_idx, fake_grads, timestamp=10.0)
    
    state1 = model.ltm.fast_vals.clone()
    timestamps1 = model.ltm.timestamps.clone()
    
    # Verify update actually happened
    if torch.all(state1 == 0):
        print("[FAIL] Memory state is still all zeros after update!")
        return False
    else:
        print(f"[INFO] Memory state updated successfully. Mean: {state1.mean().item()}")
    
    # Run 2 (Reset and repeat)
    set_seed(42) # Reset seed to ensure same initialization if we re-created model, but here we just reset memory
    model.reset_memory()
    
    # Verify reset worked
    assert torch.all(model.ltm.fast_vals == 0), "Memory not reset correctly"
    assert torch.all(model.ltm.timestamps == 0), "Timestamps not reset correctly"
    
    outputs2 = model(input_ids)
    topk_idx2 = outputs2['topk_idx']
    
    # Reuse fake_grads from Run 1 to ensure we are testing the update logic deterministically
    # independent of RNG drift from model initialization steps in Run 1.
    fake_grads2 = fake_grads.clone()
    
    # Update with SAME timestamp 10.0
    model.update_memory(topk_idx2, fake_grads2, timestamp=10.0)
    
    state2 = model.ltm.fast_vals.clone()
    timestamps2 = model.ltm.timestamps.clone()
    
    # Compare
    print(f"Topk1 sum: {topk_idx.float().sum().item()}")
    print(f"Topk2 sum: {topk_idx2.float().sum().item()}")
    print(f"Grads1 sum: {fake_grads.sum().item()}")
    print(f"Grads2 sum: {fake_grads2.sum().item()}")
    print(f"State1 sum: {state1.sum().item()}")
    print(f"State2 sum: {state2.sum().item()}")
    
    if torch.allclose(state1, state2) and torch.allclose(timestamps1, timestamps2):
        print("[PASS] Memory state and timestamps are deterministic.")
    else:
        print("[FAIL] Memory state or timestamps mismatch!")
        print(f"State diff: {(state1 - state2).abs().max().item()}")
        print(f"Timestamp diff: {(timestamps1 - timestamps2).abs().max().item()}")
        return False
        
    # Test 3: Different timestamp should produce different result (if time encoding is used)
    # Actually, the update logic stores the timestamp. So checking timestamps is enough.
    
    model.reset_memory()
    model.update_memory(topk_idx, fake_grads, timestamp=20.0)
    timestamps3 = model.ltm.timestamps.clone()
    
    if not torch.allclose(timestamps1, timestamps3):
        print("[PASS] Different timestamps stored correctly.")
    else:
        print("[FAIL] Timestamps did not update!")
        return False

    return True

if __name__ == "__main__":
    try:
        if test_deterministic_memory():
            print("\nAll coherence tests passed!")
            exit(0)
        else:
            print("\nCoherence tests failed!")
            exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
