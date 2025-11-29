import torch
import torch.nn as nn
from hierarchos import HierarchosCore

def test_momentum_inference():
    print("=== Test: Momentum in Inference LTM Updates ===")
    
    config = dict(
        vocab_size=100,
        context_dim=32, # Renamed from n_embd to match HierarchosCore
        persistent_dim=32, # Added required
        ltm_slots=100, # Renamed
        ltm_key_dim=32,
        ltm_val_dim=32,
        ltm_topk=1, 
        ltm_lr=0.1,
        ltm_momentum=0.9, 
        ltm_forget_rate=0.0, # Renamed from alpha
        h_hidden=32,
        l_hidden=32,
        max_h_steps=2,
        max_l_steps=2,
        l_conv_atol=1e-4,
        device='cpu'
    )
    
    model = HierarchosCore(config)
    model.eval()
    
    # Initialize memory to zeros
    model.ltm.fast_vals.zero_()
    model.ltm._mom_vals.zero_()
    
    # Create a dummy input that maps to a specific memory slot
    # We'll force the retrieval to pick slot 0 by manually setting keys if needed,
    # but for now let's just rely on the model's projection.
    # Actually, to be precise, let's mock retrieve_topk to always return slot 0.
    
    original_retrieve = model.ltm.retrieve_topk
    
    def mock_retrieve(*args, **kwargs):
        # Return slot 0
        topk_idx = torch.zeros((1, 1), dtype=torch.long)
        topk_vals = torch.zeros((1, 1, 32))
        topk_ts = torch.zeros((1, 1))
        return topk_vals, topk_idx, topk_ts
        
    model.ltm.retrieve_topk = mock_retrieve
    
    # Input sequence (constant)
    input_ids = torch.randint(0, 100, (1, 1))
    
    # Step 1: Run inference
    # This should trigger an update.
    # Target value (val_to_store) will be generated.
    # Gradient = -Target
    # Momentum_new = 0 * 0.9 + (-Target) = -Target
    # Memory_new = Memory_old - lr * (-Target) = +lr * Target
    
    with torch.no_grad():
        out1 = model(input_ids)
    
    mem_step1 = model.ltm.fast_vals[0].clone()
    mom_step1 = model.ltm._mom_vals[0].clone()
    
    print(f"Step 1 Memory Norm: {mem_step1.norm().item()}")
    print(f"Step 1 Momentum Norm: {mom_step1.norm().item()}")
    
    # Step 2: Run inference again with same input
    # Target is approx same.
    # Gradient = -Target
    # Momentum_new = Momentum_old * 0.9 + (-Target) = -Target * 0.9 - Target = -1.9 * Target
    # Memory_new = Memory_old - lr * (-1.9 * Target) = Memory_old + 1.9 * lr * Target
    
    with torch.no_grad():
        out2 = model(input_ids)
        
    mem_step2 = model.ltm.fast_vals[0].clone()
    mom_step2 = model.ltm._mom_vals[0].clone()
    
    print(f"Step 2 Memory Norm: {mem_step2.norm().item()}")
    print(f"Step 2 Momentum Norm: {mom_step2.norm().item()}")
    
    # Verification
    # If momentum is working, the update in step 2 should be larger than step 1 (approx 1.9x)
    # Update 1 = mem_step1 - 0
    # Update 2 = mem_step2 - mem_step1
    
    update1 = mem_step1.norm().item()
    update2 = (mem_step2 - mem_step1).norm().item()
    
    print(f"Update 1 Magnitude: {update1}")
    print(f"Update 2 Magnitude: {update2}")
    
    if update2 > update1 * 1.5:
        print("[PASS] Momentum is amplifying updates!")
    else:
        print("[FAIL] Momentum not observed (updates are constant or decaying).")
        exit(1)

if __name__ == "__main__":
    test_momentum_inference()
