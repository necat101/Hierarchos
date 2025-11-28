import torch
import numpy as np
from hierarchos import HierarchosCore, QuantizedHierarchos, LTMModule

def test_inference_memory_update():
    print("\n=== Test: Inference Memory Update ===")
    
    # Config
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
    model.eval() # Set to eval mode (inference)
    
    # Create dummy input
    input_ids = torch.randint(0, 100, (1, 10))
    
    # 1. Initial State
    model.reset_memory()
    initial_fast_vals = model.ltm.fast_vals.clone()
    
    # 2. Run Forward Pass (Inference)
    # We expect the model to update memory IF we enable it. 
    # Currently, it likely won't update in eval mode without our fix.
    outputs = model(input_ids)
    
    # 3. Check if memory changed
    final_fast_vals = model.ltm.fast_vals.clone()
    
    diff = (final_fast_vals - initial_fast_vals).abs().sum().item()
    print(f"Memory Diff (HierarchosCore.eval): {diff}")
    
    if diff == 0:
        print("[FAIL] Memory did not update during inference (HierarchosCore)!")
    else:
        print("[PASS] Memory updated during inference (HierarchosCore).")
        
    # Note: QuantizedHierarchos requires a quantized model file to load, 
    # so we can't easily test it here without creating one. 
    # We will focus on HierarchosCore first as it shares logic.

if __name__ == "__main__":
    test_inference_memory_update()
