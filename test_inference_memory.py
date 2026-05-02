import torch
import numpy as np
from hierarchos import HierarchosCore, QuantizedHierarchos, LTMModule, AttrDict

def test_inference_memory_update():
    print("\n=== Test: Inference Memory Gating ===")
    
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
    
    model = HierarchosCore(AttrDict(config))
    model.eval() # Set to eval mode (inference)
    
    # Create dummy input
    input_ids = torch.randint(0, 100, (1, 10))
    
    # 1. Initial State
    model.reset_memory()
    initial_fast_vals = model.ltm.fast_vals.clone()
    
    # 2. Plain inference is read-only for LTM. Hebbian writes are reserved
    # for explicit validation/praise paths.
    outputs = model(input_ids)
    
    # 3. Check if memory changed
    final_fast_vals = model.ltm.fast_vals.clone()
    
    diff = (final_fast_vals - initial_fast_vals).abs().sum().item()
    print(f"Memory Diff (plain HierarchosCore.eval): {diff}")
    
    if diff == 0:
        print("[PASS] Plain inference left LTM unchanged.")
    else:
        print("[FAIL] Plain inference wrote to LTM unexpectedly!")
        raise AssertionError("Plain inference wrote to LTM unexpectedly")

    # 4. Explicit validation/praise path still allows Hebbian memory writes.
    with torch.no_grad():
        model(input_ids, allow_hebbian_update=True)
    validation_diff = (model.ltm.fast_vals - initial_fast_vals).abs().sum().item()
    print(f"Memory Diff (validation Hebbian): {validation_diff}")

    if validation_diff > 0:
        print("[PASS] Validation Hebbian update wrote to LTM.")
    else:
        print("[FAIL] Validation Hebbian update did not write to LTM!")
        raise AssertionError("Validation Hebbian update did not write to LTM")
        
    # Note: QuantizedHierarchos requires a quantized model file to load, 
    # so we can't easily test it here without creating one. 
    # We will focus on HierarchosCore first as it shares logic.

if __name__ == "__main__":
    test_inference_memory_update()
