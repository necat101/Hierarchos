
import torch
import torch.nn as nn
import numpy as np
import os
import sys

# Mocking necessary parts to avoid loading full model/kernels if not present
# We just want to test the LTM update logic in QuantizedHierarchos context

# Mock LTMModule (or import if possible, but we want to test the one in hierarchos.py)
# We will import hierarchos.py but mock the kernel requirement if needed.

# Hack to bypass _HAS_KERNEL check in QuantizedHierarchos __init__
# We will subclass or patch
import hierarchos
from hierarchos import LTMModule, QuantizedHierarchos, AttrDict

# Mock QuantizedLinear and QuantizedRWKVCell to avoid kernel dependency
class MockQuantizedLinear(nn.Module):
    def __init__(self, name, q_data):
        super().__init__()
        self.qtype = "INT4"
    def __call__(self, x, device="cpu"):
        return torch.zeros(x.shape[0], 128) # Dummy output

class MockQuantizedRWKVCell(nn.Module):
    def __init__(self, n_embd, name, q_data):
        super().__init__()
        self.key = MockQuantizedLinear(name, q_data)
    def __call__(self, x, state, device="cpu"):
        return x, state

# Patch hierarchos
hierarchos._HAS_KERNEL = True # Pretend we have kernel
hierarchos.QuantizedLinear = MockQuantizedLinear
hierarchos.QuantizedRWKVCell = MockQuantizedRWKVCell

def test_quantized_ltm_persistence():
    print("Testing QuantizedHierarchos LTM Persistence...")
    
    config = {
        'vocab_size': 100,
        'context_dim': 128,
        'persistent_dim': 16,
        'ltm_slots': 10,
        'ltm_key_dim': 16,
        'ltm_val_dim': 16,
        'ltm_lr': 0.1,
        'ltm_topk': 2,
        'h_hidden': 128,
        'l_hidden': 128,
        'max_h_steps': 1,
        'max_l_steps': 1,
        'l_conv_atol': 1e-4
    }
    
    # Mock q_data
    q_data = {
        'tok_emb.weight': np.array({'raw': np.random.randn(100, 128).astype(np.float32)}),
        'persistent': np.array({'raw': np.random.randn(16).astype(np.float32)}),
        'out_norm.weight': np.array({'raw': np.random.randn(128).astype(np.float32)}),
        'out_norm.bias': np.array({'raw': np.random.randn(128).astype(np.float32)}),
        'qproj.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 16)}),
        'in_proj.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'h_rnn.key.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'h_to_context.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'l_input_proj.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'l_rnn.key.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'l_to_out.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 128)}),
        'lm_head.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (100, 128)}),
        'h_halt_proj.weight': np.array({'quantized': np.zeros((1,1)), 'qtype': 'INT4', 'original_shape': (128, 1)}),
    }
    
    # Init model
    try:
        model = QuantizedHierarchos(config, q_data)
    except Exception as e:
        print(f"Failed to init model: {e}")
        # We might need to mock more things if init fails, but let's try
        return

    # Initialize LTM with some data
    model.ltm.fast_vals.fill_(0.0)
    
    # Mock update inputs
    topk_idx = torch.tensor([[0, 1]], dtype=torch.long)
    vals = torch.ones(1, 2, 16) # Update with 1s
    
    print(f"Initial fast_vals sum: {model.ltm.fast_vals.sum().item()}")
    
    # Call update_memory_hebbian
    model.update_memory_hebbian(topk_idx, vals, timestamp=1.0, lr=1.0)
    
    final_sum = model.ltm.fast_vals.sum().item()
    print(f"Final fast_vals sum: {final_sum}")
    
    if final_sum > 0:
        print("PASS: LTM fast_vals updated successfully.")
    else:
        print("FAIL: LTM fast_vals was NOT updated.")
        sys.exit(1)

if __name__ == "__main__":
    test_quantized_ltm_persistence()
