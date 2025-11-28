import torch
from hierarchos import HierarchosCore, AttrDict
import sys

def test_device_detection():
    print("Testing Device Detection in HierarchosCore...")
    
    # Mock config
    config = AttrDict({
        'vocab_size': 100,
        'context_dim': 32,
        'h_hidden': 32,
        'l_hidden': 32,
        'compile': True, # Enable compile to trigger the check
        'device': 'cpu',  # Explicitly set device
        'persistent_dim': 32,
        'ltm_slots': 10,
        'ltm_key_dim': 32,
        'ltm_val_dim': 32,
        'h_stride': 4,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'l_conv_atol': 1e-4,
        'ltm_topk': 2,
        'max_length': 128,
        'ltm_lr': 0.01
    })
    
    print("Initializing model with device='cpu' in config...")
    try:
        model = HierarchosCore(config)
        print("Model initialized successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Check if the warning was printed (manual check of output)
    
    print("\nTest 2: Implicit device (default)")
    config2 = AttrDict({
        'vocab_size': 100,
        'context_dim': 32,
        'h_hidden': 32,
        'l_hidden': 32,
        'compile': True,
        # No device specified
        'persistent_dim': 32,
        'ltm_slots': 10,
        'ltm_key_dim': 32,
        'ltm_val_dim': 32,
        'h_stride': 4,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'l_conv_atol': 1e-4,
        'ltm_topk': 2,
        'max_length': 128,
        'ltm_lr': 0.01
    })
    try:
        model2 = HierarchosCore(config2)
        print("Model 2 initialized.")
    except Exception as e:
        print(f"FAILED 2: {e}")

if __name__ == "__main__":
    test_device_detection()
