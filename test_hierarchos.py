
import torch
import torch.nn as nn
import os
from hierarchos import HierarchosCore, AttrDict

def test_training_step():
    print("--- Testing Hierarchos Training Step ---")
    
    config = {
        'vocab_size': 1000,
        'context_dim': 64,
        'persistent_dim': 16,
        'ltm_slots': 128,
        'ltm_key_dim': 32,
        'ltm_val_dim': 32,
        'ltm_lr': 0.01,
        'ltm_topk': 4,
        'h_hidden': 64,
        'l_hidden': 64,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'l_conv_atol': 1e-4,
        'h_stride': 4,
        'commitment_threshold': 0.05,
        'compile': False, # Start with False
        'gradient_checkpointing': False
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = HierarchosCore(config).to(device)
    model.train()
    
    # Create dummy batch
    B, T = 2, 32
    input_ids = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    labels = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    
    # Run forward pass
    print("Running forward pass...")
    outputs = model(input_ids=input_ids, labels=labels)
    
    loss = outputs['loss']
    print(f"Loss: {loss.item()}")
    
    if torch.isnan(loss):
        print("FAILURE: Loss is NaN")
        return
        
    # Run backward pass
    print("Running backward pass...")
    loss.backward()
    print("Backward pass complete.")
    
    # Check gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"WARNING: NaN gradient in {name}")
                
    # Test LTM Update
    print("Testing LTM Update...")
    if outputs.get("topk_vals") is not None and outputs["topk_vals"].grad is not None:
        ltm_grads = outputs["topk_vals"].grad
        model.ltm.inner_update(
            outputs["topk_idx"], 
            ltm_grads, 
            current_lr=0.01, 
            source=2,
            tokens_covered=T
        )
        print("LTM Update complete.")
    else:
        print("Skipping LTM update (no grads or topk_vals)")

def test_compile():
    if os.name == 'nt' and not torch.cuda.is_available():
        print("Skipping compile test on Windows CPU (known issue)")
        return

    print("\n--- Testing Hierarchos with torch.compile ---")
    config = {
        'vocab_size': 1000,
        'context_dim': 64,
        'persistent_dim': 16,
        'ltm_slots': 128,
        'ltm_key_dim': 32,
        'ltm_val_dim': 32,
        'ltm_lr': 0.01,
        'ltm_topk': 4,
        'h_hidden': 64,
        'l_hidden': 64,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'l_conv_atol': 1e-4,
        'h_stride': 4,
        'commitment_threshold': 0.05,
        'compile': True, # Enable compile
        'force_compile': True,
        'gradient_checkpointing': False
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchosCore(config).to(device)
    model.train()
    
    B, T = 2, 32
    input_ids = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    labels = torch.randint(0, config['vocab_size'], (B, T)).to(device)
    
    try:
        print("Running compiled forward pass...")
        outputs = model(input_ids=input_ids, labels=labels)
        print(f"Compiled Loss: {outputs['loss'].item()}")
    except Exception as e:
        print(f"FAILURE: Compiled run failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_step()
    test_compile()
