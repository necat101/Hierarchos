import torch
import torch.nn as nn
from hierarchos import HierarchosCore

def test_l_state_gradient_flow():
    print("\n=== Test: Worker State Gradient Flow Depth ===")
    
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
        'max_length': 64,
        'detach_every_n_steps': 100 # Set high to avoid intended TBPTT for this test
    }
    
    model = HierarchosCore(config)
    model.train()
    
    B, T = 1, 5
    input_ids = torch.randint(0, 100, (B, T))
    labels = torch.randint(0, 100, (B, T))
    
    # Create initial l_state with grad
    l_state_init = torch.zeros(B, config['l_hidden'], 5, requires_grad=True)
    
    # Zero out l_feedback_proj to isolate l_state recurrent path
    if hasattr(model, 'l_feedback_proj'):
        nn.init.constant_(model.l_feedback_proj.weight, 0.0)
    
    # Forward pass
    outputs = model(input_ids, labels=labels, l_state=l_state_init)
    logits = outputs['logits'] # [B, T, vocab_size]
    
    # We want to check if gradients flow from the LAST step back to init
    # So we only compute loss on the last step
    last_step_logits = logits[:, -1, :]
    last_step_labels = labels[:, -1]
    
    loss = torch.nn.functional.cross_entropy(last_step_logits, last_step_labels)
    
    print(f"Loss (last step only): {loss.item()}")
    
    loss.backward()
    
    if l_state_init.grad is not None:
        grad_norm = l_state_init.grad.norm().item()
        print(f"Initial l_state grad norm: {grad_norm}")
        if grad_norm > 0:
            print("[PASS] Gradients flowed from last step back to initial l_state.")
        else:
            print("[FAIL] Initial l_state grad is zero. Chain broken.")
    else:
        print("[FAIL] Initial l_state has no grad.")

if __name__ == "__main__":
    test_l_state_gradient_flow()
