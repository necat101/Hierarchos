
import torch
import torch.nn as nn
from hierarchos import HierarchosCore, AttrDict

def test_coherence():
    print("--- Testing Hierarchos Coherence (Train vs Inference) ---")
    
    # 1. Setup Config
    config = AttrDict({
        'context_dim': 32,
        'persistent_dim': 16,
        'h_hidden': 32,
        'l_hidden': 32,
        'ltm_slots': 64,
        'ltm_key_dim': 16,
        'ltm_val_dim': 16,
        'ltm_topk': 2,
        'h_stride': 4,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'vocab_size': 100,
        'max_length': 20,
        'l_conv_atol': 1e-4,
        'ltm_lr': 0.01,
        'device': 'cpu'
    })
    
    torch.manual_seed(42)
    model = HierarchosCore(config)
    model.eval() # Test in eval mode first to avoid dropout noise
    
    # 2. Create Dummy Input
    B, T = 1, 12
    input_ids = torch.randint(0, config.vocab_size, (B, T))
    
    # 3. Run "Training" Forward (Full Sequence)
    print("Running Parallel Forward (Training Mode)...")
    with torch.no_grad():
        train_out = model(input_ids)
        train_logits = train_out['logits']
        train_h_state = train_out['h_state']
        train_l_state = train_out['l_state']
    
    # 4. Run "Inference" Loop (Step-by-Step)
    print("Running Recurrent Loop (Inference Mode)...")
    
    # Initialize States
    h_state = None
    l_state = None
    prev_context = None
    target_context = None
    drift_state = None
    ltm_memory_state = None
    
    infer_logits_list = []
    
    with torch.no_grad():
        for t in range(T):
            token = input_ids[:, t:t+1]
            
            # We need to simulate the forward pass but for a single token
            # HierarchosCore.forward handles this if we pass states
            
            out = model(
                token,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=drift_state,
                ltm_memory_state=ltm_memory_state,
                global_pos_offset=t # Important for Stride/Lerp
            )
            
            infer_logits_list.append(out['logits'])
            
            # Update States
            h_state = out['h_state']
            l_state = out['l_state']
            prev_context = out['prev_context']
            target_context = out['target_context']
            drift_state = out['drift_state']
            ltm_memory_state = out['ltm_memory_state']
            
    infer_logits = torch.cat(infer_logits_list, dim=1)
    
    # 5. Compare
    print("\n--- Results ---")
    
    # Logits
    diff = (train_logits - infer_logits).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"Logits Max Diff: {max_diff:.6f}")
    print(f"Logits Mean Diff: {mean_diff:.6f}")
    
    if max_diff > 1e-4:
        print("FAIL: Significant drift detected!")
        # Analyze where it starts
        for t in range(T):
            d = (train_logits[:, t] - infer_logits[:, t]).abs().max().item()
            if d > 1e-4:
                print(f"Divergence starts at step {t} (Diff: {d:.6f})")
                break
    else:
        print("PASS: Training and Inference match.")

if __name__ == "__main__":
    test_coherence()
