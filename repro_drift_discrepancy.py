
import torch
import torch.nn as nn
from hierarchos import HierarchosCore, QuantizedHierarchos, AttrDict

def test_drift_discrepancy():
    print("--- Testing Drift Discrepancy ---")
    
    # 1. Setup Config
    config = {
        'vocab_size': 100,
        'context_dim': 16,
        'persistent_dim': 4,
        'ltm_slots': 10,
        'ltm_key_dim': 8,
        'ltm_val_dim': 8,
        'ltm_lr': 0.01,
        'ltm_topk': 2,
        'h_hidden': 16,
        'l_hidden': 16,
        'max_h_steps': 2,
        'max_l_steps': 2,
        'l_conv_atol': 1e-4,
        'h_stride': 2,
        'compile': False
    }
    
    # 2. Initialize Model
    model = HierarchosCore(config)
    model.train() # Enable training mode to disable early exit in worker loop
    
    # 3. Create Inputs
    B, T = 1, 2
    input_ids = torch.randint(0, 100, (B, T))
    
    # 4. Run Training Forward (Simulated)
    print("\nRunning HierarchosCore.forward (Training Logic)...")
    out_train = model(input_ids)
    drift_state_train = out_train['drift_state']
    print(f"Train Final Drift State Mean: {drift_state_train.mean().item():.6f}")
    
    # 5. Run Inference Logic (Simulated via QuantizedHierarchos wrapper logic)
    # We can't easily run QuantizedHierarchos without quantizing, but we can inspect the logic.
    # Instead, let's manually run the inference logic using the SAME model weights
    
    print("\nRunning Simulated Inference Logic...")
    
    # Initialize States
    h_state = torch.zeros(B, config['h_hidden'], 5)
    h_state[:, :, 3] = -1e30
    l_state = torch.zeros(B, config['l_hidden'], 5)
    l_state[:, :, 3] = -1e30
    prev_context = torch.zeros(B, config['context_dim'])
    target_context = torch.zeros(B, config['context_dim'])
    
    # We need to mimic QuantizedHierarchos.__call__ loop
    # But using the unquantized model components
    
    final_l_state = l_state
    curr_prev_context = prev_context
    curr_target_context = target_context
    final_h_state = h_state
    
    stride = config['h_stride']
    
    for t in range(T):
        token_ids = input_ids[:, t:t+1] # Keep dim
        
        # ... (Skip embedding/LTM for brevity, assume they match) ...
        # We focus on the Worker/Drift logic
        
        # Assume we have 'enc' from the model's components
        # To get exact 'enc', we need to run the first part of the model
        # Let's just use the model's components
        
        x = model.tok_emb(token_ids).squeeze(1)
        q = torch.clamp(model.qproj(x), min=-10, max=10)
        # Real LTM Retrieval (Match Forward Logic)
        # We need to pass fast_vals=model.ltm.fast_vals to match forward's default behavior
        topk_vals, topk_idx, topk_ts = model.ltm.retrieve_topk(q, config['ltm_topk'], fast_vals=model.ltm.fast_vals)
        
        # Time Encoding (Match Forward Logic)
        # We need time_freqs from model (it might not be exposed directly if it's local in forward?)
        # Wait, time_freqs is a buffer in HierarchosCore?
        # Let's check HierarchosCore init.
        # It seems time_freqs is created in __init__.
        args = topk_ts.unsqueeze(-1) * model.time_freqs.unsqueeze(0).unsqueeze(0)
        pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if config['ltm_val_dim'] % 2 == 1: pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
        topk_vals = topk_vals + pe
        
        # Gate
        gate_input = torch.clamp(model.ltm_gate_logit, min=-50.0, max=50.0)
        gate = torch.sigmoid(gate_input)
        gated_vals = topk_vals * gate
        
        ltm_summary = gated_vals.view(B, -1)
        p_read = model.persistent.unsqueeze(0).expand(B, -1)
        mac_input = torch.cat([x, p_read, ltm_summary], dim=-1)
        enc = torch.nn.functional.gelu(model.in_proj(mac_input))
        enc = torch.clamp(enc, min=-30.0, max=30.0)
        
        # Manager
        l_feedback = model.l_feedback_proj(final_l_state[:, :, 0])
        enc_with_feedback = enc + l_feedback
        h_out_real, new_h_state = model.h_rnn(enc_with_feedback, final_h_state, timestep=t)
        final_h_state = new_h_state
        
        if t % stride == 0:
            curr_prev_context = curr_target_context
            # Manager Pondering (Match Training Logic)
            h_step_outputs = [h_out_real]
            halt_logit = model.h_halt_proj(h_out_real).squeeze(-1)
            h_halt_probs = [torch.sigmoid(halt_logit)]
            
            shadow_h_state = new_h_state.clone()
            current_enc_h = enc_with_feedback
            
            for step_idx in range(config['max_h_steps'] - 1):
                # Run H-RNN on shadow state
                h_out_ponder, shadow_h_state = model.h_rnn(current_enc_h, shadow_h_state, timestep=-(step_idx+1))
                
                halt_logit = model.h_halt_proj(h_out_ponder).squeeze(-1)
                h_step_outputs.append(h_out_ponder)
                h_halt_probs.append(torch.sigmoid(halt_logit))

            # Compute Weighted Average (ACT)
            h_stack = torch.stack(h_step_outputs, dim=0)
            halt_stack = torch.stack(h_halt_probs, dim=0)
            
            remain = 1.0 - halt_stack
            remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
            cum_remain = torch.cumprod(remain_shifted, dim=0)
            weights = halt_stack * cum_remain
            remainder = cum_remain[-1] * (1.0 - halt_stack[-1])
            
            total = weights.sum(dim=0) + remainder + 1e-8
            weights = weights / total.unsqueeze(0)
            remainder = remainder / total
            
            final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]

            curr_target_context = model.h_to_context(final_h_out)
            
        step_in_stride = t % stride
        alpha = step_in_stride / float(stride)
        static_context = torch.lerp(curr_prev_context, curr_target_context, alpha)
        
        # --- INFERENCE DRIFT LOGIC (FIXED) ---
        prev_worker_h = final_l_state[:, :, 0]
        current_drift = torch.tanh(model.context_drift_proj(prev_worker_h))
        current_drift = torch.clamp(current_drift, min=-5.0, max=5.0)
        
        print(f"Step {t}: Inference Initial Drift Mean: {current_drift.mean().item():.6f}")
        
        # Use Shadow State for Pondering
        shadow_l_state = final_l_state.clone()
        
        for step_idx in range(config['max_l_steps']):
            dynamic_context = static_context + current_drift
            l_input_raw = torch.cat([enc, dynamic_context], dim=-1)
            l_input = model.l_input_proj(l_input_raw)
            # Run on Shadow State
            l_out, shadow_l_state = model.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1)) 
            # Add clamping
            shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
            
            drift_delta = torch.tanh(model.context_drift_proj(l_out))
            current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
            
            if torch.mean(torch.abs(drift_delta)) < config['l_conv_atol']:
                break
            
        # Update Real State ONCE
        dynamic_context = static_context + current_drift
        l_input_raw = torch.cat([enc, dynamic_context], dim=-1)
        l_input = model.l_input_proj(l_input_raw)
        l_out, final_l_state = model.l_rnn(l_input, final_l_state, timestep=t)
        # Add clamping
        final_l_state = torch.clamp(final_l_state, min=-50.0, max=50.0)
            
    print(f"Inference Final Drift State Mean: {current_drift.mean().item():.6f}")
    
    # Compare
    diff = abs(drift_state_train.mean().item() - current_drift.mean().item())
    print(f"\nDifference: {diff:.6f}")
    if diff > 1e-5:
        print("FAIL: Drift logic mismatch detected!")
    else:
        print("PASS: Drift logic matches.")

if __name__ == "__main__":
    test_drift_discrepancy()
