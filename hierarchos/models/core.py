"""
HierarchosCore - Full parity version with original forward method.
This is a direct port from hierarchos.py to achieve exact training parity.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .rwkv_cell import RWKVCell
from .ltm import LTMModule
from ..utils.device import setup_msvc_environment, is_directml_device


class WorkerLoop:
    """
    Encapsulates the Worker's iterative refinement loop.
    Direct port from original hierarchos.py for full parity.
    NOTE: This is a plain class, NOT nn.Module, to avoid state_dict key prefixing.
    """
    def __init__(self, config, l_rnn, l_input_proj, context_drift_proj, l_to_out):
        self.config = config
        self.l_rnn = l_rnn
        self.l_input_proj = l_input_proj
        self.context_drift_proj = context_drift_proj
        self.l_to_out = l_to_out
        self.max_l_steps = config.max_l_steps
        self.l_conv_atol = getattr(config, 'l_conv_atol', 0.01)
        self.commitment_threshold = getattr(config, 'commitment_threshold', 0.1)

    def __call__(self, enc: torch.Tensor, static_context: torch.Tensor, l_state: torch.Tensor, 
                initial_drift: torch.Tensor, timestep: Optional[int] = None):
        # Handle batch dimension squeezing for torch.compile compatibility
        if enc.dim() == 3 and enc.shape[0] == 1: enc = enc.squeeze(0)
        if static_context.dim() == 3 and static_context.shape[0] == 1: static_context = static_context.squeeze(0)
        if l_state.dim() == 4 and l_state.shape[0] == 1: l_state = l_state.squeeze(0)
        if initial_drift.dim() == 3 and initial_drift.shape[0] == 1: initial_drift = initial_drift.squeeze(0)

        current_drift = initial_drift
        drift_costs = [] 
        current_enc = enc

        # Shadow state for exploration (pondering)
        shadow_l_state = l_state.clone()
        
        # Initialize dynamic_context
        dynamic_context = static_context + current_drift

        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        l_input = torch.clamp(l_input, min=-50.0, max=50.0)
        
        check_idx = [0, 1, 2, 4]

        if not self.l_rnn.training:
            prev_shadow = shadow_l_state.clone()
            for step_idx in range(self.max_l_steps):
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1))
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out))
                current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = self.l_input_proj(l_input_vec)
                
                drift_converged = torch.mean(torch.abs(drift_delta)) < self.l_conv_atol
                state_converged = torch.allclose(shadow_l_state[..., check_idx], prev_shadow[..., check_idx], atol=self.l_conv_atol)
                if drift_converged or state_converged: break
                prev_shadow = shadow_l_state.clone()
        else:
            for step_idx in range(self.max_l_steps):
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1))
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out))
                current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                drift_sq = torch.sum(current_drift ** 2, dim=-1).mean()
                hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                hinge_cost = torch.clamp(hinge_cost, max=100.0)
                drift_costs.append(hinge_cost)
                
                if torch.mean(torch.abs(drift_delta)) < self.l_conv_atol:
                   break
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = self.l_input_proj(l_input_vec)

        # Use original l_state (not shadow) for the actual state update
        ts = timestep if timestep is not None else 0
        
        # Recalculate l_input with final drift
        dynamic_context = static_context + current_drift
        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        
        final_l_out, next_l_state = self.l_rnn(l_input, l_state, timestep=ts)
        next_l_state = torch.clamp(next_l_state, min=-50.0, max=50.0)
        
        final_enc = current_enc + self.l_to_out(final_l_out)
        commitment_cost = torch.tensor(0.0, device=enc.device)
        if len(drift_costs) > 0:
            commitment_cost = torch.stack(drift_costs).mean()

        return final_enc, next_l_state, commitment_cost, current_drift


class HierarchosCore(nn.Module):
    """
    Full parity version of HierarchosCore - direct port from hierarchos.py.
    """
    
    def reset_memory(self):
        """Resets the short-term 'fast' associative memory."""
        self.ltm.reset_working_memory()
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Tokenizer-dependent
        self.tok_emb = nn.Embedding(config.vocab_size, config.context_dim)
        
        # Global Learnable State
        self.persistent_dim = getattr(config, 'persistent_dim', 128)
        self.persistent = nn.Parameter(torch.randn(self.persistent_dim))
        
        # LTM System
        self.ltm = LTMModule(n_slots=config.ltm_slots, key_dim=config.ltm_key_dim, val_dim=config.ltm_val_dim)
        self.qproj = nn.Linear(config.context_dim * 2, config.ltm_key_dim, bias=False)
        self.val_proj = nn.Linear(config.context_dim, config.ltm_val_dim, bias=False)
        
        # Encoder Projection
        in_dim = config.context_dim + self.persistent_dim + config.ltm_val_dim * config.ltm_topk
        self.in_proj = nn.Linear(in_dim, config.context_dim)
        
        # Manager Components
        self.l_feedback_proj = nn.Linear(config.l_hidden, config.h_hidden, bias=False)
        self.h_rnn = RWKVCell(config.h_hidden)
        self.h_to_context = nn.Linear(config.h_hidden, config.context_dim)
        self.h_halt_proj = nn.Linear(config.h_hidden, 1)
        
        # Worker Components
        self.l_input_proj = nn.Linear(config.context_dim * 2, config.l_hidden)
        self.l_rnn = RWKVCell(config.l_hidden)
        self.context_drift_proj = nn.Linear(config.l_hidden, config.context_dim, bias=False)
        self.l_to_out = nn.Linear(config.l_hidden, config.context_dim)
        
        # Output Head
        self.out_norm = nn.LayerNorm(config.context_dim)
        self.lm_head = nn.Linear(config.context_dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying
        
        # Worker Loop Wrapper - pass actual module references
        self.worker_loop_module = WorkerLoop(config, self.l_rnn, self.l_input_proj, 
                                             self.context_drift_proj, self.l_to_out)
        
        # System Meta-parameters
        self.register_buffer("ltm_gate_logit", torch.tensor(0.0))
        self.register_buffer("time_freqs", torch.randn(config.ltm_val_dim // 2))

    def compile(self):
        """Applies torch.compile to the worker loop if enabled in config."""
        if not getattr(self.config, 'compile', False):
            return
        
        device = next(self.parameters()).device
        is_dml = is_directml_device(device)
        
        if not is_dml:
            if os.name == 'nt':
                setup_msvc_environment()

            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            
            print("INFO: Compiling WorkerLoop...")
            try:
                self.worker_loop_module = torch.compile(self.worker_loop_module, dynamic=True)
            except Exception as e:
                print(f"Warning: Compilation failed! {e}")

    def forward(self, input_ids, attention_mask=None, labels=None, 
                h_state=None, l_state=None, 
                prev_context=None, target_context=None,
                drift_state=None, ltm_memory_state=None,
                global_pos_offset=0, min_timestamp=0.0, source_filter=None, **kwargs):
        """
        Full forward method - direct port from hierarchos.py for exact parity.
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)

        # ==================================================================
        # 1. STATE INITIALIZATION (With Context Recovery)
        # ==================================================================
        if h_state is None:
            h_state = torch.zeros(B, self.config.h_hidden, 5, device=device)
            h_state[:, :, 3] = -1e30   
            prev_context = torch.zeros(B, self.config.context_dim, device=device)
            target_context = torch.zeros(B, self.config.context_dim, device=device)
        else:
            h_state = h_state.to(device)
            if prev_context is None:
                prev_context = self.h_to_context(h_state[:, :, 0])
            else:
                prev_context = prev_context.to(device)
            if target_context is None:
                target_context = self.h_to_context(h_state[:, :, 0])
            else:
                target_context = target_context.to(device)

        if l_state is None:
            l_state = torch.zeros(B, self.config.l_hidden, 5, device=device)
            l_state[:, :, 3] = -1e30
        else:
            l_state = l_state.to(device)

        if ltm_memory_state is None:
            curr_fast_vals = self.ltm.fast_vals
            curr_mom_vals = self.ltm._mom_vals
        else:
            curr_fast_vals, curr_mom_vals = ltm_memory_state

        final_embs = []
        ponder_costs = []
        commitment_costs = []
        all_topk_vals = []
        all_topk_idx = []

        stride = self.config.h_stride
        final_drift = None

        # ==================================================================
        # 2. MAIN TIME LOOP
        # ==================================================================
        for t in range(T):
            token_x = x[:, t]
            abs_t = global_pos_offset + t
            
            # --- LTM Retrieval ---
            p = self.persistent.unsqueeze(0).expand(B, -1)
            q_in = torch.cat([token_x, prev_context], dim=-1)
            q = torch.clamp(self.qproj(q_in), min=-10, max=10)
            
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(
                q, self.config.ltm_topk, min_timestamp, source_filter, fast_vals=curr_fast_vals
            )
            
            all_topk_vals.append(topk_vals)
            all_topk_idx.append(topk_idx)
            
            # Positional encoding
            args = topk_ts.unsqueeze(-1) * self.time_freqs.unsqueeze(0).unsqueeze(0)
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1: 
                pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
            topk_vals = topk_vals + pe
            
            gate_input = torch.clamp(self.ltm_gate_logit, min=-50.0, max=50.0)
            gate = torch.sigmoid(gate_input)
            gated_vals = topk_vals * gate
            mac_in = torch.cat([token_x, p, gated_vals.view(B, -1)], dim=-1)
            
            enc = F.gelu(self.in_proj(mac_in))
            enc = torch.clamp(enc, min=-30.0, max=30.0)

            # ==================================================================
            # 3. HIERARCHICAL MANAGER (Continuous Watch, Strided Plan)
            # ==================================================================
            l_feedback = self.l_feedback_proj(l_state[:, :, 0])
            enc_with_feedback = enc + l_feedback
            
            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, timestep=t)
            h_out_real = torch.clamp(h_out_real, min=-100.0, max=100.0)
            
            step_ponder_cost = torch.tensor(0.0, device=device)
            
            # PLANNING STEP (Strided with ACT)
            if abs_t % stride == 0:
                prev_context = target_context

                # Pondering on Shadow State
                h_step_outputs = [h_out_real]
                halt_logit = self.h_halt_proj(h_out_real).squeeze(-1)
                h_halt_probs = [torch.sigmoid(halt_logit)]
                
                shadow_h_state = h_state.clone()
                current_enc_h = enc_with_feedback

                for step_idx in range(self.config.max_h_steps - 1):
                    if not self.training and h_halt_probs[-1].mean() > getattr(self.config, 'h_halt_thresh', 0.9): 
                        break
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, timestep=-(step_idx+1))
                    halt_logit = self.h_halt_proj(h_out_ponder).squeeze(-1)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(halt_logit))

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
                
                target_context = self.h_to_context(final_h_out)
                target_context = torch.clamp(target_context, min=-50.0, max=50.0)
                
                step_ponder_cost = cum_remain.sum(dim=0).mean()
                ponder_costs.append(step_ponder_cost)
            
            # LERP (Interpolation)
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            sliding_context = torch.lerp(prev_context, target_context, alpha)

            # ==================================================================
            # 4. WORKER STEP
            # ==================================================================
            if self.context_drift_proj is not None:
                prev_worker_h = l_state[:, :, 0]
                initial_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                initial_drift = torch.clamp(initial_drift, min=-5.0, max=5.0)
            else:
                initial_drift = torch.zeros(B, self.config.context_dim, device=device)

            detach_freq = getattr(self.config, 'detach_every_n_steps', 32)
            if self.training and detach_freq is not None and t > 0 and t % detach_freq == 0:
                l_state = l_state.detach()

            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                enc, l_state, cc, final_drift = checkpoint(
                    self.worker_loop_module, enc, sliding_context, l_state, initial_drift, None, 
                    use_reentrant=False
                )
            else:
                enc, l_state, cc, final_drift = self.worker_loop_module(
                    enc, sliding_context, l_state, initial_drift, timestep=None
                )
            
            final_embs.append(enc)
            commitment_costs.append(cc)

            # ==================================================================
            # 5. MEMORY UPDATE (Differentiable Hebbian)
            # ==================================================================
            if self.training:
                val_to_store = self.val_proj(enc)
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded,
                    current_lr=getattr(self.config, 'ltm_lr', 0.01),
                    timestamp=float(abs_t),
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals
                )
            else:
                val_to_store = self.val_proj(enc)
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded,
                    current_lr=getattr(self.config, 'ltm_lr', 0.01),
                    timestamp=0.0,
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals,
                    inplace=True
                )

        # ==================================================================
        # 5. FINAL OUTPUTS
        # ==================================================================
        final = self.out_norm(torch.stack(final_embs, dim=1))
        logits = self.lm_head(final)
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN/Inf detected in logits. Replacing with zeros and clamping...")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
        
        logits = torch.clamp(logits, min=-30.0, max=30.0)

        loss = None
        ponder_cost_out = None
        commitment_cost_out = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            valid_mask = shift_labels != -100
            if not valid_mask.any():
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size).float(), 
                    shift_labels.view(-1)
                )
            
            if ponder_costs: 
                ponder_cost_out = torch.stack(ponder_costs).mean()
            if commitment_costs: 
                commitment_cost_out = torch.stack(commitment_costs).mean()

        return {
            "loss": loss, 
            "logits": logits, 
            "ponder_cost": ponder_cost_out, 
            "commitment_cost": commitment_cost_out,
            "topk_vals": torch.stack(all_topk_vals, dim=1) if all_topk_vals else None, 
            "raw_topk_vals": all_topk_vals,
            "topk_idx": torch.stack(all_topk_idx, dim=1) if all_topk_idx else None,
            "h_state": h_state,
            "l_state": l_state,
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": final_drift,
            "ltm_memory_state": (curr_fast_vals, curr_mom_vals),
        }

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def update_memory(self, topk_idx, grads, timestamp, lr=1e-3):
        """Updates the LTM memory using gradients (Titans style)."""
        self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp)

    def update_memory_hebbian(self, topk_idx, vals, timestamp, lr=1e-3, tokens_covered=1):
        """Updates the LTM memory using Hebbian rule (Fallback for Inference)."""
        self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, 
                                       timestamp=timestamp, tokens_covered=tokens_covered)
