"""
HierarchosCore - Full parity version with original forward method.
This is a direct port from hierarchos.py to achieve exact training parity.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional
from torch.utils.checkpoint import checkpoint

from .rwkv_cell import RWKVCell
from .ltm import LTMModule
from ..utils.device import setup_msvc_environment, is_directml_device
from ..utils.rosa import ROSA, rosa_async_pipeline, ROSAState

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
                initial_drift: torch.Tensor, timestep: Optional[int] = None, l_deepemb_vec: Optional[torch.Tensor] = None):
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
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1), deepemb_vec=l_deepemb_vec)
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
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, timestep=-(step_idx+1), deepemb_vec=l_deepemb_vec)
                shadow_l_state = torch.clamp(shadow_l_state, min=-50.0, max=50.0)
                
                drift_delta = torch.tanh(self.context_drift_proj(l_out))
                current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                drift_sq = torch.sum(current_drift ** 2, dim=-1)
                hinge_cost = torch.relu(drift_sq - self.commitment_threshold)
                hinge_cost = torch.clamp(hinge_cost, max=100.0)
                drift_costs.append(hinge_cost)
                
                # Update dynamic_context BEFORE convergence check (matches inference path)
                dynamic_context = static_context + current_drift
                l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
                l_input = self.l_input_proj(l_input_vec)
                
                if torch.mean(torch.abs(drift_delta)) < self.l_conv_atol:
                   break

        # Use original l_state (not shadow) for the actual state update
        ts = timestep if timestep is not None else 0
        
        # Recalculate l_input with final drift
        dynamic_context = static_context + current_drift
        l_input_vec = torch.cat([current_enc, dynamic_context], dim=-1)
        l_input = self.l_input_proj(l_input_vec)
        
        final_l_out, next_l_state = self.l_rnn(l_input, l_state, timestep=ts, deepemb_vec=l_deepemb_vec)
        next_l_state = torch.clamp(next_l_state, min=-50.0, max=50.0)
        
        final_enc = current_enc + self.l_to_out(final_l_out)
        commitment_cost = torch.zeros(enc.shape[0], device=enc.device, dtype=enc.dtype)
        if len(drift_costs) > 0:
            commitment_cost = torch.stack(drift_costs, dim=0).mean(dim=0)

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
        if not torch.cuda.is_available():
            torch.set_flush_denormal(True)
            
        self.tok_emb = nn.Embedding(config.vocab_size, config.context_dim)
        
        # V8 DeepEmbed (4x scale) - default to True
        self.use_deepembed = getattr(config, 'use_deepembed', True)
        if self.use_deepembed:
            self.h_deepemb = nn.Embedding(config.vocab_size, config.h_hidden * 4)
            self.l_deepemb = nn.Embedding(config.vocab_size, config.l_hidden * 4)
            nn.init.ones_(self.h_deepemb.weight)
            nn.init.ones_(self.l_deepemb.weight)
        
        # V8 ROSA Embed - default to True
        self.use_rosa = getattr(config, 'use_rosa', True)
        if self.use_rosa:
            self.rosa_emb = nn.Embedding(config.vocab_size + 1, config.context_dim)
            nn.init.zeros_(self.rosa_emb.weight)
            # Learnable gate: sigmoid(-1.0) ≈ 0.27 initial injection strength
            self.rosa_gate_logit = nn.Parameter(torch.tensor(-1.0))
        
        # Global Learnable State
        self.persistent_dim = getattr(config, 'persistent_dim', 128)
        self.persistent = nn.Parameter(torch.randn(self.persistent_dim) * 0.02)
        
        # Learnable LTM Gate
        self.ltm_gate_logit = nn.Parameter(torch.tensor(-2.0))

        # LTM System
        self.ltm = LTMModule(
            n_slots=config.ltm_slots, 
            key_dim=config.ltm_key_dim, 
            val_dim=config.ltm_val_dim,
            lr=getattr(config, 'ltm_lr', 1e-3),
            momentum=getattr(config, 'ltm_momentum', 0.9),
            wd=getattr(config, 'ltm_weight_decay', 1e-4),
            forget_rate=getattr(config, 'ltm_forget_rate', 0.01),
            reference_chunk_len=getattr(config, 'reference_chunk_len', getattr(config, 'training_chunk_size', 128))
        )
        self.qproj = nn.Linear(config.context_dim * 2, config.ltm_key_dim, bias=False)
        self.val_proj = nn.Linear(config.context_dim, config.ltm_val_dim, bias=False)
        
        # Encoder Projection
        in_dim = config.context_dim + self.persistent_dim + config.ltm_val_dim * config.ltm_topk
        self.in_proj = nn.Linear(in_dim, config.context_dim)
        
        # Manager Components
        self.l_feedback_proj = nn.Linear(config.l_hidden, config.h_hidden, bias=False)
        # Initialize with small weights to introduce feedback gradually
        nn.init.normal_(self.l_feedback_proj.weight, mean=0.0, std=0.01)

        self.h_rnn = RWKVCell(config.h_hidden)
        self.h_to_context = nn.Linear(config.h_hidden, config.context_dim)
        self.h_halt_proj = nn.Linear(config.h_hidden, 1)
        # Initialize bias to encourage max_h_steps pondering steps initially
        # Formula: logit(1/N) = -log(N-1). This sets initial halt prob to 1/max_h_steps.
        with torch.no_grad():
            initial_steps = max(2.0, float(config.max_h_steps))
            initial_bias = -math.log(initial_steps - 1.0)
            self.h_halt_proj.bias.fill_(initial_bias)
        
        # Worker Components
        self.l_input_proj = nn.Linear(config.context_dim * 2, config.l_hidden)
        self.l_rnn = RWKVCell(config.l_hidden)
        
        # Configure truncated BPTT for RWKV cells
        detach_freq = getattr(config, 'detach_every_n_steps', 32)
        self.h_rnn.detach_every_n_steps = detach_freq
        self.l_rnn.detach_every_n_steps = detach_freq

        self.context_drift_proj = nn.Linear(config.l_hidden, config.context_dim, bias=False)
        nn.init.normal_(self.context_drift_proj.weight, mean=0.0, std=0.01)

        self.l_to_out = nn.Linear(config.l_hidden, config.context_dim)
        
        # Output Head
        self.out_norm = nn.LayerNorm(config.context_dim)
        self.lm_head = nn.Linear(config.context_dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight  # Weight tying
        
        # Worker Loop Wrapper - pass actual module references
        self.worker_loop_module = WorkerLoop(config, self.l_rnn, self.l_input_proj, 
                                             self.context_drift_proj, self.l_to_out)
        
        # Sinusoidal Encoding for Timestamps
        half_dim = config.ltm_val_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        self.register_buffer('time_freqs', emb)

    def compile(self):
        """Applies torch.compile to the worker loop if enabled in config (Robust Parity)."""
        if not getattr(self.config, 'compile', False):
            return
        
        device = next(self.parameters()).device
        device_type = 'cpu'
        if device.type == 'cuda': device_type = 'cuda'
        elif is_directml_device(device): device_type = 'dml'

        # Check for DirectML (doesn't support torch.compile)
        if device_type == 'dml':
            print("INFO: DirectML detected - torch.compile is not supported. Using eager mode.")
            self.config.compile = False
            return

        # Check for Windows CPU + Compile (Known Hang Issue)
        if os.name == 'nt' and device_type == 'cpu' and not getattr(self.config, 'force_compile', False):
            print("WARNING: torch.compile on Windows CPU is known to hang with complex RNN loops.")
            print("         Disabling compilation for stability. Use force_compile=True to override.")
            self.config.compile = False
            return

        try:
            if hasattr(torch, "compile"):
                print("INFO: Compiling WorkerLoop...")
                if os.name == 'nt' and device_type != 'dml':
                    setup_msvc_environment()

                import torch._dynamo as dynamo
                dynamo.config.suppress_errors = True
                
                compile_options = {"triton.cudagraphs": False} if device_type == 'cuda' else {}
                
                self.worker_loop_module = torch.compile(
                    self.worker_loop_module, 
                    dynamic=True,
                    fullgraph=False,
                    options=compile_options
                )
                print("INFO: WorkerLoop compiled successfully.")
        except Exception as e:
            print(f"Warning: Compilation failed! Falling back to eager mode. {e}")
            self.config.compile = False

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
        allow_hebbian_update = kwargs.pop("allow_hebbian_update", False)
        return_logits = kwargs.pop("return_logits", True)
        return_topk_values = kwargs.pop("return_topk_values", True)
        suppress_hebbian = kwargs.pop("suppress_hebbian", getattr(self, "suppress_hebbian", True))
        if allow_hebbian_update:
            suppress_hebbian = False

        x = self.tok_emb(input_ids)

        # Unpack LTM Memory State early so we can use past_tokens + ROSA states
        rosa_states = None
        memory_timestamps = None
        memory_sources = None
        if ltm_memory_state is None:
            isolate_batch_ltm = getattr(self.config, 'isolate_batch_ltm', True)
            isolate_runtime_ltm = isolate_batch_ltm and (self.training or B > 1)
            if isolate_runtime_ltm:
                curr_fast_vals = self.ltm.fast_vals.unsqueeze(0).expand(B, -1, -1).clone()
                curr_mom_vals = self.ltm._mom_vals.unsqueeze(0).expand(B, -1, -1).clone()
                memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(B, -1).clone()
                memory_sources = self.ltm.sources.unsqueeze(0).expand(B, -1).clone()
            else:
                curr_fast_vals = self.ltm.fast_vals
                curr_mom_vals = self.ltm._mom_vals
                memory_timestamps = self.ltm.timestamps
                memory_sources = self.ltm.sources
            past_tokens = None
        else:
            if len(ltm_memory_state) >= 6:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states, memory_timestamps, memory_sources = ltm_memory_state[:6]
            elif len(ltm_memory_state) >= 4:
                curr_fast_vals, curr_mom_vals, past_tokens, rosa_states = ltm_memory_state[:4]
            elif len(ltm_memory_state) >= 3:
                curr_fast_vals, curr_mom_vals, past_tokens = ltm_memory_state[:3]
            else:
                curr_fast_vals, curr_mom_vals = ltm_memory_state
                past_tokens = None
            if memory_timestamps is None:
                if curr_fast_vals.dim() == 3:
                    memory_timestamps = self.ltm.timestamps.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                    memory_sources = self.ltm.sources.unsqueeze(0).expand(curr_fast_vals.shape[0], -1).clone()
                else:
                    memory_timestamps = self.ltm.timestamps
                    memory_sources = self.ltm.sources

        # V8 ROSA Precomputation (only when enabled)
        new_rosa_states = None
        if self.use_rosa:
            rosa_max_ctx = getattr(self.config, 'rosa_max_context', 512)

            # --- Datacenter-Optimized Async ROSA Pipeline ---
            # Launch CPU suffix automaton work immediately (overlaps with GPU tok_emb)
            # Uses: Numba JIT, parallel batch threads, pinned memory, CUDA streams,
            #       and persistent automaton state across TBPTT chunks
            rosa_finalize = rosa_async_pipeline(
                input_ids=input_ids,
                past_tokens=past_tokens,
                rosa_states=rosa_states,
                vocab_size=self.config.vocab_size,
                device=device,
                rosa_max_ctx=rosa_max_ctx,
            )

            # Finalize: wait for CPU work, async H2D transfer
            rosa_batch_tensor, rosa_input, new_rosa_states = rosa_finalize()

            rosa_embs = self.rosa_emb(rosa_batch_tensor)
            # Learnable injection gate (controls ROSA signal strength)
            rosa_gate = torch.sigmoid(torch.clamp(self.rosa_gate_logit, min=-50.0, max=50.0))
            x = x + rosa_gate * rosa_embs  # Gated Neurosymbolic Inner Monologue Mix

            # Store past_tokens for cross-chunk continuity (capped by rosa_max_ctx)
            # Detach and move to CPU immediately to prevent GPU memory leak across chunks
            new_past_tokens = rosa_input.detach().cpu()
        else:
            new_past_tokens = None


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

        # (ltm_memory_state already unpacked above)
        if ltm_memory_state is not None:
            # Ensure they are on the correct device
            curr_fast_vals = curr_fast_vals.to(device)
            curr_mom_vals = curr_mom_vals.to(device)
            memory_timestamps = memory_timestamps.to(device)
            memory_sources = memory_sources.to(device)

        final_embs = []
        ponder_costs = []
        ponder_weights = []
        commitment_costs = []
        commitment_weights = []
        all_topk_vals = []
        all_topk_idx = []
        aux_attention_mask = attention_mask.to(device=device, dtype=torch.float32) if attention_mask is not None else None

        stride = self.config.h_stride
        final_drift = None

        # ==================================================================
        # 2. MAIN TIME LOOP
        # ==================================================================
        for t in range(T):
            token_x_idx = input_ids[:, t]
            token_x = x[:, t]
            abs_t = global_pos_offset + t
            
            h_deepemb_vec = self.h_deepemb(token_x_idx) if self.use_deepembed else None
            l_deepemb_vec = self.l_deepemb(token_x_idx) if self.use_deepembed else None
            
            # --- LTM Retrieval ---
            p = self.persistent.unsqueeze(0).expand(B, -1)
            q_in = torch.cat([token_x, prev_context], dim=-1)
            q = torch.clamp(self.qproj(q_in), min=-12, max=12)
            
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(
                q, self.config.ltm_topk, min_timestamp, source_filter, fast_vals=curr_fast_vals,
                timestamps=memory_timestamps, sources=memory_sources
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
            l_feedback = self.l_feedback_proj(l_state[:, :, 0].to(device))
            enc_with_feedback = enc + l_feedback
            
            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, timestep=t, deepemb_vec=h_deepemb_vec)
            h_out_real = torch.clamp(h_out_real, min=-100.0, max=100.0)
            
            if getattr(self.config, 'debug_numerics', False) and (torch.isnan(h_out_real).any() or torch.isinf(h_out_real).any()):
                print(f"WARNING: NaN/Inf detected in h_out_real at step {t}")
            
            step_ponder_cost = torch.zeros(B, device=device, dtype=enc.dtype)
            
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
                    h_out_ponder, shadow_h_state = self.h_rnn(current_enc_h, shadow_h_state, timestep=-(step_idx+1), deepemb_vec=h_deepemb_vec)
                    halt_logit = self.h_halt_proj(h_out_ponder).squeeze(-1)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(halt_logit))

                # BUG #4 FIX: Force ACT weighting to float32 for numerical stability.
                # BFloat16's limited precision (~7 bits) causes underflow in cumprod chains,
                # leading to NaN weights. Mirrors the autocast(enabled=False) pattern in rwkv_cell.py.
                h_stack = torch.stack(h_step_outputs, dim=0).float()
                halt_stack = torch.stack(h_halt_probs, dim=0).float()
                remain = 1.0 - halt_stack
                remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
                cum_remain = torch.cumprod(remain_shifted, dim=0)
                
                weights = halt_stack * cum_remain
                remainder = cum_remain[-1] * (1.0 - halt_stack[-1])
                total = weights.sum(dim=0) + remainder + 1e-8
                weights = weights / total.unsqueeze(0)
                remainder = remainder / total
                final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]
                final_h_out = final_h_out.to(enc.dtype)  # Cast back to working precision
                
                target_context = self.h_to_context(final_h_out)
                target_context = torch.clamp(target_context, min=-50.0, max=50.0)
                
                step_ponder_cost = cum_remain.sum(dim=0).to(enc.dtype)
                ponder_costs.append(step_ponder_cost)
                if aux_attention_mask is not None:
                    ponder_weights.append(aux_attention_mask[:, t])
                else:
                    ponder_weights.append(torch.ones(B, device=device, dtype=torch.float32))
            
            # LERP (Interpolation)
            step_in_stride = abs_t % stride
            alpha = step_in_stride / float(stride)
            sliding_context = (prev_context.float() + alpha * (target_context.float() - prev_context.float())).to(prev_context.dtype)

            # ==================================================================
            # 4. WORKER STEP
            # ==================================================================
            if self.context_drift_proj is not None:
                prev_worker_h = l_state[:, :, 0].to(device)
                initial_drift = torch.tanh(self.context_drift_proj(prev_worker_h))
                initial_drift = torch.clamp(initial_drift, min=-5.0, max=5.0)
            else:
                initial_drift = torch.zeros(B, self.config.context_dim, device=device)

            detach_freq = getattr(self.config, 'detach_every_n_steps', 32)
            if self.training and detach_freq is not None and t > 0 and t % detach_freq == 0:
                l_state = l_state.detach()

            if getattr(self.config, 'gradient_checkpointing', False) and self.training:
                enc, l_state, cc, final_drift = checkpoint(
                    self.worker_loop_module, enc, sliding_context, l_state, initial_drift, None, l_deepemb_vec,
                    use_reentrant=False
                )
            else:
                enc, l_state, cc, final_drift = self.worker_loop_module(
                    enc, sliding_context, l_state, initial_drift, timestep=None, l_deepemb_vec=l_deepemb_vec
                )
            
            final_embs.append(enc)
            commitment_costs.append(cc)
            if aux_attention_mask is not None:
                commitment_weights.append(aux_attention_mask[:, t])
            else:
                commitment_weights.append(torch.ones(B, device=device, dtype=torch.float32))

            # ==================================================================
            # 5. MEMORY UPDATE (Differentiable Hebbian — Inference Only)
            # ==================================================================
            # During training, the trainer handles LTM updates via gradient-based
            # Titans inner_update after backward(). Running Hebbian here too would
            # cause double-decay and conflicting update signals on the same slots.
            if not self.training and not suppress_hebbian:
                val_to_store = self.val_proj(enc)
                val_to_store = torch.clamp(val_to_store, min=-20.0, max=20.0)
                val_expanded = val_to_store.unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                
                curr_fast_vals, curr_mom_vals = self.ltm.update_memory_hebbian(
                    topk_idx, None, val_expanded,
                    current_lr=getattr(self.config, 'ltm_lr', 0.001),  # COH #1: Use config LR, not hardcoded 0.01
                    timestamp=0.0,
                    tokens_covered=1,
                    fast_vals=curr_fast_vals,
                    mom_vals=curr_mom_vals,
                    timestamps=memory_timestamps,
                    sources=memory_sources,
                    inplace=True
                )

        # ==================================================================
        # 5. FINAL OUTPUTS
        # ==================================================================
        final = self.out_norm(torch.stack(final_embs, dim=1))
        logits = None

        loss = None
        ponder_cost_out = None
        commitment_cost_out = None

        if labels is not None and not return_logits:
            loss = self._compute_cuda_chunked_lm_loss(
                final,
                labels,
                getattr(self.config, 'z_loss_weight', 1e-4),
            )
        else:
            logits = self.lm_head(final)
            
            if getattr(self.config, 'debug_numerics', False) and (torch.isnan(logits).any() or torch.isinf(logits).any()):
                print("WARNING: NaN/Inf detected in logits. Replacing with zeros and clamping...")
                logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0)
            
            logits = torch.clamp(logits, min=-30.0, max=30.0)

            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                
                valid_mask = shift_labels != -100
                if not valid_mask.any():
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    flat_logits = shift_logits.view(-1, self.config.vocab_size).float()
                    flat_labels = shift_labels.view(-1)
                    
                    # Base CE Loss computes natively ignoring -100
                    loss = F.cross_entropy(flat_logits, flat_labels)
                    
                    # Z-Loss Regularization built to prevent exploding logits
                    z_loss_weight = getattr(self.config, 'z_loss_weight', 1e-4)
                    if z_loss_weight > 0:
                        # AMP FIX: Disable autocast for the z-loss block. Boolean indexing
                        # (flat_logits[valid_mask_flat]) uses masked_scatter_ in its backward
                        # pass. Under BFloat16 AMP, logsumexp can produce BF16 gradients that
                        # flow back into the float32 flat_logits via masked_scatter_, crashing
                        # with "expected self and source to have same dtypes".
                        _zloss_device = device.type if device.type in ('cuda', 'cpu') else 'cpu'
                        with torch.amp.autocast(device_type=_zloss_device, enabled=False):
                            valid_mask_flat = flat_labels != -100
                            valid_logits = flat_logits[valid_mask_flat]
                            z_loss = torch.logsumexp(valid_logits, dim=-1).pow(2).mean() * z_loss_weight
                        loss = loss + z_loss

        if labels is not None:
            
            # Compute auxiliary costs for reporting (trainer handles loss composition)
            def _weighted_aux_mean(costs, weights):
                if not costs:
                    return None
                cost_tensor = torch.stack([c.float().view(B) for c in costs], dim=0)
                weight_tensor = torch.stack(weights, dim=0).float()
                denom = weight_tensor.sum()
                if denom <= 0:
                    return torch.zeros((), device=device, dtype=cost_tensor.dtype)
                return (cost_tensor * weight_tensor).sum() / denom

            ponder_cost_out = _weighted_aux_mean(ponder_costs, ponder_weights)
            commitment_cost_out = _weighted_aux_mean(commitment_costs, commitment_weights)

        return {
            "loss": loss, 
            "logits": logits, 
            "ponder_cost": ponder_cost_out, 
            "commitment_cost": commitment_cost_out,
            "topk_vals": torch.stack(all_topk_vals, dim=1) if (return_topk_values and all_topk_vals) else None, 
            "raw_topk_vals": all_topk_vals,
            "topk_idx": torch.stack(all_topk_idx, dim=1) if all_topk_idx else None,
            "h_state": h_state,
            "l_state": l_state,
            "prev_context": prev_context,
            "target_context": target_context,
            "drift_state": final_drift,
            "ltm_memory_state": (curr_fast_vals, curr_mom_vals, new_past_tokens, new_rosa_states, memory_timestamps, memory_sources),
        }

    def _compute_cuda_chunked_lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor,
                                      z_loss_weight: float = 1e-4) -> torch.Tensor:
        """
        Memory-friendly CUDA loss path for large vocabularies.

        This intentionally recomputes lm_head by row chunks instead of materializing
        the full shifted logits tensor for loss calculation. The reduction matches
        PyTorch's mean cross-entropy with ignore_index=-100, and the z-loss is
        averaged over the same valid-token rows as the dense path.
        """
        shift_hidden = hidden[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_hidden = shift_hidden.view(-1, hidden.shape[-1])
        flat_labels = shift_labels.view(-1)

        valid_mask = flat_labels != -100
        valid_count = valid_mask.sum()
        denom = valid_count.clamp_min(1).to(dtype=torch.float32)

        chunk_rows = int(getattr(self.config, "cuda_loss_chunk_rows", 0) or 0)
        if chunk_rows <= 0:
            chunk_rows = flat_hidden.shape[0]

        total_ce = torch.zeros((), device=hidden.device, dtype=torch.float32)
        total_z = torch.zeros((), device=hidden.device, dtype=torch.float32)

        for start in range(0, flat_hidden.shape[0], chunk_rows):
            end = min(start + chunk_rows, flat_hidden.shape[0])
            chunk_hidden = flat_hidden[start:end]
            chunk_labels = flat_labels[start:end]
            chunk_logits = torch.clamp(self.lm_head(chunk_hidden), min=-30.0, max=30.0).float()

            total_ce = total_ce + F.cross_entropy(chunk_logits, chunk_labels, reduction="sum")

            if z_loss_weight > 0:
                row_z = torch.logsumexp(chunk_logits, dim=-1).pow(2)
                total_z = total_z + (row_z * (chunk_labels != -100).to(dtype=row_z.dtype)).sum()

        loss = total_ce / denom
        if z_loss_weight > 0:
            loss = loss + (total_z / denom) * z_loss_weight
        return loss

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def update_memory(self, topk_idx, grads, timestamp, lr=1e-3):
        """Updates the LTM memory using gradients (Titans style)."""
        self.ltm.inner_update(topk_idx, grads, current_lr=lr, timestamp=timestamp, inplace=True)

    def update_memory_hebbian(self, topk_idx, vals, timestamp, lr=1e-3, tokens_covered=1):
        """Updates the LTM memory using Hebbian rule (Fallback for Inference)."""
        self.ltm.update_memory_hebbian(topk_idx, None, vals, current_lr=lr, 
                                       timestamp=timestamp, tokens_covered=tokens_covered, inplace=True)
