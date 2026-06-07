import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class LTMModule(nn.Module):
    """
    Titans-style Neural Memory Module with Dual-Store Architecture.
    
    MERGED VERSION (V3.2 - Time-Invariant Decay): 
    1. Retains 'Fast State' (Titans) for test-time updates.
    2. Scales forgetting rate based on token count to fix training/inference mismatch.
    """
    # SOURCE_ID definitions
    SRC_UNKNOWN = 0
    SRC_USER_INTERACTION = 1
    SRC_TRAINING_DATA = 2
    SRC_CORRECTION = 3 

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4, forget_rate=0.01, reference_chunk_len=128):
        super().__init__()
        
        # --- Slow Weights (Long-Term Consolidation) ---
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        
        # Vals represent the consolidated content. 
        vals_init = torch.empty(n_slots, val_dim)
        nn.init.orthogonal_(vals_init)
        self.vals = nn.Parameter(vals_init * 0.02)
        
        # --- Fast State (Associative Working Memory - Titans Feature) ---
        self.register_buffer("fast_vals", torch.zeros(n_slots, val_dim))
        
        # --- Metadata & Optimizer Stats ---
        self.register_buffer("_mom_vals", torch.zeros_like(self.vals.data))
        self.lr, self.momentum, self.weight_decay = lr, momentum, wd
        
        # Base forget rate per REFERENCE chunk size
        self.forget_rate = forget_rate 
        self.reference_chunk_len = reference_chunk_len

        # Buffers for tracking history context
        self.register_buffer("timestamps", torch.zeros(n_slots, dtype=torch.float32))
        self.register_buffer("sources", torch.full((n_slots,), self.SRC_UNKNOWN, dtype=torch.long))

        # Buffer for neg_inf to avoid creation in hot loop
        self.register_buffer("neg_inf", torch.tensor(-float('inf')), persistent=False)

        # Pre-allocate buffers for update calculations
        self.register_buffer("update_counts", torch.zeros(n_slots, dtype=torch.float32), persistent=False)
        self.register_buffer("update_slots", torch.zeros(n_slots, val_dim, dtype=torch.float32), persistent=False)

        # <<< NEW: Online Learning Accumulator >>>
        self.register_buffer("ltm_deltas", torch.zeros(n_slots, val_dim), persistent=False)
        self.accumulate_deltas = False

    def reset_working_memory(self):
        """Zeros out the Fast State (Working Memory) and associated momentum buffers."""
        self.fast_vals.zero_()
        self._mom_vals.zero_()
        self.timestamps.zero_()
        self.sources.fill_(self.SRC_UNKNOWN)

    def get_effective_memory(self):
        """Returns the combined memory (Slow + Fast)."""
        return self.vals + self.fast_vals

    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4, min_timestamp: float = 0.0,
                      source_filter: Optional[int] = None, fast_vals: Optional[torch.Tensor] = None,
                      timestamps: Optional[torch.Tensor] = None,
                      sources: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        # CRITICAL AMP FIX: Disable autocast for the entire retrieval path.
        # Under BFloat16 AMP, torch.matmul is in the BF16-eligible list and will
        # silently recast explicit .float() operands back to BF16. The backward pass
        # of this BF16 matmul then propagates BF16 gradients into float32 parameters
        # via masked_scatter_, which crashes with:
        #   "masked_scatter_: expected self and source to have same dtypes but got BFloat16 and Float"
        # This mirrors the autocast(enabled=False) pattern used in rwkv_cell.py for WKV stability.
        _device_type = queries.device.type if queries.device.type in ('cuda', 'cpu') else 'cpu'
        with torch.amp.autocast(device_type=_device_type, enabled=False):
            # Cast queries to float32 to ensure all downstream ops stay in float32
            queries = queries.float()

            scale_factor = self.keys.shape[1] ** -0.5
            sim = (queries @ self.keys.t()) * scale_factor

            current_timestamps = timestamps if timestamps is not None else self.timestamps
            current_sources = sources if sources is not None else self.sources
            current_timestamps = current_timestamps.to(device=queries.device)
            current_sources = current_sources.to(device=queries.device)

            if min_timestamp > 0.0 or source_filter is not None:
                with torch.no_grad():
                    valid_mask = torch.ones_like(sim, dtype=torch.bool)
                    if min_timestamp > 0.0:
                        valid_mask = valid_mask & (current_timestamps >= min_timestamp)
                    if source_filter is not None:
                        valid_mask = valid_mask & (current_sources == source_filter)

                    sim = torch.nan_to_num(sim, nan=-torch.inf, posinf=torch.finfo(sim.dtype).max, neginf=-torch.inf)
                    sim = torch.where(valid_mask, sim, self.neg_inf.to(dtype=sim.dtype))

            num_slots = self.vals.shape[0]
            native_topk = min(int(topk), num_slots)

            if native_topk <= 0:
                query_shape = list(queries.shape)
                vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
                idx_shape = query_shape[:-1] + [topk]
                
                return (torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), 
                        torch.full(idx_shape, -1, device=queries.device, dtype=torch.long), 
                        torch.zeros(idx_shape, device=queries.device, dtype=torch.float32))

            sim_for_topk = torch.nan_to_num(
                sim.detach(),
                nan=-torch.inf,
                posinf=torch.finfo(sim.dtype).max,
                neginf=-torch.inf,
            )
            topk_scores, idx = torch.topk(sim_for_topk, k=native_topk, dim=-1)
            valid_selected = torch.isfinite(topk_scores)
            
            current_fast_vals = fast_vals if fast_vals is not None else self.fast_vals
            effective_memory = self.vals + current_fast_vals
            
            idx_clamped = idx.clamp(min=0, max=num_slots - 1)

            if effective_memory.dim() == 2:
                gathered_vals = effective_memory.float().index_select(0, idx_clamped.reshape(-1))
                gathered_vals = gathered_vals.view(*idx_clamped.shape, self.vals.shape[-1])
            else:
                memory_for_gather = effective_memory.float()
                while memory_for_gather.ndim < idx_clamped.ndim + 1:
                    memory_for_gather = memory_for_gather.unsqueeze(1)
                if memory_for_gather.shape[:-2] != idx_clamped.shape[:-1]:
                    memory_for_gather = memory_for_gather.expand(*idx_clamped.shape[:-1], num_slots, self.vals.shape[-1])
                gather_idx = idx_clamped.unsqueeze(-1).expand(*idx_clamped.shape, self.vals.shape[-1])
                gathered_vals = torch.gather(memory_for_gather, dim=-2, index=gather_idx)
            
            # --- FINAL LTM FIX: Clean Concatenation Signal ---
            # With normalization handling the bleeding, we can now use the 
            # raw memories as the model expects. Softmax is no longer needed.
            weighted_vals = torch.where(
                valid_selected.unsqueeze(-1),
                gathered_vals,
                torch.zeros_like(gathered_vals),
            ).contiguous()
            
            ts_for_gather = current_timestamps.to(device=idx_clamped.device, dtype=torch.float32)
            if ts_for_gather.dim() == 1:
                ts_retrieved = ts_for_gather.index_select(0, idx_clamped.reshape(-1))
                ts_retrieved = ts_retrieved.view_as(idx_clamped)
            else:
                while ts_for_gather.ndim < idx_clamped.ndim:
                    ts_for_gather = ts_for_gather.unsqueeze(1)
                if ts_for_gather.shape[:-1] != idx_clamped.shape[:-1]:
                    ts_for_gather = ts_for_gather.expand(*idx_clamped.shape[:-1], num_slots)
                ts_retrieved = torch.gather(ts_for_gather, dim=-1, index=idx_clamped)
            ts_retrieved = torch.where(valid_selected, ts_retrieved, torch.zeros_like(ts_retrieved))

            idx = torch.where(valid_selected, idx, torch.full_like(idx, -1))
            
            if native_topk < topk:
                pad_size = topk - native_topk
                batch_shape_list = list(idx_clamped.shape[:-1])
                
                idx_pad = torch.full(batch_shape_list + [pad_size], -1, device=idx.device, dtype=idx.dtype)
                idx_ret = torch.cat([idx, idx_pad], dim=-1)
                
                vals_pad = torch.zeros(batch_shape_list + [pad_size, weighted_vals.shape[-1]], 
                                       device=weighted_vals.device, dtype=weighted_vals.dtype)
                vals_ret = torch.cat([weighted_vals, vals_pad], dim=-2)
                
                ts_pad = torch.zeros(batch_shape_list + [pad_size], 
                                   device=ts_retrieved.device, dtype=ts_retrieved.dtype)
                ts_ret = torch.cat([ts_retrieved, ts_pad], dim=-1)
                
                return vals_ret, idx_ret, ts_ret
            else:
                return weighted_vals, idx, ts_retrieved

    def inner_update(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor, current_lr: float, timestamp: float, 
                    source: int = SRC_USER_INTERACTION, tokens_covered: int = None,
                    fast_vals: Optional[torch.Tensor] = None, mom_vals: Optional[torch.Tensor] = None,
                    timestamps: Optional[torch.Tensor] = None, sources: Optional[torch.Tensor] = None,
                    inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Fast Weight Associative Update (Titans Style).
        """
        curr_fast = fast_vals if fast_vals is not None else self.fast_vals
        curr_mom = mom_vals if mom_vals is not None else self._mom_vals

        if curr_fast.dim() == 3:
            return self._inner_update_batched(
                topk_idx=topk_idx,
                grads_tensor=grads_tensor,
                current_lr=current_lr,
                timestamp=timestamp,
                source=source,
                tokens_covered=tokens_covered,
                curr_fast=curr_fast,
                curr_mom=curr_mom,
                timestamps=timestamps,
                sources=sources,
                inplace=inplace
            )
        
        if grads_tensor is None: 
            return curr_fast, curr_mom

        device = curr_fast.device
        topk_idx = topk_idx.to(device)
        valid_mask = topk_idx >= 0
        if not valid_mask.any(): 
            return curr_fast, curr_mom

        # AMP FIX: Cast grads_tensor to float32 BEFORE boolean indexing.
        # Under BFloat16 AMP, grads_tensor arrives as BFloat16 but downstream
        # computations promote to float32. The backward pass of boolean indexing
        # uses masked_scatter_ which requires self and source to have the same dtype.
        # If grads_tensor is BFloat16 but the incoming gradient is float32, it crashes.
        grads_tensor = grads_tensor.to(device=device, dtype=torch.float32)
        curr_fast = curr_fast.float()
        curr_mom = curr_mom.float()

        idx_flat = topk_idx[valid_mask].view(-1)
        grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))
        grads_flat = grads_flat.contiguous()

        num_slots = self.vals.size(0)
        counts = torch.zeros(num_slots, device=device, dtype=grads_flat.dtype)
        counts.scatter_add_(0, idx_flat, torch.ones_like(idx_flat, dtype=counts.dtype))

        slot_grads = torch.zeros(num_slots, self.vals.size(1), device=device, dtype=grads_flat.dtype)
        scatter_idx = idx_flat.unsqueeze(-1).expand(-1, self.vals.size(1))
        slot_grads.scatter_add_(0, scatter_idx, grads_flat)
        
        # <<< STABILITY FIX: div + 1e-8 for safe division >>>
        slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)
        
        nonzero_mask = counts > 0

        if inplace:
            curr_mom.mul_(self.momentum).add_(slot_grads)
            curr_mom.clamp_(min=-50.0, max=50.0)
            new_mom = curr_mom
        else:
            new_mom = self.momentum * curr_mom + slot_grads
            new_mom = torch.clamp(new_mom, min=-50.0, max=50.0)
        
        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        
        decay_scaler = tokens_covered / float(self.reference_chunk_len)
        retention_rate = (1.0 - self.forget_rate) ** decay_scaler
        
        if inplace:
            update_step = (new_mom + self.weight_decay * curr_fast) * (-current_lr)
            
            # Global decay
            curr_fast.mul_(retention_rate)
            
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(curr_fast)
            update_step = torch.where(nonzero_mask_expanded, update_step, torch.zeros_like(update_step))
            curr_fast.add_(update_step)
            curr_fast.clamp_(min=-50.0, max=50.0)
            new_fast = curr_fast

            if getattr(self, "accumulate_deltas", False):
                self.ltm_deltas.add_(update_step)
        else:
            update_delta = (new_mom + self.weight_decay * curr_fast)
            update_step = update_delta * (-current_lr)
            
            # Apply the same global decay/update rule as the trainer's batched path.
            decayed_fast = curr_fast * retention_rate
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(decayed_fast)
            
            update_mask = torch.zeros_like(decayed_fast)
            update_step = update_step.to(dtype=update_mask.dtype)
            update_mask = torch.where(nonzero_mask_expanded, update_step, update_mask)
            
            new_fast = decayed_fast + update_mask
            new_fast = torch.clamp(new_fast, min=-50.0, max=50.0)

            # Record change for LoRA accumulation if enabled
            if getattr(self, "accumulate_deltas", False):
                self.ltm_deltas.add_(update_mask)

        target_timestamps = timestamps if timestamps is not None else self.timestamps
        target_sources = sources if sources is not None else self.sources
        with torch.no_grad():
            target_timestamps.data[nonzero_mask] = timestamp
            target_sources.data[nonzero_mask] = source
            
        return new_fast, new_mom

    def _inner_update_batched(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                              current_lr: float, timestamp: float, source: int,
                              tokens_covered: int, curr_fast: torch.Tensor,
                              curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                              sources: Optional[torch.Tensor], inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        topk_idx = topk_idx.to(curr_fast.device)
        valid_mask_all = topk_idx >= 0
        if not valid_mask_all.any():
            return curr_fast, curr_mom

        grads_tensor = grads_tensor.to(device=curr_fast.device, dtype=torch.float32)
        if not inplace:
            curr_fast = curr_fast.float().clone()
            curr_mom = curr_mom.float().clone()
        else:
            curr_fast = curr_fast.float()
            curr_mom = curr_mom.float()

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        B = curr_fast.shape[0]
        num_slots = self.vals.size(0)
        val_dim = self.vals.size(1)

        idx_flat = topk_idx.reshape(B, -1).clamp(min=0, max=num_slots - 1)
        valid_flat = valid_mask_all.reshape(B, -1)
        grads_flat = grads_tensor.reshape(B, -1, val_dim)
        grads_flat = torch.where(valid_flat.unsqueeze(-1), grads_flat, torch.zeros_like(grads_flat))

        counts = torch.zeros(B, num_slots, device=curr_fast.device, dtype=grads_flat.dtype)
        counts.scatter_add_(1, idx_flat, valid_flat.to(dtype=counts.dtype))

        slot_grads = torch.zeros(B, num_slots, val_dim, device=curr_fast.device, dtype=grads_flat.dtype)
        scatter_idx = idx_flat.unsqueeze(-1).expand(-1, -1, val_dim)
        slot_grads.scatter_add_(1, scatter_idx, grads_flat)
        slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)

        nonzero_mask = counts > 0
        has_updates = valid_flat.any(dim=1).view(B, 1, 1)

        updated_mom = torch.clamp(self.momentum * curr_mom + slot_grads, min=-50.0, max=50.0)
        if inplace:
            curr_mom.copy_(torch.where(has_updates, updated_mom, curr_mom))
            new_mom = curr_mom
        else:
            new_mom = torch.where(has_updates, updated_mom, curr_mom)
            curr_mom = new_mom

        update_step = (curr_mom + self.weight_decay * curr_fast).mul(-current_lr)
        curr_fast.mul_(retention_rate)
        update_step = torch.where(nonzero_mask.unsqueeze(-1), update_step, torch.zeros_like(update_step))
        curr_fast.add_(update_step)
        curr_fast.clamp_(min=-50.0, max=50.0)

        # Record change for LoRA accumulation if enabled
        if getattr(self, "accumulate_deltas", False):
            self.ltm_deltas.add_(update_step.sum(dim=0))

        if timestamps is not None and sources is not None:
            with torch.no_grad():
                if timestamps.dim() == 1:
                    slot_mask = nonzero_mask.any(dim=0)
                    timestamps.data[slot_mask] = timestamp
                    sources.data[slot_mask] = source
                else:
                    timestamps.data[nonzero_mask] = timestamp
                    sources.data[nonzero_mask] = source

        return curr_fast, curr_mom

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, keys: torch.Tensor, vals: torch.Tensor, current_lr: float, timestamp: float, 
                              source: int = SRC_USER_INTERACTION, tokens_covered: int = None, fast_vals: Optional[torch.Tensor] = None,
                              mom_vals: Optional[torch.Tensor] = None, timestamps: Optional[torch.Tensor] = None,
                              sources: Optional[torch.Tensor] = None, inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Hebbian-style associative update, unified with Titans Momentum logic.
        """
        pseudo_grads = -vals
        return self.inner_update(
            topk_idx=topk_idx,
            grads_tensor=pseudo_grads,
            current_lr=current_lr,
            timestamp=timestamp,
            source=source,
            tokens_covered=tokens_covered,
            fast_vals=fast_vals,
            mom_vals=mom_vals,
            timestamps=timestamps,
            sources=sources,
            inplace=inplace
        )
