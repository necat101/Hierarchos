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
        device = self.fast_vals.device
        zeros_vals = torch.zeros(self.fast_vals.shape, dtype=self.fast_vals.dtype, device='cpu').to(device)
        self.fast_vals.copy_(zeros_vals)
        self._mom_vals.copy_(zeros_vals)
        
        zeros_ts = torch.zeros(self.timestamps.shape, dtype=self.timestamps.dtype, device='cpu').to(device)
        self.timestamps.copy_(zeros_ts)
        
        sources_init = torch.full(self.sources.shape, self.SRC_UNKNOWN, dtype=self.sources.dtype, device='cpu').to(device)
        self.sources.copy_(sources_init)

    def get_effective_memory(self):
        """Returns the combined memory (Slow + Fast)."""
        return self.vals + self.fast_vals

    def retrieve_topk(self, queries: torch.Tensor, topk: int = 4, min_timestamp: float = 0.0, source_filter: Optional[int] = None, fast_vals: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
        scale_factor = self.keys.shape[1] ** -0.5
        sim = (queries @ self.keys.t()) * scale_factor

        if min_timestamp > 0.0 or source_filter is not None:
            with torch.no_grad():
                valid_mask = torch.ones(self.keys.size(0), dtype=torch.bool, device=self.keys.device)
                if min_timestamp > 0.0:
                    valid_mask &= (self.timestamps >= min_timestamp)
                if source_filter is not None:
                    valid_mask &= (self.sources == source_filter)

                sim = torch.nan_to_num(sim, nan=-torch.inf, posinf=torch.finfo(sim.dtype).max, neginf=-torch.inf)
                sim = torch.where(valid_mask, sim, self.neg_inf)

        num_valid_slots_per_query = sim.isfinite().sum(dim=-1)
        num_valid_slots = num_valid_slots_per_query.min().item()
        effective_topk = min(topk, int(num_valid_slots))

        if effective_topk <= 0:
            query_shape = list(queries.shape)
            vals_shape = query_shape[:-1] + [topk, self.vals.shape[-1]]
            idx_shape = query_shape[:-1] + [topk]
            
            return (torch.zeros(vals_shape, device=queries.device, dtype=self.vals.dtype), 
                    torch.full(idx_shape, -1, device=queries.device, dtype=torch.long), 
                    torch.zeros(idx_shape, device=queries.device, dtype=torch.float32))

        sim_detached = sim.detach()
        _, idx = torch.topk(sim_detached, k=effective_topk, dim=-1)
        
        current_fast_vals = fast_vals if fast_vals is not None else self.fast_vals
        effective_memory = self.vals + current_fast_vals
        
        effective_memory_size = self.vals.shape[0]
        idx_clamped = idx.clamp(min=0, max=effective_memory_size-1)
        
        num_classes = effective_memory_size
        range_tensor = torch.arange(num_classes, device=idx_clamped.device)
        view_shape = [1] * idx.ndim + [-1]
        range_tensor = range_tensor.view(*view_shape)
        
        one_hot = (idx_clamped.unsqueeze(-1) == range_tensor).to(dtype=sim.dtype)
        
        sim_expanded = sim.unsqueeze(-2)
        sim_topk = (one_hot * sim_expanded).sum(dim=-1)
        
        gathered_vals = torch.matmul(one_hot, effective_memory)
        attn_weights = F.softmax(sim_topk, dim=-1)
        
        if torch.isnan(attn_weights).any():
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        weighted_vals = gathered_vals * attn_weights.unsqueeze(-1)
        weighted_vals = weighted_vals.contiguous()
        
        flat_ts = torch.matmul(one_hot, self.timestamps.unsqueeze(1)).squeeze(-1)
        ts_retrieved = flat_ts

        if effective_topk < topk:
            pad_size = topk - effective_topk
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
                    inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Fast Weight Associative Update (Titans Style).
        """
        curr_fast = fast_vals if fast_vals is not None else self.fast_vals
        curr_mom = mom_vals if mom_vals is not None else self._mom_vals
        
        if grads_tensor is None: 
            return curr_fast, curr_mom

        device = self.vals.device
        valid_mask = topk_idx >= 0
        if not valid_mask.any(): 
            return curr_fast, curr_mom

        idx_flat = topk_idx[valid_mask].view(-1)
        grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))
        grads_flat = grads_flat.contiguous()

        num_classes = self.vals.size(0)
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0)
        one_hot = (idx_flat.unsqueeze(1) == range_tensor).to(dtype=grads_flat.dtype)
        
        torch.sum(one_hot, dim=0, out=self.update_counts)
        counts = self.update_counts
        slot_grads = torch.matmul(one_hot.t(), grads_flat.to(device))
        
        # <<< STABILITY FIX: div + 1e-8 for safe division >>>
        slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)
        
        nonzero_mask = counts > 0

        # TITANS UPDATE LOGIC
        if inplace:
            curr_mom.mul_(self.momentum).add_(slot_grads)
            curr_mom.clamp_(min=-50.0, max=50.0)
            new_mom = curr_mom
        else:
            new_mom = curr_mom * self.momentum + slot_grads
            new_mom = torch.clamp(new_mom, min=-50.0, max=50.0)
        
        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        
        decay_scaler = tokens_covered / float(self.reference_chunk_len)
        retention_rate = (1.0 - self.forget_rate) ** decay_scaler
        
        if inplace:
            self.update_slots.copy_(new_mom)
            self.update_slots.add_(curr_fast, alpha=self.weight_decay)
            self.update_slots.mul_(-current_lr)
            
            curr_fast.mul_(retention_rate)
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(curr_fast)
            self.update_slots.masked_fill_(~nonzero_mask_expanded, 0.0)
            
            curr_fast.add_(self.update_slots)
            curr_fast.clamp_(min=-20.0, max=20.0)
            new_fast = curr_fast
        else:
            update_delta = (new_mom + self.weight_decay * curr_fast)
            update_step = update_delta * (-current_lr)
            decayed_fast = curr_fast * retention_rate
            
            update_mask = torch.zeros_like(decayed_fast)
            nonzero_mask_expanded = nonzero_mask.unsqueeze(-1).expand_as(update_mask)
            update_mask = torch.where(nonzero_mask_expanded, update_step, update_mask)
            
            new_fast = decayed_fast + update_mask
            new_fast = torch.clamp(new_fast, min=-20.0, max=20.0)

        with torch.no_grad():
            self.timestamps.data[nonzero_mask] = timestamp
            self.sources.data[nonzero_mask] = source
            
        return new_fast, new_mom

    def update_memory_hebbian(self, topk_idx: torch.LongTensor, keys: torch.Tensor, vals: torch.Tensor, current_lr: float, timestamp: float, 
                              source: int = SRC_USER_INTERACTION, tokens_covered: int = None, fast_vals: Optional[torch.Tensor] = None,
                              mom_vals: Optional[torch.Tensor] = None, inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
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
            inplace=inplace
        )
