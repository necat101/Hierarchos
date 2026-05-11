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

    def __init__(self, n_slots=1024, key_dim=64, val_dim=64, lr=1e-3, momentum=0.9, wd=1e-4, forget_rate=0.01, reference_chunk_len=128, score_grad_scale=1.0):
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
        self.score_grad_scale = float(score_grad_scale)

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
        # Internal architecture switch: CUDA uses gather/scatter-heavy math,
        # CPU keeps the dense one-hot path that benchmarks well on small tensors.
        self._cuda_friendly_math = True

    def reset_working_memory(self):
        """Zeros out the Fast State (Working Memory) and associated momentum buffers."""
        self.fast_vals.zero_()
        self._mom_vals.zero_()
        self.timestamps.zero_()
        self.sources.fill_(self.SRC_UNKNOWN)

    def get_effective_memory(self):
        """Returns the combined memory (Slow + Fast)."""
        return self.vals + self.fast_vals

    def _use_cuda_math(self, tensor: torch.Tensor) -> bool:
        return bool(self._cuda_friendly_math and tensor.device.type == "cuda")

    @staticmethod
    def _expand_slot_tensor_for_index(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Broadcast a (..., slots, value_dim) tensor to index leading dims."""
        if tensor.dim() == 2:
            return tensor
        index_leading = tuple(index.shape[:-1])
        tensor_leading = tuple(tensor.shape[:-2])
        if len(tensor_leading) > len(index_leading):
            raise ValueError("LTM memory has more leading dimensions than top-k indices")
        view_shape = tensor_leading + (1,) * (len(index_leading) - len(tensor_leading)) + tuple(tensor.shape[-2:])
        expand_shape = index_leading + tuple(tensor.shape[-2:])
        return tensor.reshape(view_shape).expand(expand_shape)

    @staticmethod
    def _expand_slot_metadata_for_index(tensor: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Broadcast a (..., slots) metadata tensor to index leading dims."""
        if tensor.dim() == 1:
            return tensor
        index_leading = tuple(index.shape[:-1])
        tensor_leading = tuple(tensor.shape[:-1])
        if len(tensor_leading) > len(index_leading):
            raise ValueError("LTM metadata has more leading dimensions than top-k indices")
        view_shape = tensor_leading + (1,) * (len(index_leading) - len(tensor_leading)) + (tensor.shape[-1],)
        expand_shape = index_leading + (tensor.shape[-1],)
        return tensor.reshape(view_shape).expand(expand_shape)

    def _gather_topk_cuda(self, idx_clamped: torch.LongTensor, effective_memory: torch.Tensor,
                          current_timestamps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        effective_memory = effective_memory.float()
        if effective_memory.dim() == 2:
            flat_idx = idx_clamped.reshape(-1)
            gathered_vals = effective_memory.index_select(0, flat_idx)
            gathered_vals = gathered_vals.view(*idx_clamped.shape, effective_memory.shape[-1])
        else:
            memory_view = self._expand_slot_tensor_for_index(effective_memory, idx_clamped)
            gather_idx = idx_clamped.unsqueeze(-1).expand(*idx_clamped.shape, memory_view.shape[-1])
            gathered_vals = torch.gather(memory_view, dim=-2, index=gather_idx)

        current_timestamps = current_timestamps.to(device=idx_clamped.device, dtype=torch.float32)
        if current_timestamps.dim() == 1:
            ts_retrieved = current_timestamps.index_select(0, idx_clamped.reshape(-1))
            ts_retrieved = ts_retrieved.view_as(idx_clamped)
        else:
            ts_view = self._expand_slot_metadata_for_index(current_timestamps, idx_clamped)
            ts_retrieved = torch.gather(ts_view, dim=-1, index=idx_clamped)

        return gathered_vals.contiguous(), ts_retrieved

    def _inject_score_gradients(self, gathered_vals: torch.Tensor, selected_sim: torch.Tensor) -> torch.Tensor:
        """Give the address path gradients without changing retrieved values."""
        scale = getattr(self, "score_grad_scale", 1.0)
        if scale == 0.0 or not torch.is_grad_enabled() or not selected_sim.requires_grad:
            return gathered_vals
        score_signal = (selected_sim - selected_sim.detach()).unsqueeze(-1)
        return gathered_vals + score_signal.to(dtype=gathered_vals.dtype) * gathered_vals.detach() * scale

    def _inner_update_flat_cuda(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                                current_lr: float, timestamp: float, source: int,
                                tokens_covered: int, curr_fast: torch.Tensor,
                                curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                                sources: Optional[torch.Tensor],
                                inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        device = curr_fast.device
        topk_idx = topk_idx.to(device=device, dtype=torch.long)
        grads_tensor = grads_tensor.to(device=device).float()
        valid_mask = topk_idx >= 0
        if not valid_mask.any():
            return curr_fast, curr_mom

        curr_fast = curr_fast.float() if inplace else curr_fast.float().clone()
        curr_mom = curr_mom.float() if inplace else curr_mom.float().clone()

        idx_flat = topk_idx[valid_mask].reshape(-1)
        grads_flat = grads_tensor[valid_mask].reshape(-1, self.vals.size(1)).contiguous()

        num_slots = self.vals.size(0)
        unique_idx, inverse, counts = torch.unique(
            idx_flat,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        slot_grads = grads_flat.new_zeros((unique_idx.numel(), self.vals.size(1)))
        slot_grads.index_add_(0, inverse, grads_flat)
        slot_grads = slot_grads / (counts.to(dtype=torch.float32).unsqueeze(-1) + 1e-8)

        curr_mom.mul_(self.momentum)
        touched_mom = curr_mom.index_select(0, unique_idx).add(slot_grads)
        touched_mom.clamp_(min=-50.0, max=50.0)
        curr_mom.index_copy_(0, unique_idx, touched_mom)

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        touched_fast = curr_fast.index_select(0, unique_idx)
        update_step = (touched_mom + self.weight_decay * touched_fast).mul(-current_lr)
        curr_fast.mul_(retention_rate)
        touched_fast = touched_fast.mul(retention_rate).add(update_step)
        touched_fast.clamp_(min=-50.0, max=50.0)
        curr_fast.index_copy_(0, unique_idx, touched_fast)

        if getattr(self, "accumulate_deltas", False):
            self.ltm_deltas.index_add_(0, unique_idx, update_step)

        target_timestamps = timestamps if timestamps is not None else self.timestamps
        target_sources = sources if sources is not None else self.sources
        with torch.no_grad():
            target_timestamps = target_timestamps.to(device=device)
            target_sources = target_sources.to(device=device)
            target_timestamps.index_fill_(0, unique_idx, float(timestamp))
            target_sources.index_fill_(0, unique_idx, int(source))

        return curr_fast, curr_mom

    def _inner_update_batched_cuda(self, topk_idx: torch.LongTensor, grads_tensor: torch.Tensor,
                                   current_lr: float, timestamp: float, source: int,
                                   tokens_covered: int, curr_fast: torch.Tensor,
                                   curr_mom: torch.Tensor, timestamps: Optional[torch.Tensor],
                                   sources: Optional[torch.Tensor],
                                   inplace: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        if grads_tensor is None:
            return curr_fast, curr_mom

        device = curr_fast.device
        topk_idx = topk_idx.to(device=device, dtype=torch.long)
        grads_tensor = grads_tensor.to(device=device).float()
        valid_mask = topk_idx >= 0
        if not valid_mask.any():
            return curr_fast, curr_mom

        curr_fast = curr_fast.float() if inplace else curr_fast.float().clone()
        curr_mom = curr_mom.float() if inplace else curr_mom.float().clone()

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        batch_size, num_slots, val_dim = curr_fast.shape
        idx_safe = topk_idx.clamp(min=0)
        flat_idx = idx_safe.reshape(batch_size, -1)
        valid_flat = valid_mask.reshape(batch_size, -1)
        batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_slots
        linear_idx = (flat_idx + batch_offsets)[valid_flat].reshape(-1)
        grads_flat = grads_tensor.reshape(batch_size, -1, val_dim)[valid_flat].reshape(-1, val_dim).contiguous()

        unique_linear, inverse, counts = torch.unique(
            linear_idx,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        slot_grads = grads_flat.new_zeros((unique_linear.numel(), val_dim))
        slot_grads.index_add_(0, inverse, grads_flat)
        slot_grads = slot_grads / (counts.to(dtype=torch.float32).unsqueeze(-1) + 1e-8)

        batch_idx = torch.div(unique_linear, num_slots, rounding_mode='floor')
        slot_idx = unique_linear.remainder(num_slots)

        curr_mom.mul_(self.momentum)
        touched_mom = curr_mom[batch_idx, slot_idx].add(slot_grads)
        touched_mom.clamp_(min=-50.0, max=50.0)
        curr_mom[batch_idx, slot_idx] = touched_mom

        touched_fast = curr_fast[batch_idx, slot_idx]
        update_step = (touched_mom + self.weight_decay * touched_fast).mul(-current_lr)
        curr_fast.mul_(retention_rate)
        touched_fast = touched_fast.mul(retention_rate).add(update_step)
        touched_fast.clamp_(min=-50.0, max=50.0)
        curr_fast[batch_idx, slot_idx] = touched_fast

        if getattr(self, "accumulate_deltas", False):
            self.ltm_deltas.index_add_(0, slot_idx, update_step)

        if timestamps is not None and sources is not None:
            with torch.no_grad():
                timestamps = timestamps.to(device=device)
                sources = sources.to(device=device)
                timestamps[batch_idx, slot_idx] = float(timestamp)
                sources[batch_idx, slot_idx] = int(source)

        return curr_fast, curr_mom

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

            use_cuda_math = self._use_cuda_math(queries)
            if use_cuda_math and min_timestamp <= 0.0 and source_filter is None:
                effective_topk = min(topk, self.vals.shape[0])
            else:
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
            selected_sim = torch.gather(sim, dim=-1, index=idx_clamped)

            if use_cuda_math:
                weighted_vals, ts_retrieved = self._gather_topk_cuda(
                    idx_clamped,
                    effective_memory,
                    current_timestamps,
                )
                weighted_vals = self._inject_score_gradients(weighted_vals, selected_sim)
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
                return weighted_vals, idx, ts_retrieved
            
            num_classes = effective_memory_size
            range_tensor = torch.arange(num_classes, device=idx_clamped.device)
            view_shape = [1] * idx.ndim + [-1]
            range_tensor = range_tensor.view(*view_shape)
            
            one_hot = (idx_clamped.unsqueeze(-1) == range_tensor).float()

            gathered_vals = torch.matmul(one_hot, effective_memory.float())
            
            # --- FINAL LTM FIX: Clean Concatenation Signal ---
            # With normalization handling the bleeding, we can now use the 
            # raw memories as the model expects. Softmax is no longer needed.
            weighted_vals = self._inject_score_gradients(gathered_vals.contiguous(), selected_sim)
            
            ts_for_gather = current_timestamps.to(device=one_hot.device, dtype=one_hot.dtype)
            while ts_for_gather.ndim < one_hot.ndim:
                ts_for_gather = ts_for_gather.unsqueeze(-2)
            ts_retrieved = (one_hot * ts_for_gather).sum(dim=-1)
            
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
                    timestamps: Optional[torch.Tensor] = None, sources: Optional[torch.Tensor] = None,
                    inplace: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a Fast Weight Associative Update (Titans Style).
        """
        curr_fast = fast_vals if fast_vals is not None else self.fast_vals
        curr_mom = mom_vals if mom_vals is not None else self._mom_vals

        if curr_fast.dim() == 3:
            if self._use_cuda_math(curr_fast):
                return self._inner_update_batched_cuda(
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

        if self._use_cuda_math(curr_fast):
            return self._inner_update_flat_cuda(
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

        device = curr_fast.device
        valid_mask = topk_idx >= 0
        if not valid_mask.any(): 
            return curr_fast, curr_mom

        # AMP FIX: Cast grads_tensor to float32 BEFORE boolean indexing.
        # Under BFloat16 AMP, grads_tensor arrives as BFloat16 but downstream
        # computations promote to float32. The backward pass of boolean indexing
        # uses masked_scatter_ which requires self and source to have the same dtype.
        # If grads_tensor is BFloat16 but the incoming gradient is float32, it crashes.
        grads_tensor = grads_tensor.float()
        curr_fast = curr_fast.float()
        curr_mom = curr_mom.float()

        idx_flat = topk_idx[valid_mask].view(-1)
        grads_flat = grads_tensor[valid_mask].view(-1, self.vals.size(1))
        grads_flat = grads_flat.contiguous()

        num_classes = self.vals.size(0)
        range_tensor = torch.arange(num_classes, device=device).unsqueeze(0)
        one_hot = (idx_flat.unsqueeze(1) == range_tensor).float()
        
        counts = one_hot.sum(dim=0)
        slot_grads = torch.matmul(one_hot.t(), grads_flat.to(device))
        
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

        valid_mask_all = topk_idx >= 0
        if not valid_mask_all.any():
            return curr_fast, curr_mom

        grads_tensor = grads_tensor.float()
        if not inplace:
            curr_fast = curr_fast.float().clone()
            curr_mom = curr_mom.float().clone()
        else:
            curr_fast = curr_fast.float()
            curr_mom = curr_mom.float()

        if tokens_covered is None:
            tokens_covered = self.reference_chunk_len
        retention_rate = (1.0 - self.forget_rate) ** (tokens_covered / float(self.reference_chunk_len))

        num_slots = self.vals.size(0)
        range_tensor = torch.arange(num_slots, device=curr_fast.device).unsqueeze(0)

        for batch_idx in range(curr_fast.shape[0]):
            valid_mask = valid_mask_all[batch_idx]
            if not valid_mask.any():
                curr_fast[batch_idx].mul_(retention_rate)
                continue

            idx_flat = topk_idx[batch_idx][valid_mask].view(-1).to(curr_fast.device)
            grads_flat = grads_tensor[batch_idx][valid_mask].view(-1, self.vals.size(1)).to(curr_fast.device)
            one_hot = (idx_flat.unsqueeze(1) == range_tensor).float()
            counts = one_hot.sum(dim=0)
            slot_grads = torch.matmul(one_hot.t(), grads_flat)
            slot_grads = slot_grads / (counts.unsqueeze(-1) + 1e-8)
            nonzero_mask = counts > 0

            curr_mom[batch_idx].mul_(self.momentum).add_(slot_grads)
            curr_mom[batch_idx].clamp_(min=-50.0, max=50.0)

            update_step = curr_mom[batch_idx] + self.weight_decay * curr_fast[batch_idx]
            update_step = update_step.mul(-current_lr)
            curr_fast[batch_idx].mul_(retention_rate)
            update_step = torch.where(nonzero_mask.unsqueeze(-1), update_step, torch.zeros_like(update_step))
            curr_fast[batch_idx].add_(update_step)
            curr_fast[batch_idx].clamp_(min=-50.0, max=50.0)

            # Record change for LoRA accumulation if enabled
            if getattr(self, "accumulate_deltas", False):
                self.ltm_deltas.add_(update_step)

            if timestamps is not None and sources is not None:
                with torch.no_grad():
                    timestamps[batch_idx].data[nonzero_mask] = timestamp
                    sources[batch_idx].data[nonzero_mask] = source

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
