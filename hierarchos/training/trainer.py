import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import sys
import traceback
import numpy as np

from .optimizers import DirectMLAdamW
from ..utils.device import is_directml_device, set_threads
from ..utils.checkpoint import save_checkpoint_safely, load_full_model_with_config, sanitize_model_state_dict
from ..models.core import HierarchosCore

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def validate_loss(loss: torch.Tensor, name: str = "loss") -> bool:
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        print(f"ERROR: Invalid loss ({loss.item()}) detected in {name}!")
        return False
    return True

def compute_chunk_training_weights(labels: torch.Tensor, attention_mask: torch.Tensor = None, chunk_size: int = 128):
    """
    Build TBPTT chunk weights that match the causal objective.

    CrossEntropy is averaged over valid shifted labels inside each chunk, so the
    trainer must weight chunk CE by the number of supervised answer tokens, not
    by raw chunk count or total padded length. Auxiliary costs are token-level
    dynamics, so they use real attention-mask tokens instead.
    """
    B, T = labels.shape
    if chunk_size <= 0 or chunk_size > T:
        chunk_size = T

    chunks = []
    total_valid_predictions = 0
    total_real_tokens = 0

    for start_t in range(0, T, chunk_size):
        end_t = min(start_t + chunk_size, T)
        chunk_labels = labels[:, start_t:end_t]

        if chunk_labels.shape[1] > 1:
            valid_predictions = int((chunk_labels[:, 1:] != -100).sum().item())
        else:
            valid_predictions = 0

        if attention_mask is not None:
            real_tokens = int(attention_mask[:, start_t:end_t].sum().item())
        else:
            real_tokens = B * (end_t - start_t)

        chunks.append({
            "start": start_t,
            "end": end_t,
            "valid_predictions": valid_predictions,
            "real_tokens": real_tokens,
        })
        total_valid_predictions += valid_predictions
        total_real_tokens += real_tokens

    for chunk in chunks:
        chunk["label_ratio"] = (
            chunk["valid_predictions"] / float(total_valid_predictions)
            if total_valid_predictions > 0 else 0.0
        )
        chunk["token_ratio"] = (
            chunk["real_tokens"] / float(total_real_tokens)
            if total_real_tokens > 0 else 0.0
        )

    return chunks

def compute_remaining_update_steps(dataloader_len: int, accumulation_steps: int, start_epoch: int,
                                   total_epochs: int, start_step: int = 0) -> int:
    """Count optimizer updates that will actually run after an epoch/mid-epoch resume."""
    accumulation_steps = max(1, int(accumulation_steps))
    remaining_epochs_after_current = max(0, int(total_epochs) - int(start_epoch) - 1)
    remaining_batches = max(0, int(dataloader_len) - int(start_step))
    remaining_batches += remaining_epochs_after_current * int(dataloader_len)
    return max(1, remaining_batches // accumulation_steps)

def trim_trailing_padding(input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor = None):
    """Remove trailing columns that are padding for the entire batch."""
    if attention_mask is None:
        return input_ids, labels, attention_mask
    if not isinstance(attention_mask, torch.Tensor):
        return input_ids, labels, attention_mask
    if input_ids.ndim != 2 or labels.ndim != 2 or attention_mask.ndim != 2:
        return input_ids, labels, attention_mask
    if attention_mask.shape[1] != input_ids.shape[1] or labels.shape[1] != input_ids.shape[1]:
        return input_ids, labels, attention_mask

    active_columns = attention_mask.bool().any(dim=0)
    if not bool(active_columns.any().item()):
        return input_ids, labels, attention_mask
    trim_to = int(active_columns.nonzero(as_tuple=False)[-1].item()) + 1
    if trim_to >= input_ids.shape[1]:
        return input_ids, labels, attention_mask
    return (
        input_ids[:, :trim_to].contiguous(),
        labels[:, :trim_to].contiguous(),
        attention_mask[:, :trim_to].contiguous(),
    )

def set_dataloader_epoch(dataloader, epoch: int):
    """Let length-grouped or distributed samplers reshuffle per epoch."""
    for sampler in (
        getattr(dataloader, "batch_sampler", None),
        getattr(dataloader, "sampler", None),
        getattr(dataloader, "dataset", None),
    ):
        set_epoch = getattr(sampler, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)

def train_step(model, batch, optimizer, scaler, accumulation_steps, step, args, running_states):
    """Training step with temporal chunking to match original hierarchos.py."""
    device = next(model.parameters()).device
    _nb = (device.type == 'cuda')  # non_blocking for async CUDA transfer
    full_input_ids = batch['input_ids']
    full_attention_mask = batch.get('attention_mask')
    full_labels = batch['labels']
    full_input_ids, full_labels, full_attention_mask = trim_trailing_padding(
        full_input_ids, full_labels, full_attention_mask
    )
    B, T = full_input_ids.shape
    full_input_ids = full_input_ids.to(device, non_blocking=_nb)
    if full_attention_mask is not None: full_attention_mask = full_attention_mask.to(device, non_blocking=_nb)
    full_labels = full_labels.to(device, non_blocking=_nb)
    
    # --- [NEW] Track Sequence Poisoning (Parity Fix) ---
    if torch.isnan(full_labels.float()).any():
        print(f"\nCRITICAL: NaNs detected in labels at step {step}! Skipping batch.")
        optimizer.zero_grad(set_to_none=True)
        return None, running_states
    # --------------------------------------------------
    
    h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state = running_states
    
    autocast_device = 'cpu' if is_directml_device(device) else device.type
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(model.config if hasattr(model, 'config') else args, 'amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    
    # --- INFINTIE CONTEXT: STATE RECURRENCE ---
    # Default behavior (persist_state=False): carry states ONLY between TBPTT chunks 
    # of the SAME sequence. Cross-batch persistence is disabled by default.
    persist_enabled = getattr(args, 'persist_state', False)
    
    # Safety Check: If batch size changed (e.g. last batch of epoch), we MUST reset
    if persist_enabled and h_state is not None:
        if h_state.shape[0] != B:
            print(f"INFO: Batch size changed from {h_state.shape[0]} to {B}. Resetting states.")
            persist_enabled = False

    if not persist_enabled:
        # Reset states if explicitly disabled or if batch size changed
        h_state = None
        l_state = None
        prev_ctx = None
        target_ctx = None
        drift_state = None
        ltm_state = None
        model.reset_memory() 
    else:
        # If persisting, we detach states to prevent memory issues (TBPTT truncation)
        # but the VALUES are carried forward.
        if h_state is not None: h_state = h_state.detach()
        if l_state is not None: l_state = l_state.detach()
        if prev_ctx is not None: prev_ctx = prev_ctx.detach()
        if target_ctx is not None: target_ctx = target_ctx.detach()
        if drift_state is not None: drift_state = drift_state.detach()
        if ltm_state is not None: 
            ltm_state = tuple(s.detach() if isinstance(s, torch.Tensor) else s for s in ltm_state)
    
    # Temporal chunking (critical for RWKV-based models)
    chunk_size = getattr(args, 'training_chunk_size', 128)
    if chunk_size <= 0 or chunk_size > T: chunk_size = T
    chunk_plan = compute_chunk_training_weights(full_labels, full_attention_mask, chunk_size)
    num_chunks = len(chunk_plan)
    
    total_loss = 0.0
    total_ponder = 0.0
    total_commit = 0.0
    chunks_processed = 0
    final_outputs = None
    
    try:
        for chunk_idx, chunk_info in enumerate(chunk_plan):
            start_t = chunk_info["start"]
            end_t = chunk_info["end"]
            label_ratio = chunk_info["label_ratio"]
            token_ratio = chunk_info["token_ratio"]

            # Dynamic padding can create trailing chunks with no real tokens and
            # no supervised labels. Skip them entirely so padding cannot decay or
            # momentum-step LTM state through a zero-gradient update.
            if label_ratio == 0.0 and token_ratio == 0.0:
                continue
            
            # Slice tensors for this chunk
            input_ids = full_input_ids[:, start_t:end_t]
            attention_mask = full_attention_mask[:, start_t:end_t] if full_attention_mask is not None else None
            labels = full_labels[:, start_t:end_t]
            
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=args.amp):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    h_state=h_state, l_state=l_state, prev_context=prev_ctx,
                    target_context=target_ctx, drift_state=drift_state, ltm_memory_state=ltm_state,
                    global_pos_offset=start_t
                )
                
                # LTM fast-memory update needs gradients from the exact tensors used
                # by the forward graph. retrieve_topk already keeps them float32.
                if outputs.get("raw_topk_vals") is not None:
                    for t_val in outputs["raw_topk_vals"]:
                        if t_val.requires_grad:
                            t_val.retain_grad()


                ce_loss = outputs['loss']
                ponder_cost = outputs.get('ponder_cost')
                commitment_cost = outputs.get('commitment_cost')
                
                aux_loss = torch.zeros_like(ce_loss)
                
                # --- ACT Sensitivity: Adaptive Ponder Loss ---
                if ponder_cost is not None:
                    ponder_weight = getattr(args, 'ponder_loss_weight', 0.01)
                    
                    if getattr(args, 'encourage_thinking', False):
                        # RECOVERY MODE: Invert ponder penalty to REWARD thinking
                        # Negative weight means higher ponder = lower loss
                        aux_loss = aux_loss - (abs(ponder_weight) * ponder_cost)
                    elif getattr(args, 'adaptive_ponder', False):
                        # ADAPTIVE MODE: Scale ponder target with loss
                        # Higher CE loss = more thinking needed
                        max_h_steps = getattr(args, 'max_h_steps', 5)
                        target_scale = getattr(args, 'ponder_target_scale', 0.5)
                        target_ponder = torch.clamp(ce_loss.detach() * target_scale, min=1.0, max=float(max_h_steps))
                        ponder_diff = target_ponder - ponder_cost
                        # Penalize under-thinking (when ponder < target), ignore over-thinking
                        ponder_penalty = torch.relu(ponder_diff) * ponder_weight
                        aux_loss = aux_loss + ponder_penalty
                    else:
                        # STANDARD MODE: Original additive penalty (penalizes thinking)
                        aux_loss = aux_loss + (ponder_weight * ponder_cost)
                
                if commitment_cost is not None:
                    aux_loss = aux_loss + (getattr(args, 'commitment_loss_weight', 0.5) * commitment_cost)

                # CE is already averaged over valid labels within this chunk.
                # Weight it by supervised answer-token count so long masked
                # prompts/tool traces do not dilute the actual learning signal.
                chunk_loss = ((ce_loss * label_ratio) + (aux_loss * token_ratio)) / accumulation_steps
            
            # Backprop per chunk (TBPTT)
            # retain_grad() is now handled inside autocast block above (BUG #1 fix)

            if scaler is not None: scaler.scale(chunk_loss).backward()
            else: chunk_loss.backward()
            
            # --- GRADIENT-BASED LTM UPDATE (Titans Parity) ---
            if outputs.get("raw_topk_vals") is not None:
                with torch.no_grad():
                    valid_grads = []
                    for t_val in outputs["raw_topk_vals"]:
                        g = t_val.grad
                        if g is not None:
                            # Unscale if using AMP
                            if getattr(args, 'amp', False) and scaler is not None:
                                current_scale = scaler.get_scale()
                                if current_scale > 1e-6:
                                    g = g / current_scale
                            valid_grads.append(g.detach())
                        else:
                            valid_grads.append(torch.zeros_like(t_val))
                    
                    ltm_grads_tensor = torch.stack(valid_grads, dim=1)
                    # Clear intermediate grads immediately to free memory
                    for t_val in outputs["raw_topk_vals"]:
                        t_val.grad = None
                    outputs["raw_topk_vals"] = None  # Free tensor references (BUG #3: memory leak fix)

                    if torch.isfinite(ltm_grads_tensor).all():
                        # Direct tensor clipping (clip_grad_norm_ expects parameters with .grad,
                        # but ltm_grads_tensor IS the gradient data itself — no .grad attribute)
                        _clip_val = getattr(args, 'grad_clip', 1.0)
                        if _clip_val > 0:
                            _grad_norm = ltm_grads_tensor.norm()
                            if _grad_norm > _clip_val:
                                ltm_grads_tensor = ltm_grads_tensor * (_clip_val / (_grad_norm + 1e-8))
                        
                        # Unpack current LTM state for the update
                        curr_ltm = outputs.get('ltm_memory_state')
                        curr_fast = curr_ltm[0] if curr_ltm is not None else None
                        curr_mom = curr_ltm[1] if curr_ltm is not None else None
                        curr_past_tokens = curr_ltm[2] if curr_ltm is not None and len(curr_ltm) >= 3 else None
                        curr_rosa_states = curr_ltm[3] if curr_ltm is not None and len(curr_ltm) >= 4 else None
                        curr_timestamps = curr_ltm[4] if curr_ltm is not None and len(curr_ltm) >= 5 else None
                        curr_sources = curr_ltm[5] if curr_ltm is not None and len(curr_ltm) >= 6 else None
                        
                        # Titans inner_update (Gradient-based)
                        # Pass fast_vals/mom_vals from forward pass state, not module defaults
                        new_fast, new_mom = model.ltm.inner_update(
                            outputs["topk_idx"],
                            ltm_grads_tensor,
                            current_lr=getattr(args, 'ltm_lr', 0.001),
                            source=2, # SRC_TRAINING_DATA
                            timestamp=float(end_t),
                            tokens_covered=end_t - start_t,
                            fast_vals=curr_fast,
                            mom_vals=curr_mom,
                            timestamps=curr_timestamps,
                            sources=curr_sources,
                            inplace=True
                        )
                        ltm_state = (new_fast.detach(), new_mom.detach(), 
                                     curr_past_tokens.detach() if curr_past_tokens is not None else None,
                                     curr_rosa_states,
                                     curr_timestamps.detach() if isinstance(curr_timestamps, torch.Tensor) else curr_timestamps,
                                     curr_sources.detach() if isinstance(curr_sources, torch.Tensor) else curr_sources)  # ROSA automaton states (plain Python, no detach needed)
                    else:
                        curr_ltm = outputs['ltm_memory_state']
                        ltm_state = (curr_ltm[0].detach(), curr_ltm[1].detach(),
                                     curr_ltm[2].detach() if len(curr_ltm) >= 3 and curr_ltm[2] is not None else None,
                                     curr_ltm[3] if len(curr_ltm) >= 4 else None,
                                     curr_ltm[4].detach() if len(curr_ltm) >= 5 and isinstance(curr_ltm[4], torch.Tensor) else None,
                                     curr_ltm[5].detach() if len(curr_ltm) >= 6 and isinstance(curr_ltm[5], torch.Tensor) else None)
            else:
                # No LTM outputs? Just take what we have
                if outputs.get('ltm_memory_state') is not None:
                    curr_ltm = outputs['ltm_memory_state']
                    ltm_state = (curr_ltm[0].detach(), curr_ltm[1].detach(),
                                 curr_ltm[2].detach() if len(curr_ltm) >= 3 and curr_ltm[2] is not None else None,
                                 curr_ltm[3] if len(curr_ltm) >= 4 else None,
                                 curr_ltm[4].detach() if len(curr_ltm) >= 5 and isinstance(curr_ltm[4], torch.Tensor) else None,
                                 curr_ltm[5].detach() if len(curr_ltm) >= 6 and isinstance(curr_ltm[5], torch.Tensor) else None)

            # Update states for next chunk (TBPTT - detach to limit gradient flow)
            if outputs.get('h_state') is not None:
                h_state = torch.clamp(outputs['h_state'].detach(), min=-50.0, max=50.0)
            if outputs.get('l_state') is not None:
                l_state = torch.clamp(outputs['l_state'].detach(), min=-50.0, max=50.0)
            if outputs.get('prev_context') is not None:
                prev_ctx = torch.clamp(outputs['prev_context'].detach(), min=-50.0, max=50.0)
            if outputs.get('target_context') is not None:
                target_ctx = torch.clamp(outputs['target_context'].detach(), min=-50.0, max=50.0)
            if outputs.get('drift_state') is not None:
                drift_state = torch.clamp(outputs['drift_state'].detach(), min=-5.0, max=5.0)
            
            # Accumulate for display
            total_loss += ce_loss.item() * label_ratio
            if ponder_cost is not None: 
                total_ponder += ponder_cost.item() * token_ratio
            if commitment_cost is not None: 
                total_commit += commitment_cost.item() * token_ratio
            
            chunks_processed += 1
            # final_outputs = outputs # REMOVED: Memory Leak Fix
        
        # Optimizer step after all chunks
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), getattr(args, 'grad_clip', 1.0))
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        if chunks_processed == 0:
            return None, running_states
            
        # Return averaged metrics for display
        # Since we already weighted by chunk_ratio, these are correct averages
        avg_outputs = {
            'loss': torch.tensor(total_loss),
            'ponder_cost': torch.tensor(total_ponder) if total_ponder > 0 else None,
            'commitment_cost': torch.tensor(total_commit) if total_commit > 0 else None,
        }
        
        next_states = (h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state)
        return avg_outputs, next_states
            
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("WARNING: OOM detected. Clearing cache.")
            torch.cuda.empty_cache()
            return None, running_states
        raise e


def train(args, device, tokenizer, dataloader, dataloader_len):
    print("Running in TRAIN mode...")
    if dataloader_len <= 0:
        print("ERROR: dataloader_len must be > 0. If automatic detection failed, please specify --dataset-size.")
        return
    
    
    if getattr(args, 'out_dir', None):
        os.makedirs(args.out_dir, exist_ok=True)
    config = AttrDict(vars(args))

    # Device stability
    if is_directml_device(device):
        args.compile = False
        args.amp = False
        config.compile = False
        config.amp = False

    if getattr(args, 'force_compile', False): 
        args.compile = True
        config.compile = True

    # =================================================================
    # CUDA DATACENTER OPTIMIZATIONS
    # =================================================================
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(device)
        gpu_mem = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        gpu_capability = torch.cuda.get_device_capability(device)
        print(f"INFO: CUDA GPU: {gpu_name} ({gpu_mem:.1f} GB, SM {gpu_capability[0]}.{gpu_capability[1]})")

        # TF32 matmul (Ampere+, SM >= 8.0) — 3-8x faster matmuls with negligible accuracy loss
        if gpu_capability[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')  # OPT #4: Canonical TF32 API
            print("INFO: TF32 matmul enabled (Ampere+ GPU detected).")

        # cuDNN benchmark — auto-tunes convolution algorithms for the hardware
        torch.backends.cudnn.benchmark = True

        # Auto-enable AMP on CUDA unless the user explicitly passed --amp or --no-amp.
        # Must check all argparse-accepted forms (hyphen and underscore variants).
        _amp_was_explicitly_set = any(a in sys.argv for a in ('--amp', '--no-amp', '--no_amp'))
        if not _amp_was_explicitly_set:
            args.amp = True
            config.amp = True
            print("INFO: AMP auto-enabled for CUDA training (use --no-amp to disable).")

        # Auto-enable torch.compile on CUDA (no Windows CPU hang issue)
        if not getattr(args, 'compile', False) and not getattr(args, 'force_compile', False):
            args.compile = True
            config.compile = True
            print("INFO: torch.compile auto-enabled for CUDA training.")

        # Prefer bfloat16 on Ampere+ for better dynamic range (no GradScaler needed)
        if gpu_capability[0] >= 8 and torch.cuda.is_bf16_supported():
            args.amp_dtype = 'bfloat16'
            config.amp_dtype = 'bfloat16'
            print("INFO: Using bfloat16 AMP (Ampere+ native support).")
        else:
            args.amp_dtype = 'float16'
            config.amp_dtype = 'float16'
            print("INFO: Using float16 AMP with GradScaler.")

        # Multi-GPU info
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"INFO: {num_gpus} GPUs detected. For multi-GPU training, wrap with DistributedDataParallel.")
    # =================================================================

    model = None
    optimizer = None
    start_epoch = 0
    scaler = None
    scheduler = None
    checkpoint = None
    # NOTE: use_amp MUST be read AFTER the CUDA block above, which may auto-enable AMP.
    use_amp = getattr(args, 'amp', False)
    
    # 1. Loading/Resuming Logic
    if args.resume_from_ckpt:
        print(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        checkpoint = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False)
        
        saved_config = checkpoint.get('config', {})
        model_config = AttrDict(saved_config)
        state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
        
        # ARCH Detection (Safely handling compiled checkpoints with '_orig_mod.' prefix)
        state_dict_keys = set()
        for k in state_dict.keys():
            clean_k = k.replace('_orig_mod.', '')
            state_dict_keys.add(clean_k)
            
        # 1. Detect vocab_size and context_dim from tok_emb/lm_head
        if 'tok_emb.weight' in state_dict_keys:
            key = 'tok_emb.weight' if 'tok_emb.weight' in state_dict else '_orig_mod.tok_emb.weight'
            model_config.vocab_size = state_dict[key].shape[0]
            model_config.context_dim = state_dict[key].shape[1]
        elif 'lm_head.weight' in state_dict_keys:
            key = 'lm_head.weight' if 'lm_head.weight' in state_dict else '_orig_mod.lm_head.weight'
            model_config.vocab_size = state_dict[key].shape[0]
            model_config.context_dim = state_dict[key].shape[1]
        
        # 2. Detect persistent_dim
        if 'persistent' in state_dict_keys:
            key = 'persistent' if 'persistent' in state_dict else '_orig_mod.persistent'
            model_config.persistent_dim = state_dict[key].shape[0]
        
        # 3. Detect LTM dims
        if 'val_proj.weight' in state_dict_keys:
            key = 'val_proj.weight' if 'val_proj.weight' in state_dict else '_orig_mod.val_proj.weight'
            model_config.ltm_val_dim = state_dict[key].shape[0]
        if 'qproj.weight' in state_dict_keys:
            key = 'qproj.weight' if 'qproj.weight' in state_dict else '_orig_mod.qproj.weight'
            model_config.ltm_key_dim = state_dict[key].shape[0]

        # 4. Detect RNN hidden sizes (RWKV-style states)
        if 'h_rnn.time_decay' in state_dict_keys:
            key = 'h_rnn.time_decay' if 'h_rnn.time_decay' in state_dict else '_orig_mod.h_rnn.time_decay'
            model_config.h_hidden = state_dict[key].shape[-1]
        if 'l_rnn.time_decay' in state_dict_keys:
            key = 'l_rnn.time_decay' if 'l_rnn.time_decay' in state_dict else '_orig_mod.l_rnn.time_decay'
            model_config.l_hidden = state_dict[key].shape[-1]

        # ARCH defaults / Fallbacks
        arch_defaults = {
            'ltm_slots': 1024, 'ltm_key_dim': 128, 'ltm_val_dim': 128, 'ltm_topk': 4,
            'h_stride': 4, 'max_h_steps': 5, 'max_l_steps': 5,
            'h_hidden': model_config.get('context_dim', 768),
            'l_hidden': model_config.get('context_dim', 768)
        }
        for k, v in arch_defaults.items():
            if k not in model_config:
                model_config[k] = getattr(args, k, v) if hasattr(args, k) else v

        # Runtime overrides
        model_config.compile = args.compile
        model_config.max_length = args.max_length or model_config.get('max_length', 1024)

        print(f"INFO: Final Adjusted ARCH: context_dim={model_config.context_dim}, persistent={model_config.get('persistent_dim', 128)}, ltm_val={model_config.get('ltm_val_dim', 128)}, h_hidden={model_config.h_hidden}, l_hidden={model_config.l_hidden}, vocab_size={model_config.vocab_size}")
        
        model = HierarchosCore(model_config).to(device)
        
        # Safe loading with prefix adaptation if necessary (unlikely here but good to have)
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"WARNING: Missing keys in checkpoint: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"WARNING: Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
            if not missing and not unexpected:
                print(f"INFO: Model state_dict loaded perfectly ({len(state_dict)} parameters).")
        except RuntimeError as e:
            print(f"ERROR: Failed to load state_dict! {e}")
            raise
        
        # --- SURGICAL FIX: Reset h_halt_proj.bias to encourage pondering ---
        reset_bias = getattr(args, 'reset_halt_bias', None)
        if reset_bias is not None:
            with torch.no_grad():
                if hasattr(model, 'h_halt_proj') and model.h_halt_proj.bias is not None:
                    old_bias = model.h_halt_proj.bias.item()
                    model.h_halt_proj.bias.fill_(reset_bias)
                    print(f"INFO: SURGICAL FIX - Reset h_halt_proj.bias from {old_bias:.4f} to {reset_bias:.4f}")
                    print(f"      Initial halt probability: {torch.sigmoid(torch.tensor(reset_bias)).item():.2%}")
                else:
                    print("WARNING: h_halt_proj.bias not found, surgical fix skipped.")
        
        if is_directml_device(device): optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        elif device.type == 'cuda': optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr, fused=True)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        
        if not getattr(args, 'override_scheduling', False) and 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except: print("Warning: Could not load optimizer state.")
        
        # Original script uses 'completed_epoch', modular uses 'epoch', check both for compatibility
        start_epoch = checkpoint.get('completed_epoch', checkpoint.get('epoch', 0))
        print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.")
        # BFloat16 does NOT use GradScaler — its dynamic range makes scaling unnecessary.
        # Only create scaler for float16 AMP.
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16':
            scaler = GradScaler()
            if 'scaler_state_dict' in checkpoint and not getattr(args, 'override_scheduling', False):
                try: scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except: pass
    
    elif args.model_path:
        print(f"Loading base model from: {args.model_path}")
        model, model_config = load_full_model_with_config(args.model_path, device)
        model_config.compile = args.compile
        model.config = model_config
        
        if is_directml_device(device): optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        elif device.type == 'cuda': optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr, fused=True)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()
    
    else:
        print("Starting training from scratch.")
        if 'vocab_size' not in config: config.vocab_size = len(tokenizer)
        model = HierarchosCore(config).to(device)
        if is_directml_device(device): optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        elif device.type == 'cuda': optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr, fused=True)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()

    # --- [NEW] Sync LTM reference chunk size (Parity Fix) ---
    training_chunk_size = getattr(args, 'training_chunk_size', 128)
    if hasattr(model, 'ltm'):
        if not hasattr(model.ltm, 'reference_chunk_len'):
            model.ltm.reference_chunk_len = training_chunk_size
        
        if model.ltm.reference_chunk_len != training_chunk_size:
            print(f"INFO: Updating LTM reference chunk length from {model.ltm.reference_chunk_len} to {training_chunk_size}")
            model.ltm.reference_chunk_len = training_chunk_size
    # ----------------------------------------------------

    # --- Print Model Stats ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate file size: params * 4 bytes (float32) + ~10% overhead
    estimated_bytes = total_params * 4 * 1.1
    if estimated_bytes >= 1e9:
        size_str = f"{estimated_bytes / 1e9:.2f} GB"
    else:
        size_str = f"{estimated_bytes / 1e6:.2f} MB"
    print(f"INFO: Model Parameters: {total_params:,} total ({trainable_params:,} trainable)")
    print(f"INFO: Estimated checkpoint size: ~{size_str}")
    # --------------------------

    # Compile
    model.compile()

    start_step = checkpoint.get('mid_epoch_step', 0) if checkpoint else 0

    # Scheduler
    # When override_scheduling is used during resume, calculate T_max based on REMAINING epochs
    # so that the LR decays properly to min_lr by the final epoch.
    if getattr(args, 'override_scheduling', False) and args.resume_from_ckpt:
        num_update_steps = compute_remaining_update_steps(
            dataloader_len,
            args.accumulation_steps,
            start_epoch,
            args.epochs,
            start_step,
        )
        print(f"INFO: --override-scheduling: Calculating LR schedule for remaining work ({num_update_steps} update steps)")
    else:
        num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs
    
    if not getattr(args, 'disable_lr_schedule', False) and num_update_steps > 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
        if args.resume_from_ckpt and not getattr(args, 'override_scheduling', False) and 'scheduler_state_dict' in checkpoint:
            try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except: pass
    # --- Evaluation Confirmation ---
    eval_tasks = getattr(args, 'eval_tasks', None)
    if eval_tasks:
        eval_every = getattr(args, 'eval_every_epoch', 1)
        print(f"INFO: Evaluation ENABLED - will run {eval_tasks} every {eval_every} epoch(s)")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        set_dataloader_epoch(dataloader, epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # [NEW] Restore running_states if resuming mid-epoch
        if epoch == start_epoch and checkpoint and 'running_states' in checkpoint:
            _raw_states = checkpoint['running_states']
            # Ensure tensors are on the correct device
            running_states = []
            for s in _raw_states:
                if isinstance(s, torch.Tensor):
                    running_states.append(s.to(device))
                elif isinstance(s, tuple):
                    running_states.append(tuple(ss.to(device) if isinstance(ss, torch.Tensor) else ss for ss in s))
                else:
                    running_states.append(s)
            running_states = tuple(running_states)
            print(f"INFO: Restored RNN/LTM running states from checkpoint on {device}.")
        else:
            running_states = (None, None, None, None, None, None)
        
        for step, batch in enumerate(pbar):
            # Mid-Epoch Resumption: Skip steps already processed
            if epoch == start_epoch and step < start_step:
                if step == start_step - 1: # Print only once when we are about to start processing
                    print(f"INFO: Resuming from mid-epoch step {start_step}...")
                continue

            if batch is None:
                continue
            
            # --- FIXED: Sequence-Level State Reset ---
            # If not persisting across batches, we must start each sequence with a clean slate.
            # Local sequence context is still preserved via trainer.train_step's chunk loop.
            if not getattr(args, 'persist_state', False):
                running_states = (None, None, None, None, None, None)

            outputs, running_states = train_step(model, batch, optimizer, scaler, args.accumulation_steps, step, args, running_states)
            if outputs:
                postfix = {"loss": f"{outputs['loss'].item():.4f}"}
                if outputs.get('ponder_cost') is not None:
                    postfix["ponder"] = f"{outputs['ponder_cost'].item():.2f}"
                if outputs.get('commitment_cost') is not None:
                    postfix["commit"] = f"{outputs['commitment_cost'].item():.2e}"
                if scheduler:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                pbar.set_postfix(postfix)

            if scheduler and (step + 1) % args.accumulation_steps == 0:
                scheduler.step()

            # Periodic Checkpointing (Progress Protection)
            if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                print(f"\n[Step {step+1}] Periodic Checkpoint: Saving to {args.out_dir}...")
                save_checkpoint_safely({
                    'completed_epoch': epoch, # Not yet completed
                    'mid_epoch_step': step + 1,
                    'model_state_dict': sanitize_model_state_dict(model),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'config': dict(model.config),
                    'running_states': running_states, # [NEW] Persist states for graceful resumption
                }, os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}_step_{step+1}.pt"))
            
            # --- STEP-BASED EVALUATION (runs every N steps) ---
            eval_steps = getattr(args, 'eval_steps', None)
            if eval_steps and eval_tasks and (step + 1) % eval_steps == 0:
                try:
                    from ..evaluation import run_eval, format_results, is_lm_eval_available
                    
                    if is_lm_eval_available():
                        print(f"\n[Step {step+1}] Running evaluation (--eval-steps triggered)...")
                        model.eval()
                        
                        eval_results = run_eval(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            tasks=eval_tasks,
                            batch_size=getattr(args, 'eval_batch_size', 1),
                            limit=getattr(args, 'eval_limit', None),
                            verbosity="WARNING"
                        )
                        
                        if eval_results:
                            print(format_results(eval_results, tasks=eval_tasks))
                            
                            eval_path = os.path.join(args.out_dir, f"eval_epoch_{epoch+1}_step_{step+1}.json")
                            from ..evaluation import save_results
                            save_results(eval_results, eval_path)
                        
                        model.train()
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    print(f"WARNING: Step-based evaluation failed: {e}")
                    model.train()
        
        save_checkpoint_safely({
            'completed_epoch': epoch + 1,
            'model_state_dict': sanitize_model_state_dict(model),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'config': dict(model.config),
        }, os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}.pt"))
        
        # --- OPTIONAL EVALUATION (lm-evaluation-harness) ---
        eval_tasks = getattr(args, 'eval_tasks', None)
        if eval_tasks:
            eval_every = getattr(args, 'eval_every_epoch', 1)
            if (epoch + 1) % eval_every == 0:
                try:
                    from ..evaluation import run_eval, format_results, is_lm_eval_available
                    
                    if is_lm_eval_available():
                        print(f"\n[Epoch {epoch+1}] Running evaluation on: {eval_tasks}")
                        model.eval()
                        
                        eval_results = run_eval(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            tasks=eval_tasks,
                            batch_size=getattr(args, 'eval_batch_size', 1),
                            limit=getattr(args, 'eval_limit', None),
                            verbosity="WARNING"  # Reduce lm-eval verbosity
                        )
                        
                        if eval_results:
                            print(format_results(eval_results, tasks=eval_tasks))
                            
                            # Save results to file
                            eval_path = os.path.join(args.out_dir, f"eval_epoch_{epoch+1}.json")
                            from ..evaluation import save_results
                            save_results(eval_results, eval_path)
                        
                        model.train()  # Back to training mode
                    else:
                        print("WARNING: lm-eval not installed. Install with: pip install lm-eval>=0.4.0")
                except Exception as e:
                    print(f"WARNING: Evaluation failed: {e}")
                    model.train()  # Ensure model is back in training mode

    # --- FINAL INFERENCE MODEL EXPORT ---
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Save inference-ready model (no optimizer/scheduler state = smaller file)
    final_model_path = os.path.join(args.out_dir, "hierarchos.pt")
    print(f"Saving final inference model to: {final_model_path}")
    
    # Clean state dict (remove _orig_mod. prefix from compiled models)
    clean_state_dict = sanitize_model_state_dict(model)
    
    final_checkpoint = {
        'model_state_dict': clean_state_dict,
        'config': dict(model.config),
        'completed_epoch': args.epochs,
        'training_complete': True,
    }
    save_checkpoint_safely(final_checkpoint, final_model_path)
    
    # Save tokenizer files into the directory (HuggingFace-style portability)
    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
    
    # Save config as JSON for easy inspection
    try:
        import json as json_module
        config_path = os.path.join(args.out_dir, "hierarchos_config.json")
        config_to_save = dict(model.config)
        config_to_save['completed_epoch'] = args.epochs
        with open(config_path, 'w') as f:
            json_module.dump(config_to_save, f, indent=2, default=str)
        print(f"Config saved to {config_path}")
    except Exception as e:
        print(f"Warning: Failed to save config JSON: {e}")
    
    # Calculate final model size
    model_size_bytes = os.path.getsize(final_model_path)
    if model_size_bytes >= 1e9:
        size_str = f"{model_size_bytes / 1e9:.2f} GB"
    else:
        size_str = f"{model_size_bytes / 1e6:.2f} MB"
    
    print(f"Final model size: {size_str}")
    print(f"Total epochs completed: {args.epochs}")
    print(f"\nTo use the model for inference, run:")
    print(f"  python hierarchos_cli.py chat --model-path \"{args.out_dir}\"")
    print("="*60 + "\n")

def finetune(args, device, tokenizer, dataloader, dataloader_len):
    """
    LoRA-based fine-tuning with PEFT support.
    
    Ported from hierarchos.py monolith for full feature parity.
    """
    # Try importing PEFT
    try:
        from peft import LoraConfig, get_peft_model, PeftModel
        _HAS_PEFT = True
    except ImportError:
        _HAS_PEFT = False
    
    if not _HAS_PEFT:
        raise ImportError("Please install 'peft' for fine-tuning: pip install peft")
    
    print("Running in FINETUNE mode with LoRA...")
    if dataloader_len <= 0:
        print("ERROR: dataloader_len must be > 0. If automatic detection failed, please specify --dataset-size.")
        return

    # Load the base model and its config
    model, model_config = load_full_model_with_config(args.model_path, device)

    # Ensure max_length from CLI is used if provided
    if args.max_length and args.max_length != model_config.get('max_length', 1024):
        print(f"INFO: Overriding loaded model max_length ({model_config.get('max_length')}) with CLI value ({args.max_length})")
        model_config.max_length = args.max_length
        model.pos_emb = nn.Embedding(model_config.max_length, model_config.context_dim).to(device)
    elif 'max_length' not in model_config:
        print("Warning: max_length missing from loaded config. Using default 1024.")
        model_config.max_length = 1024
        model.pos_emb = nn.Embedding(model_config.max_length, model_config.context_dim).to(device)

    # Ensure gradient_checkpointing flag from CLI is used
    gradient_checkpointing = getattr(args, 'gradient_checkpointing', False)
    if gradient_checkpointing != model_config.get('gradient_checkpointing', False):
        print(f"INFO: Setting gradient_checkpointing to {gradient_checkpointing}")
        model_config.gradient_checkpointing = gradient_checkpointing
    elif 'gradient_checkpointing' not in model_config:
        model_config.gradient_checkpointing = gradient_checkpointing

    # Ensure h_stride flag from CLI is used
    h_stride = getattr(args, 'h_stride', 4)
    if h_stride != model_config.get('h_stride', 4):
        print(f"INFO: Overriding model h_stride ({model_config.get('h_stride', 4)}) with CLI value ({h_stride})")
        model_config.h_stride = h_stride
    elif 'h_stride' not in model_config:
        model_config.h_stride = h_stride

    model.config = model_config

    # Determine LoRA rank
    lora_r = getattr(args, 'lora_r', 8)
    finetune_unlock_percent = getattr(args, 'finetune_unlock_percent', None)
    
    if finetune_unlock_percent is not None:
        if lora_r != 8:  # Default value check
            print(f"Warning: Both --lora_r ({lora_r}) and --finetune-unlock-percent were specified. Prioritizing --lora_r.")
        else:
            total_params = sum(p.numel() for p in model.parameters())
            target_modules = ["qproj", "in_proj", "h_to_context", "l_to_out", "h_halt_proj", "W_ir", "W_hr", "W_iz", "W_hz", "W_in", "W_hn"]
            lora_param_sum_per_r = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and any(tm in name for tm in target_modules):
                    lora_param_sum_per_r += module.in_features + module.out_features

            target_trainable_count = total_params * (finetune_unlock_percent / 100.0)
            if lora_param_sum_per_r > 0:
                estimated_r = target_trainable_count / lora_param_sum_per_r
                lora_r = max(1, int(round(estimated_r)))
                print(f"Targeting ~{finetune_unlock_percent}% trainable parameters. Estimated LoRA rank 'r' = {lora_r}")
            else:
                print("Warning: Could not find target modules for LoRA. Using default r=8.")

    lora_alpha = getattr(args, 'lora_alpha', 16)
    
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            # RWKV time-mixing
            "key", "value", "receptance", "output",
            # RWKV channel-mixing
            "key_cm", "receptance_cm", "value_cm",
            # Hierarchos-specific layers
            "qproj", "in_proj", "h_to_context",
            "l_input_proj", "l_to_out", "h_halt_proj",
            "context_drift_proj", "l_feedback_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["ltm"],  # LTM still updated directly
    )
    model = get_peft_model(model, lora_config)

    # Resumption Logic
    start_epoch = 0
    start_step = 0
    checkpoint = None
    if getattr(args, 'resume_from_ckpt', None):
        print(f"Resuming LoRA finetune from: {args.resume_from_ckpt}")
        checkpoint = torch.load(args.resume_from_ckpt, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False), strict=False)
        start_epoch = checkpoint.get('completed_epoch', 0)
        start_step = checkpoint.get('mid_epoch_step', 0)
        print(f"INFO: Resuming from epoch {start_epoch+1}, step {start_step}.")

    model.print_trainable_parameters()

    # Optimizer selection
    if is_directml_device(device):
        print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
        optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
    
    if checkpoint and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Successfully loaded optimizer state.")
        except:
            print("Warning: Could not load optimizer state.")

    os.makedirs(args.out_dir, exist_ok=True)

    # AMP setup (BFloat16 does NOT use GradScaler — only float16 needs it)
    scaler = None
    use_amp = getattr(args, 'amp', False)
    amp_dtype_str = getattr(args, 'amp_dtype', None) or getattr(model.config if hasattr(model, 'config') else args, 'amp_dtype', 'float16')
    amp_dtype = torch.bfloat16 if amp_dtype_str == 'bfloat16' else torch.float16
    if use_amp:
        if amp_dtype_str == 'float16':
            scaler = GradScaler()
            if checkpoint and 'scaler_state_dict' in checkpoint:
                try: scaler.load_state_dict(checkpoint['scaler_state_dict'])
                except: pass
        print(f"INFO: Automatic Mixed Precision (AMP) ENABLED for fine-tuning ({amp_dtype_str}).")

    # Scheduler setup
    scheduler = None
    if not getattr(args, 'disable_lr_schedule', False):
        accumulation_steps = getattr(args, 'accumulation_steps', 1)
        num_update_steps = (dataloader_len // accumulation_steps) * args.epochs if dataloader_len > 0 else 0
        if num_update_steps > 0:
            print(f"INFO: Cosine Annealing LR scheduler ENABLED. Total steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
            if checkpoint and 'scheduler_state_dict' in checkpoint:
                try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except: pass
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small.")

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    ponder_loss_weight = getattr(args, 'ponder_loss_weight', 0.01)
    commitment_loss_weight = getattr(args, 'commitment_loss_weight', 0.5)
    grad_clip = getattr(args, 'grad_clip', 1.0)
    ltm_lr = getattr(args, 'ltm_lr', 0.001)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        set_dataloader_epoch(dataloader, epoch)
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}")
        total_loss = 0.0
        total_ponder_cost = 0.0
        total_commitment_cost = 0.0
        
        backward_called_in_cycle = False
        steps_in_epoch = 0

        for i, batch in enumerate(pbar):
            # Mid-Epoch Resumption: Skip steps already processed
            if epoch == start_epoch and i < start_step:
                if i == start_step - 1:
                    print(f"INFO: Resuming from mid-epoch step {start_step}...")
                continue

            if batch is None:
                continue

            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch["labels"]
            input_ids, labels, attention_mask = trim_trailing_padding(
                input_ids, labels, attention_mask
            )
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            autocast_device_type = 'cpu' if is_directml_device(device) else device.type
            with autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                # AMP FIX: Cast topk_vals to float32 before retain_grad (same fix as train path).
                # Prevents masked_scatter_ dtype mismatch crash under BFloat16 AMP.
                if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad:
                    outputs["topk_vals"] = outputs["topk_vals"].float()
                    outputs["topk_vals"].retain_grad()
                
                cross_entropy_loss = outputs.get("loss")
                ponder_cost = outputs.get("ponder_cost")
                commitment_cost = outputs.get("commitment_cost")

                combined_loss = None
                ce_valid = cross_entropy_loss is not None and torch.isfinite(cross_entropy_loss).all()
                pc_valid = ponder_cost is not None and torch.isfinite(ponder_cost).all()
                cc_valid = commitment_cost is not None and torch.isfinite(commitment_cost).all()

                loss_accum = 0.0

                if ce_valid:
                    loss_accum = loss_accum + cross_entropy_loss
                elif i % accumulation_steps == 0:
                    print(f"\nWarning: CE loss is NaN/Inf at step {i+1}. Skipping.")

                if pc_valid:
                    loss_accum = loss_accum + (ponder_loss_weight * ponder_cost)
                
                if cc_valid:
                    loss_accum = loss_accum + (commitment_loss_weight * commitment_cost)

                if ce_valid:
                    combined_loss = loss_accum

            if combined_loss is not None:
                loss_to_backward = combined_loss / accumulation_steps

                if use_amp and scaler:
                    scaler.scale(loss_to_backward).backward()
                else:
                    loss_to_backward.backward()

                backward_called_in_cycle = True

                if ce_valid:
                    total_loss += cross_entropy_loss.item()
                if pc_valid:
                    total_ponder_cost += ponder_cost.item()
                if cc_valid:
                    total_commitment_cost += commitment_cost.item()
                steps_in_epoch += 1

            if (i + 1) % accumulation_steps == 0:
                if backward_called_in_cycle:
                    # LTM Update
                    ltm_grads = None
                    if outputs.get("topk_vals") is not None and outputs["topk_vals"].requires_grad:
                        if outputs["topk_vals"].grad is not None:
                            ltm_grads = outputs["topk_vals"].grad

                    if ltm_grads is not None:
                        base_ltm = model.base_model.model.ltm
                        ltm_grads_copy = ltm_grads.detach().clone()

                        valid_update = True
                        if use_amp and scaler:
                            current_scale = scaler.get_scale()
                            if current_scale > 1e-6:
                                ltm_grads_copy = ltm_grads_copy / current_scale
                            else:
                                valid_update = False

                        if valid_update and torch.isfinite(ltm_grads_copy).all():
                            if grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_([ltm_grads_copy], grad_clip)
                            base_ltm.inner_update(
                                outputs["topk_idx"],
                                ltm_grads_copy,
                                current_lr=ltm_lr,
                                timestamp=float(i + 1),
                                source=2  # SRC_TRAINING_DATA
                            )

                    # Optimizer step
                    if use_amp and scaler:
                        scaler.unscale_(optimizer)
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), grad_clip)
                        optimizer.step()

                    if scheduler:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False
                    global_step += 1

                    # Periodic Checkpointing (Progress Protection)
                    if getattr(args, 'save_steps', 0) > 0 and (i + 1) % args.save_steps == 0:
                        ckpt_path = os.path.join(args.out_dir, f"hierarchos_finetune_epoch_{epoch+1}_step_{i+1}.pt")
                        print(f"\n[Step {i+1}] Periodic Checkpoint: Saving to {ckpt_path}...")
                        save_checkpoint_safely({
                            'completed_epoch': epoch,
                            'mid_epoch_step': i + 1,
                            'model_state_dict': sanitize_model_state_dict(model),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'config': dict(model.config),
                        }, ckpt_path)
                else:
                    print(f"\nWarning: Skipping optimizer step at batch {i+1} due to invalid loss.")
                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False

            # Update progress bar
            avg_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_ponder = total_ponder_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            avg_commit = total_commitment_cost / steps_in_epoch if steps_in_epoch > 0 else 0.0
            current_lr = scheduler.get_last_lr()[0] if scheduler else args.starting_lr
            
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "ponder": f"{avg_ponder:.2f}",
                "commit": f"{avg_commit:.2e}",
                "lr": f"{current_lr:.2e}"
            })

    # Save LoRA adapter
    print(f"Saving LoRA adapter to {args.out_dir}")
    model.save_pretrained(args.out_dir)
    
    # Save tokenizer
    try:
        if tokenizer:
            tokenizer.save_pretrained(args.out_dir)
            print(f"Tokenizer files saved to {args.out_dir}")
    except Exception as e:
        print(f"Warning: Failed to save tokenizer with adapter. Error: {e}")
