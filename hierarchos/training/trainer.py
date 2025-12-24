import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
import math
import sys
import traceback
import numpy as np

from .optimizers import DirectMLAdamW
from ..utils.device import is_directml_device, set_threads
from ..utils.checkpoint import save_checkpoint_safely, load_full_model_with_config
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

def train_step(model, batch, optimizer, scaler, accumulation_steps, step, args, running_states):
    """Training step with temporal chunking to match original hierarchos.py."""
    device = next(model.parameters()).device
    full_input_ids = batch['input_ids'].to(device)
    full_attention_mask = batch.get('attention_mask')
    if full_attention_mask is not None: full_attention_mask = full_attention_mask.to(device)
    full_labels = batch['labels'].to(device)
    
    B, T = full_input_ids.shape
    h_state, l_state, prev_ctx, target_ctx, drift_state, ltm_state = running_states
    
    autocast_device = 'cpu' if is_directml_device(device) else device.type
    
    # --- CRITICAL FIX: STATE RESET PER BATCH (Conditional) ---
    # Default behavior (persist_state=False): reset states at start of each batch
    # This matches the original hierarchos.py training loop
    if not getattr(args, 'persist_state', False):
        h_state = None
        l_state = None
        prev_ctx = None
        target_ctx = None
        drift_state = None
        ltm_state = None
        model.reset_memory()  # Reset LTM working memory
    else:
        # If persisting, we detach states to prevent backprop through batches (TBPTT limited to batch)
        if h_state is not None: h_state = h_state.detach()
        if l_state is not None: l_state = l_state.detach()
        if prev_ctx is not None: prev_ctx = prev_ctx.detach()
        if target_ctx is not None: target_ctx = target_ctx.detach()
        if drift_state is not None: drift_state = drift_state.detach()
        if ltm_state is not None: 
            ltm_state = (ltm_state[0].detach(), ltm_state[1].detach())
    
    # Temporal chunking (critical for RWKV-based models)
    chunk_size = getattr(args, 'training_chunk_size', 128)
    if chunk_size <= 0 or chunk_size > T: chunk_size = T
    num_chunks = math.ceil(T / chunk_size)
    
    total_loss = 0.0
    total_ponder = 0.0
    total_commit = 0.0
    chunks_processed = 0
    final_outputs = None
    
    try:
        for chunk_idx in range(num_chunks):
            start_t = chunk_idx * chunk_size
            end_t = min((chunk_idx + 1) * chunk_size, T)
            
            # Slice tensors for this chunk
            input_ids = full_input_ids[:, start_t:end_t]
            attention_mask = full_attention_mask[:, start_t:end_t] if full_attention_mask is not None else None
            labels = full_labels[:, start_t:end_t]
            
            with autocast(device_type=autocast_device, enabled=args.amp):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    h_state=h_state, l_state=l_state, prev_context=prev_ctx,
                    target_context=target_ctx, drift_state=drift_state, ltm_memory_state=ltm_state,
                    global_pos_offset=start_t
                )
                
                ce_loss = outputs['loss']
                ponder_cost = outputs.get('ponder_cost')
                commitment_cost = outputs.get('commitment_cost')
                
                # Combine loss
                combined_loss = ce_loss
                if ponder_cost is not None:
                    combined_loss = combined_loss + (getattr(args, 'ponder_loss_weight', 0.01) * ponder_cost)
                if commitment_cost is not None:
                    combined_loss = combined_loss + (getattr(args, 'commitment_loss_weight', 0.5) * commitment_cost)
                
                chunk_loss = combined_loss / accumulation_steps
            
            if not validate_loss(chunk_loss): 
                continue
            
            # Backward per chunk (TBPTT)
            if scaler is not None: scaler.scale(chunk_loss).backward()
            else: chunk_loss.backward()
            
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
            if outputs.get('ltm_memory_state') is not None:
                ltm_state = (outputs['ltm_memory_state'][0].detach(), outputs['ltm_memory_state'][1].detach())
            
            # Accumulate for display
            total_loss += ce_loss.item()
            if ponder_cost is not None: total_ponder += ponder_cost.item()
            if commitment_cost is not None: total_commit += commitment_cost.item()
            chunks_processed += 1
            final_outputs = outputs
        
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
        avg_outputs = {
            'loss': torch.tensor(total_loss / chunks_processed),
            'ponder_cost': torch.tensor(total_ponder / chunks_processed) if total_ponder > 0 else None,
            'commitment_cost': torch.tensor(total_commit / chunks_processed) if total_commit > 0 else None,
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

    model = None
    optimizer = None
    start_epoch = 0
    scaler = None
    scheduler = None
    use_amp = getattr(args, 'amp', False)
    
    # 1. Loading/Resuming Logic
    if args.resume_from_ckpt:
        print(f"Resuming from checkpoint: {args.resume_from_ckpt}")
        checkpoint = torch.load(args.resume_from_ckpt, map_location='cpu', weights_only=False)
        
        saved_config = checkpoint.get('config', {})
        model_config = AttrDict(saved_config)
        state_dict = checkpoint['model_state_dict']
        
        # 1. Detect vocab_size and context_dim from tok_emb/lm_head
        if 'tok_emb.weight' in state_dict:
            model_config.vocab_size = state_dict['tok_emb.weight'].shape[0]
            model_config.context_dim = state_dict['tok_emb.weight'].shape[1]
        elif 'lm_head.weight' in state_dict:
            model_config.vocab_size = state_dict['lm_head.weight'].shape[0]
            model_config.context_dim = state_dict['lm_head.weight'].shape[1]
        
        # 2. Detect persistent_dim
        if 'persistent' in state_dict:
            model_config.persistent_dim = state_dict['persistent'].shape[0]
        
        # 3. Detect LTM dims
        if 'val_proj.weight' in state_dict:
            model_config.ltm_val_dim = state_dict['val_proj.weight'].shape[0]
        if 'qproj.weight' in state_dict:
            model_config.ltm_key_dim = state_dict['qproj.weight'].shape[0]

        # 4. Detect RNN hidden sizes (RWKV-style states)
        if 'h_rnn.time_decay' in state_dict:
            model_config.h_hidden = state_dict['h_rnn.time_decay'].shape[-1]
        if 'l_rnn.time_decay' in state_dict:
            model_config.l_hidden = state_dict['l_rnn.time_decay'].shape[-1]

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
        
        if is_directml_device(device): optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        
        if not getattr(args, 'override_scheduling', False) and 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except: print("Warning: Could not load optimizer state.")
        
        # Original script uses 'completed_epoch', modular uses 'epoch', check both for compatibility
        start_epoch = checkpoint.get('completed_epoch', checkpoint.get('epoch', 0))
        print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}.")
        if use_amp:
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
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        
        if use_amp: scaler = GradScaler()
    
    else:
        print("Starting training from scratch.")
        if 'vocab_size' not in config: config.vocab_size = len(tokenizer)
        model = HierarchosCore(config).to(device)
        if is_directml_device(device): optimizer = DirectMLAdamW(model.parameters(), lr=args.starting_lr)
        else: optimizer = torch.optim.AdamW(model.parameters(), lr=args.starting_lr)
        if use_amp: scaler = GradScaler()

    # Compile
    model.compile()

    # Scheduler
    num_update_steps = (dataloader_len // args.accumulation_steps) * args.epochs
    if not getattr(args, 'disable_lr_schedule', False) and num_update_steps > 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
        if args.resume_from_ckpt and not getattr(args, 'override_scheduling', False) and 'scheduler_state_dict' in checkpoint:
            try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except: pass

    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        running_states = (None, None, None, None, None, None)
        
        for step, batch in enumerate(pbar):
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
        
        save_checkpoint_safely({
            'completed_epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'config': dict(model.config),
        }, os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}.pt"))

def finetune(args, model, tokenizer, dataloader):
    pass
