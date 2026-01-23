"""
Hierarchos Chat Module - Ported from monolith for feature parity.

Features:
- Online learning with positive/negative feedback
- LTM updates with gradient-based memory system
- Proper top-k/top-p sampling
- Slash commands (/reset, /reset_ltm, /status, /settings, /temp, /topk, /topp, /filter, /learn)
- State persistence (drift_state, RWKV states, global_pos_offset)
- Save on exit logic
- Interrupt handling (Ctrl+C)
- Quantized model support with shadow model for learning
"""

import os
import sys
import signal
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..utils.device import is_directml_device
from ..utils.checkpoint import load_full_model_with_config

# --- Signal Handling for Interrupt ---
_interrupt_flag = False
_original_sigint_handler = None

def _handle_interrupt(sig, frame):
    """Sets the interrupt flag when SIGINT (Ctrl+C) is received."""
    global _interrupt_flag
    if not _interrupt_flag:
        print("\n[Interrupt received. Finishing current generation... Press Ctrl+C again to force exit.]", flush=True)
        _interrupt_flag = True
    else:
        print("\n[Forcing exit...]", flush=True)
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)
        sys.exit(1)


# --- Feedback Detection ---
def is_positive_feedback(text: str) -> bool:
    """Checks if the user input looks like positive validation."""
    text = text.lower().strip()
    positive_triggers = {
        "good", "great", "correct", "yes", "nice", "cool", "perfect",
        "thanks", "thx", "+", "right", "accurate"
    }
    if text in positive_triggers:
        return True
    first_word = text.split(' ')[0] if ' ' in text else text
    first_word = ''.join(c for c in first_word if c.isalnum())
    return first_word in positive_triggers or text.startswith("/learn")


def is_correction_or_instruction(text: str) -> bool:
    """Checks if user input looks like a correction or new instruction."""
    text_lower = text.lower().strip()
    correction_triggers = ["no", "wrong", "incorrect", "actually", "false", "not true"]
    for trigger in correction_triggers:
        if text_lower.startswith(trigger):
            return len(text.split()) > 3
    word_count = len(text.split())
    if word_count > 5 and not is_positive_feedback(text):
        return True
    return False


# --- Simple Generation Helper ---
def generate_sample(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    """Simple generation for testing/comparison."""
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    h_state, l_state, p_ctx, t_ctx = None, None, None, None
    ltm_state = None

    generated = tokens
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=tokens, h_state=h_state, l_state=l_state,
                            prev_context=p_ctx, target_context=t_ctx, ltm_memory_state=ltm_state)
            logits = outputs['logits'][:, -1, :] / max(temperature, 1e-6)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            tokens = next_token
            h_state, l_state = outputs['h_state'], outputs['l_state']
            p_ctx, t_ctx = outputs['prev_context'], outputs['target_context']
            ltm_state = outputs.get('ltm_memory_state')
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated[0], skip_special_tokens=True)


# --- Main Chat Function ---
def chat(args, device, tokenizer):
    """
    Interactive chat mode with online learning support.
    
    Ported from hierarchos.py monolith for full feature parity.
    """
    print("Running in CHAT mode...")
    
    # Import here to avoid circular imports
    from ..models.core import HierarchosCore
    from ..models.ltm import LTMModule
    
    # Try importing quantized model support
    try:
        from ..models.quantized import QuantizedHierarchos, load_quantized
        _HAS_QUANTIZED = True
    except ImportError:
        _HAS_QUANTIZED = False
    
    # =================================================================
    # 1. SETUP & SIGNAL HANDLING
    # =================================================================
    global _interrupt_flag, _original_sigint_handler
    _interrupt_flag = False
    _original_sigint_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, _handle_interrupt)
    except ValueError as e:
        print(f"Warning: Could not set SIGINT handler: {e}")
        _original_sigint_handler = None

    model = None
    shadow_model = None
    config = None
    is_quantized = False
    inference_device = device
    ltm_has_been_updated = False
    pending_training_data = None

    # =================================================================
    # 2. MODEL LOADING
    # =================================================================
    if not args.model_path or not os.path.isdir(args.model_path):
        print(f"Error: Model directory not found or invalid at {args.model_path}")
        sys.exit(1)

    try:
        npz_files = [f for f in os.listdir(args.model_path) if f.endswith('.npz')]
    except FileNotFoundError:
        print(f"Error: Model directory not found at {args.model_path}")
        sys.exit(1)

    if npz_files and _HAS_QUANTIZED:
        try:
            model, config = load_quantized(args.model_path, device=device)
            if isinstance(model, QuantizedHierarchos):
                is_quantized = True
                print(f"Loaded quantized model with {model.qtype} weights.")
                inference_device = "cpu"  # Quantized models run on CPU by default
            else:
                print("INFO: Loaded full-precision model (fallback active).")
                is_quantized = False
        except Exception as e:
            print(f"Error loading quantized model: {e}")
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            model, config = load_full_model_with_config(args.model_path, device)
            print("Loaded full-precision model.")
        except Exception as e:
            print(f"Error loading model from {args.model_path}: {e}")
            traceback.print_exc()
            sys.exit(1)

    # Handle shadow model for quantized learning
    enable_quantized_learning = getattr(args, 'enable_quantized_learning', False)
    shadow_model_path = getattr(args, 'shadow_model_path', None)
    
    if enable_quantized_learning and is_quantized:
        if not shadow_model_path:
            print("Error: --enable-quantized-learning requires --shadow-model-path")
            sys.exit(1)
        print("Loading full-precision 'shadow' model for online learning...")
        try:
            shadow_model, _ = load_full_model_with_config(shadow_model_path, device)
            shadow_model.ltm.load_state_dict(model.ltm.state_dict())
            shadow_model.eval()
        except Exception as e:
            print(f"Error loading shadow model: {e}")
            traceback.print_exc()
            sys.exit(1)

    # =================================================================
    # 3. LTM & OPTIMIZER SETUP
    # =================================================================
    ltm_lora_path = getattr(args, 'ltm_lora_path', None)
    learning_enabled = not is_quantized or enable_quantized_learning
    
    if ltm_lora_path and learning_enabled:
        print(f"LTM online learning is ACTIVE. Updates will be stored at: {ltm_lora_path}")
        updatable_model = shadow_model if is_quantized else model
        if updatable_model and hasattr(updatable_model.ltm, 'accumulate_deltas'):
            updatable_model.ltm.accumulate_deltas = True
            if os.path.exists(ltm_lora_path):
                print("Loading existing LTM deltas...")
                try:
                    deltas = torch.load(ltm_lora_path, weights_only=True)
                    updatable_model.ltm.vals.data.add_(deltas.to(updatable_model.ltm.vals.device))
                except Exception as e:
                    print(f"Warning: Failed to load LTM deltas: {e}")
    elif learning_enabled:
        print("LTM online learning is ACTIVE. Updates will modify model weights directly.")

    if not is_quantized:
        model.eval()

    # LTM Scheduler setup
    ltm_scheduler = None
    static_ltm_lr = getattr(args, 'static_ltm_lr', True)
    ltm_lr = getattr(args, 'ltm_lr', 0.001)
    
    if not static_ltm_lr and learning_enabled:
        print("INFO: Using Cosine Annealing schedule for LTM updates.")
        ltm_schedule_steps = getattr(args, 'ltm_schedule_steps', 100)
        ltm_schedule_min_lr = getattr(args, 'ltm_schedule_min_lr', 1e-5)
        dummy_param = nn.Parameter(torch.tensor(0.0))
        ltm_optimizer = torch.optim.SGD([dummy_param], lr=ltm_lr)
        ltm_scheduler = CosineAnnealingLR(ltm_optimizer, T_max=ltm_schedule_steps, eta_min=ltm_schedule_min_lr)

    # AMP Setup
    use_amp = getattr(args, 'amp', False) and learning_enabled and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    dummy_optimizer = None
    if use_amp:
        dummy_param_amp = nn.Parameter(torch.tensor(0.0)).to(device)
        dummy_optimizer = torch.optim.SGD([dummy_param_amp], lr=1.0)
        print("INFO: AMP ENABLED for online learning.")

    # =================================================================
    # 4. LOCAL HELPER FOR LTM UPDATE
    # =================================================================
    def perform_ltm_update(input_ids_tensor, label_ids_tensor, source_id, penalty=False, lr_override=None, silent=False):
        """Performs LTM update. Returns loss value if successful, else None."""
        nonlocal ltm_has_been_updated
        
        update_model = shadow_model if is_quantized else model
        if update_model is None:
            if not silent:
                print(" (No updatable model available)")
            return None
            
        target_device = device

        update_model.train()
        with torch.enable_grad():
            full_sequence = torch.cat([input_ids_tensor, label_ids_tensor], dim=0).unsqueeze(0)
            labels = torch.cat([torch.full_like(input_ids_tensor, -100), label_ids_tensor], dim=0).unsqueeze(0)
            
            max_length = getattr(config, 'max_length', 1024)
            if full_sequence.shape[1] > max_length:
                full_sequence = full_sequence[:, -max_length:]
                labels = labels[:, -max_length:]

            if use_amp and dummy_optimizer:
                dummy_optimizer.zero_grad(set_to_none=True)
            update_model.zero_grad(set_to_none=True)

            autocast_device = 'cpu' if is_directml_device(target_device) else target_device.type
            with autocast(device_type=autocast_device, enabled=use_amp):
                outputs = update_model(input_ids=full_sequence, labels=None)
                logits = outputs["logits"]
                
                if outputs.get("raw_topk_vals") is not None:
                    for t in outputs["raw_topk_vals"]:
                        if t.requires_grad:
                            t.retain_grad()

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                flat_logits = shift_logits.view(-1, config.vocab_size)
                flat_labels = shift_labels.view(-1)
                valid_mask = flat_labels != -100
                active_logits = flat_logits[valid_mask]
                active_labels = flat_labels[valid_mask]

                if penalty:
                    probs = F.softmax(active_logits, dim=-1)
                    target_probs = torch.gather(probs, 1, active_labels.unsqueeze(1)).squeeze(1)
                    target_probs = torch.clamp(target_probs, min=1e-7, max=1.0 - 1e-7)
                    loss = -torch.log(1.0 - target_probs).mean()
                else:
                    loss = F.cross_entropy(active_logits, active_labels)

            if use_amp and scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Extract and apply LTM gradients
            ltm_grads = None
            if outputs.get("raw_topk_vals") is not None:
                grads_list = [t.grad for t in outputs["raw_topk_vals"]]
                if any(g is not None for g in grads_list):
                    cleaned_grads = []
                    for i, g in enumerate(grads_list):
                        if g is None:
                            cleaned_grads.append(torch.zeros_like(outputs["raw_topk_vals"][i]))
                        else:
                            cleaned_grads.append(g)
                    ltm_grads = torch.stack(cleaned_grads, dim=1)

            if ltm_grads is not None:
                ltm_grads_copy = ltm_grads.detach().clone()
                
                current_ltm_lr = lr_override if lr_override is not None else ltm_lr
                if ltm_scheduler and lr_override is None:
                    current_ltm_lr = ltm_scheduler.get_last_lr()[0]
                    ltm_scheduler.step()

                if use_amp and scaler:
                    current_scale = scaler.get_scale()
                    if current_scale != 1.0:
                        ltm_grads_copy = ltm_grads_copy / current_scale

                update_model.ltm.inner_update(
                    outputs["topk_idx"],
                    ltm_grads_copy,
                    current_lr=current_ltm_lr,
                    timestamp=0.0,
                    source=source_id,
                    tokens_covered=full_sequence.shape[1],
                    inplace=True
                )
                ltm_has_been_updated = True

                if use_amp and scaler and dummy_optimizer:
                    scaler.unscale_(dummy_optimizer)
                    scaler.step(dummy_optimizer)
                    scaler.update()

                update_model.zero_grad(set_to_none=True)

                if is_quantized and shadow_model:
                    model.ltm.load_state_dict(update_model.ltm.state_dict())

                if penalty:
                    if not silent:
                        print(f" Done. (Unlikelihood | Loss: {loss.item():.3f})")
                else:
                    if not silent:
                        print(f" Done. (Reinforced | Loss: {loss.item():.3f})")
                return loss.item()
            else:
                if not silent:
                    print(" (No LTM gradients generated)")
                return None

        update_model.eval()
        return None

    # =================================================================
    # 5. PRINT WELCOME MESSAGE
    # =================================================================
    print("\nWelcome to Hierarchos Chat. Type 'exit' or 'quit' to end.")
    print("Commands:")
    print("  /filter time=-<seconds> | /filter source=<id>  : Constrain memory retrieval")
    print("  /settings | /topk <int> | /topp <float>        : View/Change sampling")
    print("  /reset                                         : Clear RNN & Hierarchical states")
    print("  /reset_ltm                                     : Clear LTM memory (fast_vals)")
    print("  /status                                        : Show model state info")
    print("Press Ctrl+C to stop generation at any time.")
    print("=" * 50)

    try:
        min_ts_filter = 0.0
        source_id_filter = None

        # =================================================================
        # 6. STATE INITIALIZATION
        # =================================================================
        rnn_device = "cpu" if is_quantized else device
        h_hidden = getattr(config, 'h_hidden', config.context_dim)
        l_hidden = getattr(config, 'l_hidden', config.context_dim)
        context_dim = config.context_dim

        h_state = torch.zeros(1, h_hidden, 5, device=rnn_device)
        h_state[:, :, 3] = -1e30  # PP init
        l_state = torch.zeros(1, l_hidden, 5, device=rnn_device)
        l_state[:, :, 3] = -1e30  # PP init
        
        prev_context = torch.zeros(1, context_dim, device=rnn_device)
        target_context = torch.zeros(1, context_dim, device=rnn_device)
        drift_state = torch.zeros(1, context_dim, device=rnn_device)
        
        total_tokens_generated = 0

        # =================================================================
        # 7. MAIN CHAT LOOP
        # =================================================================
        while True:
            _interrupt_flag = False
            try:
                prompt = input(">>> ")
            except EOFError:
                print("\n[EOF detected. Exiting chat.]")
                break

            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit"]:
                break

            # --- Slash Commands ---
            if prompt.startswith('/filter'):
                parts = prompt.split()
                if len(parts) >= 2:
                    for part in parts[1:]:
                        if part.startswith("time="):
                            try:
                                min_ts_filter = float(part.split("=")[1])
                                print(f"Set time filter to {min_ts_filter}")
                            except:
                                print("Usage: /filter time=<float>")
                        elif part.startswith("source="):
                            try:
                                source_id_filter = int(part.split("=")[1])
                                print(f"Set source filter to {source_id_filter}")
                            except:
                                print("Usage: /filter source=<int>")
                continue

            if prompt.startswith('/temp'):
                try:
                    val = float(prompt.split()[1])
                    args.temperature = max(0.0, val)
                    print(f"Set temperature to {args.temperature}")
                except (IndexError, ValueError):
                    print("Usage: /temp <float>")
                continue

            if prompt.startswith('/topk'):
                try:
                    val = int(prompt.split()[1])
                    args.top_k = max(0, val)
                    print(f"Set top_k to {args.top_k}")
                except (IndexError, ValueError):
                    print("Usage: /topk <int>")
                continue

            if prompt.startswith('/topp'):
                try:
                    val = float(prompt.split()[1])
                    args.top_p = max(0.0, min(1.0, val))
                    print(f"Set top_p to {args.top_p}")
                except (IndexError, ValueError):
                    print("Usage: /topp <float>")
                continue

            if prompt.startswith('/settings'):
                print(f"Current Settings:\n  Temperature: {args.temperature}\n  Top-K: {args.top_k}\n  Top-P: {args.top_p}")
                continue

            if prompt.startswith('/reset_ltm'):
                print("Resetting LTM memory...")
                if hasattr(model, 'ltm') and hasattr(model.ltm, 'reset_working_memory'):
                    model.ltm.reset_working_memory()
                if is_quantized and shadow_model and hasattr(shadow_model.ltm, 'reset_working_memory'):
                    shadow_model.ltm.reset_working_memory()
                ltm_has_been_updated = True
                print("LTM Reset complete.")
                continue

            if prompt.startswith('/reset'):
                print("Resetting all RNN and Hierarchical states...")
                h_state.zero_()
                h_state[:, :, 3] = -1e30
                l_state.zero_()
                l_state[:, :, 3] = -1e30
                prev_context.zero_()
                target_context.zero_()
                drift_state.zero_()
                total_tokens_generated = 0
                print("State Reset complete. Model is now fresh.")
                continue

            if prompt.startswith('/status'):
                print(f"Model Status:")
                print(f"  Total Tokens Generated: {total_tokens_generated}")
                print(f"  Device: {device}")
                print(f"  Quantized: {is_quantized}")
                print(f"  LTM Learning: {'ACTIVE' if learning_enabled else 'OFF'}")
                continue

            # =================================================================
            # A. CHECK FOR FEEDBACK & PERFORM UPDATES
            # =================================================================
            if learning_enabled:
                if is_positive_feedback(prompt) and pending_training_data is not None:
                    print("[Positive feedback. Reinforcing previous memory...]", end="", flush=True)
                    perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_USER_INTERACTION,
                        penalty=False
                    )
                    pending_training_data = None
                    continue

                elif prompt.strip().lower() in ["no", "n", "bad", "wrong", "bad bot"]:
                    if pending_training_data is not None:
                        print("[Negative feedback. Minimizing probability of previous output...]", end="", flush=True)
                        perform_ltm_update(
                            pending_training_data['prompt_ids'][0],
                            pending_training_data['response_ids'],
                            LTMModule.SRC_USER_INTERACTION,
                            penalty=True
                        )

            if prompt.strip() == "/learn" and pending_training_data:
                print("[Manual learn command. Reinforcing previous...]", end="", flush=True)
                perform_ltm_update(
                    pending_training_data['prompt_ids'][0],
                    pending_training_data['response_ids'],
                    LTMModule.SRC_USER_INTERACTION,
                    penalty=False
                )
                continue
            elif prompt.strip() == "/learn":
                print("[Nothing pending to learn]")
                continue

            # =================================================================
            # B. GENERATION LOGIC
            # =================================================================
            prompt_format = f"### Instruction:\n{prompt}\n\n### Response:\n"
            prompt_ids = tokenizer.encode(prompt_format, return_tensors="pt").to(device)

            print("\nhierarchos: ", end="", flush=True)
            response_ids = []

            # 1. PREFILL PASS
            with torch.no_grad():
                model_input_ids = prompt_ids.cpu() if is_quantized else prompt_ids.to(device)
                
                if is_quantized:
                    outputs = model(
                        input_ids=model_input_ids,
                        h_state=h_state.cpu(),
                        l_state=l_state.cpu(),
                        prev_context=prev_context.cpu(),
                        target_context=target_context.cpu(),
                        drift_state=drift_state.cpu(),
                        global_pos_offset=total_tokens_generated,
                        device=inference_device,
                        min_timestamp=min_ts_filter,
                        source_filter=source_id_filter
                    )
                    h_state = outputs['h_state']
                    l_state = outputs['l_state']
                    prev_context = outputs['prev_context']
                    target_context = outputs['target_context']
                    drift_state = outputs.get('drift_state', drift_state)
                else:
                    outputs = model(
                        model_input_ids.to(device),
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        global_pos_offset=total_tokens_generated,
                        min_timestamp=min_ts_filter,
                        source_filter=source_id_filter
                    )
                    if outputs.get('h_state') is not None:
                        h_state = outputs['h_state']
                    if outputs.get('l_state') is not None:
                        l_state = outputs['l_state']
                    if outputs.get('drift_state') is not None:
                        drift_state = outputs['drift_state']
                    if outputs.get('prev_context') is not None:
                        prev_context = outputs['prev_context']
                    if outputs.get('target_context') is not None:
                        target_context = outputs['target_context']

                logits = outputs["logits"].to(device)
                next_token_logits = logits[:, -1, :]

                # Apply repetition penalty
                rep_penalty = getattr(args, 'repetition_penalty', 1.2)
                if rep_penalty != 1.0 and len(response_ids) > 0:
                    for prev_token in set(response_ids):
                        if next_token_logits[0, prev_token] > 0:
                            next_token_logits[0, prev_token] /= rep_penalty
                        else:
                            next_token_logits[0, prev_token] *= rep_penalty

                # Apply sampling
                if args.temperature > 0:
                    next_token_logits = next_token_logits / args.temperature
                    if args.top_k > 0:
                        v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                    if args.top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > args.top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = -float('Inf')

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                total_tokens_generated += prompt_ids.shape[1]

                if next_token_id.item() != tokenizer.eos_token_id:
                    response_ids.append(next_token_id.item())
                    decoded_token = tokenizer.decode([next_token_id.item()])
                    print(decoded_token, end="", flush=True)
                    current_ids = next_token_id
                else:
                    current_ids = None

            # 2. INCREMENTAL GENERATION LOOP
            max_new_tokens = getattr(args, 'max_new_tokens', 512)
            if current_ids is not None:
                with torch.no_grad():
                    for i in range(max_new_tokens - 1):
                        if _interrupt_flag:
                            _interrupt_flag = False
                            print("\n[Generation interrupted by user.]", end="", flush=True)
                            break

                        model_input_ids = current_ids.cpu() if is_quantized else current_ids.to(device)

                        if is_quantized:
                            outputs = model(
                                input_ids=model_input_ids,
                                h_state=h_state.cpu(),
                                l_state=l_state.cpu(),
                                prev_context=prev_context.cpu(),
                                target_context=target_context.cpu(),
                                drift_state=drift_state.cpu(),
                                global_pos_offset=total_tokens_generated,
                                device=inference_device,
                                min_timestamp=min_ts_filter,
                                source_filter=source_id_filter
                            )
                            h_state = outputs['h_state']
                            l_state = outputs['l_state']
                            prev_context = outputs['prev_context']
                            target_context = outputs['target_context']
                            drift_state = outputs.get('drift_state', drift_state)
                        else:
                            outputs = model(
                                model_input_ids.to(device),
                                h_state=h_state,
                                l_state=l_state,
                                prev_context=prev_context,
                                target_context=target_context,
                                drift_state=drift_state,
                                global_pos_offset=total_tokens_generated,
                                min_timestamp=min_ts_filter,
                                source_filter=source_id_filter
                            )
                            if outputs.get('h_state') is not None:
                                h_state = outputs['h_state']
                            if outputs.get('l_state') is not None:
                                l_state = outputs['l_state']
                            if outputs.get('drift_state') is not None:
                                drift_state = outputs['drift_state']
                            if outputs.get('prev_context') is not None:
                                prev_context = outputs['prev_context']
                            if outputs.get('target_context') is not None:
                                target_context = outputs['target_context']

                        logits = outputs["logits"].to(device)
                        next_token_logits = logits[:, -1, :]

                        # Apply repetition penalty
                        rep_penalty = getattr(args, 'repetition_penalty', 1.2)
                        if rep_penalty != 1.0 and len(response_ids) > 0:
                            for prev_token in set(response_ids):
                                if next_token_logits[0, prev_token] > 0:
                                    next_token_logits[0, prev_token] /= rep_penalty
                                else:
                                    next_token_logits[0, prev_token] *= rep_penalty

                        if args.temperature > 0:
                            next_token_logits = next_token_logits / args.temperature
                            if args.top_k > 0:
                                v, _ = torch.topk(next_token_logits, min(args.top_k, next_token_logits.size(-1)))
                                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                            if args.top_p < 1.0:
                                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                                sorted_indices_to_remove = cumulative_probs > args.top_p
                                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                                sorted_indices_to_remove[..., 0] = 0
                                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                                next_token_logits[indices_to_remove] = -float('Inf')

                            probs = F.softmax(next_token_logits, dim=-1)
                            next_token_id = torch.multinomial(probs, num_samples=1)
                        else:
                            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

                        if next_token_id.item() == tokenizer.eos_token_id:
                            break

                        response_ids.append(next_token_id.item())
                        try:
                            decoded_token = tokenizer.decode([next_token_id.item()])
                        except Exception as e:
                            decoded_token = ""

                        if "###" in decoded_token and len(decoded_token) <= 5:
                            break

                        print(decoded_token, end="", flush=True)
                        current_ids = next_token_id
                        total_tokens_generated += 1

            print("\n")

            # =================================================================
            # C. BUFFER DATA FOR NEXT TURN
            # =================================================================
            if len(response_ids) > 0:
                pending_training_data = {
                    'prompt_ids': prompt_ids,
                    'response_ids': torch.tensor(response_ids, device=device)
                }
                
                # =================================================================
                # D. PASSIVE LEARNING (if enabled)
                # =================================================================
                passive_learning = getattr(args, 'passive_learning', False)
                passive_lr = getattr(args, 'passive_lr', 1e-5)
                surprise_threshold = getattr(args, 'surprise_threshold', 0.5)
                
                if passive_learning and learning_enabled:
                    # Compute loss first to check surprise threshold
                    loss_val = perform_ltm_update(
                        pending_training_data['prompt_ids'][0],
                        pending_training_data['response_ids'],
                        LTMModule.SRC_TRAINING_DATA,
                        penalty=False,
                        lr_override=passive_lr,
                        silent=True  # Don't print during passive updates
                    )
                    
                    if loss_val is not None:
                        if loss_val > surprise_threshold:
                            print(f"[Passive LTM update | Loss: {loss_val:.3f} > {surprise_threshold:.2f} threshold]")
                        else:
                            pass  # Below threshold - no indication needed

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C detected. Exiting chat.]")

    finally:
        # Restore original signal handler
        if _original_sigint_handler:
            signal.signal(signal.SIGINT, _original_sigint_handler)

        # =================================================================
        # 8. SAVE ON EXIT LOGIC
        # =================================================================
        MODEL_WEIGHTS_NAME = "hierarchos.pt"
        updatable_model = shadow_model if is_quantized and enable_quantized_learning else model
        can_update = updatable_model is not None and learning_enabled

        if can_update and ltm_lora_path and hasattr(updatable_model.ltm, 'accumulate_deltas') and updatable_model.ltm.accumulate_deltas:
            if hasattr(updatable_model.ltm, 'ltm_deltas') and torch.any(updatable_model.ltm.ltm_deltas != 0):
                print(f"\nSaving LTM memory deltas to {ltm_lora_path}...")
                try:
                    torch.save(updatable_model.ltm.ltm_deltas.cpu(), ltm_lora_path)
                    print("Deltas saved.")
                except Exception as e:
                    print(f"Error saving LTM deltas: {e}")
            else:
                print("\nNo new LTM updates to save as LoRA.")

        elif can_update and not ltm_lora_path and ltm_has_been_updated:
            if not is_quantized:
                while True:
                    try:
                        response = input(f"Do you want to save the learned LTM updates back to '{args.model_path}'? (y/n): ").lower()
                        if response in ["y", "yes"]:
                            print(f"\nSaving updated model to {args.model_path}...")
                            output_weights_path = os.path.join(args.model_path, MODEL_WEIGHTS_NAME)
                            try:
                                torch.save({
                                    'model_state_dict': model.state_dict(),
                                    'config': dict(model.config)
                                }, output_weights_path)
                                print("Save complete.")
                            except Exception as e:
                                print(f"Error saving model: {e}")
                            break
                        elif response in ["n", "no"]:
                            print("Changes will be discarded. Exiting.")
                            break
                        else:
                            print("Invalid input.")
                    except EOFError:
                        print("\nEOF detected. Assuming 'no' for saving.")
                        break

        elif ltm_has_been_updated:
            print("\n[Warning] LTM was updated, but no valid save configuration was found. Changes lost.")
