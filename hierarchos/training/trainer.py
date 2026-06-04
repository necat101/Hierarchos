import os
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import random
from tqdm import tqdm
import sys
import traceback
import numpy as np
import math

from .optimizers import DirectMLAdamW
from ..utils.device import is_directml_device, set_threads
from ..utils.checkpoint import (
    TRANSIENT_LTM_STATE_KEYS,
    save_checkpoint_safely,
    load_full_model_with_config,
    sanitize_model_state_dict,
)
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

def _tensor_is_nonfinite(tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return False
    return not bool(torch.isfinite(tensor).all().item())

def _describe_tensor_issue(name: str, tensor: torch.Tensor) -> str:
    detached = tensor.detach()
    nan_count = int(torch.isnan(detached).sum().item())
    inf_count = int(torch.isinf(detached).sum().item())
    finite = detached[torch.isfinite(detached)]
    if finite.numel() > 0:
        finite_min = float(finite.min().item())
        finite_max = float(finite.max().item())
        return f"{name} has {nan_count} NaN and {inf_count} Inf values; finite range=[{finite_min:.4e}, {finite_max:.4e}]"
    return f"{name} has {nan_count} NaN and {inf_count} Inf values; no finite values"

def _is_transient_ltm_state_name(name: str) -> bool:
    clean_name = str(name).replace("_orig_mod.", "")
    return any(clean_name.endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS)

def _find_first_nonfinite_model_tensor(model, include_grads: bool = False, include_transient_ltm: bool = False):
    for name, param in model.named_parameters():
        if _tensor_is_nonfinite(param):
            return _describe_tensor_issue(f"parameter {name}", param)
        if include_grads and param.grad is not None and _tensor_is_nonfinite(param.grad):
            return _describe_tensor_issue(f"gradient {name}", param.grad)
    for name, buffer in model.named_buffers():
        if not include_transient_ltm and _is_transient_ltm_state_name(name):
            continue
        if _tensor_is_nonfinite(buffer):
            return _describe_tensor_issue(f"buffer {name}", buffer)
    return None

def _find_first_nonfinite_optimizer_tensor(optimizer):
    if optimizer is None:
        return None
    for param_idx, state in enumerate(optimizer.state.values()):
        for key, value in state.items():
            if _tensor_is_nonfinite(value):
                return _describe_tensor_issue(f"optimizer state[{param_idx}].{key}", value)
    return None

def _find_first_nonfinite_payload_tensor(value, path: str = "checkpoint"):
    if _is_transient_ltm_state_name(path):
        return None
    if torch.is_tensor(value):
        if _tensor_is_nonfinite(value):
            return _describe_tensor_issue(path, value)
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            issue = _find_first_nonfinite_payload_tensor(item, f"{path}.{key}")
            if issue:
                return issue
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            issue = _find_first_nonfinite_payload_tensor(item, f"{path}[{idx}]")
            if issue:
                return issue
    return None

def _sanitize_payload_nonfinite_(value, path: str = "checkpoint", max_abs: float = 1.0) -> int:
    if torch.is_tensor(value):
        if path.endswith("[0]") and "running_states[5]" in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
        if path.endswith("[1]") and "running_states[5]" in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=max_abs, neginf=-max_abs)
        if ".model_state_dict." in path or ".grad_state_dict." in path:
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=1.0, neginf=-1.0)
        return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=1.0, neginf=-1.0)
    cleaned = 0
    if isinstance(value, dict):
        for key, item in value.items():
            cleaned += _sanitize_payload_nonfinite_(item, f"{path}.{key}", max_abs=max_abs)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            cleaned += _sanitize_payload_nonfinite_(item, f"{path}[{idx}]", max_abs=max_abs)
    return cleaned

def training_state_is_finite(model, optimizer=None, check_optimizer: bool = True, include_grads: bool = False) -> bool:
    issue = _find_first_nonfinite_model_tensor(model, include_grads=include_grads)
    if issue:
        print(f"CRITICAL: Non-finite training state detected: {issue}")
        return False
    if check_optimizer:
        issue = _find_first_nonfinite_optimizer_tensor(optimizer)
        if issue:
            print(f"CRITICAL: Non-finite training state detected: {issue}")
            return False
    return True

def _checkpoint_grad_clip(checkpoint_dict) -> float:
    config = checkpoint_dict.get("config") if isinstance(checkpoint_dict, dict) else None
    if isinstance(config, dict):
        try:
            return float(config.get("grad_clip", 1.0) or 1.0)
        except (TypeError, ValueError):
            return 1.0
    return 1.0

def _sanitize_ltm_payload_state_(value, path: str = "checkpoint", max_abs: float = 1.0) -> int:
    if torch.is_tensor(value):
        clean_path = path.replace("_orig_mod.", "")
        if clean_path.endswith("ltm.fast_vals"):
            if value.is_floating_point():
                changed = int(torch.count_nonzero(value).item())
                value.zero_()
                return changed
            return 0
        if clean_path.endswith("ltm._mom_vals"):
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=max_abs, neginf=-max_abs)
        if clean_path.endswith("ltm.timestamps"):
            return _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
        return 0
    cleaned = 0
    if isinstance(value, dict):
        for key, item in value.items():
            cleaned += _sanitize_ltm_payload_state_(item, f"{path}.{key}", max_abs=max_abs)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            cleaned += _sanitize_ltm_payload_state_(item, f"{path}[{idx}]", max_abs=max_abs)
    return cleaned

def _component_is_finite(value) -> bool:
    return value is not None and torch.is_tensor(value) and bool(torch.isfinite(value).all().item())

def _reset_after_nonfinite(optimizer, model=None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    if model is not None and hasattr(model, "reset_memory"):
        model.reset_memory()
    return (None, None, None, None, None, None)

def _sanitize_tensor_nonfinite_(tensor: torch.Tensor, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0) -> int:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return 0
    bad_count = int((~torch.isfinite(tensor)).sum().item())
    if bad_count:
        tensor.nan_to_num_(nan=nan, posinf=posinf, neginf=neginf)
    return bad_count

def _positive_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value > 0.0 else default

def _nonnegative_float(value, default: float = 0.0) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and value >= 0.0 else default

def _cap_loss_component_for_backward(value: torch.Tensor, ceiling: float) -> torch.Tensor:
    ceiling = _positive_float(ceiling, 0.0)
    if ceiling <= 0.0 or not torch.is_tensor(value):
        return value
    return torch.minimum(value, value.new_tensor(ceiling))

def _clamp_tensor_finite_magnitude_(tensor: torch.Tensor, max_abs: float) -> int:
    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return 0
    max_abs = _positive_float(max_abs, 0.0)
    if max_abs <= 0.0:
        return 0
    over = torch.abs(tensor) > max_abs
    over_count = int(over.sum().item())
    if over_count:
        tensor.clamp_(min=-max_abs, max=max_abs)
    return over_count

def _detach_finite_clamp(tensor: torch.Tensor, max_abs: float) -> torch.Tensor:
    max_abs = _positive_float(max_abs, 1.0)
    detached = tensor.detach()
    return torch.clamp(
        torch.nan_to_num(detached, nan=0.0, posinf=max_abs, neginf=-max_abs),
        min=-max_abs,
        max=max_abs,
    )

def _sanitize_optimizer_state_(optimizer) -> int:
    if optimizer is None:
        return 0
    cleaned = 0
    for state in optimizer.state.values():
        for value in state.values():
            cleaned += _sanitize_tensor_nonfinite_(value, nan=0.0, posinf=0.0, neginf=0.0)
    if cleaned:
        print(f"WARNING: Reset {cleaned} non-finite optimizer state value(s) to 0.0 for recovery resume.")
    return cleaned

def _sanitize_model_nonfinite_(model, *, include_transient_ltm: bool = False, log_prefix: str = "model") -> int:
    cleaned = 0
    first_issue = None
    with torch.no_grad():
        for name, param in model.named_parameters():
            if first_issue is None and _tensor_is_nonfinite(param):
                first_issue = _describe_tensor_issue(f"parameter {name}", param)
            cleaned += _sanitize_tensor_nonfinite_(param, nan=0.0, posinf=1.0, neginf=-1.0)
        for name, buffer in model.named_buffers():
            if not include_transient_ltm and _is_transient_ltm_state_name(name):
                continue
            if first_issue is None and _tensor_is_nonfinite(buffer):
                first_issue = _describe_tensor_issue(f"buffer {name}", buffer)
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=1.0, neginf=-1.0)
    if cleaned:
        detail = f" First repaired tensor: {first_issue}." if first_issue else ""
        print(f"WARNING: Sanitized {cleaned} non-finite {log_prefix} parameter/buffer value(s): NaN->0, +Inf->1, -Inf->-1.{detail}")
    return cleaned

def _clamp_model_finite_magnitude_(model, max_abs: float, *, include_transient_ltm: bool = False, log_prefix: str = "model") -> int:
    max_abs = _positive_float(max_abs, 0.0)
    if max_abs <= 0.0:
        return 0
    clamped = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            clamped += _clamp_tensor_finite_magnitude_(param, max_abs)
        for name, buffer in model.named_buffers():
            if not include_transient_ltm and _is_transient_ltm_state_name(name):
                continue
            clamped += _clamp_tensor_finite_magnitude_(buffer, max_abs)
    if clamped:
        print(f"WARNING: Clamped {clamped} finite {log_prefix} parameter/buffer value(s) to +/-{max_abs:g}.")
    return clamped

def _sanitize_model_transient_state_(model, max_abs: float = 1.0) -> int:
    cleaned = 0
    max_abs = float(max_abs or 0.0)
    if max_abs <= 0.0:
        max_abs = 1.0
    for name, buffer in model.named_buffers():
        clean_name = str(name).replace("_orig_mod.", "")
        if clean_name.endswith("ltm.fast_vals"):
            if torch.is_tensor(buffer) and buffer.is_floating_point():
                changed = int(torch.count_nonzero(buffer).item())
                if changed:
                    buffer.zero_()
                    cleaned += changed
        elif clean_name.endswith("ltm._mom_vals"):
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=max_abs, neginf=-max_abs)
        elif clean_name.endswith("ltm.timestamps"):
            cleaned += _sanitize_tensor_nonfinite_(buffer, nan=0.0, posinf=0.0, neginf=0.0)
        elif clean_name.endswith("ltm.sources"):
            if torch.is_tensor(buffer) and not bool(torch.isfinite(buffer.float()).all().item()):
                buffer.fill_(0)
                cleaned += int(buffer.numel())
    if cleaned:
        print(
            f"WARNING: Sanitized transient LTM state ({cleaned} value(s)): "
            "fast_vals reset; _mom_vals saturated; metadata reset."
        )
    return cleaned

def _sanitize_gradient_nonfinite_(model, max_abs: float) -> int:
    cleaned = 0
    clamped = 0
    max_abs = _positive_float(max_abs, 1.0)
    for param in model.parameters():
        if param.grad is not None:
            cleaned += _sanitize_tensor_nonfinite_(param.grad, nan=0.0, posinf=max_abs, neginf=-max_abs)
            clamped += _clamp_tensor_finite_magnitude_(param.grad, max_abs)
    if cleaned:
        print(f"WARNING: Sanitized {cleaned} non-finite gradient value(s): NaN->0, +Inf->{max_abs:g}, -Inf->{-max_abs:g} before clipping.")
    if clamped:
        print(f"WARNING: Saturated {clamped} finite gradient value(s) to +/-{max_abs:g} before global clipping.")
    return cleaned

def _find_first_nonfinite_gradient_tensor(model):
    for name, param in model.named_parameters():
        if param.grad is not None and _tensor_is_nonfinite(param.grad):
            return _describe_tensor_issue(f"gradient {name}", param.grad)
    return None

def _manual_clip_grad_norm_(params, max_norm: float):
    norms = []
    for param in params:
        grad = param.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce()._values()
        norms.append(grad.float().norm(2))
    if norms:
        total_norm = torch.stack(norms).norm(2)
    else:
        total_norm = torch.zeros(())
    if max_norm > 0.0:
        if not bool(torch.isfinite(total_norm).all().item()):
            for param in params:
                param.grad.detach().zero_()
            return total_norm
        clip_coef = max_norm / (float(total_norm.item()) + 1e-6)
        if clip_coef < 1.0:
            for param in params:
                param.grad.detach().mul_(clip_coef)
    return total_norm

def save_training_checkpoint_if_finite(checkpoint_dict, path: str, model, optimizer=None) -> bool:
    max_abs = _checkpoint_grad_clip(checkpoint_dict)
    _sanitize_model_nonfinite_(model, log_prefix="checkpoint model")
    if isinstance(checkpoint_dict, dict) and model is not None and "model_state_dict" in checkpoint_dict:
        checkpoint_dict["model_state_dict"] = sanitize_model_state_dict(model, reset_transient_ltm=False)
    _sanitize_model_transient_state_(model, max_abs=max_abs)
    _sanitize_gradient_nonfinite_(model, max_abs=1.0)
    _sanitize_optimizer_state_(optimizer)
    ltm_cleaned = _sanitize_ltm_payload_state_(checkpoint_dict, max_abs=max_abs)
    if ltm_cleaned:
        print(f"WARNING: Sanitized {ltm_cleaned} transient LTM checkpoint value(s) before saving.")
    issue = _find_first_nonfinite_payload_tensor(checkpoint_dict)
    if issue:
        cleaned = _sanitize_payload_nonfinite_(checkpoint_dict, max_abs=max_abs)
        print(f"WARNING: Non-finite checkpoint payload detected: {issue}")
        print(f"WARNING: Sanitized {cleaned} non-finite checkpoint payload value(s) before saving.")
    issue = _find_first_nonfinite_payload_tensor(checkpoint_dict)
    if issue:
        print(f"CRITICAL: Checkpoint still contains non-finite tensor after repair; refusing to save. {issue}")
        return False
    save_checkpoint_safely(checkpoint_dict, path)
    return True

def _clip_gradients_and_check(model, max_norm: float):
    params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return True, None
    max_norm = float(max_norm or 0.0)
    _sanitize_gradient_nonfinite_(model, max_abs=max_norm)
    total_norm = _manual_clip_grad_norm_(params, max_norm)
    if torch.is_tensor(total_norm) and not bool(torch.isfinite(total_norm).all().item()):
        _sanitize_gradient_nonfinite_(model, max_abs=max_norm)
        total_norm = _manual_clip_grad_norm_(params, max_norm)
    issue = _find_first_nonfinite_gradient_tensor(model)
    if issue:
        return False, issue
    return True, total_norm

def set_model_training_step(model, step: int):
    setter = getattr(model, "set_training_step", None)
    if callable(setter):
        setter(step)
        return
    base_model = getattr(model, "base_model", None)
    inner_model = getattr(base_model, "model", None)
    setter = getattr(inner_model, "set_training_step", None)
    if callable(setter):
        setter(step)

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
    dataloader_len = max(0, int(dataloader_len))
    start_step = max(0, min(int(start_step), dataloader_len))
    remaining_epochs_after_current = max(0, int(total_epochs) - int(start_epoch) - 1)
    updates_per_full_epoch = dataloader_len // accumulation_steps
    updates_already_done_this_epoch = start_step // accumulation_steps
    updates_left_this_epoch = max(0, updates_per_full_epoch - updates_already_done_this_epoch)
    remaining_updates = updates_left_this_epoch + (remaining_epochs_after_current * updates_per_full_epoch)
    return max(1, remaining_updates)

def _resolve_ltm_lr_bounds(args):
    max_lr = _positive_float(getattr(args, 'ltm_lr', 1e-3), 1e-3)
    min_ltm_lr = getattr(args, 'min_ltm_lr', None)
    if min_ltm_lr is None:
        min_lr = _nonnegative_float(getattr(args, 'min_lr', 0.0), 0.0)
    else:
        min_lr = _nonnegative_float(min_ltm_lr, 0.0)
    return max_lr, min(min_lr, max_lr)

def _cosine_annealed_value(max_value: float, min_value: float, step: int, total_steps: int) -> float:
    max_value = _nonnegative_float(max_value, 0.0)
    min_value = min(_nonnegative_float(min_value, 0.0), max_value)
    total_steps = max(1, int(total_steps or 1))
    step = max(0, min(int(step or 0), total_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * step / float(total_steps)))
    return min_value + (max_value - min_value) * cosine

def configure_ltm_lr_schedule(args, num_update_steps: int, checkpoint=None, *, override_schedule: bool = False, scheduler=None):
    max_lr, min_lr = _resolve_ltm_lr_bounds(args)
    schedule_enabled = not bool(getattr(args, 'disable_ltm_lr_schedule', False))
    total_steps = max(1, int(num_update_steps or 1))
    current_step = 0

    state = checkpoint.get('ltm_scheduler_state') if isinstance(checkpoint, dict) else None
    if schedule_enabled and state and not override_schedule:
        total_steps = max(1, int(state.get('total_steps', total_steps) or total_steps))
        current_step = max(0, int(state.get('step', 0) or 0))
    elif schedule_enabled and scheduler is not None and not override_schedule:
        current_step = max(0, int(getattr(scheduler, 'last_epoch', 0) or 0))

    args._ltm_lr_schedule_enabled = schedule_enabled
    args._ltm_lr_schedule_total_steps = total_steps
    args._ltm_lr_schedule_step = min(current_step, total_steps)
    args._ltm_lr_max = max_lr
    args._ltm_lr_min = min_lr
    args._current_ltm_lr = get_current_ltm_lr(args)
    if schedule_enabled:
        print(
            f"INFO: Cosine Annealing LTM LR scheduler ENABLED. "
            f"Total steps: {total_steps}, Max LTM LR: {max_lr:.2e}, Min LTM LR: {min_lr:.2e}"
        )
    else:
        print(f"INFO: LTM LR scheduler disabled. Fixed LTM LR: {max_lr:.2e}")
    return args._current_ltm_lr

def get_current_ltm_lr(args) -> float:
    max_lr = _positive_float(getattr(args, '_ltm_lr_max', getattr(args, 'ltm_lr', 1e-3)), 1e-3)
    min_lr = _nonnegative_float(getattr(args, '_ltm_lr_min', getattr(args, 'min_ltm_lr', 0.0) or 0.0), 0.0)
    if not bool(getattr(args, '_ltm_lr_schedule_enabled', True)):
        return max_lr
    return _cosine_annealed_value(
        max_lr,
        min_lr,
        getattr(args, '_ltm_lr_schedule_step', 0),
        getattr(args, '_ltm_lr_schedule_total_steps', 1),
    )

def advance_ltm_lr_schedule(args):
    if bool(getattr(args, '_ltm_lr_schedule_enabled', True)):
        total_steps = max(1, int(getattr(args, '_ltm_lr_schedule_total_steps', 1) or 1))
        current_step = max(0, int(getattr(args, '_ltm_lr_schedule_step', 0) or 0))
        args._ltm_lr_schedule_step = min(current_step + 1, total_steps)
    args._current_ltm_lr = get_current_ltm_lr(args)
    return args._current_ltm_lr

def capture_ltm_lr_scheduler_state(args):
    return {
        "enabled": bool(getattr(args, '_ltm_lr_schedule_enabled', True)),
        "step": int(getattr(args, '_ltm_lr_schedule_step', 0) or 0),
        "total_steps": int(getattr(args, '_ltm_lr_schedule_total_steps', 1) or 1),
        "max_lr": float(getattr(args, '_ltm_lr_max', getattr(args, 'ltm_lr', 1e-3))),
        "min_lr": float(getattr(args, '_ltm_lr_min', getattr(args, 'min_ltm_lr', 0.0) or 0.0)),
    }

def build_hierarchos_optimizer(model, args, device):
    """RWKV-style AdamW grouping: decay matrices/embeddings, never norms or scalars."""
    lr = args.starting_lr
    weight_decay = float(getattr(args, "rwkv_weight_decay", 0.1))
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (".weight" in name or "emb" in name) and ("ln" not in name and "norm" not in name):
            decay.append(param)
        else:
            no_decay.append(param)

    param_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    if is_directml_device(device):
        return DirectMLAdamW(param_groups, lr=lr)
    if device.type == 'cuda':
        return torch.optim.AdamW(param_groups, lr=lr, fused=True)
    return torch.optim.AdamW(param_groups, lr=lr)

def estimate_cuda_loss_chunk_rows(free_bytes: int, batch_size: int, chunk_size: int,
                                  vocab_size: int, requested_rows: int = 0) -> int:
    """
    Pick a CUDA lm_head loss chunk size from live free VRAM and current batch shape.

    On 96GB-class GPUs this targets enough rows to cover batch=64 at a
    256-token TBPTT chunk in one loss pass while still reserving most VRAM for
    activations, optimizer state, CUDA graphs, and fragmentation.
    """
    requested_rows = int(requested_rows or 0)
    if requested_rows > 0:
        return requested_rows

    free_bytes = max(0, int(free_bytes or 0))
    batch_size = max(1, int(batch_size or 1))
    chunk_size = max(1, int(chunk_size or 1))
    vocab_size = max(1, int(vocab_size or 1))

    free_gb = free_bytes / float(1024 ** 3)
    if free_gb >= 72.0:
        base_rows = 16834
    elif free_gb >= 48.0:
        base_rows = 12288
    elif free_gb >= 24.0:
        base_rows = 8192
    elif free_gb >= 12.0:
        base_rows = 4096
    else:
        base_rows = 2048

    batch_rows = batch_size * max(1, chunk_size - 1)
    batch_target_rows = int(math.ceil(batch_rows * 1.05))

    # FP32 logits dominate; reserve room for backward/temp buffers and leave most
    # free VRAM for activations, optimizer state, CUDA graphs, and fragmentation.
    estimated_bytes_per_row = vocab_size * 4 * 3
    memory_budget = max(512 * 1024 ** 2, int(free_bytes * 0.20))
    memory_cap_rows = max(512, memory_budget // max(1, estimated_bytes_per_row))

    rows = max(base_rows, min(batch_target_rows, memory_cap_rows))
    rows = min(rows, memory_cap_rows)
    return max(512, int(rows))

def tune_cuda_loss_chunk_rows_once(model, args, batch_size: int, chunk_size: int):
    """Auto-tune CUDA loss chunking once after startup/model allocation."""
    if not (torch.cuda.is_available() and getattr(args, 'cuda_chunked_lm_loss', True)):
        return
    if not getattr(args, '_auto_cuda_loss_chunk_rows', False):
        return

    device = next(model.parameters()).device
    if device.type != 'cuda':
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    vocab_size = int(getattr(model.config, 'vocab_size', getattr(args, 'vocab_size', 1)))
    rows = estimate_cuda_loss_chunk_rows(
        free_bytes=free_bytes,
        batch_size=batch_size,
        chunk_size=chunk_size,
        vocab_size=vocab_size,
    )

    previous = int(getattr(args, 'cuda_loss_chunk_rows', 0) or 0)
    if rows != previous:
        args.cuda_loss_chunk_rows = rows
        if hasattr(model, 'config'):
            model.config.cuda_loss_chunk_rows = rows
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        print(
            f"INFO: Startup CUDA loss chunk rows set to {rows} "
            f"(free VRAM {free_gb:.1f}/{total_gb:.1f} GB, batch={batch_size}, chunk={chunk_size})."
        )

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

def pad_training_batch_to_multiple(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor = None,
    multiple: int = 128,
    pad_token_id: int = 0,
):
    """Pad sequence length to a chunk multiple so torch.compile sees stable shapes."""
    multiple = int(multiple or 0)
    if multiple <= 1 or input_ids.ndim != 2 or labels.ndim != 2:
        return input_ids, labels, attention_mask
    T = input_ids.shape[1]
    target_T = int(math.ceil(T / multiple) * multiple)
    pad_cols = target_T - T
    if pad_cols <= 0:
        return input_ids, labels, attention_mask

    ids_pad = input_ids.new_full((input_ids.shape[0], pad_cols), int(pad_token_id))
    label_pad = labels.new_full((labels.shape[0], pad_cols), -100)
    input_ids = torch.cat([input_ids, ids_pad], dim=1).contiguous()
    labels = torch.cat([labels, label_pad], dim=1).contiguous()
    if attention_mask is not None:
        mask_pad = attention_mask.new_zeros((attention_mask.shape[0], pad_cols))
        attention_mask = torch.cat([attention_mask, mask_pad], dim=1).contiguous()
    return input_ids, labels, attention_mask

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

def should_update_progress(step: int, args, total_steps: int = None, first_step: int = 0) -> bool:
    """Throttle CUDA-to-CPU metric syncs caused by progress-bar scalar logging."""
    interval = max(1, int(getattr(args, 'progress_log_steps', 10) or 1))
    return (
        step == first_step
        or (step + 1) % interval == 0
        or (total_steps is not None and (step + 1) >= int(total_steps))
    )

def _state_to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_state_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_state_to_cpu(item) for item in value]
    if isinstance(value, dict):
        return {key: _state_to_cpu(item) for key, item in value.items()}
    return value

def _state_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_state_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_state_to_device(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _state_to_device(item, device) for key, item in value.items()}
    return value

def capture_model_grad_state(model):
    grad_state = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_state[name.replace("_orig_mod.", "")] = param.grad.detach().cpu().clone()
    return grad_state or None

def restore_model_grad_state(model, grad_state, device):
    if not grad_state:
        return False
    restored = 0
    clean_grad_state = {str(k).replace("_orig_mod.", ""): v for k, v in grad_state.items()}
    for name, param in model.named_parameters():
        clean_name = name.replace("_orig_mod.", "")
        grad = clean_grad_state.get(clean_name)
        if grad is None:
            continue
        param.grad = grad.to(device=device, dtype=param.dtype)
        restored += 1
    if restored:
        print(f"INFO: Restored {restored} pending gradient tensor(s) for accumulation parity.")
    return restored > 0

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state

def restore_rng_state(state):
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
        if "torch" in state:
            torch.random.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_all"])
        print("INFO: Restored RNG state from checkpoint.")
    except Exception as exc:
        print(f"Warning: Could not restore RNG state: {exc}")

def _capture_loader_component_state(component):
    if component is None:
        return None
    state = {"class": component.__class__.__name__}
    found = False
    for attr in ("seed", "epoch"):
        if hasattr(component, attr):
            try:
                state[attr] = int(getattr(component, attr))
                found = True
            except Exception:
                pass
    generator = getattr(component, "generator", None)
    if isinstance(generator, torch.Generator):
        try:
            state["generator_state"] = generator.get_state()
            found = True
        except Exception:
            pass
    return state if found else None

def capture_dataloader_state(dataloader):
    if dataloader is None:
        return None
    state = {}
    for name in ("batch_sampler", "sampler", "dataset"):
        component_state = _capture_loader_component_state(getattr(dataloader, name, None))
        if component_state:
            state[name] = component_state
    return state or None

def _restore_loader_component_state(component, state):
    if component is None or not state:
        return
    for attr in ("seed", "epoch"):
        if attr in state and hasattr(component, attr):
            try:
                setattr(component, attr, int(state[attr]))
            except Exception:
                pass
    generator = getattr(component, "generator", None)
    if isinstance(generator, torch.Generator) and "generator_state" in state:
        try:
            generator.set_state(state["generator_state"])
        except Exception:
            pass

def restore_dataloader_state(dataloader, state):
    if dataloader is None or not state:
        return
    for name in ("batch_sampler", "sampler", "dataset"):
        _restore_loader_component_state(getattr(dataloader, name, None), state.get(name))
    print("INFO: Restored dataloader sampler state from checkpoint.")

def build_training_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    args,
    dataloader,
    completed_epoch: int,
    mid_epoch_step: int = 0,
    running_states=None,
):
    _sanitize_model_nonfinite_(model, log_prefix="pre-checkpoint model")
    _clamp_model_finite_magnitude_(
        model,
        getattr(args, 'startup_weight_max_abs', 100.0),
        log_prefix="pre-checkpoint model",
    )
    grad_state = capture_model_grad_state(model)
    checkpoint = {
        "checkpoint_version": 2,
        "checkpoint_kind": "training",
        "completed_epoch": int(completed_epoch),
        "mid_epoch_step": int(mid_epoch_step or 0),
        "model_state_dict": sanitize_model_state_dict(model, reset_transient_ltm=False),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "config": dict(model.config),
        "rng_state": capture_rng_state(),
        "data_state": capture_dataloader_state(dataloader),
        "grad_state_dict": grad_state,
        "grad_accumulation_active": bool(grad_state),
        "ltm_scheduler_state": capture_ltm_lr_scheduler_state(args),
        "training_complete": False,
    }
    if running_states is not None:
        checkpoint["running_states"] = _state_to_cpu(running_states)
    return checkpoint

def train_step(model, batch, optimizer, scaler, accumulation_steps, step, args, running_states, collect_metrics=True):
    """Training step with temporal chunking to match original hierarchos.py."""
    args._optimizer_step_was_taken = False
    args._train_step_had_backward = False
    args._train_step_had_nonfinite = False
    device = next(model.parameters()).device
    set_model_training_step(model, getattr(args, '_current_global_step', step))
    _nb = (device.type == 'cuda')  # non_blocking for async CUDA transfer
    full_input_ids = batch['input_ids']
    full_attention_mask = batch.get('attention_mask')
    full_labels = batch['labels']
    full_rosa_ids = batch.get('rosa_ids')
    full_input_ids, full_labels, full_attention_mask = trim_trailing_padding(
        full_input_ids, full_labels, full_attention_mask
    )
    if full_rosa_ids is not None:
        full_rosa_ids = full_rosa_ids[:, :full_input_ids.shape[1]].contiguous()
    chunk_size = getattr(args, 'training_chunk_size', 128)
    padding_metric_steps = int(getattr(args, 'padding_metric_steps', 0) or 0)
    collect_padding_metrics = (
        collect_metrics
        and bool(getattr(args, 'padding_metrics', True))
        and (padding_metric_steps < 0 or step < padding_metric_steps)
    )
    padding_stats = None
    if collect_padding_metrics:
        pre_static_tokens = int(full_input_ids.numel())
        pre_static_seq_len = int(full_input_ids.shape[1]) if full_input_ids.ndim == 2 else 0
        if isinstance(full_attention_mask, torch.Tensor):
            real_tokens = int(full_attention_mask.sum().item())
        else:
            real_tokens = pre_static_tokens
    if (
        device.type == 'cuda'
        and getattr(args, 'compile', False)
        and getattr(args, 'compile_pad_to_chunk_size', True)
    ):
        pre_pad_seq_len = full_input_ids.shape[1]
        full_input_ids, full_labels, full_attention_mask = pad_training_batch_to_multiple(
            full_input_ids,
            full_labels,
            full_attention_mask,
            multiple=chunk_size,
            pad_token_id=getattr(args, 'pad_token_id', 0),
        )
        if full_rosa_ids is not None and full_input_ids.shape[1] > pre_pad_seq_len:
            pad_cols = full_input_ids.shape[1] - pre_pad_seq_len
            rosa_sentinel = int(getattr(model.config, 'vocab_size', getattr(args, 'vocab_size', 0)))
            rosa_pad = full_rosa_ids.new_full((full_rosa_ids.shape[0], pad_cols), rosa_sentinel)
            full_rosa_ids = torch.cat([full_rosa_ids, rosa_pad], dim=1).contiguous()
    if collect_padding_metrics:
        total_tokens = int(full_input_ids.numel())
        padded_seq_len = int(full_input_ids.shape[1]) if full_input_ids.ndim == 2 else 0
        padding_tokens = max(0, total_tokens - real_tokens)
        padding_stats = {
            "token_efficiency": real_tokens / float(max(1, total_tokens)),
            "padding_fraction": padding_tokens / float(max(1, total_tokens)),
            "bucket_padding_tokens": max(0, pre_static_tokens - real_tokens),
            "compile_padding_tokens": max(0, total_tokens - pre_static_tokens),
            "seq_len": padded_seq_len,
            "pre_static_seq_len": pre_static_seq_len,
        }
    
    # --- [NEW] Track Sequence Poisoning (Parity Fix) ---
    if full_labels.is_floating_point() and torch.isnan(full_labels).any():
        print(f"\nCRITICAL: NaNs detected in labels at step {step}! Skipping batch.")
        optimizer.zero_grad(set_to_none=True)
        return None, running_states
    # --------------------------------------------------
    
    B, T = full_input_ids.shape
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

    # Temporal chunking (critical for RWKV-based models). Build this on CPU before
    # moving labels/masks to CUDA so per-chunk .item() accounting does not sync GPU.
    if chunk_size <= 0 or chunk_size > T: chunk_size = T
    chunk_plan = compute_chunk_training_weights(full_labels, full_attention_mask, chunk_size)
    num_chunks = len(chunk_plan)

    full_input_ids = full_input_ids.to(device, non_blocking=_nb)
    if full_attention_mask is not None: full_attention_mask = full_attention_mask.to(device, non_blocking=_nb)
    full_labels = full_labels.to(device, non_blocking=_nb)
    if full_rosa_ids is not None: full_rosa_ids = full_rosa_ids.to(device, non_blocking=_nb)
    
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_ponder = torch.zeros((), device=device, dtype=torch.float32)
    total_commit = torch.zeros((), device=device, dtype=torch.float32)
    has_ponder = False
    has_commitment = False
    chunks_processed = 0
    final_outputs = None
    fast_lm_loss = (
        (device.type == 'cuda' and getattr(args, 'cuda_chunked_lm_loss', True))
        or (device.type == 'cpu' and getattr(args, 'cpu_chunked_lm_loss', True))
    )
    
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
            rosa_ids = full_rosa_ids[:, start_t:end_t] if full_rosa_ids is not None else None
            
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=args.amp):
                outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    h_state=h_state, l_state=l_state, prev_context=prev_ctx,
                    target_context=target_ctx, drift_state=drift_state, ltm_memory_state=ltm_state,
                    global_pos_offset=start_t,
                    return_logits=not fast_lm_loss,
                    return_topk_values=False,
                    rosa_ids=rosa_ids,
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

                ce_valid = _component_is_finite(ce_loss)
                ponder_valid = ponder_cost is None or _component_is_finite(ponder_cost)
                commitment_valid = commitment_cost is None or _component_is_finite(commitment_cost)
                if not (ce_valid and ponder_valid and commitment_valid):
                    print(
                        f"\nCRITICAL: Non-finite training loss at step {step+1}, "
                        f"chunk {chunk_idx} ({start_t}:{end_t})."
                    )
                    if not ce_valid and ce_loss is not None:
                        print("  " + _describe_tensor_issue("cross_entropy_loss", ce_loss))
                    if not ponder_valid and ponder_cost is not None:
                        print("  " + _describe_tensor_issue("ponder_cost", ponder_cost))
                    if not commitment_valid and commitment_cost is not None:
                        print("  " + _describe_tensor_issue("commitment_cost", commitment_cost))
                    print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)

                ce_loss_for_backward = _cap_loss_component_for_backward(
                    ce_loss,
                    getattr(args, 'max_ce_loss_for_backward', 10.0),
                )
                
                aux_loss = torch.zeros_like(ce_loss)
                
                # --- ACT Sensitivity: Adaptive Ponder Loss ---
                if ponder_cost is not None:
                    ponder_weight = getattr(args, 'ponder_loss_weight', 0.01)
                    ponder_cost_for_backward = _cap_loss_component_for_backward(
                        ponder_cost,
                        getattr(args, 'max_ponder_cost_for_backward', 0.0),
                    )
                    
                    if getattr(args, 'encourage_thinking', False):
                        # RECOVERY MODE: Invert ponder penalty to REWARD thinking
                        # Negative weight means higher ponder = lower loss
                        aux_loss = aux_loss - (abs(ponder_weight) * ponder_cost_for_backward)
                    elif getattr(args, 'adaptive_ponder', False):
                        # ADAPTIVE MODE: Scale ponder target with loss
                        # Higher CE loss = more thinking needed
                        max_h_steps = getattr(args, 'max_h_steps', 5)
                        target_scale = getattr(args, 'ponder_target_scale', 0.5)
                        target_ponder = torch.clamp(ce_loss.detach() * target_scale, min=1.0, max=float(max_h_steps))
                        ponder_diff = target_ponder - ponder_cost_for_backward
                        # Penalize under-thinking (when ponder < target), ignore over-thinking
                        ponder_penalty = torch.relu(ponder_diff) * ponder_weight
                        aux_loss = aux_loss + ponder_penalty
                    else:
                        # STANDARD MODE: Original additive penalty (penalizes thinking)
                        aux_loss = aux_loss + (ponder_weight * ponder_cost_for_backward)
                
                if commitment_cost is not None:
                    commitment_cost_for_backward = _cap_loss_component_for_backward(
                        commitment_cost,
                        getattr(args, 'max_commitment_cost_for_backward', 2.0),
                    )
                    aux_loss = aux_loss + (getattr(args, 'commitment_loss_weight', 0.5) * commitment_cost_for_backward)

                # CE is already averaged over valid labels within this chunk.
                # Weight it by supervised answer-token count so long masked
                # prompts/tool traces do not dilute the actual learning signal.
                chunk_loss = ((ce_loss_for_backward * label_ratio) + (aux_loss * token_ratio)) / accumulation_steps

                chunk_loss_valid = _component_is_finite(chunk_loss)
                if not chunk_loss_valid:
                    print(
                        f"\nCRITICAL: Non-finite training loss at step {step+1}, "
                        f"chunk {chunk_idx} ({start_t}:{end_t})."
                    )
                    print("  " + _describe_tensor_issue("chunk_loss", chunk_loss))
                    print("  Skipping this batch and clearing recurrent/LTM state before it can poison optimizer state.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)
            
            # Backprop per chunk (TBPTT)
            # retain_grad() is now handled inside autocast block above (BUG #1 fix)

            if scaler is not None: scaler.scale(chunk_loss).backward()
            else: chunk_loss.backward()
            args._train_step_had_backward = True
            
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
                    _clip_val = _positive_float(getattr(args, 'grad_clip', 1.0), 1.0)
                    ltm_grads_tensor = torch.nan_to_num(ltm_grads_tensor, nan=0.0, posinf=_clip_val, neginf=-_clip_val)
                    ltm_grads_tensor.clamp_(min=-_clip_val, max=_clip_val)

                    if ltm_grads_tensor is not None:
                        # Direct tensor clipping (clip_grad_norm_ expects parameters with .grad,
                        # but ltm_grads_tensor IS the gradient data itself — no .grad attribute)
                        if _clip_val > 0:
                            _grad_norm = ltm_grads_tensor.float().norm()
                            _clip_coef = torch.clamp(
                                ltm_grads_tensor.new_tensor(float(_clip_val)) / (_grad_norm + 1e-8),
                                max=1.0,
                            )
                            ltm_grads_tensor = ltm_grads_tensor * _clip_coef
                        
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
                            current_lr=get_current_ltm_lr(args),
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
            recurrent_state_clamp = getattr(args, 'recurrent_state_clamp', 50.0)
            context_state_clamp = getattr(args, 'context_state_clamp', 50.0)
            drift_state_clamp = getattr(args, 'drift_state_clamp', 5.0)
            if outputs.get('h_state') is not None:
                h_state = _detach_finite_clamp(outputs['h_state'], recurrent_state_clamp)
            if outputs.get('l_state') is not None:
                l_state = _detach_finite_clamp(outputs['l_state'], recurrent_state_clamp)
            if outputs.get('prev_context') is not None:
                prev_ctx = _detach_finite_clamp(outputs['prev_context'], context_state_clamp)
            if outputs.get('target_context') is not None:
                target_ctx = _detach_finite_clamp(outputs['target_context'], context_state_clamp)
            if outputs.get('drift_state') is not None:
                drift_state = _detach_finite_clamp(outputs['drift_state'], drift_state_clamp)
            
            # Accumulate for display
            total_loss = total_loss + ce_loss.detach().float() * label_ratio
            if ponder_cost is not None: 
                total_ponder = total_ponder + ponder_cost.detach().float() * token_ratio
                has_ponder = True
            if commitment_cost is not None: 
                total_commit = total_commit + commitment_cost.detach().float() * token_ratio
                has_commitment = True
            
            chunks_processed += 1
            # final_outputs = outputs # REMOVED: Memory Leak Fix
        
        # Optimizer step after all chunks
        if (step + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                grads_ok, grad_issue = _clip_gradients_and_check(model, getattr(args, 'grad_clip', 1.0))
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)
                scaler.step(optimizer)
                scaler.update()
            else:
                grads_ok, grad_issue = _clip_gradients_and_check(model, getattr(args, 'grad_clip', 1.0))
                if not grads_ok:
                    print(f"\nCRITICAL: Non-finite gradient at step {step+1}. {grad_issue}")
                    print("  Skipping optimizer step and clearing accumulated gradients.")
                    args._train_step_had_nonfinite = True
                    return None, _reset_after_nonfinite(optimizer, model)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            args._optimizer_step_was_taken = True
        
        if chunks_processed == 0:
            return None, running_states
            
        avg_outputs = None
        if collect_metrics:
            # Keep scalars on-device until the throttled progress update calls
            # .item(); doing this every batch forces a CUDA sync on fast GPUs.
            avg_outputs = {
                'loss': total_loss.detach(),
                'ponder_cost': total_ponder.detach() if has_ponder else None,
                'commitment_cost': total_commit.detach() if has_commitment else None,
            }
            if padding_stats is not None:
                avg_outputs.update(padding_stats)
        
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

        if getattr(args, 'cuda_chunked_lm_loss', True):
            loss_chunk_rows = int(getattr(args, 'cuda_loss_chunk_rows', 0) or 0)
            if loss_chunk_rows <= 0:
                args._auto_cuda_loss_chunk_rows = True
                config.cuda_loss_chunk_rows = 0
                print("INFO: CUDA chunked LM loss enabled (startup auto rows from free VRAM and batch shape).")
            else:
                args._auto_cuda_loss_chunk_rows = False
                config.cuda_loss_chunk_rows = loss_chunk_rows
                print(f"INFO: CUDA chunked LM loss enabled ({loss_chunk_rows} fixed rows/chunk, logits omitted in train_step).")
            config.cuda_loss_chunk_rows = loss_chunk_rows
            config.cuda_chunked_lm_loss = True
        else:
            args._auto_cuda_loss_chunk_rows = False
            config.cuda_chunked_lm_loss = False

        # Multi-GPU info
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            print(f"INFO: {num_gpus} GPUs detected. For multi-GPU training, wrap with DistributedDataParallel.")
    # =================================================================

    model = None
    optimizer = None
    start_epoch = 0
    start_step = 0
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
        load_cleaned = _sanitize_payload_nonfinite_(
            state_dict,
            "model_state_dict",
            max_abs=getattr(args, 'grad_clip', 1.0),
        )
        if load_cleaned:
            print(
                f"WARNING: Sanitized {load_cleaned} non-finite checkpoint model_state_dict "
                "value(s) before loading. Future checkpoints will be saved clean."
            )
        checkpoint['model_state_dict'] = state_dict
        
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

        # 4. Detect RNN hidden sizes and RWKV matrix-state head geometry.
        if 'h_rnn.key.weight' in state_dict_keys:
            model_config.h_hidden = state_dict['h_rnn.key.weight'].shape[0]
        elif 'h_rnn.time_decay' in state_dict_keys:
            model_config.h_hidden = state_dict['h_rnn.time_decay'].shape[-1]
        if 'l_rnn.key.weight' in state_dict_keys:
            model_config.l_hidden = state_dict['l_rnn.key.weight'].shape[0]
        elif 'l_rnn.time_decay' in state_dict_keys:
            model_config.l_hidden = state_dict['l_rnn.time_decay'].shape[-1]
        if 'h_rnn.r_k' in state_dict_keys:
            model_config.rwkv_head_size = state_dict['h_rnn.r_k'].shape[1]

        # ARCH defaults / Fallbacks
        arch_defaults = {
            'ltm_slots': 1024, 'ltm_key_dim': 128, 'ltm_val_dim': 128, 'ltm_topk': 4,
            'h_stride': 4, 'max_h_steps': 5, 'max_l_steps': 5,
            'h_hidden': model_config.get('context_dim', 448),
            'l_hidden': model_config.get('context_dim', 448),
            'rwkv_head_size': getattr(args, 'rwkv_head_size', None),
        }
        for k, v in arch_defaults.items():
            if k not in model_config:
                model_config[k] = getattr(args, k, v) if hasattr(args, k) else v

        # Runtime overrides
        model_config.compile = args.compile
        model_config.max_length = args.max_length or model_config.get('max_length', 1024)
        model_config.ltm_lr = getattr(args, 'ltm_lr', model_config.get('ltm_lr', 1e-3))
        model_config.min_ltm_lr = getattr(args, 'min_ltm_lr', model_config.get('min_ltm_lr', None))
        model_config.disable_ltm_lr_schedule = getattr(args, 'disable_ltm_lr_schedule', False)

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
        
        optimizer = build_hierarchos_optimizer(model, args, device)
        
        if not getattr(args, 'override_scheduling', False) and 'optimizer_state_dict' in checkpoint:
            try: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except: print("Warning: Could not load optimizer state.")
        
        # Original script uses 'completed_epoch', modular uses 'epoch', check both for compatibility.
        start_epoch = int(checkpoint.get('completed_epoch', checkpoint.get('epoch', 0)) or 0)
        start_step = int(checkpoint.get('mid_epoch_step', 0) or 0)
        if start_step >= dataloader_len:
            start_epoch += 1
            start_step = 0
        if start_step > 0:
            print(f"Successfully loaded model state. Resuming from epoch {start_epoch + 1}, step {start_step}.")
        else:
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
        
        optimizer = build_hierarchos_optimizer(model, args, device)
        
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()
    
    else:
        print("Starting training from scratch.")
        if 'vocab_size' not in config: config.vocab_size = len(tokenizer)
        model = HierarchosCore(config).to(device)
        optimizer = build_hierarchos_optimizer(model, args, device)
        if use_amp and getattr(config, 'amp_dtype', 'float16') == 'float16': scaler = GradScaler()

    # --- [NEW] Sync LTM reference chunk size (Parity Fix) ---
    training_chunk_size = getattr(args, 'training_chunk_size', 128)
    if hasattr(model, 'ltm'):
        if not hasattr(model.ltm, 'reference_chunk_len'):
            model.ltm.reference_chunk_len = training_chunk_size
        
        if model.ltm.reference_chunk_len != training_chunk_size:
            print(f"INFO: Updating LTM reference chunk length from {model.ltm.reference_chunk_len} to {training_chunk_size}")
            model.ltm.reference_chunk_len = training_chunk_size
        model.ltm.cpu_gather_retrieval = bool(getattr(args, 'ltm_cpu_gather_retrieval', True))
        model.ltm.cpu_sparse_update = bool(getattr(args, 'ltm_cpu_sparse_update', True))
    if hasattr(model, 'config'):
        model.config.cpu_chunked_lm_loss = bool(getattr(args, 'cpu_chunked_lm_loss', True))
        model.config.cpu_loss_chunk_rows = int(getattr(args, 'cpu_loss_chunk_rows', 0) or 0)
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
    tune_cuda_loss_chunk_rows_once(
        model,
        args,
        batch_size=getattr(args, 'batch_size', 1),
        chunk_size=getattr(args, 'training_chunk_size', 128),
    )

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
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=bool(getattr(args, 'override_scheduling', False)),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = getattr(args, 'ltm_lr', getattr(model.config, 'ltm_lr', 1e-3))
        model.config.min_ltm_lr = getattr(args, 'min_ltm_lr', getattr(model.config, 'min_ltm_lr', None))
        model.config.disable_ltm_lr_schedule = getattr(args, 'disable_ltm_lr_schedule', False)
    if checkpoint:
        restore_dataloader_state(dataloader, checkpoint.get('data_state'))
        restore_rng_state(checkpoint.get('rng_state'))
        restore_model_grad_state(model, checkpoint.get('grad_state_dict'), device)
        if checkpoint.get('running_states') is not None:
            running_issue = _find_first_nonfinite_payload_tensor(checkpoint['running_states'], "running_states")
            if running_issue:
                cleaned = _sanitize_payload_nonfinite_(
                    checkpoint['running_states'],
                    "running_states",
                    max_abs=getattr(args, 'grad_clip', 1.0),
                )
                print(f"WARNING: Sanitized saved running state before resume: {running_issue}")
                print(f"WARNING: Repaired {cleaned} non-finite running-state value(s).")
        if start_step > 0:
            if checkpoint.get('data_state') is None:
                print("Warning: Mid-epoch checkpoint has no dataloader state; resume will skip to the saved step but exact batch order may differ.")
            if checkpoint.get('rng_state') is None:
                print("Warning: Mid-epoch checkpoint has no RNG state; stochastic components may not be exactly replayed.")
            if getattr(args, 'persist_state', False) and 'running_states' not in checkpoint:
                print("Warning: Mid-epoch checkpoint has no running RWKV/LTM states while --persist-state is enabled; continuing with reset states.")

    _sanitize_model_nonfinite_(model, log_prefix="startup model")
    _clamp_model_finite_magnitude_(
        model,
        getattr(args, 'startup_weight_max_abs', 100.0),
        log_prefix="startup model",
    )
    _sanitize_model_transient_state_(model, max_abs=getattr(args, 'grad_clip', 1.0))
    _sanitize_optimizer_state_(optimizer)
    _sanitize_gradient_nonfinite_(model, max_abs=1.0)
    # --- Evaluation Confirmation ---
    eval_tasks = getattr(args, 'eval_tasks', None)
    if eval_tasks:
        eval_every = getattr(args, 'eval_every_epoch', 1)
        print(f"INFO: Evaluation ENABLED - will run {eval_tasks} every {eval_every} epoch(s)")
    
    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        model.train()
        set_dataloader_epoch(dataloader, epoch)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}", total=dataloader_len)
        
        # Restore recurrent/LTM states only for true mid-epoch checkpoints.
        if epoch == start_epoch and start_step > 0 and checkpoint and 'running_states' in checkpoint:
            running_states = _state_to_device(checkpoint['running_states'], device)
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

            first_logged_step = start_step if epoch == start_epoch else 0
            collect_metrics = should_update_progress(step, args, dataloader_len, first_logged_step)
            args._current_global_step = epoch * dataloader_len + step
            outputs, running_states = train_step(
                model,
                batch,
                optimizer,
                scaler,
                args.accumulation_steps,
                step,
                args,
                running_states,
                collect_metrics=collect_metrics,
            )
            if outputs:
                postfix = {"loss": f"{outputs['loss'].item():.4f}"}
                if outputs.get('ponder_cost') is not None:
                    postfix["ponder"] = f"{outputs['ponder_cost'].item():.2f}"
                if outputs.get('commitment_cost') is not None:
                    postfix["commit"] = f"{outputs['commitment_cost'].item():.2e}"
                if outputs.get('token_efficiency') is not None:
                    postfix["tok_eff"] = f"{outputs['token_efficiency'] * 100.0:.1f}%"
                    postfix["seq"] = int(outputs.get('seq_len', 0) or 0)
                if scheduler:
                    postfix["lr"] = f"{scheduler.get_last_lr()[0]:.2e}"
                postfix["ltm_lr"] = f"{get_current_ltm_lr(args):.2e}"
                pbar.set_postfix(postfix)

            if scheduler and getattr(args, '_optimizer_step_was_taken', False):
                scheduler.step()
            if getattr(args, '_optimizer_step_was_taken', False):
                advance_ltm_lr_schedule(args)

            # Periodic Checkpointing (Progress Protection)
            if args.save_steps > 0 and (step + 1) % args.save_steps == 0:
                print(f"\n[Step {step+1}] Periodic Checkpoint: Saving to {args.out_dir}...")
                save_training_checkpoint_if_finite(
                    build_training_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        args,
                        dataloader,
                        completed_epoch=epoch,
                        mid_epoch_step=step + 1,
                        running_states=running_states,
                    ),
                    os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}_step_{step+1}.pt"),
                    model,
                    optimizer,
                )
            
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
        
        save_training_checkpoint_if_finite(
            build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                dataloader,
                completed_epoch=epoch + 1,
                mid_epoch_step=0,
            ),
            os.path.join(args.out_dir, f"hierarchos_epoch_{epoch+1}.pt"),
            model,
            optimizer,
        )
        
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
    save_training_checkpoint_if_finite(final_checkpoint, final_model_path, model, optimizer=None)
    
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
            state_dict = sanitize_model_state_dict(checkpoint['model_state_dict'], reset_transient_ltm=False)
            load_cleaned = _sanitize_payload_nonfinite_(
                state_dict,
                "model_state_dict",
                max_abs=getattr(args, 'grad_clip', 1.0),
            )
            if load_cleaned:
                print(
                    f"WARNING: Sanitized {load_cleaned} non-finite checkpoint model_state_dict "
                    "value(s) before loading. Future checkpoints will be saved clean."
                )
            checkpoint['model_state_dict'] = state_dict
            model.load_state_dict(state_dict, strict=False)
        start_epoch = checkpoint.get('completed_epoch', 0)
        start_step = checkpoint.get('mid_epoch_step', 0)
        print(f"INFO: Resuming from epoch {start_epoch+1}, step {start_step}.")

    model.print_trainable_parameters()

    # Optimizer selection
    if is_directml_device(device):
        print("INFO: DirectML detected. Using optimized DirectMLAdamW optimizer.")
        optimizer = build_hierarchos_optimizer(model, args, device)
    else:
        optimizer = build_hierarchos_optimizer(model, args, device)
    
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
    accumulation_steps = getattr(args, 'accumulation_steps', 1)
    num_update_steps = (dataloader_len // accumulation_steps) * args.epochs if dataloader_len > 0 else 0
    if not getattr(args, 'disable_lr_schedule', False):
        if num_update_steps > 0:
            print(f"INFO: Cosine Annealing LR scheduler ENABLED. Total steps: {num_update_steps}, Max LR: {args.starting_lr}, Min LR: {args.min_lr}")
            scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=args.min_lr)
            if checkpoint and 'scheduler_state_dict' in checkpoint:
                try: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except: pass
        else:
            print("Warning: Cannot enable LR schedule, dataset might be too small.")
    configure_ltm_lr_schedule(
        args,
        num_update_steps,
        checkpoint=checkpoint,
        override_schedule=bool(getattr(args, 'override_scheduling', False)),
        scheduler=scheduler,
    )
    if hasattr(model, 'config'):
        model.config.ltm_lr = getattr(args, 'ltm_lr', getattr(model.config, 'ltm_lr', 1e-3))
        model.config.min_ltm_lr = getattr(args, 'min_ltm_lr', getattr(model.config, 'min_ltm_lr', None))
        model.config.disable_ltm_lr_schedule = getattr(args, 'disable_ltm_lr_schedule', False)

    _sanitize_model_nonfinite_(model, log_prefix="fine-tune startup model")
    _clamp_model_finite_magnitude_(
        model,
        getattr(args, 'startup_weight_max_abs', 100.0),
        log_prefix="fine-tune startup model",
    )
    _sanitize_model_transient_state_(model, max_abs=getattr(args, 'grad_clip', 1.0))
    _sanitize_optimizer_state_(optimizer)
    _sanitize_gradient_nonfinite_(model, max_abs=1.0)

    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    ponder_loss_weight = getattr(args, 'ponder_loss_weight', 0.01)
    commitment_loss_weight = getattr(args, 'commitment_loss_weight', 0.5)
    grad_clip = getattr(args, 'grad_clip', 1.0)

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- LoRA Finetune Epoch {epoch + 1} / {args.epochs} ---")
        set_dataloader_epoch(dataloader, epoch)
        pbar = tqdm(dataloader, desc=f"Finetune Epoch {epoch + 1}", total=dataloader_len)
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
            rosa_ids = batch.get("rosa_ids")
            input_ids, labels, attention_mask = trim_trailing_padding(
                input_ids, labels, attention_mask
            )
            if rosa_ids is not None:
                rosa_ids = rosa_ids[:, :input_ids.shape[1]].contiguous()
            non_blocking = device.type == 'cuda'
            input_ids = input_ids.to(device, non_blocking=non_blocking)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if rosa_ids is not None:
                rosa_ids = rosa_ids.to(device, non_blocking=non_blocking)
            set_model_training_step(model, epoch * dataloader_len + i)

            autocast_device_type = 'cpu' if is_directml_device(device) else device.type
            with autocast(device_type=autocast_device_type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, rosa_ids=rosa_ids)
                
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
                    loss_accum = loss_accum + _cap_loss_component_for_backward(
                        cross_entropy_loss,
                        getattr(args, 'max_ce_loss_for_backward', 10.0),
                    )
                elif i % accumulation_steps == 0:
                    print(f"\nWarning: CE loss is NaN/Inf at step {i+1}. Skipping.")

                if pc_valid:
                    loss_accum = loss_accum + (
                        ponder_loss_weight
                        * _cap_loss_component_for_backward(
                            ponder_cost,
                            getattr(args, 'max_ponder_cost_for_backward', 0.0),
                        )
                    )
                
                if cc_valid:
                    loss_accum = loss_accum + (
                        commitment_loss_weight
                        * _cap_loss_component_for_backward(
                            commitment_cost,
                            getattr(args, 'max_commitment_cost_for_backward', 2.0),
                        )
                    )

                if ce_valid:
                    combined_loss = loss_accum

                combined_valid = combined_loss is not None and torch.isfinite(combined_loss).all()
                if combined_loss is not None and not bool(combined_valid.item()):
                    print(
                        f"\nCRITICAL: Non-finite fine-tune loss at step {i+1}; "
                        "skipping batch and clearing gradients."
                    )
                    print("  " + _describe_tensor_issue("combined_loss", combined_loss))
                    combined_loss = None
                    optimizer.zero_grad(set_to_none=True)

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

                        if valid_update:
                            ltm_clip = _positive_float(grad_clip, 1.0)
                            ltm_grads_copy = torch.nan_to_num(
                                ltm_grads_copy,
                                nan=0.0,
                                posinf=ltm_clip,
                                neginf=-ltm_clip,
                            )
                            ltm_grads_copy.clamp_(min=-ltm_clip, max=ltm_clip)
                            if ltm_clip > 0:
                                ltm_norm = ltm_grads_copy.float().norm()
                                clip_coef = torch.clamp(
                                    ltm_grads_copy.new_tensor(float(ltm_clip)) / (ltm_norm + 1e-8),
                                    max=1.0,
                                )
                                ltm_grads_copy = ltm_grads_copy * clip_coef
                            base_ltm.inner_update(
                                outputs["topk_idx"],
                                ltm_grads_copy,
                                current_lr=get_current_ltm_lr(args),
                                timestamp=float(i + 1),
                                source=2  # SRC_TRAINING_DATA
                            )

                    # Optimizer step
                    if use_amp and scaler:
                        scaler.unscale_(optimizer)
                        grads_ok, grad_issue = _clip_gradients_and_check(model, grad_clip)
                        if not grads_ok:
                            print(f"\nCRITICAL: Non-finite fine-tune gradient at step {i+1}. {grad_issue}")
                            print("  Skipping optimizer step and clearing accumulated gradients.")
                            _reset_after_nonfinite(optimizer, model)
                            backward_called_in_cycle = False
                            continue
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        grads_ok, grad_issue = _clip_gradients_and_check(model, grad_clip)
                        if not grads_ok:
                            print(f"\nCRITICAL: Non-finite fine-tune gradient at step {i+1}. {grad_issue}")
                            print("  Skipping optimizer step and clearing accumulated gradients.")
                            _reset_after_nonfinite(optimizer, model)
                            backward_called_in_cycle = False
                            continue
                        optimizer.step()

                    if scheduler:
                        scheduler.step()
                    advance_ltm_lr_schedule(args)

                    optimizer.zero_grad(set_to_none=True)
                    backward_called_in_cycle = False
                    global_step += 1

                    # Periodic Checkpointing (Progress Protection)
                    if getattr(args, 'save_steps', 0) > 0 and (i + 1) % args.save_steps == 0:
                        ckpt_path = os.path.join(args.out_dir, f"hierarchos_finetune_epoch_{epoch+1}_step_{i+1}.pt")
                        print(f"\n[Step {i+1}] Periodic Checkpoint: Saving to {ckpt_path}...")
                        save_training_checkpoint_if_finite({
                            'completed_epoch': epoch,
                            'mid_epoch_step': i + 1,
                            'model_state_dict': sanitize_model_state_dict(model),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'ltm_scheduler_state': capture_ltm_lr_scheduler_state(args),
                            'config': dict(model.config),
                        }, ckpt_path, model, optimizer)
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
                "lr": f"{current_lr:.2e}",
                "ltm_lr": f"{get_current_ltm_lr(args):.2e}"
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
