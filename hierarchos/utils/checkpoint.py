import json
import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .device import is_directml_device

TRANSIENT_LTM_STATE_KEYS = (
    "ltm.fast_vals",
    "ltm._mom_vals",
    "ltm.timestamps",
    "ltm.sources",
)


def sanitize_model_state_dict(model_or_state_dict, reset_transient_ltm: bool = True) -> Dict[str, torch.Tensor]:
    """Return a save-ready state_dict with compile prefixes removed and transient LTM state zeroed."""
    source_state = model_or_state_dict.state_dict() if hasattr(model_or_state_dict, "state_dict") else model_or_state_dict
    clean_state = {}
    for key, value in source_state.items():
        clean_key = key.replace("_orig_mod.", "")
        if reset_transient_ltm and any(clean_key.endswith(suffix) for suffix in TRANSIENT_LTM_STATE_KEYS):
            clean_state[clean_key] = torch.zeros_like(value)
        else:
            clean_state[clean_key] = value
    return clean_state


# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def _resolve_weights_path(model_path: str) -> Tuple[str, str]:
    """Resolve a Hierarchos model source to (weights_path, model_dir)."""
    if not model_path:
        raise FileNotFoundError("No model path was provided.")

    resolved = os.path.abspath(os.path.expanduser(model_path))
    if os.path.isfile(resolved):
        if not resolved.lower().endswith(".pt"):
            raise FileNotFoundError(f"Model file must be a .pt checkpoint: {resolved}")
        return resolved, os.path.dirname(resolved)

    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Model path not found: {model_path}")

    preferred = ("hierarchos.pt", "model.pt", "hierarchos_final.pt")
    for name in preferred:
        candidate = os.path.join(resolved, name)
        if os.path.exists(candidate):
            return candidate, resolved

    pt_files = sorted(
        f for f in os.listdir(resolved)
        if f.lower().endswith(".pt")
    )
    if pt_files:
        return os.path.join(resolved, pt_files[0]), resolved

    raise FileNotFoundError(f"Model weights file not found in '{model_path}'")


def _load_json_config(model_dir: str) -> Optional[Dict[str, Any]]:
    for name in ("hierarchos_config.json", "config.json"):
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return dict(loaded)
    return None


def _has_v8_rwkv_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("h_rnn.x_r") or key.startswith("h_rnn.r_k")
        for key in state_dict
    )


def _has_legacy_rwkv_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return any(
        key.startswith("h_rnn.time_decay") or key.startswith("h_rnn.time_mix_")
        for key in state_dict
    )


def _reject_unsupported_rwkv_state_dict(state_dict: Dict[str, torch.Tensor], source: str = "checkpoint") -> None:
    if _has_legacy_rwkv_state_dict(state_dict):
        raise ValueError(
            f"Unsupported legacy scalar-RWKV checkpoint in {source}. "
            "The active modular Hierarchos path is v8-only and requires "
            "matrix-state RWKV keys such as 'h_rnn.x_r' and 'h_rnn.r_k'. "
            "Do not continue paid v8 training from this checkpoint."
        )


def _reject_rwkv_load_mismatch(missing_keys, unexpected_keys, source: str = "checkpoint") -> None:
    critical_prefixes = ("h_rnn.", "l_rnn.")
    missing = [key for key in missing_keys if str(key).startswith(critical_prefixes)]
    unexpected = [key for key in unexpected_keys if str(key).startswith(critical_prefixes)]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing RWKV keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            details.append(f"unexpected RWKV keys: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
        raise ValueError(
            f"RWKV v8 checkpoint mismatch in {source}; "
            + "; ".join(details)
            + ". Refusing to continue because partial recurrent-block loading can produce incoherence."
        )


def _infer_arch_flags_from_state_dict(config_dict: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> None:
    """Backfill architecture toggles for checkpoints saved before these flags existed."""
    if "use_deepembed" not in config_dict:
        config_dict["use_deepembed"] = any(
            key.startswith("h_deepemb.") or key.startswith("l_deepemb.")
            for key in state_dict
        )

    if "use_rosa" not in config_dict:
        config_dict["use_rosa"] = any(
            key.startswith("rosa_emb.") or key == "rosa_gate_logit"
            for key in state_dict
        )

    if config_dict.get("use_rosa", True) and "rosa_max_context" not in config_dict:
        config_dict["rosa_max_context"] = 512

    if "memory_token_routers" not in config_dict:
        has_router_weights = any(
            key.startswith("rosa_router.") or key.startswith("ltm_router.")
            for key in state_dict
        )
        config_dict["memory_token_routers"] = has_router_weights

    if "rwkv_head_size" not in config_dict:
        head_shape = state_dict.get("h_rnn.r_k")
        if torch.is_tensor(head_shape) and head_shape.ndim == 2:
            config_dict["rwkv_head_size"] = int(head_shape.shape[1])

    if "rwkv_channel_mix_key_clamp" not in config_dict:
        config_dict["rwkv_channel_mix_key_clamp"] = 12.0


def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model from a directory or direct .pt file."""
    weights_path, model_dir = _resolve_weights_path(model_path)

    try:
        # Compatibility with different PyTorch versions and custom classes
        try:
            from torch.serialization import safe_globals
            with safe_globals([AttrDict]):
                checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        except (ImportError, AttributeError):
            checkpoint = torch.load(weights_path, map_location="cpu")
            
        if not isinstance(checkpoint, dict) or ('config' not in checkpoint and 'model_state_dict' not in checkpoint):
            # Fallback for old style checkpoints
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format: expected a dict-like .pt file.")

    config_dict = checkpoint.get('config') or _load_json_config(model_dir)
    if config_dict is None:
        raise ValueError(
            "Model config not found. Include 'config' in the checkpoint or add "
            "hierarchos_config.json next to the .pt file."
        )

    config_dict = dict(config_dict)
    if 'model_type' not in config_dict: config_dict['model_type'] = 'hierarchos'
    
    # Strip _orig_mod. prefix from compiled model checkpoints (torch.compile adds this)
    # Without this, strict=False silently drops ALL weights from compiled checkpoints
    if 'model_state_dict' in checkpoint:
        state_source = checkpoint['model_state_dict']
    else:
        state_source = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
        if not state_source:
            raise ValueError("Model state_dict not found in checkpoint.")
    state_dict = sanitize_model_state_dict(state_source, reset_transient_ltm=False)
    _reject_unsupported_rwkv_state_dict(state_dict, weights_path)
    _infer_arch_flags_from_state_dict(config_dict, state_dict)

    config = AttrDict(config_dict)

    from ..models.core import HierarchosCore
    model = HierarchosCore(config)
    
    # Handle qproj shape mismatch (Old -> New Architecture)
    if 'qproj.weight' in state_dict:
        old_w = state_dict['qproj.weight']
        new_w = model.qproj.weight
        if old_w.shape != new_w.shape:
            print(f"INFO: Adapting qproj.weight from {old_w.shape} to {new_w.shape}")
            if old_w.shape[0] == new_w.shape[0] and old_w.shape[1] * 2 == new_w.shape[1]:
                adapted_w = torch.randn_like(new_w) * 0.02
                adapted_w[:, :old_w.shape[1]] = old_w
                state_dict['qproj.weight'] = adapted_w
            else:
                del state_dict['qproj.weight']

    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"WARNING: {len(load_result.missing_keys)} missing keys: {load_result.missing_keys[:5]}{'...' if len(load_result.missing_keys) > 5 else ''}")
    if load_result.unexpected_keys:
        print(f"WARNING: {len(load_result.unexpected_keys)} unexpected keys: {load_result.unexpected_keys[:5]}{'...' if len(load_result.unexpected_keys) > 5 else ''}")
    _reject_rwkv_load_mismatch(load_result.missing_keys, load_result.unexpected_keys, weights_path)
    if not load_result.missing_keys and not load_result.unexpected_keys:
        print(f"INFO: All {len(state_dict)} weight tensors loaded successfully.")
    model.to(device)
    if checkpoint.get('training_complete', False) and hasattr(model, 'reset_memory'):
        model.reset_memory()
    model.eval()
    return model, config

def save_checkpoint_safely(checkpoint_dict: Dict[str, Any], path: str):
    """Saves a checkpoint safely with validation and backup."""
    temp_path = path + ".tmp"
    backup_path = path + ".bak"
    
    try:
        torch.save(checkpoint_dict, temp_path)
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            raise RuntimeError("Failed to save checkpoint: Temp file is missing or empty.")
        
        if os.path.exists(path):
            if os.path.exists(backup_path): os.remove(backup_path)
            os.rename(path, backup_path)
        
        os.rename(temp_path, path)
        print(f"INFO: Checkpoint saved safely to {path}")
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint safely: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        raise e
