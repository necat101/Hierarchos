import os
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from .device import is_directml_device

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_full_model_with_config(model_path: str, device):
    """Loads a full-precision model and its config from a directory."""
    MODEL_WEIGHTS_NAME = "hierarchos.pt"
    weights_path = os.path.join(model_path, MODEL_WEIGHTS_NAME)
    if not os.path.exists(weights_path):
        # Try finding any .pt file if hierarchos.pt is missing
        pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        if pt_files:
            weights_path = os.path.join(model_path, pt_files[0])
        else:
            raise FileNotFoundError(f"Model weights file not found in '{model_path}'")

    try:
        # Compatibility with different PyTorch versions and custom classes
        try:
            from torch.serialization import safe_globals
            with safe_globals([AttrDict]):
                checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
        except (ImportError, AttributeError):
            checkpoint = torch.load(weights_path, map_location=device)
            
        if 'config' not in checkpoint:
            # Fallback for old style checkpoints
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    if 'config' not in checkpoint:
        raise ValueError("Model config not found in checkpoint.")

    config_dict = checkpoint['config']
    if 'model_type' not in config_dict: config_dict['model_type'] = 'hierarchos'
    config = AttrDict(config_dict)

    from ..models.core import HierarchosCore
    model = HierarchosCore(config)
    
    # Handle qproj shape mismatch (Old -> New Architecture)
    state_dict = checkpoint['model_state_dict']
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

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
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
