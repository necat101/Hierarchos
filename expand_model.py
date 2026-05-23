import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from hierarchos import AttrDict, HierarchosCore


MODEL_WEIGHTS_NAME = "hierarchos.pt"

LATEST_CONFIG_DEFAULTS = {
    "max_length": 1024,
    "ltm_lr": 1e-3,
    "ltm_topk": 4,
    "max_h_steps": 5,
    "max_l_steps": 5,
    "h_stride": 4,
    "l_conv_atol": 1e-4,
    "commitment_threshold": 0.05,
    "detach_every_n_steps": 32,
    "h_halt_thresh": 0.9,
    "gradient_checkpointing": False,
    "compile": False,
    "compile_mode": "max-autotune-no-cudagraphs",
    "compile_backend": None,
    "compile_dynamic": False,
    "compile_fullgraph_worker": False,
    "compile_cudagraphs": False,
    "compile_pad_to_chunk_size": True,
    "compile_static_worker_loop": None,
    "compile_h_rnn": True,
    "compile_quiet": True,
    "use_deepembed": True,
    "use_rosa": True,
    "rosa_max_context": 512,
    "rwkv_head_size": None,
}

ARCH_UPDATE_KEYS = [
    "vocab_size",
    "context_dim",
    "persistent_dim",
    "ltm_slots",
    "ltm_key_dim",
    "ltm_val_dim",
    "ltm_lr",
    "ltm_topk",
    "h_hidden",
    "l_hidden",
    "h_stride",
    "max_h_steps",
    "max_l_steps",
    "l_conv_atol",
    "commitment_threshold",
    "detach_every_n_steps",
    "h_halt_thresh",
    "ltm_forget_rate",
    "use_deepembed",
    "use_rosa",
    "rosa_max_context",
    "rwkv_head_size",
]

DERIVED_OR_RUNTIME_KEYS = {
    "time_freqs",
    "ltm.neg_inf",
    "ltm.update_counts",
    "ltm.update_slots",
    "ltm.ltm_deltas",
}

LOW_VARIANCE_NEW_WEIGHTS = {
    "context_drift_proj.weight",
    "l_feedback_proj.weight",
}


def _plain_dict(value: Any) -> Any:
    if isinstance(value, AttrDict):
        return {k: _plain_dict(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {k: _plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_plain_dict(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_plain_dict(v) for v in value)
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _load_torch(path: Path, device: str) -> Any:
    try:
        with torch.serialization.safe_globals([AttrDict]):
            return torch.load(path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(path, map_location=device, weights_only=False)


def _resolve_checkpoint_path(model_path: str) -> Tuple[Path, Path]:
    path = Path(model_path)
    if path.is_file():
        return path, path.parent

    candidates = [
        path / MODEL_WEIGHTS_NAME,
        path / "Hierarchos.pt",
        path / "model.pt",
        path / "hierarchos_final.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, path

    raise FileNotFoundError(
        f"No checkpoint found for '{model_path}'. Expected a .pt file or a "
        f"directory containing {MODEL_WEIGHTS_NAME}."
    )


def _resolve_output_paths(output_target: str) -> Tuple[Path, Path]:
    path = Path(output_target)
    if path.suffix.lower() == ".pt":
        output_dir = path.parent if str(path.parent) else Path(".")
        if path.name != MODEL_WEIGHTS_NAME:
            print(
                f"Treating '{output_target}' as a legacy output file path. "
                f"Latest Hierarchos loads model directories, so weights will be saved as "
                f"'{output_dir / MODEL_WEIGHTS_NAME}'."
            )
        return output_dir, output_dir / MODEL_WEIGHTS_NAME

    return path, path / MODEL_WEIGHTS_NAME


def _extract_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return _normalize_state_dict_keys(checkpoint[key])

        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return _normalize_state_dict_keys(checkpoint)

    raise ValueError("Could not find a model state dict in the checkpoint.")


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        clean_key = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if clean_key.startswith(prefix):
                    clean_key = clean_key[len(prefix):]
                    changed = True
        normalized[clean_key] = value
    return normalized


def _load_config(checkpoint: Any, model_root: Path) -> Dict[str, Any]:
    if isinstance(checkpoint, dict) and checkpoint.get("config"):
        return _plain_dict(checkpoint["config"])

    for name in ("hierarchos_config.json", "config.json"):
        config_path = model_root / name
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)

    raise ValueError(
        "Could not find config in the checkpoint or alongside it. "
        "Expected checkpoint['config'] or hierarchos_config.json."
    )


def _infer_missing_config(config: Dict[str, Any], state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    inferred = copy.deepcopy(config)

    if "tok_emb.weight" in state_dict:
        inferred.setdefault("vocab_size", int(state_dict["tok_emb.weight"].shape[0]))
        inferred.setdefault("context_dim", int(state_dict["tok_emb.weight"].shape[1]))
    elif "lm_head.weight" in state_dict:
        inferred.setdefault("vocab_size", int(state_dict["lm_head.weight"].shape[0]))
        inferred.setdefault("context_dim", int(state_dict["lm_head.weight"].shape[1]))

    if "persistent" in state_dict and state_dict["persistent"].ndim == 1:
        inferred.setdefault("persistent_dim", int(state_dict["persistent"].shape[0]))

    if "ltm.keys" in state_dict and state_dict["ltm.keys"].ndim == 2:
        inferred.setdefault("ltm_slots", int(state_dict["ltm.keys"].shape[0]))
        inferred.setdefault("ltm_key_dim", int(state_dict["ltm.keys"].shape[1]))

    if "ltm.vals" in state_dict and state_dict["ltm.vals"].ndim == 2:
        inferred.setdefault("ltm_slots", int(state_dict["ltm.vals"].shape[0]))
        inferred.setdefault("ltm_val_dim", int(state_dict["ltm.vals"].shape[1]))

    if "qproj.weight" in state_dict and state_dict["qproj.weight"].ndim == 2:
        inferred.setdefault("ltm_key_dim", int(state_dict["qproj.weight"].shape[0]))

    if "h_rnn.key.weight" in state_dict and state_dict["h_rnn.key.weight"].ndim == 2:
        inferred.setdefault("h_hidden", int(state_dict["h_rnn.key.weight"].shape[0]))

    if "h_rnn.r_k" in state_dict and state_dict["h_rnn.r_k"].ndim == 2:
        inferred.setdefault("rwkv_head_size", int(state_dict["h_rnn.r_k"].shape[1]))

    if "l_rnn.key.weight" in state_dict and state_dict["l_rnn.key.weight"].ndim == 2:
        inferred.setdefault("l_hidden", int(state_dict["l_rnn.key.weight"].shape[0]))

    if "h_hidden" not in inferred and "context_dim" in inferred:
        inferred["h_hidden"] = inferred["context_dim"]
    if "l_hidden" not in inferred and "context_dim" in inferred:
        inferred["l_hidden"] = inferred["context_dim"]

    for key, value in LATEST_CONFIG_DEFAULTS.items():
        if key not in inferred or inferred[key] is None:
            inferred[key] = value

    inferred["compile"] = False
    inferred.setdefault("model_type", "hierarchos")
    return inferred


def _source_key_for(new_key: str, old_state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    if new_key in old_state_dict:
        return new_key

    tied_aliases = {
        "tok_emb.weight": "lm_head.weight",
        "lm_head.weight": "tok_emb.weight",
    }
    alias = tied_aliases.get(new_key)
    if alias in old_state_dict:
        return alias

    return None


def _copy_overlap_(target: torch.Tensor, source: torch.Tensor) -> Optional[Tuple[int, ...]]:
    if target.shape == source.shape:
        target.copy_(source.to(device=target.device, dtype=target.dtype))
        return tuple(target.shape)

    if target.ndim != source.ndim:
        return None

    slices = tuple(slice(0, min(new_dim, old_dim)) for new_dim, old_dim in zip(target.shape, source.shape))
    if any(s.stop == 0 for s in slices):
        return None

    source_view = source.to(device=target.device, dtype=target.dtype)
    target[slices].copy_(source_view[slices])
    return tuple(s.stop for s in slices)


def _maybe_reinitialize_missing_weight(name: str, tensor: torch.Tensor) -> bool:
    if name in LOW_VARIANCE_NEW_WEIGHTS and tensor.is_floating_point():
        nn.init.normal_(tensor, mean=0.0, std=0.01)
        return True
    return False


def _layer_note(name: str) -> str:
    if name.startswith(("h_rnn.", "l_rnn.")):
        return "RWKV cell parameter"
    if name.startswith("val_proj."):
        return "latest LTM value-projection layer"
    if name.startswith(("h_deepemb.", "l_deepemb.")):
        return "latest RWKV-v8 DeepEmbed layer"
    if name.startswith("rosa_emb.") or name == "rosa_gate_logit":
        return "latest ROSA state layer"
    if name.startswith("l_feedback_proj."):
        return "latest worker-to-manager feedback layer"
    if name.startswith("context_drift_proj."):
        return "context-drift layer"
    if name.startswith("h_halt_proj."):
        return "manager halt layer"
    if name == "ltm_gate_logit":
        return "LTM gate"
    return "new/current layer"


def scan_dataset_for_max_length(dataset_path: str, tokenizer, kayla_mode: bool, alpaca_mode: bool = False) -> int:
    """Scan a JSON or JSONL dataset and return the max token length rounded to 8."""
    max_found_length = 0
    print(f"Scanning dataset '{dataset_path}' to determine max length...")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    def get_text_from_obj(obj: Dict[str, Any], kayla: bool, alpaca: bool) -> str:
        try:
            if kayla:
                feelings_part = f"### Feelings:\n{obj.get('feelings')}\n\n" if obj.get("feelings") else ""
                return (
                    f"### Instruction:\n{obj.get('Instruction', '')}\n\n"
                    f"{feelings_part}"
                    f"### Thought Process:\n{obj.get('thought-process', '')}\n\n"
                    f"### Response:\n{obj.get('output', '')}"
                )
            if alpaca:
                input_part = f"### Input:\n{obj.get('input', '')}\n\n" if obj.get("input") else ""
                return (
                    f"### Instruction:\n{obj.get('instruction', '')}\n\n"
                    f"{input_part}"
                    f"### Response:\n{obj.get('output', '') or obj.get('response', '')}"
                )
            return (
                f"### Instruction:\n{obj.get('instruction', '')}\n\n"
                f"### Response:\n{obj.get('output', '') or obj.get('response', '')}"
            )
        except Exception:
            return ""

    with open(dataset_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                data = [data]
            if isinstance(data, list):
                for obj in tqdm(data, desc="Scanning JSON"):
                    if not isinstance(obj, dict):
                        continue
                    length = len(tokenizer.encode(get_text_from_obj(obj, kayla_mode, alpaca_mode))) + 1
                    max_found_length = max(max_found_length, length)
        except json.JSONDecodeError:
            f.seek(0)
            for line in tqdm(f, desc="Scanning JSONL"):
                try:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        continue
                    length = len(tokenizer.encode(get_text_from_obj(obj, kayla_mode, alpaca_mode))) + 1
                    max_found_length = max(max_found_length, length)
                except (json.JSONDecodeError, AttributeError, TypeError):
                    continue

    if max_found_length > 0:
        adjusted_length = (max_found_length + 16 + 7) & -8
        print(f"[OK] Auto-scan complete. max_length={adjusted_length} (found max: {max_found_length}).")
        return adjusted_length

    print("[WARN] Auto-scan did not find any valid entries.")
    return 0


def transplant_weights(old_model_path: str, new_config: Dict[str, Any], output_target: str, device: str) -> None:
    """Create a latest-layout Hierarchos model and transplant compatible old weights into it."""
    checkpoint_path, tokenizer_source_dir = _resolve_checkpoint_path(old_model_path)
    output_dir, output_weights_path = _resolve_output_paths(output_target)

    print(f"Loading old checkpoint: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path, device)
    old_state_dict = _extract_state_dict(checkpoint)

    print("Initializing latest HierarchosCore layout...")
    new_model = HierarchosCore(AttrDict(new_config)).to(device)
    new_state_dict = new_model.state_dict()

    stats = {
        "copied": 0,
        "resized": 0,
        "initialized": 0,
        "skipped": 0,
    }

    print("Transplanting weights into latest layer set...")
    for name, new_tensor in tqdm(new_state_dict.items()):
        source_key = _source_key_for(name, old_state_dict)

        if source_key is None:
            if _maybe_reinitialize_missing_weight(name, new_tensor):
                print(f"  - Initialized missing {name} ({_layer_note(name)}) with low-variance weights.")
            else:
                print(f"  - Missing {name} ({_layer_note(name)}); keeping latest initialization.")
            stats["initialized"] += 1
            continue

        old_tensor = old_state_dict[source_key]
        if not torch.is_tensor(old_tensor):
            print(f"  - Skipping {name}: source value is not a tensor.")
            stats["skipped"] += 1
            continue

        if name in DERIVED_OR_RUNTIME_KEYS and old_tensor.shape != new_tensor.shape:
            print(
                f"  - Keeping derived/runtime {name} initialized by latest code "
                f"(old {list(old_tensor.shape)} -> new {list(new_tensor.shape)})."
            )
            stats["initialized"] += 1
            continue

        if name == "qproj.weight" and old_tensor.shape != new_tensor.shape:
            if old_tensor.ndim == 2 and new_tensor.ndim == 2 and old_tensor.shape[1] * 2 == new_tensor.shape[1]:
                print(
                    "  - Adapting qproj.weight for context-aware LTM query "
                    f"(old {list(old_tensor.shape)} -> new {list(new_tensor.shape)})."
                )
            else:
                print(f"  - Resizing qproj.weight (old {list(old_tensor.shape)} -> new {list(new_tensor.shape)}).")

        copied_shape = _copy_overlap_(new_tensor, old_tensor)
        if copied_shape is None:
            print(
                f"  - Could not map {name} from {source_key} "
                f"(old {list(old_tensor.shape)} -> new {list(new_tensor.shape)}); keeping latest initialization."
            )
            stats["skipped"] += 1
        elif old_tensor.shape == new_tensor.shape:
            stats["copied"] += 1
        else:
            print(
                f"  - Partially copied {name} from {source_key}: "
                f"old {list(old_tensor.shape)} -> new {list(new_tensor.shape)}, overlap {list(copied_shape)}."
            )
            stats["resized"] += 1

    new_model.load_state_dict(new_state_dict, strict=True)

    print(f"\nSaving expanded model directory to: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    final_config = _json_safe(_plain_dict(new_model.config))

    torch.save(
        {
            "model_state_dict": new_model.state_dict(),
            "config": final_config,
        },
        output_weights_path,
    )

    with open(output_dir / "hierarchos_config.json", "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    print(
        "Transplant summary: "
        f"{stats['copied']} copied, {stats['resized']} resized, "
        f"{stats['initialized']} initialized, {stats['skipped']} skipped."
    )

    try:
        print("Copying tokenizer files...")
        old_tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source_dir), trust_remote_code=True)
        old_tokenizer.save_pretrained(str(output_dir))
        print("[OK] Tokenizer copied successfully.")
    except Exception as exc:
        print(
            f"[WARN] Could not load/save tokenizer from '{tokenizer_source_dir}'. "
            f"You may need to copy tokenizer files manually. Error: {exc}"
        )

    print("[OK] Model expansion complete.")


def build_expanded_config(args: argparse.Namespace, device: str) -> Dict[str, Any]:
    checkpoint_path, model_root = _resolve_checkpoint_path(args.old_model_path)
    print(f"Loading configuration from old checkpoint: {checkpoint_path}")
    checkpoint = _load_torch(checkpoint_path, device)
    old_state_dict = _extract_state_dict(checkpoint)
    final_config = _infer_missing_config(_load_config(checkpoint, model_root), old_state_dict)

    updated_dims = {key: getattr(args, key) for key in ARCH_UPDATE_KEYS if getattr(args, key, None) is not None}
    if updated_dims:
        print("Updating model/config values:")
        for key, value in updated_dims.items():
            print(f"  - {key}: {final_config.get(key, 'N/A')} -> {value}")
        final_config.update(updated_dims)

    current_ctx = final_config.get("context_dim")
    if args.h_hidden is None and current_ctx is not None and final_config.get("h_hidden") != current_ctx:
        print(f"  - [Auto-Sync] h_hidden: {final_config.get('h_hidden', 'N/A')} -> {current_ctx}")
        final_config["h_hidden"] = current_ctx

    if args.l_hidden is None and current_ctx is not None and final_config.get("l_hidden") != current_ctx:
        print(f"  - [Auto-Sync] l_hidden: {final_config.get('l_hidden', 'N/A')} -> {current_ctx}")
        final_config["l_hidden"] = current_ctx

    new_max_len = None
    if args.auto_max_length:
        if not args.dataset_for_length:
            raise ValueError("--auto-max-length requires --dataset-for-length.")

        try:
            tokenizer_source = model_root
            print(f"Loading tokenizer from old model path: {tokenizer_source}")
            tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            determined_len = scan_dataset_for_max_length(args.dataset_for_length, tokenizer, args.kayla, args.alpaca)
            if determined_len > 0:
                new_max_len = determined_len
        except Exception as exc:
            print(f"[WARN] Error during auto-scan for max length: {exc}. Falling back.")
    elif args.new_max_length is not None:
        new_max_len = args.new_max_length

    if new_max_len is not None:
        print(f"Updating max_length: {final_config.get('max_length', 'N/A')} -> {new_max_len}")
        final_config["max_length"] = new_max_len

    for key, value in LATEST_CONFIG_DEFAULTS.items():
        if key not in final_config or final_config[key] is None:
            final_config[key] = value

    final_config["compile"] = False
    final_config.setdefault("model_type", "hierarchos")
    return final_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand a trained Hierarchos model into the latest architecture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--old-model-path",
        type=str,
        required=True,
        help="Path to a trained model directory or .pt checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        "--output-path",
        dest="output_dir",
        type=str,
        required=True,
        help="Directory for the expanded model. Legacy .pt output paths are treated as their parent directory.",
    )

    dim_group = parser.add_argument_group("Latest Architecture Overrides")
    dim_group.add_argument("--vocab_size", "--vocab-size", dest="vocab_size", type=int)
    dim_group.add_argument("--context_dim", "--context-dim", dest="context_dim", type=int)
    dim_group.add_argument("--persistent_dim", "--persistent-dim", dest="persistent_dim", type=int)
    dim_group.add_argument("--ltm_slots", "--ltm-slots", dest="ltm_slots", type=int)
    dim_group.add_argument("--ltm_key_dim", "--ltm-key-dim", dest="ltm_key_dim", type=int)
    dim_group.add_argument("--ltm_val_dim", "--ltm-val-dim", dest="ltm_val_dim", type=int)
    dim_group.add_argument("--ltm_lr", "--ltm-lr", dest="ltm_lr", type=float)
    dim_group.add_argument("--ltm_topk", "--ltm-topk", dest="ltm_topk", type=int)
    dim_group.add_argument("--h_hidden", "--h-hidden", dest="h_hidden", type=int)
    dim_group.add_argument("--l_hidden", "--l-hidden", dest="l_hidden", type=int)
    dim_group.add_argument("--h_stride", "--h-stride", dest="h_stride", type=int)
    dim_group.add_argument("--max_h_steps", "--max-h-steps", dest="max_h_steps", type=int)
    dim_group.add_argument("--max_l_steps", "--max-l-steps", dest="max_l_steps", type=int)
    dim_group.add_argument("--l_conv_atol", "--l-conv-atol", dest="l_conv_atol", type=float)
    dim_group.add_argument("--commitment_threshold", "--commitment-threshold", dest="commitment_threshold", type=float)
    dim_group.add_argument("--detach_every_n_steps", "--detach-every-n-steps", dest="detach_every_n_steps", type=int)
    dim_group.add_argument("--h_halt_thresh", "--h-halt-thresh", dest="h_halt_thresh", type=float)
    dim_group.add_argument("--ltm_forget_rate", "--ltm-forget-rate", dest="ltm_forget_rate", type=float)
    dim_group.add_argument("--rosa_max_context", "--rosa-max-context", dest="rosa_max_context", type=int)
    dim_group.add_argument("--rwkv_head_size", "--rwkv-head-size", dest="rwkv_head_size", type=int)
    deepembed_group = dim_group.add_mutually_exclusive_group()
    deepembed_group.add_argument("--use-deepembed", dest="use_deepembed", action="store_true", default=None)
    deepembed_group.add_argument("--no-deepembed", dest="use_deepembed", action="store_false")
    rosa_group = dim_group.add_mutually_exclusive_group()
    rosa_group.add_argument("--use-rosa", dest="use_rosa", action="store_true", default=None)
    rosa_group.add_argument("--no-rosa", dest="use_rosa", action="store_false")

    length_group = parser.add_argument_group("Sequence Length Expansion")
    length_group.add_argument("--new-max-length", type=int, help="Manually specify the new maximum sequence length.")
    length_group.add_argument(
        "--auto-max-length",
        action="store_true",
        help="Scan a dataset to determine max_length. Requires --dataset-for-length.",
    )
    length_group.add_argument("--dataset-for-length", type=str, help="Dataset (.jsonl or .json) for --auto-max-length.")
    scan_format_group = length_group.add_mutually_exclusive_group()
    scan_format_group.add_argument("--kayla", action="store_true", help="Use Kayla formatting for auto length scanning.")
    scan_format_group.add_argument("--alpaca", action="store_true", help="Use Alpaca instruction/input/output formatting for auto length scanning.")

    args = parser.parse_args()
    device = "cpu"

    final_config = build_expanded_config(args, device)
    transplant_weights(args.old_model_path, final_config, args.output_dir, device)


if __name__ == "__main__":
    main()
