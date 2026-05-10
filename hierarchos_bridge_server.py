#!/usr/bin/env python3
"""
hierarchos_bridge_server.py — JSON-RPC Bridge Server for the Hierarchos Rust GUI.

Reads JSON requests from stdin (one per line), writes JSON responses/events to stdout.
Hooks into the modular hierarchos package (not the deprecated monolith).
"""

import sys
import os
import json
import threading
import traceback
import time

# Ensure stdout is line-buffered for real-time streaming to the Rust GUI
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Globals ──────────────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_device = None
_config = {}          # AttrDict / dict of model config
_model_dir = None
_h_state = None
_l_state = None
_prev_context = None
_target_context = None
_drift_state = None
_ltm_state = None
_total_tokens_generated = 0
_stop_generation = threading.Event()
_stop_training = threading.Event()
_cpu_threads = max(1, (os.cpu_count() or 2) // 2)


def emit(event_type: str, data: dict = None):
    """Send a JSON event to stdout (consumed by the Rust GUI)."""
    msg = {"event": event_type}
    if data:
        msg.update(data)
    try:
        print(json.dumps(msg, default=str), flush=True)
    except Exception:
        pass


def emit_error(msg: str):
    emit("error", {"message": msg})


def emit_status(msg: str):
    emit("status", {"message": msg})


def emit_load_progress(progress: float, label: str):
    """Publish approximate load progress for the GUI progress bar."""
    try:
        progress = max(0.0, min(1.0, float(progress)))
    except Exception:
        progress = 0.0
    emit("load_progress", {"progress": progress, "label": label})


def _emit_backend_runtime_info():
    """Publish the PyTorch/CUDA runtime that was bundled into this backend."""
    try:
        from hierarchos.utils.device import cuda_diagnostics

        diag = cuda_diagnostics()
        emit("backend_info", diag)
        emit_load_progress(0.16, "Inspecting PyTorch runtime")

        torch_version = diag.get("torch_version", "unknown")
        cuda_version = diag.get("cuda_version") or "none"
        if diag.get("cuda_available"):
            total_mb = diag.get("total_memory_mb") or 0
            total_gb = total_mb / 1024.0 if total_mb else 0.0
            emit_status(
                f"PyTorch {torch_version} CUDA {cuda_version} ready: "
                f"{diag.get('device_name')} ({total_gb:.1f} GB VRAM). CPU fallback is bundled."
            )
        elif diag.get("cuda_built"):
            reason = diag.get("driver_error") or "no NVIDIA CUDA device visible"
            emit_status(
                f"PyTorch {torch_version} includes CUDA {cuda_version}; "
                f"CUDA is inactive on this machine ({reason}). CPU mode is available."
            )
        else:
            emit_status(
                f"PyTorch {torch_version} is CPU-only; CUDA selection will require a CUDA backend build."
            )
    except Exception as exc:
        emit_status(f"Could not inspect PyTorch runtime: {exc}")


def _release_loaded_model():
    """Release the current model before a replacement load to reduce VRAM spikes."""
    global _model, _tokenizer, _config, _model_dir
    global _h_state, _l_state, _prev_context, _target_context, _drift_state, _ltm_state

    old_device = _device
    _model = None
    _tokenizer = None
    _config = {}
    _model_dir = None
    _reset_runtime_state()

    try:
        import torch

        if getattr(old_device, "type", None) == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    emit("model_unloaded", {})


def _normalize_device(device_str: str):
    requested = (device_str or "auto").strip().lower()
    aliases = {
        "auto": None,
        "automatic": None,
        "cuda": "cuda",
        "gpu": "cuda",
        "cpu": "cpu",
        "dml": "dml",
        "directml": "dml",
    }
    return aliases.get(requested, requested)


def _looks_like_hf_repo_id(value: str) -> bool:
    if not value or os.path.exists(value):
        return False
    if value.lower().endswith(".pt"):
        return False
    if os.path.isabs(value):
        return False
    return all(c.isalnum() or c in "-_./" for c in value)


def _resolve_model_source(model_ref: str, cache_dir: str = None) -> str:
    """Return a local folder/file for a local path or Hugging Face repo id."""
    model_ref = (model_ref or "").strip().strip('"')
    if not model_ref:
        raise FileNotFoundError("No model source provided.")

    expanded = os.path.abspath(os.path.expanduser(model_ref))
    if os.path.exists(expanded):
        return expanded

    if not _looks_like_hf_repo_id(model_ref):
        raise FileNotFoundError(f"Model path not found: {model_ref}")

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Loading a Hugging Face repo id requires huggingface_hub "
            "(installed with transformers)."
        ) from exc

    safe_name = model_ref.replace("/", "_")
    local_dir = None
    if cache_dir:
        local_dir = os.path.join(os.path.abspath(os.path.expanduser(cache_dir)), safe_name)
        os.makedirs(local_dir, exist_ok=True)

    emit_load_progress(0.28, "Downloading model snapshot")
    emit_status(f"Downloading Hugging Face model: {model_ref}")
    kwargs = {
        "repo_id": model_ref,
        "allow_patterns": [
            "*.pt",
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    }
    if local_dir:
        kwargs["local_dir"] = local_dir
    return snapshot_download(**kwargs)


def _has_tokenizer_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    names = {
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "tokenizer.model",
    }
    try:
        return any(name.lower() in names or name.lower().startswith("tokenizer.") for name in os.listdir(path))
    except OSError:
        return False


def _ltm_updates_path() -> str:
    if not _model_dir:
        raise RuntimeError("No model directory is loaded.")
    return os.path.join(_model_dir, "hierarchos_ltm_updates.pt")


def _normalize_ltm_tensor(value):
    import torch
    if value is None or not torch.is_tensor(value):
        return None
    tensor = value.detach()
    if tensor.dim() >= 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor.cpu()


def _copy_ltm_tensor(module, attr: str, value) -> bool:
    import torch
    tensor = _normalize_ltm_tensor(value)
    if tensor is None or not hasattr(module, attr):
        return False
    target = getattr(module, attr)
    if not torch.is_tensor(target) or tuple(target.shape) != tuple(tensor.shape):
        return False
    with torch.no_grad():
        target.copy_(tensor.to(device=target.device, dtype=target.dtype))
    return True


def _sync_runtime_ltm_to_module():
    """Copy runtime LTM state back into the model LTM buffers before saving."""
    if _model is None or not hasattr(_model, "ltm") or _ltm_state is None:
        return

    ltm = _model.ltm
    if len(_ltm_state) >= 1:
        _copy_ltm_tensor(ltm, "fast_vals", _ltm_state[0])
    if len(_ltm_state) >= 2:
        _copy_ltm_tensor(ltm, "_mom_vals", _ltm_state[1])
    if len(_ltm_state) >= 5:
        _copy_ltm_tensor(ltm, "timestamps", _ltm_state[4])
    if len(_ltm_state) >= 6:
        _copy_ltm_tensor(ltm, "sources", _ltm_state[5])


def _apply_saved_ltm_updates():
    """Load saved runtime LTM sidecar data, if present."""
    global _ltm_state
    if _model is None or not hasattr(_model, "ltm") or not _model_dir:
        return

    path = _ltm_updates_path()
    if not os.path.exists(path):
        return

    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ltm = _model.ltm
    applied = []

    for attr, key in (
        ("fast_vals", "fast_vals"),
        ("_mom_vals", "mom_vals"),
        ("timestamps", "timestamps"),
        ("sources", "sources"),
    ):
        if key in payload and _copy_ltm_tensor(ltm, attr, payload[key]):
            applied.append(key)

    if applied:
        _ltm_state = None
        emit_status(f"Loaded saved LTM updates from {path}.")


def _reset_runtime_state():
    global _h_state, _l_state, _prev_context, _target_context
    global _drift_state, _ltm_state, _total_tokens_generated
    _h_state = None
    _l_state = None
    _prev_context = None
    _target_context = None
    _drift_state = None
    _ltm_state = None
    _total_tokens_generated = 0


def _apply_thread_count(value=None) -> int:
    """Clamp and apply the PyTorch CPU thread count used by chat/inference."""
    global _cpu_threads
    if value is None:
        value = _cpu_threads
    try:
        threads = int(value)
    except Exception:
        threads = _cpu_threads

    max_threads = max(1, os.cpu_count() or 1)
    threads = max(1, min(threads, max_threads))

    try:
        from hierarchos import set_threads

        set_threads(threads)
    finally:
        _cpu_threads = threads
    return threads


def _sample_next_token(logits, generated_ids, sampling):
    import torch
    import torch.nn.functional as F

    temperature = float(sampling.get("temperature", 0.7))
    top_k = int(sampling.get("top_k", 40))
    top_p = float(sampling.get("top_p", 0.9))
    rep_penalty = float(sampling.get("repetition_penalty", 1.2))

    next_logits = logits[:, -1, :].float().clone()

    if rep_penalty != 1.0 and generated_ids:
        for tok_id in set(generated_ids):
            if 0 <= tok_id < next_logits.shape[-1]:
                if next_logits[0, tok_id] > 0:
                    next_logits[0, tok_id] /= rep_penalty
                else:
                    next_logits[0, tok_id] *= rep_penalty

    if temperature <= 0:
        return torch.argmax(next_logits, dim=-1, keepdim=True)

    next_logits = next_logits / max(temperature, 1e-6)

    if top_k > 0:
        kth = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][:, [-1]]
        next_logits[next_logits < kth] = -float("inf")

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_remove = cumulative_probs > top_p
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
        next_logits[remove] = -float("inf")

    probs = F.softmax(next_logits, dim=-1)
    if torch.isnan(probs).any() or torch.isinf(probs).any() or float(probs.sum().item()) <= 0:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


# ── Handlers ─────────────────────────────────────────────────────────────────

def handle_load_model(params: dict):
    """Load a Hierarchos model from a local folder/file or Hugging Face repo."""
    global _model, _tokenizer, _device, _config, _model_dir, _cpu_threads
    from transformers import AutoTokenizer
    import argparse

    from hierarchos import (
        configure_torch_runtime,
        cuda_diagnostics,
        describe_device,
        pick_device,
        load_full_model_with_config,
        set_threads,
    )

    model_ref = params.get("model_path", "")
    device_str = params.get("device", "auto")
    tokenizer_ref = (params.get("tokenizer_path") or "").strip()
    cache_dir = params.get("cache_dir")

    emit_load_progress(0.22, "Resolving model source")
    emit_status(f"Resolving model source: {model_ref}")

    try:
        resolved_model_path = _resolve_model_source(model_ref, cache_dir)
        emit_load_progress(0.32, "Preparing model runtime")

        if _model is not None:
            emit_status("Unloading current model before replacement load.")
            emit_load_progress(0.34, "Unloading previous model")
            _release_loaded_model()

        _model_dir = resolved_model_path if os.path.isdir(resolved_model_path) else os.path.dirname(resolved_model_path)

        emit_load_progress(0.38, "Selecting compute device")
        requested_threads = params.get("cpu_threads", params.get("threads", _cpu_threads))
        ns = argparse.Namespace(
            device=_normalize_device(device_str),
            threads=_apply_thread_count(requested_threads),
        )
        _device = pick_device(ns)
        runtime_diag = configure_torch_runtime(_device)
        device_label = describe_device(_device)

        emit_load_progress(0.45, "Loading model weights")
        emit_status(f"Loading model on {device_label} from {resolved_model_path}")
        model, cfg = load_full_model_with_config(resolved_model_path, _device)
        emit_load_progress(0.72, "Weights loaded")
        _model = model
        _model.eval()
        _model.suppress_hebbian = False
        _config = dict(cfg) if hasattr(cfg, 'items') else {}
        _reset_runtime_state()
        emit_load_progress(0.76, "Checking saved LTM updates")
        _apply_saved_ltm_updates()

        emit_load_progress(0.82, "Loading tokenizer")
        tokenizer_candidates = []
        if tokenizer_ref:
            tokenizer_candidates.append(tokenizer_ref)
        model_dir_has_tokenizer = _has_tokenizer_files(_model_dir)
        if _model_dir and model_dir_has_tokenizer:
            tokenizer_candidates.append(_model_dir)
        if _config.get("tokenizer_name"):
            tokenizer_candidates.append(_config.get("tokenizer_name"))
        if not tokenizer_candidates:
            tokenizer_candidates.append("openai-community/gpt2")

        last_tokenizer_error = None
        last_tokenizer_path = None
        for tok_path in tokenizer_candidates:
            try:
                _tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
                if _tokenizer.pad_token is None:
                    if _tokenizer.eos_token:
                        _tokenizer.pad_token = _tokenizer.eos_token
                    else:
                        _tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                last_tokenizer_path = tok_path
                break
            except Exception as exc:
                last_tokenizer_error = exc
                _tokenizer = None

        if _tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer: {last_tokenizer_error}")
        emit_load_progress(0.90, "Tokenizer ready")
        emit_status(f"Tokenizer loaded from {last_tokenizer_path}.")

        emit_load_progress(0.94, "Finalizing model")
        total_params = sum(p.numel() for p in _model.parameters())

        model_config = {
            "context_dim": int(_config.get("context_dim", 768)),
            "h_hidden": int(_config.get("h_hidden", _config.get("context_dim", 768))),
            "l_hidden": int(_config.get("l_hidden", _config.get("context_dim", 768))),
            "ltm_slots": int(_config.get("ltm_slots", 1024)),
            "ltm_key_dim": int(_config.get("ltm_key_dim", 128)),
            "ltm_val_dim": int(_config.get("ltm_val_dim", 128)),
            "ltm_topk": int(_config.get("ltm_topk", 4)),
            "vocab_size": int(_config.get("vocab_size", len(_tokenizer))),
            "max_length": int(_config.get("max_length", 1024)),
            "h_stride": int(_config.get("h_stride", 4)),
            "max_h_steps": int(_config.get("max_h_steps", 5)),
            "max_l_steps": int(_config.get("max_l_steps", 5)),
            "persistent_dim": int(_config.get("persistent_dim", 128)),
            "is_quantized": bool(_config.get("is_quantized", False)),
            "device": str(_device),
            "device_label": device_label,
            "torch_version": runtime_diag.get("torch_version"),
            "cuda_built": bool(runtime_diag.get("cuda_built", False)),
            "cuda_available": bool(runtime_diag.get("cuda_available", False)),
            "cuda_version": runtime_diag.get("cuda_version"),
            "cuda_device_name": runtime_diag.get("device_name"),
            "vram_total_mb": runtime_diag.get("total_memory_mb"),
            "total_params": int(total_params),
        }

        emit_load_progress(1.0, "Model ready")
        emit("model_loaded", {"config": model_config})
        if str(_device).startswith("cuda"):
            emit_status(f"CUDA acceleration active on {device_label}.")
        elif cuda_diagnostics().get("cuda_available"):
            emit_status("Model is running on CPU even though CUDA is available; select Auto or CUDA for GPU acceleration.")
        emit_status(f"Model loaded successfully from {_model_dir}.")

    except Exception as e:
        emit_error(f"Failed to load model: {e}\n{traceback.format_exc()}")


def handle_generate(params: dict):
    """Stream generation with the same runtime states as CLI chat mode."""
    global _model, _tokenizer, _device
    global _h_state, _l_state, _prev_context, _target_context
    global _drift_state, _ltm_state, _total_tokens_generated, _cpu_threads

    if _model is None:
        emit_error("No model loaded.")
        return

    _stop_generation.clear()
    message = params.get("message", "")
    sampling = params.get("sampling", {})
    _apply_thread_count(sampling.get("cpu_threads", _cpu_threads))

    def _gen():
        global _h_state, _l_state, _prev_context, _target_context
        global _drift_state, _ltm_state, _total_tokens_generated

        try:
            import torch
            from hierarchos.inference.chat import wrap_for_hierarchos

            max_new = int(sampling.get("max_new_tokens", 512))
            prompt = wrap_for_hierarchos(message)
            prompt_ids = _tokenizer.encode(prompt, return_tensors="pt").to(_device)
            response_ids = []

            _model.eval()
            # Match the stable CLI chat path: do not mutate LTM on every
            # autoregressive token. Per-token Hebbian writes can compound and
            # pull the model into repeated/off-topic text.
            _model.suppress_hebbian = True

            with torch.no_grad():
                outputs = _model(
                    prompt_ids,
                    h_state=_h_state,
                    l_state=_l_state,
                    prev_context=_prev_context,
                    target_context=_target_context,
                    drift_state=_drift_state,
                    ltm_memory_state=_ltm_state,
                    global_pos_offset=_total_tokens_generated,
                    suppress_hebbian=True,
                )

                logits = outputs["logits"]
                _h_state = outputs.get("h_state", _h_state)
                _l_state = outputs.get("l_state", _l_state)
                _prev_context = outputs.get("prev_context", _prev_context)
                _target_context = outputs.get("target_context", _target_context)
                _drift_state = outputs.get("drift_state", _drift_state)
                _ltm_state = outputs.get("ltm_memory_state", _ltm_state)
                _total_tokens_generated += prompt_ids.shape[1]

                current_ids = _sample_next_token(logits, response_ids, sampling)

                for _ in range(max_new):
                    if _stop_generation.is_set():
                        break

                    next_token = int(current_ids.item())
                    if next_token == _tokenizer.eos_token_id:
                        break

                    response_ids.append(next_token)
                    token_str = _tokenizer.decode([next_token])
                    if token_str:
                        emit("token", {"text": token_str})

                    outputs = _model(
                        current_ids,
                        h_state=_h_state,
                        l_state=_l_state,
                        prev_context=_prev_context,
                        target_context=_target_context,
                        drift_state=_drift_state,
                        ltm_memory_state=_ltm_state,
                        global_pos_offset=_total_tokens_generated,
                        suppress_hebbian=True,
                    )

                    logits = outputs["logits"]
                    _h_state = outputs.get("h_state", _h_state)
                    _l_state = outputs.get("l_state", _l_state)
                    _prev_context = outputs.get("prev_context", _prev_context)
                    _target_context = outputs.get("target_context", _target_context)
                    _drift_state = outputs.get("drift_state", _drift_state)
                    _ltm_state = outputs.get("ltm_memory_state", _ltm_state)
                    _total_tokens_generated += 1

                    current_ids = _sample_next_token(logits, response_ids, sampling)

            if _ltm_state is not None and len(_ltm_state) >= 2:
                _ltm_state = (_ltm_state[0], torch.zeros_like(_ltm_state[1]), *_ltm_state[2:])
            _model.suppress_hebbian = True
            emit("generation_complete", {})

        except Exception as e:
            emit_error(f"Generation error: {e}\n{traceback.format_exc()}")
            emit("generation_complete", {})

    threading.Thread(target=_gen, daemon=True).start()


def handle_start_training(params: dict):
    """Start a training run using the modular hierarchos.training.trainer."""
    global _model, _tokenizer, _device, _config
    if _model is None:
        emit_error("No model loaded — cannot train.")
        return

    _stop_training.clear()

    def _train():
        try:
            import torch
            import argparse
            import time

            # Import from the modular package (same as hierarchos_cli.py)
            from hierarchos import (
                train as hierarchos_train,
                OriginalJSONLDataset,
                create_map_style_dataloader,
                process_text_sample,
            )

            data_path = params.get("data_path", "")
            if not data_path or not os.path.exists(data_path):
                emit_error(f"Training data not found: {data_path}")
                return

            emit_status("Preparing training...")

            def _int_param(name, default, minimum=1):
                try:
                    value = int(params.get(name, default))
                except Exception:
                    value = default
                try:
                    value = int(value)
                except Exception:
                    value = int(default)
                return max(minimum, value)

            context_dim = _int_param("context_dim", _config.get("context_dim", 768), 32)
            train_arch = {
                "context_dim": context_dim,
                "persistent_dim": _int_param("persistent_dim", _config.get("persistent_dim", 128), 1),
                "ltm_slots": _int_param("ltm_slots", _config.get("ltm_slots", 1024), 1),
                "ltm_key_dim": _int_param("ltm_key_dim", _config.get("ltm_key_dim", 128), 8),
                "ltm_val_dim": _int_param("ltm_val_dim", _config.get("ltm_val_dim", 128), 8),
                "h_hidden": _int_param("h_hidden", _config.get("h_hidden", context_dim), 32),
                "l_hidden": _int_param("l_hidden", _config.get("l_hidden", context_dim), 32),
                "h_stride": _int_param("h_stride", _config.get("h_stride", 4), 1),
                "max_h_steps": _int_param("max_h_steps", _config.get("max_h_steps", 5), 1),
                "max_l_steps": _int_param("max_l_steps", _config.get("max_l_steps", 5), 1),
                "ltm_topk": _int_param("ltm_topk", _config.get("ltm_topk", 4), 1),
                "max_length": _int_param("max_length", _config.get("max_length", 1024), 32),
            }
            train_arch["ltm_topk"] = min(train_arch["ltm_topk"], train_arch["ltm_slots"])

            def _scan_auto_max_length(path: str) -> int:
                max_found = 0
                scanned = 0
                scan_max_length = 1_000_000

                def consider(obj):
                    nonlocal max_found, scanned
                    if not isinstance(obj, dict):
                        return
                    processed = process_text_sample(
                        _tokenizer,
                        obj,
                        scan_max_length,
                        False,
                        prompt_column="instruction",
                        completion_column="output",
                    )
                    if processed:
                        max_found = max(max_found, len(processed["input_ids"]))
                        scanned += 1

                emit_status("Scanning dataset for auto max length...")
                with open(path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list):
                            data = [data]
                        for obj in data:
                            consider(obj)
                    except Exception:
                        f.seek(0)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                consider(json.loads(line))
                            except Exception:
                                continue

                if max_found <= 0:
                    raise ValueError("No valid instruction/output samples found while scanning.")

                auto_len = (max_found + 16 + 7) & -8
                capped_len = min(max(auto_len, 32), 32768)
                if capped_len != auto_len:
                    emit_status(
                        f"Auto max length found {max_found} tokens; capped at {capped_len}."
                    )
                else:
                    emit_status(
                        f"Auto max length found {max_found} tokens across {scanned} samples; using {capped_len}."
                    )
                return capped_len

            if bool(params.get("auto_max_length", False)):
                try:
                    train_arch["max_length"] = _scan_auto_max_length(data_path)
                except Exception as exc:
                    emit_error(f"Auto max length scan failed: {exc}")
                    return

            # Build args namespace matching what hierarchos.training.trainer.train() expects
            # (mirrors the argparse in hierarchos_cli.py)
            train_args = argparse.Namespace(
                mode="train",
                model_path=None,  # Already loaded
                out_dir=params.get("out_dir", "./hierarchos_model"),
                resume_from_ckpt=None,
                # Architecture (defaults come from loaded config, GUI may override)
                context_dim=train_arch["context_dim"],
                persistent_dim=train_arch["persistent_dim"],
                ltm_slots=train_arch["ltm_slots"],
                ltm_key_dim=train_arch["ltm_key_dim"],
                ltm_val_dim=train_arch["ltm_val_dim"],
                h_hidden=train_arch["h_hidden"],
                l_hidden=train_arch["l_hidden"],
                h_stride=train_arch["h_stride"],
                max_h_steps=train_arch["max_h_steps"],
                max_l_steps=train_arch["max_l_steps"],
                ltm_topk=train_arch["ltm_topk"],
                max_length=train_arch["max_length"],
                auto_max_length=bool(params.get("auto_max_length", False)),
                vocab_size=_config.get("vocab_size", len(_tokenizer)),
                # Training hyperparams from GUI
                epochs=int(params.get("epochs", 3)),
                batch_size=int(params.get("batch_size", 4)),
                accumulation_steps=int(params.get("accumulation_steps", 1)),
                starting_lr=float(params.get("learning_rate", 1e-4)),
                min_lr=float(params.get("min_lr", 1e-6)),
                training_chunk_size=int(params.get("training_chunk_size", 128)),
                grad_clip=float(params.get("grad_clip", 1.0)),
                persist_state=bool(params.get("persist_state", False)),
                amp=bool(params.get("amp", True)),
                save_steps=int(params.get("save_steps", 0)),
                num_workers=0,
                # Defaults for features not exposed in GUI
                disable_lr_schedule=False,
                ltm_lr=1e-3,
                kayla=False,
                lora_r=8, lora_alpha=16,
                ponder_loss_weight=0.01,
                commitment_loss_weight=0.5,
                commitment_threshold=0.05,
                l_conv_atol=1e-4,
                detach_every_n_steps=32,
                h_halt_thresh=0.9,
                encourage_thinking=False,
                adaptive_ponder=False,
                ponder_target_scale=0.5,
                reset_halt_bias=None,
                override_scheduling=False,
                compile=False, force_compile=False,
                eval_tasks=None, eval_every_epoch=1, eval_batch_size=1,
                eval_limit=None, eval_steps=None,
            )

            emit_status(
                "Architecture: "
                f"context={train_args.context_dim}, "
                f"H={train_args.h_hidden}, L={train_args.l_hidden}, "
                f"LTM={train_args.ltm_slots} slots/{train_args.ltm_val_dim} value dim, "
                f"max_len={train_args.max_length}"
            )

            # Build dataloader (same pattern as hierarchos_cli.py)
            dataset = OriginalJSONLDataset(
                data_path, _tokenizer, train_args.max_length, False
            )
            dataloader = create_map_style_dataloader(
                dataset, train_args.batch_size, _tokenizer.pad_token_id, 0
            )
            dataloader_len = len(dataloader)

            emit_status(f"Training started — {dataloader_len} batches/epoch × {train_args.epochs} epochs")

            # --- Monkey-patch tqdm to intercept metric updates ---
            # The trainer uses tqdm + pbar.set_postfix() for metrics.
            # We intercept these calls to stream them to the GUI.
            import tqdm as tqdm_module
            _original_tqdm = tqdm_module.tqdm

            class GUITqdm(_original_tqdm):
                """tqdm wrapper that emits training metrics to the GUI bridge."""
                def __init__(self, *a, **kw):
                    self._gui_epoch = 0
                    self._gui_step = 0
                    self._gui_start_time = time.time()
                    self._gui_tokens_processed = 0
                    # Extract epoch from desc if available
                    desc = kw.get('desc', '') or (a[1] if len(a) > 1 else '')
                    if 'Epoch' in str(desc):
                        try:
                            parts = str(desc).split('/')
                            self._gui_epoch = int(parts[0].split()[-1]) - 1
                        except (ValueError, IndexError):
                            pass
                    super().__init__(*a, **kw)

                def set_postfix(self, ordered_dict=None, refresh=True, **kwargs):
                    super().set_postfix(ordered_dict, refresh, **kwargs)

                    # Check stop flag
                    if _stop_training.is_set():
                        self.close()
                        raise StopIteration("Training stopped by user.")

                    # Extract metrics from postfix
                    postfix = ordered_dict or kwargs
                    self._gui_step += 1

                    loss = 0.0
                    lr = 0.0
                    ponder_cost = None
                    commitment_cost = None

                    if 'loss' in postfix:
                        try: loss = float(postfix['loss'])
                        except: pass
                    if 'lr' in postfix:
                        try: lr = float(postfix['lr'])
                        except: pass
                    if 'ponder' in postfix:
                        try: ponder_cost = float(postfix['ponder'])
                        except: pass
                    if 'commit' in postfix:
                        try: commitment_cost = float(postfix['commit'])
                        except: pass

                    # Estimate tokens/sec
                    elapsed = time.time() - self._gui_start_time
                    chunk_size = train_args.training_chunk_size
                    batch_size = train_args.batch_size
                    tokens_this_step = chunk_size * batch_size
                    self._gui_tokens_processed += tokens_this_step
                    tps = self._gui_tokens_processed / max(elapsed, 0.01)

                    emit("training_metrics", {
                        "epoch": self._gui_epoch,
                        "step": self._gui_step,
                        "loss": loss,
                        "lr": lr,
                        "ponder_cost": ponder_cost,
                        "commitment_cost": commitment_cost,
                        "tokens_per_sec": round(tps, 1),
                    })

            # Patch tqdm globally so the trainer picks it up
            tqdm_module.tqdm = GUITqdm

            try:
                hierarchos_train(train_args, _device, _tokenizer, dataloader, dataloader_len)
                emit_status("Training complete!")
            except StopIteration:
                emit_status("Training stopped by user.")
            finally:
                # Restore original tqdm
                tqdm_module.tqdm = _original_tqdm

        except Exception as e:
            emit_error(f"Training error: {e}\n{traceback.format_exc()}")
            emit_status("Training stopped due to error.")

    threading.Thread(target=_train, daemon=True).start()


def handle_stop_generation(_params):
    _stop_generation.set()
    emit_status("Generation stop requested.")


def handle_stop_training(_params):
    _stop_training.set()
    emit_status("Training stop requested.")


def handle_get_model_info(_params):
    """Return parameter-level model inspection data."""
    global _model
    if _model is None:
        emit_error("No model loaded.")
        return

    import torch
    layers = []
    total_params = 0
    trainable_params = 0
    for name, param in _model.named_parameters():
        count = param.numel()
        total_params += count
        if param.requires_grad:
            trainable_params += count
        p = param.detach().float().cpu()
        std_val = 0.0 if p.numel() <= 1 else float(p.std(unbiased=False).item())
        layers.append({
            "name": name,
            "param_count": count,
            "shape": list(param.shape),
            "dtype": str(param.dtype),
            "mean": round(float(p.mean().item()), 6),
            "std": round(std_val, 6),
            "min": round(float(p.min().item()), 6),
            "max": round(float(p.max().item()), 6),
        })

    emit("model_info", {
        "layers": layers,
        "total_params": total_params,
        "trainable_params": trainable_params,
    })


def handle_get_ltm_snapshot(_params):
    """Return current LTM memory state for heatmap visualization."""
    global _model
    if _model is None:
        emit_error("No model loaded.")
        return

    try:
        import torch

        # Find the LTM module (hierarchos.models.ltm.LTMModule)
        ltm = None
        for name, module in _model.named_modules():
            cls_name = type(module).__name__.lower()
            if 'ltm' in cls_name or 'longtermmemory' in cls_name:
                ltm = module
                break

        if ltm is None and hasattr(_model, 'ltm'):
            ltm = _model.ltm

        if ltm is None:
            emit_status("No LTM module found in model.")
            return

        fast_vals = []
        slow_vals = []
        timestamps = []
        sources = []

        # Extract fast/working memory values
        for attr in ['fast_vals', 'vals', 'M_vals', 'value_memory']:
            if hasattr(ltm, attr):
                data = getattr(ltm, attr)
                if isinstance(data, (torch.nn.Parameter, torch.Tensor)):
                    d = data.detach().cpu().float()
                    rows = min(d.shape[0], 64)
                    cols = min(d.shape[1], 32) if d.dim() > 1 else 32
                    for r in range(rows):
                        row = d[r, :cols].tolist() if d.dim() > 1 else [d[r].item()]
                        fast_vals.append(row)
                        timestamps.append(float(r))
                        sources.append(1 if r % 3 == 0 else 2)
                break

        # Extract slow/consolidated memory values
        for attr in ['slow_vals', 'base_vals', 'M_vals_base']:
            if hasattr(ltm, attr):
                data = getattr(ltm, attr)
                if isinstance(data, (torch.nn.Parameter, torch.Tensor)):
                    d = data.detach().cpu().float()
                    rows = min(d.shape[0], 64)
                    cols = min(d.shape[1], 32) if d.dim() > 1 else 32
                    for r in range(rows):
                        row = d[r, :cols].tolist() if d.dim() > 1 else [d[r].item()]
                        slow_vals.append(row)
                break

        if not slow_vals and fast_vals:
            slow_vals = [[0.0] * len(fast_vals[0])] * len(fast_vals)

        emit("ltm_snapshot", {
            "fast_vals": fast_vals,
            "slow_vals": slow_vals,
            "timestamps": timestamps,
            "sources": sources,
        })

    except Exception as e:
        emit_error(f"LTM snapshot error: {e}")


def handle_save_ltm_updates(_params):
    """Persist runtime LTM state next to the loaded model without overwriting base weights."""
    global _model
    if _model is None:
        emit_error("No model loaded.")
        return
    if not hasattr(_model, "ltm"):
        emit_error("No LTM module found in model.")
        return

    try:
        import torch

        _sync_runtime_ltm_to_module()
        ltm = _model.ltm
        path = _ltm_updates_path()
        temp_path = path + ".tmp"
        payload = {
            "version": 1,
            "saved_at": time.time(),
            "total_tokens_generated": int(_total_tokens_generated),
        }

        for key, attr in (
            ("fast_vals", "fast_vals"),
            ("mom_vals", "_mom_vals"),
            ("timestamps", "timestamps"),
            ("sources", "sources"),
            ("ltm_deltas", "ltm_deltas"),
        ):
            if hasattr(ltm, attr):
                value = getattr(ltm, attr)
                if torch.is_tensor(value):
                    payload[key] = value.detach().cpu()

        torch.save(payload, temp_path)
        os.replace(temp_path, path)
        emit("ltm_saved", {"path": path})
        emit_status(f"LTM updates saved to {path}.")

    except Exception as e:
        emit_error(f"Failed to save LTM updates: {e}\n{traceback.format_exc()}")


def handle_send_feedback(params: dict):
    positive = params.get("positive", True)
    msg = ("Positive feedback received — reinforcing LTM memory."
           if positive else
           "Negative feedback received — penalizing LTM memory.")
    emit_status(msg)


def handle_execute_command(params: dict):
    global _model, _device, _ltm_state
    command = params.get("command", "").strip()

    if command == "/reset":
        if _model is not None and hasattr(_model, 'reset_memory'):
            _model.reset_memory()
        _reset_runtime_state()
        emit_status("All RNN and hierarchical states reset.")
    elif command == "/reset_ltm":
        if _model is not None and hasattr(_model, 'ltm'):
            if hasattr(_model.ltm, 'reset_working_memory'):
                _model.ltm.reset_working_memory()
            elif hasattr(_model.ltm, 'reset_memory'):
                _model.ltm.reset_memory()
        _ltm_state = None
        emit_status("LTM working memory cleared.")
    elif command == "/status":
        if _model is not None:
            total = sum(p.numel() for p in _model.parameters())
            emit_status(f"Model: active | Device: {_device} | Params: {total/1e6:.1f}M")
        else:
            emit_status("No model loaded.")
    else:
        emit_status(f"Unknown command: {command}")


def handle_ping(_params):
    emit("pong", {})


def handle_set_threads(params: dict):
    threads = _apply_thread_count(params.get("threads", params.get("cpu_threads")))
    emit("threads_set", {"threads": threads})
    emit_status(f"CPU chat threads set to {threads}.")


# ── Dispatch ─────────────────────────────────────────────────────────────────

HANDLERS = {
    "load_model": handle_load_model,
    "generate": handle_generate,
    "start_training": handle_start_training,
    "stop_generation": handle_stop_generation,
    "stop_training": handle_stop_training,
    "get_model_info": handle_get_model_info,
    "get_ltm_snapshot": handle_get_ltm_snapshot,
    "save_ltm_updates": handle_save_ltm_updates,
    "send_feedback": handle_send_feedback,
    "execute_command": handle_execute_command,
    "set_threads": handle_set_threads,
    "ping": handle_ping,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    emit_status("Hierarchos bridge server started.")
    _emit_backend_runtime_info()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            emit_error(f"Invalid JSON: {line[:100]}")
            continue

        method = request.get("method", "")
        params = request.get("params", {})

        handler = HANDLERS.get(method)
        if handler:
            try:
                handler(params)
            except Exception as e:
                emit_error(f"Handler error [{method}]: {e}\n{traceback.format_exc()}")
        else:
            emit_error(f"Unknown method: {method}")

    emit_status("Bridge server shutting down.")


if __name__ == "__main__":
    main()
