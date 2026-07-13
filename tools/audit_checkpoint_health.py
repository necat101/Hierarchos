"""Reproducible full-precision Hierarchos checkpoint health and parity audit."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos.inference.chat import (  # noqa: E402
    advance_chat_model_state,
    boundary_drift_seed,
    tbptt_chunk_ranges,
)
from hierarchos.utils.checkpoint import (  # noqa: E402
    _resolve_weights_path,
    load_full_model_with_config,
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stats(tensor):
    value = tensor.detach().float()
    return {
        "mean": value.mean().item(),
        "std": value.std(unbiased=False).item(),
        "min": value.min().item(),
        "max": value.max().item(),
        "abs_max": value.abs().max().item(),
    }


def _gate_stats(model, input_ids, name):
    base = getattr(model, f"{name}_gate_logit", None)
    if base is None:
        return None
    embedded = model.tok_emb(input_ids)
    router = getattr(model, f"{name}_router", None)
    logits = base + router(embedded) if router is not None else base
    return _stats(torch.sigmoid(torch.clamp(logits, -50.0, 50.0)))


def _make_parity_ids(tokenizer, target_length, vocab_size):
    text = (
        "### Instruction:\nHello\n\n### Response:\n"
        "Hello! I am here to help. Please tell me what you would like to work on. "
    )
    seed = tokenizer.encode(text, add_special_tokens=False)
    if not seed:
        seed = [0]
    ids = (seed * ((target_length + len(seed) - 1) // len(seed)))[:target_length]
    ids = [int(token) % int(vocab_size) for token in ids]
    return torch.tensor([ids], dtype=torch.long)


def _run_partition(model, token_ids, chunk_size, *, one_token):
    state = (None, None, None, None, None, None)
    offset = 0
    logits = []
    ranges = (
        [(index, index + 1) for index in range(token_ids.shape[1])]
        if one_token
        else tbptt_chunk_ranges(token_ids.shape[1], chunk_size, global_offset=0)
    )
    for start, end in ranges:
        outputs, state = advance_chat_model_state(
            model,
            token_ids[:, start:end],
            device=torch.device("cpu"),
            h_state=state[0],
            l_state=state[1],
            prev_context=state[2],
            target_context=state[3],
            drift_state=state[4],
            drift_seed=boundary_drift_seed(state[4], offset, chunk_size),
            ltm_state=state[5],
            global_pos_offset=offset,
        )
        logits.append(outputs["logits"].detach().float())
        offset += end - start
    return torch.cat(logits, dim=1), state


def audit_checkpoint(model_path, parity_tokens=24, include_hash=False):
    weights_path, model_dir = _resolve_weights_path(model_path)
    model, config = load_full_model_with_config(model_path, torch.device("cpu"))
    model.eval()
    model.suppress_hebbian = True

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if len(tokenizer) != int(config.vocab_size):
        raise ValueError(
            f"Tokenizer/checkpoint vocabulary mismatch: {len(tokenizer)} != {config.vocab_size}"
        )

    state = model.state_dict()
    nonfinite = [
        name for name, tensor in state.items()
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor))
        and not torch.isfinite(tensor).all()
    ]
    unique_parameters = sum(parameter.numel() for parameter in model.parameters())
    tied_embeddings = model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

    prompt_ids = torch.tensor(
        [tokenizer.encode("### Instruction:\nHello\n\n### Response:\n", add_special_tokens=False)],
        dtype=torch.long,
    )
    with torch.no_grad():
        prompt_output = model(prompt_ids, suppress_hebbian=True)
        next_logits = prompt_output["logits"][0, -1].float()
        probabilities = F.softmax(next_logits, dim=-1)
        top_probabilities, top_indices = torch.topk(probabilities, k=5)

    model.zero_grad(set_to_none=True)
    writer_output = model(
        prompt_ids,
        labels=prompt_ids,
        suppress_hebbian=True,
        compute_ltm_value_alignment=True,
        return_logits=False,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
    )
    writer_cost = writer_output["ltm_value_alignment_cost"]
    writer_cost.backward()
    writer_grad = model.val_proj.weight.grad
    other_writer_grads = [
        name
        for name, parameter in model.named_parameters()
        if name != "val_proj.weight" and parameter.grad is not None
    ]

    chunk_size = max(1, int(getattr(config, "training_chunk_size", 256) or 256))
    parity_ids = _make_parity_ids(tokenizer, max(2, parity_tokens), config.vocab_size)
    with torch.no_grad():
        chunked_logits, chunked_state = _run_partition(
            model, parity_ids, chunk_size, one_token=False
        )
        streamed_logits, streamed_state = _run_partition(
            model, parity_ids, chunk_size, one_token=True
        )
    logit_delta = (chunked_logits - streamed_logits).abs()
    state_deltas = {}
    for index, name in enumerate(
        ("h_state", "l_state", "prev_context", "target_context", "drift_state")
    ):
        left, right = chunked_state[index], streamed_state[index]
        if torch.is_tensor(left) and torch.is_tensor(right):
            state_deltas[name] = (left.float() - right.float()).abs().max().item()

    deepembed = {}
    for name in ("h_deepemb", "l_deepemb"):
        module = getattr(model, name, None)
        if module is None:
            continue
        parameter = module.weight if hasattr(module, "weight") else module
        stats = _stats(parameter)
        sample = parameter.detach().float()[:: max(1, parameter.shape[0] // 1024)]
        token_centered = sample - sample.mean(dim=0, keepdim=True)
        stats["sample_token_centered_rms"] = token_centered.square().mean().sqrt().item()
        stats["mean_abs_distance_from_identity"] = (sample - 1.0).abs().mean().item()
        deepembed[name] = stats

    report = {
        "weights_path": weights_path,
        "file_bytes": os.path.getsize(weights_path),
        "sha256": _sha256(weights_path) if include_hash else None,
        "state_tensors": len(state),
        "unique_parameters": unique_parameters,
        "tied_embeddings": tied_embeddings,
        "nonfinite_state_tensors": nonfinite,
        "config": {
            "vocab_size": int(config.vocab_size),
            "context_dim": int(config.context_dim),
            "h_hidden": int(config.h_hidden),
            "l_hidden": int(config.l_hidden),
            "rwkv_head_size": int(config.rwkv_head_size),
            "training_chunk_size": chunk_size,
            "ltm_training_mode": getattr(config, "ltm_training_mode", "inner-update"),
            "val_proj_trained": bool(getattr(config, "val_proj_trained", False)),
        },
        "prompt_top5": [
            {
                "token_id": int(index),
                "token": tokenizer.decode([int(index)]),
                "probability": float(probability),
            }
            for probability, index in zip(top_probabilities, top_indices)
        ],
        "gates": {
            "ltm": _gate_stats(model, prompt_ids, "ltm"),
            "rosa": _gate_stats(model, prompt_ids, "rosa"),
        },
        "hebbian_writer": {
            "config_marks_trained": bool(getattr(config, "val_proj_trained", False)),
            "normalized_alignment_cost": writer_cost.detach().item(),
            "val_proj_gradient_norm": writer_grad.detach().float().norm().item(),
            "other_auxiliary_gradient_tensors": other_writer_grads,
            "weight": _stats(model.val_proj.weight),
        },
        "deepembed": deepembed,
        "out_norm": _stats(model.out_norm.weight),
        "parity": {
            "tokens": int(parity_ids.shape[1]),
            "chunk_size": chunk_size,
            "max_abs_logit_delta": logit_delta.max().item(),
            "mean_abs_logit_delta": logit_delta.mean().item(),
            "state_max_abs_deltas": state_deltas,
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--parity-tokens", type=int, default=24)
    parser.add_argument("--sha256", action="store_true")
    parser.add_argument("--json-output")
    args = parser.parse_args()

    report = audit_checkpoint(
        args.model_path,
        parity_tokens=args.parity_tokens,
        include_hash=args.sha256,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=True)
    print(rendered)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
