"""Cheap response-boundary rescue fine-tuning for Hierarchos checkpoints.

This script keeps an existing checkpoint, masks prompt tokens, freezes most of
the model, and trains a small response-facing subset of parameters. It is meant
for salvage passes when a model learned token statistics but free-runs poorly at
the `### Response:` boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hierarchos.inference.chat_state import clear_ltm_working_memory
from hierarchos.training.datasets import _format_alpaca_prompt
from hierarchos.utils.checkpoint import save_checkpoint_safely, sanitize_model_state_dict
from hierarchos.utils.checkpoint import load_full_model_with_config


DEFAULT_PROBES = [
    "Hello",
    "what is 4 + 4",
    "what is 8 + 8",
    "I was thinking about confidence versus arrogance. How does that apply to learning a hard skill?",
    "Write a simple Hello World program in Rust.",
]


TRAINABLE_PRESETS = {
    "head": (
        "tok_emb.weight",
        "out_norm.",
    ),
    "lite": (
        "tok_emb.weight",
        "out_norm.",
        "l_to_out.",
        "context_drift_proj.",
        "l_feedback_proj.",
        "l_input_proj.",
        "l_rnn.output.",
        "l_rnn.value_cm.",
    ),
    "worker": (
        "tok_emb.weight",
        "out_norm.",
        "l_to_out.",
        "context_drift_proj.",
        "l_feedback_proj.",
        "l_input_proj.",
        "l_rnn.",
    ),
    "all": ("",),
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            instruction = str(row.get("instruction") or row.get("Instruction") or "").strip()
            output = str(row.get("output") or row.get("Output") or "").strip()
            if instruction and output:
                rows.append(row)
    if not rows:
        raise ValueError(f"No usable instruction/output rows found in {path}")
    return rows


def encode_row(
    tokenizer,
    row: dict,
    max_length: int,
    train_prompt_tokens: bool = False,
    prompt_loss_weight: float = 0.0,
    response_loss_weight: float = 1.0,
    response_boundary_loss_weight: float = 5.0,
    response_boundary_tokens: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    instruction = str(row.get("instruction") or row.get("Instruction") or "")
    input_text = str(row.get("input") or row.get("Input") or "").strip()
    output = str(row.get("output") or row.get("Output") or "")
    response_boundary_tokens = max(0, int(response_boundary_tokens or 0))

    prompt = _format_alpaca_prompt(instruction, input_text)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(output, add_special_tokens=False)
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        response_ids = response_ids + [int(eos_id)]

    input_ids = prompt_ids + response_ids
    labels = (prompt_ids if train_prompt_tokens else ([-100] * len(prompt_ids))) + response_ids
    loss_weights = (
        [float(prompt_loss_weight if train_prompt_tokens else 0.0)] * len(prompt_ids)
        + [float(response_loss_weight)] * len(response_ids)
    )
    # Leave EOS at the normal response weight. Boosting EOS on short arithmetic
    # answers teaches the model to terminate immediately after `### Response:`.
    boundary = min(max(0, len(response_ids) - 1), response_boundary_tokens)
    for idx in range(boundary):
        loss_weights[len(prompt_ids) + idx] = float(response_loss_weight) * float(response_boundary_loss_weight)
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_trim = min(overflow, max(0, len(prompt_ids) - 1))
        if prompt_trim:
            input_ids = input_ids[prompt_trim:]
            labels = labels[prompt_trim:]
            loss_weights = loss_weights[prompt_trim:]
            overflow -= prompt_trim
        if overflow > 0:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            loss_weights = loss_weights[:max_length]

    return (
        torch.tensor(input_ids, dtype=torch.long).unsqueeze(0),
        torch.tensor(labels, dtype=torch.long).unsqueeze(0),
        torch.tensor(loss_weights, dtype=torch.float32).unsqueeze(0),
    )


def iter_batches(encoded_rows: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], batch_size: int, pad_token_id: int):
    for start in range(0, len(encoded_rows), batch_size):
        chunk = encoded_rows[start:start + batch_size]
        max_len = max(row[0].shape[1] for row in chunk)
        input_batch = torch.full((len(chunk), max_len), int(pad_token_id), dtype=torch.long)
        label_batch = torch.full((len(chunk), max_len), -100, dtype=torch.long)
        mask_batch = torch.zeros((len(chunk), max_len), dtype=torch.long)
        weight_batch = torch.zeros((len(chunk), max_len), dtype=torch.float32)
        for idx, (input_ids, labels, loss_weights) in enumerate(chunk):
            length = input_ids.shape[1]
            input_batch[idx, :length] = input_ids[0]
            label_batch[idx, :length] = labels[0]
            mask_batch[idx, :length] = 1
            weight_batch[idx, :length] = loss_weights[0]
        yield input_batch, label_batch, mask_batch, weight_batch


def select_trainable(model, preset: str, extra_patterns: Iterable[str] = ()) -> tuple[int, int]:
    patterns = tuple(TRAINABLE_PRESETS[preset]) + tuple(extra_patterns)
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        total += param.numel()
        enabled = any(name.startswith(pattern) for pattern in patterns)
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
    if trainable == 0:
        raise RuntimeError(f"Preset {preset!r} selected no trainable parameters.")
    return total, trainable


def format_prompt(text: str) -> str:
    return _format_alpaca_prompt(text, "")


@torch.no_grad()
def probe_model(model, tokenizer, device, prompts: Iterable[str], max_tokens: int = 48) -> None:
    model.eval()
    for prompt_text in prompts:
        prompt = format_prompt(prompt_text)
        ids = torch.tensor(tokenizer.encode(prompt, add_special_tokens=False), dtype=torch.long, device=device).unsqueeze(0)
        prefill_chunk_size = int(getattr(getattr(model, "config", None), "training_chunk_size", 0) or 0)
        prefill_step = prefill_chunk_size if prefill_chunk_size > 0 else ids.shape[1]
        prefill_step = max(1, int(prefill_step))
        outputs = None
        h_state = None
        l_state = None
        prev_context = None
        target_context = None
        drift_state = None
        ltm_state = None
        chunk_drift_state = None
        for start in range(0, ids.shape[1], prefill_step):
            end = min(start + prefill_step, ids.shape[1])
            outputs = model(
                ids[:, start:end],
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                drift_state=chunk_drift_state,
                ltm_memory_state=ltm_state,
                global_pos_offset=start,
            )
            h_state = outputs.get("h_state")
            l_state = outputs.get("l_state")
            prev_context = outputs.get("prev_context")
            target_context = outputs.get("target_context")
            drift_state = outputs.get("drift_state")
            ltm_state = outputs.get("ltm_memory_state")
            chunk_drift_state = drift_state
        logits = outputs["logits"][:, -1, :].float()
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(probs, 5)
        top = [(tokenizer.decode([int(i)]), round(float(v), 4)) for v, i in zip(top_vals[0], top_idx[0])]

        current = top_idx[:, :1]
        generated: list[int] = []
        total_tokens_seen = int(ids.shape[1])
        for _ in range(max_tokens):
            token_id = int(current.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            generation_drift_state = None
            if (
                prefill_chunk_size > 0
                and total_tokens_seen > 0
                and total_tokens_seen % prefill_chunk_size == 0
            ):
                generation_drift_state = drift_state
            outputs = model(
                current,
                h_state=h_state,
                l_state=l_state,
                prev_context=prev_context,
                target_context=target_context,
                # Epoch-13 TBPTT parity: drift is fed only at chunk boundaries.
                drift_state=generation_drift_state,
                ltm_memory_state=ltm_state,
                global_pos_offset=total_tokens_seen,
            )
            total_tokens_seen += 1
            h_state = outputs.get("h_state")
            l_state = outputs.get("l_state")
            prev_context = outputs.get("prev_context")
            target_context = outputs.get("target_context")
            drift_state = outputs.get("drift_state")
            ltm_state = outputs.get("ltm_memory_state")
            current = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
        print(f"\nPROMPT: {prompt_text}")
        print(f"TOP5: {top}")
        print(f"GREEDY: {tokenizer.decode(generated).strip()!r}")


def save_inference_dir(model, config, tokenizer, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": sanitize_model_state_dict(model),
        "config": dict(config),
        "completed_epoch": int(getattr(config, "completed_epoch", 0) or 0),
        "training_complete": True,
    }
    save_checkpoint_safely(checkpoint, str(out_dir / "hierarchos.pt"))
    tokenizer.save_pretrained(str(out_dir))
    with (out_dir / "hierarchos_config.json").open("w", encoding="utf-8") as handle:
        json.dump(dict(config), handle, indent=2, default=str)


def main() -> int:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Run a cheap response-only rescue fine-tune.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--train",
        action="append",
        default=None,
        help="JSONL rescue file. Can be passed multiple times. Default: tools/rescue_alpaca_seed.jsonl",
    )
    parser.add_argument("--out-dir", default=str(ROOT / "salvaged_kortexHOS"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--preset", choices=sorted(TRAINABLE_PRESETS), default="head")
    parser.add_argument("--include", action="append", default=[], help="Additional parameter-name prefix to train.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no explicit step limit.")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--train-prompt-tokens", action="store_true", help="Also train prompt tokens during rescue instead of response-only masking.")
    parser.add_argument("--prompt-loss-weight", type=float, default=0.15)
    parser.add_argument("--response-loss-weight", type=float, default=1.0)
    parser.add_argument("--response-boundary-loss-weight", type=float, default=5.0)
    parser.add_argument("--response-boundary-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--no-probe", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, config = load_full_model_with_config(args.model_path, device)
    clear_ltm_working_memory(model)
    model.suppress_hebbian = True
    if hasattr(config, "compile"):
        config.compile = False
    if hasattr(model, "config"):
        model.config.compile = False

    if not args.no_probe:
        print("\n=== Before Rescue Probe ===")
        probe_model(model, tokenizer, device, DEFAULT_PROBES)

    if args.probe_only:
        return 0

    train_paths = [Path(p) for p in (args.train or [str(ROOT / "tools" / "rescue_alpaca_seed.jsonl")])]
    rows = []
    for train_path in train_paths:
        rows.extend(load_jsonl(train_path))
    encoded = [
        encode_row(
            tokenizer,
            row,
            args.max_length,
            train_prompt_tokens=args.train_prompt_tokens,
            prompt_loss_weight=args.prompt_loss_weight,
            response_loss_weight=args.response_loss_weight,
            response_boundary_loss_weight=args.response_boundary_loss_weight,
            response_boundary_tokens=args.response_boundary_tokens,
        )
        for row in rows
    ]
    total, trainable = select_trainable(model, args.preset, args.include)
    print(
        f"\nTrainable preset: {args.preset} | "
        f"{trainable:,}/{total:,} parameters ({100.0 * trainable / max(1, total):.2f}%)"
    )
    print(f"Rows: {len(encoded)} | epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr:g}")
    print(
        "Loss weights: "
        f"prompt={args.prompt_loss_weight:g} ({'trained' if args.train_prompt_tokens else 'masked'}), "
        f"response={args.response_loss_weight:g}, "
        f"boundary={args.response_boundary_loss_weight:g}x first {args.response_boundary_tokens} token(s)"
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    global_step = 0
    model.train()
    for epoch in range(args.epochs):
        random.shuffle(encoded)
        running = []
        for input_ids, labels, attention_mask, loss_weights in iter_batches(encoded, args.batch_size, tokenizer.pad_token_id):
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            loss_weights = loss_weights.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_weights=loss_weights,
                return_topk_values=False,
                suppress_hebbian=True,
            )
            loss = outputs["loss"]
            if loss is None or not torch.isfinite(loss).all():
                print(f"Skipping non-finite loss at step {global_step + 1}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()

            global_step += 1
            running.append(float(loss.detach().cpu()))
            if global_step == 1 or global_step % 10 == 0:
                window = running[-10:]
                print(
                    f"epoch={epoch + 1} step={global_step} "
                    f"loss={float(loss):.4f} avg10={sum(window) / len(window):.4f}"
                )
            if args.max_steps and global_step >= args.max_steps:
                break
        if args.max_steps and global_step >= args.max_steps:
            break

    save_inference_dir(model, config, tokenizer, Path(args.out_dir))
    print(f"\nSaved salvaged inference model to: {Path(args.out_dir).resolve()}")

    if not args.no_probe:
        clear_ltm_working_memory(model)
        print("\n=== After Rescue Probe ===")
        probe_model(model, tokenizer, device, DEFAULT_PROBES)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
