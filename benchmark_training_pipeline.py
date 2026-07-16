"""Synthetic loader and training-pipeline benchmark for Hierarchos.

The benchmark deliberately uses a tiny language model rather than the real
Hierarchos architecture.  That keeps the result focused on dataset iteration,
dynamic padding, host-to-device transfer, and optimizer-loop overhead.  It also
builds the same mmap-backed binary token cache used by production training and
checks that it is mathematically equivalent to an in-memory reference dataset.

Examples
--------
Portable smoke run::

    python benchmark_training_pipeline.py --samples 256 --batches 3 \
        --warmup-batches 1 --repeats 2

RTX 6000 Blackwell-oriented run::

    python benchmark_training_pipeline.py --device cuda --samples 32768 \
        --batch-size 64 --min-length 128 --max-length 1024 \
        --vocab-size 4096 --hidden-size 256 --workers 8 \
        --prefetch-factor 2 --warmup-batches 20 --batches 100 --repeats 5

Wall-clock speed is never treated as a regression assertion.  Only batch and
optimizer-update parity can fail the process.
"""

import argparse
import copy
import gc
import json
import os
import statistics
import tempfile
import time
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from hierarchos.training.datasets import (
    TokenizedBinaryDataset,
    create_dataloader_for_tokenized_cache,
    create_map_style_dataloader,
)


class DummyTokenDataset(Dataset):
    """Map-style reference dataset with the same records as the binary cache."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
        self.sample_lengths = [int(sample["_length"]) for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class LegacyBinaryTokenDataset(TokenizedBinaryDataset):
    """Pre-optimization mmap path: per-record fetch followed by generic collate."""

    __getitems__ = None


class TinyLanguageModel(nn.Module):
    """Small causal LM used only to exercise forward/backward/optimizer work."""

    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def loss(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(input_ids)
        logits = self.projection(hidden)
        return F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[:, 1:].to(dtype=torch.long).contiguous().view(-1),
            ignore_index=-100,
        )


class CyclingBatches:
    """Consume a finite DataLoader continuously without discarding iterators."""

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def next(self):
        while True:
            try:
                batch = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)
                continue
            if batch is not None:
                return batch


def generate_dummy_samples(
    sample_count: int,
    min_length: int,
    max_length: int,
    vocab_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Generate deterministic, deliberately high-variance token sequences."""
    generator = torch.Generator().manual_seed(int(seed))
    span = max_length - min_length + 1
    random_lengths = torch.randint(
        min_length,
        max_length + 1,
        (sample_count,),
        generator=generator,
    )
    samples = []
    for index in range(sample_count):
        # Force short/long modes into the random distribution so padding and
        # length-bucketing effects remain visible even for small smoke runs.
        if index % 4 == 0:
            length = min_length + (index % min(span, 8))
        elif index % 4 == 1:
            length = max_length - (index % min(span, 8))
        else:
            length = int(random_lengths[index])
        length = max(min_length, min(max_length, int(length)))
        input_ids = torch.randint(
            3,
            vocab_size,
            (length,),
            dtype=torch.long,
            generator=generator,
        )
        samples.append({
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "_length": length,
        })
    return samples


def write_binary_token_cache(cache_dir: str, samples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Write production-compatible compact schema-v6 token records."""
    os.makedirs(cache_dir, exist_ok=True)
    offsets = []
    lengths = []
    total_bytes = 0
    data_path = os.path.join(cache_dir, "tokens.bin")
    with open(data_path, "wb") as data_file:
        for sample in samples:
            input_ids = sample["input_ids"].to(dtype=torch.uint16).contiguous()
            labels_long = sample["labels"].to(dtype=torch.long).contiguous()
            labels = torch.where(
                labels_long == -100,
                torch.full_like(labels_long, 65535),
                labels_long,
            ).to(dtype=torch.uint16).contiguous()
            length = min(int(input_ids.numel()), int(labels.numel()))
            if length <= 0:
                continue
            input_ids = input_ids[:length].contiguous()
            labels = labels[:length].contiguous()
            offsets.append(total_bytes)
            lengths.append(length)
            input_bytes = input_ids.numpy().tobytes()
            label_bytes = labels.numpy().tobytes()
            data_file.write(input_bytes)
            data_file.write(label_bytes)
            total_bytes += len(input_bytes) + len(label_bytes)

    torch.save(
        {
            "format": "map-token-bin-v6-compact",
            "storage_schema_version": 6,
            "byte_order": "little",
            "token_dtype": "uint16",
            "label_dtype": "uint16",
            "rosa_dtype": None,
            "label_ignore_sentinel": 65535,
            "loss_weight_encoding": None,
            "loss_weight_palette": None,
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.int32),
            "has_loss_weights": False,
            "has_rosa_ids": False,
        },
        os.path.join(cache_dir, "index.pt"),
    )
    with open(os.path.join(cache_dir, "_SUCCESS"), "w", encoding="utf-8") as success_file:
        json.dump({"samples": len(lengths), "bytes": total_bytes}, success_file)
    return {"samples": len(lengths), "bytes": total_bytes}


def _make_dataloader(
    kind: str,
    samples: List[Dict[str, Any]],
    cache_dir: str,
    args,
    device: torch.device,
    workers: Optional[int] = None,
):
    workers = args.workers if workers is None else int(workers)
    bucket_size = int(args.bucket_size) if int(args.bucket_size or 0) > 0 else None
    prefetch_factor = args.prefetch_factor
    torch.manual_seed(args.seed)
    if kind == "in_memory_reference":
        return create_map_style_dataloader(
            DummyTokenDataset(samples),
            batch_size=args.batch_size,
            pad_token_id=0,
            num_workers=workers,
            shuffle=True,
            use_length_bucketing=not args.no_length_bucketing,
            bucket_size=bucket_size,
            device=device,
            prefetch_factor=prefetch_factor,
        )
    if kind == "binary_token_cache":
        return create_dataloader_for_tokenized_cache(
            cache_dir,
            max_length=args.max_length,
            batch_size=args.batch_size,
            pad_token_id=0,
            num_workers=workers,
            use_length_bucketing=not args.no_length_bucketing,
            bucket_size=bucket_size,
            device=device,
            prefetch_factor=prefetch_factor,
        )
    if kind == "legacy_binary_token_cache":
        return create_map_style_dataloader(
            LegacyBinaryTokenDataset(
                cache_dir,
                max_length=args.max_length,
                pad_token_id=0,
            ),
            batch_size=args.batch_size,
            pad_token_id=0,
            num_workers=workers,
            shuffle=True,
            use_length_bucketing=not args.no_length_bucketing,
            bucket_size=bucket_size,
            device=device,
            prefetch_factor=prefetch_factor,
        )
    raise ValueError("Unknown dataloader kind: %s" % kind)


def _shutdown_dataloader(dataloader):
    """Release persistent workers and mmap handles before temp-dir cleanup."""
    iterator = getattr(dataloader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:
            pass
    close = getattr(getattr(dataloader, "dataset", None), "close", None)
    if callable(close):
        close()
    try:
        dataloader._iterator = None
    except Exception:
        pass


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _collect_batches(dataloader, count: int) -> List[Dict[str, torch.Tensor]]:
    batches = []
    iterator = iter(dataloader)
    while len(batches) < count:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        if batch is not None:
            batches.append(_clone_batch(batch))
    return batches


def _move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    non_blocking = device.type == "cuda"
    return {
        key: value.to(device, non_blocking=non_blocking)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def _batch_counts(batch: Dict[str, torch.Tensor]) -> Tuple[int, int, int, int]:
    input_ids = batch["input_ids"]
    sample_count = int(input_ids.shape[0])
    padded_tokens = int(input_ids.numel())
    attention_mask = batch.get("attention_mask")
    valid_tokens = (
        int(attention_mask.sum().item())
        if isinstance(attention_mask, torch.Tensor)
        else padded_tokens
    )
    labels = batch["labels"]
    supervised_tokens = int(labels[:, 1:].ne(-100).sum().item()) if labels.shape[1] > 1 else 0
    return sample_count, valid_tokens, padded_tokens, supervised_tokens


def _compare_batches(
    reference_batches: List[Dict[str, torch.Tensor]],
    cached_batches: List[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    mismatch = None
    if len(reference_batches) != len(cached_batches):
        mismatch = "batch count differs (%d vs %d)" % (
            len(reference_batches),
            len(cached_batches),
        )
    else:
        for batch_index, (reference, cached) in enumerate(zip(reference_batches, cached_batches)):
            if set(reference) != set(cached):
                mismatch = "batch %d keys differ" % batch_index
                break
            for key in sorted(reference):
                left = reference[key]
                right = cached[key]
                if isinstance(left, torch.Tensor):
                    if left.shape != right.shape or not torch.equal(left.to(right.dtype), right):
                        mismatch = "batch %d tensor %s differs" % (batch_index, key)
                        break
                elif left != right:
                    mismatch = "batch %d value %s differs" % (batch_index, key)
                    break
            if mismatch:
                break
    return {
        "passed": mismatch is None,
        "batches_compared": min(len(reference_batches), len(cached_batches)),
        "mismatch": mismatch,
    }


def _one_optimizer_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(batch["input_ids"], batch["labels"])
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _compare_optimizer_updates(
    reference_batches: List[Dict[str, torch.Tensor]],
    cached_batches: List[Dict[str, torch.Tensor]],
    args,
    device: torch.device,
) -> Dict[str, Any]:
    torch.manual_seed(args.seed + 17)
    base_model = TinyLanguageModel(args.vocab_size, args.hidden_size)
    reference_model = copy.deepcopy(base_model).to(device)
    cached_model = copy.deepcopy(base_model).to(device)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=args.learning_rate)
    cached_optimizer = torch.optim.AdamW(cached_model.parameters(), lr=args.learning_rate)

    reference_losses = []
    cached_losses = []
    for reference_batch, cached_batch in zip(reference_batches, cached_batches):
        reference_losses.append(
            _one_optimizer_step(reference_model, reference_optimizer, _move_batch(reference_batch, device))
        )
        cached_losses.append(
            _one_optimizer_step(cached_model, cached_optimizer, _move_batch(cached_batch, device))
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    max_parameter_delta = 0.0
    for reference_parameter, cached_parameter in zip(
        reference_model.parameters(), cached_model.parameters()
    ):
        delta = float(
            (reference_parameter.detach() - cached_parameter.detach()).abs().max().cpu().item()
        )
        max_parameter_delta = max(max_parameter_delta, delta)

    loss_tolerance = 5e-5 if device.type == "cuda" else 1e-7
    parameter_tolerance = 5e-6 if device.type == "cuda" else 1e-8
    max_loss_delta = max(
        (abs(left - right) for left, right in zip(reference_losses, cached_losses)),
        default=0.0,
    )
    passed = (
        len(reference_losses) == len(cached_losses)
        and max_loss_delta <= loss_tolerance
        and max_parameter_delta <= parameter_tolerance
    )
    return {
        "passed": passed,
        "steps_compared": len(reference_losses),
        "reference_losses": reference_losses,
        "cached_losses": cached_losses,
        "max_loss_delta": max_loss_delta,
        "max_parameter_delta": max_parameter_delta,
        "loss_tolerance": loss_tolerance,
        "parameter_tolerance": parameter_tolerance,
    }


def run_parity_checks(samples, cache_dir, args, device) -> Dict[str, Any]:
    # Parity uses workers=0 so process startup and worker scheduling cannot hide
    # a real record/order mismatch.
    reference_loader = _make_dataloader(
        "in_memory_reference", samples, cache_dir, args, device, workers=0
    )
    cached_loader = _make_dataloader(
        "binary_token_cache", samples, cache_dir, args, device, workers=0
    )
    try:
        reference_batches = _collect_batches(reference_loader, args.parity_batches)
        cached_batches = _collect_batches(cached_loader, args.parity_batches)
        batch_parity = _compare_batches(reference_batches, cached_batches)
        if batch_parity["passed"]:
            optimizer_parity = _compare_optimizer_updates(
                reference_batches,
                cached_batches,
                args,
                device,
            )
        else:
            optimizer_parity = {
                "passed": False,
                "steps_compared": 0,
                "mismatch": "skipped because batch parity failed",
            }
        return {
            "passed": bool(batch_parity["passed"] and optimizer_parity["passed"]),
            "batch_parity": batch_parity,
            "optimizer_parity": optimizer_parity,
        }
    finally:
        _shutdown_dataloader(reference_loader)
        _shutdown_dataloader(cached_loader)


def _median_summary(runs: List[Dict[str, float]]) -> Dict[str, Any]:
    return {
        "median_seconds": statistics.median(run["seconds"] for run in runs),
        "median_samples_per_second": statistics.median(
            run["samples_per_second"] for run in runs
        ),
        "median_valid_tokens_per_second": statistics.median(
            run["valid_tokens_per_second"] for run in runs
        ),
        "median_padded_tokens_per_second": statistics.median(
            run["padded_tokens_per_second"] for run in runs
        ),
        "median_supervised_tokens_per_second": statistics.median(
            run["supervised_tokens_per_second"] for run in runs
        ),
        "median_token_efficiency": statistics.median(
            run["token_efficiency"] for run in runs
        ),
        "runs": runs,
    }


def _consume_loader_batches(cycler: CyclingBatches, batch_count: int) -> Dict[str, int]:
    totals = {"batches": 0, "samples": 0, "valid": 0, "padded": 0, "supervised": 0}
    for _ in range(batch_count):
        batch = cycler.next()
        samples, valid, padded, supervised = _batch_counts(batch)
        totals["batches"] += 1
        totals["samples"] += samples
        totals["valid"] += valid
        totals["padded"] += padded
        totals["supervised"] += supervised
    return totals


def _rate_run(seconds: float, totals: Dict[str, int]) -> Dict[str, float]:
    seconds = max(float(seconds), 1e-12)
    return {
        "seconds": seconds,
        "batches": int(totals["batches"]),
        "samples": int(totals["samples"]),
        "valid_tokens": int(totals["valid"]),
        "padded_tokens": int(totals["padded"]),
        "supervised_tokens": int(totals["supervised"]),
        "samples_per_second": totals["samples"] / seconds,
        "valid_tokens_per_second": totals["valid"] / seconds,
        "padded_tokens_per_second": totals["padded"] / seconds,
        "supervised_tokens_per_second": totals["supervised"] / seconds,
        "token_efficiency": totals["valid"] / float(max(1, totals["padded"])),
    }


def benchmark_loader(dataloader, args) -> Dict[str, Any]:
    cycler = CyclingBatches(dataloader)
    first_start = time.perf_counter()
    first_batch = cycler.next()
    first_batch_seconds = time.perf_counter() - first_start
    pinned_tensors = [
        value.is_pinned()
        for value in first_batch.values()
        if isinstance(value, torch.Tensor) and value.device.type == "cpu"
    ]
    if args.warmup_batches:
        _consume_loader_batches(cycler, args.warmup_batches)

    runs = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        totals = _consume_loader_batches(cycler, args.batches)
        elapsed = time.perf_counter() - start
        runs.append(_rate_run(elapsed, totals))
    summary = _median_summary(runs)
    summary.update({
        "first_batch_seconds": first_batch_seconds,
        "first_batch_all_cpu_tensors_pinned": bool(pinned_tensors and all(pinned_tensors)),
    })
    return summary


def _resolve_amp_dtype(name: str, device: torch.device):
    if name == "auto":
        name = "bfloat16" if device.type == "cuda" else "float32"
    if name == "float32":
        return name, None
    if name == "bfloat16":
        return name, torch.bfloat16
    if name == "float16":
        if device.type != "cuda":
            raise ValueError("float16 AMP training requires CUDA")
        return name, torch.float16
    raise ValueError("Unsupported AMP dtype: %s" % name)


def _make_grad_scaler(amp_name: str, device: torch.device):
    if amp_name != "float16" or device.type != "cuda":
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def _autocast_context(device: torch.device, amp_dtype):
    if amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=amp_dtype)


def _training_step(model, optimizer, scaler, batch, device, amp_dtype):
    batch = _move_batch(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(device, amp_dtype):
        loss = model.loss(batch["input_ids"], batch["labels"])
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()
    return loss


def _consume_training_batches(
    cycler: CyclingBatches,
    batch_count: int,
    model,
    optimizer,
    scaler,
    device,
    amp_dtype,
):
    totals = {"batches": 0, "samples": 0, "valid": 0, "padded": 0, "supervised": 0}
    last_loss = None
    for _ in range(batch_count):
        batch = cycler.next()
        samples, valid, padded, supervised = _batch_counts(batch)
        totals["batches"] += 1
        totals["samples"] += samples
        totals["valid"] += valid
        totals["padded"] += padded
        totals["supervised"] += supervised
        last_loss = _training_step(
            model,
            optimizer,
            scaler,
            batch,
            device,
            amp_dtype,
        )
    return totals, last_loss


def benchmark_training(dataloader, args, device) -> Dict[str, Any]:
    amp_name, amp_dtype = _resolve_amp_dtype(args.amp_dtype, device)
    torch.manual_seed(args.seed + 29)
    model = TinyLanguageModel(args.vocab_size, args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = _make_grad_scaler(amp_name, device)
    cycler = CyclingBatches(dataloader)
    model.train()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_loss = None
    if args.warmup_batches:
        _, last_loss = _consume_training_batches(
            cycler,
            args.warmup_batches,
            model,
            optimizer,
            scaler,
            device,
            amp_dtype,
        )

    runs = []
    for _ in range(args.repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        totals, last_loss = _consume_training_batches(
            cycler,
            args.batches,
            model,
            optimizer,
            scaler,
            device,
            amp_dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        runs.append(_rate_run(elapsed, totals))

    summary = _median_summary(runs)
    summary["amp_dtype"] = amp_name
    summary["final_loss"] = float(last_loss.detach().cpu().item()) if last_loss is not None else None
    summary["peak_cuda_memory_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return summary


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _cuda_environment(device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {
            "device_capability": None,
            "cuda_arch_list": [],
            "sm_120_supported": False,
        }
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    try:
        arch_list = list(torch.cuda.get_arch_list())
    except (AttributeError, RuntimeError):
        arch_list = []
    sm_120_supported = any(
        arch == "sm_120" or arch.startswith("sm_120")
        for arch in arch_list
    )
    return {
        "device_capability": list(capability),
        "cuda_arch_list": arch_list,
        "sm_120_supported": sm_120_supported,
    }


def _check_blackwell_support(device: torch.device, cuda_environment: Dict[str, Any], required: bool):
    if device.type != "cuda":
        if required:
            raise RuntimeError("--require-blackwell requires a visible CUDA device")
        return

    capability = tuple(cuda_environment.get("device_capability") or ())
    sm_120_supported = bool(cuda_environment.get("sm_120_supported"))
    if not sm_120_supported:
        print(
            "\n*** WARNING: This PyTorch build does not advertise sm_120 support. "
            "An RTX 6000 Blackwell GPU may fail at kernel launch or fall back to "
            "unsupported/slow paths. Install a PyTorch CUDA build that includes sm_120. ***\n"
        )
    if required and (capability != (12, 0) or not sm_120_supported):
        raise RuntimeError(
            "Blackwell validation failed: capability=%s, sm_120_supported=%s, cuda_arch_list=%s"
            % (
                capability or None,
                sm_120_supported,
                cuda_environment.get("cuda_arch_list", []),
            )
        )


def _speedup(results: Dict[str, Dict[str, Any]], section: str) -> Optional[float]:
    reference = results.get("in_memory_reference", {}).get(section)
    cached = results.get("binary_token_cache", {}).get(section)
    if not reference or not cached:
        return None
    denominator = float(reference["median_valid_tokens_per_second"])
    if denominator <= 0:
        return None
    return float(cached["median_valid_tokens_per_second"]) / denominator


def _pipeline_ratio(results, section: str, numerator: str, denominator: str):
    numerator_result = results.get(numerator, {}).get(section)
    denominator_result = results.get(denominator, {}).get(section)
    if not numerator_result or not denominator_result:
        return None
    denominator_rate = float(denominator_result["median_valid_tokens_per_second"])
    if denominator_rate <= 0:
        return None
    return float(numerator_result["median_valid_tokens_per_second"]) / denominator_rate


def _print_rate(label: str, result: Dict[str, Any]):
    first_batch = result.get("first_batch_seconds")
    startup_text = "first=%7.3fs  " % float(first_batch) if first_batch is not None else ""
    print(
        "  %-22s %smedian=%10.0f valid tok/s  "
        "samples/s=%9.1f  efficiency=%6.2f%%"
        % (
            label,
            startup_text,
            result["median_valid_tokens_per_second"],
            result["median_samples_per_second"],
            result["median_token_efficiency"] * 100.0,
        )
    )


def _print_results(report: Dict[str, Any]):
    config = report["config"]
    print(
        "Environment: torch=%s device=%s CUDA=%s workers=%d"
        % (
            report["environment"]["torch_version"],
            report["environment"]["device"],
            report["environment"]["cuda_version"],
            config["workers"],
        )
    )
    if report["environment"]["device"].startswith("cuda"):
        print(
            "CUDA architecture: capability=%s compiled=%s"
            % (
                report["environment"]["device_capability"],
                ",".join(report["environment"]["cuda_arch_list"]) or "none advertised",
            )
        )
    parity = report.get("parity")
    if parity is not None:
        optimizer = parity["optimizer_parity"]
        print(
            "Parity: %s (%d batches, max loss delta %.3g, max parameter delta %.3g)"
            % (
                "PASS" if parity["passed"] else "FAIL",
                parity["batch_parity"]["batches_compared"],
                optimizer.get("max_loss_delta", float("nan")),
                optimizer.get("max_parameter_delta", float("nan")),
            )
        )
    if any("loader" in value for value in report["pipelines"].values()):
        print("Loader-only steady state:")
        for name, value in report["pipelines"].items():
            if "loader" in value:
                _print_rate(name, value["loader"])
        speedup = report["comparisons"].get("binary_cache_loader_speedup")
        if speedup is not None:
            print("  Binary-cache/reference loader ratio: %.3fx" % speedup)
        fused_speedup = report["comparisons"].get("fused_vs_legacy_loader_speedup")
        if fused_speedup is not None:
            print("  Fused/legacy binary loader speedup: %.3fx" % fused_speedup)
    if any("training" in value for value in report["pipelines"].values()):
        print("Tiny-LM forward+backward+AdamW (loader included):")
        for name, value in report["pipelines"].items():
            if "training" in value:
                _print_rate(name, value["training"])
        speedup = report["comparisons"].get("binary_cache_training_speedup")
        if speedup is not None:
            print("  Binary-cache/reference training ratio: %.3fx" % speedup)
        fused_speedup = report["comparisons"].get("fused_vs_legacy_training_speedup")
        if fused_speedup is not None:
            print("  Fused/legacy binary training speedup: %.3fx" % fused_speedup)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark Hierarchos' synthetic data/training pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-length", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--bucket-size", type=int, default=0, help="0 uses the loader default")
    parser.add_argument("--no-length-bucketing", action="store_true")
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--batches", type=int, default=20, help="Timed batches per repeat")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--parity-batches", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--require-blackwell",
        action="store_true",
        help="Fail unless the selected GPU is capability 12.0 and this PyTorch build advertises sm_120",
    )
    parser.add_argument("--skip-loader", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--json-out", default=None)
    return parser


def _validate_args(args):
    for name in (
        "samples",
        "batch_size",
        "min_length",
        "max_length",
        "vocab_size",
        "hidden_size",
        "batches",
        "repeats",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError("--%s must be positive" % name.replace("_", "-"))
    if args.min_length < 2:
        raise ValueError("--min-length must be at least 2 for next-token loss")
    if args.max_length < args.min_length:
        raise ValueError("--max-length must be >= --min-length")
    if args.vocab_size <= 3:
        raise ValueError("--vocab-size must be greater than 3")
    if args.workers < 0:
        raise ValueError("--workers cannot be negative")
    if args.prefetch_factor is not None and args.prefetch_factor <= 0:
        raise ValueError("--prefetch-factor must be positive")
    if args.warmup_batches < 0 or args.parity_batches < 0:
        raise ValueError("warmup/parity batch counts cannot be negative")
    if args.parity_batches == 0:
        args.skip_parity = True


def main():
    args = build_parser().parse_args()
    _validate_args(args)
    device = _resolve_device(args.device)
    cuda_environment = _cuda_environment(device)
    _check_blackwell_support(device, cuda_environment, args.require_blackwell)
    torch.manual_seed(args.seed)

    setup_start = time.perf_counter()
    samples = generate_dummy_samples(
        args.samples,
        args.min_length,
        args.max_length,
        args.vocab_size,
        args.seed,
    )
    generation_seconds = time.perf_counter() - setup_start

    report = {
        "environment": {
            "torch_version": torch.__version__,
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "device_capability": cuda_environment["device_capability"],
            "cuda_arch_list": cuda_environment["cuda_arch_list"],
            "sm_120_supported": cuda_environment["sm_120_supported"],
        },
        "config": vars(args).copy(),
        "setup": {"dummy_generation_seconds": generation_seconds},
        "pipelines": {},
        "comparisons": {},
    }

    parity_failed = False
    with tempfile.TemporaryDirectory(prefix="hierarchos_pipeline_bench_") as tmp_dir:
        cache_dir = os.path.join(tmp_dir, "binary_cache")
        cache_start = time.perf_counter()
        cache_stats = write_binary_token_cache(cache_dir, samples)
        report["setup"].update({
            "cache_build_seconds": time.perf_counter() - cache_start,
            "cache_samples": cache_stats["samples"],
            "cache_bytes": cache_stats["bytes"],
        })

        if not args.skip_parity:
            report["parity"] = run_parity_checks(samples, cache_dir, args, device)
            parity_failed = not report["parity"]["passed"]

        for kind in (
            "in_memory_reference",
            "legacy_binary_token_cache",
            "binary_token_cache",
        ):
            pipeline = {}
            if not args.skip_loader:
                dataloader = _make_dataloader(kind, samples, cache_dir, args, device)
                try:
                    pipeline["loader"] = benchmark_loader(dataloader, args)
                finally:
                    _shutdown_dataloader(dataloader)
                    del dataloader
                    gc.collect()
            if not args.skip_training:
                dataloader = _make_dataloader(kind, samples, cache_dir, args, device)
                try:
                    pipeline["training"] = benchmark_training(dataloader, args, device)
                finally:
                    _shutdown_dataloader(dataloader)
                    del dataloader
                    gc.collect()
            report["pipelines"][kind] = pipeline

    report["comparisons"]["binary_cache_loader_speedup"] = _speedup(
        report["pipelines"], "loader"
    )
    report["comparisons"]["binary_cache_training_speedup"] = _speedup(
        report["pipelines"], "training"
    )
    report["comparisons"]["fused_vs_legacy_loader_speedup"] = _pipeline_ratio(
        report["pipelines"],
        "loader",
        "binary_token_cache",
        "legacy_binary_token_cache",
    )
    report["comparisons"]["fused_vs_legacy_training_speedup"] = _pipeline_ratio(
        report["pipelines"],
        "training",
        "binary_token_cache",
        "legacy_binary_token_cache",
    )
    _print_results(report)

    if args.json_out:
        output_path = os.path.abspath(args.json_out)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2)
        print("JSON report: %s" % output_path)

    if parity_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
