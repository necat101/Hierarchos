import os
import sys
import argparse
import json
import hashlib
import shutil
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm
import signal
import traceback

from hierarchos import (
    pick_device,
    set_threads,
    load_full_model_with_config,
    HierarchosCore,
    train,
    finetune,
    chat,
    is_directml_device,
    setup_msvc_environment,
    create_dataloader_for_jsonl,
    create_dataloader_for_hf_streaming,
    create_map_style_dataloader,
    HuggingFaceMapStyleDataset,
    OriginalJSONLDataset,
    process_text_sample,
    create_dataloader_for_chunked,
    create_dataloader_pt_chunked,
    create_dataloader_for_tokenized_cache
)


def load_hf_dataset(dataset_name, dataset_config=None, split="train", streaming=False):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Hugging Face dataset loading requires the 'datasets' package.") from exc
    kwargs = {"split": split, "streaming": bool(streaming)}
    if dataset_config:
        return load_dataset(dataset_name, dataset_config, **kwargs)
    return load_dataset(dataset_name, **kwargs)


def _parse_hf_split_count(base_count, selector):
    selector = selector.strip()
    if not selector:
        return base_count
    if ":" not in selector:
        return base_count

    start_text, end_text = selector.split(":", 1)
    start_text = start_text.strip()
    end_text = end_text.strip()
    percent_mode = start_text.endswith("%") or end_text.endswith("%")

    if percent_mode:
        start_pct = float(start_text[:-1]) if start_text.endswith("%") and start_text[:-1] else 0.0
        end_pct = float(end_text[:-1]) if end_text.endswith("%") and end_text[:-1] else 100.0
        start = int(base_count * (start_pct / 100.0))
        end = int(base_count * (end_pct / 100.0))
    else:
        start = int(start_text) if start_text else 0
        end = int(end_text) if end_text else base_count

    start = max(0, min(base_count, start))
    end = max(start, min(base_count, end))
    return end - start


def estimate_hf_dataset_size(dataset_name, dataset_config=None, split="train"):
    from datasets import load_dataset_builder

    builder = load_dataset_builder(dataset_name, dataset_config)
    info = builder.info
    split = split or "train"
    if split in info.splits:
        return info.splits[split].num_examples

    if "[" in split and split.endswith("]"):
        base_split, selector = split.split("[", 1)
        selector = selector[:-1]
        if base_split in info.splits:
            return _parse_hf_split_count(info.splits[base_split].num_examples, selector)

    return None


def _steps_from_samples(sample_count, batch_size):
    sample_count = max(0, int(sample_count or 0))
    batch_size = max(1, int(batch_size or 1))
    return max(1, (sample_count + batch_size - 1) // batch_size)


def _is_jsonl_path(path):
    return str(path).lower().endswith((".jsonl", ".ndjson"))


def _jsonl_source_files(path):
    if _is_jsonl_path(path):
        return [path]
    if not path or not os.path.isdir(path):
        return []
    paths = []
    for _root, _dirs, files in os.walk(path):
        for filename in files:
            lower = filename.lower()
            if lower != "manifest.jsonl" and lower.endswith((".jsonl", ".ndjson")):
                paths.append(os.path.join(_root, filename))
    paths.sort()
    return paths


def _is_jsonl_source(path):
    return bool(_jsonl_source_files(path))


def count_jsonl_source_rows(path):
    count = 0
    for jsonl_path in _jsonl_source_files(path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            count += sum(1 for line in f if line.strip())
    return count


def _local_text_columns(args):
    return args.text_column, args.prompt_column, args.completion_column


def _hf_shard_cache_key(args, num_shards):
    payload = {
        "format": "tokenized-pt-shards-v2",
        "formatter": "alpaca-input-section-v1",
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "num_shards": int(num_shards),
        "chunks_per_file": int(getattr(args, "hf_cache_chunks_per_file", 2048) or 2048),
        "tokenizer": (
            getattr(args, "tokenizer_path", None)
            or getattr(args, "model_path", None)
            or "openai-community/gpt2"
        ),
        "max_length": int(getattr(args, "max_length", 0) or 0),
        "kayla": bool(getattr(args, "kayla", False)),
        "alpaca": bool(getattr(args, "alpaca", False)),
        "text_column": getattr(args, "text_column", None),
        "prompt_column": getattr(args, "prompt_column", None),
        "completion_column": getattr(args, "completion_column", None),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _estimate_hf_total_for_progress(args):
    total_samples = getattr(args, "dataset_size", 0) or None
    if total_samples is not None:
        return total_samples
    try:
        return estimate_hf_dataset_size(
            args.hf_dataset,
            args.hf_dataset_config,
            args.hf_dataset_split,
        )
    except Exception:
        return None


def _processed_sample_to_cached_item(processed):
    input_ids = processed["input_ids"].detach().cpu().to(dtype=torch.int32)
    labels = processed["labels"].detach().cpu().to(dtype=torch.int32)
    length = int(processed.get("_length", input_ids.numel()))
    length = max(1, min(int(input_ids.numel()), length))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones(input_ids.numel(), dtype=torch.uint8),
        "_length": length,
    }


def materialize_hf_dataset_pt_cache(args, tokenizer, num_shards):
    num_shards = max(1, int(num_shards or 1))
    cache_root = getattr(args, "hf_shard_cache_dir", None) or os.path.join(os.getcwd(), ".hierarchos_hf_shards")
    shard_key = _hf_shard_cache_key(args, num_shards)
    shard_dir = os.path.join(cache_root, shard_key)
    success_path = os.path.join(shard_dir, "_SUCCESS")
    manifest_path = os.path.join(shard_dir, "manifest.jsonl")

    if (
        not getattr(args, "refresh_hf_shards", False)
        and os.path.exists(success_path)
        and os.path.exists(manifest_path)
    ):
        print(f"INFO: Reusing cached HF tokenized PT shards from {shard_dir}")
        return shard_dir

    tmp_dir = shard_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_hf_shards", False) and os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print(
        "INFO: Materializing HF dataset into "
        f"{num_shards} tokenized PT shard stream(s) for multi-worker loading..."
    )
    hf_dataset = load_hf_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
        streaming=True,
    )

    chunks_per_file = max(1, int(getattr(args, "hf_cache_chunks_per_file", 2048) or 2048))
    shard_buffers = [[] for _ in range(num_shards)]
    shard_counts = [0] * int(num_shards)
    shard_parts = [0] * int(num_shards)
    accepted = 0
    skipped = 0

    def flush_shard(manifest_file, shard_idx):
        buffer = shard_buffers[shard_idx]
        if not buffer:
            return
        filename = f"shard_{shard_idx:05d}_part_{shard_parts[shard_idx]:05d}.pt"
        path = os.path.join(tmp_dir, filename)
        torch.save(buffer, path)
        for item_idx, item in enumerate(buffer):
            length = int(item.get("_length", item["input_ids"].numel()))
            manifest_file.write(json.dumps({
                "file_path": filename,
                "index_in_file": item_idx,
                "length": length,
                "valid_length": length,
            }) + "\n")
        shard_buffers[shard_idx] = []
        shard_parts[shard_idx] += 1

    with open(os.path.join(tmp_dir, "manifest.jsonl"), "w", encoding="utf-8") as manifest_file:
        total_samples = _estimate_hf_total_for_progress(args)
        skipped = 0
        for sample_idx, sample in enumerate(tqdm(
            hf_dataset,
            desc="Tokenizing/sharding HF dataset",
            unit="sample",
            total=total_samples,
        )):
            processed = process_text_sample(
                tokenizer,
                sample,
                getattr(args, "max_length", 1024),
                getattr(args, "kayla", False),
                text_column=getattr(args, "text_column", None),
                prompt_column=getattr(args, "prompt_column", None),
                completion_column=getattr(args, "completion_column", None),
                alpaca_mode=getattr(args, "alpaca", False),
            )
            if processed is None:
                skipped += 1
                continue
            item = _processed_sample_to_cached_item(processed)
            item["_source_idx"] = sample_idx
            shard_idx = accepted % num_shards
            shard_buffers[shard_idx].append(item)
            shard_counts[shard_idx] += 1
            accepted += 1
            if len(shard_buffers[shard_idx]) >= chunks_per_file:
                flush_shard(manifest_file, shard_idx)

        for shard_idx in range(num_shards):
            flush_shard(manifest_file, shard_idx)

    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({
            "format": "tokenized-pt-shards-v2",
            "formatter": "alpaca-input-section-v1",
            "num_shards": num_shards,
            "counts": shard_counts,
            "parts": shard_parts,
            "skipped": skipped,
            "max_length": getattr(args, "max_length", 1024),
            "chunks_per_file": chunks_per_file,
        }, f, indent=2)

    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.replace(tmp_dir, shard_dir)
    print(
        f"INFO: HF tokenized PT shards ready in {shard_dir} "
        f"({sum(shard_counts)} samples, skipped {skipped})."
    )
    return shard_dir


def materialize_hf_dataset_shards(args, tokenizer, num_shards):
    return materialize_hf_dataset_pt_cache(args, tokenizer, num_shards)


def _hf_token_cache_key(args):
    payload = {
        "format": "map-token-bin-v2",
        "formatter": "alpaca-input-section-v1",
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "tokenizer": (
            getattr(args, "tokenizer_path", None)
            or getattr(args, "model_path", None)
            or "openai-community/gpt2"
        ),
        "max_length": int(getattr(args, "max_length", 0) or 0),
        "kayla": bool(getattr(args, "kayla", False)),
        "alpaca": bool(getattr(args, "alpaca", False)),
        "text_column": getattr(args, "text_column", None),
        "prompt_column": getattr(args, "prompt_column", None),
        "completion_column": getattr(args, "completion_column", None),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def materialize_hf_token_cache(args, tokenizer):
    cache_root = (
        getattr(args, "hf_token_cache_dir", None)
        or getattr(args, "hf_shard_cache_dir", None)
        or os.path.join(os.getcwd(), ".hierarchos_token_cache")
    )
    cache_key = _hf_token_cache_key(args)
    cache_dir = os.path.join(cache_root, cache_key)
    success_path = os.path.join(cache_dir, "_SUCCESS")
    index_path = os.path.join(cache_dir, "index.pt")
    data_path = os.path.join(cache_dir, "tokens.bin")

    if (
        not getattr(args, "refresh_hf_token_cache", False)
        and os.path.exists(success_path)
        and os.path.exists(index_path)
        and os.path.exists(data_path)
    ):
        print(f"INFO: Reusing HF random-access token cache from {cache_dir}")
        return cache_dir

    tmp_dir = cache_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_hf_token_cache", False) and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print("INFO: Building HF random-access token cache...")
    hf_dataset = load_hf_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
        streaming=False,
    )
    try:
        total_samples = len(hf_dataset)
    except Exception:
        total_samples = _estimate_hf_total_for_progress(args)

    offsets = []
    lengths = []
    skipped = 0
    total_bytes = 0
    tmp_data_path = os.path.join(tmp_dir, "tokens.bin")

    def iter_hf_samples():
        nonlocal skipped
        if total_samples is None:
            yield from enumerate(hf_dataset)
            return
        for sample_idx in range(total_samples):
            try:
                yield sample_idx, hf_dataset[sample_idx]
            except Exception:
                skipped += 1

    with open(tmp_data_path, "wb") as data_file:
        for sample_idx, sample in tqdm(
            iter_hf_samples(),
            desc="Tokenizing HF cache",
            unit="sample",
            total=total_samples,
        ):
            try:
                processed = process_text_sample(
                    tokenizer,
                    sample,
                    getattr(args, "max_length", 1024),
                    getattr(args, "kayla", False),
                    text_column=getattr(args, "text_column", None),
                    prompt_column=getattr(args, "prompt_column", None),
                    completion_column=getattr(args, "completion_column", None),
                    alpaca_mode=getattr(args, "alpaca", False),
                )
            except Exception:
                processed = None
            if processed is None:
                skipped += 1
                continue

            input_ids = processed["input_ids"].detach().cpu().to(dtype=torch.int32).contiguous()
            labels = processed["labels"].detach().cpu().to(dtype=torch.int32).contiguous()
            length = min(int(input_ids.numel()), int(labels.numel()))
            if length <= 0:
                skipped += 1
                continue
            if input_ids.numel() != length:
                input_ids = input_ids[:length].contiguous()
            if labels.numel() != length:
                labels = labels[:length].contiguous()

            offsets.append(total_bytes)
            lengths.append(length)
            input_bytes = input_ids.numpy().tobytes()
            label_bytes = labels.numpy().tobytes()
            data_file.write(input_bytes)
            data_file.write(label_bytes)
            total_bytes += len(input_bytes) + len(label_bytes)

    if not lengths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("HF token cache build produced no usable samples.")

    torch.save({
        "format": "map-token-bin-v2",
        "formatter": "alpaca-input-section-v1",
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
    }, os.path.join(tmp_dir, "index.pt"))
    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({
            "format": "map-token-bin-v2",
            "formatter": "alpaca-input-section-v1",
            "samples": len(lengths),
            "skipped": skipped,
            "bytes": total_bytes,
            "max_length": getattr(args, "max_length", 1024),
        }, f, indent=2)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.replace(tmp_dir, cache_dir)
    print(
        f"INFO: HF random-access token cache ready in {cache_dir} "
        f"({len(lengths)} samples, skipped {skipped}, {total_bytes / (1024 ** 3):.2f} GiB)."
    )
    return cache_dir


def create_hf_training_dataloader(args, tokenizer, device):
    def create_indexed_hf_dataloader(reason=None):
        if reason:
            print(f"INFO: {reason}")
        hf_dataset = load_hf_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
        dataset = HuggingFaceMapStyleDataset(
            hf_dataset,
            tokenizer,
            args.max_length,
            args.kayla,
            args.text_column,
            args.prompt_column,
            args.completion_column,
            getattr(args, "alpaca", False),
        )
        return create_map_style_dataloader(
            dataset,
            args.batch_size,
            tokenizer.pad_token_id,
            args.num_workers,
            use_length_bucketing=getattr(args, "length_bucketing", True),
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    if getattr(args, "hf_token_cache", False):
        cache_dir = materialize_hf_token_cache(args, tokenizer)
        print("INFO: Using HF random-access token cache with length bucketing.")
        return create_dataloader_for_tokenized_cache(
            cache_dir,
            args.max_length,
            args.batch_size,
            tokenizer.pad_token_id,
            num_workers=args.num_workers,
            use_length_bucketing=getattr(args, "length_bucketing", True),
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    if getattr(args, "streaming_datasets", True):
        try:
            hf_dataset = load_hf_dataset(
                args.hf_dataset,
                args.hf_dataset_config,
                split=args.hf_dataset_split,
                streaming=True,
            )
            hf_num_workers = args.num_workers
            hf_num_shards = getattr(hf_dataset, "num_shards", None)
            try:
                hf_num_shards = int(hf_num_shards) if hf_num_shards is not None else None
            except (TypeError, ValueError):
                hf_num_shards = None
            if hf_num_shards is not None and hf_num_shards > 0 and hf_num_workers > hf_num_shards:
                if getattr(args, "hf_auto_shard", False) and hf_num_workers > 1:
                    shard_dir = materialize_hf_dataset_shards(args, tokenizer, hf_num_workers)
                    print(
                        "INFO: Using local tokenized PT shard cache for HF dataset "
                        f"({hf_num_workers} worker shard(s))."
                    )
                    return create_dataloader_pt_chunked(
                        shard_dir,
                        args.max_length,
                        args.batch_size,
                        hf_num_workers,
                        use_length_bucketing=getattr(args, "length_bucketing", True),
                        bucket_size=getattr(args, "length_bucket_size", None),
                        device=device,
                        prefetch_factor=args.prefetch_factor,
                        cache_size=max(getattr(args, "pt_cache_size", 2), hf_num_workers),
                    )
                return create_indexed_hf_dataloader(
                    "HF streaming exposes "
                    f"{hf_num_shards} physical shard(s), but --num_workers={hf_num_workers}. "
                    "Using indexed HF loading with length bucketing so PyTorch can use all requested workers."
                )
            dataloader = create_dataloader_for_hf_streaming(
                hf_dataset,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=hf_num_workers,
                kayla_mode=args.kayla,
                text_column=args.text_column,
                prompt_column=args.prompt_column,
                completion_column=args.completion_column,
                alpaca_mode=getattr(args, "alpaca", False),
                use_length_bucketing=getattr(args, "length_bucketing", True),
                bucket_size=getattr(args, "length_bucket_size", None),
                shuffle_buffer_size=getattr(args, "hf_streaming_shuffle_buffer", 10000),
                device=device,
                prefetch_factor=args.prefetch_factor,
            )
            print("INFO: Hugging Face dataset streaming enabled.")
            return dataloader
        except Exception as exc:
            print(f"WARNING: HF streaming loader failed ({exc}); falling back to map-style loading.")

    return create_indexed_hf_dataloader()


def create_local_training_dataloader(args, tokenizer, device):
    text_column, prompt_column, completion_column = _local_text_columns(args)
    if getattr(args, "streaming_datasets", True) and _is_jsonl_source(args.train):
        print("INFO: Local JSONL dataset streaming enabled.")
        return create_dataloader_for_jsonl(
            args.train,
            tokenizer,
            args.max_length,
            args.batch_size,
            tokenizer.pad_token_id,
            num_workers=args.num_workers,
            kayla_mode=args.kayla,
            text_column=text_column,
            prompt_column=prompt_column,
            completion_column=completion_column,
            alpaca_mode=getattr(args, "alpaca", False),
            use_length_bucketing=getattr(args, "length_bucketing", True),
            bucket_size=getattr(args, "length_bucket_size", None),
            device=device,
            prefetch_factor=args.prefetch_factor,
        )

    dataset = OriginalJSONLDataset(
        args.train,
        tokenizer,
        args.max_length,
        args.kayla,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=getattr(args, "alpaca", False),
    )
    return create_map_style_dataloader(
        dataset,
        args.batch_size,
        tokenizer.pad_token_id,
        args.num_workers,
        use_length_bucketing=getattr(args, "length_bucketing", True),
        bucket_size=getattr(args, "length_bucket_size", None),
        device=device,
        prefetch_factor=args.prefetch_factor,
    )


def configure_tokenizer_length(tokenizer, max_length):
    """Keep tokenizer warnings aligned with Hierarchos' runtime context length."""
    if tokenizer is None or not max_length:
        return

    try:
        tokenizer.model_max_length = int(max_length)
        if hasattr(tokenizer, "init_kwargs"):
            tokenizer.init_kwargs["model_max_length"] = int(max_length)
    except Exception:
        pass


def resolve_num_workers(requested_workers, device, batch_size):
    requested_workers = int(requested_workers)
    if requested_workers >= 0:
        return requested_workers

    cpu_count = os.cpu_count() or 1
    if getattr(device, "type", "cpu") == "cuda":
        # Four workers is the usual sweet spot for pre-tokenized CUDA training:
        # enough to keep pinned batches ready without spending too much CPU/RAM
        # bandwidth on loader overhead. Users can still override for heavy HF
        # tokenization or slow remote storage.
        by_cpu = max(1, cpu_count // 4)
        if cpu_count >= 8:
            by_cpu = max(4, by_cpu)
        by_batch = max(1, int(batch_size or 1) * 2)
        return min(4, by_cpu, by_batch)

    # CPU and DirectML training usually want the cores for model math. Users can
    # still override this for tokenizer-heavy Hugging Face datasets.
    return 0


def main():
    parser = argparse.ArgumentParser(description="hierarchos: A Hybrid Memory-Reasoning Architecture")
    parser.add_argument("mode", type=str, choices=["train", "finetune", "chat", "quantize", "merge-lora", "ckpt-2-inf"], help="Operation mode.")

    # --- Data and Path Arguments ---
    path_group = parser.add_argument_group('Paths and Data')
    path_group.add_argument("--train", type=str, nargs='?', default=None, const=True, help="Path to local training data.")
    path_group.add_argument("--hf_dataset", type=str, default=None, help="Name or path to a Hugging Face dataset.")
    path_group.add_argument("--hf_dataset_config", type=str, default=None, help="Optional configuration name for the HF dataset.")
    path_group.add_argument("--hf_dataset_split", type=str, default="train", help="Dataset split to use.")
    path_group.add_argument("--text_column", type=str, default=None, help="Column name for text completion data.")
    path_group.add_argument("--prompt_column", type=str, default=None, help="Column name for prompt/instruction.")
    path_group.add_argument("--completion_column", type=str, default=None, help="Column name for completion/response.")
    
    path_group.add_argument("--model-path", type=str, default=None, help="Path to the model directory.")
    path_group.add_argument("--out-dir", type=str, default="./hierarchos_model", help="Directory to save the new model/adapter.")
    path_group.add_argument("--tokenizer-path", type=str, default=None, help="Path or HF name of the tokenizer.") 
    path_group.add_argument("--resume-from-ckpt", type=str, default=None, help="Path to a training checkpoint .pt file.")
    path_group.add_argument("--shadow-model-path", type=str, default=None, help="Path to full-precision model for quantized learning.")

    data_fmt_group = parser.add_mutually_exclusive_group()
    data_fmt_group.add_argument("--pre_chunked_dataset", action="store_true")
    data_fmt_group.add_argument("--pre_pt_dataset", action="store_true")

    # --- Architecture Arguments ---
    arch_group = parser.add_argument_group('Architecture')
    arch_group.add_argument("--context_dim", type=int, default=768) 
    arch_group.add_argument("--persistent_dim", type=int, default=128)
    arch_group.add_argument("--ltm_slots", type=int, default=1024)
    arch_group.add_argument("--ltm_key_dim", type=int, default=128)
    arch_group.add_argument("--ltm_val_dim", type=int, default=128)
    arch_group.add_argument("--h_hidden", type=int, default=None)
    arch_group.add_argument("--l_hidden", type=int, default=None)
    arch_group.add_argument("--h_stride", type=int, default=4)
    arch_group.add_argument("--max_h_steps", type=int, default=5)
    arch_group.add_argument("--max_l_steps", type=int, default=5)
    arch_group.add_argument("--ltm_topk", type=int, default=4)
    arch_group.add_argument("--max_length", type=int, default=1024)
    arch_group.add_argument("--auto-max-length", action="store_true")

    # --- Training Arguments ---
    train_group = parser.add_argument_group('Training')
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch_size", type=int, default=4)
    train_group.add_argument("--accumulation-steps", "--accumulation_steps", dest="accumulation_steps", type=int, default=1)
    train_group.add_argument("--starting-lr", type=float, default=1e-4)
    train_group.add_argument("--min-lr", type=float, default=1e-6)
    train_group.add_argument("--disable-lr-schedule", action="store_true")
    train_group.add_argument("--ltm_lr", type=float, default=1e-3)
    train_group.add_argument("--ltm-score-grad-scale", "--ltm_score_grad_scale", type=float, default=1.0, help="Straight-through gradient scale for LTM query/key addressing. Set 0 to keep retrieval addressing frozen.")
    format_group = train_group.add_mutually_exclusive_group()
    format_group.add_argument("--kayla", action="store_true")
    format_group.add_argument(
        "--alpaca",
        action="store_true",
        help="Use Alpaca instruction/input/output formatting and default columns instruction/output.",
    )
    train_group.add_argument("--lora_r", type=int, default=8)
    train_group.add_argument("--lora_alpha", type=int, default=16)
    train_group.add_argument("--grad-clip", type=float, default=1.0)
    train_group.add_argument("--ponder-loss-weight", type=float, default=0.01)
    train_group.add_argument("--commitment-loss-weight", type=float, default=0.5)
    train_group.add_argument("--commitment-threshold", type=float, default=0.05)
    train_group.add_argument("--l_conv_atol", "--l-conv-atol", type=float, default=1e-4, help="Converge tolerance for WorkerLoop. Default: 1e-4.")
    train_group.add_argument("--detach_every_n_steps", "--detach-every-n-steps", type=int, default=32, help="RWKV state detachment frequency. Default: 32.")
    train_group.add_argument("--h_halt_thresh", "--h-halt-thresh", type=float, default=0.9, help="H-RNN halt probability threshold. Default: 0.9.")
    train_group.add_argument("--encourage-thinking", action="store_true", help="Invert ponder loss to REWARD thinking (for recovery training).")
    train_group.add_argument("--adaptive-ponder", action="store_true", help="Scale ponder target based on CE loss (more thinking for harder content).")
    train_group.add_argument("--ponder-target-scale", type=float, default=0.5, help="Scaling factor for adaptive ponder target. Default: 0.5.")
    train_group.add_argument("--reset-halt-bias", type=float, default=None, metavar="BIAS", help="SURGICAL FIX: Reset h_halt_proj.bias to this value on load (e.g., -2.0 for ~12%% halt prob).")
    train_group.add_argument("--override-scheduling", action="store_true")
    train_group.add_argument("--persist-state", action="store_true", default=False, help="Persist RNN/LTM states between batches. Default: False.")
    train_group.add_argument("--no-persist-state", dest="persist_state", action="store_false", help="Disable state persistence between chunks.")
    train_group.add_argument("--training-chunk-size", "--training_chunk_size", type=int, default=128, help="TBPTT chunk size (Default: 128).")
    train_group.add_argument("--cuda-loss-chunk-rows", "--cuda_loss_chunk_rows", type=int, default=0, help="Rows per lm_head loss chunk on CUDA (0 = auto).")
    train_group.add_argument("--no-cuda-chunked-lm-loss", dest="cuda_chunked_lm_loss", action="store_false", help="Disable CUDA chunked LM loss and return full logits during training.")
    train_group.add_argument("--cpu-loss-chunk-rows", "--cpu_loss_chunk_rows", type=int, default=0, help="Rows per lm_head loss chunk on CPU (0 = all supervised rows).")
    train_group.add_argument("--no-cpu-chunked-lm-loss", dest="cpu_chunked_lm_loss", action="store_false", help="Disable CPU supervised-row LM loss and return full logits during training.")
    train_group.add_argument("--no-ltm-cpu-gather-retrieval", dest="ltm_cpu_gather_retrieval", action="store_false", help="Use the old dense one-hot CPU retrieval path.")
    train_group.add_argument("--no-ltm-cpu-sparse-update", dest="ltm_cpu_sparse_update", action="store_false", help="Use the old dense one-hot CPU LTM update path.")
    train_group.set_defaults(
        cuda_chunked_lm_loss=True,
        cpu_chunked_lm_loss=True,
        ltm_cpu_gather_retrieval=True,
        ltm_cpu_sparse_update=True,
    )
    train_group.add_argument("--debug-numerics", action="store_true", help="Enable per-token NaN/Inf debug checks. Slower on CUDA.")
    train_group.add_argument("--save-steps", type=int, default=0, help="Save a checkpoint/adapter every N steps during training/finetuning (0 to disable).")
    train_group.add_argument("--num_workers", type=int, default=-1, help="DataLoader workers (-1 = auto; CUDA uses prefetched workers, CPU/DML uses 0).")
    train_group.add_argument("--prefetch-factor", "--prefetch_factor", dest="prefetch_factor", type=int, default=None, help="Batches prefetched per DataLoader worker (auto keeps total queued batches tied to worker count).")
    train_group.add_argument("--pt-cache-size", "--pt_cache_size", dest="pt_cache_size", type=int, default=2, help="Number of .pt chunk files to keep hot per worker for --pre_pt_dataset.")
    train_group.add_argument("--no-length-bucketing", dest="length_bucketing", action="store_false", help="Disable length-aware batching for training datasets.")
    train_group.add_argument("--length-bucket-size", "--length_bucket_size", dest="length_bucket_size", type=int, default=None, help="Samples per length bucket/window. Larger lowers padding; smaller lowers streaming latency.")
    train_group.add_argument("--no-streaming-datasets", dest="streaming_datasets", action="store_false", help="Disable streaming for raw JSONL/Hugging Face datasets and use map-style loading.")
    train_group.add_argument("--hf-streaming-shuffle-buffer", "--hf_streaming_shuffle_buffer", dest="hf_streaming_shuffle_buffer", type=int, default=10000, help="Buffered shuffle size for Hugging Face streaming datasets.")
    train_group.add_argument("--hf-auto-shard", dest="hf_auto_shard", action="store_true", help="Opt into local tokenized shard caching for single-shard HF streaming datasets.")
    train_group.add_argument("--no-hf-auto-shard", dest="hf_auto_shard", action="store_false", help="Use indexed HF loading instead of local tokenized shard caching for single-shard HF datasets.")
    train_group.set_defaults(hf_auto_shard=False)
    train_group.add_argument("--hf-shard-cache-dir", "--hf_shard_cache_dir", dest="hf_shard_cache_dir", type=str, default=None, help="Directory for cached local HF tokenized shard files.")
    train_group.add_argument("--refresh-hf-shards", "--refresh_hf_shards", dest="refresh_hf_shards", action="store_true", help="Rebuild cached local HF tokenized shards before training.")
    train_group.add_argument("--hf-cache-chunks-per-file", "--hf_cache_chunks_per_file", dest="hf_cache_chunks_per_file", type=int, default=2048, help="Tokenized samples per cached HF .pt shard chunk.")
    train_group.add_argument("--hf-token-cache", "--hf_token_cache", dest="hf_token_cache", action="store_true", help="Build/reuse a random-access binary token cache for HF datasets before training.")
    train_group.add_argument("--hf-token-cache-dir", "--hf_token_cache_dir", dest="hf_token_cache_dir", type=str, default=None, help="Directory for random-access HF token caches.")
    train_group.add_argument("--refresh-hf-token-cache", "--refresh_hf_token_cache", dest="refresh_hf_token_cache", action="store_true", help="Rebuild the random-access HF token cache before training.")
    train_group.add_argument("--amp", action="store_true", help="Enable mixed precision (auto-enabled on CUDA).")
    train_group.add_argument("--no-amp", dest="amp", action="store_false", help="Explicitly disable mixed precision.")
    train_group.add_argument("--dataset-size", type=int, default=None, help="Force a specific dataset size (total samples) to calculate steps for the LR scheduler.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (auto-enabled on CUDA).")
    parser.add_argument("--force-compile", action="store_true")

    # --- Evaluation Arguments (lm-evaluation-harness) ---
    eval_group = parser.add_argument_group('Evaluation')
    eval_group.add_argument("--eval-tasks", type=str, nargs='+', default=None,
        help="Benchmark tasks to run during training (e.g., 'hellaswag arc_easy'). Disabled by default. Requires: pip install lm-eval")
    eval_group.add_argument("--eval-every-epoch", type=int, default=1,
        help="Run evaluation every N epochs (default: 1).")
    eval_group.add_argument("--eval-batch-size", type=int, default=1,
        help="Batch size for evaluation (default: 1).")
    eval_group.add_argument("--eval-limit", type=int, default=None,
        help="Limit samples per task for fast evaluation runs (e.g., 10 for quick tests).")
    eval_group.add_argument("--eval-steps", type=int, default=None,
        help="Run evaluation every N training steps (for quick testing). Triggers periodically.")

    # --- Inference & Sampling ---
    infer_group = parser.add_argument_group('Inference')
    infer_group.add_argument("--temperature", type=float, default=1.0)
    infer_group.add_argument("--top-k", type=int, default=40)
    infer_group.add_argument("--top-p", type=float, default=0.9)
    infer_group.add_argument("--repetition-penalty", type=float, default=1.2, help="Penalty for repeating tokens (1.0=none, >1.0=discourage). Default: 1.2.")
    infer_group.add_argument("--max-new-tokens", type=int, default=512)
    infer_group.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", "dml"])
    infer_group.add_argument("--threads", type=int, default=max(1, os.cpu_count() // 2))
    
    # --- Chat-specific (Online Learning) ---
    chat_group = parser.add_argument_group('Chat Online Learning')
    chat_group.add_argument("--enable-quantized-learning", action="store_true", help="Enable LTM updates for quantized models.")
    chat_group.add_argument("--ltm-lora-path", type=str, default=None, help="Path to save/load LTM updates as delta file.")
    chat_group.add_argument("--static-ltm-lr", action="store_true", default=True, help="Use a fixed LR for LTM updates (default).")
    chat_group.add_argument("--dynamic-ltm-lr", dest="static_ltm_lr", action="store_false", help="Enable cosine annealing for LTM updates.")
    chat_group.add_argument("--ltm-schedule-steps", type=int, default=100, help="Cosine cycle steps for LTM learning.")
    chat_group.add_argument("--ltm-schedule-min-lr", type=float, default=1e-5, help="Min LR for LTM cosine annealing.")
    chat_group.add_argument("--finetune-unlock-percent", type=float, default=None, help="Target %% of params to train (overrides lora_r).")
    chat_group.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing.")
    chat_group.add_argument("--passive-learning", action="store_true", default=True, help="Enable passive LTM learning from user prompts/observed context (ON by default).")
    chat_group.add_argument("--no-passive-learning", dest="passive_learning", action="store_false", help="Disable passive LTM learning.")
    chat_group.add_argument("--passive-response-learning", action="store_true", default=False, help="Also allow passive learning from self-generated responses after confidence/quality gates. Default: OFF.")
    chat_group.add_argument("--passive-lr", type=float, default=5e-6, help="Learning rate for passive LTM updates (default: 5e-6, very conservative).")
    chat_group.add_argument("--surprise-threshold", type=float, default=1.0, help="Passive learning only writes when loss <= this threshold (default: 1.0; lower is stricter).")
    
    # --- Utility Arguments ---
    util_group = parser.add_argument_group('Utilities')
    util_group.add_argument("--ckpt-input", type=str, default=None, help="Input checkpoint path for ckpt-2-inf mode.")
    util_group.add_argument("--inf-output", type=str, default=None, help="Output inference model path for ckpt-2-inf mode.")
    util_group.add_argument("--ckpt-tok-path", type=str, default=None, help="HuggingFace tokenizer name/path to embed in the inference model (e.g., 'gpt2', 'openai-community/gpt2').")
    
    args = parser.parse_args()

    # Parity: hidden size auto-sync
    if args.mode == 'train' and not args.resume_from_ckpt:
        if args.h_hidden is None: args.h_hidden = args.context_dim
        if args.l_hidden is None: args.l_hidden = args.context_dim

    set_threads(args.threads)
    if args.compile or args.force_compile:
        setup_msvc_environment()
    pt_device = pick_device(args)
    args.num_workers = resolve_num_workers(args.num_workers, pt_device, args.batch_size)
    print(f"INFO: DataLoader workers set to {args.num_workers} for device={pt_device}.")
    
    # Tokenizer Loading
    tokenizer = None
    tokenizer_source = args.tokenizer_path or args.model_path or "openai-community/gpt2"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token: tokenizer.pad_token = tokenizer.eos_token
            else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        configure_tokenizer_length(tokenizer, args.max_length)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer: {e}"); sys.exit(1)

    # --- Auto Max Length Scan ---
    if args.auto_max_length and args.mode in ['train', 'finetune']:
        max_found = 0
        if args.hf_dataset:
            print(f"Scanning HF dataset: {args.hf_dataset}...")
            temp_ds = load_hf_dataset(args.hf_dataset, args.hf_dataset_config, split=args.hf_dataset_split)
            for sample in tqdm(temp_ds, desc="Scanning HF"):
                processed = process_text_sample(tokenizer, sample, 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column, args.alpaca)
                if processed: max_found = max(max_found, len(processed['input_ids']))
        elif args.train and isinstance(args.train, str):
            jsonl_files = _jsonl_source_files(args.train)
            if jsonl_files:
                print(f"Scanning local JSONL source: {args.train}...")
                for jsonl_path in jsonl_files:
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Scanning {os.path.basename(jsonl_path)}"):
                            try:
                                processed = process_text_sample(tokenizer, json.loads(line), 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column, args.alpaca)
                                if processed: max_found = max(max_found, len(processed['input_ids']))
                            except: continue
            else:
                print(f"Scanning local file: {args.train}...")
                with open(args.train, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list): data = [data]
                        for obj in tqdm(data, desc="Scanning JSON"):
                            processed = process_text_sample(tokenizer, obj, 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column, args.alpaca)
                            if processed: max_found = max(max_found, len(processed['input_ids']))
                    except:
                        f.seek(0)
                        for line in tqdm(f, desc="Scanning JSONL"):
                            try:
                                processed = process_text_sample(tokenizer, json.loads(line), 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column, args.alpaca)
                                if processed: max_found = max(max_found, len(processed['input_ids']))
                            except: continue
        if max_found > 0:
            args.max_length = (max_found + 16 + 7) & -8 # Align to 8
            print(f"Auto-scan found max length {max_found}. Setting max_length={args.max_length}")

    # Execution
    if args.mode == "train":
        dataloader = None
        if args.hf_dataset:
            dataloader = create_hf_training_dataloader(args, tokenizer, pt_device)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor, cache_size=args.pt_cache_size)
        elif args.train and isinstance(args.train, str):
            dataloader = create_local_training_dataloader(args, tokenizer, pt_device)
        
        if dataloader is None:
            print("ERROR: No dataset provided for training. Use --train or --hf_dataset."); sys.exit(1)

        if args.dataset_size:
            dataloader_len = _steps_from_samples(args.dataset_size, args.batch_size)
            print(f"INFO: Manual session override: {args.dataset_size} samples -> {dataloader_len} steps per epoch.")
        else:
            try: 
                dataloader_len = len(dataloader)
            except: 
                # Heuristic for IterableDataset (JSONL) or HF Metadata
                if args.hf_dataset:
                    print(f"INFO: Estimating size for HF dataset: {args.hf_dataset}...")
                    try:
                        count = estimate_hf_dataset_size(args.hf_dataset, args.hf_dataset_config, args.hf_dataset_split)
                        if count is not None:
                            dataloader_len = _steps_from_samples(count, args.batch_size)
                            print(f"INFO: Automatic HF session detection found {count} samples -> {dataloader_len} steps per epoch.")
                        else:
                            print(f"ERROR: Could not find split '{args.hf_dataset_split}' in HF dataset info. Please use --dataset-size."); sys.exit(1)
                    except Exception as e:
                        print(f"ERROR: Failed to query HF metadata: {e}. Please specify --dataset-size manually."); sys.exit(1)
                elif hasattr(args, 'train') and isinstance(args.train, str) and _is_jsonl_source(args.train):
                    print(f"INFO: Estimating dataset size from JSONL source: {args.train}...")
                    try:
                        count = count_jsonl_source_rows(args.train)
                        dataloader_len = _steps_from_samples(count, args.batch_size)
                        print(f"INFO: Automatic session detection found {count} samples -> {dataloader_len} steps per epoch.")
                    except Exception as e:
                        print(f"ERROR: Could not count JSONL rows in {args.train}: {e}. Please specify --dataset-size manually."); sys.exit(1)
                else:
                    print("ERROR: Dataset size could not be determined automatically. Please use --dataset-size to specify the number of steps per epoch."); sys.exit(1)

        train(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "chat":
        chat(args, pt_device, tokenizer)
    elif args.mode == "finetune":
        # Build dataloader for finetune
        dataloader = None
        if args.hf_dataset:
            dataloader = create_hf_training_dataloader(args, tokenizer, pt_device)
        elif args.pre_chunked_dataset:
            dataloader = create_dataloader_for_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor)
        elif args.pre_pt_dataset:
            dataloader = create_dataloader_pt_chunked(args.train, args.max_length, args.batch_size, args.num_workers, use_length_bucketing=args.length_bucketing, bucket_size=args.length_bucket_size, device=pt_device, prefetch_factor=args.prefetch_factor, cache_size=args.pt_cache_size)
        elif args.train and isinstance(args.train, str):
            dataloader = create_local_training_dataloader(args, tokenizer, pt_device)
        
        if dataloader is None:
            print("ERROR: No dataset provided for finetuning. Use --train or --hf_dataset."); sys.exit(1)

        if args.dataset_size:
            dataloader_len = _steps_from_samples(args.dataset_size, args.batch_size)
            print(f"INFO: Manual session override: {args.dataset_size} samples -> {dataloader_len} steps per epoch.")
        else:
            try: 
                dataloader_len = len(dataloader)
            except: 
                # Heuristic for IterableDataset (JSONL) or HF Metadata
                if args.hf_dataset:
                    print(f"INFO: Estimating size for HF dataset: {args.hf_dataset}...")
                    try:
                        count = estimate_hf_dataset_size(args.hf_dataset, args.hf_dataset_config, args.hf_dataset_split)
                        if count is not None:
                            dataloader_len = _steps_from_samples(count, args.batch_size)
                            print(f"INFO: Automatic HF session detection found {count} samples -> {dataloader_len} steps per epoch.")
                        else:
                            print(f"ERROR: Could not find split '{args.hf_dataset_split}' in HF dataset info. Please use --dataset-size."); sys.exit(1)
                    except Exception as e:
                        print(f"ERROR: Failed to query HF metadata: {e}. Please specify --dataset-size manually."); sys.exit(1)
                elif hasattr(args, 'train') and isinstance(args.train, str) and _is_jsonl_source(args.train):
                    print(f"INFO: Estimating dataset size from JSONL source: {args.train}...")
                    try:
                        count = count_jsonl_source_rows(args.train)
                        dataloader_len = _steps_from_samples(count, args.batch_size)
                        print(f"INFO: Automatic session detection found {count} samples -> {dataloader_len} steps per epoch.")
                    except Exception as e:
                        print(f"ERROR: Could not count JSONL rows in {args.train}: {e}. Please specify --dataset-size manually."); sys.exit(1)
                else:
                    print("ERROR: Dataset size could not be determined automatically. Please use --dataset-size to specify the number of steps per epoch."); sys.exit(1)

        finetune(args, pt_device, tokenizer, dataloader, dataloader_len)
    elif args.mode == "ckpt-2-inf":
        # Convert checkpoint to inference model (HuggingFace-style directory)
        ckpt_path = args.ckpt_input or args.resume_from_ckpt or args.model_path
        if not ckpt_path:
            print("ERROR: No checkpoint specified. Use --ckpt-input, --resume-from-ckpt, or --model-path.")
            sys.exit(1)
        
        # Determine output directory (strip .pt extension if provided)
        if args.inf_output:
            output_dir = args.inf_output.replace('.pt', '')
        else:
            base_dir = os.path.dirname(ckpt_path)
            output_dir = os.path.join(base_dir, "hierarchos_final")
        
        print(f"Converting checkpoint to inference model...")
        print(f"  Input:  {ckpt_path}")
        print(f"  Output: {output_dir}/")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            sys.exit(1)
        
        # Extract and clean state dict
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {}
        for k, v in state_dict.items():
            # Remove _orig_mod. prefix from compiled models
            clean_key = k.replace('_orig_mod.', '')
            clean_state_dict[clean_key] = v
        
        # Extract config
        config = checkpoint.get('config', {})
        completed_epoch = checkpoint.get('completed_epoch', checkpoint.get('epoch', 'unknown'))
        
        # Handle tokenizer
        tokenizer_name = args.ckpt_tok_path or config.get('tokenizer_name', 'openai-community/gpt2')
        print(f"  Tokenizer: {tokenizer_name}")
        
        # Load and verify tokenizer
        try:
            inf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            vocab_size = len(inf_tokenizer)
            print(f"  Vocab size: {vocab_size}")
            
            # Verify vocab size matches model
            model_vocab = config.get('vocab_size')
            if model_vocab and model_vocab != vocab_size:
                print(f"  WARNING: Model vocab_size ({model_vocab}) != tokenizer vocab_size ({vocab_size})")
                print(f"           Make sure you're using the same tokenizer as training!")
        except Exception as e:
            print(f"ERROR: Failed to load tokenizer '{tokenizer_name}': {e}")
            sys.exit(1)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer files
        print(f"  Saving tokenizer files...")
        inf_tokenizer.save_pretrained(output_dir)
        
        # Create inference-ready checkpoint
        model_path = os.path.join(output_dir, "model.pt")
        inference_checkpoint = {
            'model_state_dict': clean_state_dict,
            'config': config,
            'completed_epoch': completed_epoch,
            'training_complete': True,
            'converted_from': os.path.basename(ckpt_path),
            'tokenizer_name': tokenizer_name,
        }
        torch.save(inference_checkpoint, model_path)
        
        # Save config as JSON for easy inspection
        config_path = os.path.join(output_dir, "hierarchos_config.json")
        import json as json_module
        config_to_save = dict(config)
        config_to_save['completed_epoch'] = completed_epoch
        config_to_save['tokenizer_name'] = tokenizer_name
        config_to_save['converted_from'] = os.path.basename(ckpt_path)
        with open(config_path, 'w') as f:
            json_module.dump(config_to_save, f, indent=2, default=str)
        
        # Report
        input_size = os.path.getsize(ckpt_path)
        output_size = os.path.getsize(model_path)
        reduction = (1 - output_size / input_size) * 100
        
        # Count all files in output dir
        total_output_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
        
        print(f"\n" + "="*60)
        print(f"CONVERSION COMPLETE!")
        print(f"="*60)
        print(f"Input checkpoint: {input_size / 1e6:.2f} MB")
        print(f"Output directory: {output_dir}/")
        print(f"  - model.pt:     {output_size / 1e6:.2f} MB  ({reduction:.1f}% reduction)")
        print(f"  - Total size:   {total_output_size / 1e6:.2f} MB")
        print(f"  - Epoch:        {completed_epoch}")
        print(f"  - Tokenizer:    {tokenizer_name}")
        print(f"\nTo use the model for inference:")
        print(f"  python hierarchos_cli.py chat --model-path \"{output_dir}\"")
        print(f"="*60)
    else:
        print(f"INFO: Mode '{args.mode}' is not yet fully integrated in the CLI.")

if __name__ == "__main__":
    main()
