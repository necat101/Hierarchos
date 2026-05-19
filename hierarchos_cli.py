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
    create_dataloader_pt_chunked
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
        "dataset": args.hf_dataset,
        "config": args.hf_dataset_config,
        "split": args.hf_dataset_split,
        "num_shards": int(num_shards),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def materialize_hf_dataset_shards(args, num_shards):
    num_shards = max(1, int(num_shards or 1))
    cache_root = getattr(args, "hf_shard_cache_dir", None) or os.path.join(os.getcwd(), ".hierarchos_hf_shards")
    shard_key = _hf_shard_cache_key(args, num_shards)
    shard_dir = os.path.join(cache_root, shard_key)
    success_path = os.path.join(shard_dir, "_SUCCESS")

    expected_shards = [
        os.path.join(shard_dir, f"shard_{idx:05d}.jsonl")
        for idx in range(num_shards)
    ]
    if (
        not getattr(args, "refresh_hf_shards", False)
        and os.path.exists(success_path)
        and all(os.path.exists(path) for path in expected_shards)
    ):
        print(f"INFO: Reusing cached HF JSONL shards from {shard_dir}")
        return shard_dir

    tmp_dir = shard_dir + ".tmp"
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if getattr(args, "refresh_hf_shards", False) and os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    print(
        "INFO: Materializing HF dataset into "
        f"{num_shards} local JSONL shard(s) for multi-worker streaming..."
    )
    hf_dataset = load_hf_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
        streaming=True,
    )

    shard_files = []
    shard_counts = [0] * int(num_shards)
    try:
        for shard_idx in range(num_shards):
            shard_path = os.path.join(tmp_dir, f"shard_{shard_idx:05d}.jsonl")
            shard_files.append(open(shard_path, "w", encoding="utf-8"))

        total_samples = getattr(args, "dataset_size", 0) or None
        if total_samples is None:
            try:
                total_samples = estimate_hf_dataset_size(
                    args.hf_dataset,
                    args.hf_dataset_config,
                    args.hf_dataset_split,
                )
            except Exception:
                total_samples = None
        for sample_idx, sample in enumerate(tqdm(
            hf_dataset,
            desc="Sharding HF dataset",
            unit="sample",
            total=total_samples,
        )):
            shard_idx = sample_idx % num_shards
            shard_files[shard_idx].write(json.dumps(sample, ensure_ascii=False, default=str) + "\n")
            shard_counts[shard_idx] += 1
    finally:
        for shard_file in shard_files:
            try:
                shard_file.close()
            except Exception:
                pass

    with open(os.path.join(tmp_dir, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({"num_shards": num_shards, "counts": shard_counts}, f, indent=2)

    if os.path.exists(shard_dir):
        shutil.rmtree(shard_dir)
    os.replace(tmp_dir, shard_dir)
    print(f"INFO: HF JSONL shards ready in {shard_dir} ({sum(shard_counts)} samples).")
    return shard_dir


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
                if getattr(args, "hf_auto_shard", True) and hf_num_workers > 1:
                    shard_dir = materialize_hf_dataset_shards(args, hf_num_workers)
                    print(
                        "INFO: Using local sharded JSONL streaming for HF dataset "
                        f"({hf_num_workers} worker shard(s))."
                    )
                    return create_dataloader_for_jsonl(
                        shard_dir,
                        tokenizer,
                        args.max_length,
                        args.batch_size,
                        tokenizer.pad_token_id,
                        num_workers=hf_num_workers,
                        kayla_mode=args.kayla,
                        text_column=args.text_column,
                        prompt_column=args.prompt_column,
                        completion_column=args.completion_column,
                        use_length_bucketing=getattr(args, "length_bucketing", True),
                        bucket_size=getattr(args, "length_bucket_size", None),
                        device=device,
                        prefetch_factor=args.prefetch_factor,
                    )
                return create_indexed_hf_dataloader(
                    "HF streaming exposes "
                    f"{hf_num_shards} physical shard(s), but --num_workers={hf_num_workers}. "
                    "Using indexed HF loading so PyTorch can use all requested workers. "
                    "For true multi-worker iterable streaming, publish/download the dataset as multiple data files."
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
    train_group.add_argument("--kayla", action="store_true")
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
    train_group.add_argument("--no-hf-auto-shard", dest="hf_auto_shard", action="store_false", help="Disable automatic local JSONL shard materialization for single-shard HF streaming datasets.")
    train_group.add_argument("--hf-shard-cache-dir", "--hf_shard_cache_dir", dest="hf_shard_cache_dir", type=str, default=None, help="Directory for cached local HF JSONL shards.")
    train_group.add_argument("--refresh-hf-shards", "--refresh_hf_shards", dest="refresh_hf_shards", action="store_true", help="Rebuild cached local HF JSONL shards before training.")
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
                processed = process_text_sample(tokenizer, sample, 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column)
                if processed: max_found = max(max_found, len(processed['input_ids']))
        elif args.train and isinstance(args.train, str):
            jsonl_files = _jsonl_source_files(args.train)
            if jsonl_files:
                print(f"Scanning local JSONL source: {args.train}...")
                for jsonl_path in jsonl_files:
                    with open(jsonl_path, 'r', encoding='utf-8') as f:
                        for line in tqdm(f, desc=f"Scanning {os.path.basename(jsonl_path)}"):
                            try:
                                processed = process_text_sample(tokenizer, json.loads(line), 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column)
                                if processed: max_found = max(max_found, len(processed['input_ids']))
                            except: continue
            else:
                print(f"Scanning local file: {args.train}...")
                with open(args.train, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if not isinstance(data, list): data = [data]
                        for obj in tqdm(data, desc="Scanning JSON"):
                            processed = process_text_sample(tokenizer, obj, 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column)
                            if processed: max_found = max(max_found, len(processed['input_ids']))
                    except:
                        f.seek(0)
                        for line in tqdm(f, desc="Scanning JSONL"):
                            try:
                                processed = process_text_sample(tokenizer, json.loads(line), 9999, args.kayla, args.text_column, args.prompt_column, args.completion_column)
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
