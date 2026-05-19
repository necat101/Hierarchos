import argparse
import json
import os
import tempfile
import time

import torch

from hierarchos.training.datasets import (
    HuggingFaceMapStyleDataset,
    OriginalJSONLDataset,
    create_dataloader_for_hf_streaming,
    create_dataloader_for_jsonl,
    create_dataloader_for_tokenized_cache,
    create_dataloader_pt_chunked,
    create_map_style_dataloader,
    process_text_sample,
)


class BenchmarkTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        text = str(text)
        return [3 + (ord(ch) % 251) for ch in text]


class FakeHFDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, buffer_size=10000, seed=0):
        del buffer_size
        rows = list(self.rows)
        if rows:
            offset = int(seed) % len(rows)
            rows = rows[offset:] + rows[:offset]
        return FakeHFDataset(rows)

    def shard(self, num_shards, index, contiguous=False):
        del contiguous
        return FakeHFDataset(self.rows[index::num_shards])


def make_rows(samples):
    base = (
        "Hierarchos dataset streaming benchmark sample. "
        "This line is intentionally long enough to make tokenization visible. "
    )
    rows = []
    for idx in range(samples):
        repeat = 1 + (idx % 8)
        rows.append({
            "instruction": f"Summarize record {idx}: " + (base * repeat),
            "output": f"Record {idx} summary with repeat factor {repeat}.",
        })
    return rows


def write_benchmark_jsonl(path, samples):
    with open(path, "w", encoding="utf-8") as f:
        for row in make_rows(samples):
            f.write(json.dumps(row) + "\n")


def write_pt_cache(directory, rows, tokenizer, max_length, num_shards=4, chunks_per_file=256):
    os.makedirs(directory, exist_ok=True)
    buffers = [[] for _ in range(num_shards)]
    parts = [0 for _ in range(num_shards)]

    def flush(manifest, shard_idx):
        buffer = buffers[shard_idx]
        if not buffer:
            return
        filename = f"shard_{shard_idx:05d}_part_{parts[shard_idx]:05d}.pt"
        import torch
        torch.save(buffer, os.path.join(directory, filename))
        for item_idx, item in enumerate(buffer):
            length = int(item["_length"])
            manifest.write(json.dumps({
                "file_path": filename,
                "index_in_file": item_idx,
                "length": length,
                "valid_length": length,
            }) + "\n")
        buffers[shard_idx] = []
        parts[shard_idx] += 1

    accepted = 0
    with open(os.path.join(directory, "manifest.jsonl"), "w", encoding="utf-8") as manifest:
        for row in rows:
            processed = process_text_sample(tokenizer, row, max_length)
            if processed is None:
                continue
            input_ids = processed["input_ids"].detach().cpu()
            labels = processed["labels"].detach().cpu()
            item = {
                "input_ids": input_ids.to(dtype=torch.int32),
                "labels": labels.to(dtype=torch.int32),
                "attention_mask": torch.ones(input_ids.numel(), dtype=torch.uint8),
                "_length": int(processed["_length"]),
            }
            shard_idx = accepted % num_shards
            buffers[shard_idx].append(item)
            accepted += 1
            if len(buffers[shard_idx]) >= chunks_per_file:
                flush(manifest, shard_idx)
        for shard_idx in range(num_shards):
            flush(manifest, shard_idx)


def write_binary_token_cache(directory, rows, tokenizer, max_length):
    os.makedirs(directory, exist_ok=True)
    offsets = []
    lengths = []
    total_bytes = 0
    with open(os.path.join(directory, "tokens.bin"), "wb") as f:
        for row in rows:
            processed = process_text_sample(tokenizer, row, max_length)
            if processed is None:
                continue
            input_ids = processed["input_ids"].detach().cpu().to(dtype=torch.int32).contiguous()
            labels = processed["labels"].detach().cpu().to(dtype=torch.int32).contiguous()
            length = min(int(input_ids.numel()), int(labels.numel()))
            if length <= 0:
                continue
            offsets.append(total_bytes)
            lengths.append(length)
            input_bytes = input_ids[:length].numpy().tobytes()
            label_bytes = labels[:length].numpy().tobytes()
            f.write(input_bytes)
            f.write(label_bytes)
            total_bytes += len(input_bytes) + len(label_bytes)
    torch.save({
        "format": "map-token-bin-v1",
        "offsets": torch.tensor(offsets, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.int32),
    }, os.path.join(directory, "index.pt"))
    with open(os.path.join(directory, "_SUCCESS"), "w", encoding="utf-8") as f:
        json.dump({"samples": len(lengths), "bytes": total_bytes}, f)


def consume_batches(dataloader, batches):
    total_samples = 0
    total_tokens = 0
    total_padded_tokens = 0
    iterator = iter(dataloader)

    first_start = time.perf_counter()
    first_batch = next(iterator)
    first_batch_seconds = time.perf_counter() - first_start

    batch_count = 1
    total_samples += first_batch["input_ids"].shape[0]
    total_tokens += int(first_batch["attention_mask"].sum().item())
    total_padded_tokens += int(first_batch["input_ids"].numel())

    rest_start = time.perf_counter()
    while batch_count < batches:
        try:
            batch = next(iterator)
        except StopIteration:
            break
        batch_count += 1
        total_samples += batch["input_ids"].shape[0]
        total_tokens += int(batch["attention_mask"].sum().item())
        total_padded_tokens += int(batch["input_ids"].numel())
    rest_seconds = time.perf_counter() - rest_start

    return {
        "batches": batch_count,
        "samples": total_samples,
        "tokens": total_tokens,
        "padded_tokens": total_padded_tokens,
        "first_batch_seconds": first_batch_seconds,
        "rest_seconds": rest_seconds,
    }


def print_result(name, create_seconds, result):
    total_iter_seconds = result["first_batch_seconds"] + result["rest_seconds"]
    samples_per_second = result["samples"] / max(total_iter_seconds, 1e-9)
    tokens_per_second = result["tokens"] / max(total_iter_seconds, 1e-9)
    padding_waste = result["padded_tokens"] - result["tokens"]
    print(
        f"{name}: create={create_seconds:.3f}s, "
        f"first_batch={result['first_batch_seconds']:.3f}s, "
        f"{result['batches']} batches in {total_iter_seconds:.3f}s, "
        f"{samples_per_second:.1f} samples/s, {tokens_per_second:.0f} tokens/s, "
        f"padding_waste={padding_waste}"
    )


def run_case(path, tokenizer, args, name, factory):
    create_start = time.perf_counter()
    dataloader = factory()
    create_seconds = time.perf_counter() - create_start
    result = consume_batches(dataloader, args.batches)
    close_fn = getattr(getattr(dataloader, "dataset", None), "close", None)
    if callable(close_fn):
        close_fn()
    print_result(name, create_seconds, result)


def main():
    parser = argparse.ArgumentParser(description="Local benchmark for Hierarchos dataset loaders.")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=80)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--stream-workers", type=int, default=0)
    parser.add_argument("--stream-workers-extra", type=int, default=2)
    args = parser.parse_args()

    tokenizer = BenchmarkTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "benchmark.jsonl")
        write_benchmark_jsonl(path, args.samples)
        print(
            f"Benchmark dataset: {args.samples} JSONL samples, "
            f"batch_size={args.batch_size}, max_length={args.max_length}"
        )

        run_case(
            path,
            tokenizer,
            args,
            "map_style_jsonl",
            lambda: create_map_style_dataloader(
                OriginalJSONLDataset(path, tokenizer, args.max_length),
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=0,
                use_length_bucketing=True,
            ),
        )

        run_case(
            path,
            tokenizer,
            args,
            f"streaming_jsonl_bucketed_workers_{args.stream_workers}",
            lambda: create_dataloader_for_jsonl(
                path,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=args.stream_workers,
                use_length_bucketing=True,
            ),
        )

        run_case(
            path,
            tokenizer,
            args,
            f"streaming_jsonl_unbucketed_workers_{args.stream_workers}",
            lambda: create_dataloader_for_jsonl(
                path,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=args.stream_workers,
                use_length_bucketing=False,
            ),
        )

        if args.stream_workers_extra > 0:
            run_case(
                path,
                tokenizer,
                args,
                f"streaming_jsonl_bucketed_workers_{args.stream_workers_extra}",
                lambda: create_dataloader_for_jsonl(
                    path,
                    tokenizer,
                    args.max_length,
                    args.batch_size,
                    tokenizer.pad_token_id,
                    num_workers=args.stream_workers_extra,
                    use_length_bucketing=True,
                ),
            )

        pt_cache_dir = os.path.join(tmpdir, "pt_cache")
        write_pt_cache(
            pt_cache_dir,
            make_rows(args.samples),
            tokenizer,
            args.max_length,
            num_shards=max(1, args.stream_workers_extra),
        )
        run_case(
            path,
            tokenizer,
            args,
            "hf_auto_pt_cache_bucketed_workers_0",
            lambda: create_dataloader_pt_chunked(
                pt_cache_dir,
                args.max_length,
                args.batch_size,
                num_workers=0,
                use_length_bucketing=True,
                cache_size=max(2, args.stream_workers_extra),
            ),
        )
        if args.stream_workers_extra > 0:
            run_case(
                path,
                tokenizer,
                args,
                f"hf_auto_pt_cache_bucketed_workers_{args.stream_workers_extra}",
                lambda: create_dataloader_pt_chunked(
                    pt_cache_dir,
                    args.max_length,
                    args.batch_size,
                    num_workers=args.stream_workers_extra,
                    use_length_bucketing=True,
                    cache_size=max(2, args.stream_workers_extra),
                ),
            )

        token_cache_dir = os.path.join(tmpdir, "token_cache")
        write_binary_token_cache(
            token_cache_dir,
            make_rows(args.samples),
            tokenizer,
            args.max_length,
        )
        run_case(
            path,
            tokenizer,
            args,
            "hf_binary_token_cache_bucketed_workers_0",
            lambda: create_dataloader_for_tokenized_cache(
                token_cache_dir,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=0,
                use_length_bucketing=True,
            ),
        )
        if args.stream_workers_extra > 0:
            run_case(
                path,
                tokenizer,
                args,
                f"hf_binary_token_cache_bucketed_workers_{args.stream_workers_extra}",
                lambda: create_dataloader_for_tokenized_cache(
                    token_cache_dir,
                    args.max_length,
                    args.batch_size,
                    tokenizer.pad_token_id,
                    num_workers=args.stream_workers_extra,
                    use_length_bucketing=True,
                ),
            )

        hf_rows = FakeHFDataset(make_rows(args.samples))
        run_case(
            path,
            tokenizer,
            args,
            "hf_map_style",
            lambda: create_map_style_dataloader(
                HuggingFaceMapStyleDataset(
                    hf_rows,
                    tokenizer,
                    args.max_length,
                    prompt_column="instruction",
                    completion_column="output",
                ),
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=0,
                use_length_bucketing=True,
            ),
        )

        run_case(
            path,
            tokenizer,
            args,
            "hf_streaming_bucketed_workers_0",
            lambda: create_dataloader_for_hf_streaming(
                hf_rows,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=0,
                prompt_column="instruction",
                completion_column="output",
                use_length_bucketing=True,
                shuffle=False,
            ),
        )

        run_case(
            path,
            tokenizer,
            args,
            "hf_streaming_unbucketed_workers_0",
            lambda: create_dataloader_for_hf_streaming(
                hf_rows,
                tokenizer,
                args.max_length,
                args.batch_size,
                tokenizer.pad_token_id,
                num_workers=0,
                prompt_column="instruction",
                completion_column="output",
                use_length_bucketing=False,
                shuffle=False,
            ),
        )


if __name__ == "__main__":
    main()
