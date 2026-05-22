import json
import os
import types
import tempfile

import torch

from hierarchos.training.datasets import (
    HuggingFaceStreamingDataset,
    LengthGroupedBatchSampler,
    OriginalJSONLDataset,
    StreamingJSONLDataset,
    TokenizedBinaryDataset,
    _iter_hf_worker_samples,
    _iter_jsonl_lines_for_worker,
    _iter_jsonl_shards_for_worker,
    create_dataloader_for_chunked,
    create_dataloader_for_hf_streaming,
    create_dataloader_for_jsonl,
    create_dataloader_for_tokenized_cache,
    process_text_sample,
)
from hierarchos.training.trainer import pad_training_batch_to_multiple


class TinyTokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        text = str(text)
        if not text:
            return []
        return [3 + (ord(ch) % 97) for ch in text]


class RecordingTokenizer(TinyTokenizer):
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=True):
        self.calls.append((str(text), add_special_tokens))
        return super().encode(text, add_special_tokens=add_special_tokens)


class FakeHFStream:
    def __init__(self, rows):
        self.rows = list(rows)

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, buffer_size=10000, seed=0):
        del buffer_size
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        if len(self.rows) <= 1:
            return FakeHFStream(self.rows)
        order = torch.randperm(len(self.rows), generator=generator).tolist()
        return FakeHFStream([self.rows[idx] for idx in order])

    def shard(self, num_shards, index, contiguous=False):
        del contiguous
        return FakeHFStream(self.rows[index::num_shards])


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _padding_waste(batches):
    padded_tokens = sum(int(batch["input_ids"].numel()) for batch in batches)
    real_tokens = sum(int(batch["attention_mask"].sum().item()) for batch in batches)
    return padded_tokens - real_tokens


def _batch_length_spreads(batches):
    spreads = []
    for batch in batches:
        lengths = batch["attention_mask"].sum(dim=1)
        spreads.append(int(lengths.max().item() - lengths.min().item()))
    return spreads


def _compile_padded_batches(batches, multiple):
    padded = []
    for batch in batches:
        ids, labels, mask = pad_training_batch_to_multiple(
            batch["input_ids"],
            batch["labels"],
            batch.get("attention_mask"),
            multiple=multiple,
            pad_token_id=0,
        )
        assert ids.shape[1] % multiple == 0
        padded.append({"input_ids": ids, "labels": labels, "attention_mask": mask})
    return padded


def _alternating_length_rows():
    lengths = [8, 180, 12, 170, 16, 160, 20, 150, 24, 140, 28, 130, 32, 120, 36, 110]
    return [{"text": "x" * length} for length in lengths]


def _write_binary_token_cache(directory, rows, tokenizer, max_length=256):
    os.makedirs(directory, exist_ok=True)
    offsets = []
    lengths = []
    total_bytes = 0
    with open(os.path.join(directory, "tokens.bin"), "wb") as f:
        for row in rows:
            processed = process_text_sample(tokenizer, row, max_length)
            assert processed is not None
            input_ids = processed["input_ids"].to(dtype=torch.int32).contiguous()
            labels = processed["labels"].to(dtype=torch.int32).contiguous()
            length = min(int(input_ids.numel()), int(labels.numel()))
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
        json.dump({"samples": len(lengths)}, f)


def test_process_text_sample_autodetects_common_columns():
    tokenizer = TinyTokenizer()
    text_sample = process_text_sample(tokenizer, {"text": "plain text"}, max_length=128)
    instruct_sample = process_text_sample(
        tokenizer,
        {"instruction": "say hi", "output": "hi"},
        max_length=128,
    )

    assert text_sample is not None
    assert instruct_sample is not None
    assert text_sample["input_ids"].dtype == torch.long
    assert instruct_sample["labels"].shape == instruct_sample["input_ids"].shape
    assert (instruct_sample["labels"] == -100).any()


def test_alpaca_instruction_input_output_uses_input_section():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Summarize the passage.",
            "input": "The passage to summarize.",
            "output": "A short summary.",
        },
        max_length=256,
        prompt_column="instruction",
        completion_column="output",
    )

    assert processed is not None
    assert tokenizer.calls[0] == (
        "### Instruction:\nSummarize the passage.\n\n"
        "### Input:\nThe passage to summarize.\n\n"
        "### Response:\n",
        True,
    )
    assert tokenizer.calls[1] == ("A short summary.", False)
    assert "User:" not in tokenizer.calls[0][0]
    prompt_len = len(TinyTokenizer().encode(tokenizer.calls[0][0]))
    assert torch.equal(
        processed["labels"][:prompt_len],
        torch.full((prompt_len,), -100, dtype=torch.long),
    )


def test_alpaca_flag_defaults_instruction_output_columns():
    tokenizer = RecordingTokenizer()
    processed = process_text_sample(
        tokenizer,
        {
            "instruction": "Classify sentiment.",
            "input": "I love this.",
            "output": "positive",
        },
        max_length=256,
        alpaca_mode=True,
    )

    assert processed is not None
    assert "### Input:\nI love this.\n\n" in tokenizer.calls[0][0]
    assert tokenizer.calls[1] == ("positive", False)


def test_streaming_jsonl_dataloader_batches_without_materializing():
    tokenizer = TinyTokenizer()
    rows = [{"text": f"sample {idx}"} for idx in range(11)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        dataloader = create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=64,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        )

        assert isinstance(dataloader.dataset, StreamingJSONLDataset)
        assert not hasattr(dataloader.dataset, "samples")

        batches = list(dataloader)
        assert len(batches) == 3
        assert sum(batch["input_ids"].shape[0] for batch in batches) == len(rows)
        assert all(batch["input_ids"].dtype == torch.long for batch in batches)


def test_jsonl_byte_shards_cover_lines_once():
    rows = [{"text": f"row {idx}", "idx": idx} for idx in range(37)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        seen = []
        for worker_id in range(4):
            for _line_num, line in _iter_jsonl_lines_for_worker(path, worker_id, 4):
                seen.append(json.loads(line)["idx"])

    assert sorted(seen) == list(range(len(rows)))


def test_jsonl_file_shards_assign_whole_files_to_workers():
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        expected = []
        for shard_idx in range(4):
            path = os.path.join(tmpdir, f"shard_{shard_idx:05d}.jsonl")
            rows = [{"text": f"row {shard_idx}-{idx}", "idx": shard_idx * 10 + idx} for idx in range(3)]
            expected.extend(row["idx"] for row in rows)
            _write_jsonl(path, rows)
            paths.append(path)

        seen = []
        for worker_id in range(4):
            for _line_num, line in _iter_jsonl_shards_for_worker(paths, worker_id, 4):
                seen.append(json.loads(line)["idx"])

    assert sorted(seen) == sorted(expected)


def test_streaming_jsonl_directory_shards_batch_once():
    tokenizer = TinyTokenizer()
    with tempfile.TemporaryDirectory() as tmpdir:
        expected_rows = 0
        for shard_idx in range(3):
            rows = [{"text": f"directory shard {shard_idx} sample {idx}"} for idx in range(5)]
            expected_rows += len(rows)
            _write_jsonl(os.path.join(tmpdir, f"shard_{shard_idx:05d}.jsonl"), rows)
        _write_jsonl(os.path.join(tmpdir, "manifest.jsonl"), [{"ignored": True}])

        batches = list(create_dataloader_for_jsonl(
            tmpdir,
            tokenizer,
            max_length=96,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
        ))

    assert sum(batch["input_ids"].shape[0] for batch in batches) == expected_rows


def test_hf_materialization_partitions_rows_without_copying_shards():
    import hierarchos_cli

    original_loader = hierarchos_cli.load_hf_dataset
    rows = [{"text": f"hf row {idx}", "idx": idx} for idx in range(23)]

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: iter(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_shard_cache_dir=tmpdir,
                refresh_hf_shards=True,
                dataset_size=len(rows),
            )
            shard_dir = hierarchos_cli.materialize_hf_dataset_shards(args, TinyTokenizer(), 4)

            seen = []
            shard_counts = []
            tokenized_rows = 0
            with open(os.path.join(shard_dir, "manifest.jsonl"), "r", encoding="utf-8") as f:
                entries = [json.loads(line) for line in f if line.strip()]
            by_file = {}
            for entry in entries:
                by_file.setdefault(entry["file_path"], []).append(entry["index_in_file"])

            shard_counts_by_prefix = {idx: 0 for idx in range(4)}
            for file_path, indices in by_file.items():
                shard_idx = int(os.path.basename(file_path).split("_")[1])
                shard_items = torch.load(os.path.join(shard_dir, file_path), map_location="cpu")
                shard_counts_by_prefix[shard_idx] += len(indices)
                for index in indices:
                    row = shard_items[index]
                    if "input_ids" in row and "labels" in row:
                        tokenized_rows += 1
                    seen.append(row["_source_idx"])
            shard_counts = list(shard_counts_by_prefix.values())

    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert sorted(seen) == list(range(len(rows)))
    assert len(seen) == len(set(seen))
    assert tokenized_rows == len(rows)
    assert max(shard_counts) - min(shard_counts) <= 1


def test_streaming_jsonl_reads_pretokenized_rows_without_tokenizer():
    class ExplodingTokenizer(TinyTokenizer):
        def encode(self, text, add_special_tokens=True):
            raise AssertionError("pre-tokenized JSONL should not call tokenizer.encode")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tokenized.jsonl")
        _write_jsonl(path, [
            {"input_ids": [10, 11, 12], "labels": [-100, 11, 12], "attention_mask": [1, 1, 1]},
            {"input_ids": [20, 21], "labels": [20, 21], "length": 2},
        ])

        batches = list(create_dataloader_for_jsonl(
            path,
            ExplodingTokenizer(),
            max_length=16,
            batch_size=2,
            pad_token_id=0,
            num_workers=0,
            use_length_bucketing=True,
        ))

    assert len(batches) == 1
    assert int(batches[0]["attention_mask"].sum().item()) == 5


def test_cli_detects_jsonl_shard_directory():
    import hierarchos_cli

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_jsonl(os.path.join(tmpdir, "manifest.jsonl"), [{"ignored": True}])
        _write_jsonl(os.path.join(tmpdir, "shard_00000.jsonl"), [{"text": "visible shard"}])
        _write_jsonl(os.path.join(tmpdir, "shard_00001.ndjson"), [{"text": "second shard"}])

        assert hierarchos_cli._is_jsonl_source(tmpdir)
        assert hierarchos_cli.count_jsonl_source_rows(tmpdir) == 2


def test_cli_prefers_indexed_hf_for_single_streaming_shard_by_default():
    import hierarchos_cli
    from hierarchos.training.datasets import HuggingFaceMapStyleDataset

    class SingleShardStream(FakeHFStream):
        num_shards = 1

    original_loader = hierarchos_cli.load_hf_dataset
    rows = [{"text": f"hf row {idx}"} for idx in range(8)]

    def fake_loader(*args, **kwargs):
        if kwargs.get("streaming"):
            return SingleShardStream(rows)
        return list(rows)

    try:
        hierarchos_cli.load_hf_dataset = fake_loader
        args = types.SimpleNamespace(
            hf_dataset="fake/hf",
            hf_dataset_config=None,
            hf_dataset_split="train",
            max_length=64,
            kayla=False,
            text_column=None,
            prompt_column=None,
            completion_column=None,
            batch_size=4,
            num_workers=4,
            length_bucketing=True,
            length_bucket_size=None,
            streaming_datasets=True,
            hf_auto_shard=False,
            hf_streaming_shuffle_buffer=100,
            prefetch_factor=None,
        )
        dataloader = hierarchos_cli.create_hf_training_dataloader(args, TinyTokenizer(), torch.device("cpu"))
    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert isinstance(dataloader.dataset, HuggingFaceMapStyleDataset)


def test_tokenized_binary_cache_is_map_style_and_bucketed():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()

    with tempfile.TemporaryDirectory() as tmpdir:
        _write_binary_token_cache(tmpdir, rows, tokenizer, max_length=256)
        dataset = TokenizedBinaryDataset(tmpdir, max_length=256)
        assert len(dataset) == len(rows)
        assert dataset.get_sample_lengths()
        first = dataset[0]
        assert first["input_ids"].dtype == torch.int32
        assert first["labels"].dtype == torch.int32
        dataset.close()

        unbucketed_loader = create_dataloader_for_tokenized_cache(
            tmpdir,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        )
        unbucketed = list(unbucketed_loader)
        unbucketed_loader.dataset.close()
        bucketed_loader = create_dataloader_for_tokenized_cache(
            tmpdir,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        )
        bucketed = list(bucketed_loader)
        bucketed_loader.dataset.close()

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_cli_builds_and_uses_hf_token_cache():
    import hierarchos_cli
    from hierarchos.training.datasets import TokenizedBinaryDataset

    original_loader = hierarchos_cli.load_hf_dataset
    rows = _alternating_length_rows()

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: list(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_token_cache_dir=tmpdir,
                hf_shard_cache_dir=None,
                refresh_hf_token_cache=True,
                tokenizer_path=None,
                model_path=None,
                max_length=96,
                kayla=False,
                text_column=None,
                prompt_column=None,
                completion_column=None,
                batch_size=4,
                num_workers=0,
                length_bucket_size=len(rows),
                hf_token_cache=True,
                streaming_datasets=True,
                prefetch_factor=None,
            )
            dataloader = hierarchos_cli.create_hf_training_dataloader(
                args,
                TinyTokenizer(),
                torch.device("cpu"),
            )
            batches = list(dataloader)
            dataloader.dataset.close()
    finally:
        hierarchos_cli.load_hf_dataset = original_loader

    assert isinstance(dataloader.dataset, TokenizedBinaryDataset)
    assert isinstance(dataloader.batch_sampler, LengthGroupedBatchSampler)
    assert sum(batch["input_ids"].shape[0] for batch in batches) == len(rows)
    assert max(_batch_length_spreads(batches)) <= 32


def test_cli_hf_token_cache_respects_disabled_length_bucketing():
    import hierarchos_cli

    original_loader = hierarchos_cli.load_hf_dataset
    rows = _alternating_length_rows()

    try:
        hierarchos_cli.load_hf_dataset = lambda *args, **kwargs: list(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = types.SimpleNamespace(
                hf_dataset="fake/hf",
                hf_dataset_config=None,
                hf_dataset_split="train",
                hf_token_cache_dir=tmpdir,
                hf_shard_cache_dir=None,
                refresh_hf_token_cache=True,
                tokenizer_path=None,
                model_path=None,
                max_length=256,
                kayla=False,
                text_column=None,
                prompt_column=None,
                completion_column=None,
                batch_size=4,
                num_workers=0,
                length_bucketing=False,
                length_bucket_size=len(rows),
                hf_token_cache=True,
                streaming_datasets=True,
                prefetch_factor=None,
            )
            dataloader = hierarchos_cli.create_hf_training_dataloader(
                args,
                TinyTokenizer(),
                torch.device("cpu"),
            )
            try:
                assert not isinstance(dataloader.batch_sampler, LengthGroupedBatchSampler)
            finally:
                dataloader.dataset.close()
    finally:
        hierarchos_cli.load_hf_dataset = original_loader


def test_hf_streaming_dataloader_batches():
    tokenizer = TinyTokenizer()
    hf_rows = FakeHFStream({"text": f"streamed sample {idx}"} for idx in range(10))

    dataloader = create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=64,
        batch_size=3,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=False,
        shuffle=False,
    )

    assert isinstance(dataloader.dataset, HuggingFaceStreamingDataset)
    batches = list(dataloader)
    assert len(batches) == 4
    assert sum(batch["input_ids"].shape[0] for batch in batches) == 10


def test_hf_worker_shards_cover_samples_once():
    hf_rows = FakeHFStream({"text": f"row {idx}", "idx": idx} for idx in range(19))
    seen = []

    for worker_id in range(3):
        for sample in _iter_hf_worker_samples(hf_rows, worker_id, 3):
            seen.append(sample["idx"])

    assert sorted(seen) == list(range(19))


def test_streaming_jsonl_length_buckets_reduce_padding():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_hf_streaming_length_buckets_reduce_padding():
    tokenizer = TinyTokenizer()
    hf_rows = FakeHFStream(_alternating_length_rows())

    unbucketed = list(create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=256,
        batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=False,
        shuffle=False,
    ))
    bucketed = list(create_dataloader_for_hf_streaming(
        hf_rows,
        tokenizer,
        max_length=256,
        batch_size=4,
        pad_token_id=tokenizer.pad_token_id,
        num_workers=0,
        use_length_bucketing=True,
        bucket_size=len(hf_rows.rows),
        shuffle=False,
    ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(hf_rows.rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_prechunked_jsonl_streaming_length_buckets_reduce_padding():
    max_length = 64
    valid_lengths = [4, 60, 8, 56, 12, 52, 16, 48, 20, 44, 24, 40]
    rows = []
    for idx, valid_length in enumerate(valid_lengths):
        input_ids = list(range(1, valid_length + 1)) + [0] * (max_length - valid_length)
        labels = list(input_ids)
        attention_mask = [1] * valid_length + [0] * (max_length - valid_length)
        rows.append({
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "idx": idx,
        })

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "chunked.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_chunked(
            path,
            max_length=max_length,
            batch_size=4,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_chunked(
            path,
            max_length=max_length,
            batch_size=4,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    assert sum(batch["input_ids"].shape[0] for batch in bucketed) == len(rows)
    assert _padding_waste(bucketed) < _padding_waste(unbucketed)


def test_length_bucketing_still_reduces_padding_with_static_compile_chunks():
    tokenizer = TinyTokenizer()
    rows = _alternating_length_rows()
    multiple = 32

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        unbucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=False,
        ))
        bucketed = list(create_dataloader_for_jsonl(
            path,
            tokenizer,
            max_length=256,
            batch_size=4,
            pad_token_id=tokenizer.pad_token_id,
            num_workers=0,
            use_length_bucketing=True,
            bucket_size=len(rows),
        ))

    unbucketed_static = _compile_padded_batches(unbucketed, multiple)
    bucketed_static = _compile_padded_batches(bucketed, multiple)

    assert sum(batch["input_ids"].shape[0] for batch in bucketed_static) == len(rows)
    assert _padding_waste(bucketed_static) < _padding_waste(unbucketed_static)


def test_map_style_jsonl_fallback_still_loads_text_samples():
    tokenizer = TinyTokenizer()
    rows = [{"text": f"fallback {idx}"} for idx in range(5)]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train.jsonl")
        _write_jsonl(path, rows)

        dataset = OriginalJSONLDataset(path, tokenizer, max_length=64)

    assert len(dataset) == 5
    assert dataset.get_sample_lengths()


def main():
    tests = [
        test_process_text_sample_autodetects_common_columns,
        test_streaming_jsonl_dataloader_batches_without_materializing,
        test_jsonl_byte_shards_cover_lines_once,
        test_jsonl_file_shards_assign_whole_files_to_workers,
        test_streaming_jsonl_directory_shards_batch_once,
        test_hf_materialization_partitions_rows_without_copying_shards,
        test_streaming_jsonl_reads_pretokenized_rows_without_tokenizer,
        test_cli_detects_jsonl_shard_directory,
        test_cli_prefers_indexed_hf_for_single_streaming_shard_by_default,
        test_tokenized_binary_cache_is_map_style_and_bucketed,
        test_cli_builds_and_uses_hf_token_cache,
        test_cli_hf_token_cache_respects_disabled_length_bucketing,
        test_hf_streaming_dataloader_batches,
        test_hf_worker_shards_cover_samples_once,
        test_streaming_jsonl_length_buckets_reduce_padding,
        test_hf_streaming_length_buckets_reduce_padding,
        test_prechunked_jsonl_streaming_length_buckets_reduce_padding,
        test_length_bucketing_still_reduces_padding_with_static_compile_chunks,
        test_map_style_jsonl_fallback_still_loads_text_samples,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
