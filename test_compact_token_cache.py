import json
import os
from types import SimpleNamespace

import pytest
import torch

import hierarchos_cli
from hierarchos.training.datasets import TokenizedBinaryDataset


class _GPT2SizedTokenizer:
    pad_token_id = 0

    def __len__(self):
        return 50257


def _cache_args(source_path, cache_root):
    return SimpleNamespace(
        train=str(source_path),
        local_token_cache_dir=str(cache_root),
        hf_token_cache_dir=None,
        hf_dataset=None,
        hf_dataset_config=None,
        hf_dataset_split="train",
        hf_dataset_revision=None,
        tokenizer_path="openai-community/gpt2",
        model_path=None,
        max_length=8,
        kayla=False,
        alpaca=True,
        train_prompt_tokens=True,
        prompt_loss_weight=0.10,
        response_loss_weight=1.0,
        response_boundary_loss_weight=2.0,
        response_boundary_tokens=2,
        min_response_tokens=1,
        drop_empty_completions=True,
        text_column=None,
        prompt_column=None,
        completion_column=None,
        use_rosa=True,
        training_chunk_size=4,
        rosa_max_context=8,
        token_cache_build_batch_size=2,
        token_cache_write_buffer_mb=1,
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
        dataset_size=2,
        refresh_local_token_cache=False,
    )


def _dummy_weighted_batch():
    return {
        "input_ids": torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, 0, 0]],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [[10, 11, 12, 13, 14], [20, 21, 22, -100, -100]],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]],
            dtype=torch.long,
        ),
        "loss_weights": torch.tensor(
            [[0.10, 0.10, 2.0, 2.0, 1.0], [0.10, 2.0, 1.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "rosa_ids": torch.tensor(
            [[50257, 10, 11, 12, 13], [50257, 20, 21, 50257, 50257]],
            dtype=torch.long,
        ),
    }


def test_local_compact_cache_is_lossless_small_and_backward_ready(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    cache_root = tmp_path / "cache"
    args = _cache_args(source_path, cache_root)
    batch = _dummy_weighted_batch()
    build_calls = []
    builder_kwargs = {}

    def _fake_builder(*_args, **kwargs):
        build_calls.append(1)
        builder_kwargs.update(kwargs)
        return [batch]

    monkeypatch.setattr(hierarchos_cli, "create_dataloader_for_jsonl", _fake_builder)
    cache_dir = hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer())

    index = torch.load(os.path.join(cache_dir, "index.pt"), map_location="cpu", weights_only=True)
    assert index["storage_schema_version"] == 6
    assert index["token_dtype"] == "uint16"
    assert index["label_dtype"] is None
    assert index["label_encoding"] == "input_ids_alias"
    assert index["rosa_dtype"] == "uint16"
    assert index["label_ignore_sentinel"] is None
    assert index["loss_weight_encoding"] == "float32_palette_rle"
    assert builder_kwargs["in_order"] is False

    # Eight real tokens at four bytes/token: uint16 input plus ROSA. Labels are
    # reconstructed exactly from input ids after the writer verifies equality.
    data_path = os.path.join(cache_dir, "tokens.bin")
    assert os.path.getsize(data_path) == 8 * 4
    with open(os.path.join(cache_dir, "_SUCCESS"), "r", encoding="utf-8") as success_file:
        success = json.load(success_file)
    assert success["bytes"] == 8 * 4

    dataset = TokenizedBinaryDataset(cache_dir, max_length=4, pad_token_id=99)
    first = dataset[0]
    assert first["input_ids"].dtype == torch.int32
    assert torch.equal(first["input_ids"], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(first["labels"], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(first["loss_weights"], batch["loss_weights"][0, :4])
    assert torch.equal(first["rosa_ids"], batch["rosa_ids"][0, :4].to(torch.int32))

    fused = dataset.__getitems__([0, 1])
    assert fused["input_ids"].dtype == torch.int32
    assert fused["labels"].dtype == torch.int32
    assert fused["attention_mask"].dtype == torch.bool
    assert fused["rosa_ids"].dtype == torch.int32
    assert torch.equal(
        fused["labels"],
        torch.tensor([[10, 11, 12, 13], [20, 21, 22, -100]], dtype=torch.int32),
    )
    assert torch.equal(fused["loss_weights"][0], batch["loss_weights"][0, :4])
    assert torch.equal(fused["loss_weights"][1, :3], batch["loss_weights"][1, :3])
    assert fused["loss_weights"][1, 3].item() == 0.0
    reordered = dataset.__getitems__([1, 0, 1])
    assert torch.equal(reordered["input_ids"][0, :3], torch.tensor([20, 21, 22], dtype=torch.int32))
    assert torch.equal(reordered["input_ids"][1], torch.tensor([10, 11, 12, 13], dtype=torch.int32))
    assert torch.equal(reordered["input_ids"][2, :3], torch.tensor([20, 21, 22], dtype=torch.int32))
    assert torch.equal(reordered["labels"][1], reordered["input_ids"][1])
    dataset.close()

    # A completed immutable-key cache is reused rather than rebuilt.
    assert hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer()) == cache_dir
    assert len(build_calls) == 1


def test_compact_cache_rejects_corrupt_loss_run_metadata(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    args = _cache_args(source_path, tmp_path / "cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [_dummy_weighted_batch()],
    )
    cache_dir = hierarchos_cli.materialize_local_token_cache(args, _GPT2SizedTokenizer())
    index_path = os.path.join(cache_dir, "index.pt")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    index["loss_run_ends"][-1] -= 1
    torch.save(index, index_path)

    with pytest.raises(ValueError, match="final loss run"):
        TokenizedBinaryDataset(cache_dir)


def test_explicit_label_cache_preserves_masking_and_alias_mode_rejects_it(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    source_path.write_text('{"instruction":"a","output":"b"}\n', encoding="utf-8")
    masked = _dummy_weighted_batch()
    masked["labels"] = masked["labels"].clone()
    masked["labels"][0, :2] = -100
    masked["labels"][1, 0] = -100
    masked["loss_weights"] = masked["loss_weights"].clone()
    masked["loss_weights"][0, :2] = 0.0
    masked["loss_weights"][1, 0] = 0.0

    alias_args = _cache_args(source_path, tmp_path / "alias-cache")
    monkeypatch.setattr(
        hierarchos_cli,
        "create_dataloader_for_jsonl",
        lambda *_args, **_kwargs: [masked],
    )
    with pytest.raises(ValueError, match="Cannot elide token-cache labels"):
        hierarchos_cli.materialize_local_token_cache(alias_args, _GPT2SizedTokenizer())

    explicit_args = _cache_args(source_path, tmp_path / "explicit-cache")
    explicit_args.train_prompt_tokens = False
    cache_dir = hierarchos_cli.materialize_local_token_cache(
        explicit_args,
        _GPT2SizedTokenizer(),
    )
    index = torch.load(os.path.join(cache_dir, "index.pt"), map_location="cpu", weights_only=True)
    assert index["label_encoding"] is None
    assert index["label_dtype"] == "uint16"
    assert index["label_ignore_sentinel"] == 65535
    assert os.path.getsize(os.path.join(cache_dir, "tokens.bin")) == 8 * 6

    dataset = TokenizedBinaryDataset(cache_dir)
    try:
        assert torch.equal(dataset[0]["labels"], masked["labels"][0])
        assert torch.equal(dataset[1]["labels"], masked["labels"][1, :3])
    finally:
        dataset.close()
