import os
import json
import mmap
import torch
import traceback
import functools
import itertools
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset, Sampler
from typing import Optional, List, Dict, Any

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class LengthGroupedBatchSampler(Sampler):
    """
    Shuffled batch sampler that keeps similarly sized samples together.

    This preserves epoch-level randomness while reducing the amount of padding
    introduced by the dynamic collator.
    """
    def __init__(self, lengths, batch_size: int, shuffle: bool = True,
                 drop_last: bool = False, bucket_size: Optional[int] = None,
                 seed: Optional[int] = None, preserve_order: bool = False):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.lengths = [max(1, int(length)) for length in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.preserve_order = bool(preserve_order)
        target_bucket_size = int(bucket_size or (self.batch_size * 50))
        self.bucket_size = max(self.batch_size, target_bucket_size)
        self.bucket_size = max(
            self.batch_size,
            (self.bucket_size // self.batch_size) * self.batch_size,
        )
        self.seed = int(seed if seed is not None else torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))

        if self.shuffle and not self.preserve_order and len(self.lengths) > 1:
            indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            indices = list(range(len(self.lengths)))

        batches = []
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket = indices[bucket_start:bucket_start + self.bucket_size]
            if self.shuffle:
                bucket.sort(key=self.lengths.__getitem__, reverse=True)
            bucket_batches = []
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    bucket_batches.append(batch)
            if self.shuffle and self.preserve_order and len(bucket_batches) > 1:
                order = torch.randperm(len(bucket_batches), generator=generator).tolist()
                batches.extend(bucket_batches[batch_idx] for batch_idx in order)
            else:
                batches.extend(bucket_batches)

        if self.shuffle and not self.preserve_order and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            for batch_idx in order:
                yield batches[batch_idx]
        else:
            for batch in batches:
                yield batch

class EpochShuffleSampler(Sampler):
    """Deterministic map-style sampler with an epoch hook for resume parity."""
    def __init__(self, data_source, shuffle: bool = True, seed: Optional[int] = None):
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed if seed is not None else torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        length = len(self.data_source)
        if self.shuffle and length > 1:
            generator = torch.Generator()
            generator.manual_seed((self.seed + self.epoch) % (2**63 - 1))
            yield from torch.randperm(length, generator=generator).tolist()
        else:
            yield from range(length)

def _normalize_sample_lengths(dataset, lengths):
    if lengths is None:
        return None
    try:
        lengths = [max(1, int(length)) for length in lengths]
    except (TypeError, ValueError):
        return None
    if len(lengths) != len(dataset):
        return None
    return lengths

def _get_dataset_sample_lengths(dataset):
    lengths = _normalize_sample_lengths(dataset, getattr(dataset, "sample_lengths", None))
    if lengths is not None:
        return lengths

    getter = getattr(dataset, "get_sample_lengths", None)
    if callable(getter):
        return _normalize_sample_lengths(dataset, getter())
    return None

def _device_type(device=None):
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device, torch.device):
        return device.type
    device_str = str(device).lower()
    if device_str.startswith("cuda"):
        return "cuda"
    if device_str.startswith("dml") or device_str.startswith("privateuseone"):
        return "dml"
    return "cpu"

def _pin_memory_for_device(device=None):
    return _device_type(device) == "cuda" and torch.cuda.is_available()

def _worker_init_fn(_worker_id):
    # Keep loader workers from each spinning up a full BLAS/OpenMP thread pool.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(1)

def _resolve_prefetch_factor(num_workers: int, prefetch_factor=None, pin_memory: bool = False):
    if num_workers <= 0:
        return None
    if prefetch_factor is not None:
        return max(1, int(prefetch_factor))

    target_prefetched_batches = 16 if pin_memory else 4
    return max(1, (target_prefetched_batches + num_workers - 1) // num_workers)

def _create_dataloader(dataset, *, batch_size=None, collate_fn=None, num_workers=0,
                       pin_memory=False, shuffle=None, drop_last=False,
                       batch_sampler=None, sampler=None, prefetch_factor=None):
    resolved_num_workers = int(num_workers or 0)
    if resolved_num_workers < 0:
        resolved_num_workers = 0

    kwargs = {
        "dataset": dataset,
        "collate_fn": collate_fn,
        "num_workers": resolved_num_workers,
        "pin_memory": bool(pin_memory),
    }
    if batch_sampler is not None:
        kwargs["batch_sampler"] = batch_sampler
    else:
        kwargs["batch_size"] = batch_size
        if sampler is not None:
            kwargs["sampler"] = sampler
        elif shuffle is not None:
            kwargs["shuffle"] = shuffle
        kwargs["drop_last"] = drop_last

    if kwargs["num_workers"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["worker_init_fn"] = _worker_init_fn
        kwargs["prefetch_factor"] = _resolve_prefetch_factor(
            kwargs["num_workers"],
            prefetch_factor=prefetch_factor,
            pin_memory=kwargs["pin_memory"],
        )
    else:
        kwargs["persistent_workers"] = False

    return DataLoader(**kwargs)

def _as_long_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)

def _as_float_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.tensor(value, dtype=torch.float32)

def _normalize_loss_weight_args(prompt_loss_weight: float = 1.0,
                                response_loss_weight: float = 1.0,
                                response_boundary_loss_weight: float = 1.0,
                                response_boundary_tokens: int = 0):
    try:
        prompt_loss_weight = max(0.0, float(prompt_loss_weight))
    except (TypeError, ValueError):
        prompt_loss_weight = 1.0
    try:
        response_loss_weight = max(0.0, float(response_loss_weight))
    except (TypeError, ValueError):
        response_loss_weight = 1.0
    try:
        response_boundary_loss_weight = max(0.0, float(response_boundary_loss_weight))
    except (TypeError, ValueError):
        response_boundary_loss_weight = 1.0
    try:
        response_boundary_tokens = max(0, int(response_boundary_tokens))
    except (TypeError, ValueError):
        response_boundary_tokens = 0
    return (
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    )

def _uses_custom_loss_weights(prompt_loss_weight: float = 1.0,
                              response_loss_weight: float = 1.0,
                              response_boundary_loss_weight: float = 1.0,
                              response_boundary_tokens: int = 0):
    prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
        _normalize_loss_weight_args(
            prompt_loss_weight,
            response_loss_weight,
            response_boundary_loss_weight,
            response_boundary_tokens,
        )
    )
    return (
        abs(prompt_loss_weight - 1.0) > 1e-8
        or abs(response_loss_weight - 1.0) > 1e-8
        or (response_boundary_tokens > 0 and abs(response_boundary_loss_weight - 1.0) > 1e-8)
    )

def _prompt_response_loss_weights(prompt_len: int, response_len: int,
                                  train_prompt_tokens: bool,
                                  prompt_loss_weight: float = 1.0,
                                  response_loss_weight: float = 1.0,
                                  response_boundary_loss_weight: float = 1.0,
                                  response_boundary_tokens: int = 0):
    prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
        _normalize_loss_weight_args(
            prompt_loss_weight,
            response_loss_weight,
            response_boundary_loss_weight,
            response_boundary_tokens,
        )
    )
    if not _uses_custom_loss_weights(
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    ):
        return None
    prompt_weight = prompt_loss_weight if train_prompt_tokens else 0.0
    weights = [prompt_weight] * max(0, int(prompt_len))
    response_weights = [response_loss_weight] * max(0, int(response_len))
    # The final response position is EOS for prompt/completion rows. Do not
    # boost EOS, or short arithmetic rescue examples teach "end immediately."
    boundary = min(max(0, len(response_weights) - 1), response_boundary_tokens)
    for idx in range(boundary):
        response_weights[idx] = response_loss_weight * response_boundary_loss_weight
    return weights + response_weights

def _compose_prompt_response_sample(prompt_ids, response_ids, eos_token_id, max_length: int,
                                    train_prompt_tokens: bool,
                                    prompt_loss_weight: float = 1.0,
                                    response_loss_weight: float = 1.0,
                                    response_boundary_loss_weight: float = 1.0,
                                    response_boundary_tokens: int = 0,
                                    min_response_tokens: int = 1):
    """
    Compose prompt/completion rows while preserving supervised answer tokens.

    Overlong instruction/input fields can otherwise fill the entire sequence and
    leave only an EOS after the response marker. For small assistant models that
    is especially harmful: it teaches "stop" where the answer should start.
    """
    try:
        max_length = int(max_length)
    except (TypeError, ValueError):
        max_length = 0
    try:
        min_response_tokens = max(0, int(min_response_tokens or 0))
    except (TypeError, ValueError):
        min_response_tokens = 1

    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    if eos_token_id is None:
        eos_token_id = response_ids[-1] if response_ids else 0
    eos_token_id = int(eos_token_id)
    full_response_ids = response_ids + [eos_token_id]

    if max_length > 0 and len(prompt_ids) + len(full_response_ids) > max_length:
        if not full_response_ids:
            return None

        min_answer_tokens = min(min_response_tokens, len(response_ids))
        min_response_total = min(max_length, min_answer_tokens + 1)
        if min_response_total <= 0:
            min_response_total = min(max_length, 1)
        prompt_budget = max_length - min_response_total
        if prompt_budget < 0:
            return None
        if len(prompt_ids) > prompt_budget:
            # Preserve the prompt suffix because it contains the response marker.
            prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []

        response_budget = max_length - len(prompt_ids)
        if response_budget <= 0:
            return None
        if len(full_response_ids) > response_budget:
            full_response_ids = full_response_ids[:response_budget]
            full_response_ids[-1] = eos_token_id

    ids = prompt_ids + full_response_ids
    if not ids:
        return None

    prompt_labels = prompt_ids if train_prompt_tokens else ([-100] * len(prompt_ids))
    labels = prompt_labels + full_response_ids
    loss_weights = _prompt_response_loss_weights(
        len(prompt_ids),
        len(full_response_ids),
        train_prompt_tokens,
        prompt_loss_weight,
        response_loss_weight,
        response_boundary_loss_weight,
        response_boundary_tokens,
    )
    return ids, labels, loss_weights

def _sample_effective_length(item, fallback_length: Optional[int] = None):
    if item is None:
        return fallback_length or 1

    for key in ("_length", "length", "valid_length", "seq_len"):
        if key in item and item[key] is not None:
            try:
                return max(1, int(item[key]))
            except (TypeError, ValueError):
                pass

    attention_mask = item.get("attention_mask")
    if attention_mask is not None:
        try:
            if isinstance(attention_mask, torch.Tensor):
                active = attention_mask.to(dtype=torch.bool).nonzero(as_tuple=False)
                length = int(active[-1].item()) + 1 if active.numel() > 0 else 0
            else:
                length = 0
                for idx, value in enumerate(attention_mask):
                    if int(value) != 0:
                        length = idx + 1
            if length > 0:
                return length
        except Exception:
            pass

    input_ids = item.get("input_ids")
    if input_ids is not None:
        try:
            return max(1, len(input_ids))
        except TypeError:
            pass

    return fallback_length or 1

def _yield_length_bucket(buffer, batch_size: int, shuffle: bool, generator: torch.Generator):
    if not buffer:
        return
    buffer.sort(key=_sample_effective_length, reverse=True)
    if shuffle and batch_size > 1:
        groups = [buffer[i:i + batch_size] for i in range(0, len(buffer), batch_size)]
        order = torch.randperm(len(groups), generator=generator).tolist() if len(groups) > 1 else [0]
        for group_idx in order:
            for item in groups[group_idx]:
                yield item
    else:
        for item in buffer:
            yield item

_TEXT_COLUMN_CANDIDATES = ("text", "content")
_PROMPT_COMPLETION_COLUMN_CANDIDATES = (
    ("instruction", "output"),
    ("prompt", "completion"),
    ("question", "answer"),
)

def _column_has_value(sample: dict, column: Optional[str]) -> bool:
    if not column or column not in sample:
        return False
    value = sample.get(column)
    return value is not None and str(value).strip() != ""

def _resolve_text_sample_columns(text_dict: dict, text_column: Optional[str] = None,
                                 prompt_column: Optional[str] = None,
                                 completion_column: Optional[str] = None,
                                 alpaca_mode: bool = False):
    if text_column:
        return text_column, None, None
    if alpaca_mode and not (prompt_column or completion_column):
        return None, "instruction", "output"
    if prompt_column or completion_column:
        return None, prompt_column, completion_column

    for candidate in _TEXT_COLUMN_CANDIDATES:
        if _column_has_value(text_dict, candidate):
            return candidate, None, None

    for candidate_prompt, candidate_completion in _PROMPT_COMPLETION_COLUMN_CANDIDATES:
        if (
            _column_has_value(text_dict, candidate_prompt)
            or _column_has_value(text_dict, candidate_completion)
        ):
            return None, candidate_prompt, candidate_completion

    return None, None, None

def _is_alpaca_prompt_pair(prompt_column: Optional[str], completion_column: Optional[str]) -> bool:
    return (
        str(prompt_column or "").lower() == "instruction"
        and str(completion_column or "").lower() == "output"
    )

def _format_alpaca_prompt(instruction: str, inp: str) -> str:
    prompt = ""
    if inp:
        prompt += f"### Previous Context:\n{inp}\n\n"
    prompt += f"### Instruction:\n{instruction}\n\n"
    return prompt + "### Response:\n"

def _estimate_text_token_length(text_dict: dict, max_length: int, kayla_mode: bool,
                                text_column: Optional[str] = None,
                                prompt_column: Optional[str] = None,
                                completion_column: Optional[str] = None,
                                alpaca_mode: bool = False):
    text_column, prompt_column, completion_column = _resolve_text_sample_columns(
        text_dict,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
    )
    if text_column:
        text_len = len(str(text_dict.get(text_column, "")))
    elif prompt_column and completion_column:
        text_len = len(str(text_dict.get(prompt_column, "")))
        text_len += len(str(text_dict.get(completion_column, "")))
        text_len += len(str(text_dict.get("input", "")))
        if kayla_mode:
            text_len += len(str(text_dict.get("feelings", "")))
            text_len += len(str(text_dict.get("thought-process", "")))
    else:
        return None

    approx_tokens = (text_len + 3) // 4 + 1
    return max(1, min(int(max_length), approx_tokens))

def _streaming_bucket_size(batch_size: int, use_length_bucketing: bool, bucket_size=None):
    if not use_length_bucketing or batch_size <= 1:
        return 0
    if bucket_size is not None:
        return max(batch_size, int(bucket_size))
    # Keep enough lookahead for dynamic padding wins. HF single-shard datasets
    # use the tokenized PT cache path, so this no longer forces repeated HF
    # tokenization on every epoch.
    return max(batch_size, int(batch_size) * 8)

def _resolve_jsonl_paths(path):
    path = os.fspath(path)
    if os.path.isdir(path):
        paths = []
        for root, _dirs, files in os.walk(path):
            for filename in files:
                lower = filename.lower()
                if lower == "manifest.jsonl":
                    continue
                if lower.endswith((".jsonl", ".ndjson")):
                    paths.append(os.path.join(root, filename))
        paths.sort()
        if not paths:
            raise FileNotFoundError(f"No JSONL/NDJSON shard files found in directory: {path}")
        return paths
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    return [path]

def _iter_jsonl_lines_for_worker(path: str, worker_id: int, num_workers: int):
    if num_workers <= 1:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                yield line_num, line
        return

    file_size = os.path.getsize(path)
    start_byte = (file_size * worker_id) // num_workers
    end_byte = (file_size * (worker_id + 1)) // num_workers
    with open(path, "rb") as f:
        f.seek(start_byte)
        if start_byte > 0:
            f.readline()
        while True:
            if worker_id != num_workers - 1 and f.tell() >= end_byte:
                break
            raw_line = f.readline()
            if not raw_line:
                break
            try:
                yield None, raw_line.decode("utf-8")
            except UnicodeDecodeError:
                continue

def _iter_jsonl_shards_for_worker(paths, worker_id: int, num_workers: int):
    paths = list(paths)
    if not paths:
        return

    # Prefer whole-file assignment when there are enough physical shards. This
    # avoids multiple workers seeking/reading the same file at once.
    if num_workers <= 1 or len(paths) >= num_workers:
        assigned_paths = paths if num_workers <= 1 else [
            path for path_idx, path in enumerate(paths) if path_idx % num_workers == worker_id
        ]
        for path in assigned_paths:
            yield from _iter_jsonl_lines_for_worker(path, 0, 1)
        return

    # If a dataset has fewer files than workers, still split by byte range so
    # extra workers do useful work without duplicating samples.
    for path in paths:
        yield from _iter_jsonl_lines_for_worker(path, worker_id, num_workers)

class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage.
    """
    def __init__(self, path: str, max_length: int, bucket_size: int = 0,
                 batch_size: int = 1, shuffle_buckets: bool = True):
        super().__init__()
        self.path = path
        self.paths = _resolve_jsonl_paths(path)
        self.max_length = max_length
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle_buckets = bool(shuffle_buckets)
        self.seed = int(torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        print(f"[Worker {worker_id}/{num_workers}] Opening {len(self.paths)} chunked JSONL shard(s): {self.path}")
        skipped_count = 0
        processed_count = 0
        bucket = []
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch + worker_id) % (2**63 - 1))

        try:
            for _line_num, line in _iter_jsonl_shards_for_worker(self.paths, worker_id, num_workers):
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                        if not all(k in obj for k in ["input_ids", "labels", "attention_mask"]):
                            skipped_count += 1
                            continue
                        seq_len = len(obj["input_ids"])
                        if seq_len != self.max_length:
                            skipped_count += 1
                            continue
                        attention_mask = obj["attention_mask"]
                        try:
                            valid_length = 0
                            for mask_idx, mask_value in enumerate(attention_mask):
                                if int(mask_value) != 0:
                                    valid_length = mask_idx + 1
                        except Exception:
                            valid_length = seq_len
                        processed_count += 1
                        item = {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                            "_length": valid_length,
                        }
                        if self.bucket_size > 0:
                            bucket.append(item)
                            if len(bucket) >= self.bucket_size:
                                yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
                                bucket.clear()
                        else:
                            yield item
                    except Exception:
                        skipped_count += 1
                        continue
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR: {e}")
            raise e
        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
        print(f"[Worker {worker_id}] Finished. Processed: {processed_count}, Skipped: {skipped_count}")

def create_dataloader_for_chunked(path, max_length, batch_size, num_workers=0,
                                  use_length_bucketing=True, bucket_size=None,
                                  device=None, prefetch_factor=None):
    streaming_bucket_size = _streaming_bucket_size(batch_size, use_length_bucketing, bucket_size)
    dataset = IterableChunkedJSONLDataset(
        path,
        max_length=max_length,
        bucket_size=streaming_bucket_size,
        batch_size=batch_size,
        shuffle_buckets=True,
    )
    collate_fn_simple = functools.partial(_collate_training_batch, pad_token_id=0)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_simple,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

class PTChunkedDataset(Dataset):
    def __init__(self, directory_path: str, max_length: int, cache_size: int = 2):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = max_length
        self.cache_size = max(1, int(cache_size or 1))
        self.chunk_pointers = []
        self.sample_lengths = []
        self.last_loaded_path = None
        self.last_loaded_data = None
        self._chunk_cache = OrderedDict()
        manifest_file = os.path.join(directory_path, "manifest.jsonl")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                entry = json.loads(line)
                self.chunk_pointers.append((os.path.join(self.directory_path, entry["file_path"]), entry["index_in_file"]))
                length = entry.get("length", entry.get("seq_len", entry.get("valid_length")))
                try:
                    self.sample_lengths.append(max(1, int(length)))
                except (TypeError, ValueError):
                    self.sample_lengths.append(None)

    def __len__(self): return len(self.chunk_pointers)
    def _get_chunk(self, path):
        cached = self._chunk_cache.get(path)
        if cached is not None:
            self._chunk_cache.move_to_end(path)
            self.last_loaded_path = path
            self.last_loaded_data = cached
            return cached

        data = torch.load(path, map_location='cpu')
        self._chunk_cache[path] = data
        self._chunk_cache.move_to_end(path)
        while len(self._chunk_cache) > self.cache_size:
            self._chunk_cache.popitem(last=False)
        self.last_loaded_path = path
        self.last_loaded_data = data
        return data

    def __getitem__(self, idx):
        path, index = self.chunk_pointers[idx]
        item = self._get_chunk(path)[index]
        if isinstance(item, dict):
            length = self.sample_lengths[idx] if idx < len(self.sample_lengths) else None
            if length is not None and "_length" not in item:
                item = dict(item)
                item["_length"] = length
        return item
    def get_sample_lengths(self):
        if self.sample_lengths and all(length is not None for length in self.sample_lengths):
            return self.sample_lengths

        lengths = list(self.sample_lengths) if self.sample_lengths else [None] * len(self.chunk_pointers)
        positions_by_path = {}
        for position, (path, index) in enumerate(self.chunk_pointers):
            if lengths[position] is None:
                positions_by_path.setdefault(path, []).append((position, index))

        for path, entries in positions_by_path.items():
            try:
                data = torch.load(path, map_location='cpu')
            except Exception:
                return None
            for position, index in entries:
                try:
                    lengths[position] = _sample_effective_length(data[index], self.max_length)
                except Exception:
                    lengths[position] = self.max_length

        self.sample_lengths = lengths
        return self.sample_lengths

def create_dataloader_pt_chunked(directory_path, max_length, batch_size, num_workers=0,
                                 use_length_bucketing=True, bucket_size=None,
                                 device=None, prefetch_factor=None, cache_size: int = 2):
    dataset = PTChunkedDataset(directory_path, max_length=max_length, cache_size=cache_size)
    collate_fn_pt = functools.partial(_collate_training_batch, pad_token_id=0)
    use_cuda = _pin_memory_for_device(device)
    drop_last = False
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and batch_size > 1 else None
    if lengths:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            bucket_size=bucket_size,
            preserve_order=True,
        )
        return _create_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_pt,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor,
        )

    sampler = EpochShuffleSampler(dataset, shuffle=True)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn_pt,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        prefetch_factor=prefetch_factor,
    )

class TokenizedBinaryDataset(Dataset):
    """
    Random-access token cache backed by one binary file plus a compact index.

    Each record is stored as variable-length int32 input ids followed by int32
    labels. Attention masks are generated during collation from the record
    length, so the cache avoids padded max_length tensors.
    """
    def __init__(self, directory_path: str, max_length: Optional[int] = None):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = int(max_length or 0)
        index_path = os.path.join(directory_path, "index.pt")
        data_path = os.path.join(directory_path, "tokens.bin")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Token cache index not found: {index_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Token cache data file not found: {data_path}")

        index = torch.load(index_path, map_location="cpu")
        self.offsets = index["offsets"].to(dtype=torch.long).contiguous()
        self.lengths = index["lengths"].to(dtype=torch.long).contiguous()
        self.has_rosa_ids = bool(index.get("has_rosa_ids", False))
        self.has_loss_weights = bool(index.get("has_loss_weights", False))
        self.loss_weight_dtype = str(index.get("loss_weight_dtype", "float16"))
        if self.loss_weight_dtype == "float32":
            self._loss_weight_torch_dtype = torch.float32
            self._loss_weight_bytes = 4
        else:
            self._loss_weight_torch_dtype = torch.float16
            self._loss_weight_bytes = 2
        self.rosa_sentinel = int(index.get("rosa_sentinel", 0))
        if self.offsets.numel() != self.lengths.numel():
            raise ValueError("Token cache index offsets/lengths size mismatch")
        self.sample_lengths = [
            max(1, min(int(length), self.max_length)) if self.max_length > 0 else max(1, int(length))
            for length in self.lengths.tolist()
        ]
        self.data_path = data_path
        self._file = None
        self._mmap = None

    def __len__(self):
        return int(self.lengths.numel())

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_file"] = None
        state["_mmap"] = None
        return state

    def _ensure_open(self):
        if self._mmap is not None:
            return
        self._file = open(self.data_path, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()

    def __getitem__(self, idx):
        self._ensure_open()
        offset = int(self.offsets[idx])
        stored_length = int(self.lengths[idx])
        length = stored_length
        if self.max_length > 0:
            length = min(length, self.max_length)
        if length <= 0:
            return None

        label_offset = offset + (stored_length * 4)
        after_labels_offset = label_offset + (stored_length * 4)
        input_ids = torch.frombuffer(self._mmap, dtype=torch.int32, count=length, offset=offset)
        labels = torch.frombuffer(self._mmap, dtype=torch.int32, count=length, offset=label_offset)
        item = {
            "input_ids": input_ids,
            "labels": labels,
            "_length": length,
        }
        if self.has_loss_weights:
            item["loss_weights"] = torch.frombuffer(
                self._mmap,
                dtype=self._loss_weight_torch_dtype,
                count=length,
                offset=after_labels_offset,
            ).to(dtype=torch.float32)
        if self.has_rosa_ids:
            rosa_offset = after_labels_offset
            if self.has_loss_weights:
                rosa_offset += stored_length * self._loss_weight_bytes
            item["rosa_ids"] = torch.frombuffer(
                self._mmap,
                dtype=torch.int32,
                count=length,
                offset=rosa_offset,
            )
            item["_rosa_sentinel"] = self.rosa_sentinel
        return item

    def get_sample_lengths(self):
        return self.sample_lengths


def create_dataloader_for_tokenized_cache(directory_path, max_length, batch_size, pad_token_id,
                                          num_workers=0, use_length_bucketing=True,
                                          bucket_size=None, device=None, prefetch_factor=None):
    dataset = TokenizedBinaryDataset(directory_path, max_length=max_length)
    return create_map_style_dataloader(
        dataset,
        batch_size=batch_size,
        pad_token_id=pad_token_id,
        num_workers=num_workers,
        shuffle=True,
        use_length_bucketing=use_length_bucketing,
        bucket_size=bucket_size,
        device=device,
        prefetch_factor=prefetch_factor,
    )

def process_text_sample(tokenizer, text_dict: dict, max_length: int, kayla_mode: bool = False,
                         text_column: Optional[str] = None,
                         prompt_column: Optional[str] = None, completion_column: Optional[str] = None,
                         alpaca_mode: bool = False,
                         train_prompt_tokens: bool = True,
                         prompt_loss_weight: float = 1.0,
                         response_loss_weight: float = 1.0,
                         response_boundary_loss_weight: float = 1.0,
                         response_boundary_tokens: int = 0,
                         min_response_tokens: int = 1,
                         drop_empty_completions: bool = True):
    try:
        if not isinstance(text_dict, dict):
            return None
        prompt_loss_weight, response_loss_weight, response_boundary_loss_weight, response_boundary_tokens = (
            _normalize_loss_weight_args(
                prompt_loss_weight,
                response_loss_weight,
                response_boundary_loss_weight,
                response_boundary_tokens,
            )
        )
        loss_weights = None

        text_column, prompt_column, completion_column = _resolve_text_sample_columns(
            text_dict,
            text_column=text_column,
            prompt_column=prompt_column,
            completion_column=completion_column,
            alpaca_mode=alpaca_mode,
        )

        if text_column:
            text = str(text_dict.get(text_column, ""))
            if not text.strip(): return None
            ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
            labels = list(ids)
            if _uses_custom_loss_weights(
                prompt_loss_weight,
                response_loss_weight,
                response_boundary_loss_weight,
                response_boundary_tokens,
            ):
                loss_weights = [response_loss_weight] * len(ids)
        elif prompt_column and completion_column:
            instruction = str(text_dict.get(prompt_column, ""))
            output = str(text_dict.get(completion_column, ""))
            inp = str(text_dict.get('input', "")).strip()
            if drop_empty_completions and not output.strip():
                return None
            if not instruction.strip() and not output.strip() and not inp:
                return None
            if kayla_mode:
                feelings = str(text_dict.get('feelings', ''))
                thought = str(text_dict.get('thought-process', ''))
                prompt_text = f"### Instruction:\n{instruction}\n\n" + (f"### Feelings:\n{feelings}\n\n" if feelings else "")
                thought_text = f"### Thought Process:\n{thought}\n\n"
                response_text = f"### Response:\n{output}"
                p_ids = tokenizer.encode(prompt_text)
                t_ids = tokenizer.encode(thought_text, add_special_tokens=False)
                r_ids = tokenizer.encode(response_text, add_special_tokens=False)
                composed = _compose_prompt_response_sample(
                    p_ids,
                    t_ids + r_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    min_response_tokens,
                )
                if composed is None:
                    return None
                ids, labels, loss_weights = composed
            elif alpaca_mode or _is_alpaca_prompt_pair(prompt_column, completion_column):
                prompt = _format_alpaca_prompt(instruction, inp)
                p_ids = tokenizer.encode(prompt)
                c_ids = tokenizer.encode(output, add_special_tokens=False)
                composed = _compose_prompt_response_sample(
                    p_ids,
                    c_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    min_response_tokens,
                )
                if composed is None:
                    return None
                ids, labels, loss_weights = composed
            else:
                if inp:
                    prompt = f"User: {inp}\n\nUser: {instruction}\n\nAssistant: "
                else:
                    prompt = f"User: {instruction}\n\nAssistant: "
                p_ids = tokenizer.encode(prompt)
                c_ids = tokenizer.encode(output, add_special_tokens=False)
                composed = _compose_prompt_response_sample(
                    p_ids,
                    c_ids,
                    tokenizer.eos_token_id,
                    max_length,
                    train_prompt_tokens,
                    prompt_loss_weight,
                    response_loss_weight,
                    response_boundary_loss_weight,
                    response_boundary_tokens,
                    min_response_tokens,
                )
                if composed is None:
                    return None
                ids, labels, loss_weights = composed
        else: return None
        if len(ids) > max_length:
            ids = ids[:max_length-1] + [tokenizer.eos_token_id]
            labels = labels[:max_length-1] + [tokenizer.eos_token_id]
            if loss_weights is not None:
                eos_weight = loss_weights[-1] if loss_weights else 1.0
                loss_weights = loss_weights[:max_length-1] + [eos_weight]
        sample = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "_length": len(ids),
        }
        if loss_weights is not None:
            if len(loss_weights) < len(ids):
                loss_weights = loss_weights + ([1.0] * (len(ids) - len(loss_weights)))
            elif len(loss_weights) > len(ids):
                loss_weights = loss_weights[:len(ids)]
            sample["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return sample
    except: return None

def process_tokenized_sample(text_dict: dict, max_length: Optional[int] = None):
    """Load a pre-tokenized JSONL row without running the tokenizer again."""
    if not isinstance(text_dict, dict) or "input_ids" not in text_dict:
        return None
    try:
        input_ids = text_dict.get("input_ids")
        labels = text_dict.get("labels", input_ids)
        attention_mask = text_dict.get("attention_mask")
        loss_weights = text_dict.get("loss_weights")

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.detach().cpu().tolist()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().tolist()
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = attention_mask.detach().cpu().tolist()
        if isinstance(loss_weights, torch.Tensor):
            loss_weights = loss_weights.detach().cpu().tolist()

        input_ids = [int(value) for value in input_ids]
        labels = [int(value) for value in labels]
        if loss_weights is not None:
            loss_weights = [float(value) for value in loss_weights]
        if not input_ids:
            return None

        max_len = int(max_length or 0)
        if max_len > 0:
            input_ids = input_ids[:max_len]
            labels = labels[:max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:max_len]
            if loss_weights is not None:
                loss_weights = loss_weights[:max_len]

        if len(labels) < len(input_ids):
            labels = labels + ([-100] * (len(input_ids) - len(labels)))
        elif len(labels) > len(input_ids):
            labels = labels[:len(input_ids)]

        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        else:
            attention_mask = [int(value) for value in attention_mask]
            if len(attention_mask) < len(input_ids):
                attention_mask = attention_mask + ([1] * (len(input_ids) - len(attention_mask)))
            elif len(attention_mask) > len(input_ids):
                attention_mask = attention_mask[:len(input_ids)]
        if loss_weights is not None:
            if len(loss_weights) < len(input_ids):
                loss_weights = loss_weights + ([1.0] * (len(input_ids) - len(loss_weights)))
            elif len(loss_weights) > len(input_ids):
                loss_weights = loss_weights[:len(input_ids)]

        length = text_dict.get("_length", text_dict.get("length", text_dict.get("valid_length")))
        try:
            length = int(length)
        except (TypeError, ValueError):
            length = _sample_effective_length(
                {"input_ids": input_ids, "attention_mask": attention_mask},
                fallback_length=len(input_ids),
            )
        length = max(1, min(len(input_ids), length))

        sample = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "_length": length,
        }
        if loss_weights is not None:
            sample["loss_weights"] = torch.tensor(loss_weights, dtype=torch.float32)
        return sample
    except Exception:
        return None

class OriginalJSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, kayla_mode=False,
                 text_column: Optional[str] = None,
                 prompt_column: Optional[str] = None,
                 completion_column: Optional[str] = None,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True):
        super().__init__()
        self.tokenizer, self.max_length, self.kayla_mode, self.samples = tokenizer, max_length, kayla_mode, []
        self.sample_lengths = []
        skipped = 0
        line_num = 0

        def add_sample(data):
            processed = process_text_sample(
                tokenizer,
                data,
                max_length,
                kayla_mode,
                text_column=text_column,
                prompt_column=prompt_column,
                completion_column=completion_column,
                alpaca_mode=alpaca_mode,
                train_prompt_tokens=train_prompt_tokens,
                prompt_loss_weight=prompt_loss_weight,
                response_loss_weight=response_loss_weight,
                response_boundary_loss_weight=response_boundary_loss_weight,
                response_boundary_tokens=response_boundary_tokens,
                min_response_tokens=min_response_tokens,
                drop_empty_completions=drop_empty_completions,
            )
            if processed:
                self.samples.append(processed)
                self.sample_lengths.append(len(processed["input_ids"]))
                return True
            return False

        with open(path, "r", encoding="utf-8") as f:
            if not str(path).lower().endswith((".jsonl", ".ndjson")):
                try:
                    data = json.load(f)
                    if isinstance(data, dict):
                        data = [data]
                    if not isinstance(data, list):
                        raise ValueError("JSON dataset must be an object, a list of objects, or JSONL.")
                    for obj in tqdm(data, desc="Loading JSON"):
                        if not add_sample(obj):
                            skipped += 1
                    if skipped:
                        print(f"WARNING: Skipped {skipped} invalid JSON samples.")
                    return
                except json.JSONDecodeError:
                    f.seek(0)
                except Exception:
                    f.seek(0)

            for line_num, line in enumerate(tqdm(f, desc="Loading JSONL"), 1):
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    if skipped <= 5:
                        print(f"WARNING: Skipping malformed JSON at line {line_num} (char preview: {line[:80]}...)")
                    continue
                if not add_sample(data):
                    skipped += 1
        if skipped:
            print(f"WARNING: Skipped {skipped} malformed JSONL lines out of {line_num} total.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def get_sample_lengths(self): return self.sample_lengths

class StreamingJSONLDataset(IterableDataset):
    """
    Stream raw JSONL text samples from disk and tokenize inside DataLoader
    workers. This avoids materializing a tokenized tensor list before training.
    """
    def __init__(self, path, tokenizer, max_length, kayla_mode=False,
                 text_column: Optional[str] = None,
                 prompt_column: Optional[str] = None,
                 completion_column: Optional[str] = None,
                 bucket_size: int = 0, batch_size: int = 1,
                 shuffle_buckets: bool = True,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True):
        super().__init__()
        self.path = path
        self.paths = _resolve_jsonl_paths(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kayla_mode = kayla_mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle_buckets = bool(shuffle_buckets)
        self.seed = int(torch.initial_seed()) % (2**63 - 1)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        bucket = []
        generator = torch.Generator()
        generator.manual_seed((self.seed + self.epoch + worker_id) % (2**63 - 1))

        for _line_num, line in _iter_jsonl_shards_for_worker(self.paths, worker_id, num_workers):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            processed = process_tokenized_sample(obj, self.max_length)
            if processed is None:
                processed = process_text_sample(
                    self.tokenizer,
                    obj,
                    self.max_length,
                    self.kayla_mode,
                    text_column=self.text_column,
                    prompt_column=self.prompt_column,
                    completion_column=self.completion_column,
                    alpaca_mode=self.alpaca_mode,
                    train_prompt_tokens=self.train_prompt_tokens,
                    prompt_loss_weight=self.prompt_loss_weight,
                    response_loss_weight=self.response_loss_weight,
                    response_boundary_loss_weight=self.response_boundary_loss_weight,
                    response_boundary_tokens=self.response_boundary_tokens,
                    min_response_tokens=self.min_response_tokens,
                    drop_empty_completions=self.drop_empty_completions,
                )
            if processed is None:
                continue
            if self.bucket_size > 0:
                bucket.append(processed)
                if len(bucket) >= self.bucket_size:
                    yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)
                    bucket.clear()
            else:
                yield processed

        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle_buckets, generator)

def _maybe_shuffle_hf_dataset(hf_dataset, shuffle: bool, shuffle_buffer_size: int, seed: int):
    if not shuffle or shuffle_buffer_size <= 0:
        return hf_dataset
    shuffle_fn = getattr(hf_dataset, "shuffle", None)
    if not callable(shuffle_fn):
        return hf_dataset
    try:
        return shuffle_fn(buffer_size=int(shuffle_buffer_size), seed=int(seed))
    except TypeError:
        try:
            return shuffle_fn(seed=int(seed))
        except Exception:
            return hf_dataset
    except Exception:
        return hf_dataset

def _iter_hf_worker_samples(hf_dataset, worker_id: int, num_workers: int):
    dataset = hf_dataset
    sharded = False
    if num_workers > 1:
        shard_fn = getattr(dataset, "shard", None)
        if callable(shard_fn):
            try:
                dataset = shard_fn(num_shards=num_workers, index=worker_id, contiguous=False)
                sharded = True
            except TypeError:
                try:
                    dataset = shard_fn(num_shards=num_workers, index=worker_id)
                    sharded = True
                except Exception:
                    sharded = False
            except Exception:
                sharded = False

    iterator = iter(dataset)
    if num_workers > 1 and not sharded:
        iterator = itertools.islice(iterator, worker_id, None, num_workers)
    return iterator

class HuggingFaceStreamingDataset(IterableDataset):
    """
    Iterable wrapper for Hugging Face streaming datasets. It performs buffered
    stream shuffling, worker sharding, tokenization, and optional length buckets
    without building a full tokenized dataset in memory.
    """
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False,
                 text_column=None, prompt_column=None, completion_column=None,
                 bucket_size: int = 0, batch_size: int = 1,
                 shuffle: bool = True, shuffle_buffer_size: int = 10000,
                 alpaca_mode: bool = False,
                 train_prompt_tokens: bool = True,
                 prompt_loss_weight: float = 1.0,
                 response_loss_weight: float = 1.0,
                 response_boundary_loss_weight: float = 1.0,
                 response_boundary_tokens: int = 0,
                 min_response_tokens: int = 1,
                 drop_empty_completions: bool = True):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.kayla_mode = kayla_mode
        self.text_column = text_column
        self.prompt_column = prompt_column
        self.completion_column = completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.shuffle_buffer_size = max(0, int(shuffle_buffer_size or 0))
        self.seed = int(torch.initial_seed()) % (2**32 - 1)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1
        seed = int((self.seed + (self.epoch * 100003) + worker_id) % (2**32 - 1))
        dataset = _maybe_shuffle_hf_dataset(
            self.hf_dataset,
            shuffle=self.shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=seed,
        )
        iterator = _iter_hf_worker_samples(dataset, worker_id, num_workers)

        bucket = []
        generator = torch.Generator()
        generator.manual_seed(seed)

        for sample in iterator:
            processed = process_text_sample(
                self.tokenizer,
                sample,
                self.max_length,
                self.kayla_mode,
                self.text_column,
                self.prompt_column,
                self.completion_column,
                self.alpaca_mode,
                self.train_prompt_tokens,
                self.prompt_loss_weight,
                self.response_loss_weight,
                self.response_boundary_loss_weight,
                self.response_boundary_tokens,
                self.min_response_tokens,
                self.drop_empty_completions,
            )
            if processed is None:
                continue
            if self.bucket_size > 0:
                bucket.append(processed)
                if len(bucket) >= self.bucket_size:
                    yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle, generator)
                    bucket.clear()
            else:
                yield processed

        if bucket:
            yield from _yield_length_bucket(bucket, self.batch_size, self.shuffle, generator)

class HuggingFaceMapStyleDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False, text_column=None, prompt_column=None, completion_column=None, alpaca_mode: bool = False, train_prompt_tokens: bool = True, prompt_loss_weight: float = 1.0, response_loss_weight: float = 1.0, response_boundary_loss_weight: float = 1.0, response_boundary_tokens: int = 0, min_response_tokens: int = 1, drop_empty_completions: bool = True):
        super().__init__()
        self.hf_dataset, self.tokenizer, self.max_length, self.kayla_mode = hf_dataset, tokenizer, max_length, kayla_mode
        self.text_column, self.prompt_column, self.completion_column = text_column, prompt_column, completion_column
        self.alpaca_mode = alpaca_mode
        self.train_prompt_tokens = bool(train_prompt_tokens)
        self.prompt_loss_weight = prompt_loss_weight
        self.response_loss_weight = response_loss_weight
        self.response_boundary_loss_weight = response_boundary_loss_weight
        self.response_boundary_tokens = response_boundary_tokens
        self.min_response_tokens = min_response_tokens
        self.drop_empty_completions = bool(drop_empty_completions)
        self.sample_lengths = None
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        return process_text_sample(self.tokenizer, self.hf_dataset[idx], self.max_length, self.kayla_mode, self.text_column, self.prompt_column, self.completion_column, self.alpaca_mode, self.train_prompt_tokens, self.prompt_loss_weight, self.response_loss_weight, self.response_boundary_loss_weight, self.response_boundary_tokens, self.min_response_tokens, self.drop_empty_completions)
    def get_sample_lengths(self):
        if self.sample_lengths is not None:
            return self.sample_lengths
        lengths = []
        for idx in range(len(self.hf_dataset)):
            try:
                sample = self.hf_dataset[idx]
                length = _estimate_text_token_length(
                    sample, self.max_length, self.kayla_mode,
                    self.text_column, self.prompt_column, self.completion_column,
                    self.alpaca_mode,
                )
                if length is None:
                    processed = process_text_sample(
                        self.tokenizer,
                        sample,
                        self.max_length,
                        self.kayla_mode,
                        self.text_column,
                        self.prompt_column,
                        self.completion_column,
                        self.alpaca_mode,
                        self.train_prompt_tokens,
                        self.prompt_loss_weight,
                        self.response_loss_weight,
                        self.response_boundary_loss_weight,
                        self.response_boundary_tokens,
                        self.min_response_tokens,
                        self.drop_empty_completions,
                    )
                    length = len(processed["input_ids"]) if processed is not None else None
            except Exception:
                return None
            if length is None:
                length = self.max_length
            lengths.append(length)
        self.sample_lengths = lengths
        return self.sample_lengths

def _collate_training_batch(batch, pad_token_id):
    batch = [i for i in batch if i is not None]
    if not batch: return None
    ml = max(_sample_effective_length(i) for i in batch)
    ids = torch.full((len(batch), ml), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), ml), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), ml), dtype=torch.long)
    has_loss_weights = any("loss_weights" in item for item in batch)
    loss_weights = torch.zeros((len(batch), ml), dtype=torch.float32) if has_loss_weights else None
    has_rosa_ids = any("rosa_ids" in item for item in batch)
    rosa_ids = None
    if has_rosa_ids:
        sentinel = next((int(item.get("_rosa_sentinel")) for item in batch if "_rosa_sentinel" in item), None)
        if sentinel is None:
            raise ValueError("Cached ROSA samples must include _rosa_sentinel")
        rosa_ids = torch.full((len(batch), ml), sentinel, dtype=torch.long)
    for i, item in enumerate(batch):
        item_ids = _as_long_tensor(item["input_ids"])
        item_labels = _as_long_tensor(item["labels"])
        item_mask = item.get("attention_mask")
        item_mask = _as_long_tensor(item_mask) if item_mask is not None else None
        item_weights = item.get("loss_weights")
        item_weights = _as_float_tensor(item_weights) if item_weights is not None else None

        sl = min(ml, item_ids.numel(), item_labels.numel())
        if item_mask is not None:
            sl = min(sl, item_mask.numel())
        if item_weights is not None:
            sl = min(sl, item_weights.numel())
        ids[i, :sl] = item_ids[:sl]
        labels[i, :sl] = item_labels[:sl]
        mask[i, :sl] = item_mask[:sl] if item_mask is not None else 1
        if loss_weights is not None:
            loss_weights[i, :sl] = item_weights[:sl] if item_weights is not None else 1.0
        if rosa_ids is not None:
            if "rosa_ids" not in item:
                raise ValueError("Cannot mix cached-ROSA and live-ROSA samples in one batch")
            item_rosa = _as_long_tensor(item["rosa_ids"])
            rosa_ids[i, :min(sl, item_rosa.numel())] = item_rosa[:min(sl, item_rosa.numel())]
    batch_out = {"input_ids": ids, "labels": labels, "attention_mask": mask}
    if loss_weights is not None:
        batch_out["loss_weights"] = loss_weights
    if rosa_ids is not None:
        batch_out["rosa_ids"] = rosa_ids
    return batch_out

def _collate_fn_dynamic_padding(batch, pad_token_id):
    return _collate_training_batch(batch, pad_token_id)

def create_dataloader_for_jsonl(path, tokenizer, max_length, batch_size, pad_token_id,
                                num_workers=0, kayla_mode=False,
                                text_column: Optional[str] = None,
                                prompt_column: Optional[str] = None,
                                completion_column: Optional[str] = None,
                                use_length_bucketing=True, bucket_size=None,
                                device=None, prefetch_factor=None,
                                alpaca_mode: bool = False,
                                train_prompt_tokens: bool = True,
                                prompt_loss_weight: float = 1.0,
                                response_loss_weight: float = 1.0,
                                response_boundary_loss_weight: float = 1.0,
                                response_boundary_tokens: int = 0,
                                min_response_tokens: int = 1,
                                drop_empty_completions: bool = True):
    dataset = StreamingJSONLDataset(
        path,
        tokenizer,
        max_length,
        kayla_mode=kayla_mode,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
        train_prompt_tokens=train_prompt_tokens,
        prompt_loss_weight=prompt_loss_weight,
        response_loss_weight=response_loss_weight,
        response_boundary_loss_weight=response_boundary_loss_weight,
        response_boundary_tokens=response_boundary_tokens,
        min_response_tokens=min_response_tokens,
        drop_empty_completions=drop_empty_completions,
        bucket_size=_streaming_bucket_size(batch_size, use_length_bucketing, bucket_size),
        batch_size=batch_size,
        shuffle_buckets=True,
    )
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

def create_dataloader_for_hf_streaming(hf_dataset, tokenizer, max_length, batch_size, pad_token_id,
                                       num_workers=0, kayla_mode=False,
                                       text_column=None, prompt_column=None, completion_column=None,
                                       use_length_bucketing=True, bucket_size=None,
                                       shuffle=True, shuffle_buffer_size=10000,
                                       device=None, prefetch_factor=None,
                                       alpaca_mode: bool = False,
                                       train_prompt_tokens: bool = True,
                                       prompt_loss_weight: float = 1.0,
                                       response_loss_weight: float = 1.0,
                                       response_boundary_loss_weight: float = 1.0,
                                       response_boundary_tokens: int = 0,
                                       min_response_tokens: int = 1,
                                       drop_empty_completions: bool = True):
    dataset = HuggingFaceStreamingDataset(
        hf_dataset,
        tokenizer,
        max_length,
        kayla_mode=kayla_mode,
        text_column=text_column,
        prompt_column=prompt_column,
        completion_column=completion_column,
        alpaca_mode=alpaca_mode,
        train_prompt_tokens=train_prompt_tokens,
        prompt_loss_weight=prompt_loss_weight,
        response_loss_weight=response_loss_weight,
        response_boundary_loss_weight=response_boundary_loss_weight,
        response_boundary_tokens=response_boundary_tokens,
        min_response_tokens=min_response_tokens,
        drop_empty_completions=drop_empty_completions,
        bucket_size=_streaming_bucket_size(batch_size, use_length_bucketing, bucket_size),
        batch_size=batch_size,
        shuffle=shuffle,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    pin_memory = _pin_memory_for_device(device)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

def create_map_style_dataloader(dataset, batch_size, pad_token_id, num_workers=0, shuffle=True,
                                use_length_bucketing=True, bucket_size=None,
                                device=None, prefetch_factor=None):
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    use_cuda = _pin_memory_for_device(device)
    drop_last = use_cuda and len(dataset) > batch_size
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and shuffle and batch_size > 1 else None
    if lengths:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            bucket_size=bucket_size,
        )
        return _create_dataloader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            prefetch_factor=prefetch_factor,
        )

    sampler = EpochShuffleSampler(dataset, shuffle=shuffle)
    return _create_dataloader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )
