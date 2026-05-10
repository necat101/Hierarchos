import os
import json
import torch
import traceback
import functools
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
                 seed: Optional[int] = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.lengths = [max(1, int(length)) for length in lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
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

        if self.shuffle and len(self.lengths) > 1:
            indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            indices = list(range(len(self.lengths)))

        batches = []
        for bucket_start in range(0, len(indices), self.bucket_size):
            bucket = indices[bucket_start:bucket_start + self.bucket_size]
            if self.shuffle:
                bucket.sort(key=self.lengths.__getitem__, reverse=True)
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle and len(batches) > 1:
            order = torch.randperm(len(batches), generator=generator).tolist()
            for batch_idx in order:
                yield batches[batch_idx]
        else:
            for batch in batches:
                yield batch

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

def _as_long_tensor(value):
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.long)
    return torch.tensor(value, dtype=torch.long)

def _sample_effective_length(item, fallback_length: Optional[int] = None):
    if item is None:
        return fallback_length or 1

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

def _estimate_text_token_length(text_dict: dict, max_length: int, kayla_mode: bool,
                                text_column: Optional[str] = None,
                                prompt_column: Optional[str] = None,
                                completion_column: Optional[str] = None):
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

class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage.
    """
    def __init__(self, path: str, max_length: int, bucket_size: int = 0,
                 batch_size: int = 1, shuffle_buckets: bool = True):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        self.max_length = max_length
        self.bucket_size = max(0, int(bucket_size or 0))
        self.batch_size = max(1, int(batch_size))
        self.shuffle_buckets = bool(shuffle_buckets)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        print(f"[Worker {worker_id}/{num_workers}] Opening dataset file: {self.path}")
        skipped_count = 0
        processed_count = 0
        bucket = []
        generator = torch.Generator()
        generator.manual_seed((torch.initial_seed() + self.epoch + worker_id) % (2**63 - 1))

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f):
                    if line_num % num_workers != worker_id: continue
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
                        processed_count += 1
                        item = {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(obj["attention_mask"], dtype=torch.long)
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
                                  use_length_bucketing=True, bucket_size=None):
    streaming_bucket_size = int(bucket_size or (batch_size * 50)) if use_length_bucketing and batch_size > 1 else 0
    dataset = IterableChunkedJSONLDataset(
        path,
        max_length=max_length,
        bucket_size=streaming_bucket_size,
        batch_size=batch_size,
        shuffle_buckets=True,
    )
    collate_fn_simple = functools.partial(_collate_training_batch, pad_token_id=0)
    pin_memory = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_simple,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0))

class PTChunkedDataset(Dataset):
    def __init__(self, directory_path: str, max_length: int):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = max_length
        self.chunk_pointers = []
        self.sample_lengths = []
        self.last_loaded_path = None
        self.last_loaded_data = None
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
    def __getitem__(self, idx):
        path, index = self.chunk_pointers[idx]
        if path != self.last_loaded_path:
            self.last_loaded_data = torch.load(path, map_location='cpu')
            self.last_loaded_path = path
        return self.last_loaded_data[index]
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
                                 use_length_bucketing=True, bucket_size=None):
    dataset = PTChunkedDataset(directory_path, max_length=max_length)
    collate_fn_pt = functools.partial(_collate_training_batch, pad_token_id=0)
    use_cuda = torch.cuda.is_available()
    drop_last = False
    lengths = _get_dataset_sample_lengths(dataset) if use_length_bucketing and batch_size > 1 else None
    if lengths:
        batch_sampler = LengthGroupedBatchSampler(
            lengths,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            bucket_size=bucket_size,
        )
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn_pt,
                          num_workers=num_workers, pin_memory=use_cuda,
                          persistent_workers=(num_workers > 0))

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pt, shuffle=True,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=(num_workers > 0))

def process_text_sample(tokenizer, text_dict: dict, max_length: int, kayla_mode: bool = False,
                         text_column: Optional[str] = None,
                         prompt_column: Optional[str] = None, completion_column: Optional[str] = None):
    try:
        if text_column:
            text = str(text_dict.get(text_column, ""))
            if not text: return None
            ids = tokenizer.encode(text) + [tokenizer.eos_token_id]
            labels = list(ids)
        elif prompt_column and completion_column:
            instruction = str(text_dict.get(prompt_column, ""))
            output = str(text_dict.get(completion_column, ""))
            if kayla_mode:
                feelings = str(text_dict.get('feelings', ''))
                thought = str(text_dict.get('thought-process', ''))
                prompt_text = f"### Instruction:\n{instruction}\n\n" + (f"### Feelings:\n{feelings}\n\n" if feelings else "")
                thought_text = f"### Thought Process:\n{thought}\n\n"
                response_text = f"### Response:\n{output}"
                p_ids = tokenizer.encode(prompt_text)
                t_ids = tokenizer.encode(thought_text, add_special_tokens=False)
                r_ids = tokenizer.encode(response_text, add_special_tokens=False)
                ids = p_ids + t_ids + r_ids + [tokenizer.eos_token_id]
                labels = ([-100] * len(p_ids)) + t_ids + r_ids + [tokenizer.eos_token_id]
            else:
                inp = str(text_dict.get('input', "")).strip()
                if inp:
                    prompt = f"User: {inp}\n\nUser: {instruction}\n\nAssistant: "
                else:
                    prompt = f"User: {instruction}\n\nAssistant: "
                p_ids = tokenizer.encode(prompt)
                c_ids = tokenizer.encode(output, add_special_tokens=False)
                ids = p_ids + c_ids + [tokenizer.eos_token_id]
                labels = ([-100] * len(p_ids)) + c_ids + [tokenizer.eos_token_id]
        else: return None
        if len(ids) > max_length: ids, labels = ids[:max_length-1] + [tokenizer.eos_token_id], labels[:max_length-1] + [tokenizer.eos_token_id]
        return {"input_ids": torch.tensor(ids, dtype=torch.long), "labels": torch.tensor(labels, dtype=torch.long)}
    except: return None

class OriginalJSONLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, kayla_mode=False):
        super().__init__()
        self.tokenizer, self.max_length, self.kayla_mode, self.samples = tokenizer, max_length, kayla_mode, []
        self.sample_lengths = []
        skipped = 0
        with open(path, "r", encoding="utf-8") as f:
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
                # Pass kwargs explicitly to allow overriding in the CLI
                processed = process_text_sample(tokenizer, data, max_length, kayla_mode, prompt_column='instruction', completion_column='output')
                if processed:
                    self.samples.append(processed)
                    self.sample_lengths.append(len(processed["input_ids"]))
        if skipped:
            print(f"WARNING: Skipped {skipped} malformed JSONL lines out of {line_num} total.")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def get_sample_lengths(self): return self.sample_lengths

class HuggingFaceMapStyleDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False, text_column=None, prompt_column=None, completion_column=None):
        super().__init__()
        self.hf_dataset, self.tokenizer, self.max_length, self.kayla_mode = hf_dataset, tokenizer, max_length, kayla_mode
        self.text_column, self.prompt_column, self.completion_column = text_column, prompt_column, completion_column
        self.sample_lengths = None
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        return process_text_sample(self.tokenizer, self.hf_dataset[idx], self.max_length, self.kayla_mode, self.text_column, self.prompt_column, self.completion_column)
    def get_sample_lengths(self):
        if self.sample_lengths is not None:
            return self.sample_lengths
        lengths = []
        for idx in range(len(self.hf_dataset)):
            try:
                sample = self.hf_dataset[idx]
                processed = process_text_sample(
                    self.tokenizer,
                    sample,
                    self.max_length,
                    self.kayla_mode,
                    self.text_column,
                    self.prompt_column,
                    self.completion_column,
                )
                length = len(processed["input_ids"]) if processed is not None else _estimate_text_token_length(
                    sample, self.max_length, self.kayla_mode,
                    self.text_column, self.prompt_column, self.completion_column,
                )
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
    for i, item in enumerate(batch):
        item_ids = _as_long_tensor(item["input_ids"])
        item_labels = _as_long_tensor(item["labels"])
        item_mask = item.get("attention_mask")
        item_mask = _as_long_tensor(item_mask) if item_mask is not None else None

        sl = min(ml, item_ids.numel(), item_labels.numel())
        if item_mask is not None:
            sl = min(sl, item_mask.numel())
        ids[i, :sl] = item_ids[:sl]
        labels[i, :sl] = item_labels[:sl]
        mask[i, :sl] = item_mask[:sl] if item_mask is not None else 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}

def _collate_fn_dynamic_padding(batch, pad_token_id):
    return _collate_training_batch(batch, pad_token_id)

def create_map_style_dataloader(dataset, batch_size, pad_token_id, num_workers=0, shuffle=True,
                                use_length_bucketing=True, bucket_size=None):
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    use_cuda = torch.cuda.is_available()
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
        return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate,
                          num_workers=num_workers, pin_memory=use_cuda,
                          persistent_workers=(num_workers > 0))

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=use_cuda,
                      persistent_workers=(num_workers > 0),
                      drop_last=drop_last)
