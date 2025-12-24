import os
import json
import torch
import traceback
import functools
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset
from typing import Optional, List, Dict, Any

# Helper for AttrDict access
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage.
    """
    def __init__(self, path: str, max_length: int):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        self.max_length = max_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        num_workers = worker_info.num_workers if worker_info is not None else 1

        print(f"[Worker {worker_id}/{num_workers}] Opening dataset file: {self.path}")
        skipped_count = 0
        processed_count = 0

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
                        yield {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(obj["attention_mask"], dtype=torch.long)
                        }
                    except Exception:
                        skipped_count += 1
                        continue
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR: {e}")
            raise e
        print(f"[Worker {worker_id}] Finished. Processed: {processed_count}, Skipped: {skipped_count}")

def create_dataloader_for_chunked(path, max_length, batch_size, num_workers=0):
    dataset = IterableChunkedJSONLDataset(path, max_length=max_length)
    def collate_fn_simple(batch):
        if not batch: return None
        return {
            "input_ids": torch.stack([item['input_ids'] for item in batch]),
            "labels": torch.stack([item['labels'] for item in batch]),
            "attention_mask": torch.stack([item['attention_mask'] for item in batch])
        }
    pin_memory = torch.cuda.is_available() and num_workers > 0
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_simple,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=(num_workers > 0))

class PTChunkedDataset(Dataset):
    def __init__(self, directory_path: str, max_length: int):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = max_length
        self.chunk_pointers = []
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

    def __len__(self): return len(self.chunk_pointers)
    def __getitem__(self, idx):
        path, index = self.chunk_pointers[idx]
        if path != self.last_loaded_path:
            self.last_loaded_data = torch.load(path, map_location='cpu')
            self.last_loaded_path = path
        return self.last_loaded_data[index]

def create_dataloader_pt_chunked(directory_path, max_length, batch_size, num_workers=0):
    dataset = PTChunkedDataset(directory_path, max_length=max_length)
    def collate_fn_pt(batch):
        batch = [item for item in batch if item is not None]
        if not batch: return None
        return {
            "input_ids": torch.stack([item['input_ids'] for item in batch]),
            "labels": torch.stack([item['labels'] for item in batch]),
            "attention_mask": torch.stack([item['attention_mask'] for item in batch])
        }
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pt, shuffle=True,
                      num_workers=num_workers, pin_memory=(torch.cuda.is_available() and num_workers > 0),
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
                inp = str(text_dict.get('input', ""))
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n" if inp else f"### Instruction:\n{instruction}\n\n### Response:\n"
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
        with open(path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading JSONL"):
                processed = process_text_sample(tokenizer, json.loads(line), max_length, kayla_mode, prompt_column='instruction', completion_column='output')
                if processed: self.samples.append(processed)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

class HuggingFaceMapStyleDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length, kayla_mode=False, text_column=None, prompt_column=None, completion_column=None):
        super().__init__()
        self.hf_dataset, self.tokenizer, self.max_length, self.kayla_mode = hf_dataset, tokenizer, max_length, kayla_mode
        self.text_column, self.prompt_column, self.completion_column = text_column, prompt_column, completion_column
    def __len__(self): return len(self.hf_dataset)
    def __getitem__(self, idx):
        return process_text_sample(self.tokenizer, self.hf_dataset[idx], self.max_length, self.kayla_mode, self.text_column, self.prompt_column, self.completion_column)

def _collate_fn_dynamic_padding(batch, pad_token_id):
    batch = [i for i in batch if i is not None]
    if not batch: return None
    ml = max(len(i['input_ids']) for i in batch)
    ids = torch.full((len(batch), ml), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), ml), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), ml), dtype=torch.long)
    for i, item in enumerate(batch):
        sl = len(item['input_ids'])
        ids[i, :sl], labels[i, :sl], mask[i, :sl] = item['input_ids'], item['labels'], 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}

def create_map_style_dataloader(dataset, batch_size, pad_token_id, num_workers=0, shuffle=True):
    collate = functools.partial(_collate_fn_dynamic_padding, pad_token_id=pad_token_id)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(torch.cuda.is_available() and num_workers > 0),
                      persistent_workers=(num_workers > 0))
