import os
import sys
import json
import argparse
import time
import subprocess
import functools
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm
import traceback # Added for better error reporting
import signal # <<< MODIFIED: Import signal >>>
from torch.compiler import cudagraph_mark_step_begin # <<< ADD THIS LINE
import math # For ceil in dataset chunking (though that script is separate)

# <<< MODIFIED: Set Tokenizers Parallelism Environment Variable >>>
# Set this early, before tokenizers might be implicitly loaded by other imports
# Setting to "true" forces parallelism despite potential fork issues (use with caution)
# Setting to "false" explicitly disables parallelism in worker processes (safer, suppresses warning)
# Only run this environment setup logic if on Windows
if os.name == 'nt':
    CACHE_FILE = 'vcvars_path.cache.txt'
    vcvars_path = None

    # 1. Try to load the path from the cache file
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cached_path = f.read().strip()
            if os.path.exists(cached_path):
                print(f"INFO: Found cached vcvars64.bat path: {cached_path}")
                vcvars_path = cached_path
            else:
                print(f"INFO: Cached path '{cached_path}' no longer exists. Will try auto-detect.")
        except Exception as e:
            print(f"WARNING: Could not read cache file. Will try auto-detect. Error: {e}")

    # 2. If no valid path from cache, try auto-detection
    if vcvars_path is None:
        print("INFO: No cached vcvars path. Attempting auto-detection...")
        # Check standard paths for vswhere.exe
        vswhere_path_progfiles_x86 = os.path.expandvars(r"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe")
        vswhere_path_progfiles = os.path.expandvars(r"%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe")
        
        vswhere_path = None
        if os.path.exists(vswhere_path_progfiles_x86):
            vswhere_path = vswhere_path_progfiles_x86
        elif os.path.exists(vswhere_path_progfiles):
            vswhere_path = vswhere_path_progfiles
            
        if vswhere_path:
            try:
                # <<< MODIFICATION: Use vswhere to find vcvars64.bat directly >>>
                # This is more robust than finding the installationPath and guessing.
                # It asks for the path to 'vcvars64.bat' from the latest product
                # that *requires* the C++ tools.
                print("INFO: vswhere.exe found. Querying directly for vcvars64.bat...")
                cmd = [
                    vswhere_path,
                    "-latest",
                    "-products", "*",
                    "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                    "-find", r"VC\Auxiliary\Build\vcvars64.bat",
                    "-nologo"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8')
                found_path = result.stdout.strip()

                if found_path and os.path.exists(found_path):
                    vcvars_path = found_path
                    print(f"✅ INFO: vswhere.exe successfully found vcvars64.bat at: {vcvars_path}")
                else:
                    print(f"INFO: vswhere.exe ran but did not return a valid path.")
                    print(f"   (stdout: {result.stdout})")

            except Exception as e:
                print(f"WARNING: 'vswhere.exe' direct find failed. Will fall back to manual prompt. Error: {e}")
                # Note: stdout/stderr might be in e.args if check=True failed
                if hasattr(e, 'stdout'): print(f"   (stdout: {e.stdout})")
                if hasattr(e, 'stderr'): print(f"   (stderr: {e.stderr})")
        else:
            print(f"INFO: 'vswhere.exe' not found in standard locations. Will fall back to manual prompt.")

    # 3. If auto-detection failed, fall back to user prompt
    if vcvars_path is None:
        print("---" * 20)
        print("Microsoft Visual C++ (MSVC) 64-bit Compiler Setup (Manual Fallback)")
        print("---" * 20)
        print("Could not automatically find your 64-bit C++ compiler.")
        print("To use 'torch.compile' (e.g., with --gradient-checkpointing), this script")
        print("needs to find your 'vcvars64.bat' file.")
        print("\nIt is usually located in a path like:")
        print(r"  C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat")
        print(r"  C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat")
        print("---" * 20)
        
        try:
            user_path = input("Enter the full path to vcvars64.bat (or press Enter to skip): ").strip()

            if user_path and os.path.exists(user_path):
                vcvars_path = user_path
                print("INFO: Path validated.")
            elif user_path: # User provided a path, but it's invalid
                print(f"WARNING: Path '{user_path}' not found or is invalid.")
                print("         Skipping environment setup. torch.compile may fail.")
            else: # User pressed Enter
                print("INFO: Skipping 64-bit environment setup.")
                print("         'torch.compile' may fail unless you are in the correct dev terminal.")
        
        except EOFError:
             print("\nINFO: No input received. Skipping 64-bit environment setup.")
        
        print("---" * 20)

    # 4. If we have a valid path (from cache, auto-detect, or input), run it and set the environment
    if vcvars_path:
        # Cache the valid path for next time, regardless of how we found it
        try:
            with open(CACHE_FILE, 'w') as f:
                f.write(vcvars_path)
            print(f"INFO: 64-bit compiler path cached to '{CACHE_FILE}' for future runs.")
        except Exception as e:
            print(f"WARNING: Could not write cache file. You may be asked again. Error: {e}")

        print(f"INFO: Loading 64-bit MSVC environment from '{vcvars_path}'...")
        try:
            python_exe = sys.executable
            vcvars_dir = os.path.dirname(vcvars_path)
            vcvarsall_path = os.path.join(vcvars_dir, "vcvarsall.bat")

            if not os.path.exists(vcvarsall_path):
                print(f"❌ ERROR: 'vcvarsall.bat' not found in the same directory as 'vcvars64.bat'.")
                print(f"  Looked for: {vcvarsall_path}")
                raise FileNotFoundError("vcvarsall.bat not found")

            # This command runs vcvarsall.bat x64, which sets up the env,
            # then (&&) runs a python one-liner (using the *exact* python.exe path)
            # to print that new environment as a JSON string.
            # --- FIX: Add >NUL 2>&1 to redirect stdout and stderr ---
            cmd = f'"{vcvarsall_path}" x64 >NUL 2>&1 && echo ^"---ENV-JSON-START---^" && "{python_exe}" -c "import json, os; print(json.dumps(dict(os.environ)))"'
            
            # Hide the command window
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,
                check=True, # This will raise CalledProcessError if it fails
                startupinfo=startupinfo,
                encoding='utf-8',
                errors='ignore',
                cwd=vcvars_dir # Set CWD just in case
            )
            
            json_start = result.stdout.find("---ENV-JSON-START---")
            if json_start != -1:
                # --- FIX: More robust JSON finding to skip control characters ---
                raw_output_after_marker = result.stdout[json_start + len("---ENV-JSON-START---"):]
                
                # Find the first '{'
                json_obj_start = raw_output_after_marker.find('{')
                # Find the last '}'
                json_obj_end = raw_output_after_marker.rfind('}')
                
                if json_obj_start != -1 and json_obj_end != -1 and json_obj_end > json_obj_start:
                    json_str = raw_output_after_marker[json_obj_start : json_obj_end + 1]
                    
                    # Now try to parse this cleaned string
                    try:
                        env_vars = json.loads(json_str)
                        
                        # ==========================================================
                        # --- ❗️❗️❗️ BEGIN KEY FIX ❗️❗️❗️ ---
                        # ==========================================================
                        # We must use env_vars.get() to retrieve the new values
                        # and only assign them to os.environ if they are not None.
                        # This avoids the KeyError: 'LIB' when os.environ['LIB']
                        # is used as a fallback but doesn't exist.

                        new_path = env_vars.get('PATH')
                        new_lib = env_vars.get('LIB')
                        new_include = env_vars.get('INCLUDE')
                        
                        if new_path is not None:
                            os.environ['PATH'] = new_path
                        else:
                            print("WARNING: 'PATH' not found in vcvarsall.bat output.")
                            
                        if new_lib is not None:
                            os.environ['LIB'] = new_lib
                        else:
                            # This was the variable causing the KeyError.
                            # Set it to an empty string if not provided by vcvarsall.
                            print("INFO: 'LIB' not found in vcvarsall.bat output. Setting to empty string.")
                            os.environ['LIB'] = "" 
                            
                        if new_include is not None:
                            os.environ['INCLUDE'] = new_include
                        else:
                            # Also set INCLUDE to empty string just in case.
                            print("INFO: 'INCLUDE' not found in vcvarsall.bat output. Setting to empty string.")
                            os.environ['INCLUDE'] = ""
                        
                        # ==========================================================
                        # --- ❗️❗️❗️ END KEY FIX ❗️❗️❗️ ---
                        # ==========================================================

                        print("INFO: 64-bit MSVC environment loaded successfully.")
                    except json.JSONDecodeError as json_e:
                        print(f"ERROR: Failed to parse the environment JSON. {json_e}")
                        print("     --- Raw JSON String (cleaned) ---")
                        print(json_str[:500] + "...") # Print start of what was parsed
                        print("     --- Full stdout after marker ---")
                        print(raw_output_after_marker[:500] + "...")
                else:
                    print(f"WARNING: Could not find valid JSON object markers '{{' and '}}' after '---ENV-JSON-START---'.")
                    print("STDOUT after marker:", raw_output_after_marker)
            else:
                print(f"WARNING: Could not find JSON marker in vcvarsall.bat output. Compiler might still fail.")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)

        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to run vcvarsall.bat. torch.compile may fail.")
            print(f"   Return Code: {e.returncode}")
            print(f"   --- STDOUT ---")
            print(e.stdout if e.stdout else "<No stdout>")
            print(f"   --- STDERR ---")
            print(e.stderr if e.stderr else "<No stderr>")
            print(f"   --- End Output ---")
            
            # If the cached path failed, delete it so we ask again next time
            if os.path.exists(CACHE_FILE):
                print("INFO: Deleting bad cached path...")
                try:
                    os.remove(CACHE_FILE)
                except Exception as del_e:
                    print(f"WARNING: Could not delete cache file: {del_e}")
                    
        except Exception as e:
            # Catch other errors like file-not-found for subprocess itself or JSONDecodeError
            print(f"ERROR: An unexpected error occurred while trying to run vcvarsall.bat.")
            print(f"   Error: {e}")
            print(f"   Error Type: {type(e)}") # Added type for debugging
            traceback.print_exc(limit=2) # Print traceback for this
        print("---" * 20)

os.environ["TOKENIZERS_PARALLELISM"] = "true" 

import torch
import torch.nn as nn
import torch.nn.functional as F
# <<< MODIFIED: Import IterableDataset and Dataset >>>
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.serialization import safe_globals # May not be needed if not using legacy save/load
# <<< NEW: Import gradient checkpointing >>>
from torch.utils.checkpoint import checkpoint
# <<< END >>>


# <<< NEW: Import Hugging Face datasets library >>>
try:
    from datasets import load_dataset
    _HAS_HF_DATASETS = True
except ImportError:
    print("Warning: 'datasets' library not found. Loading datasets from Hugging Face Hub or local files (e.g., CSV) using --hf_dataset will be unavailable.")
    print("         Please install it: pip install datasets")
    _HAS_HF_DATASETS = False


# --- Optional Imports ---
try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    _HAS_PEFT = True
except ImportError:
    print("Warning: 'peft' library not found. LoRA fine-tuning and merging will be unavailable.")
    _HAS_PEFT = False

# <<< MODIFIED: Removed keyboard import >>>

# --- Optimizer Selection (Handles CUDA, bitsandbytes, and CPU fallback) ---
_HAS_BNB = False # Default assumption
_HAS_AMP = False # Default assumption
_torch_amp_available = False # Track if torch.amp or torch.cuda.amp was found

# --- Try importing autocast and GradScaler first (unconditionally) ---
try:
    # Prefer torch.amp (newer PyTorch versions)
    from torch.amp import GradScaler, autocast
    print("INFO: torch.amp (autocast/GradScaler) found.")
    _torch_amp_available = True
except ImportError:
    try:
        # Fallback for older PyTorch versions (requires CUDA context, but import might work)
        from torch.cuda.amp import GradScaler, autocast
        print("INFO: torch.cuda.amp (autocast/GradScaler) found.")
        _torch_amp_available = True
    except ImportError:
        print("Warning: torch.amp and torch.cuda.amp not found.")
        _torch_amp_available = False
        # --- Define dummy autocast if import failed ---
        import contextlib
        @contextlib.contextmanager
        def autocast(device_type, enabled=True, dtype=None): # Match signature
            # print("Warning: Using dummy autocast context manager.") # Optional: uncomment for debugging
            yield
        # GradScaler is only needed if _HAS_AMP is True later, so no dummy needed now

# --- Now, determine optimizer and final AMP status based on CUDA ---
if torch.cuda.is_available():
    # Attempt to use bitsandbytes 8-bit AdamW if available
    try:
        import bitsandbytes as bnb
        ADAM_OPTIMIZER = bnb.optim.AdamW8bit
        _HAS_BNB = True
        print("INFO: CUDA detected and bitsandbytes found. Using bitsandbytes 8-bit AdamW.")
    except ImportError:
        print("Warning: bitsandbytes not found.")
        print("INFO: Falling back to standard torch.optim.AdamW optimizer.")
        ADAM_OPTIMIZER = torch.optim.AdamW
        _HAS_BNB = False

    # Check if the actual AMP components were successfully imported earlier
    if _torch_amp_available:
        _HAS_AMP = True
        print("INFO: AMP support is enabled (CUDA available).")
    else:
        _HAS_AMP = False
        print("Warning: AMP support is disabled (torch.amp/torch.cuda.amp import failed).")

else: # No CUDA detected
    print("Warning: CUDA not detected. Using CPU training.")
    print("INFO: Falling back to standard torch.optim.AdamW optimizer (bitsandbytes requires CUDA).")
    ADAM_OPTIMIZER = torch.optim.AdamW
    _HAS_BNB = False
    _HAS_AMP = False # AMP not usable on CPU
    if _torch_amp_available:
        # This case means torch.amp was importable but we are on CPU
        print("INFO: AMP components found but disabled (running on CPU).")
    # If _torch_amp_available is False, the warning about dummy autocast will show if used


# --- END Optimizer Selection ---


# --- C++ Kernel Import ---
try:
    import hierarchos_matmul
    _HAS_KERNEL = True
    print("Successfully imported C++ quantization kernel.")
    if hasattr(hierarchos_matmul, "VULKAN_SUPPORT") and hierarchos_matmul.VULKAN_SUPPORT:
        print("INFO: Vulkan support is enabled in the compiled kernel.")
        _HAS_VULKAN = True
    else:
        print("INFO: Vulkan support is disabled in the compiled kernel.")
        _HAS_VULKAN = False
except ImportError:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!! WARNING: The compiled C++ kernel 'hierarchos_matmul' was not found.       !!!")
    print("!!!          Quantization and quantized inference will be unavailable.     !!!")
    print("!!!                                                                       !!!")
    print("!!! To enable these features, please run the appropriate setup script:    !!!")
    print("!!!  - On Windows:   Run setup.bat                                        !!!")
    print("!!!  - On Linux/macOS: Run bash setup.sh                                    !!!")
    print("!!!                                                                       !!!")
    print("!!! Make sure you have CMake and a C++ compiler installed.                 !!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    _HAS_KERNEL = False
    _HAS_VULKAN = False


# --- CONSTANTS ---
MODEL_WEIGHTS_NAME = "hierarchos.pt"
QUANTIZED_MODEL_WEIGHTS_NAME_TPL = "hierarchos-{qtype}.npz" # Template for the name


# --- HELPER CLASS: AttrDict ---
class AttrDict(dict):
    """A dictionary that allows for attribute-style access."""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# --- Device Utilities ---
def pick_device():
    """Picks the best available device for PyTorch training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Commenting out MPS check for stability, can be re-enabled if needed
    # if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    #      return torch.device("mps")
    return torch.device("cpu")

def set_threads(n: int):
    try:
        torch.set_num_threads(n)
        os.environ['OMP_NUM_THREADS'] = str(n)
    except Exception as e:
        print(f"Warning: Could not set thread count. {e}")


# --- Dataset & Dataloader ---

# <<< START: New Iterable Dataset for Pre-Chunked JSONL Data >>>
class IterableChunkedJSONLDataset(IterableDataset):
    """
    An IterableDataset for loading pre-tokenized, chunked, masked, and padded
    data from a JSONL file line by line. Reduces RAM usage compared to loading
    the entire dataset into memory.

    Expects each line to be a JSON object containing 'input_ids', 'labels',
    and 'attention_mask' as lists of integers, all of the *same* pre-defined length.
    """
    def __init__(self, path: str, max_length: int):
        super().__init__()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")
        self.path = path
        self.max_length = max_length # Used for validation

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
                    # Distribute lines across workers
                    if line_num % num_workers != worker_id:
                        continue

                    line = line.strip()
                    if not line: continue

                    try:
                        obj = json.loads(line)
                        # --- Basic Validation ---
                        if not all(k in obj for k in ["input_ids", "labels", "attention_mask"]):
                            # Print less frequently in multi-worker scenarios to avoid spam
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Missing required keys.")
                            skipped_count += 1
                            continue
                        if not all(isinstance(obj[k], list) for k in ["input_ids", "labels", "attention_mask"]):
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Required keys are not lists.")
                            skipped_count += 1
                            continue

                        # --- Length Validation ---
                        seq_len = len(obj["input_ids"])
                        if seq_len != self.max_length:
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Expected length {self.max_length}, found {seq_len}.")
                            skipped_count += 1
                            continue
                        elif len(obj["labels"]) != seq_len or len(obj["attention_mask"]) != seq_len:
                            if skipped_count % (100 * num_workers) == 0:
                                print(f"[Worker {worker_id}] Warning: Skipping line ~{line_num+1}. Length mismatch between input_ids, labels, and attention_mask.")
                            skipped_count += 1
                            continue

                        # Convert lists to tensors just before yielding
                        processed_count += 1
                        yield {
                            "input_ids": torch.tensor(obj["input_ids"], dtype=torch.long),
                            "labels": torch.tensor(obj["labels"], dtype=torch.long),
                            "attention_mask": torch.tensor(obj["attention_mask"], dtype=torch.long)
                        }

                    except json.JSONDecodeError:
                        if skipped_count % (100 * num_workers) == 0:
                            print(f"[Worker {worker_id}] Warning: Skipping invalid JSON on line ~{line_num+1}: {line[:100]}...")
                        skipped_count += 1
                        continue
                    except Exception as e:
                        if skipped_count % (100 * num_workers) == 0:
                            print(f"[Worker {worker_id}] Warning: Error processing line ~{line_num+1}: {e}. Line: {line[:100]}...")
                        skipped_count += 1
                        continue
        except Exception as e:
            print(f"[Worker {worker_id}] ERROR during dataset iteration: {e}")
            traceback.print_exc() # Print full traceback if file reading fails etc.
            raise e # Re-raise the exception

        print(f"[Worker {worker_id}] Finished iterating. Processed: {processed_count}, Skipped: {skipped_count}")


def create_dataloader_for_chunked(path, max_length, batch_size, num_workers=0):
    """
    Creates a DataLoader specifically for the pre-chunked JSONL dataset using
    an IterableDataset to save RAM. Padding is assumed handled by the chunking script.
    """
    # Use the IterableDataset
    dataset = IterableChunkedJSONLDataset(path, max_length=max_length)

    def collate_fn_simple(batch):
        # Batch items are dictionaries with tensors of the same length
        if not batch: return None

        # Check if items are already dictionaries (expected from IterableDataset)
        if not isinstance(batch[0], dict):
            print(f"Warning: Unexpected item type in collate_fn_simple: {type(batch[0])}. Expected dict.")
            # Attempt to handle if it's a list/tuple, otherwise raise error
            if isinstance(batch[0], (list, tuple)) and len(batch[0]) == 3: # Assuming order: ids, labels, mask
                input_ids_batch = torch.stack([item[0] for item in batch])
                labels_batch = torch.stack([item[1] for item in batch])
                attention_mask_batch = torch.stack([item[2] for item in batch])
            else:
                raise TypeError(f"Collate function received unexpected data structure: {type(batch[0])}")
        else:
            input_ids_batch = torch.stack([item['input_ids'] for item in batch])
            labels_batch = torch.stack([item['labels'] for item in batch])
            attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])

        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    pin_memory = torch.cuda.is_available() and num_workers > 0
    # persistent_workers is generally recommended with num_workers > 0
    # It avoids worker startup overhead for each epoch.
    persistent_workers = num_workers > 0

    # <<< MODIFIED: Removed shuffle=True, as it's not applicable here >>>
    # IterableDatasets handle shuffling differently (often requiring buffering)
    # or rely on the inherent order/distribution for large datasets.
    # Simple line-by-line iteration per worker is used here.
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_simple,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers)
# <<< END: Iterable Dataset/Loader for Pre-Chunked JSONL Data >>>

# <<< START: Modified Map-Style Dataset/Loader for Consolidated PT Tensors >>>
class PTChunkedDataset(Dataset):
    """
    A map-style Dataset for loading pre-tokenized, chunked, masked, and padded
    data directly from individual chunk entries listed in a manifest.jsonl,
    where multiple chunks are consolidated into single .pt files.
    Reduces RAM usage during startup compared to loading all text.

    Expects a directory containing consolidated .pt files (each a list of dicts)
    and a manifest.jsonl. Each line in manifest.jsonl should contain
    'file_path' (relative) and 'index_in_file' (int).
    Each chunk dict within the .pt files should have 'input_ids', 'labels',
    and 'attention_mask' as torch.tensors of the *same* pre-defined length.
    Implements caching to reduce redundant file reads.
    """
    def __init__(self, directory_path: str, max_length: int):
        super().__init__()
        self.directory_path = directory_path
        self.max_length = max_length # For potential validation
        # <<< MODIFIED: Store (filepath, index) tuples >>>
        self.chunk_pointers = []
        # <<< MODIFIED: Add caching attributes >>>
        self.last_loaded_path = None
        self.last_loaded_data = None # This will hold the list loaded from a .pt file

        manifest_file = os.path.join(directory_path, "manifest.jsonl")
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest file not found: {manifest_file}")

        print(f"Loading chunk pointers from manifest: {manifest_file}")
        try:
            with open(manifest_file, "r", encoding="utf-8") as f_manifest:
                for line_num, line in enumerate(f_manifest): # Added line_num for better warnings
                    line = line.strip()
                    if not line: continue
                    try:
                        entry = json.loads(line)
                        relative_path = entry.get("file_path")
                        # <<< MODIFIED: Get index_in_file >>>
                        index_in_file = entry.get("index_in_file")

                        # <<< MODIFIED: Validate entry >>>
                        if relative_path and isinstance(relative_path, str) and \
                           index_in_file is not None and isinstance(index_in_file, int):
                            full_path = os.path.join(self.directory_path, relative_path)
                            self.chunk_pointers.append((full_path, index_in_file))
                        else:
                            print(f"Warning: Manifest line ~{line_num+1} missing or invalid 'file_path' (str) or 'index_in_file' (int): {line}")
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON in manifest line ~{line_num+1}: {line[:100]}...")
            if not self.chunk_pointers:
                raise ValueError(f"No valid chunk pointers found in manifest: {manifest_file}")
            print(f"Found {len(self.chunk_pointers)} total logical chunks.")
        except Exception as e:
            print(f"Error reading manifest file {manifest_file}: {e}")
            raise e

    def __len__(self):
        # <<< MODIFIED: Length is the total number of pointers >>>
        return len(self.chunk_pointers)

    def __getitem__(self, idx):
        # <<< MODIFIED: Retrieve path and index >>>
        try:
            chunk_path, index_in_file = self.chunk_pointers[idx]
        except IndexError:
            # This shouldn't happen with standard DataLoader usage but is a safeguard
            print(f"Error: Index {idx} out of bounds for chunk pointers (len: {len(self.chunk_pointers)}).")
            return None # Indicate failure

        try:
            # <<< MODIFIED: Implement Caching >>>
            if chunk_path == self.last_loaded_path:
                # Cache Hit: Use the already loaded list
                if self.last_loaded_data is None:
                    # Should not happen if last_loaded_path is set, but handle defensively
                    print(f"Warning: Cache inconsistency for file {chunk_path}. Reloading.")
                    self.last_loaded_data = torch.load(chunk_path, map_location='cpu')
                    if not isinstance(self.last_loaded_data, list):
                            raise TypeError(f"Loaded data from {chunk_path} is not a list.")
                # Retrieve the specific chunk dictionary from the cached list
                data = self.last_loaded_data[index_in_file]

            else:
                # Cache Miss: Load the new consolidated file
                # print(f"Cache miss. Loading file: {chunk_path}") # Optional: for debugging
                loaded_list = torch.load(chunk_path, map_location='cpu') # Load to CPU initially
                if not isinstance(loaded_list, list):
                    raise TypeError(f"Loaded data from {chunk_path} is not a list.")

                # Update cache
                self.last_loaded_path = chunk_path
                self.last_loaded_data = loaded_list

                # Retrieve the specific chunk dictionary
                data = self.last_loaded_data[index_in_file]

            # --- Validation (Optional but recommended) ---
            if not isinstance(data, dict):
                raise TypeError(f"Chunk at index {index_in_file} in {chunk_path} is not a dictionary.")
            if not all(k in data for k in ["input_ids", "labels", "attention_mask"]):
                print(f"Warning: Chunk dict at index {index_in_file} in {chunk_path} missing required keys. Skipping.")
                return None # Indicate failure
            if not all(isinstance(data[k], torch.Tensor) for k in ["input_ids", "labels", "attention_mask"]):
                print(f"Warning: Data in chunk dict at index {index_in_file} in {chunk_path} are not tensors. Skipping.")
                return None
            if data["input_ids"].shape[0] != self.max_length:
                print(f"Warning: Chunk tensor 'input_ids' at index {index_in_file} in {chunk_path} has unexpected length {data['input_ids'].shape[0]}. Expected {self.max_length}. Skipping.")
                return None
            # --- End Validation ---

            return data

        except FileNotFoundError:
             print(f"Error: Consolidated chunk file not found: {chunk_path}")
             self.last_loaded_path = None # Invalidate cache if file not found
             self.last_loaded_data = None
             return None
        except IndexError:
             print(f"Error: index_in_file {index_in_file} out of bounds for loaded list from {chunk_path} (len: {len(self.last_loaded_data) if self.last_loaded_data else 'N/A'}). Check manifest/chunking script.")
             # Consider invalidating cache here too if the file structure seems wrong
             # self.last_loaded_path = None
             # self.last_loaded_data = None
             return None
        except TypeError as e:
             print(f"Error: Type error processing chunk at index {index_in_file} in {chunk_path}: {e}")
             return None
        except Exception as e:
             # Catch other potential errors during loading or processing
             print(f"Error loading or processing chunk from {chunk_path} at index {index_in_file}: {e}")
             # Optionally invalidate cache on unexpected errors
             # self.last_loaded_path = None
             # self.last_loaded_data = None
             return None


def create_dataloader_pt_chunked(directory_path, max_length, batch_size, num_workers=0):
    """
    Creates a DataLoader for the pre-chunked consolidated .pt dataset using PTChunkedDataset.
    Handles shuffling and batching of pre-loaded tensors. Caching is handled within the Dataset.
    """
    dataset = PTChunkedDataset(directory_path, max_length=max_length) # Uses the MODIFIED dataset

    def collate_fn_pt(batch):
        # Filter out None items potentially returned by dataset __getitem__ on error
        batch = [item for item in batch if item is not None]
        if not batch: return None # Return None if batch becomes empty after filtering

        # Items are dictionaries with tensors of the same length
        try:
            input_ids_batch = torch.stack([item['input_ids'] for item in batch])
            labels_batch = torch.stack([item['labels'] for item in batch])
            attention_mask_batch = torch.stack([item['attention_mask'] for item in batch])
        except Exception as e:
            print(f"Error during collate_fn_pt: {e}. One of the items might be malformed.")
            # Decide how to handle this - skip batch or raise error?
            # Returning None might be safer if errors are expected, but hides issues.
            # Raising the error stops training but makes the problem explicit.
            # Let's return None for now to avoid crashing training on rare errors.
            return None


        return {
            "input_ids": input_ids_batch,
            "labels": labels_batch,
            "attention_mask": attention_mask_batch
        }

    pin_memory = torch.cuda.is_available() and num_workers > 0
    persistent_workers = num_workers > 0

    # Map-style dataset allows shuffling
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn_pt, shuffle=True,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=persistent_workers)

# <<< END: Modified Map-Style Dataset/Loader for Consolidated PT Tensors >>>


# <<< START: Utility function for processing text based on args (shared by JSONL and HF dataset) >>>
def process_text_sample(tokenizer, text_dict: dict, max_length: int, kayla_mode: bool = False,
                         text_column: Optional[str] = None,
                         prompt_column: Optional[str] = None, completion_column: Optional[str] = None):
    """
    Processes a dictionary containing text into tokenized input_ids and labels
    based on the specified mode (Kayla, instruction tuning, or text completion).
    Returns a dictionary {"input_ids": tensor, "labels": tensor} or None on error.
    """
    input_ids = []
    labels = []

    try:
        if text_column:
            # --- Text Completion Mode (e.g., pre-training) ---
            text = text_dict.get(text_column, "")
            if not isinstance(text, str): text = str(text) # Attempt conversion
            if not text: return None # Skip empty text

            input_ids = tokenizer.encode(text, add_special_tokens=True) + [tokenizer.eos_token_id]
            labels = list(input_ids) # Predict every token

        elif prompt_column and completion_column:
            # --- Instruction Tuning Mode (Standard or Kayla) ---
            if kayla_mode:
                # --- Kayla Format ---
                instruction = text_dict.get(prompt_column, "") # Use prompt_column for instruction
                completion = text_dict.get(completion_column, "") # Use completion_column for output
                # Assumes 'feelings' and 'thought-process' are also present in text_dict if needed
                feelings = text_dict.get('feelings', '')
                thought = text_dict.get('thought-process', '')

                if not isinstance(instruction, str): instruction = str(instruction)
                if not isinstance(completion, str): completion = str(completion)
                if not isinstance(feelings, str): feelings = str(feelings)
                if not isinstance(thought, str): thought = str(thought)


                instruction_text = f"### Instruction:\n{instruction}\n\n"
                feelings_text = f"### Feelings:\n{feelings}\n\n" if feelings else ""
                prompt_context_text = instruction_text + feelings_text
                thought_text = f"### Th
