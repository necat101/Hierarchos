-----

# Hierarchos v0.10.0 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

Due to Amazon's "Chronos" forecasting models (still based on transformers BTW) I've decided to rename the project to "Hierarchos" from this point forward. This should prevent any naming confusion that may occur.

-----

### 🚀 **New in v0.10.0: The "Free Will" Update (Hinge Loss & Stability)**

> This update addresses the "Posterior Collapse" phenomenon where the Worker module would become "lazy" and ignore its internal context.
>
> 1.  **Hinge Loss (Free Bit Budget):** 🔓 To fix posterior collapse, we've implemented a Hinge Loss mechanism. The Worker is now given a "free budget" of drift (defined by `--commitment-threshold`). As long as the Worker's modifications to the plan stay within this threshold (e.g., for syntax or immediate context), **no penalty is applied**. This encourages the model to actually use its hierarchical features rather than collapsing into a standard RNN.
> 2.  **Commitment Control:** 🔗 If the Worker drifts *beyond* the threshold, a penalty is applied. This prevents the "Rogue Worker" problem, ensuring the Low-Level module doesn't stray so far that it ignores the High-Level Manager's plan.
> 3.  **Robust Training Loop:** 🛡️ Fixed a critical "Last Batch Mismatch" bug that caused training crashes on datasets that didn't divide evenly by the batch size. The model now dynamically slices internal states to handle variable batch sizes at the end of epochs.

### 🚀 **Major Paradigm Shift (v0.9.0): The RWKV Backbone**

> 1.  **RWKV Architecture:** 🧠 Hierarchos leverages Linear Transformers within an RNN framework, allowing parallelizable training and constant inference memory.
> 2.  **State-Invariant Processing:** 🚫 Explicit Positional Embeddings (`pos_emb`) have been removed. The model handles temporal relationships via RWKV's internal state decay (`time_decay`), improving generalization to infinite sequence lengths.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Hierarchos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are structured with timestamps and source metadata.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine powered by **RWKV Recurrence**.

  * **The Manager (H-RNN):** Sets the high-level plan (Static Context).
  * **The Worker (L-RNN):** Executes the plan. It operates under a "Commitment Contract"—it can drift from the plan to handle syntax (Hinge Loss), but is penalized if it drifts too far (Commitment Loss).
  * **Context Drift:** The projection layer allows the Worker to evolve the context vector dynamically during generation.

## Features ✨

  * 🧠 **RWKV-Based Recurrence**: Replaces legacy GRU cells with RWKV for linear scaling.
  * ⚖️ **Hinge Loss Regularization**: Solves posterior collapse by allowing "free" state manipulation within a threshold.
  * 🌐 **Hugging Face `datasets` Integration**: Stream datasets directly from the HF Hub (e.g., Wikitext, C4, Alpaca).
  * 🌊 **Sliding Context Mechanism**: New projection layer allowing the Worker to evolve the context vector dynamically.
  * 🛡️ **Negative Reinforcement Learning**: In-chat capability to penalize specific memory recalls via gradient inversion.
  * 🔥 **PyTorch 2.0+ Compiled Training**: Automatically uses `torch.compile` for massive speedups on NVIDIA GPUs.
  * 💾 **Optimized Consolidated Chunk Loading**: Support for instant loading of massive pre-tokenized datasets via `.pt` files.
  * 📉 **Gradient Checkpointing**: Reduce VRAM usage during training (`--gradient-checkpointing`).
  * 🕰️ **Structured & Queryable Memory**: LTM slots include timestamps and source IDs, allowing temporal filtering.
  * ⚡ **High-Performance Quantized Inference**: Custom C++ kernel (AVX2/AVX512/NEON) for `INT4`, `Q4_0`, `Q8_0`, and `Q2_K`.
  * 🎮 **Vulkan Acceleration**: Optional GPU backend for quantized inference via `setup.bat --vulkan` / `setup.sh --vulkan`.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * Python 3.8+
  * **PyTorch 2.0+ (Required for `torch.compile`)**
  * `pip install datasets`
  * **Optional:** C++ compiler (MSVC/GCC/Clang), CMake, Vulkan SDK.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/necat101/Hierarchos.git
    cd Hierarchos
    ```

2.  **Run the Setup Script:**
    This script builds the C++ kernel required for quantization and optimized inference.

      * **Default (CPU):** `setup.bat` (Windows) or `bash setup.sh` (Linux/macOS)
      * **Vulkan (GPU):** `setup.bat --vulkan` or `bash setup.sh --vulkan`

-----

## 📚 User Guide: Comprehensive Workflows

### Workflow 1: Training a New Model

**Note:** You no longer need to specify `h_hidden` or `l_hidden`. Setting `--context_dim` automatically scales the entire architecture.

#### **(A) Hugging Face Datasets (Recommended)**

**Example: Pre-training on Wikitext-103**

```bash
python hierarchos.py train \
    --hf_dataset "wikitext" \
    --hf_dataset_config "wikitext-103-raw-v1" \
    --hf_dataset_split "train" \
    --text_column "text" \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./hierarchos_wikitext" \
    --auto-max-length \
    --context_dim 768 \
    --batch_size 4 \
    --gradient-checkpointing \
    --amp
```

#### **(B) Local JSON/JSONL File**

```bash
python hierarchos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_model" \
    --epochs 3 \
    --batch_size 4 \
    --accumulation-steps 2 \
    --context_dim 768 \
    --max_h_steps 5 \
    --max_l_steps 5 \
    --auto-max-length \
    --amp
```

#### **(C) Pre-Chunked Local Dataset (Massive Scale)**

For datasets larger than RAM, use the chunking tool first.

1.  **Create Chunks:**
    ```bash
    python dataset_chunk_create.py \
        --dataset "path/to/huge_corpus.jsonl" \
        --tokenizer-path "openai-community/gpt2" \
        --output-dir "./chunked_data" \
        --chunks-per-file 1000
    ```
2.  **Train:**
    ```bash
    python hierarchos.py train \
        --pre_pt_dataset \
        --train "./chunked_data" \
        --max_length 3153 `# Must match output of chunk script` \
        --tokenizer-path "openai-community/gpt2" \
        --out-dir "./my_large_model"
    ```

### Workflow 2: Running Chat with Online Learning

To enable learning while running a quantized model (INT4/Q4\_0), you must provide the path to the original full-precision weights (`--shadow-model-path`).

```bash
python hierarchos.py chat \
    --model-path "./my_model-Q4_0" \
    --shadow-model-path "./my_model" \
    --enable-quantized-learning \
    --device vulkan `# Optional`
```

#### **Chat Interaction Guide**

  * **Positive Feedback:** Type `"Good"`, `"Yes"`, or `"Correct"` to reinforce the *previous* interaction.
  * **Negative Reinforcement:** Type `"No"`, `"Bad"`, or `"Wrong"` to **penalize** the *previous* interaction via gradient inversion.
  * **Corrections:** Start a reply with `"No, actually..."` to trigger immediate learning of new facts.

### Workflow 3: Expanding a Model

Upgrade a trained model to a larger size. The script automatically handles the new layers and enforces size symmetry.

```bash
python expand_model.py \
    --old-model-path "./my_Hierarchos_model" \
    --output-dir "./expanded_model" \
    --context_dim 1024 `# Will automatically set h_hidden and l_hidden to 1024`
```

-----

## ⚙️ Command-Line Reference

### `hierarchos.py` Arguments

| Argument | Mode(s) | Description | Default |
| :--- | :--- | :--- | :--- |
| **Paths & Data** | | | |
| `--train` | `train`, `finetune` | Path to **local** data: JSON/JSONL file, or directory for `--pre_pt_dataset`. Use flag without path if using `--hf_dataset`. Mutually Exclusive with `--hf_dataset` path. | `None` |
| `--hf_dataset` | `train`, `finetune` | Name or path to a Hugging Face dataset (e.g., 'wikitext', 'c4'). Mutually Exclusive with `--train` path. | `None` |
| `--hf_dataset_config` | `train`, `finetune` | Optional configuration name for the HF dataset (e.g., 'wikitext-103-raw-v1'). | `None` |
| `--hf_dataset_split` | `train`, `finetune` | Dataset split to use (e.g., 'train', 'validation', 'train[:10%]'). | `train` |
| `--text_column` | `train`, `finetune` | Column name for text completion data in HF dataset. Defaults to 'text' if available. | `None` |
| `--prompt_column` | `train`, `finetune` | Column name for prompt/instruction in HF dataset. Use with `--completion_column`. | `None` |
| `--completion_column` | `train`, `finetune` | Column name for completion/response in HF dataset. Use with `--prompt_column`. | `None` |
| `--pre_chunked_dataset` | `train`, `finetune` | Load pre-chunked **JSONL** dataset iteratively (requires `--max_length`). | `False` |
| `--pre_pt_dataset` | `train`, `finetune` | Load pre-chunked **consolidated .pt tensor** dataset from directory specified in `--train` (requires `--max_length`). | `False` |
| `--model-path` | `train`, `finetune`, `merge`, `quantize`, `chat` | Path to model directory. [Train]: Loads weights only (fresh start). [Other]: Loads for specified mode. | `None` |
| `--out-dir` | `train`, `finetune`, `merge`, `quantize` | Directory to save new models, checkpoints, or adapters. | `./Hierarchos_model` |
| `--tokenizer-path` | `train`, `finetune`, `merge`, `quantize` | Path or HF name of tokenizer (if not loading from model-path). | `openai-community/gpt2` |
| `--resume-from-ckpt` | `train` | Path to `.pt` checkpoint to **resume full training state** (optimizer, etc.). | `None` |
| `--shadow-model-path` | `chat` | Path to full-precision model dir for online learning with quantized model. | `None` |
| `--lora-adapter-path` | `merge`, `finetune` | Path to the trained LoRA adapter directory. | `None` |
| **Training/Fine-Tuning** | | | |
| `--epochs` | `train`, `finetune` | Number of training epochs. | `3` |
| `--batch_size` | `train`, `finetune` | Number of samples per forward pass. | `4` |
| `--accumulation-steps` | `train`, `finetune` | Number of steps to accumulate gradients over (simulates larger batch size). | `1` |
| `--gradient-checkpointing` | `train`, `finetune` | Enable gradient checkpointing to save VRAM (trades compute for memory). | `False` |
| `--grad-clip` | `train`, `finetune` | Gradient clipping value. Prevents gradient explosion (0 to disable). | `1.0` |
| `--ponder-loss-weight` | `train`, `finetune` | Weight for the Ponder Cost auxiliary loss. | `0.01` |
| `--commitment-loss-weight` | `train`, `finetune` | **[NEW]** Weight for the Commitment auxiliary loss (drifting too far from plan). | `0.1` |
| `--commitment-threshold` | `train`, `finetune` | **[NEW]** Hinge loss threshold ("Free Bit Budget"). Drift below this value is ignored. | `0.05` |
| `--override-scheduling` | `train` | [If resuming] Ignore checkpoint's schedule state and use new LR args. | `False` |
| `--starting-lr` | `train`, `finetune` | Max Learning Rate for the schedule, or fixed LR if schedule disabled. | `1e-4` |
| `--min-lr` | `train`, `finetune` | Minimum Learning Rate for cosine annealing schedule. | `1e-6` |
| `--disable-lr-schedule` | `train`, `finetune` | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing. | `False` |
| `--ltm_lr` | `train`, `finetune`, `chat` | Learning Rate for LTM "surprise" updates (or max LR for LTM schedule in chat). | `0.01` |
| `--amp` | `train`, `finetune`, `chat` | Enable Automatic Mixed Precision (requires CUDA). | `False` |
| `--num_workers` | `train`, `finetune` | Number of CPU workers for data loading (and HF dataset mapping if applicable). | `0` |
| `--lora_r` | `finetune` | LoRA rank 'r'. | `8` |
| `--lora_alpha` | `finetune` | LoRA alpha scaling factor. | `16` |
| `--finetune-unlock-percent`| `finetune` | Target % of params to train (approx.). Overrides `--lora_r` if set. | `None` |
| `--kayla` | `train`, `finetune` | Enable Kayla-style instruction tuning format (with thought-process). Ignored if using pre-chunked formats. | `False` |
| **Quantization/Inference** | | | |
| `--qtype` | `quantize`, `train` | Quantization format (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). Used by `quantize` or `--quantize-on-complete`. | `INT4` |
| `--quantize-on-complete` | `train` | Automatically run quantization after training finishes. Requires compiled kernel. | `False` |
| `--device` | `chat` | Device for **quantized** inference (`cpu`, `vulkan`). Requires kernel compiled with `--vulkan` flag. | `cpu` |
| `--h-halt-thresh` | `chat` | Probability threshold for early exiting the HRM reasoning loop during inference. | `0.9` |
| `--max-new-tokens` | `chat` | Maximum number of tokens to generate in chat mode. | `512` |
| `--enable-quantized-learning`| `chat` | Enable LTM updates for quantized models (requires `--shadow-model-path`). | `False` |
| `--ltm-lora-path` | `chat` | Optional: Path to save/load LTM updates as a separate delta file in chat mode. | `None` |
| `--static-ltm-lr` | `chat` | Disable cosine annealing for chat LTM updates, use fixed `--ltm_lr`. | `False` |
| `--ltm-schedule-steps` | `chat` | Number of chat updates per LTM LR cosine cycle. | `100` |
| `--ltm-schedule-min-lr` | `chat` | Minimum LR for chat LTM cosine schedule. | `1e-5` |
| **Architecture (Train)** | | *(Used only if starting train from scratch)* | |
| `--context_dim` | `train` | Core embedding dimension. **Sets `h_hidden` and `l_hidden` automatically.** | `768` |
| `--persistent_dim` | `train` | Dimension of the fixed Persistent Memory. | `128` |
| `--ltm_slots` | `train` | Number of slots in the Long-Term Memory. | `1024` |
| `--ltm_key_dim` | `train` | Dimension of LTM keys. | `128` |
| `--ltm_val_dim` | `train` | Dimension of LTM values. | `128` |
| `--max_h_steps` | `train` | Maximum number of reasoning steps H-module can take. Impacts training speed. | `5` |
| `--max_l_steps` | `train` | Maximum number of iterations for L-module convergence per H-step. Impacts training speed. | `5` |
| `--l_conv_atol` | `train` | Absolute tolerance for checking L-module state convergence. | `1e-4` |
| `--ltm_topk` | `train` | Number of LTM slots to retrieve per token. | `2` |
| `--max_length` | `train`, `finetune` | Maximum sequence length. Required if using pre-chunked formats. | `1024` |
| `--auto-max-length` | `train`, `finetune` | Automatically scan dataset (`--train` or `--hf_dataset`) to set `max_length`. | `False` |
| **Other** | | | |
| `--threads` | `All` | Number of CPU threads for PyTorch/OpenMP. | `CPU/2` |

### `dataset_chunk_create.py` Arguments ✂️

| Argument | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `--dataset` | Path to the input JSONL dataset file (Kayla format recommended). | **Yes** | |
| `--tokenizer-path` | Path or Hugging Face name of the tokenizer to use for chunking. | No | `openai-community/gpt2` |
| `--output-dir` | Directory to save the output **consolidated** `.pt` chunk files and `manifest.jsonl`. | No | `train_Hierarchos_chunked_tensors` |
| `--overlap` | Number of tokens to overlap between consecutive chunks. | No | `1024` |
| `--chunks-per-file` | Number of individual chunks to **consolidate** into a single `.pt` file. | No | `1000` |

### `expand_model.py` Arguments 🌱

| Argument | Description | Required | Default |
| :--- | :--- | :--- | :--- |
| `--old-model-path` | Path to the trained smaller model `.pt` checkpoint file. | **Yes** | |
| `--output-path` | Path to save the new, expanded `.pt` model file. | **Yes** | |
| `--context_dim` | **Required:** New context dimension (Auto-syncs hidden sizes). | **Yes** | |
| **Other Arch Args** | Optional: Add other architectural args like `--ltm_slots`, `--max_length`, etc., if changing them. | No | *(Uses old model's value)* |

-----

## Roadmap

  * [ ] Develop a user-friendly GUI wrapper for easier interaction.
  * [ ] Extend the architecture to support multi-modal inputs (images, audio).
  * [ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.

## License

The source code of Hierarchos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Hierarchos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  * This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  * The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  * **RWKV** community for the recurrent architecture inspiration.
  * **PyTorch Team** for `torch.compile` and gradient checkpointing functionality.

## Changelog

### v0.10.0 (alpha)

  * **Hinge Loss (Free Bit Budget):** Implemented `ReLU(drift - threshold)` to prevent Posterior Collapse in the Worker module.
  * **Last Batch Fix:** Fixed crash when dataset size isn't perfectly divisible by batch size.
  * **Commitment Control:** Added tunable threshold and weights for drift regularization.
  * **Robustness:** Improved state slicing and shadow state management during training.

### v0.9.5 (alpha)

  * **Enforced Symmetry**: Removed `h_hidden`/`l_hidden` flags. Both now sync to `context_dim` for stability.
  * **Context Drift**: Added `context_drift_proj` layer for dynamic context adaptation.
  * **Hugging Face Integration**: Added native support for HF datasets via `--hf_dataset`.
  * **Negative Reinforcement**: Chat loop now supports penalizing memory ("No", "Bad").
  * **Robust Signals**: Improved `Ctrl+C` handling for graceful exits.

### v0.9.0 (alpha)

  * **Architectural Overhaul (RWKV)**: Replaced GRU controllers with **RWKV** cells.
  * **Positional Embedding Removal**: Removed explicit `pos_emb`.
  * **Quantization Update**: Updated C++ kernels for RWKV layers.

### v0.8.5 (alpha)

  * **Reworked Kernel Build System**: CPU-only by default, optional `--vulkan` flag.

-----

© 2025 Makhi Burroughs
