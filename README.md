-----

# Hierarchos v0.11.15 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

Due to Amazon's "Chronos" forecasting models (still based on transformers BTW) I've decided to rename the project to "Hierarchos" from this point forward. This should prevent any naming confusion that may occur.

-----

### 🚀 **New in v0.11.15:**

> This update resolves critical architectural flaws that were causing **coherence problems** and **hallucinations** during inference.
>
> 1.  **Fixed Worker Loop Mismatch:** 🎯 Corrected a major discrepancy where the inference engine (`QuantizedHierarchos`) was advancing the RNN state `N` times per token (where `N` is `max_l_steps`), while training only advanced it once. This caused the model's internal state to drift ahead of the input, leading to severe incoherence. Inference now correctly uses a **shadow state** for pondering, matching the training logic.
> 2.  **Fixed Memory Persistence:** 🧠 Fixed a bug where LTM updates calculated during inference were being discarded. The model now correctly **persists** the new memory state (`fast_vals`, `mom_vals`) to its buffers, enabling true test-time learning.
> 3.  **Manager Pondering (ACT) in Inference:** ⚖️ Implemented the "Manager Pondering" (Adaptive Computation Time) logic in `QuantizedHierarchos` to match the training behavior. This resolves the "drift discrepancy" by ensuring the Manager's goal setting is consistent between training and inference.
> 4.  **Verified Stability:** ✅ Validated with reproduction scripts confirming that training and inference drift dynamics are now identical. Training results show stable convergence (e.g., loss=10.6880, ponder=3.09, commit=4.30e-01).

### 🚀 **New in v0.11.5:**

> This update resolves a critical **training-inference discrepancy** and further stabilizes the memory system.
>
> 1.  **Fixed Context Drift Discrepancy:** 🎯 Aligned the "Context Drift" logic between training and inference. Previously, training ignored the initial drift state, causing a massive discontinuity (jitter) at every token. Now, both modes calculate drift consistently from the hidden state, ensuring smooth transitions.
> 2.  **LTM Stability Clamps:** 🔒 Added safety clamps to the Long-Term Memory (LTM) update mechanism to prevent memory saturation and numerical instability during extended training runs.
> 3.  **Robust torch.compile Support:** 🛠️ Fixed a regression in the worker loop's robustness check that was causing `torch.compile` to fail on Windows CPU. Training with `--force-compile` is now stable.



### 📢 **Major Update in v0.7.0: Hugging Face `datasets` Integration & HRM Compute Note**

> This version significantly expanded dataset compatibility by integrating the Hugging Face `datasets` library and clarified the computational cost of the HRM's reasoning process.
>
> 1.  **Load Datasets from Anywhere:** 🌍 Hierarchos can now directly load and process datasets from the **Hugging Face Hub** or local paths using the `datasets` library. This adds support for numerous formats like **CSV, Parquet, JSON, Arrow**, etc., beyond the original JSONL support. New command-line arguments (`--hf_dataset`, `--hf_dataset_config`, `--hf_dataset_split`, `--text_column`, `--prompt_column`, `--completion_column`) allow specifying the dataset, configuration, split, and relevant text columns.
> 2.  **Clarified HRM Training Cost:** ⏱️ Added notes explaining how the HRM's iterative convergence (`--max_h_steps`, `--max_l_steps`) directly impacts **training speed and compute requirements**. Higher step limits allow for deeper reasoning but increase training time significantly compared to fixed-depth architectures.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Hierarchos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Hierarchos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are now structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth through **iterative convergence**. This enables it to solve complex, multi-step algorithmic problems where massive LLMs fail, though the depth of reasoning directly impacts computational cost during training.

## Features ✨

  * 🎓 **Proper Temporal Learning**: Configurable truncated BPTT (`--detach-every-n-steps`) enables learning across multiple timesteps while managing memory. Default 32-step gradients flow allows the model to **learn temporal dependencies** effectively.
  * 🔗 **End-to-End Gradient Flow**: All architectural components (Manager, Worker, LTM) receive proper gradients during training. No more detachment-induced coherence problems or NaN errors.
  * 🎯 **Train/Test Consistency**: Fixes train/test mismatch from unconditional state detachment, improving model coherence and stability.
  * 🌐 **Hugging Face `datasets` Integration**: Load datasets directly from the HF Hub or local paths in various formats (CSV, Parquet, JSON, etc.) using `--hf_dataset`.
  * 💾 **Optimized Consolidated Chunk Loading**: Dramatically reduces RAM usage and speeds up training startup for large datasets using pre-processed, consolidated `.pt` tensor files and a manifest (`--pre_pt_dataset`). Includes file caching for efficiency.
  * 📜 **Iterable Dataset Support**: Option to load pre-chunked JSONL datasets line-by-line (`--pre_chunked_dataset`) for minimal memory overhead during training.
  * ✂️ **Dataset Consolidation Script (`dataset_chunk_create.py`)**: Enhanced tool to prepare large datasets, chunking them into **consolidated `.pt` files** and creating a `manifest.jsonl` for efficient loading. Handles tokenization, anchoring, padding, and masking.
  * 📉 **Gradient Checkpointing**: Significantly reduces VRAM usage during training/fine-tuning (`--gradient-checkpointing`), enabling larger models or batches on memory-constrained hardware by trading compute for memory.
  * 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  * 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  * 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  * 🚀 **PyTorch 2.0+ torch.compile Support**: Optional compilation of the Worker loop with `--compile` for potential training speedups on NVIDIA GPUs (experimental, not recommended for Windows CPU).
  * ⚡ **Accelerated Training with AMP**: Supports Automatic Mixed Precision (`--amp`) for faster training and reduced memory usage on compatible NVIDIA GPUs.
  * 🛡️ **Stable Training**: Built-in gradient clipping (`--grad-clip`) to prevent model instability and ensure smoother convergence.
  * 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config for easy sharing and use.
  * 💾 **Automatic Re-quantization**: After a learning session, Hierarchos can automatically re-quantize a model to persist the new knowledge (`--enable-quantized-learning` in `chat`). *(Requires compiled kernel)*
  * 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones, now supporting changes in `max_length` and automatic length detection from datasets.
  * ✨ **Flexible Training Initiation**: Supports starting training runs using weights from existing model directories (inference or expanded models via `--model-path` in `train` mode), not just resuming full training checkpoints (`--resume-from-ckpt`).
  * ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). *(Requires compiled kernel)*
  * 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX/NEON) or on GPUs via Vulkan for broad hardware compatibility. *(Requires compiled kernel)*
  * 🔧 **Comprehensive Tooling**: Includes a single script (`hierarchos.py`) for training, LoRA fine-tuning, merging, quantization, and interactive chat, plus the model expansion and dataset chunking scripts.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * Python 3.8+
  * **For Hugging Face Datasets:** `pip install datasets`
  * **Optional (Quantization/Vulkan):**
      * A C++ compiler (e.g., MSVC on Windows, GCC on Linux)
      * CMake (must be available in your system's `PATH`)
      * Vulkan-compatible GPU and installed drivers (for Vulkan inference)
      * Vulkan SDK (if recompiling kernel with Vulkan support)
  * **Optional (AMP Training/Gradient Checkpointing):** NVIDIA GPU with CUDA support (Compute Capability 7.0+ recommended) and a PyTorch build with CUDA enabled.
  * **Optional (Kernel Build Dependencies):** `pip install pybind11 cmake`

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/Hierarchos.git
    cd Hierarchos
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\Activate
    # On Linux/macOS
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**

      * **Core (required for training/chat without quantization):**
        ```bash
        pip install -r core_requirements.txt
        ```
      * **Full (includes dependencies for kernel build, LoRA, quantization, etc.):**
        ```bash
        pip install -r requirements_kernel.txt
        ```

    *(Note: `requirements_kernel.txt` includes `datasets`)*

4.  **Compile C++ Kernel (Optional, for Quantization/Vulkan Inference):**
    If you need quantization or Vulkan support:

    ```bash
    # Ensure you have CMake, a C++ compiler, and installed dependencies from requirements_kernel.txt
    # On Windows
    setup.bat
    # On Linux/macOS
    bash setup.sh
    ```

    This creates `Hierarchos_matmul.*` in your project root. If you don't compile this, quantization modes (`quantize`, `--quantize-on-complete`, quantized `chat`) and Vulkan inference will be disabled.

-----

## 📚 User Guide: Comprehensive Workflows

This guide covers common scenarios from data preparation to inference.

### Workflow 1: Training a New Model

Choose **one** data source option:

**(A) Local JSON/JSONL File (Fits in RAM):**

```bash
python hierarchos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" `# Or your preferred tokenizer` \
    --out-dir "./my_Hierarchos_model" \
    --epochs 3 \
    --batch_size 4 \
    --accumulation-steps 2 `# Effective batch size = 8` \
    --auto-max-length `# Automatically determines max sequence length` \
    --context_dim 768 `# Example architecture` \
    --h_hidden 768 \
    --l_hidden 768 \
    --max_h_steps 5 \
    --max_l_steps 5 \
    --amp `# Enable Mixed Precision for speed` \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(B) Hugging Face Dataset (Text Completion):**

```bash
python hierarchos.py train \
    --hf_dataset "wikitext" \
    --hf_dataset_config "wikitext-2-raw-v1" \
    --hf_dataset_split "train" \
    --text_column "text" `# Column containing the text` \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_wikitext_model" \
    --epochs 1 \
    --batch_size 2 \
    --accumulation-steps 4 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(C) Hugging Face Dataset (Instruction/Kayla Format):**

```bash
python hierarchos.py train \
    --hf_dataset "databricks/databricks-dolly-15k" \
    --prompt_column "Instruction" \
    --completion_column "output" \
    # --kayla # Add if your HF data structure matches Kayla format (instruction, output, thought-process, feelings) \
    # --text_column "context" # Example: Map 'context' field if needed for your format \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_dolly_model" \
    --epochs 2 \
    --batch_size 1 \
    --accumulation-steps 8 \
    --auto-max-length \
    --amp \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(D) Pre-Chunked Local Dataset (Very Large Dataset):**

  * **Step 1: Create Chunks**
    ```bash
    python dataset_chunk_create.py \
        --dataset "path/to/very_large_data.jsonl" \
        --tokenizer-path "openai-community/gpt2" \
        --output-dir "./very_large_data_chunked" \
        --overlap 512 \
        --chunks-per-file 1000
    # Note the MAX_SEQ_LENGTH printed by the script (e.g., 3153)
    ```
  * **Step 2: Train using Chunks**
    ```bash
    python hierarchos.py train \
        --pre_pt_dataset `# Enable loading via manifest` \
        --train "./very_large_data_chunked" `# Directory with .pt files & manifest` \
        --max_length 3153 `# MUST match chunker output` \
        --tokenizer-path "openai-community/gpt2" `# Still needed for model init` \
        --out-dir "./my_large_model" \
        --epochs 1 \
        --batch_size 1 \
        --accumulation-steps 8 \
        --amp \
        --gradient-checkpointing # Add this if VRAM is limited
    ```

-----

💡 **Accelerating Training with AMP:** Use `--amp` for faster training and lower VRAM usage on NVIDIA GPUs.
💾 **Training on Low Memory:** Use `--gradient-checkpointing` to significantly reduce VRAM usage at the cost of some extra computation.

## ⚠️ **HRM Convergence & Training Speed:** Higher `--max_h_steps` and `--max_l_steps` allow deeper reasoning but **significantly increase training time** per batch due to the iterative HRM process. Adjust based on your task and compute resources.

### Workflow 2: Fine-Tuning with LoRA

Adapt a pre-trained model using new data (any supported format).

```bash
python hierarchos.py finetune \
    --model-path "./my_Hierarchos_model" `# Path to your trained base model` \
    --hf_dataset "squad" `# Example: Use SQuAD for QA fine-tuning` \
    --prompt_column "question" \
    --completion_column "answers" `# Might need custom processing depending on format` \
    --text_column "context" `# Use context as part of the prompt` \
    --out-dir "./my_squad_lora" \
    --epochs 1 \
    --lora_r 16 \
    --lora_alpha 32 \
    --amp \
    --gradient-checkpointing `# Use if fine-tuning large models on limited VRAM`
```

### Workflow 3: Merging LoRA Adapter

Combine the base model and the LoRA adapter into a new, standalone model.

```bash
python hierarchos.py merge-lora \
    --model-path "./my_Hierarchos_model" \
    --lora-adapter-path "./my_squad_lora" \
    --out-dir "./my_model_merged_squad"
```

### Workflow 4: Quantizing a Model *(Requires Compiled Kernel)*

Convert a full-precision model to a quantized format for faster, lower-resource inference.

```bash
python hierarchos.py quantize \
    --model-path "./my_model_merged_squad" \
    --out-dir "./my_model_merged_squad-Q4_0" \
    --qtype Q4_0 `# Choose format: INT4, Q4_0, Q8_0, Q2_K`
```

### Workflow 5: Running Chat Inference

Interact with your trained or fine-tuned model.

**Full Precision:**

```bash
python hierarchos.py chat --model-path "./my_model_merged_squad"
```

**Quantized *(Requires Compiled Kernel)*:**

```bash
python hierarchos.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --device cpu `# Or vulkan if compiled with Vulkan support`
```

**Chat with Online Learning (Quantized Example - Requires Compiled Kernel):**

```bash
python hierarchos.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged_squad" `# Path to original full-precision model` \
    --amp `# Optional: Speed up the learning step on CUDA` \
    # --ltm-lora-path "./my_chat_ltm_updates.pt" # Optional: Save LTM updates separately
```

### Workflow 6: Resuming Interrupted Training

Continue a `train` run from a saved checkpoint (`.pt` file).

```bash
python hierarchos.py train \
    # Dataset args might be loaded from checkpoint, specify only if needed \
    --out-dir "./my_large_model" \
    --resume-from-ckpt "./my_large_model/Hierarchos_epoch_1.pt" \
    --epochs 3 `# Total desired epochs` \
    --amp \
    --gradient-checkpointing # Ensure flag is consistent with the resumed run if needed
```

  * Use `--override-scheduling` with `--starting-lr`/`--min-lr` to change the learning rate schedule upon resuming.

### Workflow 7: Expanding a Model *(Requires `expand_model.py`)*

Create a larger model architecture initialized with weights from a smaller trained one.

```bash
python expand_model.py \
    --old-model-path "./my_Hierarchos_model/Hierarchos.pt" `# Trained smaller model .pt file` \
    --output-path "./expanded_model/Hierarchos.pt" `# Path for the new, expanded .pt file` \
    --context_dim 1024 `# New larger dimension` \
    --h_hidden 1024 \
    --l_hidden 1024
    # Note: expand_model.py takes specific architecture args to change.
    # Other config values are copied from the old model's checkpoint.
```

### Workflow 8: Continuing Training (After Expanding or from Inference Checkpoint)

Start a *new* training session using only the *weights* from an existing model directory (not resuming optimizer/scheduler state).

```bash
python hierarchos.py train \
    --hf_dataset "new_dataset_for_larger_model" \
    --text_column "text" \
    --model-path "./expanded_model" `# Load weights from expanded/previous model directory` \
    --tokenizer-path "./expanded_model" `# Use its tokenizer (assuming it was copied)` \
    --out-dir "./expanded_model_trained" \
    --epochs 2 \
    --starting-lr 5e-5 `# Start with a potentially smaller LR` \
    --amp \
    --gradient-checkpointing # Add if VRAM is limited
```

-----

## ⚙️ Command-Line Reference

### `hierarchos.py` Arguments

| Argument                     | Mode(s)                             | Description                                                                                                                              | Default                 |
| :----------------------------- | :---------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------- | :---------------------- |
| **Paths & Data** |                                     |                                                                                                                                          |                         |
| `--train`                      | `train`, `finetune`                 | Path to **local** data: JSON/JSONL file, or directory for `--pre_pt_dataset`. Use flag without path if using `--hf_dataset`. Mutually Exclusive with `--hf_dataset` path. | `None`                  |
| `--hf_dataset`                 | `train`, `finetune`                 | Name or path to a Hugging Face dataset (e.g., 'wikitext', 'c4', 'path/to/my\_csv/'). Mutually Exclusive with `--train` path.         | `None`                  |
| `--hf_dataset_config`          | `train`, `finetune`                 | Optional configuration name for the HF dataset (e.g., 'wikitext-103-raw-v1').                                                            | `None`                  |
| `--hf_dataset_split`           | `train`, `finetune`                 | Dataset split to use (e.g., 'train', 'validation', 'train[:10%]').                                                                       | `train`                 |
| `--text_column`                | `train`, `finetune`                 | Column name for text completion data in HF dataset (mutually exclusive with prompt/completion). Defaults to 'text' if available.           | `None`                  |
| `--prompt_column`              | `train`, `finetune`                 | Column name for prompt/instruction in HF dataset. Use with `--completion_column`.                                                        | `None`                  |
| `--completion_column`          | `train`, `finetune`                 | Column name for completion/response in HF dataset. Use with `--prompt_column`.                                                           | `None`                  |
| `--pre_chunked_dataset`        | `train`, `finetune`                 | Load pre-chunked **JSONL** dataset iteratively (requires `--max_length`). Mutually Exclusive with `--pre_pt_dataset` & `--hf_dataset`.     | `False`                 |
| `--pre_pt_dataset`             | `train`, `finetune`                 | Load pre-chunked **consolidated `.pt` tensor** dataset from directory specified in `--train` (requires `--max_length`). Mutually Exclusive with `--pre_chunked_dataset` & `--hf_dataset`. | `False`                 |
| `--model-path`                 | `train`, `finetune`, `merge`, `quantize`, `chat` | Path to model directory. **[Train]**: Loads weights only (starts fresh training). **[Other]**: Loads for the specified mode. | `None`                  |
| `--out-dir`                    | `train`, `finetune`, `merge`, `quantize` | Directory to save new models, checkpoints, or adapters.                                                                                | `./Hierarchos_model`       |
| `--tokenizer-path`             | `train`, `finetune`, `merge`, `quantize` | Path or HF name of tokenizer (if not loading from model-path).                                                                           | `openai-community/gpt2` |
| `--resume-from-ckpt`           | `train`                             | Path to `.pt` checkpoint to **resume full training state** (optimizer, etc.).                                                            | `None`                  |
| `--shadow-model-path`          | `chat`                              | Path to full-precision model dir for online learning with quantized model.                                                               | `None`                  |
| `--lora-adapter-path`          | `merge`, `finetune`                 | Path to the trained LoRA adapter directory.                                                                                            | `None`                  |
| **Training/Fine-Tuning** |                                     |                                                                                                                                          |                         |
| `--epochs`                     | `train`, `finetune`                 | Number of training epochs.                                                                                                               | `3`                     |
| `--batch_size`                 | `train`, `finetune`                 | Number of samples per forward pass.                                                                                                      | `4`                     |
| `--accumulation-steps`         | `train`, `finetune`                 | Number of steps to accumulate gradients over (simulates larger batch size).                                                              | `1`                     |
| `--gradient-checkpointing`     | `train`, `finetune`                 | **Enable gradient checkpointing to save VRAM (trades compute for memory).** | `False`                 |
| `--grad-clip`                  | `train`, `finetune`                 | Gradient clipping value. Prevents gradient explosion (0 to disable).                                                                     | `1.0`                   |
| `--ponder-loss-weight`         | `train`, `finetune`                 | Weight for the Ponder Cost auxiliary loss.                                                                                               | `0.01`                  |
| `--commitment-loss-weight`     | `train`, `finetune`                 | Weight for the commitment auxiliary loss to prevent posterior collapse.                                                                  | `0.5`                   |
| `--commitment-threshold`       | `train`, `finetune`                 | Hinge loss threshold for drift penalty. Drift^2 below this is not penalized.                                                             | `0.05`                  |
| `--override-scheduling`        | `train`                             | **[If resuming]** Ignore checkpoint's schedule state and use new LR args.                                                                | `False`                 |
| `--starting-lr`                | `train`, `finetune`                 | Max Learning Rate for the schedule, or fixed LR if schedule disabled.                                                                    | `1e-4`                  |
| `--min-lr`                     | `train`, `finetune`                 | Minimum Learning Rate for cosine annealing schedule.                                                                                     | `1e-6`                  |
| `--disable-lr-schedule`        | `train`, `finetune`                 | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing.                                                                 | `False`                 |
| `--ltm_lr`                     | `train`, `finetune`, `chat`         | Learning Rate for LTM "surprise" updates (or max LR for LTM schedule in chat).                                                         | `0.01`                  |
| `--compile`                    | `train`, `finetune`                 | **Enable torch.compile for faster training (experimental).** Compiles the Worker (L-RNN) loop for potential speedups on NVIDIA GPUs. **WARNING:** Known to hang on Windows CPU. | `False`                 |
| `--force-compile`              | `train`, `finetune`                 | Force torch.compile even on Windows CPU (overrides safety check). **Use with caution - may cause system hangs.** Requires `--compile`. | `False`                 |
| `--amp`                        | `train`, `finetune`, `chat`         | **Enable Automatic Mixed Precision (requires CUDA).** | `False`                 |
| `--num_workers`                | `train`, `finetune`                 | Number of CPU workers for data loading (and HF dataset mapping if applicable).                                                         | `0`                     |
| `--lora_r`                     | `finetune`                          | LoRA rank 'r'.                                                                                                                           | `8`                     |
| `--lora_alpha`                 | `finetune`                          | LoRA alpha scaling factor.                                                                                                               | `16`                    |\n| `--finetune-unlock-percent`    | `finetune`                          | Target % of params to train (approx.). Overrides `--lora_r` if set.                                                                     | `None`                  |
| `--kayla`                      | `train`, `finetune`                 | Enable Kayla-style instruction tuning format (with thought-process). **Ignored if using pre-chunked formats or --text\_column.** | `False`                 |
| **Quantization/Inference** |                                     |                                                                                                                                          |                         |
| `--qtype`                      | `quantize`, `train`                 | Quantization format (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). Used by `quantize` or `--quantize-on-complete`. **Requires compiled kernel.** | `INT4`                  |
| `--quantize-on-complete`       | `train`                             | Automatically run quantization after training finishes. **Requires compiled kernel.** | `False`                 |
| `--device`                     | `chat`                              | Device for *quantized* inference (`cpu`, `vulkan`). **Requires compiled kernel.** | `cpu`                   |
| `--h-halt-thresh`              | `chat`                              | Probability threshold for early exiting the HRM reasoning loop during inference.                                                         | `0.9`                   |
| `--max-new-tokens`             | `chat`                              | Maximum number of tokens to generate in chat mode.                                                                                       | `512`                   |
| `--enable-quantized-learning`  | `chat`                              | Enable LTM updates for quantized models (requires `--shadow-model-path` and **compiled kernel**).                                          | `False`                 |
| `--ltm-lora-path`              | `chat`                              | Optional: Path to save/load LTM updates as a separate delta file in chat mode.                                                           | `None`                  |
| `--static-ltm-lr`              | `chat`                              | Disable cosine annealing for chat LTM updates, use fixed `--ltm_lr`.                                                                     | `False`                 |
| `--ltm-schedule-steps`         | `chat`                              | Number of chat updates per LTM LR cosine cycle.                                                                                          | `100`                   |
| `--ltm-schedule-min-lr`        | `chat`                              | Minimum LR for chat LTM cosine schedule.                                                                                                 | `1e-5`                  |
| **Architecture (Train)** |                                     | *(Used only if starting train from scratch)* |                         |
| `--context_dim`                | `train`                             | Core embedding dimension.                                                                                                                | `768`                   |
| `--persistent_dim`             | `train`                             | Dimension of the fixed Persistent Memory.                                                                                                | `128`                   |
| `--ltm_slots`                  | `train`                             | Number of slots in the Long-Term Memory.                                                                                                 | `1024`                  |
| `--ltm_key_dim`                | `train`                             | Dimension of LTM keys.                                                                                                                   | `128`                   |
| `--ltm_val_dim`                | `train`                             | Dimension of LTM values.                                                                                                                 | `128`                   |
| `--h_hidden`                   | `train`                             | Hidden size of the High-Level (CEO) RNN.                                                                                                 | `768`                   |
| `--l_hidden`                   | `train`                             | Hidden size of the Low-Level (Worker) RNN.                                                                                               | `768`                   |
| `--max_h_steps`                | `train`                             | **Maximum** number of reasoning steps H-module can take. **Impacts training speed.** | `5`                     |
| `--max_l_steps`                | `train`                             | **Maximum** number of iterations for L-module convergence per H-step. **Impacts training speed.** | `5`                     |
| `--l_conv_atol`                | `train`                             | Absolute tolerance for checking L-module state convergence.                                                                              | `1e-4`                  |
| `--ltm_topk`                   | `train`                             | Number of LTM slots to retrieve per token.                                                                                               | `4`                     |
| `--detach-every-n-steps`       | `train`                             | **Truncated BPTT:** Detach RNN state gradients every N timesteps. Set to `None` for full BPTT (memory intensive). Lower values = less memory, less temporal learning. | `32`                    |
| `--max_length`                 | `train`, `finetune`                 | Maximum sequence length. **Required if using pre-chunked formats.** Set via scan (`--auto-max-length`), manually, or loaded from config. | `1024`                  |
| `--auto-max-length`            | `train`, `finetune`                 | Automatically scan dataset (`--train` or `--hf_dataset`) to set `max_length`. **Ignored if using pre-chunked formats.** | `False`                 |
| **Other** |                                     |                                                                                                                                          |                         |
| `--threads`                    | `All`                               | Number of CPU threads for PyTorch/OpenMP.                                                                                                | `CPU_Count/2`           |

### `dataset_chunk_create.py` Arguments ✂️

*(No changes)*

| Argument            | Description                                                                                       | Required | Default                         |
| :------------------ | :------------------------------------------------------------------------------------------------ | :------- | :------------------------------ |
| `--dataset`         | Path to the input **JSONL** dataset file (Kayla format recommended).                              | Yes      |                                 |
| `--tokenizer-path`  | Path or Hugging Face name of the tokenizer to use for chunking.                                   | No       | `openai-community/gpt2`         |
| `--output-dir`      | Directory to save the output **consolidated** `.pt` chunk files and `manifest.jsonl`.             | No       | `train_Hierarchos_chunked_tensors` |
| `--overlap`         | Number of tokens to overlap between consecutive chunks.                                           | No       | `1024`                          |
| `--chunks-per-file` | Number of individual chunks to **consolidate** into a single `.pt` file.                          | No       | `1000`                          |

### `expand_model.py` Arguments 🌱

| Argument             | Description                                                                         | Required | Default |
| :------------------- | :-------------------------------------------------------------------------------- | :------- | :------ |
| `--old-model-path`   | Path to the trained smaller model ***.pt checkpoint file***.                        | Yes      |         |
| `--output-path`      | Path to save the new, expanded ***.pt model file***.                                | Yes      |         |
| `--context_dim`      | ***Required:*** New context dimension.                                                | Yes      |         |
| `--h_hidden`         | ***Required:*** New H-RNN hidden size.                                                | Yes      |         |
| `--l_hidden`         | ***Required:*** New L-RNN hidden size.                                                | Yes      |         |
| *Other Arch Args* | *Optional:* Add other architectural args like `--ltm_slots`, `--max_length`, etc., if changing them. | No       | *(Uses old model's value)* |

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
  * **pybind11** for seamless C++/Python integration.
  * **Hugging Face `datasets`** library for broad data compatibility.
  * **PyTorch Team** for gradient checkpointing functionality.

## Changelog

### v0.11.15 (alpha)

  * **Critical Inference Fixes**:
      * **Worker Loop Correction**: Fixed logic error in `QuantizedHierarchos` where the RNN state was advanced multiple times per token. Now uses shadow state for pondering, matching training behavior.
      * **Memory Persistence**: Fixed bug where LTM updates were discarded during inference. Now correctly persists `fast_vals` and `mom_vals` to buffers.
      * **Manager Pondering (ACT)**: Implemented Manager Pondering in `QuantizedHierarchos` to match training logic, resolving drift discrepancy.
      * **Verified Coherence**: Drift dynamics now match between training and inference. Verified with stable training convergence.

### v0.11.5 (alpha)

  * **Coherence & Stability Fixes**:
      * **Fixed Drift Discrepancy**: Aligned training/inference logic for Context Drift. Training now correctly initializes drift from the previous hidden state, eliminating token-to-token jitter.
      * **LTM Clamping**: Added value clamping `[-20, 20]` to LTM updates to prevent saturation.
      * **Fixed torch.compile Regression**: Resolved a `TypeError` in the worker loop's NaN check that prevented compilation.

### v0.11.0 (alpha)

  * **Critical Gradient Flow Fixes**:
      * **Implemented Configurable Truncated BPTT**: Replaced unconditional state detachment in RWKV cells with proper truncated Back-Propagation Through Time. Added `--detach-every-n-steps` parameter (default: 32) to control gradient flow across timesteps.
      * **Fixed Worker Loop Shadow State**: Removed detachment in Worker (L-RNN) pondering loop that was breaking gradient flow. The Worker now learns to ponder effectively through proper gradient propagation.
      * **Fixed Manager State Flow**: Manager (H-RNN) now properly receives gradients from Worker feedback via `l_feedback_proj`. Hierarchical reasoning can learn from both architectural levels.
      * **Comprehensive Test Suite**: Added `test_gradient_flow.py` with 3 validation tests proving gradient flow, state continuity, and training convergence (22.8% loss reduction in 20 steps).
  * **Training Improvements**:
      * Gradients now flow properly through all critical components (tok_emb, h_rnn, l_rnn, ltm).
      * Fixed train/test mismatch from unconditional detachment, improving model coherence.
      * No more NaN/Inf errors or training hangs with proper gradient management.
      * State values remain numerically stable (< 3.0) with smooth transitions between batches.
  * **Architecture Changes**:
      * RWKV cells now accept `timestep` parameter for proper BPTT control.
      * Negative timesteps used for pondering loops to prevent detachment during exploration.
      * Removed redundant state clamping operations in Manager flow.
      * Improved torch.compile compatibility with better handling of dynamic control flow.
  * **Updated Documentation**: Added comprehensive walkthrough of fixes, added new features to README, updated command-line reference with `--detach-every-n-steps`, `--compile`, and `--force-compile` flags.

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

### v0.7.5 (alpha)

  * **Added Gradient Checkpointing**:
      * Implemented gradient checkpointing (`torch.utils.checkpoint.checkpoint`) within the `HierarchosCore` model's forward pass, specifically targeting the Adaptive HRM loop (`_adaptive_hrm_step`).
      * Added the `--gradient-checkpointing` command-line flag for `train` and `finetune` modes to enable this feature.
      * When enabled, this significantly reduces VRAM usage by recomputing activations during the backward pass instead of storing them, allowing for larger models or batches on memory-constrained GPUs.
      * Updated `train` function to save the `gradient_checkpointing` state in model config/checkpoints.
  * **Updated Documentation**: Added comprehensive documentation for gradient checkpointing in README (Features, User Guide, Command-Line Reference, Changelog). Updated version number. Corrected `expand_model.py` usage/arguments. Restored previously removed documentation sections.

### v0.7.0 (alpha)

  * **Added Hugging Face `datasets` Support**:
      * Integrated `datasets` library to load data directly from the Hub or local paths (CSV, Parquet, JSON, Arrow, text, etc.).
      * Added new arguments: `--hf_dataset`, `--hf_dataset_config`, `--hf_dataset_split`, `--text_column`, `--prompt_column`, `--completion_column`.
      * `--train` and `--hf_dataset` are now mutually exclusive sources.
      * Updated `train`, `finetune`, and `main` functions to handle the new loading mechanism.
      * Added `HuggingFaceMapStyleDataset` class and refactored dataloader creation.
      * Added `datasets` to requirements files.
  * **Clarified HRM Training Cost**: Added explanation in README about the impact of `--max_h_steps` and `--max_l_steps` on training speed and compute requirements due to iterative convergence.
  * **Updated Documentation**: Modified User Guide examples and Command-Line Reference to include HF dataset usage and arguments. Corrected defaults and argument descriptions based on latest code.

### v0.6.2 (alpha)

  * **Migrated from keyboard to signal**: Now uses Python standard "signal" library for chat interruption.

### v0.6.1 (alpha)

  * **Optimized Pre-Chunked Tensor Loading (`--pre_pt_dataset`)**:
      * `dataset_chunk_create.py` now saves **consolidated `.pt` files**.
      * A `manifest.jsonl` file is created for mapping chunks.
      * `PTChunkedDataset` updated to use manifest and **caching**.
  * **Documentation**: Updated README for consolidated chunking.

### v0.6 (alpha)

  * **Added Dataset Pre-processing Script (`dataset_chunk_create.py`)**: Chunks large `.jsonl` datasets into `.pt` tensor files.
  * **Implemented Direct Tensor Dataset Loading (`--pre_pt_dataset`)**: Load from `.pt` files + manifest.
  * **Implemented Iterable Pre-Chunked JSONL Loading (`--pre_chunked_dataset`)**: Load large JSONL line-by-line.
  * **Updated Dataloader Logic**: Conditional loading based on flags.
  * **Refined Training State Saving**: Checkpoints save dataset type flags.
  * **Documentation**: Updated for new chunking workflow.

### v0.5.2 (alpha)

  * **Added Flexible Training Initiation**: `--model-path` in `train` mode loads weights only for a new session.
  * **Enhanced `expand_model.py` Script**: Added `max_length` expansion and auto-detection.
  * **Added Automatic Mixed Precision (AMP)**: `--amp` flag for `train`, `finetune`, `chat`.
  * **Documentation**: Updated for new features.

### v0.5.1 (alpha)

  * **Added `--override-scheduling` flag**: Force new LR schedule when resuming.
  * **Documentation**: Updated for `--override-scheduling`.

### v0.5 (alpha)

  * **Implemented Structured Long-Term Memory**: Added timestamps and source metadata.
  * **Implemented Adaptive Reasoning Depth (Ponder Time)**: Dynamic HRM steps.
  * **Added Ponder Cost**: Auxiliary loss for efficiency.
  * **Added Halting Threshold**: Inference control (`--h-halt-thresh`).

### v0.4 (alpha)

  * **Implemented Dynamic LTM Learning Rate**: Default Cosine Annealing schedule in chat.
  * **Added Static LR Fallback**: `--static-ltm-lr` flag for chat.
  * **Added Gradient Clipping**: `--grad-clip` for training stability.

-----

© 2025 Makhi Burroughs
