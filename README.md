-----

# Hierarchos v0.17 (alpha): A Hybrid Memory-Reasoning Architecture

**🎉 First Coherent Release!** — Hierarchos has successfully trained a 25M parameter model from scratch on Alpaca data, producing coherent instruction-following responses. See "Using Your Trained Model" below.

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 🚀 **New in v0.17: The "Standardized Evaluation" Update**

- **LM-Evaluation-Harness Integration**: Run benchmarks like HellaSwag or ARC-Easy directly during training or after.
- **Optional Dependency**: `lm-eval` is not required for core training, keeping the environment lightweight.
- **Periodic Evaluation**: Use `--eval-steps` or `--eval-every-epoch` to track model logic progress throughout training.
- **Step-Based Progress**: Trigger benchmarks every N steps (e.g., every 500 steps) for high-granularity progress tracking.


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

  * 📊 **Integrated Benchmarking**: Optional support for `lm-evaluation-harness`. Track model accuracy on standard benchmarks (HellaSwag, ARC, etc.) during or after training with `--eval-tasks`.
  * 🎮 **AMD GPU Support (DirectML/ZLUDA)**: Train on AMD Radeon GPUs using DirectML backend on Windows. Opt-in via `--device dml` with automatic compatibility handling and optimized fallbacks.
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
  * 🚀 **PyTorch 2.0+ torch.compile Support**: Optional compilation of the Worker loop with `--compile` for potential training speedups on NVIDIA GPUs and CPU (with `--force-compile`). Improved stability and performance on Windows.
  * ⚡ **Accelerated Training with AMP**: Supports Automatic Mixed Precision (`--amp`) for faster training and reduced memory usage on compatible NVIDIA GPUs. Automatically disabled for DirectML and CPU for stability.
  * 🛡️ **Stable Training**: Built-in gradient clipping (`--grad-clip`) to prevent model instability and ensure smoother convergence.
  * 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config for easy sharing and use.
  * 💾 **Automatic Re-quantization**: After a learning session, Hierarchos can automatically re-quantize a model to persist the new knowledge (`--enable-quantized-learning` in `chat`). *(Requires compiled kernel)*
  * 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones, now supporting changes in `max_length` and automatic length detection from datasets.
  * ✨ **Flexible Training Initiation**: Supports starting training runs using weights from existing model directories (inference or expanded models via `--model-path` in `train` mode), not just resuming full training checkpoints (`--resume-from-ckpt`).
  * ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). *(Requires compiled kernel)*
  * 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX/NEON) or on GPUs via Vulkan for broad hardware compatibility. *(Requires compiled kernel)*
  * 🔧 **Comprehensive Tooling**: Includes a single script (`hierarchos.py`) for training, LoRA fine-tuning, merging, quantization, and interactive chat, plus the model expansion and dataset chunking scripts.
  * 🐍 **Python 3.13 Support**: Full compatibility with Python 3.13, including automatic build environment setup and path detection.

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * **Python 3.8+ (Python 3.13 recommended)**
  * **For Hugging Face Datasets:** `pip install datasets`
  * **For AMD GPU Training (Windows):** Install DirectML via `pip install torch-directml` and follow [README_ZLUDA.md](README_ZLUDA.md)
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
      * **DirectML (AMD GPU on Windows):**
        ```bash
        pip install -r requirements_dml.txt
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

### Choosing Your Entry Point

> ⚠️ **Important:** The modular CLI (`hierarchos_cli.py`) is the **only supported entry point**. The original `hierarchos.py` is legacy and no longer maintained.

| Entry Point | Status | Description |
|-------------|--------|-------------|
| `hierarchos_cli.py` | ✅ **Recommended** | Modular CLI - faster, stable, actively maintained |
| `hierarchos.py` | ⚠️ **Legacy** | Unmaintained monolith (5,600 lines). Kept only as reference for agentic AI workflows. |

**Example:**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --prompt_column "instruction" \
    --completion_column "output" \
    --out-dir "./my_model" \
    --epochs 3 \
    --force-compile
```


### Workflow 1: Training a New Model

Choose **one** data source option:

**(A) Local JSON/JSONL File (Fits in RAM):**

```bash
python hierarchos_cli.py train \
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
    --amp `# Enable Mixed Precision for speed (NVIDIA GPUs only)` \
    --gradient-checkpointing # Add this if VRAM is limited
```

**(B) Hugging Face Dataset (Text Completion):**

```bash
python hierarchos_cli.py train \
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
python hierarchos_cli.py train \
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
    python hierarchos_cli.py train \
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

**(E) Training on AMD GPU (DirectML/Windows):**

```bash
python hierarchos_cli.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_amd_model" \
    --device dml `# Explicitly enable DirectML` \
    --epochs 3 \
    --batch_size 2 \
    --accumulation-steps 4 \
    --auto-max-length \
    --gradient-checkpointing # Recommended for AMD GPUs
```

-----

💡 **Accelerating Training with AMP:** Use `--amp` for faster training and lower VRAM usage on NVIDIA GPUs. Automatically disabled for DirectML and CPU.
💾 **Training on Low Memory:** Use `--gradient-checkpointing` to significantly reduce VRAM usage at the cost of some extra computation.
🎮 **AMD GPU Training:** Use `--device dml` to train on AMD Radeon GPUs via DirectML. AMP is automatically disabled for stability.

## ⚠️ **HRM Convergence & Training Speed:** Higher `--max_h_steps` and `--max_l_steps` allow deeper reasoning but **significantly increase training time** per batch due to the iterative HRM process. Adjust based on your task and compute resources.

### Workflow 2: Fine-Tuning with LoRA

Adapt a pre-trained model using new data (any supported format).

```bash
python hierarchos_cli.py finetune \
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
python hierarchos_cli.py merge-lora \
    --model-path "./my_Hierarchos_model" \
    --lora-adapter-path "./my_squad_lora" \
    --out-dir "./my_model_merged_squad"
```

### Workflow 4: Quantizing a Model *(Requires Compiled Kernel)*

Convert a full-precision model to a quantized format for faster, lower-resource inference.

```bash
python hierarchos_cli.py quantize \
    --model-path "./my_model_merged_squad" \
    --out-dir "./my_model_merged_squad-Q4_0" \
    --qtype Q4_0 `# Choose format: INT4, Q4_0, Q8_0, Q2_K`
```

### Workflow 5: Running Chat Inference

Interact with your trained or fine-tuned model.

**Full Precision:**

```bash
python hierarchos_cli.py chat --model-path "./my_model_merged_squad"
```

> ⚠️ **Important for Alpaca-Trained Models:** If you trained on instruction datasets like Alpaca, your model expects **instruction-formatted prompts**, not casual conversation. See "Using Your Trained Model" section below.

**Quantized *(Requires Compiled Kernel)*:**

```bash
python hierarchos_cli.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --device cpu `# Or vulkan if compiled with Vulkan support`
```

**Chat with Online Learning (Quantized Example - Requires Compiled Kernel):**

```bash
python hierarchos_cli.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --enable-quantized-learning \
    --shadow-model-path "./my_model_merged_squad" `# Path to original full-precision model` \
    --amp `# Optional: Speed up the learning step on CUDA` \
    # --ltm-lora-path "./my_chat_ltm_updates.pt" # Optional: Save LTM updates separately
```

### Workflow 6: Resuming Interrupted Training

Continue a `train` run from a saved checkpoint (`.pt` file).

```bash
python hierarchos_cli.py train \
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
python hierarchos_cli.py train \
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

### Workflow 9: Converting Checkpoints to Inference Models

Convert a training checkpoint to a clean, inference-ready model directory.

```bash
python hierarchos_cli.py ckpt-2-inf \
    --ckpt-input "./my_model/hierarchos_epoch_60.pt" \
    --inf-output "./my_inference_model" \
    --ckpt-tok-path "openai-community/gpt2"  # Tokenizer used during training
```

This creates a HuggingFace-style directory:
```
my_inference_model/
├── model.pt              # Clean model weights (~66% smaller than checkpoint)
├── hierarchos_config.json # Model configuration
├── tokenizer.json         # Tokenizer files
├── vocab.json
└── merges.txt
```

### Workflow 10: Benchmark Evaluation (lm-eval)

Run standardized LLM benchmarks on your model. Requires `pip install lm-eval` (automatically installed through the setup script if you used it).

**During Training (End of Epoch):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --eval-tasks hellaswag arc_easy \
    --eval-every-epoch 1 \
    --eval-limit 100 # Optional: test on only 100 samples for speed
```

**Step-Based Evaluation (Frequent tracking):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --eval-tasks arc_easy \
    --eval-steps 500 # Runs every 500 steps
    --eval-limit 10
```

-----

## 🎯 Using Your Trained Model

### Instruction-Trained Models (Alpaca, Dolly, etc.)

If you trained on **instruction-following datasets** like Alpaca, your model expects prompts formatted as instructions, not casual conversation.

**❌ This won't work well:**
```
>>> hello!
hierarchos: Journey.  (incoherent)
```

**✅ Use instruction-style prompts:**
```
>>> Explain what machine learning is in simple terms.
hierarchos: Machine learning is a type of artificial intelligence that uses 
algorithms to learn from data and improve performance...
```

**Good prompt examples:**
```
>>> Write a short poem about learning.
>>> List 3 benefits of exercise.
>>> What is the capital of France?
>>> Explain photosynthesis to a 5-year-old.
```

### Sampling Parameters

Adjust generation quality with:
```bash
python hierarchos_cli.py chat --model-path "./my_model" --temperature 0.5 --top-k 40 --top-p 0.9 --repetition-penalty 1.2
```

| Parameter | Effect | Recommended |
|-----------|--------|-------------|
| `--temperature` | Lower = more focused, higher = more creative | 0.5-0.7 |
| `--top-k` | Limit vocab to top K tokens | 40 |
| `--top-p` | Nucleus sampling threshold | 0.9 |
| `--repetition-penalty` | Penalize repeated tokens (1.0=off, >1.0=stronger) | 1.2 |

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
| `--encourage-thinking`         | `train`                             | **Invert ponder loss to REWARD thinking.** Useful for ACT recovery training.                                                              | `False`                 |
| `--adaptive-ponder`            | `train`                             | **Scale ponder target with CE loss.** Harder content triggers more thinking.                                                              | `False`                 |
| `--ponder-target-scale`        | `train`                             | Scaling factor for adaptive ponder target (target = loss × scale).                                                                        | `0.5`                   |
| `--reset-halt-bias`            | `train`                             | **SURGICAL FIX:** Reset `h_halt_proj.bias` to this value on checkpoint load (e.g., `-2.0` for ~12% halt prob).                            | `None`                  |
| `--commitment-loss-weight`     | `train`, `finetune`                 | Weight for the commitment auxiliary loss to prevent posterior collapse.                                                                  | `0.5`                   |
| `--commitment-threshold`       | `train`, `finetune`                 | Hinge loss threshold for drift penalty. Drift^2 below this is not penalized.                                                             | `0.05`                  |
| `--override-scheduling`        | `train`                             | **[If resuming]** Ignore checkpoint's schedule state and use new LR args.                                                                | `False`                 |
| `--starting-lr`                | `train`, `finetune`                 | Max Learning Rate for the schedule, or fixed LR if schedule disabled.                                                                    | `1e-4`                  |
| `--min-lr`                     | `train`, `finetune`                 | Minimum Learning Rate for cosine annealing schedule.                                                                                     | `1e-6`                  |
| `--disable-lr-schedule`        | `train`, `finetune`                 | Use a fixed Learning Rate (`--starting-lr`) instead of cosine annealing.                                                                 | `False`                 |
| `--ltm_lr`                     | `train`, `finetune`, `chat`         | Learning Rate for LTM "surprise" updates (or max LR for LTM schedule in chat).                                                         | `0.01`                  |
| `--compile`                    | `train`, `finetune`                 | **Enable torch.compile for faster training (experimental).**                                                                              | `False`                 |
| `--force-compile`              | `train`, `finetune`                 | Force torch.compile even on Windows CPU (overrides safety check).                                                                         | `False`                 |
| `--amp`                        | `train`, `finetune`, `chat`         | **Enable Automatic Mixed Precision (requires CUDA).**                                                                                     | `False`                 |
| `--num_workers`                | `train`, `finetune`                 | Number of CPU workers for data loading (and HF dataset mapping if applicable).                                                         | `0`                     |
| `--lora_r`                     | `finetune`                          | LoRA rank 'r'.                                                                                                                           | `8`                     |
| `--lora_alpha`                 | `finetune`                          | LoRA alpha scaling factor.                                                                                                               | `16`                    |\n| `--finetune-unlock-percent`    | `finetune`                          | Target % of params to train (approx.). Overrides `--lora_r` if set.                                                                     | `None`                  |
| `--kayla`                      | `train`, `finetune`                 | Enable Kayla-style instruction tuning format (with thought-process). **Ignored if using pre-chunked formats or --text\_column.** | `False`                 |
| **Quantization/Inference** |                                     |                                                                                                                                          |                         |
| `--qtype`                      | `quantize`, `train`                 | Quantization format (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). Used by `quantize` or `--quantize-on-complete`. **Requires compiled kernel.** | `INT4`                  |
| `--quantize-on-complete`       | `train`                             | Automatically run quantization after training finishes. **Requires compiled kernel.** | `False`                 |
| `--device`                     | `chat`, `train`                     | Device for inference/training (`cpu`, `cuda`, `dml`/`directml`, `vulkan`). **Note:** `dml` requires `torch-directml` and Windows. DirectML requires explicit opt-in. | `auto`                   |
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
  * [ ] Expand DirectML support to Linux via ROCm.
  * [ ] Optimize LTM retrieval with approximate nearest neighbor search for larger memory capacities.

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
  * **DirectML/ZLUDA communities** for enabling AMD GPU acceleration on Windows.

## Changelog

### v0.17 (alpha)

  * **LM-Evaluation-Harness Integration**: Added optional benchmarking during/after training.
  * **HierarchosLM Wrapper**: Custom implementation of `loglikelihood`, `loglikelihood_rolling`, and `generate_until` for full compatibility with `lm-eval`.
  * **Periodic Step-Based Eval**: Added `--eval-steps` to trigger evaluation every N steps for high-granularity progress tracking.
  * **Configurable Eval**: Added `--eval-every-epoch`, `--eval-batch-size`, and `--eval-limit` control flags.
  * **Startup Confirmation**: Training now confirms if evaluation is enabled at launch.

### v0.16.2.1 (alpha)

  * **⚠️ CRITICAL: LTM Threshold Bugfix**:
      * Fixed bug where passive learning updated LTM on *every* turn, regardless of threshold
      * Could corrupt model weights over time — **restore from backup if you used v0.16.1-v0.16.2**
      * Added `compute_only` parameter to separate loss computation from actual updates
  * **Repetition Penalty**: `--repetition-penalty` (default 1.2) prevents output loops
  * **Passive Learning**: LTM learns from conversations automatically (threshold-gated)
  * **Checkpoint Converter**: `ckpt-2-inf` mode for HuggingFace-style directories
  * **First Coherent Release**: 25M model trained on Alpaca produces coherent output

### v0.15.2 (alpha)

  * **ACT Sensitivity Fixes**:
      * **Surgical Halt Bias Reset**: Added `--reset-halt-bias` to directly reset `h_halt_proj.bias` on checkpoint load, immediately fixing "ponder stickiness" where ACT gets stuck at minimal values.
      * **Encourage Thinking Mode**: Added `--encourage-thinking` flag to invert ponder loss (reward thinking instead of penalize).
      * **Adaptive Ponder Targeting**: Added `--adaptive-ponder` with `--ponder-target-scale` to automatically scale target ponder with CE loss.
  * **Training Visibility**:
      * **Model Stats Display**: Training now prints total/trainable parameter count and estimated checkpoint size at startup.

### v0.15 (alpha)

  * **Modular Architecture**: Reorganized `hierarchos.py` into `hierarchos/` package. Added `hierarchos_cli.py` as recommended entry point.
  * **Training Loop Parity**: Implemented 128-token temporal chunking with TBPTT, per-batch state reset, and full forward parity with original.

### v0.14 (alpha)

  * **Critical Training Fix**:
      * **Fixed Positional Jitter**: Passed `global_pos_offset` in the training loop. This ensures the Manager's stride/interpolation logic is continuous across chunk boundaries, resolving the 1.92 loss plateau.
  * **Architecture & Learning**:
      * **Differentiable Ponder Cost**: Switched to a differentiable ACT sum (`cum_remain.sum()`), enabling the model to learn halting efficiency.
  * **Chat & Recovery**:
      * **Incremental Generation**: Refactored chat to use prefill + single-token steps for perfect RNN state management.
      * **Full State Reset**: Added `/reset` command to zero out all internal states (h_state, l_state, context, drift).
      * **Thorough LTM Reset**: Updated `/reset_ltm` to clear all memory buffers including timestamps and sources.
      * **State Persistence**: Fixed a bug where hierarchical states were being reset during incremental steps.

### v0.13.10 (alpha)

### v0.13.5 (alpha)

  * **Coherence & Stability Fixes**:
      * **Fixed Manager Stride Logic**: Corrected the Manager's strided update check to use global position instead of local chunk index, eliminating drift between training (chunked) and inference (sequential).
      * **Unified Lerp Interpolation**: Updated the context interpolation (Lerp) to use global position, ensuring smooth and consistent context transitions in all modes.
      * **LTM Persistence**: Verified and fixed persistence of LTM `fast_vals` and `mom_vals` during inference, enabling reliable test-time learning.
      * **Chat Engine Update**: Updated `chat` function to pass `global_pos_offset` to the full-precision model, propagating coherence fixes to interactive sessions.

### v0.13.0 (alpha)

  * **Interactive Sampling Parameters**:
      * Added `/temp <float>`, `/topk <int>`, and `/topp <float>` commands to `chat` mode for dynamic control over generation creativity.
      * Added `/settings` command to view current sampling parameters.
  * **LTM Stability & Persistence Fixes**:
      * **Fixed Gradient Flow**: Resolved a critical issue where LTM gradients were detached, preventing the model from learning to use its memory effectively during training.
      * **Fixed Passive Updates**: Corrected a logic error that caused the model to skip generation after passive memory updates.
      * **Crash Fixes**: Resolved `NameError` and `TypeError` in the LTM update routine.

### v0.12.0 (alpha)

  * **DirectML Support (AMD GPU Acceleration)**:
      * Added native support for AMD Radeon GPUs on Windows via DirectML backend
      * Implemented device auto-detection with explicit opt-in via `--device dml` or `--device directml`
      * Auto-disables AMP for DirectML devices for stability
      * Added DirectML-compatible implementations for operations (replaced `torch.lerp`, `torch.index_add_`, etc.)
      * Prevents custom kernel loading on DirectML devices to avoid incompatibilities
      * Includes comprehensive compatibility checks and optimized fallbacks
  * **LTM Memory Optimization & Stability**:
      * Refactored `LTMModule.inner_update` to use in-place operations, significantly reducing memory overhead
      * Fixed LTM memory persistence bug where updates were calculated but not saved during training
      * Added LTM value clamping `[-20, 20]` to prevent saturation and numerical instability
      * Improved surprise-based update mechanism for more stable memory consolidation
  * **Python 3.13 Support**:
      * Updated `setup.bat` and `setup.ps1` to fully support Python 3.13
      * Added automatic detection for various Python installation paths (PATH, standard, Windows Store, `py` launcher)
      * Fixed C++ compilation issues related to spaces in Python installation paths
      * Added Python libs path to LIB environment variable for successful kernel compilation
  * **torch.compile Stability Improvements**:
      * Fixed device detection for `torch.compile` - now correctly identifies device type for autocast
      * Improved dynamic shape support on Windows CPU
      * Added `--force-compile` safety check with better warnings
      * Fixed compilation errors related to NaN checks in worker loop
      * Improved compatibility with Windows CPU compilation
  * **Enhanced Device Management**:
      * Improved `pick_device()` auto-detection logic with priority: CUDA → CPU (DirectML is opt-in)
      * Added `is_directml_device()` and `get_device_type()` utility functions
      * DirectML now requires explicit opt-in to prevent accidental use with incompatible operations
      * Better handling of device selection in training and inference modes
  * **Numerical Stability Fixes**:
      * Added comprehensive state clamping in `_worker_loop` to prevent NaN propagation
      * Added `gate_input` clamping before sigmoid operations
      * Improved stability checks for convergence detection
      * Added safety guards for AMP/DirectML compatibility in RWKV cells
  * **MSVC Build Environment Improvements**:
      * Enhanced `setup_msvc_environment()` with better vswhere.exe integration
      * Added automatic vcvars64.bat path detection and caching
      * Fixed vcvarsall.bat environment variable parsing
      * Improved error handling and user guidance for manual compiler setup
  * **Performance Optimizations**:
      * Reduced memory allocations in LTM update paths
      * Improved gradient computation efficiency during training
      * Better cache utilization in dataset loading
  * **Documentation Updates**:
      * Added DirectML setup instructions and workflow examples
      * Updated command-line reference with device selection options
      * Added notes on AMP auto-disable for DirectML and CPU
      * Enhanced troubleshooting guidance for various hardware configurations

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

  * **Hinge Loss:** Implemented `ReLU(drift - threshold)` to prevent Posterior Collapse.
  * **Commitment Control:** Added tunable threshold and weights for drift regularization.

*(Older changelog entries have been archived for brevity. See git history for versions prior to v0.10.)*

-----

© 2026 Makhi Burroughs
