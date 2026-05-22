-----

# Hierarchos v0.19.3 (alpha): Progress Sync Throttling

**hierarchos/models/ltm.py** (line 122): CUDA LTM updates now aggregate only touched slots with torch.unique plus indexed writes, avoiding full dense [slots, val_dim] and [batch, slots, val_dim] scatter buffers.

**hierarchos/models/core.py** (line 628): added _compute_cuda_chunked_lm_loss, matching dense CE plus z-loss behavior while chunking lm_head rows for large-vocab CUDA memory pressure.

**hierarchos/training/trainer.py**: progress-bar scalar logging is now throttled with `--progress-log-steps`, avoiding a CUDA sync from `.item()` every single batch on fast GPUs.

**hierarchos/training/datasets.py**: Alpaca formatting is standardized around the explicit `### Instruction:`, optional `### Input:`, and `### Response:` prompt string while preserving the same supervised completion labels.

**The "Progress Sync Throttling" update** — Hierarchos now avoids unnecessary CUDA-to-CPU metric syncs during training progress display. The training objective, batch data, labels, and model architecture are unchanged.

**The "DataLoader Throughput Tuning" update** — Hierarchos now keeps CUDA input queues bounded by tying auto prefetch to worker count, uses conservative CUDA worker defaults for pre-tokenized datasets, and keeps pinned memory specific to CUDA training.

**The "Optimization and GUI Update"** — Hierarchos keeps its CPU-friendly math paths intact while automatically switching hot LTM memory operations to GPU-friendly gather/scatter math on CUDA. This release also highlights the Windows GUI bundle workflow for easier local inference and experimentation.

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) and RWKV linear attention to move beyond the limitations of scale and take a decisive step on the path to AGI.

-----

### 🚀 **New in v0.19.3: Progress Sync Throttling**

#### CUDA Training Loop
- **Throttled Progress Metrics**: `train_step` only returns display metrics on scheduled progress updates, so CUDA scalar `.item()` calls no longer synchronize every batch by default.
- **Configurable Logging Interval**: Use `--progress-log-steps N` to update tqdm scalar metrics every N steps. Default is `10`; use `1` for the old every-step behavior.
- **Non-Destructive Runtime Patch**: This changes only metric reporting cadence. Model architecture, dataset formatting, labels, losses, optimizer steps, and scheduler behavior are unchanged.
- **Alpaca Prompt String Documented**: `--alpaca` uses `### Instruction:`, optional `### Input:`, and `### Response:` formatting before the supervised output text.

### 🚀 **New in v0.19.2: DataLoader Throughput Tuning**

#### CUDA/CPU Input Pipeline
- **Conservative CUDA Auto Workers**: `--num_workers -1` now auto-selects a smaller CUDA worker pool, targeting the common pre-tokenized sweet spot instead of oversubscribing CPU and pinned-memory bandwidth.
- **Worker-Tied Prefetch**: When `--prefetch-factor` is omitted, total queued batches stay bounded relative to worker count. More workers lower per-worker prefetch instead of silently multiplying queued batches.
- **CUDA-Only Pinned Memory**: DataLoader pinning is enabled for CUDA training and avoided for CPU/DirectML paths where it adds overhead without async H2D benefit.
- **Override Still Available**: Use explicit `--num_workers` and `--prefetch-factor` when slow storage, Hugging Face tokenization, or JSON parsing makes the input pipeline the bottleneck.

### 🚀 **New in v0.19: The "Optimization and GUI Update"**

#### Optimization and GUI
- **Internal CUDA Math Switch**: LTM retrieval and memory updates automatically use CUDA-friendly gather/scatter paths on NVIDIA GPUs while preserving the existing CPU-friendly dense math on CPU.
- **No User Flag Required**: The architecture selects the math path internally based on tensor device placement, keeping CLI and GUI configuration simple.
- **ROSA Preserved**: ROSA remains CPU-side and VRAM-light by design.
- **Windows GUI Release Flow**: The README documents the portable GUI bundle workflow for shipping `Hierarchos.exe` with the bundled backend.

#### 🧠 Architecture
- **RWKV v8 Backbone**: Replaced GRU cells with full RWKV v8 (Receptance Weighted Key Value) cells featuring linear attention, Time Mixing with WKV recurrence, and SwiGLU Channel Mixing.
- **DeepEmbed (4x Scale)**: New learnable token embeddings at 4× hidden dimension that gate the RWKV channel mixing FFN, providing richer per-token modulation.
- **ROSA (Receptive Ordered Suffix Automaton)**: A neurosymbolic inner monologue — a CPU-side Suffix Automaton predicts likely next tokens, which are embedded and added to the input representation. Gives the model a "heads up" about upcoming patterns.
- **V7 Backward Compatibility**: Set `use_deepembed=False, use_rosa=False` in config to run in pure V7 mode. All V7 checkpoints load cleanly.

#### ⚡ CUDA Datacenter Optimizations (Zero Config)
- **Auto-AMP**: Mixed precision auto-enables on CUDA — no `--amp` flag needed.
- **bfloat16 on Ampere+**: SM ≥ 8.0 GPUs automatically use bfloat16 (better dynamic range, no GradScaler overhead).
- **TF32 Matmul**: 3-8× faster linear layers on Ampere+ GPUs, enabled automatically.
- **cuDNN Benchmark**: Auto-tunes convolution kernels for hardware.
- **torch.compile Auto-Enable**: Worker loop compiled on CUDA (no Windows CPU hang issue).
- **Non-blocking Transfers**: Host-to-device copies overlap with GPU computation via `non_blocking=True`.
- **Pinned Memory**: DataLoader uses `pin_memory=True` for CUDA training.
- **Bounded DataLoader Prefetch**: Auto prefetch is tied to worker count to avoid over-queuing pinned batches.
- **`--no-amp` Flag**: Explicitly disable AMP if needed.

#### 🧪 Test Suite
- **11/11 Tests Pass**: Full architectural validation including gradient flow, state continuity, training convergence, memory gradients, sampling logic, coherence, forward/backward, inference generation, V7 backward compat, LTM decay parity, and momentum amplification.
- **Self-Contained Tests**: All tests create models in-memory — no hardcoded checkpoint paths.


## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Hierarchos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Hierarchos is built on three revolutionary, brain-inspired pillars:

🔄 **RWKV v8 Backbone (The Neural Engine)**
A modernized RNN with linear attention that achieves the parallel training speed of Transformers with the O(1) inference cost of RNNs. Features Time Mixing (WKV recurrence with exponential decay), SwiGLU Channel Mixing, DeepEmbed gating, and ROSA neurosymbolic embeddings.

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are now structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Its dual-module design (a high-level "Manager" and low-level "Worker") allows for profound computational depth through **iterative convergence**. This enables it to solve complex, multi-step algorithmic problems where massive LLMs fail.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│ Input Tokens → tok_emb + ROSA(Suffix Automaton) + DeepEmbed(4x) │
│                    ↓                                             │
│ LTM Retrieval (differentiable top-k attention via qproj)         │
│                    ↓                                             │
│ Encoder → in_proj(GELU)                                         │
│                    ↓                                             │
│ Manager H-RNN (RWKV v8) ← l_feedback_proj                      │
│   · ACT Pondering (shadow state, halt probabilities)            │
│   · Strided Context Plan + LERP interpolation                   │
│                    ↓                                             │
│ Worker L-RNN (RWKV v8, torch.compiled on CUDA)                  │
│   · Shadow-state exploration + convergence detection             │
│   · Drift commitment cost                                       │
│                    ↓                                             │
│ out_norm → lm_head → Logits (weight-tied with tok_emb)          │
│                    ↓                                             │
│ Titans LTM Update (gradient-based surprise + Hebbian)           │
│                    ↓                                             │
│ CE Loss + Z-Loss + Ponder Cost + Commitment Cost                │
└──────────────────────────────────────────────────────────────────┘
```

## Features ✨

  * 🔄 **RWKV v8 Backbone**: Linear-complexity attention with O(1) inference cost. WKV recurrence, SwiGLU FFN, DeepEmbed gating, and ROSA neurosymbolic embeddings.
  * ⚙️ **Adaptive CPU/CUDA LTM Math**: Keeps CPU-friendly dense LTM math on CPU while automatically using GPU-friendly gather/scatter retrieval and update paths on CUDA.
  * ⚡ **CUDA Datacenter Ready**: Auto-enables AMP (bfloat16 on Ampere+), TF32 matmul, cuDNN benchmark, torch.compile, non-blocking transfers, CUDA pinned memory, and bounded DataLoader prefetch — zero configuration needed.
  * 🪟 **Windows GUI Bundle**: Build a portable GUI release with `Hierarchos.exe` and a bundled backend for normal inference without requiring users to clone the repo.
  * 📊 **Integrated Benchmarking**: Optional support for `lm-evaluation-harness`. Track model accuracy on standard benchmarks (HellaSwag, ARC, etc.) during or after training with `--eval-tasks`.
  * 🎮 **AMD GPU Support (DirectML/ZLUDA)**: Train on AMD Radeon GPUs using DirectML backend on Windows. Opt-in via `--device dml` with automatic compatibility handling and optimized fallbacks.
  * 🎓 **Proper Temporal Learning**: Configurable truncated BPTT (`--detach-every-n-steps`) enables learning across multiple timesteps while managing memory. Default 32-step gradients flow allows the model to **learn temporal dependencies** effectively.
  * 🔗 **End-to-End Gradient Flow**: All architectural components (Manager, Worker, LTM) receive proper gradients during training. No more detachment-induced coherence problems or NaN errors.
  * 🎯 **Train/Test Consistency**: Fixes train/test mismatch from unconditional state detachment, improving model coherence and stability.
  * 🌐 **Hugging Face `datasets` Integration**: Load datasets directly from the HF Hub or local paths in various formats (CSV, Parquet, JSON, etc.) using `--hf_dataset`.
  * 💾 **Optimized Consolidated Chunk Loading**: Dramatically reduces RAM usage and speeds up training startup for large datasets using pre-processed, consolidated `.pt` tensor files and a manifest (`--pre_pt_dataset`). Includes file caching for efficiency.
  * 📜 **Iterable Dataset Support**: Option to load pre-chunked JSONL datasets line-by-line (`--pre_chunked_dataset`) for minimal memory overhead during training.
  * ✂️ **Dataset Consolidation Script (`dataset_chunk_create.py`)**: Enhanced tool to prepare large datasets, chunking them into **consolidated `.pt` files** and creating a `manifest.jsonl` for efficient loading.
  * 📉 **Gradient Checkpointing**: Significantly reduces VRAM usage during training/fine-tuning (`--gradient-checkpointing`), enabling larger models or batches on memory-constrained hardware.
  * 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  * 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  * 🧠 **Dynamic "Online" Learning**: Learns from experience during chat with a Cosine Annealing LR schedule by default for more stable knowledge consolidation.
  * 🚀 **PyTorch 2.0+ torch.compile Support**: Auto-enabled on CUDA, optional on CPU with `--compile` / `--force-compile`.
  * 🛡️ **Stable Training**: Built-in gradient clipping (`--grad-clip`), Z-loss regularization, and state clamping to prevent instability.
  * 📦 **Self-Contained & Portable Models**: Models are saved as HuggingFace-style directories containing weights, tokenizer, and architecture config for easy sharing and deployment.
  * 💾 **Automatic Re-quantization**: After a learning session, Hierarchos can automatically re-quantize a model to persist the new knowledge (`--enable-quantized-learning` in `chat`). *(Requires compiled kernel)*
  * 🌱 **Enhanced Model Expansion**: Includes `expand_model.py` script to transplant weights from smaller models to larger ones.
  * ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). *(Requires compiled kernel)*
  * 💻 **CPU & GPU Support**: Runs quantized inference on CPUs (AVX/NEON) or GPUs via Vulkan. *(Requires compiled kernel)*
  * 🐍 **Python 3.13 Support**: Full compatibility with Python 3.13, including automatic build environment setup.

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
| `hierarchos.py` | ⚠️ **Legacy** | Unmaintained monolith (5,600 lines). Kept only as reference for agentic AI workflows. | <-- DO NOT USE THIS! ITS 15 VERSIONS OUT OF DATE!!

**Example:**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
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
    --context_dim 448 `# ~233M params with GPT-2 vocab, DeepEmbed, ROSA` \
    --h_hidden 448 \
    --l_hidden 448 \
    --rwkv-head-size 64 \
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

**(C) Hugging Face Dataset (Instruction/Alpaca/Kayla Format):**

```bash
python hierarchos_cli.py train \
    --hf_dataset "databricks/databricks-dolly-15k" \
    --prompt_column "Instruction" \
    --completion_column "output" \
    # --alpaca # Add for instruction/input/output datasets; defaults columns to instruction/output \
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

💡 **CUDA Auto-Optimization:** On NVIDIA GPUs, AMP, TF32, cuDNN benchmark, and torch.compile are **auto-enabled**. Long CUDA runs default to `--compile-mode max-autotune`, static RWKV worker-loop capture, H-RNN cell compilation, and CUDA graphs. Use `--no-amp` to disable AMP.
💾 **Training on Low Memory:** Use `--gradient-checkpointing` to significantly reduce VRAM usage at the cost of some extra computation.
🎮 **AMD GPU Training:** Use `--device dml` to train on AMD Radeon GPUs via DirectML. AMP is automatically disabled for stability.
🚀 **Datacenter Training:** start with auto workers or `--num_workers 4 --batch_size 32 --training-chunk-size 512 --persist-state`, then benchmark higher worker counts only if the input pipeline is still the bottleneck.

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

### Windows GUI Release Bundle

Build a portable Windows GUI bundle with a bundled PyTorch/Transformers backend:

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\tools\build_windows_release.ps1
```

The release is written to `dist\Hierarchos-Windows\` and can be zipped for
distribution. Users run `Hierarchos.exe`; the GUI launches
`backend\hierarchos-backend.exe`, so they do not need to clone this repo or
install Python for normal inference. The GUI accepts a Hugging Face repo id,
a local model directory containing `hierarchos.pt` or `model.pt`, or a direct
`.pt` checkpoint with embedded config or a neighboring `hierarchos_config.json`.

### Workflow 10: Benchmark Evaluation (lm-eval)

Run standardized LLM benchmarks on your model. Requires `pip install lm-eval` (automatically installed through the setup script if you used it).

**During Training (End of Epoch):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
    --eval-tasks hellaswag arc_easy \
    --eval-every-epoch 1 \
    --eval-limit 100 # Optional: test on only 100 samples for speed
```

**Step-Based Evaluation (Frequent tracking):**
```bash
python hierarchos_cli.py train \
    --hf_dataset "tatsu-lab/alpaca" \
    --alpaca \
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

For models trained with `--alpaca`, the training formatter uses this prompt string before the supervised output:
```text
### Instruction:
<instruction>

### Input:
<optional input>

### Response:
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
| `--compile`                    | `train`, `finetune`                 | **Enable torch.compile (auto-enabled on CUDA).**                                                                              | `False`                 |
| `--force-compile`              | `train`, `finetune`                 | Force torch.compile even on Windows CPU (overrides safety check).                                                                         | `False`                 |
| `--compile-mode`               | `train`, `finetune`                 | torch.compile mode for the RWKV hot path. `max-autotune` has longer startup but best amortized CUDA throughput for long runs. | `max-autotune`          |
| `--no-compile-cudagraphs`      | `train`, `finetune`                 | Disable CUDA graph capture inside torch.compile.                                                                                         | `False`                 |
| `--no-compile-pad-to-chunk-size` | `train`, `finetune`               | Disable CUDA compile padding to `training_chunk_size` multiples; leaving it enabled reduces shape recompiles.                            | `False`                 |
| `--no-compile-static-worker-loop` | `train`, `finetune`              | Disable the compile-friendly fixed WorkerLoop used by CUDA compile.                                                                       | `False`                 |
| `--amp`                        | `train`, `finetune`, `chat`         | **Enable Automatic Mixed Precision (auto-enabled on CUDA).**                                                                                     | `False`                 |
| `--no-amp`                     | `train`, `finetune`                 | **Explicitly disable AMP** (overrides auto-detection on CUDA).                                                                                   | N/A                     |
| `--num_workers`                | `train`, `finetune`                 | Number of CPU workers for data loading (`-1` = auto; CUDA defaults conservatively, CPU/DML use 0).                                      | `-1`                    |
| `--prefetch-factor`            | `train`, `finetune`                 | Batches prefetched per DataLoader worker. Omit it to keep total queued batches tied to worker count.                                    | `None`                  |
| `--progress-log-steps`         | `train`                             | Update tqdm scalar metrics every N steps to reduce CUDA sync overhead from progress logging (`1` = every step).                         | `10`                    |
| `--pt-cache-size`              | `train`, `finetune`                 | Number of `.pt` chunk files to keep hot per worker when using `--pre_pt_dataset`.                                                       | `2`                     |
| `--lora_r`                     | `finetune`                          | LoRA rank 'r'.                                                                                                                           | `8`                     |
| `--lora_alpha`                 | `finetune`                          | LoRA alpha scaling factor.                                                                                                               | `16`                    |\n| `--finetune-unlock-percent`    | `finetune`                          | Target % of params to train (approx.). Overrides `--lora_r` if set.                                                                     | `None`                  |
| `--kayla`                      | `train`, `finetune`                 | Enable Kayla-style instruction tuning format (with thought-process). **Ignored if using pre-chunked formats or --text\_column.** | `False`                 |
| `--alpaca`                     | `train`, `finetune`                 | Enable Alpaca `instruction`/`input`/`output` formatting. Defaults prompt/completion columns to `instruction`/`output` and includes `input` as a `### Input:` context block. | `False`                 |
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
| `--context_dim`                | `train`                             | Core embedding dimension. The default 448 keeps full RWKV matrix-state heads at 64-wide geometry and lands near 233M params with GPT-2 vocab + DeepEmbed/ROSA. | `448`                   |
| `--persistent_dim`             | `train`                             | Dimension of the fixed Persistent Memory.                                                                                                | `128`                   |
| `--ltm_slots`                  | `train`                             | Number of slots in the Long-Term Memory.                                                                                                 | `1024`                  |
| `--ltm_key_dim`                | `train`                             | Dimension of LTM keys.                                                                                                                   | `128`                   |
| `--ltm_val_dim`                | `train`                             | Dimension of LTM values.                                                                                                                 | `128`                   |
| `--h_hidden`                   | `train`                             | Hidden size of the High-Level (CEO) RNN. Defaults to `context_dim`.                                                                       | `448`                   |
| `--l_hidden`                   | `train`                             | Hidden size of the Low-Level (Worker) RNN. Defaults to `context_dim`.                                                                     | `448`                   |
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
  * [ ] Implement multi-GPU training with DistributedDataParallel / FSDP.
  * [ ] Implement the entire training loop in Vulkan/CUDA for end-to-end GPU acceleration.
  * [ ] Expand DirectML support to Linux via ROCm.
  * [ ] Optimize LTM retrieval with approximate nearest neighbor search for larger memory capacities.
  * [ ] Explore RWKV v8 custom CUDA kernels for fused WKV computation.

## License

The source code of Hierarchos is available to the public under a custom license. It is free for non-commercial use, research, and evaluation. However, any commercial use resulting in profit is subject to a profit-sharing agreement. See `LICENSE.md` for full details.

## Support This Project

Please consider supporting my work on Patreon. I have motor cortex damage, which prevents me from working in a traditional tech role. I work on Hierarchos in my spare time while working full-time at a grocery store.

**[https://www.patreon.com/cw/MakhiBurroughs](https://www.patreon.com/cw/MakhiBurroughs)**

## Acknowledgements

  * This architecture is inspired by the concepts in Google's **Titans** and Sapient Intelligence's **HRM** papers.
  * **RWKV** architecture by BlinkDL — linear attention with RNN efficiency.
  * The quantization kernel design is heavily influenced by the groundbreaking work in **llama.cpp**.
  * **pybind11** for seamless C++/Python integration.
  * **Hugging Face `datasets`** library for broad data compatibility.
  * **PyTorch Team** for gradient checkpointing functionality.
  * **DirectML/ZLUDA communities** for enabling AMD GPU acceleration on Windows.

## Changelog

### v0.19.3 (alpha)

  * **Progress Sync Throttling**: Training progress metrics are now collected only on scheduled tqdm updates, reducing CUDA synchronization from per-step `.item()` logging.
  * **`--progress-log-steps`**: New train flag controls metric update cadence; default `10`, with `1` restoring every-step logging.
  * **Alpaca Prompt String**: The `--alpaca` formatter is documented as `### Instruction:`, optional `### Input:`, and `### Response:` before the supervised output string.

### v0.19.2 (alpha)

  * **DataLoader Throughput Tuning**: CUDA auto worker selection now defaults to a conservative worker pool for pre-tokenized training, avoiding unnecessary CPU contention and pinned-memory churn.
  * **Worker-Tied Prefetch**: Auto `prefetch_factor` now keeps total queued batches bounded relative to `num_workers`; explicit `--prefetch-factor` still overrides this when the input pipeline is truly the bottleneck.
  * **CUDA-Only Pinning**: DataLoader pinned memory is selected from the requested device, so CPU and DirectML training avoid pinning overhead while CUDA keeps async host-to-device transfers.
  * **Chunked Dataset Efficiency**: Pre-tokenized JSONL workers shard by file byte range, and `.pt` chunk loading keeps a small hot cache per worker.

### v0.19 (alpha)

  * **Optimization and GUI Update**: Release focus for CUDA/CPU math selection and the Windows GUI workflow.
  * **Adaptive LTM Math Paths**: LTM retrieval and memory updates keep the existing CPU-friendly dense one-hot/matmul path on CPU, while CUDA tensors automatically use gather/scatter-based math for better GPU utilization.
  * **Zero-Config Device Selection**: The architecture chooses the GPU-friendly LTM path internally when running on CUDA; no new CLI or GUI flag is required.
  * **ROSA Remains CPU-Side**: ROSA is intentionally unchanged so it stays fast, VRAM-light, and CPU-friendly.
  * **GUI Release Documentation**: Windows GUI bundle instructions are called out for portable `Hierarchos.exe` distribution with the bundled backend.

### v0.18 (alpha)

  * **🧠 RWKV v8 Backbone**: Complete replacement of GRU cells with RWKV v8 cells featuring:
      * **Time Mixing**: WKV (Weighted Key Value) recurrence with exponential decay and `time_first` / `time_decay` learnable parameters.
      * **Channel Mixing**: SwiGLU-gated feed-forward network with 4× expansion.
      * **5-Slot State**: `(sx, aa, bb, pp, sx_cm)` replaces the old 3-slot GRU state for richer temporal representation.
      * **Float32 WKV**: Critical exponential calculations run in float32 for numerical stability, even under AMP.
  * **🎨 DeepEmbed (4× Scale)**: New `h_deepemb` and `l_deepemb` embeddings at `hidden_dim × 4` that gate the RWKV channel mixing FFN, providing per-token modulation of the feed-forward pathway.
  * **🔮 ROSA (Receptive Ordered Suffix Automaton)**: A neurosymbolic inner monologue module:
      * CPU-side Suffix Automaton predicts likely next tokens from input history.
      * Predictions are embedded via `rosa_emb` and added to the input representation.
      * Gives the model a "heads up" about upcoming patterns (O(n) precomputation).
      * `past_tokens` state maintained across inference turns for continuity.
  * **⚡ CUDA Datacenter Auto-Optimization** (zero config):
      * **AMP auto-enable**: Mixed precision activates on CUDA without `--amp` flag.
      * **bfloat16 on Ampere+**: SM ≥ 8.0 GPUs use bf16 (no GradScaler overhead).
      * **TF32 matmul**: 3-8× faster linear layers on Ampere+.
      * **cuDNN benchmark**: Auto-tunes kernel selection for hardware.
      * **torch.compile auto-enable**: Worker loop compiled on CUDA.
      * **Non-blocking transfers**: `to(device, non_blocking=True)` for async H2D.
      * **pin_memory on CUDA**: Enables async H2D when CUDA training is active.
      * **bounded DataLoader prefetch**: Auto prefetch stays tied to worker count.
      * **drop_last on CUDA**: Prevents irregular batch OOM.
  * **🧪 Test Suite Modernized**: 11/11 tests pass. Rewrote 3 stale tests (`test_forward.py`, `test_inference.py`, `verify_parity_deep.py`) to be self-contained — create models in-memory instead of loading hardcoded checkpoints.
  * **🛡️ Stability Hardening**:
      * `ltm_state` detach handles both 2-tuple and 3-tuple formats (forward compat).
      * `verify_ltm_decay.py` and `verify_momentum_inference.py` fixed for correct tuple unpacking.
  * **🔙 V7 Backward Compatibility**: Setting `use_deepembed=False, use_rosa=False` produces a valid V7 model. All V7 checkpoints load cleanly.
  * **📦 HuggingFace Directory Output Restored**: Training exports `hierarchos.pt` + full tokenizer suite + `hierarchos_config.json` in a self-contained directory.
  * **🆕 CLI Additions**: `--no-amp` flag, improved help text for `--amp`, `--compile`, `--num_workers`.
  * **📊 GPU Diagnostics**: Training startup prints GPU name, VRAM, SM version, and all auto-enabled optimizations.

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

*(Older changelog entries have been archived for brevity. See git history for versions prior to v0.16.)*

-----

© 2026 Makhi Burroughs
