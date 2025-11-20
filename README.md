-----
# Hierarchos v0.9 (alpha): A Hybrid Memory-Reasoning Architecture

A novel AI architecture that synergistically integrates Google's Titans memory system with a Hierarchical Reasoning Model (HRM) to move beyond the limitations of scale and take a decisive step on the path to AGI.

Due to Amazon's "Chronos" forecasting models (still based on transformers BTW) I've decided to rename the project to "Hierarchos" from this point forward. This should prevent any naming confusion that may occur.

-----

### 🚀 **Major Update in v0.9.0: The RWKV Paradigm Shift**

> This version marks a fundamental architectural evolution, replacing the previous Gated Recurrent Unit (GRU) controllers with **RWKV (Receptance Weighted Key Value)** cells.
>
> 1.  **RWKV Architecture:** 🧠 Hierarchos now leverages the benefits of Linear Transformers within an RNN framework. This allows for parallelizable training (like Transformers) while maintaining constant memory usage during inference (like RNNs), offering significantly better performance/efficiency trade-offs than standard GRUs.
> 2.  **Removal of Positional Embeddings:** 🚫 We have removed explicit learned Positional Embeddings (`pos_emb`). RWKV handles temporal relationships and token positions naturally via its internal state decay (`time_decay`). This creates a model that is **state-invariant** rather than position-dependent, improving generalization to sequence lengths far beyond the training window (e.g., training on 1024 tokens and inferencing on 4096+).
> 3.  **Quantized RWKV Support:** ⚡ The C++ kernel and quantization logic have been updated to support the new RWKV cell structure (Time Mixing and Channel Mixing) for high-performance INT4/Q4_0 inference.

### 🚀 **Major Update in v0.8.5: Reworked Build System & Vulkan Auto-Setup**

> This version completely overhauls the C++ kernel build process for simplicity and cross-platform reliability, especially for the optional Vulkan backend.
>
> 1.  **CPU-First Default:** ⚙️ The build scripts (`setup.bat`, `setup.sh`) now compile the **CPU-optimized kernel by default**.
> 2.  **Opt-in Vulkan Build:** 🔥 To enable the Vulkan backend for GPU-accelerated quantized inference, simply pass the `--vulkan` flag.
> 3.  **Automatic Dependency Installation (Linux):** 🐧 The `setup.sh` script will automatically detect missing Vulkan tools and install them via `apt`.

## About The Project

The field of AI has been dominated by a paradigm of unprecedented scale, yet fundamental limitations in today's Transformer models are becoming apparent. The path to Artificial General Intelligence (AGI) may not be paved with scale alone. Hierarchos challenges this paradigm by focusing on **architectural intelligence**.

This project introduces a novel hybrid model where a deep reasoning engine operates within a dynamic, lifelong learning memory environment. Hierarchos is conceived not merely to process information, but to **think, learn, and remember** in a cohesive, integrated, and human-like manner.

## Core Concepts

Hierarchos is built on two revolutionary, brain-inspired pillars:

🧠 **Titans Architecture (The Cognitive Substrate)**
A sophisticated, multi-tiered memory workspace that enables dynamic, lifelong learning. It learns *what to remember* based on the principle of "surprise," and its memory slots are structured with timestamps and source metadata, allowing for sophisticated, context-aware queries.

⚙️ **Hierarchical Reasoning Model (The Cognitive Process)**
A powerful, data-efficient, and deep reasoning engine. Now powered by **RWKV Recurrence**, the dual-module design (a high-level "CEO" and low-level "Workers") allows for profound computational depth through **iterative convergence**. This enables it to solve complex, multi-step algorithmic problems where massive LLMs fail.

## Features ✨

  * 🧠 **RWKV-Based Recurrence**: Replaces legacy GRU cells with RWKV, offering better scalability, numerical stability, and "time-decay" based temporal processing.
  * 📏 **Infinite Length Generalization**: By removing absolute positional embeddings, the model relies on state carry-over, allowing it to process sequences significantly longer than those seen during training.
  * 🔥 **PyTorch 2.0+ Compiled Training**: **Automatically uses `torch.compile`** on the core HRM loop for massive speedups (1.5x-3x+) on modern NVIDIA GPUs after an initial "warmup" compilation.
  * 🌐 **Hugging Face `datasets` Integration**: Load datasets directly from the HF Hub or local paths in various formats (CSV, Parquet, JSON, etc.) using `--hf_dataset`.
  * 💾 **Optimized Consolidated Chunk Loading**: Dramatically reduces RAM usage and speeds up training startup for large datasets using pre-processed, consolidated `.pt` tensor files (`--pre_pt_dataset`).
  * ✂️ **Dataset Consolidation Script (`dataset_chunk_create.py`)**: Enhanced tool to prepare large datasets, chunking them into **consolidated `.pt` files** and creating a `manifest.jsonl`.
  * 📉 **Gradient Checkpointing**: Significantly reduces VRAM usage during training/fine-tuning (`--gradient-checkpointing`), enabling larger models or batches on memory-constrained hardware.
  * 🤔 **Adaptive "Ponder" Time**: Dynamically adjusts its reasoning depth, "thinking" longer for complex problems and saving computation on simpler ones.
  * 🕰️ **Structured & Queryable Memory**: LTM slots are augmented with timestamps and source data, enabling powerful temporal and contextual queries during chat.
  * ⚡ **Accelerated Training with AMP**: Supports Automatic Mixed Precision (`--amp`) for faster training and reduced memory usage on compatible NVIDIA GPUs.
  * 📦 **Self-Contained & Portable Models**: Models are saved as directories containing weights, tokenizer, and architecture config.
  * 🌱 **Enhanced Model Expansion**: `expand_model.py` allows transplanting weights to larger models. (Updated in v0.9 to automatically drop legacy positional embeddings).
  * ⚡ **High-Performance Inference**: Utilizes a custom C++ kernel inspired by `llama.cpp` for state-of-the-art quantization (`INT4`, `Q4_0`, `Q8_0`, `Q2_K`). *(Requires compiled kernel)*
  * 💻 **CPU & GPU Support**: Runs fast quantized inference on standard CPUs (with AVX/NEON) or on GPUs via Vulkan for broad hardware compatibility. *(Requires compiled kernel)*

-----

## 🚀 Getting Started

Follow these steps to get a local copy up and running.

### Prerequisites

  * Python 3.8+
  * **PyTorch 2.0+ (Required for `torch.compile` speedups)**
  * `pip install datasets`
  * **Optional (Quantization/Vulkan):** C++ compiler, CMake, Vulkan SDK.

### Installation

-----

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/necat101/Hierarchos.git](https://github.com/necat101/Hierarchos.git)
    cd Hierarchos
    ```

2.  **Run the Setup Script:**
    This script builds the C++ kernel required for quantization and optimized inference.

    **Default CPU Build (Recommended):**
    ```bash
    # On Windows
    setup.bat

    # On Linux/macOS
    bash setup.sh
    ```

    **Vulkan Build (Optional):**
    ```bash
    setup.bat --vulkan   # Windows
    bash setup.sh --vulkan # Linux/macOS
    ```

-----

## 📚 User Guide: Comprehensive Workflows

### Workflow 1: Training a New Model

**(A) Local JSON/JSONL File (Fits in RAM):**

```bash
python hierarchos.py train \
    --train "path/to/your_data.jsonl" \
    --tokenizer-path "openai-community/gpt2" \
    --out-dir "./my_Hierarchos_model" \
    --epochs 3 \
    --batch_size 4 \
    --accumulation-steps 2 \
    --auto-max-length \
    --context_dim 768 \
    --h_hidden 768 \
    --l_hidden 768 \
    --max_h_steps 5 \
    --max_l_steps 5 \
    --amp \
    --gradient-checkpointing
````

**(B) Pre-Chunked Local Dataset (Very Large Dataset):**

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
        --pre_pt_dataset \
        --train "./very_large_data_chunked" \
        --max_length 3153 `# MUST match chunker output` \
        --tokenizer-path "openai-community/gpt2" \
        --out-dir "./my_large_model" \
        --epochs 1 \
        --batch_size 1 \
        --accumulation-steps 8 \
        --amp \
        --gradient-checkpointing
    ```

-----

## ⚠️ **HRM Convergence & Training Speed:** Higher `--max_h_steps` and `--max_l_steps` allow deeper reasoning but **significantly increase training time** per batch due to the iterative HRM process. Adjust based on your task and compute resources.

### Workflow 2: Fine-Tuning with LoRA

```bash
python hierarchos.py finetune \
    --model-path "./my_Hierarchos_model" \
    --hf_dataset "squad" \
    --prompt_column "question" \
    --completion_column "answers" \
    --text_column "context" \
    --out-dir "./my_squad_lora" \
    --epochs 1 \
    --lora_r 16 \
    --amp \
    --gradient-checkpointing
```

### Workflow 3: Merging LoRA Adapter

```bash
python hierarchos.py merge-lora \
    --model-path "./my_Hierarchos_model" \
    --lora-adapter-path "./my_squad_lora" \
    --out-dir "./my_model_merged_squad"
```

### Workflow 4: Quantizing a Model *(Requires Compiled Kernel)*

```bash
python hierarchos.py quantize \
    --model-path "./my_model_merged_squad" \
    --out-dir "./my_model_merged_squad-Q4_0" \
    --qtype Q4_0 `# Choose format: INT4, Q4_0, Q8_0, Q2_K`
```

### Workflow 5: Running Chat Inference

**Full Precision:**

```bash
python hierarchos.py chat --model-path "./my_model_merged_squad"
```

**Quantized:**

```bash
python hierarchos.py chat \
    --model-path "./my_model_merged_squad-Q4_0" \
    --device cpu `# Use "vulkan" if compiled with support`
```

### Workflow 7: Expanding a Model

Create a larger model architecture initialized with weights from a smaller trained one. *Note: Legacy positional embeddings are automatically dropped.*

```bash
python expand_model.py \
    --old-model-path "./my_Hierarchos_model" \
    --output-dir "./expanded_model" \
    --context_dim 1024 \
    --h_hidden 1024 \
    --l_hidden 1024
```

-----

## ⚙️ Command-Line Reference

*(Refer to the code or previous documentation for full argument list. Key additions in v0.9 below)*

### `hierarchos.py` Arguments

| Argument | Mode(s) | Description | Default |
| :--- | :--- | :--- | :--- |
| `--max_length` | `train`, `finetune` | Maximum sequence length. Even though `pos_emb` is removed, this is used for data chunking and buffer sizing. | `1024` |
| `--context_dim` | `train` | Core embedding dimension. | `768` |
| `--h_hidden` | `train` | Hidden size of the High-Level (CEO) RNN (now RWKV). | `768` |
| `--l_hidden` | `train` | Hidden size of the Low-Level (Worker) RNN (now RWKV). | `768` |

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

### v0.9 (alpha)

  * **Architectural Overhaul (RWKV)**: Replaced GRU controllers with **RWKV** cells. This improves efficiency and scalability.
  * **Positional Embedding Removal**: Removed explicit `pos_emb`. The model now relies on RWKV's state decay for time perception, improving generalization to sequences longer than the training context.
  * **Quantization Update**: Updated `Quantizedhierarchos` and C++ kernels to support quantized RWKV layers (Time Mixing/Channel Mixing).
  * **Model Expansion Fix**: Updated `expand_model.py` to handle the removal of positional embeddings when transplanting weights.

### v0.8.6 (alpha)

  * **Improved alpaca dataset handling**: "Instruction" and "input" are now merged during training as needed to follow the alpaca dataset format\!

### v0.8.5 (alpha)

  * **Reworked Kernel Build System**: CPU-only by default, optional `--vulkan` flag.
  * **Linux Auto-Install**: `setup.sh` detects missing Vulkan tools and installs them.

### v0.8.0 (alpha)

  * **Added Experimental `torch.compile` Support**: Massive speedups on NVIDIA GPUs.

### v0.7.5 (alpha)

  * **Added Gradient Checkpointing**: Significantly reduces VRAM usage.

### v0.7.0 (alpha)

  * **Added Hugging Face `datasets` Support**.

-----

© 2025 Makhi Burroughs

