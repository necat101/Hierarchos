# Running Hierarchos with ZLUDA (AMD GPUs)

ZLUDA allows you to run CUDA applications (like PyTorch with CUDA support) on AMD GPUs. This enables you to use features like `torch.compile` and `safe_globals` on Python 3.13.

## Prerequisites

1.  **Python 3.13** (Installed)
2.  **PyTorch with CUDA support**:
    You must install the standard PyTorch version with CUDA support, NOT the CPU version.
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    (Check [pytorch.org](https://pytorch.org/) for the latest command compatible with your ZLUDA version. CUDA 11.8 or 12.1 is typically recommended).

3.  **ZLUDA**:
    -   Download the latest release from [vosen/ZLUDA](https://github.com/vosen/ZLUDA/releases).
    -   Extract it to a folder (e.g., `C:\ZLUDA`).
    -   Add the `bin` folder to your PATH (optional but recommended).

## Running Training

Use `zluda.exe` to launch your Python script. This intercepts the CUDA calls and translates them to Vulkan/ROCm for your AMD GPU.

```bash
<path_to_zluda>\zluda.exe -- python hierarchos.py train --train --hf_dataset "tatsu-lab/alpaca" --device cuda ...
```

**Note**: You must specify `--device cuda` because ZLUDA makes your AMD GPU appear as a CUDA device to PyTorch.

## Troubleshooting

-   **"CUDA not available"**: Ensure you installed the CUDA version of PyTorch, not the CPU version.
-   **Crashes**: ZLUDA is experimental. If it crashes, try a different PyTorch/CUDA version (e.g., cu118 instead of cu121).
