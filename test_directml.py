"""
DirectML Verification Script for Hierarchos.

Run directly before attempting DirectML training:
    python test_directml.py

When collected by pytest on a machine without torch-directml, this module skips
instead of exiting during import so the rest of the pre-flight suite can run.
"""

import sys

import torch


def _print_header():
    print("=" * 60)
    print("DirectML Verification Script")
    print("=" * 60)


def _load_directml():
    print("\n[Test 1] Checking torch-directml availability...")
    try:
        import torch_directml
    except ImportError as exc:
        print("FAIL: torch-directml is NOT installed")
        print("  Install it with: pip install torch-directml")
        raise RuntimeError("torch-directml is not installed") from exc
    print("OK: torch-directml is installed")
    return torch_directml


def run_directml_verification():
    _print_header()
    torch_directml = _load_directml()

    print("\n[Test 2] Creating DirectML device...")
    dml_device = torch_directml.device()
    print(f"OK: DirectML device created: {dml_device}")

    print("\n[Test 3] Testing basic tensor operations on DirectML...")
    a = torch.randn(100, 100).to(dml_device)
    b = torch.randn(100, 100).to(dml_device)
    c = torch.matmul(a, b)
    d = a + b
    e = torch.relu(a)
    print("OK: Basic tensor operations successful")
    print(f"  - Matrix multiplication: {c.shape}")
    print(f"  - Addition: {d.shape}")
    print(f"  - ReLU activation: {e.shape}")

    print("\n[Test 4] Testing neural network operations...")
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
    ).to(dml_device)
    x = torch.randn(32, 128).to(dml_device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    print("OK: Neural network operations successful")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Forward pass output shape: {y.shape}")
    print("  - Backward pass completed")

    print("\n[Test 5] Testing optimizer operations...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    x = torch.randn(32, 128).to(dml_device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    print("OK: Optimizer operations successful")
    print(f"  - Loss value: {loss.item():.6f}")

    print("\n[Test 6] Testing mixed precision (AMP) support...")
    try:
        from torch.amp import GradScaler, autocast

        optimizer.zero_grad()
        scaler = GradScaler()
        with autocast(device_type="cpu", enabled=True):
            x = torch.randn(32, 128).to(dml_device)
            y = model(x)
            loss = y.sum()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        print("OK: Mixed precision (AMP) operations successful")
        print(f"  - AMP loss value: {loss.item():.6f}")
        print("  - Note: DirectML AMP is experimental")
    except Exception as exc:
        print(f"WARN: Mixed precision (AMP) operations failed: {exc}")
        print("  This is not critical; training can proceed without AMP.")
        print("  Use --no-amp when training if AMP fails.")

    print("\n[Test 7] Testing GPU<->CPU memory transfers...")
    gpu_tensor = torch.randn(1000, 1000).to(dml_device)
    cpu_tensor = gpu_tensor.cpu()
    cpu_tensor2 = torch.randn(1000, 1000)
    gpu_tensor2 = cpu_tensor2.to(dml_device)
    print("OK: Memory transfers successful")
    print(f"  - GPU->CPU transfer: {cpu_tensor.shape}")
    print(f"  - CPU->GPU transfer: {gpu_tensor2.shape}")

    print("\n" + "=" * 60)
    print("OK: All DirectML tests passed!")
    print("=" * 60)
    print("\nYou can now use DirectML for training Hierarchos:")
    print("  python hierarchos_cli.py train --device dml [other args]")
    print("\nOr let it auto-detect DirectML:")
    print("  python hierarchos_cli.py train [other args]")
    print("\nIf you encounter issues, try disabling AMP:")
    print("  python hierarchos_cli.py train --no-amp [other args]")
    print("=" * 60)


def test_directml_verification():
    pytest = __import__("pytest")
    try:
        import torch_directml  # noqa: F401
    except ImportError:
        pytest.skip("torch-directml is not installed")
    run_directml_verification()


if __name__ == "__main__":
    try:
        run_directml_verification()
    except Exception as exc:
        print(f"\nDirectML verification failed: {exc}")
        sys.exit(1)
