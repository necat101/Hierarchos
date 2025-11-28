"""
DirectML Verification Script for Hierarchos

This script tests DirectML GPU acceleration support.
Run this before attempting full training to ensure DirectML is working correctly.
"""

import sys
import torch

print("=" * 60)
print("DirectML Verification Script")
print("=" * 60)

# Test 1: Check torch-directml availability
print("\n[Test 1] Checking torch-directml availability...")
try:
    import torch_directml
    print("✓ torch-directml is installed")
    HAS_DIRECTML = True
except ImportError:
    print("✗ torch-directml is NOT installed")
    print("  Install it with: pip install torch-directml")
    HAS_DIRECTML = False
    sys.exit(1)

# Test 2: Create DirectML device
print("\n[Test 2] Creating DirectML device...")
try:
    dml_device = torch_directml.device()
    print(f"✓ DirectML device created: {dml_device}")
except Exception as e:
    print(f"✗ Failed to create DirectML device: {e}")
    sys.exit(1)

# Test 3: Basic tensor operations
print("\n[Test 3] Testing basic tensor operations on DirectML...")
try:
    # Create tensors
    a = torch.randn(100, 100).to(dml_device)
    b = torch.randn(100, 100).to(dml_device)
    
    # Matrix multiplication
    c = torch.matmul(a, b)
    
    # Element-wise operations
    d = a + b
    e = torch.relu(a)
    
    print("✓ Basic tensor operations successful")
    print(f"  - Matrix multiplication: {c.shape}")
    print(f"  - Addition: {d.shape}")
    print(f"  - ReLU activation: {e.shape}")
except Exception as e:
    print(f"✗ Tensor operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Neural network operations
print("\n[Test 4] Testing neural network operations...")
try:
    import torch.nn as nn
    import torch.nn.functional as F
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 64)
    ).to(dml_device)
    
    # Forward pass
    x = torch.randn(32, 128).to(dml_device)
    y = model(x)
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    print("✓ Neural network operations successful")
    print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - Forward pass output shape: {y.shape}")
    print(f"  - Backward pass completed")
except Exception as e:
    print(f"✗ Neural network operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Optimizer operations
print("\n[Test 5] Testing optimizer operations...")
try:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Simulate training step
    optimizer.zero_grad()
    x = torch.randn(32, 128).to(dml_device)
    y = model(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    print("✓ Optimizer operations successful")
    print(f"  - Loss value: {loss.item():.6f}")
except Exception as e:
    print(f"✗ Optimizer operations failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Mixed precision (AMP) - Optional
print("\n[Test 6] Testing mixed precision (AMP) support...")
try:
    from torch.amp import autocast, GradScaler
    
    # Note: DirectML uses 'cpu' as device_type for autocast
    scaler = GradScaler()
    
    with autocast(device_type='cpu', enabled=True):
        x = torch.randn(32, 128).to(dml_device)
        y = model(x)
        loss = y.sum()
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    print("✓ Mixed precision (AMP) operations successful")
    print(f"  - AMP loss value: {loss.item():.6f}")
    print("  - Note: DirectML AMP is experimental")
except Exception as e:
    print(f"⚠ Mixed precision (AMP) operations failed: {e}")
    print("  This is not critical - training can proceed without AMP")
    print("  Use --no-amp flag when training if AMP fails")

# Test 7: Memory transfer test
print("\n[Test 7] Testing GPU<->CPU memory transfers...")
try:
    # GPU to CPU
    gpu_tensor = torch.randn(1000, 1000).to(dml_device)
    cpu_tensor = gpu_tensor.cpu()
    
    # CPU to GPU
    cpu_tensor2 = torch.randn(1000, 1000)
    gpu_tensor2 = cpu_tensor2.to(dml_device)
    
    print("✓ Memory transfers successful")
    print(f"  - GPU->CPU transfer: {cpu_tensor.shape}")
    print(f"  - CPU->GPU transfer: {gpu_tensor2.shape}")
except Exception as e:
    print(f"✗ Memory transfer failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All DirectML tests passed!")
print("=" * 60)
print("\nYou can now use DirectML for training Hierarchos:")
print("  python hierarchos.py --mode train --device dml [other args]")
print("\nOr let it auto-detect DirectML:")
print("  python hierarchos.py --mode train [other args]")
print("\nIf you encounter issues, try disabling AMP:")
print("  python hierarchos.py --mode train --no-amp [other args]")
print("=" * 60)
