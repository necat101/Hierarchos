import torch
import sys

def test_op(name, func, tensor):
    print(f"Testing {name}...", end=" ")
    try:
        func(tensor)
        print("OK")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        return False

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n--- Setup ---")
    # Try creating on device directly (zeros)
    try:
        print("Attempting torch.zeros(..., device=device)...", end=" ")
        t = torch.zeros(1024, 64, device=device)
        print("OK")
    except Exception as e:
        print(f"FAILED: {e}")
        print("Fallback: Creating on CPU and moving to device...")
        t = torch.zeros(1024, 64).to(device)
    
    # Fill with random data (on CPU then move, to be safe)
    t.copy_(torch.randn(1024, 64).to(device))
    
    print("\n--- Testing In-Place Zeroing Operations ---")
    
    # Test 1: fill_(0)
    try:
        t1 = t.clone()
        test_op("fill_(0)", lambda x: x.fill_(0), t1)
    except Exception as e:
        print(f"Clone failed? {e}")

    # Test 2: zero_()
    try:
        t2 = t.clone()
        test_op("zero_()", lambda x: x.zero_(), t2)
    except: pass
    
    # Test 3: data.fill_(0)
    try:
        t3 = t.clone()
        test_op("data.fill_(0)", lambda x: x.data.fill_(0), t3)
    except: pass
    
    # Test 4: copy_(zeros)
    try:
        t4 = t.clone()
        zeros = torch.zeros_like(t4)
        test_op("copy_(zeros)", lambda x: x.copy_(zeros), t4)
    except: pass
    
    # Test 5: slicing [:] = 0
    try:
        t5 = t.clone()
        test_op("[:]=0", lambda x: x.__setitem__(slice(None), 0), t5)
    except: pass
    
    # Test 6: mul_(0)
    try:
        t6 = t.clone()
        test_op("mul_(0)", lambda x: x.mul_(0), t6)
    except: pass

    print("\nDone.")

if __name__ == "__main__":
    main()
