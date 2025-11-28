import torch
import torch.nn as nn
import sys

def test_embedding(device_name):
    print(f"Testing Embedding on {device_name}...")
    device = torch.device(device_name)
    
    vocab_size = 100
    dim = 32
    batch_size = 4
    seq_len = 10
    
    # Create Embedding layer
    emb = nn.Embedding(vocab_size, dim).to(device)
    print(f"Embedding layer created on {device}.")
    
    # Create input
    # Try creating on CPU first then move, as we know direct creation might fail
    input_cpu = torch.randint(0, vocab_size, (batch_size, seq_len))
    input_ids = input_cpu.to(device)
    print(f"Input tensor created: {input_ids.shape} on {input_ids.device}")
    
    try:
        out = emb(input_ids)
        print(f"Forward pass successful. Output shape: {out.shape}")
        return True
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
        return False

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("--- Testing CUDA (ZLUDA) ---")
        if not test_embedding("cuda"):
            print("\n--- Testing Workaround: Embedding on CPU ---")
            try:
                device = torch.device("cuda")
                emb_cpu = nn.Embedding(100, 32).to("cpu")
                input_ids = torch.randint(0, 100, (4, 10)).to(device)
                print("Input on CUDA, Embedding on CPU...")
                # PyTorch might not support this directly without input on CPU
                # But let's check if we can move input to CPU for the lookup
                input_cpu_fallback = input_ids.cpu()
                out = emb_cpu(input_cpu_fallback)
                print(f"CPU Fallback successful. Output shape: {out.shape}")
                
                # Move result back to device
                out_device = out.to(device)
                print(f"Moved output back to {device}. Success.")
            except Exception as e:
                print(f"CPU Fallback FAILED: {e}")

if __name__ == "__main__":
    main()
