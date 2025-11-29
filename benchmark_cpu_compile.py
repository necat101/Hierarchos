import torch
import time
from hierarchos import HierarchosCore

def benchmark_cpu_compile():
    print("=== Benchmark: CPU Training with torch.compile ===")
    
    # Force compile=True in config
    config = dict(
        vocab_size=100,
        context_dim=64,
        persistent_dim=32,
        ltm_slots=100,
        ltm_key_dim=32,
        ltm_val_dim=32,
        ltm_topk=2,
        ltm_lr=0.01,
        h_hidden=64,
        l_hidden=64,
        max_h_steps=2,
        max_l_steps=4,
        l_conv_atol=1e-4,
        compile=True, # Enable compilation
        force_compile=True, # Override Windows CPU safety check
        device='cpu'
    )
    
    print("Initializing model...")
    model = HierarchosCore(config)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    input_ids = torch.randint(0, 100, (1, 32)) # Batch 1, Seq 32
    labels = input_ids.clone() # Self-supervised
    
    print("Starting training loop (5 steps)...")
    start_time = time.time()
    
    for i in range(5):
        step_start = time.time()
        optimizer.zero_grad()
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        step_end = time.time()
        print(f"Step {i+1}: Loss={loss.item():.4f}, Time={step_end - step_start:.4f}s")
        
    total_time = time.time() - start_time
    print(f"Total Time: {total_time:.4f}s")
    print("[PASS] Benchmark completed without hang.")

if __name__ == "__main__":
    benchmark_cpu_compile()
