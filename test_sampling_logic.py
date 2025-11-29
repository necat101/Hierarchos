import torch
import torch.nn.functional as F
import argparse

def sample_token(logits, temperature=1.0, top_k=0, top_p=1.0):
    """
    Standalone sampling logic matching the implementation in hierarchos.py
    """
    next_token_logits = logits.clone()
    
    # Apply temperature
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    
    # Apply Top-K
    if top_k > 0:
        v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
        next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')

    # Apply Top-P (Nucleus Sampling)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        next_token_logits[indices_to_remove] = -float('Inf')

    # Sample or Greedy
    if temperature > 0:
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1)
    else:
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
    return next_token_id

def test_sampling():
    print("Testing Sampling Logic...")
    torch.manual_seed(42)
    
    # Create dummy logits [1, 10]
    # Token 0: very high, Token 1: high, others: low
    logits = torch.tensor([[10.0, 8.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.1, 0.1, 0.1]])
    
    # 1. Test Greedy (Temperature = 0)
    token = sample_token(logits, temperature=0.0)
    print(f"Greedy (Temp=0): {token.item()} (Expected: 0)")
    assert token.item() == 0, "Greedy sampling failed"

    # 2. Test Temperature Scaling
    # High temp -> flatter distribution -> more random
    # Low temp -> sharper distribution -> more deterministic
    # We can't easily assert randomness, but we can check if it runs.
    token = sample_token(logits, temperature=2.0)
    print(f"High Temp (2.0): {token.item()}")
    
    token = sample_token(logits, temperature=0.1)
    print(f"Low Temp (0.1): {token.item()} (Likely 0)")
    
    # 3. Test Top-K
    # Set Top-K=1 -> Should be greedy equivalent
    token = sample_token(logits, temperature=1.0, top_k=1)
    print(f"Top-K=1: {token.item()} (Expected: 0)")
    assert token.item() == 0, "Top-K=1 failed"
    
    # Set Top-K=2 -> Should be 0 or 1
    # Force logits to be equal for 0 and 1 to test randomness if we wanted, 
    # but here just check it's in set.
    token = sample_token(logits, temperature=1.0, top_k=2)
    print(f"Top-K=2: {token.item()} (Expected: 0 or 1)")
    assert token.item() in [0, 1], "Top-K=2 failed"

    # 4. Test Top-P
    # Top-P=0.0 -> Should be greedy (only top token) - wait, implementation keeps at least one.
    # Actually logic: "keep also the first token above the threshold"
    # If top token prob > top_p, it's kept.
    probs = F.softmax(logits, dim=-1)
    print(f"Probs: {probs}")
    # Token 0 has prob ~0.88, Token 1 ~0.11
    
    # Top-P = 0.5 -> Should only keep Token 0
    token = sample_token(logits, temperature=1.0, top_p=0.5)
    print(f"Top-P=0.5: {token.item()} (Expected: 0)")
    assert token.item() == 0, "Top-P=0.5 failed"
    
    # Top-P = 0.95 -> Should keep Token 0 and 1
    token = sample_token(logits, temperature=1.0, top_p=0.95)
    print(f"Top-P=0.95: {token.item()} (Expected: 0 or 1)")
    assert token.item() in [0, 1], "Top-P=0.95 failed"

    print("All sampling tests passed!")

if __name__ == "__main__":
    test_sampling()
