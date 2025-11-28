import torch
import torch.nn.functional as F

def test_nan_loss():
    print("Testing CrossEntropyLoss with all-masked labels...")
    logits = torch.randn(1, 10, 100) # B, T, V
    labels = torch.full((1, 10), -100, dtype=torch.long) # All -100
    
    loss = F.cross_entropy(logits.view(-1, 100), labels.view(-1))
    print(f"Loss with all -100 labels: {loss}")
    
    if torch.isnan(loss):
        print("CONFIRMED: Loss is NaN when all labels are -100.")
    else:
        print("Loss is NOT NaN.")

if __name__ == "__main__":
    test_nan_loss()
