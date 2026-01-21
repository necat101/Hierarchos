"""Quick test to check model inference behavior."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')
from hierarchos import load_full_model_with_config
from transformers import AutoTokenizer

# Load model and tokenizer
print("Loading model...")
model, config = load_full_model_with_config('./rog_ally_model/hierarchos_v1RC', 'cpu')
tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2')

# Test prompt
prompt = '### Instruction:\nWhat is 2+2?\n\n### Response:\n'
input_ids = tokenizer.encode(prompt, return_tensors='pt')

print(f'Input tokens: {input_ids.shape}')
print(f'Prompt: {repr(prompt)}\n')

# Run generation
model.eval()
temperature = 0.5  # Lower for more focused output
max_tokens = 50
generated = []

h_state, l_state, prev_ctx, target_ctx, drift_state = None, None, None, None, None

current_ids = input_ids
print("Generating response:")
print("="*50)

with torch.no_grad():
    for step in range(max_tokens):
        outputs = model(
            current_ids,
            h_state=h_state,
            l_state=l_state,
            prev_context=prev_ctx,
            target_context=target_ctx,
            drift_state=drift_state,
            global_pos_offset=len(generated) + input_ids.shape[1]
        )
        
        # Update states
        h_state = outputs.get('h_state')
        l_state = outputs.get('l_state')
        prev_ctx = outputs.get('prev_context')
        target_ctx = outputs.get('target_context')
        drift_state = outputs.get('drift_state')
        
        # Get next token
        logits = outputs['logits'][:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        if next_token.item() == tokenizer.eos_token_id:
            print(" [EOS]")
            break
            
        decoded = tokenizer.decode([next_token.item()])
        print(decoded, end='', flush=True)
        generated.append(next_token.item())
        
        current_ids = next_token
        
        # Check for repetition
        if len(generated) > 10:
            last_5 = generated[-5:]
            prev_5 = generated[-10:-5]
            if last_5 == prev_5:
                print(" [REPETITION DETECTED]")
                break

print("\n" + "="*50)
print(f"Generated {len(generated)} tokens")
print(f"\nFull response: {tokenizer.decode(generated)}")

# Check state magnitudes
print(f"\nState diagnostics:")
if h_state is not None:
    print(f"  h_state: min={h_state.min().item():.3f}, max={h_state.max().item():.3f}, mean={h_state.mean().item():.3f}")
if l_state is not None:
    print(f"  l_state: min={l_state.min().item():.3f}, max={l_state.max().item():.3f}, mean={l_state.mean().item():.3f}")
if drift_state is not None:
    print(f"  drift_state: min={drift_state.min().item():.3f}, max={drift_state.max().item():.3f}")

