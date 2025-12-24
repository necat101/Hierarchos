import torch
import torch.nn.functional as F
import time
from ..utils.device import is_directml_device

def is_positive_feedback(text):
    return any(term in text.lower() for term in ['good', 'great', 'yes', 'correct', 'nice', 'accurate'])

def generate_sample(model, tokenizer, prompt, device, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9):
    model.eval()
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    h_state, l_state, p_ctx, t_ctx = None, None, None, None
    ltm_state = None
    
    generated = tokens
    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=tokens, h_state=h_state, l_state=l_state, 
                            prev_context=p_ctx, target_context=t_ctx, ltm_memory_state=ltm_state)
            logits = outputs['logits'][:, -1, :] / temperature
            # Apply sampling
            # (Simplified sampling for brevity)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            tokens = next_token
            h_state, l_state, p_ctx, t_ctx = outputs['h_state'], outputs['l_state'], outputs['prev_context'], outputs['target_context']
            ltm_state = outputs['ltm_memory_state']
            if next_token.item() == tokenizer.eos_token_id: break
            
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def chat(model, tokenizer, device, args):
    print("\n--- Hierarchos Interactive Chat ---")
    print("Type 'exit' to quit. Type 'reset' to clear context.")
    
    h_state, l_state, p_ctx, t_ctx = None, None, None, None
    ltm_state = None
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit': break
        if user_input.lower() == 'reset':
            h_state, l_state, p_ctx, t_ctx, ltm_state = None, None, None, None, None
            print("Context reset.")
            continue
            
        prompt = f"### Instruction:\n{user_input}\n\n### Response:\n"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generation loop with online learning possibility
        print("Assistant: ", end="", flush=True)
        response_tokens = []
        for _ in range(args.max_new_tokens):
            with torch.no_grad():
                outputs = model(input_ids=input_ids, h_state=h_state, l_state=l_state,
                                prev_context=p_ctx, target_context=t_ctx, ltm_memory_state=ltm_state)
                logits = outputs['logits'][:, -1, :] / args.temperature
                prob = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(prob, 1)
                
                word = tokenizer.decode(next_token[0])
                print(word, end="", flush=True)
                response_tokens.append(next_token.item())
                
                input_ids = next_token
                h_state, l_state, p_ctx, t_ctx = outputs['h_state'], outputs['l_state'], outputs['prev_context'], outputs['target_context']
                ltm_state = outputs['ltm_memory_state']
                if next_token.item() == tokenizer.eos_token_id: break
        print()
