"""
LM-Evaluation-Harness Wrapper for Hierarchos models.

This module provides a wrapper class that makes HierarchosCore compatible with
EleutherAI's lm-evaluation-harness for standardized benchmark evaluation.
"""
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from tqdm import tqdm

try:
    from lm_eval.api.model import LM
    from lm_eval.api.instance import Instance
    _HAS_LM_EVAL = True
except ImportError:
    _HAS_LM_EVAL = False
    # Create dummy classes if lm-eval not installed
    class LM:
        pass
    class Instance:
        pass


class HierarchosLM(LM):
    """
    lm-evaluation-harness compatible wrapper for HierarchosCore models.
    
    Implements the three core methods required by lm-eval:
    - loglikelihood: Compute log-prob for context+continuation
    - loglikelihood_rolling: Compute perplexity for entire sequences
    - generate_until: Generate text until stop condition
    
    Usage:
        from hierarchos.evaluation import HierarchosLM
        
        lm = HierarchosLM(model, tokenizer, device, batch_size=4)
        results = lm_eval.simple_evaluate(model=lm, tasks=["hellaswag"])
    """
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        device: torch.device,
        batch_size: int = 1,
        max_length: Optional[int] = None
    ):
        """
        Initialize the wrapper.
        
        Args:
            model: HierarchosCore model instance
            tokenizer: Tokenizer (HuggingFace compatible)
            device: torch.device to run inference on
            batch_size: Batch size for processing requests
            max_length: Optional max sequence length (uses model config if None)
        """
        if not _HAS_LM_EVAL:
            raise ImportError(
                "lm-evaluation-harness is not installed. "
                "Install it with: pip install lm-eval>=0.4.0"
            )
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._batch_size = batch_size
        self._max_length = max_length or getattr(model.config, 'max_length', 1024)
        
        # Cache the eot token id
        if tokenizer.eos_token_id is not None:
            self._eot_token_id = tokenizer.eos_token_id
        else:
            self._eot_token_id = tokenizer.pad_token_id or 0
    
    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @property
    def max_length(self) -> int:
        return self._max_length
    
    @property
    def eot_token_id(self) -> int:
        return self._eot_token_id
    
    @property  
    def device(self) -> torch.device:
        return self._device
    
    @device.setter
    def device(self, value: torch.device):
        self._device = value
    
    def tok_encode(self, string: str) -> List[int]:
        """Encode a string to token ids."""
        return self.tokenizer.encode(string, add_special_tokens=False)
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the model and return logits.
        
        Args:
            input_ids: Input token ids [batch, seq_len]
            
        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids.to(self.device))
            return outputs['logits']
    
    def loglikelihood(
        self, 
        requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuation given context.
        
        Each request contains (context, continuation) and we compute:
        - The log probability of the continuation given the context
        - Whether the continuation is the greedy choice
        
        Args:
            requests: List of Instance objects with args=(context, continuation)
            
        Returns:
            List of (log_prob, is_greedy) tuples
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(requests), self.batch_size), 
                      desc="loglikelihood", disable=len(requests) < 10):
            batch_requests = requests[i:i + self.batch_size]
            
            batch_results = []
            for req in batch_requests:
                context, continuation = req.args
                
                # Tokenize
                context_enc = self.tok_encode(context)
                continuation_enc = self.tok_encode(continuation)
                
                # Combine and truncate if needed
                full_enc = context_enc + continuation_enc
                if len(full_enc) > self.max_length:
                    # Truncate from the left (keep continuation)
                    full_enc = full_enc[-(self.max_length):]
                    context_enc = full_enc[:-len(continuation_enc)]
                
                input_ids = torch.tensor([full_enc], dtype=torch.long, device=self.device)
                
                # Get logits
                logits = self._model_call(input_ids)
                
                # Compute log-probs for continuation tokens
                # logits[0, context_len-1:, :] predicts tokens at positions context_len onwards
                cont_start = len(context_enc) - 1
                cont_logits = logits[0, cont_start:-1, :]  # [cont_len, vocab]
                cont_targets = input_ids[0, len(context_enc):]  # [cont_len]
                
                # Log softmax
                log_probs = F.log_softmax(cont_logits.float(), dim=-1)
                
                # Gather log probs for actual tokens
                target_log_probs = log_probs.gather(
                    dim=-1, 
                    index=cont_targets.unsqueeze(-1)
                ).squeeze(-1)
                
                # Total log probability
                total_log_prob = target_log_probs.sum().item()
                
                # Check if greedy (each predicted token is the argmax)
                greedy_tokens = cont_logits.argmax(dim=-1)
                is_greedy = (greedy_tokens == cont_targets).all().item()
                
                batch_results.append((total_log_prob, is_greedy))
            
            results.extend(batch_results)
        
        return results
    
    def loglikelihood_rolling(
        self, 
        requests: List[Instance]
    ) -> List[Tuple[float, bool]]:
        """
        Compute rolling log-likelihood (perplexity) for entire sequences.
        
        This is used for perplexity evaluation on datasets like WikiText.
        
        Args:
            requests: List of Instance objects with args=(sequence,)
            
        Returns:
            List of (log_prob, True) tuples (is_greedy always True for rolling)
        """
        results = []
        
        for i in tqdm(range(0, len(requests), self.batch_size),
                      desc="loglikelihood_rolling", disable=len(requests) < 10):
            batch_requests = requests[i:i + self.batch_size]
            
            for req in batch_requests:
                (sequence,) = req.args
                
                # Tokenize
                tokens = self.tok_encode(sequence)
                
                if len(tokens) == 0:
                    results.append((0.0, True))
                    continue
                
                # Truncate if too long
                if len(tokens) > self.max_length:
                    tokens = tokens[:self.max_length]
                
                input_ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                
                # Get logits
                logits = self._model_call(input_ids)
                
                # Compute log-probs for all tokens (conditioned on previous)
                # logits[0, :-1, :] predicts tokens at positions 1 onwards
                pred_logits = logits[0, :-1, :]
                targets = input_ids[0, 1:]
                
                log_probs = F.log_softmax(pred_logits.float(), dim=-1)
                target_log_probs = log_probs.gather(
                    dim=-1,
                    index=targets.unsqueeze(-1)
                ).squeeze(-1)
                
                total_log_prob = target_log_probs.sum().item()
                results.append((total_log_prob, True))
        
        return results
    
    def generate_until(
        self, 
        requests: List[Instance]
    ) -> List[str]:
        """
        Generate text until a stop condition is met.
        
        This is used for generative tasks like question answering.
        
        Args:
            requests: List of Instance objects with args=(context, gen_kwargs)
                      gen_kwargs may contain: until, max_gen_toks, temperature, etc.
            
        Returns:
            List of generated strings (continuation only, not including context)
        """
        results = []
        
        for req in tqdm(requests, desc="generate_until", disable=len(requests) < 10):
            context, gen_kwargs = req.args
            
            # Parse generation kwargs
            until = gen_kwargs.get("until", [self.tokenizer.eos_token or "</s>"])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", 128)
            temperature = gen_kwargs.get("temperature", 0.0)  # 0 = greedy
            
            # Tokenize context
            context_enc = self.tok_encode(context)
            
            # Truncate context if needed
            if len(context_enc) > self.max_length - max_gen_toks:
                context_enc = context_enc[-(self.max_length - max_gen_toks):]
            
            input_ids = torch.tensor([context_enc], dtype=torch.long, device=self.device)
            
            # Generate tokens autoregressively
            generated = []
            self.model.eval()
            
            # State management for Hierarchos
            h_state = None
            l_state = None
            prev_context = None
            target_context = None
            drift_state = None
            ltm_state = None
            
            with torch.no_grad():
                # Prefill with context
                outputs = self.model(input_ids=input_ids)
                h_state = outputs.get('h_state')
                l_state = outputs.get('l_state')
                prev_context = outputs.get('prev_context')
                target_context = outputs.get('target_context')
                drift_state = outputs.get('drift_state')
                ltm_state = outputs.get('ltm_memory_state')
                
                # Get last logits for next token prediction
                logits = outputs['logits'][0, -1, :]
                
                for _ in range(max_gen_toks):
                    # Sample or greedy
                    if temperature <= 0 or temperature < 1e-4:
                        next_token = logits.argmax(dim=-1)
                    else:
                        probs = F.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    generated.append(next_token.item())
                    
                    # Check stop conditions
                    gen_text = self.tok_decode(generated)
                    should_stop = False
                    for stop_str in until:
                        if stop_str in gen_text:
                            # Truncate at stop string
                            gen_text = gen_text.split(stop_str)[0]
                            should_stop = True
                            break
                    
                    if should_stop or next_token.item() == self._eot_token_id:
                        break
                    
                    # Next step
                    next_input = next_token.unsqueeze(0).unsqueeze(0)
                    outputs = self.model(
                        input_ids=next_input,
                        h_state=h_state,
                        l_state=l_state,
                        prev_context=prev_context,
                        target_context=target_context,
                        drift_state=drift_state,
                        ltm_memory_state=ltm_state,
                        global_pos_offset=len(context_enc) + len(generated) - 1
                    )
                    h_state = outputs.get('h_state')
                    l_state = outputs.get('l_state')
                    prev_context = outputs.get('prev_context')
                    target_context = outputs.get('target_context')
                    drift_state = outputs.get('drift_state')
                    ltm_state = outputs.get('ltm_memory_state')
                    logits = outputs['logits'][0, -1, :]
            
            # Decode final result
            gen_text = self.tok_decode(generated)
            
            # Truncate at first stop string
            for stop_str in until:
                if stop_str in gen_text:
                    gen_text = gen_text.split(stop_str)[0]
                    break
            
            results.append(gen_text)
        
        return results
