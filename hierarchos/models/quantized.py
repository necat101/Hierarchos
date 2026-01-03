import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from typing import Optional, Tuple, Dict, Any
from .rwkv_cell import RWKVCell
from .ltm import LTMModule
from ..utils.device import is_directml_device, get_device_type
from torch.utils.data import Dataset, DataLoader

# Global for kernel availability
_HAS_KERNEL = False
try:
    import hierarchos_matmul
    _HAS_KERNEL = True
except ImportError:
    # print("Warning: 'hierarchos_matmul' not found. Quantization disabled.")
    _HAS_KERNEL = False

# Helper for AttrDict access (from trainer/datasets)
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_q_block_size(qtype: str) -> int:
    """Returns the block size for a given quantization type."""
    if qtype in ["INT4", "Q4_0", "Q8_0"]:
        return 32
    elif qtype == "Q2_K":
        return 256
    else:
        return 32

class QuantizedLinear:
    """A wrapper for a quantized linear layer that uses the C++ kernel for inference."""
    def __init__(self, name: str, q_data: dict):
        self.name = name
        weight_data_key = f'{name}.weight'
        bias_data_key = f'{name}.bias'

        if weight_data_key not in q_data:
            raise KeyError(f"Weight data '{weight_data_key}' not found in quantized file.")

        weight_meta = q_data[weight_data_key].item() # .item() needed for numpy object arrays
        if 'quantized' not in weight_meta:
            raise ValueError(f"Weight '{weight_data_key}' is not quantized (missing 'quantized' key).")

        self.quantized_w = weight_meta['quantized']
        self.qtype = str(weight_meta['qtype'])
        self.original_shape = weight_meta['original_shape']
        self.M, self.K = self.original_shape

        if bias_data_key in q_data:
            bias_meta = q_data[bias_data_key].item() # .item() needed
            if 'raw' not in bias_meta:
                raise ValueError(f"Bias '{bias_data_key}' is missing 'raw' data.")
            self.bias = bias_meta['raw']
        else:
            self.bias = None

    def __call__(self, x: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        if not _HAS_KERNEL: raise ImportError("C++ kernel required for quantized matmul")

        x_np = x.cpu().float().numpy()
        # Ensure x_np is 2D for matmul
        original_ndim = x_np.ndim
        original_shape = x_np.shape
        if original_ndim == 1:
            x_np = x_np.reshape(1, -1)
        elif original_ndim > 2:
            # Flatten leading dimensions if any (e.g., batch, sequence length)
            x_np = x_np.reshape(-1, x_np.shape[-1])

        # Ensure input K matches the quantized weight K (which is the original K + padding)
        padded_k = self.K
        block_size = get_q_block_size(self.qtype)
        if self.K % block_size != 0:
            padded_k += block_size - (self.K % block_size)

        if x_np.shape[-1] != padded_k:
            # Pad input if needed to match the kernel's expectation
            pad_k = padded_k - x_np.shape[-1]
            if pad_k > 0:
                x_np = np.pad(x_np, ((0, 0), (0, pad_k)), 'constant')
            elif pad_k < 0:
                x_np = x_np[..., :padded_k]

        y_np = hierarchos_matmul.matmul_quantized(x_np, self.quantized_w, self.M, self.qtype, device)

        # Output shape should match original input dimensions + output features M
        if original_ndim > 2:
            output_shape = list(original_shape[:-1]) + [self.M]
            y_np = y_np.reshape(output_shape)
        elif original_ndim == 1:
            y_np = y_np.reshape(-1) # Reshape back to 1D

        if y_np.shape[-1] != self.M:
            y_np = y_np[..., :self.M]

        if self.bias is not None: y_np += self.bias
        return torch.from_numpy(y_np)

class QuantizedRWKVCell:
    def __init__(self, n_embd, name_prefix, q_data):
        self.n_embd = n_embd
        self.key = QuantizedLinear(f'{name_prefix}.key', q_data)
        self.value = QuantizedLinear(f'{name_prefix}.value', q_data)
        self.receptance = QuantizedLinear(f'{name_prefix}.receptance', q_data)
        self.output = QuantizedLinear(f'{name_prefix}.output', q_data)
        self.key_cm = QuantizedLinear(f'{name_prefix}.key_cm', q_data)
        self.receptance_cm = QuantizedLinear(f'{name_prefix}.receptance_cm', q_data)
        self.value_cm = QuantizedLinear(f'{name_prefix}.value_cm', q_data)

        def load_raw(name):
            return torch.from_numpy(q_data[f'{name_prefix}.{name}'].item()['raw'])

        self.time_decay = load_raw('time_decay')
        self.time_first = load_raw('time_first')
        self.time_mix_k = load_raw('time_mix_k')
        self.time_mix_v = load_raw('time_mix_v')
        self.time_mix_r = load_raw('time_mix_r')
        self.time_mix_k_cm = load_raw('time_mix_k_cm')
        self.time_mix_r_cm = load_raw('time_mix_r_cm')

        self.ln1_w = load_raw('ln1.weight')
        self.ln1_b = load_raw('ln1.bias')
        self.ln2_w = load_raw('ln2.weight')
        self.ln2_b = load_raw('ln2.bias')

    def __call__(self, x, state, device="cpu"):
        # Move raw tensors to device if needed
        for p in [self.time_decay, self.time_first, self.time_mix_k, self.time_mix_v,
                  self.time_mix_r, self.time_mix_k_cm, self.time_mix_r_cm,
                  self.ln1_w, self.ln1_b, self.ln2_w, self.ln2_b]:
            if p.device.type != device:
                p.data = p.data.to(device)

        # Capture input for next Time Mixing state (Token Shift)
        x_in = x 

        sx, aa, bb, pp, sx_cm = state.unbind(dim=2)

        # --- Time mixing ---
        x_norm = F.layer_norm(x, (self.n_embd,), weight=self.ln1_w, bias=self.ln1_b)

        xk = x_norm * self.time_mix_k + sx * (1 - self.time_mix_k)
        xv = x_norm * self.time_mix_v + sx * (1 - self.time_mix_v)
        xr = x_norm * self.time_mix_r + sx * (1 - self.time_mix_r)

        r = torch.sigmoid(self.receptance(xr, device))
        k = self.key(xk, device)
        k = torch.clamp(k, max=60)
        v = self.value(xv, device)

        ww = self.time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = (e1 * aa + e2 * v) / (e1 * bb + e2 + 1e-8)
        
        # Time Mixing Output / Channel Mixing Input
        x = x + self.output(r * wkv, device)
        
        # Capture input for next Channel Mixing state
        x_tm = x 

        ww = pp + self.time_decay
        p = torch.maximum(ww, k)
        e1 = torch.exp(ww - p)
        e2 = torch.exp(k - p)
        aa = e1 * aa + e2 * v
        bb = e1 * bb + e2
        pp = p

        # --- Channel mixing ---
        x_norm2 = F.layer_norm(x, (self.n_embd,), weight=self.ln2_w, bias=self.ln2_b)

        xk = x_norm2 * self.time_mix_k_cm + sx_cm * (1 - self.time_mix_k_cm)
        xr = x_norm2 * self.time_mix_r_cm + sx_cm * (1 - self.time_mix_r_cm)
        
        r = torch.sigmoid(self.receptance_cm(xr, device))
        k = torch.square(torch.relu(self.key_cm(xk, device)))
        x = x + r * self.value_cm(k, device)

        # Update state: [x_in, aa, bb, pp, x_tm]
        new_state = torch.stack([x_in, aa, bb, pp, x_tm], dim=2)
        return x, new_state

class QuantizedHierarchos:
    """The quantized hierarchos model for CPU/Vulkan inference."""
    def __init__(self, config: dict, q_data: dict):
        if not _HAS_KERNEL:
            raise ImportError("Cannot initialize QuantizedHierarchos: C++ kernel not found.")
        
        self.config = AttrDict(config)
        
        if 'h_stride' not in self.config: self.config['h_stride'] = 4 
        if 'l_conv_atol' not in self.config: self.config['l_conv_atol'] = 1e-4

        self.qtype = None

        # Load raw parameters
        try:
            self.tok_emb = nn.Embedding.from_pretrained(torch.from_numpy(q_data['tok_emb.weight'].item()['raw']))
            self.persistent = torch.from_numpy(q_data['persistent'].item()['raw'])
            self.out_norm = nn.LayerNorm(self.config.context_dim)
            self.out_norm.load_state_dict({
                'weight': torch.from_numpy(q_data['out_norm.weight'].item()['raw']),
                'bias': torch.from_numpy(q_data['out_norm.bias'].item()['raw'])
            })
            
            self.ltm = LTMModule(n_slots=self.config.ltm_slots,
                                 key_dim=self.config.ltm_key_dim,
                                 val_dim=self.config.ltm_val_dim,
                                 reference_chunk_len=getattr(self.config, 'reference_chunk_len', 128))
                                 
            ltm_state = {}
            for k in ['ltm.keys', 'ltm.vals', 'ltm.timestamps', 'ltm.sources']:
                if k in q_data:
                    key_name = k.split('.', 1)[1]
                    ltm_state[key_name] = torch.from_numpy(q_data[k].item()['raw'])
            
            self.ltm.load_state_dict(ltm_state, strict=False)
            
            half_dim = self.config.ltm_val_dim // 2
            emb = math.log(10000) / (half_dim - 1)
            emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
            self.time_freqs = emb 

        except Exception as e:
            raise RuntimeError(f"Error loading raw parameters: {e}")

        if 'ltm_gate_logit' in q_data:
            self.ltm_gate_logit = torch.from_numpy(q_data['ltm_gate_logit'].item()['raw'])
        else:
            self.ltm_gate_logit = torch.tensor(0.0)

        expected_quantized = [
            'qproj', 'in_proj', 'h_rnn', 'h_to_context',
            'l_input_proj', 'l_rnn', 'l_to_out', 'lm_head', 'h_halt_proj',
            'context_drift_proj', 'l_feedback_proj', 'val_proj'
        ]
        quantized_layers = {}
        for layer_name in expected_quantized:
            if layer_name in ['h_rnn', 'l_rnn']:
                hidden = self.config.h_hidden if layer_name == 'h_rnn' else self.config.l_hidden
                quantized_layers[layer_name] = QuantizedRWKVCell(hidden, layer_name, q_data)
                if self.qtype is None: self.qtype = quantized_layers[layer_name].key.qtype
            else:
                if f'{layer_name}.weight' in q_data:
                    quantized_layers[layer_name] = QuantizedLinear(layer_name, q_data)
                    if self.qtype is None: self.qtype = quantized_layers[layer_name].qtype
                elif layer_name == 'context_drift_proj':
                    quantized_layers[layer_name] = None 

        self.qproj          = quantized_layers['qproj']
        self.in_proj        = quantized_layers['in_proj']
        self.h_rnn          = quantized_layers['h_rnn']
        self.h_to_context   = quantized_layers['h_to_context']
        self.l_input_proj   = quantized_layers['l_input_proj']
        self.l_rnn          = quantized_layers['l_rnn']
        self.l_to_out       = quantized_layers['l_to_out']
        self.lm_head        = quantized_layers['lm_head']
        self.h_halt_proj    = quantized_layers['h_halt_proj']
        self.context_drift_proj = quantized_layers.get('context_drift_proj')
        self.l_feedback_proj = quantized_layers.get('l_feedback_proj')
        self.val_proj = quantized_layers.get('val_proj')
        
        self.use_context_aware_query = (self.qproj.K == self.config.context_dim * 2)
        print(f"Initialized QuantizedHierarchos ({self.qtype}) with RWKV recurrence.")

    def __call__(self, input_ids: torch.LongTensor, 
                 h_state: torch.Tensor, l_state: torch.Tensor, 
                 prev_context: torch.Tensor, target_context: torch.Tensor,
                 global_pos_offset: int = 0,
                 device: str = "cpu", min_timestamp: float = 0.0, source_filter: int = None):
        
        B, T = input_ids.shape
        curr_prev_context = prev_context.to(device if device == 'vulkan' else 'cpu')
        logits = None
        stride = self.config.h_stride
        
        # --- [PORT] Context Recovery ---
        # If we have an existing h_state but contexts are zero, recover them to avoid shock
        if h_state.abs().sum() > 0:
            if prev_context.abs().sum() == 0:
                prev_context = self.h_to_context(h_state[:, :, 0].to(device), device=device).cpu()
            if target_context.abs().sum() == 0:
                target_context = self.h_to_context(h_state[:, :, 0].to(device), device=device).cpu()
                
        curr_prev_context = prev_context.to(device if device == 'vulkan' else 'cpu')
        curr_target_context = target_context.to(device if device == 'vulkan' else 'cpu')
        
        all_topk_vals, all_topk_idx = [], []

        for t in range(T):
            abs_t = global_pos_offset + t
            token_ids = input_ids[:, t].cpu().long()
            token_emb = self.tok_emb(token_ids) 
            p_read = self.persistent.unsqueeze(0).expand(B, -1)

            if self.use_context_aware_query:
                q_in = torch.cat([token_emb.to(device), curr_prev_context.to(device)], dim=-1)
                query = self.qproj(q_in, device=device)
            else:
                query = self.qproj(token_emb.to(device), device=device)
            
            query = torch.clamp(query, min=-10.0, max=10.0)
            topk_vals, topk_idx, topk_ts = self.ltm.retrieve_topk(query, topk=self.config.ltm_topk, 
                                                           min_timestamp=min_timestamp, 
                                                           source_filter=source_filter)
            all_topk_vals.append(topk_vals); all_topk_idx.append(topk_idx)
            
            args = topk_ts.unsqueeze(-1) * self.time_freqs.to(device).unsqueeze(0).unsqueeze(0)
            pe = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            if self.config.ltm_val_dim % 2 == 1: pe = torch.cat([pe, torch.zeros_like(pe[..., :1])], dim=-1)
            topk_vals = topk_vals + pe
            
            gate = torch.sigmoid(torch.clamp(self.ltm_gate_logit.to(device), min=-50.0, max=50.0))
            topk_vals = topk_vals * gate
            ltm_summary = topk_vals.view(B, -1)

            mac_input = torch.cat([token_emb.to(device), p_read.to(device), ltm_summary], dim=-1)
            enc = F.gelu(self.in_proj(mac_input, device=device))
            enc = torch.clamp(enc, min=-30.0, max=30.0)

            if self.l_feedback_proj is not None:
                l_feedback = self.l_feedback_proj(l_state[:, :, 0].to(device), device=device)
                enc_with_feedback = enc + l_feedback
            else:
                enc_with_feedback = enc
            h_out_real, h_state = self.h_rnn(enc_with_feedback, h_state, device=device)
            h_out_real = torch.clamp(h_out_real, min=-100.0, max=100.0)
            
            if torch.isnan(h_out_real).any() or torch.isinf(h_out_real).any():
                print(f"WARNING: NaN/Inf detected in h_out_real during inference at step {t}")

            if abs_t % stride == 0:
                curr_prev_context = curr_target_context
                h_step_outputs = [h_out_real]
                h_halt_probs = [torch.sigmoid(self.h_halt_proj(h_out_real, device=device).squeeze(-1))]
                shadow_h_state = h_state.clone()
                for step_idx in range(self.config.max_h_steps - 1):
                    if h_halt_probs[-1].mean() > getattr(self.config, 'h_halt_thresh', 0.9): break
                    h_out_ponder, shadow_h_state = self.h_rnn(enc_with_feedback, shadow_h_state, device=device)
                    h_step_outputs.append(h_out_ponder)
                    h_halt_probs.append(torch.sigmoid(self.h_halt_proj(h_out_ponder, device=device).squeeze(-1)))

                h_stack = torch.stack(h_step_outputs, dim=0)
                halt_stack = torch.stack(h_halt_probs, dim=0)
                remain = 1.0 - halt_stack
                remain_shifted = torch.cat([torch.ones_like(remain[:1]), remain[:-1]], dim=0)
                cum_remain = torch.cumprod(remain_shifted, dim=0)
                weights, remainder = halt_stack * cum_remain, cum_remain[-1] * (1.0 - halt_stack[-1])
                total = weights.sum(dim=0) + remainder + 1e-8
                weights, remainder = weights / total, remainder / total
                final_h_out = (weights.unsqueeze(-1) * h_stack).sum(dim=0) + remainder.unsqueeze(-1) * h_stack[-1]
                curr_target_context = self.h_to_context(final_h_out, device=device)
                curr_target_context = torch.clamp(curr_target_context, min=-50.0, max=50.0)
            
            alpha = (abs_t % stride) / float(stride)
            static_context = curr_prev_context + (curr_target_context - curr_prev_context) * alpha

            if self.context_drift_proj is not None:
                current_drift = torch.clamp(torch.tanh(self.context_drift_proj(l_state[:, :, 0].to(device), device=device)), min=-5.0, max=5.0)
            else:
                current_drift = torch.zeros_like(static_context)
            
            shadow_l_state = l_state.clone()
            for _ in range(self.config.max_l_steps):
                l_input = self.l_input_proj(torch.cat([enc.to(device), (static_context + current_drift).to(device)], dim=-1), device=device)
                l_out, shadow_l_state = self.l_rnn(l_input, shadow_l_state, device=device)
                if self.context_drift_proj is not None:
                    drift_delta = torch.tanh(self.context_drift_proj(l_out, device=device))
                    current_drift = torch.clamp(current_drift + drift_delta, min=-5.0, max=5.0)
                    if torch.mean(torch.abs(drift_delta)) < self.config.l_conv_atol: break
                else: break
            
            l_input = self.l_input_proj(torch.cat([enc.to(device), (static_context + current_drift).to(device)], dim=-1), device=device)
            l_out, l_state = self.l_rnn(l_input, l_state, device=device)
            l_state = torch.clamp(l_state, min=-50.0, max=50.0)
            enc = enc + self.l_to_out(l_out, device=device)
            logits = self.lm_head(self.out_norm(enc.cpu()), device=device)

            if self.val_proj is not None:
                 val_to_store = self.val_proj(enc.to(device), device=device) if hasattr(self.val_proj, 'qtype') else self.val_proj(enc.to(device))
                 val_expanded = torch.clamp(val_to_store, min=-20.0, max=20.0).unsqueeze(1).expand(-1, self.config.ltm_topk, -1)
                 self.ltm.update_memory_hebbian(topk_idx, None, val_expanded, current_lr=self.config.ltm_lr, timestamp=float(abs_t), tokens_covered=1, inplace=True)

        return {
            "logits": logits.unsqueeze(1) if logits is not None else None,
            "h_state": h_state.cpu(), "l_state": l_state.cpu(),
            "prev_context": curr_prev_context.cpu(), "target_context": curr_target_context.cpu(),
            "drift_state": current_drift.cpu()
        }

def load_quantized(model_path: str, device=None):
    if device and is_directml_device(device):
        from ..utils.checkpoint import load_full_model_with_config
        return load_full_model_with_config(model_path, device)
    if not _HAS_KERNEL: raise ImportError("Cannot load quantized model: C++ kernel not found.")
    npz_files = [f for f in os.listdir(model_path) if f.endswith('.npz')]
    if not npz_files: raise FileNotFoundError(f"No quantized model .npz file found in {model_path}")
    q_data = np.load(os.path.join(model_path, npz_files[0]), allow_pickle=True)
    config = AttrDict(q_data['_config'].item())
    if 'model_type' not in config: config['model_type'] = 'hierarchos'
    return QuantizedHierarchos(config, q_data), config
