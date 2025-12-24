import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RWKVCell(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        decay_speed = torch.arange(0, n_embd) / n_embd
        self.time_decay = nn.Parameter(-5 + 4 * decay_speed) 
        self.time_first = nn.Parameter(torch.ones(n_embd) * 0.5)
        
        curve = torch.arange(0, n_embd) / n_embd
        curve = torch.pow(curve, 0.5) 

        self.time_mix_k = nn.Parameter(curve.view(1, 1, n_embd))
        self.time_mix_v = nn.Parameter(curve.view(1, 1, n_embd) + 0.1 * torch.randn(1, 1, n_embd)) 
        self.time_mix_r = nn.Parameter(0.5 * curve.view(1, 1, n_embd)) 

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)

        self.time_mix_k_cm = nn.Parameter(torch.ones(1, 1, n_embd) * 0.05)
        self.time_mix_r_cm = nn.Parameter(torch.ones(1, 1, n_embd) * 0.05)

        self.key_cm = nn.Linear(n_embd, n_embd * 4, bias=False)
        self.receptance_cm = nn.Linear(n_embd, n_embd, bias=False)
        self.value_cm = nn.Linear(n_embd * 4, n_embd, bias=False)

    def forward(self, x, state, timestep=None):
        # Handle torch.compile artifacts
        if x.dim() == 3 and x.shape[0] == 1: x = x.squeeze(0)
        if state.dim() == 4 and state.shape[0] == 1: state = state.squeeze(0)

        # Truncated BPTT logic
        detach_every_n_steps = getattr(self, 'detach_every_n_steps', None)
        if self.training and detach_every_n_steps is not None and timestep is not None:
            if timestep > 0 and timestep % detach_every_n_steps == 0:
                state = state.detach()

        x_resid_tm = x 
        x_norm = self.ln1(x)
        x_in = x_norm 

        tm_k = self.time_mix_k.view(-1)
        tm_v = self.time_mix_v.view(-1)
        tm_r = self.time_mix_r.view(-1)
        tm_k_cm = self.time_mix_k_cm.view(-1)
        tm_r_cm = self.time_mix_r_cm.view(-1)

        # Unbind state: Slot 0 (sx) is the previous timestep's input
        sx, aa, bb, pp, sx_cm = state.unbind(dim=-1)

        # Time mixing
        xk = x_norm * tm_k + sx * (1 - tm_k)
        xv = x_norm * tm_v + sx * (1 - tm_v)
        xr = x_norm * tm_r + sx * (1 - tm_r)

        r = torch.sigmoid(self.receptance(xr))
        k = self.key(xk)
        k = torch.clamp(k, max=60) 
        v = self.value(xv)

        # WKV Calculation (Float32 Stability)
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            k_f, v_f, pp_f, aa_f, bb_f = k.float(), v.float(), pp.float(), aa.float(), bb.float()
            time_first_f = self.time_first.float()
            time_decay_f = self.time_decay.float()

            ww = time_first_f + k_f
            ww = torch.clamp(ww, max=30.0)
            pp_f = torch.clamp(pp_f, max=30.0)
            
            p = torch.maximum(pp_f, ww)
            e1 = torch.exp(pp_f - p)
            e2 = torch.exp(ww - p)
            
            wkv = (e1 * aa_f + e2 * v_f) / (e1 * bb_f + e2 + 1e-8)
            wkv = wkv.to(dtype=x.dtype)

            # Update State
            ww = pp_f + time_decay_f
            ww = torch.clamp(ww, max=30.0)
            
            p = torch.maximum(ww, k_f)
            e1 = torch.exp(ww - p)
            e2 = torch.exp(k_f - p)
            aa = (e1 * aa_f + e2 * v_f).to(dtype=x.dtype)
            bb = (e1 * bb_f + e2).to(dtype=x.dtype)
            pp = p.to(dtype=x.dtype)

        x = x_resid_tm + self.output(r * wkv)
        
        # Channel mixing
        x_resid_cm = x
        x_norm2 = self.ln2(x)
        xk = x_norm2 * tm_k_cm + sx_cm * (1 - tm_k_cm)
        xr = x_norm2 * tm_r_cm + sx_cm * (1 - tm_r_cm)
        
        r = torch.sigmoid(self.receptance_cm(xr))
        k = torch.square(torch.relu(self.key_cm(xk)))
        x = x_resid_cm + r * self.value_cm(k)

        new_state = torch.stack([x_in, aa, bb, pp, x_norm2], dim=-1)
        new_state = torch.clamp(new_state, min=-50.0, max=50.0)

        return x, new_state
