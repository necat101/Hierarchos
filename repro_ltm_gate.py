import torch
import torch.nn as nn
import numpy as np

# Mock QuantizedHierarchos to test logic without kernel
class MockQuantizedHierarchos:
    def __init__(self, q_data):
        self.config = {'ltm_topk': 2, 'ltm_val_dim': 4, 'h_stride': 1, 'max_l_steps': 1}
        
        # Load LTM Gate Logit (Logic copied from hierarchos.py)
        if 'ltm_gate_logit' in q_data:
            self.ltm_gate_logit = torch.from_numpy(q_data['ltm_gate_logit'].item()['raw'])
        else:
            print("Warning: 'ltm_gate_logit' not found in quantized weights. Initializing to 0.0 (open gate).")
            self.ltm_gate_logit = torch.tensor(0.0) # Scalar 0.0
            
    def apply_gate(self, topk_vals):
        # Logic copied from hierarchos.py
        gate_input = torch.clamp(self.ltm_gate_logit, min=-50.0, max=50.0)
        gate = torch.sigmoid(gate_input)
        return topk_vals * gate

def test_ltm_gate():
    print("Testing LTM Gating Logic...")
    
    # 1. Test Fallback (No gate in data)
    print("\n--- Test 1: Fallback (No gate in data) ---")
    q_data_empty = {}
    model_fallback = MockQuantizedHierarchos(q_data_empty)
    print(f"Gate Logit: {model_fallback.ltm_gate_logit}")
    assert model_fallback.ltm_gate_logit.item() == 0.0, "Fallback init failed"
    
    vals = torch.ones(1, 2, 4) # [B, K, D]
    gated_vals = model_fallback.apply_gate(vals)
    # Sigmoid(0) = 0.5
    print(f"Input Vals: {vals[0,0,0]}")
    print(f"Gated Vals: {gated_vals[0,0,0]}")
    assert torch.allclose(gated_vals, vals * 0.5), "Fallback gating math failed"

    # 2. Test Loaded Gate (High value -> Open)
    print("\n--- Test 2: Loaded Gate (High value) ---")
    # Mock numpy structure for q_data
    # q_data['ltm_gate_logit'].item()['raw']
    raw_gate = np.array(10.0, dtype=np.float32)
    item_dict = {'raw': raw_gate}
    # Mock the .item() behavior using a simple class or just a dict wrapper if numpy allows
    # Numpy arrays have .item(), but here q_data is a dict where values are 0-d arrays containing dicts?
    # Based on export code: np.savez_compressed(..., ltm_gate_logit={'raw': ...})
    # So q_data['ltm_gate_logit'] is an array containing a dict.
    
    # Let's simulate the structure exactly
    gate_container = np.array(item_dict) # 0-d array containing dict
    q_data_loaded = {'ltm_gate_logit': gate_container}
    
    model_loaded = MockQuantizedHierarchos(q_data_loaded)
    print(f"Gate Logit: {model_loaded.ltm_gate_logit}")
    assert model_loaded.ltm_gate_logit.item() == 10.0, "Loading gate failed"
    
    gated_vals = model_loaded.apply_gate(vals)
    # Sigmoid(10) ~= 0.99995
    expected_gate = torch.sigmoid(torch.tensor(10.0))
    print(f"Expected Gate: {expected_gate}")
    print(f"Gated Vals: {gated_vals[0,0,0]}")
    assert torch.allclose(gated_vals, vals * expected_gate), "Loaded gating math failed"

    # 3. Test Loaded Gate (Low value -> Closed)
    print("\n--- Test 3: Loaded Gate (Low value) ---")
    raw_gate_low = np.array(-10.0, dtype=np.float32)
    q_data_low = {'ltm_gate_logit': np.array({'raw': raw_gate_low})}
    
    model_low = MockQuantizedHierarchos(q_data_low)
    gated_vals = model_low.apply_gate(vals)
    # Sigmoid(-10) ~= 0.000045
    expected_gate = torch.sigmoid(torch.tensor(-10.0))
    print(f"Expected Gate: {expected_gate}")
    print(f"Gated Vals: {gated_vals[0,0,0]}")
    assert torch.allclose(gated_vals, vals * expected_gate), "Low gating math failed"

    print("\nAll LTM gating tests passed!")

if __name__ == "__main__":
    test_ltm_gate()
