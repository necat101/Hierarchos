import unittest

import torch

from hierarchos.utils.checkpoint import _infer_arch_flags_from_state_dict


class ArchitectureConfigFlagTests(unittest.TestCase):
    def test_infers_deepembed_and_rosa_from_legacy_state_dict(self):
        config = {}
        state = {
            "h_deepemb.weight": torch.empty(2, 4),
            "l_deepemb.weight": torch.empty(2, 4),
            "rosa_emb.weight": torch.empty(3, 2),
            "rosa_gate_logit": torch.empty(()),
            "h_rnn.r_k": torch.empty(7, 64),
        }

        _infer_arch_flags_from_state_dict(config, state)

        self.assertTrue(config["use_deepembed"])
        self.assertTrue(config["use_rosa"])
        self.assertEqual(config["rosa_max_context"], 512)
        self.assertEqual(config["rwkv_head_size"], 64)

    def test_infers_disabled_when_legacy_state_dict_has_no_optional_modules(self):
        config = {}
        state = {"tok_emb.weight": torch.empty(2, 4)}

        _infer_arch_flags_from_state_dict(config, state)

        self.assertFalse(config["use_deepembed"])
        self.assertFalse(config["use_rosa"])
        self.assertNotIn("rosa_max_context", config)

    def test_does_not_override_explicit_config(self):
        config = {"use_deepembed": False, "use_rosa": True, "rosa_max_context": 128}
        state = {
            "h_deepemb.weight": torch.empty(2, 4),
            "rosa_emb.weight": torch.empty(3, 2),
        }

        _infer_arch_flags_from_state_dict(config, state)

        self.assertFalse(config["use_deepembed"])
        self.assertTrue(config["use_rosa"])
        self.assertEqual(config["rosa_max_context"], 128)


if __name__ == "__main__":
    unittest.main()
