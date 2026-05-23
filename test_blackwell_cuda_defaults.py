import unittest
from unittest import mock

from hierarchos.training.datasets import _resolve_prefetch_factor
from hierarchos.training.trainer import estimate_cuda_loss_chunk_rows
from hierarchos_cli import (
    choose_length_bucket_size_from_lengths,
    estimate_bucket_token_efficiency,
    resolve_length_bucket_size,
    resolve_num_workers,
)


class _Device:
    type = "cuda"


class BlackwellCudaDefaultTests(unittest.TestCase):
    def test_cuda_batch64_auto_workers_target_blackwell_profile(self):
        with mock.patch("os.cpu_count", return_value=24):
            self.assertEqual(resolve_num_workers(-1, _Device(), 64), 8)
            self.assertEqual(resolve_num_workers(-1, _Device(), 32), 4)

    def test_cuda_hf_token_cache_uses_large_bucket_window(self):
        size, message = resolve_length_bucket_size(None, _Device(), 64, hf_token_cache=True, auto_tune=False)
        self.assertEqual(size, 65536)
        self.assertIn("HF token cache", message)

        size, message = resolve_length_bucket_size(None, _Device(), 64, hf_token_cache=True, auto_tune=True)
        self.assertIsNone(size)
        self.assertIn("auto-tune", message)

        size, message = resolve_length_bucket_size(None, _Device(), 64, hf_token_cache=False)
        self.assertEqual(size, 8192)
        self.assertIn("CUDA batch>=64", message)

    def test_auto_bucket_chooses_smallest_near_best_efficiency(self):
        lengths = [8, 250, 12, 245, 16, 240, 20, 235] * 16

        small = estimate_bucket_token_efficiency(
            lengths,
            batch_size=8,
            chunk_size=64,
            bucket_size=8,
        )
        tuned, summary = choose_length_bucket_size_from_lengths(
            lengths,
            batch_size=8,
            chunk_size=64,
            candidates=[8, 32, len(lengths)],
            tolerance=0.0,
        )

        self.assertIn(tuned, {32, 128})
        self.assertGreater(summary["chosen"]["token_efficiency"], small["token_efficiency"])

    def test_pin_memory_prefetch_keeps_two_batches_per_worker_at_eight_workers(self):
        self.assertEqual(_resolve_prefetch_factor(8, prefetch_factor=None, pin_memory=True), 2)

    def test_96gb_loss_chunk_rows_cover_batch64_chunk256(self):
        rows = estimate_cuda_loss_chunk_rows(
            free_bytes=int(93.6 * 1024**3),
            batch_size=64,
            chunk_size=256,
            vocab_size=50257,
        )

        self.assertGreaterEqual(rows, 64 * 255)


if __name__ == "__main__":
    unittest.main()
