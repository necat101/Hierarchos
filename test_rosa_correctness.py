import random
import unittest

import torch

from hierarchos.utils.rosa import ROSA, rosa_async_pipeline, rosa_batch_parallel, rosa_single


def _decode_sentinel(values, sentinel):
    return [(-1 if int(v) == sentinel else int(v)) for v in values]


def _wiki_rosa_reference(x):
    """Literal expanded form of the RWKV wiki ROSA reference."""
    n = len(x)
    y = [-1] * n
    s = 2 * n + 1
    b = [None] * s
    c = [-1] * s
    d = [0] * s
    e = [-1] * s
    b[0] = {}
    g = 0
    z = 1
    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u
        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]
                break
            v = c[v]
        y[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]
    return y


def _bruteforce_rosa_reference(x):
    """Slow semantic reference: longest previous suffix, newest end-position tie."""
    y = []
    for i in range(len(x)):
        best_len = 0
        best_end = -1
        for match_len in range(1, i + 2):
            suffix_start = i - match_len + 1
            suffix = x[suffix_start:i + 1]
            for end in range(match_len - 1, i):
                start = end - match_len + 1
                if x[start:end + 1] == suffix and (match_len > best_len or end > best_end):
                    best_len = match_len
                    best_end = end
        y.append(x[best_end + 1] if best_end >= 0 else -1)
    return y


class ROSACorrectnessTests(unittest.TestCase):
    def test_reference_matches_rwkv_wiki_and_semantics(self):
        cases = [
            [],
            [1],
            [1, 2, 1, 2, 3, 1, 2],
            [4, 4, 4, 5, 4, 4, 6],
            [0, 1, 0, 1, 0, 2, 0, 1, 0],
        ]
        rng = random.Random(20260522)
        cases.extend([[rng.randrange(5) for _ in range(length)] for length in range(1, 24)])

        for seq in cases:
            self.assertEqual(ROSA(seq), _wiki_rosa_reference(seq))
            self.assertEqual(ROSA(seq), _bruteforce_rosa_reference(seq))

    def test_incremental_matches_reference(self):
        rng = random.Random(1337)
        for length in range(1, 64):
            for _ in range(25):
                seq = [rng.randrange(9) for _ in range(length)]
                expected = _wiki_rosa_reference(seq)

                state = None
                actual = []
                pos = 0
                while pos < length:
                    step = rng.randrange(1, 8)
                    preds, state = rosa_single(seq[pos:pos + step], state)
                    actual.extend(preds)
                    pos += step

                self.assertEqual(actual, expected)

    def test_batch_matches_reference(self):
        batch = [
            [1, 2, 1, 2, 3, 1, 2],
            [4, 4, 4, 5, 4, 4, 6],
            [7, 8, 9],
        ]
        preds, states = rosa_batch_parallel(batch)
        self.assertEqual(preds, [ROSA(seq) for seq in batch])
        self.assertEqual([state.tokens for state in states], batch)

    def test_async_chunking_matches_reference_without_sliding(self):
        seq = [1, 2, 1, 2, 3, 1, 2, 4, 1, 2, 3]
        chunks = [3, 1, 4, 3]
        sentinel = 99
        device = torch.device("cpu")
        past = None
        states = None
        actual = []
        pos = 0

        for step in chunks:
            cur = seq[pos:pos + step]
            finalize = rosa_async_pipeline(
                torch.tensor([cur], dtype=torch.long),
                past,
                states,
                sentinel,
                device,
                rosa_max_ctx=64,
            )
            out, past, states = finalize()
            actual.extend(_decode_sentinel(out[0].tolist(), sentinel))
            pos += step

        self.assertEqual(actual, ROSA(seq))

    def test_async_sliding_window_alignment(self):
        seq = [6, 3, 6, 3, 7, 6, 3, 8, 6]
        chunks = [2, 3, 1, 3]
        cap = 4
        sentinel = 99
        device = torch.device("cpu")
        past = None
        states = None
        pos = 0

        for step in chunks:
            cur = seq[pos:pos + step]
            end = pos + len(cur)
            window = seq[max(0, end - cap):end]
            overlap_len = min(len(cur), len(window))
            expected = [-1] * (len(cur) - overlap_len)
            if overlap_len:
                expected.extend(ROSA(window)[-overlap_len:])

            finalize = rosa_async_pipeline(
                torch.tensor([cur], dtype=torch.long),
                past,
                states,
                sentinel,
                device,
                rosa_max_ctx=cap,
            )
            out, past, states = finalize()
            actual = _decode_sentinel(out[0].tolist(), sentinel)

            self.assertEqual(len(actual), len(cur))
            self.assertEqual(actual, expected)
            self.assertEqual(states[0].tokens, window)
            pos = end

    def test_async_chunk_longer_than_window_preserves_model_length(self):
        sentinel = 99
        finalize = rosa_async_pipeline(
            torch.tensor([[6, 3]], dtype=torch.long),
            past_tokens=None,
            rosa_states=None,
            vocab_size=sentinel,
            device=torch.device("cpu"),
            rosa_max_ctx=1,
        )
        out, past, states = finalize()

        self.assertEqual(out.shape, (1, 2))
        self.assertEqual(_decode_sentinel(out[0].tolist(), sentinel), [-1, -1])
        self.assertEqual(past.tolist(), [[3]])
        self.assertEqual(states[0].tokens, [3])


if __name__ == "__main__":
    unittest.main()
