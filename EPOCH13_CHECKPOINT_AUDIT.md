# Hierarchos Epoch-13 Checkpoint Audit

This audit uses the re-downloaded full-precision epoch-13 model as the control:

- Source: `C:\Users\User\Downloads\chatHRM\chatHRM\hierarchos.pt`
- SHA-256: `396787af4a7811e5fc87286c9e2f93db37c50b7accbfc72aa45a9e3a08b9a751`
- Size: `931,161,249` bytes
- Architecture: GPT-2 vocabulary, `448/448/448`, RWKV head size `64`
- Unique parameters: `232,516,229`
- State tensors: `95`

The machine-readable control report is in
[`checkpoint_audits/epoch13-control.json`](checkpoint_audits/epoch13-control.json).
Reproduce it with:

```powershell
python tools\audit_checkpoint_health.py `
  --model-path "C:\Users\User\Downloads\chatHRM" `
  --parity-tokens 270 `
  --sha256
```

## Bottom Line

Epoch 13 is a valid, coherent warm-start checkpoint. It is not corrupted, it does
not have a remaining catastrophic train/chat logit mismatch, and it can save money
on a v2 continuation by preserving learned language, recurrent, and assistant-format
behavior. It is not an optimal or fully rehabilitated base: training left DeepEmbed
heavily attenuated, the learned network relies far more on ROSA than LTM, and the
Hebbian value writer was never trained.

Use it as the budget-efficient v2 branch. Keep a from-scratch run as the preferred
scientific control when funding permits. Do not claim that continuation guarantees a
particular capability level; model selection still needs held-out loss, generation
canaries, and standard benchmarks.

## Compatibility Results

- All `95` checkpoint tensors strict-loaded into the current full-precision model.
- No learned state tensor contains NaN or Inf.
- Token embedding and LM-head storage remain tied.
- No parameter name or shape was added, removed, or changed by the fixes.
- Chunked epoch-13 recurrence and one-token chat streaming match across the real
  `256`-token TBPTT boundary: maximum logit delta `1.144409e-05`, mean delta
  `8.011716e-07` over `270` tokens.
- Final H/L/context/drift states matched exactly in that control.
- The outer downloaded directory now resolves its single nested model/tokenizer
  directory automatically and unambiguously.
- On `### Instruction: Hello`, the first-token distribution is coherent: `Hello`
  is top-1 at `0.47993` probability.

These results rule out the old per-token drift reseeding failure. Under that incorrect
path the same checkpoint could diverge by roughly `8.93` logit points; the corrected
path is within normal floating-point ordering noise.

## Weight Findings

### 1. DeepEmbed is functional but severely attenuated

DeepEmbed represents about `77.47%` of the unique parameters. Both tables began at
the multiplicative identity value `1.0`, but epoch 13 contains:

| Tensor | Mean | Std. dev. | Sample token-centered RMS |
|---|---:|---:|---:|
| `h_deepemb.weight` | `0.354874` | `0.004239` | `0.004179` |
| `l_deepemb.weight` | `0.354072` | `0.019352` | `0.019364` |

The historical optimizer applied AdamW decay to these multiplicative tables. The
current optimizer excludes them from decay, preventing the same quiet collapse in a
new run. The current checkpoint has adapted around its roughly `0.355` scale:
resetting DeepEmbed to `1.0` changes real prompt logits by more than `10` points and
is unsafe. Recovery must be gradual, through normal gradients on diverse data.

The output norm has also adapted, with mean scale `3.7564`. This is consistent with
the rest of the network compensating for a weakened channel-mix modulation path.

### 2. Memory routing is strongly ROSA-dominant

On the control prompt, mean token gates are:

- LTM: `0.01261`
- ROSA: `0.94126`

This is learned behavior rather than a load failure. ROSA is carrying useful exact
context signal, while the differentiable LTM path is nearly closed. A v2 continuation
can update the routers and LTM weights through ordinary CE gradients in `read-only`
fast-memory mode, but the gate should not be reset or forced open. Track gate
statistics on held-out prompts and let useful LTM behavior earn its way back.

### 3. The Hebbian value writer was never trained

`val_proj.weight` is the only learned parameter with no gradient under the historical
language objective. Its distribution remains consistent with its original random
linear-layer initialization, and epoch 13 has no `val_proj_trained` marker. This
projection is used only when validated text is written through the Hebbian path.
Ordinary static chat does not use it, so it did not cause the old broken first turn.
It could, however, inject random fast-memory values after praise or `/learn`.

The current code therefore:

- Blocks Hebbian writes for legacy/unmarked checkpoints at the model boundary.
- Leaves gradient-derived feedback learning available in chat.
- Adds opt-in `--ltm-value-alignment-weight` training for the existing `val_proj`.
- Normalizes that auxiliary by target activation energy.
- Excludes `val_proj` from AdamW decay.
- Requires `--ltm-value-alignment-min-updates` successful optimizer updates before
  marking the writer safe; the default is `100`.

On the reproducible control prompt, the normalized alignment cost is `0.9795`, its isolated
`val_proj.weight` gradient norm is `0.6287`, and no other tensor receives gradient
from this auxiliary. The default weight is `0.0`, so historical training dynamics
remain unchanged unless the feature is explicitly enabled.

### 4. Coherence is real, but capability remains brittle

Controlled generation is sentence-level coherent and follows assistant formatting.
It still fails elementary arithmetic, loses semantic constraints in stories, and can
produce plausible but incorrect factual or code answers. This agrees with the local
ARC/HellaSwag/TruthfulQA results. The remaining limitation is model/data quality and
learned allocation, not catastrophic inference drift.

## Budget-Efficient V2 Path

The epoch-13 weights can be used as a v2 base and should save the early cost of
relearning token statistics, grammar, recurrent dynamics, Alpaca boundaries, and
basic assistant behavior. Use the following contract:

1. Start from the inference export with `--model-path`, a fresh optimizer, and a new
   warmup/cosine schedule. Do not restore the epoch-13 optimizer, scaler, or exhausted
   scheduler.
2. Preserve the exact tokenizer and all `95` learned tensors. Reset transient fast
   LTM state, not learned LTM keys/values or recurrent weights.
3. Train all weights on a deduplicated, diverse text/code/math mixture with a real
   held-out split. Repeating one instruction dataset many times does not substitute
   for novel tokens.
4. Keep `--ltm-training-mode read-only` for static train/chat parity. Learned LTM
   keys, values, routers, and gates still receive CE gradients; only supervised
   fast-memory inner writes are disabled.
5. Enable `--ltm-value-alignment-weight 0.01` as a measured pilot. Its gradient is
   isolated to the writer, so it cannot directly distort language logits. Keep the
   default `100` readiness updates.
6. Keep both channel-mix clamps, finite-gradient rejection, gradient norm clipping,
   DeepEmbed/`val_proj` no-decay, and `torch.compile` with
   `max-autotune-no-cudagraphs`.
7. Do not reset DeepEmbed, ROSA/LTM gates, hierarchy weights, or output norm. Let new
   data move them continuously.

For a 232M warm start, a conservative pilot is a peak model LR around `1e-5` to
`2e-5`, `1-2%` warmup, cosine decay, and `--grad-clip 0.75`. Select the range with a
short pilot and held-out loss rather than committing the full budget immediately.
As a planning heuristic, several billion genuinely varied tokens are more valuable
than many additional passes over the same one-billion-token instruction set.

After broad continued pretraining, use a shorter high-quality assistant phase at a
lower LR. Save and evaluate milestone checkpoints; do not assume the final epoch is
best merely because it is last.

## From-Scratch V2 Control

With sufficient funding, also run a smaller-scale pilot from a fresh initialization
under the fixed code before funding a full clean v2. This is the only way to measure
how much the warm start saves and how much old specialization it retains. The clean
run should use the same tokenizer/shape if weight and tooling compatibility matters,
the same diverse mixture and validation split, and the writer auxiliary from the
beginning if validated Hebbian memory is a release goal.

## Verification

- `python -m pytest -q`: `230 passed, 3 skipped`
- `python tools/check_architecture_integrity.py`: `65 passed`, `1` documented
  legacy-quantization warning, `0` failures
- Real epoch-13 checkpoint audit: strict load, `270`-token boundary parity, finite
  state, exact tokenizer vocabulary, and isolated writer-gradient probe all passed

The remaining architecture warning is unchanged: the quantized CPU/Vulkan loader
targets an older scalar-RWKV archive format. The coherent v8 checkpoint should remain
on the full-precision path.
