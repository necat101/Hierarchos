# Hierarchos at 232M Parameters: Preliminary Findings From a Recurrent Memory-Augmented Assistant Model

**Status:** Draft technical report  
**Date:** July 2026  
**Project:** Hierarchos / KortexHOS  
**Authors:** Makhi Burroughs / netcat420, Lost Time, and the Hierarchos project  

## Abstract

We present preliminary findings from Hierarchos, an experimental recurrent memory-augmented language model trained as a 232M-parameter assistant checkpoint. Hierarchos combines an RWKV-style recurrent backbone, a two-level hierarchical manager/worker refinement loop, differentiable slot-based long-term memory, DeepEmbed channel-mix modulation, and a deterministic suffix-automaton feature path called ROSA. The current release was trained for 13 epochs on an in-house Alpaca-style instruction dataset and evaluated with a local CPU benchmark preset on ARC Easy, HellaSwag, and TruthfulQA MC1.

The central finding is not that the 232M model reaches GPT-3-class capability. It does not. Rather, the project demonstrates that the architecture can produce coherent assistant behavior and measurable benchmark signal after several important train/inference parity and numerical-stability fixes. The most important engineering discoveries were: (1) streamed chat must not reseed hierarchical drift state every generated token; (2) supervised LTM inner updates during training create an inference mismatch unless mirrored by a label-free inference mechanism; (3) RWKV channel-mix activations require explicit key and DeepEmbed clamps for long assistant runs; (4) DeepEmbed identity gates should be excluded from weight decay; and (5) local static benchmarking must suppress passive memory writes and use TBPTT-style chunked recurrence to remain aligned with chat.

On a bounded local benchmark run with `--eval-limit 100`, the release checkpoint achieved ARC Easy accuracy 0.3600, HellaSwag accuracy 0.3400, HellaSwag normalized accuracy 0.3700, and TruthfulQA MC1 accuracy 0.2200. These results are preliminary smoke-test metrics rather than leaderboard-comparable claims, but they indicate that the checkpoint is not collapsed and that it has learned nontrivial commonsense and question-answering signal. We conclude with a proposed scaling recipe for a future funded retrain based on broad pretraining, staged midtraining, assistant SFT, preference/distillation polish, strict parity tests, and controlled matched-parameter baselines.

## 1. Introduction

Modern language modeling is dominated by Transformer scaling, where broad pretraining data, large parameter counts, and long training budgets produce general capability. Hierarchos explores a different research direction: whether recurrent state, explicit memory retrieval, hierarchical iterative computation, and bounded local inference can improve small-model parameter efficiency.

The first public coherent Hierarchos release is a 232M-parameter model. It was trained from scratch on an in-house dataset in Alpaca-style instruction/input/output format:

- Dataset: [netcat420/Experiment_0.1](https://huggingface.co/datasets/netcat420/Experiment_0.1)
- Model size: approximately 232M parameters
- Training: 13 epochs, project-reported RTX 6000 Blackwell generation 96GB GPU rental
- Release mode: full precision
- Recommended chat baseline: static inference with passive learning off and no previous-turn prompt history

The model should be understood as an early research checkpoint. It is more coherent than the failed intermediate runs, but it is not a GPT-3.5-level model and should not be marketed as such. Its significance is that a non-Transformer hybrid architecture survived a difficult training process, reached usable short-form assistant behavior, and produced measurable benchmark signal at small scale.

## 2. Background and Related Work

Hierarchos draws inspiration from several research lines:

- **RWKV-style recurrence.** RWKV investigates language modeling with recurrent inference and parallelizable training, offering a non-Transformer path toward efficient sequence processing.
- **Titans-style neural memory.** Titans and related neural memory systems investigate test-time or persistent memory mechanisms that can store and retrieve information beyond ordinary activations.
- **Hierarchical reasoning.** HRM-style work studies multi-level recurrent reasoning, where high-level and low-level modules can iteratively refine internal state.
- **Scaling laws.** GPT-3 and Chinchilla-style scaling results show that general language ability depends heavily on a balance of parameter count, data scale, and compute. These results caution against expecting GPT-3-class capability from a 232M model trained mostly on a narrow assistant dataset.

Hierarchos is not a faithful reproduction of these systems. It is best described as an experimental hybrid inspired by them: an RWKV-like recurrent model with explicit hierarchical control and differentiable memory.

## 3. Architecture Overview

Hierarchos is implemented as a recurrent language model rather than a Transformer. Each token passes through a sequence of recurrent, memory, and refinement components before being projected to next-token logits.

### 3.1 Token and Feature Input

Tokens are embedded with a tied input/output embedding matrix. The model can also inject two auxiliary signals:

1. **ROSA:** a deterministic suffix-automaton path that predicts likely continuation token IDs from exact repeated suffix patterns.
2. **DeepEmbed:** a learned token-specific modulation path that influences RWKV channel mixing.

ROSA should not be described as standalone symbolic reasoning. It is better understood as an exact-pattern feature generator whose usefulness depends on learned gates and downstream recurrence.

### 3.2 Long-Term Memory

The LTM subsystem uses learned slow-memory keys and values plus fast working-memory values. Retrieval is top-k associative lookup from the current token/context query. During older training runs, retrieved values could receive supervised gradient-derived fast-memory updates between TBPTT chunks.

This became one of the core lessons of the project: if training gives the model supervised memory updates that normal chat generation cannot receive, the model may learn to depend on a hidden training-only helper signal. v0.20.4 introduced read-only LTM training mode for assistant recovery, preserving recurrent and ROSA state while disabling supervised fast-memory writes.

### 3.3 Hierarchical Manager/Worker Loop

Hierarchos uses a high-level manager recurrent module and a lower-level worker recurrent module.

The manager maintains broader context and periodically produces a target context plan. The worker refines token-local state against that context using a learned drift vector. Training regularizes this drift through a commitment cost so the worker does not diverge without constraint.

This design is intended to provide bounded iterative computation without full Transformer self-attention. In practice, its stability depends heavily on drift-state handling and on matching training/inference recurrence exactly.

### 3.4 RWKV-Style Recurrent Backbone

The active recurrent cell uses RWKV-like time mixing and channel mixing with a packed recurrent state. DeepEmbed multiplicatively modulates the channel-mix path. Long training exposed rare but serious channel-mix instability: large key or DeepEmbed activations could be squared or amplified into non-finite gradients.

The release path therefore uses:

- `--rwkv-channel-mix-key-clamp 12.0`
- `--rwkv-channel-mix-deepembed-clamp 4.0`
- DeepEmbed exclusion from AdamW weight decay
- finite gradient clipping plus non-finite gradient rejection

## 4. Training Setup

The 232M release checkpoint was trained on an in-house Alpaca-style dataset. The dataset format uses:

```text
### Instruction:
<instruction>

### Input:
<optional previous context>

### Response:
<assistant response>
```

Training used a long-context assistant configuration around:

- `context_dim=448`
- `h_hidden=448`
- `l_hidden=448`
- `rwkv_head_size=64`
- `max_h_steps=5`
- `max_l_steps=5`
- `max_length=8880`
- Alpaca formatting
- prompt-token training with low prompt weight
- response-boundary weighting
- adaptive ponder
- drift/commit stabilization
- torch.compile on CUDA

The final coherent release was not produced by a perfectly clean first attempt. Several early runs were discarded or resumed after architectural fixes. This makes the current model useful as a research artifact, but it also means the release should not be treated as a controlled scaling result.

## 5. Major Findings

### 5.1 Chat/Training Drift Was a Primary Coherence Failure

The most important inference bug was a train/chat mismatch in drift-state handling. During chat generation, the loop was feeding the previous drift state back into the model on every generated token. During training, drift state is naturally reseeded at TBPTT chunk boundaries, not every token.

This mismatch caused streamed chat logits to diverge sharply from the teacher-forced or chunked path. Local diagnostics found pre-fix full-forward vs streamed logits diverging by multiple logit points. After the boundary-only drift fix, streamed chat and TBPTT-style recurrence matched to approximately single-digit micro-logit error in local tests. The architecture audit now includes regression checks for this behavior.

**Finding:** for recurrent hierarchical models, tiny state-contract mismatches can destroy apparent coherence even when training loss is low.

### 5.2 Supervised LTM Inner Updates Created an Inference Mismatch

Older training allowed supervised gradient-driven fast-memory updates between chunks. Chat generation does not have labels, so it cannot reproduce those updates. This created a plausible path for low training loss but weak open-ended chat behavior.

v0.20.4 introduced:

- `--ltm-training-mode read-only`
- `--inference-like-ltm-training`
- assistant-recovery defaulting to read-only LTM training unless overridden

In read-only mode, training still carries recurrent state, ROSA history, and LTM state structures, but it does not perform supervised fast-memory writes. This better aligns training with actual inference.

**Finding:** LTM can be useful, but the training-time memory update rule must exist at inference or be explicitly treated as an auxiliary-only path.

### 5.3 Drift and Commitment Require Explicit Bounds

The manager/worker design introduces a learned drift state. During unstable resumes, loss, ponder, and commitment cost could rise together. v0.20.1 added:

- `--drift-state-clamp`
- `--drift-norm-clamp`
- `--drift-delta-scale`
- straight-through commitment-cost capping

The important detail is that commitment capping should bound the forward auxiliary value without making high raw drift gradient-dead. Otherwise, once raw drift exceeds the cap, the regularizer may stop providing corrective signal.

**Finding:** hierarchical recurrence needs explicit drift contracts. Without them, late-run instability can damage coherence even when individual gradients remain finite.

### 5.4 RWKV Channel-Mix Needed Key and DeepEmbed Clamps

Late-run non-finite gradients repeatedly implicated RWKV channel-mix paths, especially `key_cm`, `value_cm`, and related upstream projections. The failure mode was consistent with rare activation spikes entering ReLU-squared channel mixing and being amplified by DeepEmbed modulation.

v0.20.2 added a channel-mix key clamp. v0.20.3 added a DeepEmbed modulation clamp. v0.20.4 excluded DeepEmbed from AdamW weight decay.

**Finding:** ReLU-squared recurrent FFN paths can be efficient and expressive, but they need numerical bounds when combined with learned token-specific multiplicative modulation.

### 5.5 Low Loss Alone Did Not Guarantee Coherent Chat

The project observed cases where training loss looked promising while chat output remained incoherent. The causes were not simply "bad weights." Inference drift, LTM training/inference mismatch, and prompt/history handling all created conditions where the evaluated chat path differed from the trained path.

**Finding:** for stateful recurrent-memory models, loss must be interpreted together with path parity. A low-loss model can still appear broken if chat is not executing the same state contract as training.

### 5.6 Passive Learning Should Not Be Used as the Release Baseline

Passive chat learning is potentially useful for online adaptation, but it adds another moving part during evaluation. The release profile disables passive learning and previous-turn prompt history:

```bash
python hierarchos_cli.py chat \
    --model-path "./chatHRM" \
    --temperature 0.4 \
    --top-k 40 \
    --top-p 0.9 \
    --repetition-penalty 1.15 \
    --max-new-tokens 256 \
    --no-passive-learning \
    --chat-input-history-turns 0
```

**Finding:** static chat should be the first evaluation target. Online learning and history carry can be layered back in only after static coherence is established.

## 6. Evaluation

### 6.1 Local ROG Ally Benchmark Preset

To avoid repeated cloud-rental costs, the project added a bounded local benchmark preset:

```bash
python hierarchos_cli.py benchmark \
    --model-path "./chatHRM" \
    --benchmark-preset rog-ally \
    --eval-limit 100
```

This preset uses CPU execution, batch size 1, sequential tasks, static LTM behavior, Hebbian/passive-write suppression, checkpoint-sized TBPTT chunking, and a light suite:

- ARC Easy
- HellaSwag
- TruthfulQA MC1

The benchmark path was aligned with static chat by clearing transient LTM working memory and suppressing Hebbian writes. This makes it a cheap sanity check for core weights, though it is not a replacement for full benchmark sweeps.

### 6.2 Current Smoke Results

The current release checkpoint produced:

| Benchmark | Metric | Score | Std. Err. |
| --- | ---: | ---: | ---: |
| ARC Easy | acc | 0.3600 | 0.0482 |
| ARC Easy | acc_norm | 0.3200 | 0.0469 |
| HellaSwag | acc | 0.3400 | 0.0476 |
| HellaSwag | acc_norm | 0.3700 | 0.0485 |
| TruthfulQA MC1 | acc | 0.2200 | 0.0416 |

These results should be interpreted as a local smoke test. They show non-collapse and measurable signal, especially in HellaSwag normalized accuracy and ARC Easy. They do not establish superiority over same-size Transformers.

### 6.3 Coherence Interpretation

Qualitatively, the 232M checkpoint is best described as:

- coherent on some short instruction prompts
- assistant-shaped due to Alpaca-style training
- brittle on long responses
- weak on arithmetic and broad factual recall
- closer to GPT-2-era general coherence than GPT-3-era general capability
- not GPT-3.5-class in coding, math, reasoning, or breadth

The architecture is promising for its size, but matched baselines are still required.

## 7. Limitations

This work has several important limitations:

1. **No matched Transformer baseline yet.** The project has not yet run comparable 232M Transformer, RWKV-only, or ablated Hierarchos baselines under the same data and token budget.
2. **Dataset scope is narrow.** The in-house dataset is assistant-shaped but not a broad foundation pretraining corpus.
3. **Training history was not clean.** Several earlier runs were discarded or resumed after fixes. The final checkpoint is useful, but not a controlled scaling experiment.
4. **Benchmarks are local smoke tests.** The ROG Ally suite is intentionally small and bounded. It is useful for fast iteration but not sufficient for publication-grade comparisons.
5. **Quantization remains unresolved.** The release is full precision because hierarchical drift/state dynamics were sensitive to quantization error in current experiments.
6. **LTM value is not isolated.** The release does not yet quantify how much LTM, ROSA, DeepEmbed, or hierarchical worker drift individually contribute.

## 8. Proposed Ablation Plan

To turn Hierarchos from a promising prototype into a stronger scientific result, the next study should include:

| Ablation | Purpose |
| --- | --- |
| No LTM | Measures whether slot memory improves loss, chat, or benchmarks. |
| Read-only LTM vs inner-update LTM | Tests whether supervised memory updates help or harm inference-aligned behavior. |
| No ROSA | Measures whether suffix-automaton pattern hints contribute beyond recurrence. |
| No DeepEmbed | Measures whether token-specific channel-mix modulation improves sample efficiency. |
| No hierarchical worker loop | Tests whether manager/worker refinement beats a simpler recurrent backbone. |
| Fixed ponder vs adaptive ponder | Measures whether dynamic compute allocation is useful. |
| Transformer 232M baseline | Establishes whether Hierarchos is competitive at equal scale. |
| RWKV-only 232M baseline | Separates recurrent-backbone strength from Hierarchos-specific components. |

The key scientific question is not whether the current checkpoint is impressive in isolation. It is whether Hierarchos delivers better sample efficiency, context behavior, or local inference efficiency than matched baselines.

## 9. Scaling Recipe for a Funded Retrain

A funded retrain should not repeat the same assistant-only setup at larger scale. The next serious model should be trained as a foundation model first and an assistant second.

### 9.1 Recommended Scale Tiers

| Tier | Model Size | Unique Token Target | Purpose |
| --- | ---: | ---: | --- |
| Scout | 300M-500M | 20B-50B | Validate stability, parity, and scaling slope. |
| Real v1 | 1B-1.5B | 100B-300B | Test whether Hierarchos scales beyond small-model behavior. |
| Serious | 3B | 600B-1.5T | Establish competitive open-model behavior if scaling remains stable. |
| Ambitious | 7B-13B | 2T-5T+ | Requires strong funding and mature infra. |

The 232M release should not be used as proof that a 7B run will work. A 1B-1.5B pilot is the rational next step.

### 9.2 Data Mixture

A scaled Hierarchos model should use broad pretraining data rather than only in-house SFT data. A reasonable starting mixture:

| Data Type | Approx. Share |
| --- | ---: |
| High-quality web, e.g. FineWeb/FineWeb-Edu style | 35-50% |
| Broad curated web, e.g. Dolma/DCLM style | 20-30% |
| Code and technical documentation | 8-15% |
| Math, science, proofs, textbooks where licensed | 5-12% |
| Wikipedia/reference/public-domain books | 5-10% |
| In-house assistant/conversation data | 1-5%, mostly late-stage |

All data should pass a license audit, deduplication, benchmark contamination checks, and tokenizer coverage checks before training.

### 9.3 Training Stages

1. **Architecture scout.** Train a 300M-500M model on 20B-50B broad tokens to validate loss slope and stability.
2. **Foundation pretraining.** Train the target model on broad text/code/math with read-only/inference-like LTM behavior unless a label-free memory update is proven.
3. **Quality midtraining.** Anneal into higher quality educational, code, math, and reference data.
4. **Assistant SFT.** Train on the in-house Alpaca format and high-quality instruction data.
5. **Distillation/preference polish.** Use teacher-generated responses, DPO/ORPO-style preference data, or curated rejection sampling.
6. **Release eval.** Run local smoke tests, full benchmarks, and a fixed human-readable coherence prompt suite before release.

### 9.4 Required Engineering Before Scaling

Before a funded run, the repo should have:

- weighted multi-HF-dataset streaming
- dataset provenance and license manifests
- exact and near deduplication
- benchmark decontamination
- a custom tokenizer trained on the final mixture
- automatic chat/train/logit parity tests for every release candidate
- matched baseline training scripts
- checkpoint comparison tooling
- robust full-precision and eventually quantized v8-compatible inference

## 10. Responsible Claims for Release Communication

Supported:

- Hierarchos is a coherent 232M-parameter experimental assistant checkpoint.
- It combines recurrent sequence modeling, hierarchical refinement, and memory-augmented inference.
- It shows measurable local benchmark signal.
- It is promising enough to justify controlled scaling and ablation studies.

Not yet supported:

- GPT-3.5-level coding or math performance.
- GPT-3-level general language capability.
- Transformer superiority at equal parameter count.
- AGI claims.
- Production-ready quantized inference.

The most accurate release description is:

> Hierarchos 232M is an experimental recurrent memory-augmented assistant model. It shows coherent short-form instruction behavior and measurable benchmark signal at small scale, but remains brittle and requires broader pretraining, baselines, and ablations before stronger capability claims can be made.

## 11. Conclusion

The 232M Hierarchos release is not a GPT-3-class model, but it is a meaningful research milestone. The project found and fixed several failure modes that are likely common to recurrent memory-augmented language models: subtle streaming-state drift, supervised memory updates that cannot be reproduced at inference, unstable recurrent drift accumulation, and unbounded channel-mix activations.

The current checkpoint demonstrates that the architecture can produce coherent assistant behavior and measurable benchmark signal after path-parity and stability fixes. The next phase should prioritize a controlled broad-pretraining run, strict ablations, matched Transformer/RWKV baselines, and a staged scaling plan. If these experiments show consistent sample-efficiency gains, Hierarchos could become a credible alternative research path for small and local language models.

## References

1. Brown et al. **Language Models are Few-Shot Learners.** arXiv:2005.14165. https://arxiv.org/abs/2005.14165
2. Hoffmann et al. **Training Compute-Optimal Large Language Models.** arXiv:2203.15556. https://arxiv.org/abs/2203.15556
3. Peng et al. **RWKV: Reinventing RNNs for the Transformer Era.** arXiv:2305.13048. https://arxiv.org/abs/2305.13048
4. Behrouz et al. **Titans: Learning to Memorize at Test Time.** arXiv:2501.00663. https://arxiv.org/abs/2501.00663
5. Wang et al. **Hierarchical Reasoning Model.** arXiv:2506.21734. https://arxiv.org/abs/2506.21734
6. Zellers et al. **HellaSwag: Can a Machine Really Finish Your Sentence?** arXiv:1905.07830. https://arxiv.org/abs/1905.07830
7. Clark et al. **Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge.** arXiv:1803.05457. https://arxiv.org/abs/1803.05457
8. Lin et al. **TruthfulQA: Measuring How Models Mimic Human Falsehoods.** arXiv:2109.07958. https://arxiv.org/abs/2109.07958
9. Hugging Face. **FineWeb dataset.** https://huggingface.co/datasets/HuggingFaceFW/fineweb
10. Hugging Face. **FineWeb-Edu dataset.** https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
11. Allen AI. **Dolma dataset.** https://huggingface.co/datasets/allenai/dolma
12. DataComp-LM. **DCLM Baseline dataset.** https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0
