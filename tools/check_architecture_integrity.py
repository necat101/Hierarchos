#!/usr/bin/env python
"""Architecture integrity checks for the active Hierarchos v8 code path.

This is intentionally broader than a single unit test. It probes the pieces
that can silently drift apart during rescue runs: RWKV state shape, LTM
train/chat mode alignment, DeepEmbed clamp/decay behavior, checkpoint guards,
and quantized-loader architecture status.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hierarchos.models.core import HierarchosCore
from hierarchos.models.quantized import (
    detect_quantized_rwkv_format,
    validate_quantized_rwkv_format,
)
from hierarchos.training.trainer import (
    build_hierarchos_optimizer,
    train_step,
)
from hierarchos.utils.checkpoint import (
    _infer_arch_flags_from_state_dict,
    _reject_unsupported_rwkv_state_dict,
)


class Audit:
    def __init__(self, strict: bool = False):
        self.strict = bool(strict)
        self.failures: list[str] = []
        self.warnings: list[str] = []
        self.passes = 0

    def ok(self, message: str) -> None:
        self.passes += 1
        print(f"PASS: {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        prefix = "FAIL" if self.strict else "WARN"
        print(f"{prefix}: {message}")
        if self.strict:
            self.failures.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL: {message}")

    def require(self, condition: bool, message: str) -> None:
        if condition:
            self.ok(message)
        else:
            self.fail(message)

    def finish(self) -> int:
        print()
        print(f"Architecture audit: {self.passes} passed, {len(self.warnings)} warning(s), {len(self.failures)} failure(s).")
        if self.failures:
            print("Failures:")
            for failure in self.failures:
                print(f"  - {failure}")
            return 1
        if self.warnings:
            print("Warnings:")
            for warning in self.warnings:
                print(f"  - {warning}")
        return 0


def tiny_config(**overrides):
    base = dict(
        vocab_size=64,
        max_length=32,
        context_dim=16,
        h_hidden=16,
        l_hidden=16,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_stride=2,
        max_h_steps=3,
        max_l_steps=3,
        rwkv_head_size=4,
        rwkv_n_layer_hint=2,
        use_deepembed=True,
        use_rosa=True,
        memory_token_routers=True,
        rosa_max_context=32,
        compile=False,
        gradient_checkpointing=False,
        training_chunk_size=4,
        detach_every_n_steps=4,
        ltm_lr=1e-3,
        min_ltm_lr=1e-8,
        ltm_momentum=0.9,
        ltm_weight_decay=1e-4,
        ltm_forget_rate=0.01,
        ltm_score_grad_scale=1.0,
        ltm_cpu_gather_retrieval=True,
        ltm_cpu_sparse_update=True,
        isolate_batch_ltm=True,
        memory_gate_warmup_steps=0,
        memory_gate_warmup_floor=0.0,
        activation_clamp=30.0,
        recurrent_state_clamp=20.0,
        context_state_clamp=20.0,
        drift_state_clamp=2.0,
        drift_norm_clamp=1.25,
        drift_delta_scale=0.35,
        commitment_threshold=0.05,
        max_commitment_cost_for_backward=4.0,
        halt_logit_clamp=10.0,
        h_halt_thresh=0.9,
        l_conv_atol=1e-4,
        rwkv_channel_mix_key_clamp=3.0,
        rwkv_channel_mix_deepembed_clamp=2.0,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        z_loss_weight=1e-4,
        debug_numerics=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def tensor_tree_is_finite(value) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(tensor_tree_is_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(tensor_tree_is_finite(v) for v in value if v is not None)
    return True


def make_batch(batch_size=2, length=8, vocab_size=64):
    ids = torch.arange(batch_size * length, dtype=torch.long).reshape(batch_size, length)
    ids = (ids % (vocab_size - 1)) + 1
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
    }


def make_train_args(mode: str):
    return SimpleNamespace(
        amp=False,
        training_chunk_size=4,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode=mode,
        ltm_lr=1e-4,
        min_ltm_lr=1e-8,
        min_lr=1e-8,
        disable_ltm_lr_schedule=True,
        max_ce_loss_for_backward=0.0,
        max_ponder_cost_for_backward=0.0,
        max_commitment_cost_for_backward=4.0,
        commitment_loss_weight=0.5,
        ponder_loss_weight=0.003,
        adaptive_ponder=True,
        ponder_target_scale=0.65,
        max_h_steps=3,
        max_sanitized_gradient_values=0,
    )


def check_core_forward(audit: Audit) -> None:
    torch.manual_seed(7)
    config = tiny_config()
    model = HierarchosCore(config)

    audit.require(model.tok_emb.weight is model.lm_head.weight, "token embedding and LM head weights are tied")
    audit.require(model.h_rnn.state_size == 3 + model.h_rnn.head_size, "H RWKV state size is 3 + head_size")
    audit.require(model.l_rnn.state_size == 3 + model.l_rnn.head_size, "L RWKV state size is 3 + head_size")
    audit.require(float(model.h_deepemb.weight.mean().item()) == 1.0, "DeepEmbed starts as identity multiplier")

    batch = make_batch(batch_size=2, length=8, vocab_size=config.vocab_size)
    model.train()
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_raw_topk_values=False,
        return_topk_indices=False,
    )
    audit.require(outputs["raw_topk_vals"] is None, "read-only forward can omit retained LTM tensors")
    audit.require(outputs["topk_idx"] is None, "read-only forward can omit LTM top-k indices")
    audit.require(outputs["logits"].shape == (2, 8, config.vocab_size), "forward logits shape is stable")
    audit.require(outputs["loss"] is not None and torch.isfinite(outputs["loss"]), "forward loss is finite")
    audit.require(tensor_tree_is_finite(outputs), "forward output tensor tree is finite")
    audit.require(len(outputs["ltm_memory_state"]) == 6, "LTM runtime state carries fast/momentum/ROSA/timestamps/sources")
    drift_norm = outputs["drift_state"].float().norm(dim=-1).max().item()
    audit.require(drift_norm <= config.drift_norm_clamp + 1e-4, "drift state respects L2 norm clamp")

    inner_outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        return_raw_topk_values=True,
        return_topk_indices=True,
    )
    audit.require(isinstance(inner_outputs["raw_topk_vals"], list), "inner-update forward exposes raw retrieval tensors")
    audit.require(inner_outputs["topk_idx"] is not None, "inner-update forward exposes top-k indices")


def check_ltm_train_modes(audit: Audit) -> None:
    torch.manual_seed(11)
    batch = make_batch(batch_size=2, length=8)

    read_only_model = HierarchosCore(tiny_config())
    read_only_model.train()
    optimizer = torch.optim.AdamW(read_only_model.parameters(), lr=1e-3)

    def forbidden_inner_update(*_args, **_kwargs):
        raise AssertionError("read-only mode called LTM inner_update")

    read_only_model.ltm.inner_update = forbidden_inner_update
    outputs, states = train_step(
        read_only_model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=make_train_args("read-only"),
        running_states=(None, None, None, None, None, None),
    )
    audit.require(outputs is not None, "read-only train_step completed")
    audit.require(states[5] is not None and len(states[5]) == 6, "read-only train_step carries LTM/ROSA state")

    inner_model = HierarchosCore(tiny_config())
    inner_model.train()
    optimizer = torch.optim.AdamW(inner_model.parameters(), lr=1e-3)
    original_inner_update = inner_model.ltm.inner_update
    calls = {"count": 0}

    def counted_inner_update(*args, **kwargs):
        calls["count"] += 1
        return original_inner_update(*args, **kwargs)

    inner_model.ltm.inner_update = counted_inner_update
    outputs, _states = train_step(
        inner_model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=make_train_args("inner-update"),
        running_states=(None, None, None, None, None, None),
    )
    audit.require(outputs is not None, "inner-update train_step completed")
    audit.require(calls["count"] > 0, "inner-update mode calls LTM inner_update")


def check_inference_memory_contract(audit: Audit) -> None:
    torch.manual_seed(13)
    model = HierarchosCore(tiny_config(isolate_batch_ltm=False))
    ids = make_batch(batch_size=1, length=6)["input_ids"]

    model.eval()
    model.suppress_hebbian = True
    before = model.ltm.fast_vals.detach().clone()
    with torch.no_grad():
        outputs = model(ids, suppress_hebbian=True)
    after = model.ltm.fast_vals.detach()
    audit.require(torch.equal(before, after), "normal inference suppresses Hebbian LTM writes")
    audit.require(outputs["logits"].shape == (1, 6, model.config.vocab_size), "normal inference logits shape is stable")

    with torch.no_grad():
        model(ids, allow_hebbian_update=True)
    changed = not torch.equal(before, model.ltm.fast_vals.detach())
    audit.require(changed, "explicit validation path can write fast LTM memory")


def check_clamps_and_optimizer(audit: Audit) -> None:
    torch.manual_seed(17)
    model = HierarchosCore(tiny_config(rwkv_channel_mix_key_clamp=4.0, rwkv_channel_mix_deepembed_clamp=1.5))
    model.config.rwkv_channel_mix_key_clamp = 5.0
    model.config.rwkv_channel_mix_deepembed_clamp = 2.0
    model.refresh_runtime_config()
    audit.require(model.h_rnn.channel_mix_key_clamp == 5.0, "runtime refresh updates H channel-mix key clamp")
    audit.require(model.l_rnn.channel_mix_deepembed_clamp == 2.0, "runtime refresh updates L DeepEmbed clamp")

    x = torch.randn(2, model.config.h_hidden)
    state = model.h_rnn.initial_state(2)
    extreme_deepemb = torch.full((2, model.config.h_hidden * 4), 1e6)
    y, next_state = model.h_rnn(x, state, deepemb_vec=extreme_deepemb)
    audit.require(torch.isfinite(y).all() and torch.isfinite(next_state).all(), "RWKV channel-mix clamps keep extreme DeepEmbed finite")

    opt = build_hierarchos_optimizer(
        model,
        SimpleNamespace(starting_lr=1e-3, rwkv_weight_decay=0.1),
        torch.device("cpu"),
    )
    decay_params = {id(p) for p in opt.param_groups[0]["params"]}
    no_decay_params = {id(p) for p in opt.param_groups[1]["params"]}
    audit.require(id(model.h_deepemb.weight) in no_decay_params, "H DeepEmbed is excluded from AdamW decay")
    audit.require(id(model.l_deepemb.weight) in no_decay_params, "L DeepEmbed is excluded from AdamW decay")
    audit.require(id(model.h_deepemb.weight) not in decay_params, "H DeepEmbed is not in decay group")


def check_checkpoint_and_quantized_guards(audit: Audit) -> None:
    config = {}
    state = {
        "h_deepemb.weight": torch.empty(4, 16),
        "l_deepemb.weight": torch.empty(4, 16),
        "rosa_emb.weight": torch.empty(65, 16),
        "rosa_gate_logit": torch.empty(()),
        "h_rnn.r_k": torch.empty(4, 4),
    }
    _infer_arch_flags_from_state_dict(config, state)
    audit.require(config["use_deepembed"] is True, "checkpoint backfills DeepEmbed flag")
    audit.require(config["use_rosa"] is True, "checkpoint backfills ROSA flag")
    audit.require(config["rwkv_channel_mix_key_clamp"] == 12.0, "checkpoint backfills channel-mix key clamp")
    audit.require(config["rwkv_channel_mix_deepembed_clamp"] == 4.0, "checkpoint backfills DeepEmbed clamp")

    try:
        _reject_unsupported_rwkv_state_dict({"h_rnn.time_decay": torch.empty(4)}, "legacy.pt")
    except ValueError:
        audit.ok("legacy scalar-RWKV checkpoint is rejected")
    else:
        audit.fail("legacy scalar-RWKV checkpoint was accepted")

    legacy_q = {"h_rnn.time_decay": None, "h_rnn.time_mix_k": None}
    v8_q = {"h_rnn.x_r": None, "h_rnn.r_k": None}
    mixed_q = {"h_rnn.time_decay": None, "h_rnn.x_r": None}
    audit.require(detect_quantized_rwkv_format(legacy_q) == "legacy-scalar", "quantized legacy RWKV format is detected")
    try:
        validate_quantized_rwkv_format(v8_q, "fake-v8.npz")
    except ValueError:
        audit.ok("v8 quantized archive is refused by legacy quantized loader")
    else:
        audit.fail("v8 quantized archive was accepted by legacy quantized loader")
    try:
        validate_quantized_rwkv_format(mixed_q, "fake-mixed.npz")
    except ValueError:
        audit.ok("mixed quantized archive is refused")
    else:
        audit.fail("mixed quantized archive was accepted")


def check_static_source_contracts(audit: Audit) -> None:
    chat_source = (ROOT / "hierarchos" / "inference" / "chat.py").read_text(encoding="utf-8")
    trainer_source = (ROOT / "hierarchos" / "training" / "trainer.py").read_text(encoding="utf-8")
    quantized_source = (ROOT / "hierarchos" / "models" / "quantized.py").read_text(encoding="utf-8")

    audit.require("model.suppress_hebbian = True" in chat_source, "chat explicitly suppresses normal Hebbian writes")
    audit.require("allow_hebbian_update=True" in chat_source, "chat validation path is the explicit Hebbian write gate")
    audit.require("ltm_inner_updates_enabled(args)" in trainer_source, "trainer gates supervised LTM inner updates")
    audit.require('"ltm_lr": f"{get_current_ltm_lr(args):.2e}" if use_ltm_inner_updates else "off"' in trainer_source, "finetune reports LTM LR as off in read-only mode")
    audit.require("validate_quantized_rwkv_format" in quantized_source, "quantized loader validates recurrent architecture format")
    audit.warn("quantized CPU/Vulkan inference remains legacy scalar-RWKV only; full-precision v8 is the coherent path for current rescue runs")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Hierarchos architecture integrity checks.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args()

    audit = Audit(strict=args.strict)
    checks = [
        check_core_forward,
        check_ltm_train_modes,
        check_inference_memory_contract,
        check_clamps_and_optimizer,
        check_checkpoint_and_quantized_guards,
        check_static_source_contracts,
    ]
    for check in checks:
        print()
        print(f"== {check.__name__} ==")
        try:
            check(audit)
        except Exception as exc:
            audit.fail(f"{check.__name__} crashed: {exc}")
            traceback.print_exc()
    return audit.finish()


if __name__ == "__main__":
    raise SystemExit(main())
