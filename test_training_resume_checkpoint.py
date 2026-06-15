from types import SimpleNamespace
import argparse
import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hierarchos.training.datasets import EpochShuffleSampler, LengthGroupedBatchSampler
from hierarchos.training.trainer import (
    build_training_checkpoint,
    build_lr_scheduler,
    capture_ltm_lr_scheduler_state,
    capture_dataloader_state,
    accumulation_divisor_for_step,
    configure_ltm_lr_schedule,
    compute_update_steps,
    compute_remaining_update_steps,
    get_current_ltm_lr,
    advance_ltm_lr_schedule,
    restore_dataloader_state,
    restore_model_grad_state,
    save_training_checkpoint_if_finite,
    should_step_accumulation,
    train_step,
    training_state_is_finite,
    _sanitize_model_nonfinite_,
    _sanitize_model_transient_state_,
    _sanitize_gradient_nonfinite_,
    _clamp_model_finite_magnitude_,
    _clip_gradients_and_check,
)
from hierarchos.utils.checkpoint import sanitize_model_state_dict
import hierarchos_cli


class _FakeLTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fast_vals", torch.zeros(3, 2))
        self.register_buffer("_mom_vals", torch.zeros(3, 2))
        self.register_buffer("timestamps", torch.zeros(3))
        self.register_buffer("sources", torch.zeros(3, dtype=torch.long))
        self.register_buffer("neg_inf", torch.tensor(-float("inf")), persistent=False)


class _FakeTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = _FakeLTM()
        self.proj = nn.Linear(2, 2)
        self.config = {"context_dim": 2}


class _NaNLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = SimpleNamespace(vocab_size=8, cpu_chunked_lm_loss=False, cuda_chunked_lm_loss=False)
        self.reset_called = False

    def reset_memory(self):
        self.reset_called = True

    def forward(self, **kwargs):
        nan_loss = self.weight * torch.tensor(float("nan"), device=self.weight.device)
        return {
            "loss": nan_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _InfGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        return weight.detach() * 0.0 + 1.0

    @staticmethod
    def backward(ctx, grad_output):
        return torch.full_like(grad_output, float("inf"))


class _InfGradModel(_NaNLossModel):
    def forward(self, **kwargs):
        finite_loss = _InfGradient.apply(self.weight)
        return {
            "loss": finite_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _HighFiniteLossModel(_NaNLossModel):
    def forward(self, **kwargs):
        high_loss = self.weight * 20.0
        high_commitment = self.weight * 20.0
        return {
            "loss": high_loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": high_commitment,
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _FiniteLinearLossModel(_NaNLossModel):
    def forward(self, **kwargs):
        return {
            "loss": self.weight,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


class _BoundaryRecordingModel(_NaNLossModel):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, **kwargs):
        input_ids = kwargs["input_ids"]
        labels = kwargs["labels"]
        self.seen.append((
            kwargs.get("global_pos_offset"),
            input_ids.detach().cpu().clone(),
            labels.detach().cpu().clone(),
        ))
        shifted_labels = labels[:, 1:]
        has_supervision = (shifted_labels != -100).any()
        loss = self.weight * (1.0 if bool(has_supervision.item()) else 0.0)
        return {
            "loss": loss,
            "ponder_cost": torch.zeros((), device=self.weight.device),
            "commitment_cost": torch.zeros((), device=self.weight.device),
            "raw_topk_vals": None,
            "ltm_memory_state": None,
            "h_state": None,
            "l_state": None,
            "prev_context": None,
            "target_context": None,
            "drift_state": None,
        }


def test_training_checkpoint_preserves_resume_only_state():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)
    model.proj.weight.grad = torch.ones_like(model.proj.weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace()
    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=2,
        mid_epoch_step=7,
        running_states=(torch.ones(1), None, None, None, None, None),
    )

    assert checkpoint["checkpoint_kind"] == "training"
    assert checkpoint["completed_epoch"] == 2
    assert checkpoint["mid_epoch_step"] == 7
    assert torch.equal(checkpoint["model_state_dict"]["ltm.fast_vals"], torch.full((3, 2), 3.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm._mom_vals"], torch.full((3, 2), 4.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm.timestamps"], torch.full((3,), 5.0))
    assert torch.equal(checkpoint["model_state_dict"]["ltm.sources"], torch.full((3,), 2, dtype=torch.long))
    assert checkpoint["grad_accumulation_active"] is True
    assert "proj.weight" in checkpoint["grad_state_dict"]
    assert checkpoint["running_states"][0].device.type == "cpu"


def test_training_checkpoint_repairs_nonfinite_model_before_snapshot():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.data.view(-1)[0] = float("inf")
    args = SimpleNamespace(startup_weight_max_abs=0.0)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=1,
    )

    saved_weight = checkpoint["model_state_dict"]["proj.weight"]
    assert torch.isfinite(saved_weight).all()
    assert saved_weight.view(-1)[0].item() == 1.0
    assert model.proj.weight.data.view(-1)[0].item() == 1.0


def test_training_checkpoint_does_not_apply_startup_weight_clamp_mid_run():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.data.view(-1)[0] = 123.0
    args = SimpleNamespace(startup_weight_max_abs=100.0)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=1,
    )

    assert checkpoint["model_state_dict"]["proj.weight"].view(-1)[0].item() == 123.0
    assert model.proj.weight.data.view(-1)[0].item() == 123.0


def test_mid_epoch_checkpoint_preserves_v8_running_ltm_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ltm_state = (
        torch.full((2, 3, 2), 1.0),
        torch.full((2, 3, 2), 2.0),
        torch.arange(4).reshape(1, 4),
        [{"rosa": "state"}],
        torch.full((2, 3), 3.0),
        torch.full((2, 3), 2, dtype=torch.long),
    )
    running_states = (
        torch.ones(2, 4),
        torch.ones(2, 4) * 2,
        torch.ones(2, 2),
        torch.ones(2, 2) * 3,
        torch.ones(2, 2) * 4,
        ltm_state,
    )

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(),
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=5,
        running_states=running_states,
    )

    saved_ltm_state = checkpoint["running_states"][5]
    assert len(saved_ltm_state) == 6
    assert torch.equal(saved_ltm_state[0], ltm_state[0])
    assert torch.equal(saved_ltm_state[1], ltm_state[1])
    assert torch.equal(saved_ltm_state[2], ltm_state[2])
    assert saved_ltm_state[3] == [{"rosa": "state"}]
    assert torch.equal(saved_ltm_state[4], ltm_state[4])
    assert torch.equal(saved_ltm_state[5], ltm_state[5])


def test_remaining_update_steps_counts_mid_accumulation_boundaries():
    assert compute_update_steps(dataloader_len=100, accumulation_steps=4) == 25
    assert compute_update_steps(dataloader_len=101, accumulation_steps=4) == 26
    assert compute_remaining_update_steps(
        dataloader_len=100,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=5,
    ) == 24
    assert compute_remaining_update_steps(
        dataloader_len=100,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=2,
        start_step=5,
    ) == 49
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=100,
    ) == 1
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=1,
        start_step=101,
    ) == 1
    assert compute_remaining_update_steps(
        dataloader_len=101,
        accumulation_steps=4,
        start_epoch=0,
        total_epochs=2,
        start_step=101,
    ) == 26


def test_accumulation_helpers_flush_tail_window():
    assert accumulation_divisor_for_step(0, dataloader_len=5, accumulation_steps=4) == 4
    assert accumulation_divisor_for_step(3, dataloader_len=5, accumulation_steps=4) == 4
    assert accumulation_divisor_for_step(4, dataloader_len=5, accumulation_steps=4) == 1
    assert should_step_accumulation(2, dataloader_len=5, accumulation_steps=4) is False
    assert should_step_accumulation(3, dataloader_len=5, accumulation_steps=4) is True
    assert should_step_accumulation(4, dataloader_len=5, accumulation_steps=4) is True


def test_lr_scheduler_warms_up_then_cosine_decays():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        disable_lr_schedule=False,
        starting_lr=1e-3,
        min_lr=1e-5,
        warmup_steps=2,
        warmup_ratio=0.0,
    )

    scheduler = build_lr_scheduler(optimizer, args, num_update_steps=10)

    initial_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()
    warmed_lr = optimizer.param_groups[0]["lr"]
    for _ in range(9):
        optimizer.step()
        scheduler.step()
    final_lr = optimizer.param_groups[0]["lr"]

    assert 1e-5 < initial_lr < 1e-3
    assert abs(warmed_lr - 1e-3) < 1e-12
    assert abs(final_lr - 1e-5) < 1e-12


def test_ltm_lr_cosine_schedule_decays_and_advances():
    args = SimpleNamespace(
        ltm_lr=1e-3,
        min_ltm_lr=1e-5,
        min_lr=1e-7,
        disable_ltm_lr_schedule=False,
    )

    configure_ltm_lr_schedule(args, num_update_steps=10)

    assert abs(get_current_ltm_lr(args) - 1e-3) < 1e-12
    for _ in range(5):
        advance_ltm_lr_schedule(args)
    midpoint_lr = get_current_ltm_lr(args)
    assert 1e-5 < midpoint_lr < 1e-3
    for _ in range(5):
        advance_ltm_lr_schedule(args)
    assert abs(get_current_ltm_lr(args) - 1e-5) < 1e-12


def test_ltm_lr_scheduler_state_round_trips_in_training_checkpoint():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        ltm_lr=1e-4,
        min_ltm_lr=1e-8,
        min_lr=1e-8,
        disable_ltm_lr_schedule=False,
    )
    configure_ltm_lr_schedule(args, num_update_steps=20)
    for _ in range(7):
        advance_ltm_lr_schedule(args)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=args,
        dataloader=None,
        completed_epoch=0,
        mid_epoch_step=7,
    )

    expected = capture_ltm_lr_scheduler_state(args)
    assert checkpoint["ltm_scheduler_state"] == expected
    assert checkpoint["ltm_scheduler_state"]["step"] == 7
    assert checkpoint["ltm_scheduler_state"]["total_steps"] == 20


def test_epoch_boundary_checkpoint_has_clean_resume_position():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint = build_training_checkpoint(
        model,
        optimizer,
        scheduler=None,
        scaler=None,
        args=SimpleNamespace(),
        dataloader=None,
        completed_epoch=3,
        mid_epoch_step=0,
    )

    assert checkpoint["completed_epoch"] == 3
    assert checkpoint["mid_epoch_step"] == 0
    assert "running_states" not in checkpoint
    assert checkpoint["grad_accumulation_active"] is False
    assert checkpoint["grad_state_dict"] is None


def _continuation_parser_and_args(model_path=None, resume_from_ckpt=None, epochs=3):
    defaults = {
        "mode": "train",
        "model_path": model_path,
        "resume_from_ckpt": resume_from_ckpt,
        "out_dir": "./hierarchos_model",
        "epochs": epochs,
        "train": None,
        "hf_dataset": None,
        "hf_dataset_config": None,
        "hf_dataset_split": "train",
        "text_column": None,
        "prompt_column": None,
        "completion_column": None,
        "max_length": 1024,
        "training_chunk_size": 256,
        "batch_size": 64,
        "starting_lr": 1e-4,
        "min_lr": 1e-6,
        "warmup_steps": 0,
        "warmup_ratio": 0.0,
        "ltm_lr": 1e-3,
        "min_ltm_lr": None,
        "alpaca": False,
        "kayla": False,
        "compile": False,
        "force_compile": False,
        "amp": False,
        "train_prompt_tokens": True,
        "prompt_loss_weight": 1.0,
        "response_loss_weight": 1.0,
        "response_boundary_loss_weight": 1.0,
        "response_boundary_tokens": 0,
        "min_response_tokens": 1,
        "drop_empty_completions": True,
        "ponder_loss_weight": 0.01,
        "memory_gate_warmup_steps": 2000,
        "assistant_recovery": False,
        "refresh_hf_token_cache": False,
        "refresh_hf_shards": False,
    }
    parser = argparse.ArgumentParser()
    for key, value in defaults.items():
        parser.add_argument(f"--{key}", dest=key, default=value)
    return parser, SimpleNamespace(**defaults)


def test_cli_model_path_continuation_hydrates_saved_training_config():
    saved_config = {
        "hf_dataset": "netcat420/Experiment_0.1",
        "hf_dataset_split": "train",
        "alpaca": True,
        "max_length": 8880,
        "starting_lr": 7.5e-5,
        "min_lr": 9e-9,
        "ltm_lr": 5e-7,
        "min_ltm_lr": 1e-10,
        "compile": True,
        "force_compile": True,
        "amp": True,
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "hierarchos_config.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(saved_config, f)

        parser, args = _continuation_parser_and_args(model_path=tmp, epochs=3)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 7.5e-5
    assert args.min_lr == 9e-9
    assert args.ltm_lr == 5e-7
    assert args.min_ltm_lr == 1e-10
    assert args.compile is True
    assert args.force_compile is True
    assert args.amp is True
    assert args.train_prompt_tokens is True
    assert args.epochs == 3
    assert args.base_completed_epoch == 11


def test_assistant_recovery_defaults_target_large_assistant_sft():
    parser, args = _continuation_parser_and_args(epochs=3)
    args.assistant_recovery = True

    hierarchos_cli._apply_assistant_recovery_defaults(args, explicit_dests=set())

    assert args.alpaca is True
    assert args.epochs == 4
    assert args.starting_lr == 6e-5
    assert args.min_lr == 1e-6
    assert args.warmup_ratio == 0.03
    assert args.prompt_loss_weight == 0.10
    assert args.response_loss_weight == 1.0
    assert args.response_boundary_loss_weight == 2.0
    assert args.response_boundary_tokens == 32
    assert args.min_response_tokens == 16
    assert args.ponder_loss_weight == 0.003
    assert args.memory_gate_warmup_steps == 5000


def test_assistant_recovery_respects_explicit_overrides():
    parser, args = _continuation_parser_and_args(epochs=7)
    args.assistant_recovery = True

    hierarchos_cli._apply_assistant_recovery_defaults(
        args,
        explicit_dests={"epochs", "prompt_loss_weight", "warmup_ratio"},
    )

    assert args.epochs == 7
    assert args.prompt_loss_weight == 1.0
    assert args.warmup_ratio == 0.0
    assert args.response_boundary_tokens == 32


def test_resume_hydration_does_not_persist_refresh_cache_flags(tmp_path):
    ckpt_path = tmp_path / "hierarchos_epoch_1_step_600.pt"
    torch.save({
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "refresh_hf_token_cache": True,
            "refresh_hf_shards": True,
            "completed_epoch": 1,
        },
        "model_state_dict": {},
    }, ckpt_path)
    parser, args = _continuation_parser_and_args(resume_from_ckpt=str(ckpt_path))

    hierarchos_cli._hydrate_training_args_from_model_config(args, parser, explicit_dests=set())

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.refresh_hf_token_cache is False
    assert args.refresh_hf_shards is False


def test_cli_resume_checkpoint_hydrates_config_without_base_epoch_offset():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "alpaca": True,
            "max_length": 8880,
            "starting_lr": 7.5e-5,
            "ltm_lr": 5e-7,
        },
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "epoch11.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=14)
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={"epochs", "out_dir"},
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 7.5e-5
    assert args.ltm_lr == 5e-7
    assert args.train_prompt_tokens is True
    assert args.epochs == 14
    assert args.resume_completed_epoch == 11
    assert not hasattr(args, "base_completed_epoch")


def test_cli_resume_checkpoint_rejects_non_advancing_epoch_target():
    args = SimpleNamespace(
        mode="train",
        resume_from_ckpt="epoch11.pt",
        epochs=3,
        resume_completed_epoch=11,
    )

    try:
        hierarchos_cli._validate_resume_epoch_target(args)
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("Expected non-advancing resume target to exit")


def test_cli_resume_checkpoint_preserves_explicit_colab_overrides():
    checkpoint = {
        "config": {
            "hf_dataset": "netcat420/Experiment_0.1",
            "alpaca": True,
            "max_length": 1024,
            "starting_lr": 7.5e-5,
            "min_lr": 9e-9,
            "ltm_lr": 5e-7,
            "min_ltm_lr": 1e-10,
            "train_prompt_tokens": False,
        },
        "completed_epoch": 11,
    }
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "hierarchos_epoch_11.pt")
        torch.save(checkpoint, ckpt_path)

        parser, args = _continuation_parser_and_args(resume_from_ckpt=ckpt_path, epochs=14)
        args.max_length = 8880
        args.starting_lr = 1e-5
        args.min_lr = 1e-8
        args.ltm_lr = 1e-5
        args.min_ltm_lr = 1e-9
        args.train_prompt_tokens = True
        hierarchos_cli._hydrate_training_args_from_model_config(
            args,
            parser,
            explicit_dests={
                "epochs",
                "out_dir",
                "max_length",
                "starting_lr",
                "min_lr",
                "ltm_lr",
                "min_ltm_lr",
            },
        )

    assert args.hf_dataset == "netcat420/Experiment_0.1"
    assert args.alpaca is True
    assert args.max_length == 8880
    assert args.starting_lr == 1e-5
    assert args.min_lr == 1e-8
    assert args.ltm_lr == 1e-5
    assert args.min_ltm_lr == 1e-9
    assert args.train_prompt_tokens is True
    assert args.resume_completed_epoch == 11


def test_inference_sanitization_still_clears_transient_ltm():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm.timestamps.fill_(5.0)
    model.ltm.sources.fill_(2)

    state = sanitize_model_state_dict(model)

    assert torch.count_nonzero(state["ltm.fast_vals"]) == 0
    assert torch.count_nonzero(state["ltm._mom_vals"]) == 0
    assert torch.count_nonzero(state["ltm.timestamps"]) == 0
    assert torch.count_nonzero(state["ltm.sources"]) == 0


def test_restore_model_grad_state_round_trips_pending_accumulation():
    model = _FakeTrainModel()
    grad_state = {"proj.weight": torch.full_like(model.proj.weight, 9.0)}

    restored = restore_model_grad_state(model, grad_state, torch.device("cpu"))

    assert restored is True
    assert torch.equal(model.proj.weight.grad, torch.full_like(model.proj.weight, 9.0))


def test_training_state_finite_rejects_poisoned_optimizer_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.ones(1, 2)
    loss = model.proj(x).sum()
    loss.backward()
    optimizer.step()

    first_state = next(iter(optimizer.state.values()))
    first_state["exp_avg"].view(-1)[0] = float("nan")

    assert training_state_is_finite(model, optimizer) is False


def test_training_state_finite_rejects_poisoned_pending_grad():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.weight.grad.view(-1)[0] = float("nan")

    assert training_state_is_finite(model, include_grads=True) is False


def test_clip_gradients_and_check_saturates_nonfinite_gradients():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.bias.grad = torch.ones_like(model.proj.bias)
    model.proj.weight.grad.view(-1)[0] = float("inf")
    model.proj.weight.grad.view(-1)[1] = float("-inf")
    model.proj.bias.grad.view(-1)[0] = float("nan")

    ok, issue = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is True
    assert issue is not None or torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.bias.grad).all()


def test_clip_gradients_and_check_saturates_huge_finite_gradients():
    model = _FakeTrainModel()
    model.proj.weight.grad = torch.full_like(model.proj.weight, 1e30)
    model.proj.bias.grad = torch.full_like(model.proj.bias, -1e30)

    ok, issue = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is True
    assert issue is not None or torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.weight.grad).all()
    assert torch.isfinite(model.proj.bias.grad).all()
    assert model.proj.weight.grad.abs().max().item() <= 1.0
    assert model.proj.bias.grad.abs().max().item() <= 1.0


def test_gradient_sanitizer_preserves_finite_gradients_for_global_norm_clip():
    model = _FakeTrainModel()
    finite_grad = torch.tensor([[2.0, -3.0], [0.5, 1.5]])
    model.proj.weight.grad = finite_grad.clone()

    cleaned = _sanitize_gradient_nonfinite_(model, max_abs=1.0)

    assert cleaned == 0
    torch.testing.assert_close(model.proj.weight.grad, finite_grad)

    ok, total_norm = _clip_gradients_and_check(model, max_norm=1.0)

    assert ok is True
    assert total_norm.item() > 1.0
    expected = finite_grad * (1.0 / (finite_grad.norm().item() + 1e-6))
    torch.testing.assert_close(model.proj.weight.grad, expected)


def test_model_nonfinite_sanitizer_repairs_parameters_and_buffers():
    model = _FakeTrainModel()
    model.proj.weight.data.view(-1)[0] = float("inf")
    model.proj.weight.data.view(-1)[1] = float("-inf")
    model.proj.bias.data.view(-1)[0] = float("nan")
    model.ltm.fast_vals[0, 0] = float("inf")

    cleaned = _sanitize_model_nonfinite_(model)

    assert cleaned == 3
    assert model.proj.weight.data.view(-1)[0].item() == 1.0
    assert model.proj.weight.data.view(-1)[1].item() == -1.0
    assert model.proj.bias.data.view(-1)[0].item() == 0.0
    assert torch.isinf(model.ltm.fast_vals[0, 0])


def test_model_sanitizer_preserves_intentional_ltm_neg_inf_buffer():
    model = _FakeTrainModel()

    cleaned = _sanitize_model_nonfinite_(model)

    assert cleaned == 0
    assert torch.isneginf(model.ltm.neg_inf)
    assert training_state_is_finite(model) is True


def test_model_startup_magnitude_clamp_repairs_all_weights_and_buffers():
    model = _FakeTrainModel()
    model.proj.weight.data.fill_(123.0)
    model.proj.bias.data.fill_(-123.0)
    model.register_buffer("finite_buffer", torch.tensor([77.0]))

    clamped = _clamp_model_finite_magnitude_(model, 0.75)

    assert clamped == model.proj.weight.numel() + model.proj.bias.numel() + 1
    assert model.proj.weight.data.abs().max().item() == 0.75
    assert model.proj.bias.data.abs().max().item() == 0.75
    assert model.finite_buffer.abs().max().item() == 0.75


def test_train_step_skips_nonfinite_loss_before_backward():
    model = _NaNLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is True
    assert model.weight.grad is None
    assert model.reset_called is True


def test_train_step_saturates_nonfinite_gradient_before_optimizer_step():
    model = _InfGradModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    before = model.weight.detach().clone()
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert states == (None, None, None, None, None, None)
    assert args._train_step_had_nonfinite is False
    assert model.weight.grad is None
    assert not torch.equal(model.weight.detach(), before)
    assert model.reset_called is True


def test_train_step_caps_finite_loss_explosion_for_backward():
    model = _HighFiniteLossModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        max_ce_loss_for_backward=10.0,
        max_commitment_cost_for_backward=2.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    before = model.weight.detach().clone()
    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert outputs["loss"].item() == 20.0
    assert outputs["commitment_cost"].item() == 20.0
    assert args._train_step_had_nonfinite is False
    assert torch.equal(model.weight.detach(), before)
    assert states == (None, None, None, None, None, None)


def test_train_step_flushes_tail_accumulation_with_real_divisor():
    model = _FiniteLinearLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=10.0,
        max_ce_loss_for_backward=0.0,
    )
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
    }

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=4,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
        force_optimizer_step=False,
        accumulation_divisor=2,
    )

    assert outputs is not None
    assert states == (None, None, None, None, None, None)
    assert args._optimizer_step_was_taken is False
    assert model.weight.grad is not None
    assert abs(model.weight.grad.item() - 0.5) < 1e-6

    outputs, states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=4,
        step=1,
        args=args,
        running_states=states,
        force_optimizer_step=True,
        accumulation_divisor=2,
    )

    assert outputs is not None
    assert args._optimizer_step_was_taken is True
    assert model.weight.grad is None
    assert abs(model.weight.item() - 0.9) < 1e-6


def test_train_step_passes_one_token_label_lookahead_across_chunks():
    model = _BoundaryRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=3,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long),
        "labels": torch.tensor([[10, 11, 12, 13, 14, 15, 16]], dtype=torch.long),
        "attention_mask": torch.ones(1, 7, dtype=torch.long),
    }

    outputs, _states = train_step(
        model,
        batch,
        optimizer,
        scaler=None,
        accumulation_steps=1,
        step=0,
        args=args,
        running_states=(None, None, None, None, None, None),
    )

    assert outputs is not None
    assert [entry[0] for entry in model.seen] == [0, 3, 6]
    assert [entry[1].shape[1] for entry in model.seen] == [3, 3, 1]
    assert [entry[2].shape[1] for entry in model.seen] == [4, 4, 1]
    assert model.seen[0][2].tolist() == [[10, 11, 12, 13]]
    assert model.seen[1][2].tolist() == [[13, 14, 15, 16]]
    assert model.seen[2][2].tolist() == [[16]]


def test_train_step_rejects_masked_active_labels_for_alpaca_all_token_recovery():
    model = _BoundaryRecordingModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = SimpleNamespace(
        amp=False,
        training_chunk_size=8,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        alpaca=True,
        train_prompt_tokens=True,
        strict_all_token_loss=True,
    )
    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 13]], dtype=torch.long),
        "labels": torch.tensor([[10, -100, 12, 13]], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }

    try:
        train_step(
            model,
            batch,
            optimizer,
            scaler=None,
            accumulation_steps=1,
            step=0,
            args=args,
            running_states=(None, None, None, None, None, None),
        )
    except RuntimeError as exc:
        assert "All-token loss audit failed" in str(exc)
    else:
        raise AssertionError("masked active Alpaca labels should fail the recovery audit")

    assert model.seen == []


def test_ltm_transient_recovery_resets_fast_and_saturates_momentum():
    model = _FakeTrainModel()
    model.ltm.fast_vals.fill_(3.0)
    model.ltm._mom_vals.fill_(4.0)
    model.ltm._mom_vals[0, 0] = float("inf")
    model.ltm._mom_vals[0, 1] = float("-inf")
    model.ltm._mom_vals[1, 0] = float("nan")
    model.ltm.timestamps[0] = float("inf")
    model.ltm.sources[0] = 2

    cleaned = _sanitize_model_transient_state_(model, max_abs=0.75)

    assert cleaned > 0
    assert torch.count_nonzero(model.ltm.fast_vals) == 0
    assert model.ltm._mom_vals[0, 0].item() == 0.75
    assert model.ltm._mom_vals[0, 1].item() == -0.75
    assert model.ltm._mom_vals[1, 0].item() == 0.0
    assert model.ltm._mom_vals[1, 1].item() == 4.0
    assert model.ltm.timestamps[0].item() == 0.0


def test_checkpoint_save_allows_clean_state_and_writes_file():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "clean.pt")

        saved = save_training_checkpoint_if_finite({"ok": torch.tensor(1)}, path, model, optimizer)

        assert saved is True
        assert os.path.exists(path)


def test_checkpoint_save_sanitizes_poisoned_gradient():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.proj.weight.grad = torch.ones_like(model.proj.weight)
    model.proj.weight.grad.view(-1)[0] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_grad.pt")

        saved = save_training_checkpoint_if_finite({"bad": torch.tensor(1)}, path, model, optimizer)

        assert saved is True
        assert os.path.exists(path)
        assert model.proj.weight.grad.view(-1)[0].item() == 0.0


def test_checkpoint_save_sanitizes_poisoned_optimizer_state():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.ones(1, 2)
    loss = model.proj(x).sum()
    loss.backward()
    optimizer.step()
    next(iter(optimizer.state.values()))["exp_avg"].view(-1)[0] = float("nan")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_optimizer.pt")

        saved = save_training_checkpoint_if_finite({"bad": torch.tensor(1)}, path, model, optimizer)

        assert saved is True
        assert os.path.exists(path)
        assert next(iter(optimizer.state.values()))["exp_avg"].view(-1)[0].item() == 0.0


def test_checkpoint_save_sanitizes_poisoned_running_state_payload():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "running_states": (torch.tensor([float("nan")]), None, None, None, None, None),
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poisoned_running_state.pt")

        saved = save_training_checkpoint_if_finite(checkpoint, path, model, optimizer)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        assert saved is True
        assert os.path.exists(path)
        assert torch.equal(loaded["running_states"][0], torch.zeros(1))


def test_checkpoint_save_refreshes_stale_poisoned_model_snapshot():
    model = _FakeTrainModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    stale_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale_state["proj.weight"].view(-1)[0] = float("inf")
    checkpoint = {"model_state_dict": stale_state}

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "stale_model_snapshot.pt")

        saved = save_training_checkpoint_if_finite(checkpoint, path, model, optimizer)
        loaded = torch.load(path, map_location="cpu", weights_only=False)

        assert saved is True
        assert os.path.exists(path)
        assert torch.isfinite(loaded["model_state_dict"]["proj.weight"]).all()
        assert torch.equal(loaded["model_state_dict"]["proj.weight"], model.state_dict()["proj.weight"])


def test_length_grouped_sampler_state_restores_epoch_order():
    dataset = TensorDataset(torch.arange(12))
    sampler = LengthGroupedBatchSampler(
        lengths=list(range(1, 13)),
        batch_size=3,
        shuffle=True,
        seed=123,
    )
    dataloader = DataLoader(dataset, batch_sampler=sampler)
    sampler.set_epoch(4)
    expected_order = list(iter(sampler))
    saved_state = capture_dataloader_state(dataloader)

    sampler.seed = 999
    sampler.set_epoch(0)
    restore_dataloader_state(dataloader, saved_state)

    assert sampler.seed == 123
    assert sampler.epoch == 4
    assert list(iter(sampler)) == expected_order


def test_epoch_shuffle_sampler_state_restores_epoch_order():
    dataset = TensorDataset(torch.arange(10))
    sampler = EpochShuffleSampler(dataset, shuffle=True, seed=321)
    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    sampler.set_epoch(3)
    expected_order = list(iter(sampler))
    saved_state = capture_dataloader_state(dataloader)

    sampler.seed = 111
    sampler.set_epoch(0)
    restore_dataloader_state(dataloader, saved_state)

    assert sampler.seed == 321
    assert sampler.epoch == 3
    assert list(iter(sampler)) == expected_order
