from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hierarchos.training.datasets import EpochShuffleSampler, LengthGroupedBatchSampler
from hierarchos.training.trainer import (
    build_training_checkpoint,
    capture_dataloader_state,
    compute_remaining_update_steps,
    restore_dataloader_state,
    restore_model_grad_state,
)
from hierarchos.utils.checkpoint import sanitize_model_state_dict


class _FakeLTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("fast_vals", torch.zeros(3, 2))
        self.register_buffer("_mom_vals", torch.zeros(3, 2))
        self.register_buffer("timestamps", torch.zeros(3))
        self.register_buffer("sources", torch.zeros(3, dtype=torch.long))


class _FakeTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ltm = _FakeLTM()
        self.proj = nn.Linear(2, 2)
        self.config = {"context_dim": 2}


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
