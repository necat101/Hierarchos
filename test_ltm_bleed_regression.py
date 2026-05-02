import torch

from hierarchos import AttrDict, HierarchosCore, LTMModule
from hierarchos.inference.chat import extract_correction_text, parse_temperature_setting, passive_response_quality
from hierarchos.training.trainer import compute_chunk_training_weights


def _config():
    return AttrDict(
        vocab_size=128,
        context_dim=24,
        persistent_dim=8,
        ltm_slots=24,
        ltm_key_dim=12,
        ltm_val_dim=12,
        ltm_lr=0.05,
        ltm_topk=2,
        ltm_forget_rate=0.0,
        h_hidden=24,
        l_hidden=24,
        max_h_steps=1,
        max_l_steps=1,
        h_stride=2,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        isolate_batch_ltm=True,
        use_deepembed=False,
        use_rosa=False,
        compile=False,
    )


def test_plain_eval_forward_is_read_only_by_default():
    torch.manual_seed(7)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3], [31, 32, 33]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        out = model(input_ids)

    ltm_state = out["ltm_memory_state"]
    assert ltm_state[0].shape == (2, model.config.ltm_slots, model.config.ltm_val_dim)
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert torch.allclose(ltm_state[0], initial_fast.unsqueeze(0).expand_as(ltm_state[0]))


def test_validation_hebbian_batch_ltm_state_is_isolated_from_global_buffer():
    torch.manual_seed(7)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[1, 2, 3], [31, 32, 33]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        out = model(input_ids, allow_hebbian_update=True)

    ltm_state = out["ltm_memory_state"]
    assert ltm_state[0].shape == (2, model.config.ltm_slots, model.config.ltm_val_dim)
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert not torch.allclose(ltm_state[0], initial_fast.unsqueeze(0).expand_as(ltm_state[0]))


def test_suppressed_hebbian_leaves_ltm_unchanged_during_generation():
    torch.manual_seed(11)
    model = HierarchosCore(_config())
    model.eval()
    model.suppress_hebbian = True

    input_ids = torch.tensor([[4, 5, 6]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()
    initial_mom = model.ltm._mom_vals.clone()

    with torch.no_grad():
        out = model(input_ids)

    ltm_state = out["ltm_memory_state"]
    assert torch.allclose(model.ltm.fast_vals, initial_fast)
    assert torch.allclose(model.ltm._mom_vals, initial_mom)
    assert torch.allclose(ltm_state[0], initial_fast)
    assert torch.allclose(ltm_state[1], initial_mom)


def test_explicit_hebbian_still_writes_single_sequence_working_memory():
    torch.manual_seed(13)
    model = HierarchosCore(_config())
    model.eval()

    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
    initial_fast = model.ltm.fast_vals.clone()

    with torch.no_grad():
        model(input_ids, allow_hebbian_update=True)

    diff = (model.ltm.fast_vals - initial_fast).abs().sum()
    assert diff > 0


def test_ltm_2d_update_matches_trainer_batched_update():
    torch.manual_seed(17)
    ltm = LTMModule(
        n_slots=8,
        key_dim=4,
        val_dim=4,
        momentum=0.9,
        wd=0.01,
        forget_rate=0.05,
        reference_chunk_len=4,
    )

    topk_idx = torch.tensor([[[0, 1], [2, 1], [1, -1]]], dtype=torch.long)
    grads = torch.randn(1, 3, 2, 4)
    fast = torch.randn(8, 4)
    mom = torch.randn(8, 4) * 0.1
    timestamps = torch.zeros(8)
    sources = torch.zeros(8, dtype=torch.long)

    fast_2d, mom_2d = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.03,
        timestamp=5.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        timestamps=timestamps.clone(),
        sources=sources.clone(),
        inplace=True,
    )
    fast_3d, mom_3d = ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.03,
        timestamp=5.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=3,
        fast_vals=fast.unsqueeze(0).clone(),
        mom_vals=mom.unsqueeze(0).clone(),
        timestamps=timestamps.unsqueeze(0).clone(),
        sources=sources.unsqueeze(0).clone(),
        inplace=True,
    )

    assert torch.allclose(fast_2d, fast_3d[0], atol=1e-6)
    assert torch.allclose(mom_2d, mom_3d[0], atol=1e-6)


def test_hebbian_update_uses_same_rule_as_negative_value_gradient():
    torch.manual_seed(19)
    ltm = LTMModule(n_slots=8, key_dim=4, val_dim=4, momentum=0.9, wd=0.01, forget_rate=0.02)
    topk_idx = torch.tensor([[[0, 1], [3, 1]]], dtype=torch.long)
    vals = torch.randn(1, 2, 2, 4)
    fast = torch.randn(1, 8, 4)
    mom = torch.randn(1, 8, 4) * 0.1

    fast_grad, mom_grad = ltm.inner_update(
        topk_idx,
        -vals,
        current_lr=0.02,
        timestamp=7.0,
        source=LTMModule.SRC_USER_INTERACTION,
        tokens_covered=2,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        inplace=True,
    )
    fast_hebb, mom_hebb = ltm.update_memory_hebbian(
        topk_idx,
        None,
        vals,
        current_lr=0.02,
        timestamp=7.0,
        source=LTMModule.SRC_USER_INTERACTION,
        tokens_covered=2,
        fast_vals=fast.clone(),
        mom_vals=mom.clone(),
        inplace=True,
    )

    assert torch.allclose(fast_grad, fast_hebb, atol=1e-6)
    assert torch.allclose(mom_grad, mom_hebb, atol=1e-6)


def test_feedback_path_raw_topk_tensors_receive_gradients():
    torch.manual_seed(23)
    model = HierarchosCore(_config())
    model.train()

    input_ids = torch.tensor([[11, 12, 13, 14]], dtype=torch.long)
    labels = input_ids.clone()

    outputs = model(input_ids=input_ids, labels=labels, suppress_hebbian=True)
    raw_topk_vals = outputs["raw_topk_vals"]
    for value_tensor in raw_topk_vals:
        if value_tensor.requires_grad:
            value_tensor.retain_grad()

    outputs["loss"].backward()
    grad_norm = sum(
        value_tensor.grad.abs().sum().item()
        for value_tensor in raw_topk_vals
        if value_tensor.grad is not None
    )

    assert grad_norm > 0


def test_inplace_ltm_update_accumulates_deltas_like_state_delta():
    torch.manual_seed(29)
    ltm = LTMModule(n_slots=6, key_dim=4, val_dim=4, momentum=0.0, wd=0.0, forget_rate=0.0)
    ltm.accumulate_deltas = True

    topk_idx = torch.tensor([[[0, 1], [1, 2]]], dtype=torch.long)
    grads = torch.randn(1, 2, 2, 4)
    before = ltm.fast_vals.clone()

    ltm.inner_update(
        topk_idx,
        grads,
        current_lr=0.04,
        timestamp=3.0,
        source=LTMModule.SRC_TRAINING_DATA,
        tokens_covered=2,
        inplace=True,
    )

    assert torch.allclose(ltm.ltm_deltas, ltm.fast_vals - before, atol=1e-6)
    assert ltm.ltm_deltas.abs().sum() > 0


def test_chat_correction_and_passive_quality_gates():
    assert extract_correction_text("its 4") == "4"
    assert extract_correction_text("Actually, the answer is four") == "the answer is four"
    assert passive_response_quality([1, 2, 3, 1, 2, 3, 1, 2, 3])[0] is False
    assert passive_response_quality(list(range(12)))[0] is True


def test_runtime_temperature_setting_grid():
    assert parse_temperature_setting("0") == 0.0
    assert parse_temperature_setting("0.05") == 0.05
    assert parse_temperature_setting("1") == 1.0

    for bad_value in ("-0.05", "1.05", "0.03"):
        try:
            parse_temperature_setting(bad_value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad_value} should not be accepted")


def test_training_chunk_weights_follow_supervised_answer_tokens():
    labels = torch.full((1, 260), -100, dtype=torch.long)
    labels[0, 250:260] = torch.arange(10)

    weights = compute_chunk_training_weights(labels, chunk_size=128)

    assert len(weights) == 3
    assert weights[0]["valid_predictions"] == 0
    assert weights[1]["valid_predictions"] == 6
    assert weights[2]["valid_predictions"] == 3
    assert weights[0]["label_ratio"] == 0.0
    assert abs(sum(chunk["label_ratio"] for chunk in weights) - 1.0) < 1e-8


def test_training_chunk_token_weights_ignore_padding_only_chunks():
    labels = torch.full((1, 260), -100, dtype=torch.long)
    attention_mask = torch.zeros((1, 260), dtype=torch.long)
    attention_mask[0, :130] = 1

    weights = compute_chunk_training_weights(labels, attention_mask=attention_mask, chunk_size=128)

    assert weights[0]["real_tokens"] == 128
    assert weights[1]["real_tokens"] == 2
    assert weights[2]["real_tokens"] == 0
    assert weights[2]["token_ratio"] == 0.0
    assert abs(sum(chunk["token_ratio"] for chunk in weights) - 1.0) < 1e-8


def test_masked_padding_tokens_do_not_change_training_costs():
    torch.manual_seed(31)
    model = HierarchosCore(_config())
    model.train()

    input_a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0]], dtype=torch.long)
    input_b = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 91, 92, 93, 94, 95]], dtype=torch.long)
    labels = input_a.clone()
    labels[:, 7:] = -100
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.long)

    out_a = model(input_a, labels=labels, attention_mask=attention_mask, suppress_hebbian=True)
    out_b = model(input_b, labels=labels, attention_mask=attention_mask, suppress_hebbian=True)

    assert torch.allclose(out_a["loss"], out_b["loss"], atol=1e-6)
    assert torch.allclose(out_a["ponder_cost"], out_b["ponder_cost"], atol=1e-6)
    assert torch.allclose(out_a["commitment_cost"], out_b["commitment_cost"], atol=1e-6)
