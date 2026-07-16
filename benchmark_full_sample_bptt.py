"""Reproducible full-sample BPTT activation/timing probe.

CPU runs validate the implementation when CUDA is unavailable. On the rental,
pass ``--device cuda --amp`` to collect synchronized step time and peak allocated
VRAM for the exact same direct/checkpointed graphs.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch

from hierarchos import AttrDict, HierarchosCore
from hierarchos.training.trainer import train_step
from hierarchos.utils.rosa import precompute_rosa_ids_for_chunks


def tiny_config(sequence_length):
    return AttrDict(
        vocab_size=128,
        context_dim=16,
        persistent_dim=8,
        ltm_slots=16,
        ltm_key_dim=8,
        ltm_val_dim=8,
        ltm_topk=2,
        h_hidden=16,
        l_hidden=16,
        h_stride=2,
        max_h_steps=2,
        max_l_steps=2,
        max_length=sequence_length,
        l_conv_atol=1e-4,
        commitment_threshold=0.05,
        use_deepembed=True,
        use_rosa=True,
        rosa_max_context=sequence_length,
        detach_every_n_steps=None,
        gradient_checkpointing=False,
        compile=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def model_kwargs(input_ids, rosa_ids):
    return dict(
        input_ids=input_ids,
        labels=input_ids.clone(),
        attention_mask=torch.ones_like(input_ids),
        h_state=None,
        l_state=None,
        prev_context=None,
        target_context=None,
        drift_state=None,
        ltm_memory_state=None,
        global_pos_offset=0,
        return_logits=True,
        return_topk_values=False,
        return_raw_topk_values=False,
        return_topk_indices=False,
        compute_ltm_value_alignment=False,
        rosa_ids=rosa_ids,
        loss_weights=None,
    )


def combined_loss(outputs):
    return (
        outputs["loss"]
        + 0.01 * outputs["ponder_cost"]
        + 0.5 * outputs["commitment_cost"]
    )


def training_args(*, checkpointed, checkpoint_segment_size, amp):
    return SimpleNamespace(
        amp=bool(amp),
        amp_dtype="bfloat16",
        training_chunk_size=16,
        full_sample_bptt=True,
        full_sample_activation_checkpointing=bool(checkpointed),
        full_sample_checkpoint_segment_size=int(checkpoint_segment_size),
        persist_state=False,
        compile=False,
        pad_token_id=0,
        padding_metrics=False,
        cpu_chunked_lm_loss=False,
        cuda_chunked_lm_loss=False,
        grad_clip=1.0,
        ltm_training_mode="read-only",
        adaptive_ponder=False,
        ponder_loss_weight=0.01,
        commitment_loss_weight=0.5,
        max_ce_loss_for_backward=0.0,
        max_ponder_cost_for_backward=0.0,
        max_commitment_cost_for_backward=0.0,
        ltm_value_alignment_weight=0.0,
    )


def run_step(model, optimizer, kwargs, train_args, device):
    model.zero_grad(set_to_none=True)
    model.reset_memory()
    saved_forward_bytes = 0

    def pack(tensor):
        nonlocal saved_forward_bytes
        saved_forward_bytes += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(tensor):
        return tensor

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_start = torch.cuda.memory_allocated(device)
    else:
        memory_start = None

    synchronize(device)
    started = time.perf_counter()
    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        outputs, states = train_step(
            model,
            {
                "input_ids": kwargs["input_ids"],
                "labels": kwargs["labels"],
                "attention_mask": kwargs["attention_mask"],
                "rosa_ids": kwargs["rosa_ids"],
            },
            optimizer,
            scaler=None,
            # Keep gradients for the correctness comparison without taking an
            # optimizer step; both paths receive the same divisor.
            accumulation_steps=2,
            step=0,
            args=train_args,
            running_states=(None, None, None, None, None, None),
            collect_metrics=True,
        )
    synchronize(device)
    elapsed = time.perf_counter() - started

    peak_increment = None
    if device.type == "cuda":
        peak_increment = torch.cuda.max_memory_allocated(device) - memory_start
    grads = {
        name: None if parameter.grad is None else parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    combined = outputs["loss"].detach().float()
    if outputs.get("ponder_cost") is not None:
        combined = combined + 0.01 * outputs["ponder_cost"].detach().float()
    if outputs.get("commitment_cost") is not None:
        combined = combined + 0.5 * outputs["commitment_cost"].detach().float()
    return {
        "seconds": elapsed,
        "saved_forward_bytes": saved_forward_bytes,
        "peak_cuda_increment_bytes": peak_increment,
        "loss": float(combined.cpu().item()),
        "states": states,
        "grads": grads,
    }


def summarize(samples, tokens):
    seconds = statistics.median(sample["seconds"] for sample in samples)
    saved = statistics.median(sample["saved_forward_bytes"] for sample in samples)
    peaks = [sample["peak_cuda_increment_bytes"] for sample in samples]
    peak = statistics.median(peaks) if peaks and peaks[0] is not None else None
    return {
        "median_step_seconds": seconds,
        "median_tokens_per_second": tokens / seconds,
        "median_saved_forward_bytes_proxy": saved,
        "median_peak_cuda_increment_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--checkpoint-segment-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--require-blackwell",
        action="store_true",
        help="Fail unless capability 12.0 and this PyTorch build advertises sm_120.",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(args.device)
    capability = None
    cuda_arch_list = []
    if device.type == "cuda":
        capability = tuple(torch.cuda.get_device_capability(device))
        cuda_arch_list = list(torch.cuda.get_arch_list())
        has_native_arch = any(arch.startswith("sm_120") for arch in cuda_arch_list)
        if args.require_blackwell and (capability != (12, 0) or not has_native_arch):
            raise RuntimeError(
                "Blackwell preflight failed: "
                f"capability={capability}, cuda_arch_list={cuda_arch_list}. "
                "Install a PyTorch CUDA build with sm_120 before renting training time."
            )
    torch.manual_seed(20260715)

    config = tiny_config(args.sequence_length)
    direct_model = HierarchosCore(config).to(device).train()
    segmented_model = HierarchosCore(AttrDict(dict(config))).to(device).train()
    segmented_model.load_state_dict(direct_model.state_dict())
    direct_optimizer = torch.optim.SGD(direct_model.parameters(), lr=0.0)
    segmented_optimizer = torch.optim.SGD(segmented_model.parameters(), lr=0.0)
    direct_args = training_args(
        checkpointed=False,
        checkpoint_segment_size=args.checkpoint_segment_size,
        amp=args.amp,
    )
    segmented_args = training_args(
        checkpointed=True,
        checkpoint_segment_size=args.checkpoint_segment_size,
        amp=args.amp,
    )

    base = torch.arange(args.sequence_length, dtype=torch.long).remainder(31).add(2)
    input_ids = base.unsqueeze(0).repeat(args.batch_size, 1).to(device)
    rosa_row = precompute_rosa_ids_for_chunks(
        base.tolist(),
        vocab_size=config.vocab_size,
        chunk_size=16,
        rosa_max_ctx=config.rosa_max_context,
    )
    rosa_ids = torch.tensor(rosa_row, dtype=torch.long, device=device).unsqueeze(0)
    rosa_ids = rosa_ids.repeat(args.batch_size, 1)
    kwargs = model_kwargs(input_ids, rosa_ids)

    # Warm both paths before collecting synchronized medians.
    run_step(direct_model, direct_optimizer, kwargs, direct_args, device)
    run_step(segmented_model, segmented_optimizer, kwargs, segmented_args, device)
    direct = [
        run_step(direct_model, direct_optimizer, kwargs, direct_args, device)
        for _ in range(args.repeats)
    ]
    segmented = [
        run_step(segmented_model, segmented_optimizer, kwargs, segmented_args, device)
        for _ in range(args.repeats)
    ]

    direct_probe = direct[-1]
    checkpoint_probe = segmented[-1]
    max_grad_delta = 0.0
    grad_none_mismatches = 0
    for name, direct_grad in direct_probe["grads"].items():
        segmented_grad = checkpoint_probe["grads"][name]
        if (direct_grad is None) != (segmented_grad is None):
            grad_none_mismatches += 1
        elif direct_grad is not None:
            max_grad_delta = max(
                max_grad_delta,
                float((direct_grad - segmented_grad).abs().max().item()),
            )

    tokens = args.batch_size * args.sequence_length
    direct_summary = summarize(direct, tokens)
    checkpoint_summary = summarize(segmented, tokens)
    result = {
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "device_capability": capability,
        "cuda_arch_list": cuda_arch_list,
        "amp": bool(args.amp),
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "checkpoint_segment_size": args.checkpoint_segment_size,
        "repeats": args.repeats,
        "direct_full_bptt": direct_summary,
        "segmented_checkpointed_full_bptt": checkpoint_summary,
        "segmented_checkpoint_to_direct_step_time_ratio": (
            checkpoint_summary["median_step_seconds"]
            / direct_summary["median_step_seconds"]
        ),
        "segmented_saved_forward_proxy_reduction_fraction": (
            1.0
            - checkpoint_summary["median_saved_forward_bytes_proxy"]
            / max(1, direct_summary["median_saved_forward_bytes_proxy"])
        ),
        "correctness": {
            "loss_absolute_delta": abs(direct_probe["loss"] - checkpoint_probe["loss"]),
            "maximum_parameter_gradient_absolute_delta": max_grad_delta,
            "gradient_none_mismatches": grad_none_mismatches,
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
