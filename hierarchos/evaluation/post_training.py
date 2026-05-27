"""
Post-training benchmark orchestration for Hierarchos models.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .benchmarks import (
    BenchmarkSpec,
    benchmark_manifest,
    missing_harness_tasks,
    resolve_task_names,
    specs_for_tasks,
)
from .evaluator import format_results, is_lm_eval_available, run_eval


ARC_AGI_KEYS = {"arc_agi", "arc_agi_2"}


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    cleaned = cleaned.strip("-._")
    return cleaned or "benchmark"


def merge_results(*results_list: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {"results": {}, "versions": {}, "config": {}}
    samples: Dict[str, Any] = {}
    for results in results_list:
        if not results:
            continue
        if isinstance(results.get("results"), dict):
            merged["results"].update(results["results"])
        if isinstance(results.get("versions"), dict):
            merged["versions"].update(results["versions"])
        if isinstance(results.get("config"), dict):
            merged["config"].update(results["config"])
        if isinstance(results.get("samples"), dict):
            samples.update(results["samples"])
    if samples:
        merged["samples"] = samples
    return merged


def write_benchmark_artifacts(
    output_dir: str,
    run_name: Optional[str],
    results: Dict[str, Any],
    manifest: Dict[str, Any],
    skipped: Sequence[BenchmarkSpec],
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = _slug(run_name or f"hierarchos-benchmark-{timestamp}")
    run_dir = os.path.abspath(os.path.join(output_dir, name))
    os.makedirs(run_dir, exist_ok=True)

    results_path = os.path.join(run_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    manifest_path = os.path.join(run_dir, "manifest.json")
    manifest_to_write = dict(manifest)
    manifest_to_write["generated_at"] = timestamp
    manifest_to_write["skipped"] = [spec.to_dict() for spec in skipped]
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_to_write, f, indent=2, default=str)

    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# Hierarchos Benchmark Summary\n\n")
        f.write("```text\n")
        f.write(format_results(results))
        f.write("\n```\n")
        if skipped:
            f.write("\n## Skipped External Benchmarks\n\n")
            for spec in skipped:
                f.write(f"- {spec.display_name} (`{spec.key}`): {spec.notes or spec.description}\n")

    return run_dir


def run_post_training_benchmarks(
    model,
    tokenizer,
    device,
    benchmark_names: Optional[Sequence[str]] = None,
    suite_names: Optional[Sequence[str]] = None,
    raw_tasks: Optional[Sequence[str]] = None,
    batch_size: int = 1,
    limit: Optional[int] = None,
    num_fewshot: Optional[int] = None,
    verbosity: str = "WARNING",
    strict: bool = False,
    arc_agi_path: Optional[str] = None,
    arc_agi_max_tasks: Optional[int] = None,
    arc_agi_max_test_items: Optional[int] = None,
    arc_agi_keep_samples: bool = False,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    sequential: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[BenchmarkSpec]]:
    requested = list(benchmark_names or [])
    if suite_names:
        requested = list(suite_names) + requested

    task_names, external_specs, unknown = resolve_task_names(
        raw_tasks or requested,
        default_suite=None if raw_tasks else "frontier-text",
        include_external=True,
    )

    missing = missing_harness_tasks(task_names) if strict else []
    if missing:
        raise ValueError(
            "lm-eval does not know these tasks: "
            + ", ".join(missing)
            + ". Install a newer lm-eval or pass different benchmark names."
        )

    results_parts: List[Optional[Dict[str, Any]]] = []
    skipped: List[BenchmarkSpec] = list(external_specs)

    failed_tasks: List[str] = []

    if task_names:
        if not is_lm_eval_available():
            raise ImportError("lm-evaluation-harness is not installed. Install with: pip install lm-eval")
        task_batches = [[task] for task in task_names] if sequential else [task_names]
        for index, task_batch in enumerate(task_batches, 1):
            if sequential:
                task_label = task_batch[0]
                print(f"\n[{index}/{len(task_batches)}] Running benchmark: {task_label}")
            task_results = run_eval(
                model=model,
                tokenizer=tokenizer,
                device=device,
                tasks=task_batch,
                batch_size=batch_size,
                limit=limit,
                num_fewshot=num_fewshot,
                verbosity=verbosity,
            )
            if task_results is None:
                failed_tasks.extend(task_batch)
                continue
            results_parts.append(task_results)

    if arc_agi_path:
        from .arc_agi import run_arc_agi_eval

        arc_specs = [spec for spec in external_specs if spec.key in ARC_AGI_KEYS]
        for spec in arc_specs:
            results_parts.append(
                run_arc_agi_eval(
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    dataset_path=arc_agi_path,
                    max_tasks=arc_agi_max_tasks,
                    max_test_items=arc_agi_max_test_items,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    keep_samples=arc_agi_keep_samples,
                    result_key=spec.key,
                )
            )
        skipped = [spec for spec in skipped if spec.key not in ARC_AGI_KEYS]

    results = merge_results(*results_parts)
    manifest = benchmark_manifest(
        task_names=task_names,
        external=external_specs,
        requested=benchmark_names or raw_tasks or [],
        suites=suite_names or [],
    )
    manifest["resolved_lm_eval_tasks"] = task_names
    manifest["unknown_raw_tasks"] = unknown
    manifest["runnable_benchmarks"] = [spec.to_dict() for spec in specs_for_tasks(task_names)]
    manifest["sequential"] = sequential
    manifest["failed_lm_eval_tasks"] = failed_tasks
    return results, manifest, skipped
