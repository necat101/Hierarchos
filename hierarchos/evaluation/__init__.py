"""
Hierarchos Evaluation Module

Optional integration with lm-evaluation-harness for standardized benchmarking.

Usage:
    from hierarchos.evaluation import run_eval, is_lm_eval_available
    
    if is_lm_eval_available():
        results = run_eval(model, tokenizer, device, tasks=["hellaswag"])
"""

from .evaluator import run_eval, is_lm_eval_available, format_results, save_results
from .benchmarks import (
    BENCHMARKS,
    SUITES,
    BenchmarkSpec,
    format_benchmark_catalog,
    get_benchmark,
    list_benchmarks,
    list_suites,
    resolve_task_names,
)
from .post_training import (
    run_post_training_benchmarks,
    write_benchmark_artifacts,
)

# Conditionally export HierarchosLM
try:
    from .lm_eval_wrapper import HierarchosLM
except ImportError:
    HierarchosLM = None

__all__ = [
    "run_eval",
    "is_lm_eval_available", 
    "format_results",
    "save_results",
    "HierarchosLM",
    "BENCHMARKS",
    "SUITES",
    "BenchmarkSpec",
    "format_benchmark_catalog",
    "get_benchmark",
    "list_benchmarks",
    "list_suites",
    "resolve_task_names",
    "run_post_training_benchmarks",
    "write_benchmark_artifacts",
]
