"""
Hierarchos Evaluation Module

Optional integration with lm-evaluation-harness for standardized benchmarking.

Usage:
    from hierarchos.evaluation import run_eval, is_lm_eval_available
    
    if is_lm_eval_available():
        results = run_eval(model, tokenizer, device, tasks=["hellaswag"])
"""

from .evaluator import run_eval, is_lm_eval_available, format_results, save_results

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
    "HierarchosLM"
]
