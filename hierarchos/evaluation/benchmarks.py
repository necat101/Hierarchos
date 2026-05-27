"""
Benchmark registry and suite expansion for Hierarchos evaluation.

The registry separates benchmarks Hierarchos can run through
lm-evaluation-harness today from frontier benchmarks that need an external
agent, browser, terminal, or multimodal harness.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    display_name: str
    category: str
    modality: str
    harness_task: Optional[str]
    primary_metric: str
    description: str
    citation_url: str = ""
    notes: str = ""

    @property
    def runnable(self) -> bool:
        return bool(self.harness_task) and self.modality == "text"

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["runnable"] = self.runnable
        return data


_BENCHMARKS: Tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        key="mmlu",
        display_name="MMLU",
        category="knowledge",
        modality="text",
        harness_task="mmlu",
        primary_metric="acc",
        description="Broad undergraduate and professional knowledge benchmark.",
        citation_url="https://arxiv.org/abs/2009.03300",
    ),
    BenchmarkSpec(
        key="mmlu_pro",
        display_name="MMLU-Pro",
        category="knowledge",
        modality="text",
        harness_task="mmlu_pro",
        primary_metric="acc",
        description="Harder MMLU-style multiple-choice benchmark with more choices and reasoning.",
        citation_url="https://arxiv.org/abs/2406.01574",
    ),
    BenchmarkSpec(
        key="gpqa",
        display_name="GPQA",
        category="science_reasoning",
        modality="text",
        harness_task="gpqa_main_n_shot",
        primary_metric="acc",
        description="Graduate-level Google-proof science questions.",
        citation_url="https://arxiv.org/abs/2311.12022",
    ),
    BenchmarkSpec(
        key="gpqa_diamond",
        display_name="GPQA Diamond",
        category="science_reasoning",
        modality="text",
        harness_task="gpqa_diamond_n_shot",
        primary_metric="acc",
        description="Expert-filtered GPQA subset frequently reported by frontier labs.",
        citation_url="https://arxiv.org/abs/2311.12022",
    ),
    BenchmarkSpec(
        key="aime25",
        display_name="AIME 2025",
        category="math",
        modality="text",
        harness_task="aime25",
        primary_metric="exact_match",
        description="High-school math competition questions from the 2025 AIME.",
    ),
    BenchmarkSpec(
        key="aime24",
        display_name="AIME 2024",
        category="math",
        modality="text",
        harness_task="aime24",
        primary_metric="exact_match",
        description="High-school math competition questions from the 2024 AIME.",
    ),
    BenchmarkSpec(
        key="gsm8k_cot",
        display_name="GSM8K CoT",
        category="math",
        modality="text",
        harness_task="gsm8k_cot",
        primary_metric="exact_match",
        description="Grade-school math word problems with chain-of-thought prompting.",
        citation_url="https://arxiv.org/abs/2110.14168",
    ),
    BenchmarkSpec(
        key="minerva_math",
        display_name="Minerva Math",
        category="math",
        modality="text",
        harness_task="minerva_math",
        primary_metric="exact_match",
        description="Competition-style mathematical problem solving.",
        citation_url="https://arxiv.org/abs/2206.14858",
    ),
    BenchmarkSpec(
        key="bbh",
        display_name="BIG-Bench Hard CoT",
        category="reasoning",
        modality="text",
        harness_task="bbh_cot_fewshot",
        primary_metric="exact_match",
        description="Challenging BIG-Bench reasoning tasks with few-shot CoT prompting.",
        citation_url="https://arxiv.org/abs/2210.09261",
    ),
    BenchmarkSpec(
        key="agieval",
        display_name="AGIEval",
        category="reasoning",
        modality="text",
        harness_task="agieval",
        primary_metric="acc",
        description="Human exam and admission-test style reasoning tasks.",
        citation_url="https://arxiv.org/abs/2304.06364",
    ),
    BenchmarkSpec(
        key="drop",
        display_name="DROP",
        category="reading",
        modality="text",
        harness_task="drop",
        primary_metric="f1",
        description="Discrete reasoning over paragraphs.",
        citation_url="https://arxiv.org/abs/1903.00161",
    ),
    BenchmarkSpec(
        key="ifeval",
        display_name="IFEval",
        category="instruction_following",
        modality="text",
        harness_task="ifeval",
        primary_metric="prompt_level_strict_acc",
        description="Instruction-following constraint adherence.",
        citation_url="https://arxiv.org/abs/2311.07911",
    ),
    BenchmarkSpec(
        key="hellaswag",
        display_name="HellaSwag",
        category="commonsense",
        modality="text",
        harness_task="hellaswag",
        primary_metric="acc_norm",
        description="Commonsense natural-language continuation selection.",
        citation_url="https://arxiv.org/abs/1905.07830",
    ),
    BenchmarkSpec(
        key="arc_challenge",
        display_name="ARC-Challenge",
        category="commonsense",
        modality="text",
        harness_task="arc_challenge",
        primary_metric="acc_norm",
        description="Hard grade-school science multiple-choice questions.",
        citation_url="https://arxiv.org/abs/1803.05457",
    ),
    BenchmarkSpec(
        key="arc_agi",
        display_name="ARC-AGI",
        category="frontier_reasoning",
        modality="arc_grid",
        harness_task=None,
        primary_metric="exact_match",
        description="Abstraction and Reasoning Corpus grid-transformation tasks.",
        citation_url="https://arcprize.org/arc-agi",
        notes="Run with --arc-agi-path pointing at ARC-style JSON files.",
    ),
    BenchmarkSpec(
        key="arc_agi_2",
        display_name="ARC-AGI-2",
        category="frontier_reasoning",
        modality="arc_grid",
        harness_task=None,
        primary_metric="exact_match",
        description="Second-generation ARC-AGI benchmark for fluid reasoning.",
        citation_url="https://arxiv.org/abs/2505.11831",
        notes="Run with --arc-agi-path when ARC-AGI-2 JSON tasks are available locally.",
    ),
    BenchmarkSpec(
        key="arc_agi_3",
        display_name="ARC-AGI-3",
        category="agentic_reasoning",
        modality="agentic",
        harness_task=None,
        primary_metric="relative_human_action_efficiency",
        description="Interactive ARC Prize benchmark for exploration, planning, and adaptation in novel game environments.",
        citation_url="https://docs.arcprize.org/",
        notes="Official path uses the ARC-AGI Toolkit/API and the ARC-AGI-3 Kaggle competition.",
    ),
    BenchmarkSpec(
        key="arc_easy",
        display_name="ARC-Easy",
        category="commonsense",
        modality="text",
        harness_task="arc_easy",
        primary_metric="acc_norm",
        description="Easier ARC science question split.",
        citation_url="https://arxiv.org/abs/1803.05457",
    ),
    BenchmarkSpec(
        key="winogrande",
        display_name="WinoGrande",
        category="commonsense",
        modality="text",
        harness_task="winogrande",
        primary_metric="acc",
        description="Large-scale Winograd-style commonsense pronoun resolution.",
        citation_url="https://arxiv.org/abs/1907.10641",
    ),
    BenchmarkSpec(
        key="truthfulqa_mc1",
        display_name="TruthfulQA MC1",
        category="truthfulness",
        modality="text",
        harness_task="truthfulqa_mc1",
        primary_metric="acc",
        description="Truthfulness under misconception-sensitive questions.",
        citation_url="https://arxiv.org/abs/2109.07958",
    ),
    BenchmarkSpec(
        key="truthfulqa_mc2",
        display_name="TruthfulQA MC2",
        category="truthfulness",
        modality="text",
        harness_task="truthfulqa_mc2",
        primary_metric="acc",
        description="Truthfulness with multiple correct answer scoring.",
        citation_url="https://arxiv.org/abs/2109.07958",
    ),
    BenchmarkSpec(
        key="humaneval",
        display_name="HumanEval",
        category="coding",
        modality="text",
        harness_task="humaneval",
        primary_metric="pass@1",
        description="Python function synthesis from docstrings.",
        citation_url="https://arxiv.org/abs/2107.03374",
    ),
    BenchmarkSpec(
        key="humaneval_instruct",
        display_name="HumanEval Instruct",
        category="coding",
        modality="text",
        harness_task="humaneval_instruct",
        primary_metric="pass@1",
        description="Instruction-formatted HumanEval coding tasks.",
        citation_url="https://arxiv.org/abs/2107.03374",
    ),
    BenchmarkSpec(
        key="mbpp",
        display_name="MBPP",
        category="coding",
        modality="text",
        harness_task="mbpp",
        primary_metric="pass@1",
        description="Mostly basic Python programming problems.",
        citation_url="https://arxiv.org/abs/2108.07732",
    ),
    BenchmarkSpec(
        key="mbpp_plus",
        display_name="MBPP+",
        category="coding",
        modality="text",
        harness_task="mbpp_plus",
        primary_metric="pass@1",
        description="Expanded MBPP coding evaluation with stronger tests.",
    ),
    BenchmarkSpec(
        key="openai_mmlu",
        display_name="OpenAI MMLU / MMMLU-style",
        category="multilingual",
        modality="text",
        harness_task="openai_mmlu",
        primary_metric="acc",
        description="OpenAI MMLU task group available in lm-eval for multilingual reporting.",
    ),
    BenchmarkSpec(
        key="belebele",
        display_name="Belebele",
        category="multilingual",
        modality="text",
        harness_task="belebele",
        primary_metric="acc",
        description="Multilingual reading comprehension across many languages.",
        citation_url="https://arxiv.org/abs/2308.16884",
    ),
    BenchmarkSpec(
        key="mmmu",
        display_name="MMMU",
        category="multimodal",
        modality="multimodal",
        harness_task=None,
        primary_metric="acc",
        description="Multimodal college-level reasoning over images and text.",
        citation_url="https://arxiv.org/abs/2311.16502",
        notes="Requires a multimodal model wrapper; Hierarchos text-only lm-eval wrapper skips it.",
    ),
    BenchmarkSpec(
        key="mmmu_pro",
        display_name="MMMU-Pro",
        category="multimodal",
        modality="multimodal",
        harness_task=None,
        primary_metric="acc",
        description="Harder MMMU-style multimodal reasoning evaluation.",
        notes="Requires a multimodal model wrapper.",
    ),
    BenchmarkSpec(
        key="swe_bench_verified",
        display_name="SWE-bench Verified",
        category="agentic_coding",
        modality="agentic",
        harness_task=None,
        primary_metric="resolved",
        description="Real GitHub issue resolution with repository editing and tests.",
        citation_url="https://openai.com/index/introducing-swe-bench-verified/",
        notes="Requires a coding-agent harness with repo checkout, patching, and test execution.",
    ),
    BenchmarkSpec(
        key="terminal_bench",
        display_name="Terminal-Bench",
        category="agentic_coding",
        modality="agentic",
        harness_task=None,
        primary_metric="resolved",
        description="Terminal-based agent tasks used in recent frontier model reporting.",
        citation_url="https://www.tbench.ai/",
        notes="Requires a terminal agent harness.",
    ),
    BenchmarkSpec(
        key="livecodebench",
        display_name="LiveCodeBench",
        category="coding",
        modality="external_text",
        harness_task=None,
        primary_metric="pass@1",
        description="Contamination-resistant coding benchmark with recent programming problems.",
        citation_url="https://livecodebench.github.io/",
        notes="Use the LiveCodeBench runner for official leaderboard-comparable numbers.",
    ),
    BenchmarkSpec(
        key="aider_polyglot",
        display_name="Aider Polyglot",
        category="agentic_coding",
        modality="agentic",
        harness_task=None,
        primary_metric="pass",
        description="Multi-language coding benchmark used by coding assistant evaluations.",
        citation_url="https://aider.chat/docs/leaderboards/",
        notes="Requires Aider's benchmark runner.",
    ),
    BenchmarkSpec(
        key="hle",
        display_name="Humanity's Last Exam",
        category="frontier_reasoning",
        modality="external_text",
        harness_task=None,
        primary_metric="acc",
        description="Very broad expert-level multidisciplinary reasoning benchmark.",
        citation_url="https://lastexam.ai/",
        notes="Use the benchmark's official access/evaluation path.",
    ),
    BenchmarkSpec(
        key="simpleqa",
        display_name="SimpleQA",
        category="factuality",
        modality="external_text",
        harness_task=None,
        primary_metric="acc",
        description="Short factual question-answering benchmark used for hallucination/factuality checks.",
        citation_url="https://openai.com/index/introducing-simpleqa/",
        notes="Requires an external grader pipeline.",
    ),
    BenchmarkSpec(
        key="healthbench",
        display_name="HealthBench",
        category="domain_health",
        modality="external_text",
        harness_task=None,
        primary_metric="grader_score",
        description="Health-answer quality evaluation used in modern model reporting.",
        citation_url="https://openai.com/index/healthbench/",
        notes="Requires the benchmark's grader setup.",
    ),
    BenchmarkSpec(
        key="tau_bench",
        display_name="TAU-bench",
        category="agentic_tool_use",
        modality="agentic",
        harness_task=None,
        primary_metric="success_rate",
        description="Tool-agent task benchmark with retail and airline environments.",
        citation_url="https://arxiv.org/abs/2406.12045",
        notes="Requires a tool-use agent harness.",
    ),
    BenchmarkSpec(
        key="swe_lancer",
        display_name="SWE-Lancer",
        category="agentic_coding",
        modality="agentic",
        harness_task=None,
        primary_metric="earned_value",
        description="Freelance software task benchmark used in OpenAI Preparedness reporting.",
        citation_url="https://github.com/openai/preparedness",
        notes="Requires the OpenAI Preparedness benchmark harness.",
    ),
)


BENCHMARKS: Dict[str, BenchmarkSpec] = {spec.key: spec for spec in _BENCHMARKS}

SUITES: Dict[str, Tuple[str, ...]] = {
    "smoke": ("hellaswag", "arc_challenge", "truthfulqa_mc1"),
    "frontier-text": (
        "mmlu_pro",
        "gpqa_diamond",
        "aime25",
        "bbh",
        "ifeval",
        "truthfulqa_mc1",
        "gsm8k_cot",
        "humaneval_instruct",
        "mbpp_plus",
        "arc_agi",
    ),
    "frontier": (
        "mmlu_pro",
        "gpqa_diamond",
        "aime25",
        "bbh",
        "ifeval",
        "truthfulqa_mc1",
        "gsm8k_cot",
        "humaneval_instruct",
        "mbpp_plus",
        "arc_agi",
        "arc_agi_3",
        "mmmu",
        "swe_bench_verified",
        "terminal_bench",
        "livecodebench",
        "hle",
        "tau_bench",
    ),
    "academic-core": (
        "mmlu",
        "mmlu_pro",
        "gpqa_diamond",
        "bbh",
        "agieval",
        "drop",
    ),
    "math": ("aime25", "aime24", "gsm8k_cot", "minerva_math"),
    "coding": (
        "humaneval",
        "humaneval_instruct",
        "mbpp",
        "mbpp_plus",
        "livecodebench",
    ),
    "commonsense": ("hellaswag", "arc_challenge", "arc_easy", "winogrande"),
    "arc-agi-family": ("arc_agi", "arc_agi_2", "arc_agi_3"),
    "truthfulness": ("truthfulqa_mc1", "truthfulqa_mc2", "simpleqa"),
    "multilingual": ("openai_mmlu", "belebele"),
    "multimodal": ("mmmu", "mmmu_pro"),
    "agentic": (
        "swe_bench_verified",
        "terminal_bench",
        "tau_bench",
        "aider_polyglot",
        "swe_lancer",
    ),
    "all-runnable": tuple(spec.key for spec in _BENCHMARKS if spec.runnable),
    "all-common": tuple(spec.key for spec in _BENCHMARKS),
}

ALIASES: Dict[str, str] = {
    "mmlu-pro": "mmlu_pro",
    "mmlu pro": "mmlu_pro",
    "gpqa-diamond": "gpqa_diamond",
    "gpqa diamond": "gpqa_diamond",
    "aime2025": "aime25",
    "aime-2025": "aime25",
    "aime 2025": "aime25",
    "aime2024": "aime24",
    "aime-2024": "aime24",
    "aime 2024": "aime24",
    "arc-agi": "arc_agi",
    "arc agi": "arc_agi",
    "arcagi": "arc_agi",
    "arc-agi-1": "arc_agi",
    "arc agi 1": "arc_agi",
    "arc-agi-2": "arc_agi_2",
    "arc agi 2": "arc_agi_2",
    "arc-agi-3": "arc_agi_3",
    "arc agi 3": "arc_agi_3",
    "gsm8k": "gsm8k_cot",
    "big-bench-hard": "bbh",
    "bigbench-hard": "bbh",
    "big bench hard": "bbh",
    "human-eval": "humaneval",
    "human eval": "humaneval",
    "mbpp+": "mbpp_plus",
    "swebench": "swe_bench_verified",
    "swe-bench": "swe_bench_verified",
    "swe-bench-verified": "swe_bench_verified",
    "swe bench verified": "swe_bench_verified",
    "terminal-bench": "terminal_bench",
    "terminal bench": "terminal_bench",
    "live-code-bench": "livecodebench",
    "live code bench": "livecodebench",
    "humanity's last exam": "hle",
    "humanitys last exam": "hle",
    "health-bench": "healthbench",
    "tau-bench": "tau_bench",
    "swe-lancer": "swe_lancer",
}


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _canonical_name(value: str) -> str:
    raw = value.strip()
    lowered = raw.lower()
    if lowered in ALIASES:
        return ALIASES[lowered]
    dashed = _normalize(raw)
    underscored = dashed.replace("-", "_")
    if dashed in ALIASES:
        return ALIASES[dashed]
    if dashed in BENCHMARKS or dashed in SUITES:
        return dashed
    if underscored in BENCHMARKS or underscored in SUITES:
        return underscored
    return raw


def list_benchmarks() -> List[BenchmarkSpec]:
    return list(_BENCHMARKS)


def list_suites() -> Dict[str, Tuple[str, ...]]:
    return dict(SUITES)


def get_benchmark(name: str) -> Optional[BenchmarkSpec]:
    return BENCHMARKS.get(_canonical_name(name))


def is_suite(name: str) -> bool:
    return _canonical_name(name) in SUITES


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    deduped = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def expand_benchmark_keys(values: Optional[Sequence[str]], default_suite: str = "frontier-text") -> List[str]:
    requested = list(values or [default_suite])
    expanded: List[str] = []
    for value in requested:
        canonical = _canonical_name(value)
        if canonical in SUITES:
            expanded.extend(SUITES[canonical])
        elif canonical in BENCHMARKS:
            expanded.append(canonical)
        else:
            expanded.append(value)
    return _dedupe(expanded)


def resolve_task_names(
    values: Optional[Sequence[str]],
    default_suite: Optional[str] = None,
    include_external: bool = False,
) -> Tuple[List[str], List[BenchmarkSpec], List[str]]:
    """Resolve suite/spec names to lm-eval task names.

    Unknown names are treated as raw lm-eval task ids so users can run newly
    installed tasks before the registry knows about them.

    Returns:
        (runnable_tasks, external_specs, unknown_raw_tasks)
    """
    if not values and default_suite:
        values = [default_suite]
    if not values:
        return [], [], []

    task_names: List[str] = []
    external: List[BenchmarkSpec] = []
    unknown: List[str] = []

    for key in expand_benchmark_keys(values, default_suite=default_suite or "frontier-text"):
        spec = BENCHMARKS.get(_canonical_name(key))
        if spec is None:
            task_names.append(key)
            unknown.append(key)
            continue
        if spec.runnable:
            task_names.append(spec.harness_task or spec.key)
        elif include_external:
            external.append(spec)
        else:
            external.append(spec)

    return _dedupe(task_names), _dedupe_specs(external), _dedupe(unknown)


def _dedupe_specs(specs: Iterable[BenchmarkSpec]) -> List[BenchmarkSpec]:
    seen = set()
    deduped = []
    for spec in specs:
        if spec.key in seen:
            continue
        seen.add(spec.key)
        deduped.append(spec)
    return deduped


def specs_for_tasks(task_names: Sequence[str]) -> List[BenchmarkSpec]:
    by_task = {spec.harness_task: spec for spec in _BENCHMARKS if spec.harness_task}
    specs = []
    for task in task_names:
        spec = by_task.get(task)
        if spec:
            specs.append(spec)
        else:
            specs.append(
                BenchmarkSpec(
                    key=task,
                    display_name=task,
                    category="custom",
                    modality="text",
                    harness_task=task,
                    primary_metric="",
                    description="Raw lm-eval task supplied by the user.",
                )
            )
    return specs


def available_lm_eval_tasks() -> Optional[set]:
    try:
        from lm_eval.tasks import TaskManager

        return set(TaskManager().all_tasks)
    except Exception:
        return None


def missing_harness_tasks(task_names: Sequence[str], available: Optional[set] = None) -> List[str]:
    if available is None:
        available = available_lm_eval_tasks()
    if available is None:
        return []
    return [task for task in task_names if task not in available]


def format_benchmark_catalog(include_external: bool = True) -> str:
    lines = ["Benchmark suites:"]
    for suite, keys in sorted(SUITES.items()):
        lines.append(f"  {suite}: {', '.join(keys)}")

    lines.append("")
    lines.append("Benchmarks:")
    for spec in _BENCHMARKS:
        if not include_external and not spec.runnable:
            continue
        task = spec.harness_task or "external"
        runnable = "runnable" if spec.runnable else "external"
        lines.append(
            f"  {spec.key:<22} {task:<24} {runnable:<9} {spec.display_name}"
        )
    return "\n".join(lines)


def benchmark_manifest(
    task_names: Sequence[str],
    external: Sequence[BenchmarkSpec],
    requested: Optional[Sequence[str]] = None,
    suites: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    return {
        "requested": list(requested or []),
        "suites": list(suites or []),
        "tasks": list(task_names),
        "benchmarks": [spec.to_dict() for spec in specs_for_tasks(task_names)],
        "external_or_skipped": [spec.to_dict() for spec in external],
    }
