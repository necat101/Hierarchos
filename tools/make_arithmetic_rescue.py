"""Build a deterministic arithmetic rescue set for response-boundary repair."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


ADD_TEMPLATES = [
    "what is {a} + {b}",
    "What is {a} + {b}?",
    "Calculate {a} + {b}.",
    "Answer directly: {a} plus {b}.",
    "Do not explain: {a} + {b}",
]

SUB_TEMPLATES = [
    "what is {a} - {b}",
    "What is {a} minus {b}?",
    "Calculate {a} - {b}.",
    "Answer directly: {a} minus {b}.",
    "Do not explain: {a} - {b}",
]

MUL_TEMPLATES = [
    "what is {a} * {b}",
    "What is {a} times {b}?",
    "Calculate {a} * {b}.",
    "Answer directly: {a} times {b}.",
    "Do not explain: {a} * {b}",
]


def row(instruction: str, answer: int) -> dict:
    # Start with the answer token. This directly trains the first logit after
    # `### Response:` instead of teaching a verbose arithmetic explanation first.
    return {"instruction": instruction, "input": "", "output": f"{answer}."}


def build_focus_rows(repeats: int, family: str = "all") -> list[dict]:
    focus_4_plus_4 = [
        ("what is 4 + 4", 8),
        ("What is 4 + 4?", 8),
        ("4 + 4", 8),
        ("Answer directly: 4 + 4", 8),
        ("Do not explain: 4 + 4", 8),
    ]
    focus_8_plus_8 = [
        ("what is 8 + 8", 16),
        ("What is 8 + 8?", 16),
        ("8 + 8", 16),
        ("Answer directly: 8 + 8", 16),
        ("Do not explain: 8 + 8", 16),
    ]
    if family == "4plus4":
        focus = focus_4_plus_4
    elif family == "8plus8":
        focus = focus_8_plus_8
    else:
        focus = focus_4_plus_4 + focus_8_plus_8
    rows: list[dict] = []
    for _ in range(max(0, int(repeats or 0))):
        rows.extend(row(prompt, answer) for prompt, answer in focus)
    return rows


def build_rows(seed: int, count: int, focus_repeats: int = 0, focus_family: str = "all") -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    rows.extend(build_focus_rows(focus_repeats, focus_family))

    must_have = [
        ("what is 4 + 4", 8),
        ("what is 8 + 8", 16),
        ("What is 12 + 7?", 19),
        ("Answer directly: 5 times 6.", 30),
        ("Answer directly: 10 minus 3.", 7),
    ]
    rows.extend(row(prompt, answer) for prompt, answer in must_have)

    candidates: list[dict] = []
    for a in range(0, 31):
        for b in range(0, 31):
            candidates.append(row(rng.choice(ADD_TEMPLATES).format(a=a, b=b), a + b))
            if a >= b:
                candidates.append(row(rng.choice(SUB_TEMPLATES).format(a=a, b=b), a - b))
    for a in range(0, 13):
        for b in range(0, 13):
            candidates.append(row(rng.choice(MUL_TEMPLATES).format(a=a, b=b), a * b))

    rng.shuffle(candidates)
    seen = set()
    if focus_repeats <= 0:
        seen = {(item["instruction"], item["output"]) for item in rows}
    for item in candidates:
        key = (item["instruction"], item["output"])
        if key in seen:
            continue
        rows.append(item)
        seen.add(key)
        if len(rows) >= count:
            break
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an arithmetic rescue JSONL file.")
    parser.add_argument("--out", default="tools/rescue_math_seed.jsonl")
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--focus-repeats", type=int, default=0, help="Repeat exact failed prompts before adding the broad arithmetic mix.")
    parser.add_argument("--focus-family", choices=("all", "4plus4", "8plus8"), default="all")
    parser.add_argument("--seed", type=int, default=448)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows(args.seed, args.count, args.focus_repeats, args.focus_family)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in rows:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
