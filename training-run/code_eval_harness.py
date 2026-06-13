#!/usr/bin/env python3
"""Offline code benchmark harness — verifiable pass@1 without training.

Usage:
  cd training-run
  python generate_code_dataset.py
  python code_eval_harness.py                    # stub reference solutions
  python code_eval_harness.py --model API_MODEL  # if completions provided via file
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from code_rewards import score_hidden_tests
from code_sandbox import run_hidden_tests
from code_tasks import CODE_TASKS
from training_utils import extract_python_code

ROOT = Path(__file__).parent
EVAL_PATH = ROOT / "code_eval_dataset.jsonl"
RESULTS_PATH = ROOT / "code_eval_results.jsonl"

# Minimal reference solutions for sanity-checking the harness (not for training).
_REFERENCE: dict[str, str] = {
    "he_has_close_elements": """
def has_close_elements(numbers, threshold):
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
""",
    "he_below_zero": """
def below_zero(operations):
    bal = 0
    for op in operations:
        bal += op
        if bal < 0:
            return True
    return False
""",
    "he_palindrome": """
def is_palindrome(text):
    t = text.lower()
    return t == t[::-1]
""",
    "he_prime": """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
""",
    "mbpp_list_sum": """
def list_sum(nums):
    return sum(nums)
""",
}


def _load_eval_rows() -> list[dict]:
    if EVAL_PATH.exists():
        return [json.loads(line) for line in EVAL_PATH.open()]
    return [
        {k: v for k, v in t.items() if k != "test" or True}
        for t in CODE_TASKS
    ]


def evaluate_completion(task: dict, completion: str) -> dict:
    scored = score_hidden_tests(completion, task)
    return {
        "task_id": task.get("task_id"),
        "category": task.get("category"),
        **scored,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Code eval harness (RLVR)")
    parser.add_argument("--limit", type=int, default=0, help="Max tasks (0=all)")
    parser.add_argument(
        "--completions",
        type=str,
        default="",
        help="JSONL with {task_id, completion} per line",
    )
    parser.add_argument("--write-results", action="store_true")
    args = parser.parse_args()

    tasks_by_id = {t["task_id"]: t for t in CODE_TASKS}
    rows = _load_eval_rows()
    if args.limit:
        rows = rows[: args.limit]

    completions: dict[str, str] = {}
    if args.completions:
        for line in Path(args.completions).read_text().splitlines():
            if line.strip():
                obj = json.loads(line)
                completions[obj["task_id"]] = obj.get("completion", "")
    else:
        for tid, code in _REFERENCE.items():
            completions[tid] = f"```python\n{code}\n```"

    results = []
    passed = 0
    total = 0
    for row in rows:
        tid = row.get("task_id", "")
        task = tasks_by_id.get(tid, row)
        if tid not in completions and not args.completions:
            continue
        completion = completions.get(tid, "")
        if not completion:
            continue
        total += 1
        res = evaluate_completion(task, completion)
        results.append(res)
        if res.get("hidden_tests_pass", 0) >= 1.0:
            passed += 1
        status = "PASS" if res.get("hidden_tests_pass") else "FAIL"
        print(f"{status} {tid} ({task.get('category')})")

    if total:
        print(f"\nVerifiable pass@1: {passed}/{total} = {100*passed/total:.1f}%")
    else:
        print("No completions to evaluate. Use --completions or extend _REFERENCE.")

    if args.write_results and results:
        with RESULTS_PATH.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
