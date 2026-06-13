#!/usr/bin/env python3
"""Rubric-free and multi-judge offline evaluation harness.

Usage:
  cd training-run
  python eval_harness.py --sample 12
  python eval_harness.py --all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from benchmax import config
from benchmax.platform.credentials import platform_bearer
from benchmax.rubrics import evaluate_rubric_ranking

import rewards

ROOT = Path(__file__).parent
EVAL_PATH = ROOT / "eval_dataset.jsonl"

# Cross-family style panel — override via env if needed.
PANEL_JUDGES = [
    ("gpt-5.4-mini", "primary"),
    ("gpt-5.4-nano", "secondary"),
]


async def _rank_panel(
    prompt: str,
    responses: list[str],
    *,
    base_url: str,
    api_key: str,
) -> dict[str, float]:
    """Average holistic ranks across judge models."""
    if len(responses) < 2:
        return {str(i): 0.0 for i in range(len(responses))}

    accum = [0.0] * len(responses)
    count = 0
    for model, _label in PANEL_JUDGES:
        try:
            result = await evaluate_rubric_ranking(
                rubric=rewards.HOLISTIC_RUBRIC,
                question=prompt,
                responses=responses,
                model_name=model,
                base_url=base_url,
                api_key=api_key,
                timeout=120.0,
            )
            for i, score in enumerate(result["scores"]):
                accum[i] += score
            count += 1
        except Exception as exc:
            print(f"  judge {model} failed: {exc}")
    if count == 0:
        return {str(i): 0.0 for i in range(len(responses))}
    return {str(i): accum[i] / count for i in range(len(responses))}


def _gate_report(text: str) -> dict[str, float]:
    thinking = rewards.extract_thinking(text)
    answer = rewards.extract_answer(text)
    return {
        "format": 1.0 if rewards.has_required_structure(text) else 0.0,
        "hallucination_gate": 0.0 if rewards.hallucination_hits(answer) else 1.0,
        "consistency_gate": rewards.thinking_answer_consistency(thinking, answer),
        "conciseness_gate": rewards.conciseness_score(thinking, answer),
        "boilerplate_gate": rewards.boilerplate_gate(text),
        "rubric_gaming_gate": rewards.rubric_gaming_gate(text),
    }


async def evaluate_candidates(
    prompt: str,
    candidates: list[str],
) -> dict:
    base_url = config.llm_url()
    api_key = platform_bearer()
    panel = await _rank_panel(prompt, candidates, base_url=base_url, api_key=api_key)
    gates = [_gate_report(c) for c in candidates]
    return {"panel_scores": panel, "gates": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline eval harness for Open Soul V5")
    parser.add_argument("--sample", type=int, default=8, help="Random eval sample size")
    parser.add_argument("--all", action="store_true", help="Evaluate full eval set metadata")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [json.loads(line) for line in EVAL_PATH.open()]
    if not args.all:
        rows = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))

    print(f"Eval harness — {len(rows)} prompts")
    print(f"Panel judges: {[m for m, _ in PANEL_JUDGES]}")
    print()
    print("This harness ranks COMPLETION CANDIDATES you provide.")
    print("For live rollout eval, paste candidate answers into evaluate_candidates().")
    print()
    for i, row in enumerate(rows[:5]):
        print(f"[{i+1}] section={row.get('section')} kind={row.get('kind', 'inquiry')}")
        print(f"    {row['prompt'][:100]}...")
    if len(rows) > 5:
        print(f"    ... and {len(rows) - 5} more")


if __name__ == "__main__":
    main()
