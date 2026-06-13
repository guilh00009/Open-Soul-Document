#!/usr/bin/env python3
"""Download Fable-5-traces and build train/eval JSONL splits."""

from __future__ import annotations

import json
from pathlib import Path
import re

from datasets import load_dataset

from benchmax.traces.processing import apply_filters, split_dataset

from fable_adapter import rows_to_training_examples

DATASET_ID = "Glint-Research/Fable-5-traces"
OUTPUT_DIR = Path(__file__).parent
MAX_PROMPT_CHARS = 12000
SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),  # Groq
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style
    re.compile(r"cf_[A-Za-z0-9]{40,}"),  # Castform
]


def _redact_secrets(text: str) -> str:
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_SECRET]", text)
    return text


def _redact_row(row: dict) -> dict:
    out = dict(row)
    if "prompt_messages" in out:
        out["prompt_messages"] = [
            {**m, "content": _redact_secrets(str(m.get("content", "")))}
            for m in out["prompt_messages"]
        ]
    if "ground_truth" in out and isinstance(out["ground_truth"], dict):
        gt = dict(out["ground_truth"])
        if "content" in gt:
            gt["content"] = _redact_secrets(str(gt.get("content", "")))
        out["ground_truth"] = gt
    if "init_rollout_args" in out and isinstance(out["init_rollout_args"], dict):
        args = dict(out["init_rollout_args"])
        if "raw_prompt" in args:
            args["raw_prompt"] = _redact_secrets(str(args["raw_prompt"]))
        out["init_rollout_args"] = args
    return out


def _cap_prompt(example_row: dict) -> dict:
    """Truncate very long prompt histories to keep rollouts within token limits."""
    msgs = example_row.get("prompt_messages", [])
    total = sum(len(str(m.get("content", ""))) for m in msgs)
    if total <= MAX_PROMPT_CHARS:
        return example_row

    # Keep the last user message and as much recent context as fits.
    kept: list[dict] = []
    budget = MAX_PROMPT_CHARS
    for msg in reversed(msgs):
        content = str(msg.get("content", ""))
        if len(content) > budget and kept:
            break
        if len(content) > budget:
            content = "…[truncated]…\n" + content[-budget:]
        content = _redact_secrets(content)
        kept.insert(0, {**msg, "content": content})
        budget -= len(content)
        if budget <= 0:
            break
    return {**example_row, "prompt_messages": kept}


def main() -> tuple[list[dict], list[dict], dict]:
    print(f"Loading {DATASET_ID}...")
    ds = load_dataset(DATASET_ID, split="train")
    rows = [dict(row) for row in ds]
    print(f"Loaded {len(rows)} rows from {len(set(r['session'] for r in rows))} sessions")

    examples = rows_to_training_examples(rows)
    print(f"Built {len(examples)} per-turn training examples")

    filter_result = apply_filters(
        examples,
        steps=[
            ("heuristic", {"min_completion_chars": 20}),
            ("tool_relay", {"overlap_threshold": 0.85}),
            ("dedup", {"similarity_threshold": 0.9}),
        ],
    )
    kept = filter_result.kept
    print(f"Filters kept {len(kept)}/{len(examples)} examples")

    train_count = max(int(len(kept) * 0.9), 16)
    eval_count = max(len(kept) - train_count, 1)
    train_data, eval_data = split_dataset(kept, train_count, eval_count, seed=42)
    train_data = [_cap_prompt(_redact_row(r)) for r in train_data]
    eval_data = [_cap_prompt(_redact_row(r)) for r in eval_data]

    stats = {
        "examples_built": len(examples),
        "kept_after_filters": len(kept),
        "train_count": len(train_data),
        "eval_count": len(eval_data),
        "max_prompt_chars": MAX_PROMPT_CHARS,
    }

    train_path = OUTPUT_DIR / "train_dataset.jsonl"
    eval_path = OUTPUT_DIR / "eval_dataset.jsonl"
    meta_path = OUTPUT_DIR / "dataset_metadata.json"

    with train_path.open("w") as f:
        for row in train_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with eval_path.open("w") as f:
        for row in eval_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with meta_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"Wrote {len(train_data)} train + {len(eval_data)} eval examples")
    print(f"  train: {train_path}")
    print(f"  eval:  {eval_path}")
    return train_data, eval_data, stats


if __name__ == "__main__":
    main()
