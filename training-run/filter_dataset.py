#!/usr/bin/env python3
"""DEPO / curriculum dataset filtering — tag rows by difficulty band.

Usage:
  python filter_dataset.py --dataset code_train_dataset.jsonl --pass-rates rates.json
  python filter_dataset.py --dataset code_train_dataset.jsonl --simulate-random

`rates.json` format: {"task_id": 0.35, ...} from a base-model pass@8 probe.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from training_utils import difficulty_band, filter_by_difficulty

ROOT = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter training rows by difficulty")
    parser.add_argument("--dataset", required=True, help="Input JSONL path")
    parser.add_argument("--output", default="", help="Output JSONL (default: <stem>_filtered.jsonl)")
    parser.add_argument("--pass-rates", default="", help="JSON file task_id -> pass_rate")
    parser.add_argument(
        "--simulate-random",
        action="store_true",
        help="Assign random pass rates (for pipeline testing only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    path = Path(args.dataset)
    if not path.is_absolute():
        path = ROOT / path
    rows = [json.loads(line) for line in path.open()]

    if args.simulate_random:
        rng = random.Random(args.seed)
        pass_rates = {
            str(r.get("task_id", i)): rng.random()
            for i, r in enumerate(rows)
        }
    elif args.pass_rates:
        pass_rates = json.loads(Path(args.pass_rates).read_text())
    else:
        print("Provide --pass-rates or --simulate-random", flush=True)
        raise SystemExit(1)

    filtered = filter_by_difficulty(rows, pass_rates=pass_rates)
    bands = Counter(difficulty_band(pass_rates.get(str(r.get("task_id")), 0.5)) for r in rows)
    print(f"Input: {len(rows)} bands={dict(bands)}")
    print(f"Output: {len(filtered)} (kept hard/sweet_spot/easy)")

    out = args.output or str(path.with_name(path.stem + "_filtered.jsonl"))
    with open(out, "w", encoding="utf-8") as f:
        for row in filtered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
