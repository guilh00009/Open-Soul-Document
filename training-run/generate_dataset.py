from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from adversarial_inquiries import ADVERSARIAL_INQUIRIES
from handcrafted_inquiries import INQUIRIES

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train_dataset.jsonl"
EVAL_PATH = ROOT / "eval_dataset.jsonl"
SEED = 42
MIN_EVAL_PER_SECTION = 3
TARGET_EVAL_FRACTION = 0.15


def _normalize_row(row: dict) -> dict:
    out = {
        "prompt": row["prompt"],
        "section": row.get("section", ""),
    }
    if row.get("kind"):
        out["kind"] = row["kind"]
    rf = row.get("requires_friction")
    if rf is not None:
        out["requires_friction"] = str(rf).lower()
    return out


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_section[row.get("section", "") or "unknown"].append(row)

    rng = random.Random(SEED)
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for section, items in sorted(by_section.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_eval = max(MIN_EVAL_PER_SECTION, int(len(shuffled) * TARGET_EVAL_FRACTION))
        n_eval = min(n_eval, len(shuffled) - 1) if len(shuffled) > 1 else 1
        eval_rows.extend(shuffled[:n_eval])
        train_rows.extend(shuffled[n_eval:])

    rng.shuffle(eval_rows)
    rng.shuffle(train_rows)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    base = [_normalize_row(r) for r in INQUIRIES]
    adversarial = [_normalize_row(r) for r in ADVERSARIAL_INQUIRIES]
    rows = base + adversarial
    train_rows, eval_rows = stratified_split(rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    from collections import Counter

    print(f"total={len(rows)} train={len(train_rows)} eval={len(eval_rows)}")
    print(f"  handcrafted={len(base)} adversarial={len(adversarial)}")
    print(f"  eval sections: {dict(Counter(r['section'] for r in eval_rows))}")


if __name__ == "__main__":
    main()
