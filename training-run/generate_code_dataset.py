from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from code_tasks import CODE_TASKS

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "code_train_dataset.jsonl"
EVAL_PATH = ROOT / "code_eval_dataset.jsonl"
SEED = 42
MIN_EVAL_PER_CATEGORY = 2


def _row(task: dict) -> dict:
    return {
        k: v
        for k, v in task.items()
        if k in (
            "task_id",
            "prompt",
            "category",
            "difficulty",
            "entry_point",
            "test",
            "starter_code",
            "language",
            "reward_mode",
        )
        and v is not None
        and v != ""
    }


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[row.get("category", "general")].append(row)

    rng = random.Random(SEED)
    train, eval_rows = [], []
    for _cat, items in sorted(by_cat.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_eval = max(MIN_EVAL_PER_CATEGORY, len(shuffled) // 4)
        n_eval = min(n_eval, len(shuffled) - 1) if len(shuffled) > 1 else 1
        eval_rows.extend(shuffled[:n_eval])
        train.extend(shuffled[n_eval:])
    rng.shuffle(train)
    rng.shuffle(eval_rows)
    return train, eval_rows


def main() -> None:
    rows = [_row(t) for t in CODE_TASKS]
    train, eval_rows = stratified_split(rows)
    for path, data in ((TRAIN_PATH, train), (EVAL_PATH, eval_rows)):
        with path.open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"total={len(rows)} train={len(train)} eval={len(eval_rows)}")
    print("categories:", dict(Counter(r["category"] for r in rows)))
    print("difficulty:", dict(Counter(r.get("difficulty", "?") for r in rows)))


if __name__ == "__main__":
    main()
