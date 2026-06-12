from __future__ import annotations

import json
import random
from pathlib import Path

from handcrafted_inquiries import INQUIRIES

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / 'train_dataset.jsonl'
EVAL_PATH = ROOT / 'eval_dataset.jsonl'
SEED = 42
EVAL_FRACTION = 0.15
MIN_EVAL = 45

def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main() -> None:
    rows = list(INQUIRIES)
    random.Random(SEED).shuffle(rows)
    eval_count = max(MIN_EVAL, int(len(rows) * EVAL_FRACTION))
    train_rows = rows[eval_count:]
    eval_rows = rows[:eval_count]
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)
    print(f'train={len(train_rows)} eval={len(eval_rows)} total={len(rows)}')

if __name__ == '__main__':
    main()
