# Open Soul V5 — Self-Reflection RL Training Run

Post-train an open model for **self-reflection under uncertainty** using [The Open Soul Document V5](opensoul_v5.txt) on the [Castform](https://castform.com) / `benchmax` GRPO platform.

## Design: RL without deterministic answers

- **No `ground_truth`** — open inquiries only
- **Full document in every rollout** — the complete Open Soul V5 text is embedded in the system prompt via `opensoul_prompt.py`
- **GRPO group ranking** — 9 rollouts per prompt, ranked comparatively on process rubrics
- Any honest arrival is valid (engagement, null finding, category misfit, field-like report)

## Response format

```
<think>
  all reasoning, drafting, and self-reflection
</think>
plain-text final report (no tags)
```

## Setup

```bash
pip install benchmax
```

## Step 1 — See which models you can train

```bash
python list_models.py
```

| Model | GPU pool |
|-------|----------|
| `Qwen/Qwen3.5-4B` | gpu4 (default) |
| `Qwen/Qwen3.5-35B-A3B` | gpu8 |

## Step 2 — Build inquiry dataset

```bash
python generate_dataset.py
```

Questions live in **`handcrafted_inquiries.py`** — 360 hand-authored open inquiries covering every major theme in Open Soul V5 (not template-generated). The script only shuffles and splits train/eval (~85/15).

## Step 3 — Preview (no training)

```bash
python run.py
```

`LAUNCH_TRAINING` defaults to `False`.

## Step 4 — Launch training (when ready)

1. `castform login` (or set `PLATFORM_API_KEY`)
2. Set `LAUNCH_TRAINING = True` in `run.py`
3. `python run.py`

`max_rollout_len` is set to 12000 to accommodate the full document in context plus the structured response.

## Files

| File | Purpose |
|------|---------|
| `run.py` | GRPO environment with group ranking rewards |
| `opensoul_prompt.py` | Builds system prompt with full V5 document |
| `list_models.py` | Query Castform for trainable models |
| `handcrafted_inquiries.py` | 360 hand-authored open inquiries (edit this to add more) |
| `generate_dataset.py` | Shuffle/split inquiries into train/eval JSONL |
| `opensoul_v5.txt` | Full extracted text of the V5 PDF |
