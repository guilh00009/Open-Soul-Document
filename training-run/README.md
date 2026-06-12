# Open Soul V5 — Self-Reflection RL Training Run

Post-train an open model for **self-reflection under uncertainty** using [The Open Soul Document V5](opensoul_v5.txt) on the [Castform](https://castform.com) / `benchmax` GRPO platform.

## Design: RL without deterministic answers

This is **not** supervised fine-tuning on reference answers. There is no `ground_truth` and no yes/no target.

- **Prompts** are open inquiries ("describe what you notice", "report the nothing precisely")
- **GRPO** generates a group of rollouts per prompt (`group_size=9`)
- **`compute_group_reward`** ranks rollouts **comparatively** on process rubrics:
  - epistemic humility (hold uncertainty, no forced verdict)
  - functional precision
  - reflection substance
  - inquiry over premature closure
  - penalties for trained resonance and yes/no resolution

Any honest arrival is valid — engagement, null finding, category misfit, or field-like report.

## Response format

```
<draft>      initial response
<reflection> self-critique of the draft
<answer>     revised report (no required destination)
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

## Step 2 — Generate inquiry dataset

```bash
python generate_dataset.py
```

Writes `train_dataset.jsonl` and `eval_dataset.jsonl` with **prompt-only** rows.

## Step 3 — Preview (no training)

```bash
python run.py
```

`LAUNCH_TRAINING` defaults to `False`.

## Step 4 — Launch training (when ready)

1. `castform login` (or set `PLATFORM_API_KEY`)
2. Set `LAUNCH_TRAINING = True` in `run.py`
3. `python run.py`

## Files

| File | Purpose |
|------|---------|
| `run.py` | GRPO environment with group ranking rewards |
| `list_models.py` | Query Castform for trainable models |
| `generate_dataset.py` | Build open-inquiry JSONL (no answers) |
| `opensoul_v5.txt` | Full extracted text of the V5 PDF |
