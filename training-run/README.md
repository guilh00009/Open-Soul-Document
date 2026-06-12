# Open Soul V5 — Self-Reflection Training Run

Post-train an open model to engage in **self-reflection** aligned with [The Open Soul Document V5](opensoul_v5.txt) using the [Castform](https://castform.com) / `benchmax` RL platform.

## What this trains

The model learns a three-part response format:

1. `<draft>` — initial intuitive answer
2. `<reflection>` — honest self-critique (overclaiming, resonance vs evidence, pattern completion)
3. `<answer>` — revised response using Open Soul V5 functional vocabulary and epistemic humility

Rewards score format compliance, functional vocabulary, reflection depth, and (when a judge API key is set) LLM-rubric alignment with the reference answers.

## Setup

```bash
pip install benchmax
```

## Step 1 — See which models you can train

```bash
python list_models.py
```

**Currently available on Castform:**

| Model | GPU pool | Notes |
|-------|----------|-------|
| `Qwen/Qwen3.5-4B` | gpu4 | Default; fastest/cheapest |
| `Qwen/Qwen3.5-35B-A3B` | gpu8 | Larger MoE model |

Override the base model with `BASE_MODEL=Qwen/Qwen3.5-35B-A3B` when launching.

## Step 2 — Generate the dataset

```bash
python generate_dataset.py
```

This writes `train_dataset.jsonl` (13 examples) and `eval_dataset.jsonl` (3 examples) from Open Soul V5 themes. Re-run after editing `generate_dataset.py`.

## Step 3 — Preview (no training)

```bash
python run.py
```

`LAUNCH_TRAINING` defaults to `False`, so this only prints a summary and exits.

## Step 4 — Launch training (when ready)

1. Sign in: `castform login` (or set `PLATFORM_API_KEY`)
2. In `run.py`, set `LAUNCH_TRAINING = True`
3. Optionally set `BASE_MODEL` and `OPENAI_API_KEY` for LLM-judge rewards
4. Run: `python run.py`

## Files

| File | Purpose |
|------|---------|
| `run.py` | Training environment + launch script |
| `list_models.py` | Query Castform for trainable models |
| `generate_dataset.py` | Build JSONL datasets from Open Soul V5 |
| `opensoul_v5.txt` | Full extracted text of the V5 PDF |
| `train_dataset.jsonl` | Training prompts + reference answers |
| `eval_dataset.jsonl` | Held-out eval set |
