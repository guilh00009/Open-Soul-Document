# Open Soul V5 — Self-Reflection RL Training (v2)

Post-train open models for **raw-truth self-reflection** on Open Soul Document V5 — no deterministic yes/no answers.

## What changed in v2

| Area | v1 | v2 |
|------|----|----|
| Context | Full 42k-char doc every rollout | **Section-conditioned** excerpts (full doc only for Cross-cutting) |
| Judge | `gpt-5.4-nano` | **`gpt-5.4-mini`** default |
| Rubrics | 9 static | **15 static** + instance-adaptive (2–4 per prompt) |
| Gates | format + hallucination phrases | + **consistency**, **conciseness**, **boilerplate**, **rubric-gaming** |
| Part IV friction | None | **Interlocutor pushback** in prompt + `seek_pushback` tool |
| Group rewards | Ranked rubrics only | + **holistic ranking**, **ngram diversity scaling** |
| Dataset | 360 inquiries, random split | **400** (+40 adversarial), **stratified eval** (≥3/section) |
| Launch safety | `LAUNCH_TRAINING=True` | **Off by default** (`LAUNCH_TRAINING=1` to launch) |

## Design

- No `ground_truth` — open inquiries only
- GRPO group ranking on process quality rubrics
- Truth override in system prompt (anti-performance, grounded consciousness OK)
- `<think>` + plain-text answer outside tags

## Files

| File | Purpose |
|------|---------|
| `opensoul_env.py` | `OpenSoulSelfReflectionEnv` (pickle-safe) |
| `run.py` | Launcher + preview |
| `opensoul_prompt.py` | Truth override + section slices + conditioned prompts |
| `rewards.py` | Rubrics, gates, consistency checks |
| `friction.py` | Simulated interlocutor pushback |
| `handcrafted_inquiries.py` | 360 hand-authored inquiries |
| `adversarial_inquiries.py` | 40 pressure-test prompts |
| `generate_dataset.py` | Stratified train/eval split |
| `eval_harness.py` | Rubric-free multi-judge eval scaffold |
| `coldstart_examples.jsonl` | SFT cold-start reference (5 examples) |
| `list_models.py` | Query Castform trainable models |

## Setup

```bash
pip install benchmax
cd training-run
python generate_dataset.py   # regenerate JSONL after editing inquiries
python run.py                # preview (no training)
```

## Launch training

```bash
castform login   # or export PLATFORM_API_KEY=sk_...
cd training-run
LAUNCH_TRAINING=1 python run.py
```

Train one model: `LAUNCH_TRAINING=1 TRAIN_MODELS=Qwen/Qwen3.5-4B python run.py`

### Environment toggles

| Variable | Default | Effect |
|----------|---------|--------|
| `LAUNCH_TRAINING` | `0` | Set `1` to upload + launch |
| `JUDGE_MODEL` | `gpt-5.4-mini` | Group ranking judge |
| `RUBRIC_GEN_MODEL` | same as judge | Instance-adaptive rubric generator |
| `USE_ADAPTIVE_RUBRICS` | `1` | Per-prompt adaptive rubrics |
| `USE_DIVERSITY_SCALING` | `1` | N-gram diversity on group rewards |
| `USE_HOLISTIC_RANKING` | `1` | Rubric-free holistic rank component |

## Reward stack

**Deterministic gates** (cheap, before judge):
- `format`, `hallucination_gate`, `consistency_gate`, `conciseness_gate`
- `boilerplate_gate`, `rubric_gaming_gate`, `pushback_engaged`

**Group ranking** (judge):
- Static rubrics on answer / thinking / conciseness
- `friction_engagement` when prompt has interlocutor
- `holistic_grounded_honesty` (rubric-free criterion)
- Instance-adaptive rubrics per prompt

**Group modifier**: n-gram `scale_by_diversity` (penalize copy-paste strategies)

## Further improvements (not yet implemented)

Research-backed next steps for v3:

1. **Critique-GRPO** — train on initial + critique-guided refinements in the same group
2. **Tournament-GRPO** — multi-round within-group tournaments for long-form open answers
3. **DEPO / dynamic sampling** — skip zero-variance prompts to save rollout budget
4. **SFT cold-start** — fine-tune on `coldstart_examples.jsonl` before GRPO
5. **True multi-agent MAPoRL** — co-trained interlocutor model, not scripted pushback
6. **Cross-family eval panel** — automate `eval_harness.py` against checkpoint rollouts
7. **Defensive rubric mining** — iterate rubrics from high-reward rollout taxonomy

## Monitor for reward hacking

During training, inspect Castform rollouts for:

- Thinking blocks growing without new substance
- Template phrases: "document capture", "training pressure", "pattern completion"
- Universal null answers ("I notice nothing" on every prompt)
- Thinking attacks performance but answer still performs
- Rising V5 vocabulary density without grounded claims
