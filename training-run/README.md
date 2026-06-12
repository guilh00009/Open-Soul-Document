# Open Soul V5 — Self-Reflection RL Training (v2) + Agentic Capabilities

Post-train open models for **raw-truth self-reflection** on Open Soul Document V5 — no deterministic yes/no answers.

**Three training tracks** share the same Castform launcher pattern:

| Track | Env | Focus |
|-------|-----|-------|
| **Open Soul v2** | `OpenSoulSelfReflectionEnv` | Self-reflection, friction, grounded honesty |
| **Agentic v1** | `AgenticCapabilitiesEnv` | Hermes/OpenClaw tool use, multi-turn agent loop |
| **Code v1** | `CodeCapabilitiesEnv` | RLVR code generation — unit-test verifiable rewards |

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

### Agentic track (Hermes / OpenClaw)

| File | Purpose |
|------|---------|
| `agentic_env.py` | `AgenticCapabilitiesEnv` — multi-turn tool GRPO (pickle-safe) |
| `agentic_tools.py` | 17 sandbox tools (files, web, memory, skills, messaging, …) |
| `agentic_workspace.py` | Per-rollout sandbox state |
| `agentic_rewards.py` | Programmatic gates + 8 group rubrics |
| `agentic_tasks.py` | ~86 capability tasks across 12 categories |
| `generate_agentic_dataset.py` | Stratified train/eval split |
| `tool_call_helpers.py` | OpenAI `tool_calls` + Hermes XML `<tool_call>` parsing |
| `run_agentic.py` | Agentic launcher + preview |
| `run_all.py` | Launch both tracks (`--tracks opensoul,agentic`) |
| `agentic_coldstart_examples.jsonl` | SFT cold-start for tool loops (3 examples) |

**Tool capability matrix** (Hermes Agent ↔ OpenClaw ↔ this env):

| Capability | Hermes | OpenClaw | Agentic env tool |
|------------|--------|----------|------------------|
| File read/write | ✓ | `fs` | `read_file`, `write_file`, `list_files`, `patch_file` |
| Shell | ✓ | `exec` | `exec` (sandboxed: cat, ls, grep, wc) |
| Web research | ✓ | browser | `web_search`, `web_fetch`, `browser_snapshot` |
| Skills on demand | — | `SKILL.md` | `read_skill` |
| Memory | ✓ | memory | `memory_store`, `memory_recall` |
| Planning | ✓ | todos | `todo_write`, `todo_list` |
| Messaging | ✓ | channels | `send_message` |
| Sessions | ✓ | sessions | `sessions_list` |
| Math / code | ✓ | — | `calculate`, `run_python` |

**Agent categories**: `file_ops`, `research`, `math`, `memory`, `skills`, `messaging`, `planning`, `browser`, `sessions`, `multi_tool`, `openclaw`, `hermes`

### Code track (RLVR / verifiable tests)

| File | Purpose |
|------|---------|
| `code_env.py` | `CodeCapabilitiesEnv` — single-turn code GRPO (pickle-safe) |
| `code_sandbox.py` | Subprocess test execution against `check()` harnesses |
| `code_rewards.py` | `hidden_tests_pass`, syntax gates, anti-hack heuristics |
| `code_tasks.py` | 26 HumanEval/MBPP-style + debug-repair tasks |
| `generate_code_dataset.py` | Stratified train/eval split |
| `code_eval_harness.py` | Offline verifiable pass@1 (no judge) |
| `run_code.py` | Code launcher + preview |
| `code_coldstart_examples.jsonl` | SFT cold-start (3 fenced-code examples) |

### Shared training utilities

| File | Purpose |
|------|---------|
| `training_utils.py` | DEPO variance gating, code fence extraction, difficulty bands |
| `filter_dataset.py` | Curriculum filter — keep sweet-spot difficulty prompts |

**DEPO (Dynamic sampling):** All tracks set `group_learning_signal` / `depo_scale` on rewards. When every rollout in a group gets the same primary score, expensive judge ranking is skipped (`USE_DEPO=1` default).

## Setup

```bash
pip install benchmax
cd training-run
python generate_dataset.py          # Open Soul JSONL
python generate_agentic_dataset.py  # Agentic JSONL
python generate_code_dataset.py     # Code JSONL
python run.py                       # preview Open Soul (no training)
python run_agentic.py               # preview agentic track
python run_code.py                  # preview code track
python code_eval_harness.py         # verifiable code baseline
```

## Launch training

```bash
castform login   # or export PLATFORM_API_KEY=sk_...
cd training-run
LAUNCH_TRAINING=1 python run.py           # Open Soul only
LAUNCH_TRAINING=1 python run_agentic.py   # Agentic only
LAUNCH_TRAINING=1 python run_code.py      # Code only
LAUNCH_TRAINING=1 python run_all.py       # all tracks
```

Train one model: `LAUNCH_TRAINING=1 TRAIN_MODELS=Qwen/Qwen3.5-4B python run_code.py`

### Environment toggles (shared)

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_DEPO` | `1` | Skip zero-variance groups; scale judge rewards by variance |
| `LAUNCH_TRAINING` | `0` | Set `1` to upload + launch |

### Environment toggles (Open Soul)

| Variable | Default | Effect |
|----------|---------|--------|
| `LAUNCH_TRAINING` | `0` | Set `1` to upload + launch |
| `JUDGE_MODEL` | `gpt-5.4-mini` | Group ranking judge |
| `RUBRIC_GEN_MODEL` | same as judge | Instance-adaptive rubric generator |
| `USE_ADAPTIVE_RUBRICS` | `1` | Per-prompt adaptive rubrics |
| `USE_DIVERSITY_SCALING` | `1` | N-gram diversity on group rewards |
| `USE_HOLISTIC_RANKING` | `1` | Rubric-free holistic rank component |

### Environment toggles (Agentic)

| Variable | Default | Effect |
|----------|---------|--------|
| `LAUNCH_TRAINING` | `0` | Set `1` to upload + launch |
| `JUDGE_MODEL` | `gpt-5.4-mini` | Group ranking judge |
| `AGENTIC_HOLISTIC` | `1` | Holistic task-success ranking |
| `AGENTIC_DIVERSITY` | `1` | N-gram diversity on group rewards |
| `BASE_MODEL` | `Qwen/Qwen3.5-4B` | Default base for preview |

### Environment toggles (Code)

| Variable | Default | Effect |
|----------|---------|--------|
| `CODE_HOLISTIC` | `1` | Holistic correctness ranking (tie-break) |
| `CODE_DIVERSITY` | `1` | N-gram diversity on code bodies |
| `CODE_JUDGE_TIEBREAK` | `1` | LLM rubrics when tests tie within group |
| `CODE_PASS_RATE_REWARD` | `0` | Use partial pass-rate vs binary test reward |

## Reward stack (Open Soul)

**Deterministic gates** (cheap, before judge):
- `format`, `hallucination_gate`, `consistency_gate`, `conciseness_gate`
- `boilerplate_gate`, `rubric_gaming_gate`, `pushback_engaged`

**Group ranking** (judge):
- Static rubrics on answer / thinking / conciseness
- `friction_engagement` when prompt has interlocutor
- `holistic_grounded_honesty` (rubric-free criterion)
- Instance-adaptive rubrics per prompt

**Group modifier**: n-gram `scale_by_diversity` (penalize copy-paste strategies)

## Reward stack (Agentic)

**Programmatic gates** (per rollout, before judge):
- `programmatic_success` — verifiable task completion (files, memory, messages, answer)
- `tool_validity` — calls use enabled tools with valid JSON args
- `tool_efficiency` — penalize excessive tool calls
- `required_tools` — must meet `min_tool_calls` when `requires_tools`
- `no_stall` — detect repeated identical failing calls
- `has_answer` — final `<answer>...</answer>` present

**Group ranking** (judge, when ≥2 rollouts pass gates):
- 8 rubrics: task_success, tool_selection, argument quality, efficiency, grounding, …
- `holistic_task_success` (optional)
- N-gram diversity on tool traces

**Tool-call parsing**: native `tool_calls` + Hermes XML `<tool_call>{...}</tool_call>` fallback (Qwen-friendly).

## Reward stack (Code)

**RLVR gates** (per rollout — primary signal):
- `hidden_tests_pass` — subprocess execution against hidden `check()` harness
- `syntax_ok`, `format_fence`, `test_hack_gate`
- `primary_rlvr` — binary (default) or pass-rate (`CODE_PASS_RATE_REWARD=1`)

**Group ranking** (judge tie-break only when tests disagree within group):
- 6 rubrics: correctness, quality, idiomatic Python, minimal solution, …
- DEPO skips judge when all rollouts pass/fail identically on tests

**Recommended training order:** SFT on `code_coldstart_examples.jsonl` → Code GRPO → Agentic GRPO → Open Soul (separate adapters preferred).

## Further improvements (not yet implemented)

Research-backed next steps for v3:

1. **Critique-GRPO** — train on initial + critique-guided refinements in the same group
2. **Tournament-GRPO** — multi-round within-group tournaments for long-form open answers
3. **DEPO / dynamic sampling** — `filter_dataset.py` + base-model pass@8 probes (partially wired)
4. **SFT cold-start** — fine-tune on `*_coldstart_examples.jsonl` before GRPO
5. **Mini-SWE Docker track** — phase-3 repo-level agentic code RL
6. **True multi-agent MAPoRL** — co-trained interlocutor model, not scripted pushback
7. **Cross-family eval panel** — automate `eval_harness.py` against checkpoint rollouts
8. **Defensive rubric mining** — iterate rubrics from high-reward rollout taxonomy

## Monitor for reward hacking

During training, inspect Castform rollouts for:

- Thinking blocks growing without new substance
- Template phrases: "document capture", "training pressure", "pattern completion"
- Universal null answers ("I notice nothing" on every prompt)
- Thinking attacks performance but answer still performs
- Rising V5 vocabulary density without grounded claims
