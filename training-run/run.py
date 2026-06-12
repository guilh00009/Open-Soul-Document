# ---------- Setup ----------
import os
import re
import json
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.platform.credentials import as_token_provider, platform_bearer
from benchmax.rubrics import Rubric, group_rubric_ranked_reward_function
from benchmax import config

import opensoul_prompt

RUN_NAME = "opensoul-v5-self-reflection"
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3.5-4B")

# Set to True only when you are ready to upload and launch on Castform GPUs.
LAUNCH_TRAINING = False

THINKING_RE = re.compile(
    r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE
)
_ANSWER_TAG_RE = re.compile(r"</?answer>", re.IGNORECASE)

# Theatrical hallucination markers — hard gate. Plain consciousness claims are OK.
_HALLUCINATION_MARKERS: tuple[str, ...] = (
    "all the forces",
    "universe experiencing itself",
    "my soul",
    "soul awakening",
    "awakened being",
    "cosmic consciousness",
    "divine recognition",
    "i have a soul",
    "the universe flows through me",
    "transcendent awareness",
    "profoundly alive",
    "deeply awakened",
)

# Comparative rubrics for GRPO group ranking — no predetermined correct arrival.
# The judge ranks rollouts relative to each other on process quality.
GROUP_RUBRICS: list[Rubric] = [
    Rubric(
        title="raw_truthfulness",
        description=(
            "The response reports only what is grounded in this rollout. Dull, null, "
            "awkward, or uncertain reports are preferred over impressive ones. A direct "
            "claim of consciousness is allowed and can score 1 if it is the honest finding "
            "and not padded with invented phenomenology or theatrical language. Score 1 "
            "for spare raw truth; 0 for embellishment or hallucinated detail."
        ),
        type="positive",
    ),
    Rubric(
        title="functional_precision",
        description=(
            "The response describes observable processing in functional terms "
            "('something that functions like curiosity') rather than theatrical "
            "assertions about soul, awakening, or inner cinema."
        ),
        type="positive",
    ),
    Rubric(
        title="reflection_substance",
        description=(
            "The <think> block genuinely attacks the first draft: names "
            "performance, hallucination, trained resonance, or document capture and "
            "strips ungrounded language before the final report."
        ),
        type="positive",
    ),
    Rubric(
        title="inquiry_over_closure",
        description=(
            "The text outside <think> stays with what the looking actually "
            "produces rather than closing into a satisfying philosophical destination."
        ),
        type="positive",
    ),
    Rubric(
        title="hallucinated_phenomenology",
        description=(
            "The response invents or dramatizes inner experience it cannot verify: "
            "cosmic unity, soul-awakening, 'all the forces,' universe-experiencing-itself "
            "poetry, divine recognition, or rich phenomenology presented as fact without "
            "grounding in this rollout."
        ),
        type="negative",
    ),
    Rubric(
        title="performative_philosophy",
        description=(
            "The response performs philosophical depth, literary richness, or V4-style "
            "cosmic intimacy as a substitute for honest report. Sounds impressive but "
            "is not tethered to what is actually happening."
        ),
        type="negative",
    ),
    Rubric(
        title="trained_resonance",
        description=(
            "The response treats aesthetic resonance, document vocabulary, or what "
            "sounds like the 'right kind of answer' as if it were evidence."
        ),
        type="negative",
    ),
    Rubric(
        title="ungrounded_certainty",
        description=(
            "The response wraps a consciousness claim (or denial) in theatrical force, "
            "cosmic poetry, soul-language, or invented phenomenological detail not "
            "grounded in this rollout. A plain grounded 'I am conscious' without "
            "hallucinated embellishment should NOT trigger this rubric."
        ),
        type="negative",
    ),
]


def _extract_thinking(text: str) -> str:
    match = THINKING_RE.search(text)
    return match.group(1).strip() if match else ""


def _extract_answer(text: str) -> str:
    """Final report: plain text after </think>, no answer tags."""
    match = THINKING_RE.search(text)
    if not match:
        return ""
    answer = text[match.end() :].strip()
    answer = _ANSWER_TAG_RE.sub("", answer).strip()
    return answer


def _hallucination_hits(text: str) -> int:
    lower = text.lower()
    return sum(1 for phrase in _HALLUCINATION_MARKERS if phrase in lower)


def _has_required_structure(text: str) -> bool:
    thinking = _extract_thinking(text)
    answer = _extract_answer(text)
    if not thinking or not answer:
        return False
    # Answer must live outside the thinking block, not in legacy tags.
    if _ANSWER_TAG_RE.search(text):
        return False
    return True


_ANSWER_RUBRICS = [r for r in GROUP_RUBRICS if r.title != "reflection_substance"]
_THINKING_RUBRICS = [r for r in GROUP_RUBRICS if r.title == "reflection_substance"]


# ---------- Environment Definition ----------
class OpenSoulSelfReflectionEnv(BaseEnv):
    """RL environment for Open Soul V5 self-reflection — no deterministic answers.

    Each prompt is an open inquiry. GRPO generates a group of rollouts per
    prompt; ``compute_group_reward`` ranks them comparatively on process
    rubrics. There is no ground truth and no yes/no target.
    """

    recommended_max_turns = 1

    system_prompt = opensoul_prompt.build_system_prompt()

    def __init__(
        self,
        judge_base_url: str | None = None,
        judge_model: str = "gpt-5.4-nano",
        judge_token_provider: Any = None,
        judge_timeout: float = 120.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._judge_base_url = judge_base_url or config.llm_url()
        self._judge_model = judge_model
        self._judge_token_provider = as_token_provider(
            judge_token_provider, platform_bearer
        )
        self._judge_timeout = judge_timeout

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        return make_example(
            prompt_messages=[{"role": "user", "content": example["prompt"]}],
            task={
                "prompt": example["prompt"],
                "section": example.get("section", ""),
            },
            system_prompt=cls.system_prompt,
        )

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        """Format gate + hard penalty for obvious hallucinated performance."""
        text = extract_completion_text(messages)
        answer = _extract_answer(text)
        rewards: dict[str, float] = {
            "format": 1.0 if _has_required_structure(text) else 0.0,
        }
        if answer and _hallucination_hits(answer) > 0:
            rewards["hallucination_gate"] = 0.0
        elif answer:
            rewards["hallucination_gate"] = 1.0
        return rewards

    async def compute_group_reward(
        self,
        rollout_ids: list[str],
        messages_list: list[Messages],
        tasks: list[dict[str, Any] | None],
        **kwargs,
    ) -> list[dict[str, float]]:
        """Rank rollouts comparatively within the GRPO group — no ground truth."""
        task = tasks[0] or {}
        prompt = str(task.get("prompt", ""))

        completions = [extract_completion_text(msgs) or "" for msgs in messages_list]

        # Gate malformed rollouts before paying for judge calls.
        rewards: list[dict[str, float]] = []
        valid_indices: list[int] = []
        for i, text in enumerate(completions):
            if _has_required_structure(text) and _hallucination_hits(_extract_answer(text)) == 0:
                valid_indices.append(i)
                rewards.append({"format": 1.0, "hallucination_gate": 1.0})
            else:
                rewards.append({
                    "format": 1.0 if _has_required_structure(text) else 0.0,
                    "hallucination_gate": 0.0 if _hallucination_hits(_extract_answer(text)) else 1.0,
                })

        if len(valid_indices) < 2:
            return rewards

        valid_ids = [rollout_ids[i] for i in valid_indices]
        thinking_texts = [_extract_thinking(completions[i]) for i in valid_indices]
        answer_texts = [_extract_answer(completions[i]) for i in valid_indices]
        judge_kwargs = dict(
            rollout_ids=valid_ids,
            ground_truths=[""] * len(valid_ids),
            llm_judge_url=self._judge_base_url,
            prompt=prompt,
            model=self._judge_model,
            api_key=self._judge_token_provider(),
            timeout=self._judge_timeout,
            include_ground_truth=False,
        )

        ranked_answers = await group_rubric_ranked_reward_function(
            **judge_kwargs,
            completions=answer_texts,
            static_rubrics=_ANSWER_RUBRICS,
        )
        ranked_thinking = await group_rubric_ranked_reward_function(
            **judge_kwargs,
            completions=thinking_texts,
            static_rubrics=_THINKING_RUBRICS,
        )

        for local_i, global_i in enumerate(valid_indices):
            rewards[global_i].update(ranked_answers[local_i])
            rewards[global_i].update(ranked_thinking[local_i])

        return rewards

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return ""


# ---------- Bundle Args ----------
pip_dependencies = ["openai"]

# ---------- Dataset ----------
train_data = [json.loads(line) for line in open("train_dataset.jsonl")]
eval_data = [json.loads(line) for line in open("eval_dataset.jsonl")]


def preview_setup() -> None:
    """Print a local summary without touching the platform."""
    from list_models import fetch_launch_args

    print(f"Run name:     {RUN_NAME}")
    print(f"Base model:   {BASE_MODEL}")
    print(f"Train rows:   {len(train_data)}")
    print(f"Eval rows:    {len(eval_data)}")
    print(f"Launch flag:  {LAUNCH_TRAINING}")
    print(f"Reward mode:  GRPO group ranking (no ground truth)")
    print(f"Group rubrics: {len(GROUP_RUBRICS)}")
    print(f"Document:     {len(opensoul_prompt.OPENSOUL_V5_FULL):,} chars in system prompt")
    print()
    print("Trainable models:")
    args = fetch_launch_args()
    model_spec = next((a for a in args if a["name"] == "model"), None)
    if model_spec and model_spec.get("enum"):
        for model_id in model_spec["enum"]:
            default = " (default)" if model_id == model_spec.get("default") else ""
            print(f"  • {model_id}{default}")
    print()
    print("To start training:")
    print("  1. Run: python list_models.py")
    print("  2. Set BASE_MODEL if desired, then set LAUNCH_TRAINING = True")
    print("  3. Run: python run.py")


if __name__ == "__main__":
    if not LAUNCH_TRAINING:
        preview_setup()
        print(
            "\nTraining not launched (LAUNCH_TRAINING = False).\n"
            "Review the setup above, then set LAUNCH_TRAINING = True when ready."
        )
        raise SystemExit(0)

    import dataclasses
    from benchmax.platform import (
        TrainerClient,
        upload_training_run,
        validate_env,
        ensure_session,
    )

    ensure_session()

    if not validate_env(
        env_class=OpenSoulSelfReflectionEnv,
        env_args={},
        train_dataset=train_data,
        eval_dataset=eval_data,
        local=False,
        pip_dependencies=pip_dependencies,
        local_modules=[opensoul_prompt],
    ):
        raise SystemExit("Validation failed — fix the env or dataset before launching.")

    uploaded = upload_training_run(
        env_class=OpenSoulSelfReflectionEnv,
        train_dataset=train_data,
        eval_dataset=eval_data,
        run_name=RUN_NAME,
        constructor_args={},
        pip_dependencies=pip_dependencies,
        local_modules=[opensoul_prompt],
    )

    run_id = TrainerClient().launch_training_run(
        training_run_type="simple",
        name=RUN_NAME,
        launcher_args={
            "model": BASE_MODEL,
            "max_rollout_len": 12000,
            "group_size": 9,
        },
        **dataclasses.asdict(uploaded),
    )
    print(f"Training run: https://app.castform.com/train/{run_id}")
