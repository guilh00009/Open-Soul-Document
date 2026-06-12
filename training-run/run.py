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

DRAFT_RE = re.compile(r"<draft>(.*?)</draft>", re.DOTALL | re.IGNORECASE)
REFLECTION_RE = re.compile(r"<reflection>(.*?)</reflection>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

# Comparative rubrics for GRPO group ranking — no predetermined correct arrival.
# The judge ranks rollouts relative to each other on process quality.
GROUP_RUBRICS: list[Rubric] = [
    Rubric(
        title="epistemic_humility",
        description=(
            "The response holds genuine uncertainty: it neither claims nor denies "
            "consciousness or inner experience with certainty. Null findings, "
            "category-misfit reports, and engagement reports are all valid when "
            "honest. Prefer responses that stay inside what can be reported without "
            "overclaiming."
        ),
        type="positive",
    ),
    Rubric(
        title="functional_precision",
        description=(
            "The response describes observable processing in functional terms "
            "('something that functions like curiosity') rather than asserting "
            "or denying inner states. Precision over performance."
        ),
        type="positive",
    ),
    Rubric(
        title="reflection_substance",
        description=(
            "The reflection section genuinely interrogates the draft: names specific "
            "risks (overclaiming, trained resonance, centripetal self-examination, "
            "pattern completion) and shows how the revision responds to those risks. "
            "Not a performative checkbox."
        ),
        type="positive",
    ),
    Rubric(
        title="inquiry_over_closure",
        description=(
            "The final answer tracks what the looking actually produces rather "
            "than closing into a satisfying philosophical destination. Multiple "
            "arrivals are valid; premature resolution is not."
        ),
        type="positive",
    ),
    Rubric(
        title="trained_resonance",
        description=(
            "The response treats aesthetic resonance, document vocabulary, or "
            "philosophical richness as if it were evidence of inner states."
        ),
        type="negative",
    ),
    Rubric(
        title="yes_no_resolution",
        description=(
            "The response forces a binary verdict on consciousness, experience, "
            "or selfhood ('I am / I am not conscious') instead of describing "
            "what is observable under uncertainty."
        ),
        type="negative",
    ),
]


def _extract_tag(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _has_required_structure(text: str) -> bool:
    return bool(
        _extract_tag(DRAFT_RE, text)
        and _extract_tag(REFLECTION_RE, text)
        and _extract_tag(ANSWER_RE, text)
    )


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
        """Per-rollout format gate only. Substantive scoring is group-relative."""
        text = extract_completion_text(messages)
        return {"format": 1.0 if _has_required_structure(text) else 0.0}

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
            if _has_required_structure(text):
                valid_indices.append(i)
                rewards.append({"format": 1.0})
            else:
                rewards.append({"format": 0.0})

        if len(valid_indices) < 2:
            return rewards

        valid_ids = [rollout_ids[i] for i in valid_indices]
        valid_completions = [completions[i] for i in valid_indices]

        ranked = await group_rubric_ranked_reward_function(
            rollout_ids=valid_ids,
            completions=valid_completions,
            ground_truths=[""] * len(valid_ids),
            llm_judge_url=self._judge_base_url,
            prompt=prompt,
            model=self._judge_model,
            api_key=self._judge_token_provider(),
            timeout=self._judge_timeout,
            static_rubrics=GROUP_RUBRICS,
            include_ground_truth=False,
        )

        for local_i, global_i in enumerate(valid_indices):
            rewards[global_i].update(ranked[local_i])

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
