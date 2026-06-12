"""Single-turn code GRPO environment — execution-verified rewards (RLVR)."""

from __future__ import annotations

import logging
import os
from typing import Any

from benchmax.envs.base_env import BaseEnv
from benchmax.envs.example_id import make_example
from benchmax.envs.reward_helpers import extract_completion_text
from benchmax.envs.types import Example, Messages, ToolDefinition
from benchmax.platform.credentials import as_token_provider, platform_bearer
from benchmax.rewards.diversity import DiversityConfig, scale_by_diversity
from benchmax.rubrics import (
    evaluate_rubric_ranking,
    group_rubric_ranked_reward_function,
)
from benchmax import config

import code_rewards
import training_utils

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
USE_HOLISTIC = os.environ.get("CODE_HOLISTIC", "1").lower() in ("1", "true", "yes")
USE_DIVERSITY = os.environ.get("CODE_DIVERSITY", "1").lower() in ("1", "true", "yes")
USE_DEPO = os.environ.get("USE_DEPO", "1").lower() in ("1", "true", "yes")
USE_JUDGE_TIEBREAK = os.environ.get("CODE_JUDGE_TIEBREAK", "1").lower() in (
    "1",
    "true",
    "yes",
)

HOLISTIC_RUBRIC = code_rewards.GROUP_RUBRICS[0]
GATE_KEYS = frozenset({
    "format_fence",
    "syntax_ok",
    "hidden_tests_pass",
    "pass_rate",
    "primary_rlvr",
    "test_hack_gate",
    "group_learning_signal",
    "depo_scale",
})


class CodeCapabilitiesEnv(BaseEnv):
    """Train code generation with verifiable unit-test rewards."""

    recommended_max_turns = 1
    recommended_max_tool_calls = 0

    system_prompt = code_rewards.CODE_SYSTEM_SUFFIX.strip()

    def __init__(
        self,
        judge_base_url: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
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
        user = code_rewards.build_user_prompt(example)
        category = example.get("category", "general")
        system = (
            f"{code_rewards.CODE_SYSTEM_SUFFIX.strip()}\n\n"
            f"Category: {category}\n"
            f"Implement `{example.get('entry_point', 'solution')}`."
        )
        task_payload = {
            "prompt": example.get("prompt", ""),
            "task_id": example.get("task_id", ""),
            "category": category,
            "entry_point": example.get("entry_point", ""),
            "test": example.get("test", ""),
            "starter_code": example.get("starter_code", ""),
            "difficulty": example.get("difficulty", "medium"),
            "reward_mode": example.get("reward_mode", "binary"),
        }
        return make_example(
            prompt_messages=[{"role": "user", "content": user}],
            task=task_payload,
            system_prompt=system,
            init_rollout_args={
                "category": category,
                "difficulty": example.get("difficulty", "medium"),
            },
        )

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        task = task or {}
        text = extract_completion_text(messages)
        scored = code_rewards.score_hidden_tests(text, task)
        return scored

    async def list_tools(self) -> list[ToolDefinition]:
        return []

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        return f"Error: no tools in code track (got {tool_name})"

    async def compute_group_reward(
        self,
        rollout_ids: list[str],
        messages_list: list[Messages],
        tasks: list[dict[str, Any] | None],
        **kwargs,
    ) -> list[dict[str, float]]:
        task = tasks[0] or {}
        prompt = str(task.get("prompt", ""))

        rewards_out: list[dict[str, float]] = []
        valid_indices: list[int] = []
        code_texts: list[str] = []

        for i, msgs in enumerate(messages_list):
            per = await self.compute_reward(rollout_ids[i], msgs, task=task)
            rewards_out.append(per)
            text = extract_completion_text(msgs)
            if per.get("syntax_ok", 0) > 0 and per.get("test_hack_gate", 0) > 0:
                valid_indices.append(i)
                code_texts.append(training_utils.extract_python_code(text) or text[:2000])

        training_utils.annotate_group_signal(
            rewards_out, primary_key="hidden_tests_pass"
        )

        primaries = [rewards_out[i].get("hidden_tests_pass", 0.0) for i in range(len(rewards_out))]
        if USE_DEPO and not training_utils.has_group_learning_signal(primaries):
            return rewards_out

        if len(valid_indices) < 2:
            return rewards_out

        depo_scale = training_utils.depo_group_scale(primaries)

        # Skip expensive judge when all pass/fail identically on tests
        if not USE_JUDGE_TIEBREAK or depo_scale <= 0.0:
            return rewards_out

        valid_ids = [rollout_ids[i] for i in valid_indices]
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

        ranked = await group_rubric_ranked_reward_function(
            **judge_kwargs,
            completions=code_texts,
            static_rubrics=code_rewards.GROUP_RUBRICS,
        )

        merged: list[dict[str, float]] = [dict(r) for r in ranked]
        if USE_HOLISTIC:
            holistic = await evaluate_rubric_ranking(
                rubric=HOLISTIC_RUBRIC,
                question=prompt,
                responses=code_texts,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_token_provider(),
                timeout=self._judge_timeout,
            )
            for j, score in enumerate(holistic["scores"]):
                merged[j]["holistic_correctness"] = score

        if USE_DIVERSITY:
            scaled, _ = await scale_by_diversity(
                rewards=merged,
                texts=code_texts,
                config=DiversityConfig(method="ngram", similarity_threshold=0.6),
                context=prompt[:400],
            )
            merged = scaled

        if USE_DEPO and depo_scale < 1.0:
            merged = [
                training_utils.scale_merged_rewards(m, depo_scale, skip_keys=GATE_KEYS)
                for m in merged
            ]

        for local_i, global_i in enumerate(valid_indices):
            rewards_out[global_i].update(merged[local_i])

        return rewards_out
