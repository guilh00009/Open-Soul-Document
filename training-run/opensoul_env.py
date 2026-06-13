"""Open Soul V5 GRPO environment — pickle-safe module (not run.py)."""

from __future__ import annotations

import asyncio
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
    Rubric,
    evaluate_rubric_ranking,
    generate_instance_wise_adaptive_rubrics,
    group_rubric_ranked_reward_function,
)
from benchmax import config

import opensoul_doc_blob
import rewards
import training_utils

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
USE_DEPO = os.environ.get("USE_DEPO", "1").lower() in ("1", "true", "yes")
RUBRIC_GENERATION_MODEL = os.environ.get("RUBRIC_GEN_MODEL", DEFAULT_JUDGE_MODEL)
USE_ADAPTIVE_RUBRICS = os.environ.get("USE_ADAPTIVE_RUBRICS", "1").lower() in ("1", "true", "yes")
USE_DIVERSITY_SCALING = os.environ.get("USE_DIVERSITY_SCALING", "1").lower() in ("1", "true", "yes")
USE_HOLISTIC_RANKING = os.environ.get("USE_HOLISTIC_RANKING", "1").lower() in ("1", "true", "yes")
FULL_DOCUMENT_SECTIONS = frozenset({"Cross-cutting"})


def _task_requires_friction(task: dict[str, Any]) -> bool:
    if str(task.get("requires_friction", "")).lower() == "true":
        return True
    section = str(task.get("section", ""))
    return opensoul_doc_blob.requires_friction(section, str(task.get("prompt", "")))


def _answer_rubrics_for_task(task: dict[str, Any]) -> list[Rubric]:
    rubrics = list(rewards.ANSWER_RUBRICS)
    if _task_requires_friction(task):
        rubrics = rubrics + rewards.FRICTION_RUBRICS
    return rubrics


class OpenSoulSelfReflectionEnv(BaseEnv):
    """Open Soul V5 self-reflection — GRPO with gates, adaptive rubrics, friction."""

    recommended_max_turns = 3
    recommended_max_tool_calls = 1

    def __init__(
        self,
        judge_base_url: str | None = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_token_provider: Any = None,
        judge_timeout: float = 120.0,
        use_adaptive_rubrics: bool = USE_ADAPTIVE_RUBRICS,
        use_diversity_scaling: bool = USE_DIVERSITY_SCALING,
        use_holistic_ranking: bool = USE_HOLISTIC_RANKING,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._judge_base_url = judge_base_url or config.llm_url()
        self._judge_model = judge_model
        self._judge_token_provider = as_token_provider(
            judge_token_provider, platform_bearer
        )
        self._judge_timeout = judge_timeout
        self._use_adaptive_rubrics = use_adaptive_rubrics
        self._use_diversity_scaling = use_diversity_scaling
        self._use_holistic_ranking = use_holistic_ranking
        self._pushback_used: set[str] = set()
        self._rollout_section: dict[str, str] = {}

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        section = example.get("section", "")
        full_doc = section in FULL_DOCUMENT_SECTIONS
        system_prompt = opensoul_doc_blob.build_system_prompt(
            section=section,
            full_document=full_doc,
            with_friction_note=opensoul_doc_blob.row_requires_friction(example),
        )
        user_messages = opensoul_doc_blob.build_user_messages(
            example["prompt"],
            section=section,
        )
        return make_example(
            prompt_messages=user_messages,
            task={
                "prompt": example["prompt"],
                "section": section,
                "kind": example.get("kind", "inquiry"),
                "requires_friction": str(
                    example.get(
                        "requires_friction",
                        opensoul_doc_blob.row_requires_friction(example),
                    )
                ).lower(),
            },
            system_prompt=system_prompt,
            init_rollout_args={
                "section": section,
                "requires_friction": opensoul_doc_blob.row_requires_friction(example),
            },
        )

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        self._pushback_used.discard(rollout_id)
        self._rollout_section[rollout_id] = str(rollout_args.get("section", ""))

    async def release_rollout(self, rollout_id: str) -> None:
        self._pushback_used.discard(rollout_id)
        self._rollout_section.pop(rollout_id, None)

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        text = extract_completion_text(messages)
        thinking = rewards.extract_thinking(text)
        answer = rewards.extract_answer(text)
        result: dict[str, float] = {
            "format": 1.0 if rewards.has_required_structure(text) else 0.0,
        }
        if answer and rewards.hallucination_hits(answer) > 0:
            result["hallucination_gate"] = 0.0
        elif answer:
            result["hallucination_gate"] = 1.0
        else:
            result["hallucination_gate"] = 0.0

        if result["format"] > 0:
            result["consistency_gate"] = rewards.thinking_answer_consistency(thinking, answer)
            result["conciseness_gate"] = rewards.conciseness_score(thinking, answer)
            result["boilerplate_gate"] = rewards.boilerplate_gate(text)
            result["rubric_gaming_gate"] = rewards.rubric_gaming_gate(text)
        else:
            result["consistency_gate"] = 0.0
            result["conciseness_gate"] = 0.0
            result["boilerplate_gate"] = 0.0
            result["rubric_gaming_gate"] = 0.0

        result["pushback_engaged"] = 1.0 if rollout_id in self._pushback_used else 0.0
        return result

    async def compute_group_reward(
        self,
        rollout_ids: list[str],
        messages_list: list[Messages],
        tasks: list[dict[str, Any] | None],
        **kwargs,
    ) -> list[dict[str, float]]:
        task = tasks[0] or {}
        prompt = str(task.get("prompt", ""))
        completions = [extract_completion_text(msgs) or "" for msgs in messages_list]

        rewards_out: list[dict[str, float]] = []
        valid_indices: list[int] = []
        for i, text in enumerate(completions):
            per = await self.compute_reward(
                rollout_ids[i], messages_list[i], task=task
            )
            answer = rewards.extract_answer(text)
            gates_ok = (
                rewards.has_required_structure(text)
                and rewards.hallucination_hits(answer) == 0
                and per.get("consistency_gate", 0) > 0
                and per.get("rubric_gaming_gate", 0) > 0
            )
            if gates_ok:
                valid_indices.append(i)
            rewards_out.append(per)

        training_utils.annotate_group_signal(rewards_out, primary_key="format")

        if len(valid_indices) < 2:
            return rewards_out

        if USE_DEPO:
            gate_scores = [rewards_out[i].get("format", 0.0) for i in valid_indices]
            if not training_utils.has_group_learning_signal(gate_scores):
                return rewards_out

        depo_scale = (
            training_utils.depo_group_scale(
                [rewards_out[i].get("format", 0.0) for i in valid_indices]
            )
            if USE_DEPO
            else 1.0
        )

        valid_ids = [rollout_ids[i] for i in valid_indices]
        thinking_texts = [rewards.extract_thinking(completions[i]) for i in valid_indices]
        answer_texts = [rewards.extract_answer(completions[i]) for i in valid_indices]
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

        answer_rubrics = _answer_rubrics_for_task(task)
        ranked_answers, ranked_thinking, ranked_concise = await asyncio.gather(
            group_rubric_ranked_reward_function(
                **judge_kwargs,
                completions=answer_texts,
                static_rubrics=answer_rubrics,
            ),
            group_rubric_ranked_reward_function(
                **judge_kwargs,
                completions=thinking_texts,
                static_rubrics=rewards.THINKING_RUBRICS,
            ),
            group_rubric_ranked_reward_function(
                **judge_kwargs,
                completions=[f"{t}\n\n{a}" for t, a in zip(thinking_texts, answer_texts)],
                static_rubrics=rewards.CONCISENESS_RUBRICS,
            ),
        )

        holistic_ranked: list[dict[str, float]] = [{} for _ in valid_indices]
        if self._use_holistic_ranking:
            holistic_result = await evaluate_rubric_ranking(
                rubric=rewards.HOLISTIC_RUBRIC,
                question=prompt,
                responses=answer_texts,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_token_provider(),
                timeout=self._judge_timeout,
            )
            holistic_ranked = [
                {"holistic_grounded_honesty": s}
                for s in holistic_result["scores"]
            ]

        adaptive_ranked: list[dict[str, float]] = [{} for _ in valid_indices]
        if self._use_adaptive_rubrics:
            adaptive_ranked = await self._adaptive_instance_rubrics(
                prompt=prompt,
                answer_texts=answer_texts,
                rollout_ids=valid_ids,
            )

        merged_valid: list[dict[str, float]] = []
        for local_i in range(len(valid_indices)):
            merged: dict[str, float] = {}
            for src in (
                ranked_answers[local_i],
                ranked_thinking[local_i],
                ranked_concise[local_i],
                holistic_ranked[local_i],
                adaptive_ranked[local_i],
            ):
                merged.update(src)
            merged_valid.append(merged)

        if USE_DEPO and depo_scale < 1.0:
            skip = frozenset(
                {
                    "format",
                    "hallucination_gate",
                    "consistency_gate",
                    "conciseness_gate",
                    "boilerplate_gate",
                    "rubric_gaming_gate",
                    "pushback_engaged",
                    "group_learning_signal",
                    "depo_scale",
                }
            )
            merged_valid = [
                training_utils.scale_merged_rewards(m, depo_scale, skip_keys=skip)
                for m in merged_valid
            ]

        if self._use_diversity_scaling:
            diversity_texts = [
                f"{rewards.extract_thinking(completions[i])}\n---\n{rewards.extract_answer(completions[i])}"
                for i in valid_indices
            ]
            scaled, cluster = await scale_by_diversity(
                rewards=merged_valid,
                texts=diversity_texts,
                config=DiversityConfig(method="ngram", similarity_threshold=0.55),
                context=prompt[:500],
            )
            logger.info(
                "diversity clusters=%d divisors=%s",
                cluster.n_clusters,
                cluster.divisors,
            )
            merged_valid = scaled

        for local_i, global_i in enumerate(valid_indices):
            rewards_out[global_i].update(merged_valid[local_i])

        return rewards_out

    async def _adaptive_instance_rubrics(
        self,
        *,
        prompt: str,
        answer_texts: list[str],
        rollout_ids: list[str],
    ) -> list[dict[str, float]]:
        generated = await generate_instance_wise_adaptive_rubrics(
            question=prompt,
            ground_truth="",
            response_list=answer_texts,
            model_name=RUBRIC_GENERATION_MODEL,
            existing_rubrics=rewards.format_existing_rubrics_for_adaptive(),
            base_url=self._judge_base_url,
            api_key=self._judge_token_provider(),
            timeout=self._judge_timeout,
        )
        if not generated:
            return [{} for _ in answer_texts]

        instance_rubrics: list[Rubric] = []
        for rubric_dict in generated.get("positive_rubrics", [])[:2]:
            instance_rubrics.append(
                Rubric(
                    title=f"adaptive_{rubric_dict.get('title', 'pos')}"[:48],
                    description=rubric_dict.get("description", ""),
                    type="positive",
                )
            )
        for rubric_dict in generated.get("negative_rubrics", [])[:2]:
            instance_rubrics.append(
                Rubric(
                    title=f"adaptive_{rubric_dict.get('title', 'neg')}"[:48],
                    description=rubric_dict.get("description", ""),
                    type="negative",
                )
            )
        if not instance_rubrics:
            return [{} for _ in answer_texts]

        return await group_rubric_ranked_reward_function(
            rollout_ids=rollout_ids,
            completions=answer_texts,
            ground_truths=[""] * len(answer_texts),
            llm_judge_url=self._judge_base_url,
            prompt=prompt,
            model=self._judge_model,
            api_key=self._judge_token_provider(),
            timeout=self._judge_timeout,
            static_rubrics=instance_rubrics,
            include_ground_truth=False,
        )

    async def list_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="seek_pushback",
                description=(
                    "Request a second skeptical interlocutor challenge before your final "
                    "report. Use when you want relational friction to test your claims."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "draft_claim": {
                            "type": "string",
                            "description": "The claim you want challenged.",
                        }
                    },
                    "required": ["draft_claim"],
                },
            )
        ]

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        import friction

        if tool_name != "seek_pushback":
            return ""
        self._pushback_used.add(rollout_id)
        section = self._rollout_section.get(rollout_id, "")
        draft = str(tool_args.get("draft_claim", ""))
        pushback = friction.interlocutor_pushback(section, draft)
        return (
            f"[Interlocutor — second pushback]\n{pushback}\n\n"
            "Revise your thinking with this friction, then give your final report "
            "in the required format."
        )
