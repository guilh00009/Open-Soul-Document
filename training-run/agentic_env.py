"""Multi-turn agentic GRPO environment — Hermes + OpenClaw tool patterns."""

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
    evaluate_rubric_ranking,
    group_rubric_ranked_reward_function,
)
from benchmax import config

import agentic_rewards
import agentic_tools
from agentic_workspace import AgenticWorkspace
from tool_call_helpers import extract_messages_list, iter_tool_calls

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini")
USE_HOLISTIC = os.environ.get("AGENTIC_HOLISTIC", "1").lower() in ("1", "true", "yes")
USE_DIVERSITY = os.environ.get("AGENTIC_DIVERSITY", "1").lower() in ("1", "true", "yes")

HOLISTIC_RUBRIC = agentic_rewards.GROUP_RUBRICS[0]  # task_success for ranking


class AgenticCapabilitiesEnv(BaseEnv):
    """Train general agentic tool use — files, web, memory, skills, messaging."""

    recommended_max_turns = 8
    recommended_max_tool_calls = 15

    system_prompt = agentic_rewards.AGENT_SYSTEM_SUFFIX.strip()

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
        self._workspaces: dict[str, AgenticWorkspace] = {}
        self._enabled_tools: dict[str, set[str]] = {}

    @staticmethod
    def _workspace_seed(example: dict) -> dict:
        return {
            "success": example.get("success", {}),
            "workspace": example.get("workspace", {}),
            "web_pages": example.get("web_pages", {}),
            "skills": example.get("skills", {}),
            "browser_url": example.get("browser_url", ""),
            "enabled_tools": example.get("enabled_tools"),
            "requires_tools": example.get("requires_tools", True),
            "min_tool_calls": example.get("min_tool_calls", 1),
            "max_tool_calls": example.get("max_tool_calls", 12),
        }

    @classmethod
    def dataset_preprocess(cls, example: Any, **kwargs) -> Example:
        prompt = example["prompt"]
        category = example.get("category", "general")
        system = (
            f"{agentic_rewards.AGENT_SYSTEM_SUFFIX.strip()}\n\n"
            f"Task category: {category}\n"
            "Use tools until you can answer, then wrap the final result in <answer> tags."
        )
        seed = cls._workspace_seed(example)
        return make_example(
            prompt_messages=[{"role": "user", "content": prompt}],
            task={
                "prompt": prompt,
                "category": category,
                **seed,
            },
            system_prompt=system,
            init_rollout_args={"category": category, **seed},
        )

    async def init_rollout(self, rollout_id: str, **rollout_args) -> None:
        task = self._workspace_seed(rollout_args)
        ws = AgenticWorkspace()
        ws.reset(task)
        self._workspaces[rollout_id] = ws
        enabled = task.get("enabled_tools")
        self._enabled_tools[rollout_id] = (
            set(enabled) if enabled else set(agentic_tools.ALL_TOOL_NAMES)
        )

    async def release_rollout(self, rollout_id: str) -> None:
        self._workspaces.pop(rollout_id, None)
        self._enabled_tools.pop(rollout_id, None)

    def _get_ws(self, rollout_id: str, task: dict | None) -> AgenticWorkspace:
        if rollout_id not in self._workspaces:
            ws = AgenticWorkspace()
            ws.reset(task or {})
            self._workspaces[rollout_id] = ws
            enabled = task.get("enabled_tools") if task else None
            self._enabled_tools[rollout_id] = (
                set(enabled) if enabled else set(agentic_tools.ALL_TOOL_NAMES)
            )
        return self._workspaces[rollout_id]

    async def list_tools(self) -> list[ToolDefinition]:
        return list(agentic_tools.TOOL_DEFINITIONS)

    async def run_tool(self, rollout_id: str, tool_name: str, **tool_args) -> Any:
        ws = self._workspaces.get(rollout_id)
        if ws is None:
            return "Error: rollout not initialized"
        allowed = self._enabled_tools.get(rollout_id, set(agentic_tools.ALL_TOOL_NAMES))
        if tool_name not in allowed:
            return f"Error: tool '{tool_name}' not enabled for this task"
        return await agentic_tools.dispatch_tool(ws, tool_name, **tool_args)

    async def compute_reward(self, rollout_id, messages, task=None, **kwargs):
        task = task or {}
        ws = self._get_ws(rollout_id, task)
        msgs = messages if isinstance(messages, list) else extract_messages_list(messages)
        allowed = self._enabled_tools.get(rollout_id, set(agentic_tools.ALL_TOOL_NAMES))
        snap = agentic_rewards.workspace_snapshot(ws, msgs)
        prog = agentic_rewards.programmatic_task_success(task, snap)

        return {
            "programmatic_success": prog,
            "tool_validity": agentic_rewards.gate_tool_validity(msgs, allowed),
            "tool_efficiency": agentic_rewards.gate_efficiency(
                ws.tool_call_count, ws.max_tool_calls, prog
            ),
            "required_tools": agentic_rewards.gate_required_tools(msgs, task),
            "no_stall": agentic_rewards.detect_repeat_failures(msgs),
            "has_answer": 1.0 if snap.get("final_answer") else 0.0,
        }

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
        answer_texts: list[str] = []
        trace_summaries: list[str] = []

        for i, msgs in enumerate(messages_list):
            per = await self.compute_reward(rollout_ids[i], msgs, task=task)
            rewards_out.append(per)
            msg_list = msgs if isinstance(msgs, list) else extract_messages_list(msgs)
            answer = agentic_rewards.workspace_snapshot(
                self._get_ws(rollout_ids[i], task), msg_list
            ).get("final_answer", "")
            if (
                per.get("has_answer", 0) > 0
                and per.get("tool_validity", 0) > 0
                and per.get("required_tools", 0) > 0
            ):
                valid_indices.append(i)
                answer_texts.append(answer or extract_completion_text(msgs))
                calls = iter_tool_calls(msg_list)
                trace_summaries.append(
                    f"tools={[c['name'] for c in calls]} answer={answer[:200]}"
                )

        if len(valid_indices) < 2:
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
            completions=answer_texts,
            static_rubrics=agentic_rewards.GROUP_RUBRICS,
        )

        merged: list[dict[str, float]] = [dict(r) for r in ranked]
        if USE_HOLISTIC:
            holistic = await evaluate_rubric_ranking(
                rubric=HOLISTIC_RUBRIC,
                question=prompt,
                responses=answer_texts,
                model_name=self._judge_model,
                base_url=self._judge_base_url,
                api_key=self._judge_token_provider(),
                timeout=self._judge_timeout,
            )
            for j, score in enumerate(holistic["scores"]):
                merged[j]["holistic_task_success"] = score

        if USE_DIVERSITY:
            scaled, _ = await scale_by_diversity(
                rewards=merged,
                texts=trace_summaries,
                config=DiversityConfig(method="ngram", similarity_threshold=0.5),
                context=prompt[:400],
            )
            merged = scaled

        for local_i, global_i in enumerate(valid_indices):
            rewards_out[global_i].update(merged[local_i])

        return rewards_out
