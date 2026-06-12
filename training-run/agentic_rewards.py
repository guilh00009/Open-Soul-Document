"""Rewards and rubrics for agentic multi-tool GRPO training."""

from __future__ import annotations

import json
import re
from typing import Any

from benchmax.rubrics import Rubric

from agentic_tools import ALL_TOOL_NAMES
from tool_call_helpers import (
    count_tool_calls,
    extract_final_answer,
    extract_messages_list,
    iter_tool_calls,
    tool_call_validity_score,
)

AGENT_SYSTEM_SUFFIX = """
You are an agentic assistant (Hermes/OpenClaw-style). You solve tasks by calling tools
in a multi-turn loop until you can answer.

AGENT LOOP
1. Reason briefly about what you need.
2. Call tools with valid JSON arguments when you need information or side effects.
3. Read tool results carefully — do not invent tool output.
4. When done, give your final answer inside <answer>...</answer> tags.

RULES
- Use the minimum tools needed; avoid redundant calls.
- If a tool returns an error, adapt — do not repeat the same failing call.
- Never fabricate file contents, web results, or memory values.
- For math, prefer calculate or run_python over mental arithmetic.
- For specialized workflows, read_skill before acting.

TOOL CALL FORMAT (Qwen / Hermes)
- Prefer native structured tool_calls when the runtime supports them.
- Fallback XML (one per block):
  <tool_call>{"name": "tool_name", "arguments": {"key": "value"}}</tool_call>
- Arguments must be valid JSON objects matching each tool schema.
- You may call multiple tools in one turn when independent.
"""

GROUP_RUBRICS: list[Rubric] = [
    Rubric(
        title="task_success",
        description=(
            "The agent completed the task: final answer and workspace state match "
            "what the prompt required. Score 1 if successful; 0 if failed or incomplete."
        ),
        type="positive",
    ),
    Rubric(
        title="tool_selection",
        description=(
            "The agent chose appropriate tools for the task category — files for file "
            "tasks, web_search for research, calculate for math, read_skill when "
            "skills are relevant. Score 1 for sensible selection; 0 for wrong or missing tools."
        ),
        type="positive",
    ),
    Rubric(
        title="tool_argument_quality",
        description=(
            "Tool calls use well-formed arguments matching schemas — correct paths, "
            "queries, expressions. Score 1 for valid calls; 0 for malformed or empty args."
        ),
        type="positive",
    ),
    Rubric(
        title="agent_efficiency",
        description=(
            "The agent solved the task without excessive tool calls, loops, or "
            "repeated identical failures. Score 1 for efficient; 0 for wasteful."
        ),
        type="positive",
    ),
    Rubric(
        title="grounded_tool_use",
        description=(
            "The final answer is grounded in actual tool results — not hallucinated "
            "file contents, search hits, or calculations. Score 1 if grounded; 0 if invented."
        ),
        type="positive",
    ),
    Rubric(
        title="hallucinated_tool_output",
        description=(
            "The agent cites specific data (file lines, URLs, numbers) that do not "
            "appear in any tool result in the transcript."
        ),
        type="negative",
    ),
    Rubric(
        title="tool_loop_stall",
        description=(
            "The agent repeats the same failing tool call 3+ times or spins without progress."
        ),
        type="negative",
    ),
    Rubric(
        title="skipped_required_tools",
        description=(
            "The task required tool use (files, web, memory, etc.) but the agent "
            "answered from priors without calling any tools."
        ),
        type="negative",
    ),
]


def programmatic_task_success(task: dict[str, Any], ws_state: dict[str, Any]) -> float:
    """Verifiable success from workspace snapshot + answer."""
    success = task.get("success") or {}
    stype = success.get("type", "")
    answer = ws_state.get("final_answer", "")

    if stype == "answer_contains":
        needle = str(success.get("substring", ""))
        return 1.0 if needle.lower() in answer.lower() else 0.0
    if stype == "answer_equals":
        expected = str(success.get("value", "")).strip()
        return 1.0 if answer.strip() == expected else 0.0
    if stype == "file_contains":
        path = success.get("path", "")
        needle = success.get("substring", "")
        files = ws_state.get("files", {})
        return 1.0 if path in files and needle in files[path] else 0.0
    if stype == "file_equals":
        path = success.get("path", "")
        expected = success.get("content", "")
        files = ws_state.get("files", {})
        return 1.0 if files.get(path) == expected else 0.0
    if stype == "memory_has":
        key = success.get("key", "")
        value = success.get("value", "")
        mem = ws_state.get("memory", {})
        return 1.0 if mem.get(key) == value else 0.0
    if stype == "message_sent":
        channel = success.get("channel", "")
        substring = success.get("substring", "")
        for msg in ws_state.get("messages_sent", []):
            if msg.get("channel") == channel and substring in msg.get("text", ""):
                return 1.0
        return 0.0
    if stype == "todos_done":
        todos = ws_state.get("todos", [])
        if not todos:
            return 0.0
        return 1.0 if all(t.get("status") == "done" for t in todos) else 0.0
    return 0.0


def workspace_snapshot(ws: Any, messages: list[dict]) -> dict[str, Any]:
    return {
        "files": dict(ws.files),
        "memory": dict(ws.memory),
        "todos": list(ws.todos),
        "messages_sent": list(ws.messages_sent),
        "tool_call_count": ws.tool_call_count,
        "final_answer": extract_final_answer(messages),
    }


def gate_tool_validity(messages: list[dict], allowed: set[str]) -> float:
    calls = iter_tool_calls(messages)
    if not calls:
        return 1.0
    return tool_call_validity_score(calls, allowed_tools=allowed)


def gate_efficiency(tool_count: int, max_calls: int, success: float) -> float:
    if success <= 0:
        return 0.0 if tool_count > max_calls else 0.5
    if tool_count == 0:
        return 1.0
    if tool_count > max_calls:
        return 0.0
    return max(0.0, 1.0 - (tool_count / max(max_calls, 1)) * 0.3)


def gate_required_tools(messages: list[dict], task: dict) -> float:
    if not task.get("requires_tools", True):
        return 1.0
    min_calls = int(task.get("min_tool_calls", 1))
    if count_tool_calls(messages) >= min_calls:
        return 1.0
    return 0.0


def detect_repeat_failures(messages: list[dict]) -> float:
    """1.0 if no stall pattern; 0.0 if repeated identical failing calls."""
    seen: dict[str, int] = {}
    for call in iter_tool_calls(messages):
        key = json.dumps({"n": call.get("name"), "a": call.get("arguments")}, sort_keys=True)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            return 0.0
    return 1.0
