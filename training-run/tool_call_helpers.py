"""Parse and score tool calls from rollout message transcripts."""

from __future__ import annotations

import json
import re
from typing import Any

_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)

def extract_messages_list(completion: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(completion, list):
        return [m for m in completion if isinstance(m, dict)]
    return []


def iter_tool_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Yield normalized tool calls: {name, arguments, id, source}."""
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        # OpenAI-style structured tool_calls
        for tc in msg.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
            name = fn.get("name") or tc.get("name") or ""
            raw_args = fn.get("arguments") if isinstance(fn, dict) else tc.get("arguments", {})
            args = _parse_args(raw_args)
            calls.append({
                "name": str(name),
                "arguments": args,
                "id": tc.get("id", ""),
                "source": "structured",
            })
        # Hermes / Qwen XML-in-content fallback
        content = msg.get("content") or ""
        if isinstance(content, str):
            for match in _TOOL_CALL_XML_RE.finditer(content):
                try:
                    payload = json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
                calls.append({
                    "name": str(payload.get("name", "")),
                    "arguments": payload.get("arguments") or payload.get("parameters") or {},
                    "id": "",
                    "source": "xml",
                })
    return calls


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def count_tool_calls(messages: list[dict[str, Any]]) -> int:
    return len(iter_tool_calls(messages))


def _final_assistant_content(messages: list[dict[str, Any]]) -> str:
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content") or ""
        if isinstance(content, str) and content.strip():
            return content
    return ""


def extract_final_answer(messages: list[dict[str, Any]]) -> str:
    """Plain-text answer after </think> on the last assistant turn."""
    from rewards import extract_answer

    content = _final_assistant_content(messages)
    if not content:
        return ""
    answer = extract_answer(content)
    if answer:
        return answer
    if not iter_tool_calls([{"role": "assistant", "content": content}]):
        return content.strip()
    return ""


def tool_call_validity_score(
    calls: list[dict[str, Any]],
    *,
    allowed_tools: set[str],
) -> float:
    if not calls:
        return 0.0
    ok = 0
    for call in calls:
        name = call.get("name", "")
        if name not in allowed_tools:
            continue
        if not isinstance(call.get("arguments"), dict):
            continue
        ok += 1
    return ok / len(calls)


def unique_tools_used(calls: list[dict[str, Any]]) -> set[str]:
    return {c["name"] for c in calls if c.get("name")}
