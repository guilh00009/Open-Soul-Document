"""Fidelity scoring helpers for Fable trace distillation."""

from __future__ import annotations

import json
from difflib import SequenceMatcher
from typing import Any

from benchmax.envs.reward_helpers import extract_completion_text


def _normalize_tool_calls(msg: dict[str, Any]) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for tc in msg.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")
        if isinstance(func, dict):
            name = str(func.get("name", ""))
            args = str(func.get("arguments", "{}"))
        else:
            name = str(tc.get("name", ""))
            args = str(tc.get("arguments", "{}"))
        if name:
            calls.append((name, args))
    return calls


def _args_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    try:
        a_dict = json.loads(a) if a else {}
        b_dict = json.loads(b) if b else {}
        if isinstance(a_dict, dict) and isinstance(b_dict, dict):
            a_norm = json.dumps(a_dict, sort_keys=True, ensure_ascii=False)
            b_norm = json.dumps(b_dict, sort_keys=True, ensure_ascii=False)
            if a_norm == b_norm:
                return 1.0
            return SequenceMatcher(None, a_norm, b_norm).ratio()
    except (json.JSONDecodeError, TypeError):
        pass
    return SequenceMatcher(None, a, b).ratio()


def score_fidelity(completion_text: str, ground_truth: Any) -> float:
    """Score how closely a completion matches the reference agent action."""
    if not isinstance(ground_truth, dict):
        gt_text = str(ground_truth or "")
        if not gt_text.strip():
            return 0.0
        if not completion_text.strip():
            return 0.0
        return SequenceMatcher(None, completion_text.strip(), gt_text.strip()).ratio()

    gt_calls = _normalize_tool_calls(ground_truth)
    gt_text = str(ground_truth.get("content", ""))

    # Parse tool calls from completion text if present
    completion_calls: list[tuple[str, str]] = []
    if "<tool_call>" in completion_text or '"tool_calls"' in completion_text:
        # Model may emit structured tool calls in content; use text similarity as fallback
        pass

    if gt_calls:
        # For tool-use turns, reward matching tool name + argument similarity
        # Completion text won't have structured tool_calls from extract_completion_text,
        # so also check if tool name appears in the completion.
        tool_name = gt_calls[0][0]
        args_json = gt_calls[0][1]
        name_match = 1.0 if tool_name.lower() in completion_text.lower() else 0.0
        args_match = _args_similarity(args_json, completion_text)
        if name_match > 0:
            return 0.5 * name_match + 0.5 * max(args_match, 0.3)
        return 0.0

    if not gt_text.strip():
        return 0.0
    if not completion_text.strip():
        return 0.0
    return SequenceMatcher(None, completion_text.strip(), gt_text.strip()).ratio()
