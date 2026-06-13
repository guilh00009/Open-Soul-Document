"""Convert Glint-Research/Fable-5-traces rows into benchmax TrainingExamples."""

from __future__ import annotations

import json
import re
from typing import Any

from benchmax.traces.adapter import NormalizedTrace, ToolCall, TraceMessage
from benchmax.traces.processing import TrainingExample, _serialize_messages

USER_SPLIT_RE = re.compile(r"(?=^USER:)", re.MULTILINE)
TRUNCATED_PREFIX = "…"


def _parse_context(context: str) -> list[TraceMessage]:
    """Parse Fable context text into user messages."""
    if not context:
        return []

    if context.startswith(TRUNCATED_PREFIX) or not USER_SPLIT_RE.search(context):
        return [TraceMessage(role="user", content=context)]

    messages: list[TraceMessage] = []
    for part in USER_SPLIT_RE.split(context):
        part = part.strip()
        if not part:
            continue
        if part.startswith("USER:"):
            part = part[5:].strip()
        if part:
            messages.append(TraceMessage(role="user", content=part))
    return messages


def _output_to_assistant(output: Any, output_type: str) -> TraceMessage:
    """Convert a Fable output field into an assistant TraceMessage."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return TraceMessage(role="assistant", content=output)

    if not isinstance(output, dict):
        return TraceMessage(role="assistant", content=str(output))

    if output_type == "tool_use" or "tool" in output:
        tool_name = str(output.get("tool", ""))
        tool_input = output.get("input", {})
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}
        return TraceMessage(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    name=tool_name,
                    arguments=json.dumps(tool_input, ensure_ascii=False),
                )
            ],
        )

    text = str(output.get("text", ""))
    return TraceMessage(role="assistant", content=text)


def row_to_training_example(row: dict[str, Any]) -> TrainingExample | None:
    """Convert one Fable-5-traces row into a TrainingExample."""
    prompt_messages = _parse_context(str(row.get("context", "")))
    if not prompt_messages:
        return None

    output = row.get("output")
    output_type = str(row.get("output_type", "text"))
    completion_messages = [_output_to_assistant(output, output_type)]

    session = str(row.get("session", ""))
    uid = str(row.get("uid", ""))
    turn_index = 0
    if "#" in uid:
        try:
            turn_index = int(uid.rsplit("#", 1)[-1])
        except ValueError:
            pass

    return TrainingExample(
        prompt_messages=prompt_messages,
        completion_messages=completion_messages,
        prompt=_serialize_messages(prompt_messages),
        ground_truth=_serialize_messages(completion_messages),
        trace_id=session or uid,
        turn_index=turn_index,
        scores={},
        metadata={
            "model": row.get("model"),
            "output_type": output_type,
            "origin": row.get("origin"),
            "source_file": row.get("source_file"),
        },
    )


def rows_to_training_examples(rows: list[dict[str, Any]]) -> list[TrainingExample]:
    """Convert Fable rows, skipping any that fail to parse."""
    examples: list[TrainingExample] = []
    for row in rows:
        ex = row_to_training_example(row)
        if ex is not None:
            examples.append(ex)
    return examples


def rows_to_normalized_traces(rows: list[dict[str, Any]]) -> list[NormalizedTrace]:
    """Group Fable rows by session into NormalizedTrace objects."""
    from collections import defaultdict

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sessions[str(row.get("session", row.get("uid", "")))].append(row)

    traces: list[NormalizedTrace] = []
    for session_id, session_rows in sessions.items():
        session_rows.sort(key=lambda r: str(r.get("uid", "")))
        messages: list[TraceMessage] = []
        for row in session_rows:
            prompt = _parse_context(str(row.get("context", "")))
            completion = _output_to_assistant(
                row.get("output"), str(row.get("output_type", "text"))
            )
            if not messages and prompt:
                messages.extend(prompt)
            messages.append(completion)

        if messages:
            traces.append(
                NormalizedTrace(
                    id=session_id,
                    messages=messages,
                    metadata={"model": session_rows[0].get("model")},
                )
            )
    return traces
