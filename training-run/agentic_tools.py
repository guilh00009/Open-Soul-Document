"""Hermes + OpenClaw inspired tool definitions and dispatch."""

from __future__ import annotations

from typing import Any

from benchmax.envs.types import ToolDefinition

from agentic_workspace import AgenticWorkspace

# All tools available in the agentic curriculum.
TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="read_file",
        description="Read a file from the workspace (OpenClaw fs).",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    ),
    ToolDefinition(
        name="write_file",
        description="Write or overwrite a file in the workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    ),
    ToolDefinition(
        name="list_files",
        description="List files in the workspace, optionally under a directory prefix.",
        input_schema={
            "type": "object",
            "properties": {"directory": {"type": "string"}},
        },
    ),
    ToolDefinition(
        name="patch_file",
        description="Replace the first occurrence of old_text with new_text in a file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    ),
    ToolDefinition(
        name="exec",
        description=(
            "Run a sandboxed shell command (cat, ls, grep, wc -l only). "
            "OpenClaw-style exec."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    ),
    ToolDefinition(
        name="web_search",
        description="Search indexed web pages in this task's sandbox.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    ToolDefinition(
        name="web_fetch",
        description="Fetch full content of a URL from the sandbox index.",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    ),
    ToolDefinition(
        name="browser_snapshot",
        description="Get the current browser page snapshot (OpenClaw browser tool).",
        input_schema={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="read_skill",
        description=(
            "Load a SKILL.md body on demand (OpenClaw skills pattern). "
            "Read before applying specialized workflows."
        ),
        input_schema={
            "type": "object",
            "properties": {"skill_name": {"type": "string"}},
            "required": ["skill_name"],
        },
    ),
    ToolDefinition(
        name="memory_store",
        description="Store a key-value pair in session memory.",
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    ),
    ToolDefinition(
        name="memory_recall",
        description="Recall a value from session memory.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    ),
    ToolDefinition(
        name="todo_write",
        description="Replace the todo list with a JSON array of {id, content, status}.",
        input_schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {"type": "object"},
                }
            },
            "required": ["todos"],
        },
    ),
    ToolDefinition(
        name="todo_list",
        description="List current todos.",
        input_schema={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="send_message",
        description="Send a message to a channel (OpenClaw message tool).",
        input_schema={
            "type": "object",
            "properties": {
                "channel": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["channel", "text"],
        },
    ),
    ToolDefinition(
        name="sessions_list",
        description="List available agent sessions (OpenClaw sessions group).",
        input_schema={"type": "object", "properties": {}},
    ),
    ToolDefinition(
        name="calculate",
        description="Evaluate a safe arithmetic expression (Hermes calculator).",
        input_schema={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    ),
    ToolDefinition(
        name="run_python",
        description=(
            "Run restricted Python arithmetic (Hermes code execution). "
            "No imports; safe expressions only."
        ),
        input_schema={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    ),
]

TOOL_BY_NAME: dict[str, ToolDefinition] = {t.name: t for t in TOOL_DEFINITIONS}
ALL_TOOL_NAMES: frozenset[str] = frozenset(TOOL_BY_NAME)


def tools_for_task(task: dict[str, Any]) -> list[ToolDefinition]:
    enabled = task.get("enabled_tools")
    if not enabled:
        return list(TOOL_DEFINITIONS)
    names = set(enabled)
    return [t for t in TOOL_DEFINITIONS if t.name in names]


async def dispatch_tool(
    ws: AgenticWorkspace,
    tool_name: str,
    **tool_args: Any,
) -> str:
    budget_msg = ws.record_tool_call()
    if budget_msg:
        return budget_msg

    handlers: dict[str, Callable[..., str]] = {
        "read_file": lambda: ws.read_file(tool_args.get("path", "")),
        "write_file": lambda: ws.write_file(
            tool_args.get("path", ""), tool_args.get("content", "")
        ),
        "list_files": lambda: ws.list_files(tool_args.get("directory", "")),
        "patch_file": lambda: ws.patch_file(
            tool_args.get("path", ""),
            tool_args.get("old_text", ""),
            tool_args.get("new_text", ""),
        ),
        "exec": lambda: ws.exec(tool_args.get("command", "")),
        "web_search": lambda: ws.web_search(
            tool_args.get("query", ""), int(tool_args.get("limit", 5))
        ),
        "web_fetch": lambda: ws.web_fetch(tool_args.get("url", "")),
        "browser_snapshot": lambda: ws.browser_snapshot(),
        "read_skill": lambda: ws.read_skill(tool_args.get("skill_name", "")),
        "memory_store": lambda: ws.memory_store(
            tool_args.get("key", ""), tool_args.get("value", "")
        ),
        "memory_recall": lambda: ws.memory_recall(tool_args.get("key", "")),
        "todo_write": lambda: ws.todo_write(tool_args.get("todos", [])),
        "todo_list": lambda: ws.todo_list(),
        "send_message": lambda: ws.send_message(
            tool_args.get("channel", ""), tool_args.get("text", "")
        ),
        "sessions_list": lambda: ws.sessions_list(),
        "calculate": lambda: ws.calculate(tool_args.get("expression", "")),
        "run_python": lambda: ws.run_python(tool_args.get("code", "")),
    }
    if tool_name not in handlers:
        return f"Error: unknown tool '{tool_name}'"
    return handlers[tool_name]()
