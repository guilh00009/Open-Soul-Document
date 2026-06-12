"""Per-rollout in-memory workspace — OpenClaw/Hermes-style agent state."""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from dataclasses import dataclass, field
from typing import Any

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


@dataclass
class AgenticWorkspace:
    files: dict[str, str] = field(default_factory=dict)
    memory: dict[str, str] = field(default_factory=dict)
    todos: list[dict[str, Any]] = field(default_factory=list)
    sessions: dict[str, list[str]] = field(default_factory=dict)
    messages_sent: list[dict[str, str]] = field(default_factory=list)
    web_pages: dict[str, str] = field(default_factory=dict)
    skills: dict[str, str] = field(default_factory=dict)
    browser_url: str = ""
    tool_call_count: int = 0
    max_tool_calls: int = 15

    def reset(self, task: dict[str, Any]) -> None:
        self.files = dict(task.get("workspace", {}).get("files", {}))
        self.memory = {}
        self.todos = []
        self.sessions = dict(task.get("workspace", {}).get("sessions", {}))
        self.messages_sent = []
        self.web_pages = dict(task.get("web_pages", {}))
        self.skills = dict(task.get("skills", {}))
        self.browser_url = task.get("browser_url", "")
        self.tool_call_count = 0
        self.max_tool_calls = int(task.get("max_tool_calls", 15))

    def budget_ok(self) -> bool:
        return self.tool_call_count < self.max_tool_calls

    def record_tool_call(self) -> str | None:
        self.tool_call_count += 1
        if self.tool_call_count > self.max_tool_calls:
            return (
                f"Error: tool budget exceeded ({self.max_tool_calls} calls). "
                "Provide your final report after </think>."
            )
        return None

    # ── File tools (OpenClaw group:fs) ───────────────────────────────────

    def read_file(self, path: str) -> str:
        path = path.strip().lstrip("/")
        if path not in self.files:
            return f"Error: file not found: {path}"
        return self.files[path]

    def write_file(self, path: str, content: str) -> str:
        path = path.strip().lstrip("/")
        self.files[path] = content
        return f"ok: wrote {len(content)} bytes to {path}"

    def list_files(self, directory: str = "") -> str:
        prefix = directory.strip().lstrip("/")
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        names = sorted(
            p for p in self.files if (not prefix or p.startswith(prefix))
        )
        return "\n".join(names) if names else "(empty)"

    def patch_file(self, path: str, old_text: str, new_text: str) -> str:
        path = path.strip().lstrip("/")
        if path not in self.files:
            return f"Error: file not found: {path}"
        if old_text not in self.files[path]:
            return "Error: old_text not found in file"
        self.files[path] = self.files[path].replace(old_text, new_text, 1)
        return "ok: patched"

    # ── Exec (sandboxed — OpenClaw exec) ─────────────────────────────────

    def exec(self, command: str) -> str:
        cmd = (command or "").strip()
        if cmd.startswith("cat "):
            return self.read_file(cmd[4:].strip())
        if cmd in ("ls", "ls ."):
            return self.list_files()
        if cmd.startswith("ls "):
            return self.list_files(cmd[3:].strip())
        if cmd.startswith("grep "):
            parts = cmd.split(" ", 2)
            if len(parts) < 3:
                return "Error: usage grep <pattern> <path>"
            pattern, path = parts[1], parts[2].strip()
            text = self.read_file(path)
            if text.startswith("Error:"):
                return text
            lines = [ln for ln in text.splitlines() if pattern in ln]
            return "\n".join(lines) if lines else "(no matches)"
        if cmd.startswith("wc -l "):
            text = self.read_file(cmd[6:].strip())
            if text.startswith("Error:"):
                return text
            return str(len(text.splitlines()))
        return (
            "Error: sandbox exec allows only: cat, ls, grep, wc -l. "
            "Use read_file/write_file for other operations."
        )

    # ── Web (OpenClaw web_search / web_fetch) ──────────────────────────

    def web_search(self, query: str, limit: int = 5) -> str:
        q = (query or "").lower()
        hits: list[tuple[int, str, str]] = []
        for title, body in self.web_pages.items():
            score = sum(1 for w in q.split() if w in (title + body).lower())
            if score:
                hits.append((score, title, body[:300]))
        hits.sort(reverse=True)
        if not hits:
            return "No results."
        lines = []
        for i, (_, title, snippet) in enumerate(hits[: max(1, limit)]):
            lines.append(f"[{i+1}] {title}\n{snippet}")
        return "\n\n".join(lines)

    def web_fetch(self, url: str) -> str:
        url = (url or "").strip()
        for key, body in self.web_pages.items():
            if url in key or key in url:
                return body
        return f"Error: URL not in sandbox: {url}"

    def browser_snapshot(self) -> str:
        if self.browser_url and self.browser_url in self.web_pages:
            return f"URL: {self.browser_url}\n\n{self.web_pages[self.browser_url]}"
        if self.web_pages:
            first = next(iter(self.web_pages.items()))
            return f"URL: {first[0]}\n\n{first[1]}"
        return "Error: no browser content in this task"

    # ── Skills (OpenClaw on-demand SKILL.md) ───────────────────────────

    def read_skill(self, skill_name: str) -> str:
        key = (skill_name or "").strip()
        if key in self.skills:
            return self.skills[key]
        for name, body in self.skills.items():
            if key.lower() in name.lower():
                return body
        return f"Error: skill not found: {skill_name}"

    # ── Memory (agent session continuity) ────────────────────────────────

    def memory_store(self, key: str, value: str) -> str:
        self.memory[str(key)] = str(value)
        return "ok: stored"

    def memory_recall(self, key: str) -> str:
        if key not in self.memory:
            return f"Error: no memory for key: {key}"
        return self.memory[key]

    # ── Planning (Hermes / OpenClaw automation) ────────────────────────

    def todo_write(self, todos: list[dict[str, Any]] | str) -> str:
        if isinstance(todos, str):
            try:
                todos = json.loads(todos)
            except json.JSONDecodeError:
                return "Error: todos must be a JSON list"
        if not isinstance(todos, list):
            return "Error: todos must be a list"
        self.todos = todos
        return f"ok: {len(todos)} todo(s)"

    def todo_list(self) -> str:
        if not self.todos:
            return "(no todos)"
        return json.dumps(self.todos, indent=2)

    # ── Messaging (OpenClaw message / sessions) ────────────────────────

    def send_message(self, channel: str, text: str) -> str:
        entry = {"channel": str(channel), "text": str(text)}
        self.messages_sent.append(entry)
        return f"ok: sent to {channel}"

    def sessions_list(self) -> str:
        if not self.sessions:
            return "(no sessions)"
        return "\n".join(f"- {sid}: {len(msgs)} messages" for sid, msgs in self.sessions.items())

    # ── Calculate / Python (Hermes code tools) ─────────────────────────────

    def calculate(self, expression: str) -> str:
        try:
            return str(_safe_eval(expression))
        except Exception as exc:
            return f"Error: {exc}"

    def run_python(self, code: str) -> str:
        code = (code or "").strip()
        if code.startswith("print(") and code.endswith(")"):
            inner = code[6:-1]
            try:
                return str(_safe_eval(inner))
            except Exception as exc:
                return f"Error: {exc}"
        try:
            return str(_safe_eval(code))
        except Exception as exc:
            return f"Error: {exc} — sandbox allows only safe arithmetic expressions"


def _safe_eval(expr: str) -> float:
    node = ast.parse((expr or "").strip(), mode="eval")

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp) and type(n.op) in _SAFE_OPS:
            return _SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError("unsupported expression")

    return _eval(node)
