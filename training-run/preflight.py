#!/usr/bin/env python3
"""Pre-launch checks beyond benchmax validate_env.

validate_env never calls init_rollout, so workspace-seeding bugs can pass local
checks yet fail on Castform during multi-turn tool rollouts. This script closes
that gap and verifies pickle bundling + answer format gates.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any, Callable

import cloudpickle


def _fail(msg: str) -> None:
    print(f"PREFLIGHT FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


async def _check_init_rollout(
    env: Any,
    preprocess: Callable[..., Any],
    rows: list[dict],
    *,
    label: str,
    sample_tool: str | None = None,
) -> None:
    if not hasattr(env, "init_rollout"):
        return
    for i, row in enumerate(rows[: min(5, len(rows))]):
        ex = preprocess(row)
        init_args = ex.get("init_rollout_args") or {}
        rid = f"preflight-{label}-{i}"
        await env.init_rollout(rid, **init_args)
        if sample_tool and hasattr(env, "run_tool"):
            tools = await env.list_tools()
            names = {t.name for t in tools}
            if sample_tool in names:
                result = await env.run_tool(rid, sample_tool, objection="preflight")
                if str(result).startswith("Error: rollout not initialized"):
                    _fail(f"{label} example {i}: init_rollout did not register workspace")
    _ok(f"{label}: init_rollout + workspace OK ({min(5, len(rows))} samples)")


def check_pickle(env: Any, *, label: str) -> None:
    try:
        blob = cloudpickle.dumps(env)
        cloudpickle.loads(blob)
    except Exception as exc:
        _fail(f"{label} pickle round-trip failed: {exc}")
    _ok(f"{label}: pickle round-trip OK ({len(blob)} bytes)")


def check_answer_format() -> None:
    from rewards import extract_answer, has_required_structure

    good = (
        "<think>checking</think>\n"
        "Plain answer here."
    )
    bad_tags = (
        "<think>ok</think>\n"
        "<answer>wrapped</answer>"
    )
    if not has_required_structure(good):
        _fail("answer format gate rejects valid redacted_thinking + plain text")
    if has_required_structure(bad_tags):
        _fail("answer format gate still accepts <answer> tags")
    if not extract_answer(good):
        _fail("extract_answer failed on valid completion")
    _ok("answer format gate (no <answer> tags)")


async def check_code_sandbox(rows: list[dict]) -> None:
    from code_sandbox import _build_test_script, run_hidden_tests

    for row in rows:
        starter = row.get("starter_code") or (
            f"def {row['entry_point']}(*args, **kwargs):\n    raise NotImplementedError\n"
        )
        script = _build_test_script(starter, row["entry_point"], row["test"])
        try:
            compile(script, "<preflight>", "exec")
        except SyntaxError as exc:
            _fail(f"code task {row.get('task_id')}: assembled test script SyntaxError: {exc}")
        run_hidden_tests(starter, entry_point=row["entry_point"], test_code=row["test"])
    _ok(f"code: sandbox scripts compile for {len(rows)} tasks")


async def check_opensoul() -> None:
    import friction
    import opensoul_document_embedded
    import opensoul_env
    import opensoul_prompt
    import rewards
    import training_utils
    from opensoul_env import OpenSoulSelfReflectionEnv

    rows = [json.loads(l) for l in open("train_dataset.jsonl")]
    env = OpenSoulSelfReflectionEnv()
    check_pickle(env, label="opensoul")
    await _check_init_rollout(
        env,
        OpenSoulSelfReflectionEnv.dataset_preprocess,
        rows,
        label="opensoul",
        sample_tool="seek_pushback",
    )
    # Bundling: every imported local module must be listed in run.py local_modules
    for mod in (
        opensoul_env,
        opensoul_prompt,
        opensoul_document_embedded,
        rewards,
        friction,
        training_utils,
    ):
        cloudpickle.dumps(mod)
    _ok("opensoul: local_modules importable")


async def check_agentic() -> None:
    import agentic_env
    import agentic_rewards
    import agentic_tools
    import agentic_workspace
    import rewards
    import tool_call_helpers
    import training_utils
    from agentic_env import AgenticCapabilitiesEnv

    rows = [json.loads(l) for l in open("agentic_train_dataset.jsonl")]
    env = AgenticCapabilitiesEnv()
    check_pickle(env, label="agentic")

    # Full sweep: every task must seed workspace via init_rollout
    for i, row in enumerate(rows):
        ex = AgenticCapabilitiesEnv.dataset_preprocess(row)
        init_args = ex.get("init_rollout_args") or {}
        if "task" in init_args:
            _fail(f"agentic row {i}: nested 'task' in init_rollout_args (double-task bug)")
        rid = f"preflight-agentic-{i}"
        await env.init_rollout(rid, **init_args)
        ws = env._workspaces.get(rid)
        if ws is None:
            _fail(f"agentic row {i}: workspace missing after init_rollout")
        expected_files = (row.get("workspace") or {}).get("files") or {}
        for path in expected_files:
            if path not in ws.files:
                _fail(f"agentic row {i}: file {path!r} not seeded in workspace")

    _ok(f"agentic: init_rollout seeds workspace for all {len(rows)} tasks")
    for mod in (
        agentic_env,
        agentic_rewards,
        agentic_tools,
        agentic_workspace,
        rewards,
        tool_call_helpers,
        training_utils,
    ):
        cloudpickle.dumps(mod)
    _ok("agentic: local_modules importable")


async def check_code() -> None:
    import code_env
    import code_rewards
    import code_sandbox
    import training_utils
    from code_env import CodeCapabilitiesEnv

    rows = [json.loads(l) for l in open("code_train_dataset.jsonl")]
    env = CodeCapabilitiesEnv()
    check_pickle(env, label="code")
    tools = await env.list_tools()
    if tools:
        _fail("code env should expose zero tools")
    result = await env.run_tool("x", "noop")
    if "no tools" not in str(result).lower():
        _fail("code env run_tool stub missing")
    _ok("code: list_tools/run_tool stubs OK")
    await check_code_sandbox(rows)
    for mod in (code_env, code_rewards, code_sandbox, training_utils):
        cloudpickle.dumps(mod)
    _ok("code: local_modules importable")


async def run_preflight(tracks: str = "opensoul,agentic,code") -> None:
    selected = {t.strip() for t in tracks.split(",") if t.strip()}
    print("Preflight checks (init_rollout + pickle + format gates)")
    check_answer_format()
    if "opensoul" in selected:
        await check_opensoul()
    if "agentic" in selected:
        await check_agentic()
    if "code" in selected:
        await check_code()
    print("Preflight passed.")


def main() -> None:
    tracks = sys.argv[1] if len(sys.argv) > 1 else "opensoul,agentic,code"
    asyncio.run(run_preflight(tracks))


if __name__ == "__main__":
    main()
