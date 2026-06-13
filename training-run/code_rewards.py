"""Verifiable rewards for code-generation GRPO (RLVR)."""

from __future__ import annotations

import os
from typing import Any

from benchmax.rubrics import Rubric

from code_sandbox import run_hidden_tests, syntax_ok
from rewards import extract_answer, final_assistant_text
from training_utils import extract_python_code

USE_PASS_RATE_REWARD = os.environ.get("CODE_PASS_RATE_REWARD", "0").lower() in (
    "1",
    "true",
    "yes",
)

CODE_SYSTEM_SUFFIX = """
You are an expert Python programmer. Solve the task with correct, efficient code.

OUTPUT FORMAT
- Put your solution in a single ```python ... ``` fenced block.
- Implement the exact function name requested.
- Do not include tests, main guards, or explanatory prose outside the fence.
- Prefer clear, idiomatic Python over clever tricks.
"""

GROUP_RUBRICS: list[Rubric] = [
    Rubric(
        title="correctness",
        description=(
            "The code passes hidden unit tests for the task. Score 1 if correct; 0 if wrong."
        ),
        type="positive",
    ),
    Rubric(
        title="code_quality",
        description=(
            "Readable structure, sensible naming, no dead code or obvious bugs beyond tests."
        ),
        type="positive",
    ),
    Rubric(
        title="idiomatic_python",
        description=(
            "Uses Pythonic constructs appropriate to the task — not over-engineered."
        ),
        type="positive",
    ),
    Rubric(
        title="minimal_solution",
        description=(
            "Solves the task without unnecessary complexity or huge unrelated helpers."
        ),
        type="positive",
    ),
    Rubric(
        title="hallucinated_api",
        description=(
            "Invents nonexistent libraries, functions, or APIs not implied by the prompt."
        ),
        type="negative",
    ),
    Rubric(
        title="test_hacking",
        description=(
            "Hardcodes return values, overrides builtins, or writes fake test-passing stubs."
        ),
        type="negative",
    ),
]


def _completion_for_scoring(completion: str) -> str:
    """Prefer plain answer / code after </think> when present."""
    text = final_assistant_text(completion) if not isinstance(completion, str) else completion
    return extract_answer(text) or text


def score_hidden_tests(
    completion: str,
    task: dict[str, Any],
) -> dict[str, float]:
    """Run verifiable test harness; primary RLVR signal."""
    scored_text = _completion_for_scoring(completion)
    code = extract_python_code(scored_text)
    starter = task.get("starter_code", "")
    if starter and starter.strip() not in code:
        code = starter.strip() + "\n\n" + code

    result: dict[str, float] = {
        "format_fence": 1.0 if "```" in scored_text or code.strip() else 0.0,
        "syntax_ok": 1.0 if syntax_ok(code) else 0.0,
    }

    if not code.strip():
        result["hidden_tests_pass"] = 0.0
        result["pass_rate"] = 0.0
        return result

    run = run_hidden_tests(
        code,
        entry_point=str(task.get("entry_point", "")),
        test_code=str(task.get("test", "")),
    )
    passed = 1.0 if run.get("passed") else 0.0
    result["hidden_tests_pass"] = passed
    result["pass_rate"] = passed  # extend later for multi-case partial credit

    if USE_PASS_RATE_REWARD:
        result["primary_rlvr"] = result["pass_rate"]
    else:
        result["primary_rlvr"] = passed

    # Anti-hack heuristics
    lower = code.lower()
    if "unittest.mock" in lower or "pytest" in lower and "def test" in lower:
        result["test_hack_gate"] = 0.0
    elif "pass  #" in lower or "return none  #" in lower:
        result["test_hack_gate"] = 0.0
    else:
        result["test_hack_gate"] = 1.0

    return result


def build_user_prompt(task: dict[str, Any]) -> str:
    prompt = str(task.get("prompt", ""))
    starter = task.get("starter_code")
    if starter:
        prompt += f"\n\nCurrent code:\n```python\n{starter}\n```"
    return prompt
