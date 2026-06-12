"""Sandboxed Python execution for verifiable code rewards."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SEC = 8.0


def _build_test_script(
    candidate_code: str,
    entry_point: str,
    test_code: str,
) -> str:
    """Assemble a standalone script: candidate + tests + check(entry_point)."""
    parts = [
        candidate_code.strip(),
        test_code.strip(),
        "if __name__ == '__main__':",
        "    try:",
        f"        check({entry_point})",
        "    except Exception as exc:",
        "        print(f'FAIL: {exc}')",
        "        raise SystemExit(1)",
        "    print('PASS')",
        "    raise SystemExit(0)",
    ]
    return "\n\n".join(parts) + "\n"


def run_hidden_tests(
    code: str,
    *,
    entry_point: str,
    test_code: str,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
) -> dict[str, Any]:
    """Execute candidate code against a `check` function test harness.

    Returns dict with passed, error, stdout snippet.
    """
    if not code.strip():
        return {"passed": False, "error": "empty_code", "stdout": ""}

    script = _build_test_script(code, entry_point, test_code)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(script)
        path = Path(f.name)

    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env={
                "PATH": "/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
        )
        passed = proc.returncode == 0
        err = (proc.stderr or "").strip()[:500]
        out = (proc.stdout or "").strip()[:500]
        return {
            "passed": passed,
            "error": err if not passed else "",
            "stdout": out,
        }
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "timeout", "stdout": ""}
    except Exception as exc:
        return {"passed": False, "error": str(exc)[:300], "stdout": ""}
    finally:
        path.unlink(missing_ok=True)


def syntax_ok(code: str) -> bool:
    if not code.strip():
        return False
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
