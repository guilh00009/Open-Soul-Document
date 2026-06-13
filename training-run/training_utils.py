"""Shared GRPO training utilities — DEPO, variance gates, code extraction."""

from __future__ import annotations

import math
import re
import statistics
from typing import Any

_CODE_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_python_code(text: str) -> str:
    """Last fenced Python block, or full text if it looks like raw code."""
    if not text:
        return ""
    blocks = _CODE_FENCE_RE.findall(text)
    if blocks:
        return blocks[-1].strip()
    stripped = text.strip()
    if stripped.startswith("def ") or stripped.startswith("import "):
        return stripped
    return ""


def reward_std(values: list[float]) -> float:
    """Population std-dev; 0 when fewer than 2 values or all equal."""
    if len(values) < 2:
        return 0.0
    try:
        return float(statistics.pstdev(values))
    except statistics.StatisticsError:
        return 0.0


def has_group_learning_signal(
    values: list[float],
    *,
    min_std: float = 1e-6,
) -> bool:
    """DEPO-style: True when the group has reward variance worth learning from."""
    if len(values) < 2:
        return False
    if min(values) == max(values):
        return False
    return reward_std(values) >= min_std


def depo_group_scale(
    values: list[float],
    *,
    min_std: float = 1e-6,
    floor: float = 0.0,
) -> float:
    """Scale factor for group-level judge rewards when variance is low."""
    if not has_group_learning_signal(values, min_std=min_std):
        return floor
    # Normalize std into (0, 1] — higher variance → full judge weight
    std = reward_std(values)
    return min(1.0, std / max(min_std * 10, 1e-9))


def scale_merged_rewards(
    merged: dict[str, float],
    scale: float,
    *,
    skip_keys: frozenset[str] | None = None,
) -> dict[str, float]:
    """Multiply judge-derived rubric scores by DEPO scale; leave gates untouched."""
    skip = skip_keys or frozenset()
    out = dict(merged)
    for key, val in merged.items():
        if key in skip:
            continue
        if isinstance(val, (int, float)) and math.isfinite(val):
            out[key] = float(val) * scale
    return out


def primary_reward_vector(reward_dicts: list[dict[str, float]], key: str) -> list[float]:
    return [float(d.get(key, 0.0)) for d in reward_dicts]


def annotate_group_signal(
    reward_dicts: list[dict[str, float]],
    *,
    primary_key: str,
    min_std: float = 1e-6,
) -> None:
    """In-place: add group_learning_signal and depo_scale to each rollout reward."""
    primaries = primary_reward_vector(reward_dicts, primary_key)
    signal = 1.0 if has_group_learning_signal(primaries, min_std=min_std) else 0.0
    scale = depo_group_scale(primaries, min_std=min_std)
    for d in reward_dicts:
        d["group_learning_signal"] = signal
        d["depo_scale"] = scale


def merge_reward_dicts(*dicts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for d in dicts:
        merged.update(d)
    return merged


def difficulty_band(pass_rate: float) -> str:
    """Bucket pass@k rate for curriculum tagging."""
    if pass_rate <= 0.05:
        return "too_hard"
    if pass_rate < 0.15:
        return "hard"
    if pass_rate <= 0.85:
        return "sweet_spot"
    if pass_rate < 0.98:
        return "easy"
    return "too_easy"


def filter_by_difficulty(
    rows: list[dict[str, Any]],
    *,
    pass_rates: dict[str, float],
    keep_bands: frozenset[str] = frozenset({"hard", "sweet_spot", "easy"}),
    id_key: str = "task_id",
) -> list[dict[str, Any]]:
    """DEPO / curriculum: keep rows whose estimated pass_rate is in keep_bands."""
    out: list[dict[str, Any]] = []
    for row in rows:
        tid = str(row.get(id_key, row.get("prompt", ""))[:80])
        rate = pass_rates.get(tid, pass_rates.get(row.get("task_id", ""), 0.5))
        if difficulty_band(rate) in keep_bands:
            out.append(row)
    return out
