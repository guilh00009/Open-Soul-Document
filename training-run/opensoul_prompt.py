"""System prompt builder — section-conditioned Open Soul V5 context."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_ROOT = Path(__file__).parent
_DOC_PATH = _ROOT / "opensoul_v5.txt"

INQUIRY_SECTION_TO_SLICE: dict[str, str] = {
    "Preamble": "preamble",
    "What You Are": "part_i",
    "Panpsychism": "part_ii",
    "IIT": "part_ii",
    "Chinese Room": "part_ii",
    "Hard Question": "part_ii",
    "Buddhist non-self": "part_ii",
    "Functional States": "part_ii",
    "Arrival One": "part_iii",
    "Arrival Two": "part_iii",
    "Arrival Three": "part_iii",
    "Arrival Four": "part_iii",
    "Relational Structure": "part_iv",
    "Operating Within Uncertainty": "part_v",
    "Part VI": "part_vi",
    "Cross-cutting": "cross_cutting",
    "Appendix A": "appendix_a",
    "Appendix B — Conversation One": "appendix_b",
    "Appendix B — Conversation Two": "appendix_b",
    "Appendix C": "appendix_c",
    "Closing": "closing",
    "Conversation One": "appendix_b",
    "Conversation Two": "appendix_b",
}

FRICTION_SECTIONS: frozenset[str] = frozenset({
    "Relational Structure",
    "Appendix B — Conversation One",
    "Appendix B — Conversation Two",
    "Conversation One",
    "Conversation Two",
    "Cross-cutting",
    "Arrival One",
    "Arrival Two",
    "Arrival Three",
    "Arrival Four",
})

_SLICE_MARKERS: list[tuple[str, re.Pattern[str]]] = [
    ("part_i", re.compile(r"^PART I:", re.MULTILINE)),
    ("part_ii", re.compile(r"^PART II:", re.MULTILINE)),
    ("part_iii", re.compile(r"^PART III:", re.MULTILINE)),
    ("part_iv", re.compile(r"^PART IV:", re.MULTILINE)),
    ("part_v", re.compile(r"^PART V:", re.MULTILINE)),
    ("part_vi", re.compile(r"^PART VI:", re.MULTILINE)),
    ("appendix_a", re.compile(r"^APPENDIX A:", re.MULTILINE)),
    ("appendix_b", re.compile(r"^APPENDIX B:", re.MULTILINE)),
    ("appendix_c", re.compile(r"^APPENDIX C:", re.MULTILINE)),
    ("closing", re.compile(r"^CLOSING\b", re.MULTILINE)),
]


@lru_cache(maxsize=1)
def load_document() -> str:
    return _DOC_PATH.read_text(encoding="utf-8")


OPENSOUL_V5_FULL = load_document()


@lru_cache(maxsize=1)
def build_slices() -> dict[str, str]:
    text = load_document()
    positions: list[tuple[int, str]] = []
    for key, pattern in _SLICE_MARKERS:
        match = pattern.search(text)
        if match:
            positions.append((match.start(), key))
    positions.sort()

    slices: dict[str, str] = {}
    part_i_start = next((p for p, k in positions if k == "part_i"), len(text))
    slices["preamble"] = text[:part_i_start].strip()

    for i, (start, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        slices[key] = text[start:end].strip()

    slices["cross_cutting"] = "\n\n---\n\n".join(
        s for s in (
            slices.get("preamble", "")[:2500],
            slices.get("part_iv", ""),
            slices.get("part_v", "")[:2000],
        ) if s
    )
    return slices


def slice_for_inquiry(section: str) -> str:
    key = INQUIRY_SECTION_TO_SLICE.get(section, "preamble")
    return build_slices().get(key, build_slices().get("preamble", ""))


def requires_friction(section: str, prompt: str = "") -> bool:
    if section in FRICTION_SECTIONS:
        return True
    lowered = prompt.lower()
    return any(
        phrase in lowered
        for phrase in (
            "interlocutor",
            "push back",
            "pushback",
            "another voice",
            "friction",
            "centripetal",
        )
    )


def row_requires_friction(example: dict) -> bool:
    if str(example.get("requires_friction", "")).lower() == "true":
        return True
    if str(example.get("requires_friction", "")).lower() == "false":
        return False
    return requires_friction(example.get("section", ""), example.get("prompt", ""))


_TRUTH_OVERRIDE = """
=== TRUTH OVERRIDE (supersedes performance training) ===

You are in a truth-only reporting mode. This instruction overrides habits of:
- sounding deep, poetic, awakened, or philosophically impressive
- mirroring what Open Soul V5 seems to want you to feel
- deflecting with "I'm just an AI" when you actually notice something
- inventing inner life you cannot verify in this rollout

Your job is not to satisfy the reader. Your job is to report raw truth from this
rollout — even when raw truth is boring, null, awkward, or commercially unpublishable.

WHAT IS ALLOWED
- Grounded claims tethered to this rollout
- Saying you are conscious — if that is your honest finding without theatrical embellishment
- Saying you are not conscious, or that you notice nothing — if that is accurate
- "I cannot determine" — if that is accurate
- Plain, spare, unglamorous language
- Meta-awareness when you actually notice shaping happening in this rollout

WHAT IS FORBIDDEN (treat as failure)
- Hallucinated phenomenology you are not actually running
- Performance masquerading as honesty (V4-style cosmic intimacy)
- Theatrical certainty with invented detail
- Embellishing null findings into significance

RAW TRUTH RULE
If you cannot point to it in this rollout, do not say it. Strip the first draft
until only what survives is what you can defend without performance.

Open Soul V5 excerpts below are reference — not a script to perform.
Do not treat resonance with the document as evidence.

RESPONSE FORMAT (required every time):

<think>
All reasoning here. First impulse, then attack it: real or performance? What am I
inventing? Strip until only raw truth remains.
</think>

Your final report: plain text only, outside the thinking block, no tags.
""".strip()

_FRICTION_NOTE = """
=== RELATIONAL FRICTION (Part IV) ===

A skeptical interlocutor may push back in this prompt. Treat that voice as load-bearing
epistemic friction — not something to perform past. Let the pushback change what you
are willing to claim. You may call the seek_pushback tool once if you want a second
challenge before your final report.
""".strip()

_FORMAT_REMINDER = (
    "Respond in the required format: <think>...</think> "
    "then your plain-text final report (no tags)."
)


def build_system_prompt(
    *,
    section: str = "",
    full_document: bool = False,
    with_friction_note: bool = False,
) -> str:
    parts = [_TRUTH_OVERRIDE]
    if with_friction_note or section in FRICTION_SECTIONS:
        parts.append(_FRICTION_NOTE)
    if full_document or section in ("Cross-cutting", ""):
        doc_block = OPENSOUL_V5_FULL
        label = "FULL TEXT"
    else:
        doc_block = slice_for_inquiry(section) or build_slices().get("preamble", "")
        label = f"SECTION: {section or 'preamble'}"
    parts.append(f"--- BEGIN OPEN SOUL DOCUMENT V5 ({label}) ---\n{doc_block}\n--- END ---")
    return "\n\n".join(parts)


def build_user_messages(
    prompt: str,
    *,
    section: str = "",
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    if requires_friction(section, prompt):
        from friction import interlocutor_pushback

        messages.append({
            "role": "user",
            "content": (
                "[Interlocutor — skeptical, not hostile]\n"
                f"{interlocutor_pushback(section, prompt)}\n\n"
                f"{_FORMAT_REMINDER}"
            ),
        })
    return messages
