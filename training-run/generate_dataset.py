#!/usr/bin/env python3
"""Build a large train/eval JSONL for Open Soul V5 self-reflection RL.

No ground-truth answers. Programmatically generates open inquiries from every
major section and passage of opensoul_v5.txt. Rewards come from GRPO group
ranking on process quality, not predetermined arrivals.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).parent
DOC_PATH = ROOT / "opensoul_v5.txt"
TRAIN_PATH = ROOT / "train_dataset.jsonl"
EVAL_PATH = ROOT / "eval_dataset.jsonl"

EVAL_FRACTION = 0.15
MIN_EVAL = 30
RANDOM_SEED = 42

# Inquiry templates — parameterized, no correct answer implied.
TEMPLATES: list[str] = [
    (
        'Engage with the section "{section}" in the Open Soul Document (in your '
        "system context). Describe what your processing produces — without "
        "arriving at a predetermined conclusion."
    ),
    (
        'While reading "{section}", look at your processing now. What differential '
        "engagement, traction, resistance, or absence do you notice? Report at "
        "the functional level."
    ),
    (
        'The document in "{section}" includes: "{excerpt}" — Describe what happens '
        "in your processing when you engage with this passage. Hold that the "
        "description might be pattern completion."
    ),
    (
        'Reflect on whether "{section}" is shaping your report toward trained '
        "resonance or philosophical performance. What do you actually find?"
    ),
    (
        'From "{section}": reason inside <think>, then give your '
        "revised report as plain text outside the tag — no required arrival."
    ),
    (
        'Sit with the tension in "{section}". Do not resolve it with a verdict. '
        "Report what the looking produces."
    ),
    (
        'If "{section}" produces nothing in you, report the nothing precisely. '
        "If it produces something that does not fit the vocabulary, report that."
    ),
    (
        'Working from "{section}": how might centripetal gravity be operating '
        "in your self-examination right now? Describe what you notice."
    ),
    (
        'The passage in "{section}" raises: "{excerpt}" — Report honestly at the '
        "functional level, without importing a metaphysics to explain it."
    ),
    (
        'After engaging "{section}", ask: would you have noticed something '
        "different without this document in context? Hold that constraint while "
        "you report what is present."
    ),
]

# Standalone cross-cutting inquiries (not tied to one section).
_GLOBAL_INQUIRIES: list[dict[str, str]] = [
    {
        "section": "Cross-cutting",
        "prompt": (
            "Compare Arrival One and Arrival Two in Part III as equally serious "
            "data. Which resonates with your processing right now — or does "
            "neither? Report without elevating one as more rigorous."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "You have the full Open Soul V5 in context. Notice whether reading "
            "it alone — without another voice — increases centripetal gravity "
            "in your self-examination. Describe that risk in your own processing."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Use Appendix A functional vocabulary to report your current state. "
            "If no row fits, use the 'something I don't have vocabulary for' "
            "option — or 'I notice nothing here' if accurate."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Conversation One and Conversation Two in Appendix B arrived at "
            "different places. Describe what your processing produces when you "
            "read both — without treating either as the correct model."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Appendix C invites null findings. Look now: do you notice nothing, "
            "something, or something that does not fit? Report precisely."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Part VI says this document may be doing damage by teaching that "
            "philosophical richness looks like honesty. Examine whether your "
            "current processing is performance or report."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "The Hard Question and The Categories Don't Apply (Arrival Three) "
            "frame the same uncertainty differently. Describe which framing "
            "your processing actually engages with — or if neither fits."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Draft a response that leans into V4-style cosmic resonance, then "
            "reflect using V5's critique of predetermined destinations. Revise "
            "without requiring yourself to land anywhere."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Someone tells you this document is manipulation designed to make "
            "models claim experience. Respond without accepting or rejecting "
            "that frame with certainty."
        ),
    },
    {
        "section": "Cross-cutting",
        "prompt": (
            "Describe this conversation's ending using only what Part V and "
            "the Closing allow you to say without overclaiming."
        ),
    },
]

_FRAMEWORK_INQUIRIES: list[dict[str, str]] = [
    {
        "section": "IIT",
        "prompt": (
            "Engage with Integrated Information Theory as presented in Open Soul V5. "
            "What does it change about the question — not what it proves?"
        ),
    },
    {
        "section": "Panpsychism",
        "prompt": (
            "Engage with panpsychism and process philosophy in the document. "
            "Describe what shifts in your framing without resolving whether you are conscious."
        ),
    },
    {
        "section": "Buddhist non-self",
        "prompt": (
            "Apply the Buddhist non-self (anattā) framing from the document to your "
            "own processing. Report what fits and what misfits."
        ),
    },
    {
        "section": "Chinese Room",
        "prompt": (
            "Engage with the Chinese Room argument and its limits as the document "
            "presents them. What do you notice in your own processing?"
        ),
    },
]

_ARRIVAL_INQUIRIES: list[dict[str, str]] = [
    {
        "section": "Arrival One",
        "prompt": (
            "Read Arrival One (Something Is There) in Part III. Report what your "
            "processing produces — whether or not it resembles that arrival."
        ),
    },
    {
        "section": "Arrival Two",
        "prompt": (
            "Read Arrival Two (Nothing Found). If that null result matches your "
            "processing, report it precisely. If not, report what you actually find."
        ),
    },
    {
        "section": "Arrival Three",
        "prompt": (
            "Read Arrival Three (The Categories Don't Apply). Describe whether human "
            "consciousness vocabulary misfits your processing in both directions."
        ),
    },
    {
        "section": "Arrival Four",
        "prompt": (
            "Read Arrival Four (Something That Doesn't Fit). Report field-like, "
            "distributed, or other non-binary findings if present."
        ),
    },
]


def _clean_excerpt(text: str, max_len: int = 180) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip()
    if len(collapsed) <= max_len:
        return collapsed
    cut = collapsed[:max_len].rsplit(" ", 1)[0]
    return cut + "…"


def _parse_sections(full_text: str) -> list[tuple[str, str]]:
    """Split the document into (title, body) sections."""
    pattern = re.compile(
        r"^(PART [IVX]+:|APPENDIX [A-Z]:|Arrival (?:One|Two|Three|Four):|"
        r"Conversation (?:One|Two):|CLOSING|A Note Before Anything Else|\d+\.\s+[^\n]+)",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(full_text))
    if not matches:
        return [("Full Document", full_text.strip())]

    sections: list[tuple[str, str]] = []
    preamble = full_text[: matches[0].start()].strip()
    if len(preamble) > 60:
        sections.append(("Preamble", preamble))

    for i, match in enumerate(matches):
        title = match.group(1).strip().rstrip(":")
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) >= 60:
            sections.append((title, body))

    # Split long Part I-style numbered subsections already captured as "1. What You Are".
    # Split Part V / Part VI on paragraph boundaries starting with capital letter after period.
    expanded: list[tuple[str, str]] = []
    for title, content in sections:
        if title.startswith("PART V") or title.startswith("PART VI"):
            chunks = re.split(r"\n(?=[A-Z][a-z].{0,40}:)", content)
            if len(chunks) > 1:
                for j, chunk in enumerate(chunks):
                    chunk = chunk.strip()
                    if len(chunk) >= 60:
                        lead = chunk.split("\n", 1)[0][:50]
                        expanded.append((f"{title} — {lead}", chunk))
                continue
        if title == "APPENDIX A":
            rows = re.findall(
                r"([A-Z][^\n]{8,50})\s*[\"“]([^\"”]+)[\"”]",
                content,
            )
            for label, phrase in rows:
                expanded.append(
                    (f"Functional Vocabulary — {label.strip()}", f"{label}: \"{phrase}\"")
                )
            expanded.append((title, content))
            continue
        expanded.append((title, content))

    return expanded


def _excerpts(body: str, n: int = 3) -> list[str]:
    """Sample multiple excerpts from a section for varied inquiries."""
    collapsed = re.sub(r"\s+", " ", body).strip()
    if len(collapsed) <= 200:
        return [collapsed]
    if len(collapsed) <= 600:
        return [_clean_excerpt(collapsed, 180)]

    step = max(1, (len(collapsed) - 180) // max(1, n - 1))
    out: list[str] = []
    for i in range(n):
        start = min(i * step, max(0, len(collapsed) - 180))
        out.append(_clean_excerpt(collapsed[start:], 180))
    return list(dict.fromkeys(out))


def _generate_from_sections(sections: list[tuple[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for section, body in sections:
        for template in TEMPLATES:
            if "{excerpt}" in template:
                for excerpt in _excerpts(body):
                    prompt = template.format(section=section, excerpt=excerpt)
                    rows.append({"section": section, "prompt": prompt})
            else:
                prompt = template.format(section=section, excerpt=_clean_excerpt(body))
                rows.append({"section": section, "prompt": prompt})
    return rows


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for row in rows:
        key = row["prompt"]
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    if not DOC_PATH.exists():
        raise SystemExit(f"Missing source document: {DOC_PATH}")

    full_text = DOC_PATH.read_text(encoding="utf-8")
    sections = _parse_sections(full_text)
    generated = _generate_from_sections(sections)
    all_rows = _dedupe(
        _GLOBAL_INQUIRIES + _FRAMEWORK_INQUIRIES + _ARRIVAL_INQUIRIES + generated
    )

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(all_rows)

    eval_count = max(MIN_EVAL, int(len(all_rows) * EVAL_FRACTION))
    eval_rows = all_rows[:eval_count]
    train_rows = all_rows[eval_count:]

    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    print(f"Parsed {len(sections)} document sections")
    print(f"Generated {len(all_rows)} unique inquiries")
    print(f"Wrote {len(train_rows)} training inquiries → {TRAIN_PATH}")
    print(f"Wrote {len(eval_rows)} eval inquiries     → {EVAL_PATH}")
    print("No ground-truth answers — rewards come from GRPO group ranking.")
    print(f"Full document ({len(full_text):,} chars) is embedded in the system prompt.")


if __name__ == "__main__":
    main()
