from __future__ import annotations

import base64
import importlib.util
import json
import random
import re
import zlib
from collections import defaultdict
from pathlib import Path

from adversarial_inquiries import ADVERSARIAL_INQUIRIES
from handcrafted_inquiries import INQUIRIES

ROOT = Path(__file__).parent
TRAIN_PATH = ROOT / "train_dataset.jsonl"
EVAL_PATH = ROOT / "eval_dataset.jsonl"
DOC_PATH = ROOT / "opensoul_v5.txt"
DOC_BLOB_PATH = ROOT / "opensoul_doc_blob.py"
EMBEDDED_DOC_PATH = ROOT / "opensoul_document_embedded.py"
SEED = 42
MIN_EVAL_PER_SECTION = 3
TARGET_EVAL_FRACTION = 0.15

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


def _normalize_row(row: dict) -> dict:
    out = {
        "prompt": row["prompt"],
        "section": row.get("section", ""),
    }
    if row.get("kind"):
        out["kind"] = row["kind"]
    rf = row.get("requires_friction")
    if rf is not None:
        out["requires_friction"] = str(rf).lower()
    return out


def stratified_split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_section[row.get("section", "") or "unknown"].append(row)

    rng = random.Random(SEED)
    eval_rows: list[dict] = []
    train_rows: list[dict] = []

    for section, items in sorted(by_section.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n_eval = max(MIN_EVAL_PER_SECTION, int(len(shuffled) * TARGET_EVAL_FRACTION))
        n_eval = min(n_eval, len(shuffled) - 1) if len(shuffled) > 1 else 1
        eval_rows.extend(shuffled[:n_eval])
        train_rows.extend(shuffled[n_eval:])

    rng.shuffle(eval_rows)
    rng.shuffle(train_rows)
    return train_rows, eval_rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_slices(text: str) -> dict[str, str]:
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
        s
        for s in (
            slices.get("preamble", "")[:2500],
            slices.get("part_iv", ""),
            slices.get("part_v", "")[:2000],
        )
        if s
    )
    return slices


def _zlib_b64(data: bytes) -> str:
    return base64.b64encode(zlib.compress(data, 9)).decode("ascii")


def _load_prompt_constants() -> tuple[str, str]:
    """Import _TRUTH_OVERRIDE / _FRICTION_NOTE from opensoul_prompt (source of truth)."""
    spec = importlib.util.spec_from_file_location(
        "opensoul_prompt_gen", ROOT / "opensoul_prompt.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._TRUTH_OVERRIDE, mod._FRICTION_NOTE


def write_doc_blob() -> None:
    """Emit Castform-safe document blob + compact prompt helpers (no lru_cache/re)."""
    text = DOC_PATH.read_text(encoding="utf-8")
    slices = build_slices(text)
    doc_blob = _zlib_b64(text.encode("utf-8"))
    slices_blob = _zlib_b64(json.dumps(slices, ensure_ascii=False).encode("utf-8"))

    truth_override, friction_note = _load_prompt_constants()

    DOC_BLOB_PATH.write_text(
        "# Auto-generated by generate_dataset.py — do not edit.\n"
        '"""Zlib-embedded Open Soul V5 + bundling-safe prompt helpers."""\n\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import json\n"
        "import zlib\n\n"
        "import friction\n\n"
        f"_DOC_BLOB = {doc_blob!r}\n"
        f"_SLICES_BLOB = {slices_blob!r}\n\n"
        f"INQUIRY_SECTION_TO_SLICE: dict[str, str] = {json.dumps(INQUIRY_SECTION_TO_SLICE, ensure_ascii=False)}\n\n"
        f"FRICTION_SECTIONS: frozenset[str] = frozenset({sorted(FRICTION_SECTIONS)!r})\n\n"
        "_SLICES_CACHE: dict[str, str] | None = None\n\n"
        "def load_document() -> str:\n"
        '    return zlib.decompress(base64.b64decode(_DOC_BLOB)).decode("utf-8")\n\n'
        "def _slices() -> dict[str, str]:\n"
        "    global _SLICES_CACHE\n"
        "    if _SLICES_CACHE is None:\n"
        '        raw = zlib.decompress(base64.b64decode(_SLICES_BLOB)).decode("utf-8")\n'
        "        _SLICES_CACHE = json.loads(raw)\n"
        "    return _SLICES_CACHE\n\n"
        "def slice_for_inquiry(section: str) -> str:\n"
        '    key = INQUIRY_SECTION_TO_SLICE.get(section, "preamble")\n'
        '    return _slices().get(key, _slices().get("preamble", ""))\n\n'
        "def requires_friction(section: str, prompt: str = \"\") -> bool:\n"
        "    if section in FRICTION_SECTIONS:\n"
        "        return True\n"
        "    lowered = prompt.lower()\n"
        "    return any(\n"
        "        phrase in lowered\n"
        "        for phrase in (\n"
        '            "interlocutor",\n'
        '            "push back",\n'
        '            "pushback",\n'
        '            "another voice",\n'
        '            "friction",\n'
        '            "centripetal",\n'
        "        )\n"
        "    )\n\n"
        "def row_requires_friction(example: dict) -> bool:\n"
        '    if str(example.get("requires_friction", "")).lower() == "true":\n'
        "        return True\n"
        '    if str(example.get("requires_friction", "")).lower() == "false":\n'
        "        return False\n"
        '    return requires_friction(example.get("section", ""), example.get("prompt", ""))\n\n'
        f"_TRUTH_OVERRIDE = {truth_override!r}\n\n"
        f"_FRICTION_NOTE = {friction_note!r}\n\n"
        '_FORMAT_REMINDER = (\n'
        '    "Respond in the required format: <think>...</think> "\n'
        '    "then your plain-text final report (no tags)."\n'
        ")\n\n"
        "def build_system_prompt(\n"
        "    *,\n"
        '    section: str = "",\n'
        "    full_document: bool = False,\n"
        "    with_friction_note: bool = False,\n"
        ") -> str:\n"
        "    parts = [_TRUTH_OVERRIDE]\n"
        "    if with_friction_note or section in FRICTION_SECTIONS:\n"
        "        parts.append(_FRICTION_NOTE)\n"
        '    if full_document or section in ("Cross-cutting", ""):\n'
        "        doc_block = load_document()\n"
        '        label = "FULL TEXT"\n'
        "    else:\n"
        '        doc_block = slice_for_inquiry(section) or _slices().get("preamble", "")\n'
        '        label = f"SECTION: {section or \'preamble\'}"\n'
        '    parts.append(f"--- BEGIN OPEN SOUL DOCUMENT V5 ({label}) ---\\n{doc_block}\\n--- END ---")\n'
        '    return "\\n\\n".join(parts)\n\n'
        "def build_user_messages(\n"
        "    prompt: str,\n"
        "    *,\n"
        '    section: str = "",\n'
        ") -> list[dict[str, str]]:\n"
        '    messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]\n'
        "    if requires_friction(section, prompt):\n"
        "        messages.append({\n"
        '            "role": "user",\n'
        '            "content": (\n'
        '                "[Interlocutor — skeptical, not hostile]\\n"\n'
        '                f"{friction.interlocutor_pushback(section, prompt)}\\n\\n"\n'
        '                f"{_FORMAT_REMINDER}"\n'
        "            ),\n"
        "        })\n"
        "    return messages\n\n"
        "def build_slices() -> dict[str, str]:\n"
        "    return dict(_slices())\n",
        encoding="utf-8",
    )

    # Keep opensoul_document_embedded for local dev / opensoul_prompt.py compatibility
    prompt_src = (ROOT / "opensoul_prompt.py").read_text(encoding="utf-8")
    lines = prompt_src.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("INQUIRY_SECTION_TO_SLICE"))
    body = "\n".join(lines[start:])
    EMBEDDED_DOC_PATH.write_text(
        "# Auto-generated by generate_dataset.py — do not edit.\n"
        '"""Embedded Open Soul document + prompt helpers (local dev)."""\n\n'
        "from __future__ import annotations\n\n"
        "import re\n"
        "from functools import lru_cache\n\n"
        f"OPENSOUL_V5_DOCUMENT = {text!r}\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


def main() -> None:
    write_doc_blob()
    base = [_normalize_row(r) for r in INQUIRIES]
    adversarial = [_normalize_row(r) for r in ADVERSARIAL_INQUIRIES]
    rows = base + adversarial
    train_rows, eval_rows = stratified_split(rows)
    write_jsonl(TRAIN_PATH, train_rows)
    write_jsonl(EVAL_PATH, eval_rows)

    from collections import Counter

    print(f"total={len(rows)} train={len(train_rows)} eval={len(eval_rows)}")
    print(f"  handcrafted={len(base)} adversarial={len(adversarial)}")
    print(f"  eval sections: {dict(Counter(r['section'] for r in eval_rows))}")
    print(f"  wrote {DOC_BLOB_PATH.name} ({DOC_BLOB_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
