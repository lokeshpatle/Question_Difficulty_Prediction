from __future__ import annotations

import re
from dataclasses import dataclass

from qdp.preprocessing.code_detector import detect_code_spans
from qdp.preprocessing.normalize import normalize_working_text


@dataclass
class TextViews:
    raw_text: str
    code_spans: list[str]
    code_ranges: list[tuple[int, int]]
    option_block: list[str]
    option_start: int | None
    option_end: int | None
    nl_view: str
    nl_stem: str
    sentences: list[str]
    semantic_truncated: bool
    code_detection_tiers: list[tuple[int, int, str]]
    mcq_detection_tier: str | None


def mask_math(text: str) -> str:
    return re.sub(r"\\\(.*?\\\)|\\\[.*?\\\]|\$[^$]+\$", " MATHEXPR ", text, flags=re.DOTALL)


def _mask_code_lines(raw_text: str, ranges):
    lines = raw_text.splitlines()
    keep = []
    for start, end, _ in ranges:
        for idx in range(start, min(end, len(lines))):
            keep.append(idx)
    out = []
    for idx, line in enumerate(lines):
        out.append(" " if idx in set(keep) else line)
    return "\n".join(out)


def _truncate_for_semantic(text: str, encoder, limit: int):
    if encoder is None:
        words = text.split()
        return " ".join(words[:limit]), len(words) > limit
    return encoder.truncate_text(text, limit)


def make_views(raw_text, signatures, mcq_detector, max_chars=20000, semantic_piece_limit=256, encoder=None, sentence_abbreviations=None):
    raw_text = normalize_working_text(raw_text)
    if len(raw_text) > max_chars:
        raise ValueError("oversized")
    codes, code_ranges = detect_code_spans(raw_text, signatures)
    masked = _mask_code_lines(raw_text, code_ranges)
    masked = mask_math(masked)
    mcq = mcq_detector(raw_text)
    options = mcq["options"] if mcq["detected"] else []
    option_start = mcq.get("option_start")
    option_end = mcq.get("option_end")
    # Remove the option block from the already masked natural-language view.
    masked_lines = masked.splitlines()
    if option_start is not None and option_end is not None:
        stem_lines = [line for idx, line in enumerate(masked_lines) if not (option_start <= idx < option_end)]
        nl_stem = "\n".join(stem_lines).strip()
    else:
        nl_stem = masked.strip()
    from qdp.preprocessing.tokenizer import split_sentences
    sentences = split_sentences(masked, sentence_abbreviations)
    _, semantic_truncated = _truncate_for_semantic(masked, encoder, semantic_piece_limit)
    return TextViews(
        raw_text=raw_text,
        code_spans=codes,
        code_ranges=[(a, b) for a, b, _ in code_ranges],
        option_block=options,
        option_start=option_start,
        option_end=option_end,
        nl_view=masked,
        nl_stem=nl_stem,
        sentences=sentences,
        semantic_truncated=semantic_truncated,
        code_detection_tiers=code_ranges,
        mcq_detection_tier=("sequential_markers" if mcq["detected"] else None),
    )
