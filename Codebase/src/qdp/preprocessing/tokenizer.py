from __future__ import annotations

import re
from typing import Iterable


_WORD_RE = re.compile(r"\b[\w]+(?:'[\w]+)?\b", flags=re.UNICODE)
_DEFAULT_ABBREVIATIONS = {
    "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "vs.", "etc.", "fig.", "eq.", "no.", "approx.", "inc.", "dept.",
}


def word_tokens_regex(text: str):
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def split_sentences(text: str, abbreviations: Iterable[str] | None = None) -> list[str]:
    """Deterministic fallback sentence segmentation required by the TDD.

    Boundaries occur after . ! ? when followed by whitespace + uppercase/digit/quote,
    plus line breaks. Common abbreviations are protected from false boundaries.
    """
    protected = {str(item).strip().casefold() for item in (abbreviations or _DEFAULT_ABBREVIATIONS)}
    sentences: list[str] = []
    start = 0
    text_len = len(text)
    i = 0
    while i < text_len:
        if text[i] in ".!?":
            j = i + 1
            while j < text_len and text[j].isspace() and text[j] != "\n":
                j += 1
            if j < text_len and text[j] == "\n":
                j += 1
            # Extract the token immediately ending at the punctuation mark.
            k = i
            while k > start and not text[k - 1].isspace():
                k -= 1
            token = text[k : i + 1].casefold()
            next_char = text[j] if j < text_len else ""
            boundary = token not in protected and (j >= text_len or next_char.isupper() or next_char.isdigit() or next_char in '"\'([[')
            if boundary and (j > i + 1 or i + 1 == text_len):
                piece = text[start : i + 1].strip()
                if piece:
                    sentences.append(piece)
                start = j
                i = j
                continue
        if text[i] == "\n":
            piece = text[start:i].strip()
            if piece:
                sentences.append(piece)
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences
