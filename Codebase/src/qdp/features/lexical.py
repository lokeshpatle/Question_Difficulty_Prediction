from __future__ import annotations

from collections import Counter
import re


def parser_tokens(doc):
    if doc is None:
        return []
    return [
        token.text
        for token in doc
        if not token.is_punct and not token.is_space and any(ch.isalnum() for ch in token.text)
    ]


def regex_tokens(text):
    return re.findall(r"\b[\w]+(?:'[\w]+)?\b", text, flags=re.UNICODE)


def fit_rare_word_df(token_lists):
    df = Counter()
    for tokens in token_lists:
        df.update(set(str(token).casefold() for token in tokens))
    return dict(df)


def extract(tokens, rare_df=None, rare_threshold=5, n_sentences=1):
    # Public compatibility: accept either a token sequence (production) or raw text (legacy/tests).
    words = regex_tokens(tokens) if isinstance(tokens, str) else list(tokens)
    n = len(words)
    rare_df = rare_df or {}
    rare = sum(1 for token in words if rare_df.get(token.casefold(), 0) < rare_threshold)
    return {
        "F01": n,
        "F02": n / max(1, n_sentences),
        "F03": sum(len(token) for token in words) / max(1, n),
        "F04": len({token.casefold() for token in words}) / max(1, n),
        "F05": rare / max(1, n),
    }
