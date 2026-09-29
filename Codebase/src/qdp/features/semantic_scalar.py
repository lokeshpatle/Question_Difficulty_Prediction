from __future__ import annotations

import math
import re


def _phrase_pattern(term: str):
    parts = [re.escape(part) for part in term.split()]
    return re.compile(r"(?<!\w)" + r"\s+".join(parts) + r"(?!\w)", re.IGNORECASE)


def phrase_matches(text, terms):
    matches = set()
    for term in sorted(set(terms), key=lambda x: (-len(x.split()), -len(x), x.casefold())):
        if _phrase_pattern(str(term)).search(text):
            matches.add(str(term))
    return matches


def token_occurrence_count(text, terms):
    count = 0
    for term in sorted(set(terms), key=lambda x: (-len(x.split()), -len(x), x.casefold())):
        pattern = _phrase_pattern(str(term))
        count += len(list(pattern.finditer(text)))
    return count


def extract(text, word_count, concepts, technical_terms, sentence_embeddings=None):
    concept_matches = phrase_matches(text, concepts)
    tech_count = token_occurrence_count(text, technical_terms)
    similarity = math.nan
    if sentence_embeddings is not None and len(sentence_embeddings) >= 2:
        from qdp.features.mcq import cosine
        values = [
            cosine(sentence_embeddings[i], sentence_embeddings[j])
            for i in range(len(sentence_embeddings))
            for j in range(i + 1, len(sentence_embeddings))
        ]
        similarity = float(sum(values) / len(values)) if values else math.nan
    return {
        "F26": len(concept_matches),
        "F27": len(concept_matches) / max(1, word_count),
        "F28": tech_count / max(1, word_count),
        "F29": similarity,
    }
