from __future__ import annotations

import re

Q_TYPES = ["What", "Why", "How", "Which", "When", "Where", "YesNo", "Other"]
I_TYPES = ["Define", "Explain", "Calculate", "Compare", "Analyze", "Design", "Prove", "Describe", "Identify", "Other"]
BLOOM = {"Remember": 1, "Understand": 2, "Apply": 3, "Analyze": 4, "Evaluate": 5, "Create": 6}
_AUX = {"is", "are", "was", "were", "do", "does", "did", "can", "could", "will", "would", "should", "has", "have", "had"}
_INTERROGATIVES = ("what", "why", "how", "which", "when", "where")


def question_type(text: str) -> str:
    from qdp.preprocessing.tokenizer import split_sentences
    sentences = split_sentences(text)
    first_question = next((s for s in sentences if s.endswith("?")), sentences[0] if sentences else "")
    first_lower = first_question.casefold()
    for word in _INTERROGATIVES:
        if re.match(rf"^{re.escape(word)}\b", first_lower):
            return word.capitalize()
    if first_lower.endswith("?") and re.match(r"^(?:" + "|".join(sorted(_AUX)) + r")\b", first_lower):
        return "YesNo"
    positions = []
    lower = text.casefold()
    for order, word in enumerate(_INTERROGATIVES):
        match = re.search(rf"\b{re.escape(word)}\b", lower)
        if match:
            positions.append((match.start(), order, word.capitalize()))
    return min(positions)[2] if positions else "Other"


def instruction_type(lemmas, lexicon) -> str:
    normalized = {cls: {str(word).casefold() for word in words} for cls, words in lexicon.items()}
    for lemma in lemmas:
        value = str(lemma).casefold()
        for cls in I_TYPES[:-1]:
            if value in normalized.get(cls, set()):
                return cls
    return "Other"


def cognitive_demand(lemmas, lexicon) -> int:
    normalized = {cls: {str(word).casefold() for word in words} for cls, words in lexicon.items()}
    level = 1
    for lemma in lemmas:
        value = str(lemma).casefold()
        for cls, words in normalized.items():
            if value in words:
                level = max(level, BLOOM[cls])
    return level


def subquestion_count(text: str, option_line_indexes: set[int] | None = None) -> int:
    """Count enumerators only when they form a sequential line-anchored run."""
    option_line_indexes = option_line_indexes or set()
    candidates = []
    pattern = re.compile(r"^\s*(\([ivxlcdm]+\)|\([a-z]\)|\d+[.)])\s+", flags=re.I)
    for line_idx, line in enumerate(text.splitlines()):
        if line_idx in option_line_indexes:
            continue
        match = pattern.match(line)
        if not match:
            continue
        raw = match.group(1).strip().lower().strip("()").rstrip(".)")
        if raw.isdigit():
            kind_value = ("num", int(raw.rstrip('.)')))
        elif len(raw) == 1 and raw.isalpha():
            kind_value = ("alpha", ord(raw) - ord('a') + 1)
        else:
            roman_map = {"i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10}
            if raw not in roman_map:
                continue
            kind_value = ("roman", roman_map[raw])
        candidates.append((line_idx, kind_value))

    counted = 0
    run_start = 0
    while run_start < len(candidates):
        run_end = run_start + 1
        while run_end < len(candidates):
            prev_idx, prev_value = candidates[run_end - 1]
            idx, value = candidates[run_end]
            if idx != prev_idx + 1 or value[0] != prev_value[0] or value[1] != prev_value[1] + 1:
                break
            run_end += 1
        if run_end - run_start >= 2:
            counted += run_end - run_start
        run_start = run_end
    return counted + max(0, text.count("?") - 1)
