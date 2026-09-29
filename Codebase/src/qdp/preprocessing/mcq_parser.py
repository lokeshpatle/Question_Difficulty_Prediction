from __future__ import annotations

import re


def _marker_value(marker: str):
    raw = marker.strip().lower().strip("()[]")
    raw = re.sub(r"^option\s+", "", raw)
    raw = raw.rstrip(".)")
    if raw.isdigit():
        return ("num", int(raw))
    if len(raw) == 1 and raw.isalpha():
        return ("alpha", ord(raw) - ord("a") + 1)
    return None


def detect_mcq(raw_text: str, patterns: list[str], min_options: int = 2, max_options: int = 12):
    lines = raw_text.splitlines()
    hits = []
    for idx, line in enumerate(lines):
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                marker = match.group(0)
                hits.append((idx, marker, match.end(), line[match.end():].strip()))
                break
    if len(hits) < min_options:
        return {"detected": False, "options": [], "stem": raw_text, "option_start": None, "option_end": None, "option_lines": []}

    chosen = []
    expected = None
    previous_idx = None
    for hit in hits:
        if previous_idx is not None and hit[0] != previous_idx + 1:
            break
        current = _marker_value(hit[1])
        if current is None:
            break
        if expected is None:
            expected = current
        elif current != expected:
            break
        chosen.append(hit)
        expected = (current[0], current[1] + 1)
        previous_idx = hit[0]
        if len(chosen) >= max_options:
            break

    if len(chosen) < min_options:
        return {"detected": False, "options": [], "stem": raw_text, "option_start": None, "option_end": None, "option_lines": []}

    start = chosen[0][0]
    end = chosen[-1][0] + 1
    options = [item[3] for item in chosen]
    stem = "\n".join(lines[:start]).strip()
    return {
        "detected": True,
        "options": options,
        "stem": stem,
        "option_start": start,
        "option_end": end,
        "option_lines": [item[0] for item in chosen],
    }
