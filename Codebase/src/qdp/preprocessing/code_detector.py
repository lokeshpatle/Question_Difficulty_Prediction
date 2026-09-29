from __future__ import annotations

import re
import textwrap


def detect_code_spans(raw_text: str, signatures: list[str]):
    lines = raw_text.splitlines()
    spans = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            marker = stripped[:3]
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith(marker):
                j += 1
            if j < len(lines):
                spans.append((i, j + 1, "fenced"))
                i = j + 1
                continue
        if i < len(lines) - 3:
            block = []
            j = i
            while j < len(lines) and (lines[j].startswith("    ") or lines[j].startswith("\t")):
                block.append(lines[j])
                j += 1
            if len(block) >= 4 and any(re.search(pattern, "\n".join(block)) for pattern in signatures):
                spans.append((i, j, "indented"))
                i = j
                continue
        if any(re.search(pattern, line) for pattern in signatures):
            spans.append((i, i + 1, "signature"))
        i += 1

    merged = []
    for a, b, kind in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]), merged[-1][2])
        else:
            merged.append((a, b, kind))

    code = []
    for a, b, kind in merged:
        block = lines[a:b]
        if kind == "fenced":
            block = block[1:-1]  # remove fence markers, retaining only actual code
            if block and not block[0].strip():
                block = block[1:]
        elif kind == "indented":
            block = [textwrap.dedent(item) for item in block]
        code.append("\n".join(block))
    return code, merged
