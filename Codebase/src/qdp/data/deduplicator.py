from __future__ import annotations

import re
import unicodedata
from collections import defaultdict

from qdp.utils.hashing import text_hash


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", q)).strip().casefold()


def exact_deduplicate(records):
    groups = defaultdict(list)
    for i, record in enumerate(records):
        groups[text_hash(norm(record.question))].append((i, record))
    kept, conflicts, report = [], [], []
    for group_hash, items in sorted(groups.items()):
        labels = {record.label for _, record in items}
        if len(labels) > 1:
            conflicts.append(items)
            report.append({
                "group": group_hash,
                "reason": "duplicate_label_conflict",
                "records": [record.as_dict() for _, record in items],
            })
            continue
        retained = min(items, key=lambda pair: (pair[1].question_id, pair[0]))[1]
        kept.append(retained)
        report.append({"group": group_hash, "reason": "exact_group", "retained": retained.question_id, "size": len(items)})
    kept.sort(key=lambda r: r.question_id)
    return kept, conflicts, report
