from __future__ import annotations

from collections import Counter


class ValidationError(Exception):
    pass


def validate_training(records, min_records: int, min_class_count: int):
    if len(records) < int(min_records):
        raise ValidationError(f"record_count_below_min:{len(records)}<{min_records}")
    labels = [r.label for r in records]
    if any(l not in {"Easy", "Moderate", "Hard"} for l in labels):
        raise ValidationError("invalid_label")
    counts = Counter(labels)
    missing = [x for x in ("Easy", "Moderate", "Hard") if counts[x] == 0]
    if missing:
        raise ValidationError(f"missing_classes:{missing}")
    low = {k: v for k, v in counts.items() if v < int(min_class_count)}
    if low:
        raise ValidationError(f"class_minimum:{low}")
    ids = [r.question_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValidationError("duplicate_question_id_after_dedup")
    return {"record_count": len(records), "class_counts": dict(counts)}


def validate_splits(records, min_class_count: int = 1) -> None:
    for split in ("train", "validation", "test"):
        rows = [r for r in records if r.split == split]
        counts = Counter(r.label for r in rows)
        missing = [label for label in ("Easy", "Moderate", "Hard") if counts[label] < min_class_count]
        if missing:
            raise ValidationError(f"{split}_split_missing_or_low_classes:{missing}")
