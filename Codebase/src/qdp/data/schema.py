from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from qdp.preprocessing.normalize import duplicate_norm
from qdp.utils.hashing import text_hash

Label = Literal["Easy", "Moderate", "Hard"]
LABEL_TO_INT = {"Easy": 0, "Moderate": 1, "Hard": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}


@dataclass(frozen=True)
class Record:
    question: str
    label: str | None
    question_id: str
    raw_hash: str
    split: str | None = None
    family_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "label": self.label,
            "question_id": self.question_id,
            "raw_hash": self.raw_hash,
            "split": self.split,
            "family_id": self.family_id,
        }


def make_record(
    question: str,
    label: str | None,
    question_id: str | None = None,
    split: str | None = None,
) -> Record:
    normalized = duplicate_norm(question)
    qh = text_hash(normalized)
    qid = str(question_id) if question_id else qh[:16]
    return Record(
        question=question,
        label=label,
        question_id=qid,
        raw_hash=qh,
        split=split,
        family_id=None,
    )


def with_split(record: Record, split: str, family_id: str | None = None) -> Record:
    return Record(
        question=record.question,
        label=record.label,
        question_id=record.question_id,
        raw_hash=record.raw_hash,
        split=split,
        family_id=family_id if family_id is not None else record.family_id,
    )
