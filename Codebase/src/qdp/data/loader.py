from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterator

from qdp.data.schema import Record, make_record


class IngestError(Exception):
    """Raised when a dataset-level ingestion gate fails."""


class CorpusLoader:
    def __init__(self, label_map: dict, max_reject_ratio: float = 0.02, max_question_chars: int | None = None):
        self.label_map = {str(k).strip().casefold(): str(v) for k, v in label_map.items()}
        self.max_reject_ratio = float(max_reject_ratio)
        self.max_question_chars = max_question_chars

    def normalize_label(self, value):
        if value is None:
            return None
        return self.label_map.get(str(value).strip().casefold())

    def _iter_objects(self, path: Path, rejects: list[dict]) -> Iterator[tuple[int, object]]:
        try:
            fh = path.open("r", encoding="utf-8", errors="strict")
        except (OSError, UnicodeError) as exc:
            raise IngestError(f"encoding_error:{exc}") from exc
        with fh:
            first_non_ws = None
            while True:
                try:
                    ch = fh.read(1)
                except UnicodeDecodeError as exc:
                    raise IngestError(f"encoding_error:{exc}") from exc
                if not ch:
                    break
                if not ch.isspace():
                    first_non_ws = ch
                    break
            if first_non_ws is None:
                return
            fh.seek(0)
            if first_non_ws == "[":
                try:
                    data = json.load(fh)
                except UnicodeDecodeError as exc:
                    raise IngestError(f"encoding_error:{exc}") from exc
                except json.JSONDecodeError as exc:
                    raise IngestError(f"malformed_json:{exc}") from exc
                if not isinstance(data, list):
                    raise IngestError("malformed_json:root must be array")
                for i, obj in enumerate(data, 1):
                    yield i, obj
                return

            try:
                for line_no, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        yield line_no, json.loads(line)
                    except json.JSONDecodeError:
                        rejects.append({
                            "line": line_no,
                            "reason": "malformed_json",
                            "record": line.rstrip("\n"),
                        })
            except UnicodeDecodeError as exc:
                raise IngestError(f"encoding_error:{exc}") from exc

    def load(
        self,
        path: Path,
        rejection_path: Path | None = None,
        audit_event: Callable | None = None,
    ) -> list[Record]:
        if not path.exists():
            raise IngestError(f"missing_corpus:{path}")
        records: list[Record] = []
        rejects: list[dict] = []
        for line_no, obj in self._iter_objects(path, rejects):
            reason = None
            if not isinstance(obj, dict):
                reason = "malformed_record"
            question = obj.get("question") if isinstance(obj, dict) else None
            label = obj.get("label") if isinstance(obj, dict) else None
            if reason is None and (not isinstance(question, str) or not question.strip()):
                reason = "missing_question"
            if reason is None and self.max_question_chars is not None and len(question) > self.max_question_chars:
                reason = "oversized"
            normalized = self.normalize_label(label) if reason is None else None
            if reason is None and normalized not in {"Easy", "Moderate", "Hard"}:
                reason = "unknown_label"
            if reason:
                item = {"line": line_no, "reason": reason, "record": obj}
                rejects.append(item)
                if audit_event:
                    audit_event(item)
                continue
            qid = obj.get("question_id")
            records.append(make_record(question, normalized, qid))

        total = len(records) + len(rejects)
        reject_ratio = len(rejects) / total if total else 0.0
        if rejection_path is not None:
            rejection_path.parent.mkdir(parents=True, exist_ok=True)
            with rejection_path.open("w", encoding="utf-8") as fh:
                for item in rejects:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        if reject_ratio > self.max_reject_ratio:
            raise IngestError(f"rejection_rate_exceeded:{len(rejects)}/{total}:{reject_ratio:.6f}")
        return records
