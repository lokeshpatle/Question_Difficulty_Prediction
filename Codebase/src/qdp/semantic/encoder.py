from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


class SemanticEncoder:
    def __init__(self, model_name: str, cache_dir: Path | None = None, batch_size: int = 64, max_seq_length: int = 256, revision: str | None = None):
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_seq_length = int(max_seq_length)
        self.revision = revision or "configured"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model = None
        self.tokenizer = None

    def load(self):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            raise RuntimeError(f"sentence-transformers unavailable:{exc}") from exc
        kwargs = {
            "cache_folder": str(self.cache_dir) if self.cache_dir else None,
            "local_files_only": True,
        }
        if self.revision and self.revision != "configured":
            kwargs["revision"] = self.revision
        try:
            self.model = SentenceTransformer(self.model_name, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                "semantic_model_not_available_locally:"
                f"model={self.model_name}:revision={self.revision}:error={exc}"
            ) from exc
        if hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = self.max_seq_length
        try:
            self.tokenizer = self.model.tokenizer
        except Exception:
            self.tokenizer = None
        return self

    def truncate_text(self, text: str, limit: int):
        if self.tokenizer is None and self.model is None:
            self.load()
        tokenizer = self.tokenizer
        if tokenizer is None:
            words = text.split()
            return " ".join(words[:limit]), len(words) > limit
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        if len(ids) <= limit:
            return text, False
        truncated_ids = ids[:limit]
        truncated = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        return truncated, True

    def _cache_key(self, question_id: str, role: str, text: str):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        payload = f"{question_id}|{role}|{digest}|{self.model_name}|{self.revision}|{self.max_seq_length}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def encode_units(self, units):
        if not units:
            return np.empty((0, 0), dtype=np.float32)
        if self.model is None:
            self.load()
        cache_dir = self.cache_dir
        cached = [None] * len(units)
        missing = []
        missing_indices = []
        for idx, (qid, role, text) in enumerate(units):
            key = self._cache_key(qid, role, text)
            if cache_dir:
                path = cache_dir / "embeddings" / f"{key}.npy"
                if path.exists():
                    try:
                        cached[idx] = np.load(path, allow_pickle=False)
                        continue
                    except Exception:
                        path.unlink(missing_ok=True)
            truncated, _ = self.truncate_text(text, self.max_seq_length)
            missing.append(truncated)
            missing_indices.append(idx)
        if missing:
            embeddings = self.model.encode(
                missing,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            for local_idx, global_idx in enumerate(missing_indices):
                vector = embeddings[local_idx]
                cached[global_idx] = vector
                if cache_dir:
                    key = self._cache_key(units[global_idx][0], units[global_idx][1], units[global_idx][2])
                    path = cache_dir / "embeddings" / f"{key}.npy"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, vector, allow_pickle=False)
        return np.vstack(cached)

    def encode(self, texts):
        units = [(f"item_{i}", "question", text) for i, text in enumerate(texts)]
        return self.encode_units(units)
