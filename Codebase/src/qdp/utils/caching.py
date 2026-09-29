from __future__ import annotations

import pickle
from pathlib import Path


def cache_path(root: Path, key: str, suffix: str = ".pkl") -> Path:
    path = Path(root) / f"{key}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_pickle(path: Path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def load_pickle(path: Path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)
