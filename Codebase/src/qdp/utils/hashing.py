from __future__ import annotations
from hashlib import sha256
import json
from pathlib import Path

def text_hash(text: str) -> str:
    return sha256(text.encode('utf-8')).hexdigest()

def file_sha256(path: Path) -> str:
    h = sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def stable_json_hash(obj) -> str:
    return sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
