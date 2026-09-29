from __future__ import annotations

from pathlib import Path

from qdp.utils.hashing import file_sha256


def checksum_dir(root: Path, exclude=None):
    excluded = set(exclude or ())
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def verify_checksums(root: Path, expected: dict):
    actual = checksum_dir(root, exclude={"artifact_checksums.json", "run_manifest.json"})
    mismatches = []
    for path, digest in expected.items():
        if actual.get(path) != digest:
            mismatches.append(path)
    extra = sorted(set(actual) - set(expected))
    if mismatches or extra:
        raise ValueError(f"artifact_checksum_mismatch:mismatched={mismatches}:extra={extra}")
    return True
