from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdp.artifacts.checksums import verify_checksums
from qdp.artifacts.save_load import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-dir", default="artifacts/latest")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    run = root / args.run_dir
    manifest = load_json(run / "run_manifest.json")
    verify_checksums(run, manifest["artifact_checksums"])
    metrics = load_json(run / "metrics.json")
    if "production" not in metrics or "test" not in metrics["production"]:
        raise RuntimeError("final_test_metrics_missing")
    print(json.dumps({"run_id": metrics.get("selected_run_id"), "dependency_profile": manifest["dependency_profile"], "production": metrics["production"]}, indent=2))


if __name__ == "__main__":
    main()
