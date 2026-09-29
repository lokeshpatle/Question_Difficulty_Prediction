from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qdp.config.loader import load_config
from qdp.config.schema import validate_config
from qdp.pipeline import QDPTrainer

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--config',default='config/default.yaml'); ap.add_argument('--corpus',default=None); ap.add_argument('--run-dir',default='artifacts/latest'); ap.add_argument('--allow-test-reevaluation', action='store_true'); args=ap.parse_args()
    root=Path(args.root).resolve(); cfg=load_config(root,args.config); validate_config(cfg); result=QDPTrainer(root,cfg,root/args.run_dir,profile=cfg["runtime"].get("dependency_profile","P0")).train(args.corpus, allow_test_reevaluation=args.allow_test_reevaluation); print(result)
if __name__=='__main__': main()
