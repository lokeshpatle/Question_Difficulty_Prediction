import json, logging
from pathlib import Path

def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger('qdp')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(run_dir/'run.log', encoding='utf-8')
    sh = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger

def write_event(path: Path, stage: str, severity: str, reason: str, question_id: str | None = None, **extra):
    rec = {'stage': stage, 'severity': severity, 'reason': reason}
    if question_id is not None: rec['question_id'] = question_id
    rec.update(extra)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
