from __future__ import annotations
import json, joblib, os
from pathlib import Path

def atomic_json(path: Path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True,default=str),encoding='utf-8'); tmp.replace(path)

def save_joblib(path,obj):
    tmp=path.with_suffix(path.suffix+'.tmp'); joblib.dump(obj,tmp); tmp.replace(path)

def load_joblib(path): return joblib.load(path)

def load_json(path): return json.loads(path.read_text(encoding='utf-8'))
