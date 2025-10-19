
import json, os, sys, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

def load_json(p):
    with open(p, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj, p):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def ensure_dirs():
    for d in ['data/logs', 'output']:
        Path(BASE / d).mkdir(parents=True, exist_ok=True)

def now_str():
    return datetime.datetime.now().isoformat()
