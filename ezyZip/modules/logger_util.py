"""
logger_util.py
Simple logging helper to append prediction logs to a daily file.
"""

import os, json, datetime

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

def ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

def append_log(entry: dict):
    ensure_log_dir()
    date = datetime.datetime.now().strftime("%Y%m%d")
    fname = os.path.join(LOG_DIR, f"prediction_{date}.log")
    with open(fname, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
