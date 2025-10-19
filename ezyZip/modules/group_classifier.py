"""
group_classifier.py
Module for classifying messages into topic groups.
This module first tries to use Hugging Face API if configured (hf_online).
If HF call is not available or fails, it falls back to a simple keyword-based classifier.
"""

import os, json, logging
from typing import Dict

# Load topic list if present
def load_topics(project_root="."):
    topics_path = os.path.join(project_root, "data", "subjects.json")
    if os.path.exists(topics_path):
        try:
            with open(topics_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # default topics
    return ["Academics","Teachers","Health","Canteen / Food","Mental Health","Sports","Library","Transportation","Safety","Others"]

TOPIC_KEYWORDS = {
    "Academics": ["exam","test","homework","study","grade","lesson","math","science","project"],
    "Teachers": ["teacher","professor","mr.","ms.","miss","sir","madam","taught","teaches","punish","fair"],
    "Health": ["sick","ill","doctor","hospital","fever","headache","pain","hurt","injury"],
    "Canteen / Food": ["canteen","food","lunch","breakfast","eat","meal","taste","smell"],
    "Mental Health": ["stress","anxious","depressed","sad","lonely","counselor","therapy"],
    "Sports": ["sport","football","soccer","basketball","pe class","exercise","gym"],
    "Library": ["library","books","quiet","study in library","borrow"],
    "Transportation": ["bus","ride","late","traffic","transportation","bus stop"],
    "Safety": ["bully","bullying","safe","unsafe","security","danger"],
    "Others": []
}

def classify_with_keywords(text: str) -> Dict[str, float]:
    """Simple keyword-based topic detection. Returns best topic and score (0-1)."""
    t = text.lower()
    best = ("Others", 0.0)
    for topic, kws in TOPIC_KEYWORDS.items():
        if not kws: continue
        count = sum(1 for kw in kws if kw in t)
        score = count / max(len(kws),1)
        if score > best[1]:
            best = (topic, score)
    return {"group": best[0], "confidence": float(best[1])}

# HF API hook: attempts to call project hf client if available
def classify_with_hf_api(text: str, cfg: dict) -> Dict[str, float]:
    """
    Try to use Hugging Face Inference API via modules/hf_api_client if configured.
    Expected to return a parsed dict {"group": label, "confidence": score}.
    """
    try:
        # import local client if present
        from src.models.hf_api_client import send_hf_api_request
    except Exception:
        try:
            from modules.hf_api_client import send_hf_api_request
        except Exception:
            send_hf_api_request = None
    if send_hf_api_request is None:
        raise RuntimeError("HF API client not available")
    # prepare candidate labels from config or default
    candidates = cfg.get("topics") if cfg and cfg.get("topics") else load_topics(WORK_DIR if 'WORK_DIR' in globals() else ".")
    params = {"candidate_labels": candidates}
    resp = send_hf_api_request(text, cfg.get("hf_model", "bert-base-uncased"), task=None, parameters=params, cfg=cfg)
    # resp likely is {'labels':[...], 'scores':[...]} or similar
    if isinstance(resp, dict) and "labels" in resp and "scores" in resp:
        labels = resp.get("labels", [])
        scores = resp.get("scores", [])
        if labels and scores:
            return {"group": labels[0], "confidence": float(scores[0])}
    # fallback parse list/dict style
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        top = max(resp, key=lambda x: x.get("score",0))
        return {"group": top.get("label"), "confidence": float(top.get("score",0))}
    return {"group":"Others","confidence":0.0}

def classify(text: str, cfg: dict=None) -> Dict[str, float]:
    """Main entry point to classify group/topic for a text."""
    # If config asks hf_online, attempt HF API
    try:
        if cfg and cfg.get("mode","hf_online") == "hf_online" and cfg.get("hf_token",""):
            res = classify_with_hf_api(text, cfg)
            # only accept if confidence > threshold if provided
            thr = cfg.get("thresholds",{}).get("topic_confidence", 0.6) if cfg else 0.6
            if res and res.get("confidence",0.0) >= thr:
                return res
    except Exception as e:
        # log and fallback
        logging = __import__("logging")
        logging.getLogger(__name__).warning(f"HF API classification failed: {e}")
    # fallback to keyword rules
    return classify_with_keywords(text)
