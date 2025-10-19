"""
emotion_detector.py
Robust emotion detection module.
- Tries to use HuggingFace API / local model based on cfg (not performing heavy imports here).
- Falls back to keyword-based detection.
- Default fallback emotion label is "Neutral" (capitalized) to match label_enforcer.
"""
import os, logging
from typing import Dict

# Simple keyword mapping (lowercase keys)
EMOTION_KEYWORDS = {
    "Happy": ["happy","glad","joy","excited","fun","enjoy","vui","hạnh phúc"],
    "Sad": ["sad","unhappy","depressed","down","tear","cry","buồn","thất vọng"],
    "Angry": ["angry","mad","furious","hate","annoyed","tức giận","giận"],
    "Anxious": ["anxious","nervous","worried","scared","fear","lo lắng","căng thẳng"],
    "Neutral": []
}

def detect_with_keywords(text: str) -> Dict[str, float]:
    t = (text or "").lower()
    best_label = "Neutral"
    best_score = 0.0
    for label, kws in EMOTION_KEYWORDS.items():
        if not kws:
            continue
        count = sum(1 for kw in kws if kw in t)
        score = count / len(kws) if kws else 0.0
        if score > best_score:
            best_score = score
            best_label = label
    if best_score == 0.0:
        # default fallback
        return {"label":"Neutral", "confidence":0.0}
    return {"label": best_label, "confidence": float(best_score)}

def detect_with_hf_api(text: str, cfg: dict) -> Dict[str, float]:
    # Placeholder for potential HF API or local model integration.
    # In this environment we avoid heavy imports; caller will configure to enable real models.
    # Return None to indicate HF detection not available.
    return None

def detect(text: str, cfg: dict=None) -> Dict[str, float]:
    """Detect emotion for given text using configured backends.
    Returns dict: { 'label': <EmotionLabel>, 'confidence': <0..1 float> }
    """
    # Try HF/local model if configured (cfg keys vary across versions)
    try:
        if cfg and cfg.get("huggingface",{}).get("enabled", False):
            # attempt HF/local detection via detect_with_hf_api (implementation optional)
            res = detect_with_hf_api(text, cfg)
            thr = cfg.get("thresholds",{}).get("emotion_confidence", 0.55)
            if res and isinstance(res, dict) and res.get("confidence",0.0) >= thr:
                # ensure Label capitalization consistent
                label = res.get("label") or res.get("emotion") or "Neutral"
                return {"label": str(label), "confidence": float(res.get("confidence",0.0))}
    except Exception as e:
        logging.getLogger(__name__).warning(f"HF/local emotion detection failed: {e}")
    # Fallback to keywords; default label is "Neutral"
    try:
        return detect_with_keywords(text)
    except Exception as e:
        logging.getLogger(__name__).error(f"Keyword fallback failed: {e}")
        return {"label":"Neutral", "confidence":0.0}
