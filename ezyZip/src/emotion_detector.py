
"""
emotion_detector.py
Module to detect emotions from text. 
Provides a unified interface for both online (HF API) and offline (local model) usage.
"""
from typing import Dict, Any

# This function expects either:
# - For online API: the HF client returns a list of dicts like [{"label": "...", "score": 0.9}, ...]
# - For local model: a dict with keys 'label' and 'confidence'
def parse_hf_emotion_response(response) -> Dict[str, Any]:
    """
    Parse the HF API or local pipeline response to a standard dict:
    {"label": "<label>", "confidence": 0.0}
    """
    # If response is list of dicts:
    if isinstance(response, list) and len(response) > 0 and isinstance(response[0], dict):
        # pick the top-scoring label if present
        top = max(response, key=lambda x: x.get('score', 0))
        return {"label": top.get('label'), "confidence": float(top.get('score', 0))}
    # If response is dict like {'label': 'POSITIVE', 'score': 0.9}
    if isinstance(response, dict):
        if 'label' in response and 'score' in response:
            return {"label": response['label'], "confidence": float(response['score'])}
        # maybe single-item list in dict form
        for k,v in response.items():
            if isinstance(v, (list,tuple)) and len(v)>0 and isinstance(v[0], dict):
                top = max(v, key=lambda x: x.get('score',0))
                return {"label": top.get('label'), "confidence": float(top.get('score',0))}
    # Unknown format: return uncertain
    return {"label": "unknown", "confidence": 0.0}

def detect_emotion_standard(response, threshold=0.5):
    """
    Convert parsed response to final label with thresholding.
    """
    parsed = parse_hf_emotion_response(response)
    label = parsed.get('label')
    conf = parsed.get('confidence',0.0)
    if conf < threshold:
        return {"label": "uncertain", "confidence": conf}
    return {"label": label, "confidence": conf}
