
"""
group_classifier.py
Module to classify text into topic groups.
Provides parsing utilities to normalize outputs from HF API or local pipeline.
"""
from typing import Dict, Any

def parse_hf_topic_response(response, candidate_labels=None):
    """
    Expecting HF zero-shot response: {'labels': [...], 'scores': [...]}
    Or list of dicts. Normalize to {"group": label, "confidence": score}
    """
    if isinstance(response, dict):
        labels = response.get('labels') or []
        scores = response.get('scores') or []
        if labels and scores:
            return {"group": labels[0], "confidence": float(scores[0])}
    if isinstance(response, list) and len(response)>0 and isinstance(response[0], dict):
        # HF sometimes returns list of label-score dicts
        top = max(response, key=lambda x: x.get('score',0))
        return {"group": top.get('label'), "confidence": float(top.get('score',0))}
    # Unknown format
    return {"group": "unknown", "confidence": 0.0}

def classify_group_standard(response, threshold=0.5):
    parsed = parse_hf_topic_response(response)
    if parsed.get('confidence',0.0) < threshold:
        parsed['group'] = 'uncertain'
    return parsed
