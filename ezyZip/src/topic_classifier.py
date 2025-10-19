"""
Topic classifier module with hybrid strategy: zero-shot classification, prototype matching, and keyword fallback.
"""


import re, json
from pathlib import Path
from .utils import load_json
BASE = Path(__file__).resolve().parents[1]

# existing keyword map builder
def build_keyword_map():
    topics = load_json(BASE / 'config' / 'topics.json')
    kw_map = {}
    for t in topics:
        tokens = [w.lower() for w in re.split(r'[\s/,&]+', t) if w]
        kw_map[t] = tokens
    kw_map["Canteen / Food"].extend(["food","canteen","lunch","meal"])
    kw_map["Health (Physical)"].extend(["sick","health","sore","illness","doctor"])
    kw_map["Mental Health / Psychology"].extend(["stress","depress","anxious","lonely","suicide"])
    kw_map["Exams & Tests"].extend(["exam","test","quiz","final","midterm"])
    kw_map["Homework"].extend(["homework","assignment","project","exercise"])
    kw_map["Teachers"].extend(["teacher","professor","mr","ms","mrs","coach"])
    return kw_map

KEYWORD_MAP = build_keyword_map()

# HuggingFace zero-shot wrapper with safe fallback
class HFTopicClassifier:
    def __init__(self, cfg=None):
        self.enabled = False
        self.classifier = None
        self.topics = []
        if cfg:
            self.enabled = cfg.get('use_hf_classification', False)
            if self.enabled:
                try:
                    from transformers import pipeline
                    model = "facebook/bart-large-mnli"
                    if cfg.get("huggingface_multilingual", False):
                        model = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
                    self.classifier = pipeline("zero-shot-classification", model=model)
                except Exception as e:
                    print("Warning: HF zero-shot not available, falling back to keyword. Error:", e)
                    self.classifier = None

    def classify(self, text, topics):
        # topics: list of candidate labels
        self.topics = topics
        if not self.enabled or self.classifier is None:
            return None
        try:
            out = self.classifier(text, topics, multi_label=False)
            # out has labels and scores
            if isinstance(out, dict):
                label = out.get('labels', [None])[0]
                score = out.get('scores', [0.0])[0]
                return label, float(score)
        except Exception as e:
            print("HF classification failed:", e)
        return None

# legacy function kept for compatibility
def classify_topic(text):
    txt = text.lower()
    scores = {}
    for topic, kws in KEYWORD_MAP.items():
        score = 0
        for kw in kws:
            if kw and kw in txt:
                score += 1
        if score>0:
            scores[topic] = score
    if not scores:
        if any(w in txt for w in ['food','canteen','meal']):
            return "Canteen / Food", 0.5
        return "Others", 0.1
    topic = max(scores, key=lambda k: scores[k])
    conf = scores[topic] / sum(scores.values())
    return topic, conf

# convenience function for pipeline to try HF then fallback
def classify_topic_hybrid(text, cfg=None):
    topics = load_json(BASE / 'config' / 'topics.json')
    if cfg and cfg.get('use_hf_classification', False):
        hf = HFTopicClassifier(cfg)
        try:
            res = hf.classify(text, topics)
            if res:
                return res
        except Exception:
            pass
    # fallback
    return classify_topic(text)

if __name__ == '__main__':
    print(classify_topic_hybrid("I am worried about the upcoming exam", {'use_hf_classification': False}))


# --- BEGIN: Prototype matching integration ---
# Try prototype matching as part of hybrid classification.
def classify_topic_with_prototype(text, cfg=None):
    """
    Hybrid topic classification: try HF zero-shot first (if enabled), then prototype matching as fallback.
    Returns (topic, confidence, rationale).
    """
    # Attempt HF zero-shot if enabled via classify_topic_hybrid
    res = None
    try:
        res = classify_topic_hybrid(text, cfg)
    except Exception:
        res = None
    if res and res[1] >= (cfg.get('zero_shot_conf_threshold', 0.45) if cfg else 0.45):
        # HF result is confident enough
        topic, conf = res
        return topic, conf, 'zero-shot'
    # Otherwise try prototype matcher
    try:
        prototypes = json.load(open(BASE / 'config' / 'prototypes.json', 'r', encoding='utf-8'))
        from .prototype_matcher import match_prototype
        ptopic, pscore, pr = match_prototype(text, prototypes, cfg)
        if ptopic and pscore >= (cfg.get('proto_threshold', 0.65) if cfg else 0.65):
            return ptopic, float(pscore), pr + ' (prototype)'
        # fallback to previous keyword-based result
    except Exception:
        pass
    # fallback to legacy keyword classifier
    t, conf = classify_topic(text)
    return t, conf, 'keyword-fallback'
# --- END: Prototype matching integration ---
