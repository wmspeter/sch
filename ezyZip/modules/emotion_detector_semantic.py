"""
emotion_detector_semantic.py
Semantic emotion detector using embeddings from a BERT model (offline-first via bert_loader).
Similar to topic_classifier_semantic but for emotion labels.
"""
from typing import List, Dict
import numpy as np

def detect_emotions_semantic(texts: List[str], emotions: List[str], embed_func, top_k=1):
    import numpy as np
    text_embs = np.array(embed_func(texts))
    emotion_embs = np.array(embed_func(emotions))
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(text_embs, emotion_embs)
    results = []
    for i,row in enumerate(sims):
        idx = int(row.argmax())
        results.append({"text": texts[i], "emotion": emotions[idx], "score": float(row[idx]), "all_scores": row.tolist()})
    return results
