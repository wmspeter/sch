"""
topic_classifier_semantic.py
Semantic topic classifier using embeddings from a BERT model (offline-first via bert_loader).
If BERT not available, it will raise an exception and caller can fallback to keyword-based classifier.
"""
from typing import List, Dict
import numpy as np

def build_label_embeddings(labels: List[str], embed_func):
    # embed label texts and return numpy array
    embs = embed_func(labels)
    return np.array(embs)

def classify_semantic(texts: List[str], labels: List[str], embed_func, top_k=1):
    # embed texts and labels and compute cosine similarity
    import numpy as np
    text_embs = np.array(embed_func(texts))
    label_embs = np.array(embed_func(labels))
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(text_embs, label_embs)
    results = []
    for i, row in enumerate(sims):
        idx = int(row.argmax())
        results.append({"text": texts[i], "label": labels[idx], "score": float(row[idx]), "all_scores": row.tolist()})
    return results
