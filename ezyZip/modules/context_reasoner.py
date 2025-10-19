"""
context_reasoner.py
- Contextual reasoning layer that adjusts group/emotion confidence based on semantic similarity
  between a message and cluster centroids (or other messages).
- Uses sentence-transformers for embeddings if available; otherwise falls back to TF-IDF vectors.
- Adjustment formula:
    adjusted = (1 - weight) * original_score + weight * similarity
  but only applied if similarity >= min_similarity_threshold.
"""

from typing import List, Dict, Any
import numpy as np, os

def try_sbert(model_name="all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None

def embed_texts(texts: List[str], model_name="all-MiniLM-L6-v2"):
    model = try_sbert(model_name)
    if model is not None:
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs, "sbert"
    # fallback tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=512)
    X = vec.fit_transform(texts)
    return X.toarray(), "tfidf"

def cosine_sim(a: np.ndarray, b: np.ndarray):
    # handle 1D arrays
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0: return 0.0
        return float(np.dot(a, b) / denom)
    # a is 1D, b is 2D
    if a.ndim == 1 and b.ndim == 2:
        bn = np.linalg.norm(b, axis=1)
        an = np.linalg.norm(a)
        denom = bn * an
        denom[denom==0]=1e-8
        sims = np.dot(b, a) / denom
        return sims.tolist()
    # both 2D
    from sklearn.metrics.pairwise import cosine_similarity
    return cosine_similarity(a, b)

def adjust_scores(final_results: List[Dict[str,Any]], cluster_centers: List[List[float]], texts: List[str],
                  weight: float=0.3, min_similarity_threshold: float=0.6, model_name="all-MiniLM-L6-v2"):
    """
    final_results: list of dict with keys including 'text', 'group_score', 'emotion_score'
    cluster_centers: list of centroid vectors (may be TF-IDF or sbert dims)
    texts: list of original texts corresponding to final_results (for embedding fallback)
    Returns: new list with fields 'group_score_adj', 'emotion_score_adj', 'group_sim', 'emotion_sim'
    """
    # embed cluster centers and texts using same method: try sbert; if not, assume centers are compatible
    sbert = try_sbert(model_name)
    use_method = "sbert" if sbert is not None else "tfidf"
    # embed texts
    if use_method == "sbert":
        text_embs = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        centers = np.array(cluster_centers)
    else:
        # fallback to TF-IDF embedding of texts and approximate centers (recompute centers from texts grouped by cluster if possible)
        from sklearn.feature_extraction.text import TfidfVectorizer
        # sanitize texts: replace None with empty string
        safe_texts = [t if isinstance(t, str) else "" for t in texts]
        vec = TfidfVectorizer(max_features=512)
        try:
            X = vec.fit_transform(safe_texts).toarray()
            text_embs = X
            centers = np.array(cluster_centers) if cluster_centers and len(cluster_centers[0])==X.shape[1] else None
        except Exception:
            # fallback: use zero vectors to avoid crashing (no adjustment will be applied)
            n = len(safe_texts)
            text_embs = np.zeros((n, 128))
            centers = np.array(cluster_centers) if cluster_centers else None

    adjusted = []
    for idx, res in enumerate(final_results):
        text_vec = text_embs[idx] if idx < len(text_embs) else None
        # compute similarity to its cluster center if possible
        cluster_id = res.get("cluster", None)
        group_sim = 0.0
        emotion_sim = 0.0
        if cluster_id is not None and centers is not None and cluster_id < len(centers) and text_vec is not None:
            # similarity between text and cluster center
            group_sim = cosine_sim(text_vec, centers[cluster_id])
            emotion_sim = group_sim  # use same sim for emotion as approximation, can be enhanced
        # adjust scores
        orig_group = float(res.get("group_score", 0.0) or 0.0)
        orig_emotion = float(res.get("emotion_score", 0.0) or 0.0)
        if group_sim >= min_similarity_threshold:
            group_adj = (1.0 - weight) * orig_group + weight * group_sim
        else:
            group_adj = orig_group
        if emotion_sim >= min_similarity_threshold:
            emo_adj = (1.0 - weight) * orig_emotion + weight * emotion_sim
        else:
            emo_adj = orig_emotion
        new = dict(res)
        new.update({
            "group_sim": float(group_sim),
            "emotion_sim": float(emotion_sim),
            "group_score_adj": float(group_adj),
            "emotion_score_adj": float(emo_adj)
        })
        adjusted.append(new)
    return adjusted
