"""
topic_discovery.py
- Compute sentence embeddings for input messages (tries sentence-transformers first)
  otherwise falls back to TF-IDF vectors.
- Cluster messages into topic clusters using AgglomerativeClustering or KMeans.
- Export cluster assignments and cluster centroids.
"""
from typing import List, Dict, Any
import os, numpy as np

def _try_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception:
        return None

def embed_messages(texts: List[str]):
    """
    Return numpy array of shape (n_texts, dim).
    Prefer sentence-transformers; fallback to TF-IDF vectors.
    """
    model = _try_sentence_transformers()
    if model is not None:
        embeds = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeds, 'st'
    # fallback TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=512)
    X = vec.fit_transform(texts)
    return X.toarray(), 'tfidf'

def cluster_embeddings(embeds, n_clusters: int = 8):
    """
    Cluster embeddings. Use KMeans when number of points >= n_clusters, otherwise Agglomerative.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    n = embeds.shape[0]
    k = min(max(2, n_clusters), n)  # at least 2, at most n
    if n >= k:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=8)
            labs = km.fit_predict(embeds)
            centers = km.cluster_centers_
            return labs, centers
        except Exception:
            ac = AgglomerativeClustering(n_clusters=k)
            labs = ac.fit_predict(embeds)
            # compute simple centroids
            centers = []
            for i in range(k):
                pts = embeds[labs==i]
                centers.append(np.mean(pts, axis=0))
            return labs, np.vstack(centers)
    else:
        ac = AgglomerativeClustering(n_clusters=k)
        labs = ac.fit_predict(embeds)
        centers = []
        for i in range(k):
            pts = embeds[labs==i]
            centers.append(np.mean(pts, axis=0))
        return labs, np.vstack(centers)

def discover_topics(texts: List[str], n_topics: int = 8):
    """
    Given list of texts, return:
      - cluster_labels: list[int] per text
      - cluster_centers: ndarray (k, dim)
      - emb_type: 'st' or 'tfidf'
    """
    embeds, emb_type = embed_messages(texts)
    labs, centers = cluster_embeddings(embeds, n_clusters=n_topics)
    return labs.tolist(), centers.tolist(), emb_type
