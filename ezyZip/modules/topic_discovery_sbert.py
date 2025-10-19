"""
topic_discovery_sbert.py - SBERT-based topic discovery and mapping to fixed labels.
"""
from typing import List
import numpy as np, os

DEFAULT_GROUPS = [
"Academics","Teachers","Classmates","Homework","Exams & Tests","Learning Motivation","School Environment",
"Classroom Facilities","School Cleanliness","Library","Canteen / Food","Health (Physical)",
"Mental Health / Psychology","Sports / Physical Education","Extracurricular Activities","School Events",
"Technology / Devices / Online Learning","Transportation / School Bus","Safety / Bullying","School Management / Rules",
"Family / Home","Parents","Future / Career Orientation","Finance / Allowance","Friends / Relationships","Love / Crush",
"Sleep / Fatigue","Stress / Pressure","Appearance / Body Image","Teachers’ Attitude / Behavior","Peer Pressure",
"School Schedule / Time Table","Homework Load","Holidays / Vacations","Others"
]

DEFAULT_EMOTIONS = [
"Happy","Sad","Angry","Anxious","Excited","Bored","Neutral","Proud","Embarrassed","Lonely",
"Relieved","Frustrated","Hopeful","Fearful","Disappointed","Grateful","Curious","Confused","Tired","Annoyed"
]

def try_load_sbert(model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None

def embed_texts(texts, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    model = try_load_sbert(model_name)
    if model is not None:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False), 'sbert'
    # fallback tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=512)
    X = vec.fit_transform(texts)
    return X.toarray(), 'tfidf'

def cluster_texts(embs, n_clusters=8):
    from sklearn.cluster import KMeans, AgglomerativeClustering
    n = embs.shape[0]
    k = min(max(2, n_clusters), n)
    try:
        km = KMeans(n_clusters=k, random_state=42, n_init=8)
        labs = km.fit_predict(embs)
        centers = km.cluster_centers_
        return labs.tolist(), centers.tolist()
    except Exception:
        ac = AgglomerativeClustering(n_clusters=k)
        labs = ac.fit_predict(embs)
        centers = []
        for i in range(k):
            pts = embs[[j for j in range(len(labs)) if labs[j]==i]]
            centers.append(list(np.mean(pts, axis=0)))
        return labs.tolist(), centers

def map_centers_to_labels(centers, groups=None, emotions=None, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    groups = groups or DEFAULT_GROUPS
    emotions = emotions or DEFAULT_EMOTIONS
    sbert = try_load_sbert(model_name)
    if sbert is not None:
        label_texts = groups + emotions
        label_embs = sbert.encode(label_texts, convert_to_numpy=True, show_progress_bar=False)
        from sklearn.metrics.pairwise import cosine_similarity
        sims = cosine_similarity(centers, label_embs)
        mappings = []
        for i in range(sims.shape[0]):
            row = sims[i]
            g_len = len(groups)
            g_idx = int(row[:g_len].argmax())
            e_idx = int(row[g_len:].argmax())
            mappings.append({
                "cluster": i,
                "suggested_group": groups[g_idx],
                "group_score": float(row[g_idx]),
                "suggested_emotion": emotions[e_idx],
                "emotion_score": float(row[g_len + e_idx])
            })
        return mappings, 'sbert'
    else:
        mappings = []
        for i in range(len(centers)):
            mappings.append({
                "cluster": i,
                "suggested_group": groups[i % len(groups)],
                "group_score": 0.0,
                "suggested_emotion": emotions[i % len(emotions)],
                "emotion_score": 0.0
            })
        return mappings, 'tfidf'

def discover_and_map(texts, n_topics=8, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
    embs, method = embed_texts(texts, model_name=model_name)
    labs, centers = cluster_texts(embs, n_topics)
    mappings, map_method = map_centers_to_labels(centers, groups=DEFAULT_GROUPS, emotions=DEFAULT_EMOTIONS, model_name=model_name)
    return labs, centers, mappings, method, map_method
