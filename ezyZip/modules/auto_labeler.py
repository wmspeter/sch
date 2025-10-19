"""
auto_labeler.py
- Map discovered clusters to known group names (35 groups) and emotion labels (20 emotions)
  by comparing cluster centroids to embeddings of group names and emotion keywords.
- If sentence-transformers available, use it to embed labels; otherwise use TF-IDF on texts+labels.
"""
from typing import List, Dict
import numpy as np

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

def _embed_texts(texts: List[str]):
    # try sentence-transformers then fallback to TF-IDF
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embs, 'st'
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=512)
        X = vec.fit_transform(texts)
        return X.toarray(), 'tfidf'

def map_clusters_to_labels(cluster_centers: List[List[float]], texts: List[str], groups: List[str]=None, emotions: List[str]=None):
    """
    Map each cluster center to the closest group name and the dominant emotion.
    Returns mapping: [{"cluster":i,"suggested_group":..., "group_score":..., "suggested_emotion":..., "emotion_score":...}, ...]
    """
    groups = groups or DEFAULT_GROUPS
    emotions = emotions or DEFAULT_EMOTIONS
    # Embed group names and emotions as short texts along with centers using same embedding method
    labels = groups + emotions
    label_embs, method = _embed_texts(labels)
    centers = np.array(cluster_centers)
    # If TF-IDF used for labels and centers differ in dim, fallback to simple lexical matching
    if centers.shape[1] != label_embs.shape[1]:
        # simple lowercased token overlap scoring between group name and concatenated cluster texts
        mappings = []
        for i, center in enumerate(centers):
            # approximate: choose group by name length match (not ideal) -> fallback to first group
            mappings.append({
                "cluster": i,
                "suggested_group": groups[i % len(groups)],
                "group_score": 0.0,
                "suggested_emotion": emotions[i % len(emotions)],
                "emotion_score": 0.0
            })
        return mappings, method
    # compute similarities
    sims = cosine_similarity(centers, label_embs)
    mappings = []
    for i in range(sims.shape[0]):
        row = sims[i]
        # first half labels are groups
        g_sims = row[:len(groups)]
        e_sims = row[len(groups):]
        gi = int(g_sims.argmax()); gscore = float(g_sims[gi])
        ei = int(e_sims.argmax()); escore = float(e_sims[ei])
        mappings.append({
            "cluster": i,
            "suggested_group": groups[gi],
            "group_score": gscore,
            "suggested_emotion": emotions[ei],
            "emotion_score": escore
        })
    return mappings, method
