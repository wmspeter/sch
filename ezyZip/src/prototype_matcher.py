"""
Prototype matcher module: matches input text to stored prototype examples using semantic embeddings (sentence-transformers) or token overlap fallback.
"""


"""
prototype_matcher.py
- Compute semantic similarity between input message and stored prototypes for each topic.
- Prefer sentence-transformers embeddings when available; otherwise fallback to token overlap scoring.
- Exposes function match_prototype(text, prototypes, cfg) -> (best_topic, score, rationale)
"""

from pathlib import Path
import os, json, math
BASE = Path(__file__).resolve().parents[1]

def _token_overlap_score(a, b):
    # Simple bag-of-words overlap normalized by length
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    inter = sa.intersection(sb)
    return len(inter) / max(len(sa), len(sb))

def match_prototype(text, prototypes, cfg=None):
    """
    Try to match text to prototypes.
    If sentence-transformers available, use embeddings and cosine similarity.
    Otherwise, use token overlap heuristic.
    Returns (best_topic, best_score, rationale).
    """
    try:
        # Try to use sentence-transformers
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        # build corpus: one embedding per prototype grouped by topic
        topic_embeds = {}
        for topic, examples in prototypes.items():
            topic_embeds[topic] = model.encode(examples, convert_to_tensor=True)
        text_emb = model.encode([text], convert_to_tensor=True)[0]
        # compute average max similarity per topic
        best_topic = None
        best_score = -1.0
        for topic, embeds in topic_embeds.items():
            # compute cosine similarities between text and each example, then take max
            sims = util.cos_sim(text_emb, embeds).cpu().numpy().tolist()[0]
            max_sim = max(sims) if sims else 0.0
            if max_sim > best_score:
                best_score = float(max_sim)
                best_topic = topic
        rationale = f\"Matched prototype for topic '{best_topic}' with cosine {best_score:.3f}\"
        return best_topic, best_score, rationale
    except Exception as e:
        # fallback token overlap scoring across all prototypes
        best_topic = None
        best_score = 0.0
        for topic, examples in prototypes.items():
            scores = [_token_overlap_score(text, ex) for ex in examples]
            sc = max(scores) if scores else 0.0
            if sc > best_score:
                best_score = sc
                best_topic = topic
        rationale = f\"Fallback token-overlap matched '{best_topic}' with score {best_score:.3f}\"
        return best_topic, best_score, rationale
