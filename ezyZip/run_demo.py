"""
run_demo.py
Quick script to run the v1.3.0 demo:
- loads sample messages
- classifies group and detects emotion for each
- logs results and produces a PDF report with emotion distribution
"""

import os, json, yaml, datetime
from modules.group_classifier import classify
from modules.emotion_detector import detect
from modules.logger_util import append_log
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from collections import Counter

BASE = os.path.dirname(__file__)
DATA_FILE = os.path.join(BASE, "data", "sample_messages_v1.3.0.json")
CONFIG_FILE = os.path.join(BASE, "config", "config.yaml")
REPORT_DIR = os.path.join(BASE, "reports")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR, exist_ok=True)

# load config if present
cfg = {}
if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

# load sample messages
with open(DATA_FILE, "r", encoding="utf-8") as f:
    messages = json.load(f)

results = []
group_counts = Counter()
emotion_counts = Counter()

for i, msg in enumerate(messages, start=1):
    text = msg.get("text_en") or msg.get("text") or msg.get("text_vi") or ""
    group_res = classify(text, cfg)
    emo_res = detect(text, cfg)
    entry = {
        "id": msg.get("id", i),
        "text": text,
        "group": group_res.get("group"),
        "group_confidence": group_res.get("confidence"),
        "emotion": emo_res.get("label"),
        "emotion_confidence": emo_res.get("confidence")
    }
    results.append(entry)
    group_counts[entry["group"]] += 1
    emotion_counts[entry["emotion"]] += 1
    # append to log file
    append_log(entry)

# Save processed results
out_json = os.path.join(BASE, "data", "processed_results_v1.3.0.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# Create report PDF with emotion distribution chart
report_file = os.path.join(REPORT_DIR, f"summary_report_v1.3.0.pdf")
with PdfPages(report_file) as pdf:
    # Page 1: Summary text
    fig1 = plt.figure(figsize=(8.27, 11.69))  # A4
    plt.axis("off")
    total = len(results)
    lines = [
        "School Health AI - v1.3.0 Summary Report",
        f"Date: {datetime.datetime.now().isoformat()}",
        f"Total messages processed: {total}",
        "",
        "Top groups:",
    ]
    for g, c in group_counts.most_common(10):
        lines.append(f"- {g}: {c}")
    lines.append("")
    lines.append("Top emotions:")
    for e, c in emotion_counts.most_common(10):
        lines.append(f"- {e}: {c}")
    txt = "\n".join(lines)
    plt.text(0.01, 0.99, txt, va="top", wrap=True)
    pdf.savefig(fig1); plt.close(fig1)

    # Page 2: Emotion distribution - pie chart
    fig2 = plt.figure(figsize=(8,6))
    labels = list(emotion_counts.keys())
    sizes = [emotion_counts[k] for k in labels]
    if sum(sizes) == 0:
        labels = ["neutral"]; sizes = [1]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Emotion Distribution")
    pdf.savefig(fig2); plt.close(fig2)

    # Page 3: Group distribution - bar chart
    fig3 = plt.figure(figsize=(10,6))
    groups = [g for g,c in group_counts.items()]
    counts = [c for g,c in group_counts.items()]
    plt.bar(groups, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Group / Topic Distribution")
    pdf.savefig(fig3); plt.close(fig3)

print("Report written to", report_file)
print("Logs written to logs/ and processed results saved to data/processed_results_v1.3.0.json")



# --- BEGIN: Topic discovery and automatic labeling ---
print("Running topic discovery and automatic label mapping...")
from modules.topic_discovery import discover_topics
from modules.auto_labeler import map_clusters_to_labels, DEFAULT_GROUPS, DEFAULT_EMOTIONS

# prepare texts from sample messages
texts = [m.get("text_en") or m.get("text_vi") or m.get("text") for m in messages]
# discover clusters (choose up to 10 clusters for 50 messages)
n_topics = min(10, max(2, int(len(texts)/5)))
labs, centers, emb_type = discover_topics(texts, n_topics)
print(f"Discovered {len(set(labs))} clusters using embedding type: {emb_type}")

mappings, method = map_clusters_to_labels(centers, texts, groups=None, emotions=None)
# save auto topics and emotions
auto_topics = {"mappings": mappings, "method": method}
with open(os.path.join(BASE, "data", "auto_topics_v1.3.1.json"), "w", encoding="utf-8") as f:
    json.dump(auto_topics, f, ensure_ascii=False, indent=2)
print("Auto topics saved to data/auto_topics_v1.3.1.json")

# Create simple auto_emotions summary per cluster based on assigned messages + keyword detector
cluster_emotion_summary = {}
from collections import defaultdict, Counter
for idx, lab in enumerate(labs):
    emo = results[idx]["emotion"]
    cluster_emotion_summary.setdefault(lab, []).append(emo)

cluster_emotion_summary = {str(k): dict(Counter(v)) for k,v in cluster_emotion_summary.items()}
with open(os.path.join(BASE, "data", "auto_emotions_v1.3.1.json"), "w", encoding="utf-8") as f:
    json.dump(cluster_emotion_summary, f, ensure_ascii=False, indent=2)
print("Auto emotions saved to data/auto_emotions_v1.3.1.json")
# --- END ---



# --- v1.3.2: topic discovery (SBERT) and final enforced mapping ---
print("Running v1.3.2 discovery + enforce mapping...")
from modules.topic_discovery_sbert import discover_and_map
from modules.label_enforcer import enforce_group_label, enforce_emotion_label
texts = [m.get("text_en") or m.get("text_vi") or m.get("text") for m in messages]
n_topics = min(10, max(2, int(len(texts)/5)))
labs, centers, mappings, emb_method, map_method = discover_and_map(texts, n_topics, model_name=cfg.get('embedding',{}).get('model_name','paraphrase-multilingual-MiniLM-L12-v2'))
print(f"Discovery method: {emb_method}, mapping method: {map_method}")
with open(os.path.join(BASE, "data", "auto_clusters_v1.3.2.json"), "w", encoding="utf-8") as f:
    json.dump({"labs": labs, "mappings": mappings}, f, ensure_ascii=False, indent=2)
final_results = []
from collections import Counter
for i, msg in enumerate(messages):
    cluster = labs[i]
    mapping = next((m for m in mappings if m["cluster"]==cluster), None)
    suggested_group = mapping.get("suggested_group") if mapping else "Others"
    suggested_emotion = mapping.get("suggested_emotion") if mapping else "Neutral"
    group_label = enforce_group_label(suggested_group)
    emotion_label = enforce_emotion_label(suggested_emotion)
    final_results.append({
        "id": msg.get("id", i+1),
        "text": msg.get("text_en") or msg.get("text_vi") or msg.get("text"),
        "cluster": int(cluster),
        "group_label": group_label,
        "group_score": mapping.get("group_score",0.0) if mapping else 0.0,
        "emotion_label": emotion_label,
        "emotion_score": mapping.get("emotion_score",0.0) if mapping else 0.0
    })
with open(os.path.join(BASE, "data", "final_results_v1.3.2.json"), "w", encoding="utf-8") as f:
    json.dump(final_results, f, ensure_ascii=False, indent=2)
print("Saved final_results_v1.3.2.json")



# --- v1.3.3: Contextual Reasoning Adjustment ---
print("Applying contextual reasoning adjustments...")
from modules.context_reasoner import adjust_scores
# load final results and cluster centers if present
final_path = os.path.join(BASE, "data", "final_results_v1.3.2.json")
if os.path.exists(final_path):
    with open(final_path, "r", encoding="utf-8") as f:
        final_results = json.load(f)
else:
    final_results = []
centers_path = os.path.join(BASE, "data", "auto_clusters_v1.3.2.json")
cluster_centers = []
if os.path.exists(centers_path):
    try:
        with open(centers_path, "r", encoding="utf-8") as f:
            ac = json.load(f)
            # ac expected to have 'mappings' and labs maybe; if centers not present, leave blank
            if isinstance(ac, dict) and "mappings" in ac:
                # try to reconstruct centers from mappings if possible (not always available)
                cluster_centers = [m.get("center", []) for m in ac.get("mappings", [])]
    except Exception:
        cluster_centers = []

texts = [m.get("text") for m in messages]
# load thresholds from config if exist
cfg = {}
cfg_path = os.path.join(BASE, "config", "config.yaml")
try:
    import yaml
    if os.path.exists(cfg_path):
        cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
except Exception:
    cfg = {}
weight = cfg.get("contextual_reasoning", {}).get("weight", 0.3)
min_sim = cfg.get("contextual_reasoning", {}).get("min_similarity_threshold", 0.6)
# If cluster_centers are empty, try to use auto_clusters_v1.3.2.json centers if available
# For simplicity, if centers are empty, we will not adjust (leave as is)
adjusted = adjust_scores(final_results, cluster_centers or [], texts, weight=weight, min_similarity_threshold=min_sim, model_name=cfg.get('contextual_reasoning',{}).get('similarity_model','all-MiniLM-L6-v2'))
# save adjusted results
with open(os.path.join(BASE, "data", "final_results_v1.3.3.json"), "w", encoding="utf-8") as f:
    json.dump(adjusted, f, ensure_ascii=False, indent=2)
print("Saved adjusted final_results_v1.3.3.json")

# Optionally regenerate a short report page for adjusted scores
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from collections import Counter
emotion_counts = Counter([r.get("emotion_label","Neutral") for r in adjusted])
group_counts = Counter([r.get("group_label","Others") for r in adjusted])
report_file_adj = os.path.join(BASE, "reports", "summary_report_v1.3.3.pdf")
with PdfPages(report_file_adj) as pdf:
    fig = plt.figure(figsize=(8,6))
    labels = list(emotion_counts.keys())
    sizes = [emotion_counts[l] for l in labels]
    if sum(sizes)==0:
        labels=["Neutral"]; sizes=[1]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Emotion Distribution (Adjusted)")
    pdf.savefig(fig); plt.close(fig)
    fig2 = plt.figure(figsize=(10,6))
    groups = list(group_counts.keys())
    counts = [group_counts[g] for g in groups]
    plt.bar(groups, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Group Distribution (Adjusted)")
    pdf.savefig(fig2); plt.close(fig2)
print("Adjusted report written to", report_file_adj)
# --- end context reasoning ---


# --- v1.3.4: OpenAI refinement (optional) ---
try:
    from modules.openai_refiner import refine_with_openai
    # load allowed labels
    from modules.label_enforcer import GROUPS as ALLOWED_GROUPS, EMOTIONS as ALLOWED_EMOTIONS
    input_path = os.path.join(BASE, "data", "final_results_v1.3.3.json")
    output_path = os.path.join(BASE, "data", "final_results_v1.3.4.json")
    refine_with_openai(input_path, output_path, ALLOWED_GROUPS, ALLOWED_EMOTIONS, cfg)
except Exception as e:
    print("OpenAI refinement step failed or skipped:", e)
# --- end v1.3.4 ---
