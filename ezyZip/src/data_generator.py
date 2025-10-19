
import random, json, os
from pathlib import Path
from .utils import save_json, now_str

BASE = Path(__file__).resolve().parents[1]
TOPICS = json.load(open(BASE / 'config' / 'topics.json', 'r', encoding='utf-8'))

# Templates for generating messages (EN and simple VI)
TEMPLATES_EN = [
    "I am worried about the upcoming {topic}.",
    "I feel happy when we have {topic} class.",
    "The {topic} in our school is terrible, I'm disappointed.",
    "I love the {topic}, it makes me excited.",
    "I am tired because of too much {topic}.",
    "Our {topic} makes me feel nervous.",
    "I want to complain about the {topic}.",
    "Can the school improve the {topic}?",
    "I feel lonely since my {topic} changed.",
    "I'm proud of my achievement in {topic}."
]

TEMPLATES_VI = [
    "Em lo lắng về {topic}.",
    "Em cảm thấy vui khi có {topic}.",
    "{topic} ở trường mình tệ quá, em thất vọng.",
    "Em rất thích {topic}, cảm thấy hào hứng.",
    "Em mệt vì quá nhiều {topic}.",
    "Em muốn góp ý về {topic}.",
]

def generate_messages(n=30, include_vi=True):
    msgs = []
    for i in range(n):
        topic = random.choice(TOPICS)
        if include_vi and random.random() < 0.25:
            template = random.choice(TEMPLATES_VI)
            text = template.format(topic=topic)
            lang = "vi"
        else:
            template = random.choice(TEMPLATES_EN)
            # simplify topic mention by removing symbols
            tclean = topic.replace("/", " ").replace("&", "and")
            text = template.format(topic=tclean.lower())
            lang = "en"
        msgs.append({
            "id": f"m{i+1:03d}",
            "text": text,
            "lang": lang,
            "created_at": now_str()
        })
    # save
    save_json(msgs, BASE / 'data' / 'messages_generated.json')
    return msgs

if __name__ == '__main__':
    generate_messages(30)
