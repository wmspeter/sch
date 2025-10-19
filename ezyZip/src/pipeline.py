# src/pipeline.py
from src.translator import translate_vi_to_en
from src.topic_classifier import classify_topic
from src.emotion_detector import detect_emotion

def run_pipeline(text):
    """
    Quy trình xử lý chính:
    1. Dịch tiếng Việt sang tiếng Anh (translate_vi_to_en)
    2. Phân loại chủ đề (classify_topic)
    3. Phát hiện cảm xúc (detect_emotion)
    """
    en_text = translate_vi_to_en(text)
    topic = classify_topic(en_text)
    emotion = detect_emotion(en_text)

    return {
        "original_text": text,
        "translated_text": en_text,
        "topic": topic,
        "emotion": emotion
    }
