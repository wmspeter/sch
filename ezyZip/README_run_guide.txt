HƯỚNG DẪN CHẠY NHANH - School Health AI v1.3.0
--------------------------------------------

1. Yêu cầu:
   - Python 3.8+
   - Cài dependencies: pip install -r requirements.txt

2. Cấu hình:
   - config/config.yaml: điều chỉnh mode: hf_online | hf_offline | openai
   - Nếu dùng hf_online: đặt hf_token trong config hoặc biến môi trường HUGGINGFACE_HUB_TOKEN

3. Chạy demo:
   - python run_demo.py
   - Kết quả: reports/summary_report_v1.3.0.pdf, data/processed_results_v1.3.0.json, logs/prediction_YYYYMMDD.log

4. Thay đổi ngưỡng:
   - Mở config/config.yaml, sửa thresholds.topic_confidence hoặc thresholds.emotion_confidence

5. Ghi chú:
   - Module group_classifier.py và emotion_detector.py sẽ cố gọi HF API nếu mode=hf_online,
     nếu không có token thì sẽ dùng cơ chế nhận diện theo từ khóa (fallback).