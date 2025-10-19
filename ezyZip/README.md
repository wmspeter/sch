# School Health AI - v1.2.1

This project is a demo system for analyzing anonymous student messages to detect topics and emotions.
It includes hybrid topic classification (zero-shot + prototype matching) and hybrid emotion detection
(Hugging Face emotion model with LLM fallback).

## How to run
1. Create virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. (Optional) To enable Hugging Face or OpenAI features, update `config/config.yaml` and `config/thresholds.yaml`.
3. Run the pipeline:
   ```bash
   python main.py
   ```
4. Output report will be in `output/report.pdf` and per-message logs in `data/logs/`.

All configuration and thresholds are stored in `config/` in YAML format.

## v1.2.3 - New Modules
- Added `src/group_classifier.py` and `src/emotion_detector.py`.
- These modules normalize HF API/local pipeline outputs to a common format.
