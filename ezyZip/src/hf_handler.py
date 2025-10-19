from src.emotion_detector import detect_emotion_standard
from src.group_classifier import classify_group_standard

# hf_handler.py
# Responsible for loading Hugging Face pipelines with optional offline caching.
from pathlib import Path
import os

def load_hf_pipeline(task, model_name, cfg, **kwargs):
    """
    Load a Hugging Face pipeline.
    If huggingface.offline_mode is True, try to use local cached model directory first.
    Returns pipeline instance or raises exception if not available.
    """
    # delayed import to avoid hard dependency if not used
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    except Exception as e:
        raise RuntimeError("transformers is required for HF handler") from e

    hf_cfg = cfg.get('huggingface', {})
    offline = hf_cfg.get('offline_mode', False)
    local_dir = hf_cfg.get('local_model_dir', 'models')
    use_auth = hf_cfg.get('hf_api_key', None) or None

    # sanitize model name to local folder name
    safe_name = model_name.replace("/", "_")
    local_path = os.path.join(local_dir, safe_name)

    if offline:
        if os.path.exists(local_path):
            # load from local path
            return pipeline(task, model=local_path, **kwargs)
        else:
            # try to download and save to local_path
            pipe = pipeline(task, model=model_name, use_auth_token=use_auth, **kwargs)
            Path(local_path).mkdir(parents=True, exist_ok=True)
            try:
                pipe.save_pretrained(local_path)
            except Exception:
                # some pipelines may not support save_pretrained; ignore
                pass
            return pipe
    else:
        # online mode
        return pipeline(task, model=model_name, use_auth_token=use_auth, **kwargs)
