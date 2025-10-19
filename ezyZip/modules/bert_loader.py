"""
bert_loader.py
Offline-first BERT loader with optional Hugging Face Inference API fallback.
Provides functions to initialize tokenizer/model or HF Inference client.
"""

import os

def load_local_model(model_name="bert-base-uncased", cache_dir="./models"):
    """
    Try to load local tokenizer and model using transformers (local_files_only=True).
    Returns a dict with mode 'local' and tokenizer/model objects if successful, else None.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        return {"mode":"local", "tokenizer":tokenizer, "model":model}
    except Exception as e:
        # local load failed
        return None

def init_hf_api_client(repo_id, token, endpoint=None):
    """
    Initialize Hugging Face Inference API client if huggingface_hub is available.
    """
    try:
        from huggingface_hub import InferenceApi
        client = InferenceApi(repo_id=repo_id, token=token, base_url=endpoint or None)
        return {"mode":"hf_api", "client":client}
    except Exception:
        return None

def init_bert(cfg):
    """
    Attempt to initialize local model first (offline). If fails and HF API enabled, init API client.
    Returns a dict describing the mode: 'local', 'hf_api', or 'fallback'.
    """
    model_name = cfg.get("huggingface",{}).get("model","bert-base-uncased")
    cache_dir = cfg.get("huggingface",{}).get("cache_dir","./models")
    offline_only = cfg.get("huggingface",{}).get("offline_only",False)
    # try local
    local = load_local_model(model_name=model_name, cache_dir=cache_dir)
    if local:
        return local
    # if local fails and offline_only not set, try HF API
    if not offline_only and cfg.get("huggingface_api",{}).get("enabled",False):
        token = cfg.get("huggingface_api",{}).get("token") or os.environ.get("HF_API_TOKEN")
        repo = cfg.get("huggingface_api",{}).get("repo_id", model_name)
        api = init_hf_api_client(repo, token, cfg.get("huggingface_api",{}).get("endpoint"))
        if api:
            return api
    # else fallback
    return {"mode":"fallback"}
