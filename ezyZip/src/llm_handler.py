
# llm_handler.py
# Minimal OpenAI wrapper supporting on/off via config.
import os

def init_openai_client(cfg):
    cfg_llm = cfg.get('llm', {})
    if not cfg_llm.get('use_openai', False):
        return None
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai package required for llm handler") from e
    api_key = cfg_llm.get('openai_api_key') or os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise RuntimeError("OpenAI API key not provided in config or OPENAI_API_KEY env var")
    openai.api_key = api_key
    return openai

def analyze_with_openai(text, cfg, system_prompt=None):
    openai = init_openai_client(cfg)
    if openai is None:
        raise RuntimeError("OpenAI not enabled in config")
    model = cfg.get('llm', {}).get('openai_model', 'gpt-4o-mini')
    messages = [{"role":"system", "content": system_prompt or "You are a helpful assistant."},
                {"role":"user", "content": text}]
    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=cfg.get('llm',{}).get('temperature',0.0))
    return resp['choices'][0]['message']['content']
