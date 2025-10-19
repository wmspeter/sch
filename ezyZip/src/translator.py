
# Translator module with optional HuggingFace pipeline.
from pathlib import Path
import os
BASE = Path(__file__).resolve().parents[1]

def _simple_translate(text):
    mapping = {
        "Em lo lắng về": "I am worried about",
        "Em cảm thấy vui khi có": "I feel happy when we have",
        "tệ quá": "is terrible",
        "Em rất thích": "I love",
        "Em mệt vì": "I am tired because of",
        "Em muốn góp ý về": "I want to complain about",
    }
    out = text
    for k,v in mapping.items():
        out = out.replace(k, v)
    return out

class Translator:
    def __init__(self, cfg=None):
        self.enabled = False
        self.use_hf = False
        if cfg:
            self.enabled = cfg.get("translate_vi_en", False)
            self.use_hf = cfg.get("use_hf_translation", False)
        self.translator = None
        if self.enabled and self.use_hf:
            try:
                # delayed import to avoid heavy dependency if not used
                from transformers import pipeline
                # use Helsinki model for vi->en
                self.translator = pipeline("translation", model="Helsinki-NLP/opus-mt-vi-en")
            except Exception as e:
                print("Warning: HuggingFace translation pipeline not available, falling back. Error:", e)
                self.translator = None

    def translate(self, text):
        if not self.enabled:
            return text
        if self.translator is None:
            return _simple_translate(text)
        try:
            out = self.translator(text, max_length=512)
            if isinstance(out, list) and len(out)>0 and 'translation_text' in out[0]:
                return out[0]['translation_text']
            # fallback
            return out[0].get('generated_text', text)
        except Exception as e:
            print("Translation failed, fallback. Error:", e)
            return _simple_translate(text)

# module-level helper for backward compatibility
_default_translator = None
def get_translator(cfg=None):
    global _default_translator
    if _default_translator is None:
        _default_translator = Translator(cfg)
    return _default_translator

if __name__ == '__main__':
    import yaml
    cfg = yaml.safe_load(open(BASE / 'config' / 'config.yaml'))
    tr = Translator(cfg)
    print(tr.translate("Em lo lắng về kỳ thi"))
