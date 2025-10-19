"""
semantic_runner.py - example utilities to initialize bert_loader and generate embeddings.
This module is a light adapter: if local model available, uses transformer AutoModel to create mean-pooled embeddings.
If HF API client is returned, uses its feature-extraction inference to get embeddings.
"""
import os, numpy as np
from modules.bert_loader import init_bert

def default_embed_func_from_client(client_info):
    mode = client_info.get("mode")
    if mode == "local":
        tokenizer = client_info["tokenizer"]
        model = client_info["model"]
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        def embed_texts(texts):
            batches = []
            with torch.no_grad():
                for t in texts:
                    enc = tokenizer(t, return_tensors="pt", truncation=True, padding=True)
                    enc = {k:v.to(device) for k,v in enc.items()}
                    out = model(**enc, return_dict=True)
                    last = out.last_hidden_state
                    mask = enc["attention_mask"].unsqueeze(-1)
                    summed = (last * mask).sum(1)
                    counts = mask.sum(1)
                    emb = (summed / counts).cpu().numpy()[0]
                    batches.append(emb)
            return np.vstack(batches)
        return embed_texts
    elif mode == "hf_api":
        client = client_info["client"]
        def embed_texts(texts):
            res = []
            for t in texts:
                out = client(t, {"task":"feature-extraction"})
                import numpy as np
                arr = np.array(out)
                if arr.ndim==3:
                    arr = arr.mean(axis=1)
                res.append(arr[0])
            return np.vstack(res)
        return embed_texts
    else:
        raise RuntimeError("No embedding backend available")
