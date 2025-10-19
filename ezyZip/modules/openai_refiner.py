"""openai_refiner.py
Module to perform optional OpenAI-based refinement of predicted group and emotion.
Reads prompt templates from openai_prompts/refine_prompt.txt and calls OpenAI ChatCompletion API.
If OpenAI is disabled or API key missing, the module will skip refinement and copy input to output.
"""

import os, json, time
from typing import List

PROMPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "openai_prompts", "refine_prompt.txt")

def load_prompt_template():
    if not os.path.exists(PROMPT_PATH):
        return None
    return open(PROMPT_PATH, "r", encoding="utf-8").read()

def refine_with_openai(input_path: str, output_path: str, allowed_groups: List[str], allowed_emotions: List[str], cfg: dict):
    """Refine predictions using OpenAI. cfg should contain openai.enabled and openai.api_key and openai.model."""
    # If not enabled or no key, skip refinement
    openai_cfg = cfg.get("openai", {}) if cfg else {}
    if not openai_cfg.get("enabled", False):
        print("OpenAI refinement disabled in config; skipping.")
        # copy input to output
        if os.path.exists(input_path):
            with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
                data = json.load(f_in)
                json.dump(data, f_out, ensure_ascii=False, indent=2)
        return

    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY_ALT")
    if not api_key:
        print("OpenAI enabled but no API key configured; skipping refinement.")
        # copy input to output
        if os.path.exists(input_path):
            with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
                data = json.load(f_in)
                json.dump(data, f_out, ensure_ascii=False, indent=2)
        return

    try:
        import openai
    except Exception as e:
        print("openai package not installed; skipping refinement.", e)
        with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
            data = json.load(f_in)
            json.dump(data, f_out, ensure_ascii=False, indent=2)
        return

    # set API key
    openai.api_key = api_key
    model = openai_cfg.get("model", "gpt-4o-mini") or "gpt-4o-mini"

    # load input data
    with open(input_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    prompt_template = load_prompt_template() or "{text}"

    refined = []
    for it in items:
        text = it.get("text", "")
        group = it.get("group_label") or it.get("group") or it.get("suggested_group") or "Others"
        emotion = it.get("emotion_label") or it.get("emotion") or it.get("suggested_emotion") or "Neutral"
        prompt = prompt_template.format(text=text, group=group, emotion=emotion,
                                        allowed_groups=allowed_groups, allowed_emotions=allowed_emotions)
        # call OpenAI (chat completion)
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.0,
                max_tokens=200
            )
            content = resp["choices"][0]["message"]["content"].strip()
            # try parse JSON from content
            try:
                parsed = json.loads(content)
                refined_group = parsed.get("refined_group", group)
                refined_emotion = parsed.get("refined_emotion", emotion)
            except Exception:
                # fallback: do string matching for group and emotion in content
                refined_group = group
                refined_emotion = emotion
                for g in allowed_groups:
                    if g.lower() in content.lower():
                        refined_group = g; break
                for e in allowed_emotions:
                    if e.lower() in content.lower():
                        refined_emotion = e; break
        except Exception as e:
            print("OpenAI call failed for item; skipping refinement for this item:", e)
            refined_group = group
            refined_emotion = emotion

        it["refined_group"] = refined_group
        it["refined_emotion"] = refined_emotion
        refined.append(it)
        # gentle rate limit
        time.sleep(0.12)

    # write output
    with open(output_path, "w", encoding="utf-8") as f_out:
        json.dump(refined, f_out, ensure_ascii=False, indent=2)
    print("OpenAI refinement complete. Saved to:", output_path)
