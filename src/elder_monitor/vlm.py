import json
import re

import cv2

from .schemas import (
    LYING_IN_BED, LYING_ON_FLOOR, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING,
)

POSTURES = {"lying", "sitting", "standing", "walking", "unclear"}
SUPPORTS = {"bed", "floor", "chair", "none", "unclear"}
PROMPT = (
    "Look only inside the green box in this indoor camera image and ignore anyone outside it. "
    "Is a real person visible inside the box? If none is clearly visible, set person_visible to false. "
    '"support" is the surface directly under that person\'s body. '
    'Reply with JSON only: {"person_visible": true or false, '
    '"posture": "lying" | "sitting" | "standing" | "walking" | "unclear", '
    '"support": "bed" | "floor" | "chair" | "none" | "unclear"}'
)


def parse_answer(text):
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None
    try:
        ans = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(ans.get("person_visible"), bool):
        return None
    if ans.get("posture") not in POSTURES or ans.get("support") not in SUPPORTS:
        return None
    return {k: ans[k] for k in ("person_visible", "posture", "support")}


def answer_state(ans):
    if not ans or not ans["person_visible"]:
        return None
    posture, support = ans["posture"], ans["support"]
    if posture == "lying":
        return {"bed": LYING_IN_BED, "floor": LYING_ON_FLOOR}.get(support)
    if posture == "sitting":
        return {"bed": SITTING_ON_BED, "chair": SITTING_OUTSIDE_BED, "floor": SITTING_OUTSIDE_BED}.get(support)
    return {"standing": STANDING, "walking": WALKING}.get(posture)


class QwenVLM:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        v = cfg["vlm"]
        print(f"loading VLM {v['model']}")
        self.max_side, self.max_new_tokens = v["max_side"], v["max_new_tokens"]
        self.processor = AutoProcessor.from_pretrained(v["model"])
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForImageTextToText.from_pretrained(v["model"], dtype=dtype, device_map="auto")
        self.model.generation_config.temperature = None
        self.revision = f"{v['model']}@{getattr(self.model.config, '_commit_hash', None) or 'unknown'}"

    def ask(self, image_bgr):
        from PIL import Image

        image = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image.thumbnail((self.max_side, self.max_side))
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return parse_answer(reply), reply
