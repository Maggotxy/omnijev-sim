"""Low-frequency visual task planning; no access to the physical World."""
import base64
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from PIL import Image

from .bank import validate_update


def validate_plan(plan, allowed_ids):
    if not isinstance(plan, dict) or plan.get("goal") != "green_pad" or plan.get("stage") not in ("reach_target", "wait_view"):
        raise ValueError("Invalid task-level plan")
    if not isinstance(plan.get("target_visible"), bool):
        raise ValueError("Missing target visibility")
    point = plan.get("target_xy")
    if not isinstance(point, list) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in point):
        raise ValueError("Invalid normalized image target")
    updates = plan.get("reflex_updates", [])
    if not isinstance(updates, list) or len(updates) > 3:
        raise ValueError("Invalid reflex updates")
    for update in updates:
        if not isinstance(update, dict) or set(update)-{"id", "condition", "threshold"}:
            raise ValueError("Brain updates may only clarify a condition or threshold")
        if update.get("id") not in allowed_ids:
            raise ValueError("Brain cannot invent a motor reflex ID")
        validate_update(update)
    return {"goal": "green_pad", "stage": plan["stage"], "target_visible": plan["target_visible"],
            "target_xy": [float(x) for x in point], "reflex_updates": updates}


class VisualBrain:
    def __init__(self, timeout=15, image_dir=None):
        self.base = os.environ.get("GEMINI_BASE", "https://g.92.run").rstrip("/")
        self.key = os.environ.get("GEMINI_API_KEY", "")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
        self.thinking = os.environ.get("GEMINI_THINKING", "low")
        self.timeout = timeout
        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir:
            self.image_dir.mkdir()

    def __call__(self, observation):
        if not self.key:
            raise ValueError("GEMINI_API_KEY is required")
        buffer = io.BytesIO()
        Image.fromarray(observation.frame).save(buffer, "JPEG", quality=85)
        data = buffer.getvalue()
        if self.image_dir:
            (self.image_dir/f"{observation.sequence:06d}.jpg").write_bytes(data)
        context = observation.context or {}
        bank = context["bank"]
        prompt = ("You are the slow visual task planner of a robot. The ONLY task is to reach and hover over the GREEN/TURQUOISE CIRCULAR PAD, keeping gripper open. "
                  "YELLOW block is an obstacle and NEVER a goal. Blue outline is current planned landing point; orange outline is robot gripper. "
                  "OmniJev independently handles immediate reflexes while you think. Do not prescribe torques or immediate evasive actions. "
                  "Return goal green_pad, stage reach_target or wait_view, target_visible, and target_xy as normalized image [x,y] between0and1. "
                  "If view is obscured retain wait_view; unavailable point may be [0.5,0.5] and is ignored. "
                  "Use measured past feedback to clarify at most three existing reflex conditions or thresholds if needed; preserve each reflex's intended meaning. "
                  "Thresholds must be between 0.5 and 0.98; condition text must be 5 to 600 characters. Empty updates are valid. "
                  "Do not infer success of a past reflex solely from force decreasing. "
                  f"Current reflex bank: {json.dumps(bank)}. Retrieved past feedback: {json.dumps(context.get('memories', []))}. "
                  'Return JSON only: {"goal":"green_pad","stage":"reach_target","target_visible":true,"target_xy":[0.5,0.5],"reflex_updates":[]}.')
        schema = {"type": "OBJECT", "properties": {
            "goal": {"type": "STRING", "enum": ["green_pad"]},
            "stage": {"type": "STRING", "enum": ["reach_target", "wait_view"]},
            "target_visible": {"type": "BOOLEAN"},
            "target_xy": {"type": "ARRAY", "items": {"type": "NUMBER"}, "minItems": 2, "maxItems": 2},
            "reflex_updates": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "id": {"type": "STRING", "enum": [r["id"] for r in bank["reflexes"]]},
                "condition": {"type": "STRING"}, "threshold": {"type": "NUMBER"}}, "required": ["id"]}, "maxItems": 3}},
            "required": ["goal", "stage", "target_visible", "target_xy", "reflex_updates"]}
        generation = {"maxOutputTokens": 1024, "responseMimeType": "application/json", "responseSchema": schema,
                      "thinkingConfig": {"thinkingLevel": self.thinking}}
        body = {"contents": [{"role": "user", "parts": [{"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(data).decode()}},
                                                            {"text": prompt}]}], "generationConfig": generation}
        request = urllib.request.Request(self.base+"/v1beta/models/"+self.model+":generateContent", json.dumps(body).encode(),
                                         {"Content-Type": "application/json", "x-goog-api-key": self.key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"HTTP {exc.code}") from None
        text = "".join(p.get("text", "") for c in raw.get("candidates", []) for p in c.get("content", {}).get("parts", []) if not p.get("thought"))
        if not text:
            raise ValueError("Brain returned no task plan")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
        plan = validate_plan(json.loads(text), {r["id"] for r in bank["reflexes"]})
        return {"plan": plan, "response": {"text": text, "modelVersion": raw.get("modelVersion"), "usageMetadata": raw.get("usageMetadata")},
                "requested_model": self.model, "served_model": raw.get("modelVersion"), "image_sha256": hashlib.sha256(data).hexdigest(),
                "bank_version_at_observation": bank["version"], "thinking_level": self.thinking}
