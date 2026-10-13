import os
"""Smoke test: StepFun step-1o-turbo-vision answering a grid-cell question."""
import base64
import io
import json
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim import Sim

GRID, IMG = 4, 640
CELL_PX = IMG // GRID
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)


def bold_grid(arr):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=4)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=4)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 10, row * CELL_PX + 8), name, fill=(255, 255, 0),
               font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(im)


def b64_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


KEY = os.environ.get("STEPFUN_API_KEY", "")


def stepfun_ask(img_b64, text, model="step-1o-turbo-vision"):
    payload = {"model": model, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
        {"type": "text", "text": text}]}], "temperature": 0.0}
    req = urllib.request.Request(
        "https://api.stepfun.com/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]


sim = Sim(img_w=IMG, img_h=IMG)
sim.reset(rng=np.random.default_rng(1))
img = bold_grid(sim.render("agentview"))
print("cube true pos:", sim.cube_pos().round(3))
ans = stepfun_ask(b64_png(img),
                  "This robot scene has a labeled 4x4 grid (columns A-D left to right, "
                  "rows 1-4 top to bottom). Which cell contains the red cube? "
                  "Answer with ONLY the cell label, e.g. 'B2'.")
print("stepfun answer:", repr(ans))
ans2 = stepfun_ask(b64_png(img),
                   "Describe what you see in this robot scene in 2-3 sentences.")
print("stepfun describe:", ans2[:300])
