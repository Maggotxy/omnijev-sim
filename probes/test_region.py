import os
"""Test v1.1 region-box choice format against the served 0.8B API."""
import json
import sys
sys.path.insert(0, "/opt/LIBERO")
import urllib.request
import base64, io
from PIL import Image

img = Image.open("/opt/LIBERO/out_task0_agent.png")
buf = io.BytesIO()
img.save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

def grid(rows=4, cols=4):
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append({"key": "r%dc%d" % (r, c),
                        "region": {"box": [round(c * 1000 / cols), round(r * 1000 / rows),
                                           round((c + 1) * 1000 / cols), round((r + 1) * 1000 / rows)]}})
    return out

body = {
    "model": "omnijev-0.8b-kk",
    "image_b64": b64,
    "questions": {
        "where": {"type": "choice",
                  "instructions": "Which region contains the alphabet soup can?",
                  "options": grid()},
    },
}
req = urllib.request.Request("http://39.106.166.185:6004/v1/systemone",
                             data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + os.environ.get("OMNIJEV_KEY", "20000304"),
                                      "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=60) as r:
        out = json.loads(r.read())
    print("region format ACCEPTED")
    probs = out["where"].get("probabilities", {})
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:5]
    print("top5:", top)
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:300])
