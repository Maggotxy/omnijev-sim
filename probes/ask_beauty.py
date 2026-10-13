import os
"""Ask StepFun to describe the beauty frame (proxy visual check)."""
import base64, io, json, urllib.request
from PIL import Image

KEY = os.environ.get("STEPFUN_API_KEY", "")
img = Image.open("out/beauty_check.png")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
payload = {"model": "step-1o-turbo-vision", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    {"type": "text", "text": "Describe this image in detail: scene, objects, materials, lighting, camera angle. Is anything black/broken/invisible?"}]}],
    "temperature": 0.0}
req = urllib.request.Request("https://api.stepfun.com/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
