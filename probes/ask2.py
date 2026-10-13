import os
import base64, io, json, sys, urllib.request
from PIL import Image

KEY = os.environ.get("STEPFUN_API_KEY", "")
img = Image.open(sys.argv[1])
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()
payload = {"model": "step-1o-turbo-vision", "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    {"type": "text", "text": "Describe this robot scene: objects, materials, lighting. Rate visual realism 1-10."}]}],
    "temperature": 0.0}
req = urllib.request.Request("https://api.stepfun.com/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=120) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
