"""Check probe images: pixel stats locally, then ask OmniJev what it sees."""
import numpy as np
from PIL import Image
from omnijev_api import OmniJev

for name in ("probe_agent.png", "probe_wrist.png"):
    a = np.array(Image.open(name).convert("RGB"))
    print(name, a.shape,
          "mean RGB:", a.mean(axis=(0, 1)).round(1),
          "red px:", int(((a[:, :, 0] > 150) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)).sum()),
          "green px:", int(((a[:, :, 1] > 120) & (a[:, :, 0] < 100) & (a[:, :, 2] < 100)).sum()),
          "blue px:", int(((a[:, :, 2] > 120) & (a[:, :, 0] < 100) & (a[:, :, 1] < 120)).sum()))

api = OmniJev()
a = np.array(Image.open("probe_agent.png").convert("RGB"))
w = np.array(Image.open("probe_wrist.png").convert("RGB"))
ans = api.ask([a, w], {
    "obj": {"type": "choice", "instructions": "What objects are on the wooden table?",
            "criteria": {"red cube": "a red cube", "blue cube": "a blue cube",
                         "robot arm": "a robot arm", "green circle": "a green circle"}},
    "arm": {"type": "noul", "instructions": "There is a robot arm in the scene."},
    "red": {"type": "noul", "instructions": "There is a red cube on the table."},
    "green": {"type": "noul", "instructions": "There is a green circle marker on the table."},
})
import json
print(json.dumps(ans, indent=1))
print("X-Jev-Model:", api.last_model)
