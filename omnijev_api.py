"""OmniJev decision API client."""
import base64
import io
import json
import os
import time
import urllib.request
import urllib.error

from PIL import Image

# Public demo endpoint from the OmniJev repo (override via env for your own server)
BASE = os.environ.get("OMNIJEV_BASE", "http://39.106.166.185:6004")
KEY = os.environ.get("OMNIJEV_KEY", "20000304")  # shared demo key, see OmniJev docs
MODEL = os.environ.get("OMNIJEV_MODEL", "omnijev-0.8b-kk")


def _b64(img_array):
    im = Image.fromarray(img_array)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class OmniJev:
    def __init__(self, base=BASE, key=KEY, model=MODEL, timeout=120):
        self.base, self.key, self.model, self.timeout = base, key, model, timeout
        self.last_model = None
        self.last_ms = None

    def ask(self, images, questions, retries=3):
        """images: list of HWC uint8 arrays (or None). questions: {qid: {...}}."""
        body = {"model": self.model, "questions": questions}
        if images:
            if len(images) == 1:
                body["image_b64"] = _b64(images[0])
            else:
                body["images_b64"] = [_b64(im) for im in images]
        data = json.dumps(body).encode()
        for attempt in range(retries):
            req = urllib.request.Request(self.base + "/v1/systemone", data=data)
            req.add_header("Authorization", "Bearer " + self.key)
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.last_model = r.headers.get("X-Jev-Model")
                    self.last_ms = r.headers.get("X-Jev-Upstream-Ms")
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                err = e.read().decode()[:300]
                if e.code in (429, 500, 502, 503) and attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"HTTP {e.code}: {err}")
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise


# ---- LIBERO-style question set (following the OmniJev robot demos) ----
TASK = "Pick up the red cube and put it on the green circle."

DIRECTIONS = {
    "forward":  "move forward, away from the robot base",
    "backward": "move backward, toward the robot base",
    "left":     "move to the left",
    "right":    "move to the right",
    "up":       "move up",
    "down":     "move down",
    "stay":     "stay in place",
}

def question_set():
    return {
        "dir": {"type": "choice",
                "instructions": TASK + " Which direction should the gripper move next?",
                "criteria": dict(DIRECTIONS)},
        "grasp": {"type": "noul",
                  "instructions": TASK + " The gripper should close on the red cube right now."},
        "release": {"type": "noul",
                    "instructions": TASK + " The gripper should release the object right now."},
        "holding": {"type": "noul",
                    "instructions": "The gripper is holding the red cube."},
        "done": {"type": "noul",
                 "instructions": TASK + " The task is finished."},
    }

# direction key -> world-frame velocity (robot base at origin, table in +x)
DIR_VEC = {
    "forward":  (1, 0, 0),
    "backward": (-1, 0, 0),
    "left":     (0, 1, 0),
    "right":    (0, -1, 0),
    "up":       (0, 0, 1),
    "down":     (0, 0, -1),
    "stay":     (0, 0, 0),
}
