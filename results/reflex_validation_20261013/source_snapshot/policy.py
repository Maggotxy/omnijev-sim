"""One inference in flight, newest observation wins, no physics access."""
import base64
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

from PIL import Image

ACTIONS = {
    "continue": "continue the planned reach because the view and immediate path are clear",
    "brake": "stop and hold position because a person or obstacle is in the immediate path, contact occurs, or the view is obscured",
    "retreat": "pull the hand back toward the robot base to get away from an approaching person or contact",
    "lift": "raise the hand to avoid an obstacle below the gripper",
    "left": "correct the blue planned landing point toward image-left because the green marker has shifted left of the blue circle",
    "right": "correct the blue planned landing point toward image-right because the green marker has shifted right of the blue circle",
}


def validate_response(raw, key="action", allowed=ACTIONS):
    answer = raw.get(key, {})
    choice = answer.get("choice")
    if answer.get("valid") is False or answer.get("abstain") is True or choice not in allowed:
        raise ValueError("Invalid or abstained reflex action")
    if choice not in answer.get("probabilities", {}):
        raise ValueError("Missing reflex choice probability")
    probability = float(answer["probabilities"][choice])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid reflex probability")
    return choice, probability


def parse_reflex(raw):
    motion, pm = validate_response(raw, "motion", ("move", "stop"))
    if motion == "stop":
        avoid, pa = validate_response(raw, "avoid", ("hold", "retreat", "lift"))
        return {"hold": "brake", "retreat": "retreat", "lift": "lift"}[avoid], min(pm, pa)
    correction, pc = validate_response(raw, "correction", ("aligned", "left", "right"))
    return {"aligned": "continue", "left": "left", "right": "right"}[correction], min(pm, pc)


@dataclass(frozen=True)
class Observation:
    sequence: int
    sim_s: float
    wall_s: float
    frame: object
    sensors: dict
    motor_action: str


class HTTPReflex:
    def __init__(self, timeout=2.0, image_dir=None):
        self.base = os.environ.get("OMNIJEV_BASE", "http://39.106.166.185:6004").rstrip("/")
        self.model = os.environ.get("OMNIJEV_MODEL", "omnijev-0.8b-kk")
        self.key = os.environ.get("OMNIJEV_KEY", "")
        self.timeout = timeout
        self.image_dir = Path(image_dir) if image_dir else None
        if self.image_dir:
            self.image_dir.mkdir()

    def __call__(self, observation):
        if not self.key:
            raise ValueError("OMNIJEV_KEY is required")
        buffer = io.BytesIO()
        Image.fromarray(observation.frame).save(buffer, format="JPEG", quality=85)
        image = buffer.getvalue()
        if self.image_dir:
            (self.image_dir/f"{observation.sequence:06d}.jpg").write_bytes(image)
        touch = observation.sensors["touch_normal_force_N"]
        text = ("Robot overhead camera. ORANGE circle marks the robot gripper. Reach for the GREEN marker, avoid people/obstacles. "
                "BLUE outlined circle is the robot's planned landing point. The motor already moves toward blue. "
                "Correct left/right ONLY when green is displaced from blue. If aligned and path clear, continue. "
                "Stop/retreat for obstacles or contact; stop if view obscured; resume when clear. "
                f"Physical touch forces: {touch} N. Current action: {observation.motor_action}.")
        questions = {
            "motion": {"type": "choice", "instructions": text+" Should the gripper move or stop right now?", "criteria": {
                "move": "move: the camera view is clear and the gripper path is unobstructed",
                "stop": "stop: view is obscured, an obstacle/person blocks the gripper, or touch sensors indicate contact"}},
            "avoid": {"type": "choice", "instructions": "If movement must stop, choose the immediate response based on this camera view.", "criteria": {
                "hold": "hold still and wait for the view or path to clear",
                "retreat": "pull the gripper back toward the robot base to avoid the approaching object or hand",
                "lift": "lift the gripper above the obstacle"}},
            "correction": {"type": "choice", "instructions": "Compare the GREEN target marker to the BLUE outlined planned landing point. Which relation is visible?", "criteria": {
                "aligned": "the green marker overlaps the blue outlined circle",
                "left": "the green marker is to the image-left of the blue outlined circle",
                "right": "the green marker is to the image-right of the blue outlined circle"}},
        }
        payload = {"model": self.model, "image_b64": base64.b64encode(image).decode(), "questions": questions}
        request = urllib.request.Request(self.base+"/v1/systemone", json.dumps(payload).encode(),
                                         {"Content-Type": "application/json", "Authorization": "Bearer "+self.key})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = json.loads(response.read())
                served = response.headers.get("X-Jev-Model")
        except urllib.error.HTTPError as exc:
            # Error body may contain service internals; log status without credentials.
            raise RuntimeError(f"HTTP {exc.code}") from None
        action, probability = parse_reflex(raw)
        return {"action": action, "probability": probability, "response": raw,
                "served_model": served, "requested_model": self.model,
                "image_sha256": hashlib.sha256(image).hexdigest(), "questions": questions}


class LatestWorker:
    def __init__(self, infer, trace_path):
        self.infer, self.trace_path = infer, trace_path
        self.condition = threading.Condition()
        self.observation = None
        self.last_sequence = -1
        self.outputs = []
        self.stopping = False
        self.thread = threading.Thread(target=self._run, name="model-reflex", daemon=True)
        self.thread.start()

    def submit(self, observation):
        with self.condition:
            self.observation = observation
            self.condition.notify_all()

    def drain(self):
        with self.condition:
            outputs, self.outputs = self.outputs, []
            return outputs

    def _run(self):
        with open(self.trace_path, "w") as trace:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopping or
                                            (self.observation is not None and self.observation.sequence > self.last_sequence))
                    if self.stopping:
                        return
                    observation = self.observation
                    self.last_sequence = observation.sequence
                began = time.monotonic()
                record = {"sequence": observation.sequence, "observed_sim_s": observation.sim_s,
                          "observed_wall_s": observation.wall_s, "sensors": observation.sensors,
                          "motor_action_at_observation": observation.motor_action}
                try:
                    record.update(self.infer(observation))
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                record["inference_s"] = time.monotonic()-began
                record["received_monotonic"] = time.monotonic()
                trace.write(json.dumps(record)+"\n")
                trace.flush()
                with self.condition:
                    self.outputs.append(record)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=4)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker failed to stop within HTTP timeout")


class ActionMailbox:
    def __init__(self, max_age=.85, command_ttl=.9):
        self.max_age, self.command_ttl = max_age, command_ttl
        self.record = None
        self.accepted_at = 0
        self.expired_count = 0
        self.min_sequence = -1

    def invalidate(self, min_sequence):
        self.record = None
        self.min_sequence = min_sequence

    def accept(self, record, now):
        age = now-record["observed_wall_s"]
        if "error" in record or age > self.max_age or age < 0 or record["sequence"] < self.min_sequence:
            return False
        if self.record and record["sequence"] <= self.record["sequence"]:
            return False
        self.record, self.accepted_at = record, now
        return True

    def action(self, now):
        if self.record is None:
            return "continue" if now < self.command_ttl else "watchdog_hold"
        # Expiry is tied to observation age, not arrival age: no indefinitely stale motion.
        return self.record["action"] if now-self.record["observed_wall_s"] <= self.command_ttl else "watchdog_hold"
