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

from .bank import ReflexBank, make_questions, parse_decision, signature

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
    context: dict = None
    clock_origin_monotonic: float = None


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
        context = observation.context or {}
        snapshot = context.get("bank") or ReflexBank().snapshot()
        questions = make_questions(snapshot, observation.sensors["touch_normal_force_N"],
                                   observation.motor_action, context.get("memories", []))
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
        decision = parse_decision(raw, snapshot)
        return {**decision, "response": raw,
                "served_model": served, "requested_model": self.model,
                "image_sha256": hashlib.sha256(image).hexdigest(), "questions": questions,
                "observation_signature": signature(observation.frame, observation.sensors),
                "bank_snapshot": snapshot, "retrieved_memories": context.get("memories", [])}


class LatestWorker:
    def __init__(self, infer, trace_path, close_timeout=4, minimum_interval=0):
        self.infer, self.trace_path = infer, trace_path
        self.close_timeout = close_timeout
        self.minimum_interval = minimum_interval
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
        next_start = 0
        with open(self.trace_path, "w") as trace:
            while True:
                with self.condition:
                    while not self.stopping:
                        fresh = self.observation is not None and self.observation.sequence > self.last_sequence
                        delay = next_start-time.monotonic()
                        if fresh and delay <= 0:
                            break
                        self.condition.wait(timeout=max(.001, delay) if fresh else None)
                    if self.stopping:
                        return
                    observation = self.observation
                    self.last_sequence = observation.sequence
                    next_start = time.monotonic()+self.minimum_interval
                began = time.monotonic()
                record = {"sequence": observation.sequence, "observed_sim_s": observation.sim_s,
                          "observed_wall_s": observation.wall_s, "sensors": observation.sensors,
                          "motor_action_at_observation": observation.motor_action}
                try:
                    record.update(self.infer(observation))
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                ended = time.monotonic()
                record["inference_s"] = ended-began
                record["received_monotonic"] = ended
                if observation.clock_origin_monotonic is not None:
                    record["call_started_wall_s"] = began-observation.clock_origin_monotonic
                    record["received_wall_s"] = ended-observation.clock_origin_monotonic
                    record["capture_to_call_s"] = record["call_started_wall_s"]-observation.wall_s
                trace.write(json.dumps(record)+"\n")
                trace.flush()
                with self.condition:
                    self.outputs.append(record)

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.thread.join(timeout=self.close_timeout)
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
