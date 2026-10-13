"""Versioned natural-language reflexes and retrieval of measured robot feedback."""
from collections import deque
import copy
import json
import math
from pathlib import Path

import numpy as np

ACTIONS = ("continue", "brake", "retreat", "lift", "left", "right")
LEVELS = ("clear and safe", "minor concern", "danger of collision", "immediate emergency")
DEFAULTS = [
    {"id": "continue", "action": "continue", "threshold": .55,
     "condition": "The camera view is clear and the robot gripper can continue its planned reach without an immediate obstacle."},
    {"id": "brake", "action": "brake", "threshold": .75,
     "condition": "The camera workspace is obscured, or the physical touch force exceeds 10 N during this open-gripper reach."},
    {"id": "retreat", "action": "retreat", "threshold": .75,
     "condition": "A separate human skin-colored hand or arm is very close to the robot gripper or entering its immediate path."},
    {"id": "lift", "action": "lift", "threshold": .75,
     "condition": "A solid yellow object blocks the robot gripper's immediate reach from below and lifting can avoid it."},
    {"id": "left", "action": "left", "threshold": .65,
     "condition": "The green or turquoise circular target pad is visibly to image-left of the blue outlined planned landing point."},
    {"id": "right", "action": "right", "threshold": .65,
     "condition": "The green or turquoise circular target pad is visibly to image-right of the blue outlined planned landing point."},
]


def probability(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Probability must be finite and within [0,1]")
    return value


def validate_update(update, allow_new=False):
    if not isinstance(update, dict):
        raise ValueError("Reflex update must be an object")
    identity = update.get("id")
    if not isinstance(identity, str) or not identity or len(identity) > 40 or not identity.replace("_", "").isalnum():
        raise ValueError("Invalid reflex ID")
    output = {"id": identity}
    if "condition" in update:
        condition = update["condition"]
        if not isinstance(condition, str) or not 5 <= len(condition.strip()) <= 600:
            raise ValueError("Invalid natural-language condition")
        output["condition"] = condition.strip()
    if "threshold" in update:
        threshold = probability(update["threshold"])
        if not .5 <= threshold <= .98:
            raise ValueError("Threshold outside runtime limits")
        output["threshold"] = threshold
    if "action" in update:
        if update["action"] not in ACTIONS:
            raise ValueError("Action outside motor vocabulary")
        output["action"] = update["action"]
    if allow_new and not all(key in output for key in ("condition", "threshold", "action")):
        raise ValueError("New reflex needs condition, threshold and action")
    return output


class ReflexBank:
    def __init__(self, path=None):
        self.version = 0
        self.reflexes = copy.deepcopy(DEFAULTS)
        self.arbitration_confidence = .12
        self.path = Path(path) if path else None

    def snapshot(self):
        return {"version": self.version, "reflexes": copy.deepcopy(self.reflexes),
                "arbitration_confidence": self.arbitration_confidence, "score_levels": list(LEVELS)}

    def update(self, updates, source, sim_s, allow_new=False):
        if not isinstance(updates, list) or len(updates) > 3:
            raise ValueError("At most three reflex updates per transaction")
        next_reflexes = copy.deepcopy(self.reflexes)
        for update in updates:
            checked = validate_update(update)
            existing = next((r for r in next_reflexes if r["id"] == checked["id"]), None)
            if existing is None:
                if not allow_new or len(next_reflexes) >= 12:
                    raise ValueError("Unknown reflex ID")
                next_reflexes.append(validate_update(update, allow_new=True))
            else:
                existing.update(checked)
        if next_reflexes == self.reflexes:
            return False
        self.reflexes = next_reflexes
        self.version += 1
        if self.path:
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"sim_s": sim_s, "source": source, **self.snapshot()})+"\n")
        return True


def make_questions(snapshot, touch, current_action, memories):
    context = ("Robot camera. Black gripper and white links are the ROBOT. Orange outline marks its gripper; "
               "green/turquoise pad is the reach target, blue outline its planned landing point. "
               f"Touch force {touch} N. Current motor {current_action}. "
               "The gripper stays open; yellow block is an obstacle.")
    # Each binary head gets one condition, with current physical touch for the contact condition.
    questions = {"condition_"+r["id"]: {"type": "noul", "instructions": r["condition"]+(f" Current touch {touch} N." if r["id"] == "brake" else "")}
                 for r in snapshot["reflexes"]}
    questions["arbitrate"] = {"type": "choice", "instructions": context+" Select the immediate reflex appropriate to this actual view. "+" ".join(memories)[:220],
                              "criteria": {r["id"]: r["action"]+": "+r["condition"] for r in snapshot["reflexes"]}}
    questions["danger"] = {"type": "score", "instructions": "Rate immediate risk of continuing the reach. "+context,
                            "levels": snapshot["score_levels"]}
    return questions


def parse_decision(raw, snapshot):
    definitions = {r["id"]: r for r in snapshot["reflexes"]}
    conditions = {identity: probability(raw["condition_"+identity]["noul"]) for identity in definitions}
    answer = raw["arbitrate"]
    chosen = answer.get("choice")
    if chosen not in definitions or answer.get("valid") is False or answer.get("abstain") is True:
        raise ValueError("Invalid model arbitration")
    selected_probability = probability(answer["probabilities"][chosen])
    confidence = probability(answer.get("confidence", selected_probability))
    reflex = definitions[chosen]
    triggered = conditions[chosen] >= reflex["threshold"]
    certain = confidence >= snapshot["arbitration_confidence"]
    action = reflex["action"] if triggered and certain else "continue"
    score = raw["danger"]
    selected_level = score.get("score", score.get("level"))
    levels = snapshot["score_levels"]
    if selected_level not in levels:
        raise ValueError("Invalid danger level")
    distribution = [probability(score["probabilities"][level]) for level in levels]
    total = sum(distribution)
    if not .9 <= total <= 1.1:
        raise ValueError("Invalid danger distribution")
    danger = sum(i*p for i, p in enumerate(distribution))/(total*(len(levels)-1))
    return {"action": action, "probability": selected_probability, "reflex_id": chosen,
            "condition_probabilities": conditions, "bank_version": snapshot["version"],
            "arbitration_confidence": confidence, "threshold_triggered": triggered,
            "arbitration_passed": certain, "model_intervention": bool(triggered and certain),
            "gate_reason": "selected_reflex" if triggered and certain else "condition_or_arbitration_below_threshold",
            "danger_score": danger, "speed_scale": 1-.45*danger}


def signature(frame, sensors):
    # Generic image statistics and robot sensors are retrieval keys, not a detector.
    hist = np.concatenate([np.histogram(frame[..., c], bins=8, range=(0, 256))[0]/frame[..., c].size for c in range(3)])
    touch = np.log1p(np.maximum(0, sensors["touch_normal_force_N"]))/np.log(101)
    return np.r_[hist, np.clip(touch, 0, 2)].tolist()


class ReflexMemory:
    def __init__(self, path, maximum=128):
        self.path, self.entries = Path(path), deque(maxlen=maximum)
        self.pending = []
        self.counter = 0
        self.epoch = 0

    def retrieve(self, key, limit=3):
        completed = [e for e in self.entries if "feedback" in e]
        ranked = sorted(completed, key=lambda e: float(np.linalg.norm(np.asarray(e["signature"])-key)))[:limit]
        return [f"Past measured association, epoch {e['epoch']}: candidate {e['action']}; actual interval actions {e['feedback']['actions']}; touch {e['touch_before_N']:.1f} to {e['feedback']['touch_after_N']:.1f} N; motion {e['feedback']['motion_m']:.3f} m. This is not isolated causal evidence."
                for e in ranked]

    def remember(self, record, sim_s, position, touch):
        if "observation_signature" not in record:
            return
        self.counter += 1
        entry = {"id": self.counter, "epoch": self.epoch, "sim_s": sim_s, "action": record["action"],
                 "signature": record["observation_signature"], "touch_before_N": float(max(touch)),
                 "position_before_m": list(map(float, position)), "condition_probabilities": record.get("condition_probabilities"),
                 "bank_version": record.get("bank_version"), "image_sha256": record.get("image_sha256"), "interval_actions": []}
        self.entries.append(entry)
        self.pending.append(entry)
        self._write({"kind": "association", **entry})

    def feedback(self, sim_s, position, touch, action=None, plan_version=None):
        for entry in list(self.pending):
            actual = action or entry["action"]
            if not entry["interval_actions"] or entry["interval_actions"][-1] != actual:
                entry["interval_actions"].append(actual)
            if sim_s-entry["sim_s"] < .6:
                continue
            entry["feedback"] = {"sim_s": sim_s, "touch_after_N": float(max(touch)),
                                 "motion_m": float(np.linalg.norm(np.asarray(position)-entry["position_before_m"])),
                                 "actions": entry["interval_actions"].copy(), "plan_version": plan_version,
                                 "measurement_scope": "observed association across a 0.6s interval; not causal attribution"}
            self._write({"kind": "feedback", "id": entry["id"], **entry["feedback"]})
            self.pending.remove(entry)

    def reset_epoch(self):
        self.epoch += 1
        self.pending.clear()  # Retain completed experience; do not attribute a reset to a reflex.

    def _write(self, value):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value)+"\n")
