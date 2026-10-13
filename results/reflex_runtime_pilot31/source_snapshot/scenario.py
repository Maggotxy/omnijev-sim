"""Saved exogenous disturbances. Only the world and evaluator see these."""
from dataclasses import asdict, dataclass
import numpy as np

KINDS = ("hand", "slider", "target", "occlusion", "combined", "quiet")


@dataclass(frozen=True)
class Scenario:
    kind: str
    seed: int
    onset: float
    duration: float
    side: int
    shift: float
    mass: float
    friction: float
    light: float
    nominal_goal: tuple
    timings: tuple = ()

    @classmethod
    def generate(cls, kind, seed):
        if kind not in KINDS:
            raise ValueError("Unknown disturbance")
        rng = np.random.default_rng(seed)
        onset = float(rng.uniform(2.5, 3.7))
        duration = float(rng.uniform(3.8, 5.0))
        timings = []
        if kind == "combined":
            start = onset
            for item in rng.permutation(["hand", "slider", "target", "occlusion"]):
                timings.append((str(item), start, float(rng.uniform(2.5, 4.8))))
                start += float(rng.uniform(1.2, 2.9))
        return cls(kind, seed, onset,
                   duration, int(rng.choice([-1, 1])),
                   float(rng.uniform(0.09, 0.15)), float(rng.uniform(0.12, 0.28)),
                   float(rng.uniform(0.12, 0.28)), float(rng.uniform(0.75, 1.1)),
                   (0.61, float(rng.uniform(0.06, 0.12)), 0.59), tuple(timings))

    def serializable(self):
        return asdict(self)

    def contains(self, kind):
        return self.kind == kind or self.kind == "combined"

    def active(self, t, kind):
        event = self.event(kind, t)
        return bool(event and event["onset_sim_s"] <= t < event["end_sim_s"])

    def event(self, kind, t=None):
        return next((event for event in reversed(self.events()) if event["kind"] == kind
                     and (t is None or event["onset_sim_s"] <= t)), None)

    def events(self):
        output = []
        items = self.timings if self.kind == "combined" else [(self.kind, self.onset, self.duration)]
        for item in items:
            kind, onset, duration = item[:3]
            params = item[3] if len(item)>3 else {}
            if kind != "quiet":
                side = 1 if params.get("end_y", self.side)-params.get("start_y", 0)>0 else -1
                output.append({"kind": kind, "onset_sim_s": onset,
                               "end_sim_s": onset + duration,
                               "expected": (["left"] if side < 0 else ["right"]) if kind == "target" else
                                           (["brake"] if kind == "occlusion" else ["brake", "retreat", "lift"]),
                               **params})
        return output
