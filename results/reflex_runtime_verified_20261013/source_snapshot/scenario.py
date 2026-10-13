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
        scenario = cls(kind, seed, onset,
                   duration, int(rng.choice([-1, 1])),
                   float(rng.uniform(0.09, 0.15)), float(rng.uniform(0.12, 0.28)),
                   float(rng.uniform(0.12, 0.28)), float(rng.uniform(0.75, 1.1)),
                   (0.61, float(rng.uniform(0.06, 0.12)), 0.59))
        timings, goal_y = [], scenario.nominal_goal[1]
        kinds = [str(k) for k in rng.permutation(["hand", "slider", "target", "occlusion"])]
        if kind == "combined":
            kinds += [str(k) for k in rng.choice(["hand", "slider", "target", "occlusion"], 5)]
        else:
            kinds = [] if kind == "quiet" else [kind]
        start, occupied_until = onset, {}
        for item in kinds:
            # A single actor cannot start a second instance while it is still present.
            start = max(start, occupied_until.get(item, 0)+.2)
            span = float(rng.uniform(1.4, 5.8)) if kind == "combined" else duration
            params = {"side": int(rng.choice([-1, 1]))}
            if item == "hand":
                params.update(x=float(rng.uniform(.50, .64)), z=float(rng.uniform(.57, .69)),
                              approach_fraction=float(rng.uniform(.18, .40)))
            elif item == "slider":
                params.update(x=float(rng.uniform(.49, .64)), speed_m_s=float(rng.uniform(.35, .9)),
                              entry_y_m=float(rng.uniform(.30, .40)))
            elif item == "target":
                end_y = float(np.clip(goal_y+params["side"]*rng.uniform(.09, .16), -.22, .25))
                params.update(start_y=goal_y, end_y=end_y)
                goal_y = end_y
            elif item == "occlusion":
                params.update(camera_block_x=float(rng.uniform(.30, .65)),
                              camera_block_y=float(rng.uniform(-.26, .26)))
            timings.append((item, start, span, params))
            occupied_until[item] = start+span
            start += float(np.clip(.8+rng.exponential(2), 1, 5.5))
        from dataclasses import replace
        return replace(scenario, timings=tuple(timings))

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
        items = self.timings if self.timings or self.kind == "combined" else [(self.kind, self.onset, self.duration)]
        for item in items:
            kind, onset, duration = item[:3]
            params = item[3] if len(item)>3 else {}
            if kind != "quiet":
                side = 1 if params.get("end_y", params.get("side", self.side))-params.get("start_y", 0)>0 else -1
                output.append({"kind": kind, "onset_sim_s": onset,
                               "end_sim_s": onset + duration,
                               "expected": (["left"] if side < 0 else ["right"]) if kind == "target" else
                                           (["brake"] if kind == "occlusion" else ["brake", "retreat", "lift"]),
                               **params})
        return output
