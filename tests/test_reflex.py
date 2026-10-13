"""Causality/freshness tests for asynchronous reflex control."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from jev_reflex.policy import ActionMailbox, LatestWorker, Observation, parse_reflex, validate_response
from jev_reflex.scenario import Scenario


class ReflexTests(unittest.TestCase):
    def test_model_stop_decision_takes_priority_over_spatial_correction(self):
        raw = {"motion": {"choice": "stop", "probabilities": {"stop": .8}},
               "avoid": {"choice": "retreat", "probabilities": {"retreat": .9}},
               "correction": {"choice": "right", "probabilities": {"right": .95}}}
        self.assertEqual(parse_reflex(raw), ("retreat", .8))
        raw["motion"] = {"choice": "move", "probabilities": {"move": .85}}
        raw["correction"] = {"choice": "aligned", "probabilities": {"aligned": .9}}
        self.assertEqual(parse_reflex(raw), ("continue", .85))

    def test_rejects_invalid_and_nonfinite_actions(self):
        for answer in ({"choice": "teleport"}, {"choice": "brake"}, {"choice": "brake", "valid": False},
                       {"choice": "brake", "abstain": True},
                       {"choice": "brake", "probabilities": {"brake": float("nan")}}):
            with self.assertRaises(ValueError):
                validate_response({"action": answer})

    def test_nominal_startup_window_is_bounded_and_separately_named(self):
        box = ActionMailbox(.8, .9)
        self.assertEqual(box.action(.1), "continue")
        self.assertEqual(box.action(.89), "continue")
        self.assertEqual(box.action(.9), "watchdog_hold")

    def test_freshness_uses_capture_time_not_arrival_time(self):
        box = ActionMailbox(.8, .9)
        self.assertFalse(box.accept({"sequence": 0, "observed_wall_s": 0, "action": "retreat"}, 1))
        self.assertEqual(box.action(1), "watchdog_hold")
        self.assertTrue(box.accept({"sequence": 1, "observed_wall_s": .4, "action": "brake"}, 1))
        self.assertEqual(box.action(1.1), "brake")
        self.assertEqual(box.action(1.4), "watchdog_hold")

    def test_older_reply_cannot_override_newer_command(self):
        box = ActionMailbox()
        self.assertTrue(box.accept({"sequence": 9, "observed_wall_s": .4, "action": "brake"}, .5))
        self.assertFalse(box.accept({"sequence": 8, "observed_wall_s": .45, "action": "continue"}, .6))
        self.assertEqual(box.action(.7), "brake")

    def test_scene_reset_invalidates_inflight_reply(self):
        box = ActionMailbox()
        box.invalidate(20)
        self.assertFalse(box.accept({"sequence": 19, "observed_wall_s": .4, "action": "retreat"}, .5))
        self.assertTrue(box.accept({"sequence": 20, "observed_wall_s": .4, "action": "continue"}, .5))

    def test_busy_inference_does_not_block_submission_and_skips_old_frames(self):
        entered, release, second = threading.Event(), threading.Event(), threading.Event()
        seen = []
        def infer(obs):
            seen.append(obs.sequence)
            if obs.sequence == 0:
                entered.set()
                self.assertTrue(release.wait(2))
            else:
                second.set()
            return {"action": "continue", "probability": 1}
        with tempfile.TemporaryDirectory() as tmp:
            worker = LatestWorker(infer, Path(tmp)/"trace.jsonl")
            worker.submit(Observation(0, 0, 0, None, {}, "continue"))
            self.assertTrue(entered.wait(1))
            start = time.monotonic()
            for sequence in range(1, 101):
                worker.submit(Observation(sequence, sequence*.05, sequence*.05, None, {}, "continue"))
            self.assertLess(time.monotonic()-start, .1)
            release.set()
            self.assertTrue(second.wait(1))
            worker.close()
            self.assertEqual(seen, [0, 100])
            rows = [json.loads(x) for x in (Path(tmp)/"trace.jsonl").read_text().splitlines()]
            self.assertEqual([r["sequence"] for r in rows], [0, 100])

    def test_saved_seed_reproduces_disturbance_without_policy_state(self):
        self.assertEqual(Scenario.generate("hand", 31), Scenario.generate("hand", 31))
        self.assertNotEqual(Scenario.generate("hand", 31), Scenario.generate("hand", 73))
        self.assertEqual(Scenario.generate("quiet", 31).events(), [])
        combined = Scenario.generate("combined", 31)
        self.assertEqual({x["kind"] for x in combined.events()}, {"hand", "slider", "target", "occlusion"})
        self.assertEqual(len(combined.events()), 9)
        events = combined.events()
        for kind in ("hand", "slider", "target", "occlusion"):
            actor_events = [event for event in events if event["kind"] == kind]
            self.assertTrue(all(a["end_sim_s"] < b["onset_sim_s"] for a, b in zip(actor_events, actor_events[1:])))
        self.assertNotEqual(combined.timings, Scenario.generate("combined", 73).timings)

    def test_future_same_kind_event_does_not_hide_current_event(self):
        from dataclasses import replace
        scenario = replace(Scenario.generate("combined", 31), timings=(("hand", 2, 3), ("hand", 8, 3)))
        self.assertTrue(scenario.active(3, "hand"))
        self.assertFalse(scenario.active(6, "hand"))
        self.assertTrue(scenario.active(9, "hand"))
        self.assertEqual(scenario.event("hand", 3)["onset_sim_s"], 2)


if __name__ == "__main__":
    unittest.main()
