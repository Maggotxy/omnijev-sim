"""Runtime contracts and mechanical fixtures; not model performance evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from jev_reflex.bank import ReflexBank, ReflexMemory, parse_decision, signature, LEVELS
from jev_reflex.brain import validate_plan


def response(bank, choice="retreat", condition=.9, confidence=.8):
    answer = {"condition_"+r["id"]: {"noul": condition} for r in bank["reflexes"]}
    answer["arbitrate"] = {"choice": choice, "probabilities": {r["id"]: .8 if r["id"] == choice else .04 for r in bank["reflexes"]},
                            "valid": True, "confidence": confidence}
    answer["danger"] = {"score": LEVELS[2], "probabilities": dict(zip(LEVELS, [.1, .2, .6, .1]))}
    return answer


class BankContracts(unittest.TestCase):
    def test_probability_and_arbitration_both_gate_the_declared_action(self):
        bank = ReflexBank().snapshot()
        self.assertEqual(parse_decision(response(bank), bank)["action"], "retreat")
        self.assertEqual(parse_decision(response(bank, condition=.2), bank)["action"], "continue")
        self.assertEqual(parse_decision(response(bank, confidence=.01), bank)["action"], "continue")

    def test_score_modulates_speed_without_selecting_an_action(self):
        bank = ReflexBank().snapshot()
        parsed = parse_decision(response(bank, choice="brake"), bank)
        self.assertEqual(parsed["action"], "brake")
        self.assertGreater(parsed["danger_score"], .5)
        self.assertGreaterEqual(parsed["speed_scale"], .55)
        self.assertLess(parsed["speed_scale"], 1)

    def test_invalid_probabilities_and_abstained_choices_are_rejected(self):
        bank = ReflexBank().snapshot()
        for bad in (float("nan"), float("inf"), -1, 2):
            answer = response(bank)
            answer["condition_retreat"]["noul"] = bad
            with self.assertRaises(ValueError):
                parse_decision(answer, bank)
        answer = response(bank)
        answer["arbitrate"]["abstain"] = True
        with self.assertRaises(ValueError):
            parse_decision(answer, bank)

    def test_bank_transaction_is_atomic_and_versions_real_updates(self):
        bank = ReflexBank()
        before = bank.snapshot()
        with self.assertRaises(ValueError):
            bank.update([{"id": "retreat", "threshold": .8}, {"id": "brake", "threshold": 2}], "test", 0)
        self.assertEqual(bank.snapshot(), before)
        self.assertTrue(bank.update([{"id": "retreat", "threshold": .8}], "test", 0))
        self.assertEqual(bank.version, 1)
        self.assertFalse(bank.update([{"id": "retreat", "threshold": .8}], "test", 0))

    def test_runtime_authoring_adds_a_question_and_keeps_motor_vocabulary(self):
        bank = ReflexBank()
        bank.update([{"id": "new_condition", "condition": "A hand is entering from the left", "threshold": .8, "action": "brake"}], "operator", 0, allow_new=True)
        self.assertEqual(len(bank.snapshot()["reflexes"]), 7)
        with self.assertRaises(ValueError):
            bank.update([{"id": "unsafe", "condition": "Any observation", "threshold": .8, "action": "run_code"}], "operator", 0, allow_new=True)

    def test_memory_retrieves_measured_completed_feedback_and_excludes_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            memory = ReflexMemory(Path(temp)/"memory.jsonl")
            key = signature(np.full((16, 16, 3), 100, dtype=np.uint8), {"touch_normal_force_N": [0, 0, 0]})
            record = {"action": "retreat", "observation_signature": key}
            memory.remember(record, 1, [0, 0, 0], [10, 0, 0])
            self.assertEqual(memory.retrieve(key), [])
            memory.feedback(1.7, [.06, 0, 0], [1, 0, 0])
            self.assertIn("10.0 to 1.0", memory.retrieve(key)[0])
            memory.remember(record, 2, [0, 0, 0], [5, 0, 0])
            memory.reset_epoch()
            memory.feedback(3, [1, 0, 0], [0, 0, 0])
            entries = [json.loads(s) for s in memory.path.read_text().splitlines()]
            self.assertEqual(sum(e["kind"] == "feedback" for e in entries), 1)
            self.assertEqual(len(memory.retrieve(key)), 1)

    def test_memory_retains_intervening_actions_without_claiming_causality(self):
        with tempfile.TemporaryDirectory() as temp:
            memory = ReflexMemory(Path(temp)/"memory.jsonl")
            key = [0]*27
            memory.remember({"action": "retreat", "observation_signature": key}, 0, [0, 0, 0], [0, 0, 0])
            memory.feedback(.3, [.01, 0, 0], [0, 0, 0], action="retreat")
            memory.feedback(.7, [.02, 0, 0], [0, 0, 0], action="brake")
            self.assertEqual(memory.entries[0]["feedback"]["actions"], ["retreat", "brake"])
            self.assertIn("not isolated causal evidence", memory.retrieve(key)[0])
            memory.reset_epoch()
            self.assertEqual(memory.epoch, 1)


class BrainContracts(unittest.TestCase):
    def plan(self):
        return {"goal": "green_pad", "stage": "reach_target", "target_visible": True,
                "target_xy": [.57, .62], "reflex_updates": []}

    def test_brain_updates_task_and_conditions_but_cannot_replace_motor_mapping(self):
        plan = self.plan()
        plan["reflex_updates"] = [{"id": "retreat", "threshold": .8}]
        self.assertEqual(validate_plan(plan, {"retreat"})["target_xy"], [.57, .62])
        plan["reflex_updates"] = [{"id": "retreat", "action": "continue"}]
        with self.assertRaises(ValueError):
            validate_plan(plan, {"retreat"})

    def test_wrong_goal_and_invalid_pixels_are_rejected(self):
        for bad in ([float("nan"), .5], [True, .5], [-.1, .5], [.5], [1.5, .5]):
            plan = self.plan()
            plan["target_xy"] = bad
            with self.assertRaises(ValueError):
                validate_plan(plan, set())
        plan = self.plan()
        plan["goal"] = "yellow_block"
        with self.assertRaises(ValueError):
            validate_plan(plan, set())


if __name__ == "__main__":
    unittest.main()
