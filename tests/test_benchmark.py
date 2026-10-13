import json
import os
import unittest
from unittest.mock import patch

from jev_benchmark.clients import APIError, Gemini, Omni, parse_json, redact, validate_plan


class ClientContracts(unittest.TestCase):
    def test_plan_rejects_unknown_and_identical_objects(self):
        for value in ({"object": "cup", "target": "basket"}, {"object": "soup", "target": "soup"}):
            with self.assertRaises(ValueError):
                validate_plan(value, ["soup", "basket"])
        self.assertEqual(validate_plan({"object": "finish"}, ["soup"])["object"], "finish")

    def test_fenced_json_and_invalid_output(self):
        self.assertEqual(parse_json('```json\n{"action":"wait"}\n```'), {"action": "wait"})
        for text in ('[]', 'yes', '{"action": "wait"} trailing'):
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_credentials_are_redacted(self):
        with patch.dict(os.environ, GEMINI_API_KEY="test-secret"):
            self.assertNotIn("test-secret", redact("URL test-secret projects/private-project/locations/global"))
            self.assertNotIn("private-project", redact("projects/private-project/locations/global"))

    def test_streaming_requires_distinct_transport(self):
        with patch.dict(os.environ, GEMINI_MODEL="gemini-robotics-er-2-streaming-preview"):
            with self.assertRaises(ValueError):
                Gemini()

    def test_native_request_has_role_and_preserves_model(self):
        with patch.dict(os.environ, GEMINI_API_KEY="test", GEMINI_MODEL="gemini-robotics-er-2-preview", GEMINI_PROTOCOL="generateContent"):
            api = Gemini()
            response = {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}
            with patch.object(api, "post", return_value=response) as post:
                self.assertEqual(api.health(), "OK")
                args = post.call_args.args
                self.assertIn("gemini-robotics-er-2-preview:generateContent", args[0])
                self.assertEqual(args[1]["contents"][0]["role"], "user")

    def test_http_errors_do_not_fallback_to_other_model(self):
        with patch.dict(os.environ, GEMINI_API_KEY="test"):
            api = Gemini()
            with patch.object(api, "post", side_effect=APIError(404, "unavailable")) as post:
                with self.assertRaises(APIError):
                    api.health()
                self.assertEqual(post.call_count, 1)

    def test_low_confidence_proceed_holds(self):
        api = Omni()
        for action, p, expected in (("proceed", .49, "wait"), ("proceed", .8, "proceed"), ("replan", .9, "replan")):
            with patch.object(api, "ask", return_value={"action": {"choice": action, "probabilities": {action: p}}}):
                self.assertEqual(api.gate(None, "test")["action"], expected)

    def test_nonfinite_probabilities_and_abstentions_stop(self):
        api = Omni()
        cases = [{"choice": "proceed", "probabilities": {"proceed": p}} for p in (float("nan"), float("inf"), -1, 2)]
        cases.append({"choice": "proceed", "valid": False, "probabilities": {"proceed": .9}})
        for case in cases:
            with patch.object(api, "ask", return_value={"action": case}):
                with self.assertRaises(ValueError):
                    api.gate(None, "test")


class ControlContracts(unittest.TestCase):
    def test_wait_budget_never_executes_skill(self):
        from jev_benchmark.run_suite import Episode
        from types import SimpleNamespace
        ep = object.__new__(Episode)
        ep.args = SimpleNamespace(max_gate_waits=3)
        ep.task = SimpleNamespace(language="put soup in basket")
        ep.target, ep.completed, ep.gate_counts = "soup", [], {"proceed": 0, "wait": 0, "replan": 0}
        ep.reflex = SimpleNamespace(gate=lambda *args: {"action": "wait"})
        ep.frame = lambda: None
        ep.event = lambda *args, **kwargs: None
        ep.hold = lambda seconds: None
        with self.assertRaisesRegex(RuntimeError, "model_gate_timeout"):
            ep.gate("GRASP", "close gripper")
        self.assertEqual(ep.gate_counts, {"proceed": 0, "wait": 3, "replan": 0})

    def test_pairing_rejects_mismatched_states(self):
        from jev_benchmark.compose import verify_pairing
        row = {"mode": "omni", "seed": 1, "init_index": 0, "task_name": "test", "limits": {},
               "initial_state_sha256": "one", "settled_state_sha256": "one", "controller": {"hz": 20}}
        with self.assertRaises(ValueError):
            verify_pairing([row, {**row, "mode": "r2", "initial_state_sha256": "two"}])
        self.assertFalse(verify_pairing([row, {"mode": "r2", "status": "unavailable"}]))
        self.assertTrue(verify_pairing([row, {**row, "mode": "r2"}, {**row, "mode": "r2_omni"}]))
        for change in ({"seed": 7}, {"task_name": "another"}):
            with self.assertRaises(ValueError):
                verify_pairing([row, {**row, "mode": "r2", **change}])
        with self.assertRaises(ValueError):
            verify_pairing([row, row])
        with self.assertRaises(ValueError):
            verify_pairing([{"mode": "omni", "initial_state_sha256": "one"}])

    def test_policy_roles(self):
        from jev_benchmark.run_suite import POLICIES
        self.assertEqual(POLICIES, {"r2": ("gemini", "gemini"), "r2_omni": ("gemini", "omni"), "omni": ("omni", "omni")})


if __name__ == "__main__":
    unittest.main()
