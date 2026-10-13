"""Real MuJoCo/GL timing check; fixed test responses are not model evidence."""
import argparse
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from jev_reflex.run import episode
from jev_reflex.scenario import Scenario
from jev_reflex.world import World


class SlowMechanicalResponse:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, observation):
        time.sleep(.5)
        return {"action": "brake", "probability": 1, "served_model": "mechanical-test-stub"}


class SlowVisualResponse:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, observation):
        time.sleep(1)
        return {"plan": {"goal": "green_pad", "stage": "reach_target", "target_visible": True,
                         "target_xy": [.62, .6], "reflex_updates": []},
                "bank_version_at_observation": observation.context["bank"]["version"],
                "served_model": "mechanical-brain-stub"}


@unittest.skipUnless(Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim"), "panda.xml").exists(), "Requires Panda assets and MuJoCo offscreen GL")
class RealTimeIntegration(unittest.TestCase):
    def test_future_scheduled_events_are_not_scored_as_missed_reactions(self):
        from dataclasses import replace
        from jev_reflex.run import reactions
        scenario = replace(Scenario.generate("quiet", 31), kind="combined", timings=(("hand", 2, 3), ("occlusion", 20, 3)))
        self.assertEqual([r["kind"] for r in reactions(scenario, [], horizon=10)], ["hand"])

    def test_slow_brain_and_reflex_run_independently_of_physical_time(self):
        args = argparse.Namespace(asset_root=Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")),
                                  image_size=256, api_timeout=2, max_observation_age=1.2,
                                  command_ttl=1.4, seconds=2, render_hz=8,
                                  brain_timeout=2, brain_interval=4, max_brain_age=12)
        with tempfile.TemporaryDirectory() as tmp, patch("jev_reflex.run.HTTPReflex", SlowMechanicalResponse), patch("jev_reflex.run.VisualBrain", SlowVisualResponse):
            output = Path(tmp)/"integration"
            result = episode(args, Scenario.generate("quiet", 31), "vlm_omni", output)
            import json
            reflex = [json.loads(s) for s in (output/"applications.jsonl").read_text().splitlines()]
            brain = [json.loads(s) for s in (output/"brain_applications.jsonl").read_text().splitlines()]
            first_reflex = next(r for r in reflex if r["accepted"])
            first_brain = next(r for r in brain if r["accepted"])
            self.assertLess(first_reflex["applied_sim_s"], first_brain["applied_sim_s"])
            self.assertAlmostEqual(result["sim_seconds"], 2, places=5)
            self.assertGreater(result["real_time_factor"], .9)
            self.assertEqual(result["macro_plan_changes"], 1)

    def test_visual_plan_uses_camera_ray_and_never_tracks_target_truth(self):
        world = World(Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")), Scenario.generate("quiet", 31), 256)
        try:
            first = world.apply_visual_plan([.6, .62])["calibrated_goal_m"]
            import numpy as np
            import mujoco
            world.data.mocap_pos[world.model.body_mocapid[world.goal]] = [.7, -.25, .418]
            mujoco.mj_forward(world.model, world.data)
            second = world.apply_visual_plan([.6, .62])["calibrated_goal_m"]
            np.testing.assert_allclose(first, second)
            self.assertGreater(first[1], 0)
        finally:
            world.close()

    def test_retreat_is_local_and_nominal_motion_can_resume(self):
        world = World(Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")), Scenario.generate("quiet", 31), 256)
        try:
            origin = world.position()
            for _ in range(160):
                world.motor("retreat")
                world.step()
            import numpy as np
            self.assertLess(np.linalg.norm(world.position()-origin), .12)
            self.assertLess(np.linalg.norm(world.position()-world.reflex_anchor), .025)
            for _ in range(160):
                world.motor("continue")
                world.step()
            self.assertTrue(world.evaluate()["goal_reached"])
            # Actual camera calibration: positive world Y is image-right.
            import mujoco
            cid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
            R = world.data.cam_xmat[cid].reshape(3, 3)
            self.assertGreater((R.T @ np.array([0, .1, 0]))[0], 0)
        finally:
            world.close()

    def test_physics_advances_during_half_second_inference(self):
        args = argparse.Namespace(asset_root=Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")),
                                  image_size=256, api_timeout=2, max_observation_age=1.2,
                                  command_ttl=1.4, seconds=2, render_hz=8)
        with tempfile.TemporaryDirectory() as tmp, patch("jev_reflex.run.HTTPReflex", SlowMechanicalResponse):
            output = Path(tmp)/"integration"
            result = episode(args, Scenario.generate("quiet", 31), "reflex", output)
            import json
            applications = [json.loads(s) for s in (output/"applications.jsonl").read_text().splitlines()]
            first = next(r for r in applications if r["accepted"])
            self.assertGreaterEqual(first["applied_sim_s"]-first["observed_sim_s"], .4)
            self.assertLess(first["observation_age_s"], 1.2)
            self.assertAlmostEqual(result["sim_seconds"], 2, places=5)
            self.assertGreater(result["real_time_factor"], .9)
            self.assertTrue(first["motor_changed"])
            self.assertGreater(result["motor_action_ticks"].get("brake", 0), 0)

    def test_hand_starts_outside_camera_and_exits_on_its_own_side(self):
        from dataclasses import replace
        import numpy as np
        original = Scenario.generate("quiet", 31)
        scenario = replace(original, kind="combined", timings=(("hand", 1, 2, {"side": -original.side, "x": .6, "z": .62}),))
        world = World(Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")), scenario, 256)
        try:
            driver = world.model.body_mocapid[world.human_driver]
            # The finite weld permits vertical sag under gravity; lateral placement must match.
            np.testing.assert_allclose(world.data.xpos[world.human, :2], world.data.mocap_pos[driver, :2], atol=.002)
            self.assertGreater(abs(world.data.xpos[world.human, 1]), .65)
            world.data.time = world.zero+2.999
            world.disturbances()
            before = world.data.mocap_pos[driver].copy()
            world.data.time = world.zero+3.001
            world.disturbances()
            self.assertLess(np.linalg.norm(before-world.data.mocap_pos[driver]), .01)
            self.assertEqual(np.sign(world.data.mocap_pos[driver, 1]), -original.side)
        finally:
            world.close()

    def test_new_slider_preserves_the_previous_free_object(self):
        from dataclasses import replace
        import numpy as np
        import mujoco
        scenario = replace(Scenario.generate("quiet", 31), kind="combined", timings=(("slider", 1, .5), ("slider", 3, .5)))
        world = World(Path(os.environ.get("PANDA_ASSET_ROOT", "/opt/omnijev-sim")), scenario, 256)
        try:
            jid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_JOINT, "slider_joint")
            qa = world.model.jnt_qposadr[jid]
            world.data.time = world.zero+1
            world.disturbances()
            world.data.qpos[qa:qa+3] = [.4, .2, .505]
            previous = world.data.qpos[qa:qa+7].copy()
            world.data.time = world.zero+3
            world.disturbances()
            np.testing.assert_array_equal(previous, world.data.qpos[qa:qa+7])
            self.assertEqual(world.slider_introduced, 2)
        finally:
            world.close()


if __name__ == "__main__":
    unittest.main()
