"""Shared motor and completion regression; scripted outputs are not model evidence."""
import os, unittest
from pathlib import Path
import numpy as np
from jev_reflex.world import World
from jev_reflex.scenario import Scenario
from jev_reflex.completion import task_completion

class CompletionTests(unittest.TestCase):
    def test_transient_touch_does_not_count_as_retained_completion(self):
        rows=[{"sim_s":(i+1)*.05,"wall_s":(i+1)*.05,"goal_reached":i<50,"measured_eef_velocity_m_s":[0,0,0]} for i in range(80)]
        result=task_completion(rows,Scenario.generate("quiet",701))
        self.assertFalse(result["success"]);self.assertIsNotNone(result["first_dwell_wall_s"])
        for row in rows:row["goal_reached"]=True
        self.assertTrue(task_completion(rows,Scenario.generate("quiet",701))["success"])
        for row in rows:row["measured_eef_velocity_m_s"]=[.03,0,0]
        self.assertFalse(task_completion(rows,Scenario.generate("quiet",701))["success"])

    def test_before_target_shift_is_not_task_completion(self):
        rows=[{"sim_s":(i+1)*.05,"wall_s":(i+1)*.05,"goal_reached":True,"measured_eef_velocity_m_s":[0,0,0]} for i in range(160)]
        self.assertFalse(task_completion(rows,Scenario.generate("target_jump",701))["success"])

    def test_one_lateral_reply_is_bounded_and_cached_reply_cannot_repeat(self):
        world=World(Path(os.environ["PANDA_ASSET_ROOT"]),Scenario.generate("quiet",701),256)
        try:
            for _ in range(200):world.motor("right",command_id=("brain",1));world.step()
            self.assertAlmostEqual(world.offset[1],.015)
            self.assertLess(np.linalg.norm(world.position()-(world.plan_goal+world.offset)),.025)
            world.motor("left",command_id=("brain",2));world.step()
            world.motor("right",command_id=("brain",1));world.step()
            self.assertAlmostEqual(world.offset[1],0)
        finally:world.close()

    def test_visual_start_holds_and_absolute_plan_clears_local_offset(self):
        world=World(Path(os.environ["PANDA_ASSET_ROOT"]),Scenario.generate("quiet",701),256)
        try:
            world.await_visual_plan();initial=world.position().copy()
            for _ in range(80):world.motor("continue");world.step()
            self.assertLess(np.linalg.norm(world.position()-initial),.005)
            world.apply_visual_plan([.59,.62])
            world.motor("right",command_id=("brain",1));world.step()
            world.apply_visual_plan([.59,.62])
            self.assertAlmostEqual(world.offset[1],0)
            for _ in range(240):world.motor("continue");world.step()
            position=world.position().copy()
            for _ in range(160):world.motor("continue");world.step()
            self.assertLess(np.linalg.norm(world.position()-position),.005)
            self.assertLess(np.linalg.norm(world.position()-world.plan_goal),.015)
        finally:world.close()

    def test_visual_reset_holds_restored_pose_until_new_visual_plan(self):
        world=World(Path(os.environ["PANDA_ASSET_ROOT"]),Scenario.generate("quiet",701),256)
        try:
            world.await_visual_plan();origin=world.position().copy()
            world.apply_visual_plan([.59,.62])
            for _ in range(120):world.motor("continue");world.step()
            self.assertGreater(np.linalg.norm(world.position()-origin),.05)
            world.reset_live()
            np.testing.assert_allclose(world.position(),origin,atol=1e-8)
            np.testing.assert_allclose(world.plan_goal,origin,atol=1e-8)
            for _ in range(80):world.motor("continue");world.step()
            self.assertLess(np.linalg.norm(world.position()-origin),.005)
        finally:world.close()

if __name__=="__main__":unittest.main()
