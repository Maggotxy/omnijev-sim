"""Brain/cerebellum mechanism checks; fixed responses are not model evidence."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from jev_reflex.brain import validate_plan
from jev_reflex.control import choose_motor
from jev_reflex.motion import motion_responses
from jev_reflex.policy import ActionMailbox,LatestWorker,Observation
from jev_reflex.run import episode
from jev_reflex.scenario import Scenario


def plan(action="retreat"):
    return {"goal":"green_pad","stage":"wait_view","target_visible":False,"target_xy":[.5,.5],"reflex_updates":[],"action":action,"confidence":.9,"danger_score":.8}


class DirectBrain:
    def __init__(self,*args,**kwargs):pass
    def __call__(self,observation):
        time.sleep(.3)
        return {"plan":plan(),"action":"retreat","probability":.9,"bank_version_at_observation":observation.context["bank"]["version"],"served_model":"mechanical-brain-test"}


class BrainComparisonContracts(unittest.TestCase):
    def test_motor_vocabulary_and_finite_numbers(self):
        for value in ("teleport",None,"run_code"):
            p=plan(value)
            with self.assertRaises(ValueError):validate_plan(p,set())
        for field in ("confidence","danger_score"):
            for value in (True,float("nan"),float("inf"),-1,2):
                p=plan();p[field]=value
                with self.assertRaises(ValueError):validate_plan(p,set())
        self.assertEqual(validate_plan(plan(),set())["action"],"retreat")

    def test_fresh_local_avoidance_overrides_brain_and_expired_local_does_not(self):
        local=ActionMailbox(1,1.4);brain=ActionMailbox(8,10)
        local.accept({"sequence":1,"observed_wall_s":1,"action":"retreat"},1.1)
        brain.accept({"sequence":1,"observed_wall_s":0,"action":"continue"},1.1)
        self.assertEqual(choose_motor(1.2,local,brain,True,True)[2],"cerebellum")
        self.assertEqual(choose_motor(3,local,brain,True,True)[2],"brain")

    def test_untriggered_local_continue_does_not_veto_brain_retreat(self):
        local=ActionMailbox(1,1.4);brain=ActionMailbox(8,10)
        local.accept({"sequence":2,"observed_wall_s":1,"action":"continue","model_intervention":False},1.1)
        brain.accept({"sequence":1,"observed_wall_s":.2,"action":"retreat"},1.1)
        self.assertEqual(choose_motor(1.2,local,brain,True,True)[:3],("retreat",1,"brain"))

    def test_new_explicit_clear_can_release_an_older_brain_stop(self):
        local=ActionMailbox(1,1.4);brain=ActionMailbox(8,10)
        local.accept({"sequence":2,"observed_wall_s":1,"action":"continue","model_intervention":True,"reflex_id":"continue"},1.1)
        brain.accept({"sequence":1,"observed_wall_s":.2,"action":"brake"},1.1)
        self.assertEqual(choose_motor(1.2,local,brain,True,True)[:3],("continue",1,"cerebellum"))

    def test_worker_timings_use_common_clock_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            origin=time.monotonic()
            worker=LatestWorker(lambda observation:{"action":"brake"},Path(tmp)/"trace.jsonl")
            worker.submit(Observation(1,0,0,np.zeros((2,2,3),dtype=np.uint8),{},"continue",{},origin))
            limit=time.monotonic()+1
            records=[]
            while not records and time.monotonic()<limit:
                time.sleep(.01);records=worker.drain()
            worker.close()
            self.assertTrue(records)
            self.assertGreaterEqual(records[0]["received_wall_s"],records[0]["call_started_wall_s"])
            self.assertGreaterEqual(records[0]["capture_to_call_s"],0)

    def test_commanded_brake_is_not_a_mechanical_stop(self):
        scenario=Scenario.generate("hand_close",541);event=scenario.events()[0]
        at=event["onset_sim_s"]+.1
        apps=[{"accepted":True,"motor_changed":True,"action":"brake","source":"brain","observed_sim_s":at,"applied_sim_s":at,"applied_wall_s":at}]
        states=[{"sim_s":at+i*.05,"wall_s":at+i*.05,"eef_position_m":[0,0,0],"measured_eef_velocity_m_s":[.2,0,0]} for i in range(40)]
        result=motion_responses(scenario,apps,states)[0]
        self.assertIsNone(result["command_to_motion_s"])
        self.assertEqual(result["motion_status"],"criterion_not_reached_within_window")

    def test_event_scoring_includes_selected_brain_commands_and_excludes_unselected(self):
        from jev_reflex.run import reactions
        scenario=Scenario.generate("hand_close",541)
        event=scenario.events()[0];at=event["onset_sim_s"]+.1
        def record(source,changed):
            return {"accepted":True,"motor_changed":changed,"action":"retreat","source":source,
                    "observed_sim_s":at,"applied_sim_s":at+.5,"observation_age_s":.5}
        self.assertEqual(reactions(scenario,[record("brain",True)])[0]["first_source"],"brain")
        self.assertIsNone(reactions(scenario,[record("brain",False)])[0]["first_action"])
        self.assertEqual(reactions(scenario,[record("cerebellum",True)])[0]["first_source"],"cerebellum")

    def test_focused_events_are_seeded_and_do_not_track_robot(self):
        for kind in ("hand_close","slider_fast","occlusion_brief","target_jump"):
            a,b=Scenario.generate(kind,541),Scenario.generate(kind,541)
            self.assertEqual(a.serializable(),b.serializable())
            self.assertEqual(len(a.events()),1)
            self.assertGreaterEqual(a.onset,10.5)
            self.assertLessEqual(a.onset,12.5)

    @unittest.skipUnless(Path(os.environ.get("PANDA_ASSET_ROOT","/opt/omnijev-sim"),"panda.xml").exists(),"Requires Panda physics")
    def test_hold_reference_does_not_follow_gravity_drift(self):
        from jev_reflex.world import World
        for action in ("brake","watchdog_hold"):
            world=World(Path(os.environ["PANDA_ASSET_ROOT"]),Scenario.generate("quiet",541),256)
            try:
                initial=world.position();reference=world.qref.copy()
                for _ in range(160):
                    world.motor(action);world.step()
                np.testing.assert_array_equal(world.qref,reference)
                self.assertLess(float(np.linalg.norm(world.position()-initial)),.005)
                self.assertEqual(world.evaluate()["human_contact_N"],0)
                self.assertEqual(world.evaluate()["object_contact_N"],0)
            finally:world.close()

    @unittest.skipUnless(Path(os.environ.get("PANDA_ASSET_ROOT","/opt/omnijev-sim"),"panda.xml").exists(),"Requires native MuJoCo/GL and Panda")
    def test_pure_brain_never_creates_an_omni_client_and_changes_real_motor(self):
        args=argparse.Namespace(asset_root=Path(os.environ["PANDA_ASSET_ROOT"]),image_size=256,api_timeout=2,max_observation_age=1.2,command_ttl=1.4,seconds=1.5,render_hz=8,brain_timeout=2,brain_interval=0,max_brain_age=8,brain_command_ttl=10,freeze_bank=True)
        with tempfile.TemporaryDirectory() as tmp,patch("jev_reflex.run.HTTPReflex",side_effect=AssertionError("Pure brain must not create Omni client")),patch("jev_reflex.run.VisualBrain",DirectBrain):
            out=Path(tmp)/"episode"
            result=episode(args,Scenario.generate("quiet",541),"vlm",out)
            self.assertFalse((out/"api.jsonl").exists())
            self.assertEqual(result["api_results"],0)
            self.assertGreater(result["motor_action_ticks"].get("retreat",0),0)
            records=[json.loads(s) for s in (out/"motor_applications.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["motor_changed"] and r["source"]=="brain" for r in records))
            self.assertGreater(result["real_time_factor"],.9)

if __name__=="__main__":unittest.main()
