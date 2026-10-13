"""Render a probe frame (agentview + wrist) to verify the scene before the loop."""
import os
from PIL import Image
from sim import Sim

ROOT = os.path.dirname(os.path.abspath(__file__))
sim = Sim()
a = sim.render("agentview")
w = sim.render_wrist()
Image.fromarray(a).save(os.path.join(ROOT, "probe_agent.png"))
Image.fromarray(w).save(os.path.join(ROOT, "probe_wrist.png"))
print("ee:", sim.ee_pos().round(3), "cube:", sim.cube_pos().round(3), "zone:", sim.zone_pos().round(3))
print("grip_act:", sim.grip_act, "ee_body:", sim.ee_body)
print("OK")
