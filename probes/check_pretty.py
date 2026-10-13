"""Sanity check the pretty scene: render beauty frame, save PNG, ask VLM to describe."""
import numpy as np
import mujoco
from PIL import Image
from sim import Sim

sim = Sim(scene_path="pretty_scene.xml", img_w=640, img_h=640)
sim.reset(rng=np.random.default_rng(1))
r = mujoco.Renderer(sim.model, height=540, width=960)
r.update_scene(sim.data, camera="beauty")
img = r.render()
Image.fromarray(img).save("out/beauty_check.png")
print("beauty frame:", img.shape, "mean:", img.mean().round(1))
r2 = mujoco.Renderer(sim.model, height=640, width=640)
r2.update_scene(sim.data, camera="agentview")
Image.fromarray(r2.render()).save("out/agent_check.png")
print("agent frame saved")
