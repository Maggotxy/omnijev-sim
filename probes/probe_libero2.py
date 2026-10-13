"""Probe LIBERO env: controller, action mapping, gripper sign, camera orientation."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
suite = bd["libero_10"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file),
    camera_heights=512, camera_widths=512)
obs = env.reset()
env.env.hard_reset = False
print("action dim:", env.env.action_dim)
print("eef pos:", obs["robot0_eef_pos"].round(3))
print("gripper qpos:", obs["robot0_gripper_qpos"].round(4))

def run(action, steps=20):
    global obs
    for _ in range(steps):
        obs, r, d, _ = env.step(action)
    return obs

# 1. +x action -> where does eef go in world?
p0 = obs["robot0_eef_pos"].copy()
run(np.array([0.6, 0, 0, 0, 0, 0, 0]))
p1 = obs["robot0_eef_pos"].copy()
print("action +x -> eef delta:", (p1 - p0).round(3))
Image.fromarray(obs["agentview_image"][::-1]).save("/opt/LIBERO/calib_plusx.png")

# 2. +y
run(np.array([0, 0.6, 0, 0, 0, 0, 0]))
p2 = obs["robot0_eef_pos"].copy()
print("action +y -> eef delta:", (p2 - p1).round(3))
Image.fromarray(obs["agentview_image"][::-1]).save("/opt/LIBERO/calib_plusy.png")

# 3. +z
run(np.array([0, 0, 0.6, 0, 0, 0, 0]))
p3 = obs["robot0_eef_pos"].copy()
print("action +z -> eef delta:", (p3 - p2).round(3))

# 4. gripper sign
q0 = obs["robot0_gripper_qpos"].copy()
run(np.array([0, 0, 0, 0, 0, 0, 1.0]))
q1 = obs["robot0_gripper_qpos"].copy()
print("gripper action +1 -> qpos delta:", (q1 - q0).round(4), "(qpos now", q1.round(4), ")")
run(np.array([0, 0, 0, 0, 0, 0, -1.0]))
q2 = obs["robot0_gripper_qpos"].copy()
print("gripper action -1 -> qpos delta:", (q2 - q1).round(4), "(qpos now", q2.round(4), ")")

# camera pose
cam_id = env.env.sim.model.camera_name2id("agentview")
print("agentview pos:", env.env.sim.data.cam_xpos[cam_id].round(3))
env.close()
