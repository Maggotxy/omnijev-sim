"""Smoke test: instantiate LIBERO-10 env, render frames."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from PIL import Image

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
suite = bd["libero_10"]()
print("tasks:", suite.n_tasks)
for i in range(min(3, suite.n_tasks)):
    t = suite.get_task(i)
    print(i, t.name, "|", t.language)

task = suite.get_task(0)
env_args = {
    "bddl_file_name": os.path.join(get_libero_path("bddl_files"),
                                   task.problem_folder, task.bddl_file),
    "camera_heights": 512,
    "camera_widths": 512,
}
env = OffScreenRenderEnv(**env_args)
obs = env.reset()
print("obs keys:", [k for k in obs.keys() if "image" in k])

img = obs["agentview_image"][::-1]  # mujoco renders upside down
Image.fromarray(img).save("/opt/LIBERO/out_task0_agent.png")
if "robot0_eye_in_hand_image" in obs:
    Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save("/opt/LIBERO/out_task0_wrist.png")

# step a few times with zero action
for _ in range(10):
    obs, r, done, _ = env.step(np.zeros(7))
img = obs["agentview_image"][::-1]
Image.fromarray(img).save("/opt/LIBERO/out_task0_agent_t10.png")
print("OK, reward:", r, "done:", done)
env.close()
