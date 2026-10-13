import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bd = benchmark.get_benchmark_dict()
suite = bd["libero_10"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file),
    camera_heights=256, camera_widths=256)
obs = env.reset()
sim = env.env.sim
print("bodies with z:")
for i in range(sim.model.nbody):
    n = sim.model.body_id2name(i) or ""
    if any(k in n for k in ("table", "soup", "sauce", "basket", "counter", "milk", "cream", "butter", "ketchup", "juice")):
        print(f"  {n:40s} z={sim.data.body_xpos[i][2]:.3f}")
env.close()
