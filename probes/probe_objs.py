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
    camera_heights=512, camera_widths=512)
obs = env.reset()
sim = env.env.sim
for n in ("alphabet_soup_1_main", "tomato_sauce_1_main", "basket_1_main",
          "milk_1_main", "cream_cheese_1_main", "butter_1_main",
          "ketchup_1_main", "orange_juice_1_main"):
    try:
        bid = sim.model.body_name2id(n)
        print(f"{n:30s} {sim.data.body_xpos[bid].round(3)}")
    except Exception as e:
        print(n, "ERR", e)
# also project alphabet soup into image to compare with model answer r2c0/r2c1
cam_id = sim.model.camera_name2id("agentview")
xpos = sim.data.cam_xpos[cam_id]
xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)
F = 256 / np.tan(np.radians(sim.model.cam_fovy[cam_id]) / 2)
for n in ("alphabet_soup_1_main", "tomato_sauce_1_main", "basket_1_main"):
    pt = sim.data.body_xpos[sim.model.body_name2id(n)]
    pc = xmat.T @ (pt - xpos)
    px = 256 + pc[0] / (-pc[2]) * F
    py = 256 - pc[1] / (-pc[2]) * F
    # region in 4x4 (0-1000): col = px/512*4, row = py/512*4
    print(f"{n:30s} px=({px:.0f},{py:.0f}) -> r{int(py/128)}c{int(px/128)}")
env.close()
