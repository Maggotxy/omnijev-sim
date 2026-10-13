"""Debug: numeric image jacobian at the stuck EE pose."""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

IMG = 512
bd = benchmark.get_benchmark_dict()
suite = bd["libero_10"]()
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                task.problem_folder, task.bddl_file),
    camera_heights=IMG, camera_widths=IMG)
obs = env.reset()
sim = env.env.sim
cam_id = sim.model.camera_name2id("agentview")
fovy = np.radians(sim.model.cam_fovy[cam_id])
F_PX = (IMG / 2) / np.tan(fovy / 2)
xpos = sim.data.cam_xpos[cam_id]
xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)
print("cam pos", xpos.round(3), "fovy", sim.model.cam_fovy[cam_id])
print("xmat:\n", xmat.round(3))

def project(pt):
    pc = xmat.T @ (np.asarray(pt, float) - xpos)
    return np.array([IMG / 2 + pc[0] / (-pc[2]) * F_PX,
                     IMG / 2 - pc[1] / (-pc[2]) * F_PX])

pt = np.array([0.14, -0.15, 0.80])
print("project(pt) =", project(pt).round(1))
J = np.zeros((2, 3))
base = project(pt)
for i in range(3):
    dq = np.zeros(3); dq[i] = 0.01
    J[:, i] = (project(pt + dq) - base) / 0.01
print("J (dpx/dworld):\n", J.round(1))
e = np.array([0.0, 207.0])
d = J.T @ (e / F_PX)
print("J.T@e =", d.round(4))

# also check: which world dir increases py most? sample unit vectors
best = None
for th in np.linspace(0, 2 * np.pi, 16):
    w = np.array([np.cos(th), np.sin(th), 0])
    dpy = J[1] @ w
    dpx = J[0] @ w
    print(f"w=({w[0]:+.2f},{w[1]:+.2f}) -> dpx={dpx:+.1f} dpy={dpy:+.1f}")
env.close()
