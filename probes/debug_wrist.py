"""Debug the wrist cameras: pose + image stats + save frames."""
import numpy as np
import mujoco
from PIL import Image
from sim import Sim

sim = Sim()
sim.reset(rng=np.random.default_rng(1))
cube = sim.cube_pos().copy()

lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

def finger_mid():
    return (sim.data.xpos[lf] + sim.data.xpos[rf]) / 2

def servo(target, steps=400, gain=0.5, tol=0.004):
    for _ in range(steps):
        err = target - finger_mid()
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, None, sim.ee_body)
        J = jacp[:, :7]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err * gain)
        lo = np.array([sim.model.jnt_range[k][0] + 1e-3 for k in range(7)])
        hi = np.array([sim.model.jnt_range[k][1] - 1e-3 for k in range(7)])
        sim.q_target = np.clip(sim.q_target + dq, lo, hi)
        sim.data.ctrl[:7] = sim.q_target
        sim.step(20)

servo(np.array([cube[0], cube[1], cube[2] + 0.10]))
servo(np.array([cube[0], cube[1], cube[2] + 0.05]))
print("tip:", finger_mid().round(4), "cube:", sim.cube_pos().round(4))
print("hand xpos:", sim.data.xpos[sim.ee_body].round(4))
print("hand xmat:", sim.data.xmat[sim.ee_body].round(3))

for cam in ("wrist", "wrist_alt"):
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    img = sim.render(cam)
    a = img.astype(int)
    print(f"{cam}: id={cid} xpos={sim.data.cam_xpos[cid].round(3)} "
          f"mean RGB={a.mean(axis=(0,1)).round(1)} max R={a[:,:,0].max()}")
    Image.fromarray(img).save(f"calib_{cam}.png")
img = sim.render("agentview")
Image.fromarray(img).save("calib_agent.png")
print("saved calib_*.png")
