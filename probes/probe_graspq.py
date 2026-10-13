"""Probe which noul question best discriminates 'in grasp position'."""
import numpy as np
import mujoco
from sim import Sim
from omnijev_api import OmniJev

sim = Sim(img_w=640, img_h=640)
sim.reset(rng=np.random.default_rng(1))
api = OmniJev()

lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
def tip():
    return (sim.data.xpos[lf] + sim.data.xpos[rf]) / 2

def servo(target, steps=300, gain=0.5, tol=0.004):
    for _ in range(steps):
        err = target - tip()
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

Q = {
    "close_now":  "The gripper should close on the red cube right now.",
    "above":      "The robot gripper is directly above the red cube.",
    "touching":   "The gripper fingers are touching the red cube.",
    "between":    "The red cube is between the two gripper fingers.",
    "far":        "The gripper is far away from the red cube.",
}

def ask_all(tag):
    agent = sim.render("agentview")
    wrist = sim.render("wrist")
    for name, img in (("agent", agent), ("wrist", wrist)):
        ans = api.ask([img], {k: {"type": "noul", "instructions": v} for k, v in Q.items()})
        vals = " ".join(f"{k}={ans[k]['noul']:.2f}" for k in Q)
        print(f"{tag:14s} {name}: {vals}")

cube = sim.cube_pos().copy()
servo(np.array([cube[0], cube[1], cube[2] + 0.25]))
print("tip far above:", tip().round(3), "cube:", sim.cube_pos().round(3))
ask_all("high(+25cm)")
servo(np.array([cube[0], cube[1], cube[2] + 0.06]))
print("tip hover:", tip().round(3))
ask_all("hover(+6cm)")
servo(np.array([cube[0], cube[1], cube[2] + 0.02]))
print("tip grasp height:", tip().round(3))
ask_all("grasp_h(+2cm)")
