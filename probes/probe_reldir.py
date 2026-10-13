"""Probe: can the model judge the cube's direction relative to the gripper tip?"""
import numpy as np
import mujoco
from sim import Sim
from omnijev_api import OmniJev, TASK

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

DIRS = ["center", "left", "right", "up", "down",
        "up-left", "up-right", "down-left", "down-right"]

def ask_dir():
    img = sim.render("agentview")
    ans = api.ask([img], {"d": {"type": "choice", "instructions":
        TASK + " Look at the gripper tip and the red cube. In the image, where is the red cube relative to the gripper tip?",
        "criteria": {d: f"the cube is {d} of the tip" for d in DIRS}}})
    d = ans["d"]
    return d.get("choice"), max(d.get("probabilities", {}).values() or [0])

cube = sim.cube_pos().copy()
H = cube[2] + 0.06
# (name, world offset of tip relative to cube, expected image direction of cube)
# agent cam: image right = (0.45,0.89,0)/|, image up xy part = (-0.42,0.21)
tests = [
    ("tip +x5 (cube left-ish in img)",  np.array([ 0.05, 0.0, 0.0])),
    ("tip -x5 (cube right-ish)",        np.array([-0.05, 0.0, 0.0])),
    ("tip +y5",                          np.array([0.0,  0.05, 0.0])),
    ("tip -y5",                          np.array([0.0, -0.05, 0.0])),
    ("tip +x3+y3",                       np.array([ 0.03, 0.03, 0.0])),
    ("centered",                         np.array([ 0.0,  0.0,  0.0])),
]
for name, off in tests:
    servo(np.array([cube[0] + off[0], cube[1] + off[1], H]))
    # expected image direction of cube relative to tip:
    d_world = -off  # cube relative to tip
    img_x = d_world[0] * 0.45 + d_world[1] * 0.89
    img_y = -(d_world[0] * -0.42 + d_world[1] * 0.21)  # image up xy ~(-0.42,0.21); y down positive
    ang = np.degrees(np.arctan2(img_y, img_x))
    sectors = {0: "right", 45: "down-right", 90: "down", 135: "down-left",
               180: "left", 225: "up-left", 270: "up", 315: "up-right"}
    a360 = (ang + 360) % 360
    expected = sectors[int(round(a360 / 45)) % 8 * 45] if min(a360, 360 - a360) > 22.5 else "center"
    if np.linalg.norm(off) < 0.01:
        expected = "center"
    choice, p = ask_dir()
    ok = "OK " if choice == expected else "XXX"
    print(f"{ok} {name}: expected={expected} model={choice} (p={p:.2f})")
