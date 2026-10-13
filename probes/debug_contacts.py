"""Instrument the grasp descent: report every contact involving the cube."""
import numpy as np
import mujoco
from sim import Sim

sim = Sim()
sim.reset(rng=np.random.default_rng(1))
cube = sim.cube_pos().copy()
lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
cg = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_red_geom")

def gname(g):
    n = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, g)
    b = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_BODY, sim.model.geom_bodyid[g])
    return f"{b}/{n or g}"

def finger_mid():
    return (sim.data.xpos[lf] + sim.data.xpos[rf]) / 2

def servo(target, steps=400, gain=0.5, tol=0.004):
    for i in range(steps):
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
        for c in range(sim.data.ncon):
            con = sim.data.contact[c]
            if con.geom1 == cg or con.geom2 == cg:
                other = con.geom2 if con.geom1 == cg else con.geom1
                print(f"  step {i}: cube contact with {gname(other)} "
                      f"fm={finger_mid().round(3)} cube={sim.cube_pos().round(4)}")

print("cube start", cube.round(4))
servo(np.array([cube[0], cube[1], cube[2] + 0.12]))
print("after above: fm", finger_mid().round(4), "cube", sim.cube_pos().round(4))
servo(np.array([cube[0], cube[1], cube[2] + 0.005]))
print("after descend: fm", finger_mid().round(4), "cube", sim.cube_pos().round(4))
