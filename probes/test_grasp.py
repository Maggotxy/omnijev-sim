"""Ground-truth reach+grasp test: is the sim itself capable of grasping?"""
import numpy as np
import mujoco
from sim import Sim

sim = Sim()
sim.reset(rng=np.random.default_rng(1))
cube = sim.cube_pos().copy()
ee = sim.ee_pos().copy()
lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
hand = sim.ee_body

def finger_mid():
    return (sim.data.xpos[lf] + sim.data.xpos[rf]) / 2

print("cube", cube.round(4))
print("hand origin", ee.round(4), "finger mid", finger_mid().round(4),
      "-> hand-to-fingertip offset", (finger_mid() - ee).round(4))
print("finger sep", np.linalg.norm(sim.data.xpos[lf] - sim.data.xpos[rf]).round(4))

def servo(target, steps=400, gain=0.5, tol=0.004, label=""):
    for i in range(steps):
        cur = finger_mid()
        err = target - cur
        if np.linalg.norm(err) < tol:
            break
        jacp = np.zeros((3, sim.model.nv))
        mujoco.mj_jacBody(sim.model, sim.data, jacp, None, hand)
        J = jacp[:, :7]
        lam = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err * gain)
        lo = np.array([sim.model.jnt_range[i][0] + 1e-3 for i in range(7)])
        hi = np.array([sim.model.jnt_range[i][1] - 1e-3 for i in range(7)])
        sim.q_target = np.clip(sim.q_target + dq, lo, hi)
        sim.data.ctrl[:7] = sim.q_target
        sim.step(20)
    print(f"  servo[{label}] target={target.round(3)} -> finger_mid={finger_mid().round(3)} "
          f"err={np.linalg.norm(target - finger_mid()):.4f} steps={i}")

# 1) approach from above
servo(np.array([cube[0], cube[1], cube[2] + 0.12]), label="above")
print("  finger_mid now", finger_mid().round(4), "cube", sim.cube_pos().round(4))
# 2) descend so the pads straddle the cube centre
servo(np.array([cube[0], cube[1], cube[2] + 0.005]), label="grasp height")
print("  finger_mid now", finger_mid().round(4), "cube", sim.cube_pos().round(4))
# 3) close
sim.set_gripper(False)
sim.settle(1.0)
print("holding:", sim.holding())
# 4) lift
servo(np.array([cube[0], cube[1], cube[2] + 0.20]), label="lift")
print("after lift: cube", sim.cube_pos().round(4), "holding:", sim.holding())
# 5) move to zone xy
zone = sim.zone_pos().copy()
servo(np.array([zone[0], zone[1], cube[2] + 0.20]), label="carry")
print("carried: cube", sim.cube_pos().round(4), "zone", zone.round(4))
sim.set_gripper(True)
sim.settle(0.8)
print("after release: cube", sim.cube_pos().round(4), "dist_xy_to_zone",
      np.linalg.norm(sim.cube_pos()[:2] - zone[:2]).round(4), "success:", sim.success())
