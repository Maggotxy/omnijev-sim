"""Check gripper opening range vs ctrl value and settle time."""
import numpy as np
import mujoco
from sim import Sim

sim = Sim()
sim.reset(rng=np.random.default_rng(1))
lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

def sep():
    return np.linalg.norm(sim.data.xpos[lf] - sim.data.xpos[rf])

act = sim.grip_act
print("actuator", act, "ctrlrange", sim.model.actuator_ctrlrange[act],
      "gainprm", sim.model.actuator_gainprm[act][:3], "biastype", sim.model.actuator_biastype[act])
fj1 = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
print("finger_joint1 range", sim.model.jnt_range[fj1])

for ctrl in (0, 128, 255):
    sim.data.ctrl[act] = ctrl
    sim.settle(2.0)
    q = sim.data.qpos[sim.model.jnt_qposadr[fj1]]
    print(f"ctrl={ctrl:4d} sep={sep():.4f} finger_joint1 qpos={q:.4f}")
