"""Calibrate the wrist camera: which orientation sees the cube, and how do
image pixels map to world xy? Prints the affine mapping for run_grid."""
import numpy as np
import mujoco
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

def red_centroid(img):
    a = img.astype(int)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mask = (r - np.maximum(g, b) > 25) & (r > 60)
    n = int(mask.sum())
    if n < 30:
        return None, n
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]), n

# hover directly above the cube
servo(np.array([cube[0], cube[1], cube[2] + 0.10]))
servo(np.array([cube[0], cube[1], cube[2] + 0.05]))
print("tip:", finger_mid().round(4), "cube:", sim.cube_pos().round(4))

for cam in ("wrist", "wrist_alt"):
    img = sim.render(cam)
    c, n = red_centroid(img)
    print(f"{cam}: red px={n} centroid={None if c is None else c.round(1)} "
          f"(center={img.shape[1]/2},{img.shape[0]/2})")

# affine mapping for the better camera: nudge +x and +y in world, measure centroid shift
CAM = "wrist"
base = sim.render(CAM)
c0, n0 = red_centroid(base)
if c0 is None:
    CAM = "wrist_alt"
    base = sim.render(CAM)
    c0, n0 = red_centroid(base)
print("using", CAM)

rows = []
for d_world in (np.array([0.03, 0, 0]), np.array([0, 0.03, 0])):
    target = finger_mid() + d_world
    servo(target, steps=60, tol=0.002)
    img = sim.render(CAM)
    c1, n1 = red_centroid(img)
    if c1 is None:
        print("lost cube for world delta", d_world)
        rows.append(None)
    else:
        shift = c1 - c0
        rows.append(shift)
        print(f"world {d_world.round(3)} -> img shift {shift.round(1)} px  (px={n1})")
    servo(finger_mid() - d_world, steps=80, tol=0.003)
print("tip:", finger_mid().round(4), "cube:", sim.cube_pos().round(4))
print("AFFINE rows (img px shift per +3cm world):", rows)
