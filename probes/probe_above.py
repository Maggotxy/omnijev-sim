"""Map P(gripper directly above the cube) on a 5x5 sweep — last candidate signal."""
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

def servo(target, steps=300, gain=0.5, tol=0.003):
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

cube = sim.cube_pos().copy()
H = cube[2] + 0.055
print("cube:", cube.round(3))
for qname, instr in [("above", "The robot gripper is directly above the red cube."),
                     ("between", "The red cube is between the two gripper fingers.")]:
    print(f"--- {qname} ---")
    grid = np.zeros((5, 5))
    for iy, dy in enumerate(np.linspace(-0.04, 0.04, 5)):
        for ix, dx in enumerate(np.linspace(-0.04, 0.04, 5)):
            servo(np.array([cube[0] + dx, cube[1] + dy, H]))
            vals = []
            for _ in range(2):
                img = sim.render("agentview")
                vals.append(api.ask([img], {"q": {"type": "noul", "instructions": instr}})["q"]["noul"])
            grid[iy, ix] = sum(vals) / len(vals)
        print(" ".join(f"{v:.2f}" for v in grid[iy]))
    iy, ix = np.unravel_index(np.argmax(grid), grid.shape)
    print(f"peak {grid[iy, ix]:.2f} at dx={np.linspace(-4,4,5)[ix]} dy={np.linspace(-4,4,5)[iy]}; "
          f"center value {grid[2,2]:.2f}")
