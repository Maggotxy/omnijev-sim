"""Ground-truth check of the model's grid-cell localization on the agent view.
Renders at 640px with a bold grid, asks cells, compares to projected truth."""
import json
import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont
from sim import Sim
from omnijev_api import OmniJev

sim = Sim(img_w=640, img_h=640)
sim.reset(rng=np.random.default_rng(1))
api = OmniJev()

GRID = 4
IMG = 640
CELL_PX = IMG // GRID
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]

def bold_grid(arr):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=4)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=4)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 10, row * CELL_PX + 8), name, fill=(255, 255, 0), font=font,
               stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(im)

def project(world_pt, cam_name="agentview"):
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    xpos = sim.data.cam_xpos[cid]
    xmat = sim.data.cam_xmat[cid].reshape(3, 3)
    fovy = np.radians(sim.model.cam_fovy[cid])
    f_px = (IMG / 2) / np.tan(fovy / 2)
    pc = xmat.T @ (world_pt - xpos)
    px = IMG / 2 + (pc[0] / (-pc[2])) * f_px
    py = IMG / 2 - (pc[1] / (-pc[2])) * f_px
    return np.array([px, py])

def cell_of(px):
    col = int(np.clip(px[0] // CELL_PX, 0, GRID - 1))
    row = int(np.clip(px[1] // CELL_PX, 0, GRID - 1))
    return CELLS[row * GRID + col]

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

cube = sim.cube_pos().copy()
positions = [
    np.array([cube[0], cube[1], cube[2] + 0.13]),          # right over the cube
    np.array([0.45, -0.15, 0.60]),
    np.array([0.60, 0.15, 0.55]),
]
hits_g = hits_t = 0
for i, p in enumerate(positions):
    servo(p)
    img = bold_grid(sim.render("agentview"))
    Image.fromarray(img).save(f"cell_test_{i}.png")
    true_g = cell_of(project(tip()))
    true_t = cell_of(project(sim.cube_pos()))
    ans = api.ask([img], {
        "grip": {"type": "choice",
                 "instructions": "This image has a labeled 4x4 grid (columns A-D left to right, rows 1-4 top to bottom). Which cell contains the robot gripper tip?",
                 "criteria": {c: f"cell {c}" for c in CELLS}},
        "tgt": {"type": "choice",
                "instructions": "Same grid. Which cell contains the red cube?",
                "criteria": {c: f"cell {c}" for c in CELLS}},
    })
    mg = ans["grip"].get("choice")
    mt = ans["tgt"].get("choice")
    hits_g += mg == true_g
    hits_t += mt == true_t
    print(f"[{i}] tip={tip().round(3)} true_grip={true_g} model={mg} | "
          f"true_cube={true_t} model={mt} (p={max(ans['tgt'].get('probabilities',{}).values() or [0]):.2f})")
print(f"accuracy: grip {hits_g}/{len(positions)} cube {hits_t}/{len(positions)}")
