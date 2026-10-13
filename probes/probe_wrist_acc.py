"""Decisive test: wrist-camera 4x4 cell accuracy while the cube IS in view."""
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
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)

def bold_grid(arr):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=4)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=4)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 10, row * CELL_PX + 8), name, fill=(255, 255, 0),
               font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(im)

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

def project(pt, cam):
    cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    xpos = sim.data.cam_xpos[cid]
    xmat = sim.data.cam_xmat[cid].reshape(3, 3)
    fovy = np.radians(sim.model.cam_fovy[cid])
    f_px = (IMG / 2) / np.tan(fovy / 2)
    pc = xmat.T @ (pt - xpos)
    return np.array([IMG / 2 + (pc[0] / (-pc[2])) * f_px,
                     IMG / 2 - (pc[1] / (-pc[2])) * f_px])

def cell_of(px):
    if not (0 <= px[0] < IMG and 0 <= px[1] < IMG):
        return "OUT"
    col = int(px[0] // CELL_PX)
    row = int(px[1] // CELL_PX)
    return CELLS[row * GRID + col]

cube = sim.cube_pos().copy()
H = cube[2] + 0.13
tests = [(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4), (4, 4), (-4, -4), (8, 0), (0, 8)]
hits = 0
for dx, dy in tests:
    servo(np.array([cube[0] + dx / 100, cube[1] + dy / 100, H]))
    img = bold_grid(sim.render("wrist"))
    true_cell = cell_of(project(sim.cube_pos(), "wrist"))
    ans = api.ask([img], {"c": {"type": "choice", "instructions":
        "This wrist-camera image has a labeled 4x4 grid (columns A-D left to right, rows 1-4 top to bottom). Which cell contains the red cube?",
        "criteria": {c: f"cell {c}" for c in CELLS}}})
    model_cell = ans["c"].get("choice")
    p = max(ans["c"].get("probabilities", {}).values() or [0])
    ok = "OK " if model_cell == true_cell else "XXX"
    hits += model_cell == true_cell
    print(f"{ok} tip offset ({dx:+d},{dy:+d})cm: true={true_cell} model={model_cell} (p={p:.2f})")
print(f"accuracy {hits}/{len(tests)}")
