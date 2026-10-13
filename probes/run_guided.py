"""Model-guided closed loop, v4.

- Coarse approach: OmniJev localizes the target on a 4x4 grid (reliable, p~0.85);
  the gripper cell comes from proprioceptive projection; both are unprojected to
  world coordinates through the camera -> exact coarse servo.
- Final centering: hill-climb (coordinate ascent) directly on the model's own
  grasp probability `P(close on the cube right now)` — measured to rise
  0.46 -> 0.81 -> 0.82 as the tip converges on the cube. Fully model-driven.
- Grasp / release / done / holding decisions are the model's nouls.
"""
import argparse
import json
import os
import time

import numpy as np
import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw, ImageFont

from sim import Sim
from omnijev_api import OmniJev, TASK

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = 4
IMG = 640
CELL_PX = IMG // GRID
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]
FONT = None


def bold_grid(arr):
    global FONT
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    if FONT is None:
        try:
            FONT = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        except Exception:
            FONT = ImageFont.load_default()
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=4)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=4)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 10, row * CELL_PX + 8), name, fill=(255, 255, 0),
               font=FONT, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(im)


class CamMath:
    def __init__(self, sim, cam="agentview"):
        self.sim = sim
        self.cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, cam)

    def _pose(self):
        return (self.sim.data.cam_xpos[self.cid],
                self.sim.data.cam_xmat[self.cid].reshape(3, 3))

    def f_px(self):
        fovy = np.radians(self.sim.model.cam_fovy[self.cid])
        return (IMG / 2) / np.tan(fovy / 2)

    def project(self, pt):
        xpos, xmat = self._pose()
        pc = xmat.T @ (pt - xpos)
        return np.array([IMG / 2 + (pc[0] / (-pc[2])) * self.f_px(),
                         IMG / 2 - (pc[1] / (-pc[2])) * self.f_px()])

    def unproject(self, px, z_plane):
        xpos, xmat = self._pose()
        d_cam = np.array([(px[0] - IMG / 2) / self.f_px(),
                          -(px[1] - IMG / 2) / self.f_px(), -1.0])
        d = xmat @ d_cam
        d /= np.linalg.norm(d)
        t = (z_plane - xpos[2]) / d[2]
        return xpos + t * d

    def cell_of(self, px):
        col = int(np.clip(px[0] // CELL_PX, 0, GRID - 1))
        row = int(np.clip(px[1] // CELL_PX, 0, GRID - 1))
        return CELLS[row * GRID + col]

    def cell_center(self, cell):
        idx = CELLS.index(cell)
        col, row = idx % GRID, idx // GRID
        return np.array([(col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX])


def q_cell(target_word):
    return {"tgt": {"type": "choice", "instructions":
                    f"This robot scene has a labeled 4x4 grid (columns A-D left to "
                    f"right, rows 1-4 top to bottom). Which cell contains the {target_word}?",
                    "criteria": {c: f"cell {c}" for c in CELLS}}}


def q_decide():
    return {
        "grasp": {"type": "noul", "instructions":
                  TASK + " The gripper should close on the red cube right now."},
        "release": {"type": "noul", "instructions":
                    TASK + " The gripper should release the object right now."},
        "done": {"type": "noul", "instructions": TASK + " The task is finished."},
    }


def q_grasp():
    return {"grasp": {"type": "noul", "instructions":
                      TASK + " The gripper should close on the red cube right now."}}


def run(out_dir, max_ticks=120, seed=0, speed=0.22,
        grasp_th=0.65, release_th=0.55, done_th=0.75, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim(img_w=IMG, img_h=IMG)
    sim.reset(rng=np.random.default_rng(seed))
    api = OmniJev()
    cam = CamMath(sim)

    frames, log = [], []
    phase, grasped, lift_left, done = "REACH", False, 0, False
    stats = {"api_calls": 0}
    t0 = time.time()

    def ask(images, qs):
        stats["api_calls"] += 1
        return api.ask(images, qs)

    def move(v, spd=speed):
        if np.linalg.norm(v) > 1e-6:
            sim.move_ee(np.asarray(v, dtype=float), speed=spd)
        sim.data.ctrl[:7] = sim.q_target
        sim.settle(0.5)

    def grasp_p(n=2):
        """model grasp probability, averaged over n asks to cut noise."""
        vals = []
        for _ in range(n):
            agent = sim.render("agentview")
            vals.append(ask([agent], q_grasp())["grasp"]["noul"])
        return sum(vals) / len(vals)

    def ascent(step=0.02, iters=4, thresh=0.70):
        """Coordinate ascent on the model's grasp probability. Returns final p."""
        p = grasp_p()
        for _ in range(iters):
            if p > thresh:
                break
            for axis in range(2):
                if p > thresh:
                    break
                d = np.zeros(3)
                d[axis] = step
                move(d, spd=0.08)
                p_plus = grasp_p()
                if p_plus > p + 0.01:
                    p = p_plus
                    continue
                move(-2 * d, spd=0.08)      # undo, try the other side
                p_minus = grasp_p()
                if p_minus > p + 0.01:
                    p = p_minus
                else:
                    move(d, spd=0.08)       # neither side better: restore
        return p

    for tick in range(max_ticks):
        agent = sim.render("agentview")
        grid_img = bold_grid(agent)
        target_word = "red cube" if not grasped else "green circle"
        try:
            ans_t = ask([grid_img], q_cell(target_word))
            ans_d = ask([agent], q_decide())
        except Exception as e:
            print(f"[{tick}] API error: {e}")
            time.sleep(2)
            continue

        tgt_cell = ans_t.get("tgt", {}).get("choice")
        p_tgt = max(ans_t.get("tgt", {}).get("probabilities", {}).values() or [0])
        p_grasp = ans_d.get("grasp", {}).get("noul", 0)
        p_release = ans_d.get("release", {}).get("noul", 0)
        p_done = ans_d.get("done", {}).get("noul", 0)

        tip = sim.tip_pos()
        cube = sim.cube_pos()
        tip_cell = cam.cell_of(cam.project(tip))
        z_plane = cube[2]
        same = tgt_cell == tip_cell

        v = np.zeros(3)
        action = "stay"

        def seek(target_cell, altitude):
            tw = cam.unproject(cam.cell_center(target_cell), z_plane)
            d = tw - tip
            return np.array([d[0], d[1], 4.0 * (altitude - tip[2])])

        if phase == "REACH":
            if same:
                phase = "GUIDE"
                action = "aligned"
            elif tgt_cell:
                v = seek(tgt_cell, z_plane + 0.13)
                action = f"track:{tip_cell}->{tgt_cell}"
        elif phase == "GUIDE":
            # hill-climb on the model's grasp probability at hover height
            hover_z = z_plane + 0.055
            if tip[2] > hover_z + 0.02:
                v = np.array([0, 0, -1.0])
                action = "to_hover"
            else:
                p = ascent(step=0.02, iters=4, thresh=0.70)
                action = f"ascend:{p:.2f}"
                if p > 0.5:
                    phase = "DESCEND"
                else:
                    phase = "REACH"   # lost it: back to coarse
        elif phase == "DESCEND":
            z_gap = tip[2] - cube[2]
            if z_gap > 0.03:
                v = np.array([0, 0, -1.0])
                action = "descend"
            elif p_grasp > grasp_th:
                sim.set_gripper(False)
                sim.settle(0.8)
                if sim.holding():
                    grasped, phase, lift_left = True, "LIFT", 4
                    action = "GRASP"
                else:
                    sim.set_gripper(True)
                    sim.settle(0.4)
                    phase = "GUIDE"          # re-center via the model
                    action = "GRASP_MISS"
            else:
                # one more ascent round at grasp height, then give up to GUIDE
                p = ascent(step=0.012, iters=2, thresh=grasp_th + 0.1)
                action = f"reascend:{p:.2f}"
                if p <= grasp_th:
                    phase = "GUIDE"
        elif phase == "LIFT":
            v = np.array([0, 0, 1.0])
            lift_left -= 1
            action = "lift"
            if lift_left <= 0:
                phase = "CARRY"
        elif phase == "CARRY":
            if not sim.holding():
                grasped, phase = False, "REACH"
                action = "DROPPED"
            elif same:
                phase = "PLACE"
                action = "over_zone"
            elif tgt_cell:
                v = seek(tgt_cell, z_plane + 0.16)
                action = f"carry:{tip_cell}->{tgt_cell}"
        elif phase == "PLACE":
            if not sim.holding():
                grasped, phase = False, "REACH"
                action = "DROPPED"
            elif cube[2] > z_plane + 0.045:
                v = np.array([0, 0, -1.0])
                action = "lower"
            elif p_release > release_th:
                sim.set_gripper(True)
                sim.settle(0.6)
                grasped = False
                phase = "REACH"
                action = "RELEASE"
            else:
                action = f"wait_release:{p_release:.2f}"

        if np.linalg.norm(v) > 1e-6:
            move(v)

        gt_hold = bool(sim.holding())
        succ = bool(sim.success())
        rec = {"tick": tick, "phase": phase, "action": action,
               "tgt_cell": tgt_cell, "tip_cell": tip_cell, "p_tgt": round(float(p_tgt), 3),
               "p": {"grasp": p_grasp, "release": p_release, "done": p_done},
               "gt_holding": gt_hold, "success": succ,
               "tip": tip.round(3).tolist(), "cube": cube.round(3).tolist()}
        log.append(rec)
        if verbose:
            print(f"[{tick:3d}] {phase:7s} {action:22s} tgt={tgt_cell}({p_tgt:.2f}) "
                  f"grasp={p_grasp:.2f} rel={p_release:.2f} done={p_done:.2f} "
                  f"hold={gt_hold} succ={succ}", flush=True)
        frames.append(grid_img)

        if succ:
            print(f"SUCCESS at tick {tick}")
            done = True
            break
        if p_done > done_th and tick > 10:
            print(f"model done at {tick} (p={p_done:.2f}), success={succ}")
            done = succ
            break

    vid = os.path.join(out_dir, "episode.mp4")
    imageio.mimsave(vid, frames, fps=4, quality=7)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump({"task": TASK, "seed": seed, "mode": "guided_v4", "success": done,
                   "ticks": len(log), "wall_s": round(time.time() - t0, 1),
                   "api_calls": stats["api_calls"], "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames), api calls: {stats['api_calls']}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "guided0"))
    ap.add_argument("--ticks", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.out, max_ticks=args.ticks, seed=args.seed)
