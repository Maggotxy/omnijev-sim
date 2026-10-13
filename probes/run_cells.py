"""Cell-servo closed loop, v2.

- Perception: OmniJev localizes the target (red cube / green circle) on a bold
  4x4 grid over a 640px agent view (measured 3/3 accurate, p=0.72-0.91).
- Proprioception: the gripper cell is computed by projecting the true tip
  position into the same camera (a robot knows its own kinematics).
- Motion: both cells are unprojected onto the table plane through the camera,
  giving an exact world-frame delta; the arm servos the tip there.
- Decisions (grasp / release / done / holding) stay with the model (noul).
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
        self.cam = cam
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
        """world point where the camera ray through px hits plane z=z_plane."""
        xpos, xmat = self._pose()
        d_cam = np.array([(px[0] - IMG / 2) / self.f_px(),
                          -(px[1] - IMG / 2) / self.f_px(), -1.0])
        d = xmat @ d_cam
        d = d / np.linalg.norm(d)
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


def questions_tgt(target_word):
    return {
        "tgt": {"type": "choice", "instructions":
                f"This robot scene has a labeled 4x4 grid (columns A-D left to right, "
                f"rows 1-4 top to bottom). Which cell contains the {target_word}?",
                "criteria": {c: f"cell {c}" for c in CELLS}},
    }


class Zoomer:
    """Two-stage zoom pointing (like the official grid demo):
    crop a cell, upscale, overlay a finer grid, ask again."""

    def __init__(self, api, cam):
        self.api = api
        self.cam = cam

    @staticmethod
    def crop_cell(img, cell, margin=0.30):
        idx = CELLS.index(cell)
        col, row = idx % GRID, idx // GRID
        m = int(CELL_PX * margin)
        x0 = max(col * CELL_PX - m, 0)
        y0 = max(row * CELL_PX - m, 0)
        x1 = min((col + 1) * CELL_PX + m, IMG)
        y1 = min((row + 1) * CELL_PX + m, IMG)
        crop = img[y0:y1, x0:x1]
        im = Image.fromarray(crop).resize((IMG, IMG), Image.BILINEAR)
        return np.array(im), (x0, y0, x1 - x0, y1 - y0)

    def ask_subcell(self, img, cell, target_word):
        """Returns (sub_cell, p, crop_info) for the target inside `cell`."""
        zoomed, info = self.crop_cell(img, cell)
        grid_img = bold_grid(zoomed)
        ans = self.api.ask([grid_img], questions_tgt(target_word))
        sub = ans.get("tgt", {}).get("choice")
        p = max(ans.get("tgt", {}).get("probabilities", {}).values() or [0])
        return sub, p, info

    def subcell_world(self, cell, sub_cell, info, z_plane):
        """world xy of a sub-cell center inside the zoomed crop of `cell`."""
        x0, y0, w, h = info
        c = self.cam.cell_center(sub_cell)          # in 640px zoomed coords
        orig_px = np.array([x0 + c[0] * w / IMG, y0 + c[1] * h / IMG])
        return self.cam.unproject(orig_px, z_plane)

    def tip_subcell(self, cell, info, tip):
        """which sub-cell of the zoomed crop the tip projects into (or None)."""
        px = self.cam.project(tip)
        x0, y0, w, h = info
        zx = (px[0] - x0) * IMG / w
        zy = (px[1] - y0) * IMG / h
        if not (0 <= zx < IMG and 0 <= zy < IMG):
            return None
        return self.cam.cell_of(np.array([zx, zy]))


def questions_decide():
    return {
        "grasp": {"type": "noul", "instructions":
                  TASK + " The gripper should close on the red cube right now."},
        "release": {"type": "noul", "instructions":
                    TASK + " The gripper should release the object right now."},
        "done": {"type": "noul", "instructions": TASK + " The task is finished."},
    }


def run(out_dir, max_ticks=100, seed=0, decision_hz=2.0, speed=0.22,
        grasp_th=0.7, release_th=0.7, done_th=0.75, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim(img_w=IMG, img_h=IMG)
    sim.reset(rng=np.random.default_rng(seed))
    api = OmniJev()
    cam = CamMath(sim)
    zoom = Zoomer(api, cam)

    frames, log = [], []
    phase, grasped, lift_left, done = "REACH", False, 0, False
    center_rounds = 0
    wait_grasp = 0
    t0 = time.time()

    for tick in range(max_ticks):
        agent = sim.render("agentview")
        grid_img = bold_grid(agent)
        target_word = "red cube" if not grasped else "green circle"
        try:
            ans_t = api.ask([grid_img], questions_tgt(target_word))
            ans_d = api.ask([agent], questions_decide())
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

        v = np.zeros(3)
        action = "stay"
        same = tgt_cell == tip_cell

        def seek(target_cell, altitude):
            """velocity toward the target cell's world point, holding altitude."""
            tw = cam.unproject(cam.cell_center(target_cell), z_plane)
            d = tw - tip
            out = np.array([d[0], d[1], 4.0 * (altitude - tip[2])])
            return out

        if phase == "REACH":
            servo_z = z_plane + 0.13
            if same:
                phase, center_rounds = "CENTER", 0
                action = "aligned"
            elif tgt_cell:
                v = seek(tgt_cell, servo_z)
                action = f"track:{tip_cell}->{tgt_cell}"
        elif phase == "CENTER":
            # zoom into the target cell and center the tip precisely
            servo_z = z_plane + 0.12
            try:
                sub, p_sub, info = zoom.ask_subcell(agent, tgt_cell, target_word)
            except Exception as e:
                print(f"[{tick}] zoom API error: {e}")
                sub, p_sub, info = None, 0, None
            if sub is None:
                action = "zoom_fail"
                center_rounds += 1
            else:
                tip_sub = zoom.tip_subcell(tgt_cell, info, tip)
                tgt_w = zoom.subcell_world(tgt_cell, sub, info, z_plane)
                d = tgt_w - tip
                action = f"center:{tip_sub}->{sub}({p_sub:.2f})"
                if tip_sub == sub or np.linalg.norm(d[:2]) < 0.015:
                    center_rounds += 1
                else:
                    v = np.array([d[0], d[1], 4.0 * (servo_z - tip[2])])
            if center_rounds >= 2:
                phase, wait_grasp = "DESCEND", 0
        elif phase == "DESCEND":
            z_gap = tip[2] - cube[2]
            if z_gap > 0.03:
                v = np.array([0, 0, -1.0])
                action = "descend"
                wait_grasp = 0
            elif p_grasp > grasp_th or (wait_grasp > 8 and p_grasp > 0.3):
                sim.set_gripper(False)
                sim.settle(0.8)
                if sim.holding():
                    grasped, phase, lift_left = True, "LIFT", 4
                    action = "GRASP"
                else:
                    sim.set_gripper(True)
                    sim.settle(0.4)
                    phase = "REACH"
                    action = "GRASP_MISS"
            else:
                wait_grasp += 1
                action = f"wait_grasp:{p_grasp:.2f}"
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
            else:
                z_gap = tip[2] - cube[2]  # tip vs held cube: lower both together
                if cube[2] > z_plane + 0.045:
                    v = np.array([0, 0, -1.0])
                    action = "lower"
                elif p_release > release_th:
                    sim.set_gripper(True)
                    sim.settle(0.6)
                    grasped = False
                    action = "RELEASE"
                    phase = "REACH"
                else:
                    action = f"wait_release:{p_release:.2f}"

        if np.linalg.norm(v) > 1e-6:
            sim.move_ee(v, speed=speed)
        sim.data.ctrl[:7] = sim.q_target
        sim.settle(1.0 / decision_hz)

        gt_hold = bool(sim.holding())
        succ = bool(sim.success())
        rec = {"tick": tick, "phase": phase, "action": action,
               "tgt_cell": tgt_cell, "tip_cell": tip_cell, "p_tgt": round(float(p_tgt), 3),
               "p": {"grasp": p_grasp, "release": p_release, "done": p_done},
               "gt_holding": gt_hold, "success": succ,
               "tip": tip.round(3).tolist(), "cube": cube.round(3).tolist()}
        log.append(rec)
        if verbose:
            print(f"[{tick:3d}] {phase:5s} {action:24s} tgt={tgt_cell}({p_tgt:.2f}) "
                  f"tip_cell={tip_cell} grasp={p_grasp:.2f} rel={p_release:.2f} "
                  f"done={p_done:.2f} hold={gt_hold} succ={succ}")
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
        json.dump({"task": TASK, "seed": seed, "mode": "cells_v2", "success": done,
                   "ticks": len(log), "wall_s": round(time.time() - t0, 1),
                   "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames)")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "cells0"))
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.out, max_ticks=args.ticks, seed=args.seed)
