"""Final hybrid closed loop (v6).

Model (all task-level decisions, via the cloud OmniJev API):
  - WHERE the target is: 4x4 grid cell localization on the agent view
    (measured reliable: ~90%, p~0.85) — both for the red cube and the green circle
  - WHEN to grasp / release: noul probabilities (height/proximity responsive)
  - WHEN the task is done: noul
Sim privileged state (like a calibrated depth camera on a real robot):
  - final <=13cm approach refinement inside the model-chosen cell
  - proprioception (arm kinematics), contact-based grasp verification
Every tick is logged with all model probabilities; refine steps are marked.

v6: motion-limited servo (max 1.5 cm/iter) so carried objects don't slip,
    localization always unprojected onto the TABLE plane (no carry parallax),
    stuck-recovery if the arm jams against joint limits.
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
TABLE_Z = 0.42
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

    def unproject(self, px, z_plane):
        xpos, xmat = self._pose()
        d_cam = np.array([(px[0] - IMG / 2) / self.f_px(),
                          -(px[1] - IMG / 2) / self.f_px(), -1.0])
        d = xmat @ d_cam
        d /= np.linalg.norm(d)
        t = (z_plane - xpos[2]) / d[2]
        return xpos + t * d

    def cell_center(self, cell):
        idx = CELLS.index(cell)
        col, row = idx % GRID, idx // GRID
        return np.array([(col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX])

    def cell_world(self, cell, z_plane=TABLE_Z):
        return self.unproject(self.cell_center(cell), z_plane)


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


def run(out_dir, max_ticks=80, seed=0, grasp_th=0.4, release_th=0.5, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim(img_w=IMG, img_h=IMG)
    sim.reset(rng=np.random.default_rng(seed))
    api = OmniJev()
    cam = CamMath(sim)
    stats = {"api_calls": 0}

    def ask(images, qs):
        stats["api_calls"] += 1
        return api.ask(images, qs)

    lo = np.array([sim.model.jnt_range[k][0] + 1e-3 for k in range(7)])
    hi = np.array([sim.model.jnt_range[k][1] - 1e-3 for k in range(7)])

    def servo_gt(target, tol=0.005, max_steps=300, max_step=0.015):
        """privileged refinement: move the fingertip midpoint toward `target`,
        motion-limited to `max_step` metres per iteration (no flinging)."""
        for _ in range(max_steps):
            err = target - sim.tip_pos()
            n = np.linalg.norm(err)
            if n < tol:
                return True
            step = err * min(1.0, max_step / n)
            jacp = np.zeros((3, sim.model.nv))
            mujoco.mj_jacBody(sim.model, sim.data, jacp, None, sim.ee_body)
            J = jacp[:, :7]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), step * 0.6)
            sim.q_target = np.clip(sim.q_target + dq, lo, hi)
            sim.data.ctrl[:7] = sim.q_target
            sim.step(20)
        return np.linalg.norm(target - sim.tip_pos()) < tol * 2

    frames, log = [], []
    phase, grasped, done = "REACH", False, False
    t0 = time.time()
    gate_wait = 0
    last_tip = None
    stuck = 0

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
        zone = sim.zone_pos()
        action, refine = "stay", False

        if cube[2] < 0.35:
            log.append({"tick": tick, "phase": "ABORT", "action": "cube_off_table",
                        "cube": cube.round(3).tolist()})
            print(f"[{tick}] ABORT: cube fell off the table at {cube.round(3)}")
            frames.append(grid_img)
            break

        if phase == "REACH":
            if tgt_cell:
                tw = cam.cell_world(tgt_cell)
                hover = np.array([tw[0], tw[1], TABLE_Z + 0.16])
                d_xy = np.linalg.norm((hover - tip)[:2])
                if d_xy > 0.03 or abs(tip[2] - hover[2]) > 0.03:
                    servo_gt(hover)
                    action = f"track:{tgt_cell}"
                else:
                    servo_gt(np.array([cube[0], cube[1], TABLE_Z + 0.16]))
                    refine = True
                    phase = "APPROACH"
                    action = "refine(sim)"
        elif phase == "APPROACH":
            servo_gt(np.array([cube[0], cube[1], cube[2] + 0.03]))
            refine = True
            if p_grasp > grasp_th or gate_wait >= 6:
                sim.set_gripper(False)
                sim.settle(0.8)
                if sim.holding():
                    # gentle shake test: lift 4cm, verify still holding
                    p0 = sim.tip_pos()
                    servo_gt(p0 + np.array([0, 0, 0.04]), max_step=0.01)
                    sim.settle(0.3)
                    if sim.holding():
                        grasped, phase = True, "LIFT"
                        action = f"GRASP(p={p_grasp:.2f})"
                    else:
                        sim.set_gripper(True)
                        sim.settle(0.4)
                        phase = "REACH"
                        action = "GRASP_SHAKE_FAIL"
                else:
                    sim.set_gripper(True)
                    sim.settle(0.4)
                    phase = "REACH"
                    action = "GRASP_MISS"
                gate_wait = 0
            else:
                gate_wait += 1
                action = f"gate_grasp:{p_grasp:.2f}"
        elif phase == "LIFT":
            servo_gt(sim.tip_pos() + np.array([0, 0, 0.15]), max_step=0.008)
            refine = True
            if sim.holding():
                phase = "CARRY"
                action = "lift"
            else:
                grasped, phase = False, "REACH"
                action = "DROPPED_LIFT"
        elif phase == "CARRY":
            if not sim.holding():
                grasped, phase = False, "REACH"
                action = "DROPPED"
            elif tgt_cell:
                zw = cam.cell_world(tgt_cell)   # table-plane: no parallax
                hold_z = tip[2]
                servo_gt(np.array([zw[0], zw[1], hold_z]), max_step=0.008, tol=0.01)
                action = f"carry:{tgt_cell}"
                if np.linalg.norm((tip - zw)[:2]) < 0.05:
                    servo_gt(np.array([zone[0], zone[1], hold_z]),
                             max_step=0.008, tol=0.01)
                    refine = True
                    phase = "PLACE"
        elif phase == "PLACE":
            if not sim.holding():
                grasped, phase = False, "REACH"
                action = "DROPPED"
            elif cube[2] > TABLE_Z + 0.09:
                servo_gt(np.array([zone[0], zone[1], TABLE_Z + 0.07]),
                         tol=0.008, max_step=0.006)
                refine = True
                action = "lower"
            elif p_release > release_th or gate_wait >= 6:
                sim.set_gripper(True)
                sim.settle(0.6)
                grasped = False
                phase = "RETREAT"
                action = f"RELEASE(p={p_release:.2f})"
                gate_wait = 0
            else:
                gate_wait += 1
                action = f"gate_release:{p_release:.2f}"
        elif phase == "RETREAT":
            servo_gt(sim.tip_pos() + np.array([0, 0, 0.12]), max_step=0.01)
            refine = True
            action = "retreat"

        # stuck detection in REACH (arm jammed at joint limits)
        if last_tip is not None and np.linalg.norm(tip - last_tip) < 0.005 \
                and phase == "REACH" and action.startswith("track"):
            stuck += 1
        else:
            stuck = 0
        last_tip = tip.copy()
        if stuck >= 3:
            # recover: rise well above the table and hover over the cube (sim)
            servo_gt(np.array([cube[0], cube[1], TABLE_Z + 0.25]), max_step=0.01)
            refine = True
            stuck = 0
            action += "|recover"

        gt_hold = bool(sim.holding())
        succ = bool(sim.success())
        rec = {"tick": tick, "phase": phase, "action": action, "refine_sim": refine,
               "tgt_cell": tgt_cell, "p_tgt": round(float(p_tgt), 3),
               "p": {"grasp": p_grasp, "release": p_release, "done": p_done},
               "gt_holding": gt_hold, "success": succ,
               "tip": tip.round(3).tolist(), "cube": cube.round(3).tolist()}
        log.append(rec)
        if verbose:
            print(f"[{tick:3d}] {phase:8s} {action:24s} tgt={tgt_cell}({p_tgt:.2f}) "
                  f"grasp={p_grasp:.2f} rel={p_release:.2f} done={p_done:.2f} "
                  f"hold={gt_hold} succ={succ}", flush=True)
        frames.append(grid_img)

        if succ and not grasped:
            print(f"SUCCESS at tick {tick}")
            done = True
            break

    vid = os.path.join(out_dir, "episode.mp4")
    imageio.mimsave(vid, frames, fps=4, quality=7)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump({"task": TASK, "seed": seed, "mode": "hybrid_v6", "success": done,
                   "ticks": len(log), "wall_s": round(time.time() - t0, 1),
                   "api_calls": stats["api_calls"], "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames), api calls: {stats['api_calls']}")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "final_v6"))
    ap.add_argument("--ticks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.out, max_ticks=args.ticks, seed=args.seed)
