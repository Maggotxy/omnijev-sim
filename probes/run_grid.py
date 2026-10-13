"""Grid-localization closed loop: OmniJev points at cells, geometry moves the arm.

The model does perception + grasp/release/done decisions (its proven strength:
96-cell pointing). Cartesian direction is computed from cell deltas.
"""
import argparse
import json
import os
import time

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

from sim import Sim
from omnijev_api import OmniJev, TASK

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID = 4  # 4x4 cells, labels A1..D4
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]
IMG = 448
CELL_PX = IMG // GRID

# agentview camera axes (from sim_scene.xml): x=(0.45,0.89,0) right, y=(-0.42,0.21,0.88) up
CAM_RIGHT = np.array([0.45, 0.89, 0.0]); CAM_RIGHT /= np.linalg.norm(CAM_RIGHT)
CAM_UP = np.array([-0.42, 0.21, 0.88])
CAM_UP_XY = np.array([-0.42, 0.21, 0.0]); CAM_UP_XY /= np.linalg.norm(CAM_UP_XY)
M_PER_PX = 1.04 / IMG  # rough scale at table distance


def draw_grid(arr):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=2)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=2)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 6, row * CELL_PX + 6), name, fill=(255, 255, 0))
    return np.array(im)


def cell_xy(name):
    idx = CELLS.index(name)
    col, row = idx % GRID, idx // GRID
    return np.array([(col + 0.5) * CELL_PX, (row + 0.5) * CELL_PX])  # px, y down


def questions(target_word):
    return {
        "grip": {"type": "choice", "instructions":
                 "Panel 1 shows a robot scene with a grid. Which grid cell contains the tip of the robot gripper?",
                 "criteria": {c: f"cell {c}" for c in CELLS}},
        "tgt": {"type": "choice", "instructions":
                f"Panel 1: which grid cell contains the {target_word}?",
                "criteria": {c: f"cell {c}" for c in CELLS}},
        "wtgt": {"type": "choice", "instructions":
                 f"Panel 2 is the wrist camera looking down from the gripper. "
                 f"Which grid cell contains the {target_word}? (answer a cell only if it is visible)",
                 "criteria": {c: f"cell {c}" for c in CELLS}},
        "grasp": {"type": "noul", "instructions":
                  TASK + " The gripper should close on the red cube right now."},
        "release": {"type": "noul", "instructions":
                    TASK + " The gripper should release the object right now."},
        "done": {"type": "noul", "instructions": TASK + " The task is finished."},
    }


# wrist camera: calibrated affine world-xy -> image-px (hover ~5cm above cube):
#   world +x (1m) -> img (-3598, -392) px ; world +y (1m) -> img (-42, +2136) px
_A = np.array([[-3598.0, -42.0], [-392.0, 2136.0]])
_A_INV = np.linalg.inv(_A)
IMG_CENTER = np.array([224.0, 224.0])
CENTER_CELLS = {"B2", "B3", "C2", "C3"}


def wrist_world_delta(cell):
    """world xy velocity that brings the target image point to the image center."""
    s = IMG_CENTER - cell_xy(cell)          # required image shift, px
    w = _A_INV @ s                          # meters
    return np.array([w[0], w[1], 0.0])


def run(out_dir, max_ticks=80, seed=0, decision_hz=2.0, speed=0.20,
        grasp_th=0.6, release_th=0.65, done_th=0.7, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim()
    sim.reset(rng=np.random.default_rng(seed))
    api = OmniJev()

    frames, log = [], []
    grasped, phase, lift_left, done = False, "REACH", 0, False
    center_streak = 0          # consecutive ticks with the cube in wrist center
    t0 = time.time()

    for tick in range(max_ticks):
        agent = sim.render("agentview")
        wrist = sim.render("wrist")
        grid_agent = draw_grid(agent)
        grid_wrist = draw_grid(wrist)
        target_word = "red cube" if not grasped else "green circle"
        try:
            ans = api.ask([grid_agent, grid_wrist], questions(target_word))
        except Exception as e:
            print(f"[{tick}] API error: {e}")
            time.sleep(2)
            continue

        g_cell = ans.get("grip", {}).get("choice")
        t_cell = ans.get("tgt", {}).get("choice")
        w_ans = ans.get("wtgt", {})
        w_cell = w_ans.get("choice")
        w_abstain = w_ans.get("abstain", 1)
        if w_abstain and w_abstain > 0.5:
            w_cell = None
        p_grasp = ans.get("grasp", {}).get("noul", 0)
        p_release = ans.get("release", {}).get("noul", 0)
        p_done = ans.get("done", {}).get("noul", 0)

        action = "stay"
        v = np.zeros(3)
        aligned = g_cell == t_cell

        if phase == "REACH":
            cube_z = sim.cube_pos()[2]
            servo_z = cube_z + 0.13          # stay high enough for the wrist to see
            tip_z = sim.tip_pos()[2]
            if not aligned:
                # coarse stage: agent-view cell tracking, keep altitude
                center_streak = 0
                if g_cell and t_cell:
                    delta_px = cell_xy(t_cell) - cell_xy(g_cell)
                    w = (delta_px[0] * M_PER_PX) * CAM_RIGHT + (-delta_px[1] * M_PER_PX) * CAM_UP_XY
                    v = np.array([w[0], w[1], 0.0])
                    action = f"track:{g_cell}->{t_cell}"
                if tip_z < servo_z:
                    v[2] = 0.5
            elif w_cell is not None and w_cell not in CENTER_CELLS:
                # fine stage: wrist servo at servo altitude
                center_streak = 0
                v = wrist_world_delta(w_cell)
                v[2] = 0.8 * (servo_z - tip_z) / 0.13   # hold servo altitude
                action = f"servo:{w_cell}"
            else:
                # agent-aligned and wrist centered (or wrist blind): hold & count
                center_streak += 1
                v[2] = 0.8 * (servo_z - tip_z) / 0.13
                action = f"hover:{center_streak}"
                if center_streak >= 3:
                    phase = "DESCEND"
                    center_streak = 0
        elif phase == "DESCEND":
            z_gap = sim.tip_pos()[2] - sim.cube_pos()[2]
            if p_grasp > grasp_th and z_gap < 0.025:
                sim.set_gripper(False)
                sim.settle(0.8)
                if sim.holding():
                    grasped, phase, lift_left = True, "LIFT", 4
                    action = "GRASP"
                else:
                    sim.set_gripper(True)
                    phase = "REACH"           # back to coarse approach
                    action = "GRASP_MISS"
            elif z_gap > 0.025:
                v = np.array([0, 0, -1.0])
                action = "descend"
            else:
                action = f"wait_grasp:{p_grasp:.2f}"
        elif phase == "LIFT":
            v = np.array([0, 0, 1.0])
            lift_left -= 1
            action = "lift"
            if lift_left <= 0:
                phase = "CARRY"
        elif phase == "CARRY":
            if aligned and p_release > release_th:
                sim.set_gripper(True)
                sim.settle(0.5)
                grasped, phase = False, "REACH"
                action = "RELEASE"
            elif aligned:
                v = np.array([0, 0, -0.5])          # lower onto the zone
                action = "lower"
            elif g_cell and t_cell:
                delta_px = cell_xy(t_cell) - cell_xy(g_cell)
                w = (delta_px[0] * M_PER_PX) * CAM_RIGHT + (-delta_px[1] * M_PER_PX) * CAM_UP_XY
                v = np.array([w[0], w[1], 0.0])
                action = f"carry:{g_cell}->{t_cell}"

        if np.linalg.norm(v) > 1e-6:
            sim.move_ee(v, speed=speed)
        sim.data.ctrl[:7] = sim.q_target
        sim.settle(1.0 / decision_hz)

        gt_hold = bool(sim.holding())
        succ = bool(sim.success())
        rec = {"tick": tick, "phase": phase, "action": action,
               "g_cell": g_cell, "t_cell": t_cell, "w_cell": w_cell,
               "p": {"grasp": p_grasp, "release": p_release, "done": p_done},
               "gt_holding": gt_hold, "success": succ,
               "ee": sim.ee_pos().round(3).tolist(),
               "tip": sim.tip_pos().round(3).tolist(),
               "cube": sim.cube_pos().round(3).tolist()}
        log.append(rec)
        if verbose:
            print(f"[{tick:3d}] {phase:5s} {action:22s} w={w_cell} grasp={p_grasp:.2f} "
                  f"rel={p_release:.2f} done={p_done:.2f} gt_hold={gt_hold} succ={succ}")
        frames.append(grid_agent)

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
        json.dump({"task": TASK, "seed": seed, "mode": "grid", "success": done,
                   "ticks": len(log), "wall_s": round(time.time() - t0, 1),
                   "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames)")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "grid0"))
    ap.add_argument("--ticks", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.out, max_ticks=args.ticks, seed=args.seed)
