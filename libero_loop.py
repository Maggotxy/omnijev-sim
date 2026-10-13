"""LIBERO closed loop driven by OmniJev (official question template, in-distribution).

Model (cloud OmniJev) decides per ~0.4s: gripper direction (7-way), gripper
open/close/keep, holding noul, done noul, progress score — the exact questions
from the OmniJev LIBERO demos. Control: OSC velocity at 20Hz between decisions.
Video: every control step, HUD overlay, playback at 20fps (real-time speed).
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

import sys
sys.path.insert(0, "/opt/LIBERO")
from omnijev_api import OmniJev

FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
FONT_M = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)

IMG = 512
HUD_H = 128
DIRECTIONS = ["forward", "backward", "left", "right", "up", "down", "stay still"]
GRIPPER = ["open the gripper", "close the gripper", "keep the gripper as it is"]
PROGRESS = ["just started", "less than halfway", "more than halfway", "almost done"]

# workspace safety box (world frame, measured from probe)
WS = {"x": (-0.60, 0.10), "y": (-0.45, 0.45), "z": (0.80, 1.25)}


def questions_libero(task_lang):
    return {
        "action": {"type": "choice",
                   "instructions": "Robot camera view. Instruction: %s. During the "
                                   "next second, in which direction should the gripper "
                                   "mainly move?" % task_lang,
                   "criteria": {k: k for k in DIRECTIONS}},
        "gripper": {"type": "choice",
                    "instructions": "Robot camera view. Instruction: %s. What should "
                                    "happen to the gripper next?" % task_lang,
                    "criteria": {k: k for k in GRIPPER}},
        "holding": {"type": "noul",
                    "instructions": "Robot camera view. The gripper is currently "
                                    "closed on an object."},
        "done": {"type": "noul",
                 "instructions": "Robot camera view. The instruction '%s' has been "
                                 "completed." % task_lang},
        "progress": {"type": "score",
                     "instructions": "Robot camera view. Instruction: %s. How far "
                                     "along is the task?" % task_lang,
                     "levels": list(PROGRESS)},
    }


class HUD:
    def __init__(self, task_lang):
        self.task = task_lang
        self.dir = "stay still"
        self.dir_p = 0.0
        self.grip = "keep the gripper as it is"
        self.grip_p = 0.0
        self.probs = {"holding": 0.0, "done": 0.0}
        self.progress = PROGRESS[0]
        self.sim_t = 0.0
        self.api_calls = 0
        self.api_lat = 0.0
        self.banner = None
        self.banner_until = 0.0

    def flash(self, text, color):
        self.banner = (text, color)
        self.banner_until = self.sim_t + 1.0

    def compose(self, frame):
        im = Image.fromarray(frame).convert("RGB")
        canvas = Image.new("RGB", (IMG, IMG + HUD_H), (16, 16, 22))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        y = IMG + 5
        d.text((8, y), self.task, fill=(255, 255, 255), font=FONT_M)
        y += 24
        d.text((8, y), f"dir: {self.dir} ({self.dir_p:.2f})   grip: {self.grip} "
                       f"({self.grip_p:.2f})", fill=(255, 220, 80), font=FONT_S)
        d.text((IMG - 150, y), f"t={self.sim_t:.1f}s", fill=(170, 170, 170), font=FONT_S)
        y += 20
        d.text((8, y), f"progress: {self.progress}", fill=(150, 220, 255), font=FONT_S)
        d.text((IMG - 170, y), f"api {self.api_calls} ({self.api_lat:.2f}s)",
               fill=(170, 170, 170), font=FONT_S)
        y += 22
        bar_w = 300
        for name, col in (("holding", (240, 120, 60)), ("done", (120, 230, 120))):
            p = float(self.probs.get(name, 0.0))
            d.text((8, y), f"{name:7s}", fill=(200, 200, 200), font=FONT_S)
            d.rectangle([80, y + 2, 80 + bar_w, y + 13], outline=(80, 80, 80))
            d.rectangle([80, y + 2, 80 + int(bar_w * min(p, 1.0)), y + 13], fill=col)
            d.text((390, y), f"{p:.2f}", fill=col, font=FONT_S)
            y += 18
        if self.banner and self.sim_t < self.banner_until:
            txt, color = self.banner
            d.rectangle([IMG // 2 - 170, IMG // 2 - 30, IMG // 2 + 170, IMG // 2 + 30],
                        fill=(0, 0, 0), outline=color, width=3)
            w = d.textlength(txt, font=FONT_B)
            d.text((IMG // 2 - w / 2, IMG // 2 - 18), txt, fill=color, font=FONT_B)
        return np.array(canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-idx", type=int, default=0)
    ap.add_argument("--out", default="/opt/LIBERO/out/loop0")
    ap.add_argument("--max-decisions", type=int, default=400)
    ap.add_argument("--decide-every", type=int, default=8)   # control steps (0.4s @20Hz)
    ap.add_argument("--mag", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_10"]()
    task = suite.get_task(args.task_idx)
    print("task:", task.name, "|", task.language)

    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=IMG, camera_widths=IMG)
    obs = env.reset()
    env.env.hard_reset = False
    api = OmniJev()
    hud = HUD(task.language)
    qs = questions_libero(task.language)

    # camera-based direction mapping (world-frame unit vectors)
    sim = env.env.sim
    cam_id = sim.model.camera_name2id("agentview")
    xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    # cam_xmat COLUMNS are the camera axes in world: x=right, y=up, z=backward
    right_w = xmat[:, 0].copy()
    right_w[2] = 0
    right_w /= np.linalg.norm(right_w)
    fwd_w = -xmat[:, 2].copy()
    fwd_w[2] = 0
    fwd_w /= np.linalg.norm(fwd_w)
    DIR_VEC = {
        "forward": fwd_w, "backward": -fwd_w,
        "left": -right_w, "right": right_w,
        "up": np.array([0, 0, 1.0]), "down": np.array([0, 0, -1.0]),
        "stay still": np.zeros(3),
    }
    print("forward maps to world", fwd_w.round(2), "right maps to", right_w.round(2))

    frames, log = [], []
    cur_dir = np.zeros(3)
    gripper = -1.0           # start open (robosuite: -1 open, +1 close)
    last_grip_change = 0.0
    step_i = 0
    done = False
    success = False
    t0 = time.time()
    n_decisions = 0

    def record():
        hud.sim_t = sim.data.time
        hud.api_calls = n_decisions
        hud.api_lat = float(api.last_ms or 0) / 1000.0
        frames.append(hud.compose(obs["agentview_image"][::-1]))

    while n_decisions < args.max_decisions and not done:
        # control steps with current command
        action = np.concatenate([cur_dir * args.mag, [0, 0, 0], [gripper]])
        obs, r, done_flag, info = env.step(action)
        step_i += 1
        record()
        if env.env._check_success():
            success = True

        if step_i % args.decide_every != 0:
            continue

        # ---- decision tick ----
        n_decisions += 1
        img = obs["agentview_image"][::-1]
        try:
            ans = api.ask([img], qs)
        except Exception as e:
            print(f"[d{n_decisions}] API error: {e}")
            time.sleep(2)
            continue

        a = ans["action"]
        d_name, d_p = a.get("choice"), max(a.get("probabilities", {}).values() or [0])
        g = ans["gripper"]
        g_name, g_p = g.get("choice"), max(g.get("probabilities", {}).values() or [0])
        p_hold = ans["holding"]["noul"]
        p_done = ans["done"]["noul"]
        sc = ans["progress"]
        prog = sc.get("score") or sc.get("level") or "?"

        hud.dir, hud.dir_p = d_name, d_p
        hud.grip, hud.grip_p = g_name, g_p
        hud.probs = {"holding": p_hold, "done": p_done}
        hud.progress = prog

        cur_dir = DIR_VEC.get(d_name, np.zeros(3))

        # gripper with 0.8s cooldown
        if g_p > 0.5 and sim.data.time - last_grip_change > 0.8:
            if g_name == "close the gripper" and gripper < 0:
                gripper = 1.0
                last_grip_change = sim.data.time
                hud.flash("CLOSE GRIPPER", (255, 120, 40))
            elif g_name == "open the gripper" and gripper > 0:
                gripper = -1.0
                last_grip_change = sim.data.time
                hud.flash("OPEN GRIPPER", (90, 180, 250))

        # workspace safety clamp (zero out components leaving the box)
        eef = obs["robot0_eef_pos"]
        for ax, (k, lim) in enumerate((("x", WS["x"]), ("y", WS["y"]), ("z", WS["z"]))):
            if cur_dir[ax] > 0 and eef[ax] > lim[1]:
                cur_dir[ax] = 0
            if cur_dir[ax] < 0 and eef[ax] < lim[0]:
                cur_dir[ax] = 0

        rec = {"d": n_decisions, "t": round(sim.data.time, 2),
               "dir": d_name, "dir_p": round(float(d_p), 3),
               "grip": g_name, "grip_p": round(float(g_p), 3),
               "holding": round(float(p_hold), 3), "done": round(float(p_done), 3),
               "progress": prog, "eef": eef.round(3).tolist(),
               "gt_success": bool(env.env._check_success())}
        log.append(rec)
        print(f"[d{n_decisions:3d}] dir={d_name}({d_p:.2f}) grip={g_name}({g_p:.2f}) "
              f"hold={p_hold:.2f} done={p_done:.2f} prog={prog} succ={rec['gt_success']}",
              flush=True)

        if p_done > 0.85 and success:
            print("model says done AND ground truth success")
            done = True
        if done_flag:
            done = True

    hud.flash("TASK COMPLETE" if success else "EPISODE END",
              (80, 255, 120) if success else (255, 90, 90))
    for _ in range(30):
        record()

    vid = os.path.join(args.out, "episode.mp4")
    imageio.mimsave(vid, frames, fps=20, quality=7)
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump({"task": task.language, "task_name": task.name, "seed": args.seed,
                   "success": bool(success), "decisions": n_decisions,
                   "sim_seconds": round(sim.data.time, 1),
                   "wall_s": round(time.time() - t0, 1), "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames), success={success}, "
          f"sim {sim.data.time:.1f}s, wall {time.time() - t0:.0f}s")
    env.close()


if __name__ == "__main__":
    main()
