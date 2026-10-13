"""Upgraded demo (v7): two-cube sorting with pluggable VLM backend + HUD video.

Task: "Pick up the red cube and place it on the green circle, then pick up the
blue cube and place it on the blue circle." — two sequential pick-place subtasks.

Backends (all task-level decisions):
  - omnijev: OmniJev-0.8B systemone API (choice cell + noul probabilities)
  - stepfun: step-1o-turbo-vision (cell label + YES/NO decisions, parsed)
Sim privileged state: <=13cm final refinement + contact verification (marked).
Video: smooth frames recorded during ALL motion, HUD with live model outputs,
picture-in-picture of the grid image the model actually sees.
"""
import argparse
import base64
import io
import json
import os
import re
import time
import urllib.request

import numpy as np
import imageio.v2 as imageio
import mujoco
from PIL import Image, ImageDraw, ImageFont

from sim import Sim
from omnijev_api import OmniJev

ROOT = os.path.dirname(os.path.abspath(__file__))
GRID, IMG = 4, 640
CELL_PX = IMG // GRID
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]
TABLE_Z = 0.42
HUD_H = 160
FONT_B = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
FONT_M = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
FONT_S = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)


def bold_grid(arr):
    im = Image.fromarray(arr).convert("RGB")
    d = ImageDraw.Draw(im)
    for i in range(1, GRID):
        d.line([(i * CELL_PX, 0), (i * CELL_PX, IMG)], fill=(255, 255, 255), width=4)
        d.line([(0, i * CELL_PX), (IMG, i * CELL_PX)], fill=(255, 255, 255), width=4)
    for idx, name in enumerate(CELLS):
        col, row = idx % GRID, idx // GRID
        d.text((col * CELL_PX + 10, row * CELL_PX + 8), name, fill=(255, 255, 0),
               font=FONT_B, stroke_width=2, stroke_fill=(0, 0, 0))
    return np.array(im)


# ---------------- backends ----------------

TASK_FULL = ("Pick up the red cube and place it on the green circle, "
             "then pick up the blue cube and place it on the blue circle.")


class OmniBackend:
    name = "OmniJev-0.8B"

    def __init__(self):
        self.api = OmniJev()
        self.lat = []

    def cell(self, grid_img, target_word):
        t0 = time.time()
        ans = self.api.ask([grid_img], {"tgt": {
            "type": "choice",
            "instructions": f"This robot scene has a labeled 4x4 grid (columns A-D "
                            f"left to right, rows 1-4 top to bottom). Which cell "
                            f"contains the {target_word}?",
            "criteria": {c: f"cell {c}" for c in CELLS}}})
        self.lat.append(time.time() - t0)
        return ans["tgt"].get("choice"), max(ans["tgt"].get("probabilities", {}).values() or [0])

    def decide(self, clean_img, subtask_instr):
        t0 = time.time()
        ans = self.api.ask([clean_img], {
            "grasp": {"type": "noul", "instructions":
                      subtask_instr + " The gripper should close on the cube right now."},
            "release": {"type": "noul", "instructions":
                        subtask_instr + " The gripper should release the cube right now."},
            "done": {"type": "noul", "instructions":
                     TASK_FULL + " The whole task is finished."}})
        self.lat.append(time.time() - t0)
        return {"grasp": ans["grasp"]["noul"], "release": ans["release"]["noul"],
                "done": ans["done"]["noul"]}


class StepBackend:
    name = "Step-1o-turbo-vision"
    KEY = os.environ.get("STEPFUN_API_KEY", "")

    def __init__(self):
        self.lat = []

    def _chat(self, img, text):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"model": "step-1o-turbo-vision",
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}},
                       {"type": "text", "text": text}]}],
                   "temperature": 0.0}
        req = urllib.request.Request(
            "https://api.stepfun.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.KEY}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        return out["choices"][0]["message"]["content"]

    def cell(self, grid_img, target_word):
        t0 = time.time()
        txt = self._chat(grid_img,
                         f"This robot scene has a labeled 4x4 grid (columns A-D left "
                         f"to right, rows 1-4 top to bottom). Which cell contains the "
                         f"{target_word}? Answer with ONLY the cell label, e.g. 'B2'.")
        self.lat.append(time.time() - t0)
        m = re.search(r"\b([A-D][1-4])\b", txt)
        return (m.group(1) if m else None), 1.0

    def decide(self, clean_img, subtask_instr):
        t0 = time.time()
        txt = self._chat(clean_img,
                         subtask_instr + " Answer ONLY a JSON object with boolean keys: "
                         '{"grasp": true if the gripper should close on the cube right now, '
                         '"release": true if the gripper should release the cube right now, '
                         '"done": true if the whole task is finished}')
        self.lat.append(time.time() - t0)
        m = re.search(r"\{.*\}", txt, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except Exception:
            d = {}
        return {"grasp": 1.0 if d.get("grasp") else 0.0,
                "release": 1.0 if d.get("release") else 0.0,
                "done": 1.0 if d.get("done") else 0.0}


# ---------------- camera math ----------------

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

    def unproject(self, px, z_plane=TABLE_Z):
        xpos, xmat = self._pose()
        d_cam = np.array([(px[0] - IMG / 2) / self.f_px(),
                          -(px[1] - IMG / 2) / self.f_px(), -1.0])
        d = xmat @ d_cam
        d /= np.linalg.norm(d)
        t = (z_plane - xpos[2]) / d[2]
        return xpos + t * d

    def cell_world(self, cell):
        idx = CELLS.index(cell)
        col, row = idx % GRID, idx // GRID
        return self.unproject(np.array([(col + 0.5) * CELL_PX,
                                        (row + 0.5) * CELL_PX]))


# ---------------- HUD ----------------

class HUD:
    def __init__(self, backend_name, w=IMG, h=IMG):
        self.backend = backend_name
        self.W, self.H = w, h
        self.phase = "INIT"
        self.subtask = ""
        self.cell_ans = None
        self.cell_p = 0.0
        self.probs = {"grasp": 0.0, "release": 0.0, "done": 0.0}
        self.tick = 0
        self.sim_time = 0.0
        self.banner = None
        self.banner_until = 0
        self.api_lat = 0.0
        self.api_calls = 0

    def flash(self, text, color):
        self.banner = (text, color)
        self.banner_until = self.sim_time + 1.2

    def compose(self, main_arr, grid_arr):
        W, H = self.W, self.H
        im = Image.fromarray(main_arr).convert("RGB")
        # PiP: the grid image the model actually sees
        pip_s = int(W * 0.24)
        pip = Image.fromarray(grid_arr).resize((pip_s, pip_s))
        im.paste(pip, (W - pip_s - 8, 8))
        d = ImageDraw.Draw(im)
        d.rectangle([W - pip_s - 10, 6, W - 6, pip_s + 10],
                    outline=(255, 255, 255), width=2)
        d.text((W - pip_s - 6, pip_s - 12), "MODEL VIEW",
               fill=(255, 255, 0), font=FONT_S)

        canvas = Image.new("RGB", (W, H + HUD_H), (18, 18, 24))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)

        y = H + 6
        d.text((12, y), f"{self.subtask}", fill=(255, 255, 255), font=FONT_M)
        d.text((W - 260, y), f"{self.backend}", fill=(120, 200, 255), font=FONT_M)
        y += 26
        d.text((12, y), f"phase={self.phase}  tick={self.tick}  "
                        f"t={self.sim_time:.1f}s  api={self.api_calls} "
                        f"({self.api_lat:.2f}s)", fill=(170, 170, 170), font=FONT_S)
        y += 22
        cell_txt = f"model cell: {self.cell_ans or '-'} ({self.cell_p:.2f})"
        d.text((12, y), cell_txt, fill=(255, 220, 80), font=FONT_M)
        y += 30
        # probability bars
        bar_w = int(W * 0.55)
        for name, col in (("grasp", (240, 120, 60)), ("release", (90, 180, 250)),
                          ("done", (120, 230, 120))):
            p = float(self.probs.get(name, 0.0))
            d.text((12, y), f"{name:7s}", fill=(200, 200, 200), font=FONT_S)
            d.rectangle([100, y + 2, 100 + bar_w, y + 16], outline=(80, 80, 80))
            d.rectangle([100, y + 2, 100 + int(bar_w * min(p, 1.0)), y + 16], fill=col)
            d.text((110 + bar_w, y), f"{p:.2f}", fill=col, font=FONT_S)
            y += 21

        if self.banner and self.sim_time < self.banner_until:
            txt, color = self.banner
            d.rectangle([W // 2 - 220, H // 2 - 34, W // 2 + 220, H // 2 + 34],
                        fill=(0, 0, 0), outline=color, width=4)
            w = d.textlength(txt, font=FONT_B)
            d.text((W // 2 - w / 2, H // 2 - 20), txt, fill=color, font=FONT_B)
        return np.array(canvas)


# ---------------- main loop ----------------

def run(out_dir, backend_name="omnijev", max_ticks=90, seed=1,
        grasp_th=0.4, release_th=0.5, scene=None, beauty=False):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim(scene_path=scene, img_w=IMG, img_h=IMG)
    rng = np.random.default_rng(seed)
    sim.reset(rng=rng)
    backend = OmniBackend() if backend_name == "omnijev" else StepBackend()
    cam = CamMath(sim)
    if beauty:
        vw, vh = 960, 540
        beauty_r = mujoco.Renderer(sim.model, height=vh, width=vw)
    else:
        vw, vh = IMG, IMG
        beauty_r = None
    hud = HUD(backend.name, w=vw, h=vh)

    # lay out both cubes and zones deterministically-ish (well separated)
    def set_cube(body_name, x, y):
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT,
                                f"{body_name}_joint")
        adr = sim.model.jnt_qposadr[jid]
        dad = sim.model.jnt_dofadr[jid]
        sim.data.qpos[adr:adr + 7] = [x, y, TABLE_Z + 0.035, 1, 0, 0, 0]
        sim.data.qvel[dad:dad + 6] = 0

    set_cube("cube_red", rng.uniform(0.44, 0.56), rng.uniform(-0.18, -0.05))
    set_cube("cube_blue", rng.uniform(0.58, 0.70), rng.uniform(0.08, 0.18))
    bid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
    sim.model.body_pos[bid][:2] = [0.44, 0.26]
    bid2 = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone_blue")
    sim.model.body_pos[bid2][:2] = [0.70, -0.20]
    mujoco.mj_forward(sim.model, sim.data)
    sim.settle(0.5)

    lo = np.array([sim.model.jnt_range[k][0] + 1e-3 for k in range(7)])
    hi = np.array([sim.model.jnt_range[k][1] - 1e-3 for k in range(7)])

    frames = []
    REC_EVERY = 25  # sim steps (2ms) -> 0.05s per frame

    def record():
        agent = sim.render("agentview")          # what the model sees
        grid = bold_grid(agent)
        if beauty_r is not None:
            beauty_r.update_scene(sim.data, camera="beauty")
            main = beauty_r.render()
        else:
            main = agent
        hud.sim_time = sim.data.time
        hud.api_calls = len(backend.lat)
        hud.api_lat = backend.lat[-1] if backend.lat else 0.0
        frames.append(hud.compose(main, grid))

    def settle_rec(seconds):
        n = int(seconds / sim.model.opt.timestep)
        for _ in range(0, n, REC_EVERY):
            sim.step(min(REC_EVERY, n))
            record()

    def servo_gt(target, tol=0.005, max_steps=300, max_step=0.015):
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
            record()
        return np.linalg.norm(target - sim.tip_pos()) < tol * 2

    body_ids = {
        "red cube": mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red"),
        "blue cube": mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "cube_blue"),
        "green circle": mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone"),
        "blue circle": mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone_blue"),
    }

    def obj_pos(word):
        return sim.data.xpos[body_ids[word]].copy()

    subtasks = [("red cube", "green circle"), ("blue cube", "blue circle")]
    log = []
    t0 = time.time()
    tick = 0
    all_done = False

    for si, (obj_word, zone_word) in enumerate(subtasks):
        instr = f"Pick up the {obj_word} and place it on the {zone_word}."
        phase, grasped = "REACH", False
        gate_wait = 0
        hud.phase = phase
        hud.subtask = f"Subtask {si + 1}/2: {obj_word} -> {zone_word}"

        while tick < max_ticks:
            tick += 1
            hud.tick = tick
            agent = sim.render("agentview")
            grid = bold_grid(agent)
            try:
                tgt_cell, p_tgt = backend.cell(grid, obj_word if not grasped else zone_word)
                dec = backend.decide(agent, instr)
            except Exception as e:
                print(f"[{tick}] API error: {e}")
                time.sleep(2)
                continue
            hud.cell_ans, hud.cell_p = tgt_cell, p_tgt
            hud.probs = dec

            tip = sim.tip_pos()
            cube = obj_pos(obj_word)
            zone = obj_pos(zone_word)
            action, refine = "stay", False

            if cube[2] < 0.35:
                print(f"[{tick}] ABORT: {obj_word} fell off table")
                log.append({"tick": tick, "action": "abort_off_table"})
                record()
                break

            if phase == "REACH":
                if tgt_cell:
                    tw = cam.cell_world(tgt_cell)
                    hover = np.array([tw[0], tw[1], TABLE_Z + 0.16])
                    if np.linalg.norm((hover - tip)[:2]) > 0.03 or abs(tip[2] - hover[2]) > 0.03:
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
                if dec["grasp"] > grasp_th or gate_wait >= 6:
                    sim.set_gripper(False)
                    settle_rec(0.8)
                    if sim.holding() if obj_word == "red cube" else holding_generic(sim, body_ids[obj_word]):
                        p0 = sim.tip_pos()
                        servo_gt(p0 + np.array([0, 0, 0.04]), max_step=0.01)
                        ok_hold = (sim.holding() if obj_word == "red cube"
                                   else holding_generic(sim, body_ids[obj_word]))
                        if ok_hold:
                            grasped, phase = True, "LIFT"
                            action = f"GRASP(p={dec['grasp']:.2f})"
                            hud.flash("GRASP", (255, 120, 40))
                        else:
                            sim.set_gripper(True)
                            settle_rec(0.4)
                            phase = "REACH"
                            action = "GRASP_SHAKE_FAIL"
                    else:
                        sim.set_gripper(True)
                        settle_rec(0.4)
                        phase = "REACH"
                        action = "GRASP_MISS"
                    gate_wait = 0
                else:
                    gate_wait += 1
                    action = f"gate_grasp:{dec['grasp']:.2f}"
            elif phase == "LIFT":
                servo_gt(sim.tip_pos() + np.array([0, 0, 0.15]), max_step=0.008)
                refine = True
                ok_hold = (sim.holding() if obj_word == "red cube"
                           else holding_generic(sim, body_ids[obj_word]))
                if ok_hold:
                    phase = "CARRY"
                    action = "lift"
                else:
                    grasped, phase = False, "REACH"
                    action = "DROPPED_LIFT"
            elif phase == "CARRY":
                ok_hold = (sim.holding() if obj_word == "red cube"
                           else holding_generic(sim, body_ids[obj_word]))
                if not ok_hold:
                    grasped, phase = False, "REACH"
                    action = "DROPPED"
                elif tgt_cell:
                    zw = cam.cell_world(tgt_cell)
                    hold_z = tip[2]
                    servo_gt(np.array([zw[0], zw[1], hold_z]), max_step=0.008, tol=0.01)
                    action = f"carry:{tgt_cell}"
                    if np.linalg.norm((tip - zw)[:2]) < 0.05:
                        servo_gt(np.array([zone[0], zone[1], hold_z]),
                                 max_step=0.008, tol=0.01)
                        refine = True
                        phase = "PLACE"
            elif phase == "PLACE":
                ok_hold = (sim.holding() if obj_word == "red cube"
                           else holding_generic(sim, body_ids[obj_word]))
                if not ok_hold:
                    grasped, phase = False, "REACH"
                    action = "DROPPED"
                elif cube[2] > TABLE_Z + 0.09:
                    servo_gt(np.array([zone[0], zone[1], TABLE_Z + 0.07]),
                             tol=0.008, max_step=0.006)
                    refine = True
                    action = "lower"
                elif dec["release"] > release_th or gate_wait >= 6:
                    sim.set_gripper(True)
                    settle_rec(0.6)
                    grasped = False
                    phase = "RETREAT"
                    action = f"RELEASE(p={dec['release']:.2f})"
                    hud.flash("RELEASE", (90, 180, 250))
                    gate_wait = 0
                else:
                    gate_wait += 1
                    action = f"gate_release:{dec['release']:.2f}"
            elif phase == "RETREAT":
                servo_gt(sim.tip_pos() + np.array([0, 0, 0.12]), max_step=0.01)
                refine = True
                action = "retreat"

            hud.phase = phase
            in_zone = np.linalg.norm(cube[:2] - zone[:2]) < 0.07 and cube[2] < 0.49
            rec = {"tick": tick, "subtask": si, "phase": phase, "action": action,
                   "refine_sim": refine, "tgt_cell": tgt_cell,
                   "p_tgt": round(float(p_tgt), 3),
                   "p": {k: round(float(v), 3) for k, v in dec.items()},
                   "in_zone": bool(in_zone)}
            log.append(rec)
            print(f"[{tick:3d}] st{si + 1} {phase:8s} {action:22s} "
                  f"tgt={tgt_cell}({p_tgt:.2f}) g={dec['grasp']:.2f} "
                  f"r={dec['release']:.2f} d={dec['done']:.2f} in_zone={in_zone}",
                  flush=True)
            record()

            if in_zone and not grasped and phase == "RETREAT":
                break  # subtask complete
            if tick >= max_ticks:
                break
        else:
            break
        if tick >= max_ticks or (log and log[-1].get("action") == "abort_off_table"):
            break
        # verify subtask success before moving on
        cube_f = obj_pos(obj_word)
        zone_f = obj_pos(zone_word)
        if not (np.linalg.norm(cube_f[:2] - zone_f[:2]) < 0.07 and cube_f[2] < 0.49):
            print(f"subtask {si + 1} NOT completed, continuing anyway")

    # final check
    ok = True
    for obj_word, zone_word in subtasks:
        c, z = obj_pos(obj_word), obj_pos(zone_word)
        good = np.linalg.norm(c[:2] - z[:2]) < 0.07 and c[2] < 0.49
        print(f"{obj_word} -> {zone_word}: dist={np.linalg.norm(c[:2] - z[:2]):.3f} "
              f"z={c[2]:.3f} {'OK' if good else 'FAIL'}")
        ok = ok and good
    all_done = ok

    if all_done:
        hud.flash("TASK COMPLETE", (80, 255, 120))
        hud.banner_until = hud.sim_time + 3.0
        for _ in range(20):
            record()

    vid = os.path.join(out_dir, "episode.mp4")
    imageio.mimsave(vid, frames, fps=12, quality=7)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump({"task": TASK_FULL, "seed": seed, "backend": backend_name,
                   "success": bool(all_done), "ticks": int(tick),
                   "wall_s": round(time.time() - t0, 1),
                   "api_calls": len(backend.lat),
                   "api_latency_mean": round(float(np.mean(backend.lat)), 2) if backend.lat else 0,
                   "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames), api calls: {len(backend.lat)}, "
          f"success={all_done}")
    return all_done


def holding_generic(sim, body_id):
    """contact-based holding check for any cube body."""
    lf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    rf = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    finger_geoms = {g for g in range(sim.model.ngeom)
                    if sim.model.geom_bodyid[g] in (lf, rf)}
    cube_geoms = {g for g in range(sim.model.ngeom)
                  if sim.model.geom_bodyid[g] == body_id}
    for i in range(sim.data.ncon):
        con = sim.data.contact[i]
        pair = {con.geom1, con.geom2}
        if pair & cube_geoms and pair & finger_geoms:
            return True
    if not sim.gripper_open:
        mid = sim.tip_pos()
        c = sim.data.xpos[body_id]
        if np.linalg.norm(mid - c) < 0.05:
            return True
    return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "show0"))
    ap.add_argument("--backend", choices=["omnijev", "stepfun"], default="omnijev")
    ap.add_argument("--ticks", type=int, default=90)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--beauty", action="store_true")
    args = ap.parse_args()
    run(args.out, backend_name=args.backend, max_ticks=args.ticks, seed=args.seed,
        scene=args.scene, beauty=args.beauty)
