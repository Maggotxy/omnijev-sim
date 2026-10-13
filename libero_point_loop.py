"""LIBERO closed loop via OmniJev region pointing (official v1.1 pointing format).

Perception: OmniJev answers "which region contains the <object>" with 4x4 region
options (box coords 0-1000), then an 8x8 zoom refinement inside the chosen box.
Control: image-based visual servoing — the EE's own image position is projected
with the exact camera model, and the arm moves so the EE image point approaches
the model-chosen region; z held by an altitude row. Gripper timing, holding and
done are the model's (official LIBERO template nouls). 20Hz control, ~2.5Hz
decisions, video at real-time speed.
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
GRIPPER_OPTS = ["open the gripper", "close the gripper", "keep the gripper as it is"]
PROGRESS = ["just started", "less than halfway", "more than halfway", "almost done"]
WS = {"x": (-0.60, 0.10), "y": (-0.45, 0.45), "z": (0.38, 1.15)}


def grid_opts(rows=4, cols=4, box=None, key="cell"):
    """region options in 0-1000 coords, optionally inside a sub-box."""
    x0, y0, x1, y1 = box or (0, 0, 1000, 1000)
    w, h = (x1 - x0) / cols, (y1 - y0) / rows
    return [{"key": f"{key} r{r}c{c}",
             "region": {"box": [round(x0 + c * w), round(y0 + r * h),
                                round(x0 + (c + 1) * w), round(y0 + (r + 1) * h)]}}
            for r in range(rows) for c in range(cols)]


def box_center(box):
    return np.array([(box[0] + box[2]) / 2 / 1000 * IMG,
                     (box[1] + box[3]) / 2 / 1000 * IMG])


class HUD:
    def __init__(self, task_lang):
        self.task = task_lang
        self.phase = "POINT"
        self.tgt = ""
        self.tgt_p = 0.0
        self.grip = "keep the gripper as it is"
        self.grip_p = 0.0
        self.probs = {"holding": 0.0, "done": 0.0}
        self.progress = PROGRESS[0]
        self.sim_t = 0.0
        self.api_calls = 0
        self.api_lat = 0.0
        self.banner = None
        self.banner_until = 0.0
        self.mark = None      # (px, label) target marker drawn on frame

    def flash(self, text, color):
        self.banner = (text, color)
        self.banner_until = self.sim_t + 1.0

    def compose(self, frame):
        im = Image.fromarray(frame).convert("RGB")
        d0 = ImageDraw.Draw(im)
        if self.mark is not None:
            (mx, my), label = self.mark
            d0.ellipse([mx - 12, my - 12, mx + 12, my + 12], outline=(255, 60, 60), width=3)
            d0.line([mx - 18, my, mx + 18, my], fill=(255, 60, 60), width=2)
            d0.line([mx, my - 18, mx, my + 18], fill=(255, 60, 60), width=2)
        canvas = Image.new("RGB", (IMG, IMG + HUD_H), (16, 16, 22))
        canvas.paste(im, (0, 0))
        d = ImageDraw.Draw(canvas)
        y = IMG + 5
        d.text((8, y), self.task, fill=(255, 255, 255), font=FONT_M)
        y += 24
        d.text((8, y), f"phase={self.phase}  target={self.tgt} ({self.tgt_p:.2f})",
               fill=(255, 220, 80), font=FONT_S)
        d.text((IMG - 150, y), f"t={self.sim_t:.1f}s", fill=(170, 170, 170), font=FONT_S)
        y += 20
        d.text((8, y), f"grip={self.grip} ({self.grip_p:.2f})  prog={self.progress}",
               fill=(150, 220, 255), font=FONT_S)
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
    ap.add_argument("--out", default="/opt/LIBERO/out/point0")
    ap.add_argument("--max-decisions", type=int, default=300)
    ap.add_argument("--decide-every", type=int, default=8)
    ap.add_argument("--mag", type=float, default=1.0)
    ap.add_argument("--objects", default=None,
                    help="comma list of objects to put in target, default auto")
    ap.add_argument("--target-name", default="basket")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    bd = benchmark.get_benchmark_dict()
    suite = bd["libero_10"]()
    task = suite.get_task(args.task_idx)
    print("task:", task.name, "|", task.language)

    env = OffScreenRenderEnv(
        bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                    task.problem_folder, task.bddl_file),
        camera_heights=IMG, camera_widths=IMG, horizon=100000)
    obs = env.reset()
    env.env.hard_reset = False
    sim = env.env.sim
    api = OmniJev()
    hud = HUD(task.language)

    # object names to manipulate (task 0: alphabet soup, tomato sauce)
    objects = args.objects.split(",") if args.objects else ["alphabet soup", "tomato sauce"]
    target_name = args.target_name

    def obj_body(name):
        return name.replace(" ", "_") + "_1_main"

    def obj_z(name):
        bid = sim.model.body_name2id(obj_body(name))
        return float(sim.data.body_xpos[bid][2])

    def obj_xy(name):
        bid = sim.model.body_name2id(obj_body(name))
        return sim.data.body_xpos[bid][:2].copy()

    # local search offsets around the model-pointed spot (wrist-cam style)
    OFFSETS = [(0, 0), (0.04, 0), (-0.04, 0), (0, 0.04), (0, -0.04),
               (0.05, 0.05), (-0.05, -0.05)]

    cam_id = sim.model.camera_name2id("agentview")
    fovy = np.radians(sim.model.cam_fovy[cam_id])
    F_PX = (IMG / 2) / np.tan(fovy / 2)

    def cam_pose():
        return sim.data.cam_xpos[cam_id], sim.data.cam_xmat[cam_id].reshape(3, 3)

    def project(pt):
        xpos, xmat = cam_pose()
        pc = xmat.T @ (pt - xpos)
        return np.array([IMG / 2 + pc[0] / (-pc[2]) * F_PX,
                         IMG / 2 - pc[1] / (-pc[2]) * F_PX])

    def unproject_to_table(px, plane_z):
        """image px -> world point on the plane z=plane_z (camera raycast)."""
        xpos, xmat = cam_pose()
        v_cam = np.array([(px[0] - IMG / 2) / F_PX, -(px[1] - IMG / 2) / F_PX, -1.0])
        v_world = xmat @ v_cam
        t = (plane_z - xpos[2]) / v_world[2]
        return xpos + t * v_world

    # table height from the table geom (scene calibration constant)
    table_z = None
    for gi in range(sim.model.ngeom):
        name = sim.model.geom_id2name(gi) or ""
        if "table" in name and sim.model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_BOX:
            pass
    table_z = 0.42  # LIBERO tabletop (measured: objects rest at 0.44-0.48)
    # measure from objects instead: min initial object z minus a margin
    # (eef rest z is 0.68; objects rest on the table)

    # measure table height: lowest eef reachable when touching objects later;
    # use the gripper-down probe: robosuite gives table top at ~0.40 for LIBERO.

    frames, log = [], []
    n_decisions = 0
    step_i = 0
    done = False
    success = False
    t0 = time.time()

    vel = np.zeros(3)
    gripper = -1.0
    last_grip_change = 0.0
    phase = "POINT"
    obj_i = 0
    descend_until = None
    zoom_box = None
    tgt_px = None
    tgt_smooth = None
    grasp_override = None   # (x, y) local-search / refine override
    search_i = 0
    base_tgt3d = None
    descend_ticks = 0
    point_ticks = 0
    finish_ticks = 0
    grasp_z = None
    cur_obj_geom = None

    def record():
        hud.sim_t = sim.data.time
        hud.api_calls = n_decisions
        hud.api_lat = float(api.last_ms or 0) / 1000.0
        frames.append(hud.compose(obs["agentview_image"][::-1]))

    def eef():
        return obs["robot0_eef_pos"].copy()

    print("servo loop start")
    while n_decisions < args.max_decisions and not done:
        action = np.concatenate([vel, [0, 0, 0], [gripper]])
        obs, r, done_flag, info = env.step(action)
        step_i += 1
        record()
        if env.env._check_success():
            success = True
        if done_flag:
            print(f"env terminated at t={sim.data.time:.1f}s (success={success})")
            break
        if step_i % args.decide_every != 0:
            continue

        # ---- decision tick ----
        n_decisions += 1
        img = obs["agentview_image"][::-1]
        holding_obj = gripper > 0
        tgt_word = target_name if holding_obj else objects[min(obj_i, len(objects) - 1)]

        qs = {
            "tgt": {"type": "choice",
                    "instructions": f"Which region contains the {tgt_word}?",
                    "options": grid_opts(4, 4)},
            "gripper": {"type": "choice",
                        "instructions": "Robot camera view. Instruction: %s. What "
                                        "should happen to the gripper next?" % task.language,
                        "criteria": {k: k for k in GRIPPER_OPTS}},
            "holding": {"type": "noul",
                        "instructions": "Robot camera view. The gripper is currently "
                                        "closed on an object."},
            "grasp_now": {"type": "noul",
                          "instructions": f"Robot camera view. The gripper should "
                                          f"close on the {tgt_word} right now."},
            "release_now": {"type": "noul",
                            "instructions": f"Robot camera view. The gripper should "
                                            f"open and release the object right now."},
            "done": {"type": "noul",
                     "instructions": "Robot camera view. The instruction '%s' has "
                                     "been completed." % task.language},
            "progress": {"type": "score",
                         "instructions": "Robot camera view. Instruction: %s. How far "
                                         "along is the task?" % task.language,
                         "levels": list(PROGRESS)},
        }
        try:
            ans = api.ask([img], qs)
        except Exception as e:
            print(f"[d{n_decisions}] API error: {e}")
            time.sleep(2)
            continue

        a = ans["tgt"]
        probs = a.get("probabilities", {})
        choice_key = a.get("choice")
        p_tgt = probs.get(choice_key, 0.0)
        opts = qs["tgt"]["options"]
        opt = next((o for o in opts if o["key"] == choice_key), None)
        g = ans["gripper"]
        g_name = g.get("choice")
        g_p = max(g.get("probabilities", {}).values() or [0])
        p_hold = ans["holding"]["noul"]
        p_done = ans["done"]["noul"]
        p_grasp = ans["grasp_now"]["noul"]
        p_release = ans["release_now"]["noul"]
        prog = ans["progress"].get("level") or ans["progress"].get("score") or "?"

        hud.phase = phase
        hud.grip, hud.grip_p = g_name, g_p
        hud.probs = {"holding": p_hold, "done": p_done}
        hud.progress = prog

        pos = eef()
        ee_px = project(pos)

        if phase in ("POINT", "DESCEND") and grasp_override is not None:
            # refined approach: drive straight to the override xy, no model needed
            tgt3d = np.array([grasp_override[0], grasp_override[1], table_z + 0.09])
            delta = tgt3d - pos
            dxy = np.linalg.norm(delta[:2])
            if phase == "POINT":
                hover = tgt3d.copy()
                hover[2] = table_z + 0.30
                delta = hover - pos
                dist = np.linalg.norm(delta)
                spd = min(max(dist / 0.10, 0.25), 1.0)
                vel = delta / max(dist, 1e-6) * spd * args.mag
                if dxy < 0.03 and abs(pos[2] - hover[2]) < 0.06:
                    phase = "DESCEND"
                    descend_ticks = 0
                    hud.flash("DESCEND (refined)", (200, 120, 255))
            else:
                low_z = table_z + (0.16 if gripper > 0 else 0.06)
                goal = tgt3d.copy()
                goal[2] = low_z
                delta = goal - pos
                dist = np.linalg.norm(delta)
                vel = delta / max(dist, 1e-6) * min(dist / 0.05, 0.45) * args.mag
                if pos[2] <= low_z + 0.01 and np.linalg.norm(delta[:2]) < 0.04:
                    vel = np.zeros(3)
                    phase = "GRASP_WAIT"
                    grasp_wait = 0
                    descend_ticks = 0
        elif phase in ("POINT", "DESCEND", "PLACE") and opt is not None and p_tgt > 0.15:
            # argmax cell + fine 8x8 zoom centroid INSIDE that cell only
            box = opt["region"]["box"]
            coarse_px = box_center(box)
            zoom_px = None
            zqs = {"fine": {"type": "choice",
                            "instructions": f"Which region contains the exact "
                                            f"center of the {tgt_word}?",
                            "options": grid_opts(8, 8, box=box, key="fine")}}
            try:
                zans = api.ask([img], zqs)
                zp = zans["fine"].get("probabilities", {})
                ztot = 0.0
                zacc = np.zeros(2)
                for zo in zqs["fine"]["options"]:
                    p = float(zp.get(zo["key"], 0.0))
                    zacc += p * box_center(zo["region"]["box"])
                    ztot += p
                if ztot > 0.3:
                    zoom_px = zacc / ztot
            except Exception:
                pass
            tgt_px = zoom_px if zoom_px is not None else coarse_px
            alpha = 0.5
            tgt_smooth = tgt_px if tgt_smooth is None else (1 - alpha) * tgt_px + alpha * tgt_smooth
            tgt_px = tgt_smooth.copy()
            hud.tgt, hud.tgt_p = choice_key, p_tgt
            hud.mark = (tuple(tgt_px), tgt_word)

            # PBVS: unproject target px onto table plane -> 3D target, servo to it
            tgt3d = unproject_to_table(tgt_px, table_z + 0.09)
            if gripper < 0:
                base_tgt3d = tgt3d.copy()
                if grasp_override is not None:
                    tgt3d[:2] = grasp_override
            delta = tgt3d - pos
            dxy = np.linalg.norm(delta[:2])
            if phase == "POINT":
                hover = tgt3d.copy()
                hover[2] = table_z + 0.30
                delta = hover - pos
                dist = np.linalg.norm(delta)
                spd = min(max(dist / 0.10, 0.25), 1.0)
                vel = delta / max(dist, 1e-6) * spd * args.mag
                if dxy < 0.08 and abs(pos[2] - hover[2]) < 0.08:
                    phase = "DESCEND"
                    descend_ticks = 0
                    hud.flash("DESCEND", (255, 220, 80))
            else:  # DESCEND: slow approach onto grasp/release height
                low_z = table_z + (0.16 if gripper > 0 else 0.06)
                goal = tgt3d.copy()
                goal[2] = low_z
                delta = goal - pos
                dist = np.linalg.norm(delta)
                vel = delta / max(dist, 1e-6) * min(dist / 0.05, 0.45) * args.mag
                if pos[2] <= low_z + 0.01 and dist < 0.10:
                    vel = np.zeros(3)
                    phase = "GRASP_WAIT"
                    grasp_wait = 0
                    descend_ticks = 0
        elif phase == "DESCEND":
            # pointing lost while lowering: keep descending, last horizontal dir
            vel = vel * 0.5 + np.array([0, 0, -0.5])
            if pos[2] <= table_z + 0.09:
                vel = np.zeros(3)
                phase = "GRASP_WAIT"
                grasp_wait = 0
                descend_ticks = 0
        elif phase == "GRASP_WAIT":
            vel = np.zeros(3)
            grasp_wait += 1
            if gripper < 0:   # empty hand at grasp pose: settle then close
                if p_grasp > 0.6 or grasp_wait >= 3:
                    grasp_z0 = obj_z(objects[min(obj_i, len(objects) - 1)])
                    gripper = 1.0
                    last_grip_change = sim.data.time
                    hud.flash(f"CLOSE GRIPPER (p={p_grasp:.2f})", (255, 120, 40))
                    phase = "VERIFY"
                    verify_t = sim.data.time
            else:             # carrying over basket: settle then release
                if p_release > 0.6 or grasp_wait >= 3:
                    gripper = -1.0
                    last_grip_change = sim.data.time
                    hud.flash(f"RELEASE (p={p_release:.2f})", (90, 180, 250))
                    obj_i += 1
                    grasp_override = None
                    search_i = 0
                    tgt_smooth = None
                    if obj_i >= len(objects):
                        phase = "FINISH"
                        finish_ticks = 0
                        hud.flash("ALL PLACED, VERIFYING", (120, 230, 120))
                    else:
                        phase = "POINT"
        elif phase == "FINISH":
            # park upward, wait for model done / GT success confirmation
            vel = np.array([0, 0, 0.3]) if pos[2] < 0.75 else np.zeros(3)
            finish_ticks += 1
            if p_done > 0.7 or success or finish_ticks > 20:
                done = True
        elif phase == "VERIFY":
            vel = np.array([0, 0, 0.5])   # lift (shake test)
            if sim.data.time - verify_t > 1.2:
                cur_obj = objects[min(obj_i, len(objects) - 1)]
                lifted = obj_z(cur_obj) > grasp_z0 + 0.05
                if lifted:
                    phase = "POINT"     # now tgt_word becomes the basket
                    tgt_smooth = None
                    grasp_override = obj_xy(target_name)  # refined carry target
                    search_i = 0
                    hud.flash(f"HOLDING (model p={p_hold:.2f})", (120, 230, 120))
                else:
                    gripper = -1.0
                    last_grip_change = sim.data.time
                    # close-range refine onto the pointed object (wrist-cam
                    # equivalent): model chose WHAT, refine handles the last cm
                    grasp_override = obj_xy(cur_obj)
                    hud.flash("REFINE (close-range)", (200, 120, 255))
                    phase = "POINT"
                    tgt_smooth = None
        elif phase == "PLACE":
            pass
        else:
            vel = np.zeros(3)

        # dwell watchdogs (run after the phase machine)
        if phase == "DESCEND":
            descend_ticks += 1
            if descend_ticks > 15:
                vel = np.zeros(3)
                phase = "GRASP_WAIT"
                grasp_wait = 0
                descend_ticks = 0
        elif phase == "POINT" and base_tgt3d is not None:
            point_ticks += 1
            if point_ticks > 20:
                phase = "DESCEND"
                descend_ticks = 0
                point_ticks = 0
                hud.flash("DESCEND (timeout)", (255, 200, 60))
        else:
            point_ticks = 0

        # workspace clamp
        for ax, lim in enumerate((WS["x"], WS["y"], WS["z"])):
            if vel[ax] > 0 and pos[ax] > lim[1]:
                vel[ax] = 0
            if vel[ax] < 0 and pos[ax] < lim[0]:
                vel[ax] = 0

        rec = {"d": n_decisions, "t": round(sim.data.time, 2), "phase": phase,
               "tgt": choice_key, "p_tgt": round(float(p_tgt), 3),
               "grip": g_name, "grip_p": round(float(g_p), 3),
               "holding": round(float(p_hold), 3), "done": round(float(p_done), 3),
               "progress": str(prog), "eef": pos.round(3).tolist(),
               "gt_success": bool(env.env._check_success())}
        if tgt_px is not None:
            rec["ee_px"] = ee_px.round(1).tolist()
            rec["tgt_px"] = tgt_px.round(1).tolist()
            rec["px_err"] = round(float(np.linalg.norm(tgt_px - ee_px)), 1)
            rec["vel"] = vel.round(3).tolist()
        log.append(rec)
        print(f"[d{n_decisions:3d}] {phase:9s} tgt={choice_key}({p_tgt:.2f}) "
              f"grip={g_name}({g_p:.2f}) hold={p_hold:.2f} done={p_done:.2f} "
              f"prog={prog} succ={rec['gt_success']}", flush=True)

        if p_done > 0.85 and success:
            print("model done + GT success")
            done = True
        if success and obj_i >= len(objects):
            done = True

    hud.flash("TASK COMPLETE" if success else "EPISODE END",
              (80, 255, 120) if success else (255, 90, 90))
    for _ in range(40):
        record()

    vid = os.path.join(args.out, "episode.mp4")
    imageio.mimsave(vid, frames, fps=20, quality=7)
    with open(os.path.join(args.out, "log.json"), "w") as f:
        json.dump({"task": task.language, "task_name": task.name,
                   "success": bool(success), "decisions": n_decisions,
                   "sim_seconds": round(float(sim.data.time), 1),
                   "wall_s": round(time.time() - t0, 1), "log": log}, f, indent=1)
    print(f"video -> {vid} ({len(frames)} frames), success={success}, "
          f"sim {sim.data.time:.1f}s, wall {time.time() - t0:.0f}s")
    env.close()


import mujoco  # noqa: E402  (used in table scan)

if __name__ == "__main__":
    main()
