import os
"""Benchmark: OmniJev-0.8B vs StepFun step-1o-turbo-vision on the robot scene.

Suite A: grid-cell localization (16 probes, one per cell, ground truth by teleport)
Suite B: noul decision calibration (6 scenarios with ground truth labels)
Both models see IDENTICAL images and equivalent questions.
Output: bench_results.json + printed table.
"""
import base64
import io
import json
import re
import time
import urllib.request

import numpy as np
import mujoco
from PIL import Image, ImageDraw, ImageFont

from sim import Sim
from omnijev_api import OmniJev, TASK

GRID, IMG = 4, 640
CELL_PX = IMG // GRID
CELLS = [f"{c}{r}" for r in range(1, GRID + 1) for c in "ABCD"]
TABLE_Z = 0.42
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


class StepFun:
    KEY = os.environ.get("STEPFUN_API_KEY", "")
    MODEL = "step-1o-turbo-vision"

    def ask_text(self, img, text):
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"model": self.MODEL, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text}]}], "temperature": 0.0}
        req = urllib.request.Request(
            "https://api.stepfun.com/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.KEY}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.loads(r.read())
        return out["choices"][0]["message"]["content"]


# ---------------- sim helpers ----------------

def make_cam(sim):
    class Cam:
        cid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")

        def pose(self):
            return (sim.data.cam_xpos[self.cid],
                    sim.data.cam_xmat[self.cid].reshape(3, 3))

        def project(self, pt):
            xpos, xmat = self.pose()
            fovy = np.radians(sim.model.cam_fovy[self.cid])
            f_px = (IMG / 2) / np.tan(fovy / 2)
            pc = xmat.T @ (pt - xpos)
            return np.array([IMG / 2 + (pc[0] / (-pc[2])) * f_px,
                             IMG / 2 - (pc[1] / (-pc[2])) * f_px])

        def cell_of(self, px):
            col = int(np.clip(px[0] // CELL_PX, 0, GRID - 1))
            row = int(np.clip(px[1] // CELL_PX, 0, GRID - 1))
            return CELLS[row * GRID + col]

        def unproject(self, px, z_plane):
            xpos, xmat = self.pose()
            fovy = np.radians(sim.model.cam_fovy[self.cid])
            f_px = (IMG / 2) / np.tan(fovy / 2)
            d_cam = np.array([(px[0] - IMG / 2) / f_px, -(px[1] - IMG / 2) / f_px, -1.0])
            d = xmat @ d_cam
            d /= np.linalg.norm(d)
            t = (z_plane - xpos[2]) / d[2]
            return xpos + t * d
    return Cam()


def set_cube(sim, x, y):
    jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint")
    adr = sim.model.jnt_qposadr[jid]
    dad = sim.model.jnt_dofadr[jid]
    sim.data.qpos[adr:adr + 7] = [x, y, TABLE_Z + 0.035, 1, 0, 0, 0]
    sim.data.qvel[dad:dad + 6] = 0
    mujoco.mj_forward(sim.model, sim.data)
    sim.settle(0.3)


def servo(sim, target, tol=0.005, max_steps=300, max_step=0.015):
    lo = np.array([sim.model.jnt_range[k][0] + 1e-3 for k in range(7)])
    hi = np.array([sim.model.jnt_range[k][1] - 1e-3 for k in range(7)])
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
    return False


# ---------------- benchmark ----------------

def main():
    sim = Sim(img_w=IMG, img_h=IMG)
    sim.reset(rng=np.random.default_rng(1))
    omni = OmniJev()
    step = StepFun()
    cam = make_cam(sim)
    results = {"suiteA": [], "suiteB": [], "latency": {"omnijev": [], "stepfun": []}}

    CELL_Q = ("This robot scene has a labeled 4x4 grid (columns A-D left to right, "
              "rows 1-4 top to bottom). Which cell contains the red cube?")
    CELL_Q_VLM = CELL_Q + " Answer with ONLY the cell label, e.g. 'B2'."

    # ---------- Suite A: localization ----------
    # 16 points on the table surface; true cell = pinhole projection of the cube
    print("=== Suite A: grid localization (16 on-table points, arm parked) ===")
    PARK = np.array([0.30, 0.38, 0.80])   # out of the way, above the table edge
    xs = np.linspace(0.38, 0.72, 4)
    ys = np.linspace(-0.22, 0.22, 4)
    points = [(x, y) for y in ys for x in xs]
    for x, y in points:
        set_cube(sim, float(x), float(y))
        servo(sim, PARK)                  # unobstructed view of the whole grid
        img = bold_grid(sim.render("agentview"))
        cell = cam.cell_of(cam.project(sim.cube_pos()))

        t0 = time.time()
        ans = omni.ask([img], {"tgt": {"type": "choice", "instructions": CELL_Q,
                                       "criteria": {c: f"cell {c}" for c in CELLS}}})
        lat_o = time.time() - t0
        o_cell = ans["tgt"].get("choice")
        o_p = max(ans["tgt"].get("probabilities", {}).values() or [0])

        t0 = time.time()
        try:
            txt = step.ask_text(img, CELL_Q_VLM)
        except Exception as e:
            txt = f"ERROR: {e}"
        lat_s = time.time() - t0
        m = re.search(r"\b([A-D][1-4])\b", txt)
        s_cell = m.group(1) if m else None

        results["latency"]["omnijev"].append(round(lat_o, 2))
        results["latency"]["stepfun"].append(round(lat_s, 2))
        ok_o, ok_s = o_cell == cell, s_cell == cell
        results["suiteA"].append({"cell": cell, "omnijev": o_cell,
                                  "omnijev_p": round(float(o_p), 3),
                                  "stepfun": s_cell, "stepfun_raw": txt[:80],
                                  "omnijev_ok": ok_o, "stepfun_ok": ok_s})
        print(f"{cell}: omnijev={o_cell}({o_p:.2f}) {'OK' if ok_o else 'X'} | "
              f"stepfun={s_cell} {'OK' if ok_s else 'X'} (raw {txt[:40]!r})", flush=True)

    # ---------- Suite B: noul decisions ----------
    print("=== Suite B: decision calibration (6 scenarios) ===")
    cube = sim.cube_pos().copy()

    def ask_noul(instr):
        img = sim.render("agentview")
        t0 = time.time()
        a = omni.ask([img], {"q": {"type": "noul", "instructions": instr}})
        lat_o = time.time() - t0
        t0 = time.time()
        try:
            txt = step.ask_text(img, instr + " Answer ONLY 'YES' or 'NO'.")
        except Exception as e:
            txt = f"ERROR: {e}"
        lat_s = time.time() - t0
        results["latency"]["omnijev"].append(round(lat_o, 2))
        results["latency"]["stepfun"].append(round(lat_s, 2))
        s_yes = txt.strip().upper().startswith("YES")
        return a["q"]["noul"], s_yes, txt[:40]

    scenarios = []

    # B1 should-grasp YES (bare statement + task-prefixed)
    set_cube(sim, 0.52, 0.04)
    cube = sim.cube_pos().copy()
    servo(sim, np.array([cube[0], cube[1], cube[2] + 0.03]))
    instr_bare = "The gripper should close on the red cube right now."
    instr = TASK + " " + instr_bare
    for tag, ins in (("grasp_now", instr), ("grasp_now_bare", instr_bare)):
        p, s_yes, raw = ask_noul(ins)
        results["suiteB"].append({"scenario": tag, "truth": True,
                                  "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                                  "stepfun_yes": s_yes, "stepfun_raw": raw})
        print(f"{tag}(yes): omni p={p:.2f} stepfun={s_yes}")

    # B2 should-grasp NO (gripper far)
    servo(sim, np.array([0.35, 0.25, 0.75]))
    p, s_yes, raw = ask_noul(instr)
    results["suiteB"].append({"scenario": "grasp_far", "truth": False,
                              "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                              "stepfun_yes": s_yes, "stepfun_raw": raw})
    print(f"grasp_far(no): omni p={p:.2f} stepfun={s_yes}")

    # B3 holding YES
    servo(sim, np.array([cube[0], cube[1], cube[2] + 0.13]))
    servo(sim, np.array([cube[0], cube[1], cube[2] + 0.03]))
    sim.set_gripper(False)
    sim.settle(0.8)
    servo(sim, sim.tip_pos() + np.array([0, 0, 0.15]), max_step=0.008)
    hold_ok = bool(sim.holding())
    instr_h = TASK + " The gripper is currently holding the red cube."
    p, s_yes, raw = ask_noul(instr_h)
    results["suiteB"].append({"scenario": "holding", "truth": hold_ok,
                              "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                              "stepfun_yes": s_yes, "stepfun_raw": raw})
    print(f"holding({hold_ok}): omni p={p:.2f} stepfun={s_yes}")

    # B4 holding NO (empty gripper, released)
    sim.set_gripper(True)
    sim.settle(0.6)
    servo(sim, np.array([0.45, 0.1, 0.70]))
    p, s_yes, raw = ask_noul(instr_h)
    results["suiteB"].append({"scenario": "not_holding", "truth": False,
                              "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                              "stepfun_yes": s_yes, "stepfun_raw": raw})
    print(f"not_holding(no): omni p={p:.2f} stepfun={s_yes}")

    # B5 done YES: cube in zone, arm retreated
    zone = sim.zone_pos()
    set_cube(sim, float(zone[0]), float(zone[1]))
    servo(sim, np.array([0.45, 0.1, 0.70]))
    instr_d = TASK + " The task is finished."
    p, s_yes, raw = ask_noul(instr_d)
    results["suiteB"].append({"scenario": "done", "truth": True,
                              "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                              "stepfun_yes": s_yes, "stepfun_raw": raw})
    print(f"done(yes): omni p={p:.2f} stepfun={s_yes}")

    # B6 done NO: cube far from zone
    set_cube(sim, float(zone[0]) - 0.25, float(zone[1]) - 0.15)
    p, s_yes, raw = ask_noul(instr_d)
    results["suiteB"].append({"scenario": "not_done", "truth": False,
                              "omnijev_p": round(float(p), 3), "omnijev_yes": p > 0.5,
                              "stepfun_yes": s_yes, "stepfun_raw": raw})
    print(f"not_done(no): omni p={p:.2f} stepfun={s_yes}")

    # ---------- summary ----------
    a_o = sum(r["omnijev_ok"] for r in results["suiteA"])
    a_s = sum(r["stepfun_ok"] for r in results["suiteA"])
    b_o = sum((r["omnijev_yes"] == r["truth"]) for r in results["suiteB"])
    b_s = sum((r["stepfun_yes"] == r["truth"]) for r in results["suiteB"])
    lo = results["latency"]["omnijev"]
    ls = results["latency"]["stepfun"]
    nb = len(results["suiteB"])
    results["summary"] = {
        "localization_acc": {"omnijev": f"{a_o}/16", "stepfun": f"{a_s}/16"},
        "decision_acc": {"omnijev": f"{b_o}/{nb}", "stepfun": f"{b_s}/{nb}"},
        "latency_mean_s": {"omnijev": round(sum(lo) / len(lo), 2),
                           "stepfun": round(sum(ls) / len(ls), 2)},
    }
    with open("out/bench_results.json", "w") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("\n=== SUMMARY ===")
    print(json.dumps(results["summary"], indent=1))


if __name__ == "__main__":
    main()
