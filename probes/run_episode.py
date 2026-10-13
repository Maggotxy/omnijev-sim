"""Closed-loop episode runner: render -> ask OmniJev -> act -> record."""
import argparse
import json
import os
import time

import numpy as np
import imageio.v2 as imageio

from sim import Sim
from omnijev_api import OmniJev, question_set, DIR_VEC, TASK

ROOT = os.path.dirname(os.path.abspath(__file__))


def run_episode(out_dir, max_ticks=100, seed=0, decision_hz=2.0, speed=0.18,
                grasp_th=0.5, release_th=0.5, done_th=0.7, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    sim = Sim()
    sim.reset(rng=np.random.default_rng(seed))
    api = OmniJev()
    qs = question_set()

    frames = []
    log = []
    grasped = False
    grip_cooldown = 0          # ticks before the gripper may toggle again
    done = False
    t0 = time.time()

    GRASP_TH, RELEASE_TH, COOLDOWN = 0.65, 0.70, 6
    HOLD_LOST_TH = 0.40        # model holding-confidence below this = dropped

    for tick in range(max_ticks):
        agent = sim.render("agentview")
        wrist = sim.render_wrist()
        t_ask = time.time()
        try:
            ans = api.ask([agent, wrist], qs)
        except Exception as e:
            log.append({"tick": tick, "error": str(e)})
            print(f"[{tick}] API error: {e}")
            time.sleep(2)
            continue
        dt_ask = time.time() - t_ask

        d = ans.get("dir", {})
        direc = d.get("choice")
        p_grasp = ans.get("grasp", {}).get("noul", 0)
        p_release = ans.get("release", {}).get("noul", 0)
        p_hold = ans.get("holding", {}).get("noul", 0)
        p_done = ans.get("done", {}).get("noul", 0)

        # --- decision -> action (state machine + hysteresis + cooldown) ---
        if grip_cooldown > 0:
            grip_cooldown -= 1
        if not grasped and p_grasp > GRASP_TH and grip_cooldown == 0:
            sim.set_gripper(False)
            sim.settle(0.8)  # let fingers close before judging
            grasped = True
            grip_cooldown = COOLDOWN
            # self-recovery: model *and* ground truth both say not holding -> reopen
            p_hold_now = ans.get("holding", {}).get("noul", 1)
            if not sim.holding() and p_hold_now < 0.5:
                sim.set_gripper(True)
                grasped = False
                action = "GRASP_MISS"
            else:
                action = "GRASP"
        elif grasped and (p_release > RELEASE_TH or p_hold < HOLD_LOST_TH) and grip_cooldown == 0:
            sim.set_gripper(True)
            grasped = False
            grip_cooldown = COOLDOWN
            action = "RELEASE"
        else:
            v = DIR_VEC.get(direc, (0, 0, 0))
            sim.move_ee(np.array(v, dtype=float), speed=speed)
            action = f"move:{direc}"

        # physics between decisions: ~1/decision_hz of sim time
        sim.data.ctrl[:7] = sim.q_target
        sim.settle(1.0 / decision_hz)

        gt_hold = bool(sim.holding())
        succ = bool(sim.success())
        rec = {"tick": tick, "action": action, "dir": direc,
               "p": {"grasp": p_grasp, "release": p_release,
                     "hold": p_hold, "done": p_done},
               "gt_holding": gt_hold, "success": succ,
               "ee": sim.ee_pos().round(3).tolist(),
               "cube": sim.cube_pos().round(3).tolist(),
               "api_ms": api.last_ms, "wall_ask_s": round(dt_ask, 2)}
        log.append(rec)
        if verbose:
            print(f"[{tick:3d}] {action:14s} grasp={p_grasp:.2f} rel={p_release:.2f} "
                  f"hold={p_hold:.2f} done={p_done:.2f} gt_hold={gt_hold} succ={succ} "
                  f"ask={dt_ask:.1f}s")
        frames.append(agent)

        if succ:
            print(f"SUCCESS at tick {tick}")
            done = True
            break
        if p_done > done_th and tick > 10:
            print(f"model says done at tick {tick} (p={p_done:.2f}), success={succ}")
            done = succ
            break

    # save video + log
    vid = os.path.join(out_dir, "episode.mp4")
    imageio.mimsave(vid, frames, fps=4, quality=7)
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump({"task": TASK, "seed": seed, "success": done,
                   "ticks": len(log), "wall_s": round(time.time() - t0, 1),
                   "log": log}, f, indent=1)
    print(f"video -> {vid}  ({len(frames)} frames)")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "ep0"))
    ap.add_argument("--ticks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hz", type=float, default=2.0)
    args = ap.parse_args()
    run_episode(args.out, max_ticks=args.ticks, seed=args.seed, decision_hz=args.hz)
