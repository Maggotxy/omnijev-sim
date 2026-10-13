"""Three policies, one initial state and one instrumented skill controller.

All arms receive simulator object labels and share simulator pose assistance.
This is an assisted planning/reflex ablation, not a vision-only benchmark.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .clients import Gemini, Omni, redact

MODES = ("gemini", "gemini_omni", "omni")
TITLES = {"gemini": "Gemini", "gemini_omni": "Gemini + OmniJev", "omni": "OmniJev"}
POLICIES = {"gemini": ("gemini", "gemini"), "gemini_omni": ("gemini", "omni"), "omni": ("omni", "omni")}
IMG, FPS = 512, 20
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def title(mode, planner_model=None):
    if mode == "omni":
        return TITLES[mode]
    names = {"gemini-2.5-flash": "Gemini 2.5 Flash", "gemini-robotics-er-2-preview": "Gemini Robotics ER2"}
    name = names.get(planner_model, planner_model or "Gemini")
    return name if mode == "gemini" else name + " + OmniJev"


def panel(frame, mode, phase, target, sim_s, calls, decision, success=None, planner_model=None):
    canvas = Image.new("RGB", (IMG, 736), (15, 20, 30))
    canvas.paste(Image.fromarray(frame), (0, 64))
    draw = ImageDraw.Draw(canvas)
    color = {"gemini": (122, 182, 255), "gemini_omni": (135, 223, 180), "omni": (250, 195, 108)}[mode]
    heading = title(mode, planner_model)
    draw.text((16, 10), heading, fill=color, font=font(22 if len(heading) > 23 else 25))
    draw.text((16, 43), "Shared simulator poses + OSC skills | 20 Hz", fill=(180, 190, 202), font=font(14))
    rows = [f"Phase: {phase}", f"Target: {target}",
            f"Sim {sim_s:.2f}s | API calls {calls}", f"Model gate: {decision}"]
    if success is not None:
        rows.append("Official task success: " + str(success))
    for i, row in enumerate(rows):
        draw.text((16, 588 + i * 27), row[:62], fill=color if i == 0 else (227, 233, 241), font=font(17))
    return np.asarray(canvas)


class Episode:
    def __init__(self, args, mode, seed, output, clients):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.args, self.mode, self.seed, self.output = args, mode, seed, output
        self.clients = clients
        self.planner, self.reflex = (clients[x] for x in POLICIES[mode])
        random.seed(seed)
        np.random.seed(seed)
        suite = benchmark.get_benchmark_dict()["libero_10"]()
        self.task = suite.get_task(args.task_idx)
        if args.task_idx != 0:
            raise ValueError("This calibrated pick-and-place controller currently supports LIBERO-10 task 0")
        import torch
        # LIBERO's checked-in initial states contain NumPy arrays, not weights.
        # Load only the official task file selected by this installed suite.
        state_file = Path(get_libero_path("init_states")) / self.task.problem_folder / self.task.init_states_file
        states = torch.load(state_file, weights_only=False)
        if not 0 <= args.init_index < len(states):
            raise ValueError("init-index outside the official initial-state dataset")
        initial = np.asarray(states[args.init_index], dtype=float)
        self.initial_sha = sha(initial)
        self.env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), self.task.problem_folder, self.task.bddl_file),
            camera_names=["agentview"], camera_heights=IMG, camera_widths=IMG,
            horizon=100000, control_freq=FPS)
        self.env.seed(seed)
        self.env.reset()
        self.obs = self.env.set_init_state(initial)
        self.env.env.hard_reset = False
        self.sim = self.env.sim
        self.bodies = {name[:-7].replace("_", " "): name for name in self.sim.model.body_names
                       if name and name.endswith("_1_main")}
        self.phase, self.target, self.decision = "INITIAL", "", "pending"
        self.gripper = -1.0
        self.steps = 0
        self.start_sim = float(self.sim.data.time)
        self.started = time.monotonic()
        self.completed, self.events = [], []
        self.actions = open(output / "actions.jsonl", "w")
        self.writer = imageio.get_writer(str(output / "episode.mp4"), fps=FPS, quality=7)
        self.call_start = {name: len(client.records) for name, client in clients.items()}
        self.gate_counts = {"proceed": 0, "wait": 0, "replan": 0}
        self.grasp_failures = 0
        for _ in range(10):
            self.step(np.zeros(3))
        self.settled_sha = sha(self.env.get_sim_state())
        Image.fromarray(self.frame()).save(output / "initial.png")

    def frame(self):
        return self.obs["agentview_image"][::-1].copy()

    def calls(self):
        return sum(len(c.records) - self.call_start[n] for n, c in self.clients.items())

    def pos(self, name):
        return self.sim.data.body_xpos[self.sim.model.body_name2id(self.bodies[name])].copy()

    def eef(self):
        return self.obs["robot0_eef_pos"].copy()

    def step(self, velocity):
        action = np.r_[np.clip(velocity, -1, 1), [0, 0, 0], self.gripper]
        self.obs, _, _, _ = self.env.step(action)
        self.steps += 1
        success = bool(self.env.check_success())
        self.actions.write(json.dumps({"step": self.steps, "sim_s": float(self.sim.data.time) - self.start_sim,
                                       "phase": self.phase, "action": action.tolist(), "eef": self.eef().tolist(),
                                       "official_success": success}) + "\n")
        self.writer.append_data(panel(self.frame(), self.mode, self.phase, self.target,
                                      float(self.sim.data.time) - self.start_sim, self.calls(), self.decision,
                                      planner_model=self.planner.model))
        if self.steps % 100 == 0:
            print(f"{self.mode} seed={self.seed} steps={self.steps} phase={self.phase}", flush=True)
        if self.steps >= self.args.max_steps:
            raise RuntimeError("control_step_budget")

    def hold(self, seconds):
        for _ in range(round(seconds * FPS)):
            self.step(np.zeros(3))

    def servo(self, goal, max_steps=140):
        goal = np.asarray(goal, dtype=float)
        # Same workspace constraints and controller gains in all policy arms.
        if not (-0.65 <= goal[0] <= 0.15 and -0.48 <= goal[1] <= 0.48 and 0.45 <= goal[2] <= 1.05):
            raise RuntimeError("skill_target_outside_workspace")
        for _ in range(max_steps):
            delta = goal - self.eef()
            if np.linalg.norm(delta) < 0.012:
                return
            self.step(np.clip(delta * 9, -0.7, 0.7))
        raise RuntimeError("servo_timeout")

    def event(self, kind, **data):
        entry = {"kind": kind, "step": self.steps, "phase": self.phase, **data}
        self.events.append(entry)
        with open(self.output / "decisions.jsonl", "a") as stream:
            stream.write(json.dumps(entry) + "\n")

    def gate(self, phase, description):
        self.phase = phase
        for attempt in range(self.args.max_gate_waits):
            context = (f"Task: {self.task.language}; proposed={description}; current phase={phase}; "
                       f"current subtask={self.target}; completed placements={self.completed}. "
                       "A shared calibrated pose controller executes the skill. The arm is stationary at this checkpoint.")
            result = self.reflex.gate(self.frame(), context)
            self.decision = result["action"]
            self.gate_counts[self.decision] += 1
            self.event("gate", context=context, result=result, attempt=attempt)
            if self.decision == "proceed":
                return True
            if self.decision == "replan":
                return False
            self.hold(0.4)
        raise RuntimeError("model_gate_timeout")

    def execute(self, obj, target):
        if target != "basket" or obj == "basket":
            raise RuntimeError("unsupported_skill_plan: this controller supports placing small objects into basket")
        self.target = f"{obj} -> {target}"
        self.gripper = -1.0
        start = self.pos(obj)
        if not self.gate("APPROACH", f"move the open gripper above {obj}"):
            return False
        self.servo([*start[:2], 0.75])
        if not self.gate("DESCEND", f"lower the open gripper around {obj}"):
            return False
        self.servo([*self.pos(obj)[:2], 0.48])
        if not self.gate("GRASP", f"close the gripper to grasp {obj}"):
            return False
        self.gripper = 1.0
        self.hold(0.8)
        if not self.gate("LIFT", f"lift the grasped {obj} to transport height"):
            return False
        self.servo([*self.eef()[:2], 0.75])
        lifted = bool(self.pos(obj)[2] > start[2] + 0.05)
        self.event("simulated_grasp_sensor", object=obj, lifted=lifted,
                   initial_z=float(start[2]), current_z=float(self.pos(obj)[2]))
        if not lifted:
            self.grasp_failures += 1
            return False
        if not self.gate("CARRY", f"transport the held {obj} above {target}"):
            return False
        destination = self.pos(target)
        self.servo([*destination[:2], 0.75])
        if not self.gate("LOWER", f"lower the held {obj} into {target}"):
            return False
        self.servo([*destination[:2], 0.59])
        if not self.gate("RELEASE", f"open the gripper to release {obj} into {target}"):
            return False
        self.gripper = -1.0
        self.hold(1.0)
        self.servo([*self.eef()[:2], 0.78])
        self.hold(0.4)
        placed = bool(np.linalg.norm(self.pos(obj)[:2] - self.pos(target)[:2]) < 0.12 and self.pos(obj)[2] < 0.60)
        self.event("simulated_placement_sensor", object=obj, target=target, placed=placed)
        if placed:
            self.completed.append(obj)
        return placed

    def run(self):
        status, error, model_done = "plan_budget", None, False
        try:
            for _ in range(self.args.max_plans):
                self.phase = "PLAN"
                plan = self.planner.plan(self.frame(), self.task.language, list(self.bodies), self.completed)
                self.event("plan", result=plan)
                if plan["object"] == "finish":
                    model_done = True
                    status = "model_finished"
                    break
                ok = self.execute(plan["object"], plan["target"])
                if not ok:
                    # Identical recovery primitive, logged explicitly in every arm.
                    self.phase = "RECOVER"
                    self.event("controller_recovery", reason="replan_or_failed_grasp")
                    self.gripper = -1.0
                    self.hold(0.6)
                    self.servo([*self.eef()[:2], 0.78])
                if self.env.check_success():
                    status = "success"
                    break
        except Exception as exc:
            status, error = "stopped", redact(str(exc))
        finally:
            success = bool(self.env.check_success())
            final = panel(self.frame(), self.mode, status, self.target,
                          float(self.sim.data.time) - self.start_sim, self.calls(), self.decision, success,
                          planner_model=self.planner.model)
            for _ in range(FPS * 2):
                self.writer.append_data(final)
            Image.fromarray(final).save(self.output / "final.png")
            self.writer.close()
            self.actions.close()
            records = [r for name, c in self.clients.items() for r in c.records[self.call_start[name]:]]
            result = {"mode": self.mode, "seed": self.seed, "init_index": self.args.init_index,
                      "task_name": self.task.name, "task": self.task.language, "status": status,
                      "success": success, "model_done": model_done, "error": error,
                      "initial_state_sha256": self.initial_sha, "settled_state_sha256": self.settled_sha,
                      "final_state_sha256": sha(self.env.get_sim_state()),
                      "sim_seconds": self.steps / FPS, "wall_seconds": round(time.monotonic() - self.started, 3),
                      "control_steps": self.steps, "api_calls": len(records),
                      "api_seconds": round(sum(r["latency_s"] for r in records), 3),
                      "api_errors": sum("error" in r for r in records),
                      "gate_counts": self.gate_counts, "grasp_failures": self.grasp_failures,
                      "completed": self.completed, "planner": self.planner.model, "reflex": self.reflex.model,
                      "assistance": ["simulator object labels and poses", "object height grasp verification", "placement sensor", "official success evaluator"],
                      "controller": {"hz": FPS, "gain": 9, "speed_clip": 0.7, "grasp_z": 0.48, "release_z": 0.59},
                      "limits": {"plans": self.args.max_plans, "gate_waits": self.args.max_gate_waits, "steps": self.args.max_steps}}
            (self.output / "result.json").write_text(json.dumps(result, indent=2))
            self.env.close()
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 7])
    parser.add_argument("--task-idx", type=int, default=0)
    parser.add_argument("--init-index", type=int, default=0)
    parser.add_argument("--max-plans", type=int, default=4)
    parser.add_argument("--max-gate-waits", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=2400)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Use a fresh output directory to preserve earlier evidence")
    args.out.mkdir(parents=True)
    manifest = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in Path(__file__).parent.glob("*.py")},
                "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    clients = {"gemini": Gemini(args.out / "gemini_api.jsonl"), "omni": Omni(args.out / "omni_api.jsonl")}
    availability = {}
    for key in dict.fromkeys(p for mode in args.modes for p in POLICIES[mode]):
        try:
            clients[key].health()
            availability[key] = {"available": True}
        except Exception as exc:
            availability[key] = {"available": False, "error": redact(str(exc))}
    (args.out / "availability.json").write_text(json.dumps(availability, indent=2))
    results = []
    for seed in args.seeds:
        for mode in args.modes:
            output = args.out / f"{mode}_seed{seed}"
            output.mkdir()
            blocked = [key for key in POLICIES[mode] if not availability[key]["available"]]
            if blocked:
                result = {"mode": mode, "seed": seed, "status": "unavailable", "success": None,
                          "error": availability[blocked[0]]["error"], "init_index": args.init_index,
                          "planner": clients[POLICIES[mode][0]].model, "reflex": clients[POLICIES[mode][1]].model}
                (output / "result.json").write_text(json.dumps(result, indent=2))
            else:
                try:
                    result = Episode(args, mode, seed, output, clients).run()
                except Exception as exc:
                    result = {"mode": mode, "seed": seed, "status": "initialization_error", "success": None,
                              "error": redact(str(exc)), "init_index": args.init_index}
                    (output / "result.json").write_text(json.dumps(result, indent=2))
            results.append(result)
            (args.out / "summary.json").write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
