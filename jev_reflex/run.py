"""Run real-time physics while inference and video recording work concurrently."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "osmesa")

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .policy import ActionMailbox, HTTPReflex, LatestWorker, Observation
from .scenario import KINDS, Scenario
from .render import RenderState, RenderWorker
from .world import DT, World
from .bank import ReflexBank, ReflexMemory, signature
from .brain import VisualBrain
from .control import choose_motor
from .motion import motion_responses

FPS = 20


def font(size):
    return ImageFont.truetype("C:/Windows/Fonts/arial.ttf" if os.name == "nt" else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


def panel(frame, view, mode, sim_s, wall_s, action, age, sensors, metrics, event, last_reaction, camera_age=0, brain_status="", bank_version=0, motor_source=""):
    canvas = Image.new("RGB", (960, 720), (11, 17, 27))
    d = ImageDraw.Draw(canvas)
    color = (121, 228, 173) if mode != "baseline" else (169, 183, 205)
    title = "Visual brain only" if mode == "vlm" else "Visual brain + OmniJev cerebellum" if mode == "vlm_omni" else "OmniJev reflex" if mode != "baseline" else "Same motor plan | no model reflex"
    d.text((18, 12), title, fill=color, font=font(24))
    d.text((18, 49), f"World {sim_s:.2f}s | Wall {wall_s:.2f}s | camera age {camera_age:.2f}s | continuous physics", fill=(209, 220, 237), font=font(16))
    canvas.paste(Image.fromarray(view).resize((480, 480)), (0, 88))
    canvas.paste(Image.fromarray(frame).resize((480, 480)), (480, 88))
    d.text((18, 94), "Observer camera", fill=(244, 245, 248), font=font(16))
    d.text((496, 94), "Actual policy camera", fill=(244, 245, 248), font=font(16))
    d.rectangle((10, 580, 950, 619), fill=(25, 39, 53))
    action_label = "WATCHDOG HOLD (software expiry)" if action == "watchdog_hold" else action.upper()
    action_color = (255, 198, 107) if action == "watchdog_hold" else color
    d.text((20, 586), f"MOTOR: {action_label} ({motor_source}) | obs age: {age if age is not None else '--'}", fill=action_color, font=font(20))
    d.text((20, 632), f"Touch: {sensors['touch_normal_force_N']} N | contact: human {metrics['human_contact_N']:.1f} N, object {metrics['object_contact_N']:.1f} N", fill=(216, 224, 237), font=font(16))
    d.text((20, 662), f"Goal error: {metrics['goal_distance_m']*100:.1f} cm | disturbance: {event} | reaction: {last_reaction}", fill=(248, 200, 117), font=font(16))
    d.text((20, 692), f"Reflex bank v{bank_version} | brain: {brain_status or 'shared initial intention'} | camera + robot feedback", fill=(155, 169, 190), font=font(13))
    return np.asarray(canvas)


class WallVideo:
    """Sample the newest rendered panel at wall-clock 20 fps, independently of physics."""
    def __init__(self, path, initial):
        self.path, self.frame = path, initial
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.timestamps = []
        self.error = None
        self.thread = threading.Thread(target=self._run, name="wall-video", daemon=True)
        self.thread.start()

    def update(self, frame):
        with self.lock:
            self.frame = frame

    def _run(self):
        start = time.monotonic()
        try:
            with imageio.get_writer(str(self.path), fps=FPS, quality=7, macro_block_size=16) as writer:
                tick = 0
                while not self.stop.is_set():
                    with self.lock:
                        frame = self.frame
                    writer.append_data(frame)
                    self.timestamps.append(time.monotonic()-start)
                    tick += 1
                    self.stop.wait(max(0, start+tick/FPS-time.monotonic()))
        except Exception as exc:
            self.error = repr(exc)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.error:
            raise RuntimeError(self.error or "Video writer failed to finish")


def reactions(scenario, applied, horizon=None):
    output = []
    for event in scenario.events():
        if horizon is not None and event["onset_sim_s"] >= horizon:
            continue
        candidates = [r for r in applied if r["accepted"] and r.get("motor_changed") and r.get("action") in event["expected"]
                      and r["observed_sim_s"] >= event["onset_sim_s"]
                      and r["applied_sim_s"] < event["end_sim_s"]]
        first = candidates[0] if candidates else None
        output.append({**event, "first_model_reaction_sim_s": first["applied_sim_s"] if first else None,
                       "response_from_onset_sim_s": first["applied_sim_s"]-event["onset_sim_s"] if first else None,
                       "observation_to_motor_wall_s": first["observation_age_s"] if first else None,
                       "first_action": first["action"] if first else None,
                       "first_source": first.get("source", "cerebellum") if first else None,
                       "received_to_motor_wall_s": first.get("received_to_motor_s") if first else None})
    return output


def episode(args, scenario, mode, output, live=None):
    output.mkdir(parents=True)
    world = World(args.asset_root, scenario, args.image_size)
    rng = np.random.default_rng(scenario.seed + 901)
    (output/"scene.xml").write_text(world.xml)
    (output/"scenario.json").write_text(json.dumps(scenario.serializable(), indent=2))
    sensors = world.sensor_observation(rng)
    bank = ReflexBank(output/"reflex_bank.jsonl")
    (output/"initial_bank.json").write_text(json.dumps(bank.snapshot(), indent=2))
    memory = ReflexMemory(output/"memory.jsonl")
    camera = RenderWorker(world.model, args.image_size, scenario.seed+301)
    def capture(sequence, sim_s, wall_s, sensors, action):
        return RenderState(sequence, sim_s, wall_s, world.data.qpos.copy(), world.data.qvel.copy(),
                           world.data.mocap_pos.copy(), world.data.mocap_quat.copy(), sensors, action,
                           world.plan_goal+world.offset)
    camera.submit(capture(0, 0, 0, sensors, "continue"))
    first = camera.get(wait=True)
    if first is None:
        camera.close()
        world.close()
        raise RuntimeError("Initial camera did not render")
    _, frame, view = first
    Image.fromarray(frame).save(output/"initial.png")
    metrics = world.evaluate()
    initial_panel = panel(frame, view, mode, 0, 0, "continue", None, sensors, metrics, "none", "--")
    recorder = WallVideo(output/"episode.mp4", initial_panel)
    worker = LatestWorker(HTTPReflex(args.api_timeout, output/"observations"), output/"api.jsonl") if mode in ("reflex", "omni", "vlm_omni") else None
    brain = LatestWorker(VisualBrain(getattr(args, "brain_timeout", 15), output/"brain_observations"), output/"brain.jsonl",
                         close_timeout=getattr(args, "brain_timeout", 15)+2,
                         minimum_interval=getattr(args, "brain_interval", 0)) if mode in ("vlm", "vlm_omni") else None
    mailbox = ActionMailbox(args.max_observation_age, args.command_ttl)
    brain_mailbox = ActionMailbox(getattr(args, "max_brain_age", 8), getattr(args, "brain_command_ttl", 10))
    applied, motor_applied, states, actions = [], [], [], Counter()
    peak_human, peak_object, goal_ever, tick, sequence = 0, 0, False, 0, 1
    impulse_human, impulse_object = 0.0, 0.0
    start = time.monotonic()
    next_render = 0
    last_reaction = "--"
    previous_action = "continue"
    previous_velocity = np.zeros(3)
    last_camera_sequence = -1
    brain_applications = []
    brain_min_sequence, brain_last_sequence = -1, -1
    brain_status = "thinking" if brain else "shared initial intention"
    macro_changes = 0
    try:
        with open(output/"states.jsonl", "w") as state_log, open(output/"applications.jsonl", "w") as app_log:
            while world.data.time-world.zero < args.seconds-1e-7 and not (live and live.stopping):
                now, sim_s = time.monotonic()-start, float(world.data.time-world.zero)
                if live:
                    if live.poll(world, bank):
                        mailbox.invalidate(sequence)
                        brain_min_sequence = sequence
                        brain_mailbox.invalidate(sequence)
                        memory.reset_epoch()
                    scenario = world.scenario
                brain_pending = []
                if brain:
                    for record in brain.drain():
                        age = now-record["observed_wall_s"]
                        accepted = bool("error" not in record and 0 <= age <= getattr(args, "max_brain_age", 8)
                                        and record["sequence"] >= brain_min_sequence and record["sequence"] > brain_last_sequence)
                        entry = {**record, "source": "brain", "accepted": accepted, "applied_sim_s": sim_s,
                                 "applied_wall_s": now, "observation_age_s": age,
                                 "action": record.get("plan", {}).get("action", "continue")}
                        if "received_wall_s" in record:
                            entry["received_to_motor_s"] = now-record["received_wall_s"]
                        if accepted:
                            brain_last_sequence = record["sequence"]
                            plan = record["plan"]
                            brain_status = plan["stage"]+f" @ {sim_s:.1f}s"
                            changed = False
                            if plan["target_visible"] and plan["stage"] == "reach_target":
                                entry["motor_plan"] = world.apply_visual_plan(plan["target_xy"])
                                changed = entry["motor_plan"]["changed"]
                                macro_changes += int(changed)
                            if not getattr(args, "freeze_bank", False) and record["bank_version_at_observation"] == bank.version:
                                changed |= bank.update(plan["reflex_updates"], "visual_brain", sim_s)
                            if changed:
                                mailbox.invalidate(sequence)
                            decision = {**record, "action": entry["action"],
                                        "speed_scale": record.get("speed_scale", 1-.45*plan.get("danger_score", 0))}
                            brain_mailbox.accept(decision, now)
                        brain_pending.append(entry)
                reflex_enabled = bool(worker and (live is None or live.enabled))
                pending = []
                if worker:
                    for record in worker.drain():
                        accepted = reflex_enabled and record.get("bank_version", bank.version) == bank.version and mailbox.accept(record, now)
                        entry = {**record, "source": "cerebellum", "accepted": accepted, "applied_wall_s": now,
                                 "applied_sim_s": sim_s, "observation_age_s": now-record["observed_wall_s"]}
                        if "received_wall_s" in record:
                            entry["received_to_motor_s"] = now-record["received_wall_s"]
                        pending.append(entry)
                motor_wall_s = time.monotonic()-start
                action, speed_scale, motor_source, selected_record = choose_motor(motor_wall_s, mailbox, brain_mailbox, reflex_enabled, bool(brain))
                motor = world.motor(action, speed_scale)
                for entry in pending+brain_pending:
                    entry["applied_wall_s"] = motor_wall_s
                    entry["observation_age_s"] = motor_wall_s-entry["observed_wall_s"]
                    if "received_wall_s" in entry:
                        entry["received_to_motor_s"] = motor_wall_s-entry["received_wall_s"]
                    entry["action_transition"] = action != previous_action
                    entry["motor_selected"] = bool(entry["accepted"] and entry["source"] == motor_source and selected_record
                                                   and selected_record["sequence"] == entry["sequence"])
                    entry["motor_changed"] = bool(entry["motor_selected"] and entry.get("action") == action
                                                  and entry["action_transition"]
                                                  and (np.linalg.norm(np.asarray(motor["applied_velocity"])-previous_velocity)>.005
                                                       or motor["plan_offset_changed"]))
                    entry["actual_motor_action"] = action
                    motor_applied.append(entry)
                    if entry["motor_selected"]:
                        memory.remember(entry, sim_s, world.position(), world.data.sensordata)
                        if action != "continue":
                            last_reaction = f"{motor_source}:{action} @ {sim_s:.2f}s"
                    with open(output/"motor_applications.jsonl", "a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry)+"\n")
                    if entry["source"] == "brain":
                        brain_applications.append(entry)
                        with open(output/"brain_applications.jsonl", "a", encoding="utf-8") as stream:
                            stream.write(json.dumps(entry)+"\n")
                    else:
                        applied.append(entry)
                        app_log.write(json.dumps(entry)+"\n")
                before_position = world.position()
                previous_action, previous_velocity = action, np.asarray(motor["applied_velocity"])
                world.step()
                memory.feedback(float(world.data.time-world.zero), world.position(), world.data.sensordata,
                                action=action, plan_version=world.plan_version)
                metrics = world.evaluate()
                peak_human = max(peak_human, metrics["human_contact_peak_N"])
                peak_object = max(peak_object, metrics["object_contact_peak_N"])
                impulse_human += metrics["human_contact_impulse_Ns"]
                impulse_object += metrics["object_contact_impulse_Ns"]
                goal_ever |= metrics["goal_reached"]
                actions[action] += 1
                state = {"wall_s": motor_wall_s, "measured_wall_s": time.monotonic()-start, "sim_s": float(world.data.time-world.zero), "action": action,
                         "reflex_enabled": reflex_enabled, "motor_source": motor_source,
                         "reflex_bank_version": bank.version,
                         "brain_status": brain_status,
                         "measured_eef_velocity_m_s": ((world.position()-before_position)/DT).tolist(),
                         **motor, **metrics}
                states.append(state)
                state_log.write(json.dumps(state)+"\n")
                if now >= next_render:
                    sensors = world.sensor_observation(rng)
                    camera.submit(capture(sequence, float(world.data.time-world.zero),
                                          time.monotonic()-start, sensors, action))
                    sequence += 1
                    next_render += 1/args.render_hz
                    if next_render < now-1/args.render_hz:
                        next_render = now+1/args.render_hz
                packet = camera.get()
                if packet and packet[0].sequence != last_camera_sequence:
                    captured, frame, view = packet
                    key = signature(frame, captured.sensors)
                    context = {"bank": bank.snapshot(), "memories": memory.retrieve(key), "freeze_bank": getattr(args, "freeze_bank", False)}
                    observation = Observation(captured.sequence, captured.sim_s, captured.wall_s,
                                              frame, captured.sensors, captured.motor_action, context, start)
                    if reflex_enabled and captured.sequence >= mailbox.min_sequence:
                        worker.submit(observation)
                    if brain and captured.sequence >= brain_min_sequence:
                        brain.submit(observation)
                    last_camera_sequence = captured.sequence
                    age = round(now-selected_record["observed_wall_s"], 3) if selected_record else None
                    active = [e["kind"] for e in scenario.events() if e["onset_sim_s"] <= sim_s < e["end_sim_s"]]
                    painted = panel(frame, view, mode if reflex_enabled or brain else "baseline", sim_s, now, action, age,
                                    captured.sensors, metrics, "+".join(active) or "none", last_reaction, now-captured.wall_s,
                                    brain_status, bank.version, motor_source)
                    recorder.update(painted)
                    if live:
                        live.update(painted, {"status": "running", "world_s": sim_s, "wall_s": now,
                                              "action": action, "model_enabled": reflex_enabled, "motor_source": motor_source,
                                              "measured_eef_speed_m_s": float(np.linalg.norm(state["measured_eef_velocity_m_s"])),
                                              "commanded_eef_speed_m_s": float(np.linalg.norm(motor["applied_velocity"])),
                                              "brain_action": brain_mailbox.record.get("action") if brain_mailbox.record else None,
                                              "observation_age_s": age, "camera_age_s": now-captured.wall_s,
                                              "group": mode, "brain_status": brain_status, "bank_version": bank.version,
                                              "bank": bank.snapshot(), "memory_entries": len(memory.entries),
                                              "reflex_candidate": mailbox.record.get("reflex_id") if mailbox.record else None,
                                              "reflex_gate": mailbox.record.get("model_intervention") if mailbox.record else None,
                                              "condition_probabilities": mailbox.record.get("condition_probabilities") if mailbox.record else None,
                                              "danger_score": mailbox.record.get("danger_score") if mailbox.record else None,
                                              **metrics})
                tick += 1
                if tick % 40 == 0:
                    print(f"{scenario.kind} seed={scenario.seed} {mode} sim={sim_s:.2f} wall={now:.2f} action={action}", flush=True)
                time.sleep(max(0, start+tick*DT-time.monotonic()))
    finally:
        elapsed = time.monotonic()-start
        recorder.close()
        camera.close()
        if worker:
            worker.close()
            for record in worker.drain():
                entry = {**record, "accepted": False, "discard_reason": "episode_ended",
                         "applied_wall_s": elapsed, "applied_sim_s": float(world.data.time-world.zero),
                         "observation_age_s": elapsed-record["observed_wall_s"]}
                applied.append(entry)
                with open(output/"applications.jsonl", "a") as stream:
                    stream.write(json.dumps(entry)+"\n")
        if brain:
            brain.close()
            for record in brain.drain():
                entry = {**record, "accepted": False, "discard_reason": "episode_ended",
                         "applied_wall_s": elapsed, "applied_sim_s": float(world.data.time-world.zero)}
                brain_applications.append(entry)
                with open(output/"brain_applications.jsonl", "a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry)+"\n")
        final_metrics = world.evaluate()
        Image.fromarray(panel(frame, view, mode, float(world.data.time-world.zero), elapsed, action, None,
                              sensors, final_metrics, "finished", last_reaction)).save(output/"final.png")
        world.close()
    prefix = states if scenario.kind == "quiet" else [s for s in states if s["sim_s"] < scenario.onset]
    result = {"mode": mode, "scenario": scenario.kind, "seed": scenario.seed,
              "initial_state_sha256": world.initial_sha, "scenario_parameters": scenario.serializable(),
              "wall_seconds": elapsed, "sim_seconds": states[-1]["sim_s"],
              "real_time_factor": states[-1]["sim_s"]/elapsed, "control_steps": tick,
              "peak_human_contact_N": peak_human, "peak_object_contact_N": peak_object,
              "human_contact_impulse_Ns": impulse_human, "object_contact_impulse_Ns": impulse_object,
              "goal_ever_reached": bool(goal_ever), "final_goal_reached": bool(final_metrics["goal_reached"]),
              "final_goal_distance_m": final_metrics["goal_distance_m"], "motor_action_ticks": dict(actions),
              "clean_prefix_model_intervention_ticks": sum(s["action"] not in ("continue", "watchdog_hold") for s in prefix),
              "api_results": len(applied), "accepted_results": sum(r["accepted"] for r in applied),
              "api_errors": sum("error" in r for r in applied),
              "latency_wall_s": [r["inference_s"] for r in applied],
              "reactions": reactions(scenario, motor_applied, states[-1]["sim_s"]),
              "motion_responses": motion_responses(scenario, motor_applied, states),
              "motor_source_ticks": dict(Counter(s["motor_source"] for s in states)), "video_frames": len(recorder.timestamps),
              "video_wall_sample_seconds": recorder.timestamps,
              "sensor_boundary": "policy: camera + physical touch + retrieved robot feedback; archive also records joint position/velocity/torque",
              "model": os.environ.get("OMNIJEV_MODEL", "omnijev-0.8b-kk") if worker else None,
              "nominal_motor_plan": "shared initial reach; camera-calibrated model image points may update intention",
              "brain_model": os.environ.get("GEMINI_MODEL", "gemini-3.8-flash") if brain else None,
              "brain_results": len(brain_applications), "brain_accepted": sum(r["accepted"] for r in brain_applications),
              "brain_errors": sum("error" in r for r in brain_applications), "macro_plan_changes": macro_changes,
              "reflex_bank_versions": bank.version, "memory_entries": len(memory.entries),
              "software_limits": {"max_observation_age_s": args.max_observation_age,
                                   "command_ttl_s": args.command_ttl, "joint_ranges": True,
                                   "max_brain_age_s": getattr(args, "max_brain_age", 8),
                                   "brain_command_ttl_s": getattr(args, "brain_command_ttl", 10)},
              "watchdog_ticks": actions["watchdog_hold"]}
    if live:
        result["interactive"] = True
        result["summary_scope"] = "Event summary covers the final scene epoch; all operator events and resets remain in interactive_events.jsonl"
        result["clean_prefix_model_intervention_ticks"] = None
        (output/"scenario.json").write_text(json.dumps(scenario.serializable(), indent=2))
    (output/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("scenario", "seed", "mode", "real_time_factor", "final_goal_reached", "peak_human_contact_N", "peak_object_contact_N", "accepted_results")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    bundled = Path(__file__).resolve().parents[1]/"resources"/"panda"
    parser.add_argument("--asset-root", type=Path, default=bundled if (bundled/"panda.xml").exists() else Path("/opt/omnijev-sim"))
    parser.add_argument("--scenarios", nargs="+", choices=KINDS, default=["hand", "slider", "target", "occlusion"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 73])
    parser.add_argument("--modes", nargs="+", choices=["baseline", "reflex", "omni", "vlm", "vlm_omni"], default=["vlm", "vlm_omni"])
    parser.add_argument("--seconds", type=float, default=16)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--render-hz", type=float, default=8)
    parser.add_argument("--api-timeout", type=float, default=2)
    parser.add_argument("--brain-timeout", type=float, default=15)
    parser.add_argument("--brain-interval", type=float, default=0)
    parser.add_argument("--max-brain-age", type=float, default=8)
    parser.add_argument("--brain-command-ttl", type=float, default=10)
    parser.add_argument("--freeze-bank", action="store_true")
    parser.add_argument("--max-observation-age", type=float, default=1.2)
    parser.add_argument("--command-ttl", type=float, default=1.4)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Use a fresh output directory")
    if args.seconds <= 0 or args.command_ttl < args.max_observation_age:
        raise SystemExit("Invalid duration or freshness limits")
    args.out.mkdir(parents=True)
    manifest = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
                "panda_xml_sha256": hashlib.sha256((args.asset_root/"panda.xml").read_bytes()).hexdigest(),
                "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.out/"manifest.json").write_text(json.dumps(manifest, indent=2))
    mesh_manifest = args.asset_root/"mesh_manifest.json"
    if mesh_manifest.exists():
        (args.out/"mesh_manifest.json").write_bytes(mesh_manifest.read_bytes())
    summaries = []
    for seed in args.seeds:
        for kind in args.scenarios:
            scenario = Scenario.generate(kind, seed)
            for mode in args.modes:
                result = episode(args, scenario, mode, args.out/f"{kind}_seed{seed}_{mode}")
                summaries.append(result)
                (args.out/"summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
