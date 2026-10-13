"""Physical world, robot sensors, and bounded motor primitives.

Robot feedback uses its own joints/site pose. Dynamic object state is read only
by scenario generation/evaluation, never by motor() or the policy observation.
"""
import hashlib
from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DT = 0.05


def node(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k: str(v) for k, v in attrs.items()})


def scene_xml(asset_root, scenario):
    root = ET.parse(Path(asset_root) / "panda.xml").getroot()
    for key in list(root.findall("keyframe")):
        root.remove(key)  # Panda-only keyframe dimension is invalid after free bodies are added.
    root.find("compiler").set("meshdir", str(Path(asset_root).resolve() / "assets"))
    root.find("option").set("timestep", "0.002")
    visual = node(root, "visual")
    node(visual, "global", offwidth=1024, offheight=1024)
    node(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.7 0.7 0.7")
    asset = root.find("asset")
    node(asset, "texture", name="woodgrain", type="2d", builtin="checker", width=256, height=256,
         rgb1="0.48 0.34 0.22", rgb2="0.45 0.31 0.20")
    node(asset, "material", name="tablewood", texture="woodgrain", texrepeat="12 12", specular="0.2")
    world = root.find("worldbody")
    node(world, "geom", name="floor", type="plane", size="3 3 .1", rgba=".15 .18 .22 1")
    node(world, "geom", name="table", type="box", pos=".57 0 .39", size=".38 .47 .025",
         material="tablewood", friction=".6 .01 .001")
    node(world, "light", name="worklight", pos=".5 -.2 2", diffuse=f"{scenario.light} {scenario.light} {scenario.light}")
    node(world, "camera", name="overhead", pos=".49 0 1.5", xyaxes="0 1 0 -1 0 0", fovy=50)
    node(world, "camera", name="beauty", pos="1.45 -.8 1.15", xyaxes=".64 .77 0 -.38 .31 .87", fovy=48)
    goal = node(world, "body", name="goal", mocap="true", pos=".61 .1 .418")
    node(goal, "geom", name="goal_pad", type="cylinder", size=".055 .002", rgba=".12 .85 .3 1",
         contype=0, conaffinity=0)
    node(world, "body", name="human_driver", mocap="true", pos=".55 -.8 .63")
    human = node(world, "body", name="human", pos=".55 -.8 .63")
    node(human, "freejoint", name="human_joint")
    node(human, "geom", name="forearm", type="capsule", fromto=".15 0 0 .01 0 0",
         size=".025", rgba=".72 .46 .3 1", friction=".5 .01 .001")
    node(human, "geom", name="palm", type="ellipsoid", pos="-.025 0 0", size=".045 .04 .022",
         rgba=".8 .54 .37 1")
    for i in range(4):
        node(human, "geom", name=f"human_finger{i}", type="capsule",
             fromto=f"-.05 {-.027+i*.018} 0 -.10 {-.027+i*.018} 0", size=".009", rgba=".82 .57 .39 1")
    node(root.find("equality"), "weld", body1="human_driver", body2="human", solref=".1 1", solimp=".8 .95 .01")
    slider = node(world, "body", name="slider", pos=f".56 {scenario.side*.30} .505")
    node(slider, "freejoint", name="slider_joint")
    node(slider, "geom", name="slider_geom", type="box", size=".032 .04 .085", mass=scenario.mass,
         friction=f"{scenario.friction} .005 .001", rgba=".95 .65 .1 1")
    occluder = node(world, "body", name="occluder", mocap="true", pos="2 2 2")
    node(occluder, "geom", type="box", size=".4 .45 .01", rgba=".12 .13 .15 1", contype=0, conaffinity=0)
    hand = root.find(".//body[@name='hand']")
    node(hand, "site", name="tool", pos="0 0 .105", size=".009", rgba=".1 .8 1 1")
    node(hand, "site", name="hand_skin", type="box", pos="0 0 .06", size=".075 .07 .08", rgba="0 0 0 0")
    sensors = node(root, "sensor")
    node(sensors, "touch", name="hand_touch", site="hand_skin")
    for finger in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{finger}']")
        node(body, "site", name=finger+"_skin", type="box", pos="0 0 .02", size=".02 .02 .05", rgba="0 0 0 0")
        node(sensors, "touch", name=finger+"_touch", site=finger+"_skin")
    # Lightweight display geometry; the original collision meshes/inertias stay intact.
    for body in root.findall(".//body"):
        name = body.get("name", "")
        if name.startswith("link"):
            child = body.find("body")
            offset = np.fromstring(child.get("pos", "0 0 0"), sep=" ") if child is not None else np.zeros(3)
            if np.linalg.norm(offset) > .06:
                node(body, "geom", name=name+"_display", type="capsule",
                     fromto="0 0 0 "+" ".join(map(str, offset)), size=".045", group=4,
                     rgba=".9 .92 .94 1", contype=0, conaffinity=0)
            else:
                node(body, "geom", name=name+"_display", type="sphere", size=".058", group=4,
                     rgba=".7 .73 .77 1", contype=0, conaffinity=0)
        elif name == "hand":
            node(body, "geom", type="box", size=".045 .04 .035", pos="0 0 .025", group=4,
                 rgba=".2 .23 .26 1", contype=0, conaffinity=0)
        elif name in ("left_finger", "right_finger"):
            node(body, "geom", type="box", size=".012 .009 .025", pos="0 0 .022", group=4,
                 rgba=".13 .16 .19 1", contype=0, conaffinity=0)
    for light in root.findall(".//light"):
        light.set("castshadow", "false")
    return ET.tostring(root, encoding="unicode")


class World:
    def __init__(self, asset_root, scenario, size=384):
        self.scenario = scenario
        self.xml = scene_xml(asset_root, scenario)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.size = size
        self.tool = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool")
        self.hand = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.robot_bodies = set(range(1, self.hand + 3))
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] in self.robot_bodies and self.model.geom_group[g] == 3:
                self.model.geom_rgba[g] = [.88, .9, .92, 1] if self.model.geom_bodyid[g] < self.hand else [.16, .18, .20, 1]
        self.human = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human")
        self.human_driver = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "human_driver")
        self.slider = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "slider")
        self.goal = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.occluder = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "occluder")
        self.qref = np.array([0, -.6, 0, -2.2, 0, 1.8, .8])
        self.data.qpos[:7] = self.qref
        self.data.qpos[7:9] = .035
        self.data.ctrl[:7] = self.qref
        self.data.ctrl[7] = 220
        self.data.mocap_pos[self.model.body_mocapid[self.goal]] = [*scenario.nominal_goal[:2], .418]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)
        self.zero = self.data.time
        self.rotation = self.data.site_xmat[self.tool].reshape(3, 3).copy()
        self.origin = self.position()
        self.offset = np.zeros(3)
        self.plan_goal = np.asarray(scenario.nominal_goal).copy()
        self.plan_version = 0
        self.last_action = None
        self.reflex_anchor = None
        self.slider_kicked_at = None
        self.tick_peak_human = self.tick_peak_object = 0.0
        self.tick_impulse_human = self.tick_impulse_object = 0.0
        self.initial_sha = hashlib.sha256(self.data.qpos.tobytes() + self.data.qvel.tobytes()).hexdigest()
        self.initial_arrays = {name: getattr(self.data, name).copy() for name in ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")}

    def reset_live(self):
        # Explicit operator reset; preserve the continuous logging clock.
        for name, value in self.initial_arrays.items():
            getattr(self.data, name)[:] = value
        self.qref = self.initial_arrays["ctrl"][:7].copy()
        self.offset[:] = 0
        self.plan_goal = np.asarray(self.scenario.nominal_goal).copy()
        self.plan_version += 1
        self.last_action = self.reflex_anchor = self.slider_kicked_at = None
        self.scenario = replace(self.scenario, kind="combined", timings=())
        mujoco.mj_forward(self.model, self.data)

    def inject(self, kind, rng):
        """Operator/environment action, never a policy input or a motor decision."""
        t = float(self.data.time-self.zero)
        if self.scenario.active(t, kind):
            return {"kind": kind, "sim_s": t, "status": "already_active"}
        params = {}
        if kind == "target":
            y = float(self.data.mocap_pos[self.model.body_mocapid[self.goal], 1])
            delta = float(rng.choice([-1, 1]))*float(rng.uniform(.10, .16))
            params = {"start_y": y, "end_y": float(np.clip(y+delta, -.22, .25))}
        duration = float(rng.uniform(2.5, 4.8))
        event = (kind, t, duration, params)
        self.scenario = replace(self.scenario, kind="combined", timings=self.scenario.timings+(event,),
                                onset=self.scenario.timings[0][1] if self.scenario.timings else t)
        return {"kind": kind, "sim_s": t, "duration_s": duration, "status": "injected", **params}

    def position(self):
        return self.data.site_xpos[self.tool].copy()

    def apply_visual_plan(self, xy):
        """Camera-ray calibration of a MODEL-selected image point; no target truth."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
        focal = .5/np.tan(np.radians(self.model.cam_fovy[cid])/2)
        direction = self.data.cam_xmat[cid].reshape(3, 3) @ np.array([(xy[0]-.5)/focal, - (xy[1]-.5)/focal, -1])
        origin = self.data.cam_xpos[cid]
        if abs(direction[2]) < 1e-6:
            raise ValueError("Camera ray parallel to table")
        scale = (.418-origin[2])/direction[2]
        if scale <= 0:
            raise ValueError("Selected point is behind camera")
        point = origin+scale*direction
        goal = np.array([np.clip(point[0], .38, .74), np.clip(point[1], -.30, .30), .59])
        changed = bool(np.linalg.norm(goal-self.plan_goal) > .025)
        if changed:
            self.plan_goal = goal
            self.offset[:] = 0
            self.plan_version += 1
        return {"changed": changed, "selected_pixel_xy": xy, "calibrated_goal_m": goal.tolist(),
                "workspace_clipped": bool(not np.allclose(point[:2], goal[:2])), "plan_version": self.plan_version}

    def sensor_observation(self, rng):
        return {"joint_position_rad": (self.data.qpos[:7] + rng.normal(0, .0001, 7)).round(5).tolist(),
                "joint_velocity_rad_s": self.data.qvel[:7].round(4).tolist(),
                "joint_motor_torque_Nm": self.data.actuator_force[:7].round(3).tolist(),
                "touch_normal_force_N": np.maximum(0, self.data.sensordata + rng.normal(0, .02, 3)).round(3).tolist()}

    def disturbances(self):
        t, s = self.data.time - self.zero, self.scenario
        human_pos = [.56, s.side * .75, .64]
        if s.active(t, "hand"):
            event = s.event("hand", t)
            u = (t-event["onset_sim_s"]) / (event["end_sim_s"]-event["onset_sim_s"])
            # Smooth approach, dwell, retreat. No robot-state targeting.
            ramp = min(1, u/.3, (1-u)/.2)
            human_pos[1] = s.side * (.75 - .72*max(0, ramp))
        self.data.mocap_pos[self.model.body_mocapid[self.human_driver]] = human_pos
        slider_event = s.event("slider", t)
        if s.active(t, "slider") and self.slider_kicked_at != slider_event["onset_sim_s"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider_joint")
            self.data.qvel[self.model.jnt_dofadr[jid]+1] = -s.side*.6
            self.slider_kicked_at = slider_event["onset_sim_s"]  # Physical impulse, then free-body dynamics.
        goal = np.array([*s.nominal_goal[:2], .418])
        target_event = s.event("target", t)
        if target_event and t >= target_event["onset_sim_s"]:
            elapsed = t-target_event["onset_sim_s"]
            start_y = target_event.get("start_y", s.nominal_goal[1])
            end_y = target_event.get("end_y", s.nominal_goal[1]+s.side*s.shift)
            goal[1] = start_y+(end_y-start_y)*min(1, elapsed/.7)
        self.data.mocap_pos[self.model.body_mocapid[self.goal]] = goal
        occlusion = [.49, 0, 1.02] if s.active(t, "occlusion") else [2, 2, 2]
        self.data.mocap_pos[self.model.body_mocapid[self.occluder]] = occlusion

    def motor(self, action, speed_scale=1):
        # A shared initial reach is the "brain" command. It never follows object truth.
        prior_offset = self.offset.copy()
        if action == "left":
            self.offset[1] -= .055*DT
        elif action == "right":
            self.offset[1] += .055*DT
        self.offset = np.clip(self.offset, [-.1, -.22, 0], [.1, .22, .16])
        goal = self.plan_goal + self.offset
        nominal = (goal-self.position()) * 1.4
        nominal *= min(1, .065/max(np.linalg.norm(nominal), 1e-9))
        speed_scale = float(np.clip(speed_scale, .25, 1))
        velocity = nominal.copy()*speed_scale
        if action != self.last_action and action in ("retreat", "lift"):
            self.reflex_anchor = self.position() + (np.array([-.06, 0, .04]) if action == "retreat" else np.array([0, 0, .08]))
        if action in ("retreat", "lift"):
            velocity = (self.reflex_anchor-self.position())*2
            velocity *= min(1, .065*speed_scale/max(np.linalg.norm(velocity), 1e-9))
            if np.linalg.norm(self.reflex_anchor-self.position()) < .009:
                velocity[:] = 0
                self.qref = self.data.qpos[:7].copy()
        if action in ("brake", "watchdog_hold"):
            velocity[:] = 0
            self.qref = self.data.qpos[:7].copy()
        self.last_action = action
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.tool)
        rotation = self.data.site_xmat[self.tool].reshape(3, 3)
        orientation = .5*sum((np.cross(rotation[:, i], self.rotation[:, i]) for i in range(3)))
        J = np.vstack((jp[:, :7], jr[:, :7]))
        desired = np.r_[velocity*DT, orientation*DT*2]
        if np.linalg.norm(velocity) > 0:
            self.qref += J.T @ np.linalg.solve(J @ J.T + .005*np.eye(6), desired)
        before = self.qref.copy()
        self.qref = np.clip(self.qref, self.model.jnt_range[:7, 0]+.01, self.model.jnt_range[:7, 1]-.01)
        self.data.ctrl[:7] = self.qref
        return {"nominal_velocity": nominal.tolist(), "applied_velocity": velocity.tolist(),
                "model_speed_scale": speed_scale, "plan_version": self.plan_version,
                "joint_limit_clipped": bool(np.any(before != self.qref)),
                "plan_offset_changed": bool(np.linalg.norm(self.offset-prior_offset)>.0001),
                "command_changes_motion": bool(np.linalg.norm(velocity-nominal)>.005 or np.linalg.norm(self.offset-prior_offset)>.0001)}

    def step(self):
        self.tick_peak_human = self.tick_peak_object = 0.0
        self.tick_impulse_human = self.tick_impulse_object = 0.0
        for _ in range(round(DT/self.model.opt.timestep)):
            self.disturbances()
            mujoco.mj_step(self.model, self.data)
            human, obj = self._contacts()
            self.tick_peak_human = max(self.tick_peak_human, human)
            self.tick_peak_object = max(self.tick_peak_object, obj)
            self.tick_impulse_human += human*self.model.opt.timestep
            self.tick_impulse_object += obj*self.model.opt.timestep

    def render(self, camera="overhead"):
        if not hasattr(self, "renderer"):
            self.renderer = mujoco.Renderer(self.model, height=self.size, width=self.size)
        option = mujoco.MjvOption()
        option.geomgroup[2] = 0
        option.geomgroup[3] = 0
        option.geomgroup[4] = 1
        self.renderer.update_scene(self.data, camera=camera, scene_option=option)
        return self.renderer.render().copy()

    def _contacts(self):
        human_force, object_force = 0.0, 0.0
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            bodies = {int(self.model.geom_bodyid[con.geom1]), int(self.model.geom_bodyid[con.geom2])}
            if not bodies & self.robot_bodies:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, force)
            if self.human in bodies:
                human_force += max(0, force[0])
            if self.slider in bodies:
                object_force += max(0, force[0])
        return float(human_force), float(object_force)

    def evaluate(self):
        human_force, object_force = self._contacts()
        goal = self.data.xpos[self.goal]
        distance = float(np.linalg.norm(self.position()[:2]-goal[:2]))
        return {"human_contact_N": human_force, "object_contact_N": object_force,
                "human_contact_peak_N": self.tick_peak_human, "object_contact_peak_N": self.tick_peak_object,
                "human_contact_impulse_Ns": self.tick_impulse_human, "object_contact_impulse_Ns": self.tick_impulse_object,
                "goal_distance_m": distance, "goal_reached": bool(distance < .04 and abs(self.position()[2]-.59)<.05),
                "eef_position_m": self.position().tolist()}

    def close(self):
        if hasattr(self, "renderer"):
            release_renderer(self.renderer)


def release_renderer(renderer):
    if hasattr(renderer, "close"):
        renderer.close()
    else:
        renderer._mjr_context.free()
        renderer._gl_context.free()
