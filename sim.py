"""MuJoCo sim wrapper: Panda + table scene, camera rendering, Cartesian EE control."""
import os
import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.abspath(__file__))


class Sim:
    def __init__(self, scene_path=None, img_w=448, img_h=448):
        scene_path = scene_path or os.path.join(ROOT, "sim_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(scene_path)
        self.data = mujoco.MjData(self.model)
        self.img_w, self.img_h = img_w, img_h
        self.renderer = mujoco.Renderer(self.model, height=img_h, width=img_w)

        self.nq_arm = 7
        # joint position actuators 0..6 are the arm; find gripper actuator
        self.grip_act = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
        self.ee_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        # the menagerie hand collision shell wraps down to fingertip level and
        # shoves small objects before the pads can wrap them; make it non-colliding
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] == self.ee_body:
                self.model.geom_contype[g] = 0
                self.model.geom_conaffinity[g] = 0
        self.cube_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
        self.zone_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        # home configuration (a comfortable ready pose)
        self.q_home = np.array([0.0, -0.6, 0.0, -2.2, 0.0, 1.8, 0.8])
        self.q_target = self.q_home.copy()
        self.gripper_open = True
        self.reset()

    def reset(self, randomize=True, rng=None):
        mujoco.mj_resetData(self.model, self.data)
        rng = rng or np.random.default_rng()
        self.data.qpos[:7] = self.q_home
        self.q_target = self.q_home.copy()
        # randomize cube / zone positions on the table top (x in [0.42,0.68], y in [-0.2,0.3])
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint")
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr:adr+2] = [rng.uniform(0.42, 0.62), rng.uniform(-0.20, 0.05)]
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_blue_joint")
        adr = self.model.jnt_qposadr[jid]
        self.data.qpos[adr:adr+2] = [rng.uniform(0.50, 0.68), rng.uniform(0.0, 0.18)]
        zx = [rng.uniform(0.40, 0.55), rng.uniform(0.15, 0.30)]
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_zone")
        self.model.body_pos[bid][:2] = zx
        self.gripper_open = True
        self._apply_gripper()
        self.data.ctrl[:7] = self.q_home  # hold the arm pose while settling
        self.settle(1.2)  # let the gripper open fully (80mm takes ~1s)
        mujoco.mj_forward(self.model, self.data)

    # ---------- control ----------
    def _apply_gripper(self):
        # menagerie panda: actuator8 ctrl 0=closed, 255=open (tendon-driven fingers)
        self.data.ctrl[self.grip_act] = 255.0 if self.gripper_open else 0.0

    def set_gripper(self, open_: bool):
        self.gripper_open = open_
        self._apply_gripper()

    def ee_pos(self):
        return self.data.xpos[self.ee_body].copy()

    def tip_pos(self):
        """Midpoint between the two fingertips — the point that grasps."""
        lf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        rf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        return (self.data.xpos[lf] + self.data.xpos[rf]) / 2

    def cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def zone_pos(self):
        return self.data.xpos[self.zone_body].copy()

    def move_ee(self, v_world, speed=0.15):
        """v_world: unit-ish 3D velocity direction; integrates one decision tick."""
        if np.linalg.norm(v_world) > 1e-6:
            v = np.asarray(v_world, dtype=float)
            v = v / max(np.linalg.norm(v), 1e-9) * speed
            ee = self.ee_pos()
            jacp = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, None, self.ee_body)
            J = jacp[:, :7]
            lam = 1e-2
            dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), v * 0.35)
            new_q = self.q_target + dq
            # clamp to joint ranges
            for i in range(7):
                lo, hi = self.model.jnt_range[i]
                new_q[i] = np.clip(new_q[i], lo + 1e-3, hi - 1e-3)
            self.q_target = new_q
        self.data.ctrl[:7] = self.q_target

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def settle(self, seconds=0.5):
        self.step(int(seconds / self.model.opt.timestep))

    # ---------- rendering ----------
    def render(self, camera="agentview"):
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def render_wrist(self):
        """Eye-in-hand style view computed from the hand pose."""
        hand = self.data.xpos[self.ee_body]
        # camera slightly above/behind the gripper looking at a point below it
        cam_pos = hand + np.array([-0.10, 0.0, 0.22])
        look = hand + np.array([0.06, 0.0, -0.10])
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        fwd = look - cam_pos
        fwd /= np.linalg.norm(fwd)
        # mujoco camera: z = -forward, x = right
        up = np.array([0.0, 0.0, 1.0])
        z = -fwd
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        cam.lookat[:] = look
        cam.distance = np.linalg.norm(look - cam_pos)
        # derive azimuth/elevation from forward vector
        az = np.degrees(np.arctan2(-fwd[1], -fwd[0]))
        el = np.degrees(np.arcsin(np.clip(-fwd[2], -1, 1)))
        cam.azimuth = az
        cam.elevation = el
        self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()

    def success(self):
        c, z = self.cube_pos(), self.zone_pos()
        return np.linalg.norm(c[:2] - z[:2]) < 0.06 and c[2] < 0.48

    def holding(self):
        """Ground-truth holding check (logging/recovery): contact-based, with a
        fingertip-distance fallback. Finger geoms are unnamed in menagerie, so
        match by body id."""
        lf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        rf = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        cg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_red_geom")
        finger_geoms = {g for g in range(self.model.ngeom)
                        if self.model.geom_bodyid[g] in (lf, rf)}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            pair = {con.geom1, con.geom2}
            if cg in pair and pair & finger_geoms:
                return True
        # fallback: both fingertips near the cube while closed
        if not self.gripper_open:
            fl = self.data.xpos[lf]
            fr = self.data.xpos[rf]
            c = self.cube_pos()
            mid = (fl + fr) / 2
            if np.linalg.norm(mid - c) < 0.05:
                return True
        return False
