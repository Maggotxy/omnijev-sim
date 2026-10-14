"""Render copied states in a dedicated GL thread; never stall physical time."""
from dataclasses import dataclass
import threading

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .world import release_renderer


@dataclass(frozen=True)
class RenderState:
    sequence: int
    sim_s: float
    wall_s: float
    qpos: object
    qvel: object
    mocap_pos: object
    mocap_quat: object
    sensors: dict
    motor_action: str
    planned_aim: object


class RenderWorker:
    def __init__(self, model, size, seed):
        self.model, self.size, self.seed = model, size, seed
        self.condition = threading.Condition()
        self.latest, self.output = None, None
        self.stop = False
        self.error = None
        self.thread = threading.Thread(target=self._run, name="camera-render", daemon=True)
        self.thread.start()

    def submit(self, state):
        with self.condition:
            self.latest = state
            self.condition.notify_all()

    def get(self, wait=False):
        with self.condition:
            if wait:
                self.condition.wait_for(lambda: self.output is not None or self.error is not None, timeout=15)
            if self.error:
                raise RuntimeError(self.error)
            return self.output

    def _run(self):
        renderer = None
        try:
            renderer = mujoco.Renderer(self.model, height=self.size, width=self.size)
            data = mujoco.MjData(self.model)
            option = mujoco.MjvOption()
            option.geomgroup[2] = 0  # Use low-polygon physical shells for display.
            option.geomgroup[3] = 0
            option.geomgroup[4] = 1  # Primitive display shells; original physical contacts remain unchanged.
            last_sequence = -1
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stop or
                                            (self.latest is not None and self.latest.sequence > last_sequence))
                    if self.stop:
                        return
                    state = self.latest
                data.qpos[:] = state.qpos
                data.qvel[:] = state.qvel
                data.mocap_pos[:] = state.mocap_pos
                data.mocap_quat[:] = state.mocap_quat
                mujoco.mj_forward(self.model, data)
                renderer.update_scene(data, camera="overhead", scene_option=option)
                frame = renderer.render().copy()
                rng = np.random.default_rng(self.seed+state.sequence)
                frame = np.clip(frame.astype(float)+rng.normal(0, 1, frame.shape), 0, 255).astype(np.uint8)
                # Robot's own planned landing point, projected with camera calibration.
                # This overlay never uses the actual target/obstacle position.
                camera = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "overhead")
                point = np.array([*state.planned_aim[:2], .418])
                pc = data.cam_xmat[camera].reshape(3, 3).T @ (point-data.cam_xpos[camera])
                focal = self.size/2/np.tan(np.radians(self.model.cam_fovy[camera])/2)
                u, v = self.size/2+focal*pc[0]/(-pc[2]), self.size/2-focal*pc[1]/(-pc[2])
                annotated = Image.fromarray(frame)
                draw = ImageDraw.Draw(annotated)
                draw.ellipse((u-13, v-13, u+13, v+13), outline=(20, 95, 255), width=3)
                tool = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tool")
                robot_pc = data.cam_xmat[camera].reshape(3, 3).T @ (data.site_xpos[tool]-data.cam_xpos[camera])
                ru = self.size/2+focal*robot_pc[0]/(-robot_pc[2])
                rv = self.size/2-focal*robot_pc[1]/(-robot_pc[2])
                draw.ellipse((ru-15, rv-15, ru+15, rv+15), outline=(255, 130, 20), width=3)
                frame = np.asarray(annotated).copy()
                renderer.update_scene(data, camera="beauty", scene_option=option)
                view = renderer.render().copy()
                with self.condition:
                    self.output = (state, frame, view)
                    self.condition.notify_all()
                last_sequence = state.sequence
        except Exception as exc:
            with self.condition:
                self.error = repr(exc)
                self.condition.notify_all()
        finally:
            if renderer:
                release_renderer(renderer)

    def close(self):
        with self.condition:
            self.stop = True
            self.condition.notify_all()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Camera render failed to stop")
