import mujoco, os
ROOT = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "scene.xml"))
print("njnt", m.njnt, "nq", m.nq, "nbody", m.nbody)
for i in range(m.njnt):
    print("joint", i, repr(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)), "qposadr", m.jnt_qposadr[i])
for i in range(m.nbody):
    print("body", i, repr(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)))
for i in range(m.nu):
    print("actuator", i, repr(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)))
print("cube_red_joint id:", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_red_joint"))
print("actuator8 id:", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8"))
print("hand id:", mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand"))
