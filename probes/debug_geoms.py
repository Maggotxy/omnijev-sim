import mujoco, os
ROOT = os.path.dirname(os.path.abspath(__file__))
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, "sim_scene.xml"))
for i in range(m.ngeom):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
    bid = m.geom_bodyid[i]
    bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
    if bname in ("left_finger", "right_finger", "hand", "cube_red"):
        print("geom", i, repr(name), "body", bname)
