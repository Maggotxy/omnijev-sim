# Panda asset provenance

Robot description and meshes originate from [MuJoCo Menagerie / Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda), derived from Franka's robot description. The upstream README and Apache-2.0 LICENSE are included.

This package copies the existing JEV runtime's `/opt/omnijev-sim/panda.xml` and the 67 mesh files it references. The runtime XML includes added `wrist` and `wrist_alt` cameras. An XML provenance comment was added for this distribution. The robot geometry, inertial parameters and actuators are otherwise preserved. `mesh_manifest.json` records the exact distributed mesh SHA256 values.

The continuous-reflex scene builder composes its world in memory, removes the Panda-only keyframe, adds physical bodies/touch sensors, and adds primitive display shells for CPU rendering. Original robot collision meshes remain active. The upstream README also describes files from the full Menagerie package which are not needed by this simulation.
