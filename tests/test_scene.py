import mujoco
import numpy as np

from rl_insert.scene import SceneParams, build_scene, place_peg_in_hand


def _tip_in_tcp(m, d):
    tcp, tip = m.site("tcp").id, m.site("peg_tip").id
    return d.site_xmat[tcp].reshape(3, 3).T @ (d.site_xpos[tip] - d.site_xpos[tcp])


def test_closed_gripper_holds_peg_under_gravity():
    p = SceneParams()
    m = build_scene(p).compile()
    d = mujoco.MjData(m)
    place_peg_in_hand(m, d, p.peg_hole.peg_radius, p.protrusion)
    np.testing.assert_allclose(_tip_in_tcp(m, d), [0, 0, p.protrusion], atol=1e-6)
    d.ctrl[:] = 0  # arm holds zero, gripper commanded closed
    for _ in range(2000):
        mujoco.mj_step(m, d)
    np.testing.assert_allclose(_tip_in_tcp(m, d), [0, 0, p.protrusion], atol=5e-4)


def test_arm_holds_pose_under_gravity():
    q = np.array([0, 0.3, -0.8, 0, 0.5, 0.0])
    p = SceneParams()
    m = build_scene(p).compile()
    d = mujoco.MjData(m)
    assert [m.actuator(i).name for i in range(6)] == [f"joint{i}" for i in range(1, 7)]
    d.qpos[:6] = q  # arm joints are the first 6 dofs
    d.ctrl[:6] = q
    place_peg_in_hand(m, d, p.peg_hole.peg_radius, p.protrusion)
    tcp = m.site("tcp").id
    ref = d.site_xpos[tcp].copy()
    for _ in range(int(2 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    assert np.linalg.norm(d.site_xpos[tcp] - ref) < 5e-4
    np.testing.assert_allclose(d.qpos[:6], q, atol=1e-3)
