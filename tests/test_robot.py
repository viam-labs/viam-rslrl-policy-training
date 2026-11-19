import mujoco
import numpy as np
import pytest

from rl_insert.robot import ARM_JOINTS, FLANGE, add_gripper, load_xarm6


def test_arm_joints_and_actuators_match_viam_names():
    m = load_xarm6().compile()
    assert tuple(m.joint(i).name for i in range(m.njnt)) == ARM_JOINTS
    assert tuple(m.actuator(i).name for i in range(m.nu)) == ARM_JOINTS


def test_flange_at_zero_matches_viam_home():
    """viam-ufactory-xarm reports the xArm6 flange at (207, 0, 112) mm, pointing down, at zero."""
    m = load_xarm6().compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    flange = m.body(FLANGE).id
    np.testing.assert_allclose(d.xpos[flange], [0.207, 0.0, 0.112], atol=1e-4)
    assert d.xmat[flange].reshape(3, 3)[:, 2] == pytest.approx([0, 0, -1], abs=1e-6)


def test_gripper_adds_one_actuator_and_a_tcp():
    spec = load_xarm6()
    add_gripper(spec)
    m = spec.compile()
    assert m.actuator(m.nu - 1).name == "gripper"
    assert m.site("tcp").id >= 0


def test_kinematics_spec_matches_training_fk(tmp_path):
    from rl_insert.robot import kinematics_spec

    full = load_xarm6()
    add_gripper(full)
    full = full.compile()
    path = tmp_path / "kinematics.xml"
    path.write_text(kinematics_spec().to_xml())
    kin = mujoco.MjModel.from_xml_path(str(path))  # loads on its own: no mesh files
    assert kin.nmesh == 0
    q = np.array([0.1, -0.3, -0.6, 0.2, 0.9, -0.4])
    out = []
    for m in (full, kin):
        d = mujoco.MjData(m)
        for name, v in zip(ARM_JOINTS, q):
            d.qpos[m.joint(name).qposadr[0]] = v
        mujoco.mj_kinematics(m, d)
        out.append((d.site("tcp").xpos.copy(), d.site("tcp").xmat.copy()))
    np.testing.assert_allclose(out[0][0], out[1][0], atol=1e-5)  # to_xml prints ~7 significant digits (rot error to 1.75e-6 over random poses)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-5)  # to_xml prints ~7 significant digits (rot error to 1.75e-6 over random poses)


def test_closed_g2_presses_each_pad_with_grip_force():
    from rl_insert.peg_hole import PegHoleParams, add_peg, configure_options
    from rl_insert.robot import GripperParams
    from scripts.export_sim_scene import gripper_spec

    spec = gripper_spec()
    configure_options(spec)
    spec.option.gravity = [0, 0, 0]
    add_peg(spec, spec.worldbody, PegHoleParams(clearance=0.00035), pos=(0, 0, 0.1966), quat=(0, 1, 0, 0))
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(500):  # ctrl 0: closed
        mujoco.mj_step(m, d)
    force = {m.geom("pad_left").id: 0.0, m.geom("pad_right").id: 0.0}
    f = np.zeros(6)
    for i in range(d.ncon):
        mujoco.mj_contactForce(m, d, i, f)
        for g in (d.contact[i].geom1, d.contact[i].geom2):
            if g in force:
                force[g] += f[0]
    np.testing.assert_allclose(list(force.values()), GripperParams().grip_force, rtol=0.02)


def test_stiff_servo_settles_without_limit_cycle():
    """kp 1e4 x [0.5, 2] with armature: every joint settles from a 10 mrad step (Euler, 1 ms).
    Without armature J6 limit-cycles even at kp 2000 (2026-11-18 stiffness study)."""
    from rl_insert.robot import ArmParams, GripperParams

    for scale in (0.5, 1.0, 2.0):
        spec = load_xarm6(ArmParams(kp=ArmParams().kp * scale))
        add_gripper(spec, GripperParams())
        m = spec.compile()
        m.opt.timestep, m.opt.integrator = 0.001, mujoco.mjtIntegrator.mjINT_EULER
        d = mujoco.MjData(m)
        q0 = np.array([0, -0.0728, -0.6831, 0, 0.7560, 0])
        d.qpos[:6] = q0
        d.ctrl[:6] = q0 + 0.01
        for _ in range(300):
            mujoco.mj_step(m, d)
        tail = []
        for _ in range(200):
            mujoco.mj_step(m, d)
            tail.append(d.qpos[:6].copy())
        tail = np.array(tail)
        assert np.abs(tail - (q0 + 0.01)).max() < 1e-3, scale  # within 1 mrad
        assert np.ptp(tail, axis=0).max() < 2e-4, scale  # no limit cycle
