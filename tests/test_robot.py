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
