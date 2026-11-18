"""Controller service: stream timing, pre-insert pose, and the loop against async fakes."""

import asyncio

import numpy as np
import pytest

from model.package import control_contract
from policy_controller.core import Kinematics, Policy
from policy_controller.service import PolicyController
from policy_controller.stream import nominal_tip, point_times, pre_insert
from rl_insert.robot import kinematics_spec
from rl_insert.task import TaskParams, peg_insert_env_cfg


@pytest.fixture(scope="module")
def kin_cfg(tmp_path_factory):
    path = tmp_path_factory.mktemp("kin") / "kinematics.xml"
    path.write_text(kinematics_spec().to_xml())
    cfg = control_contract(peg_insert_env_cfg(TaskParams(), play=True))
    cfg["model"] = {"inputs": [{"name": "obs"}], "outputs": [{"name": "actions"}]}
    cfg["observation"] = {"terms": [{"name": n} for n in ("tip_rel_est_hole", "ee_rot6d", "joint_pos", "joint_vel", "last_action", "ik_error")]}  # build_config adds the real ones
    return Kinematics(path, cfg["control"]), cfg


def test_point_times():
    t = point_times(0, 0.05)
    assert t == pytest.approx([0.06, 0.07, 0.08, 0.09, 0.10])
    assert point_times(3, 0.05)[0] == pytest.approx(0.21) and point_times(3, 0.05)[-1] == pytest.approx(0.25)
    assert all(b > a for a, b in zip(t + point_times(1, 0.05), (t + point_times(1, 0.05))[1:]))


def test_pre_insert_pose(kin_cfg):
    kin, cfg = kin_cfg
    hole = np.array([0.42, 0.01, 0.05])
    q, pose = pre_insert(kin, cfg, hole)
    np.testing.assert_allclose(nominal_tip(kin, cfg, q), hole + [0, 0, 0.015], atol=1e-6)
    assert kin.fk(q)[1][:, 2] == pytest.approx([0, 0, -1], abs=1e-6)
    assert (pose["o_x"], pose["o_y"], pose["o_z"]) == pytest.approx((0, 0, -1), abs=1e-6)
    assert pose["z"] == pytest.approx(1000 * (0.105 + 0.1566), abs=1e-3)


class FakeArm:
    def __init__(self, q_deg):
        self.q, self.batches, self.stopped = q_deg, [], 0

    async def get_joint_positions(self):
        return self.q

    async def move_through_joint_positions_streamed(self, batches):
        async for b in batches:
            self.batches.append(b)
        yield None

    async def stop(self):
        self.stopped += 1


class FakeModel:
    def __init__(self, action):
        self.action = action

    async def infer(self, tensors):
        return {"actions": np.array([self.action], np.float32)}


def controller(kin_cfg, hole, action):
    kin, cfg = kin_cfg
    c = PolicyController.__new__(PolicyController)
    q = np.array(cfg["control"]["default_joint_pos"])
    c.cfg, c.hole, c.timeout_s, c.max_overruns = cfg, np.array(hole), 10, 3
    c.policy = Policy(cfg, kin)
    c.arm, c.mlmodel = FakeArm(np.degrees(q).tolist()), FakeModel(action)
    c.state, c.reason, c.steps, c.overruns, c.last_obs = "running", None, 0, 0, None
    return c


def test_loop_success_half_closes(kin_cfg):
    c = controller(kin_cfg, [0.4, 0, 0.081], np.zeros(6))  # default tip is 16 mm below this mouth, centred
    asyncio.run(c.run())
    assert c.state == "succeeded" and c.arm.stopped == 0 and len(c.arm.batches) == 1


def test_loop_safety_stop_fails_and_stops_arm(kin_cfg):
    c = controller(kin_cfg, [0.4, 0, 0.05], np.full(6, np.nan))
    asyncio.run(c.run())
    assert (c.state, c.reason, c.arm.stopped) == ("failed", "non_finite", 1)


def test_loop_streams_ramped_points(kin_cfg):
    c = controller(kin_cfg, [0.4, 0, 0.05], np.array([0, 0, -1.0, 0, 0, 0]))
    c.timeout_s = 0.3
    asyncio.run(c.run())
    assert c.state == "failed" and c.reason == "timeout" and c.steps >= 3
    first, second = c.arm.batches[1], c.arm.batches[2]
    assert [p.time.total_seconds() for p in first] == pytest.approx(point_times(0, 0.05))
    assert second[0].time > first[-1].time
