"""Controller service: stream timing, pre-insert pose, and the loop against async fakes."""

import asyncio

import json

import numpy as np
import pytest
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.proto.component.arm import JointPositions
from viam.services.mlmodel import MLModel
from viam.utils import dict_to_struct

from model.package import control_contract
from policy_controller.core import Kinematics, Policy
from policy_controller.service import PolicyController
from policy_controller.stream import nominal_tip, orientation_vector, point_times, pre_insert, slip_or_jam, succeeded
from rl_insert.robot import kinematics_spec
from rl_insert.task import TaskParams, peg_insert_env_cfg


@pytest.fixture(scope="module")
def kin_cfg(tmp_path_factory):
    path = tmp_path_factory.mktemp("kin") / "kinematics.xml"
    path.write_text(kinematics_spec().to_xml())
    cfg = control_contract(peg_insert_env_cfg(TaskParams(), play=True))
    cfg["model"] = {"inputs": [{"name": "obs"}], "outputs": [{"name": "actions"}]}
    cfg["robustness"] = {"hole_bias_m": list(TaskParams().hole_bias), "grasp_offset_m": TaskParams().grasp_offset}  # build_config adds the rest
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
        return JointPositions(values=self.q)

    async def move_through_joint_positions_streamed(self, batches):
        async for b in batches:
            self.batches.append(b)
        yield None

    async def stop(self):
        self.stopped += 1


class FakeModel:
    def __init__(self, action):
        self.action = action

    delay = 0.0

    async def infer(self, tensors):
        await asyncio.sleep(self.delay)
        return {"actions": np.array([self.action], np.float32)}


def controller(kin_cfg, hole, action):
    kin, cfg = kin_cfg
    c = PolicyController.__new__(PolicyController)
    q = np.array(cfg["control"]["default_joint_pos"])
    c.cfg, c.timeout_s, c.max_overruns, c.jam_s = cfg, 10, 3, 2.0
    c.hole = c.hole_cfg = np.array(hole)
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


def test_orientation_vector_roundtrip(kin_cfg):
    kin, cfg = kin_cfg
    q = np.array(cfg["control"]["default_joint_pos"]) + [0.3, 0.2, -0.1, 0.5, 0.4, 0.7]
    rot = kin.fk(q)[1]
    o, theta = orientation_vector(rot)
    lon, lat = np.arctan2(o[1], o[0]), np.arccos(o[2])

    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    ry = np.array([[np.cos(lat), 0, np.sin(lat)], [0, 1, 0], [-np.sin(lat), 0, np.cos(lat)]])
    np.testing.assert_allclose((rz(lon) @ ry @ rz(theta))[:, 0], rot[:, 0], atol=1e-9)


def test_overruns_fail(kin_cfg):
    c = controller(kin_cfg, [0.4, 0, 0.05], np.zeros(6))
    c.mlmodel.delay, c.max_overruns = 0.08, 1  # slower than the 50 ms tick
    asyncio.run(c.run())
    assert (c.state, c.reason, c.arm.stopped) == ("failed", "overruns", 1)


def test_start_while_running_rejected_and_close_stops(kin_cfg):
    async def go():
        c = controller(kin_cfg, [0.4, 0, 0.05], np.zeros(6))
        c.state = "idle"
        assert await c.do_command({"command": "start"}) == {"status": "running"}
        with pytest.raises(RuntimeError):
            await c.do_command({"command": "start"})
        await asyncio.sleep(0.15)
        await c.close()
        assert c.task.done() and c.arm.stopped == 1
        return c

    c = asyncio.run(go())
    assert c.steps >= 1


def test_new_configures(kin_cfg, tmp_path):
    kin, cfg = kin_cfg
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    attrs = {"arm": "a", "gripper": "g", "mlmodel": "m", "package_dir": str(tmp_path), "hole_mouth_mm": [400, 0, 50]}
    conf = ComponentConfig(name="ctl", attributes=dict_to_struct(attrs))
    assert PolicyController.validate_config(conf)[0] == ["a", "g", "m"]
    deps = {Arm.get_resource_name("a"): FakeArm([0] * 6), Gripper.get_resource_name("g"): object(), MLModel.get_resource_name("m"): FakeModel(np.zeros(6))}
    c = PolicyController.new(conf, deps)
    assert c.state == "idle" and c.hole.tolist() == [0.4, 0, 0.05] and c.max_overruns == 3


def test_start_takes_a_per_episode_hole(kin_cfg):
    c = controller(kin_cfg, [0.4, 0, 0.05], np.zeros(6))
    c.state, c.task = "idle", None

    async def go():
        await c.do_command({"command": "start", "hole_mouth_mm": [401, -2, 50.5]})
        await c._halt()

    asyncio.run(go())
    np.testing.assert_allclose(c.hole, [0.401, -0.002, 0.0505])
    np.testing.assert_allclose(c.hole_cfg, [0.4, 0, 0.05])


def test_success_allows_for_the_hole_estimate_error(kin_cfg):
    kin, cfg = kin_cfg
    q = np.array(cfg["control"]["default_joint_pos"])
    tip = nominal_tip(kin, cfg, q)
    # 14.2 mm below an estimate that is 1.9 mm off sideways: in, given training's 2/2/1 mm estimate error
    assert succeeded(kin, cfg, q, tip + [0.0019, 0.0, 0.0142])
    assert not succeeded(kin, cfg, q, tip + [0.0, 0.0, 0.0135])  # too shallow even for the worst-case error
    assert not succeeded(kin, cfg, q, tip + [0.004, 0.0, 0.016])  # off the axis by more than error + tolerance


def test_slip_and_jam(kin_cfg):
    kin, cfg = kin_cfg
    cfg = {**cfg, "robustness": {"hole_bias_m": [0.002, 0.002, 0.001], "grasp_offset_m": 0.003},
           "task": {**cfg["task"], "peg_hole": {**cfg["task"].get("peg_hole", {}), "depth": 0.02}}}
    q = np.array(cfg["control"]["default_joint_pos"])
    tip = nominal_tip(kin, cfg, q)
    assert slip_or_jam(kin, cfg, q, tip + [0, 0, 0.025], 0.0, 2.0) == ("slip", True)  # 25 mm: past 20 + 1 + 3
    assert slip_or_jam(kin, cfg, q, tip + [0, 0, 0.010], 1.0, 2.0) == (None, True)
    assert slip_or_jam(kin, cfg, q, tip + [0, 0, 0.010], 2.5, 2.0) == ("jam", True)
    assert slip_or_jam(kin, cfg, q, tip - [0, 0, 0.010], 9.0, 2.0) == (None, False)  # above the mouth
