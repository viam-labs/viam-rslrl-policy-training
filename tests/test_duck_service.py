import asyncio
import json

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from duck_policy.service import DuckPolicy


def cfg(attrs):
    return ComponentConfig(name="walker", api="rdk:service:generic", model="devrel:duck-policy:controller",
                           attributes=dict_to_struct(attrs))


def test_validate_requires_duck_mlmodel_package():
    with pytest.raises(ValueError, match="duck"):
        DuckPolicy.validate_config(cfg({"mlmodel": "walk", "package_dir": "/x"}))
    deps, _ = DuckPolicy.validate_config(cfg({"duck": "duck", "mlmodel": "walk", "package_dir": "/x"}))
    assert sorted(deps) == ["duck", "walk"]


def test_set_command_and_status_without_running(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "joints": ["j"] * 14, "default_pose": [0.0] * 14, "action_scale": 1.0, "joint_limits": [[-1, 1]] * 14,
        "control_hz": 50, "model": {"input": "obs", "output": "actions"},
        "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}, "safety": {}}))
    svc = DuckPolicy("walker")
    svc._configure({"duck": "duck", "mlmodel": "walk", "package_dir": str(tmp_path)}, duck=object(), mlmodel=object())
    out = asyncio.run(svc.do_command({"command": "set_command", "vx": 0.9, "vy": 0, "wz": 0.2}))
    assert out["command"] == [0.4, 0.0, 0.2] and out["command_clipped"] is True
    status = asyncio.run(svc.do_command({"command": "status"}))
    assert status["state"] == "idle"
    with pytest.raises(ValueError, match="unknown command"):
        asyncio.run(svc.do_command({"command": "fly"}))


class StubDuck:
    def __init__(self, gz, q0=0.0):
        self.gz, self.q0, self.events = gz, q0, []

    async def do_command(self, cmd):
        if cmd["io"].get("torque"):
            self.events.append("torque")
        return {"joints": ["j"] * 14, "t": 0.0, "q": [self.q0] * 14, "dq": [0.0] * 14, "gyro": [0.0] * 3,
                "gravity": [0.0, 0.0, self.gz], "generation": 0, "torque": False}


class StubMl:
    async def infer(self, inputs):
        return {"actions": np.zeros((1, 14), dtype=np.float32)}


class StubWorld:
    def __init__(self, svc=None, fail_on=None):
        self.svc, self.fail_on, self.cmds, self.task_at_play = svc, fail_on, [], None

    async def do_command(self, cmd):
        name = cmd["command"]
        self.cmds.append(name)
        if name == "play":
            self.torque_at_play = "torque" in self.svc.duck.events
            self.task_at_play = self.svc.task is not None
        if name == self.fail_on:
            raise RuntimeError("boom")
        return {}


def make_walker(tmp_path, gz, world=True, fail_on=None, q0=0.0, command=None):
    (tmp_path / "config.json").write_text(json.dumps({
        "joints": ["j"] * 14, "default_pose": [0.0] * 14, "action_scale": 1.0, "joint_limits": [[-1, 1]] * 14,
        "control_hz": 50, "model": {"input": "obs", "output": "actions"},
        "command": command or {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}, "safety": {}}))
    svc = DuckPolicy("walker")
    w = StubWorld(svc, fail_on) if world else None
    svc._configure({"duck": "duck", "mlmodel": "walk", "package_dir": str(tmp_path)},
                   duck=StubDuck(gz, q0), mlmodel=StubMl(), world=w)
    return svc, w


async def start_and_stop(svc, command="start"):
    try:
        return await svc.do_command({"command": command})
    finally:
        await svc.do_command({"command": "stop"})


def test_start_recovers_fallen_duck(tmp_path):
    svc, world = make_walker(tmp_path, gz=0.2)
    out = asyncio.run(start_and_stop(svc))
    assert world.cmds == ["pause", "reset", "play"]
    assert world.task_at_play is True and world.torque_at_play is True
    assert out == {"state": "running", "recovered": "fallen"}


def test_start_upright_with_world_makes_no_world_calls(tmp_path):
    svc, world = make_walker(tmp_path, gz=-1.0)
    out = asyncio.run(start_and_stop(svc))
    assert world.cmds == [] and out["recovered"] is False


def test_start_fallen_without_world(tmp_path):
    svc, _ = make_walker(tmp_path, gz=0.2, world=False)
    out = asyncio.run(start_and_stop(svc))
    assert out == {"state": "running", "recovered": False}


def test_start_sends_play_even_if_reset_fails(tmp_path):
    svc, world = make_walker(tmp_path, gz=0.2, fail_on="reset")
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(start_and_stop(svc))
    assert world.cmds == ["pause", "reset", "play"]


def test_validate_world_is_a_dep():
    deps, _ = DuckPolicy.validate_config(
        cfg({"duck": "duck", "mlmodel": "walk", "package_dir": "/x", "world": "sim-world"}))
    assert sorted(deps) == ["duck", "sim-world", "walk"]


def test_concurrent_starts_make_one_task(tmp_path):
    svc, _ = make_walker(tmp_path, gz=-1.0)

    async def go():
        try:
            return await asyncio.gather(svc.do_command({"command": "start"}),
                                        svc.do_command({"command": "start"}), return_exceptions=True)
        finally:
            await svc.do_command({"command": "stop"})

    res = asyncio.run(go())
    assert sum(isinstance(r, ValueError) and "already running" in str(r) for r in res) == 1
    assert sum(isinstance(r, dict) for r in res) == 1


def test_start_recovers_when_pose_far_from_default(tmp_path):
    svc, world = make_walker(tmp_path, gz=-1.0, q0=0.3)
    out = asyncio.run(start_and_stop(svc))
    assert world.cmds == ["pause", "reset", "play"] and out["recovered"] == "off_pose"


def test_start_slightly_off_pose_hands_straight_to_policy(tmp_path):
    # the walker stands 0.063 rad off default (neck_pitch): a handover must not reset the world
    svc, world = make_walker(tmp_path, gz=-1.0, q0=0.1)
    out = asyncio.run(start_and_stop(svc))
    assert world.cmds == [] and out["recovered"] is False


def test_phase_service_starts_with_cycles_and_rejects_set_command(tmp_path):
    svc, _ = make_walker(tmp_path, gz=-1.0, world=False, command={"mode": "phase", "period_s": 0.1})

    async def go():
        try:
            out = await svc.do_command({"command": "start", "cycles": 2})
            assert svc.loop.cycles == 2
            return out
        finally:
            await svc.do_command({"command": "stop"})

    assert asyncio.run(go())["state"] == "running"
    with pytest.raises(ValueError):
        asyncio.run(svc.do_command({"command": "set_command", "vx": 0.1}))


def test_start_rejects_negative_cycles(tmp_path):
    svc, _ = make_walker(tmp_path, gz=-1.0, world=False, command={"mode": "phase", "period_s": 0.1})
    with pytest.raises(ValueError, match="cycles"):
        asyncio.run(svc.do_command({"command": "start", "cycles": -1}))
