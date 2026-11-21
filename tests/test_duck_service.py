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
