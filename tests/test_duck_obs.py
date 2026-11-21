import json
from pathlib import Path

import numpy as np
import pytest

from duck_policy.obs import Command, build_obs

DEFAULT = np.array([0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530])


def test_layout_matches_infer_policy():
    q, dq = DEFAULT + 0.01 * np.arange(14), -0.02 * np.arange(14)
    last = 0.5 * np.ones(14)
    obs = build_obs(gyro=[0.1, 0.2, 0.3], gravity=[0.0, 0.05, -0.99], q=q, dq=dq,
                    default_pose=DEFAULT, last_action=last, command=Command(0.2, -0.1, 0.5))
    assert obs.dtype == np.float32 and obs.shape == (61,)
    assert obs[0:3] == pytest.approx([0.1, 0.2, 0.3])
    assert obs[3:6] == pytest.approx([0.0, 0.05, -0.99])
    assert obs[6:20] == pytest.approx(0.01 * np.arange(14), abs=1e-6)
    assert obs[20:34] == pytest.approx(dq)
    assert obs[34:48] == pytest.approx(last)
    assert obs[48:51] == pytest.approx([0.2, -0.1, 0.5])
    assert not obs[51:61].any()  # head and body pose slots: zero for a walk


def test_command_clips_to_the_trained_ranges():
    c = Command(1.0, -1.0, 3.0).clipped([-0.4, -0.3, -1.0], [0.4, 0.3, 1.0])
    assert (c.vx, c.vy, c.wz) == (0.4, -0.3, 1.0)


GOLDEN = Path(__file__).parent / "data" / "duck_golden.json"


def test_build_obs_reproduces_infer_policy_exactly():
    g = json.loads(GOLDEN.read_text())
    assert len(g["ticks"]) == 25
    for tick in g["ticks"]:
        vx, vy, wz = tick["command"]
        obs = build_obs(gyro=tick["gyro"], gravity=tick["gravity"], q=tick["q"], dq=tick["dq"],
                        default_pose=g["default_pose"], last_action=tick["last_action"], command=Command(vx, vy, wz))
        assert obs == pytest.approx(np.array(tick["obs"], np.float32), abs=1e-6)
