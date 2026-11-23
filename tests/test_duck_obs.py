import json
from pathlib import Path

import numpy as np
import pytest

from duck_policy.obs import Command, build_obs, phase_command

DEFAULT = np.array([0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530])


def test_layout_matches_infer_policy():
    q, dq = DEFAULT + 0.01 * np.arange(14), -0.02 * np.arange(14)
    last = 0.5 * np.ones(14)
    obs = build_obs(gyro=[0.1, 0.2, 0.3], gravity=[0.0, 0.05, -0.99], q=q, dq=dq,
                    default_pose=DEFAULT, last_action=last, command=Command(0.2, -0.1, 0.5).vector())
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
    assert len(g["ticks"]) == 50
    assert any(t["command"][0] and t["command"][2] for t in g["ticks"])
    for tick in g["ticks"]:
        vx, vy, wz = tick["command"]
        obs = build_obs(gyro=tick["gyro"], gravity=tick["gravity"], q=tick["q"], dq=tick["dq"],
                        default_pose=g["default_pose"], last_action=tick["last_action"], command=Command(vx, vy, wz).vector())
        assert obs == pytest.approx(np.array(tick["obs"], np.float32), abs=1e-6)


def test_phase_command():
    assert np.allclose(phase_command(0.25)[:3], [0, 1, 0], atol=1e-7) and not phase_command(0.25)[3:].any()


from duck_policy.obs import DEFAULT_LAYOUT, obs_layout  # noqa: E402


def layout_spec(pairs, dim=None):
    return {"terms": [{"name": n, "dim": d} for n, d in pairs], **({"dim": dim} if dim is not None else {})}


def test_obs_layout_defaults_for_old_packages():
    # the 0.1.0 packages carry a descriptive string here, and none at all is the same
    assert obs_layout({"observation": "gyro(3) gravity(3) ..."}) == DEFAULT_LAYOUT
    assert obs_layout({}) == DEFAULT_LAYOUT


def test_obs_layout_from_config_drives_build_obs():
    layout = obs_layout({"observation": layout_spec([("q_rel", 14), ("gyro", 3), ("command", 4)], dim=21), "model": {"obs_len": 21}})
    obs = build_obs(layout=layout, gyro=[0.1, 0.2, 0.3], gravity=[0, 0, -1], q=DEFAULT + 0.5, dq=np.zeros(14),
                    default_pose=DEFAULT, last_action=np.zeros(14), command=Command(0.2).vector(4))
    assert obs.shape == (21,) and obs[:14] == pytest.approx(0.5) and obs[14:17] == pytest.approx([0.1, 0.2, 0.3])
    assert obs[17:] == pytest.approx([0.2, 0, 0, 0])


@pytest.mark.parametrize("obs, model", [
    (layout_spec([("torque", 3)]), {}),                       # unknown term
    (layout_spec([("gyro", 0)]), {}),                         # empty term
    (layout_spec([("command", 2)]), {}),                      # no room for a twist
    (layout_spec([("gyro", 3), ("dq", 14)]), {"obs_len": 61}),  # doesn't add up to the model's input
    (layout_spec([("gyro", 3)], dim=6), {}),                  # doesn't add up to its own dim
])
def test_obs_layout_rejects_bad_configs(obs, model):
    with pytest.raises(ValueError):
        obs_layout({"observation": obs, "model": model})


def test_build_obs_rejects_a_term_of_the_wrong_size():
    with pytest.raises(ValueError, match="dq"):
        build_obs(layout=[("dq", 12)], gyro=[0] * 3, gravity=[0] * 3, q=DEFAULT, dq=np.zeros(14),
                  default_pose=DEFAULT, last_action=np.zeros(14), command=np.zeros(13))
