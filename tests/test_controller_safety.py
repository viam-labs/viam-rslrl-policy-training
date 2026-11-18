"""Controller safety: caps, joint limits, workspace box, non-finite input. The replay test needs a measured rollout."""

import os

import numpy as np
import pytest

from model.package import control_contract
from policy_controller.core import DEFAULTS, Kinematics, Policy, SafetyStop
from rl_insert.robot import kinematics_spec
from rl_insert.task import TaskParams, peg_insert_env_cfg


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    path = tmp_path_factory.mktemp("kin") / "kinematics.xml"
    path.write_text(kinematics_spec().to_xml())
    cfg = control_contract(peg_insert_env_cfg(TaskParams(), play=True))
    return cfg, path


def make(cfg, **limits):
    cfg, path = cfg
    return Policy(cfg, Kinematics(path, cfg["control"]), limits)


def tcp(policy, q):
    return policy.kin.fk(q)[0]


def hole_at_tip(policy, q):
    pos, rot = policy.kin.fk(q)
    return pos + rot[:, 2] * policy.cfg["robot"]["nominal_tcp"]["protrusion_m"]


def test_action_is_clipped_but_recorded_raw(cfg):
    q = np.array(cfg[0]["control"]["default_joint_pos"])
    big, capped = make(cfg, workspace_mm=1e6), make(cfg, workspace_mm=1e6)
    a = np.array([100.0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        big.act(q, hole_at_tip(big, q), a), capped.act(q, hole_at_tip(capped, q), a * DEFAULTS["action_clip"] / 100)
    )
    assert big.last_action[0] == 100.0


def test_step_scaled_to_exactly_the_cap(cfg):
    q = np.array(cfg[0]["control"]["default_joint_pos"])
    p = make(cfg, action_clip=1e6, workspace_mm=1e6, max_step_mm=5.0, max_joint_step_deg=90)
    goal = p.act(q, hole_at_tip(p, q), np.array([30.0, 0, 0, 0, 0, 0]))
    assert np.linalg.norm(tcp(p, goal) - tcp(p, q)) == pytest.approx(5e-3, rel=1e-6)
    p = make(cfg, action_clip=1e6, workspace_mm=1e6, max_step_mm=1e3, max_joint_step_deg=1.0)
    goal = p.act(q, hole_at_tip(p, q), np.array([30.0, 0, 0, 0, 0, 0]))
    assert np.abs(goal - q).max() == pytest.approx(np.deg2rad(1.0), rel=1e-9)


def test_hole_far_away_stops(cfg):
    q = np.array(cfg[0]["control"]["default_joint_pos"])
    p = make(cfg)
    hole = hole_at_tip(p, q) + np.array([0.1, 0, 0])
    with pytest.raises(SafetyStop) as e:
        p.act(q, hole, np.zeros(6))
    assert e.value.reason == "workspace"


@pytest.mark.parametrize("bad", ["q", "action", "hole"])
def test_non_finite_stops(cfg, bad):
    args = {"q": np.array(cfg[0]["control"]["default_joint_pos"]), "action": np.zeros(6), "hole": np.zeros(3)}
    args[bad][0] = np.nan
    with pytest.raises(SafetyStop) as e:
        make(cfg).act(args["q"], args["hole"], args["action"])
    assert e.value.reason == "non_finite"


def test_goal_stops_at_the_joint_limit(cfg):
    q = np.array(cfg[0]["control"]["default_joint_pos"])
    p = make(cfg, action_clip=1e6, workspace_mm=1e6, max_step_mm=1e3, max_joint_step_deg=90)
    lower, upper = np.array(cfg[0]["control"]["ik"]["joint_limits_rad"]).T
    q[3] = upper[3] - 1e-3  # next to its limit; a big rotation about the tool axis pushes joint 4 past it
    goal = p.act(q, hole_at_tip(p, q), np.array([0, 0, 0, 0, 0, -100.0]))
    assert (goal <= upper + 1e-12).all() and (goal >= lower - 1e-12).all()
    assert goal[3] == pytest.approx(upper[3]) or goal[3] == pytest.approx(lower[3])
    assert np.abs(goal - q).max() <= np.deg2rad(90) + 1e-12


def test_stop_leaves_state_unchanged(cfg):
    q = np.array(cfg[0]["control"]["default_joint_pos"])
    p = make(cfg)
    p.act(q, hole_at_tip(p, q), np.ones(6))
    before = (p.last_action.copy(), p.q_prev.copy(), p.p_des.copy())
    with pytest.raises(SafetyStop):
        p.act(q + 0.01, hole_at_tip(p, q) + 0.1, np.full(6, 2.0))
    for old, new in zip(before, (p.last_action, p.q_prev, p.p_des), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.skipif("POLICY_ENVELOPE_NPZ" not in os.environ, reason="needs scripts/policy_envelope.py's npz")
def test_no_cap_binds_on_the_recorded_rollout(cfg):
    d = np.load(os.environ["POLICY_ENVELOPE_NPZ"])  # (steps, num_envs, ...)
    steps, envs = d["action"].shape[:2]
    policies, binds = [make(cfg) for _ in range(envs)], 0
    for t in range(steps):
        for i, p in enumerate(policies):
            if d["first"][t, i]:
                p.reset()
            raw = p.kin.step_goal(d["q"][t, i], d["action"][t, i], cfg[0]["action"])  # the unsafe goal
            try:
                binds += not np.array_equal(p.act(d["q"][t, i], d["hole"][t, i], d["action"][t, i]), raw)
            except SafetyStop:
                binds += 1
    assert binds <= 0.001 * steps * envs, f"{binds} of {steps * envs} steps bound a cap"
