"""The controller's IK reproduces the training env's, step for step."""

import copy

import numpy as np
import pytest
import torch

from model.package import control_contract
from policy_controller.core import Kinematics, Policy
from rl_insert.robot import kinematics_spec
from rl_insert.task import mdp
from rl_insert.task.checks import make_env


# scale 2000 makes the max_dq clamp bind; the second case turns on the joint-limit and posture terms.
@pytest.mark.parametrize(("scale", "weights", "atol"), [(1, (0.0, 0.0), 1e-5), (2000, (1.0, 0.3), 5e-5)])
def test_ik_goal_matches_training(tmp_path, scale, weights, atol):
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    cfg = control_contract(env.cfg)
    control = copy.deepcopy(cfg["control"])
    term = env.action_manager.get_term("ee_delta")
    term.cfg.joint_limit_weight, term.cfg.posture_weight = weights
    control["ik"]["joint_limit_weight"], control["ik"]["posture_weight"] = weights
    kin = Kinematics(tmp_path / "kinematics.xml", control)
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        a = torch.randn(4, 6, generator=gen) * scale
        q = term._entity.data.joint_pos[:, term._joint_ids].numpy().copy()
        env.step(a)
        for i in range(4):
            q_goal = kin.step_goal(q[i], a[i].numpy(), cfg["action"])
            np.testing.assert_allclose(q_goal, term._q_to[i].numpy(), atol=atol)


def test_observation_matches_training(tmp_path):
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    cfg = control_contract(env.cfg)
    terms = env.cfg.observations["actor"].terms
    dims = [d[0] for d in env.observation_manager.group_obs_term_dim["actor"]]
    config = cfg | {"observation": {"terms": [{"name": n, "dim": d} for n, d in zip(terms, dims, strict=True)]}}
    kin = Kinematics(tmp_path / "kinematics.xml", cfg["control"])
    policies = [Policy(config, kin) for _ in range(4)]  # one Kinematics: act() follows observe() per env
    term = env.action_manager.get_term("ee_delta")
    names = list(terms)
    offs = np.cumsum([0, *dims])
    worst = dict.fromkeys(names, 0.0)
    gen = torch.Generator().manual_seed(1)
    resets = 0
    for step in range(150):
        obs = env.observation_manager.compute()["actor"].numpy()
        hole = (mdp.hole_mouth(env) + mdp.state(env).hole_bias - env.scene.env_origins).numpy()  # base frame, as on the arm
        q = term._entity.data.joint_pos[:, term._joint_ids].numpy().copy()
        a = torch.randn(4, 6, generator=gen) * 3
        for i in range(4):
            if env.episode_length_buf[i] == 0:
                policies[i].reset()
                resets += step > 0
            mine = policies[i].observe(q[i], hole[i])
            assert mine.shape == obs[i].shape
            for j, n in enumerate(names):
                err = np.abs(mine[offs[j] : offs[j + 1]] - obs[i, offs[j] : offs[j + 1]]).max()
                worst[n] = max(worst[n], err)
            policies[i].act(q[i], hole[i], a[i].numpy())
        env.step(a)
    assert resets > 0, "no env reset mid-run: raise steps or action scale"
    assert all(e < 1e-5 for e in worst.values()), worst


def test_unknown_term_raises(tmp_path):
    env = make_env(num_envs=1, device="cpu")
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    cfg = control_contract(env.cfg)
    policy = Policy(cfg | {"observation": {"terms": [{"name": "bogus"}]}}, Kinematics(tmp_path / "kinematics.xml", cfg["control"]))
    with pytest.raises(ValueError, match="no recipe for observation term bogus"):
        policy.observe(np.zeros(6), np.zeros(3))
