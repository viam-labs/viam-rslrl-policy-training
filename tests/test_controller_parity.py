"""The controller's IK reproduces the training env's, step for step."""

import copy

import numpy as np
import pytest
import torch

from model.package import control_contract
from policy_controller.core import Kinematics
from rl_insert.robot import kinematics_spec
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
