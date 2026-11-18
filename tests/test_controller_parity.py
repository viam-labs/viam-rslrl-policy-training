"""The controller's IK reproduces the training env's, step for step."""

import numpy as np
import torch

from model.package import control_contract
from policy_controller.core import Kinematics
from rl_insert.robot import kinematics_spec
from rl_insert.task.checks import make_env


def test_ik_goal_matches_training(tmp_path):
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    cfg = control_contract(env.cfg)
    kin = Kinematics(tmp_path / "kinematics.xml", cfg["control"])
    term = env.action_manager.get_term("ee_delta")
    gen = torch.Generator().manual_seed(0)
    worst = 0.0
    for _ in range(10):
        a = torch.randn(4, 6, generator=gen)
        q = term._entity.data.joint_pos[:, term._joint_ids].numpy().copy()
        env.step(a)
        for i in range(4):
            q_goal = kin.step_goal(q[i], a[i].numpy(), cfg["action"])
            worst = max(worst, np.abs(q_goal - term._q_to[i].numpy()).max())
            np.testing.assert_allclose(q_goal, term._q_to[i].numpy(), atol=1e-5)
    print("max |q_goal - q_to|", worst)
