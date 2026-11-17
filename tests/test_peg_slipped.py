"""Slip is measured from the sampled grasp, so a randomised grasp at rest is not slip.

At rest the tip sits within 0.1 mm of the weld target (256 envs, CPU); off the nominal grasp it's
up to ~5 mm, half the envs past 2.5 mm.
"""
import torch

from rl_insert.task import mdp
from rl_insert.task.checks import make_env


def test_randomised_grasp_at_rest_is_not_slip():
    env = make_env(num_envs=64, device="cpu")
    env.reset()
    for _ in range(20):
        env.step(torch.zeros(64, env.action_manager.total_action_dim))
    off_nominal = (mdp.true_tip(env) - mdp.nominal_tip(env)).norm(dim=-1)
    assert (off_nominal > 0.001).any()  # the grasp really is randomised past the gate below
    assert not mdp.peg_slipped(env, max_slip=0.001).any()
