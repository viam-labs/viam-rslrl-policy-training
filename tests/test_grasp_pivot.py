"""The grasp's pivot stiffness spans the real G2's: holds in free air, pivots mm under side load."""

from dataclasses import replace

import torch

from rl_insert.task.checks import make_env, tip_lateral_under_load
from rl_insert.task.env_cfg import TaskParams


def _env(ts):
    # max_slip off: a soft grasp under 20 N would otherwise reset mid-measurement
    return make_env(num_envs=16, device="cpu", p=replace(TaskParams(), grasp_torquescale=(ts, ts), max_slip=1.0))


def test_soft_grasp_pivots_mm_under_20N():
    assert tip_lateral_under_load(_env(TaskParams().grasp_torquescale[0]), force_n=20.0) > 2.0


def test_stiff_grasp_barely_moves_under_20N():
    assert tip_lateral_under_load(_env(TaskParams().grasp_torquescale[1]), force_n=20.0) < 1.0


def test_torquescale_sampled_per_episode():
    env = make_env(num_envs=64, device="cpu")
    env.reset()
    ts = env.sim.model.eq_data[:, env.insert_state.weld_id, 10]
    lo, hi = TaskParams().grasp_torquescale
    assert ts.min() >= lo and ts.max() <= hi and torch.unique(ts).numel() > 32
