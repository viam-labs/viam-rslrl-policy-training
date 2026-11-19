"""Sim checks shared by the probe job (model.training --mode=probe) and the local tests."""

from __future__ import annotations

import time

import torch

from mjlab.envs import ManagerBasedRlEnv

from . import mdp
from .env_cfg import TaskParams, peg_insert_env_cfg


def make_env(
    num_envs: int, device: str, seed: int = 0, render: bool = False, p: TaskParams = TaskParams()
) -> ManagerBasedRlEnv:
    cfg = peg_insert_env_cfg(p, play=True)  # play: no observation noise or curriculum
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if render else None)


def tip_along_tcp_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Peg tip position along the TCP z axis, m: grows as the peg slides out of the grasp."""
    pos, rot = mdp.tcp_pose(env)
    return ((mdp.true_tip(env) - pos).unsqueeze(1) @ rot[:, :, 2:]).flatten()


def grasp_hold_slip(env: ManagerBasedRlEnv, seconds: float = 2.0, settle_s: float = 0.5) -> dict[str, float]:
    """Peg slip along the TCP z axis (mm, max |.| over envs) while the arm holds still.

    settle_mm: the first settle_s after reset (a pad-only grasp got knocked up to 5 mm here on
    Warp). creep_mm: the following `seconds`, the steady slip.
    """
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    def hold(duration: float) -> None:
        for _ in range(round(duration / env.step_dt)):
            _, _, term, trunc, _ = env.step(zero)
            assert not (term | trunc).any(), "an env reset during the hold; slip would be meaningless"

    z0 = tip_along_tcp_z(env)
    hold(settle_s)
    z1 = tip_along_tcp_z(env)
    hold(seconds)
    z2 = tip_along_tcp_z(env)
    return {"settle_mm": float((z1 - z0).abs().max()) * 1e3, "creep_mm": float((z2 - z1).abs().max()) * 1e3}


def tip_lateral_under_load(env: ManagerBasedRlEnv, force_n: float, seconds: float = 1.0) -> float:
    """Max over envs of the peg tip's move in the hand (mm, TCP-frame x) with `force_n` pushed on the tip
    along TCP x: the grasp's pivot + weld compliance. Measured in the TCP frame because a zero StepIK
    action re-targets from the measured joints each step, so under load the arm itself creeps."""
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(10):
        env.step(zero)  # settle
    peg = env.scene["peg"]
    bid = int(peg.indexing.body_ids[0])
    before = mdp.peg_in_hand(env)[:, 0].clone()
    for _ in range(round(seconds / env.step_dt)):
        _, rot = mdp.tcp_pose(env)
        f = rot[:, :, 0] * force_n  # TCP x in world
        # xfrc acts at the body's COM (~30 mm up the peg); add the moment that puts it at the tip
        tau = torch.cross(mdp.true_tip(env) - env.sim.data.xipos[:, bid], f, dim=-1)
        peg.write_external_wrench_to_sim(f.unsqueeze(1), tau.unsqueeze(1))
        _, _, term, trunc, _ = env.step(zero)
        assert not (term | trunc).any(), "an env reset mid-measurement"
    zero3 = torch.zeros(env.num_envs, 1, 3, device=env.device)
    peg.write_external_wrench_to_sim(zero3, zero3)  # the wrench persists until overwritten
    return float((mdp.peg_in_hand(env)[:, 0] - before).abs().max()) * 1000


def throughput(env: ManagerBasedRlEnv, steps: int = 50, warmup: int = 5) -> dict[str, float]:
    """Policy and physics steps per second (summed over envs) with uniform random actions."""
    env.reset()
    shape = (env.num_envs, env.action_manager.total_action_dim)
    for _ in range(warmup):
        env.step(torch.rand(shape, device=env.device) * 2 - 1)
    if env.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(torch.rand(shape, device=env.device) * 2 - 1)
    if env.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    policy = steps * env.num_envs / dt
    return {
        "policy_steps_per_s": policy,
        "physics_steps_per_s": policy * env.cfg.decimation,
        "seconds_per_policy_step": dt / steps,
    }
