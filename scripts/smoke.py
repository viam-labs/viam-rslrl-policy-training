"""Env smoke test: build, reset, check the reset geometry, step with zero / scripted actions, time it.

uv run python -m scripts.smoke [num_envs]
"""

import sys
import time

import torch

from mjlab.envs import ManagerBasedRlEnv

from rl_insert.task import mdp
from rl_insert.task.env_cfg import TaskParams, peg_insert_env_cfg


def main(num_envs: int = 4) -> None:
    cfg = peg_insert_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    p = TaskParams()
    obs, _ = env.reset()
    print(
        "obs dims:",
        {k: tuple(v.shape) for k, v in obs.items()},
        "action dim:",
        env.action_manager.total_action_dim,
    )
    print("actor terms:", env.observation_manager.active_terms["actor"])

    # Reset geometry: tip 10-20 mm above the mouth, within the lateral range, peg in hand.
    rel = mdp.true_tip(env) - mdp.hole_mouth(env)
    slip = (mdp.true_tip(env) - mdp.nominal_tip(env)).norm(dim=-1)
    print("tip - mouth (mm):\n", (rel * 1e3).round(decimals=2))
    print("tip vs nominal tip (mm):", (slip * 1e3).round(decimals=2))
    assert ((rel[:, 2] >= p.start_height[0] - 1e-6) & (rel[:, 2] <= p.start_height[1] + 1e-6)).all()
    assert (slip < 0.006).all()

    # Hold still for 1 s: the peg must stay in hand (grip + gravcomp), no termination.
    zero = torch.zeros(num_envs, env.action_manager.total_action_dim)
    for _ in range(20):
        obs, rew, term, trunc, _ = env.step(zero)
    slip2 = (mdp.true_tip(env) - mdp.nominal_tip(env)).norm(dim=-1)
    print("after 1 s hold, slip (mm):", (slip2 * 1e3).round(decimals=2), "terminated:", term.tolist())
    print("ik error after hold:", obs["actor"][:, -6:].abs().max().item())
    assert not term.any(), "terminated while holding still"

    # Scripted descent: move toward the hole (true hole, no bias) and down; report depth / force.
    t0 = time.perf_counter()
    steps = 60
    for i in range(steps):
        rel = mdp.true_tip(env) - mdp.hole_mouth(env)
        a = torch.zeros(num_envs, 6)
        a[:, :2] = (-rel[:, :2] / p.action_pos_scale).clamp(-1, 1)
        a[:, 2] = -0.5 if i > 5 else 0.0
        obs, rew, term, trunc, extras = env.step(a)
        if i % 10 == 0 or term.any():
            lat, depth = mdp.tip_in_hole(env)
            f = mdp.peg_hole_force(env).norm(dim=-1)
            print(
                f"step {i:3d} depth(mm) {[round(x, 1) for x in (depth * 1e3).tolist()]} "
                f"lat(mm) {[round(x, 2) for x in (lat * 1e3).tolist()]} "
                f"F(N) {[round(x, 1) for x in f.tolist()]} "
                f"rew {[round(x, 3) for x in rew.tolist()]} term {term.tolist()}"
            )
        if term.any():
            tm = env.termination_manager
            print("   why:", {n: tm.get_term(n).nonzero().flatten().tolist() for n in tm.active_terms})
    dt = (time.perf_counter() - t0) / steps
    print(f"wall time per policy step ({num_envs} envs, CPU): {dt * 1e3:.1f} ms")
    print(
        "contacts in world 0 (nacon total):",
        int(env.sim.data.nacon.numpy()[0]) if hasattr(env.sim.data.nacon, "numpy") else env.sim.data.nacon,
    )
    print("episode log keys:", sorted(extras.get("log", {}).keys())[:20])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
