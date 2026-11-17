"""Contact spike: does MuJoCo Warp handle the peg and hole like CPU MuJoCo, at scale?

Presses N pegs into the hole from random offsets, directions and tilts (the arm-driven rig from
rl_insert.peg_hole), on Warp, and the first K of them on CPU MuJoCo. Prints a JSON report and
exits 1 if the gate fails:
  - no world blows up (non-finite, left the fixture, sank through, bounced up),
  - over the K reference worlds, Warp's insertion rate is within 10 points of CPU MuJoCo's,
  - the contact buffer never passes 90% full (a full one drops contacts silently).
Per-world agreement is reported too, but not gated: pegs balanced on the rim go either way
(float32 vs float64), and RL only sees the statistics.

    uv run python spikes/contact_warp.py --device cpu --nworld 16 --reference 16
    uv run python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene

SPEED = 0.04  # m/s, the policy's top speed: 2 mm per 50 ms step
SECONDS = 3.0
CTRL_EVERY = 10  # steps between ctrl updates (0.4 mm of ramp), to keep host-device copies rare
NCONMAX = 128  # contacts per world; one wedged peg peaked at 94
NJMAX = 1024  # constraint rows per world


def initial_states(p: PegHoleParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 5) qpos: tip x, y, z and tilts, offset up to 4 mm past the chamfer in a random direction."""
    r = rng.uniform(0, p.clearance + p.chamfer + 0.004, n)
    a = rng.uniform(0, 2 * math.pi, n)
    tilt = np.radians(rng.uniform(0, 5, (n, 2)))
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, 0.003), tilt])


def target_z(step: int, dt: float) -> float:
    return 0.003 - SPEED * step * dt


def classify(p: PegHoleParams, tip: np.ndarray) -> str:
    if not np.all(np.isfinite(tip)):
        return "blew_up"
    if np.hypot(*tip[:2]) >= p.hole_radius + p.wall or tip[2] <= -p.depth - 0.0005 or tip[2] >= 0.003:
        return "blew_up"
    return "inserted" if tip[2] < -p.depth + 0.001 else "not_inserted"


def run_cpu(m: mujoco.MjModel, q0: np.ndarray) -> np.ndarray:
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.ctrl[:] = q0
    for i in range(int(SECONDS / m.opt.timestep)):
        if i % CTRL_EVERY == 0:
            d.ctrl[2] = target_z(i, m.opt.timestep)
        mujoco.mj_step(m, d)
    return d.qpos[:3].copy()


def run_warp(m: mujoco.MjModel, q0: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Final tips, world-steps per second, and peak contact-buffer use (fraction of capacity)."""
    n = len(q0)
    mw = mjw.put_model(m)
    # nconmax and njmax are per world; the total contact buffer is nconmax * nworld
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=n, nconmax=NCONMAX, njmax=NJMAX)
    peak_contacts = 0
    dw.qpos = wp.array(q0.astype(np.float32), dtype=float)
    ctrl = q0.astype(np.float32).copy()
    steps = int(SECONDS / m.opt.timestep)
    start = time.perf_counter()
    for i in range(steps):
        if i % CTRL_EVERY == 0:
            ctrl[:, 2] = target_z(i, m.opt.timestep)
            dw.ctrl = wp.array(ctrl, dtype=float)
            peak_contacts = max(peak_contacts, int(dw.nacon.numpy()[0]))
        mjw.step(mw, dw)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return dw.qpos.numpy()[:, :3], n * steps / elapsed, peak_contacts / (NCONMAX * n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--reference", type=int, default=64)
    ap.add_argument("--clearance", type=float, nargs="+", default=[0.001, 0.0005, 0.0002])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    wp.config.quiet = True
    wp.set_device(args.device)
    rng = np.random.default_rng(args.seed)
    report, ok = {"device": args.device, "nworld": args.nworld, "results": []}, True
    for c in args.clearance:
        p = PegHoleParams(clearance=c)
        m = driven_peg_scene(p).compile()
        q0 = initial_states(p, args.nworld, rng)
        tips, rate, buffer_use = run_warp(m, q0)
        warp_outcomes = [classify(p, t) for t in tips]
        k = min(args.reference, args.nworld)
        cpu_outcomes = [classify(p, run_cpu(m, q0[i])) for i in range(k)]
        agree = sum(a == b for a, b in zip(warp_outcomes[:k], cpu_outcomes)) / k
        warp_rate = warp_outcomes[:k].count("inserted") / k
        cpu_rate = cpu_outcomes.count("inserted") / k
        counts = {o: warp_outcomes.count(o) for o in ("inserted", "not_inserted", "blew_up")}
        # a full contact buffer drops contacts silently, which looks like penetration
        passed = counts["blew_up"] == 0 and abs(warp_rate - cpu_rate) <= 0.1 and buffer_use < 0.9
        ok &= passed
        report["results"].append({"clearance_mm": c * 1e3, "warp": counts, "insert_rate_warp": round(warp_rate, 3),
                                  "insert_rate_cpu": round(cpu_rate, 3), "per_world_agreement": round(agree, 3),
                                  "contact_buffer_peak_use": round(buffer_use, 3),
                                  "world_steps_per_s": round(rate), "passed": passed})
    report["passed"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
