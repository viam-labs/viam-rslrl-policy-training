# /// script
# dependencies = ["viam-sdk>=0.85", "numpy"]
# ///
"""Walk parity: the same command script through Pollen's reference (duck_reference.py, run in the
mujoco-sim worktree's environment) and through our stack on a Viam machine (duck-policy + onnx-cpu +
mujoco-sim io). Spec unit 5.

    uv run scripts/duck_parity.py ours <cell.json> <out.json> [--episodes 20]
    uv run scripts/duck_parity.py report <reference.json> <ours.json> <report.md>

Run the reference with duck_reference.py first. Metrics per side:
- falls (gravity z > -0.5 at any sample);
- per-segment velocity-tracking RMSE of body-frame vx and wz, from 0.1 s trunk pose samples;
- for ours, the walker's tick stats and the world's slips / real-time factor.

Pass (spec unit 5, with the milestone 2 decisions):
- at least 18 of 20 episodes slip-free, else the result is "machine too slow";
- falls(ours) <= falls(ref) + 1;
- RMSE(ours) <= 1.25 x RMSE(ref) for vx and for wz;
- mean tick rate >= 49.5 Hz, p99 tick time < 20 ms, under 1 % of ticks late.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from duck_reference import DURATION, SCRIPT, command_at  # noqa: E402

SKILL = Path("/Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell/scripts")
SETTLE_S = 0.5  # skip this much after each command change when scoring tracking


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tracking(samples: list[dict]) -> dict:
    """Body-frame vx and wz from consecutive pose samples, scored against the command, skipping SETTLE_S
    after each segment boundary."""
    bounds, acc = [], 0.0
    for seconds, *_ in SCRIPT:
        acc += seconds
        bounds.append(acc)
    err_vx, err_wz = [], []
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        t_mid = (a["t"] + b["t"]) / 2
        if any(0 <= t_mid - edge < SETTLE_S for edge in [0.0, *bounds]):
            continue
        ya, yb = yaw(a["trunk_quat_wxyz"]), yaw(b["trunk_quat_wxyz"])
        dyaw = math.atan2(math.sin(yb - ya), math.cos(yb - ya))
        ym = ya + dyaw / 2  # mid-interval heading: exact for a constant-curvature arc to first order
        dx, dy = b["trunk_pos"][0] - a["trunk_pos"][0], b["trunk_pos"][1] - a["trunk_pos"][1]
        vx = (math.cos(ym) * dx + math.sin(ym) * dy) / dt
        wz = dyaw / dt
        cvx, _, cwz = command_at(t_mid)
        err_vx.append(vx - cvx)
        err_wz.append(wz - cwz)
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")  # noqa: E731
    return {"rmse_vx": rms(err_vx), "rmse_wz": rms(err_wz), "n": len(err_vx)}


def fell(samples: list[dict]) -> bool:
    return any(s["gravity"][2] > -0.5 for s in samples)


async def ours(cell_json: Path, out: Path, episodes: int) -> None:
    sys.path.insert(0, str(SKILL))
    import cell
    from viam.components.generic import Generic as GenericComponent
    from viam.services.generic import Generic as GenericService

    results = []
    async with cell.connect(cell_json) as robot:
        world = GenericComponent.from_robot(robot, "sim-world")
        duck = GenericComponent.from_robot(robot, "duck")
        walker = GenericService.from_robot(robot, "walker")
        for ep in range(episodes):
            await world.do_command({"command": "pause"})
            await world.do_command({"command": "reset"})
            await walker.do_command({"command": "set_command", "vx": 0, "vy": 0, "wz": 0})
            before = await world.do_command({"command": "status"})  # reset zeroed slips
            await world.do_command({"command": "play"})
            await walker.do_command({"command": "start"})  # right after play: no ramp at STAND, takes over in a tick
            first = await duck.do_command({"io": {}})
            t0, samples, last_cmd = float(first["t"]), [], None
            while True:
                s = await duck.do_command({"io": {}})
                t = float(s["t"]) - t0
                if t >= DURATION:
                    break
                cmd = command_at(t)
                if cmd != last_cmd:
                    await walker.do_command({"command": "set_command", "vx": cmd[0], "vy": cmd[1], "wz": cmd[2]})
                    last_cmd = cmd
                if not samples or t - samples[-1]["t"] >= 0.1 - 1e-6:
                    samples.append({"t": round(t, 4), "trunk_pos": s["trunk_pos"],
                                    "trunk_quat_wxyz": s["trunk_quat_wxyz"], "gravity": s["gravity"]})
                await asyncio.sleep(0.05)  # 20 Hz is enough for 0.1 s samples and spares the sim thread
            status = await walker.do_command({"command": "status"})
            await walker.do_command({"command": "stop"})
            after = await world.do_command({"command": "status"})
            await world.do_command({"command": "pause"})
            results.append({"episode": ep, "samples": samples, "walker": status,
                            "slips": after.get("slips", 0) - before.get("slips", 0),
                            "real_time_factor": after.get("real_time_factor")})
            reason = status.get("reason")
            reason = reason if isinstance(reason, str) else "none"  # viam-sdk sends a null as int 0
            print(f"episode {ep}: fell={fell(samples)} {tracking(samples)} walker={reason} "
                  f"ticks={status.get('ticks')} p99={status.get('tick_ms_p99')} slips={results[-1]['slips']}")
    out.write_text(json.dumps(results))


def report(ref_path: Path, ours_path: Path, md: Path) -> None:
    ref = json.loads(ref_path.read_text())
    eps = json.loads(ours_path.read_text())
    ref_track, ref_fell = tracking(ref["samples"]), fell(ref["samples"])
    clean = [e for e in eps if e["slips"] == 0]
    tracks = [tracking(e["samples"]) for e in clean]
    ours_falls = sum(fell(e["samples"]) for e in clean)
    ref_falls = len(clean) * int(ref_fell)  # deterministic reference: every episode is the same
    rmse_vx = float(np.mean([t["rmse_vx"] for t in tracks])) if tracks else float("nan")
    rmse_wz = float(np.mean([t["rmse_wz"] for t in tracks])) if tracks else float("nan")
    ticks = sum(e["walker"].get("ticks", 0) for e in clean)
    late = sum(e["walker"].get("late", 0) for e in clean)
    p99 = max((e["walker"].get("tick_ms_p99") or 0) for e in clean) if clean else float("nan")
    run_s = sum(e["walker"].get("run_s") or 0 for e in clean)
    rate = ticks / run_s if run_s else 0.0  # the loop's own run time, not the episode's start/stop overhead
    checks = {
        "slip_free >= 18/20": len(clean) >= 18,
        "falls": ours_falls <= ref_falls + 1,
        "rmse_vx <= 1.25 x ref": rmse_vx <= 1.25 * ref_track["rmse_vx"],
        "rmse_wz <= 1.25 x ref": rmse_wz <= 1.25 * ref_track["rmse_wz"],
        "tick rate >= 49.5 Hz": rate >= 49.5,
        "p99 tick < 20 ms": p99 < 20,
        "late < 1 %": ticks > 0 and late / ticks < 0.01,
    }
    verdict = "machine too slow" if not checks["slip_free >= 18/20"] else ("PASS" if all(checks.values()) else "FAIL")
    lines = [
        f"# MicroDuck walk parity: {verdict}", "",
        f"- Reference: fell={ref_fell}, rmse_vx={ref_track['rmse_vx']:.4f}, rmse_wz={ref_track['rmse_wz']:.4f}",
        f"- Ours ({len(clean)}/{len(eps)} slip-free): falls={ours_falls}, rmse_vx={rmse_vx:.4f}, rmse_wz={rmse_wz:.4f}",
        f"- Timing: {rate:.1f} Hz, p99 tick {p99} ms, late {late}/{ticks}", "",
        "| Check | Result |", "|---|---|",
        *[f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in checks.items()],
    ]
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "ours":
        episodes = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 20
        asyncio.run(ours(Path(sys.argv[2]), Path(sys.argv[3]), episodes))
    elif mode == "report":
        report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
