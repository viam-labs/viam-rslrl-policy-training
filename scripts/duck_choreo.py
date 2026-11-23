# /// script
# dependencies = ["viam-sdk>=0.85", "numpy", "mujoco"]
# ///
"""Choreograph a MicroDuck dance on the choreo machine (the pinned duck, spec Part 1).

    uv run scripts/duck_choreo.py pose    <cell.json> <dance.json> --t T      # pose the duck, then capture it
    uv run scripts/duck_choreo.py preview <cell.json> <dance.json> [--cycles 2]
    uv run scripts/duck_choreo.py check   <dance.json>
    uv run scripts/duck_choreo.py record  <cell.json> <dance.json> --part-id ID [--cycles 3] [--sync-wait 90]

The arms are left-leg (5 joints), head (4) and right-leg (5), in policy order. `pose` reads measured
positions: gravity sag of ~0.05 rad on the hip rolls and head pitch shows up in captured keyframes.
`record` stops at creating the sequence; the dataset steps are the controller's (org-level access).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from duck_train.dance import STAND, interpolate, limits_of, load_dance, foot_check, validate  # noqa: E402
from duck_train.robot import JOINTS  # noqa: E402

SKILL = Path("/Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell/scripts")
ARMS = {"left-leg": slice(0, 5), "head": slice(5, 9), "right-leg": slice(9, 14)}
DEFAULT_XML = ROOT / "dist/duck-sim/duck.xml"
DT = 0.02
KEEP_TOL = 0.01  # rad: joints closer than this to STAND are left out of a keyframe


def split_arms(q) -> dict:
    q = np.asarray(q, float)
    return {name: q[s] for name, s in ARMS.items()}


def join_arms(parts: dict) -> np.ndarray:
    return np.concatenate([np.asarray(parts[name], float) for name in ARMS])


def upsert_keyframe(dance: dict, t: float, q) -> dict:
    t = round(float(t), 4)
    pose = {j: round(float(v), 4) for j, v, s in zip(JOINTS, q, STAND) if abs(v - s) > KEEP_TOL}
    kfs = [k for k in dance["keyframes"] if round(float(k["t"]), 4) != t]
    kfs.append({"t": t, "pose": pose})
    dance["keyframes"] = sorted(kfs, key=lambda k: k["t"])
    return validate(dance)


def _arm(robot, name):
    from viam.components.arm import Arm

    return Arm.from_robot(robot, name)


async def read_q(robot) -> np.ndarray:
    rs = await asyncio.gather(*(_arm(robot, n).do_command({"command": "get_joint_positions_radians"}) for n in ARMS))
    return join_arms({n: r["values"] for n, r in zip(ARMS, rs)})


async def write_q(robot, q) -> None:
    await asyncio.gather(*(
        _arm(robot, n).do_command({"command": "set_joint_targets_radians", "values": [float(v) for v in vals]})
        for n, vals in split_arms(q).items()))


def connect(cell_json: Path):
    sys.path.insert(0, str(SKILL))
    import cell
    import viam.components.arm  # noqa: F401  registers the arm API, or the client builds no arm clients on connect
    return cell.connect(cell_json)


async def play(robot, dance, limits, cycles: float) -> None:
    period = dance["period_s"]
    t0 = time.monotonic()
    k = 0
    try:
        while (now := time.monotonic() - t0) < cycles * period:
            await write_q(robot, interpolate(dance, (now / period) % 1.0, limits))
            k = max(k + 1, int((time.monotonic() - t0) / DT) + 1)  # after a stall, resync instead of bursting
            await asyncio.sleep(max(0.0, t0 + k * DT - time.monotonic()))
    finally:
        await write_q(robot, STAND)


def check(args) -> int:
    try:
        dance = load_dance(args.dance)
        errors = foot_check(dance, args.duck_xml)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    for e in errors:
        print(e)
    print("ok" if not errors else f"{len(errors)} foot check error(s)")
    return 1 if errors else 0


async def pose(args) -> int:
    if args.dance.exists():
        dance = json.loads(args.dance.read_text())
    else:
        dance = {"name": args.dance.stem, "period_s": 6.0, "keyframes": [{"t": 0.0, "pose": {}}]}
    async with connect(args.cell) as robot:
        q = await read_q(robot)
    try:
        dance = upsert_keyframe(dance, args.t, q)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    args.dance.parent.mkdir(parents=True, exist_ok=True)
    args.dance.write_text(json.dumps(dance, indent=2) + "\n")
    print(json.dumps(next(k for k in dance["keyframes"] if k["t"] == round(args.t, 4))))
    return 0


async def preview(args) -> int:
    dance = load_dance(args.dance)
    async with connect(args.cell) as robot:
        await play(robot, dance, limits_of(args.duck_xml), args.cycles)
    return 0


async def record(args) -> int:
    if check(args):
        return 1
    from viam.app.viam_client import ViamClient
    from viam.proto.app.data import SequenceResourceFilter
    from viam.rpc.dial import DialOptions

    dance = load_dance(args.dance)
    limits = limits_of(args.duck_xml)
    cell = json.loads(args.cell.read_text())
    key, key_id = cell["key"], cell["key_id"]  # fail before the duck moves
    async with connect(args.cell) as robot:
        await write_q(robot, STAND)
        await asyncio.sleep(1.0)
        start = datetime.now(timezone.utc)
        await play(robot, dance, limits, args.cycles)
    end = start + timedelta(seconds=args.cycles * dance["period_s"])
    print(f"window {start.isoformat()} .. {end.isoformat()}, resources {list(ARMS)} (JointPositions)")
    print(f"waiting {args.sync_wait} s for data sync")
    await asyncio.sleep(args.sync_wait)
    client = await ViamClient.create_from_dial_options(DialOptions.with_api_key(key, key_id))
    try:
        sid = await client.data_client.create_sequence(
            args.part_id,
            resources=[SequenceResourceFilter(resource_name=n, method_name="JointPositions") for n in ARMS],
            start_time=start, end_time=end,
            sequence_tags=[f"dance:{dance['name']}", f"period:{dance['period_s']}"])
    finally:
        client.close()
    print(sid)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("pose", "preview", "check", "record"):
        s = sub.add_parser(name)
        if name != "check":
            s.add_argument("cell", type=Path)
        s.add_argument("dance", type=Path)
        s.add_argument("--duck-xml", type=Path, default=DEFAULT_XML)
        if name == "pose":
            s.add_argument("--t", type=float, required=True)
        if name in ("preview", "record"):
            s.add_argument("--cycles", type=float, default=2 if name == "preview" else 3)
        if name == "record":
            s.add_argument("--part-id", required=True)
            s.add_argument("--sync-wait", type=float, default=90)
    args = p.parse_args()
    if args.cmd == "check":
        return check(args)
    return asyncio.run({"pose": pose, "preview": preview, "record": record}[args.cmd](args))


if __name__ == "__main__":
    sys.exit(main())
