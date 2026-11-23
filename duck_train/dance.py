"""MicroDuck dances (spec: docs/superpowers/specs/2026-11-21-microduck-dance-design.md). A dance is a looping
keyframe file; interpolate() turns it into 14 joint angles in the policy order at any phase in [0, 1). The
choreo tool, the reference builder's tests and the evaluator share this module. numpy only at import time:
MuJoCo and pandas are imported where they're used, so the training image and the Mac both load it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .robot import DEFAULT_POSE, JOINTS, TRUNK_Z

STAND = np.asarray(DEFAULT_POSE, float)
STAND_TOL = 0.05  # rad: phase 0 must be STAND (start and hold pose)


def validate(dance: dict) -> dict:
    if not isinstance(dance.get("name"), str) or not dance["name"]:
        raise ValueError("dance needs a name")
    period = float(dance.get("period_s", 0))
    if not 4.0 <= period <= 8.0:
        raise ValueError(f"period_s {period} outside 4-8 s")
    kfs = dance.get("keyframes") or []
    try:
        ts = [float(k["t"]) for k in kfs]
        for k in kfs:
            pose = k.get("pose", {})
            if not isinstance(pose, dict) or not all(np.isfinite(float(v)) for v in pose.values()):
                raise ValueError("pose must map joint names to finite numbers")
    except (KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"malformed keyframe: {e!r}") from e
    if not ts or ts[0] != 0.0 or any(b <= a for a, b in zip(ts, ts[1:])) or ts[-1] >= 1.0:
        raise ValueError(f"keyframe t must start at 0, strictly increase and stay below 1: {ts}")
    for k in kfs:
        unknown = set(k.get("pose", {})) - set(JOINTS)
        if unknown:
            raise ValueError(f"unknown joints {sorted(unknown)}")
    if np.abs(_pose_vector(kfs[0].get("pose", {})) - STAND).max() > STAND_TOL:
        raise ValueError("the t=0 keyframe must be STAND within 0.05 rad")
    return dance


def load_dance(path: Path) -> dict:
    return validate(json.loads(Path(path).read_text()))


def _pose_vector(pose: dict) -> np.ndarray:
    q = STAND.copy()
    for name, value in pose.items():
        q[JOINTS.index(name)] = float(value)
    return q


def interpolate(dance: dict, phase: float, limits: np.ndarray | None = None) -> np.ndarray:
    """Periodic cubic Hermite (Catmull-Rom tangents over the wrapped keyframes, non-uniform spacing), per joint.
    Passes through every keyframe; wraps from the last keyframe back to the first at t = 1.
    Callers that command joints must pass limits (limits_of(duck_xml)); without them the Hermite overshoot is unclipped."""
    kfs = dance["keyframes"]
    ts = np.array([float(k["t"]) for k in kfs])
    ps = np.stack([_pose_vector(k.get("pose", {})) for k in kfs])
    n = len(ts)
    phase = float(phase) % 1.0
    if n == 1:
        q = ps[0].copy()
    else:
        def t_of(k):  # keyframe k's time, unwrapped across the loop (k may be -1, n or n + 1)
            return ts[k % n] + (k // n)

        def tangent(k):
            return (ps[(k + 1) % n] - ps[(k - 1) % n]) / (t_of(k + 1) - t_of(k - 1))

        i = int(np.searchsorted(ts, phase, side="right")) - 1  # segment keyframe i -> i + 1
        t0, t1 = t_of(i), t_of(i + 1)
        h = t1 - t0
        s = (phase - t0) / h
        h00, h10, h01, h11 = 2*s**3 - 3*s**2 + 1, s**3 - 2*s**2 + s, -2*s**3 + 3*s**2, s**3 - s**2
        q = h00 * ps[i] + h10 * h * tangent(i) + h01 * ps[(i + 1) % n] + h11 * h * tangent(i + 1)
    if limits is not None:
        q = np.clip(q, limits[:, 0], limits[:, 1])
    return q


def table_at(table: np.ndarray, phase: float) -> np.ndarray:
    """Linear interpolation between bin centres ((k + 0.5) / bins), wrapping. Same rule as the torch reward."""
    bins = len(table)
    x = (float(phase) % 1.0) * bins - 0.5
    i0 = int(np.floor(x))
    w = x - i0
    return (1 - w) * table[i0 % bins] + w * table[(i0 + 1) % bins]


FREE_JOINT = "trunk_base_freejoint"
FOOT_SITES = ("left_foot", "right_foot")
_MODELS: dict = {}  # path -> (model, scratch data). The cached MjData belongs to feet_world; other callers make their own mujoco.MjData(m).


def _model(duck_xml: Path):
    import mujoco

    key = str(Path(duck_xml).resolve())
    if key not in _MODELS:
        m = mujoco.MjModel.from_xml_path(key)
        _MODELS[key] = (m, mujoco.MjData(m))
    return _MODELS[key]


def limits_of(duck_xml: Path) -> np.ndarray:
    """(14, 2) joint ranges in policy order."""
    m, _ = _model(duck_xml)
    return np.array([m.jnt_range[m.joint(n).id] for n in JOINTS])


def feet_world(duck_xml: Path, q, trunk_pos, trunk_quat) -> np.ndarray:
    """World positions (2, 3) of the left and right foot sites for joint angles q (policy order) and a trunk pose.
    Kinematics only (mj_kinematics): no physics, no contacts."""
    import mujoco

    m, d = _model(duck_xml)
    free = m.joint(FREE_JOINT)
    a = m.jnt_qposadr[free.id]
    d.qpos[:] = 0
    d.qpos[a:a + 3], d.qpos[a + 3:a + 7] = trunk_pos, trunk_quat
    for name, value in zip(JOINTS, q):
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = value
    mujoco.mj_kinematics(m, d)
    return np.stack([d.site(s).xpos.copy() for s in FOOT_SITES])


def foot_check(dance: dict, duck_xml: Path, samples: int = 200, lift_m: float = 0.01, slide_m: float = 0.02) -> list[str]:
    """Spec Part 1 'Foot check'. The trunk stays upright at the origin; each phase's lower foot defines the floor.
    Slide = per foot, the largest xy distance between any two samples over the loop (full range, not drift from
    phase 0). Returns human-readable errors (empty = pass); joint-limit errors are one line per joint."""
    limits = limits_of(duck_xml)
    phases = np.arange(samples) / samples
    qs = np.stack([interpolate(dance, p) for p in phases])
    errors = []
    over = np.maximum(limits[:, 0] - qs, qs - limits[:, 1])  # (samples, 14), > 1e-9 = outside
    for j in np.flatnonzero(over.max(axis=0) > 1e-9):
        k = int(over[:, j].argmax())
        errors.append(f"{JOINTS[j]} outside joint limits by {over[k, j]:.3f} rad at phase {phases[k]:.3f}")
    feet = np.stack([feet_world(duck_xml, q, (0, 0, TRUNK_Z), (1, 0, 0, 0)) for q in qs])  # (samples, 2, 3)
    lift = feet[:, :, 2].max(axis=1) - feet[:, :, 2].min(axis=1)
    if lift.max() > lift_m:
        errors.append(f"a foot lifts {100 * lift.max():.1f} cm at phase {phases[lift.argmax()]:.3f} (max {100 * lift_m:.0f} cm)")
    xy = feet[:, :, :2]
    slide = np.linalg.norm(xy[:, None] - xy[None], axis=-1).max(axis=(0, 1))  # (2,) per foot
    if slide.max() > slide_m:
        errors.append(f"a foot slides {100 * slide.max():.1f} cm over the loop (max {100 * slide_m:.0f} cm)")
    return errors


ARM_SLICES = {"left-leg": slice(0, 5), "head": slice(5, 9), "right-leg": slice(9, 14)}
SPREAD_TOL = 0.05  # rad: max std-dev of a bin across cycles


def joint_values(payload: str) -> list[float]:
    """Degrees from a real JointPositions payload: {"positions": {"values": [...]}}."""
    p = json.loads(payload)
    try:
        return [float(v) for v in p["positions"]["values"]]
    except (KeyError, TypeError) as e:
        raise ValueError(f"unexpected JointPositions payload, top-level keys {sorted(p) if isinstance(p, dict) else type(p)}") from e


def build_reference(binary_file, tabular_file, sequences_file, dance: str, bins: int = 100) -> tuple[float, np.ndarray]:
    """Phase-binned (bins, 14) radian table from a sequence-dataset Parquet export. binary_file is unused (no images)."""
    import pandas as pd

    seqs = pd.read_parquet(sequences_file)
    seqs = seqs[[f"dance:{dance}" in list(t) for t in seqs["tags"]]]
    if seqs.empty:
        raise ValueError(f"no sequence tagged dance:{dance}")
    periods = {float(t.split(":", 1)[1]) for tags in seqs["tags"] for t in tags if t.startswith("period:")}
    if not periods:
        raise ValueError("no period tag")
    if len(periods) > 1:
        raise ValueError(f"mixed periods {sorted(periods)}")
    period = periods.pop()
    tab = pd.read_parquet(tabular_file)
    tab = tab[tab["sequence_id"].isin(seqs["sequence_id"]) & tab["method_name"].str.contains("JointPositions")]
    total, sumsq, count = (np.zeros((bins, 14)), np.zeros((bins, 14)), np.zeros(bins))
    for s in seqs.itertuples():
        start, end = s.start_at.timestamp(), s.end_at.timestamp()
        rel = (np.arange(int(round((end - start) / 0.02))) + 0.5) * 0.02  # half-step samples: none lands on a bin edge
        grid = start + rel
        rows = tab[tab["sequence_id"] == s.sequence_id]
        q = np.zeros((len(grid), 14))
        for arm, sl in ARM_SLICES.items():
            r = rows[rows["component_name"].str.endswith(arm)].sort_values("timestamp")
            if r.empty:
                raise ValueError(f"no JointPositions rows for {arm} in sequence {s.sequence_id}")
            ts = np.array([x.timestamp() for x in r["timestamp"]])
            vals = np.array([joint_values(p) for p in r["payload"]])
            q[:, sl] = np.radians(np.stack([np.interp(grid, ts, vals[:, j]) for j in range(vals.shape[1])], axis=1))
        idx = np.minimum(((rel / period) % 1 * bins).astype(int), bins - 1)
        np.add.at(total, idx, q)
        np.add.at(sumsq, idx, q * q)
        np.add.at(count, idx, 1)
    if (count == 0).any():
        raise ValueError(f"empty bin {int(np.flatnonzero(count == 0)[0])}")
    table = total / count[:, None]
    spread = np.sqrt(np.maximum(sumsq / count[:, None] - table**2, 0))
    # ponytail: spread includes the dance's own motion inside a bin (~0.014 rad on bob-sway), so it falsely fails once a joint
    # moves faster than ~17 rad per phase unit; upgrade: second pass, std of q - table_at(table, phase) per bin.
    if spread.max() > SPREAD_TOL:
        raise ValueError(f"spread {spread.max():.3f} rad across cycles")
    if np.abs(table[0] - STAND).max() > STAND_TOL:
        raise ValueError("bin 0 isn't STAND")
    return period, table
