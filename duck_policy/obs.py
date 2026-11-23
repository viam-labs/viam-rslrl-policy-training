"""The 61-value observation Pollen's MicroDuck policies take (microduck_rl@0daab16 infer_policy.py:656-682,
duck-control obs.rs): gyro(3), projected gravity(3), joint positions minus the default pose(14), joint
velocities(14), the last raw action(14), command(13) = twist(3) + head pose(4) + body pose(6), where the twist slot holds either
a velocity twist or, for a dance, the phase clock (cos 2*pi*phi, sin 2*pi*phi, 0). No scaling,
noise, history or clock. Pure numpy: shared by the Viam service and the golden test.

That layout is the default. A package can declare its own in config.json as
`observation: {"dim": total, "terms": [{"name": ..., "dim": ...}, ...]}` (the shape policy_controller
uses), concatenated in order, with names from TERMS. The 0.1.0 packages carry a descriptive string
there instead, which means the default."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Command:
    vx: float = 0.0  # m/s forward
    vy: float = 0.0  # m/s left
    wz: float = 0.0  # rad/s yaw

    def clipped(self, lo: Sequence[float], hi: Sequence[float]) -> "Command":
        return Command(*np.clip([self.vx, self.vy, self.wz], lo, hi).tolist())

    def vector(self, size: int = 13) -> np.ndarray:
        cmd = np.zeros(size, np.float32)
        cmd[0:3] = (self.vx, self.vy, self.wz)  # the rest (MicroDuck: head 3:7, body 7:13 pose) is zero for a walk
        return cmd


TERMS = ("gyro", "gravity", "q_rel", "dq", "last_action", "command")  # q_rel = q - default_pose
DEFAULT_LAYOUT = [("gyro", 3), ("gravity", 3), ("q_rel", 14), ("dq", 14), ("last_action", 14), ("command", 13)]


def obs_layout(cfg: dict) -> list[tuple[str, int]]:
    """The package's observation layout, checked: known terms, positive sizes, room for a twist in
    command, and a total that matches observation.dim and model.obs_len when the package states them."""
    spec = cfg.get("observation")
    if not isinstance(spec, dict):
        return list(DEFAULT_LAYOUT)
    layout = [(str(t["name"]), int(t["dim"])) for t in spec["terms"]]
    for term, n in layout:
        if term not in TERMS:
            raise ValueError(f"observation term {term!r} isn't one of {TERMS}")
        if n < (3 if term == "command" else 1):
            raise ValueError(f"observation term {term!r} has dim {n}")
    total = sum(n for _, n in layout)
    for where, want in (("observation.dim", spec.get("dim")), ("model.obs_len", (cfg.get("model") or {}).get("obs_len"))):
        if want is not None and total != want:
            raise ValueError(f"observation terms add up to {total}, {where} is {want}")
    return layout


def build_obs(*, gyro, gravity, q, dq, default_pose, last_action, command, layout=DEFAULT_LAYOUT) -> np.ndarray:
    values = {"gyro": gyro, "gravity": gravity, "q_rel": np.asarray(q, np.float32) - np.asarray(default_pose, np.float32),
              "dq": dq, "last_action": last_action, "command": command}
    parts = []
    for term, n in layout:
        v = np.asarray(values[term], np.float32)
        if v.shape != (n,):
            raise ValueError(f"observation term {term!r}: got {v.shape[0] if v.ndim else 0} values, the layout says {n}")
        parts.append(v)
    return np.concatenate(parts)


def phase_command(phi: float, size: int = 13) -> np.ndarray:
    cmd = np.zeros(size, np.float32)
    cmd[0:2] = (np.cos(2 * np.pi * phi), np.sin(2 * np.pi * phi))
    return cmd
