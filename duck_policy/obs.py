"""The 61-value observation Pollen's MicroDuck policies take (microduck_rl@0daab16 infer_policy.py:656-682,
duck-control obs.rs): gyro(3), projected gravity(3), joint positions minus the default pose(14), joint
velocities(14), the last raw action(14), command(13) = twist(3) + head pose(4) + body pose(6). No scaling,
noise, history or clock. Pure numpy: shared by the Viam service and the golden test."""

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

    def vector(self) -> np.ndarray:
        cmd = np.zeros(13, np.float32)
        cmd[0:3] = (self.vx, self.vy, self.wz)  # head (3:7) and body (7:13) pose: zero for a walk
        return cmd


def build_obs(*, gyro, gravity, q, dq, default_pose, last_action, command: Command) -> np.ndarray:
    return np.concatenate([
        np.asarray(gyro, np.float32),
        np.asarray(gravity, np.float32),
        np.asarray(q, np.float32) - np.asarray(default_pose, np.float32),
        np.asarray(dq, np.float32),
        np.asarray(last_action, np.float32),
        command.vector(),
    ]).astype(np.float32)
