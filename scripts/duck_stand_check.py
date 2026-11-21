"""The MicroDuck in viam-labs:mujoco-sim with BAM: does it load, home at STAND, get driven by BAM,
and does io report the policy's joint order? Runs mujoco-sim's handles in-process, no viam-server.
Run with mujoco-sim's own environment:

    uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating python scripts/duck_stand_check.py dist/duck-sim [mujoco-sim src dir]

The source dir defaults to $MUJOCO_SIM_SRC, else the floating-base-bam worktree (viam-mujoco-sim
PR #7, unreleased).

With no policy the duck tips forward in about a second, and the walk model has no trunk/head
collision, so it then sinks through the floor: the check only asserts the first 0.3 s and prints
the 1 s state. A policy keeps it up (milestone 2).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_SRC = "/Users/nick.hehr/src/viam-mujoco-sim-floating/src"
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else os.environ.get("MUJOCO_SIM_SRC", DEFAULT_SRC))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from mujoco_module.handles import ArmHandle  # noqa: E402
from mujoco_module.robot_file import RobotHandle  # noqa: E402
from mujoco_module.sim import SimManager, WorldConfig  # noqa: E402

# The policies' joint order (microduck_rl README "Joint layout"): left leg, neck/head, right leg
POLICY_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# All three limbs hang off trunk_base, so every frame is the identity.
LIMBS = {
    "left-leg": (POLICY_ORDER[0:5], "left_foot"),
    "head": (POLICY_ORDER[5:9], "head_camera"),
    "right-leg": (POLICY_ORDER[9:14], "right_foot"),
}


def main(package: Path) -> None:
    sim = SimManager()
    sim.bind_current_thread()
    sim.configure_world(WorldConfig.from_attrs({"physics_dt": 0.005}))
    duck = RobotHandle(sim, "duck", {"mjcf": str(package / "duck.xml"), "actuators": {"model": "bam"}})
    snap = duck.io(None, None)
    assert snap["joints"] == POLICY_ORDER, snap["joints"]
    print("io joint order matches the policy's")
    for name, (joints, site) in LIMBS.items():
        ArmHandle(sim, name, {"robot": "duck", "parent": "duck", "joints": joints,
                              "flange_site": site, "home_keyframe": "STAND"})
    stand = mujoco.MjModel.from_xml_path(str(package / "duck.xml")).key("STAND").qpos[7:]
    snap = duck.io(None, None)
    assert abs(snap["trunk_pos"][2] - 0.12) < 1e-6, snap["trunk_pos"]
    assert np.allclose(snap["q"], stand, atol=1e-6), (snap["q"], stand)
    print("homed at STAND, trunk z 0.12")
    sim.advance(0.05)
    assert np.abs(sim.data.ctrl).max() > 0, "BAM is not driving"
    print("BAM drives the servos")
    sim.advance(0.25)
    snap = duck.io(None, None)
    assert snap["gravity"][2] < -0.98, snap["gravity"]
    print(f"upright at 0.3 s, gravity {np.round(snap['gravity'], 3).tolist()}")
    sim.advance(0.7)
    snap = duck.io(None, None)
    print(f"at 1 s (info only): trunk z {snap['trunk_pos'][2]:.3f} m, gravity {np.round(snap['gravity'], 3).tolist()}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
