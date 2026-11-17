"""The full insertion scene: xArm6, gripper, a peg in hand and a fixed hole."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .peg_hole import PegHoleParams, add_hole, add_peg, configure_options
from .robot import ArmParams, GripperParams, add_gripper, load_xarm6


@dataclass(frozen=True)
class SceneParams:
    peg_hole: PegHoleParams = field(default_factory=PegHoleParams)
    arm: ArmParams = field(default_factory=ArmParams)
    gripper: GripperParams = field(default_factory=GripperParams)
    hole_pos: tuple[float, float, float] = (0.4, 0.0, 0.0)
    protrusion: float = 0.02  # how far the peg tip sticks out past the fingertip centre


def build_scene(p: SceneParams = SceneParams()) -> mujoco.MjSpec:
    spec = load_xarm6(p.arm)
    configure_options(spec)
    add_gripper(spec, p.gripper)
    add_hole(spec, spec.worldbody, p.peg_hole, pos=p.hole_pos)
    add_peg(spec, spec.worldbody, p.peg_hole)
    return spec


def place_peg_in_hand(m: mujoco.MjModel, d: mujoco.MjData, peg_radius: float, protrusion: float,
                      offset=(0.0, 0.0, 0.0)) -> None:
    """Put the peg between the pads, its tip `protrusion` past the TCP along the TCP z axis.

    `offset` (m, in the TCP frame) shifts the grasp, for randomising it. Opens the fingers to
    the peg's radius so they start touching it. Call after setting the arm's qpos.
    """
    for side in ("finger_left", "finger_right"):
        d.qpos[m.jnt_qposadr[m.joint(side).id]] = peg_radius
    mujoco.mj_forward(m, d)
    tcp = m.site("tcp").id
    rot = d.site_xmat[tcp].reshape(3, 3)
    tip = d.site_xpos[tcp] + rot @ (np.array([0.0, 0.0, protrusion]) + np.asarray(offset))
    flip = rot @ np.diag([1.0, -1.0, -1.0])  # peg +z points back toward the gripper
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, flip.ravel())
    adr = m.jnt_qposadr[m.joint("peg_free").id]
    d.qpos[adr:adr + 3] = tip
    d.qpos[adr + 3:adr + 7] = quat
    d.qvel[m.jnt_dofadr[m.joint("peg_free").id]:][:6] = 0
    mujoco.mj_forward(m, d)
