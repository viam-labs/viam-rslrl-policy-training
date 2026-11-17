"""The xArm6 (from viam-ufactory-xarm's URDF, the model Viam's frame system uses) plus a
parametric two-finger gripper, as an MjSpec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

ASSETS = Path(__file__).resolve().parent / "assets"  # inside the package so pip installs it
XARM6_URDF = ASSETS / "xarm6" / "xarm6.urdf"
ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 7))
FLANGE = "link_eef"


@dataclass(frozen=True)
class ArmParams:
    # ponytail: stiff PD standing in for the xArm's position servo; tune against logged real tracking
    kp: float = 2000.0
    dampratio: float = 1.0


@dataclass(frozen=True)
class GripperParams:
    """UFactory xArm Gripper G2 (AG1200), measured from its STEP file.

    Mount face (flange contact) to pad centre is 156.6 mm; the fingertips end 176.3 mm out.
    It seats flat on the mount face (confirmed on the real arm), not on the 5 mm locating spigot.
    """
    tcp: float = 0.1566  # flange to the pad centre, along the flange z axis
    finger_length: float = 0.0375  # pad length along the gripper axis
    pad_width: float = 0.0295
    max_opening: float = 0.084  # full stroke, both fingers (84 +/- 1 mm)
    # ponytail: spec range is 10-50 N; plan 2 randomises within it
    grip_force: float = 30.0
    pad_friction: float = 1.0


def load_xarm6(arm: ArmParams = ArmParams()) -> mujoco.MjSpec:
    """The xArm6 at the world origin, with one position actuator per joint (rad)."""
    spec = mujoco.MjSpec.from_file(str(XARM6_URDF))
    for body in spec.bodies[1:]:  # skip world
        body.gravcomp = 1.0  # the real xArm servo holds pose; without this the TCP sags ~9 mm
    for name in ARM_JOINTS:
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=arm.kp, dampratio=arm.dampratio, inheritrange=True)
    return spec


def add_gripper(spec: mujoco.MjSpec, g: GripperParams = GripperParams()) -> mujoco.MjsBody:
    """Two fingers sliding along the flange y axis, coupled by an equality, closed by one actuator.

    Control is each finger's travel from centre in metres (0 closed, max_opening / 2 fully open).
    The actuator is stiff enough to saturate, so a closed command squeezes with grip_force.
    """
    flange = spec.body(FLANGE)
    root = flange.add_body(name="gripper", pos=[0, 0, g.tcp - g.finger_length], gravcomp=1.0)  # see load_xarm6
    root.add_site(name="tcp", pos=[0, 0, g.finger_length], size=[0.002] * 3)
    root.add_geom(name="gripper_palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.005],
                  pos=[0, 0, -0.005], mass=0.5)
    half = g.max_opening / 2
    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = root.add_body(name=f"finger_{side}", gravcomp=1.0)
        finger.add_joint(name=f"finger_{side}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, sign, 0],
                         range=[0, half])
        finger.add_geom(name=f"pad_{side}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[g.pad_width / 2, 0.003, g.finger_length / 2],
                        pos=[0, sign * 0.003, g.finger_length / 2], mass=0.05,
                        friction=[g.pad_friction, 0.005, 0.0001], solref=[0.002, 1.0])
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="finger_left", name2="finger_right",
                      solref=[0.002, 1.0])  # the default soft coupling gives 6 mm under grip force
    act = spec.add_actuator(name="gripper", target="finger_left", trntype=mujoco.mjtTrn.mjTRN_JOINT,
                            ctrlrange=[0, half], forcerange=[-g.grip_force, g.grip_force])
    act.set_to_position(kp=1e4, dampratio=1.0)  # saturates at grip_force once a pad touches the peg
    return root
