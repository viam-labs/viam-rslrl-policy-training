"""Parametric round peg and chamfered hole, built with MjSpec. SI units (m, kg).

Both are convex meshes, not primitives. A bore made of overlapping boxes flings a wedged peg
out of the hole, and on MuJoCo Warp a cylinder primitive is either thrown out when wedged
(multi-CCD on) or sinks 0.5 mm into the floor (multi-CCD off). The hole's pieces tile the ring
exactly. Contacts use a stiff solref: at MuJoCo's default, 20 N drives the peg 10 mm into a wall.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco

# ponytail: 2 x the 1 ms timestep, the stiffest MuJoCo recommends; < 0.07 mm wall penetration at 50 N
CONTACT_SOLREF = (0.002, 1.0)
TIMESTEP = 0.001


@dataclass(frozen=True)
class PegHoleParams:
    peg_radius: float = 0.005
    peg_length: float = 0.04
    peg_mass: float = 0.02
    clearance: float = 0.0005  # radial gap between peg and bore wall
    chamfer: float = 0.001  # 45 degree chamfer leg at the mouth
    depth: float = 0.02
    wall: float = 0.01  # radial wall thickness
    segments: int = 32  # sides of the polygonal bore and peg
    friction: float = 0.5

    @property
    def hole_radius(self) -> float:
        """Bore inradius: the closest the wall comes to the axis."""
        return self.peg_radius + self.clearance


def _ring_piece(r_in_bot, r_in_top, r_out, z_bot, z_top, a0, a1) -> list[float]:
    """8 vertices of a convex ring segment between angles a0..a1.

    Radii are face distances (polygon inradius), so corners sit at r / cos(half-angle) and
    neighbouring pieces share faces exactly, with no overlap.
    """
    k = 1 / math.cos((a1 - a0) / 2)
    verts: list[float] = []
    for z, r_in in ((z_bot, r_in_bot), (z_top, r_in_top)):
        for r in (r_in, r_out):
            for a in (a0, a1):
                verts += [r * k * math.cos(a), r * k * math.sin(a), z]
    return verts


def _geom_kwargs(p: PegHoleParams) -> dict:
    return {"friction": [p.friction, 0.005, 0.0001], "solref": list(CONTACT_SOLREF)}


def add_hole(spec: mujoco.MjSpec, parent: mujoco.MjsBody, p: PegHoleParams, name: str = "hole",
             pos=(0.0, 0.0, 0.0)) -> mujoco.MjsBody:
    """A fixed hole. Its frame is the centre of the mouth, +z out of the surface (insertion is -z)."""
    body = parent.add_body(name=name, pos=list(pos))
    body.add_site(name=f"{name}_mouth", size=[0.001] * 3)
    r_in, r_out, ch, n = p.hole_radius, p.hole_radius + p.wall, p.chamfer, p.segments
    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        for kind, verts in (
            ("wall", _ring_piece(r_in, r_in, r_out, -p.depth, -ch, a0, a1)),
            ("chamfer", _ring_piece(r_in, r_in + ch, r_out, -ch, 0.0, a0, a1)),
        ):
            mesh = f"{name}_{kind}_{i}"
            spec.add_mesh(name=mesh, uservert=verts)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, **_geom_kwargs(p))
    body.add_geom(name=f"{name}_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[r_out, r_out, 0.002],
                  pos=[0, 0, -p.depth - 0.002], **_geom_kwargs(p))
    return body


def add_peg(spec: mujoco.MjSpec, parent: mujoco.MjsBody, p: PegHoleParams, name: str = "peg",
            pos=(0.0, 0.0, 0.0), quat=(1.0, 0.0, 0.0, 0.0)) -> mujoco.MjsBody:
    """A free peg. Its frame is the tip centre, +z along the peg toward the grip."""
    body = parent.add_body(name=name, pos=list(pos), quat=list(quat))
    body.add_freejoint(name=f"{name}_free")
    body.add_site(name=f"{name}_tip", size=[0.001] * 3)
    verts: list[float] = []
    for z in (0.0, p.peg_length):
        for k in range(p.segments):
            a = 2 * math.pi * k / p.segments
            verts += [p.peg_radius * math.cos(a), p.peg_radius * math.sin(a), z]
    spec.add_mesh(name=f"{name}_mesh", uservert=verts)
    body.add_geom(name=f"{name}_geom", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh",
                  mass=p.peg_mass, **_geom_kwargs(p))
    return body


def configure_options(spec: mujoco.MjSpec) -> None:
    spec.option.timestep = TIMESTEP
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10


def peg_hole_scene(p: PegHoleParams) -> mujoco.MjSpec:
    """Peg and hole alone, for contact tests. The peg starts on the hole axis, 5 mm above the mouth."""
    spec = mujoco.MjSpec()
    configure_options(spec)
    add_hole(spec, spec.worldbody, p)
    add_peg(spec, spec.worldbody, p, pos=(0.0, 0.0, 0.005))
    return spec


DRIVEN_AXES = ("x", "y", "z", "rx", "ry")


def driven_peg_scene(p: PegHoleParams) -> mujoco.MjSpec:
    """The peg on a force-limited servo (x, y, z slides, x and y tilts), the way the arm holds it.

    Joints read the absolute tip position and tilt; ctrl is the target. For contact tests: the peg
    is never free in the task.
    """
    spec = peg_hole_scene(p)
    peg = spec.body("peg")
    spec.delete(spec.joint("peg_free"))
    peg.pos = [0, 0, 0]
    for name in DRIVEN_AXES:
        hinge = name.startswith("r")
        axis = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[name[-1]]
        kind = mujoco.mjtJoint.mjJNT_HINGE if hinge else mujoco.mjtJoint.mjJNT_SLIDE
        peg.add_joint(name=name, type=kind, axis=axis)
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT,
                                forcerange=[-0.5, 0.5] if hinge else [-50, 50])
        # omega * dt ~ 0.1 for a 20 g peg; stiffer servos are unstable with explicit integration
        act.set_to_position(kp=0.1 if hinge else 200, dampratio=1.0)
    return spec
