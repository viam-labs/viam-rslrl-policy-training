"""Export the training scene as files for viam-labs:mujoco-sim: the interactive cell on a Viam machine.

    uv run python -m scripts.export_sim_scene [out_dir] [clearance_mm]   (default dist/sim, 0.35)

Writes, all from the same code the training env is built from:
- xarm6.xml + meshes/: the arm as MJCF (mujoco-sim:arm `mjcf`), with the training's actuators and
  its nominal start pose as the `home` keyframe (a URDF arm would home at all-zero joints, which
  puts the G2 in the floor); xarm6_kinematics.json is viam-ufactory-xarm's, what the real arm serves
- g2.xml + g2 meshes: the G2 gripper (mujoco-sim:gripper `mjcf`), one actuator, 0 = closed
- insert_cell.xml: the fixture at the training's nominal hole mouth (mujoco-sim:scene `mjcf`)
  and a free peg standing on the floor (a prop: the world's set_prop_pose puts it in hand)

Episode reset on the machine (verified on rl-insert, mujoco-sim 0.16.0): world `reset` (arm to
home, peg back to the floor), gripper Open, world `pause`, `set_prop_pose` insert-cell/peg at
(0.4, 0, 0.069) orientation [1, 0, 0, 0], gripper Grab, world `play`. The peg drops ~4.1 mm while
the jaws close, which is what the +4 mm is for: it settles 0.1 mm from the training's tip pose
(protrusion 40 mm, 15 mm above the mouth), with a firm grasp at 4.96 mm a jaw.

The fixture collides with the training's hole meshes (hidden) and shows the printable hole block
(hardware/stl) on a riser, so the mouth is where training put it: TaskParams.hole_mouth.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import mujoco

from rl_insert.peg_hole import PegHoleParams, add_hole, add_peg
from rl_insert.robot import FLANGE, G2_MESHES, HIDDEN, XARM6_URDF, GripperParams, add_gripper, load_xarm6
from rl_insert.task.env_cfg import TaskParams, nominal_joint_pos

ROOT = Path(__file__).resolve().parent.parent
BLOCK_TOP = 0.024  # hole_block's top_height: its mouth above its own base (hardware/parts/hole_block.py)


def arm_spec(p: TaskParams) -> mujoco.MjSpec:
    """The xArm6 as training builds it, homed at the training's start pose."""
    spec = load_xarm6()
    spec.body(FLANGE).add_site(name="attachment_site")  # where mujoco-sim attaches the gripper
    q0 = nominal_joint_pos(p).tolist()
    spec.add_key(name="home", qpos=q0, ctrl=q0)
    return spec


def gripper_spec(g: GripperParams = GripperParams()) -> mujoco.MjSpec:
    """The G2 alone, its root at the flange (where mujoco-sim attaches it)."""
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name=FLANGE)  # add_gripper hangs off a body with the flange's name
    add_gripper(spec, g)
    spec.body(FLANGE).name = "g2_mount"
    return spec


def to_xml(spec: mujoco.MjSpec, out: Path) -> None:
    """Write the spec with mesh paths relative to the file (to_xml needs them resolvable first)."""
    text = spec.to_xml()
    for mesh in spec.meshes:
        if mesh.file:
            text = text.replace(f'"{mesh.file}"', f'"{Path(mesh.file).name}"')
    out.write_text(text)


def cell_spec(p: TaskParams, block_stl: Path) -> mujoco.MjSpec:
    ph = PegHoleParams(clearance=p.clearance)
    spec = mujoco.MjSpec()
    x, y, mouth_z = p.hole_mouth
    riser_h = mouth_z - BLOCK_TOP
    fixture = spec.worldbody.add_body(name="fixture", pos=[x, y, 0])
    fixture.add_geom(name="riser", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, riser_h / 2],
                     pos=[0, 0, riser_h / 2], rgba=[0.35, 0.35, 0.4, 1])
    spec.add_mesh(name="hole_block", file=str(block_stl), scale=[0.001] * 3)  # STL is in mm
    fixture.add_geom(name="hole_block_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="hole_block",
                     pos=[0, 0, riser_h], contype=0, conaffinity=0, rgba=[0.9, 0.55, 0.15, 1])
    hole = add_hole(spec, fixture, ph, pos=(0, 0, mouth_z))  # what the peg touches, as in training
    for i, geom in enumerate(hole.geoms):
        geom.name = geom.name or f"hole_piece_{i}"
        geom.group = HIDDEN
    add_peg(spec, spec.worldbody, ph, pos=(x - 0.1, y + 0.15, 0.0))
    spec.geom("peg_geom").rgba = [0.2, 0.5, 0.9, 1]
    return spec


def export(out: Path, clearance_mm: float = 0.35) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    p = TaskParams(clearance=clearance_mm / 1000)
    shutil.copy(XARM6_URDF.parent / "xarm6_kinematics.json", out)
    shutil.copytree(XARM6_URDF.parent / "meshes", out / "meshes", dirs_exist_ok=True)
    to_xml(arm_spec(p), out / "xarm6.xml")
    for stl in G2_MESHES.glob("*.stl"):
        shutil.copy(stl, out / stl.name)
    to_xml(gripper_spec(), out / "g2.xml")

    block = ROOT / "hardware" / "stl" / f"hole_block_c{clearance_mm:g}".replace(".", "_")
    block = block.with_suffix(".stl")
    shutil.copy(block, out / block.name)
    to_xml(cell_spec(p, block), out / "insert_cell.xml")
    for name in ("xarm6.xml", "g2.xml", "insert_cell.xml"):  # both must load on their own, meshes relative to the file
        mujoco.MjModel.from_xml_path(str(out / name))
    return out


if __name__ == "__main__":
    print(export(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "dist" / "sim",
                 float(sys.argv[2]) if len(sys.argv) > 2 else 0.35))
