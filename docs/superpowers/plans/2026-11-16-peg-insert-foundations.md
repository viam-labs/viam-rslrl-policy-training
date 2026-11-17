# Peg-insert foundations and contact spike: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested MuJoCo model of the xArm6, its gripper and a parametric peg and hole, plus proof that MuJoCo Warp simulates the peg-in-hole contact correctly on the cloud sim machine's GPU.

**Architecture:** The `rl_insert` package builds everything with `mujoco.MjSpec`:
- the arm comes from viam-ufactory-xarm's `xarm6.urdf`, the model Viam's frame system uses;
- the gripper and the peg and hole are generated from parameters.

CPU MuJoCo tests pin down geometry and contact behaviour. `spikes/contact_warp.py` replays the contact cases batched on MuJoCo Warp and compares them against CPU MuJoCo. `scripts/probe_gpu.sh` runs that spike inside the training container on the GPU machine.

**Tech Stack:** Python 3.12, uv, `mujoco-warp~=4.0.1` (brings `mujoco` 4.0.1 core bindings and `warp-lang`), numpy, pytest, Docker (linux/amd64).

**Spec:** `docs/superpowers/specs/2026-11-16-peg-insert-rl-design.md`. This is plan 1 of 4: foundations and spikes. The later plans are the training task and container, the controller module, and the sim-to-sim cell.

**Deferred from spec spike 3 (xArm6 model):**
- Loading this model in mjlab happens in plan 2 (training).
- Loading it in viam-mujoco-sim happens in plan 4 (sim-to-sim).
- `GripperParams` is the UFactory Gripper G2 (AG1200), measured from its STEP file:
  - 156.6 mm from the mount face to the pad centre (+5 mm if the flange seats on the locating spigot; confirm on the real arm);
  - 37.5 × 29.5 mm pads;
  - 84 mm stroke;
  - 10–50 N grip force (30 N default; plan 2 randomises it).

---

## What prototyping already established

Every file below was prototyped and its tests run on macOS (CPU) before this plan was written. The amd64 image was also built and run under emulation. These findings shaped the code; don't "simplify" them away:

1. **The hole is convex prism meshes that tile the ring, not boxes.**
   - A bore made of overlapping boxes flung a wedged peg out at 6 m/s.
   - Prisms whose faces meet exactly don't.
2. **The peg is a convex 32-sided mesh, not a `cylinder` primitive.**
   - On MuJoCo Warp a cylinder primitive is either thrown out when wedged (multi-CCD on) or sinks 0.5 mm into the floor (multi-CCD off).
   - The mesh peg matches CPU MuJoCo.
3. **Peg, hole and pad contacts use `solref=[0.002, 1]` at a 1 ms timestep.**
   - At MuJoCo's default (0.02), a 20 N push drove the peg 10 mm into the wall.
   - At 0.002, 50 N moves it under 0.07 mm.
4. **Contact tests drive the peg like the arm does, not by dropping it.**
   - A free-falling peg wedged at speed can be ejected even with stiff contacts. In the task the peg is always held, so the tests drive it with a force-limited servo (`driven_peg_scene`).
   - That servo must stay soft: ω·dt ≈ 0.1. A stiffer servo makes the integrator blow up before contact matters.
5. **The finger equality needs `solref=[0.002, 1]`.** At the default it gives way 6 mm under grip force and the peg slides sideways.
6. **The URDF needs `fusestatic="false"`,** or MuJoCo fuses `link_eef` (the flange) into `link6`. Mesh paths must be bare filenames with `meshdir="meshes"`.
7. **In MuJoCo 4, multi-CCD is on by default.** It's now a *disable* flag (`mjDSBL_MULTICCD`).
8. **Warp versus CPU, for pegs balanced on the rim.**
   - Starts 1.4–2.4 mm past the chamfer can snap in on CPU and jam on Warp (float32 vs float64).
   - Per-world agreement on those starts is noisy, so the spike gates on insertion *rates* (within 10 points) and on zero blow-ups.
   - Warp errs toward jamming, the conservative direction for training.

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock`, `.gitignore` | Project and pinned deps |
| `assets/xarm6/xarm6.urdf`, `assets/xarm6/meshes/*.stl`, `assets/xarm6/NOTICE` | xArm6 model copied from viam-ufactory-xarm (Apache-2.0), patched for MuJoCo |
| `rl_insert/__init__.py` | Package marker |
| `rl_insert/robot.py` | `load_xarm6()`: arm and joint actuators; `add_gripper()`: parametric two-finger gripper with a `tcp` site |
| `rl_insert/peg_hole.py` | `PegHoleParams`, `add_hole()`, `add_peg()`, `peg_hole_scene()`, `driven_peg_scene()`, physics options |
| `rl_insert/scene.py` | `SceneParams`, `build_scene()`: arm, gripper, hole and peg; `place_peg_in_hand()` |
| `tests/test_robot.py`, `tests/test_peg_hole.py`, `tests/test_scene.py` | CPU MuJoCo tests |
| `spikes/contact_warp.py` | Warp-versus-CPU contact spike with a JSON report and exit-code gate |
| `container/Dockerfile` | The training image (spike only for now; plan 2 adds mjlab) |
| `scripts/probe_gpu.sh` | GPU and Docker probe for the cloud sim machine |
| `docs/superpowers/specs/2026-11-16-contact-spike-results.md` | Results of the L40S run (Task 8) |

Conventions: SI units throughout (m, kg, rad). The hole frame is the centre of the mouth with +z out of the surface, so insertion is −z. The peg frame is the tip centre, with +z pointing back toward the grip.

---

### Task 1: Project setup

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `rl_insert/__init__.py`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "viam-rl-insert"
version = "0.1.0"
requires-python = ">=3.12,<3.14"
dependencies = ["mujoco-warp~=4.0.1", "numpy>=1.26"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["rl_insert"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

The hatchling build backend makes `uv sync` install `rl_insert` (editable), so tests and `uv run python spikes/...` can import it.

- [ ] **Step 2: Write `.gitignore`**

```
.venv/
__pycache__/
.pytest_cache/
MUJOCO_LOG.TXT
```

- [ ] **Step 3: Create the empty package marker**

```bash
mkdir -p rl_insert tests spikes container scripts assets/xarm6
touch rl_insert/__init__.py
```

- [ ] **Step 4: Lock and install**

Run: `uv lock && uv sync`
Expected: `uv.lock` created; `uv run python -c "import mujoco, mujoco_warp; print(mujoco.__version__)"` prints `4.0.1`.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .gitignore rl_insert/__init__.py
git commit -m "chore: project skeleton with mujoco-warp"
```

---

### Task 2: xArm6 model from Viam's URDF

**Files:**
- Create: `assets/xarm6/xarm6.urdf`, `assets/xarm6/meshes/*.stl`, `assets/xarm6/NOTICE`, `rl_insert/robot.py`
- Test: `tests/test_robot.py`

- [ ] **Step 1: Copy and patch the URDF and meshes**

The source is `../viam-ufactory-xarm/arm/xarm6.urdf` and `../viam-ufactory-xarm/arm/meshes/xarm6meshes/*.stl`. The patch makes two changes: it reduces mesh paths to bare filenames, and it adds a `<mujoco>` compiler block (meshdir, and no static-body fusing so `link_eef` survives).

```bash
SRC=../viam-ufactory-xarm/arm
mkdir -p assets/xarm6/meshes
cp $SRC/meshes/xarm6meshes/*.stl assets/xarm6/meshes/
sed -e 's#package://description/meshes/xarm6meshes/##' \
    -e 's#<robot name="\([^"]*\)">#<robot name="\1">\n<mujoco><compiler meshdir="meshes" fusestatic="false" discardvisual="false" balanceinertia="true"/></mujoco>#' \
    $SRC/xarm6.urdf > assets/xarm6/xarm6.urdf
head -3 assets/xarm6/xarm6.urdf
```

Expected: line 3 is the `<mujoco><compiler .../></mujoco>` block.

- [ ] **Step 2: Write `assets/xarm6/NOTICE`**

```
xarm6.urdf and meshes/*.stl are copied from viam-ufactory-xarm (https://github.com/viam-modules/viam-ufactory-xarm),
Apache License 2.0. Changes: mesh paths reduced to file names and a <mujoco> compiler block added.
```

- [ ] **Step 3: Write the failing test `tests/test_robot.py`**

```python
import mujoco
import numpy as np
import pytest

from rl_insert.robot import ARM_JOINTS, FLANGE, add_gripper, load_xarm6


def test_arm_joints_and_actuators_match_viam_names():
    m = load_xarm6().compile()
    assert tuple(m.joint(i).name for i in range(m.njnt)) == ARM_JOINTS
    assert tuple(m.actuator(i).name for i in range(m.nu)) == ARM_JOINTS


def test_flange_at_zero_matches_viam_home():
    """viam-ufactory-xarm reports the xArm6 flange at (207, 0, 112) mm, pointing down, at zero."""
    m = load_xarm6().compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    flange = m.body(FLANGE).id
    np.testing.assert_allclose(d.xpos[flange], [0.207, 0.0, 0.112], atol=1e-4)
    assert d.xmat[flange].reshape(3, 3)[:, 2] == pytest.approx([0, 0, -1], abs=1e-6)


def test_gripper_adds_one_actuator_and_a_tcp():
    spec = load_xarm6()
    add_gripper(spec)
    m = spec.compile()
    assert m.actuator(m.nu - 1).name == "gripper"
    assert m.site("tcp").id >= 0
```

- [ ] **Step 4: Run it to make sure it fails**

Run: `uv run pytest tests/test_robot.py -v`
Expected: collection error, `ModuleNotFoundError: No module named 'rl_insert.robot'`.

- [ ] **Step 5: Write `rl_insert/robot.py`**

```python
"""The xArm6 (from viam-ufactory-xarm's URDF, the model Viam's frame system uses) plus a
parametric two-finger gripper, as an MjSpec."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco

ASSETS = Path(__file__).resolve().parent.parent / "assets"
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
    ponytail: if the flange seats on the 5 mm locating spigot instead, add 5 mm to tcp.
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
    root = flange.add_body(name="gripper", pos=[0, 0, g.tcp - g.finger_length])
    root.add_site(name="tcp", pos=[0, 0, g.finger_length], size=[0.002] * 3)
    root.add_geom(name="gripper_palm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.05, 0.005],
                  pos=[0, 0, -0.005], mass=0.5)
    half = g.max_opening / 2
    for side, sign in (("left", 1.0), ("right", -1.0)):
        finger = root.add_body(name=f"finger_{side}")
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
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_robot.py -v`
Expected: 3 passed. If `test_flange_at_zero_matches_viam_home` fails, the URDF patch dropped `fusestatic="false"`.

- [ ] **Step 7: Commit**

```bash
git add assets/xarm6 rl_insert/robot.py tests/test_robot.py
git commit -m "feat: xArm6 MuJoCo model from Viam's URDF, with parametric gripper"
```

---

### Task 3: Parametric peg and hole

**Files:**
- Create: `rl_insert/peg_hole.py`
- Test: `tests/test_peg_hole.py` (geometry test only in this task)

- [ ] **Step 1: Write the failing geometry test**

Create `tests/test_peg_hole.py` with only the imports, the `CLEARANCES` constant and `test_bore_fits_peg_with_clearance`:

```python
import math

import mujoco
import numpy as np
import pytest

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene, peg_hole_scene

CLEARANCES = (0.001, 0.0005, 0.0002)


@pytest.mark.parametrize("c", CLEARANCES)
def test_bore_fits_peg_with_clearance(c):
    p = PegHoleParams(clearance=c)
    m = peg_hole_scene(p).compile()
    d = mujoco.MjData(m)
    d.qpos[:3] = [0, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon == 0, "a centred peg touches the wall"
    d.qpos[:3] = [1.5 * c, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon > 0, "a peg 1.5x the clearance off-axis doesn't touch the wall"
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_peg_hole.py -v`
Expected: `ModuleNotFoundError: No module named 'rl_insert.peg_hole'`.

- [ ] **Step 3: Write `rl_insert/peg_hole.py`**

```python
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
```

- [ ] **Step 4: Run the test**

Run: `uv run pytest tests/test_peg_hole.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add rl_insert/peg_hole.py tests/test_peg_hole.py
git commit -m "feat: parametric chamfered hole and mesh peg"
```

---

### Task 4: Contact behaviour tests

**Files:**
- Modify: `tests/test_peg_hole.py`

These pin the contact behaviour the prototype established, so later changes to contact parameters, segment counts or MuJoCo versions get caught.

- [ ] **Step 1: Replace `tests/test_peg_hole.py` with the full suite**

```python
import math

import mujoco
import numpy as np
import pytest

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene, peg_hole_scene

CLEARANCES = (0.001, 0.0005, 0.0002)


def _driven(p: PegHoleParams, x0: float, tilt_deg: float = 0.0, speed: float = 0.04, seconds: float = 3.0):
    """Press the peg straight down at `speed`; return the tip position and peak linear speed."""
    m = driven_peg_scene(p).compile()
    d = mujoco.MjData(m)
    d.qpos[:5] = [x0, 0, 0.003, 0, math.radians(tilt_deg)]
    d.ctrl[:] = d.qpos[:5]
    peak = 0.0
    for i in range(int(seconds / m.opt.timestep)):
        d.ctrl[2] = 0.003 - speed * i * m.opt.timestep
        mujoco.mj_step(m, d)
        peak = max(peak, float(np.linalg.norm(d.qvel[:3])))
    return d.qpos[:3].copy(), peak


@pytest.mark.parametrize("c", CLEARANCES)
def test_bore_fits_peg_with_clearance(c):
    p = PegHoleParams(clearance=c)
    m = peg_hole_scene(p).compile()
    d = mujoco.MjData(m)
    d.qpos[:3] = [0, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon == 0, "a centred peg touches the wall"
    d.qpos[:3] = [1.5 * c, 0, -p.depth / 2]
    mujoco.mj_forward(m, d)
    assert d.ncon > 0, "a peg 1.5x the clearance off-axis doesn't touch the wall"


@pytest.mark.parametrize("c", CLEARANCES)
@pytest.mark.parametrize("case", ["aligned", "chamfer", "tilted"])
def test_pressed_peg_reaches_floor(c, case):
    p = PegHoleParams(clearance=c)
    x0, tilt = {"aligned": (0, 0), "chamfer": (c + 0.7 * p.chamfer, 0), "tilted": (0.0005, 5)}[case]
    tip, peak = _driven(p, x0, tilt)
    assert tip[2] == pytest.approx(-p.depth, abs=0.0005)
    assert peak < 0.5, f"contact injected energy: peak {peak:.2f} m/s"


@pytest.mark.parametrize("c", CLEARANCES)
@pytest.mark.parametrize("past_chamfer", [0.002, 0.003, 0.004])
def test_peg_pressed_beyond_chamfer_stays_in_or_on_the_hole(c, past_chamfer):
    """Off the chamfer the peg may snap in, jam or rest on the rim, but it must never leave the
    fixture or sink through it. (It can move fast when it snaps in: the test servo pushes a 20 g
    peg with ~20 N and has no arm inertia behind it, so speed is not checked here.)"""
    p = PegHoleParams(clearance=c)
    tip, _ = _driven(p, c + p.chamfer + past_chamfer)
    assert np.all(np.isfinite(tip))
    assert np.hypot(*tip[:2]) < p.hole_radius + p.wall, "the peg left the fixture"
    assert -p.depth - 0.0005 < tip[2] < 0.003, "the peg sank through the floor or bounced up"


def test_wall_resists_push():
    """50 N sideways, more than the arm should ever apply, moves the peg < 0.1 mm into the wall."""
    p = PegHoleParams(clearance=0.0002)
    spec = peg_hole_scene(p)
    spec.option.gravity = [0, 0, 0]
    peg = spec.body("peg")
    spec.delete(spec.joint("peg_free"))
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        peg.add_joint(type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis)
    peg.pos = [0, 0, -p.depth / 2]
    m = spec.compile()
    d = mujoco.MjData(m)
    for _ in range(500):
        d.xfrc_applied[m.body("peg").id, :3] = [50, 0, 0]
        mujoco.mj_step(m, d)
    assert d.qpos[0] - p.clearance < 0.0001
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_peg_hole.py -v`
Expected: 22 passed, in about a second.

If `test_wall_resists_push` fails, `CONTACT_SOLREF` was softened. If a `test_pressed_peg_reaches_floor` case fails on `peak`, the driven rig's servo gains were raised (keep ω·dt ≈ 0.1).

- [ ] **Step 3: Sanity-check that the tests bite**

Temporarily set `CONTACT_SOLREF = (0.02, 1.0)` in `rl_insert/peg_hole.py`.
Run: `uv run pytest tests/test_peg_hole.py::test_wall_resists_push -v`
Expected: FAIL (the peg ends millimetres into the wall). Revert the change, then run again: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_peg_hole.py
git commit -m "test: pin peg-in-hole contact behaviour under an arm-like driven peg"
```

---

### Task 5: Full scene with the peg in hand

**Files:**
- Create: `rl_insert/scene.py`
- Test: `tests/test_scene.py`

- [ ] **Step 1: Write the failing test**

```python
import mujoco
import numpy as np

from rl_insert.scene import SceneParams, build_scene, place_peg_in_hand


def _tip_in_tcp(m, d):
    tcp, tip = m.site("tcp").id, m.site("peg_tip").id
    return d.site_xmat[tcp].reshape(3, 3).T @ (d.site_xpos[tip] - d.site_xpos[tcp])


def test_closed_gripper_holds_peg_under_gravity():
    p = SceneParams()
    m = build_scene(p).compile()
    d = mujoco.MjData(m)
    place_peg_in_hand(m, d, p.peg_hole.peg_radius, p.protrusion)
    np.testing.assert_allclose(_tip_in_tcp(m, d), [0, 0, p.protrusion], atol=1e-6)
    d.ctrl[:] = 0  # arm holds zero, gripper commanded closed
    for _ in range(2000):
        mujoco.mj_step(m, d)
    np.testing.assert_allclose(_tip_in_tcp(m, d), [0, 0, p.protrusion], atol=5e-4)
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_scene.py -v`
Expected: `ModuleNotFoundError: No module named 'rl_insert.scene'`.

- [ ] **Step 3: Write `rl_insert/scene.py`**

```python
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
```

- [ ] **Step 4: Run all tests**

Run: `uv run pytest -v`
Expected: 26 passed.

If the grasp test fails with the tip shifted a few millimetres sideways, the finger equality lost its `solref`.

- [ ] **Step 5: Commit**

```bash
git add rl_insert/scene.py tests/test_scene.py
git commit -m "feat: insertion scene with the peg held by the gripper"
```

---

### Task 6: MuJoCo Warp contact spike

**Files:**
- Create: `spikes/contact_warp.py`

- [ ] **Step 1: Write `spikes/contact_warp.py`**

```python
"""Contact spike: does MuJoCo Warp handle the peg and hole like CPU MuJoCo, at scale?

Presses N pegs into the hole from random offsets, directions and tilts (the arm-driven rig from
rl_insert.peg_hole), on Warp, and the first K of them on CPU MuJoCo. Prints a JSON report and
exits 1 if the gate fails:
  - no world blows up (non-finite, left the fixture, sank through, bounced up),
  - over the K reference worlds, Warp's insertion rate is within 10 points of CPU MuJoCo's,
  - the contact buffer never passes 90% full (a full one drops contacts silently).
Per-world agreement is reported too, but not gated: pegs balanced on the rim go either way
(float32 vs float64), and RL only sees the statistics.

    uv run python spikes/contact_warp.py --device cpu --nworld 16 --reference 16
    uv run python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from rl_insert.peg_hole import PegHoleParams, driven_peg_scene

SPEED = 0.04  # m/s, the policy's top speed: 2 mm per 50 ms step
SECONDS = 3.0
CTRL_EVERY = 10  # steps between ctrl updates (0.4 mm of ramp), to keep host-device copies rare
NCONMAX = 128  # contacts per world; one wedged peg peaked at 94
NJMAX = 1024  # constraint rows per world


def initial_states(p: PegHoleParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """(n, 5) qpos: tip x, y, z and tilts, offset up to 4 mm past the chamfer in a random direction."""
    r = rng.uniform(0, p.clearance + p.chamfer + 0.004, n)
    a = rng.uniform(0, 2 * math.pi, n)
    tilt = np.radians(rng.uniform(0, 5, (n, 2)))
    return np.column_stack([r * np.cos(a), r * np.sin(a), np.full(n, 0.003), tilt])


def target_z(step: int, dt: float) -> float:
    return 0.003 - SPEED * step * dt


def classify(p: PegHoleParams, tip: np.ndarray) -> str:
    if not np.all(np.isfinite(tip)):
        return "blew_up"
    if np.hypot(*tip[:2]) >= p.hole_radius + p.wall or tip[2] <= -p.depth - 0.0005 or tip[2] >= 0.003:
        return "blew_up"
    return "inserted" if tip[2] < -p.depth + 0.001 else "not_inserted"


def run_cpu(m: mujoco.MjModel, q0: np.ndarray) -> np.ndarray:
    d = mujoco.MjData(m)
    d.qpos[:] = q0
    d.ctrl[:] = q0
    for i in range(int(SECONDS / m.opt.timestep)):
        if i % CTRL_EVERY == 0:
            d.ctrl[2] = target_z(i, m.opt.timestep)
        mujoco.mj_step(m, d)
    return d.qpos[:3].copy()


def run_warp(m: mujoco.MjModel, q0: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Final tips, world-steps per second, and peak contact-buffer use (fraction of capacity)."""
    n = len(q0)
    mw = mjw.put_model(m)
    # nconmax and njmax are per world; the total contact buffer is nconmax * nworld
    dw = mjw.put_data(m, mujoco.MjData(m), nworld=n, nconmax=NCONMAX, njmax=NJMAX)
    peak_contacts = 0
    dw.qpos = wp.array(q0.astype(np.float32), dtype=float)
    ctrl = q0.astype(np.float32).copy()
    steps = int(SECONDS / m.opt.timestep)
    start = time.perf_counter()
    for i in range(steps):
        if i % CTRL_EVERY == 0:
            ctrl[:, 2] = target_z(i, m.opt.timestep)
            dw.ctrl = wp.array(ctrl, dtype=float)
            peak_contacts = max(peak_contacts, int(dw.nacon.numpy()[0]))
        mjw.step(mw, dw)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    return dw.qpos.numpy()[:, :3], n * steps / elapsed, peak_contacts / (NCONMAX * n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nworld", type=int, default=4096)
    ap.add_argument("--reference", type=int, default=64)
    ap.add_argument("--clearance", type=float, nargs="+", default=[0.001, 0.0005, 0.0002])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    wp.config.quiet = True
    wp.set_device(args.device)
    rng = np.random.default_rng(args.seed)
    report, ok = {"device": args.device, "nworld": args.nworld, "results": []}, True
    for c in args.clearance:
        p = PegHoleParams(clearance=c)
        m = driven_peg_scene(p).compile()
        q0 = initial_states(p, args.nworld, rng)
        tips, rate, buffer_use = run_warp(m, q0)
        warp_outcomes = [classify(p, t) for t in tips]
        k = min(args.reference, args.nworld)
        cpu_outcomes = [classify(p, run_cpu(m, q0[i])) for i in range(k)]
        agree = sum(a == b for a, b in zip(warp_outcomes[:k], cpu_outcomes)) / k
        warp_rate = warp_outcomes[:k].count("inserted") / k
        cpu_rate = cpu_outcomes.count("inserted") / k
        counts = {o: warp_outcomes.count(o) for o in ("inserted", "not_inserted", "blew_up")}
        # a full contact buffer drops contacts silently, which looks like penetration
        passed = counts["blew_up"] == 0 and abs(warp_rate - cpu_rate) <= 0.1 and buffer_use < 0.9
        ok &= passed
        report["results"].append({"clearance_mm": c * 1e3, "warp": counts, "insert_rate_warp": round(warp_rate, 3),
                                  "insert_rate_cpu": round(cpu_rate, 3), "per_world_agreement": round(agree, 3),
                                  "contact_buffer_peak_use": round(buffer_use, 3),
                                  "world_steps_per_s": round(rate), "passed": passed})
    report["passed"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it on the local CPU**

Run: `uv run python spikes/contact_warp.py --device cpu --nworld 32 --reference 32`
Expected: JSON with `"blew_up": 0` for every clearance, insertion rates within 10 points, `"passed": true`, exit 0.
The first run compiles Warp kernels (about 30 s). The prototype measured insertion rates of about 0.72/0.75, 0.63/0.69 and 0.38/0.41 (Warp/CPU) at 1.0, 0.5 and 0.2 mm.

- [ ] **Step 3: Commit**

```bash
git add spikes/contact_warp.py
git commit -m "spike: compare MuJoCo Warp and CPU MuJoCo on peg-in-hole contact"
```

---

### Task 7: Training container and GPU probe

**Files:**
- Create: `container/Dockerfile`, `scripts/probe_gpu.sh`

- [ ] **Step 1: Write `container/Dockerfile`**

```dockerfile
# Training image. For now it holds the sim package and the contact spike; the training plan adds
# mjlab and the training script on top. linux/amd64 is what Viam training jobs run.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONPATH=/app PATH="/app/.venv/bin:$PATH"
WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY rl_insert rl_insert
COPY assets assets
COPY spikes spikes

CMD ["python", "spikes/contact_warp.py", "--device", "cuda"]
```

`warp-lang` wheels bundle the CUDA runtime, so the base image needs no CUDA. `--gpus all` supplies the driver.

- [ ] **Step 2: Write `scripts/probe_gpu.sh` and make it executable**

```bash
#!/usr/bin/env bash
# Run on the cloud sim machine (L40S): can it build and run the training container on its GPU?
# Prints each step; stops at the first failure and prints FAIL.
set -euo pipefail
trap 'echo "FAIL"' ERR
cd "$(dirname "$0")/.."
IMAGE=rl-insert:probe

step() { echo "== $1"; }

step "host GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

step "docker can see the GPU (needs the NVIDIA container toolkit)"
docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L

step "build the image"
docker build --platform linux/amd64 -f container/Dockerfile -t "$IMAGE" .

step "Warp finds CUDA inside the container"
docker run --rm --gpus all "$IMAGE" python -c \
  "import warp as wp; wp.init(); assert wp.is_cuda_available(), 'no CUDA'; print(wp.get_cuda_devices())"

step "contact spike on the GPU"
docker run --rm --gpus all "$IMAGE" python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64

echo "PASS"
```

```bash
chmod +x scripts/probe_gpu.sh
```

- [ ] **Step 3: Build and run the image locally (CPU, emulated amd64)**

```bash
docker build --platform linux/amd64 -f container/Dockerfile -t rl-insert:probe .
docker run --rm --platform linux/amd64 rl-insert:probe \
  python spikes/contact_warp.py --device cpu --nworld 4 --reference 4 --clearance 0.0005
```

Expected: the build succeeds; the JSON ends with `"passed": true`. Emulated, this takes a few minutes.

- [ ] **Step 4: Commit**

```bash
git add container/Dockerfile scripts/probe_gpu.sh
git commit -m "feat: training container and GPU probe script"
```

---

### Task 8: Run the probe on the cloud sim machine (L40S) and record results

This task runs on the user's cloud sim machine, not locally. Stop and hand over to the user if you can't reach it.

- [ ] **Step 1: Get the repo onto the machine** (clone or rsync the branch), then run:

```bash
./scripts/probe_gpu.sh 2>&1 | tee probe.log
```

Expected: every step prints, the spike JSON shows `"passed": true` for all three clearances, and the script ends with `PASS`.

- [ ] **Step 2: If Docker can't use the GPU** (the step "docker can see the GPU" fails, or there's no Docker), use the uv fallback with the same pinned deps:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync --frozen --no-dev
uv run python -c "import warp as wp; wp.init(); assert wp.is_cuda_available()"
uv run python spikes/contact_warp.py --device cuda --nworld 4096 --reference 64
```

Record which path worked. Plan 2's training workflow depends on it.

- [ ] **Step 3: Write `docs/superpowers/specs/2026-11-16-contact-spike-results.md`**

Include:
- the GPU (`nvidia-smi` name, memory, driver);
- Docker or uv fallback;
- the spike JSON;
- `world_steps_per_s` at nworld 4096;
- the decision.

Gate:
- **Pass** (no blow-ups, rates within 10 points at all clearances): proceed to plan 2.
- **Blow-ups on GPU only:**
  - First check `contact_buffer_peak_use`. If it's near 0.9, raise `NCONMAX`/`NJMAX` in the spike. They are per world, so memory is nworld × these.
  - Otherwise compare with `--device cpu` on the same seed. If CUDA diverges from Warp-on-CPU, file it against mujoco_warp with the seed and clearance.
- **Rate gap over 10 points:** rerun with `--seed 1` and then `--seed 2`. If it persists, raise segments to 48 and retest before going further.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-11-16-contact-spike-results.md
git commit -m "docs: contact spike results on the cloud sim GPU"
```
