# MicroDuck milestone 1: floating base, BAM and `io` in viam-mujoco-sim — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `viam-labs:mujoco-sim`'s `robot` model can load a free-floating legged robot. It drives the robot's servos with Rhoban's BAM actuator model, and serves one atomic `io` DoCommand per control tick. The MicroDuck walk MJCF loads into it from a private archive package and stands under BAM.

**Architecture:** Implements units 1–3 and 6 of the spec,
`docs/superpowers/specs/2026-11-21-microduck-sim-to-sim-design.md` (read it first).

- `robot_file.py` accepts one free joint on a root body, and homes it relative to the robot's frame.
- A new `bam.py` in mujoco-sim:
  - rewrites the matching actuators to torque motors at spec time, mirroring Pollen's
    `load_mujoco_with_bam`;
  - drives them through `bam.mujoco.MujocoController` from a step callback that rebinds on every
    model rebuild.
- Limb arms on a BAM robot are read-only.
- The robot gets an `io` method, exposed as a DoCommand.

The duck package is built in this repo from a pinned `microduck_rl` commit.

**Tech Stack:** Python 3.12, MuJoCo 4.0 (`MjSpec`), `better-actuator-models` (BAM) at commit
`62bd8ce12154340be97e06f7f41a0ca8f116d967`, viam-sdk, pytest, uv.

**Repos and branches:**
- Tasks 1–9: `/Users/nick.hehr/src/viam-mujoco-sim`, on a new branch `floating-base-bam` from
  `origin/main` (commit `1f227d9` or later). The local checkout is on `diff-ik-policy-sim`, which is
  unrelated in-flight work. Don't build on it, and don't discard it.
- Tasks 10–11: this repo, `/Users/nick.hehr/src/viam-rslrl-policy-training`, branch
  `peg-insert-design`.

**Pinned upstream sources** (copy their numbers; do not re-derive):
- `pollen-robotics/microduck_rl` @ `0daab160d3fb1c2cbaba16a4a19052f36709dd9e`:
  - `scripts/infer_policy.py` `load_mujoco_with_bam` (lines 59–104) and the BAM constants
    (lines 29–46);
  - `src/mjlab_microduck/sim/body_server.py` `gravity_in_trunk`.
- `Rhoban/bam` @ `62bd8ce`: `bam/mujoco.py` `MujocoController(model, actuator, mujoco_model,
  mujoco_data, vin_drop_gain=None, vin_min=None)`, `.reset(qpos)`, `.update()`, `.q_target`,
  `.act_indexes`, `.qpos_indexes`, `.dof_indexes`.
- The branch head of `mjlab_frictionloss` has since renamed `vin_drop_gain` to
  `vin_drop_resistance`. **Pin the commit, never the branch.**

**Facts about mujoco-sim this plan relies on** (verified at `1f227d9`):
- `SimManager._rebuild` (`sim.py` ~741) compiles a new `model` and `data` on every
  `set_contribution`, copies state into it, runs `init` only for new contributions, then increments
  `generation`.
- `step_once` (`sim.py` ~776) runs every `step_callbacks` entry, then calls `mj_step`. A callback
  that raises is logged and **removed**, so the BAM callback must never raise while the robot is
  out of the scene.
- In the `sim` test fixture (`tests/conftest.py`), `sim.run(fn)` runs inline, and nothing steps
  unless the test calls `sim.advance(seconds)`.
- `RobotHandle` loads its file with `robots.load_mjcf`, which sets `gravcomp = 1` on every body.
  That's meant for position-servoed arms; a floating robot with it would hover weightless. Task 2
  clears it for floating robots.
- `bam.mujoco.MujocoController.__init__` (62bd8ce) calls `mj_setConst(model, data)`, which resets
  that `data`'s qpos to `qpos0`. Task 4 constructs it on scratch data.
- mujoco-sim compiles every scene with `mjINT_IMPLICITFAST` (`sim.py:666`).
- MuJoCo 4.0 `MjSpec`:
  - `MjsJoint.damping` is a shape-(3,) array;
  - `MjsActuator.forcelimited` and `ctrllimited` are `mujoco.mjtLimited` enums;
  - `MjsActuator.target` is a `str`;
  - `spec.worldbody.bodies` lists the root bodies.

---

## File structure

viam-mujoco-sim:

| File | Responsibility |
|---|---|
| `pyproject.toml`, `.python-version`, `uv.lock` | Python 3.12 and the BAM dependency (Task 1) |
| `src/mujoco_module/bam.py` (new) | `BamConfig` (attrs), `rewrite_spec` (spec-time actuator rewrite), `BamDrive` (bind, rebind, step, targets, torque) |
| `src/mujoco_module/robot_file.py` | Free-root check, floating home, `actuators` wiring, `io`, `commanded` |
| `src/mujoco_module/handles.py` (`ArmHandle` only) | Limb frames on a floating root; read-only arms under BAM |
| `src/mujoco_module/models/robot.py` | Validate `actuators`; the `io` DoCommand |
| `tests/test_floating_robot.py` (new) | Everything above, on a synthetic floating biped (no Pollen meshes) |
| `tests/test_robot_file.py:209` | Update the old "free joint yet" expectation |
| `README.md` (robot section) | Document the floating base, `actuators` and `io` |

This repo:

| File | Responsibility |
|---|---|
| `scripts/export_duck_scene.py` (new) | Build `dist/duck-sim/` and `dist/duck-sim.tar.gz` from a `microduck_rl` checkout at the pinned commit |
| `scripts/duck_stand_check.py` (new) | Local check: the real duck MJCF in mujoco-sim with BAM stands and reports the policy's joint order |
| `docs/superpowers/specs/duck-sim-machine.json` (new) | Example machine config for the duck cell |

---

### Task 1: Python 3.12 and the BAM dependency (go/no-go gate)

BAM declares `requires-python >=3.12,<3.13`. mujoco-sim is on 3.11 (`.python-version`) with
`requires-python >=3.11`. uv refuses to lock a dependency whose Python range doesn't cover the
project's.

Install from GitHub's archive tarball, not `git+`: uv needs a `git` binary for git dependencies, and
the Viam cloud-sim image may not have one.

**Files:**
- Modify: `pyproject.toml`
- Modify: `.python-version`
- Regenerate: `uv.lock`

- [ ] **Step 1: Branch from main**

```bash
cd /Users/nick.hehr/src/viam-mujoco-sim
git status --short   # expect clean; if not, stop and ask: diff-ik-policy-sim has the user's work
git fetch origin && git switch -c floating-base-bam origin/main
```

- [ ] **Step 2: Pin Python and add BAM**

In `pyproject.toml`, change `requires-python = ">=3.11"` to `requires-python = ">=3.12,<3.13"`. Add
this to `dependencies` (keep the list alphabetical):

```toml
    # Rhoban's BAM actuator models (Apache-2.0), at the commit pollen-robotics/microduck_rl locks:
    # its MujocoController takes vin_drop_gain (the branch head renamed it). Archive URL, not git+,
    # so installing needs no git binary.
    "better-actuator-models @ https://github.com/Rhoban/bam/archive/62bd8ce12154340be97e06f7f41a0ca8f116d967.tar.gz",
```

Add to the end of `pyproject.toml`:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

Only add that block if the build backend is hatchling. Check `[build-system]`: there is none today,
so uv treats the project as virtual and the block is unnecessary. Skip it unless `uv lock` complains.

Write `3.12` to `.python-version`.

- [ ] **Step 3: Lock and sync**

Run: `uv lock && uv sync`
Expected: it resolves `better-actuator-models==1.0.1` from the archive URL, and `uv run python -c "import bam.mujoco, bam.model; m = bam.model.load_model(motor_name='xl330', model='m6'); print(m.kt.value, m.R.value)"`
prints two positive floats.

- [ ] **Step 4: Gate — full existing suite on 3.12**

Run: `uv run python -m pytest -q`
Expected: the same pass count as on `main` with 3.11. Get the baseline by running
`git stash; uv sync --python 3.11; uv run python -m pytest -q; git stash pop` first, if one isn't
already recorded.

**If anything regresses, or the lock fails:** switch to the fallback. Vendor these files under
`src/mujoco_module/_bam/` and import from there instead:
- BAM's `bam/mujoco.py`, `bam/model.py`, `bam/actuator.py`, `bam/actuators.py`,
  `bam/parameter.py`;
- `bam/params/xl330/m6.json`;
- the Apache-2.0 notice, added to `NOTICE`.

Keep Python at 3.11 in that case. Record which path was taken in the commit message.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .python-version uv.lock
git commit -m "build: Python 3.12 and BAM (better-actuator-models @ 62bd8ce) for actuator physics"
```

---

### Task 2: A robot file may have one free joint on a root body

**Files:**
- Modify: `src/mujoco_module/robot_file.py:35-57` (`_QPOS_WIDTH`, `check_robot_file`)
- Modify: `src/mujoco_module/robot_file.py` `RobotHandle.__init__`
- Modify: `tests/test_robot_file.py:209`
- Create: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the test fixture and failing tests**

Create `tests/test_floating_robot.py`:

```python
"""A free-floating robot file (a legged robot's trunk on a free joint) in viam-labs:mujoco-sim:robot,
its BAM actuators, read-only limb arms and the io DoCommand. A synthetic two-legged robot stands in
for the MicroDuck, whose meshes are CC BY-NC-SA and stay out of this repo."""

import numpy as np
import pytest

from mujoco_module.handles import ArmHandle
from mujoco_module.robot_file import RobotHandle

# A 0.3 kg trunk on a free joint, and two 2-hinge legs about y with wide flat box feet, so it
# stands statically. Actuator order interleaves the legs the way the duck's file does (left, then
# right), so io's joint order is the file's actuator order. Geometry at STAND (hip 0.15,
# knee -0.3):
# - each 0.1 m segment drops 0.1*cos(0.15) = 0.0989 m;
# - the shin's world pitch is -0.15, so the foot box is pitched +0.15 on the shin to lie flat;
# - trunk centre to sole is 0.03 + 2*0.0989 + 0.005 = 0.2328, so the trunk starts at 0.234,
#   about 1 mm above the floor.
# STAND must differ from qpos0 (all zeros): several tests use that to catch a qpos wipe.
FLOATER = """
<mujoco>
  <compiler angle="radian"/>
  <default><joint damping="0.05" frictionloss="0.01" armature="0.001"/><position kp="5"/></default>
  <worldbody>
    <body name="trunk" pos="0 0 0.234">
      <freejoint name="root"/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.3"/>
      <body name="l_thigh" pos="0 0.03 -0.03">
        <joint name="l_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="l_shin" pos="0 0 -0.1">
          <joint name="l_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="l_foot" pos="0 0 -0.1"/>
        </body>
      </body>
      <body name="r_thigh" pos="0 -0.03 -0.03">
        <joint name="r_hip" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
        <body name="r_shin" pos="0 0 -0.1">
          <joint name="r_knee" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.01" mass="0.05"/>
          <geom type="box" size="0.04 0.02 0.005" pos="0 0 -0.105" euler="0 0.15 0" mass="0.02"/>
          <site name="r_foot" pos="0 0 -0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="l_hip" joint="l_hip"/>
    <position name="l_knee" joint="l_knee"/>
    <position name="r_hip" joint="r_hip"/>
    <position name="r_knee" joint="r_knee"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="0 0 0.234 1 0 0 0  0.15 -0.3 0.15 -0.3" ctrl="0.15 -0.3 0.15 -0.3"/>
  </keyframe>
</mujoco>
"""
STAND = [0.15, -0.3, 0.15, -0.3]
TRUNK_Z = 0.234


@pytest.fixture
def floater(tmp_path):
    path = tmp_path / "floater.xml"
    path.write_text(FLOATER)
    return path


def robot(sim, path, **extra):
    return RobotHandle(sim, "duck", {"mjcf": str(path), **extra})


def free_qpos(sim, name="duck/root"):
    adr = sim.model.joint(name).qposadr[0]
    return sim.data.qpos[adr : adr + 7].copy()


def test_a_free_root_loads_and_homes_from_its_keyframe(sim, floater):
    robot(sim, floater)
    assert sim.model.joint("duck/root").type == 0  # mjJNT_FREE
    assert free_qpos(sim) == pytest.approx([0, 0, TRUNK_Z, 1, 0, 0, 0])


def test_a_floating_robot_has_weight(sim, floater):
    # load_mjcf gravity-compensates every body (for arms); a floating robot must not hover
    robot(sim, floater)
    trunk = sim.model.body("duck/trunk").id
    assert sim.model.body_gravcomp[trunk] == 0.0
    assert sim.model.body_gravcomp[sim.model.body("duck/l_shin").id] == 0.0


def test_a_free_root_homes_relative_to_the_robots_frame(sim, floater):
    # the keyframe's trunk pose is in the file's world; the robot's frame moves it
    robot(sim, floater, position=[1.0, 0.5, 0.0], orientation_wxyz=[0.7071068, 0, 0, 0.7071068])
    q = free_qpos(sim)
    assert q[:3] == pytest.approx([1.0, 0.5, TRUNK_Z], abs=1e-6)
    assert q[3:] == pytest.approx([0.7071068, 0, 0, 0.7071068], abs=1e-6)


@pytest.mark.parametrize(
    "edit, message",
    [
        # a free joint below the root
        (('<body name="l_thigh" pos="0 0.03 -0.03">', '<body name="l_thigh" pos="0 0.03 -0.03"><freejoint name="loose"/>'),
         "free joint on 'l_thigh', which isn't a root body"),
        # a second floating root
        (("</worldbody>", '<body name="other"><freejoint/><geom size="0.01"/></body></worldbody>'),
         "more than one floating root"),
    ],
)
def test_free_joint_errors(sim, floater, edit, message):
    floater.write_text(FLOATER.replace(*edit, 1))
    with pytest.raises(ValueError, match=message):
        robot(sim, floater)


def test_a_floating_robot_cannot_ride_a_base(sim, floater):
    with pytest.raises(ValueError, match="floating robot .* must be framed on the world"):
        robot(sim, floater, parent="rover")
```

Change `tests/test_robot_file.py:209`. That case is a free body that is *not* the file's only root
(the torso is), so it must still fail, with the new message:

```python
    ('<body name="loose"><freejoint/><geom size="0.01"/></body>', "more than one root body|floating root"),
```

Read the surrounding parametrized test first and confirm the snippet is inserted inside the
`<worldbody>` next to `torso`. If it is inserted elsewhere, pick the message that applies.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -q`
Expected: FAIL with "a robot file can't have a free joint yet".

- [ ] **Step 3: Implement**

In `robot_file.py`, replace `_QPOS_WIDTH` and `check_robot_file`:

```python
_QPOS_WIDTH = {int(mujoco.mjtJoint.mjJNT_BALL): 4, int(_FREE): 7}  # hinge and slide: 1


def check_robot_file(label: str, spec: mujoco.MjSpec) -> str | None:
    """ValueError for what a robot file can't have: flexes, skins, plugins,
    mocap bodies, and free joints anywhere but one root body (a legged
    robot's trunk). Returns that floating root's name, or None."""
    for kind in REJECTED:
        found = list(getattr(spec, kind, ()))
        if found:
            first = getattr(found[0], "name", "") or "#0"
            raise ValueError(f"{label}: a robot file can't have {kind} (found {first!r})")
    roots = {body.name for body in spec.worldbody.bodies}
    floating: list[str] = []
    for body in spec.bodies:
        if body.mocap:
            raise ValueError(f"{label}: a robot file can't have mocap bodies (found {body.name!r})")
        for joint in body.joints:
            if joint.type != _FREE:
                continue
            if body.name not in roots:
                raise ValueError(
                    f"{label}: free joint on {body.name!r}, which isn't a root body; only a "
                    "floating base (the file's root body, e.g. a legged robot's trunk) may have one"
                )
            floating.append(body.name)
    if len(floating) > 1:
        raise ValueError(f"{label}: more than one floating root ({floating}); a robot file has one")
    if floating and len(roots) > 1:
        raise ValueError(
            f"{label}: a floating robot file has one root body (its trunk), found more than one root body {sorted(roots)}"
        )
    return floating[0] if floating else None
```

In `RobotHandle.__init__`, replace the line `check_robot_file(label, spec)` with:

```python
        self.floating_root = check_robot_file(label, spec)
        if self.floating_root:
            # load_mjcf gravity-compensates every body, as a fixed arm's controller does; a
            # floating robot carries its own weight on its legs, or it hovers
            for body in spec.bodies:
                body.gravcomp = 0
```

After `parent = attrs.get("parent")`, add:

```python
        if self.floating_root and parent:
            raise ValueError(
                f"{label}: a floating robot (free joint on {self.floating_root!r}) must be framed on "
                f"the world, not ride {parent!r}"
            )
```

Note: the "second floating root" test hits the `len(floating) > 1` branch only if both roots have
free joints. The fixture's added `other` body does, so the message is "more than one floating root".

In `_home`, replace the qpos copy inside the joint loop:

```python
            width = _QPOS_WIDTH.get(int(own.jnt_type[joint]), 1)
            src = own.jnt_qposadr[joint]
            dst = model.joint(self.prefix + name).qposadr[0]
            key = own.key_qpos[self._key, src : src + width]
            if int(own.jnt_type[joint]) == _FREE:
                # the keyframe's trunk pose is in the file's world: put it in the robot's frame
                pose = self.origin @ robots.transform(key[:3], key[3:])
                data.qpos[dst : dst + 3] = pose[:3, 3]
                data.qpos[dst + 3 : dst + 7] = robots.quat(pose)
                dof = model.joint(self.prefix + name).dofadr[0]
                data.qvel[dof : dof + 6] = 0.0
            else:
                data.qpos[dst : dst + width] = key
```

`self.origin` is already set before `build` and `init` are defined. Check that `robots.transform`
and `robots.quat` take a w-first quaternion, and return/take a 4×4 matrix. Both are used that way
above in this file (`self.origin = robots.transform(position, quat)`).

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py tests/test_robot_file.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/robot_file.py tests/test_floating_robot.py tests/test_robot_file.py
git commit -m "feat: a robot file may float on one free root joint, homed in the robot's frame"
```

---

### Task 3: Limb arms framed on a floating trunk

Today `ArmHandle` checks that a limb's chain base is welded to the world (`reference = 0`). For a
floating robot, the reference is the floating root. The arm's frame is then the chain base's pose
relative to the trunk.

**Files:**
- Modify: `src/mujoco_module/handles.py` (the `if holder is not None:` frame block, ~lines 340–372)
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_floating_robot.py`:

```python
def leg(side, **extra):
    # chain base is the trunk itself, so the arm's frame relative to the robot is the identity
    return {"robot": "duck", "parent": "duck", "joints": [f"{side}_hip", f"{side}_knee"],
            "flange_site": f"{side}_foot", "home_joints": STAND[:2], **extra}


def test_limbs_bind_to_a_floating_trunk(sim, floater):
    robot(sim, floater)
    left = ArmHandle(sim, "left-leg", leg("l"))
    ArmHandle(sim, "right-leg", leg("r"))
    assert left.joint_names() == ("l_hip", "l_knee")
    assert left.get_joint_positions() == pytest.approx(STAND[:2], abs=1e-9)
    # the arm frame site sits on the trunk and moves with it
    assert sim.data.site("left-leg/arm_frame").xpos == pytest.approx(sim.data.body("duck/trunk").xpos)


def test_a_limb_frame_off_the_trunk_says_what_to_paste(sim, floater):
    robot(sim, floater)
    # -?: inv(parent_end) @ base can leave -1e-17, printed as -0.0
    with pytest.raises(ValueError, match=r'set frame.translation to \{"x": -?0.0, "y": -?0.0, "z": -?0.0\}'):
        ArmHandle(sim, "left-leg", leg("l", position=[0, 0, 0.1]))
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k limb -q`
Expected: FAIL with "trunk moves relative to robot duck".

- [ ] **Step 3: Implement**

In `ArmHandle.__init__`'s robot branch, just after `reference, parent_end = 0, np.eye(4)`, add:

```python
            if above is None and holder.floating_root:
                # a floating robot's limbs are framed on its trunk, which moves with them
                reference = self.model.body(holder.floating_root).id
                parent_end = robots.transform(data.xpos[reference], data.xquat[reference])
```

Nothing else changes. The weld check and `expected = inv(parent_end) @ base @ inv(base_offset)` now
compare against the trunk.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py tests/test_robot_file.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/handles.py tests/test_floating_robot.py
git commit -m "feat: limb arms of a floating robot are framed on its trunk"
```

---

### Task 4: `bam.py` — config, spec rewrite, drive

**Files:**
- Create: `src/mujoco_module/bam.py`
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_floating_robot.py`. The reference rewrite is a verbatim port of Pollen's
`load_mujoco_with_bam` (microduck_rl @ 0daab16, `scripts/infer_policy.py:59-104`, Apache-2.0).
Keep it in the test as the oracle, so a change to `bam.rewrite_spec` that drifts from Pollen's fails
here.

```python
import mujoco

from mujoco_module.bam import BamConfig, rewrite_spec

POLLEN_SOLREF = (-5.0e4, -2.0e2)
POLLEN_SOLIMP = (0.99, 0.9999, 0.001, 0.5, 2.0)


def pollen_rewrite(spec, bam_model):
    """microduck_rl@0daab16 scripts/infer_policy.py load_mujoco_with_bam, the spec part."""
    force_limit = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    for act in spec.actuators:
        if act.target.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        joint = spec.joint(act.target)
        joint.damping = np.zeros(np.shape(joint.damping))
        joint.frictionloss = 0.0
        joint.solref_friction = POLLEN_SOLREF
        joint.solimp_friction = POLLEN_SOLIMP


def test_bam_config_defaults_and_errors():
    cfg = BamConfig.from_attrs({"model": "bam"}, "robot duck")
    assert (cfg.motor, cfg.bam_model, cfg.kp_fw, cfg.vin, cfg.vin_drop_gain, cfg.vin_min) == (
        "xl330", "m6", 200.0, 7.4, 0.1, 6.0)
    with pytest.raises(ValueError, match='actuators.model must be "bam"'):
        BamConfig.from_attrs({"model": "pd"}, "robot duck")
    with pytest.raises(ValueError, match="unknown actuators keys"):
        BamConfig.from_attrs({"model": "bam", "kp": 3}, "robot duck")
    with pytest.raises(ValueError, match="BAM has no motor"):
        BamConfig.from_attrs({"model": "bam", "motor": "nope"}, "robot duck")


def test_rewrite_matches_pollens_load_mujoco_with_bam():
    cfg = BamConfig.from_attrs({"model": "bam"}, "t")
    ours, theirs = mujoco.MjSpec.from_string(FLOATER), mujoco.MjSpec.from_string(FLOATER)
    names = rewrite_spec(ours, cfg, cfg.load())
    pollen_rewrite(theirs, cfg.load())
    assert names == ["l_hip", "l_knee", "r_hip", "r_knee"]
    a, b = ours.compile(), theirs.compile()
    for field in ("actuator_gainprm", "actuator_biasprm", "actuator_gear", "actuator_forcerange",
                  "actuator_forcelimited", "actuator_ctrllimited", "actuator_dyntype",
                  "dof_damping", "dof_frictionloss", "dof_solref", "dof_solimp"):
        assert getattr(a, field) == pytest.approx(getattr(b, field)), field
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k "bam_config or rewrite" -q`
Expected: FAIL with `ModuleNotFoundError: mujoco_module.bam`.

- [ ] **Step 3: Implement `src/mujoco_module/bam.py`**

```python
"""BAM actuator physics for a robot file's servos: Rhoban's better-actuator-models
(https://github.com/Rhoban/bam, Apache-2.0), as pollen-robotics/microduck_rl runs it on CPU MuJoCo.

At spec time the matching position actuators become voltage-limited torque motors with their joint
damping and friction zeroed, because BAM computes friction itself each step (rewrite_spec, a port
of microduck_rl@0daab16 scripts/infer_policy.py load_mujoco_with_bam). Each physics step,
BamDrive.step runs bam.mujoco.MujocoController.update(), which turns a position target into motor
torque through the servo's voltage control law, back-EMF and load-dependent friction.

The model is recompiled whenever a contribution is added, so BamDrive binds a fresh controller to
the current model on the first step (or io call) after each rebuild. q_target restarts at the pose
then, which follows every contribution's init.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

if TYPE_CHECKING:
    from .sim import SimManager

LOGGER = logging.getLogger(__name__)

# BAM's stiff friction-loss constraint (bam.mjlab.BamActuator stiff_frictionloss), as infer_policy.py
STIFF_SOLREF_FRICTION = (-5.0e4, -2.0e2)
STIFF_SOLIMP_FRICTION = (0.99, 0.9999, 0.001, 0.5, 2.0)
# The step BAM's XL330 fit, the duck's contact solref and armature are tuned at (body_server.py)
FIT_DT = 0.005
_STRINGS = ("motor", "bam_model", "joints")


@dataclass(frozen=True)
class BamConfig:
    motor: str = "xl330"
    bam_model: str = "m6"
    kp_fw: float = 200.0  # the servo firmware's position gain register
    vin: float = 7.4  # battery volts
    vin_drop_gain: float = 0.1  # sag: vin - gain * sum|motor torque|; 0 turns it off
    vin_min: float = 6.0  # floor on the sagged voltage
    joints: str = r"^(?!passive_).*"  # actuators whose target joint matches are BAM's

    @classmethod
    def from_attrs(cls, attrs: Any, label: str) -> BamConfig:
        if not isinstance(attrs, dict):
            raise ValueError(f"{label}: actuators must be an object")
        attrs = dict(attrs)
        if attrs.pop("model", None) != "bam":
            raise ValueError(f'{label}: actuators.model must be "bam"')
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(attrs) - known)
        if unknown:
            raise ValueError(f"{label}: unknown actuators keys {unknown}; known: {sorted(known)}")
        try:
            cfg = cls(**{k: (str(v) if k in _STRINGS else float(v)) for k, v in attrs.items()})
            re.compile(cfg.joints)
        except (TypeError, ValueError, re.error) as error:
            raise ValueError(f"{label}: actuators: {error}") from None
        try:
            cfg.load()
        except FileNotFoundError:
            raise ValueError(f"{label}: BAM has no motor {cfg.motor!r} model {cfg.bam_model!r}") from None
        return cfg

    def load(self) -> Any:
        """A fresh bam.model.Model with this config's firmware gain and battery voltage."""
        from bam.model import load_model

        model = load_model(motor_name=self.motor, model=self.bam_model)
        model.actuator.kp = self.kp_fw
        model.actuator.vin = self.vin
        model.actuator.max_current = None  # training runs without the firmware current limit
        return model


def rewrite_spec(spec: mujoco.MjSpec, cfg: BamConfig, bam_model: Any) -> list[str]:
    """Make the actuators whose target joint matches cfg.joints BAM's torque motors, in place, and
    return their names in file order. Must run before the spec compiles."""
    force = bam_model.actuator.vin * bam_model.kt.value / bam_model.R.value
    pattern = re.compile(cfg.joints)
    names = []
    for act in spec.actuators:
        if not pattern.match(act.target):
            continue
        joint = spec.joint(act.target)
        if joint is None:
            raise ValueError(f"actuator {act.name!r} drives {act.target!r}, which isn't a joint")
        act.set_to_motor()
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = (-force, force)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        act.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        joint.damping = np.zeros(np.shape(joint.damping))
        joint.frictionloss = 0.0
        joint.solref_friction = STIFF_SOLREF_FRICTION
        joint.solimp_friction = STIFF_SOLIMP_FRICTION
        names.append(act.name)
    if not names:
        raise ValueError(f"no actuator's joint matches actuators.joints {cfg.joints!r}")
    return names


class BamDrive:
    """Drives a robot's rewritten actuators (scene names, prefixed) with BAM. Sim thread only."""

    def __init__(self, cfg: BamConfig, names: Sequence[str]) -> None:
        self.cfg = cfg
        self.names = tuple(names)
        self.torque = True
        self._ctrl: Any = None
        self._generation = -1

    def present(self, sim: SimManager) -> bool:
        return sim.model is not None and mujoco.mj_name2id(
            sim.model, mujoco.mjtObj.mjOBJ_ACTUATOR, self.names[0]
        ) >= 0

    def ensure_bound(self, sim: SimManager) -> Any:
        """The controller bound to sim's current model, rebound after a rebuild (q_target then
        holds the current pose). RuntimeError while the robot is out of the scene."""
        if self._ctrl is None or self._generation != sim.generation:
            if not self.present(sim):
                raise RuntimeError("the robot is not in the scene")
            from bam.mujoco import MujocoController

            # MujocoController.__init__ sets the armature, then calls mj_setConst(model, data),
            # which resets data.qpos to qpos0. On sim.data that would teleport every joint in the
            # world. Construct it on scratch data, then point it at the live data.
            ctrl = MujocoController(
                self.cfg.load(), list(self.names), sim.model, mujoco.MjData(sim.model),
                vin_drop_gain=self.cfg.vin_drop_gain or None, vin_min=self.cfg.vin_min,
            )
            ctrl.mujoco_data = sim.data
            ctrl.last_ts = sim.data.time
            ctrl.reset(sim.data.qpos)
            self._ctrl, self._generation = ctrl, sim.generation
        return self._ctrl

    def step(self, sim: SimManager) -> None:
        """The step callback's body: never raises while the robot is absent (step_once drops a
        callback that raises)."""
        if not self.present(sim):
            return
        ctrl = self.ensure_bound(sim)
        ctrl.update()
        if not self.torque:  # limp: no motor torque; BAM's friction on the back-driven gears stays
            sim.data.ctrl[ctrl.act_indexes] = 0.0

    def set_targets(self, sim: SimManager, targets: np.ndarray) -> None:
        self.ensure_bound(sim).q_target[:] = targets

    def set_torque(self, sim: SimManager, on: bool) -> None:
        ctrl = self.ensure_bound(sim)
        if on and not self.torque:
            ctrl.reset(sim.data.qpos)  # don't fling the robot at a stale target
        self.torque = on

    def targets(self, sim: SimManager) -> np.ndarray:
        return self.ensure_bound(sim).q_target.copy()
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py -k "bam_config or rewrite" -q`
Expected: PASS. If `actuator_forcelimited` differs, that's because Pollen's `True` and our
`mjLIMITED_TRUE` are both 1. If `pytest.approx` chokes on int arrays, compare those fields with
`np.array_equal`.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/bam.py tests/test_floating_robot.py
git commit -m "feat: bam.py, BAM actuator config, spec rewrite (port of Pollen's) and a rebinding drive"
```

---

### Task 5: The robot's `actuators` attribute runs BAM

**Files:**
- Modify: `src/mujoco_module/robot_file.py` (`RobotHandle.__init__`, `close`)
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
import copy

from mujoco_module.sim import WorldConfig

BAM = {"model": "bam"}


@pytest.fixture
def sim5(sim):
    """The sim fixture's world at BAM's 5 ms step."""
    sim.configure_world(WorldConfig.from_attrs({"physics_dt": 0.005}))
    return sim


def test_bam_drives_the_robot_and_holds_a_stand(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    assert sim5.model.actuator("duck/l_hip").gainprm[0] == 1.0  # a motor now
    sim5.advance(0.005)
    # binding must not move anything (MujocoController's mj_setConst resets the data it's given)
    assert duck.drive.targets(sim5) == pytest.approx(STAND, abs=1e-9)
    assert free_qpos(sim5)[2] == pytest.approx(TRUNK_Z, abs=0.002)
    sim5.advance(1.0)
    adr = [sim5.model.joint(f"duck/{n}").qposadr[0] for n in ("l_hip", "l_knee", "r_hip", "r_knee")]
    assert sim5.data.qpos[adr] == pytest.approx(STAND, abs=0.05)  # held near the keyframe
    assert duck.io(None, None)["gravity"][2] < -0.99  # still upright: the fixture stands
    assert np.any(sim5.data.ctrl != 0.0)  # BAM is writing torques


def test_bam_rollout_equals_the_controller_run_directly(sim5, floater):
    """Same model, same start: mujoco-sim's BAM callback and a bare MujocoController agree."""
    from bam.mujoco import MujocoController

    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.005)  # bind
    ref_data = mujoco.MjData(sim5.model)
    # construct first (its mj_setConst resets ref_data), then copy the live state in exactly
    ref = MujocoController(duck.drive.cfg.load(), list(duck.drive.names), sim5.model, ref_data,
                           vin_drop_gain=0.1, vin_min=6.0)
    mujoco.mj_copyData(ref_data, sim5.model, sim5.data)
    ref.q_target[:] = duck.drive.targets(sim5)
    ref._prev_motor_torque[:] = duck.drive._ctrl._prev_motor_torque
    ref.last_ts = duck.drive._ctrl.last_ts
    steps = 200
    sim5.advance(steps * 0.005)
    for _ in range(steps):
        ref.update()
        mujoco.mj_step(sim5.model, ref_data)
    assert sim5.data.qpos == pytest.approx(ref_data.qpos, abs=1e-9)


def test_adding_a_limb_rebinds_bam_to_the_new_model(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    first = duck.drive._ctrl
    ArmHandle(sim5, "left-leg", leg("l", home_joints=[0.2, -0.5]))  # forces a rebuild
    sim5.advance(0.005)
    assert duck.drive._ctrl is not first and duck.drive._generation == sim5.generation
    dof = sim5.model.joint("duck/l_hip").dofadr[0]
    assert sim5.model.dof_armature[dof] == pytest.approx(duck.drive.cfg.load().actuator.get_extra_inertia())
    assert duck.drive.targets(sim5)[:2] == pytest.approx([0.2, -0.5], abs=1e-6)  # the arm's init pose


def test_bam_warns_off_its_fitted_step(sim, floater, caplog):
    robot(sim, floater, actuators=BAM)  # the fixture's world runs at 2 ms
    assert "fitted at 0.005" in caplog.text
```

`ref._prev_motor_torque` and `ref.last_ts` are private BAM state. Copying them makes both rollouts
start identically. If `MujocoController` at 62bd8ce names them differently, read `bam/mujoco.py`
and copy whatever `reset`/`update` keep between steps. Keep the order above: construct the
reference controller, then `mj_copyData`. Copying fields first and constructing after wipes them
(`mj_setConst`), and even `copy.copy` before construction diverged by about 4e-6 over 200 steps.
Constructing first, then `mj_copyData`, matched exactly in a probe.

Drop the now-unused `import copy` from this block.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k "bam_drives or rollout or rebinds or warns" -q`
Expected: FAIL. `RobotHandle` has no `drive`, and `actuators` is ignored.

- [ ] **Step 3: Implement**

In `robot_file.py`, add `import logging` and `from .bam import FIT_DT, BamConfig, BamDrive, rewrite_spec`,
and set `LOGGER = logging.getLogger(__name__)`. In `RobotHandle.__init__`, after `check_robot_file`
and before `self.model = spec.copy().compile()`, add:

```python
        self.drive: BamDrive | None = None
        if attrs.get("actuators") is not None:
            cfg = BamConfig.from_attrs(attrs["actuators"], label)
            names = rewrite_spec(spec, cfg, cfg.load())  # before compile, so self.model has motors too
            self.drive = BamDrive(cfg, [self.prefix + n for n in names])
            self._bam_names = tuple(names)  # file names, io's "joints" order
            dt = sim.run(lambda: sim.cfg.physics_dt if sim.cfg is not None else None)
            if dt is not None and abs(dt - FIT_DT) > 1e-9:
                LOGGER.warning(
                    "%s: BAM's servo model is fitted at 0.005 s physics steps; the world runs at %s "
                    "(set the world's physics_dt to 0.005)", label, dt,
                )
        self._callback_key = f"robot-bam:{name}"
```

Change `register` to also add the step callback:

```python
        def register() -> None:
            sim.robots[name] = self
            if self.drive is not None:
                drive = self.drive
                sim.step_callbacks[self._callback_key] = lambda _dt: drive.step(sim)
```

Change `close`'s `detach` to pop it:

```python
        def detach() -> None:
            self._sim.step_callbacks.pop(self._callback_key, None)
            if self._sim.robots.get(self.name) is self:
                del self._sim.robots[self.name]
```

Note on `_home`: it writes `data.ctrl` for unclaimed actuators from the keyframe's `ctrl`. Under BAM
those are motors, and `update()` overwrites `ctrl` before the first `mj_step`, so it's harmless.
Don't special-case it.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py -q`
Expected: PASS.

If `test_bam_drives_the_robot_and_holds_a_stand` fails, check the fixture before touching the
tolerance:
- run 2 s and print the trunk z and the gravity vector;
- the trunk should settle within about 2 mm of `TRUNK_Z` and stay upright;
- if it tips, widen the foot boxes (size x 0.06), or move the feet under the trunk's centre of mass.

The test checks that BAM holds a pose, not a balance controller.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/robot_file.py tests/test_floating_robot.py
git commit -m "feat: robot actuators: bam drives a robot file's servos, rebinding on every rebuild"
```

---

### Task 6: Limb arms on a BAM robot are read-only

**Files:**
- Modify: `src/mujoco_module/handles.py` (`ArmHandle`: `__init__` robot branch, `add()`, `get_commanded_positions`, `set_joint_targets`, `start_trajectory`, `extend_trajectory`, `stop_trajectory`, `stop`)
- Modify: `src/mujoco_module/robot_file.py` (add `RobotHandle.commanded`)
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
from mujoco_module.handles import JointTrajectoryPoint


def test_limbs_of_a_bam_robot_are_read_only(sim5, floater):
    robot(sim5, floater, actuators=BAM)
    left = ArmHandle(sim5, "left-leg", leg("l"))
    assert "arm:left-leg" not in sim5.step_callbacks  # its position hold would fight BAM for ctrl
    for call in (lambda: left.set_joint_targets([0, 0]),
                 lambda: left.start_trajectory([JointTrajectoryPoint(0.1, (0.0, 0.0))]),
                 lambda: left.extend_trajectory([JointTrajectoryPoint(0.1, (0.0, 0.0))]),
                 left.stop_trajectory):
        with pytest.raises(RuntimeError, match="read-only: robot duck's io drives it"):
            call()
    left.stop()  # Stop (session end, e-stop) must not fail
    sim5.advance(0.1)
    assert left.get_commanded_positions() == pytest.approx(STAND[:2], abs=1e-6)  # BAM's q_target
    assert left.get_joint_positions() == pytest.approx(STAND[:2], abs=0.05)


def test_a_limb_outside_bams_joints_is_an_error(sim5, floater):
    robot(sim5, floater, actuators={"model": "bam", "joints": "^r_"})
    with pytest.raises(ValueError, match="isn't driven by robot duck's BAM actuators"):
        ArmHandle(sim5, "left-leg", leg("l"))
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k "read_only or outside" -q`
Expected: FAIL.

- [ ] **Step 3: Implement**

In `RobotHandle` (`robot_file.py`), add:

```python
    def commanded(self, names: Sequence[str]) -> list[float]:
        """BAM's targets for these scene actuator names (sim thread)."""
        assert self.drive is not None
        targets = self.drive.targets(self._sim)
        return [float(targets[self.drive.names.index(n)]) for n in names]
```

In `ArmHandle.__init__`, set `self._bam_robot = None` near the top, before `if self.robot:`. After
`self._actuator_names = ...` is computed, add:

```python
        if holder is not None and holder.drive is not None:
            missing = [n for n in self._actuator_names if n not in holder.drive.names]
            if missing:
                raise ValueError(
                    f"{label}: actuator {missing[0].removeprefix(self.file_prefix)!r} isn't driven by "
                    f"robot {self.robot}'s BAM actuators (its actuators.joints); a limb of a BAM robot is read-only"
                )
            self._bam_robot = holder
```

In `add()`, register the position hold only for a writable arm:

```python
            if self._bam_robot is None:
                sim.step_callbacks[self._callback_key] = self._on_step
```

Add a helper and guard the writers:

```python
    def _check_writable(self) -> None:
        if self._bam_robot is not None:
            raise RuntimeError(
                f"arm {self.name!r} is read-only: robot {self.robot}'s io drives it (BAM actuators)"
            )
```

- First line of `set_joint_targets`, `start_trajectory`, `extend_trajectory` and `stop_trajectory`:
  `self._check_writable()`.
- First line of `stop`: `if self._bam_robot is not None: return`.
- In `get_commanded_positions`'s `read`, before reading `data.ctrl`:

```python
            if self._bam_robot is not None:
                self._now()  # raises while the arm is out of the scene
                return self._bam_robot.commanded(self._actuator_names)
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py tests/test_robot_file.py tests/test_arm.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/handles.py src/mujoco_module/robot_file.py tests/test_floating_robot.py
git commit -m "feat: limb arms of a BAM robot are read-only; commanded positions are BAM's targets"
```

---

### Task 7: `io` on the robot handle

**Files:**
- Modify: `src/mujoco_module/robot_file.py` (add `RobotHandle.io`, `gravity_in_trunk`)
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_io_reads_one_instant_and_applies_targets(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    snap = duck.io(None, None)  # binds on the spot: no step has run yet
    assert snap["joints"] == ["l_hip", "l_knee", "r_hip", "r_knee"]
    assert snap["q"] == pytest.approx(STAND, abs=1e-9)
    assert snap["gravity"] == pytest.approx([0, 0, -1], abs=1e-9)
    assert snap["trunk_pos"] == pytest.approx([0, 0, TRUNK_Z]) and snap["generation"] == sim5.generation
    t0 = snap["t"]
    crouch = [0.25, -0.5, 0.25, -0.5]  # a small crouch: the feet tilt 0.1 rad, it stays up
    duck.io(crouch, True)
    sim5.advance(1.0)
    snap = duck.io(None, None)
    assert snap["t"] == pytest.approx(t0 + 1.0, abs=0.006)
    assert snap["q"] == pytest.approx(crouch, abs=0.05)
    assert snap["gravity"][2] < -0.98


def test_io_after_a_rebuild_acts_on_the_new_model(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.01)
    ArmHandle(sim5, "left-leg", leg("l", home_joints=[0.2, -0.5]))  # rebuild; no step after it
    snap = duck.io([0.2, -0.5, 0.15, -0.3], None)  # binds to the new model itself
    assert duck.drive._generation == sim5.generation == snap["generation"]
    assert snap["q"][:2] == pytest.approx([0.2, -0.5], abs=1e-6)


def test_io_gravity_and_gyro_are_in_the_trunk_frame(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    adr = sim5.model.joint("duck/root").qposadr[0]
    dof = sim5.model.joint("duck/root").dofadr[0]
    sim5.data.qpos[adr + 3 : adr + 7] = [0.7071068, 0.7071068, 0, 0]  # rolled +90 deg about x
    sim5.data.qvel[dof + 3 : dof + 6] = [0.1, 0.2, 0.3]
    mujoco.mj_forward(sim5.model, sim5.data)
    snap = duck.io(None, None)
    assert snap["gravity"] == pytest.approx([0, -1, 0], abs=1e-6)  # world -z seen from the rolled trunk
    assert snap["gyro"] == pytest.approx([0.1, 0.2, 0.3])  # free-joint angular qvel: body frame


def test_torque_off_collapses_slowly_and_torque_on_resets_targets(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    sim5.advance(0.2)
    standing = duck.io(None, None)["trunk_pos"][2]
    duck.io(None, False)
    sim5.advance(0.05)
    assert np.all(sim5.data.ctrl[[sim5.model.actuator(f"duck/{n}").id for n in ("l_hip", "l_knee")]] == 0.0)
    sim5.advance(0.45)
    sagged = duck.io(None, None)
    assert sagged["trunk_pos"][2] < standing - 0.005  # it collapsed
    duck.io(None, True)
    assert duck.drive.targets(sim5) == pytest.approx(sagged["q"], abs=1e-9)  # not flung at STAND


def test_io_validates(sim5, floater):
    duck = robot(sim5, floater, actuators=BAM)
    with pytest.raises(ValueError, match="expected 4 targets"):
        duck.io([0.0], None)
    plain = RobotHandle(sim5, "pd", {"mjcf": str(floater), "position": [2, 0, 0]})
    with pytest.raises(ValueError, match="io needs actuators"):
        plain.io(None, None)
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k io -q`
Expected: FAIL with `AttributeError: 'RobotHandle' object has no attribute 'io'`.

- [ ] **Step 3: Implement**

In `robot_file.py`, add the module-level function (it is `body_server.py`'s `gravity_in_trunk`,
microduck_rl@0daab16):

```python
def gravity_in_trunk(quat: np.ndarray) -> np.ndarray:
    """World gravity (unit) in the trunk frame; upright is [0, 0, -1]. What a legged policy
    observes, and what the real robot's IMU filter reports."""
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    return -rotation.reshape(3, 3).T[:, 2]
```

Add the method on `RobotHandle`:

```python
    def io(self, targets: Sequence[float] | None, torque: bool | None) -> dict[str, Any]:
        """One control tick's exchange, on the sim thread between physics steps: apply torque on/off
        and BAM position targets (in "joints" order; None leaves them), then read every joint and
        the trunk at that one instant."""
        drive = self.drive
        if drive is None:
            raise ValueError(f"robot {self.name}: io needs actuators (BAM)")
        wanted = None
        if targets is not None:
            wanted = np.asarray([float(v) for v in targets])
            if wanted.shape != (len(drive.names),):
                raise ValueError(f"robot {self.name}: expected {len(drive.names)} targets, got {wanted.size}")

        def exchange() -> dict[str, Any]:
            sim = self._sim
            if torque is not None:
                drive.set_torque(sim, bool(torque))
            if wanted is not None:
                drive.set_targets(sim, wanted)
            ctrl = drive.ensure_bound(sim)
            model, data = sim.model, sim.data
            out: dict[str, Any] = {
                "t": float(data.time),
                "q": data.qpos[ctrl.qpos_indexes].tolist(),
                "dq": data.qvel[ctrl.dof_indexes].tolist(),
                "joints": list(self._bam_names),
                "generation": int(sim.generation),
                "torque": drive.torque,
            }
            if self.floating_root:
                joint = model.body_jntadr[model.body(self.prefix + self.floating_root).id]
                qa, da = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
                quat = data.qpos[qa + 3 : qa + 7]
                out["trunk_pos"] = data.qpos[qa : qa + 3].tolist()
                out["trunk_quat_wxyz"] = quat.tolist()
                out["gyro"] = data.qvel[da + 3 : da + 6].tolist()  # free joint: body-frame angular velocity
                out["gravity"] = gravity_in_trunk(quat).tolist()
            return out

        return self._sim.run(exchange)
```

In `test_io_validates`, the second `RobotHandle` named `pd` is a second floating robot in the same
world. If that trips anything (e.g. both float on the same floor), it's fine: it only needs to
construct.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest tests/test_floating_robot.py -q`
Expected: PASS. If the collapse test fails because BAM's friction holds the floater up, lower the
threshold to the measured sag over 0.5 s with torque on vs. off, and assert off sags more than on.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/robot_file.py tests/test_floating_robot.py
git commit -m "feat: robot io, one atomic exchange of BAM targets, joint state, trunk gyro and gravity"
```

---

### Task 8: The `io` DoCommand and `actuators` validation on the model

**Files:**
- Modify: `src/mujoco_module/models/robot.py`
- Test: `tests/test_floating_robot.py`

- [ ] **Step 1: Write the failing tests**

Append. This runs through the real component on the `running_sim` fixture, as production does:

```python
import asyncio

from viam.proto.app.robot import ComponentConfig
from google.protobuf.struct_pb2 import Struct

from mujoco_module.models.robot import MujocoRobot
from mujoco_module.models.world import MujocoWorld


def component(name, model, attrs, api="rdk:component:generic"):
    struct = Struct()
    struct.update(attrs)
    return ComponentConfig(name=name, api=api, model=model, attributes=struct)


def test_io_do_command(running_sim, floater):
    async def run():
        world = MujocoWorld.new(component("sim-world", "viam-labs:mujoco-sim:world", {"physics_dt": 0.005}), {})
        duck = MujocoRobot.new(
            component("duck", "viam-labs:mujoco-sim:robot", {"world": "sim-world", "mjcf": str(floater), "actuators": BAM}), {})
        first = await duck.do_command({"io": {"targets": None}})
        second = await duck.do_command({"io": {"targets": STAND, "torque": True}})
        await duck.close(); await world.close()
        return first, second

    first, second = asyncio.run(run())
    assert first["joints"] == ["l_hip", "l_knee", "r_hip", "r_knee"]
    assert second["t"] >= first["t"] and len(second["gravity"]) == 3


def test_validate_rejects_bad_actuators(floater):
    with pytest.raises(ValueError, match="unknown actuators keys"):
        MujocoRobot.validate_config(component(
            "duck", "viam-labs:mujoco-sim:robot",
            {"world": "sim-world", "mjcf": str(floater), "actuators": {"model": "bam", "kp": 1}}))
```

Check `tests/test_robot_file.py` or `tests/test_models.py` for an existing helper that builds a
`ComponentConfig` and a world component in `running_sim`, and reuse it instead of `component` above
if one exists. Match how those tests construct `MujocoWorld` (class name and constructor). The names
here are the expected ones; confirm against `models/world.py`.

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `uv run python -m pytest tests/test_floating_robot.py -k "do_command or validate" -q`
Expected: FAIL. `do_command` is unimplemented on `MujocoRobot`, and `validate_config` ignores
`actuators`.

- [ ] **Step 3: Implement**

In `models/robot.py`:
- Extend the module docstring's attribute list:

```
  actuators (object)       - {"model": "bam", ...}: BAM servo physics for the matching actuators
                             (motor, bam_model, kp_fw, vin, vin_drop_gain, vin_min, joints); see README
```

- Add `from ..bam import BamConfig` and `from viam.utils import ValueTypes`.
- In `validate_config`, after `_source(...)`:

```python
        attrs = get_attrs(config)
        if attrs.get("actuators") is not None:
            BamConfig.from_attrs(attrs["actuators"], config.name)
```

  If `get_attrs(config)` is already called, reuse it.

- Add to `MujocoRobot`:

```python
    async def do_command(
        self, command: Mapping[str, ValueTypes], *, timeout: float | None = None, **kwargs
    ) -> Mapping[str, ValueTypes]:
        if "io" in command:
            if self._handle is None:
                raise RuntimeError(f"robot {self.name} is closed")
            request = command["io"] or {}
            return await asyncio.to_thread(self._handle.io, request.get("targets"), request.get("torque"))
        raise ValueError(f"robot {self.name}: unknown command {sorted(command)}; known: ['io']")
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `uv run python -m pytest -q`
Expected: the whole suite passes. The pass count is the Task 1 baseline plus this file's tests.

- [ ] **Step 5: Commit**

```bash
git add src/mujoco_module/models/robot.py tests/test_floating_robot.py
git commit -m "feat: robot io DoCommand; actuators validated at config time"
```

---

### Task 9: README and the pull request

**Files:**
- Modify: `README.md` (section "Model viam-labs:mujoco-sim:robot")

- [ ] **Step 1: Document**

In the robot section's attribute table, add the `actuators` row. Replace the sentence "mocap bodies,
plugins, flexes, skins and free joints are errors" with the floating-base rule. Then add a
"Floating base, BAM servos and io" subsection containing:
- **Floating base.** One free joint, on the file's single root body. The robot must be framed on
  `world`. The keyframe's trunk pose is applied in the robot's frame. Limb arms are framed on the
  trunk, and their config error gives the values to paste. The robot's own Viam frame stays where
  it is configured, so read the trunk's pose from `io`.
- **`actuators`.** The JSON block from spec unit 2, with its defaults. It says that the matching
  actuators become BAM torque motors (Rhoban's BAM, Apache-2.0), that `physics_dt` should be 0.005,
  and that limb arms are then read-only.
- **The `io` DoCommand.** The request and response from spec unit 3, and that it's one exchange at
  one sim instant. `generation` changes when the scene is rebuilt, so a controller should stop if it
  changes. Torque off goes limp, with gear friction.

- [ ] **Step 2: Lint and full suite**

Run: `uv run ruff check src tests && uv run python -m pytest -q`
Expected: no lint errors, and all tests pass.

- [ ] **Step 3: Commit and open the PR (ask the user first)**

```bash
git add README.md
git commit -m "docs: floating base, BAM actuators and io on the robot model"
```

Ask the user before `git push -u origin floating-base-bam` and `gh pr create`. The repo is public
(`viam-labs`), so this is outward-facing. PR body: what each task added, the Python 3.12 bump (or
the vendoring fallback, if Task 1 took it), and that the MicroDuck itself isn't bundled (NC
meshes). Releasing a module version is also the user's call, after merge. Milestone 2 needs a
released version that the duck machine can pin.

---

### Task 10: The duck scene package (this repo)

**Files:**
- Create: `scripts/export_duck_scene.py`
- Create: `docs/superpowers/specs/duck-sim-machine.json`

- [ ] **Step 1: Write the exporter**

`scripts/export_duck_scene.py`:

```python
"""Package the MicroDuck walk model for viam-labs:mujoco-sim as a private Viam archive.

    uv run python scripts/export_duck_scene.py <microduck_rl checkout> [out_dir]   (default dist/duck-sim)

The checkout must be at MICRODUCK_RL_COMMIT. The meshes are CC BY-NC-SA 4.0 (microduck_rl's
LICENSE-MODELS.md): the package stays private to the org, and is never bundled in a public module.

Writes out_dir/ with robot_walk.xml, its assets/ meshes, LICENSE-MODELS.md and duck.xml: a wrapper
that includes robot_walk.xml and adds the STAND keyframe from scene_walk.xml as its first keyframe,
the robot's home (mujoco-sim homes a robot file from its first keyframe). Then out_dir.tar.gz,
for `viam packages upload`.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

MICRODUCK_RL_COMMIT = "0daab160d3fb1c2cbaba16a4a19052f36709dd9e"
MODEL_DIR = Path("src/mjlab_microduck/robot/microduck")


def stand_keyframe(scene_xml: str) -> str:
    """scene_walk.xml's STAND <key .../> element, verbatim. Comments are stripped first: the file
    keeps the old STAND commented out above the STAND2-tuned one (hip_pitch 0.4579, not 0.5236)."""
    live = re.sub(r"<!--.*?-->", "", scene_xml, flags=re.S)
    match = re.search(r'<key\s+name="STAND".*?/>', live, re.S)
    if match is None:
        raise SystemExit("scene_walk.xml has no STAND keyframe; is the checkout at the pinned commit?")
    return match.group(0)


def main(checkout: Path, out: Path) -> None:
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()
    if head != MICRODUCK_RL_COMMIT:
        raise SystemExit(f"{checkout} is at {head}; check out {MICRODUCK_RL_COMMIT}")
    src = checkout / MODEL_DIR
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copy(src / "robot_walk.xml", out / "robot_walk.xml")
    shutil.copytree(src / "assets", out / "assets", ignore=shutil.ignore_patterns("*.part"))
    shutil.copy(checkout / "LICENSE-MODELS.md", out / "LICENSE-MODELS.md")
    key = stand_keyframe((src / "scene_walk.xml").read_text())
    (out / "duck.xml").write_text(
        '<mujoco model="microduck">\n  <include file="robot_walk.xml"/>\n'
        f"  <keyframe>\n    {key}\n  </keyframe>\n</mujoco>\n"
    )
    archive = out.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(out.rglob("*")):
            tar.add(path, arcname=path.relative_to(out))
    print(f"wrote {out} and {archive}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist/duck-sim"))
```

- [ ] **Step 2: Run it**

```bash
git clone https://github.com/pollen-robotics/microduck_rl.git "${TMPDIR:-/tmp}/microduck_rl" 2>/dev/null || true
git -C "${TMPDIR:-/tmp}/microduck_rl" fetch -q origin 0daab160d3fb1c2cbaba16a4a19052f36709dd9e
git -C "${TMPDIR:-/tmp}/microduck_rl" checkout -q 0daab160d3fb1c2cbaba16a4a19052f36709dd9e
uv run python scripts/export_duck_scene.py "${TMPDIR:-/tmp}/microduck_rl"
```

Expected: `wrote dist/duck-sim and dist/duck-sim.tar.gz`. `dist/` is already git-ignored; check
with `git check-ignore dist/duck-sim`.

Check the keyframe: `grep -o 'qpos="[^"]*' dist/duck-sim/duck.xml | tr -s ' \n' ' '` must contain
`-0.457924`. That's the STAND2 hip pitch, not the old 0.5236.

- [ ] **Step 3: Write the example machine config**

`docs/superpowers/specs/duck-sim-machine.json` uses the field shapes of the README's robot and arm
examples:
- **World:** `physics_dt` 0.005.
- **Robot `duck`:** `mjcf: "${packages.archive.duck-sim}/duck.xml"`, `actuators: {"model": "bam"}`,
  `frame.parent: "world"`.
- **Arms:**
  - `left-leg`: joints `left_hip_yaw`, `left_hip_roll`, `left_hip_pitch`, `left_knee`,
    `left_ankle`; `flange_site: "left_foot"`.
  - `right-leg`: the right joints; `flange_site` is the right foot's site. Read its name from
    `robot_walk.xml`, near `ankle_right`.
  - `head`: `neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`; `flange_site: "head_camera"`.

  Each arm has `robot: "duck"` and `frame.parent: "duck"`. Fill in each arm's `frame.translation`
  and `frame.orientation` from the arm's own config error, the first time Task 11 constructs it.
- **Package entry:** `{"name": "duck-sim", "package": "<org>/duck-sim", "type": "archive",
  "version": "0.1.0"}`.
- **Module entry:** `viam-labs:mujoco-sim`, at the version released after Task 9's PR merges.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_duck_scene.py docs/superpowers/specs/duck-sim-machine.json
git commit -m "feat: duck-sim archive package from microduck_rl@0daab16 (STAND home, NC meshes kept private)"
```

---

### Task 11: The real duck stands under BAM (local check, then upload)

**Files:**
- Create: `scripts/duck_stand_check.py`

- [ ] **Step 1: Write the check**

`scripts/duck_stand_check.py` runs mujoco-sim's handles in-process against the exported package,
with no viam-server. Run it with mujoco-sim's own environment:
`uv run --project ../viam-mujoco-sim python scripts/duck_stand_check.py dist/duck-sim`.

```python
"""The MicroDuck in viam-labs:mujoco-sim with BAM: does it load, home at STAND and stand for 5 s,
and does io report the policy's joint order? Run with mujoco-sim's environment:

    uv run --project ../viam-mujoco-sim python scripts/duck_stand_check.py dist/duck-sim

Prints each limb's frame error (the values for duck-sim-machine.json) the first time through.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "viam-mujoco-sim" / "src"))

from mujoco_module.handles import ArmHandle  # noqa: E402
from mujoco_module.robot_file import RobotHandle  # noqa: E402
from mujoco_module.sim import SimManager, WorldConfig  # noqa: E402

# The policies' joint order (microduck_rl README "Joint layout"): left leg, neck/head, right leg
POLICY_ORDER = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
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
        try:
            ArmHandle(sim, name, {"robot": "duck", "parent": "duck", "joints": joints, "flange_site": site})
        except ValueError as error:
            print(f"{name}: {error}")  # paste into duck-sim-machine.json, then rerun
    z0 = duck.io(None, None)["trunk_pos"][2]
    sim.advance(5.0)
    snap = duck.io(None, None)
    print(f"trunk z {z0:.3f} -> {snap['trunk_pos'][2]:.3f} m, gravity {snap['gravity']}")
    assert snap["gravity"][2] < -0.95, "it fell over"
    assert snap["trunk_pos"][2] > z0 - 0.02, "it sank more than 2 cm"
    print("stands under BAM for 5 s")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
```

If a limb's error is about the frame, copy the printed translation and orientation into
`duck-sim-machine.json` and pass them as `position`/`orientation_wxyz` (metres, w-first) in `LIMBS`
here. Rerun until all three limbs bind. Fix `right_foot` if the site has another name (see Task 10
Step 3).

- [ ] **Step 2: Run it**

Run: `uv run --project ../viam-mujoco-sim python scripts/duck_stand_check.py dist/duck-sim`
Expected: "io joint order matches the policy's", then "stands under BAM for 5 s".

If it falls, standing still with no policy is plausible for an 800 g biped with feet this small.
Before changing anything, check two things:
- that the trunk starts at STAND's 0.12 m and not inside the floor;
- that `physics_dt` is 0.005.

If it still tips, relax the check to "stands for 1 s". A policy keeps it up in milestone 2. Record
what happened in the commit message.

- [ ] **Step 3: Upload the package (DevRelDemos org, private)**

```bash
viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=duck-sim --version=0.1.0 \
  --type=archive --path=dist/duck-sim.tar.gz
```

This is inside the user's standing authorization for this project's Viam uploads, which covers
DevRelDemos. If the CLI asks for visibility, keep it private. Never re-upload an existing version;
bump it instead.

- [ ] **Step 4: Commit**

```bash
git add scripts/duck_stand_check.py docs/superpowers/specs/duck-sim-machine.json
git commit -m "feat: duck stand check (io joint order, 5 s under BAM); duck-sim 0.1.0 uploaded"
```

---

## Deferred from the spec, on purpose

- **3D viewer showing the trunk walk** (spec unit 1). It isn't needed for milestone 2. Check
  `world_state` streaming when there's a reason to.
- **The rewrite on the real `robot_walk.xml`** (spec testing). The meshes can't live in
  mujoco-sim's repo. The synthetic floater test compares against Pollen's rewrite code itself, and
  Task 11 runs the real file.

## Done when

- viam-mujoco-sim's `floating-base-bam` passes its full suite. The PR is open, with the user's
  go-ahead.
- `duck_stand_check.py` passes on the exported package, and `duck-sim` 0.1.0 is in DevRelDemos.
- `duck-sim-machine.json` has the real limb frames.

Milestone 2 (duck-policy and walk parity) gets its own plan once a mujoco-sim release with this
work exists.
