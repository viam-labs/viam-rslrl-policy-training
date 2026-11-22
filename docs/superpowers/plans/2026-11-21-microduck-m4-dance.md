# MicroDuck milestone 4: an in-place dance by imitation — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** choreograph a looping in-place dance on a Viam sim machine, record it into a sequence dataset, train an imitation policy as a Viam training job, and run it with duck-policy on the preview machine, scored against a headless reference.

**Architecture:** spec `docs/superpowers/specs/2026-11-21-microduck-dance-design.md`. It has three parts, built in order:
1. **Choreographing and recording:** a pinned duck, `duck_choreo.py`, and a sequence dataset.
2. **Training:** a reference builder, the `Mjlab-Dance-Flat-MicroDuck` task, and `task=dance` in `microduck-train`.
3. **Running and evaluating:** phase mode in duck-policy, a `dancer` service, and a phase parity check.

The observation stays at 61 values. The dance is driven by a phase clock in the twist slot of the command.

**Tech stack:**
- Python 3.12, uv, numpy, MuJoCo (local); torch for the reward tests
- pandas and pyarrow (new dev dependencies; also added to the training image)
- viam-sdk 0.85 (`DataClient.create_sequence`, arm DoCommands)
- mjlab 1.3.0 and microduck_rl@0daab16 (image only)
- the `viam` CLI, Docker (linux/amd64, Docker Hub `hipsterbrown/`)

**Facts this plan relies on** (checked 2026-11-21; `MD` = microduck_rl@0daab16, at `$TMPDIR/gh-clones-3856841f-0045-481c-bb70-ba5b784e7265/microduck_rl`):
- **`MD/src/mjlab_microduck/tasks/mdp.py`:**
  - `GroundPickPhaseCommand` (about line 5210) advances `_gp_phase` by `dt/period` on every `compute`.
  - `reset(env_ids)` randomizes it when `randomize_phase`.
  - `vel_command_b[:, 0:3] = (cos 2πφ, sin 2πφ, 0)`.
  - `GroundPickPhaseCommandCfg(UniformVelocityCommandCfg)` is a `kw_only` dataclass with `period` and `randomize_phase`.
  - Ground pick builds it as `GroundPickPhaseCommandCfg(**{**vars(command), "class_type": GroundPickPhaseCommand, "period": P})` (`microduck_ground_pick_env_cfg.py:465`).
- **Contact sensor:** `env.scene.sensors["feet_ground_contact"].data.found` is `(num_envs, n_feet)` (`mdp.py:347`, `:1780`).
- **Task registration:** `register_mjlab_task(task_id=, env_cfg=, play_env_cfg=, rl_cfg=, runner_cls=)` from `mjlab.tasks.registry`. The velocity task uses `make_microduck_velocity_env_cfg(play=False, rough=False)`, `MicroduckRlCfg` (`microduck_velocity_env_cfg.py:912`) and `MicroduckOnPolicyRunner` (`MD/src/mjlab_microduck/tasks/__init__.py`).
- **Velocity task names:**
  - Commands are `twist`, `head_pose` and `body_pose`; the last two are `UniformPoseCommandCfg` with `ranges`.
  - Rewards removed for the dance (all present in the velocity config): `track_linear_velocity`, `track_angular_velocity`, `air_time`, `foot_clearance`, `foot_swing_height`, `pose`, `head_pose_tracking`, `head_pose_bias`, `body_pose_tracking`.
  - Curricula removed: `action_rate_weight`, `standing_envs`, `head_pose_range`, `body_pose_range`, `head_pose_bias_weight`.
- **Export:** `run_export(task_id, ExportConfig)` loads `load_env_cfg(task_id, play=True)`, so the dance task must be registered in the same process before export.
- **mujoco-sim arms (README in `~/src/viam-mujoco-sim-floating`):**
  - `GetJointPositions` returns degrees.
  - The DoCommands `get_joint_positions_radians` → `{values}` and `set_joint_targets_radians {values}` latch the target and return at once.
  - A floating robot without `actuators` has no `io`, but the pinned duck isn't floating.
- **viam-sdk 0.85.2 `DataClient`:** `create_sequence(part_id, resources: [SequenceResourceFilter], start_time, end_time, sequence_tags)`, `add_sequences_to_dataset(dataset_id, sequence_ids)`, and `create_dataset(name, organization_id, type)`.
- **Sequence-dataset Parquet (docs.viam.com/train/dataset-formats):**
  - `sequences.parquet` has `sequence_id`, `tags` (list), `start_at` and `end_at` (timestamps in µs, UTC).
  - `tabular_data.parquet` has `sequence_id`, `timestamp` (µs, UTC), `part_id`, `component_name`, `method_name` and `payload` (a JSON string).
  - The docs don't show the `JointPositions` payload or how component names appear. **Task 5 pins both from a real export.**
  - A job on a sequence dataset gets `--binary_data_file`, `--tabular_data_file` and `--sequences_file`.
- **duck-policy timing:** `loop.py` runs tick k in grid slot `slot`, whose start is `t0 + (slot-1)*dt`. A late tick jumps `slot` to `floor((now-t0)/dt)+1`.
- **Existing IDs:** org `16518049-9dd3-479f-9644-c5d112aa42d8`; upload part `cab989eb-90a0-4dce-8fc6-2c50d0ed1f10`; microduck-train image `docker.io/hipsterbrown/viam-microduck-train:0.1.0-1` (container `accd222e-0f61-410b-af6e-cd209e8c4c0d`); script `microduck-train` v1.

**Standing rules:**
- Never print or commit `cell.json`, `viam.json` or keys.
- Never re-push a tag or re-upload a version; bump it instead.
- Machine creation and API-key creation need the user's confirmation. The controller does those, not implementers.

## File map

| File | Status | Responsibility |
|---|---|---|
| `duck_train/robot.py` | new | `JOINTS`, `DEFAULT_POSE`, `TRUNK_Z`: one home for constants that `train.py` and `dance.py` share |
| `duck_train/dance.py` | new | dance file load and validation, `interpolate`, `table_at`, `feet_world`, `foot_check`, `build_reference` (numpy, MuJoCo and pandas imported lazily) |
| `duck_train/dance_rewards.py` | new | torch-only reward math: `phase_of`, `table_at_torch`, `dance_pose`, `dance_pose_l1`, `feet_contact` |
| `duck_train/dance_task.py` | new | image-only: `DancePhaseCommand(Cfg)`, `make_dance_env_cfg`, `register_dance_task` |
| `duck_train/train.py` | modify | `--task`, `--dance`; task id threaded through; phase package |
| `duck_train/package.py` | modify | `build_config(command=…)` |
| `build_duck.sh`, `container/Dockerfile.duck` | modify | ship the new modules; pandas and pyarrow in the image |
| `scripts/export_duck_scene.py` | modify | also writes `robot_pinned.xml` and `duck_pinned.xml` |
| `scripts/duck_choreo.py` | new | `pose`, `preview`, `check`, `record` |
| `dances/<name>.json` | new | the first dance |
| `docs/superpowers/specs/duck-choreo-machine.json` | new | the choreo machine's config |
| `duck_policy/obs.py`, `loop.py`, `service.py` | modify | phase mode |
| `scripts/duck_reference.py` | modify | `--phase`, records `q` |
| `scripts/duck_parity.py` | modify | `--phase`, `--service`, dance report, `handover` |
| `docs/superpowers/specs/duck-sim-machine.json` | modify | `dance-model` and `dancer` |
| `tests/test_duck_dance.py` | new | dance file, interpolation, foot check, reference builder |
| `tests/test_duck_dance_rewards.py` | new | reward math with a fake env |
| `tests/test_duck_loop.py`, `test_duck_service.py`, `test_duck_obs.py`, `test_duck_train.py`, `test_duck_parity.py` | modify | phase mode and the new signatures |

Run tests with `uv run pytest -q`. The full suite passes at the start, and every task leaves it passing.

---

## Part 1: choreographing and recording

### Task 1: Shared constants, dance files, interpolation

**Files:**
- Create: `duck_train/robot.py`, `duck_train/dance.py`, `tests/test_duck_dance.py`
- Modify: `duck_train/train.py` (import the constants instead of defining them), `build_duck.sh` (copy `robot.py`)

- [ ] **Step 1: Move the constants.**
  1. Create `duck_train/robot.py` with `JOINTS` and `DEFAULT_POSE` copied verbatim from `duck_train/train.py`, plus `TRUNK_Z = 0.125` (from `scripts/duck_reference.py`).
  2. In `train.py`, replace the two definitions with `from .robot import DEFAULT_POSE, JOINTS`.
  3. In `build_duck.sh`, add `duck_train/robot.py` to the `cp duck_train/...` line.
  4. Run `uv run pytest -q`; expect it to pass unchanged.

- [ ] **Step 2: Write the failing tests** in `tests/test_duck_dance.py`:

```python
import json

import numpy as np
import pytest

from duck_train.dance import interpolate, load_dance, table_at, validate
from duck_train.robot import DEFAULT_POSE, JOINTS

STAND = np.array(DEFAULT_POSE)
BOB = {"name": "t", "period_s": 6.0, "keyframes": [
    {"t": 0.0, "pose": {}},
    {"t": 0.25, "pose": {"head_roll": 0.3}},
    {"t": 0.5, "pose": {}},
    {"t": 0.75, "pose": {"head_roll": -0.3}},
]}
ROLL = JOINTS.index("head_roll")


def test_hits_keyframes_and_defaults_to_stand():
    assert np.allclose(interpolate(BOB, 0.0), STAND)
    q = interpolate(BOB, 0.25)
    assert q[ROLL] == pytest.approx(0.3) and np.allclose(np.delete(q, ROLL), np.delete(STAND, ROLL))
    assert interpolate(BOB, 0.75)[ROLL] == pytest.approx(-0.3)


def test_wraps_continuously():
    a, b = interpolate(BOB, 0.999999), interpolate(BOB, 0.0)
    assert np.abs(a - b).max() < 1e-4
    assert np.allclose(interpolate(BOB, 1.25), interpolate(BOB, 0.25))


def test_smooth_between_keyframes():
    xs = np.linspace(0, 1, 401)
    roll = np.array([interpolate(BOB, x)[ROLL] for x in xs])
    assert np.abs(np.diff(roll)).max() < 0.02  # no jumps
    assert 0.0 < interpolate(BOB, 0.125)[ROLL] < 0.3


def test_clips_to_limits():
    limits = np.tile([-0.2, 0.2], (14, 1)).astype(float)
    limits[:, 0] = np.minimum(limits[:, 0], STAND - 0.01)
    limits[:, 1] = np.maximum(limits[:, 1], STAND + 0.01)
    assert interpolate(BOB, 0.25, limits)[ROLL] == pytest.approx(0.2)


@pytest.mark.parametrize("bad", [
    {"keyframes": [{"t": 0.0, "pose": {}}, {"t": 0.5, "pose": {}}, {"t": 0.4, "pose": {}}]},  # not increasing
    {"keyframes": [{"t": 0.1, "pose": {}}]},                                                   # doesn't start at 0
    {"keyframes": [{"t": 0.0, "pose": {}}, {"t": 1.0, "pose": {}}]},                          # t must be < 1
    {"keyframes": [{"t": 0.0, "pose": {"nope": 0.1}}]},                                        # unknown joint
    {"keyframes": [{"t": 0.0, "pose": {"head_roll": 0.2}}]},                                   # t=0 not STAND
    {"period_s": 3.0},                                                                          # period outside 4-8 s
])
def test_validate_rejects(bad):
    with pytest.raises(ValueError):
        validate({**BOB, **bad})


def test_load_roundtrip(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps(BOB))
    assert load_dance(p)["name"] == "t"


def test_table_at_interpolates_between_bin_centres():
    table = np.zeros((100, 14))
    table[:, 0] = np.arange(100)
    assert table_at(table, 0.005)[0] == pytest.approx(0.0)    # bin 0 centre
    assert table_at(table, 0.010)[0] == pytest.approx(0.5)    # halfway to bin 1
    assert table_at(table, 0.999)[0] == pytest.approx(99 * 0.6, abs=1.0)  # wraps toward bin 0
```

- [ ] **Step 3:** Run `uv run pytest tests/test_duck_dance.py -q`. Expect it to fail with an `ImportError`.

- [ ] **Step 4: Implement `duck_train/dance.py`** (the first half; `foot_check` comes in Task 2, `build_reference` in Task 6):

```python
"""MicroDuck dances (spec: docs/superpowers/specs/2026-11-21-microduck-dance-design.md). A dance is a looping
keyframe file; interpolate() turns it into 14 joint angles in the policy order at any phase in [0, 1). The
choreo tool, the reference builder's tests and the evaluator share this module. numpy only at import time:
MuJoCo and pandas are imported where they're used, so the training image and the Mac both load it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .robot import DEFAULT_POSE, JOINTS

STAND = np.asarray(DEFAULT_POSE, float)
STAND_TOL = 0.05  # rad: phase 0 must be STAND (start and hold pose)


def validate(dance: dict) -> dict:
    if not isinstance(dance.get("name"), str) or not dance["name"]:
        raise ValueError("dance needs a name")
    period = float(dance.get("period_s", 0))
    if not 4.0 <= period <= 8.0:
        raise ValueError(f"period_s {period} outside 4-8 s")
    kfs = dance.get("keyframes") or []
    ts = [float(k["t"]) for k in kfs]
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
    Passes through every keyframe; wraps from the last keyframe back to the first at t = 1."""
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
```

- [ ] **Step 5:** Run `uv run pytest tests/test_duck_dance.py -q`; expect it to pass. Then run `uv run pytest -q`; expect it all to pass.

- [ ] **Step 6: Commit**

```bash
git add duck_train/robot.py duck_train/dance.py duck_train/train.py build_duck.sh tests/test_duck_dance.py
git commit -m "feat: dance files and periodic interpolation; shared duck constants"
```

### Task 2: Foot kinematics and the foot check

**Files:**
- Modify: `duck_train/dance.py`, `tests/test_duck_dance.py`

The duck model used is `dist/duck-sim/duck.xml`, which already exists (git-ignored). Tests skip without it, the same way `tests/test_duck_train.py` skips without `VELSTAND`.

- [ ] **Step 1: Write the failing tests** (append):

```python
from pathlib import Path

from duck_train.dance import feet_world, foot_check

DUCK_XML = Path("dist/duck-sim/duck.xml")
needs_duck = pytest.mark.skipif(not DUCK_XML.exists(), reason="run scripts/export_duck_scene.py first")


@needs_duck
def test_feet_world_at_stand_level():
    feet = feet_world(DUCK_XML, STAND, trunk_pos=(0, 0, 0.125), trunk_quat=(1, 0, 0, 0))
    assert feet.shape == (2, 3) and abs(feet[0, 2] - feet[1, 2]) < 0.005


@needs_duck
def test_foot_check_passes_head_only_dance():
    assert foot_check(BOB, DUCK_XML) == []


@needs_duck
def test_foot_check_fails_one_knee():
    knee = JOINTS.index("left_knee")
    lift = {"name": "lift", "period_s": 6.0, "keyframes": [
        {"t": 0.0, "pose": {}},
        {"t": 0.5, "pose": {"left_knee": float(STAND[knee] + 0.8), "left_hip_pitch": float(STAND[2] + 0.6)}},
    ]}
    errors = foot_check(lift, DUCK_XML)
    assert any("lifts" in e for e in errors), errors
```

  (Checked against `dist/duck-sim/duck.xml`: `+0.8 / +0.6` lifts the left foot 2.4 cm; `−` only lowers it 0.9 cm.)

- [ ] **Step 2:** Run the new tests; expect them to fail with an `ImportError`.

- [ ] **Step 3: Implement** (append to `dance.py`):

```python
FREE_JOINT = "trunk_base_freejoint"
FOOT_SITES = ("left_foot", "right_foot")
_MODELS: dict = {}


def _model(duck_xml: Path):
    import mujoco

    key = str(Path(duck_xml).resolve())
    if key not in _MODELS:
        m = mujoco.MjModel.from_xml_path(key)
        _MODELS[key] = (m, mujoco.MjData(m))
    return _MODELS[key]


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
    Returns human-readable errors (empty = pass)."""
    m, _ = _model(duck_xml)
    limits = np.array([m.jnt_range[m.joint(n).id] for n in JOINTS])
    errors, xy0, worst_lift, worst_slide = [], None, (0.0, 0.0), (0.0, 0.0)
    for phase in np.arange(samples) / samples:
        q = interpolate(dance, phase)
        if np.any(q < limits[:, 0] - 1e-9) or np.any(q > limits[:, 1] + 1e-9):
            bad = [JOINTS[i] for i in np.flatnonzero((q < limits[:, 0]) | (q > limits[:, 1]))]
            errors.append(f"phase {phase:.3f}: {bad} outside joint limits")
        feet = feet_world(duck_xml, q, (0, 0, TRUNK_Z), (1, 0, 0, 0))
        lift = feet[:, 2].max() - feet[:, 2].min()
        worst_lift = max(worst_lift, (lift, phase))
        xy = feet[:, :2]
        xy0 = xy if xy0 is None else xy0
        slide = float(np.linalg.norm(xy - xy0, axis=1).max())
        worst_slide = max(worst_slide, (slide, phase))
    if worst_lift[0] > lift_m:
        errors.append(f"a foot lifts {100 * worst_lift[0]:.1f} cm at phase {worst_lift[1]:.3f} (max {100 * lift_m:.0f} cm)")
    if worst_slide[0] > slide_m:
        errors.append(f"a foot slides {100 * worst_slide[0]:.1f} cm at phase {worst_slide[1]:.3f} (max {100 * slide_m:.0f} cm)")
    return errors
```

  Also add `TRUNK_Z` to the `from .robot import …` line.

- [ ] **Step 4:** Run `uv run pytest tests/test_duck_dance.py -q`; expect it to pass.

- [ ] **Step 5: Commit**

```bash
git add duck_train/dance.py tests/test_duck_dance.py
git commit -m "feat: foot kinematics and the in-place foot check for dances"
```

### Task 3: The pinned duck in duck-sim 0.3.0

**Files:**
- Modify: `scripts/export_duck_scene.py`
- Test: `tests/test_duck_dance.py` (append)

- [ ] **Step 1: Write the failing test:**

```python
@needs_duck
def test_pinned_duck_compiles_without_free_joint():
    import mujoco

    pinned = DUCK_XML.parent / "duck_pinned.xml"
    if not pinned.exists():
        pytest.skip("re-run scripts/export_duck_scene.py")
    m = mujoco.MjModel.from_xml_path(str(pinned))
    assert m.nq == 14 and m.njnt == 14 and m.nu == 14
    assert m.nkey >= 1 and m.key("STAND").qpos.shape == (14,)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("STAND").id)
    mujoco.mj_forward(m, d)
    assert d.xpos[m.body("trunk_base").id][2] == pytest.approx(0.125, abs=0.01)
```

  Confirm the trunk body's name with `grep -n "<body" dist/duck-sim/robot_groundcontact.xml | head -3`, and adjust `trunk_base` if it differs. Take the height from the STAND keyframe's free-joint z, not from `TRUNK_Z`, if they differ.

- [ ] **Step 2: Implement** in `export_duck_scene.py`, after `duck.xml` is written:
  1. Load `out / "robot_groundcontact.xml"` with `mujoco.MjSpec.from_file`.
  2. Find the free joint (`trunk_base_freejoint`) and delete it (`spec.delete(joint)`, MuJoCo ≥ 3.3; check `uv run python -c "import mujoco; print(mujoco.__version__)"`).
  3. Set the root body's `pos` and `quat` to the STAND keyframe's first 7 qpos values.
  4. Write `robot_pinned.xml` next to it with `spec.to_xml()`. Keep `meshdir` resolving: write into `out`, the same folder as the assets.
  5. Write the wrapper `duck_pinned.xml`, which includes `robot_pinned.xml` and has a STAND key whose `qpos` attribute is the original STAND qpos minus its first 7 numbers. Keep any `ctrl` attribute.

  Factor the qpos rewrite as a small function, `pinned_keyframe(key: str) -> str`, using a regex on the `qpos="…"` attribute. Then update the module docstring: it now writes `duck.xml` and `duck_pinned.xml` (the posing duck: no free joint, position actuators, trunk fixed at STAND height).

- [ ] **Step 3:** Run `uv run python scripts/export_duck_scene.py $TMPDIR/gh-clones-3856841f-0045-481c-bb70-ba5b784e7265/microduck_rl`, then `uv run pytest tests/test_duck_dance.py -q`. Expect it to pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/export_duck_scene.py tests/test_duck_dance.py
git commit -m "feat: duck-sim gains duck_pinned.xml, the posing duck (no free joint, trunk at STAND)"
```

- [ ] **Step 5 (controller):** `viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=duck-sim --version=0.3.0 --type=archive --path=dist/duck-sim.tar.gz`

### Task 4: `scripts/duck_choreo.py`

**Files:**
- Create: `scripts/duck_choreo.py`
- Test: `tests/test_duck_dance.py` (append pure-helper tests)

The tool talks to the choreo machine through the mujoco-sim skill's `cell.connect`, the same way `scripts/duck_parity.py` does (copy its `SKILL` path and its `sys.path.insert`). It uses the arms `left-leg`, `head` and `right-leg`: 5, 4 and 5 joints in policy order.

- [ ] **Step 1: Write the failing tests** for the pure helpers:

```python
import importlib.util

spec = importlib.util.spec_from_file_location("duck_choreo", "scripts/duck_choreo.py")
choreo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(choreo)


def test_upsert_keyframe_drops_joints_at_stand_and_keeps_order():
    q = STAND.copy()
    q[ROLL] += 0.3
    q[0] += 0.005  # within 0.01 of STAND: left out
    d = choreo.upsert_keyframe(json.loads(json.dumps(BOB)), 0.6, q)
    assert [k["t"] for k in d["keyframes"]] == [0.0, 0.25, 0.5, 0.6, 0.75]
    assert d["keyframes"][3]["pose"] == {"head_roll": pytest.approx(float(q[ROLL]))}
    d = choreo.upsert_keyframe(d, 0.6, STAND)  # replace in place
    assert d["keyframes"][3] == {"t": 0.6, "pose": {}}


def test_split_and_join_arms():
    parts = choreo.split_arms(np.arange(14.0))
    assert [len(p) for p in parts.values()] == [5, 4, 5] and list(parts) == ["left-leg", "head", "right-leg"]
    assert np.allclose(choreo.join_arms(parts), np.arange(14.0))
```

- [ ] **Step 2:** Run them; expect them to fail.

- [ ] **Step 3: Implement `scripts/duck_choreo.py`.** It has a PEP 723 header with `viam-sdk>=0.85`, `numpy` and `mujoco`, and puts the repo root on `sys.path` so `duck_train.dance` imports. Its contents:
  - `ARMS = {"left-leg": slice(0, 5), "head": slice(5, 9), "right-leg": slice(9, 14)}`, plus `split_arms(q)` / `join_arms(parts)`.
  - `upsert_keyframe(dance, t, q)`:
    - inserts or replaces the keyframe at `round(t, 4)`;
    - the pose holds the joints where `|q − STAND| > 0.01`, rounded to 4 decimals;
    - keeps the keyframes sorted;
    - calls `validate` before returning.
  - `async def read_q(robot)`: for each arm, `await arm.do_command({"command": "get_joint_positions_radians"})["values"]`, then joins them.
  - `async def write_q(robot, q)`: for each arm, `do_command({"command": "set_joint_targets_radians", "values": [...]})`.
  - **`pose <cell.json> <dance.json> --t T`:** reads q, upserts the keyframe, writes the file, and prints the keyframe. If the file doesn't exist, it starts from `{"name": <stem>, "period_s": 6.0, "keyframes": [{"t": 0.0, "pose": {}}]}`.
  - **`preview <cell.json> <dance.json> [--cycles 2]`:**
    - `t0 = time.monotonic()`; each 20 ms, `phase = ((now − t0) / period) % 1`, then `write_q(interpolate(dance, phase, limits))`, sleeping to the next 20 ms grid point;
    - stops after `cycles × period` and writes STAND;
    - `limits` come from `duck.xml` via `dance._model(DUCK_XML)` (default `dist/duck-sim/duck.xml`, flag `--duck-xml`).
  - **`check <dance.json> [--duck-xml …]`:** prints `foot_check` errors and exits 1 if there are any, else prints `ok`.
  - **`record <cell.json> <dance.json> --part-id ID [--cycles 3] [--sync-wait 90]`:**
    1. Run `check`, and exit on failure.
    2. Write STAND and sleep 1 s.
    3. Take `start = datetime.now(timezone.utc)` immediately before `t0` of a `preview` of N cycles, so the playback starts at φ = 0 at `start`.
    4. `end = start + timedelta(seconds=N * period)`.
    5. Wait `--sync-wait` seconds for the data manager to sync.
    6. Create the sequence with the machine key: `ViamClient.create_from_dial_options(DialOptions.with_api_key(key, key_id))` from `cell.json`, then `data_client.create_sequence(part_id, resources=[SequenceResourceFilter(...) for each arm, method "JointPositions"], start_time=start, end_time=end, sequence_tags=[f"dance:{name}", f"period:{period}"])`.
    7. Print the sequence id.

    Read `SequenceResourceFilter`'s fields in `viam.proto.app.data` before using it (`uv run python -c "from viam.proto.app.data import SequenceResourceFilter as S; print(S.DESCRIPTOR.fields_by_name.keys())"`).

  `record` runs on the same host as the choreo machine's viam-server (spec Part 2, step 5), which this plan's setup guarantees. Deviation from the spec: `record` stops at creating the sequence. Creating the dataset and adding the sequence are controller steps (Task 5), because they need org-level access the machine key doesn't have.

- [ ] **Step 4:** Run `uv run pytest tests/test_duck_dance.py -q` (pass), then `uv run scripts/duck_choreo.py check dances/does-not-exist.json` (a clear error, not a traceback).

- [ ] **Step 5: Commit**

```bash
git add scripts/duck_choreo.py tests/test_duck_dance.py
git commit -m "feat: duck_choreo.py: pose, preview, check and record dances on the choreo machine"
```

### Task 5 (controller, with the user): The choreo machine, the first dance, the dataset

Machine-facing work. The controller runs it; it needs the user's confirmation before creating the machine and its key.

- [ ] **Step 1: Write `docs/superpowers/specs/duck-choreo-machine.json`,** modelled on `duck-sim-machine.json`. It has:
  - the mujoco-sim local module (the same `executable_path`);
  - package `duck-sim` **0.3.0** (archive);
  - a world with `physics_dt` 0.005;
  - robot `duck` with `mjcf: ${packages.archive.duck-sim}/duck_pinned.xml`, no `actuators`, and the same frame as the preview duck (not floating, so no world-frame requirement beyond what the module asks for);
  - arms `left-leg`, `head` and `right-leg` as on the preview machine (`home_keyframe` `"STAND"`, the same flange sites);
  - the `overview` camera;
  - the `data_manager` service with `capture_dir` default and `sync_interval_mins: 0.05`;
  - on each arm, a `service_configs` entry for `data_manager` with `capture_methods: [{"method": "JointPositions", "capture_frequency_hz": 50}]`.
- [ ] **Step 2:** Confirm with the user, then create `microduck-choreo` in DevRelDemos (same location as the preview machine). Run `cell.py creds` into `<scratch>/microduck-choreo/`, start its viam-server in the background, write the config, `cell.py wait`, then `reset`. Tell the user the key's name and id.
- [ ] **Step 3:** Use `GetImages` on `overview` to check the pinned duck stands at STAND. Try `set_joint_targets_radians` on `head` and look again.
- [ ] **Step 4: Choreograph `dances/bob-sway.json`.** It's 6 s long, with:
  - a knee bob on both legs (hip pitch, knee and ankle together, mirrored left and right);
  - a small hip sway;
  - head roll and nod.

  Build each keyframe with `pose`, look at it with `GetImages`, and iterate until `check` passes. Show the user images of each keyframe and offer changes. Commit the dance: `git add dances/bob-sway.json && git commit -m "feat: first dance, bob-sway"`.
- [ ] **Step 5: Record.**
  1. `uv run scripts/duck_choreo.py record <cell.json> dances/bob-sway.json --part-id <part> --cycles 3` gives a sequence id.
  2. Create the `microduck-dances` sequence dataset (MCP `create_dataset`, or `DataClient.create_dataset(..., type=<sequence>)`), and add the sequence with MCP `add_sequences_to_dataset`.
  3. If `create_sequence` is denied with the machine key, stop and ask the user about an org-scoped key. Don't create one unasked.
- [ ] **Step 6: Pin the Parquet format.**
  1. Export the dataset to a fresh scratch directory: `viam dataset export --dataset-id=<id> --destination=<scratch>/dances-export --only-parquet`.
  2. Read one `JointPositions` row: its `component_name`, `method_name` and `payload`.
  3. Write `tests/data/dance_parquet_sample.json` with three real rows (one per arm): `component_name`, `method_name` and `payload` verbatim, with no ids or keys.

  This file is the contract Task 6 parses. Commit it.

---

## Part 2: training

### Task 6: `build_reference`

Blocked until Task 5 Step 6 has committed `tests/data/dance_parquet_sample.json`.

**Files:**
- Modify: `duck_train/dance.py`, `pyproject.toml` (dev deps), `tests/test_duck_dance.py`

- [ ] **Step 1:** Run `uv add --dev pandas pyarrow`.

- [ ] **Step 2: Write the failing tests.** The synthetic Parquet must mimic the real rows pinned in `tests/data/dance_parquet_sample.json`: the same `component_name` style and the same payload shape, with the angles replaced.

```python
import datetime as dt

from duck_train.dance import build_reference, joint_values

SAMPLE = json.loads(Path("tests/data/dance_parquet_sample.json").read_text())


def payload_like(sample_payload: str, degrees) -> str:
    """Rebuild a real payload with new angles (whatever the nesting, the angle list is the only list)."""
    def swap(o):
        if isinstance(o, list):
            return [float(v) for v in degrees]
        if isinstance(o, dict):
            return {k: swap(v) for k, v in o.items()}
        return o
    return json.dumps(swap(json.loads(sample_payload)))


def test_joint_values_reads_real_payload():
    for row in SAMPLE:
        assert len(joint_values(row["payload"])) in (4, 5)


def write_parquet(tmp_path, dance, cycles=3, lag_s=0.04, jitter=0.0, tags=None, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    start = dt.datetime(2026, 11, 21, tzinfo=dt.timezone.utc)
    period = dance["period_s"]
    rows, t = [], 0.0
    by_arm = {r["component_name"]: r for r in SAMPLE}
    names = list(by_arm)  # left-leg, head, right-leg order as captured; match by suffix below
    while t < cycles * period:
        q = interpolate(dance, ((t - lag_s) / period) % 1.0) + rng.normal(0, jitter, 14)
        for cname in names:
            arm = next(a for a in ("left-leg", "head", "right-leg") if cname.endswith(a))
            sl = {"left-leg": slice(0, 5), "head": slice(5, 9), "right-leg": slice(9, 14)}[arm]
            rows.append({"sequence_id": "s1", "timestamp": start + dt.timedelta(seconds=t + rng.uniform(0, 0.004)),
                         "part_id": "p", "component_name": cname, "method_name": by_arm[cname]["method_name"],
                         "payload": payload_like(by_arm[cname]["payload"], np.degrees(q[sl]))})
        t += 0.02
    pd.DataFrame(rows).to_parquet(tmp_path / "tabular.parquet")
    pd.DataFrame([{"sequence_id": "s1", "tags": tags or [f"dance:{dance['name']}", f"period:{period}"],
                   "start_at": start, "end_at": start + dt.timedelta(seconds=cycles * period)}]).to_parquet(tmp_path / "seq.parquet")
    pd.DataFrame(columns=["sequence_id"]).to_parquet(tmp_path / "bin.parquet")
    return tmp_path / "bin.parquet", tmp_path / "tabular.parquet", tmp_path / "seq.parquet"


def test_build_reference_recovers_the_dance(tmp_path):
    period, table = build_reference(*write_parquet(tmp_path, BOB, jitter=0.002), dance="t")
    assert period == 6.0 and table.shape == (100, 14)
    want = np.stack([interpolate(BOB, (k + 0.5) / 100) for k in range(100)])
    assert np.abs(table - want).max() < 0.02  # looser than the spec's 0.01: the test adds a 40 ms actuator lag on purpose


@pytest.mark.parametrize("tags,match", [(["dance:other", "period:6.0"], "no sequence"),
                                        (["dance:t"], "period")])
def test_build_reference_rejects(tmp_path, tags, match):
    with pytest.raises(ValueError, match=match):
        build_reference(*write_parquet(tmp_path, BOB, tags=tags), dance="t")


def test_build_reference_rejects_inconsistent_cycles(tmp_path):
    with pytest.raises(ValueError, match="spread"):
        build_reference(*write_parquet(tmp_path, BOB, jitter=0.2), dance="t")
```

  `period:` parsing: a sequence carries one `period:` tag. Several sequences must agree, or raise `ValueError("mixed periods …")`. Add one more test for mixed periods with two sequences if it's cheap; the single-sequence tests above are required.

- [ ] **Step 3:** Run them; expect them to fail.

- [ ] **Step 4: Implement** (append to `dance.py`), following spec Part 2, "Reference builder":
  - `joint_values(payload: str) -> list[float]` parses the pinned payload shape and returns degrees. Write it for the exact shape in `dance_parquet_sample.json`, for example `json.loads(p)["positions"]["values"]` if that's what it is. Raise `ValueError` with the payload's top-level keys when it's not that shape.
  - `ARM_SLICES = {"left-leg": slice(0, 5), "head": slice(5, 9), "right-leg": slice(9, 14)}`. Match a tabular row's `component_name` to an arm by `endswith(arm)` (or by whatever exact rule the sample shows).
  - `build_reference(binary_file, tabular_file, sequences_file, dance, bins=100) -> tuple[float, np.ndarray]`:
    1. Read `sequences_file` with pandas. Select the rows whose `tags` contain `f"dance:{dance}"`; if there are none, raise `ValueError(f"no sequence tagged dance:{dance}")`.
    2. Collect `period:` values (float). Raise `ValueError` for none (`"no period tag"`) or more than one distinct value (`"mixed periods"`).
    3. Read `tabular_file`, keeping rows of the selected sequences whose `method_name` contains `JointPositions`.
    4. For each sequence: build the grid `np.arange(start, end, 0.02)` in seconds (convert tz-aware timestamps with `.timestamp()` or by subtracting the epoch). For each arm, `np.interp` each joint onto the grid from that arm's rows sorted by timestamp, then `np.radians`. Concatenate into (N, 14). If an arm has no rows, raise `ValueError`.
    5. `phase = ((grid − start) / period) % 1`; bin index `min(int(phase × bins), bins − 1)`. Accumulate the sum, sum of squares and count over all sequences.
    6. `table = sum / count`; `spread = sqrt(max(sumsq / count − table², 0))`. Run the checks **in this order**, so that a noisy recording reports its spread rather than bin 0: raise `ValueError` if any count is 0 (`"empty bin"`), then if `spread.max() > 0.05` (`"spread … rad across cycles"`), then if `|table[0] − STAND|.max() > 0.05` (`"bin 0 isn't STAND"`).
  - `binary_file` is unused (the dance has no images). Keep the parameter so the call mirrors the three job arguments.

- [ ] **Step 5:** Run `uv run pytest -q`; expect it to pass.

- [ ] **Step 6: Commit**

```bash
git add duck_train/dance.py tests/test_duck_dance.py pyproject.toml uv.lock
git commit -m "feat: build the dance reference table from a sequence dataset"
```

### Task 7: Dance reward math

**Files:**
- Create: `duck_train/dance_rewards.py`, `tests/test_duck_dance_rewards.py`

These are torch only, with no mjlab imports. They get the env duck-typed: `env.command_manager.get_command(name)`, `env.scene[asset_name]` with `.find_joints(names)` and `.data.joint_pos`, and `env.scene.sensors[name].data.found`. That's the subset of mjlab that Pollen's `_phase_pose_error` uses.

- [ ] **Step 1: Write the failing tests:**

```python
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from duck_train.dance import table_at
from duck_train.dance_rewards import dance_pose, dance_pose_l1, feet_contact, phase_of, table_at_torch
from duck_train.robot import JOINTS


class Asset:
    def __init__(self, q):
        self.order = list(reversed(JOINTS))  # a joint order different from the policy's, as in mjlab
        self.data = SimpleNamespace(joint_pos=torch.as_tensor(q, dtype=torch.float32))

    def find_joints(self, names, preserve_order=False):
        ids = [self.order.index(n) for n in names]
        return (ids, names) if preserve_order else (sorted(ids), [self.order[i] for i in sorted(ids)])


class Scene(dict):
    pass


def env_with(q_asset_order, phases, found=None):
    cmd = torch.stack([torch.cos(2 * math.pi * torch.tensor(phases)), torch.sin(2 * math.pi * torch.tensor(phases)),
                       torch.zeros(len(phases))], 1)
    scene = Scene(robot=Asset(q_asset_order))
    scene.sensors = {"feet_ground_contact": SimpleNamespace(data=SimpleNamespace(found=torch.as_tensor(found)))} if found is not None else {}
    return SimpleNamespace(scene=scene, command_manager=SimpleNamespace(get_command=lambda name: cmd), device="cpu")


TABLE = np.random.default_rng(0).normal(0, 0.3, (100, 14))


def test_phase_of_and_table_match_numpy():
    phases = [0.0, 0.123, 0.5, 0.999]
    env = env_with(np.zeros((4, 14)), phases)
    got = phase_of(env.command_manager.get_command("twist"))
    assert torch.allclose(got, torch.tensor(phases), atol=1e-5)
    t = table_at_torch(torch.as_tensor(TABLE), got)
    for i, p in enumerate(phases):
        assert np.allclose(t[i].numpy(), table_at(TABLE, p), atol=1e-5)


def test_pose_rewards_peak_on_reference_in_asset_order():
    phases = [0.3, 0.7]
    ref_policy = np.stack([table_at(TABLE, p) for p in phases])
    asset_order = ref_policy[:, ::-1].copy()  # Asset.order is JOINTS reversed
    env = env_with(asset_order, phases)
    assert torch.allclose(dance_pose(env, table=TABLE.tolist()), torch.ones(2), atol=1e-5)
    assert torch.allclose(dance_pose_l1(env, table=TABLE.tolist()), torch.zeros(2), atol=1e-5)
    off = env_with(asset_order + 0.3, phases)
    assert (dance_pose(off, table=TABLE.tolist()) < 0.5).all()
    assert torch.allclose(dance_pose_l1(off, table=TABLE.tolist()), torch.full((2,), -0.3), atol=1e-5)


def test_feet_contact_needs_both():
    env = env_with(np.zeros((3, 14)), [0, 0, 0], found=[[1, 1], [1, 0], [0, 0]])
    assert feet_contact(env).tolist() == [1.0, 0.0, 0.0]
```

- [ ] **Step 2:** Run them; expect them to fail.

- [ ] **Step 3: Implement `duck_train/dance_rewards.py`:**

```python
"""Reward math for Mjlab-Dance-Flat-MicroDuck, torch only (tested off-GPU with a fake env). The env is used
the way microduck_rl's _phase_pose_error uses it (MD tasks/mdp.py ~2917): phase from the twist command,
joints resolved by name (the asset's joint order isn't the policy's)."""

from __future__ import annotations

import torch

from .robot import JOINTS

_TABLES: dict = {}


def phase_of(cmd: torch.Tensor) -> torch.Tensor:
    return (torch.atan2(cmd[:, 1], cmd[:, 0]) / (2 * torch.pi)) % 1.0


def table_at_torch(table: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """(bins, 14) x (B,) -> (B, 14). Linear between bin centres, wrapping: same rule as dance.table_at."""
    bins = table.shape[0]
    x = phase * bins - 0.5
    i0 = torch.floor(x)
    w = (x - i0).unsqueeze(-1)
    i0 = i0.long() % bins
    return (1 - w) * table[i0] + w * table[(i0 + 1) % bins]


def _table(table, device) -> torch.Tensor:
    # ponytail: keyed by the params list's id; one table per run, so the cache never grows past one entry per device.
    key = (id(table), str(device))
    if key not in _TABLES:
        _TABLES[key] = torch.as_tensor(table, dtype=torch.float32, device=device)
    return _TABLES[key]


def _error(env, command_name, table, asset_name):
    asset = env.scene[asset_name]
    ids, _ = asset.find_joints(JOINTS, preserve_order=True)
    cur = asset.data.joint_pos[:, ids]
    tgt = table_at_torch(_table(table, cur.device), phase_of(env.command_manager.get_command(command_name)))
    return cur - tgt


def dance_pose(env, command_name: str = "twist", table=None, std: float = 0.3, asset_name: str = "robot") -> torch.Tensor:
    return torch.exp(-(_error(env, command_name, table, asset_name) / std) ** 2).mean(dim=-1)


def dance_pose_l1(env, command_name: str = "twist", table=None, asset_name: str = "robot") -> torch.Tensor:
    return -_error(env, command_name, table, asset_name).abs().mean(dim=-1)


def feet_contact(env, sensor_name: str = "feet_ground_contact") -> torch.Tensor:
    found = env.scene.sensors[sensor_name].data.found
    return (found[:, :2] > 0).all(dim=-1).float()
```

  mjlab's `find_joints` defaults to `preserve_order=False` (asset order). The rewards pass `preserve_order=True`, and the fake `Asset` returns asset order without it, so the test fails if the flag is dropped. Confirm that mjlab 1.3.0 has the flag in the image (`docker run --rm --platform linux/amd64 microduck-train:dev grep -n "def find_joints" -A6 /opt/venv/lib/python3.12/site-packages/mjlab/entity/entity.py`). If it doesn't, resolve one name at a time, as Pollen does (`find_joints([n])[0][0]`).

- [ ] **Step 4:** Run `uv run pytest tests/test_duck_dance_rewards.py -q`; expect it to pass.

- [ ] **Step 5: Commit**

```bash
git add duck_train/dance_rewards.py tests/test_duck_dance_rewards.py
git commit -m "feat: dance reward math (phase pose tracking, both feet down)"
```

### Task 8: The dance task, `task=dance`, the image and the script

**Files:**
- Create: `duck_train/dance_task.py`
- Modify: `duck_train/train.py`, `duck_train/package.py`, `build_duck.sh`, `container/Dockerfile.duck`, `tests/test_duck_train.py`

- [ ] **Step 1: Write the failing tests** (in `tests/test_duck_train.py`):

```python
def test_parse_args_task_defaults(tmp_path):
    v = parse_args([f"--model_output_directory={tmp_path}"])
    assert v.task == "velocity" and v.max_iterations == 4000
    d = parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--dance=bob-sway",
                    "--tabular_data_file=t", "--sequences_file=s", "--binary_data_file=b"])
    assert d.task == "dance" and d.dance == "bob-sway" and d.max_iterations == 2000


def test_parse_args_dance_requires_name_and_parquet(tmp_path):
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--tabular_data_file=t", "--sequences_file=s"])
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--task=dance", "--dance=x"])


def test_build_config_phase_variant():
    cfg = build_config(joints=JOINTS, default_pose=[0.1] * 14, action_scale=1.0, joint_limits=[[-1.0, 1.0]] * 14,
                       command={"mode": "phase", "period_s": 6.0}, control_hz=50, source={"task": "Mjlab-Dance-Flat-MicroDuck"})
    assert cfg["command"] == {"mode": "phase", "period_s": 6.0} and cfg["safety"] == {}
```

  Also update `test_build_config_shape` to pass `command={"twist_min": [...], "twist_max": [...]}` instead of the two keyword arguments.

- [ ] **Step 2:** Run `uv run pytest tests/test_duck_train.py -q`; expect it to fail.

- [ ] **Step 3: `package.py`.** Replace the `twist_min, twist_max` parameters with `command: dict`, written as `"command": {k: (list(map(float, v)) if isinstance(v, (list, tuple)) else v) for k, v in command.items()}`. Update the call in `train.py` to pass `command={"twist_min": [...], "twist_max": [...]}`.

- [ ] **Step 4: `train.py`.**
  - Arguments:
    - `--task` (`velocity|dance`, default `velocity`) and `--dance` (default None).
    - Change `--max_iterations` to default `None`, resolved after parsing to 4000 (velocity) or 2000 (dance).
    - After parsing, for `task == "dance"`, require `--dance`, `--tabular_data_file` and `--sequences_file`, or `p.error(...)`, which raises `SystemExit`.
    - Update the parquet arguments' help from "ignored" to "sequence dataset (dance)".
  - **Thread the task id.** `TASK` stays the velocity id. The dance id comes from `register_dance_task(...)`'s return value, so `train.py` defines no dance constant and never imports `dance_task` at module level.
    - `_cfgs(args, task)`, `export_and_verify(task, ckpt, onnx_out)`, and `run_train(task, …)` take the id.
    - `write_package(out, log_dir, env_cfg, agent, task, command, extra)` writes the `command` it's given; it doesn't read `twist`. `source["task"]` is `task`. `extra` is a dict of file name → JSON written into the package.
    - `probe()` keeps working for velocity: update its calls to `_cfgs(args, TASK)` and `export_and_verify(TASK, …)`.
  - **In `train()`, in this order:**
    1. If `args.task == "dance"`:
       - `period, table = build_reference(args.binary_data_file, args.tabular_data_file, args.sequences_file, args.dance)`;
       - `task = register_dance_task(table, period)` (`from .dance_task import register_dance_task` inside the branch);
       - `command = {"mode": "phase", "period_s": period}`;
       - `extra = {"reference.json": {"period_s": period, "joints": JOINTS, "table": table.round(5).tolist()}}`;
       - log the table's shape.

       Otherwise: `task = TASK` and `extra = {}`.
    2. `env_cfg, agent = _cfgs(args, task)`. This must come after registration, because `load_env_cfg` needs the task registered.
    3. For velocity, build `command` from `env_cfg.commands["twist"].ranges` (the code moved out of `write_package`).
    4. `source` gets `"dance": args.dance` for dance jobs, and video `TAGS` get `args.task`.

- [ ] **Step 5: `duck_train/dance_task.py`** (image only; imports at module top are fine, since only `train.py` imports it lazily):

```python
"""Mjlab-Dance-Flat-MicroDuck (spec Part 2): Pollen's flat velocity task (MD microduck_velocity_env_cfg) with the
velocity terms swapped for phase-indexed pose tracking against the dance's reference table."""

from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from mjlab.managers import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

from . import dance_rewards

DANCE_TASK = "Mjlab-Dance-Flat-MicroDuck"
REMOVED_REWARDS = ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                   "foot_swing_height", "pose", "head_pose_tracking", "head_pose_bias", "body_pose_tracking")
REMOVED_CURRICULA = ("action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range", "head_pose_bias_weight")
EPISODE_S = 20.0
REL_FROZEN = 0.2


class DancePhaseCommand(microduck_mdp.GroundPickPhaseCommand):
    """GroundPickPhaseCommand with a fraction of envs frozen at phase 0 (STAND hold)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rel_frozen = float(cfg.rel_frozen_envs)

    def reset(self, env_ids):
        out = super().reset(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._frozen[env_ids] = torch.rand(len(env_ids), device=self.device) < self._rel_frozen
            self._gp_phase[env_ids] = torch.where(self._frozen[env_ids], torch.zeros_like(self._gp_phase[env_ids]),
                                                  self._gp_phase[env_ids])
        return out

    def compute(self, dt):
        super().compute(dt)
        self._gp_phase[self._frozen] = 0.0
        self.vel_command_b[self._frozen, 0] = 1.0
        self.vel_command_b[self._frozen, 1] = 0.0


@dataclasses.dataclass(kw_only=True)
class DancePhaseCommandCfg(microduck_mdp.GroundPickPhaseCommandCfg):
    class_type: type = DancePhaseCommand
    rel_frozen_envs: float = REL_FROZEN

    def build(self, env):
        return DancePhaseCommand(self, env)


def make_dance_env_cfg(table, period_s: float, play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    twist = cfg.commands["twist"]
    fields = {f.name for f in dataclasses.fields(DancePhaseCommandCfg)}
    cfg.commands["twist"] = DancePhaseCommandCfg(**{**{k: v for k, v in vars(twist).items() if k in fields},
                                                    "class_type": DancePhaseCommand, "period": float(period_s),
                                                    "randomize_phase": True})
    for name in ("head_pose", "body_pose"):
        ranges = cfg.commands[name].ranges
        cfg.commands[name].ranges = tuple((0.0, 0.0) for _ in ranges)
    for name in REMOVED_REWARDS:
        del cfg.rewards[name]
    for name in REMOVED_CURRICULA:
        del cfg.curriculum[name]
    table = [list(map(float, row)) for row in table]
    cfg.rewards["dance_pose"] = RewardTermCfg(func=dance_rewards.dance_pose, weight=4.0, params={"table": table, "std": 0.3})
    cfg.rewards["dance_pose_l1"] = RewardTermCfg(func=dance_rewards.dance_pose_l1, weight=1.0, params={"table": table})
    cfg.rewards["feet_contact"] = RewardTermCfg(func=dance_rewards.feet_contact, weight=0.5)
    cfg.rewards["action_rate_l2"].weight = -1.0  # the removed action_rate_weight curriculum's final value
    cfg.episode_length_s = EPISODE_S
    return cfg


def register_dance_task(table, period_s: float) -> str:
    rl = deepcopy(MicroduckRlCfg)
    rl.experiment_name = "microduck_dance"
    register_mjlab_task(task_id=DANCE_TASK, env_cfg=make_dance_env_cfg(table, period_s),
                        play_env_cfg=make_dance_env_cfg(table, period_s, play=True), rl_cfg=rl,
                        runner_cls=MicroduckOnPolicyRunner)
    return DANCE_TASK
```

  **Things to verify inside the image** before committing (`docker build` first, as in Step 7; then `docker run --rm --platform linux/amd64 microduck-train:dev /opt/venv/bin/python -c "<snippet>"`):
  - `from mjlab_microduck.tasks import MicroduckOnPolicyRunner` works. It's defined in `tasks/__init__.py`.
  - The `head_pose` / `body_pose` `ranges` type. If it's a dataclass rather than a tuple of pairs, zero each field instead.
  - Every name in `REMOVED_*` exists in both `play=False` and `play=True` configs. If one is missing in play, use `pop(name, None)` for play only, and say so in a comment.
  - `make_dance_env_cfg(np.zeros((100, 14)), 6.0)` builds without error on CPU. Building the config doesn't need a GPU.

- [ ] **Step 6: Ship it.**
  1. In `build_duck.sh`, add `duck_train/dance.py duck_train/dance_rewards.py duck_train/dance_task.py` to the `cp` line.
  2. In `container/Dockerfile.duck`, add `pandas pyarrow` to the `uv pip install` line, and `pandas, pyarrow` to the import check.
  3. In `duck_train/setup.py`, bump `version` to `0.2.0`, and update `tests/test_duck_train.py::test_duck_sdist_shim` (it hard-codes `microduck_train-0.1.0/model/training.py`) to the 0.2.0 path.

- [ ] **Step 7:** Run `uv run pytest -q` (pass), then `grep -n "_cfgs(\|export_and_verify(\|run_train(" duck_train/train.py`: every call passes a task id, including the ones in `probe()`. Then run `./build_duck.sh`. The tar listing shows `robot.py`, `dance.py`, `dance_rewards.py` and `dance_task.py`. Then run `docker build --platform linux/amd64 -f container/Dockerfile.duck -t microduck-train:dev container/` and the image checks from Step 5.

- [ ] **Step 8: Commit**

```bash
git add duck_train/ build_duck.sh container/Dockerfile.duck tests/test_duck_train.py
git commit -m "feat: Mjlab-Dance-Flat-MicroDuck and task=dance in microduck-train; pandas/pyarrow in the image"
```

### Task 9 (controller): The dance probe job

- [ ] **Step 1: Push and register a new image tag** (never re-push 0.1.0-1):

```bash
docker tag microduck-train:dev docker.io/hipsterbrown/viam-microduck-train:0.2.0-1
docker push docker.io/hipsterbrown/viam-microduck-train:0.2.0-1
viam train containers register --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --uri=docker.io/hipsterbrown/viam-microduck-train:0.2.0-1 --description="microduck-train 0.2.0-1 (dance)"
viam train containers list --org-id=16518049-9dd3-479f-9644-c5d112aa42d8   # note the container ID
viam training-script upload --path=dist/microduck-train.tar.gz --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --script-name=microduck-train --version=2 --framework=onnx --type=robot_policy --visibility=private
```

- [ ] **Step 2: Submit 200 iterations:**

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=<microduck-dances id> --model-name=microduck-dance-probe --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:microduck-train --version=2 --container-id=<new id> \
  --args=task=dance,dance=bob-sway,num_envs=4096,max_iterations=200,save_interval=100,video_every_iters=100,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=dance-probe-1
```

- [ ] **Step 3: Read the result.**
  1. Poll every 10 minutes in the background.
  2. Export the package into a fresh scratch dir.
  3. Check that `verify_onnx` passed (the job log), that `config.json` has `command.mode == "phase"`, and that `reference.json` is 100 × 14.
  4. Read the job log for `dance_pose` rising and `fell_over` falling over 200 iterations, and watch the uploaded videos.

  If `dance_pose` stays flat, the first lever is the STAND → full reference ramp (spec "No dance curriculum at first"). Bring that to the user before changing anything.

### Task 10 (controller and implementer): The full run, the reference `--phase`, the package

**Files (implementer):**
- Modify: `scripts/duck_reference.py`, `tests/test_duck_parity.py`

- [ ] **Step 1 (controller):** Submit the full run: same as Task 9 Step 2, with `--model-name=microduck-dance-train --model-version=1`, `max_iterations=2000`, `save_interval=250`, `video_every_iters=500`, `run_name=dance-1`. Poll in the background.

- [ ] **Step 2 (implementer, in parallel): write failing tests** in `tests/test_duck_parity.py` for the phase schedule:

```python
def test_phase_schedule():
    try:
        ref.set_phase(6.0, cycles=3, hold_s=2.0)
        assert ref.DURATION == 20.0
        assert ref.command_at(0.0) == pytest.approx((1.0, 0.0, 0.0))
        assert ref.command_at(1.5) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)  # phase 0.25
        assert ref.command_at(18.5) == (1.0, 0.0, 0.0)  # holding after 3 cycles
    finally:
        ref.set_phase(None)
        ref.set_script(ref.DEFAULT_SCRIPT)
```

- [ ] **Step 3: Implement in `duck_reference.py`:**
  - `PHASE = None` (or `{"period_s", "cycles", "hold_s"}`).
  - `set_phase(period_s, cycles=3, hold_s=2.0)` sets `PHASE` and `DURATION = cycles * period_s + hold_s`. `set_phase(None)` clears it and restores `DURATION` from `SCRIPT`.
  - `command_at(t)`: when `PHASE` is set and `t < cycles * period`, return `(cos 2πφ, sin 2πφ, 0.0)` with `φ = (t / period) % 1`; after that, `(1.0, 0.0, 0.0)`.
  - Each 0.1 s sample gains `"q": data.qpos[policy.joint_qpos_indices].tolist()`.
  - The result has `"phase": PHASE` when set (and still `"script"`).
  - CLI: `--phase <period_s>` (with `--cycles 3` and `--hold 2.0`).
  - Confirm that `PolicyInference` puts `vel_cmd` in obs slots 48:51 with `new_cmd_obs=True`: print `policy.get_observations()[48:51]` once in a scratch run with a set `vel_cmd`.

  Run `uv run pytest -q` (pass) and commit: `git commit -am "feat: duck_reference --phase: drive the phase clock, record joint angles"`.

- [ ] **Step 4 (controller), after the job:**
  1. Export the package: `viam packages export … --name=microduck-dance-train --version=1 --type=ml_model --destination=<scratch>/dance-train`.
  2. Copy it to `dist/microduck-dance-bob-sway/` (`policy.onnx`, `config.json`, `reference.json`, `model_metrics.json`; leave out `checkpoint.pt`).
  3. Run `duck_reference.py <checkout> dist/duck-sim dist/microduck-dance-bob-sway/policy.onnx dist/dance-reference.json --phase 6.0 --stats dist/microduck-dance-bob-sway/config.json`. Its `safety` must now be filled. The reference must not fall; if it does, stop and report.
  4. Pack and upload: `tar -czf dist/microduck-dance-bob-sway.tar.gz -C dist/microduck-dance-bob-sway .`, then `viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=microduck-dance-bob-sway --version=0.1.0 --type=ml_model --model-framework=onnx --model-type=robot_policy --path=dist/microduck-dance-bob-sway.tar.gz`.

---

## Part 3: running it and evaluating

### Task 11: Phase mode in duck-policy

**Files:**
- Modify: `duck_policy/obs.py`, `duck_policy/loop.py`, `duck_policy/service.py`
- Tests: `tests/test_duck_obs.py`, `tests/test_duck_loop.py`, `tests/test_duck_service.py`

- [ ] **Step 1: Write the failing tests.**

  In `test_duck_obs.py`: update `build_obs(...)` calls to pass `command=Command(...).vector()`, and add:

```python
from duck_policy.obs import phase_command

def test_phase_command():
    assert np.allclose(phase_command(0.25)[:3], [0, 1, 0], atol=1e-7) and not phase_command(0.25)[3:].any()
```

  In `test_duck_loop.py` (reusing its `FakeIO` and `CFG`):

```python
PHASE_CFG = {**CFG, "command": {"mode": "phase", "period_s": 0.1}}  # 5 ticks per cycle


def test_phase_loop_advances_then_holds():
    io = FakeIO()
    loop = PolicyLoop(io, lambda obs: np.zeros(14), PHASE_CFG, Limits(), cycles=2)
    seen = []
    orig = loop._infer

    async def spy(obs):
        seen.append(obs[48:51].copy())
        return await orig(obs)

    loop._infer = spy
    asyncio.run(loop.run(max_ticks=14))
    phases = [round((math.atan2(s[1], s[0]) / (2 * math.pi)) % 1.0, 2) for s in seen]
    assert phases[:5] == [0.0, 0.2, 0.4, 0.6, 0.8]  # skipped slots would show as gaps, not drift
    assert all(p == 0.0 for p in phases[10:]) and loop.status()["holding"] and loop.status()["cycles_done"] == 2


def test_phase_loop_rejects_set_command():
    loop = PolicyLoop(FakeIO(), lambda obs: np.zeros(14), PHASE_CFG, Limits())
    with pytest.raises(ValueError):
        loop.set_command(Command(0.1))
```

  (Add `import math` at the top.) If tick-timing jitter on the test machine makes the first assertion flaky, assert on slot-derived phases with `pytest.approx(abs=0.21)`. Don't weaken the hold assertions.

  In `test_duck_service.py`, following its existing fake setup: `start {"cycles": 2}` on a phase config starts, and `set_command` raises `ValueError`.

- [ ] **Step 2:** Run them; expect them to fail.

- [ ] **Step 3: Implement.**
  - **`obs.py`:**
    - `build_obs(..., command: np.ndarray)` concatenates `np.asarray(command, np.float32)`.
    - Add `phase_command(phi) -> np.ndarray` (13 values: cos, sin, then zeros).
    - Update the module docstring: command is twist or phase.
  - **`loop.py`:**
    - `__init__(..., cycles: int = 0)`.
    - `cmd = cfg["command"]`; `self.period = float(cmd["period_s"]) if cmd.get("mode") == "phase" else None`.
    - Twist limits are read only when `self.period is None`.
    - `self.cycles, self.cycles_done, self.phase, self.holding = int(cycles), 0, 0.0, False`.
    - `set_command`: when `self.period` is set, `raise ValueError("set_command isn't available in phase mode")`.
    - In `_run`, before `build_obs`:

```python
if self.period is None:
    command = self.command.vector()
else:
    s = (slot - 1) * self.dt  # this tick's scheduled time since t0: skipped slots advance it like real time
    done = int(s // self.period)
    self.holding = bool(self.cycles) and done >= self.cycles
    self.cycles_done = min(done, self.cycles) if self.cycles else done
    self.phase = 0.0 if self.holding else (s / self.period) % 1.0
    command = phase_command(self.phase)
```

    - Pass `command=command`.
    - `status()` adds `"phase"`, `"cycles_done"` and `"holding"` when `self.period` is set.
  - **`service.py`:**
    - `start` reads `cycles = int(command.get("cycles", 0))` and builds `PolicyLoop(..., cycles=cycles)`. It calls `loop.set_command(self.command)` only in velocity mode.
    - `set_command` raises `ValueError` in phase mode, before touching `twist_*`.
    - Update the module docstring.

- [ ] **Step 4:** Run `uv run pytest -q`; expect it all to pass, including the golden obs test.

- [ ] **Step 5: Commit**

```bash
git add duck_policy/ tests/test_duck_obs.py tests/test_duck_loop.py tests/test_duck_service.py
git commit -m "feat: duck-policy phase mode: phase clock from slot time, cycles then STAND hold"
```

### Task 12: Phase parity and the handover check

**Files:**
- Modify: `scripts/duck_parity.py`, `tests/test_duck_parity.py`

- [ ] **Step 1: Write the failing tests** (pure functions, fed by synthetic samples):

```python
def test_dance_metrics_perfect_and_drifting():
    import duck_parity as par
    table = np.tile(np.array(DEFAULT)[None], (100, 1))
    samples = [{"t": round(0.1 * i, 4), "q": list(DEFAULT), "trunk_pos": [0, 0, 0.125],
                "trunk_quat_wxyz": [1, 0, 0, 0], "gravity": [0, 0, -1]} for i in range(200)]
    ref.set_phase(6.0, cycles=3, hold_s=2.0)
    try:
        m = par.dance_metrics(samples, table, DUCK_XML)
        assert m["joint_rmse"] == pytest.approx(0.0, abs=1e-9) and m["yaw_drift_deg"] == pytest.approx(0.0)
        assert m["foot_drift_m"] < 1e-6
        moved = [{**s, "trunk_pos": [0.02 * s["t"], 0, 0.125]} for s in samples]  # 12 cm per 6 s cycle
        assert par.dance_metrics(moved, table, DUCK_XML)["foot_drift_m"] > 0.03
    finally:
        ref.set_phase(None)
```

  (`DEFAULT` and `DUCK_XML` as defined in the other test modules: import or redefine them, and skip when `DUCK_XML` is missing.)

- [ ] **Step 2:** Run it; expect it to fail.

- [ ] **Step 3: Implement in `duck_parity.py`:**
  - Replace the hand-rolled `sys.argv` parsing with `argparse` subcommands (`ours`, `report`, `handover`). The existing flags keep their names. This refactor is in scope.
  - Put the repo root on `sys.path` and import `duck_train.dance` (`table_at`, `feet_world`). Add `mujoco` to the PEP 723 dependencies.
  - `dance_metrics(samples, table, duck_xml) -> dict`:
    - **`joint_rmse`:** over samples with `t < cycles × period`, RMS of `q − table_at(table, (t / period) % 1)` across all joints and samples.
    - **`foot_drift_m`:** feet from `feet_world(duck_xml, q, trunk_pos, trunk_quat)`. For each cycle boundary k·period (nearest sample), take each foot's horizontal displacement between consecutive boundaries; report the max.
    - **`yaw_drift_deg`:** the wrapped yaw difference between the first and last sample, in degrees.
    - **`hold_upright`:** `gravity_z < −0.5` on the last sample.
  - `ours`:
    - `--phase <period_s>` (with `--cycles 3 --hold 2.0`) calls `ref.set_phase`.
    - `--service <name>` (default `walker`) chooses the policy service.
    - In phase mode, `start` is sent as `{"command": "start", "cycles": cycles}`, and no `set_command` is sent.
    - Samples gain `"q": s["q"]`, in both modes.
    - The episode record stores `"phase": ref.PHASE`.
    - Phase alignment: ours' `t` starts at the first `io` read after `start`, while the dancer's phase starts at its loop's `t0` (after warm-up). Record the offset as `status.run_s` vs the io time span, and report it. A few tens of ms are expected and are within the 1.25× allowance.
  - `report`:
    - If the reference JSON has `"phase"`, call `ref.set_phase(**phase)`, require `--reference-table <reference.json>` and `--duck-xml` (default `dist/duck-sim/duck.xml`), and build the phase checks (spec Part 3, "Evaluation"):
      - slip-free ≥ 18/20;
      - falls ≤ reference falls + 1;
      - mean `joint_rmse` (ours) ≤ 1.25 × the reference's;
      - every episode's `foot_drift_m` < 0.03;
      - every episode's `yaw_drift_deg` < 10;
      - the timing checks as now;
      - "policy ran every episode" (the same rule as today's "walker ran", renamed);
      - "hold": `hold_upright` and the service state is `running` in every episode.
    - Title the report `MicroDuck dance parity`.
    - The velocity path is unchanged, and so is the mismatch guard: compare `phase` the way `script` is compared.
  - `handover <cell.json> [--cycles 2]`:
    1. walker `start`, 2 s, `stop`;
    2. dancer `start {"cycles": N}`, then wait N × period + 0.5 s, where the period comes from dancer `status` or a `--period` flag;
    3. `stop`; walker `start`, 2 s, `stop`;
    4. `io` is polled at 10 Hz throughout.

    Print PASS or FAIL. It fails on any `gravity z > −0.5` or any service `reason` other than `stopped`/`None` before the stop.

- [ ] **Step 4:** Run `uv run pytest -q`; expect it to pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/duck_parity.py tests/test_duck_parity.py
git commit -m "feat: dance parity (joint RMSE, foot and yaw drift, hold) and the walker/dancer handover check"
```

### Task 13 (controller): The preview machine, the evaluation, the results

- [ ] **Step 1: Update `docs/superpowers/specs/duck-sim-machine.json`:**
  - package `microduck-dance-bob-sway` 0.1.0 (ml_model);
  - service `dance-model` (`viam:mlmodel:onnx-cpu`, `model_path: ${packages.ml_model.microduck-dance-bob-sway}/policy.onnx`);
  - service `dancer` (`devrel:duck-policy:controller`; `duck`, `mlmodel: dance-model`, `package_dir: ${packages.ml_model.microduck-dance-bob-sway}`).

  Write it to the preview machine (read the revision first), `cell.py wait`, then `reset`. Restart the duck-policy module if it was running old code: `kill -9` the `duck_policy.main` process, as in milestone 2.
- [ ] **Step 2: Smoke test.** Run `dancer start {"cycles": 2}` and take `GetImages` frames of `overview` at phases 0.25 and 0.75 of the first cycle, then `status` after 13 s: `holding` true, state `running`. Then `stop`.
- [ ] **Step 3: Parity.**
  1. `uv run scripts/duck_parity.py ours <cell.json> dist/dance-ours.json --episodes 20 --phase 6.0 --service dancer`
  2. `uv run scripts/duck_parity.py report dist/dance-reference.json dist/dance-ours.json dist/dance-parity.md --reference-table dist/microduck-dance-bob-sway/reference.json`
- [ ] **Step 4: Handover:** `uv run scripts/duck_parity.py handover <cell.json> --cycles 2`.
- [ ] **Step 5: Write `docs/superpowers/specs/2026-11-21-microduck-dance-results.md`,** in the same shape as the milestone 2 parity results:
  - the verdict table;
  - per-check numbers;
  - the explained failures;
  - training numbers (iterations, wall time, final `dance_pose`, `fell_over`);
  - two `overview` frames;
  - what's next.

  Commit the results, the machine config and `dances/`.

---

## Self-review notes

- **Spec coverage:**
  - Part 1 (pinned duck, dance file, tool, foot check, record): Tasks 1–5.
  - Part 2 (job inputs, builder, task, package, `--stats --phase`): Tasks 6–10.
  - Part 3 (phase mode, dancer, evaluation, handover): Tasks 11–13.
  - Testing section: the unit tests above, plus the probe in Task 9.
- **Pinned in a task rather than assumed:** the `JointPositions` payload shape and component naming (Task 5 Step 6), and `find_joints` ordering (Task 7). Mjlab config details are checked in the image (Task 8 Step 5).
- **Controller-only steps,** run with the user's go-ahead where needed: creating machines and keys, uploading packages, pushing images, registering containers, uploading scripts, submitting jobs.
