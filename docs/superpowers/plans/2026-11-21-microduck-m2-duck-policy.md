# MicroDuck milestone 2: duck-policy and walk parity — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Status 2026-11-21: Tasks 1–9 done.** Results are in
`docs/superpowers/specs/2026-11-21-microduck-parity-results.md`.
- **Verdict:** strict FAIL, on velstand's backward-start basin (the reference stands still from 3
  of 20 perturbed stances) and one overrun stop. Forward and turn parity is within 2–7 %, and p99
  tick time is about 6 ms.
- **Fixes beyond the plan:**
  - golden ticks from the turning segment;
  - loop warm-up, no catch-up, stale state needs 3 reads, shielded limp;
  - call timeout 50 ms (from 15);
  - parity checks per segment against the reference, including outlier counts and "walker ran".
- **Open:** whether to raise the `late_streak` overrun rule.

**Goal:** a Viam module, `duck_policy`, runs Pollen's `velstand` walk policy at 50 Hz against the
simulated MicroDuck through the robot's `io` DoCommand and the `viam:mlmodel:onnx-cpu` service. It
passes a parity check against Pollen's own CPU reference, `infer_policy.py`.

**Architecture:** spec units 4–5 of `docs/superpowers/specs/2026-11-21-microduck-sim-to-sim-design.md`.
Read its "Milestone 2 decisions" section first: it overrides the units below it.

- **Policy loop.** A pure observation builder (`obs.py`) and a loop (`loop.py`: read, infer, write,
  safety) sit behind a small IO protocol (`io.py`: `SimIO` now, `BusIO` on the robot later).
- **Viam layer.** A thin generic service (`service.py`) wires the loop to the duck robot component and
  the mlmodel service.
- **Parity.** A reference runner wraps `infer_policy.py`'s own code headless, and a harness runs the
  same command script through both stacks and compares falls, tracking and timing.

**Tech stack:** Python 3.12, numpy, viam-sdk ≥ 0.85, onnxruntime (reference and tests only), MuJoCo 4.0
and BAM (via the mujoco-sim worktree's environment), `viam:mlmodel:onnx-cpu` 0.2.2, pytest, uv.

**Where things are:**
- This repo, branch `peg-insert-design`: everything in this plan.
- mujoco-sim with the floating base, BAM and `io`, unreleased (viam-labs/viam-mujoco-sim#7): the worktree
  `/Users/nick.hehr/src/viam-mujoco-sim-floating`. Its environment has MuJoCo 4.0, BAM, numpy and
  viam-sdk. Run anything that imports `mujoco_module` or `bam` with
  `uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating ...`, adding `--with onnxruntime`
  where ONNX is needed.
- Pollen's code at commit `0daab160d3fb1c2cbaba16a4a19052f36709dd9e`: a clone at
  `${TMPDIR:-/tmp}/gh-clones-${CLAUDE_SESSION_ID}/microduck_rl` (re-clone and check out that commit if
  it's missing). Scripts take its path as an argument; nothing hard-codes it.
- Preview machine `microduck-preview` `8e5ca4b7-c352-4ea7-b8c0-9ddf52da6bb1` (DevRelDemos/test):
  - viam-server runs on this Mac, and mujoco-sim is a local module from the worktree.
  - Credentials are in `/private/tmp/claude-501/-Users-nick-hehr-src-viam-rslrl-policy-training/3856841f-0045-481c-bb70-ba5b784e7265/scratchpad/microduck-preview/cell.json`;
    never print or commit them.
  - The machine config is written with the Viam MCP tools (`read_machine_config` / `set_machine_config`).

**Facts this plan relies on** (verified 2026-11-21; file:line refer to `microduck_rl@0daab16`):

- **Observation** (61 values, `scripts/infer_policy.py:656-682`):

  | Slice | Term | Source |
  |---|---|---|
  | `[0:3]` | gyro | the `imu_ang_vel` sensor; the imu site sits on `trunk_base` with identity orientation |
  | `[3:6]` | projected gravity | world `(0,0,-1)` rotated into `xquat[trunk_base]` |
  | `[6:20]` | joint positions | `qpos - DEFAULT_POSE` |
  | `[20:34]` | joint velocities | `qvel` |
  | `[34:48]` | last action | the raw ONNX output |
  | `[48:61]` | command | `[vx, vy, wz, neck_pitch, head_pitch, head_yaw, head_roll, x, y, z, roll, pitch, yaw]` |

  - The command has no scaling. A walk sends twist only; the head and body slots are zero.
  - No noise, no history, no clock term.
  - mujoco-sim's `io` provides gyro (the free joint's angular qvel, which equals the sensor for an
    identity imu site) and gravity (the same rotation), as the Task 7 review of milestone 1 verified.
- **Action:** `target = DEFAULT_POSE + action * 1.0` (`infer_policy.py:873-875`). There is no clipping.
  Joints are in policy order: left leg ×5, neck/head ×4, right leg ×5.
- **Default pose** (`infer_policy.py:126-141`, equal to the ONNX `default_joint_pos` metadata and to the STAND keyframe):
  `[0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530]`.
- **Reference loop:**
  - Reset (`infer_policy.py:1376-1387`):
    - trunk qpos `(0, 0, 0.125, 1, 0, 0, 0)` and joints at DEFAULT_POSE;
    - `bam_ctrl.reset(qpos)`, then targets = DEFAULT_POSE;
    - `mj_forward`.
  - Each tick: `infer()`, `apply_action()`, then 4 × (`bam_ctrl.update()`; `mj_step`) at timestep 0.005
    (`infer_policy.py:1683-1777`).
  - BAM settings: xl330/m6, kp_fw 200, vin 7.4, vin_drop_gain 0.1, vin_min 6.0.
  - `PolicyInference(model, data, walking_onnx_path=..., bam_ctrl=..., use_projected_gravity=True,
    new_cmd_obs=True)`. Set `policy.vel_cmd` and call `policy._update_command()` before each `infer()`.
- **Command ranges** (training, `tasks/microduck_velocity_env_cfg.py:643-650`): vx ±0.4, vy ±0.3,
  wz ±1.0. velstand stands at an exact-zero command.
- **`io` wire types:** numbers come back as floats, so compare `int(generation)`. A JSON null in a
  request arrives as int 0; the module never sends nulls.
- **onnx-cpu:** module `viam:onnx-cpu`, model `viam:mlmodel:onnx-cpu`, API `rdk:service:mlmodel`.
  Its config reads only `model_path` (must end `.onnx`). Tensor names come from the graph: `obs` in,
  `actions` out, float32.
- **Duck in mujoco-sim:**
  - limb arms need `home_keyframe: "STAND"`;
  - `head-cam` uses quaternion W -0.5, X 0.5, Y -0.5, Z 0.5;
  - the world should be paused before `reset` so the duck starts standing;
  - with no policy it tips over in about 1 s.

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/export_duck_scene.py` (modify) | Build `duck-sim` from `robot_groundcontact.xml` and `scene.xml`'s STAND keyframe |
| `scripts/duck_stand_check.py` (modify) | Docstring only: the ground-contact model has trunk collision, so a fallen duck rests on the floor instead of sinking |
| `scripts/fetch_walk_policy.py` (new) | Download velstand at the pinned revision and verify sha256; write `dist/microduck-walk/{policy.onnx, config.json, LICENSE-POLICY.md}` |
| `scripts/duck_reference.py` (new) | Headless reference built on `infer_policy.py`'s code: runs a command script; records trajectory, golden ticks and action statistics |
| `scripts/duck_parity.py` (new) | Runs the reference and our stack on the same script; computes metrics; writes a JSON and markdown report |
| `duck_policy/__init__.py` (new) | Package marker |
| `duck_policy/obs.py` (new) | `build_obs`, `Command`: pure numpy, no Viam |
| `duck_policy/io.py` (new) | `Snapshot`, `DuckIO` protocol, `SimIO` (the robot's io DoCommand) |
| `duck_policy/loop.py` (new) | `Limits`, `Stop`, `PolicyLoop`: start ramp, the 50 Hz read-infer-write loop, stop conditions, stats |
| `duck_policy/service.py` (new) | Viam generic service `devrel:duck-policy:controller`: config, DoCommand |
| `duck_policy/main.py`, `meta.json`, `run.sh`, `requirements.txt` (new) | Module entrypoint and packaging, as `policy_controller/` does |
| `tests/test_duck_obs.py`, `tests/test_duck_loop.py`, `tests/test_duck_service.py` (new) | Unit tests |
| `tests/data/duck_golden.json` (new) | Golden reference ticks for the observation test |
| `docs/superpowers/specs/duck-sim-machine.json` (modify) | duck-sim 0.2.0, the 5 mm offset, the walk package, the mlmodel and duck-policy services |
| `docs/superpowers/specs/<date>-microduck-parity-results.md` (new) | The parity report |

---

### Task 1: duck-sim 0.2.0 from the ground-contact model

**Files:**
- Modify: `scripts/export_duck_scene.py`
- Modify: `scripts/duck_stand_check.py`

- [ ] **Step 1: Switch the exporter.** In `export_duck_scene.py`, change the two file names:
  - `shutil.copy(src / "robot_walk.xml", out / "robot_walk.xml")` copies `robot_groundcontact.xml` instead;
  - the include in `duck.xml` becomes `<include file="robot_groundcontact.xml"/>`;
  - the STAND keyframe is read from `scene.xml`, not `scene_walk.xml`.

  Update the docstring to say why: velstand, the walk policy, was trained on the ground-contact model,
  `infer_policy.py` loads it, and it has trunk and head collision. Keep the comment stripping:
  `scene.xml` also has the old STAND commented out (line 26).

  In `duck_stand_check.py`, update only the docstring sentence about the duck sinking through the
  floor: with the ground-contact model it falls onto the floor. The assertions stay.

- [ ] **Step 2: Export and check the keyframe.**

```bash
uv run python scripts/export_duck_scene.py "${TMPDIR:-/tmp}/gh-clones-${CLAUDE_SESSION_ID}/microduck_rl"
grep -c -- "-0.457924" dist/duck-sim/duck.xml   # expect >= 2 (qpos and ctrl)
```

- [ ] **Step 3: Verify the imu site is still identity on the trunk** (the gyro equivalence depends on it):

```bash
uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('dist/duck-sim/duck.xml')
s = m.site('imu'); print(m.body(s.bodyid[0]).name, np.round(s.quat, 6))"
```

Expected: `trunk_base [1. 0. 0. 0.]` (or `[-1, 0, 0, 0]`, the same rotation). If it isn't, stop: the
gyro needs rotating into the site frame in `io`, which is a mujoco-sim change.

- [ ] **Step 4: Run the stand check against the new package.**

```bash
uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating python scripts/duck_stand_check.py dist/duck-sim
```

Expected: the same five lines as in milestone 1. "upright at 0.3 s" must hold. The 1 s tilt may
differ, and the trunk should no longer end below z 0. Paste the output into the commit message.

- [ ] **Step 5: Upload duck-sim 0.2.0 and commit.**

```bash
viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=duck-sim --version=0.2.0 \
  --type=archive --path=dist/duck-sim.tar.gz
git add scripts/export_duck_scene.py scripts/duck_stand_check.py
git commit -m "feat: duck-sim 0.2.0 from the ground-contact model velstand was trained on"
```

This upload is covered by the user's standing authorization for this project's DevRelDemos uploads.
Never re-upload an existing version.

---

### Task 2: The walk policy package

**Files:**
- Create: `scripts/fetch_walk_policy.py`

- [ ] **Step 1: Write the fetcher.**

```python
"""Fetch Pollen's MicroDuck walk policy (velstand) at a pinned Hugging Face revision and package it for
viam:mlmodel:onnx-cpu and duck-policy.

    uv run --with onnxruntime python scripts/fetch_walk_policy.py <duck-sim dir> [out_dir]   (default dist/microduck-walk)

Writes out_dir/policy.onnx (sha256-checked), config.json (what duck-policy reads) and LICENSE-POLICY.md,
then out_dir.tar.gz for `viam packages upload --type ml_model`. The joint limits come from the duck-sim
robot file, so pass the exported duck-sim directory.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "pollen-robotics/microduck-policies"
REVISION = "236d1835b03ac7b45d962efbf2fe3a6940206112"
FILE = "velstand.onnx"
SHA256 = "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a"  # HF x-linked-etag at REVISION
DEFAULT_POSE = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.4530]  # infer_policy.py:126-141
JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]


def main(duck_sim: Path, out: Path) -> None:
    import mujoco
    import onnxruntime as ort

    out.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
    data = urllib.request.urlopen(url).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"{FILE} sha256 {digest} != pinned {SHA256}")
    (out / "policy.onnx").write_bytes(data)

    session = ort.InferenceSession(str(out / "policy.onnx"))
    meta = session.get_modelmeta().custom_metadata_map
    (inp,), (outp,) = session.get_inputs(), session.get_outputs()
    assert (inp.name, list(inp.shape), outp.name, list(outp.shape)) == ("obs", [1, 61], "actions", [1, 14])
    # The metadata keeps 3 decimals; infer_policy.py:126-141 has 4. Write infer_policy's, so both sides
    # of the parity check use the same pose, and check the metadata agrees within its rounding.
    default_pose = DEFAULT_POSE
    meta_pose = [float(v) for v in meta["default_joint_pos"].split(",")]
    assert all(abs(a - b) <= 6e-4 for a, b in zip(meta_pose, DEFAULT_POSE)), meta_pose
    joint_names = meta["joint_names"].split(",")
    assert joint_names == JOINTS, joint_names
    action_scale = float(meta["action_scale"])

    model = mujoco.MjModel.from_xml_path(str(duck_sim / "duck.xml"))
    limits = [[float(v) for v in model.joint(n).range] for n in JOINTS]

    config = {
        "schema": 1,
        "source": {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": SHA256, "license": "apache-2.0"},
        "model": {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14},
        "joints": JOINTS,
        "default_pose": default_pose,
        "action_scale": action_scale,
        "joint_limits": limits,
        "control_hz": 50,
        "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]},
        "safety": {},  # action_bound and max_step_rad: filled in by scripts/duck_reference.py --stats
        "observation": "gyro(3) gravity(3) q-default_pose(14) dq(14) last_action(14) command(13): microduck_rl@0daab16 infer_policy.py:656-682",
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "LICENSE-POLICY.md").write_text(
        f"policy.onnx is {FILE} from https://huggingface.co/{REPO} at revision {REVISION}, "
        "Copyright Pollen Robotics, licensed Apache-2.0.\n"
    )
    pack(out)
    print(f"wrote {out}: default_pose {default_pose}, action_scale {action_scale}")


def pack(out: Path) -> None:
    with tarfile.open(out.with_suffix(".tar.gz"), "w:gz") as tar:
        for path in sorted(out.iterdir()):
            tar.add(path, arcname=path.name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else Path("dist/microduck-walk"))
```

The SHA256 and the metadata keys (`default_joint_pos`, `joint_names`, `action_scale`) were
verified against the real file in plan review.

- [ ] **Step 2: Run it.**

```bash
uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating --with onnxruntime python scripts/fetch_walk_policy.py dist/duck-sim
```

Expected: it prints the default pose (`infer_policy`'s 4-decimal values, after the metadata check
passes) and action scale 1.0.

- [ ] **Step 3: Commit** (the package itself stays in `dist/`; it's uploaded in Task 8):

```bash
git add scripts/fetch_walk_policy.py
git commit -m "feat: walk policy package (velstand @236d183, sha256-pinned) for onnx-cpu and duck-policy"
```

---

### Task 3: Observation builder

**Files:**
- Create: `duck_policy/__init__.py` (empty)
- Create: `duck_policy/obs.py`
- Test: `tests/test_duck_obs.py`

- [ ] **Step 1: Write the failing test.**

```python
import numpy as np
import pytest

from duck_policy.obs import Command, build_obs

DEFAULT = np.array([0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530])


def test_layout_matches_infer_policy():
    q, dq = DEFAULT + 0.01 * np.arange(14), -0.02 * np.arange(14)
    last = 0.5 * np.ones(14)
    obs = build_obs(gyro=[0.1, 0.2, 0.3], gravity=[0.0, 0.05, -0.99], q=q, dq=dq,
                    default_pose=DEFAULT, last_action=last, command=Command(0.2, -0.1, 0.5))
    assert obs.dtype == np.float32 and obs.shape == (61,)
    assert obs[0:3] == pytest.approx([0.1, 0.2, 0.3])
    assert obs[3:6] == pytest.approx([0.0, 0.05, -0.99])
    assert obs[6:20] == pytest.approx(0.01 * np.arange(14), abs=1e-6)
    assert obs[20:34] == pytest.approx(dq)
    assert obs[34:48] == pytest.approx(last)
    assert obs[48:51] == pytest.approx([0.2, -0.1, 0.5])
    assert not obs[51:61].any()  # head and body pose slots: zero for a walk


def test_command_clips_to_the_trained_ranges():
    c = Command(1.0, -1.0, 3.0).clipped([-0.4, -0.3, -1.0], [0.4, 0.3, 1.0])
    assert (c.vx, c.vy, c.wz) == (0.4, -0.3, 1.0)
```

- [ ] **Step 2: Run it and confirm it fails.** `uv run pytest tests/test_duck_obs.py -q` should fail
  with `ModuleNotFoundError: duck_policy`.

- [ ] **Step 3: Implement `duck_policy/obs.py`.**

```python
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
```

- [ ] **Step 4: Run the test and confirm it passes.**
- [ ] **Step 5: Commit** with message `feat: duck_policy observation builder (Pollen's 61-value layout)`.

---

### Task 4: Reference runner and the golden test

**Files:**
- Create: `scripts/duck_reference.py`
- Create: `tests/data/duck_golden.json` (generated)
- Modify: `tests/test_duck_obs.py` (add the golden test)

- [ ] **Step 1: Write the reference runner.**

```python
"""Pollen's own CPU reference for the MicroDuck walk policy, headless: microduck_rl's scripts/infer_policy.py
code (load_mujoco_with_bam, PolicyInference), its reset sequence and its 4 x 5 ms steps per action,
driven by a command script instead of the keyboard and run as fast as it can.

    uv run --project <mujoco-sim worktree> --with onnxruntime python scripts/duck_reference.py \
        <microduck_rl checkout> <duck-sim dir> <policy.onnx> <out.json> [--golden tests/data/duck_golden.json] [--stats <walk config.json>]

The scene is the duck-sim package's duck.xml, so both sides of the parity check simulate the same file
with the same MuJoCo build. Records, every 0.1 s of sim time: t, trunk position and quaternion, gravity.
--golden saves the first 25 ticks' raw inputs and the observation infer_policy built, for the obs test.
--stats writes action_bound (2 x p99.9 |action|) and max_step_rad (2 x p99.9 |target change per tick|)
into the walk package's config.json safety block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The command script: (seconds, vx, vy, wz). velstand has a dead zone (Pollen's docs/velstand_policy.md,
# microduck_velstand_env_cfg.py:344-380): measured on infer_policy's own scene, vx 0.15-0.2, |vx| <= 0.3
# backwards, and wz +-0.5 all stand still; vx 0.4 -> ~0.17 m/s, vx -0.4 -> ~-0.14, vx 0.4 + wz +-0.6 ->
# ~+-0.38 rad/s. So every moving segment uses values that walk (spec, milestone 2 decisions).
SCRIPT = [(2.0, 0.0, 0.0, 0.0), (8.0, 0.4, 0.0, 0.0), (5.0, 0.4, 0.0, 0.6), (5.0, 0.4, 0.0, -0.6),
          (4.0, 0.0, 0.0, 0.0), (6.0, -0.4, 0.0, 0.0)]
DURATION = sum(seg[0] for seg in SCRIPT)
TRUNK_Z = 0.125  # infer_policy.py:1376-1383


def command_at(t: float) -> tuple[float, float, float]:
    for seconds, vx, vy, wz in SCRIPT:
        if t < seconds:
            return vx, vy, wz
        t -= seconds
    return SCRIPT[-1][1:]


def run(checkout: Path, duck_xml: Path, onnx: Path, golden: Path | None = None) -> dict:
    sys.path.insert(0, str(checkout / "scripts"))
    import mujoco
    from infer_policy import BAM_VIN_MIN, PolicyInference, load_bam_model, load_mujoco_with_bam

    bam_model = load_bam_model(200.0, 7.4, None)
    # duck.xml has no floor (mujoco-sim adds one when it builds the scene): wrap it with the same plane,
    # next to duck.xml so the include and meshdir resolve, and remove the wrapper so it never ships.
    scene = duck_xml.parent / "_reference_scene.xml"
    scene.write_text('<mujoco model="reference">\n  <include file="duck.xml"/>\n  <worldbody>\n'
                     '    <geom name="floor" type="plane" size="0 0 0.05"/>\n  </worldbody>\n</mujoco>\n')
    try:
        model, data, bam_ctrl, _ = load_mujoco_with_bam(str(scene), bam_model, 0.005, 0.1, BAM_VIN_MIN)
    finally:
        scene.unlink()
    policy = PolicyInference(model, data, walking_onnx_path=str(onnx), bam_ctrl=bam_ctrl,
                             use_projected_gravity=True, new_cmd_obs=True)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[free]
    data.qpos[qa:qa + 3] = (0.0, 0.0, TRUNK_Z)
    data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    samples, ticks, actions, steps = [], [], [], []
    prev_target = np.array(policy.default_pose, float)
    tick = 0
    while data.time < DURATION - 1e-9:
        policy.vel_cmd[:] = command_at(data.time)
        policy._update_command()
        if golden is not None and tick < 25:
            ticks.append({
                "gyro": policy.get_base_ang_vel().tolist(), "gravity": policy.get_projected_gravity().tolist(),
                "q": data.qpos[policy.joint_qpos_indices].tolist(), "dq": data.qvel[policy.joint_qvel_indices].tolist(),
                "last_action": policy.last_action.tolist(), "command": policy.vel_cmd.tolist(),
                "obs": policy.get_observations().tolist(),
            })
        action = policy.infer()
        policy.apply_action(action)
        target = np.array(policy.default_pose) + action * policy.action_scale
        actions.append(np.abs(action))
        steps.append(np.abs(target - prev_target))
        prev_target = target
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        tick += 1
        if tick % 5 == 0:  # every 0.1 s
            quat = data.qpos[qa + 3:qa + 7].copy()
            samples.append({"t": round(float(data.time), 4), "trunk_pos": data.qpos[qa:qa + 3].tolist(),
                            "trunk_quat_wxyz": quat.tolist(), "gravity": policy.get_projected_gravity().tolist()})
    if golden is not None:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps({"default_pose": list(map(float, policy.default_pose)), "ticks": ticks}, indent=1))
    return {
        "samples": samples,
        "action_p999": float(np.percentile(np.concatenate(actions), 99.9)),
        "step_p999": float(np.percentile(np.concatenate(steps), 99.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout", type=Path)
    ap.add_argument("duck_sim", type=Path)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--stats", type=Path, help="walk package config.json to write the safety block into")
    args = ap.parse_args()
    result = run(args.checkout.resolve(), (args.duck_sim / "duck.xml").resolve(), args.onnx.resolve(), args.golden)
    args.out.write_text(json.dumps(result))
    if args.stats:
        cfg = json.loads(args.stats.read_text())
        cfg["safety"] = {"action_bound": round(2 * result["action_p999"], 3), "max_step_rad": round(2 * result["step_p999"], 3)}
        args.stats.write_text(json.dumps(cfg, indent=2) + "\n")
    last = result["samples"][-1]
    print(f"ran {last['t']:.1f} s sim; final trunk z {last['trunk_pos'][2]:.3f}, gravity z {last['gravity'][2]:.3f}; "
          f"action p99.9 {result['action_p999']:.3f}, step p99.9 {result['step_p999']:.3f}")


if __name__ == "__main__":
    main()
```

Notes for the implementer:
- `infer_policy.py` imports `termios` and `tty` at the top, which is fine on macOS and Linux. It
  prints a few lines at construction; that's fine.
- If `PolicyInference`'s constructor needs more arguments, or if `_update_command` doesn't exist under
  that name at 0daab16, read the class (`infer_policy.py:212-520`) and adapt. Keep the reset and step
  sequence exactly as above.
- `load_bam_model(kp_fw, vin, max_current)` and
  `load_mujoco_with_bam(xml, bam_model, timestep, vin_drop_gain, vin_min)` are at
  `infer_policy.py:49-104`.
- The floor wrapper was checked against Pollen's own `scene.xml`: x = 1.668 at t = 10 s at vx 0.4,
  an exact match.
- Print `model.opt` (integrator, cone, impratio, iterations, tolerance) and compare it once with the
  scene mujoco-sim builds. To get that, read `sim.model.opt` from a `SimManager` built the way
  `duck_stand_check.py` builds it. Record any difference in the commit message: it is a parity
  difference besides real-time stepping. The two differ: the reference compiles with Euler (from the
  robot file), while mujoco-sim forces `implicitfast`. Set
  `model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST` after loading. Plan review found
  the dead-zone numbers identical to 3 decimals either way, since BAM leaves no velocity terms for
  the two integrators to differ on. The change still removes one variable.

- [ ] **Step 2: Run it** (once, creating the golden file and the safety stats):

```bash
REF="${TMPDIR:-/tmp}/gh-clones-${CLAUDE_SESSION_ID}/microduck_rl"
uv run --project /Users/nick.hehr/src/viam-mujoco-sim-floating --with onnxruntime python scripts/duck_reference.py \
  "$REF" dist/duck-sim dist/microduck-walk/policy.onnx dist/duck-reference.json \
  --golden tests/data/duck_golden.json --stats dist/microduck-walk/config.json
```

Expected: about 30 s of sim time. The final gravity z should be below -0.9 (it didn't fall). If it
falls, stop and report. velstand at these commands should walk, and a fall means the scene or BAM
setup differs from Pollen's: check the trunk start height, physics_dt 0.005 and the BAM settings
before anything else.

- [ ] **Step 3: Add the golden test** to `tests/test_duck_obs.py`:

```python
import json
from pathlib import Path

GOLDEN = Path(__file__).parent / "data" / "duck_golden.json"


def test_build_obs_reproduces_infer_policy_exactly():
    g = json.loads(GOLDEN.read_text())
    assert len(g["ticks"]) == 25
    for tick in g["ticks"]:
        vx, vy, wz = tick["command"]
        obs = build_obs(gyro=tick["gyro"], gravity=tick["gravity"], q=tick["q"], dq=tick["dq"],
                        default_pose=g["default_pose"], last_action=tick["last_action"], command=Command(vx, vy, wz))
        assert obs == pytest.approx(np.array(tick["obs"], np.float32), abs=1e-6)
```

- [ ] **Step 4: Run** `uv run pytest tests/test_duck_obs.py -q` and confirm it passes.

- [ ] **Step 5: Repack the walk package** with the safety block, by running
  `uv run python -c "from pathlib import Path; import sys; sys.path.insert(0, 'scripts'); import fetch_walk_policy as f; f.pack(Path('dist/microduck-walk'))"`.
  Then commit:

```bash
git add scripts/duck_reference.py tests/data/duck_golden.json tests/test_duck_obs.py
git commit -m "feat: headless infer_policy reference runner; golden obs test reproduces it exactly"
```

---

### Task 5: IO and the policy loop

**Files:**
- Create: `duck_policy/io.py`
- Create: `duck_policy/loop.py`
- Test: `tests/test_duck_loop.py`

- [ ] **Step 1: Write `duck_policy/io.py`.** It is thin and is tested through the loop's fakes and
  Task 6's service test.

```python
"""How duck-policy reaches the duck. SimIO talks to viam-labs:mujoco-sim's robot io DoCommand; a BusIO
(later, on the real duck) will talk to the Dynamixel bus. Units: rad, rad/s, trunk frame; gravity is a
unit vector, upright [0, 0, -1]."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Snapshot:
    t: float
    q: list[float]
    dq: list[float]
    gyro: list[float]
    gravity: list[float]
    generation: int
    torque: bool
    trunk_pos: list[float] | None = None


class DuckIO(Protocol):
    async def read(self) -> Snapshot: ...

    async def write(self, targets: list[float]) -> None: ...

    async def torque(self, on: bool) -> Snapshot: ...


class SimIO:
    """The duck robot's io DoCommand. Checks the joint order once, against the policy's."""

    def __init__(self, duck: Any, joints: list[str]) -> None:
        self._duck, self._joints, self._checked = duck, joints, False

    async def _io(self, request: dict) -> Snapshot:
        r = await self._duck.do_command({"io": request})
        if not self._checked:
            if list(r["joints"]) != self._joints:
                raise ValueError(f"duck joints {list(r['joints'])} != the policy's {self._joints}")
            self._checked = True
        return Snapshot(t=float(r["t"]), q=list(r["q"]), dq=list(r["dq"]), gyro=list(r["gyro"]),
                        gravity=list(r["gravity"]), generation=int(r["generation"]), torque=bool(r["torque"]),
                        trunk_pos=list(r.get("trunk_pos", [])) or None)

    async def read(self) -> Snapshot:
        return await self._io({})

    async def write(self, targets: list[float]) -> None:
        await self._io({"targets": [float(v) for v in targets]})

    async def torque(self, on: bool) -> Snapshot:
        return await self._io({"torque": bool(on)})
```

- [ ] **Step 2: Write the failing loop tests** in `tests/test_duck_loop.py`:

```python
import asyncio

import numpy as np
import pytest

from duck_policy.io import Snapshot
from duck_policy.loop import Limits, PolicyLoop
from duck_policy.obs import Command

DEFAULT = [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530]
CFG = {"default_pose": DEFAULT, "action_scale": 1.0, "joint_limits": [[-1.6, 1.6]] * 14,
       "control_hz": 50, "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]},
       "safety": {"action_bound": 3.0, "max_step_rad": 0.5}}


class FakeIO:
    """Sim time advances 20 ms per read; gravity and generation can be scripted per read."""

    def __init__(self, gravity=lambda i: [0, 0, -1], generation=lambda i: 1, frozen=False, hang_at=None):
        self.i, self.writes, self.torques = 0, [], []
        self.gravity, self.generation, self.frozen, self.hang_at = gravity, generation, frozen, hang_at

    def _snap(self):
        return Snapshot(t=0.0 if self.frozen else 0.02 * self.i, q=list(DEFAULT), dq=[0.0] * 14, gyro=[0, 0, 0],
                        gravity=self.gravity(self.i), generation=self.generation(self.i), torque=True)

    async def read(self):
        self.i += 1
        if self.hang_at is not None and self.i >= self.hang_at:
            await asyncio.sleep(1.0)
        return self._snap()

    async def write(self, targets):
        self.writes.append(list(targets))

    async def torque(self, on):
        self.torques.append(on)
        return self._snap()


def zero_policy(obs):
    return np.zeros(14, np.float32)


def loop(io, infer=zero_policy, ticks=200, **limits):
    lp = PolicyLoop(io, infer, CFG, Limits(ramp_s=0.1, **limits))
    lp.command = Command(0.1, 0.0, 0.0)
    return lp, asyncio.run(lp.run(max_ticks=ticks))


def test_a_healthy_run_writes_default_plus_action_and_counts_ticks():
    io = FakeIO()
    lp, reason = loop(io, ticks=20)
    assert reason == "max_ticks" and lp.ticks == 20
    assert io.writes[-1] == pytest.approx(DEFAULT, abs=1e-6)  # zero action: the default pose
    assert lp.last_obs[48:51] == pytest.approx([0.1, 0.0, 0.0])


def test_fall_goes_limp_after_three_ticks():
    io = FakeIO(gravity=lambda i: [0.9, 0, -0.3] if i > 10 else [0, 0, -1])
    lp, reason = loop(io)
    assert reason == "fall" and io.torques == [True, False]


def test_bad_action_goes_limp():
    io = FakeIO()
    lp, reason = loop(io, infer=lambda obs: np.full(14, np.nan, np.float32))
    assert reason == "bad_action" and io.torques[-1] is False
    io2 = FakeIO()
    _, reason2 = loop(io2, infer=lambda obs: np.full(14, 9.0, np.float32))
    assert reason2 == "bad_action"


def test_stale_state_and_rebuild_stop_and_hold():
    io = FakeIO(frozen=True)
    _, reason = loop(io)
    assert reason == "stale_state" and io.torques == [True]  # held, not limp
    io2 = FakeIO(generation=lambda i: 1 if i < 20 else 2)
    _, reason2 = loop(io2)
    assert reason2 == "rebuilt" and io2.torques == [True]


def test_hung_call_stops():
    io = FakeIO(hang_at=15)
    _, reason = loop(io, call_timeout_s=0.05)
    assert reason == "hung_call"


def test_targets_are_clipped_and_rate_capped():
    io = FakeIO()
    lp, _ = loop(io, infer=lambda obs: np.full(14, 2.5, np.float32), ticks=60)
    w = np.array(io.writes)
    assert (w <= 1.6 + 1e-9).all()  # joint limit
    assert np.abs(np.diff(w, axis=0)).max() <= 0.5 + 1e-9  # max_step_rad


def test_set_command_is_clipped():
    lp = PolicyLoop(FakeIO(), zero_policy, CFG, Limits())
    lp.set_command(Command(1.0, 0.0, -5.0))
    assert (lp.command.vx, lp.command.wz) == (0.4, -1.0) and lp.command_clipped
```

- [ ] **Step 3: Run them and confirm they fail** (`ModuleNotFoundError`).

- [ ] **Step 4: Implement `duck_policy/loop.py`.**

```python
"""duck-policy's 50 Hz loop: ramp to the default pose, then each tick read the duck, build the
observation, infer, and write default_pose + action * scale (clipped to joint limits, rate-capped).
Read-infer-write in that order matches infer_policy.py (observe, infer, apply, then step). Every stop
condition ends the loop with a reason; a fall or a bad action also turns torque off (limp), the rest hold
the last targets. See spec unit 4."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from .io import DuckIO, Snapshot
from .obs import Command, build_obs


@dataclass(frozen=True)
class Limits:
    fall_gravity_z: float = -0.5  # trunk tilted ~60 degrees
    fall_ticks: int = 3
    late_window: int = 50
    late_fraction: float = 0.05
    late_streak: int = 3
    call_timeout_s: float = 0.015
    ramp_s: float = 1.0


class Stop(Exception):
    def __init__(self, reason: str, limp: bool = False) -> None:
        super().__init__(reason)
        self.reason, self.limp = reason, limp


class PolicyLoop:
    def __init__(self, io: DuckIO, infer: Callable[[np.ndarray], np.ndarray | Awaitable[np.ndarray]],
                 cfg: dict, limits: Limits) -> None:
        self.io, self.infer, self.limits = io, infer, limits
        self.default = np.asarray(cfg["default_pose"], float)
        self.scale = float(cfg["action_scale"])
        lim = np.asarray(cfg["joint_limits"], float)
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.dt = 1.0 / float(cfg.get("control_hz", 50))
        self.twist_lo, self.twist_hi = cfg["command"]["twist_min"], cfg["command"]["twist_max"]
        safety = cfg.get("safety") or {}
        self.action_bound = float(safety.get("action_bound", 5.0))
        self.max_step = float(safety.get("max_step_rad", 0.5))
        self.command, self.command_clipped = Command(), False
        self.ticks, self.late, self.reason, self.last_obs, self.t0, self.t_end = 0, 0, None, None, None, None
        self.tick_ms: deque[float] = deque(maxlen=500)

    def set_command(self, command: Command) -> None:
        clipped = command.clipped(self.twist_lo, self.twist_hi)
        self.command_clipped, self.command = clipped != command, clipped

    async def _call(self, coro):
        try:
            return await asyncio.wait_for(coro, self.limits.call_timeout_s)
        except asyncio.TimeoutError:
            raise Stop("hung_call") from None

    async def _infer(self, obs: np.ndarray) -> np.ndarray:
        out = self.infer(obs)
        if asyncio.iscoroutine(out) or isinstance(out, Awaitable):
            out = await self._call(out)
        return np.asarray(out, float).reshape(14)

    def _target(self, action: np.ndarray, prev: np.ndarray) -> np.ndarray:
        target = np.clip(self.default + action * self.scale, self.lo, self.hi)
        return prev + np.clip(target - prev, -self.max_step, self.max_step)

    async def run(self, max_ticks: int | None = None) -> str:
        try:
            await self._run_timed(max_ticks)
            self.reason = "max_ticks"
        except Stop as stop:
            self.reason = stop.reason
            if stop.limp:
                await self.io.torque(False)
        except asyncio.CancelledError:
            self.reason = "stopped"
            raise
        except Exception as error:  # e.g. a gRPC error from the duck or the mlmodel service
            self.reason = f"error: {error}"
        return self.reason

    async def _run_timed(self, max_ticks: int | None) -> None:
        try:
            await self._run(max_ticks)
        finally:
            self.t_end = time.monotonic()

    async def _run(self, max_ticks: int | None) -> None:
        lim = self.limits
        snap: Snapshot = await self._call(self.io.torque(True))
        generation, prev_t = snap.generation, snap.t
        # Ramp from where the duck is to the default pose. After a reset at STAND it's already there, and
        # the ramp is skipped: holding the pose with no policy, the duck tips over within ~1.5 s (plan review).
        start = np.asarray(snap.q, float)
        n = 0 if np.abs(start - self.default).max() < 0.05 else max(1, round(lim.ramp_s / self.dt))
        for i in range(1, n + 1):
            await self._call(self.io.write((start + (self.default - start) * i / n).tolist()))
            await asyncio.sleep(self.dt)
        if n == 0:
            # Let sim time move past the torque snapshot's t. mujoco-sim steps in 5 ms, so a read 2 ms
            # later often shows the same t, which would stop with "stale_state" on tick 1.
            await asyncio.sleep(self.dt)
        prev_target, last_action = self.default.copy(), np.zeros(14)
        fall, late_hist, late_streak = 0, deque(maxlen=lim.late_window), 0
        t0 = self.t0 = time.monotonic()
        while max_ticks is None or self.ticks < max_ticks:
            deadline = t0 + (self.ticks + 1) * self.dt
            tick_start = time.monotonic()
            snap = await self._call(self.io.read())
            if snap.generation != generation:
                raise Stop("rebuilt")
            if snap.t <= prev_t:
                raise Stop("stale_state")
            prev_t = snap.t
            fall = fall + 1 if snap.gravity[2] > lim.fall_gravity_z else 0
            if fall >= lim.fall_ticks:
                raise Stop("fall", limp=True)
            obs = build_obs(gyro=snap.gyro, gravity=snap.gravity, q=snap.q, dq=snap.dq,
                            default_pose=self.default, last_action=last_action, command=self.command)
            self.last_obs = obs
            action = await self._infer(obs)
            if not np.all(np.isfinite(action)) or np.abs(action).max() > self.action_bound:
                raise Stop("bad_action", limp=True)
            target = self._target(action, prev_target)
            await self._call(self.io.write(target.tolist()))
            prev_target, last_action = target, action
            self.ticks += 1
            elapsed = time.monotonic() - tick_start
            self.tick_ms.append(1000 * elapsed)
            is_late = time.monotonic() > deadline
            self.late += is_late
            late_hist.append(is_late)
            late_streak = late_streak + 1 if is_late else 0
            if late_streak >= lim.late_streak or (len(late_hist) == lim.late_window
                                                  and sum(late_hist) > lim.late_fraction * lim.late_window):
                raise Stop("overruns")
            await asyncio.sleep(max(0.0, deadline - time.monotonic()))

    def status(self) -> dict:
        ms = sorted(self.tick_ms)
        pct = (lambda p: round(ms[min(len(ms) - 1, int(p * len(ms)))], 2)) if ms else (lambda p: None)
        run_s = None if self.t0 is None else round((self.t_end or time.monotonic()) - self.t0, 3)
        return {"ticks": self.ticks, "late": self.late, "run_s": run_s,  # run_s: since the loop's t0 (after any ramp)
                "tick_ms_p50": pct(0.5), "tick_ms_p99": pct(0.99),
                "reason": self.reason, "command": [self.command.vx, self.command.vy, self.command.wz],
                "command_clipped": self.command_clipped}
```

The `last_action` in the observation is the raw action, before scale, clipping and rate cap. That
matches `infer_policy.py:863`, which stores the ONNX output.

- [ ] **Step 5: Run** `uv run pytest tests/test_duck_loop.py -q` **and confirm the tests pass.**
  `FakeIO` with `frozen=True` returns t=0 on every read, so the first loop tick has `t <= prev_t` and
  stops with "stale_state". That is intended.

- [ ] **Step 6: Commit** with message `feat: duck_policy IO protocol (SimIO) and 50 Hz read-infer-write loop with stop conditions`.

---

### Task 6: The Viam service and module packaging

**Files:**
- Create: `duck_policy/service.py`, `duck_policy/main.py`, `duck_policy/meta.json`, `duck_policy/run.sh`, `duck_policy/requirements.txt`
- Test: `tests/test_duck_service.py`

- [ ] **Step 1: Write the failing service tests.**

```python
import asyncio
import json

import numpy as np
import pytest
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from duck_policy.service import DuckPolicy


def cfg(attrs):
    return ComponentConfig(name="walker", api="rdk:service:generic", model="devrel:duck-policy:controller",
                           attributes=dict_to_struct(attrs))


def test_validate_requires_duck_mlmodel_package():
    with pytest.raises(ValueError, match="duck"):
        DuckPolicy.validate_config(cfg({"mlmodel": "walk", "package_dir": "/x"}))
    deps, _ = DuckPolicy.validate_config(cfg({"duck": "duck", "mlmodel": "walk", "package_dir": "/x"}))
    assert sorted(deps) == ["duck", "walk"]


def test_set_command_and_status_without_running(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "joints": ["j"] * 14, "default_pose": [0.0] * 14, "action_scale": 1.0, "joint_limits": [[-1, 1]] * 14,
        "control_hz": 50, "model": {"input": "obs", "output": "actions"},
        "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}, "safety": {}}))
    svc = DuckPolicy("walker")
    svc._configure({"duck": "duck", "mlmodel": "walk", "package_dir": str(tmp_path)}, duck=object(), mlmodel=object())
    out = asyncio.run(svc.do_command({"command": "set_command", "vx": 0.9, "vy": 0, "wz": 0.2}))
    assert out["command"] == [0.4, 0.0, 0.2] and out["command_clipped"] is True
    status = asyncio.run(svc.do_command({"command": "status"}))
    assert status["state"] == "idle"
    with pytest.raises(ValueError, match="unknown command"):
        asyncio.run(svc.do_command({"command": "fly"}))
```

- [ ] **Step 2: Run them and confirm they fail.**

- [ ] **Step 3: Implement `duck_policy/service.py`.**

```python
"""Viam generic service devrel:duck-policy:controller: runs the MicroDuck walk policy at 50 Hz against a
duck that serves the io DoCommand (viam-labs:mujoco-sim's robot now; a hardware module later), with
inference through an mlmodel service (viam:mlmodel:onnx-cpu). DoCommand: start, stop, status,
set_command {vx, vy, wz}. See spec unit 4."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from viam.components.generic import Generic as GenericComponent
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.mlmodel import MLModel
from viam.utils import struct_to_dict

from .io import SimIO
from .loop import Limits, PolicyLoop
from .obs import Command

LIMIT_KEYS = ("fall_gravity_z", "call_timeout_s", "ramp_s")


class DuckPolicy(Generic, EasyResource):
    MODEL = Model(ModelFamily("devrel", "duck-policy"), "controller")

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        for key in ("duck", "mlmodel", "package_dir"):
            if not attrs.get(key):
                raise ValueError(f"attribute {key} is required")
        return [attrs["duck"], attrs["mlmodel"]], []

    @classmethod
    def new(cls, config, dependencies):
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        duck = dependencies[GenericComponent.get_resource_name(attrs["duck"])]
        mlmodel = dependencies[MLModel.get_resource_name(attrs["mlmodel"])]
        self._configure(attrs, duck=duck, mlmodel=mlmodel)

    def _configure(self, attrs: dict, *, duck: Any, mlmodel: Any) -> None:
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
        self.cfg = json.loads((Path(attrs["package_dir"]) / "config.json").read_text())
        self.duck, self.mlmodel = duck, mlmodel
        self.limits = Limits(**{k: float(attrs[k]) for k in LIMIT_KEYS if k in attrs})
        self.input_name = self.cfg["model"]["input"]
        self.output_name = self.cfg["model"]["output"]
        self.task, self.loop, self.command = None, None, Command()

    async def _infer(self, obs: np.ndarray) -> np.ndarray:
        out = await self.mlmodel.infer({self.input_name: obs[None].astype(np.float32)})
        return np.asarray(out[self.output_name])[0]

    def _state(self) -> str:
        if self.task is None:
            return "idle"
        return "running" if not self.task.done() else "stopped"

    async def do_command(self, command, *, timeout=None, **kwargs):
        match command.get("command"):
            case "start":
                if self._state() == "running":
                    raise ValueError("already running")
                self.loop = PolicyLoop(SimIO(self.duck, self.cfg["joints"]), self._infer, self.cfg, self.limits)
                self.loop.set_command(self.command)
                self.task = asyncio.create_task(self.loop.run())
                return {"state": "running"}
            case "stop":
                if self.task and not self.task.done():
                    self.task.cancel()
                    await asyncio.gather(self.task, return_exceptions=True)
                return {"state": self._state(), "reason": self.loop.reason if self.loop else None}
            case "set_command":
                c = Command(float(command.get("vx", 0)), float(command.get("vy", 0)), float(command.get("wz", 0)))
                probe = self.loop or PolicyLoop(SimIO(self.duck, self.cfg["joints"]), self._infer, self.cfg, self.limits)
                probe.set_command(c)
                self.command = probe.command
                if self.loop:
                    self.loop.set_command(c)
                return {"command": [self.command.vx, self.command.vy, self.command.wz],
                        "command_clipped": probe.command_clipped}
            case "status":
                status = self.loop.status() if self.loop else {}
                return {"state": self._state(), **status}
            case other:
                raise ValueError(f"unknown command {other!r}; known: start, stop, status, set_command")

    async def close(self):
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
```

- [ ] **Step 4: Packaging.** Model each file on `policy_controller/`'s:
  - `duck_policy/main.py` imports `DuckPolicy` (which registers the model) and runs
    `Module.run_from_registry()`.
  - `duck_policy/meta.json`:
    - module id `devrel:duck-policy`, visibility `private`;
    - model `devrel:duck-policy:controller`, api `rdk:service:generic`;
    - entrypoint `run.sh`.
  - `duck_policy/run.sh`: copy `policy_controller/run.sh` and keep its `cd "$(dirname "$0")"`, so the
    module's `.venv` and `requirements.txt` stay inside `duck_policy/`. Never point that `.venv` logic
    at the repo root: `rm -rf .venv` would delete the repo's mjlab environment. Before the `exec`, add
    `export PYTHONPATH="$(cd .. && pwd)${PYTHONPATH:+:$PYTHONPATH}"` so `duck_policy` imports from this
    checkout. Then run `exec .venv/bin/python -m duck_policy.main "$@"`. Add `duck_policy/.venv/` and
    `duck_policy/bin/` to `.gitignore`.
  - `duck_policy/requirements.txt`: `viam-sdk>=0.85`, `numpy`, `packaging`.

  Make `run.sh` executable. Don't add a module upload here: the preview machine runs it locally
  (Task 7). Uploading to the registry belongs with the real robot.

- [ ] **Step 5: Run the tests and the whole suite.** Run `uv run pytest tests/test_duck_*.py -q`, then
  `uv run pytest -q` (the existing tests must stay green).

- [ ] **Step 6: Commit** with message `feat: duck-policy generic service (start/stop/status/set_command) and module packaging`.

---

### Task 7: Wire it up on the preview machine

**Files:**
- Modify: `docs/superpowers/specs/duck-sim-machine.json`

- [ ] **Step 1: Upload the walk package** (private, DevRelDemos; covered by the standing authorization):

```bash
viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=microduck-walk --version=0.1.0 \
  --type=ml_model --path=dist/microduck-walk.tar.gz
```

If `--type=ml_model` needs extra flags (model framework or type), check
`viam packages upload --help`. Pass `--model-framework onnx` (or its equivalent) if asked.

- [ ] **Step 2: Update `duck-sim-machine.json`.** Make these changes:
  - `packages`:
    - duck-sim goes to `0.2.0`;
    - add `{"name": "microduck-walk", "package": "16518049-9dd3-479f-9644-c5d112aa42d8/microduck-walk",
      "type": "ml_model", "version": "0.1.0"}`.
  - `modules`: add the registry module `{"type": "registry", "name": "viam_onnx-cpu",
    "module_id": "viam:onnx-cpu", "version": "0.2.2"}` and a local module
    `{"type": "local", "name": "duck-policy", "executable_path": "/Users/nick.hehr/src/viam-rslrl-policy-training/duck_policy/run.sh"}`.
  - The robot `duck`: frame `"translation": {"x": 0, "y": 0, "z": 5}` (mm), so the trunk homes at
    0.125 like the reference.
  - `services`:
    - `walk-model`: api `rdk:service:mlmodel`, model `viam:mlmodel:onnx-cpu`, attributes
      `{"model_path": "${packages.ml_model.microduck-walk}/policy.onnx"}`;
    - `walker`: api `rdk:service:generic`, model `devrel:duck-policy:controller`, attributes
      `{"duck": "duck", "mlmodel": "walk-model", "package_dir": "${packages.ml_model.microduck-walk}"}`.

  Check the `${packages.ml_model.<name>}` placeholder against how the rl-insert cell referenced its
  ML model package (`docs/superpowers/plans/2026-11-17-policy-controller.md`, Task near line 375). It
  used `${packages.ml_model.rl-insert}`.

- [ ] **Step 3: Apply it to the preview machine.**
  1. Read the revision with `read_machine_config`.
  2. Write the JSON with `set_machine_config`.
  3. Wait for every resource to be ready:
     `uv run <mujoco-sim-cell skill>/scripts/cell.py wait --cell <cell.json>`. The skill directory is
     `/Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell`.
  4. Read the machine logs if anything isn't ready, then fix and re-apply.
  5. Confirm onnx-cpu 0.2.2 started on darwin/arm64. That is also the check that its release ships
     that platform.

- [ ] **Step 4: Smoke test.** Write `scripts/duck_smoke.py`, a short `cell.connect` script (PEP 723
  header, as `scratchpad/microduck-preview/actuate.py` does), not separate MCP calls. The policy must
  take over within a few ticks of `play`: the duck tips over by about 1.5 s without it, and a `start`
  while paused stops at once with "stale_state".
  1. `sim-world` pause, then reset.
  2. `walker` `set_command` with zeros.
  3. `sim-world` play, then immediately `walker` `{"command": "start"}`.
  4. After 10 s, `walker` status should be `running`, with about 450–500 ticks and
     `tick_ms_p99 < 20`.
  5. `duck` `{"io": {}}` should show gravity z < -0.95: it stands at a zero command.
  6. `walker` `{"command": "set_command", "vx": 0.4}` (lower forward speeds are in velstand's dead
     zone). After 5 s, the trunk x has increased by more than 0.3 m; about 0.85 m is expected.
  7. `walker` stop, `sim-world` pause.

  Commit `scripts/duck_smoke.py` with the config.

  Save a `GetImages` from `overview` mid-walk as evidence (note the saved path).

- [ ] **Step 5: Commit** with message `feat: duck-sim-machine.json runs duck-policy with the walk package on onnx-cpu`.

---

### Task 8: Parity harness

**Files:**
- Create: `scripts/duck_parity.py`

- [ ] **Step 1: Write the harness.** It reuses `duck_reference.run` for the reference side, so the
  command script and sampling are shared.

```python
# /// script
# dependencies = ["viam-sdk>=0.85", "numpy"]
# ///
"""Walk parity: the same command script through Pollen's reference (duck_reference.py, run in the
mujoco-sim worktree's environment) and through our stack on a Viam machine (duck-policy + onnx-cpu +
mujoco-sim io). Spec unit 5.

    uv run scripts/duck_parity.py ours <cell.json> <out.json> [--episodes 20]
    uv run scripts/duck_parity.py report <reference.json> <ours.json> <report.md>

Run the reference with duck_reference.py first. Metrics per side:
- falls (gravity z > -0.5 at any sample);
- per-segment velocity-tracking RMSE of body-frame vx and wz, from 0.1 s trunk pose samples;
- for ours, the walker's tick stats and the world's slips / real-time factor.

Pass (spec unit 5, with the milestone 2 decisions):
- at least 18 of 20 episodes slip-free, else the result is "machine too slow";
- falls(ours) <= falls(ref) + 1;
- RMSE(ours) <= 1.25 x RMSE(ref) for vx and for wz;
- mean tick rate >= 49.5 Hz, p99 tick time < 20 ms, under 1 % of ticks late.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from duck_reference import DURATION, SCRIPT, command_at  # noqa: E402

SKILL = Path("/Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell/scripts")
SETTLE_S = 0.5  # skip this much after each command change when scoring tracking


def yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def tracking(samples: list[dict]) -> dict:
    """Body-frame vx and wz from consecutive pose samples, scored against the command, skipping SETTLE_S
    after each segment boundary."""
    bounds, acc = [], 0.0
    for seconds, *_ in SCRIPT:
        acc += seconds
        bounds.append(acc)
    err_vx, err_wz = [], []
    for a, b in zip(samples, samples[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        t_mid = (a["t"] + b["t"]) / 2
        if any(0 <= t_mid - edge < SETTLE_S for edge in [0.0, *bounds]):
            continue
        ya, yb = yaw(a["trunk_quat_wxyz"]), yaw(b["trunk_quat_wxyz"])
        dyaw = math.atan2(math.sin(yb - ya), math.cos(yb - ya))
        ym = ya + dyaw / 2  # mid-interval heading: exact for a constant-curvature arc to first order
        dx, dy = b["trunk_pos"][0] - a["trunk_pos"][0], b["trunk_pos"][1] - a["trunk_pos"][1]
        vx = (math.cos(ym) * dx + math.sin(ym) * dy) / dt
        wz = dyaw / dt
        cvx, _, cwz = command_at(t_mid)
        err_vx.append(vx - cvx)
        err_wz.append(wz - cwz)
    rms = lambda e: float(np.sqrt(np.mean(np.square(e)))) if e else float("nan")  # noqa: E731
    return {"rmse_vx": rms(err_vx), "rmse_wz": rms(err_wz), "n": len(err_vx)}


def fell(samples: list[dict]) -> bool:
    return any(s["gravity"][2] > -0.5 for s in samples)


async def ours(cell_json: Path, out: Path, episodes: int) -> None:
    sys.path.insert(0, str(SKILL))
    import cell
    from viam.components.generic import Generic as GenericComponent
    from viam.services.generic import Generic as GenericService

    results = []
    async with cell.connect(cell_json) as robot:
        world = GenericComponent.from_robot(robot, "sim-world")
        duck = GenericComponent.from_robot(robot, "duck")
        walker = GenericService.from_robot(robot, "walker")
        for ep in range(episodes):
            await world.do_command({"command": "pause"})
            await world.do_command({"command": "reset"})
            await walker.do_command({"command": "set_command", "vx": 0, "vy": 0, "wz": 0})
            before = await world.do_command({"command": "status"})  # reset zeroed slips
            await world.do_command({"command": "play"})
            await walker.do_command({"command": "start"})  # right after play: no ramp at STAND, takes over in a tick
            first = await duck.do_command({"io": {}})
            t0, samples, last_cmd = float(first["t"]), [], None
            while True:
                s = await duck.do_command({"io": {}})
                t = float(s["t"]) - t0
                if t >= DURATION:
                    break
                cmd = command_at(t)
                if cmd != last_cmd:
                    await walker.do_command({"command": "set_command", "vx": cmd[0], "vy": cmd[1], "wz": cmd[2]})
                    last_cmd = cmd
                if not samples or t - samples[-1]["t"] >= 0.1 - 1e-6:
                    samples.append({"t": round(t, 4), "trunk_pos": s["trunk_pos"],
                                    "trunk_quat_wxyz": s["trunk_quat_wxyz"], "gravity": s["gravity"]})
                await asyncio.sleep(0.05)  # 20 Hz is enough for 0.1 s samples and spares the sim thread
            status = await walker.do_command({"command": "status"})
            await walker.do_command({"command": "stop"})
            after = await world.do_command({"command": "status"})
            await world.do_command({"command": "pause"})
            results.append({"episode": ep, "samples": samples, "walker": status,
                            "slips": after.get("slips", 0) - before.get("slips", 0),
                            "real_time_factor": after.get("real_time_factor")})
            print(f"episode {ep}: fell={fell(samples)} {tracking(samples)} walker={status.get('reason')} "
                  f"ticks={status.get('ticks')} p99={status.get('tick_ms_p99')} slips={results[-1]['slips']}")
    out.write_text(json.dumps(results))


def report(ref_path: Path, ours_path: Path, md: Path) -> None:
    ref = json.loads(ref_path.read_text())
    eps = json.loads(ours_path.read_text())
    ref_track, ref_fell = tracking(ref["samples"]), fell(ref["samples"])
    clean = [e for e in eps if e["slips"] == 0]
    tracks = [tracking(e["samples"]) for e in clean]
    ours_falls = sum(fell(e["samples"]) for e in clean)
    ref_falls = len(clean) * int(ref_fell)  # deterministic reference: every episode is the same
    rmse_vx = float(np.mean([t["rmse_vx"] for t in tracks])) if tracks else float("nan")
    rmse_wz = float(np.mean([t["rmse_wz"] for t in tracks])) if tracks else float("nan")
    ticks = sum(e["walker"].get("ticks", 0) for e in clean)
    late = sum(e["walker"].get("late", 0) for e in clean)
    p99 = max((e["walker"].get("tick_ms_p99") or 0) for e in clean) if clean else float("nan")
    run_s = sum(e["walker"].get("run_s") or 0 for e in clean)
    rate = ticks / run_s if run_s else 0.0  # the loop's own run time, not the episode's start/stop overhead
    checks = {
        "slip_free >= 18/20": len(clean) >= 18,
        "falls": ours_falls <= ref_falls + 1,
        "rmse_vx <= 1.25 x ref": rmse_vx <= 1.25 * ref_track["rmse_vx"],
        "rmse_wz <= 1.25 x ref": rmse_wz <= 1.25 * ref_track["rmse_wz"],
        "tick rate >= 49.5 Hz": rate >= 49.5,
        "p99 tick < 20 ms": p99 < 20,
        "late < 1 %": ticks > 0 and late / ticks < 0.01,
    }
    verdict = "machine too slow" if not checks["slip_free >= 18/20"] else ("PASS" if all(checks.values()) else "FAIL")
    lines = [
        f"# MicroDuck walk parity: {verdict}", "",
        f"- Reference: fell={ref_fell}, rmse_vx={ref_track['rmse_vx']:.4f}, rmse_wz={ref_track['rmse_wz']:.4f}",
        f"- Ours ({len(clean)}/{len(eps)} slip-free): falls={ours_falls}, rmse_vx={rmse_vx:.4f}, rmse_wz={rmse_wz:.4f}",
        f"- Timing: {rate:.1f} Hz, p99 tick {p99} ms, late {late}/{ticks}", "",
        "| Check | Result |", "|---|---|",
        *[f"| {k} | {'pass' if v else 'FAIL'} |" for k, v in checks.items()],
    ]
    md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "ours":
        episodes = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 20
        asyncio.run(ours(Path(sys.argv[2]), Path(sys.argv[3]), episodes))
    elif mode == "report":
        report(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
```

Notes for the implementer:
- **World status keys:** check what the world's `status` DoCommand returns (mujoco-sim README, world
  section) for `slips` and `real_time_factor`, and use the real key names.
- **io trunk keys:** `duck`'s io returns `trunk_pos` and `trunk_quat_wxyz` because the robot floats.
- **The `duck_reference` import:** this script imports `duck_reference` only for `SCRIPT`,
  `DURATION` and `command_at`. Keep those importable without mujoco: `duck_reference.py` must import
  mujoco and infer_policy only inside `run()`, which it already does.
- **Smoke test first:** run `ours` with `--episodes 1` before the full run.

- [ ] **Step 2: Add a unit test for `tracking`/`fell`** in `tests/test_duck_parity.py`.
  1. Synthesize samples every 0.1 s of a trunk following the commanded twist exactly. Integrate the
     unicycle in closed form per interval: for wz ≠ 0, the arc is radius vx/wz; for wz = 0, a straight
     line.
  2. Assert both RMSEs are below 1e-3. The mid-interval heading is a first-order approximation, so not
     1e-6.
  3. Assert `fell` is False.
  4. Tilt one sample's gravity to `[0.9, 0, -0.3]` and assert `fell` is True.
  5. Run it with `uv run pytest tests/test_duck_parity.py -q`. `duck_parity.py` imports
     `duck_reference` only for the script constants, so the test needs neither mujoco nor viam.
- [ ] **Step 3: Commit** with message `feat: walk parity harness (reference vs duck-policy on Viam; falls, tracking, timing)`.

---

### Task 9: Run parity and record results

**Files:**
- Create: `docs/superpowers/specs/2026-11-2X-microduck-parity-results.md` (use the date of the run)

- [ ] **Step 1: Reference** (already produced in Task 4 as `dist/duck-reference.json`). Re-run it if
  duck-sim or the policy changed since.
- [ ] **Step 2: Ours, then the report.**

```bash
uv run scripts/duck_parity.py ours <cell.json> dist/duck-ours.json --episodes 20
uv run scripts/duck_parity.py report dist/duck-reference.json dist/duck-ours.json dist/duck-parity.md
```

- [ ] **Step 3: Write the results doc** in the same style as `2026-11-18-sim-to-sim-results.md`. It
  covers the verdict table, the reference and our numbers, the tick timing, and an overview frame
  mid-walk.
  - **If it fails on tracking (not timing):** suspect first the one real difference left between the
    two stacks, which is real-time stepping against lockstep. Then MuJoCo version: it's the same build
    on both sides here, so that's unlikely.
  - **Record, don't fix, in this milestone:** if the gap needs the joint-velocity lag or the action
    filter Pollen's daemon uses (spec, milestone 2 decisions), note it for milestone 3/4.
- [ ] **Step 4: Commit** the results doc (not the dist JSONs) with message
  `docs: MicroDuck walk parity results (<verdict>)`.

---

## Done when

- The `duck_policy` tests (obs golden, loop stop conditions, service) and the existing suite pass.
- duck-sim 0.2.0 and microduck-walk 0.1.0 are in DevRelDemos.
- The preview machine runs `duck-sim-machine.json`: the duck stands at a zero command and walks at
  vx 0.4. Lower forward speeds are in velstand's dead zone.
- The parity report exists with a verdict. PASS is the goal; a FAIL with a diagnosed cause is still a
  result.
