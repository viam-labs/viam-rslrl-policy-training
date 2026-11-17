# Peg-insert training on Viam: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the PegInsert-xArm6 policy as a Viam custom training job (2× A100, Vertex AI). The job produces `policy.onnx`, `config.json`, `model_metrics.json` and a checkpoint, and it uploads sim videos recorded during training to Viam Data.

**Architecture:**
- **The task:** an mjlab manager-based env, `rl_insert/task/`, built on plan 1's MjSpec model. It has three entities:
  - robot: arm + G2 gripper;
  - peg: held by a compliant weld;
  - hole: mocap, randomized per env.
- **Action:** a 6-D end-effector delta solved with mjlab's differential IK.
- **Training:** RSL-RL PPO.
- **Entry point:** `model/training.py`, which Viam runs as `python3 -m model.training`. It picks a GL backend, trains, uploads videos in a background thread, and writes the package. A `--mode=probe` run checks new hardware cheaply.
- **Where things live:** dependencies in a Docker image; code and assets in a pip-installed script package (Viam's launcher installs it at job start).

**Tech Stack:**
- mjlab at commit `6ab9846` (MuJoCo 4.0.1, mujoco-warp 4.0.1, rsl-rl-lib 5.5.2), torch, warp-lang 1.19;
- viam-sdk 0.85, onnxruntime, tensorboard;
- Docker (linux/amd64), Viam CLI.

**Spec:** `docs/superpowers/specs/2026-11-16-peg-insert-rl-design.md` (sections Training environment, Task config and config.json, Frames, Infrastructure, Training videos to Viam Data). This is plan 2 of 4. Plan 1 (foundations) is done; its "After execution" section lists the items carried into this plan.

**Fixed IDs:**

| What | Value |
|---|---|
| Org | DevRelDemos, `16518049-9dd3-479f-9644-c5d112aa42d8` |
| Video upload part | `rl-insert` machine, part `cab989eb-90a0-4dce-8fc6-2c50d0ed1f10` |
| Placeholder dataset | `testghost`, `6581c38d39e1f5ebd5bc0e0f` (the job ignores it, but Viam requires one) |
| Docker Hub namespace | `hipsterbrown/` |

The user has authorized pushing the image, registering it, uploading the script and submitting jobs.

---

## What prototyping already established

Every code block below ran on macOS (CPU Warp) before this plan was written:
- the env and grasp tests;
- a 3-iteration PPO run;
- ONNX export checked against torch in onnxruntime (max difference 5e-8);
- a video;
- the pip-installed package run from an empty cwd;
- the linux/amd64 image running probe and train modes (emulated CPU).

Findings to keep:

1. **mjlab is pinned to a `main` commit, not the PyPI release.** mjlab 1.6.0 on PyPI pins MuJoCo 3.11; `main` at `6ab9846` uses MuJoCo 4.0.1, which plan 1 validated.
2. **The grasp is a compliant weld, not friction.**
   - On Warp the friction grasp knocked the peg up to 5.2 mm along its axis during reset, and then it crept about 0.4 mm/s. Neither pre-squeezing nor stiffer pads fixed it.
   - The weld is anchored at the TCP, with solref 0.002, and each env's grasp pose goes into `eq_data`. With it, settle is 0.14 mm and creep 0.07 mm, and 20 N moves the peg about 0.2–0.5 mm.
   - The fingers still close on the peg. Grip force is fixed at 30 N, since randomizing it no longer means anything.
   - The real G2 (10–50 N) won't slip a 20 g peg at the ≤ 40 N this task allows.
3. **Sim settings:**
   - Euler integrator at 1 ms, the setting plan 1 validated (mjlab defaults to implicitfast).
   - Decimation 50, so the policy runs at 20 Hz.
   - `nconmax=64`, `njmax=400`; mjlab's defaults overflowed.
   - Arm-link meshes don't collide.
4. **The action is world-frame.** `DifferentialIKActionCfg` runs in relative mode on site `tcp`, with 2 mm and 1° per unit.
   - Deltas are in world axes, applied to the *measured* TCP pose, and IK is re-solved every physics step.
   - The hole is upright with zero yaw, so the hole frame is the world frame translated to the mouth. The deploy side (plan 3) rotates the deltas into the base frame.
5. **Clearance is fixed per run** (`--clearance_mm`), because it's compiled mesh geometry. The curriculum scales start offsets and hole-pose bias from 0.3 to 1.0 as success rises.
6. **ONNX metadata:** mjlab's exporter asserts a `joint_pos` action. `PegInsertRunner.get_export_metadata` replaces it so the IK action exports.
7. **Logging:** rsl_rl never flushes its tensorboard writer, so the runner flushes it after `learn`. `model_metrics.json` is built from those scalars.
8. **The image needs git:** rsl_rl imports GitPython, which needs the binary.
9. **EGL can be software:** EGL can "work" through Mesa llvmpipe. The probe records the GL renderer string so a CPU renderer is visible.
10. **Driver version:** warp 1.18+ wheels are CUDA 13 builds and need driver ≥ 580. The documented fallback (torch cu126 with warp 1.17.0) is untested; the probe job decides.
11. **Reward scale:** mjlab scales rewards by dt. The success bonus (800, so 40 after dt) is set to twice the most that hovering below success depth can earn (0.2 per step / (1 − γ) = 20), or PPO would stall just short of success.
12. **Sim-to-real terms:**
    - `joint_vel` is a finite difference of measured joints at the policy rate (0 on the first step), as the deploy side computes it.
    - The start pose is randomized (±0.02 rad per joint).
    - Each episode samples an action latency of 0–2 policy steps, and the arm's PD gains vary ±20%.
    - config.json lists what isn't modelled.
13. **Termination metrics are counts:** `Episode_Termination/*` holds counts per reset batch, so `model_metrics.json` turns them into rates.
14. **The weld makes insertion harder.** With the friction grasp, a sloppy peg slid into place. A scripted descent now gets only 2–3 of 8 envs in at 0.5 mm clearance, which is the realistic case for a stiff G2 and xArm. The first job therefore uses 1.0 mm.

## File map

| File | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock` | Dev env: mjlab (git pin), viam-sdk, onnxruntime; packages `rl_insert` and `model` |
| `rl_insert/assets/` (moved from `assets/`) | xArm6 URDF and meshes, shipped inside the package |
| `rl_insert/task/__init__.py` | Registers `PegInsert-xArm6` |
| `rl_insert/task/env_cfg.py` | `TaskParams`, nominal IK pose, entity specs, grasp weld, `peg_insert_env_cfg`, `OBS_DOC` |
| `rl_insert/task/mdp.py` | Reset event (peg in hand, hole placement, bias), observations, rewards, terminations, curriculum |
| `rl_insert/task/runner.py` | PPO config; runner with ONNX metadata and tensorboard flush |
| `rl_insert/task/checks.py` | `make_env`, grasp-hold slip, throughput (used by the tests and the probe) |
| `rl_insert/task/viam_upload.py` | Upload one file to Viam Data with the job's API key |
| `model/training.py` | Viam entry point: args, GL selection, train and probe modes, video uploader |
| `model/package.py` | Writes `config.json` and `model_metrics.json` |
| `scripts/smoke.py`, `scripts/onnx_check.py` | Dev scripts: env smoke; ONNX vs torch on a training run |
| `tests/test_grasp_hold.py`, `tests/test_training_cli.py` | Grasp gate (CPU Warp); launcher argument contract |
| `setup.py`, `MANIFEST.in`, `build.sh` | Script package → `dist/rl-insert-train.tar.gz` |
| `container/Dockerfile.train`, `container/Dockerfile.train.dockerignore` | Training image (dependencies only) |
| `docs/superpowers/specs/2026-11-17-vertex-probe-results.md`, `...-first-training-run.md` | Results of Tasks 7 and 8 |

Code style: this repo uses 4-space indent; mjlab uses 2. The files below are already 4-space.

---

### Task 1: Dependencies, and assets inside the package

**Files:**
- Modify: `pyproject.toml`, `rl_insert/robot.py:11`, `container/Dockerfile:12`
- Move: `assets/` → `rl_insert/assets/`

- [ ] **Step 1: Move the assets with git, so history follows**

```bash
git mv assets rl_insert/assets
```

- [ ] **Step 2: Point `ASSETS` at the new location.** In `rl_insert/robot.py`, replace line 11 with:

```python
ASSETS = Path(__file__).resolve().parent / "assets"  # inside the package so pip installs it
```

- [ ] **Step 3: Drop the now-wrong COPY from plan 1's probe image.** In `container/Dockerfile`, delete the line `COPY assets assets`; `COPY rl_insert rl_insert` now carries the assets.

- [ ] **Step 4: Replace `pyproject.toml`**

```toml
[project]
name = "viam-rl-insert"
version = "0.2.0"
requires-python = ">=3.12,<3.14"
dependencies = [
    "mjlab[cpu] @ git+https://github.com/mujocolab/mjlab@6ab9846ea068714678fd0e9f791f898c6b697b20",
    "viam-sdk>=0.85",
    "onnxruntime>=1.20",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.build.targets.wheel]
packages = ["rl_insert", "model"]

[dependency-groups]
dev = ["pytest>=8"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 5: Lock, sync, and run plan 1's suite**

Run: `uv lock && uv sync && uv run pytest -q`
Expected: 27 passed. `uv run python -c "import mjlab, mujoco; print(mujoco.__version__)"` prints `4.0.1`. If uv resolves mujoco 3.11, the git pin was lost.

- [ ] **Step 6: Commit**

```bash
git add -A pyproject.toml uv.lock rl_insert container/Dockerfile assets
git commit -m "chore: mjlab (MuJoCo 4 pin), viam-sdk; assets inside the package"
```

---

### Task 2: The mjlab task

**Files:**
- Create: `rl_insert/task/__init__.py`, `rl_insert/task/env_cfg.py`, `rl_insert/task/mdp.py`, `rl_insert/task/runner.py`, `scripts/__init__.py` (empty), `scripts/smoke.py`

The task is prototyped code with no unit test of its own. Its gates are the smoke run (Step 6 here) and the grasp test (Task 3).

- [ ] **Step 1: Write `rl_insert/task/env_cfg.py`**

```python
"""PegInsert-xArm6: the mjlab env config.

Entities (mjlab allows one freejoint per entity, so the peg is its own entity):
- robot: xArm6 + G2 gripper from rl_insert.robot, keeping its MjSpec actuators (XmlActuatorCfg).
  Fixed base; mjlab wraps it in a mocap body at the origin. The gripper is commanded closed
  (ctrl 0, squeezing at grip_force) and is not part of the action.
- peg: free body from rl_insert.peg_hole.add_peg (frame = tip), held by a compliant weld to the
  gripper body (_add_grasp_weld) whose pose is the per-env sampled grasp.
- hole: from add_hole; fixed, so mjlab wraps it in a mocap body and the reset moves it per env.

Physics keeps what the foundations plan validated: mesh peg/hole, solref 0.002 on peg/hole/pads
and the finger equality, 1 ms timestep, elliptic cone, impratio 10, gravcomp, and the Euler
integrator the contact tests ran with (mjlab defaults to implicitfast).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import events as mj_events
from mjlab.envs.mdp import observations as mj_obs
from mjlab.envs.mdp import rewards as mj_rew
from mjlab.envs.mdp import terminations as mj_term
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from rl_insert.peg_hole import TIMESTEP, PegHoleParams, add_hole, add_peg
from rl_insert.robot import ARM_JOINTS, GripperParams, add_gripper, load_xarm6

from . import mdp


@dataclass(frozen=True)
class TaskParams:
    """Everything a run varies. Becomes the training task config / config.json `task` section."""

    clearance: float = 0.0005  # fixed per run: compiled mesh geometry, see mdp.widen_on_success
    hole_mouth: tuple[float, float, float] = (0.4, 0.0, 0.05)  # nominal, robot base frame
    protrusion: float = 0.02  # peg tip past the TCP (pad centre)
    start_height: tuple[float, float] = (0.010, 0.020)  # tip above the mouth at reset
    start_lateral: float = 0.003  # +/- m at full difficulty
    hole_bias: tuple[float, float, float] = (0.002, 0.002, 0.001)  # +/- m at full difficulty
    difficulty_init: float = 0.3
    grasp_offset: float = 0.003  # +/- m along TCP x and z
    grasp_tilt: float = float(np.deg2rad(3.0))  # +/- rad about TCP y (the pad normal)
    success_depth: float = 0.015  # of the 20 mm bore
    success_centred: float = 0.001
    force_limit: float = 40.0  # N, terminate; keep below the xArm's collision trip
    force_penalty_from: float = 15.0  # N
    max_slip: float = 0.01  # m
    action_pos_scale: float = 0.002  # m per unit action per policy step
    action_rot_scale: float = float(np.deg2rad(1.0))  # rad per unit action per policy step
    # The peg is held by a compliant weld (see _add_grasp_weld), so grip force is fixed at
    # GripperParams.grip_force and no longer randomized: it only presses the pads on a welded peg.
    grasp_solref: float = 0.002  # s, weld time constant; see _add_grasp_weld for the measured compliance
    start_joint_offset: float = 0.02  # +/- rad per arm joint at reset (TCP tilt up to a few degrees)
    pd_gain_scale: tuple[float, float] = (0.8, 1.2)  # arm servo kp and kd, scaled per episode
    action_delay_steps: tuple[int, int] = (0, 2)  # policy steps; constant within an episode
    episode_s: float = 10.0
    decimation: int = 50  # 1 ms physics, 20 Hz policy


PPO_GAMMA = 0.99  # discount, used here to size the success bonus (runner.py uses it for PPO)

# Units and frame of each actor observation term, for config.json. Keep in step with `actor` below.
OBS_DOC: dict[str, dict[str, str]] = {
    "tip_rel_est_hole": {
        "units": "m",
        "frame": "hole",
        "doc": "nominal peg tip (TCP + protrusion along TCP z) minus estimated hole mouth",
    },
    "ee_rot6d": {
        "units": "rot6d",
        "frame": "hole",
        "doc": "TCP rotation in the hole frame, columns x then y, 3 each",
    },
    "joint_pos": {"units": "rad", "frame": "joint", "doc": "arm joints minus control.default_joint_pos"},
    "joint_vel": {
        "units": "rad/s",
        "frame": "joint",
        "doc": "(q_t - q_{t-1}) / policy_dt from measured arm joint positions at consecutive policy "
        "steps; 0 on the first step after reset",
    },
    "last_action": {"units": "action", "frame": "hole", "doc": "previous raw policy output, unscaled"},
    "ik_error": {
        "units": "m (0:3), rad axis-angle (3:6)",
        "frame": "hole",
        "doc": "the IK action's desired TCP pose minus the measured TCP pose, world (= hole) axes. "
        "At the previous policy step the action a (6, unscaled) set, from the TCP pose (p, R) "
        "measured then: p_des = p + pos_scale * a[0:3], R_des = R_delta R with R_delta the "
        "rotation of axis-angle rot_scale * a[3:6]. Now, with the measured TCP pose (p_m, R_m): "
        "[0:3] = p_des - p_m, [3:6] = axis-angle (rotation vector, |.| <= pi) of q_des * q_m^-1. "
        "0 on the first step after reset",
    },
}


def nominal_joint_pos(p: TaskParams, standoff: float | None = None) -> np.ndarray:
    """Arm joints putting the nominal peg tip `standoff` above the nominal hole mouth, TCP pointing
    straight down with the same yaw as the xArm's zero pose (TCP x = -base x).

    Damped least squares on CPU MuJoCo; deterministic, ~ms, so it runs at config time instead of
    being a stale constant.
    """
    standoff = sum(p.start_height) / 2 if standoff is None else standoff
    spec = load_xarm6()
    add_gripper(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    site = m.site("tcp").id
    target_pos = np.array(p.hole_mouth) + [0.0, 0.0, p.protrusion + standoff]
    target_rot = np.diag([-1.0, 1.0, -1.0])
    q = np.array([0.0, 0.3, -1.0, 0.0, 0.7, 0.0])  # elbow up, wrist down
    arm = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(200):
        d.qpos[arm] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rot = d.site_xmat[site].reshape(3, 3)
        err_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
        err = np.concatenate([target_pos - d.site_xpos[site], err_rot])
        if np.linalg.norm(err) < 1e-9:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        jac = np.vstack([jacp, jacr])[:, arm]
        q = q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
    assert np.linalg.norm(err) < 1e-6, f"nominal IK did not converge: {err}"
    return q


def _robot_spec(g: GripperParams) -> mujoco.MjSpec:
    spec = load_xarm6()
    add_gripper(spec, g)
    # The arm-link meshes only matter for collisions the task never has; skip their narrowphase.
    for body in spec.bodies[1:]:
        if body.name.startswith("link"):
            for geom in body.geoms:
                geom.contype = geom.conaffinity = 0
    # Pads win friction mixing against the peg (MuJoCo takes the max otherwise), so finger-peg
    # friction is exactly the pads' value and can be randomized on its own.
    for name in ("pad_left", "pad_right"):
        spec.geom(name).priority = 1
    return spec


def _hole_spec(ph: PegHoleParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = add_hole(spec, spec.worldbody, ph)
    for i, geom in enumerate(body.geoms):
        geom.name = geom.name or f"hole_piece_{i}"  # add_hole leaves the ring pieces unnamed
        geom.priority = 1  # hole friction wins against the peg, as for the pads
    return spec


def _peg_spec(ph: PegHoleParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    add_peg(spec, spec.worldbody, ph)
    return spec


def _add_grasp_weld(spec: mujoco.MjSpec, timeconst: float) -> None:
    """Weld the peg to the gripper body: the grasp model.

    On MuJoCo Warp the pads alone let the peg creep (up to ~0.4 mm/s) and get knocked up to 5 mm
    along its axis in the first 0.4 s after reset, whatever the grip force or finger preload.
    The real G2 holds a 20 g peg rigidly at these loads, so the weld carries it; the fingers still
    close on it for contact and looks. The pose here is a placeholder: ResetPegInHand writes each
    world's grasp pose into eq_data. An explicit identity quat leaves eq_data[11] (the "compute
    from qpos0" mark) at 0, so set_const never overwrites the per-world pose.

    Measured on CPU Warp (64 envs): timeconst 0.002 s gives settle 0.14 mm / creep 0.07 mm over
    2 s, and 20 N moves the peg in the hand ~0.2 mm axially / ~0.5 mm laterally (max 0.7 / 1.2).
    0.005 s: settle 0.4, creep 0.28, lateral 20 N up to 2.5 mm; 0.01 s fails the 0.5 mm gates.
    """
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name=mdp.GRASP_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="robot/gripper",
        name2="peg/peg",
        solref=[timeconst, 1.0],
        data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def peg_insert_env_cfg(p: TaskParams = TaskParams(), play: bool = False) -> ManagerBasedRlEnvCfg:
    ph = PegHoleParams(clearance=p.clearance)
    g = GripperParams()
    q0 = nominal_joint_pos(p)
    pd = asdict(p) | {"peg_radius": ph.peg_radius}  # what the terms read via env.insert_state

    robot = EntityCfg(
        spec_fn=lambda: _robot_spec(g),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={**dict(zip(ARM_JOINTS, q0.tolist())), "finger_.*": ph.peg_radius},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # Action latency: the arm's joint targets reach the servos 0-2 policy steps late,
                # one lag per episode (hold_prob 1: sampled at reset, then held).
                XmlActuatorCfg(
                    target_names_expr=ARM_JOINTS,
                    delay_min_lag=p.action_delay_steps[0] * p.decimation,
                    delay_max_lag=p.action_delay_steps[1] * p.decimation,
                    delay_hold_prob=1.0,
                ),
                XmlActuatorCfg(target_names_expr=("finger_left",)),  # the gripper; target stays 0 = closed
            ),
        ),
    )
    scene = SceneCfg(
        num_envs=4,
        env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": robot,
            "peg": EntityCfg(spec_fn=lambda: _peg_spec(ph)),
            "hole": EntityCfg(
                spec_fn=lambda: _hole_spec(ph),
                init_state=EntityCfg.InitialStateCfg(pos=p.hole_mouth),
            ),
        },
        spec_fn=lambda spec: _add_grasp_weld(spec, p.grasp_solref),
        sensors=(
            ContactSensorCfg(
                name="peg_hole_contact",
                primary=ContactMatch(mode="body", pattern="peg", entity="peg"),
                secondary=ContactMatch(mode="body", pattern="hole", entity="hole"),
                fields=("found", "force"),
                reduce="netforce",  # one net wrench, world frame
            ),
        ),
    )

    arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
    actor = {
        "tip_rel_est_hole": ObservationTermCfg(
            func=mdp.tip_rel_est_hole, noise=Unoise(n_min=-5e-4, n_max=5e-4)
        ),
        "ee_rot6d": ObservationTermCfg(func=mdp.ee_rot6d, noise=Unoise(n_min=-5e-3, n_max=5e-3)),
        "joint_pos": ObservationTermCfg(
            func=mj_obs.joint_pos_rel, params={"asset_cfg": arm}, noise=Unoise(n_min=-1e-3, n_max=1e-3)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.JointVelFiniteDiff, params={"asset_cfg": arm}, noise=Unoise(n_min=-0.02, n_max=0.02)
        ),
        "last_action": ObservationTermCfg(func=mj_obs.last_action),
        "ik_error": ObservationTermCfg(func=mdp.ik_tracking_error, noise=Unoise(n_min=-2e-4, n_max=2e-4)),
    }
    critic = {
        **actor,
        "hole_pos": ObservationTermCfg(func=mdp.hole_pos),
        "tip_rel_hole": ObservationTermCfg(func=mdp.tip_rel_hole),
        "contact_force": ObservationTermCfg(func=mdp.contact_force, scale=0.05),
        "peg_in_hand": ObservationTermCfg(func=mdp.peg_in_hand),
    }

    depth = p.success_depth
    dense = {"keypoint_coarse": 1.0, "keypoint_fine": 1.0, "depth": 2.0}  # each term is in [0, 1]
    # Rewards are scaled by step_dt, so hovering just short of success can earn at most
    # sum(dense) * dt per step, sum(dense) * dt / (1 - gamma) = 20 discounted. Success ends the
    # episode, so its one-off bonus (weight * dt) must beat that: twice the bound, weight 800 (40).
    success_weight = 2 * sum(dense.values()) / (1 - PPO_GAMMA)
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations={
            "actor": ObservationGroupCfg(actor, enable_corruption=not play),
            "critic": ObservationGroupCfg(critic, enable_corruption=False),
        },
        actions={
            "ee_delta": DifferentialIKActionCfg(
                entity_name="robot",
                actuator_names=ARM_JOINTS,
                frame_type="site",
                frame_name="tcp",
                use_relative_mode=True,  # world-frame delta from the *measured* TCP pose each step
                delta_pos_scale=p.action_pos_scale,
                delta_ori_scale=p.action_rot_scale,
            ),
        },
        events={
            "reset_scene": EventTermCfg(func=mj_events.reset_scene_to_default, mode="reset"),
            # Before reset_peg_hole: it places the peg and hole from the TCP pose these joints give.
            "start_pose": EventTermCfg(
                func=mj_events.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-p.start_joint_offset, p.start_joint_offset),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": arm,
                },
            ),
            "reset_peg_hole": EventTermCfg(func=mdp.ResetPegInHand, mode="reset", params={"p": pd}),
            "arm_pd_gains": EventTermCfg(
                func=dr.pd_gains,
                mode="reset",
                params={
                    # actuator_ids index entity.actuators: [0] is the arm XmlActuatorCfg group
                    "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0]),
                    "kp_range": p.pd_gain_scale,
                    "kd_range": p.pd_gain_scale,
                },
            ),
            "friction_peg_hole": EventTermCfg(
                func=dr.geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("hole", geom_names=(".*",)),
                    "ranges": (0.3, 0.8),
                    "shared_random": True,
                },
            ),
            "friction_finger_peg": EventTermCfg(
                func=dr.geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=("pad_.*",)),
                    "ranges": (0.7, 1.3),
                    "shared_random": True,
                },
            ),
        },
        rewards={
            "keypoint_coarse": RewardTermCfg(
                func=mdp.keypoint_tanh, weight=dense["keypoint_coarse"], params={"std": 0.02, "depth": depth}
            ),
            "keypoint_fine": RewardTermCfg(
                func=mdp.keypoint_tanh, weight=dense["keypoint_fine"], params={"std": 0.003, "depth": depth}
            ),
            "depth": RewardTermCfg(func=mdp.insertion_depth, weight=dense["depth"], params={"depth": depth}),
            "success": RewardTermCfg(
                func=mdp.success, weight=success_weight, params={"depth": depth, "centred": p.success_centred}
            ),
            "action_rate": RewardTermCfg(func=mj_rew.action_rate_l2, weight=-0.01),
            "joint_vel": RewardTermCfg(func=mj_rew.joint_vel_l2, weight=-0.01, params={"asset_cfg": arm}),
            "force": RewardTermCfg(
                func=mdp.force_over, weight=-0.05, params={"threshold": p.force_penalty_from}
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mj_term.time_out, time_out=True),
            "success": TerminationTermCfg(
                func=mdp.success, params={"depth": depth, "centred": p.success_centred}
            ),
            "peg_slipped": TerminationTermCfg(func=mdp.peg_slipped, params={"max_slip": p.max_slip}),
            "excess_force": TerminationTermCfg(func=mdp.excess_force, params={"threshold": p.force_limit}),
        },
        curriculum={
            "difficulty": CurriculumTermCfg(
                func=mdp.widen_on_success, params={"up": 0.8, "down": 0.3, "step": 0.05}
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="hole",
            body_name="hole",
            distance=0.35,
            elevation=-20.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=64,  # per world; see scripts/smoke.py for the peak seen
            njmax=400,
            mujoco=MujocoCfg(timestep=TIMESTEP, integrator="euler", cone="elliptic", impratio=10),
        ),
        decimation=p.decimation,
        episode_length_s=p.episode_s,
    )
    if play:
        cfg.curriculum = {}
    return cfg
```

- [ ] **Step 2: Write `rl_insert/task/mdp.py`**

```python
"""MDP terms for PegInsert-xArm6: the batched reset, observations, rewards, terminations, curriculum.

Frames: the robot base is the world frame (one robot per Warp world, all at the origin). The hole
is upright with zero yaw, so the hole frame is the world frame translated to the mouth: vectors
"in the hole frame" here are world vectors. The deploy side must rotate actions and observations
between its measured hole frame and the arm base frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

GRASP_WELD = "grasp"  # the peg-to-gripper weld equality added by env_cfg._add_grasp_weld
FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # peg +z points back toward the gripper


def _rot_y(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).view(-1, 3, 3)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def rot6d(mat: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, flattened: continuous, unlike quaternions."""
    return mat[..., :, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)


class InsertState:
    """Per-env episode state the terms share: the reset event owns it, the rest read it.

    Lives at env.insert_state (set by ResetPegInHand.__init__). Holds the per-episode hole-pose
    bias (the actor's *estimated* hole = true + bias) and the curriculum difficulty.
    """

    def __init__(self, env: ManagerBasedRlEnv, p: dict):
        self.p = p
        n, dev = env.num_envs, env.device
        self.hole_bias = torch.zeros(n, 3, device=dev)
        self.difficulty = p["difficulty_init"]  # scales lateral start offset and hole bias, 0..1
        robot = env.scene["robot"]
        self.tcp_gid = int(robot.indexing.site_ids[robot.find_sites("tcp")[0][0]])
        self.flip = FLIP.to(dev)
        mj = env.sim.mj_model
        self.weld_id = mj.equality(GRASP_WELD).id
        self.gripper_bid = mj.body("robot/gripper").id


def state(env: ManagerBasedRlEnv) -> InsertState:
    return env.insert_state  # ty: ignore[unresolved-attribute]


# Geometry helpers (all world frame).


def tcp_pose(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    s = state(env)
    return env.sim.data.site_xpos[:, s.tcp_gid], env.sim.data.site_xmat[:, s.tcp_gid].view(-1, 3, 3)


def nominal_tip(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Peg tip assuming the nominal grasp: what the real arm can compute from FK."""
    pos, rot = tcp_pose(env)
    return pos + rot[:, :, 2] * state(env).p["protrusion"]


def true_tip(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["peg"].data.root_link_pose_w[:, :3]  # peg frame origin is its tip


def hole_mouth(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["hole"].data.root_link_pose_w[:, :3]  # mocap root = hole mouth


def tip_in_hole(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(lateral distance from the hole axis, depth below the mouth), both m, true tip."""
    rel = true_tip(env) - hole_mouth(env)
    return rel[:, :2].norm(dim=-1), -rel[:, 2]


def peg_hole_force(env: ManagerBasedRlEnv, sensor_name: str = "peg_hole_contact") -> torch.Tensor:
    """Net peg-hole contact force, world frame (N, 3)."""
    return env.scene[sensor_name].data.force[:, 0, :]


# Reset.


class ResetPegInHand:
    """Batched equivalent of rl_insert.scene.place_peg_in_hand, plus hole placement.

    Runs after reset_scene_to_default has put the arm at its nominal joints with the fingers at
    the peg radius. Then:
    1. forward kinematics, so the TCP pose is current (the arm was just teleported);
    2. the peg goes in hand: tip `protrusion` past the TCP along TCP z, grasp offset along TCP x and
       z only (a y offset is re-centred by the coupled fingers), tilt about TCP y (the pad normal);
    3. the hole goes under the true tip: mouth = tip - (dx, dy, height), so the tip starts
       height (10-20 mm) above the mouth with a lateral offset;
    4. the per-episode hole-pose bias for the actor's estimated hole is sampled.

    The grasp is a weld (env_cfg._add_grasp_weld), so step 2 also writes that world's weld pose:
    anchored at the TCP (the grasp point), relative pose = the sampled grasp. Since the peg starts
    exactly at the weld pose, nothing settles after reset.
    """

    # eq_data gets per-world memory; after the event mjlab reruns set_const KIN0, which leaves
    # welds whose eq_data[11] mark is 0 (ours) untouched.
    model_fields = ("eq_data",)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        env.insert_state = InsertState(env, cfg.params["p"])  # ty: ignore[unresolved-attribute]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, p: dict) -> None:
        s = state(env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n, dev = len(env_ids), env.device
        env.sim.forward()  # the arm was just written; site_xpos is stale until FK runs
        pos, rot = tcp_pose(env)
        pos, rot = pos[env_ids], rot[env_ids]

        g = p["grasp_offset"]
        offset = torch.stack(
            [_uniform(n, -g, g, dev), torch.zeros(n, device=dev), _uniform(n, -g, g, dev)], dim=-1
        )
        tilt = _rot_y(_uniform(n, -p["grasp_tilt"], p["grasp_tilt"], dev))
        along = tilt @ torch.tensor([0.0, 0.0, p["protrusion"]], device=dev)
        tip = pos + (rot @ (offset + along).unsqueeze(-1)).squeeze(-1)
        peg_quat = quat_from_matrix(rot @ tilt @ s.flip)
        peg_state = torch.cat([tip, peg_quat, torch.zeros(n, 6, device=dev)], dim=-1)
        env.scene["peg"].write_root_state_to_sim(peg_state, env_ids=env_ids)

        # Weld data: [anchor in peg frame (3), anchor in gripper frame (3), quat gripper->peg (4),
        # torquescale, compute-from-qpos0 mark]. Anchor = the TCP.
        peg_rot = rot @ tilt @ s.flip
        g_pos = env.sim.data.xpos[env_ids, s.gripper_bid]
        g_quat = env.sim.data.xquat[env_ids, s.gripper_bid]
        g_rot = matrix_from_quat(g_quat)
        in_peg = (peg_rot.transpose(-1, -2) @ (pos - tip).unsqueeze(-1)).squeeze(-1)
        in_grip = (g_rot.transpose(-1, -2) @ (pos - g_pos).unsqueeze(-1)).squeeze(-1)
        rel = quat_mul(quat_conjugate(g_quat), peg_quat)
        tail = torch.tensor([1.0, 0.0], device=dev).expand(n, 2)
        env.sim.model.eq_data[env_ids, s.weld_id] = torch.cat([in_peg, in_grip, rel, tail], dim=-1)

        lat = p["start_lateral"] * s.difficulty
        h = _uniform(n, *p["start_height"], dev)
        mouth = tip - torch.stack([_uniform(n, -lat, lat, dev), _uniform(n, -lat, lat, dev), h], -1)
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
        env.scene["hole"].write_mocap_pose_to_sim(torch.cat([mouth, ident], -1), env_ids=env_ids)

        b = torch.tensor(p["hole_bias"], device=dev) * s.difficulty
        s.hole_bias[env_ids] = (torch.rand(n, 3, device=dev) * 2 - 1) * b


# Observations.


def tip_rel_est_hole(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Actor: nominal peg tip relative to the estimated hole (true + episode bias), hole frame."""
    return nominal_tip(env) - (hole_mouth(env) + state(env).hole_bias)


def ee_rot6d(env: ManagerBasedRlEnv) -> torch.Tensor:
    """TCP orientation in the hole frame (= world, see module docstring), rot6d."""
    return rot6d(tcp_pose(env)[1])


def ik_tracking_error(env: ManagerBasedRlEnv, action_name: str = "ee_delta") -> torch.Tensor:
    """Commanded minus measured TCP pose (pos m, axis-angle rad), world axes: the contact proxy.

    Commanded = the IK action's desired pose (private _desired_pos/_desired_quat: mjlab has no
    public accessor), set when the last action was processed: p_des = p + pos_scale * a[:3],
    R_des = R_delta(rot_scale * a[3:]) R, from the TCP pose measured at that policy step. The
    rotation part is the axis-angle of q_des * q_meas^-1. The action's reset() zeroes that buffer,
    so the first step after reset reports zero instead of a bogus error. env_cfg.OBS_DOC holds
    the deploy-side recipe.
    """
    term = env.action_manager.get_term(action_name)
    pos, rot = tcp_pose(env)
    quat = quat_from_matrix(rot)
    dq = quat_mul(term._desired_quat, quat_conjugate(quat))
    err = torch.cat([term._desired_pos - pos, axis_angle_from_quat(dq)], dim=-1)
    return err * (env.episode_length_buf > 0).unsqueeze(-1)


class JointVelFiniteDiff:
    """(q_t - q_{t-1}) / policy_dt from joint positions at consecutive policy steps; 0 on the first
    step after reset. What the deploy side computes from measured joints (the arm API has no
    velocity), unlike qvel.

    Keyed on env.common_step_counter, so extra compute() calls within a step change nothing.
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.prev = self.cur = None
        self.step = -1

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
        if env.common_step_counter != self.step:
            self.step = env.common_step_counter
            q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids].clone()
            self.prev, self.cur = (q if self.cur is None else self.cur), q
        vel = (self.cur - self.prev) / env.step_dt
        return vel * (env.episode_length_buf > 0).unsqueeze(-1)


def hole_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true hole mouth in the robot base frame."""
    return hole_mouth(env)


def tip_rel_hole(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true peg tip relative to the true hole."""
    return true_tip(env) - hole_mouth(env)


def contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return peg_hole_force(env)


def peg_in_hand(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true peg pose in the TCP frame (tip position 3 + rot6d 6)."""
    pos, rot = tcp_pose(env)
    peg = env.scene["peg"].data.root_link_pose_w
    rt = rot.transpose(-1, -2)
    rel_pos = (rt @ (peg[:, :3] - pos).unsqueeze(-1)).squeeze(-1)
    return torch.cat([rel_pos, rot6d(rt @ matrix_from_quat(peg[:, 3:7]))], dim=-1)


# Rewards.


def keypoint_tanh(env: ManagerBasedRlEnv, std: float, depth: float) -> torch.Tensor:
    """1 - tanh(d / std), d = true tip to the point `depth` down the hole axis."""
    rel = true_tip(env) - hole_mouth(env)
    rel[:, 2] += depth
    return 1.0 - torch.tanh(rel.norm(dim=-1) / std)


def insertion_depth(env: ManagerBasedRlEnv, depth: float) -> torch.Tensor:
    """Depth below the mouth as a fraction of the success depth, counted only near the axis."""
    lateral, d = tip_in_hole(env)
    inside = lateral < state(env).p["peg_radius"]  # the tip can only be below the mouth in the bore
    return (d.clamp(0.0, depth) / depth) * inside


def success(env: ManagerBasedRlEnv, depth: float, centred: float) -> torch.Tensor:
    lateral, d = tip_in_hole(env)
    return (d >= depth) & (lateral <= centred)


def force_over(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    return (peg_hole_force(env).norm(dim=-1) - threshold).clamp(min=0.0)


# Terminations.


def peg_slipped(env: ManagerBasedRlEnv, max_slip: float) -> torch.Tensor:
    """Peg tip moved more than max_slip from where the nominal grasp says it is (or fell)."""
    return (true_tip(env) - nominal_tip(env)).norm(dim=-1) > max_slip


def excess_force(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    return peg_hole_force(env).norm(dim=-1) > threshold


# Curriculum.


def widen_on_success(env: ManagerBasedRlEnv, env_ids, up: float, down: float, step: float) -> torch.Tensor:
    """Widen the start offset and hole bias (difficulty 0..1) when resetting envs mostly succeed.

    Runs at the start of every reset on the envs about to reset, while the termination manager
    still holds why they ended. Clearance can't be in this loop: it is compiled mesh geometry, so
    it is fixed per run (TaskParams.clearance).
    """
    s = state(env)
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0 and env.common_step_counter > 0:
        rate = env.termination_manager.get_term("success")[env_ids].float().mean().item()
        if rate > up:
            s.difficulty = min(1.0, s.difficulty + step)
        elif rate < down:
            s.difficulty = max(s.p["difficulty_init"], s.difficulty - step)
    return torch.tensor(s.difficulty)
```

- [ ] **Step 3: Write `rl_insert/task/runner.py`**

```python
"""PPO config and runner for PegInsert-xArm6."""

from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfg import PPO_GAMMA


def peg_insert_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            obs_normalization=True,  # exported into the ONNX graph
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.005, learning_rate=3e-4, gamma=PPO_GAMMA),
        experiment_name="peg_insert_xarm6",
        logger="tensorboard",
        num_steps_per_env=24,
        save_interval=50,
        max_iterations=1500,
    )


class PegInsertRunner(MjlabOnPolicyRunner):
    """Exports ONNX on every save.

    mjlab's get_base_metadata asserts a "joint_pos" JointPositionAction (joint stiffness, action
    scale...), which this task doesn't have, so the metadata is replaced with the IK action's.
    """

    export_onnx_on_save = True

    def learn(self, *args, **kwargs) -> None:
        super().learn(*args, **kwargs)
        # rsl_rl never closes a plain tensorboard SummaryWriter (flush_secs=10), so the last
        # iterations' scalars may still be buffered when model_metrics.json reads the event file.
        writer = getattr(self.logger, "writer", None)
        if writer is not None and hasattr(writer, "flush"):
            writer.flush()

    def get_export_metadata(self, run_name: str) -> dict[str, list | str | float]:
        env = self.env.unwrapped
        ik = env.action_manager.get_term("ee_delta").cfg
        obs = env.observation_manager
        return {
            "run_path": run_name,
            "action_type": "ee_delta_hole_frame",
            "action_pos_scale": ik.delta_pos_scale,
            "action_rot_scale": ik.delta_ori_scale,
            "policy_dt": env.step_dt,
            "joint_names": list(env.scene["robot"].joint_names),
            "observation_names": obs.active_terms["actor"],
            "observation_dims": [d[-1] for d in obs.group_obs_term_dim["actor"]],
        }
```

- [ ] **Step 4: Write `rl_insert/task/__init__.py`**

```python
"""PegInsert-xArm6 mjlab task. Importing this registers it."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import TaskParams, peg_insert_env_cfg
from .runner import PegInsertRunner, peg_insert_ppo_cfg

TASK_ID = "PegInsert-xArm6"

register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=peg_insert_env_cfg(),
    play_env_cfg=peg_insert_env_cfg(play=True),
    rl_cfg=peg_insert_ppo_cfg(),
    runner_cls=PegInsertRunner,
)

__all__ = ["TASK_ID", "TaskParams", "peg_insert_env_cfg", "PegInsertRunner", "peg_insert_ppo_cfg"]
```

- [ ] **Step 5: Write `scripts/smoke.py`** (and an empty `scripts/__init__.py`)

```python
"""Env smoke test: build, reset, check the reset geometry, step with zero / scripted actions, time it.

uv run python -m scripts.smoke [num_envs]
"""

import sys
import time

import torch

from mjlab.envs import ManagerBasedRlEnv

from rl_insert.task import mdp
from rl_insert.task.env_cfg import TaskParams, peg_insert_env_cfg


def main(num_envs: int = 4) -> None:
    cfg = peg_insert_env_cfg()
    cfg.scene.num_envs = num_envs
    cfg.seed = 0
    env = ManagerBasedRlEnv(cfg, device="cpu")
    p = TaskParams()
    obs, _ = env.reset()
    print(
        "obs dims:",
        {k: tuple(v.shape) for k, v in obs.items()},
        "action dim:",
        env.action_manager.total_action_dim,
    )
    print("actor terms:", env.observation_manager.active_terms["actor"])

    # Reset geometry: tip 10-20 mm above the mouth, within the lateral range, peg in hand.
    rel = mdp.true_tip(env) - mdp.hole_mouth(env)
    slip = (mdp.true_tip(env) - mdp.nominal_tip(env)).norm(dim=-1)
    print("tip - mouth (mm):\n", (rel * 1e3).round(decimals=2))
    print("tip vs nominal tip (mm):", (slip * 1e3).round(decimals=2))
    assert ((rel[:, 2] >= p.start_height[0] - 1e-6) & (rel[:, 2] <= p.start_height[1] + 1e-6)).all()
    assert (slip < 0.006).all()

    # Hold still for 1 s: the peg must stay in hand (grip + gravcomp), no termination.
    zero = torch.zeros(num_envs, env.action_manager.total_action_dim)
    for _ in range(20):
        obs, rew, term, trunc, _ = env.step(zero)
    slip2 = (mdp.true_tip(env) - mdp.nominal_tip(env)).norm(dim=-1)
    print("after 1 s hold, slip (mm):", (slip2 * 1e3).round(decimals=2), "terminated:", term.tolist())
    print("ik error after hold:", obs["actor"][:, -6:].abs().max().item())
    assert not term.any(), "terminated while holding still"

    # Scripted descent: move toward the hole (true hole, no bias) and down; report depth / force.
    t0 = time.perf_counter()
    steps = 60
    for i in range(steps):
        rel = mdp.true_tip(env) - mdp.hole_mouth(env)
        a = torch.zeros(num_envs, 6)
        a[:, :2] = (-rel[:, :2] / p.action_pos_scale).clamp(-1, 1)
        a[:, 2] = -0.5 if i > 5 else 0.0
        obs, rew, term, trunc, extras = env.step(a)
        if i % 10 == 0 or term.any():
            lat, depth = mdp.tip_in_hole(env)
            f = mdp.peg_hole_force(env).norm(dim=-1)
            print(
                f"step {i:3d} depth(mm) {[round(x, 1) for x in (depth * 1e3).tolist()]} "
                f"lat(mm) {[round(x, 2) for x in (lat * 1e3).tolist()]} "
                f"F(N) {[round(x, 1) for x in f.tolist()]} "
                f"rew {[round(x, 3) for x in rew.tolist()]} term {term.tolist()}"
            )
        if term.any():
            tm = env.termination_manager
            print("   why:", {n: tm.get_term(n).nonzero().flatten().tolist() for n in tm.active_terms})
    dt = (time.perf_counter() - t0) / steps
    print(f"wall time per policy step ({num_envs} envs, CPU): {dt * 1e3:.1f} ms")
    print(
        "contacts in world 0 (nacon total):",
        int(env.sim.data.nacon.numpy()[0]) if hasattr(env.sim.data.nacon, "numpy") else env.sim.data.nacon,
    )
    print("episode log keys:", sorted(extras.get("log", {}).keys())[:20])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
```

- [ ] **Step 6: Run the smoke test**

Run: `MJLAB_WARP_QUIET=1 uv run python -m scripts.smoke 4`
Expected:
- obs dims actor `(4, 33)`, critic `(4, 51)`, action dim 6;
- the reset-geometry assertions pass;
- the scripted descent runs 60 steps, printing depth, lateral error and force every 10th step and at any termination, without NaN;
- about 25–35 ms per policy step.

- [ ] **Step 7: Commit**

```bash
git add rl_insert/task scripts/__init__.py scripts/smoke.py
git commit -m "feat: PegInsert-xArm6 mjlab task (weld grasp, IK action, curriculum)"
```

---

### Task 3: Grasp-hold gate and shared checks

**Files:**
- Create: `rl_insert/task/checks.py`, `tests/test_grasp_hold.py`

The grasp-hold test is the regression gate for prototyping finding 2.

- [ ] **Step 1: Write `tests/test_grasp_hold.py`**

```python
"""The held peg must stay put in the grasp on MuJoCo Warp (CPU here; the probe runs it on GPU).

settle: the first 0.5 s after reset (no knock when the fingers close); creep: the next 2 s.
"""

from rl_insert.task.checks import grasp_hold_slip, make_env


def test_grasp_holds_for_2s():
    env = make_env(num_envs=64, device="cpu")
    slip = grasp_hold_slip(env, seconds=2.0, settle_s=0.5)
    print(f"max slip along TCP z (mm): {slip}")
    assert slip["settle_mm"] < 0.5
    assert slip["creep_mm"] < 0.5
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `MJLAB_WARP_QUIET=1 uv run pytest tests/test_grasp_hold.py -q`
Expected: `ModuleNotFoundError: No module named 'rl_insert.task.checks'`.

- [ ] **Step 3: Write `rl_insert/task/checks.py`**

```python
"""Sim checks shared by the probe job (model.training --mode=probe) and the local tests."""

from __future__ import annotations

import time

import torch

from mjlab.envs import ManagerBasedRlEnv

from . import mdp
from .env_cfg import TaskParams, peg_insert_env_cfg


def make_env(
    num_envs: int, device: str, seed: int = 0, render: bool = False, p: TaskParams = TaskParams()
) -> ManagerBasedRlEnv:
    cfg = peg_insert_env_cfg(p, play=True)  # play: no observation noise or curriculum
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    return ManagerBasedRlEnv(cfg, device=device, render_mode="rgb_array" if render else None)


def tip_along_tcp_z(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Peg tip position along the TCP z axis, m: grows as the peg slides out of the grasp."""
    pos, rot = mdp.tcp_pose(env)
    return ((mdp.true_tip(env) - pos).unsqueeze(1) @ rot[:, :, 2:]).flatten()


def grasp_hold_slip(env: ManagerBasedRlEnv, seconds: float = 2.0, settle_s: float = 0.5) -> dict[str, float]:
    """Peg slip along the TCP z axis (mm, max |.| over envs) while the arm holds still.

    settle_mm: the first settle_s after reset (a pad-only grasp got knocked up to 5 mm here on
    Warp). creep_mm: the following `seconds`, the steady slip.
    """
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)

    def hold(duration: float) -> None:
        for _ in range(round(duration / env.step_dt)):
            _, _, term, trunc, _ = env.step(zero)
            assert not (term | trunc).any(), "an env reset during the hold; slip would be meaningless"

    z0 = tip_along_tcp_z(env)
    hold(settle_s)
    z1 = tip_along_tcp_z(env)
    hold(seconds)
    z2 = tip_along_tcp_z(env)
    return {"settle_mm": float((z1 - z0).abs().max()) * 1e3, "creep_mm": float((z2 - z1).abs().max()) * 1e3}


def throughput(env: ManagerBasedRlEnv, steps: int = 50, warmup: int = 5) -> dict[str, float]:
    """Policy and physics steps per second (summed over envs) with uniform random actions."""
    env.reset()
    shape = (env.num_envs, env.action_manager.total_action_dim)
    for _ in range(warmup):
        env.step(torch.rand(shape, device=env.device) * 2 - 1)
    if env.device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(torch.rand(shape, device=env.device) * 2 - 1)
    if env.device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    policy = steps * env.num_envs / dt
    return {
        "policy_steps_per_s": policy,
        "physics_steps_per_s": policy * env.cfg.decimation,
        "seconds_per_policy_step": dt / steps,
    }
```

- [ ] **Step 4: Run it**

Run: `MJLAB_WARP_QUIET=1 uv run pytest tests/test_grasp_hold.py -q -s`
Expected: PASS, printing settle about 0.14 mm and creep about 0.07 mm. Takes roughly 10–90 s on CPU.

- [ ] **Step 5: Check that the gate bites**

In `rl_insert/task/env_cfg.py`, temporarily set `TaskParams.grasp_solref` to `0.02` and rerun: the test FAILS (creep or settle ≥ 0.5 mm). Revert it and rerun: PASS.

- [ ] **Step 6: Commit**

```bash
git add rl_insert/task/checks.py tests/test_grasp_hold.py
git commit -m "test: grasp-hold gate on MuJoCo Warp (settle and creep < 0.5 mm)"
```

---

### Task 4: Viam training entry point

**Files:**
- Create: `rl_insert/task/viam_upload.py`, `model/__init__.py` (empty), `model/package.py`, `model/training.py`, `scripts/onnx_check.py`
- Test: `tests/test_training_cli.py`

- [ ] **Step 1: Write the failing test `tests/test_training_cli.py`**

```python
"""The contract with Viam's launcher: the args it passes, and strict parsing of ours."""

from pathlib import Path

import pytest

from model.training import parse_args


def test_viam_args_with_placeholder_dataset_and_quoted_custom_args():
    args = parse_args(
        [
            "--model_output_directory=/gcs/bucket/out",
            "--dataset_file=/gcs/bucket/dataset.jsonl",  # ignored: training is in simulation
            "--clearance_mm='1.0'",  # Viam can leave the shell quotes on
            '--run_name="first-run"',
            "--upload_part_id=abc",
        ]
    )
    assert args.model_output_directory == Path("/gcs/bucket/out")
    assert args.clearance_mm == 1.0
    assert args.run_name == "first-run"
    assert args.upload_part_id == "abc"
    assert args.mode == "train"


def test_sequence_dataset_args_are_accepted_too():
    args = parse_args(
        [
            "--model_output_directory=/o",
            "--binary_data_file=/b",
            "--tabular_data_file=/t",
            "--sequences_file=/s",
        ]
    )
    assert args.run_name.startswith("peg-insert-")


@pytest.mark.parametrize("bad", ["--bogus=1", "--max_iter=5"])  # unknown, and an abbreviation
def test_unknown_or_abbreviated_args_exit_2(bad):
    with pytest.raises(SystemExit) as e:
        parse_args(["--model_output_directory=/o", bad])
    assert e.value.code == 2
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run pytest tests/test_training_cli.py -q`
Expected: `ModuleNotFoundError: No module named 'model'`.

- [ ] **Step 3: Write `rl_insert/task/viam_upload.py`**

```python
"""Upload a training video to Viam Data (viam-sdk 0.85).

Credentials come from API_KEY / API_KEY_ID, the names Viam training jobs set. Without them, or
without a part id, upload is skipped with a warning so local runs never fail on it.
"""

from __future__ import annotations

import asyncio
import os
import warnings

from viam.app.viam_client import ViamClient
from viam.rpc.dial import DialOptions


async def upload_video_async(
    path: str, part_id: str | None, tags: list[str] | None = None, dataset_ids: list[str] | None = None
) -> str | None:
    key, key_id = os.environ.get("API_KEY"), os.environ.get("API_KEY_ID")
    if not (key and key_id and part_id):
        warnings.warn(f"Viam upload skipped for {path}: API_KEY, API_KEY_ID and part_id are all required")
        return None
    client = await ViamClient.create_from_dial_options(DialOptions.with_api_key(key, key_id))
    try:
        return await client.data_client.file_upload_from_path(
            path, part_id=part_id, tags=tags, dataset_ids=dataset_ids, mime_type="video/mp4"
        )  # the uploaded file's id
    finally:
        client.close()


def upload_video(
    path: str, part_id: str | None, tags: list[str] | None = None, dataset_ids: list[str] | None = None
) -> str | None:
    """Blocking wrapper for scripts and the training loop. Returns the file id, or None if skipped.

    Raises on upload errors; callers that must not fail (training) catch and log.
    """
    return asyncio.run(upload_video_async(path, part_id, tags, dataset_ids))
```

- [ ] **Step 4: Write `model/package.py`**

```python
"""The model package files beyond policy.onnx: config.json (the deploy contract) and model_metrics.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import onnx
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.envs import ManagerBasedRlEnvCfg
from rl_insert.peg_hole import PegHoleParams
from rl_insert.robot import ARM_JOINTS, FLANGE, XARM6_URDF, ArmParams, GripperParams
from rl_insert.task.env_cfg import OBS_DOC, TaskParams

SCHEMA_VERSION = 1

# model_metrics.json name -> tensorboard scalar tag. mjlab logs Episode_Termination/<k> as the number
# of envs that ended for reason k per reset (count_nonzero), averaged by rsl_rl over the iteration,
# so build_metrics turns those into rates: each count over the sum of all four at that iteration.
METRIC_TAGS = {
    "success_rate": "Episode_Termination/success",
    "slip_rate": "Episode_Termination/peg_slipped",
    "excess_force_rate": "Episode_Termination/excess_force",
    "timeout_rate": "Episode_Termination/time_out",
    "mean_reward": "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "curriculum_difficulty": "Curriculum/difficulty",
}
MAX_METRIC_POINTS = 100


def _tensor_contract(value_info) -> dict:
    return {
        "name": value_info.name,
        "dtype": "float32",
        "shape": [d.dim_value for d in value_info.type.tensor_type.shape.dim],
    }


def build_config(onnx_path: Path, env_cfg: ManagerBasedRlEnvCfg, p: TaskParams, run_name: str) -> dict:
    model = onnx.load(onnx_path)
    meta = {e.key: e.value for e in model.metadata_props}
    names = meta["observation_names"].split(",")
    dims = [round(float(d)) for d in meta["observation_dims"].split(",")]
    actor = env_cfg.observations["actor"].terms
    terms, offset = [], 0
    for name, dim in zip(names, dims, strict=True):
        noise = actor[name].noise
        terms.append(
            {
                "name": name,
                "offset": offset,
                "dim": dim,
                **OBS_DOC[name],
                "train_noise": None if noise is None else [noise.n_min, noise.n_max],
            }
        )
        offset += dim

    ik = env_cfg.actions["ee_delta"]
    robot = env_cfg.scene.entities["robot"]
    default_q = robot.init_state.joint_pos
    g, ph, arm = GripperParams(), PegHoleParams(clearance=p.clearance), ArmParams()
    events = env_cfg.events
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "model": {
            "format": "onnx",
            "opset": model.opset_import[0].version,
            "inputs": [_tensor_contract(i) for i in model.graph.input],
            "outputs": [_tensor_contract(o) for o in model.graph.output],
            "obs_normalization": "inside the graph",
        },
        "observation": {"dim": offset, "terms": terms},
        "action": {
            "type": "ee_delta_hole_frame",
            "dim": 6,
            "layout": "[dx, dy, dz, rx, ry, rz]: translation and axis-angle rotation of the TCP",
            "frame": "hole frame axes (origin at the mouth, +z out of the surface, insertion is -z). "
            "Rotate into the arm base frame before use. Applied to the measured TCP pose each "
            "step (target = FK(measured) + delta), rotation pre-multiplied: R_target = R_delta R_tcp.",
            "pos_scale_m": ik.delta_pos_scale,
            "rot_scale_rad": ik.delta_ori_scale,
            "clip": None,
        },
        "control": {
            "policy_dt": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
            "physics_dt": env_cfg.sim.mujoco.timestep,
            "joint_names": list(ARM_JOINTS),
            "default_joint_pos": [default_q[j] for j in ARM_JOINTS],
            "ik": {
                "method": "damped least squares, joint space: (J^T W J + damping^2 I) dq = J^T W e",
                "frame": f"{ik.frame_type} {ik.frame_name}",
                "solved_every_physics_step": True,
                "damping": ik.damping,
                "max_dq_rad": ik.max_dq,
                "position_weight": ik.position_weight,
                "orientation_weight": ik.orientation_weight,
                "joint_limit_weight": ik.joint_limit_weight,
                "posture_weight": ik.posture_weight,
            },
            "joint_servo_sim": {"kp": arm.kp, "dampratio": arm.dampratio, "gravity_compensated": True},
            "gripper": {"close_command": 0.0, **asdict(g)},
        },
        "robot": {
            "model_id": "ufactory-xarm6",
            "urdf": "rl_insert/assets/xarm6/xarm6.urdf",
            "urdf_sha256": hashlib.sha256(XARM6_URDF.read_bytes()).hexdigest(),
            "flange": FLANGE,
            "nominal_tcp": {
                "flange_to_pad_centre_m": g.tcp,
                "protrusion_m": p.protrusion,
                "flange_to_peg_tip_m": [0.0, 0.0, g.tcp + p.protrusion],
            },
        },
        "task": {
            "peg_hole": asdict(ph) | {"hole_radius": ph.hole_radius},
            "clearance_m": p.clearance,
            "nominal_hole_mouth_base_frame": list(p.hole_mouth),
            "pre_insert_tip_height_m": list(p.start_height),
            "success_depth_m": p.success_depth,
            "success_centred_m": p.success_centred,
            "time_limit_s": env_cfg.episode_length_s,
            "force_limit_n": p.force_limit,
        },
        "robustness": {
            "start_lateral_m": p.start_lateral,
            "hole_bias_m": list(p.hole_bias),
            "grasp_offset_m": p.grasp_offset,
            "grasp_tilt_rad": p.grasp_tilt,
            "difficulty_init": p.difficulty_init,
            "hole_friction": list(events["friction_peg_hole"].params["ranges"]),
            "pad_friction": list(events["friction_finger_peg"].params["ranges"]),
            "start_joint_offset_rad": p.start_joint_offset,
            "arm_pd_gain_scale": list(p.pd_gain_scale),
            "action_delay_policy_steps": list(p.action_delay_steps),
            "observation_noise": "per term: observation.terms[].train_noise (uniform)",
            "not_modelled": [
                "force/torque sensing (none on the real arm either)",
                "hole orientation error (the hole is upright, zero yaw; bias is position only)",
                "part dimension tolerances (clearance is fixed per run)",
                "real grasp slip (the grasp is a compliant weld)",
                "joint-target streaming: sim re-solves IK toward the desired pose every 1 ms",
            ],
        },
        "grasp_model": {
            "type": "weld",
            "doc": "sim holds the peg with a compliant weld (gripper body to peg, anchored at the TCP) at "
            "the sampled grasp offset; pads still close on it at a fixed grip force. Real grasp "
            "slip is not modelled: at <= 40 N the G2 (10-50 N grip) holds a 20 g peg.",
            "weld_solref_timeconst_s": p.grasp_solref,
            "grip_force_n": g.grip_force,
        },
    }


def build_metrics(event_dir: Path) -> dict[str, list[float]]:
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    series = {tag: {e.step: e.value for e in ea.Scalars(tag)} for tag in METRIC_TAGS.values() if tag in have}
    ends = [v for tag, v in series.items() if tag.startswith("Episode_Termination/")]
    total = {step: sum(e.get(step, 0.0) for e in ends) for step in set().union(*ends)}
    metrics = {}
    for name, tag in METRIC_TAGS.items():
        if tag in series:
            points = sorted(series[tag].items())
            if tag.startswith("Episode_Termination/"):  # count -> rate
                values = [v / total[step] if total[step] > 0 else 0.0 for step, v in points]
            else:
                values = [v for _, v in points]
            stride = max(1, -(-len(values) // MAX_METRIC_POINTS))
            kept = values[::stride]
            if (len(values) - 1) % stride:
                kept.append(values[-1])  # always end on the final iteration
            metrics[name] = kept
    return metrics


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
```

- [ ] **Step 5: Write `model/training.py`** (and an empty `model/__init__.py`)

```python
"""Viam custom training script for PegInsert-xArm6 (mjlab + RSL-RL PPO on MuJoCo Warp).

Viam runs `python3 -m model.training --model_output_directory=... <dataset args> --key=value ...`.
The dataset is a required placeholder and is ignored. Everything in the output directory except
tmp/ becomes the model package: policy.onnx, config.json, checkpoint.pt, model_metrics.json.

Modes:
- train (default): train on one GPU (CPU when there is none), upload training videos to Viam Data
  as they are written, then write the package.
- probe: a cheap first job on new hardware; writes probe.json (GPU, GL, grasp hold, throughput,
  video upload). Every check is independent and the job always exits 0.

mujoco reads MUJOCO_GL once, when its GL module loads, and mjlab defaults it to egl on Linux, so
the GL backend is picked (in a subprocess) before anything imports mujoco or mjlab.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

logger = logging.getLogger("rl-insert")

TAGS = ["rl-insert"]
GL_TEST = """
import mujoco
xml = "<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 32, 32)
r.update_scene(d)
assert r.render().any(), "blank frame"
try:  # best effort: OSMesa returns None here
  from OpenGL import GL
  print(GL.glGetString(GL.GL_VENDOR).decode(), "|", GL.glGetString(GL.GL_RENDERER).decode())
except Exception:
  print("unknown")
r.close()
"""


def unquote(value: str) -> str:
    """Viam passes some `--key=value` values through with the shell quotes still on."""
    return value.strip("'\"")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse strictly: an unknown or misspelled custom arg exits 2."""
    p = argparse.ArgumentParser(prog="model.training", allow_abbrev=False)
    p.add_argument("--model_output_directory", type=lambda s: Path(unquote(s)), required=True)
    # Dataset arguments Viam passes. Accepted and ignored: this task trains in simulation.
    for name in ("dataset_file", "binary_data_file", "tabular_data_file", "sequences_file"):
        p.add_argument(f"--{name}", type=unquote, default=None, help="ignored (placeholder dataset)")
    p.add_argument("--mode", type=unquote, choices=("train", "probe"), default="train")
    p.add_argument("--clearance_mm", type=lambda s: float(unquote(s)), default=0.5)
    p.add_argument(
        "--max_iterations",
        type=lambda s: int(unquote(s)),
        default=None,
        help="PPO iterations (default: the task's PPO config)",
    )
    p.add_argument("--num_envs", type=lambda s: int(unquote(s)), default=4096)
    p.add_argument("--seed", type=lambda s: int(unquote(s)), default=42)
    p.add_argument(
        "--video_every_iters",
        type=lambda s: int(unquote(s)),
        default=250,
        help="record a training clip every N PPO iterations (0: no video)",
    )
    p.add_argument(
        "--video_length",
        type=lambda s: int(unquote(s)),
        default=200,
        help="clip length in policy steps (20 Hz)",
    )
    p.add_argument(
        "--upload_part_id",
        type=unquote,
        default=None,
        help="machine part the videos are uploaded under (no upload without it)",
    )
    p.add_argument("--upload_dataset_id", type=unquote, default=None)
    p.add_argument("--run_name", type=unquote, default=None, help="default: peg-insert-<timestamp>")
    args = p.parse_args(argv)
    args.run_name = args.run_name or time.strftime("peg-insert-%Y%m%d-%H%M%S")
    return args


def select_gl() -> dict:
    """Set MUJOCO_GL to a backend that renders here: cgl on macOS, else egl, else osmesa.

    Returns what was tried; "backend" is None when nothing works (video is then disabled).
    """
    candidates = ["cgl"] if platform.system() == "Darwin" else ["egl", "osmesa"]
    info: dict = {"tried": {}, "backend": None}
    for backend in candidates:
        env = {**os.environ, "MUJOCO_GL": backend}  # mujoco sets PYOPENGL_PLATFORM to match
        try:
            r = subprocess.run(
                [sys.executable, "-c", GL_TEST], env=env, capture_output=True, text=True, timeout=120
            )
            ok, err = r.returncode == 0, r.stderr.strip().splitlines()[-1:] or [f"exit {r.returncode}"]
        except subprocess.TimeoutExpired:
            ok, err = False, ["timed out"]
        info["tried"][backend] = "ok" if ok else err[0]
        if ok:
            # Which GL actually rendered: Mesa's llvmpipe also answers EGL, so "egl ok" alone
            # doesn't mean the GPU is rendering.
            info["renderer"] = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else None
            info["backend"] = backend
            os.environ["MUJOCO_GL"] = backend
            break
    if info["backend"] is None:
        logger.warning("no working GL backend (%s): video disabled", info["tried"])
    return info


def _device() -> str:
    import torch
    import warp as wp

    return "cuda:0" if torch.cuda.is_available() and wp.is_cuda_available() else "cpu"


class VideoUploader(threading.Thread):
    """Uploads each finished mp4 in `folder` to Viam Data. Failures are logged, never raised.

    mjlab's VideoRecorder writes a whole clip at once when it ends, so a file counts as finished
    once its size has been unchanged for one poll and it is older than the poll period.
    """

    def __init__(self, folder: Path, steps_per_iter: int, args: argparse.Namespace, poll_s: float = 15.0):
        super().__init__(daemon=True)
        self.folder, self.steps_per_iter, self.args, self.poll_s = folder, steps_per_iter, args, poll_s
        self.stop = threading.Event()
        self.sizes: dict[Path, int] = {}
        self.done: dict[str, str | None] = {}  # file name -> uploaded file id (None: skipped/failed)

    def run(self) -> None:
        while not self.stop.wait(self.poll_s):
            self.sweep(final=False)

    def sweep(self, final: bool) -> None:
        from rl_insert.task.viam_upload import upload_video

        for f in sorted(self.folder.glob("*.mp4")):
            if f.name in self.done:
                continue
            size = f.stat().st_size
            stable = size > 0 and self.sizes.get(f) == size and time.time() - f.stat().st_mtime > self.poll_s
            self.sizes[f] = size
            if not (final or stable):
                continue
            m = re.search(r"step-(\d+)", f.name)
            tags = [*TAGS, self.args.run_name] + ([f"iter-{int(m[1]) // self.steps_per_iter}"] if m else [])
            try:
                ds = [self.args.upload_dataset_id] if self.args.upload_dataset_id else None
                self.done[f.name] = upload_video(str(f), self.args.upload_part_id, tags, ds)
                logger.info("video %s -> %s", f.name, self.done[f.name])
            except Exception:
                self.done[f.name] = None
                logger.error("video upload failed for %s:\n%s", f.name, traceback.format_exc())


def train(args: argparse.Namespace, gl: dict) -> None:
    from mjlab.scripts.train import TrainConfig, run_train

    from model.package import build_config, build_metrics, write_json
    from rl_insert.task import TASK_ID, TaskParams, peg_insert_env_cfg, peg_insert_ppo_cfg

    out = args.model_output_directory
    log_dir = out / "tmp" / "logs" / args.run_name  # all run output stays under tmp/
    p = TaskParams(clearance=args.clearance_mm / 1000)
    env_cfg = peg_insert_env_cfg(p)
    env_cfg.scene.num_envs = args.num_envs
    agent = peg_insert_ppo_cfg()
    agent.seed = args.seed
    agent.run_name = args.run_name
    if args.max_iterations is not None:
        agent.max_iterations = args.max_iterations
    video = gl["backend"] is not None and args.video_every_iters > 0
    cfg = TrainConfig(
        env=env_cfg,
        agent=agent,
        video=video,
        video_length=args.video_length,
        # VideoRecorder counts env.step() calls; one PPO iteration is num_steps_per_env of them.
        video_interval=max(1, args.video_every_iters) * agent.num_steps_per_env,
        log_root=str(log_dir.parent),
        gpu_ids=None,  # unused: run_train is called directly, on one device
    )
    # run_train picks its device from CUDA_VISIBLE_DEVICES: "" is CPU, otherwise cuda:LOCAL_RANK.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" if _device().startswith("cuda") else ""

    uploader = VideoUploader(log_dir / "videos" / "train", agent.num_steps_per_env, args)
    if video:
        uploader.folder.mkdir(parents=True, exist_ok=True)
        uploader.start()
    try:
        run_train(TASK_ID, cfg, log_dir)
    finally:
        uploader.stop.set()
        if video:
            uploader.join()
            uploader.sweep(final=True)

    ckpt = max(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    shutil.copy(log_dir / f"{log_dir.name}.onnx", out / "policy.onnx")
    shutil.copy(ckpt, out / "checkpoint.pt")
    write_json(out / "config.json", build_config(out / "policy.onnx", env_cfg, p, args.run_name))
    write_json(out / "model_metrics.json", build_metrics(log_dir))
    logger.info(
        "wrote %s (from %s); videos: %s", sorted(f.name for f in out.iterdir()), ckpt.name, uploader.done
    )


def _check(results: dict, name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}
    results[name] = {"check_s": round(time.perf_counter() - t0, 1), **results[name]}
    logger.info("%s: %s", name, json.dumps(results[name], default=str)[:2000])


def probe(args: argparse.Namespace, gl: dict) -> None:
    out = args.model_output_directory
    results: dict = {"gl": gl}

    def nvidia_smi() -> dict:
        exe = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        return {"path": exe, "returncode": r.returncode, "output": (r.stdout + r.stderr)[-4000:]}

    def cuda() -> dict:
        import torch
        import warp as wp

        drv = wp.get_cuda_driver_version() if wp.is_cuda_available() else None
        return {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "warp": wp.__version__,
            "warp_cuda_available": wp.is_cuda_available(),
            "warp_cuda_devices": wp.get_cuda_device_count(),
            "cuda_driver_version": drv,
            "device_used": _device(),
        }

    def grasp_hold() -> dict:
        from rl_insert.task.checks import grasp_hold_slip, make_env

        env = make_env(64, _device())
        try:
            slip = grasp_hold_slip(env, seconds=2.0)
        finally:
            env.close()
        return {"num_envs": 64, "hold_s": 2.0, **slip, "pass": max(slip.values()) < 0.5}

    def speed() -> dict:
        from rl_insert.task.checks import make_env, throughput

        env = make_env(args.num_envs, _device())
        try:
            return {"num_envs": args.num_envs, **throughput(env)}
        finally:
            env.close()

    def video() -> dict:
        if gl["backend"] is None:
            return {"skipped": "no GL backend"}
        import mediapy
        import torch

        from rl_insert.task.checks import make_env
        from rl_insert.task.viam_upload import upload_video

        env = make_env(1, _device(), render=True)
        try:
            env.reset()
            frames = []
            for _ in range(40):  # 2 s at 20 Hz
                env.step(torch.zeros(1, 6, device=env.device))
                f = env.render()
                frames.append(f[0] if f.ndim == 4 else f)
        finally:
            env.close()
        path = out / "tmp" / "probe.mp4"
        mediapy.write_video(path, frames, fps=20)
        file_id = upload_video(
            str(path),
            args.upload_part_id,
            [*TAGS, "probe"],
            [args.upload_dataset_id] if args.upload_dataset_id else None,
        )
        return {"path": str(path), "frames": len(frames), "file_id": file_id, "uploaded": file_id is not None}

    for name, fn in (
        ("nvidia_smi", nvidia_smi),
        ("cuda", cuda),
        ("grasp_hold", grasp_hold),
        ("throughput", speed),
        ("video", video),
    ):
        _check(results, name, fn)
    (out / "probe.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    print(json.dumps(results, indent=2, default=str))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_output_directory = args.model_output_directory.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = args.model_output_directory
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    os.chdir(out / "tmp")  # anything a library writes relative to cwd stays out of the package
    gl = select_gl()
    if args.mode == "probe":
        probe(args, gl)
    else:
        train(args, gl)


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the dev script `scripts/onnx_check.py`**

```python
"""Check a model package's policy.onnx against its checkpoint.pt on one real observation.

uv run python -m scripts.onnx_check <package_dir>    # model.training's --model_output_directory
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from rl_insert.task import PegInsertRunner, TaskParams, peg_insert_env_cfg, peg_insert_ppo_cfg


def main(package_dir: str) -> None:
    pkg = Path(package_dir)
    ckpt, onnx_path = pkg / "checkpoint.pt", pkg / "policy.onnx"
    clearance = json.loads((pkg / "config.json").read_text())["task"]["clearance_m"]

    model = onnx.load(onnx_path)
    print(
        "inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
    )
    print(
        "outputs:",
        [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output],
    )
    ops = [n.op_type for n in model.graph.node]
    print("first ops (normalizer = Sub/Div before the first Gemm):", ops[:6])
    print("metadata:", {p.key: p.value[:80] for p in model.metadata_props})

    env_cfg = peg_insert_env_cfg(TaskParams(clearance=clearance), play=True)
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cpu"))
    runner = PegInsertRunner(env, asdict(peg_insert_ppo_cfg()), log_dir=None, device="cpu")  # no logs written
    runner.load(str(ckpt))
    policy = runner.get_inference_policy("cpu")

    for _ in range(3):  # a few random steps, so velocities, last action and IK error are non-zero
        obs, *_ = env.step(torch.rand(1, 6) * 2 - 1)
    with torch.no_grad():
        torch_act = policy(obs).numpy()
    sess = ort.InferenceSession(str(onnx_path))
    onnx_act = sess.run(None, {"obs": obs["actor"].numpy().astype(np.float32)})[0]
    print("torch:", torch_act.round(5), "\nonnx: ", onnx_act.round(5))
    err = np.abs(torch_act - onnx_act).max()
    print(f"max abs diff: {err:.2e}")
    assert err < 1e-5

    # The normalizer must matter: running the MLP on raw obs (normalizer bypassed) differs.
    raw = policy.mlp(obs["actor"]).detach().numpy()
    print("diff vs un-normalized MLP:", np.abs(raw - onnx_act).max())


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 7: Run the CLI tests and the full suite**

Run: `uv run pytest -q`
Expected: 32 passed (27 from plan 1, 1 grasp, 4 CLI).

- [ ] **Step 8: Run a local job end to end, the way Viam would**

```bash
OUT=$(mktemp -d)/out
MJLAB_WARP_QUIET=1 uv run python -m model.training --model_output_directory=$OUT --dataset_file=x \
  --num_envs=2 --max_iterations=2 --video_every_iters=1 --video_length=20
ls $OUT; ls $OUT/tmp/logs/*/videos/train
MJLAB_WARP_QUIET=1 uv run python -m scripts.onnx_check $OUT
```

Expected:
- `$OUT` holds `checkpoint.pt`, `config.json`, `model_metrics.json`, `policy.onnx` and `tmp/`;
- two clips exist, `rl-video-step-0.mp4` and `rl-video-step-24.mp4`;
- the log says the upload was skipped (no credentials);
- `onnx_check` reports `obs` [1,33] → `actions` [1,6] and a max difference from torch below 1e-5.

`model_metrics.json` may be `{}` on a run this short, because no episode ends.

- [ ] **Step 9: Commit**

```bash
git add rl_insert/task/viam_upload.py model scripts/onnx_check.py tests/test_training_cli.py
git commit -m "feat: Viam training entry point (train/probe modes, video upload, config.json)"
```

---

### Task 5: Script package

**Files:**
- Create: `setup.py`, `MANIFEST.in`, `build.sh`
- Modify: `.gitignore` (add `dist/`)

- [ ] **Step 1: Write `setup.py`**

```python
from setuptools import find_packages, setup

# Viam custom training script package. Viam's launcher pip-installs this sdist and runs
# `python3 -m model.training`. Dependencies live in the training image
# (container/Dockerfile.train), so the job doesn't reinstall them on billed GPU time.
setup(
    name="rl-insert-train",
    version="0.1.0",
    packages=find_packages(include=["model", "rl_insert", "rl_insert.*"]),
    package_data={"rl_insert": ["assets/xarm6/*.urdf", "assets/xarm6/NOTICE", "assets/xarm6/meshes/*.stl"]},
    install_requires=[],
)
```

- [ ] **Step 2: Write `MANIFEST.in`**

```
recursive-include rl_insert/assets *
global-exclude __pycache__ *.py[co]
```

- [ ] **Step 3: Write `build.sh` and make it executable** (`chmod +x build.sh`); add `dist/` to `.gitignore`

```bash
#!/usr/bin/env bash
# Builds dist/rl-insert-train.tar.gz, the training-script package Viam's launcher pip-installs.
# Staged in its own directory: the repo's pyproject.toml describes the dev env, and pip/build
# would package that instead of setup.py.
set -euo pipefail
cd "$(dirname "$0")"
rm -rf dist && mkdir -p dist/stage
cp -R setup.py MANIFEST.in model rl_insert dist/stage/
# COPYFILE_DISABLE: macOS tar would otherwise add AppleDouble ._* entries.
(cd dist/stage && COPYFILE_DISABLE=1 uvx --from build pyproject-build --sdist --outdir .. . >/dev/null)
mv dist/rl_insert_train-*.tar.gz dist/rl-insert-train.tar.gz
rm -rf dist/stage
tar -tzf dist/rl-insert-train.tar.gz | grep -cE "\.py$|\.stl$|\.urdf$" | xargs echo "files (py/stl/urdf):"
```

- [ ] **Step 4: Build it, then install it in a fresh venv and run it from an empty directory**

```bash
./build.sh
V=$(mktemp -d); python3.12 -m venv $V/venv
echo "$PWD/.venv/lib/python3.12/site-packages" > $V/venv/lib/python3.12/site-packages/deps.pth  # stands in for the image
$V/venv/bin/pip install -q --no-deps dist/rl-insert-train.tar.gz
cd $(mktemp -d) && MJLAB_WARP_QUIET=1 $V/venv/bin/python -m model.training \
  --model_output_directory=$PWD/out --dataset_file=x --num_envs=2 --max_iterations=2 --video_every_iters=0
ls out; ls   # cwd holds only out/
cd - >/dev/null
```

Expected:
- `build.sh` reports 22 py/stl/urdf files;
- `out/` holds the four artifacts and `tmp/`;
- nothing else was written to the cwd.

- [ ] **Step 5: Commit**

```bash
git add setup.py MANIFEST.in build.sh .gitignore
git commit -m "feat: training-script package for Viam's launcher"
```

---

### Task 6: Training image

**Files:**
- Create: `container/Dockerfile.train`, `container/Dockerfile.train.dockerignore`

The ignore file is named after the Dockerfile; BuildKit uses it for this Dockerfile only. A repo-root `.dockerignore` of `*` would break plan 1's probe image, which copies from the context.

- [ ] **Step 1: Write `container/Dockerfile.train`**

```dockerfile
# syntax=docker/dockerfile:1
# Training image for the rl-insert Viam custom training script (PegInsert-xArm6, mjlab + RSL-RL).
#
# Viam (Vertex AI, linux/amd64, 2x A100) ignores ENTRYPOINT/CMD but keeps ENV. Its launcher runs
# `pip install <training script tar.gz>` then `python3 -m model.training "$@"`, so python3 and
# pip on PATH (a venv here) must already import every dependency; the package has none.
#
# No CUDA base image: the torch wheel bundles the CUDA runtime (nvidia-* wheels) and warp bundles
# NVRTC; only the driver comes from the host, mounted at /usr/local/nvidia.
#
# CUDA 13 (default) needs driver >= 580. Fallback for an older driver:
#   --build-arg TORCH_CUDA=cu126 --build-arg WARP_VERSION=1.17.0
# warp 1.18+ wheels (PyPI and pypi.nvidia alike) are CUDA 13.0 builds; 1.17.0 is the newest
# CUDA 12.9 build, and mujoco-warp 4.0.1 accepts warp >= 1.15. The fallback is unverified.
#
# Build (from this repo's root):
#   docker build --platform linux/amd64 -f container/Dockerfile.train -t rl-insert-train .
FROM ubuntu:24.04

ARG TORCH_CUDA=cu130
ARG TORCH_VERSION=2.14.1
ARG WARP_VERSION=1.19.0
ARG MJLAB_REF=6ab9846ea068714678fd0e9f791f898c6b697b20

# NVIDIA_DRIVER_CAPABILITIES: "graphics" asks the runtime to mount the EGL driver; whether Vertex
# honours it is unknown, so model.training probes EGL and falls back to OSMesa.
# MUJOCO_GL is deliberately unset: model.training picks it before importing mujoco.
# WARP_CACHE_PATH: compiled kernels; mount a volume here to reuse them across `docker run`s.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    NVIDIA_DISABLE_REQUIRE=1 \
    WARP_CACHE_PATH=/opt/warp-cache \
    MJLAB_WARP_QUIET=1

# libegl1 (glvnd) + the NVIDIA vendor file for EGL; libosmesa6 for software rendering;
# libgl1/libglx-mesa0 for mujoco's GL loader; ffmpeg for mediapy's mp4 writer.
# git: rsl_rl imports GitPython (code-state logging), which fails to import without it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev ca-certificates ffmpeg git \
        libegl1 libgl1 libglx-mesa0 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && mkdir -p /opt/warp-cache && chmod a+rwX /opt/warp-cache

RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}" "torch==${TORCH_VERSION}" \
    && pip install \
        "warp-lang==${WARP_VERSION}" \
        "mjlab @ https://github.com/mujocolab/mjlab/archive/${MJLAB_REF}.tar.gz" \
        "viam-sdk>=0.85" onnxruntime tensorboard \
    && python3 -c "import torch, mjlab, mujoco_warp, onnxruntime, viam; print('torch', torch.__version__, torch.version.cuda)"
```

- [ ] **Step 2: Write `container/Dockerfile.train.dockerignore`**

```
# The image installs nothing from the build context.
*
```

- [ ] **Step 3: Build, then run probe and train inside the image from the installed package (emulated CPU)**

```bash
docker build --platform linux/amd64 -f container/Dockerfile.train -t rl-insert-train:dev .
./build.sh
OUT=$(mktemp -d)
docker run --rm --platform linux/amd64 -v $PWD/dist:/pkg:ro -v $OUT:/out rl-insert-train:dev bash -c \
  "pip install -q --no-deps /pkg/rl-insert-train.tar.gz && cd /tmp \
   && python3 -m model.training --mode=probe --num_envs=2 --model_output_directory=/out/probe \
   && python3 -m model.training --model_output_directory=/out/train --dataset_file=x --num_envs=2 \
      --max_iterations=1 --video_every_iters=1 --video_length=20"
cat $OUT/probe/probe.json; ls $OUT/train
```

Expected:
- the image is about 7.7 GB;
- `probe.json` reports CUDA unavailable, a GL backend (egl via llvmpipe, or osmesa), a grasp hold with `pass: true`, and the video "uploaded": false (no credentials);
- `train/` holds the four artifacts.

Emulation is slow; allow 10–20 minutes.

- [ ] **Step 4: Commit**

```bash
git add container/Dockerfile.train container/Dockerfile.train.dockerignore
git commit -m "feat: training image (torch cu130, mjlab pin, EGL/OSMesa, viam-sdk)"
```

---

### Task 7: Probe job on Vertex

This task acts on external services; the user authorized it. Record every ID it produces in the results doc.

- [ ] **Step 1: Tag and push the image** (`-1` is the build suffix; never re-push a registered tag, bump the suffix instead)

```bash
docker tag rl-insert-train:dev docker.io/hipsterbrown/viam-rl-insert-train:0.1.0-1
docker push docker.io/hipsterbrown/viam-rl-insert-train:0.1.0-1
```

The Docker Hub repo must be public (Viam pulls it anonymously). If the push creates a private repo, make it public on hub.docker.com.

- [ ] **Step 2: Register the container, and note its ID**

```bash
viam train containers register --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --uri=docker.io/hipsterbrown/viam-rl-insert-train:0.1.0-1 --description="rl-insert-train 0.1.0-1"
viam train containers list --org-id=16518049-9dd3-479f-9644-c5d112aa42d8   # find the new container ID
```

- [ ] **Step 3: Upload the script package as version 1**

```bash
./build.sh
viam training-script upload --path=dist/rl-insert-train.tar.gz --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --script-name=rl-insert-train --version=1 --framework=onnx --type=robot_policy --visibility=private
```

- [ ] **Step 4: Submit the probe job, and note the job ID**

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=6581c38d39e1f5ebd5bc0e0f --model-name=rl-insert-probe --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:rl-insert-train --version=1 \
  --container-id=<container-id> \
  --args=mode=probe,num_envs=4096,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=probe-1
```

- [ ] **Step 5: Wait for it, then read `probe.json` from the model package**

```bash
viam train get --job-id=<job-id>       # poll every few minutes until it completes or fails
viam train logs --job-id=<job-id> | tail -80
viam packages export --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=rl-insert-probe --version=1 \
  --type=ml_model --destination=/tmp/rl-insert-probe && cat /tmp/rl-insert-probe/*/probe.json
```

If the package doesn't appear for export, `model.training` also prints `probe.json` to stdout: read it from `viam train logs`. Also confirm in the Viam app (DATA, tag `probe`) that the probe clip arrived under the `rl-insert` machine.

- [ ] **Step 6: Decide from `probe.json`, and fix before Task 8:**
  - **`cuda.torch_cuda_available` false, or `cuda_driver_version` below (13, 0):** the driver is older than 580. Rebuild with `--build-arg TORCH_CUDA=cu126 --build-arg WARP_VERSION=1.17.0`, push as `0.1.0-2`, register it, and re-probe. If mujoco-warp fails with warp 1.17, stop and report to the user.
  - **`gl.backend` null:** video is off; train anyway and report it. **Backend egl with an llvmpipe renderer:** fine, it's CPU rendering of env 0 only.
  - **`grasp_hold.pass` false on CUDA:** stop. That's a plan-1-style physics issue; report the numbers.
  - **`video.uploaded` false with an error:** fix the upload (key scope, part ID) before Task 8. Videos are the point of the user's request.
  - **Throughput:** `throughput.policy_steps_per_s` is a sanity check only. It uses random actions with no PPO update; Step 7 measures the real iteration time.

- [ ] **Step 7: Run a short training job on the GPU.** The probe never exercises the train path on the GPU: `run_train`, the VideoRecorder, ONNX export, tensorboard, and writing the package to `/gcs`. Run it for minutes before the multi-hour job:

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=6581c38d39e1f5ebd5bc0e0f --model-name=rl-insert-smoke --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:rl-insert-train --version=1 \
  --container-id=<container-id> \
  --args=clearance_mm=1.0,num_envs=4096,max_iterations=3,video_every_iters=1,video_length=40,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=smoke-1
```

Expected:
- the job completes;
- `viam packages export --name=rl-insert-smoke ...` gives `policy.onnx`, `config.json`, `checkpoint.pt` and `model_metrics.json` (possibly `{}`);
- about 3 clips tagged `smoke-1` appear in Viam Data.

From `viam train logs`, read the seconds per PPO iteration, which rsl_rl logs every iteration. Pick Task 8's `max_iterations` so the job runs about 2–4 hours.

- [ ] **Step 8: Write `docs/superpowers/specs/2026-11-17-vertex-probe-results.md`** (the IDs, `probe.json`, the smoke job's iteration time, the decisions) **and commit it**

```bash
git add docs/superpowers/specs/2026-11-17-vertex-probe-results.md
git commit -m "docs: Vertex probe job results"
```

---

### Task 8: First training job

- [ ] **Step 1: Submit**, using the container ID from Task 7, script version 1 (or a later version if Task 7 changed the code), and `max_iterations` from Task 7's throughput

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=6581c38d39e1f5ebd5bc0e0f --model-name=rl-insert --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:rl-insert-train --version=1 \
  --container-id=<container-id> \
  --args=clearance_mm=1.0,num_envs=4096,max_iterations=<n>,video_every_iters=250,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=c1.0-run1
```

- [ ] **Step 2: Monitor.** Check `viam train get` and `viam train logs` every 20–30 minutes. A video tagged `c1.0-run1` should appear in Viam Data about every 250 iterations. If none has arrived after the first 250 iterations, read the logs for uploader errors.

- [ ] **Step 3: When it finishes, export the package and check it**

```bash
viam packages export --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=rl-insert --version=1 \
  --type=ml_model --destination=/tmp/rl-insert-v1
ls /tmp/rl-insert-v1/*; python3 -m json.tool /tmp/rl-insert-v1/*/model_metrics.json | head -40
uv run python -c "
import glob, numpy as np, onnxruntime as ort
s = ort.InferenceSession(glob.glob('/tmp/rl-insert-v1/*/policy.onnx')[0])
out = s.run(None, {'obs': np.zeros((1, 33), np.float32)})[0]
assert out.shape == (1, 6) and np.isfinite(out).all(), out
print('policy.onnx ok', out)"
```

Expected: `policy.onnx`, `config.json`, `checkpoint.pt` and a non-empty `model_metrics.json`.

- [ ] **Step 4: Judge the run, and write `docs/superpowers/specs/2026-11-17-first-training-run.md`.** Include:
  - the job ID and hours taken;
  - the final values and trend of `success_rate`, `timeout_rate`, `excess_force_rate` and `curriculum_difficulty`;
  - links or tags for the videos, and what they show.

  Success means `success_rate` keeps rising and difficulty reaches 1.0. A flat success near 0 with mostly timeouts is a reward or curriculum problem: report it with the videos, and don't tune blind.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-11-17-first-training-run.md
git commit -m "docs: first training run on Viam"
```
