# MicroDuck milestone 3: Pollen's training recipe as a Viam training job — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** a Viam custom training job trains `Mjlab-Velocity-Flat-MicroDuck` (pollen-robotics/microduck_rl
@0daab16: mjlab 1.3.0, MuJoCo Warp, RSL-RL PPO, BAM actuators). It outputs a `policy.onnx` and
`config.json` that duck-policy runs. Our policy then goes through the milestone 2 parity check on the
preview machine.

**Architecture:** spec unit 7 (`docs/superpowers/specs/2026-11-21-microduck-sim-to-sim-design.md`).

- **Own image and own sdist.** Pollen's pins (mjlab 1.3.0, torch 2.9.1 on CUDA 12, warp 1.12.0,
  BAM @62bd8ce) can't share the peg-insert image (mjlab@6ab9846, torch 2.14.1+cu130, warp 1.19).
  Viam always runs `python3 -m model.training`, so the duck job ships its own `model/training.py`
  shim that calls `duck_train.train.main`.
- **Shared harness.** The generic pieces of `model/training.py` move into `model/harness.py` and
  `model/viam_upload.py`, so both jobs use them. The peg-insert job's behaviour and tests are
  unchanged.
- **After the job:** a local step runs our reference (`scripts/duck_reference.py`) on the trained
  ONNX to derive the safety block, then uploads the deployable `microduck-velocity` package.

**Tech stack:**
- Python 3.12, uv, Docker (buildx, linux/amd64, pushing to Docker Hub `hipsterbrown/`)
- the `viam` CLI (`train containers register`, `training-script upload`, `train submit custom`,
  `packages export/upload`)
- mjlab 1.3.0 `run_train`; microduck_rl's `run_export`
- onnxruntime, pytest

**Facts this plan relies on** (verified 2026-11-21; `MD` = microduck_rl@0daab16, `ML` = mjlab 1.3.0):

- **Training from Python** (`ML/scripts/train.py:25-47`):
  - Call `from mjlab.scripts.train import TrainConfig, run_train`, then
    `run_train(task_id, TrainConfig(env=…, agent=…, video=…, video_length=…, video_interval=…, gpu_ids=None), log_dir)`.
  - mjlab 1.3.0's `TrainConfig` has **no `log_root`**, unlike the peg-insert mjlab@6ab9846; `run_train`
    writes into `log_dir`.
  - Its fields are env, agent, registry_name, video, video_length, video_interval, enable_nan_guard,
    torchrunx_log_dir, wandb_run_path, wandb_checkpoint_name and gpu_ids.
  - `run_train` picks CPU unless `CUDA_VISIBLE_DEVICES` is set, so `set_visible_devices` must come first.
  - Before that, import `mjlab.tasks`: it loads the `mjlab.tasks` entry point, which registers
    microduck's tasks (`MD/pyproject.toml:40-41`).
  - `load_env_cfg(TASK)` and `load_rl_cfg(TASK)` come from the mjlab task registry. Read
    `ML/tasks/registry.py` for the exact names, as peg-insert's `rl_insert/task/__init__.py` uses
    them.
  - The runner is `MicroduckOnPolicyRunner` (registered).
  - Calling `run_train` directly uses one GPU (multi-GPU only via `launch_training`/torchrunx).
- **Defaults to override** (`MD/.../microduck_velocity_env_cfg.py:943-948`, `ML/rl/config.py:104`):
  - `agent.logger` defaults to `"wandb"`: set `"tensorboard"`;
  - set `agent.upload_model = False`;
  - `max_iterations` defaults to 50,000: set it from a job arg (default 4000);
  - `save_interval` is 250.
  - Set `WANDB_MODE=disabled` too.
  - wandb stays installed: `ML/tasks/velocity/rl/runner.py:1` imports it.
- **Run directory:** `model_<it>.pt` every `save_interval`, and `<run>.onnx` re-exported on every
  save, plus `params/`, tfevents files and `videos/train/*.mp4`.
- **Export without wandb** (`MD/src/mjlab_microduck/export.py:90`, `checkpoints.py:54-61`):
  - `run_export(task_id, ExportConfig(checkpoint_file=<local .pt>, num_envs=1, onnx_file=<out>))`.
  - The exported graph runs normalizer → MLP → mean action: input `obs` [1,61], output `actions`
    [1,14], opset 18.
  - Metadata keys include `joint_names`, `default_joint_pos`, `action_scale`,
    `observation_names` and `command_names`.
- **Command ranges:** `env_cfg.commands["twist"].ranges` are vx ±0.4, vy ±0.3, wz ±1.0
  (`velocity_env_cfg.py:648-650`).
- **Standing curriculum:** standing-env fraction ramps to 25 % by about iteration 2000, so early
  checkpoints don't stand at a zero command.
- **Robot model:** the velocity task trains on `robot_walk.xml`. duck-sim 0.2.0 is the ground-contact
  model: same bodies, joints and actuators, plus trunk and head collision. For walking that's fine,
  and parity uses the same `duck.xml` on both sides.
- **Viam job contract** (`model/training.py:1-15,56-67`; plan `2026-11-17-peg-insert-training.md:2117-2160`):
  - Viam runs `python3 -m model.training --model_output_directory=… --dataset_file=… --k=v …`, and
    values can arrive with quotes still on.
  - The dataset is a placeholder: `6581c38d39e1f5ebd5bc0e0f`.
  - Custom args go in `--args=k=v,k=v` (no commas inside values).
  - The output directory minus `tmp/` becomes the ML model package.
  - Videos upload with the `API_KEY`/`API_KEY_ID` env vars plus `upload_part_id`. The part is
    `cab989eb-90a0-4dce-8fc6-2c50d0ed1f10`, as before.
- **Hardware:** Vertex runs linux/amd64 on an A100 with driver 580 / CUDA 13, which runs CUDA 12.8
  wheels.
- **Python and wheels:** ubuntu:24.04's python3 is 3.12, which microduck needs (`>=3.12,<3.13`).
  `MD/uv.lock` resolves x86_64 torch 2.9.1 from PyPI with CUDA 12.8 wheels, and BAM to @62bd8ce.
- **Authorization:** the user's standing authorization for this project covers pushing training
  images, registering them, uploading training scripts and submitting jobs in DevRelDemos
  (`16518049-9dd3-479f-9644-c5d112aa42d8`). Never re-push a tag or re-upload a version; bump it.

---

## File structure

| File | Responsibility |
|---|---|
| `model/harness.py` (new) | Generic job helpers moved from `model/training.py`: `unquote`, `GL_TEST`, `select_gl`, `device`, `VideoUploader` (tags passed in), `set_visible_devices`, `check` |
| `model/viam_upload.py` (moved from `rl_insert/task/viam_upload.py`) | Video upload to Viam Data; `rl_insert/task/viam_upload.py` stays as a one-line re-export |
| `model/training.py` (modify) | Peg-insert entry, now importing helpers from `model.harness`; behaviour unchanged |
| `duck_train/__init__.py`, `duck_train/train.py` (new) | Duck job: `parse_args`, `train`, `probe`, `main` |
| `duck_train/package.py` (new) | `build_config` (pure), `verify_onnx`, `scalar_metrics`, `write_package` |
| `duck_train/setup.py` (new) | sdist metadata for `microduck-train` (packages `model`, `duck_train`) |
| `build_duck.sh` (new) | Stage and build `dist/microduck-train.tar.gz` with a `model/training.py` shim |
| `container/Dockerfile.duck` (new) | Training image: microduck_rl@0daab16 locked env in `/opt/venv` + viam-sdk + pip |
| `tests/test_duck_train.py` (new) | `parse_args`, `build_config`, `verify_onnx` (on the velstand ONNX), the shim |
| `scripts/duck_reference.py`, `scripts/duck_parity.py` (modify) | `--script` option: a command script from JSON, for our policy's measured behaviour |
| `docs/superpowers/specs/<date>-microduck-training-results.md` (new) | Job, curves, parity |

---

### Task 1: Extract the shared harness (peg-insert behaviour unchanged)

**Files:**
- Create: `model/harness.py`
- Move: `rl_insert/task/viam_upload.py` → `model/viam_upload.py` (with `git mv`), then recreate
  `rl_insert/task/viam_upload.py` as a **module alias**, not a star re-export.
  `tests/test_training_runtime.py` monkeypatches `viam_upload.upload_video` and
  `viam_upload.ViamClient` through `from rl_insert.task import viam_upload`; a star re-export would be
  a separate module object, and the patches wouldn't reach it. The alias:
  ```python
  """Moved to model/viam_upload.py; this name stays an alias of that module (tests patch through it)."""
  import sys

  from model import viam_upload as _m

  sys.modules[__name__] = _m
  ```
  `harness.VideoUploader.sweep()` must keep importing `upload_video` **at call time**
  (`from model.viam_upload import upload_video` inside `sweep`), so patches applied after import
  still take effect.
- Modify: `model/training.py`

- [ ] **Step 1: Baseline.** `uv run pytest -q`: record the pass count (expected 108 passed, 1 skipped).

- [ ] **Step 1b: Stop `build.sh` deleting `dist/`.** Its `rm -rf dist` would wipe milestone 2's
  artifacts (`dist/microduck-walk`, `dist/duck-sim`, `duck-reference.json`, `duck-ours.json`), which
  later tasks need. Change that line to
  `rm -rf dist/stage dist/rl-insert-train.tar.gz && mkdir -p dist/stage`.
- [ ] **Step 2: Move the generic code into `model/harness.py`, unchanged in behaviour.** Move:
  - `unquote`
  - `GL_TEST` and `select_gl`
  - `_device`, renamed `device`
  - `VideoUploader`, changed so `__init__` takes `tags: list[str]` and `run_name: str`, plus the
    existing `args` fields it reads (`upload_part_id`, `upload_dataset_id`), passed explicitly as
    keyword arguments rather than an `argparse.Namespace`. Build the tag list as
    `[*self.tags, self.run_name, iter-N]`. Import `upload_video` from `model.viam_upload`.
  - `set_visible_devices`
  - `_check`, renamed `check`
  - the module logger is `logging.getLogger("harness")`
- [ ] **Step 3: Update `model/training.py`** to import those names from `model.harness`. Keep:
  - the old private names as aliases (`_device = device`, `_check = check`), so existing tests and
    patches still resolve;
  - `TAGS = ["rl-insert"]`, passed into `VideoUploader(..., tags=TAGS, run_name=args.run_name,
    upload_part_id=args.upload_part_id, upload_dataset_id=args.upload_dataset_id)`.

  Keep peg-insert's old constructor working for its tests. Two tests in
  `tests/test_training_runtime.py` call `training.VideoUploader(tmp_path, 24, args, poll_s=...)`
  with a positional Namespace, so define a thin subclass in `model/training.py`:
  ```python
  class VideoUploader(harness.VideoUploader):
      """Peg-insert's signature (args Namespace) over the shared uploader."""

      def __init__(self, folder, steps_per_iter, args, poll_s: float = 15.0):
          super().__init__(folder, steps_per_iter, tags=TAGS, run_name=args.run_name,
                           upload_part_id=args.upload_part_id, upload_dataset_id=args.upload_dataset_id, poll_s=poll_s)
  ```
  Use it in `train()` unchanged (`VideoUploader(log_dir / "videos" / "train", agent.num_steps_per_env, args)`).
  Keep `import os` in `training.py` (tests patch `training.os`).

  Grep `tests/` for `training._device`, `training._check`, `VideoUploader(`, `viam_upload`,
  `training.set_visible_devices`, `training.write_package` and `training.UPLOAD_JOIN_TIMEOUT`, and
  keep every one working.
- [ ] **Step 4:** `uv run pytest -q` passes with the same count as Step 1. `./build.sh` still
  builds `dist/rl-insert-train.tar.gz`, and its file list includes `model/harness.py` and
  `model/viam_upload.py` (`find_packages` picks them up).
- [ ] **Step 5: Commit** with message `refactor: shared training-job harness in model/ (peg-insert unchanged)`.

---

### Task 2: The duck training entry and package writer

**Files:**
- Create: `duck_train/__init__.py` (empty), `duck_train/package.py`, `duck_train/train.py`
- Test: `tests/test_duck_train.py`

- [ ] **Step 1: Write the failing tests.**

```python
import json
from pathlib import Path

import pytest

from duck_train.package import build_config, verify_onnx
from duck_train.train import parse_args

JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
VELSTAND = Path("dist/microduck-walk/policy.onnx")


def test_parse_args_strict_and_unquotes(tmp_path):
    a = parse_args([f"--model_output_directory='{tmp_path}'", "--dataset_file=x", "--mode=probe",
                    "--max_iterations='20'", "--num_envs=64", "--run_name=p1"])
    assert a.model_output_directory == tmp_path and a.mode == "probe" and a.max_iterations == 20
    with pytest.raises(SystemExit):
        parse_args([f"--model_output_directory={tmp_path}", "--nmu_envs=1"])


def test_build_config_shape():
    cfg = build_config(joints=JOINTS, default_pose=[0.1] * 14, action_scale=1.0,
                       joint_limits=[[-1.0, 1.0]] * 14, twist_min=[-0.4, -0.3, -1.0],
                       twist_max=[0.4, 0.3, 1.0], control_hz=50,
                       source={"task": "Mjlab-Velocity-Flat-MicroDuck", "checkpoint": "model_3999.pt"})
    assert cfg["model"] == {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14}
    assert cfg["joints"] == JOINTS and cfg["safety"] == {} and cfg["control_hz"] == 50
    assert cfg["command"] == {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]}
    json.dumps(cfg)  # serializable


@pytest.mark.skipif(not VELSTAND.exists(), reason="needs dist/microduck-walk (scripts/fetch_walk_policy.py)")
def test_verify_onnx_accepts_a_pollen_policy():
    meta = verify_onnx(VELSTAND, JOINTS)
    assert float(meta["action_scale"]) == 1.0
```

- [ ] **Step 2: Run them and confirm they fail** (`ModuleNotFoundError: duck_train`).

- [ ] **Step 3: Implement `duck_train/package.py`.**

```python
"""What the MicroDuck training job writes: policy.onnx (the normalizer is part of the graph), config.json
in duck-policy's shape (as scripts/fetch_walk_policy.py writes for Pollen's velstand), checkpoint.pt and
model_metrics.json. The safety block is left empty: scripts/duck_reference.py --stats fills it after the job,
from rollouts in our reference."""

from __future__ import annotations

import json
from pathlib import Path


def build_config(*, joints, default_pose, action_scale, joint_limits, twist_min, twist_max, control_hz, source) -> dict:
    return {
        "schema": 1,
        "source": source,
        "model": {"input": "obs", "output": "actions", "obs_len": 61, "action_len": 14},
        "joints": list(joints),
        "default_pose": [float(v) for v in default_pose],
        "action_scale": float(action_scale),
        "joint_limits": [[float(lo), float(hi)] for lo, hi in joint_limits],
        "control_hz": int(control_hz),
        "command": {"twist_min": [float(v) for v in twist_min], "twist_max": [float(v) for v in twist_max]},
        "safety": {},
        "observation": "gyro(3) gravity(3) q-default_pose(14) dq(14) last_action(14) command(13): microduck_rl@0daab16 infer_policy.py:656-682",
    }


def verify_onnx(path: Path, joints) -> dict:
    """The contract duck-policy relies on; returns the ONNX metadata. Raises on a mismatch."""
    import onnxruntime as ort

    s = ort.InferenceSession(str(path))
    (i,), (o,) = s.get_inputs(), s.get_outputs()
    got = (i.name, list(i.shape), o.name, list(o.shape))
    if got != ("obs", [1, 61], "actions", [1, 14]):
        raise ValueError(f"{path.name}: io {got}, expected obs [1,61] -> actions [1,14]")
    meta = s.get_modelmeta().custom_metadata_map
    if meta.get("joint_names", "").split(",") != list(joints):
        raise ValueError(f"{path.name}: joint_names {meta.get('joint_names')} != the policy order")
    return meta


def scalar_metrics(log_dir: Path) -> dict:
    """The last value of every tensorboard scalar (tag names aren't hard-coded: they differ by task)."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out = {}
    for events in sorted(log_dir.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(events.parent), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            last = acc.Scalars(tag)[-1]
            out[tag] = {"value": float(last.value), "step": int(last.step)}
    return out


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")
```

  `write_package` is in Step 4's `train.py`, because it needs the live mjlab env config.

- [ ] **Step 4: Implement `duck_train/train.py`.** It follows `model/training.py`'s structure: chdir
  into `<out>/tmp`, run `select_gl()` before any mujoco import, train on local disk, write the package
  before uploads can stall, and copy logs to `<out>/tmp/logs`.

```python
"""Viam custom training script for the MicroDuck walk (Mjlab-Velocity-Flat-MicroDuck, pollen-robotics/microduck_rl
@0daab16: mjlab 1.3.0 + RSL-RL PPO on MuJoCo Warp, BAM XL330 actuators). Viam runs `python3 -m model.training`;
this sdist's model/training.py calls main() here.

Modes:
- train (default): train on one GPU, upload training videos as they're written, then write the package:
  policy.onnx (exported from the last checkpoint, normalizer included), config.json, checkpoint.pt,
  model_metrics.json.
- probe: a cheap first job; writes probe.json. Each check is independent and the job always exits 0. The
  checks: nvidia-smi, versions, a short training run (iterations, compile time, s/iteration), export and
  verify an ONNX, upload one video.

wandb is installed (mjlab imports it) but never used: logger=tensorboard and WANDB_MODE=disabled.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from model.harness import VideoUploader, check, device, select_gl, set_visible_devices, unquote

from .package import build_config, scalar_metrics, verify_onnx, write_json

logger = logging.getLogger("microduck-train")
TASK = "Mjlab-Velocity-Flat-MicroDuck"
TAGS = ["microduck"]
DEFAULT_POSE = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.4530]  # infer_policy.py:126-141
UPLOAD_JOIN_TIMEOUT = 60.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="model.training", allow_abbrev=False)
    p.add_argument("--model_output_directory", type=lambda s: Path(unquote(s)), required=True)
    for name in ("dataset_file", "binary_data_file", "tabular_data_file", "sequences_file"):
        p.add_argument(f"--{name}", type=unquote, default=None, help="ignored (placeholder dataset)")
    p.add_argument("--mode", type=unquote, choices=("train", "probe"), default="train")
    p.add_argument("--max_iterations", type=lambda s: int(unquote(s)), default=4000)
    p.add_argument("--num_envs", type=lambda s: int(unquote(s)), default=4096)
    p.add_argument("--seed", type=lambda s: int(unquote(s)), default=42)
    p.add_argument("--save_interval", type=lambda s: int(unquote(s)), default=250)
    p.add_argument("--video_every_iters", type=lambda s: int(unquote(s)), default=500)
    p.add_argument("--video_length", type=lambda s: int(unquote(s)), default=200)
    p.add_argument("--run_name", type=unquote, default=time.strftime("microduck-%Y%m%d-%H%M%S"))
    p.add_argument("--upload_part_id", type=unquote, default=None)
    p.add_argument("--upload_dataset_id", type=unquote, default=None)
    return p.parse_args(argv)


def _cfgs(args):
    import mjlab.tasks  # noqa: F401  (loads the mjlab.tasks entry points: registers microduck's tasks)
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # verified in mjlab 1.3.0 (return deep copies)

    env_cfg = load_env_cfg(TASK)
    env_cfg.scene.num_envs = args.num_envs
    agent = load_rl_cfg(TASK)
    agent.max_iterations = args.max_iterations
    agent.save_interval = args.save_interval
    agent.seed = args.seed
    agent.run_name = args.run_name
    agent.logger = "tensorboard"
    agent.upload_model = False
    return env_cfg, agent


def _last_checkpoint(log_dir: Path) -> Path:
    ckpts = sorted(log_dir.rglob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {log_dir}")
    return ckpts[-1]


def export_and_verify(ckpt: Path, onnx_out: Path) -> dict:
    from mjlab_microduck.export import ExportConfig, run_export  # verified at MD/.../export.py:90

    run_export(TASK, ExportConfig(checkpoint_file=str(ckpt), num_envs=1, onnx_file=str(onnx_out)))
    joints = json.loads(json.dumps(_policy_joints()))
    return verify_onnx(onnx_out, joints)


def _policy_joints() -> list[str]:
    return ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
            "neck_pitch", "head_pitch", "head_yaw", "head_roll",
            "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]


def write_package(out: Path, log_dir: Path, env_cfg, agent) -> None:
    import mujoco

    ckpt = _last_checkpoint(log_dir)
    meta = export_and_verify(ckpt, out / "policy.onnx")
    # The metadata keeps 3 decimals; write full precision (infer_policy.py:126-141 / the robot's STAND init state,
    # which the reference side uses) and check the metadata agrees within its rounding.
    default_pose = DEFAULT_POSE
    meta_pose = [float(v) for v in meta["default_joint_pos"].split(",")]
    if any(abs(a - b) > 6e-4 for a, b in zip(meta_pose, DEFAULT_POSE)):
        raise ValueError(f"ONNX default_joint_pos {meta_pose} != DEFAULT_POSE")
    # Joint limits: the training robot's compiled model (env_cfg.scene's robot entity spec). If that's awkward
    # to reach, compile microduck's robot_walk.xml directly (mjlab_microduck/robot/microduck/robot_walk.xml).
    xml = Path(__import__("mjlab_microduck").__file__).parent / "robot" / "microduck" / "robot_walk.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    joints = _policy_joints()
    limits = [list(model.joint(n).range) for n in joints]
    twist = env_cfg.commands["twist"].ranges
    config = build_config(
        joints=joints, default_pose=default_pose, action_scale=float(meta["action_scale"]), joint_limits=limits,
        twist_min=[twist.lin_vel_x[0], twist.lin_vel_y[0], twist.ang_vel_z[0]],
        twist_max=[twist.lin_vel_x[1], twist.lin_vel_y[1], twist.ang_vel_z[1]],
        control_hz=round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)),
        source={"task": TASK, "microduck_rl": "0daab160d3fb1c2cbaba16a4a19052f36709dd9e", "checkpoint": ckpt.name,
                "iterations": agent.max_iterations, "num_envs": env_cfg.scene.num_envs, "seed": agent.seed},
    )
    write_json(out / "config.json", config)
    shutil.copy(ckpt, out / "checkpoint.pt")
    write_json(out / "model_metrics.json", scalar_metrics(log_dir))
    logger.info("wrote %s from %s", sorted(p.name for p in out.iterdir()), ckpt.name)


def train(args, gl: dict) -> None:
    from mjlab.scripts.train import TrainConfig, run_train

    out = args.model_output_directory
    log_dir = Path(tempfile.mkdtemp(prefix="microduck-")) / "logs" / args.run_name
    env_cfg, agent = _cfgs(args)
    video = gl["backend"] is not None and args.video_every_iters > 0
    cfg = TrainConfig(env=env_cfg, agent=agent, video=video, video_length=args.video_length,
                      video_interval=max(1, args.video_every_iters) * agent.num_steps_per_env,
                      gpu_ids=None)  # mjlab 1.3.0's TrainConfig has no log_root: run_train writes into log_dir
    set_visible_devices(device().startswith("cuda"))
    uploader = VideoUploader(log_dir / "videos" / "train", agent.num_steps_per_env, tags=TAGS, run_name=args.run_name,
                             upload_part_id=args.upload_part_id, upload_dataset_id=args.upload_dataset_id)
    if video:
        uploader.folder.mkdir(parents=True, exist_ok=True)
        uploader.start()
    try:
        try:
            run_train(TASK, cfg, log_dir)
        except BaseException:
            try:
                write_package(out, log_dir, env_cfg, agent)
            except Exception:
                logger.error("no package after failed run:\n%s", traceback.format_exc())
            raise
        write_package(out, log_dir, env_cfg, agent)
    finally:
        uploader.stop.set()
        if video:
            uploader.join(timeout=UPLOAD_JOIN_TIMEOUT)
            uploader.sweep(final=True)
        try:
            shutil.copytree(log_dir.parent, out / "tmp" / "logs", dirs_exist_ok=True)
        except Exception:
            logger.error("copying logs failed:\n%s", traceback.format_exc())


def probe(args, gl: dict) -> None:
    out = args.model_output_directory
    results: dict = {"gl": gl}
    check(results, "nvidia_smi", lambda: {"out": subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=60).stdout[-3000:]}, out)

    def versions() -> dict:
        import mjlab, mujoco, mujoco_warp, torch, warp  # noqa: E401
        return {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_ok": torch.cuda.is_available(),
                "warp": warp.__version__, "mujoco": mujoco.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "?"),
                "mjlab": getattr(mjlab, "__version__", "?")}

    check(results, "versions", versions, out)
    log_dir = Path(tempfile.mkdtemp(prefix="microduck-probe-")) / "logs" / args.run_name

    def short_train() -> dict:
        from mjlab.scripts.train import TrainConfig, run_train

        env_cfg, agent = _cfgs(args)  # probe: pass max_iterations=20, save_interval=10
        set_visible_devices(device().startswith("cuda"))
        video = gl["backend"] is not None
        t0 = time.monotonic()
        run_train(TASK, TrainConfig(env=env_cfg, agent=agent, video=video, video_length=100,
                                    video_interval=10 * agent.num_steps_per_env, gpu_ids=None), log_dir)
        return {"iterations": agent.max_iterations, "num_envs": env_cfg.scene.num_envs,
                "wall_s": round(time.monotonic() - t0, 1), "checkpoints": sorted(p.name for p in log_dir.rglob("model_*.pt")),
                "metrics": scalar_metrics(log_dir)}

    check(results, "short_train", short_train, out)
    check(results, "export", lambda: {"meta": export_and_verify(_last_checkpoint(log_dir), out / "probe_policy.onnx")}, out)

    def upload() -> dict:
        vids = sorted((log_dir / "videos" / "train").glob("*.mp4"))
        if not vids:
            return {"skipped": "no video"}
        from model.viam_upload import upload_video
        return {"file_id": upload_video(str(vids[0]), args.upload_part_id, [*TAGS, args.run_name, "probe"], None)}

    check(results, "video_upload", upload, out)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_output_directory = args.model_output_directory.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("WANDB_MODE", "disabled")
    out = args.model_output_directory
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    os.chdir(out / "tmp")
    gl = select_gl()
    if args.mode == "probe":
        probe(args, gl)
    else:
        train(args, gl)
```

Notes for the implementer:
- **Names verified in plan review (inside the built image):** `load_env_cfg`, `load_rl_cfg`,
  `TrainConfig`'s fields (no `log_root`), `ExportConfig` and `run_export`, the twist ranges, and
  `sim.mujoco.timestep` and `decimation`. If anything still mismatches, fix it and note it in the
  commit. The sources are (wheel unpacked at
  `/private/tmp/claude-501/-Users-nick-hehr-src-viam-rslrl-policy-training/3856841f-0045-481c-bb70-ba5b784e7265/scratchpad/mjlab130/src/mjlab`;
  microduck at `${TMPDIR:-/tmp}/gh-clones-${CLAUDE_SESSION_ID}/microduck_rl`). Fix any mismatch and
  note it in the commit:
  - `load_env_cfg` / `load_rl_cfg` (module path);
  - `TrainConfig`'s fields (including `log_root`);
  - `ExportConfig` / `run_export`;
  - `env_cfg.commands["twist"].ranges` and its field names;
  - `env_cfg.sim.mujoco.timestep` and `env_cfg.decimation`.
- **Imports:** `duck_train.train` must import without mjlab installed, so the unit tests can run in
  the repo env. Keep every mjlab, microduck and mujoco import inside functions, as above.
- **This repo's env has a different mjlab** (@6ab9846), so nothing here can be integration-tested
  locally. The image's local smoke test (Task 4) and the probe job (Task 5) are the integration tests.

- [ ] **Step 5: Run the tests** with `uv run pytest tests/test_duck_train.py -q`. The velstand test
  runs if `dist/microduck-walk` exists.
- [ ] **Step 6: Commit** with message `feat: duck_train, Viam training entry and package writer for Mjlab-Velocity-Flat-MicroDuck`.

---

### Task 3: The duck sdist

**Files:**
- Create: `duck_train/setup.py`
- Create: `build_duck.sh`

- [ ] **Step 1: Write `duck_train/setup.py`** (copied into the stage as its `setup.py`):

```python
from setuptools import find_packages, setup

# Viam custom training script for the MicroDuck walk. Viam's launcher pip-installs this sdist and runs
# `python3 -m model.training`, which this package's model/training.py forwards to duck_train.train.main.
# Dependencies live in the image (container/Dockerfile.duck).
setup(name="microduck-train", version="0.1.0", packages=find_packages(include=["model", "duck_train"]), install_requires=[])
```

- [ ] **Step 2: Write `build_duck.sh`.**

```bash
#!/usr/bin/env bash
# Builds dist/microduck-train.tar.gz for `viam training-script upload`. Staged on its own: the repo's
# model/training.py is the peg-insert entry, so the stage gets a model/training.py shim to duck_train.
set -euo pipefail
cd "$(dirname "$0")"
stage=$(mktemp -d)
mkdir -p "$stage/model" "$stage/duck_train" dist
cp model/__init__.py model/harness.py model/viam_upload.py "$stage/model/"
cat > "$stage/model/training.py" <<'PY'
"""Viam runs `python3 -m model.training`: the MicroDuck job's entry is duck_train.train.main."""
from duck_train.train import main

if __name__ == "__main__":
    main()
PY
cp duck_train/__init__.py duck_train/train.py duck_train/package.py "$stage/duck_train/"
cp duck_train/setup.py "$stage/setup.py"
(cd "$stage" && COPYFILE_DISABLE=1 uvx --from build pyproject-build --sdist --outdir "$OLDPWD/dist" . >/dev/null)
mv dist/microduck_train-*.tar.gz dist/microduck-train.tar.gz
rm -rf "$stage"
tar -tzf dist/microduck-train.tar.gz
```

  If `model/__init__.py` doesn't exist in the repo, check how `model` is a package and adapt.
  `uvx` writes into `$OLDPWD/dist`; if `OLDPWD` is unreliable in the subshell, compute the absolute
  path first.

- [ ] **Step 3: Build and check.** Run `chmod +x build_duck.sh && ./build_duck.sh`. The listing
  must show `model/training.py`, `model/harness.py`, `model/viam_upload.py`, `duck_train/*.py` and
  `setup.py`, and nothing from `rl_insert/`.

  Add a test to `tests/test_duck_train.py`:
  1. Run `build_duck.sh` via `subprocess`, skipping if `uvx` is missing.
  2. Extract `microduck_train-0.1.0/model/training.py` from the tarball.
  3. Assert it imports `duck_train.train`.
- [ ] **Step 4: Commit** with message `build: microduck-train sdist (model.training shim -> duck_train)`.

---

### Task 4: The training image

**Files:**
- Create: `container/Dockerfile.duck`

- [ ] **Step 1: Write the Dockerfile.**

```dockerfile
# syntax=docker/dockerfile:1
# Training image for the microduck-train Viam custom training script (Mjlab-Velocity-Flat-MicroDuck).
# Viam (Vertex AI, linux/amd64, A100, driver 580/CUDA 13) ignores ENTRYPOINT/CMD but keeps ENV. Its launcher
# runs `pip install <script tar.gz>` then `python3 -m model.training "$@"`, so python3 and pip on PATH must
# already import every dependency.
#
# The environment is pollen-robotics/microduck_rl's own lock at a pinned commit (mjlab 1.3.0, torch 2.9.1 with
# CUDA 12.8 wheels, warp 1.12.0, BAM @62bd8ce), synced into /opt/venv, plus viam-sdk and pip.
#
# Build (from this repo's root):
#   docker build --platform linux/amd64 -f container/Dockerfile.duck -t microduck-train:dev .
FROM ubuntu:24.04

ARG MICRODUCK_RL_SHA=0daab160d3fb1c2cbaba16a4a19052f36709dd9e
ARG UV_VERSION=0.11.30

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PATH=/opt/venv/bin:/usr/local/nvidia/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/nvidia/lib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    NVIDIA_DISABLE_REQUIRE=1 \
    WARP_CACHE_PATH=/opt/warp-cache \
    MJLAB_WARP_QUIET=1 \
    WANDB_MODE=disabled \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_HTTP_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev ca-certificates curl ffmpeg git \
        libegl1 libgl1 libglx-mesa0 libosmesa6 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/share/glvnd/egl_vendor.d \
    && echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
    && mkdir -p /opt/warp-cache && chmod a+rwX /opt/warp-cache \
    && curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | env UV_INSTALL_DIR=/usr/local/bin sh

RUN git clone https://github.com/pollen-robotics/microduck_rl.git /opt/microduck_rl \
    && git -C /opt/microduck_rl checkout "${MICRODUCK_RL_SHA}" \
    && cd /opt/microduck_rl && uv sync --frozen --no-editable --python /usr/bin/python3 \
    && uv pip install --python /opt/venv/bin/python "viam-sdk>=0.85" tensorboard "setuptools>=70" wheel pip \
    && /opt/venv/bin/python -c "import torch, mjlab, mujoco_warp, onnxruntime, viam, bam, tensorboard, mjlab_microduck; print('torch', torch.__version__, torch.version.cuda)"

# As in Dockerfile.train: build the script sdist with the venv's setuptools (no PyPI fetch on billed time).
ENV PIP_NO_BUILD_ISOLATION=0
```

Notes:
- **If `uv pip install viam-sdk` moves protobuf, torch, mjlab or warp:** check the resolver output
  and pin them back, using `--constraint` with the versions from `uv.lock`. viam-sdk's protobuf
  range against the lock's protobuf 6.x is unverified.
- **If `uv sync --no-editable` fails on the project itself** (microduck_rl's uv_build backend): try
  `uv sync --frozen --no-install-project`, then
  `uv pip install --python /opt/venv/bin/python --no-deps /opt/microduck_rl`.
- **mjlab_microduck must import, and its `mjlab.tasks` entry point must be registered.** Check with
  `python -c "from importlib.metadata import entry_points; print(entry_points(group='mjlab.tasks'))"`.

- [ ] **Step 2: Build it** with
  `docker build --platform linux/amd64 -f container/Dockerfile.duck -t microduck-train:dev container/`.
  Plan review already built this Dockerfile verbatim: the lock installed first try, BAM built from git,
  protobuf stayed at 6.33.6 and it printed `torch 2.9.1+cu128 12.8`. If the local image
  `microduck-train:review` exists and the Dockerfile is unchanged, you may
  `docker tag microduck-train:review microduck-train:dev`, then `docker rmi microduck-train:review`.
  Expected: the final `RUN` prints `torch 2.9.1+cu128 12.8` or similar. Record the image size.

- [ ] **Step 3: Local smoke test, on CPU under emulation.** It is slow, so keep it tiny. It checks
  imports, argument parsing and that the task registry resolves, not training.

```bash
./build_duck.sh
docker run --rm --platform linux/amd64 -v "$PWD/dist:/dist" microduck-train:dev bash -c '
  pip install -q /dist/microduck-train.tar.gz &&
  python3 -c "import mjlab.tasks; from mjlab.tasks.registry import load_env_cfg, load_rl_cfg; \
    e = load_env_cfg(\"Mjlab-Velocity-Flat-MicroDuck\"); a = load_rl_cfg(\"Mjlab-Velocity-Flat-MicroDuck\"); \
    print(type(e).__name__, a.max_iterations, a.logger, e.commands[\"twist\"].ranges)" &&
  python3 -m model.training --help | head -5'
```

  Fix the registry module path to the real one found in Task 2. A short CPU training run under
  emulation is optional: try `--mode=probe --num_envs=8 --max_iterations=2` with a 20-minute
  timeout, and skip it if warp has no CPU path or it's too slow. The probe job is the real test.

- [ ] **Step 4: Commit** with message `build: microduck-train image (microduck_rl@0daab16 lock + viam-sdk)`.

---

### Task 5: Publish and run the probe job

- [ ] **Step 1: Push and register.**

```bash
docker tag microduck-train:dev docker.io/hipsterbrown/viam-microduck-train:0.1.0-1
docker push docker.io/hipsterbrown/viam-microduck-train:0.1.0-1
viam train containers register --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --uri=docker.io/hipsterbrown/viam-microduck-train:0.1.0-1 --description="microduck-train 0.1.0-1"
viam train containers list --org-id=16518049-9dd3-479f-9644-c5d112aa42d8   # note the container ID
```

- [ ] **Step 2: Upload the script** as version 1:

```bash
viam training-script upload --path=dist/microduck-train.tar.gz --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --script-name=microduck-train --version=1 --framework=onnx --type=robot_policy --visibility=private
```

- [ ] **Step 3: Submit the probe job.**

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=6581c38d39e1f5ebd5bc0e0f --model-name=microduck-probe --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:microduck-train --version=1 \
  --container-id=<container-id> \
  --args=mode=probe,num_envs=4096,max_iterations=20,save_interval=10,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=probe-1
```

- [ ] **Step 4: Read the results.**
  1. Poll `viam train get --job-id=<id>` every few minutes; read `viam train logs --job-id=<id> | tail -80`.
  2. Export the result with `viam packages export --org-id=… --name=microduck-probe --version=1
     --type=ml_model --destination=<scratch>`.
  3. Read `probe.json`. Record:
     - versions, and whether CUDA works;
     - `short_train.wall_s`, giving seconds per iteration, compile overhead included (the first
       iteration is slower; read per-iteration times from the logs if present);
     - whether the export check verified the layout;
     - the video file id.
  4. **If a check failed:** fix it. Upload a new script version (2, 3, …), or push a new image tag
     `0.1.0-2` if a dependency changed. Re-probe until every check passes.

- [ ] **Step 5: Record the probe** in the results doc (Task 9), and pick the iteration count for the
  real job. The aim is about 4000 iterations, which reaches the curricula's final stage (about
  iteration 2000) plus margin, budgeted at s/iteration × 4000. If that's over about 10 hours, ask the
  user before submitting.

---

### Task 6: The training job

- [ ] **Step 1: Submit.**

```bash
viam train submit custom from-registry --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 \
  --dataset-id=6581c38d39e1f5ebd5bc0e0f --model-name=microduck-velocity-train --model-version=1 \
  --script-name=16518049-9dd3-479f-9644-c5d112aa42d8:microduck-train --version=<probe-passing version> \
  --container-id=<container-id> \
  --args=num_envs=4096,max_iterations=4000,save_interval=250,video_every_iters=500,upload_part_id=cab989eb-90a0-4dce-8fc6-2c50d0ed1f10,run_name=velocity-1
```

- [ ] **Step 2: Monitor.** Poll `viam train get` at an interval matched to the job length (every
  20–30 minutes), and read the logs occasionally: mean reward rising, episode length rising, no NaN.
  Training videos appear in Viam Data under the rl-insert part, tagged `microduck` and `velocity-1`.
- [ ] **Step 3: Export the package** to `dist/microduck-velocity-train/` with `viam packages export`,
  and check:
  - `policy.onnx` passes `verify_onnx`;
  - `config.json` has `default_pose` within 6e-4 of DEFAULT_POSE (the STAND keyframe the policy was
    trained around);
  - `model_metrics.json` has the final reward.

---

### Task 7: The deployable package (safety from our reference)

**Files:**
- Modify: `scripts/duck_reference.py`, `scripts/duck_parity.py`, adding a `--script <json>` option
- Create: `scripts/duck_speed_sweep.py`

- [ ] **Step 1: `--script` option.** `duck_reference.py` and `duck_parity.py` take an optional
  `--script path.json`: a list of `[seconds, vx, vy, wz]`. It replaces the module's `SCRIPT`
  everywhere: `command_at`, `DURATION`, the segment bounds, and `ours`'s command schedule.
  - Implement it as `duck_reference.set_script(list)`. It mutates the module's `SCRIPT` in place
    (`SCRIPT[:] = ...`) and recomputes the module global `DURATION`.
  - **Switch `duck_parity.py` (and `tests/test_duck_parity.py`) to `import duck_reference as ref`**
    and use `ref.SCRIPT`, `ref.DURATION` and `ref.command_at` everywhere. Today it does
    `from duck_reference import DURATION, SCRIPT, command_at` (`duck_parity.py:34`): `DURATION` is
    copied by value, so a later `set_script` would leave it stale at lines 122 and 171. `SCRIPT` is
    used at 49, 78, 175 and 177.
  - `run()` saves `"script"` in its output JSON.
  - `report` calls `set_script` from the reference JSON's `"script"`, falling back to the default
    script when the key is missing (the existing `dist/duck-reference.json` has none), so the two
    sides can't drift.
  - Default behaviour is unchanged. Unit-test that `set_script` changes `command_at` and the segment
    bounds.

- [ ] **Step 2: Speed sweep.** Before choosing a parity script, write `scripts/duck_speed_sweep.py`.
  It's a loop of `duck_reference.set_script([(8.0, vx, 0.0, wz)])` and
  `duck_reference.run(checkout, duck_xml, onnx)`. Each case runs from the reset pose; report the
  mean body vx and wz from samples with t > 3 s. Don't refactor `run()`. Commands:
  - vx ∈ {0.1, 0.2, 0.3, 0.4, −0.2, −0.4}
  - wz ∈ {±0.3, ±0.6, ±1.0}, both in place and with vx 0.2

  Run it on our ONNX and on velstand, and record both tables in the results doc. Our velocity policy
  may not have velstand's dead zone, so choose the parity script from our policy's own sweep: one
  speed it walks forward at, one turn each way, a stop, and backward if it walks backward.

- [ ] **Step 3: Safety and package.**
  1. Run `duck_reference.py` with our ONNX, the chosen `--script` and `--stats` against a copy of
     the job's `config.json` in `dist/microduck-velocity/`, next to `policy.onnx` and a
     `LICENSE-POLICY.md`. The license note: trained by us from Pollen's Apache-2.0 recipe; the duck
     meshes are not included.
  2. Pack the directory (`fetch_walk_policy.pack`).
  3. Upload `microduck-velocity` 0.1.0:
     `viam packages upload --org-id=16518049-9dd3-479f-9644-c5d112aa42d8 --name=microduck-velocity --version=0.1.0 --type=ml_model --model-framework=onnx --model-type=robot_policy --path=dist/microduck-velocity.tar.gz`
     (`--model-type` is required for `ml_model`).
- [ ] **Step 4: Commit** the script changes and the sweep script.

---

### Task 8: Run our policy on the preview machine

- [ ] **Step 1: Point the machine at our policy.** In `docs/superpowers/specs/duck-sim-machine.json`,
  add the `microduck-velocity` package. Point `walk-model`'s `model_path` and `walker`'s
  `package_dir` at it, keeping `microduck-walk` in the packages list so switching back is one edit.
  Apply it with the MCP tools, then wait until ready.
- [ ] **Step 2: Smoke test.** Run `scripts/duck_smoke.py` with the vx our sweep says walks; make it
  a CLI argument with default 0.4. It must stand at zero and walk.
- [ ] **Step 3: Commit.**

---

### Task 9: Parity with our policy, and the results doc

- [ ] **Step 1: Run parity.**
  1. Reference: `duck_reference.py <ours onnx> --script <chosen>`.
  2. Ours: `duck_parity.py ours … --episodes 20 --script <chosen>`.
  3. `report`.
  4. Compare with milestone 2's velstand result. The same caveats apply: a basin of attraction at
     the edge of the command range, and the overrun rule.
- [ ] **Step 2: Write `docs/superpowers/specs/<date>-microduck-training-results.md`.** It covers:
  - the image and script versions, and the job ID;
  - the probe numbers;
  - iterations, wall time and the final reward and episode length (from `model_metrics.json`);
  - the speed sweeps (ours and velstand);
  - the parity table;
  - an overview frame mid-walk.
- [ ] **Step 3: Commit** the doc and the machine config.

---

## Done when

- The peg-insert tests still pass, and its sdist still builds.
- The probe job passes every check, and a 4000-iteration (or user-agreed) job finishes with a
  package whose ONNX passes `verify_onnx`.
- `microduck-velocity` 0.1.0 is in DevRelDemos. The preview machine walks with it.
- The parity report and the training results doc exist.
