# A reusable RL pipeline for Viam developers: design

Date: 2026-11-17

## Goal

This repo trains a peg-in-hole insertion policy for an xArm6 with a UFactory G2 gripper, using mjlab
(MuJoCo Warp and RSL-RL PPO) as Viam custom training jobs. This doc proposes how to turn it into
something Viam developers can reuse and be taught: train a policy in simulation, package it, and run
it on a machine through Viam, for a task that isn't peg insertion.

Audience: Viam DevRel and engineering. This is a proposal. Nothing below the "Layers" section is built
yet, except where it says so.

## Summary

- About 760 lines of Python in this repo (about 840 with the training Dockerfile and `build.sh`) form a
  training harness that is mostly task-agnostic. About 1000 lines are the task itself.
- Most of the GPU hours lost on this task were not RL problems. They were packaging, geometry, solver
  and measurement bugs that a cheap preview would have caught. The proposal is a `--mode=preflight`
  job and an interactive scene on a Viam machine.
- Deployment needs one generic policy controller module driven by `config.json`, not one per task. That
  only works if training and the controller share a vocabulary of observation terms and action types.
- Order: build the controller and the sim-to-sim test for this task first, then extract the harness
  once a second, simpler task (reach) exists to prove it.

## Layers

Line counts are from `wc -l` at commit `b0f795d`.

| Layer | Files | Lines | Reuse |
|---|---|---|---|
| Task | `rl_insert/robot.py`, `peg_hole.py`, `scene.py`, `task/env_cfg.py`, `task/mdp.py` | 1007 (87, 132, 52, 404, 332) | Mostly specific to peg insertion. `robot.py` (xArm6 and G2) is reusable for any task on that arm. |
| Training harness | `model/training.py`, `task/runner.py`, `model/package.py`, `task/viam_upload.py`, `container/Dockerfile.train`, `build.sh` | 758 Python (394, 126, 186, 52), plus 63 and 13 | Almost entirely task-agnostic, with the exceptions below. |
| Deployment contract | `policy.onnx` and `config.json` | n/a | The format is generic. The contents are per task. |

### Task layer

The task is defined by an env config factory (`peg_insert_env_cfg(p)` in `task/env_cfg.py`) and a PPO
config factory (`peg_insert_ppo_cfg()` in `task/runner.py`), registered under `TASK_ID`
(`rl_insert/task/__init__.py`). `TaskParams` carries the knobs that change per run. `mdp.py` holds the
observation terms, events, terminations and the `InsertState` that the curriculum reads.

Reusable on its own: `robot.py` defines the xArm6 and G2 (URDF, joint servo gains, gripper
parameters). A reach task on the same arm would take it unchanged.

### Training harness

What the harness already does, with no peg-specific code:

| Concern | Where |
|---|---|
| Viam CLI contract: `--model_output_directory`, the placeholder dataset arguments, `--key=value` custom args with stray quotes stripped, strict parsing that exits 2 on a misspelled arg | `training.py` `parse_args`, `unquote` |
| GL backend choice (cgl on macOS; EGL, then OSMesa on Linux) before mujoco imports | `select_gl` |
| Device selection and the `CUDA_VISIBLE_DEVICES` convention `run_train` expects | `_device`, `set_visible_devices` |
| Video upload to Viam Data from a background thread, with failures logged and never raised | `VideoUploader`, `viam_upload.py` |
| Probe mode: nvidia-smi, CUDA, grasp hold, throughput, a video upload; each check independent, always exit 0 | `probe`, `_check` |
| ONNX export on every save, best-checkpoint pick (success-rate EMA at full difficulty only), fail-fast | `PegInsertRunner` |
| Package writing: `policy.onnx`, `checkpoint.pt`, `config.json`, `model_metrics.json`, written before uploads can stall, and also after a failed run | `write_package`, `package.py` |
| Image and script split | `Dockerfile.train`, `build.sh` |

The harness's inputs are a task ID, an env config factory, a PPO config and a params object.

Not yet task-agnostic. These are what the extraction has to change:

- `train()` in `training.py` imports `TaskParams`, `peg_insert_env_cfg`, `peg_insert_ppo_cfg` and
  `PegInsertRunner` directly, and takes `--clearance_mm`. The default `--run_name` is
  `peg-insert-<timestamp>`.
- `PegInsertRunner` is generic in its save, best and fail-fast logic, but its `_watch` reads
  `env.unwrapped.insert_state.difficulty` and its `get_export_metadata` hard-codes
  `action_type = "ee_delta_hole_frame"` and the `ee_delta` action term. Both need to come from the task.
  The `success`, `excess_force` and `time_out` termination names it relies on are a convention to
  document.
- `package.py` `build_config` is the main exception. Its `observation`, `model` and `control` sections
  are driven by the ONNX metadata and the env config, but its `robot`, `task`, `robustness` and
  `grasp_model` sections, and the `action` frame text, are peg-specific. `METRIC_TAGS` also names
  `peg_slipped` and `gripper_hit`.
- `task/checks.py` (71 lines) is the probe's task checks. `make_env` is generic; `grasp_hold_slip` and
  `tip_along_tcp_z` are not.

`container/Dockerfile` (14 lines) is not part of the harness. It is the earlier spike image (contact
spike, `uv`); the training image is `Dockerfile.train`.

### The pattern to teach

The dependencies image and the code are separate.

- `container/Dockerfile.train` holds only dependencies: torch (CUDA 13 wheels), warp, a pinned mjlab
  commit, viam-sdk, onnxruntime, tensorboard, ffmpeg, EGL and OSMesa. It is built and pushed once.
- `build.sh` builds `dist/rl-insert-train.tar.gz`, an sdist of `model/` and `rl_insert/` with no
  dependencies. `viam training-script upload` sends it. Viam's launcher `pip install`s it and runs
  `python3 -m model.training`.
- A code change (task, rewards, harness) is a new sdist upload, not an image rebuild. Image rebuilds
  happen only when a dependency changes. The Dockerfile also turns off pip build isolation so a PyPI
  outage can't fail a billed job.

## Lessons learned

Almost none of these were RL problems, and all of them cost GPU hours or invalidated a run. The
check column is what would have caught each one before training started.

| Symptom | Cause | Check that would have caught it |
|---|---|---|
| The job failed at start | The G2 meshes were missing from the installed sdist. `package_data` listed folders and dropped `assets/g2` (`b686c5f`, `setup.py` now uses `assets/**/*`). | Build the env from the installed sdist, not from the source tree. |
| Run 1's policy couldn't work on the real G2 | The sim finger pads were 18.75 mm higher than the real CAD: they ended at the TCP instead of being centred on it. The real fingertips would have landed on the fixture (`first-training-run.md`, fixed in `e1edba4`). | A CAD comparison, and a camera frame render of the gripper against the hole. |
| In videos the gripper looked like it was floating | The gripper was drawn without meshes (fixed by `64696e0`, `448bc2a`). | A rendered frame from the training camera. |
| Contact forces read low | `mujoco_warp`'s `contact_sensor_maxmatch` defaults to 64. 61 peg-hole matches were seen at 0.35 mm clearance, so contacts were dropped near the limit. Raised to 256 (`753e391`). | Measure peak contact counts against the caps. |
| About 14% slip early in training | The grasp weld was soft in rotation: `torquescale` 1 with a 20 g peg let pad torque swing the tip (p95 7 mm under random actions; 0.23 mm at `torquescale` 10). Fixed in `206184b`. | A random-policy termination breakdown. |
| The slip check ate the slip budget | Slip was measured against the nominal grasp, not the sampled one, so grasp randomisation (up to about 6 mm) used most of the 10 mm budget (`206184b`). | The same breakdown. |
| Early training rammed the peg instead of aiming | A "ram it" local optimum: excess-force terminations ended episodes faster than timeouts did. Runs recovered on their own, but a run that didn't would burn its whole budget. | A scripted-oracle feasibility check, and fail-fast (`--fail_fast_iters`, default 300). |

Notes on the last row. The runner docstring records that runs 1 and 3 first had a successful episode
at about iteration 24 and 40. Fail-fast raises if none has by iteration 300, and still writes the
package. The oracle check doesn't exist yet; it is one of the preflight checks below.

What these have in common: each was visible in a frame, a count or a single rollout, in seconds to
minutes, on a laptop. None needed a trained policy.

## Preview before training (proposal)

### (a) `--mode=preflight`

A new mode next to `train` and `probe` in `model.training`. It runs locally, in CI, or as a cheap Viam
job (the probe job is the precedent: see `2026-11-17-vertex-probe-results.md`). Like probe, every check
is independent and the job writes a report, here with images, to the output directory.

| Check | Catches | Status |
|---|---|---|
| Build the env from the installed sdist | Missing package assets | New. `make_env` exists; the install step doesn't. |
| Render a frame from the training camera | Floating or missing geometry, bad framing | New. The probe's `video` check renders from env 0 already. |
| Grasp hold | Soft or slipping grasp | Exists: `grasp_hold_slip` in `checks.py`, run by probe. |
| Peak contacts against the Warp caps | Silent contact dropping | New. |
| Scripted oracle that must succeed under the force limit | A task that is infeasible, or feasible only by ramming | New. |
| Random-policy termination breakdown | Terminations that fire for the wrong reason (slip, hit) | New. |

The oracle is the check that tells a developer the task can be solved at all, before PPO spends hours
finding out. Its pass condition is success with peak force under `force_limit`.

A laptop on CPU Warp is enough to run all of these, so preflight doesn't depend on a GPU.

### (b) An interactive scene on a Viam machine

A MuJoCo-backed module exposes arm, gripper and camera components built from the same model the
training uses. A developer can:

- see the scene in the Viam app (3D tab, camera) with no local setup;
- drive the arm through the Viam API;
- later run the trained policy against it.

It doubles as the sim-to-sim deployment test (see below). `viam-mujoco-sim` is already referenced in
the first design spec as the test cell. Work on configuring the `rl-insert` machine as this
interactive scene is in progress, in parallel with this doc.

Because (a) already covers the local preview, (b) is mainly for teaching and for deployment
validation, not for catching training bugs.

## General deployment design

Pipeline:

```
training job -> ML model package in the registry
             -> on the machine: mlmodel service (viam:mlmodel:onnx-cpu)
                              + policy controller service (generic)
```

The package is `policy.onnx` and `config.json` (schema version 1). `checkpoint.pt` and
`model_metrics.json` ride along but aren't needed to run.

### Policy controller

One generic service module. It:

1. reads `config.json` from the package;
2. builds observations from components, per `observation.terms`;
3. runs inference at `control.policy_dt` through the mlmodel service;
4. maps the action by `action.type` to the arm API, using `MoveThroughJointPositionsStreamed`;
5. enforces safety: an action clip, a per-step limit, a workspace box, and stop on IK error;
6. exposes `DoCommand` with `start`, `stop` and `status`.

The first design spec (`2026-11-16-peg-insert-rl-design.md`, "Deploy controller") already details the
streaming scheme, the stop conditions and the peg-specific `pre_insert_pose` command. That spec is
for this task; the change here is to make everything but the task-specific commands generic.

### The key design decision: a shared vocabulary

Training and the controller must share a vocabulary of observation terms and action types:

- Each observation term has a name, units, frame and a recipe the controller can compute from
  components (joint positions from the arm, TCP pose by FK, last action, and so on). Today this lives
  in `OBS_DOC` in `env_cfg.py` and is copied into `config.json`: `tip_rel_est_hole`, `ee_rot6d`,
  `joint_pos`, `joint_vel` and the rest of the actor group.
- Each action type has a name and a mapping to the arm API. Today there is one:
  `ee_delta_hole_frame`. A reach task would add `ee_delta_base_frame` or `joint_delta`.

When a new task uses only terms and action types already in the vocabulary, it needs no new
controller code. When it needs a new one, that is added on both sides together. This is what makes the
controller reusable, and it is also the main place where generalizing from one task would go wrong
(see order of work).

### The hard part: reproducing the sim's IK

The policy outputs a pose delta, not joint targets. In sim, damped least squares IK is solved every
1 ms physics step toward the desired pose, tracked by a joint PD servo. A real deployment streams
joint targets at the policy rate. The controller therefore has to reproduce the IK closely enough that
the arm follows the same path. `config.json` documents it for this reason: `control.ik` has the method,
damping, `max_dq_rad` and the position, orientation, joint-limit and posture weights, and
`control.joint_servo_sim` has the servo gains. `robustness.not_modelled` lists "joint-target streaming:
sim re-solves IK toward the desired pose every 1 ms" as a known gap. How close the controller can get
is an open question below.

### Safety

The first training run found that an all-zero observation, which is out of distribution, gives actions
up to about 18.7 (37 mm and 18.7 degrees per step after scaling). `config.json` currently has
`action.clip: null`. The controller must therefore clip actions and cap per-step motion itself and not
trust the policy output. The workspace box and stop on IK error are the second line.

### Validation ladder

Each rung is run before the next:

1. Sim: the training env (success rate per clearance stage).
2. Sim-to-sim: the controller drives the interactive scene on a Viam machine through the Viam API.
   This catches unit, frame, TCP-offset and timing mismatches without hardware.
3. Real arm at reduced speed, inside a workspace box, with a person on the e-stop.
4. Real arm at full speed.

## Teaching plan

### A template

The harness as a library, plus a task folder with four files to fill in and three commands.

| File | Holds |
|---|---|
| robot | The arm and gripper, including joint servo gains and the TCP. Reuse `robot.py` for the xArm6. |
| scene | Objects and fixtures, and the training camera. |
| task | Observations, actions, rewards, terminations, events, curriculum (the env config factory). |
| params | The params object and the PPO config. |

| Command | Does |
|---|---|
| preview | Open the scene locally or on a machine, and render a frame. |
| preflight | Run the checks above. |
| train | Upload the script and submit the job. |

### Two tutorials

1. **Reach task on the xArm6.** No contact, minutes of training. It covers the full loop: train,
   package, deploy and sim-to-sim. It is the proof that the template and the controller are task
   agnostic.
2. **Peg insertion, the advanced one.** Built around the lessons table: each row becomes a step where
   a developer finds the bug with a preflight check rather than after a run.

Reward and curriculum design stay as examples, in the peg task and the reach task. They are not a
framework.

## Order of work

1. **Controller module and sim-to-sim for this task** (plans 3 and 4; the controller and the sim-to-sim
   test are specified in the first design spec, and neither has a plan file yet). Keep the observation
   and action vocabularies generic from the start, even though only one task uses them.
2. **Extract the harness** once a second task, the reach tutorial, exists to prove it. Generalizing
   from one example bakes in its quirks, for example hole-frame actions and the `insert_state`
   curriculum hook. Preflight is built in this step, since its checks need the same task-supplied
   hooks.
3. **Template and tutorials.**

### Non-goals

- A reward framework.
- Support for multiple simulators. It is mjlab on MuJoCo Warp.

### Open questions

- Does a MuJoCo-backed Viam module already exist in the registry (`viam-mujoco-sim` is referenced in
  the first design spec), or do we build one? Whether it can expose a 3D view and a camera from our
  model decides how much of (b) is configuration and how much is code.
- How far can the controller's IK match the sim's? The sim solves every 1 ms and the real arm takes
  streamed joint targets at 20 Hz upsampled to 10 ms. If the paths differ materially, the controller
  needs a finer inner loop, or training needs a model of the streaming.
- Where does the shared vocabulary live? The leading option is a small pip package that both the
  training script and the controller depend on. It would have to be installable in the training image
  and in the module's environment.
