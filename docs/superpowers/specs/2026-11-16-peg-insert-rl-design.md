# Peg-in-hole insertion with RSL-RL and MuJoCo Warp — design

Date: 2026-11-16

## Goal

Train an RL policy that inserts a gripped round peg into a fixtured hole with a real
UFactory xArm6 and two-finger gripper, driven through Viam. Training runs in MuJoCo Warp with
RSL-RL (via mjlab); deployment runs the exported policy through Viam's arm API.

## Decisions

| Topic | Decision |
|---|---|
| Target | Sim-to-real on an xArm6 with a two-finger gripper |
| Policy scope | Insertion only. Viam's motion service picks the peg and brings it to a pre-insert pose; the policy does the last centimeters |
| Hole pose | Fixtured at a measured frame first. The first model trains with hole-pose bias inside the chamfer's capture range (default ±2 mm, ±1°). A vision-width model (±5 mm, ±3°) is a later curriculum stage, attempted only once the narrow model works on the real arm |
| Force sensing | None. Proprioception only; commanded-vs-actual EE pose is the contact proxy |
| Fit | Moderate: 0.2–0.5 mm clearance, chamfered hole, round peg. No CAD yet, so geometry is parametric |
| Control | `MoveThroughJointPositionsStreamed` (Go: `viam-ufactory-xarm/arm/comm.go`, Python: `move_through_joint_positions_streamed` in viam-mujoco-sim) |
| Training stack | mjlab task (MuJoCo Warp + RSL-RL), packaged as a Viam custom training script and container |
| Inference | `viam:mlmodel:onnx-cpu` mlmodel service; the controller calls `Infer` |
| Artifact contract | `config.json` next to `policy.onnx` in the model package |

## Components

Repo layout follows `viam-anomalib-service` and `openvino-physical-ai-training`.

1. **`training/`** — custom training script and container.
   - mjlab task `PegInsert-xArm6`: scene, observations, rewards, events (randomization), curriculum, PPO config.
   - `model/training.py`: parses args and the task config, trains, exports `policy.onnx`
     (observation normalizer baked in), `config.json`, `model_metrics.json`.
   - `container/Dockerfile`: CUDA, mjlab, mujoco_warp, rsl_rl, pinned.
2. **`src/`** — Viam module, model `<namespace>:rl-insert:controller` (generic service, Python).
3. **`assets/`** — xArm6 + gripper model (built from UFactory's URDF) and the peg/hole generator.
   One source for training, the controller's IK/FK, and the viam-mujoco-sim test cell.
4. **`examples/`** — viam-mujoco-sim cell config (xArm6, gripper, peg prop, fixtured hole scene).

## Task config and `config.json`

Training input: a task config (geometry, randomization ranges, curriculum, PPO settings) with
defaults in the script; script args override it. Viam training jobs require a dataset, so jobs
pass a placeholder. Carrying the config in the dataset is a later option.

Training output `config.json` (schema-versioned) tells the controller everything beyond the ONNX graph:

- `model`: ONNX tensor contract for `onnx-cpu`: input and output names, dtype (float32) and shapes (`[1,N]` → `[1,6]`).
- `observation`: ordered terms with offset, dim, units and frame. Training is SI (m, rad);
  the controller converts from Viam's units (degrees for joints, mm and orientation vectors for poses).
- `action`: type (EE delta in hole frame), scale (e.g. 2 mm and 1° per unit), clip.
- `control`: policy rate, joint names/order, IK damping, robot model id + hash, nominal TCP offset
  (peg tip vs flange), gripper close command and force.
- `task`: peg/hole geometry, pre-insert offset, success depth, time limit.
- `robustness`: randomization ranges trained with (hole-pose noise etc.).

## Frames

- **Hole frame:** origin at the center of the hole mouth; +z is the outward surface normal, so
  insertion is −z. The `hole_frame` in Viam's frame system is defined the same way.
- **TCP:** the *nominal* peg tip relative to the flange, from `config.json`. Actor observations
  and deployment both use the nominal TCP. The true grasp offset (randomized) goes only to the critic.

## Training environment

**Episode start:** the peg is held 10–20 mm above the hole with random lateral offset and tilt.
Grasp offset is randomized (±3 mm, ±3°). The fingers hold the peg by friction (no weld), so slip
is possible. Finger force in sim is matched to the real gripper's grip force (randomized around it).

**Actor observations** (all available on the real arm):
- peg tip relative to the *estimated* hole frame (true pose + per-episode bias),
- EE orientation in the hole frame,
- joint positions and velocities,
- last action,
- commanded minus actual EE pose, defined as FK(last joint target) − FK(measured joints).
  Joint velocity is the finite difference of measured positions (the arm API has no velocity),
  computed the same way in training and deployment.

**Critic observations** (asymmetric, training only): adds the true hole pose, contact forces and
the true peg-in-hand pose.

**Action:** 6-D EE delta in the hole frame, at most 2 mm and 1° per step, at 20 Hz.
Damped least-squares differential IK produces joint targets, tracked by stiff joint PD tuned to
match the xArm. 0–2 steps of action latency are randomized.

**Reward:**
- tanh-shaped keypoint distance (peg vs hole),
- insertion-depth term,
- success bonus (depth reached and centered within clearance),
- penalties: action rate, joint velocity, contact force over a threshold.

**Termination:** success, dropped or slipped peg, excess force, or a 10 s timeout. There is no
force sensing on the real arm, so the excess-force threshold is set below the xArm's
collision-detection trip level. The policy learns to stay under it, and the real arm doesn't fault.

**Curriculum:** driven by success rate.
- Clearance: 1.0 mm → 0.2 mm.
- Start offsets and hole-pose bias widen up to the configured maximum (default ±2 mm / ±1°).

**Domain randomization:**
- friction (peg–hole, finger–peg),
- part dimensions within tolerance,
- joint PD gains,
- latency,
- observation noise,
- grasp offset.

**Geometry:** generated from diameter, clearance, chamfer and depth. Settled by prototyping (see the
foundations plan):
- The peg is a convex 32-sided mesh, not a cylinder primitive.
- The hole is a ring of convex prism meshes (wall plus chamfer pieces) that tile exactly, not
  overlapping boxes.
- Peg, hole and pad contacts use `solref=[0.002, 1]` at a 1 ms timestep.

## Deploy controller

**Config:**
- `arm`, `gripper`
- `mlmodel` (the onnx-cpu service)
- `config_path` (`${packages.ml_model.<name>}/config.json`)
- `hole_frame` (frame-system name)
- optional `max_step_mm`, `timeout_s`

**Kinematics:** the controller ships the same xArm6 model from `assets/`. At startup it checks the
model's hash against `config.json` and refuses to run on a mismatch.

**DoCommands:**
- `pre_insert_pose`: returns the flange pose in the world frame to hand to the motion service,
  computed from `hole_frame`, the pre-insert offset and the nominal TCP. This is the handoff from the
  scripted approach to the policy.
- `start`:
  - The gripper must be closed on the peg.
  - It rejects start states outside the trained start distribution (offset and tilt from the
    pre-insert pose).
- `stop`
- `status`: `idle | running | succeeded | failed`, with reason and last observation.

After success the controller holds position and reports `succeeded`. Releasing the gripper and
retreating are left to the caller.

**Loop (20 Hz):**
1. Read joints; compute EE pose by FK. Read the hole pose once at start.
2. Build the observation per `config.json`; call `mlmodel.Infer`.
3. Scale and clip the action; run differential IK to get a joint target.
4. Safety clip: joint limits, workspace box around the hole, max step per tick, max joint jump.
   The xArm streaming path has no limit check (RSDK-14284), so this clip is the only guard.
5. Stream to one `MoveThroughJointPositionsStreamed` call:
   - The first point is the current joints at t=0, because the xArm skips that point.
   - Each 50 ms step is upsampled to about 5 points at 10 ms, so servo mode gets smooth setpoints.
   - Point times are absolute from the stream anchor (the xArm sends each point at anchor + `Time`,
     and sends late points immediately). Tick k stamps its points in the window
     [(k+1)·50 ms, (k+2)·50 ms), one tick ahead. That is a constant one step of latency, inside the
     trained 0–2 range.
   - If a tick's inference is late enough that its window has started, the controller skips ahead
     and records an overrun. Too many overruns is a failure.

**Stop on:** success, timeout, tracking error over a threshold (jam), stream loss, inference
error, too many overruns, or `stop`.
- On success: half-close the stream (both modules then wait for motion to settle).
- On any failure: cancel the stream and call `arm.Stop`. A half-close would let queued points play
  out. Record the reason.

## Infrastructure (updated 2026-11-17)

**Training runs as Viam custom training jobs.**
- Hardware: Vertex AI, 2× A100 40GB; the script uses one GPU.
- Job contract (from viam-app):
  - Vertex ignores the image's entry point. It `pip install`s the uploaded script package and runs `python3 -m model.training` with `--model_output_directory` and the dataset args.
  - Custom args arrive as `--key=value`.
  - The env holds `API_KEY` and `API_KEY_ID`: an org-scoped key with the submitter's permissions, deleted when the job ends.
  - Everything in the output dir except `tmp/` becomes the model package. `model_metrics.json` becomes package metadata where the org has the feature flag.
- Packaging:
  - The container holds dependencies only: torch (CUDA), mjlab, viam-sdk, ffmpeg, and EGL and OSMesa libraries.
  - The script package holds `model/training.py`, `rl_insert/` and its assets.
  - So the task can change without rebuilding the container.
- mjlab is pinned to a `main` commit that uses MuJoCo 4.0.1, the version plan 1 validated. The PyPI 1.6.0 release pins MuJoCo 3.11.

**Sim sandbox (plan 1's GPU probe).** It gave an NVIDIA L4, not an L40S, and has no Docker. It's fine for quick GPU checks with `scripts/probe_sim_sandbox.sh`; it isn't the training target.

## Training videos to Viam Data

The training script records sim videos at an interval and uploads each one to Viam Data, so runs can be reviewed later.
- **Recording:**
  - mjlab's `VideoRecorder` records env 0 every `video_every_iters` PPO iterations (default 250) for `video_length` steps.
  - Frames come from `mujoco.Renderer`. The script tries EGL first and falls back to OSMesa (CPU software rendering), because EGL on Vertex nodes is unconfirmed.
- **Upload:**
  - A background thread uploads each finished MP4 with `DataClient.file_upload_from_path` and the job's API key. A last pass runs after training.
  - Uploads need a machine part. Videos go to the `rl-insert` machine in DevRelDemos, passed as `--upload_part_id`, so jobs must run in the DevRelDemos org.
  - Tags: `rl-insert`, the run name, and `iter-<n>`. An optional `--upload_dataset_id` adds each video to a dataset.
- **Missing credentials or part ID:** the script logs a warning and keeps videos local. It never fails the job.
- **Model package:** videos aren't included.

## Risks and spikes (do first)

1. **Docker probe on the cloud sim machine:**
   - `nvidia-smi`, then `docker run --gpus all`, then a mujoco_warp smoke test in the container.
   - Fallback: the same pinned deps in a `uv` venv on the host.
2. **Contact spike:** an arm-driven peg pressed into the mesh hole at 1.0, 0.5 and 0.2 mm
   clearance, batched on MuJoCo Warp and compared with CPU MuJoCo. Pass: no blow-ups, insertion
   rates within 10 points, contact buffer under 90% full.
3. **xArm6 model:** URDF → MJCF/URDF usable by both mjlab and viam-mujoco-sim; gripper model match.

## Testing

1. **Unit:**
   - the observation builder matches the `config.json` layout,
   - differential IK reaches small targets on the xArm6 model,
   - the safety clip rejects out-of-bounds targets.
2. **Training:** the spikes above, then success rate per clearance stage on the L40S as the gate.
3. **Sim-to-sim:** viam-mujoco-sim cell. The motion service makes the approach, then the controller
   runs `start` with the same config as the real cell. Measure success rate over N randomized hole
   poses. This catches unit, frame, TCP-offset and timing mismatches. It also logs stream overruns
   and checks that points reach the arm evenly paced, not in bursts.
4. **Real xArm6:** loose clearance and reduced `max_step_mm` first, with a person on the e-stop;
   then tighten.

## Out of scope (for now)

Vision hole estimation (and the vision-width noise stage), the pick itself, F/T sensing,
multi-GPU training, task config carried in a dataset.
