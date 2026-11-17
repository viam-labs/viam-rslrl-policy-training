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
| Hole pose | Fixtured at a measured frame first; training noise is wide enough (±5 mm, ±3°) for a later vision estimate |
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

Training input: a task config (geometry, randomization ranges, curriculum, PPO settings), with
script args overriding it. Viam training jobs require a dataset; if a dataset can carry a
`task.json` file (spike), the config comes from it, otherwise from args with a placeholder dataset.

Training output `config.json` (schema-versioned) tells the controller everything beyond the ONNX graph:

- `observation`: ordered terms with offset, dim, units and frame.
- `action`: type (EE delta in hole frame), scale (e.g. 2 mm and 1° per unit), clip.
- `control`: policy rate, joint names/order, IK damping, robot model id + hash, TCP offset (peg tip vs flange).
- `task`: peg/hole geometry, pre-insert offset, success depth, time limit.
- `robustness`: randomization ranges trained with (hole-pose noise etc.).
- `metrics`: success rate per curriculum clearance (also in `model_metrics.json`).

## Training environment

**Episode start:** the peg is held 10–20 mm above the hole with random lateral offset and tilt.
Grasp offset is randomized (±3 mm, ±3°). The fingers hold the peg by friction (no weld), so slip is possible.

**Actor observations** (all available on the real arm):
- peg tip relative to the *estimated* hole frame (true pose + per-episode bias),
- EE orientation in the hole frame,
- joint positions and velocities,
- last action,
- commanded minus actual EE pose.

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

**Termination:** success, dropped or slipped peg, excess force, or a 10 s timeout.

**Curriculum:** driven by success rate.
- Clearance: 1.0 mm → 0.2 mm.
- Start offsets and hole-pose bias widen up to ±5 mm / ±3°.

**Domain randomization:**
- friction (peg–hole, finger–peg),
- part dimensions within tolerance,
- joint PD gains,
- latency,
- observation noise,
- grasp offset.

**Geometry:** the peg is a cylinder; the hole is a chamfered ring of 24–32 boxes, generated from
diameter, clearance, chamfer and depth.

## Deploy controller

**Config:**
- `arm`
- `mlmodel` (the onnx-cpu service)
- `config_path` (`${packages.ml_model.<name>}/config.json`)
- `hole_frame` (frame-system name; later a vision service and camera)
- optional `max_step_mm`, `timeout_s`

**DoCommands:**
- `start`: the peg must be gripped and at the pre-insert pose.
- `stop`
- `status`: `idle | running | succeeded | failed`, with reason and last observation.

**Loop (20 Hz):**
1. Read joints; compute EE pose by FK. Read the hole pose (once now, per step with vision).
2. Build the observation per `config.json`; call `mlmodel.Infer`.
3. Scale and clip the action; run differential IK to get a joint target.
4. Safety clip: joint limits, workspace box around the hole, max step per tick, max joint jump.
   The xArm streaming path has no limit check (RSDK-14284), so this clip is the only guard.
5. Stream to one `MoveThroughJointPositionsStreamed` call:
   - The first point is the current joints at t=0, because the xArm skips that point.
   - Each 50 ms step is upsampled to about 5 points at 10 ms, so servo mode gets smooth setpoints.

**Stop on:** success, timeout, tracking error over a threshold (jam), stream loss, inference
error, or `stop`. Every failure ends the stream (which halts the arm on both modules) and records
the reason.

## Infrastructure

- Iterate on the cloud sim machine (L40S, believed 48 GB) with `docker run --gpus all` of the
  training container.
- Viam training jobs run on 2× A100 40GB; start single-GPU and add `torchrun` only if needed
  (verify mjlab multi-GPU support first).

## Risks and spikes (do first)

1. **Docker probe on the cloud sim machine:**
   - `nvidia-smi`, then `docker run --gpus all`, then a mujoco_warp smoke test in the container.
   - Fallback: the same pinned deps in a `uv` venv on the host.
2. **Contact spike:** a peg at 0.2 mm clearance in the box-ring hole under MuJoCo Warp. Check for
   no wall penetration and no false jamming, and confirm Warp supports cylinder–box collisions.
3. **Dataset-as-config:** can a Viam dataset carry a `task.json` that the training script receives?
4. **xArm6 model:** URDF → MJCF/URDF usable by both mjlab and viam-mujoco-sim; gripper model match.

## Testing

1. **Unit:**
   - the observation builder matches the `config.json` layout,
   - differential IK reaches small targets on the xArm6 model,
   - the safety clip rejects out-of-bounds targets.
2. **Training:** the spikes above, then success rate per clearance stage on the L40S as the gate.
3. **Sim-to-sim:** viam-mujoco-sim cell. The motion service makes the approach, then the controller
   runs `start` with the same config as the real cell. Measure success rate over N randomized hole
   poses. This catches unit, frame, TCP-offset and timing mismatches.
4. **Real xArm6:** loose clearance and reduced `max_step_mm` first, with a person on the e-stop;
   then tighten.

## Out of scope (for now)

Vision hole estimation, the pick itself, F/T sensing, multi-GPU training.
