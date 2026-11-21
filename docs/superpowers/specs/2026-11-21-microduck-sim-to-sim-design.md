# MicroDuck sim-to-sim on Viam: design

Date: 2026-11-21

## Goal

The office has a Pollen Robotics MicroDuck on pre-order, with deliveries expected before Christmas 2026.
Before it arrives, prove the deployment path in simulation:

- a floating-base duck, with BAM actuator physics, in `viam-mujoco-sim`;
- a Python Viam module that runs the 50 Hz policy loop against it;
- a Viam training job that produces policies for that loop.

The same module later runs on the robot, with a hardware IO backend in place of the sim one. That
backend is out of scope here.

## Context

- **Robot.** 14 actuated joints: 5 per leg, plus neck pitch, head pitch, head yaw and head roll. The
  mouth is a 15th servo that no walking policy drives. The servos are Dynamixel XL330s on one 1 Mbps
  bus. The IMU board sits on the same bus as ID 200. The computer is a Radxa Zero 3 (RK3566, arm64),
  and the robot is about 800 g.
- **Pollen's training.** `pollen-robotics/microduck_rl` (Apache-2.0 code, CC BY-NC-SA 4.0 meshes)
  trains on mjlab 1.3.0, MuJoCo Warp and RSL-RL. It uses the BAM M6 model of the XL330, fitted to the
  real servos with firmware kp 200, and battery voltage randomized over 6.5–8.2 V. Physics runs at
  5 ms with decimation 4, so the policy runs at 50 Hz. Every task shares one ONNX contract,
  `[1,61] -> [1,14]`, with the observation normalizer built into the graph.
- **Pollen's runtime.** The `robotd` daemon (Rust) runs the 50 Hz loop on the robot. Its `duck-body`
  command serves a MuJoCo body to `robotd --sim`, but with the MJCF's PD actuators, not BAM.
  `scripts/infer_policy.py` runs the ONNX policies in CPU MuJoCo with BAM
  (`bam.mujoco.MujocoController`). This design uses it as the reference.
- **viam-mujoco-sim today.** `robot_file.py` rejects free joints ("a floating base isn't supported
  yet"). `bases.py` removes a chassis's free joint. `SimManager.step_once` calls `step_callbacks`
  before every `mj_step`, which is where BAM attaches. The physics step defaults to 2 ms.
- **This repo.** The rl-insert pipeline (Viam training jobs, the `devrel:rl-policy` controller, the
  sim-to-sim results) is the precedent. Its lessons apply: bound every robot call, clip the policy's
  output, check with a reference before a long run.

## Decisions

| Question | Decision | Why |
|---|---|---|
| How the loop does IO | The module owns the loop behind a `DuckIO` interface. `SimIO` now, `BusIO` (Dynamixel bus in-process) later. | One atomic `sync_read` per tick on hardware. Pollen's `RobotIo` trait validates this split. |
| Sim API | The floating-base `robot` model, with read-only arms for the limbs, and one `io` DoCommand per tick that also carries gyro and gravity. No IMU component in milestone 1: `io` provides gyro and gravity, and `sensors.py` mounts on a parent's static Viam frame, not on the moving trunk. | A single snapshot from one sim instant. Frames and data capture keep working. |
| Where floating base and BAM live | Generalize mujoco-sim's existing `robot` model (upstream PR to `viam-labs/viam-mujoco-sim`) | A future Go1, G1 or Open Duck reuses it. A separate BAM module would cross processes every physics step. |
| Inference | `viam:mlmodel:onnx-cpu`, like rl-insert | The Viam-standard path. The extra gRPC hop is measured by the timing bar. |
| First policy | Pollen's published walk ONNX. Our trained policy is milestone 3. | Separates "our sim or loop is wrong" from "our training is wrong". |
| Success bar | Parity with `infer_policy.py` on falls and velocity tracking, plus loop timing | Catches subtle mismatches (IMU frame, gravity sign, joint order) that "it walks" doesn't. |
| Module language | Python | Reuses the `devrel:rl-policy` patterns. Pollen's `rustypot` gives Python a fast `sync_read` later. |

## Architecture

```
Viam machine (cloud simulation, like rl-insert)
  viam-mujoco-sim
    world      physics_dt 0.005
    robot duck floating base; actuators: bam xl330; DoCommand io
      arms     left-leg (5 joints), right-leg (5), head (4): read-only, frames, data capture
  viam:mlmodel:onnx-cpu    the walk policy, [1,61] -> [1,14]
  duck-policy (new module)
    50 Hz loop: io snapshot -> obs(61) -> Infer -> 14 targets -> next io
    DuckIO: SimIO (the robot's io DoCommand); BusIO later
    DoCommand: start, stop, status, set_command
```

## Units

### 1. Floating base in the `robot` model (mujoco-sim)

- A robot file may have one free joint, and only on its root body (the trunk).
- The robot's frame parent must then be `world`. A floating robot riding a base is rejected.
- Limbs stay `viam-labs:mujoco-sim:arm` components with `robot` set, each a chain whose base is fixed
  to the trunk. Their Viam frames are relative to the robot, so they are correct relative to the trunk.
- The robot's own Viam frame stays at its configured pose. RDK has no dynamic root pose, so the
  trunk's world pose is available from `io` (`trunk_pos`, `trunk_quat_wxyz`).
- With `actuators` set (unit 2), the limb arms are read-only:
  - their position-hold step callback is not registered, because it writes position targets into
    `data.ctrl`, which BAM now owns as torque;
  - motion commands return an error;
  - `JointPositions` reads `qpos`;
  - commanded positions read BAM's `q_target`.

  `io` is the only writer of targets. Without `actuators`, arms behave as today.
- Out of scope: the 3D viewer showing the duck walk. The plan checks whether `world_state` can stream
  the trunk pose cheaply, and adds it only if it can.

### 2. BAM actuators (mujoco-sim)

An optional `actuators` attribute on `robot`:

```json
"actuators": {"model": "bam", "motor": "xl330", "bam_model": "m6", "kp_fw": 200, "vin": 7.4,
              "vin_drop_gain": 0.1, "vin_min": 6.0, "joints": "^(?!passive_).*"}
```

The defaults are the values above. `vin_drop_gain` 0.1 and `vin_min` 6.0 are `infer_policy.py`'s
defaults, and 0.1 sits in the middle of the training range 0–0.2. 0 turns sag off.

At MjSpec build time, the robot mirrors `infer_policy.py`'s `load_mujoco_with_bam`, which mirrors
`bam.mjlab.BamActuator.edit_spec`. For each matching actuator:

- `set_to_motor()`;
- `forcelimited` on, with `forcerange = ±vin·kt/R` (from the BAM model);
- `ctrllimited` off, and `gear = [1,0,0,0,0,0]`;
- on its joint, `damping = 0`, `frictionloss = 0`, and `solref_friction` / `solimp_friction` set to
  BAM's stiff values `(-5e4, -2e2)` and `(0.99, 0.9999, 0.001, 0.5, 2.0)`.

The robot registers one step callback, a wrapper (callbacks take `dt`; `update()` takes no
arguments). On each step it:

1. Compares `sim.generation` with the generation it last bound to. mujoco-sim recompiles `model` and
   `data` on every `set_contribution`, for example each limb arm added after the robot, and runs
   `init` only for new contributions. So a controller made once would keep the dead arrays, and the
   armature on the dead model.
2. On a change, or the first step, it rebinds:
   - creates `MujocoController(bam_model, names, sim.model, sim.data, vin_drop_gain, vin_min)`, which
     also sets the armature on the new model;
   - calls `reset(data.qpos)`, so `q_target` holds the current pose and the voltage-sag state is
     cleared;
   - keeps the torque on/off flag, which lives on the robot handle.

   Because this happens on the first step after the rebuild, it follows every contribution's `init`,
   including an arm's `qpos = home`.
3. Calls `controller.update()`, and if torque is off, zeroes `ctrl` for the BAM actuators.

Targets go through `set_q_target` (present in BAM's `mujoco.py`) or the `q_target` array, as
`infer_policy.py` does. The bind check is one `ensure_bound()` method. The step wrapper calls it, and
so does `io` before touching the controller, because `io` can arrive after a rebuild but before the
next step, or while the world is paused. Rebuilds happen at configure time, so a mid-run rebuild is
not supported. `io` reports `generation`, and duck-policy stops if it changes while the loop runs.
Sim time stays continuous across a rebuild, so the stale-state check would not catch it.

An arm's `init` still writes `data.ctrl = home` under BAM. That is harmless, because `update()`
overwrites `ctrl` before the first `mj_step`. Without the actuator rewrite, BAM's torques would feed
the MJCF's position actuators and the physics would be silently wrong. Unit tests lock both the
rewrite and the rebind (see Testing).
- Without `actuators`, the file's own actuators run as today.
- `better-actuator-models` (numpy and colorama only) becomes a dependency, from Rhoban's git branch
  `mjlab_frictionloss` at the commit `microduck_rl` locks.
- BAM requires Python `<3.13`. The plan pins the module's Python to 3.12. Vendoring `bam/mujoco.py`
  and its model loader (Apache-2.0) is the fallback if pinning breaks other users.
- BAM's fit is valid at 5 ms. The robot warns if `physics_dt` is not 0.005 while BAM is on.

### 3. The `io` DoCommand (mujoco-sim)

Request: `{"io": {"targets": [14 floats] | null, "torque": true | false}}`

Response:

```json
{"t": 12.34, "q": [14], "dq": [14], "gyro": [3], "gravity": [3],
 "trunk_pos": [3], "trunk_quat_wxyz": [4], "joints": [14 names], "generation": 3,
 "torque": true}
```

`torque` in the request may be omitted or null, which leaves it unchanged; the response reports
the current flag.

Freshness: `io` reads state after the last physics step. Pollen's `infer_policy.py` reads sensors
and `xquat` straight after `mj_step`, so its values lag by one 5 ms substep, about 0.007 on gyro
and gravity at high velocity. That's below sensor noise, so it isn't a parity concern, but don't
compare individual ticks bit for bit.

- **One instant.** It runs on the sim thread. Targets are applied, then the state is read in the same
  task, between physics steps.
- **Joint order.** The actuator order of the joints matched by `actuators.joints`, reported in
  `joints` so the client can check it.
- **Gyro.** Trunk angular velocity in the trunk frame (`qvel` of the free joint's angular part).
- **Gravity.** The unit gravity vector in the trunk frame, computed from the free joint's
  quaternion, as `body_server.py` does.
- **Targets.** `targets: null` leaves the current `q_target` unchanged (a read-only call).
- **Torque.** `torque: false` makes the duck limp, as `body_server` does. With BAM, `ctrl` is zeroed
  after `update()` each step while torque is off. When torque is turned back on, `q_target` is reset
  to the current pose, so the duck isn't flung at a stale target.
- **Scope.** `io` requires `actuators` (unit 2). Without BAM there is no `io`; PD-only duck physics
  isn't a goal.

### 4. duck-policy module (new; directory and repo chosen in the milestone 2 plan)

Python, built like `devrel:rl-policy` (uv-built venv, bounded calls, a `generic` service with
DoCommand).

| File | Holds |
|---|---|
| `io.py` | `DuckIO` protocol (`exchange(targets, torque) -> Snapshot`), `Snapshot`, `SimIO` |
| `obs.py` | Snapshot plus command plus last action, to 61 floats. Pure, no Viam imports. |
| `loop.py` | The 50 Hz loop, the stop conditions, the start ramp, timing statistics |
| `service.py` | Viam resource: config, DoCommand, the mlmodel client |

Config: `io` (the duck robot's name), `mlmodel` (the service name), `policy_manifest` (path to the
manifest or `config.json`), and optional overrides of the safety limits.

**Each tick (20 ms):**

1. `snap = io.exchange(prev_targets, torque=True)`.
2. `obs = build_obs(snap, command, last_action)` gives 61 floats: 48 proprioceptive, then twist[3],
   head_pose[4] and body_pose[6]. The head and body slots are zero for the walk milestone.
3. `action = mlmodel.infer(obs)` gives 14 floats.
4. `targets = clip(default_pose + action * action_scale)`, clipped to the joint limits and capped per
   step.
5. Sleep to the next deadline. A missed deadline is counted and not caught up.

**Source of truth.** The 48 proprioceptive terms come from `infer_policy.py` and Pollen's policy
manifest (schema 2): term order, default pose, scales, last action, and the 15-to-14 joint mapping
with the mouth dropped. The milestone 2 plan transcribes them into `obs.py` first, with a golden test.

**Delays.** None injected. Pollen trains with 0–20 ms IMU delay and 20 ms joint-velocity delay. The
gRPC round trips add some in sim, and parity shows whether injection is needed.

**Safety.** Every condition stops the loop and records the reason in `status`:

| Condition | Detection | Action |
|---|---|---|
| Fall | gravity z > -0.5 (tilt about 60°) for 3 consecutive ticks | torque off (limp) |
| Bad action | NaN, inf, or \|action\| above the manifest's bound (2x the p99.9 of reference rollouts if the manifest has none) | torque off |
| Overruns | more than 5 % of the last 50 ticks late, or 3 in a row | stop, torque on, hold the last targets |
| Stale state | snapshot `t` not advanced since the last tick | stop, torque on, hold the last targets |
| Sim rebuilt | snapshot `generation` differs from the one at `start` | stop, torque on, hold the last targets |
| Hung call | `io` or `Infer` exceeds 15 ms | stop, torque on, hold the last targets |
| Bad command | `set_command` outside the trained twist ranges | clip, and note it in `status` |

- `start` ramps from the current pose to the default pose over 1 s, then hands over to the policy.
- `stop` holds the last targets.
- `status` reports ticks, late ticks, p50 and p99 tick time, the stop reason and the command.
- Not included: auto-restart after a fall, and stand-up or recovery policies.

### 5. Parity harness (this repo, `scripts/`)

- **Command script.** 20 episodes of 30 s each:
  - stand for 2 s;
  - forward at 0.15 m/s for 8 s;
  - turn left at 0.5 rad/s for 5 s;
  - turn right at 0.5 rad/s for 5 s;
  - stop for 4 s;
  - backward at 0.1 m/s for 6 s.

  Both sides use the same actuator settings: battery voltage 7.4 V, `vin_drop_gain` 0.1, `vin_min`
  6.0, kp_fw 200, no other domain randomization. Episode seeds vary only the start yaw.
- **Reference.** A headless runner that reuses `infer_policy.py`'s own code: `load_mujoco_with_bam`;
  `bam_ctrl.reset(data.qpos)` before each episode; and `PolicyInference.get_observations`, a method
  that needs a `PolicyInference` built with `new_cmd_obs=True` and `use_projected_gravity=True`. It
  steps exactly 4 × 5 ms per action and runs from a `microduck_rl` checkout at a pinned commit.
- **Timing difference.** mujoco-sim runs `real_time` against the wall clock, not in lockstep. Ticks
  land at arbitrary sim instants, with a varying number of physics steps between them. The 1.25×
  RMSE margin is meant to absorb that jitter. The harness logs the world's `real_time_factor` and
  `slips` for every episode, and an episode with any slip is reported separately rather than
  counted. At least 18 of the 20 episodes must be slip-free for the gate to be judged at all.
  Otherwise the result is "machine too slow", not pass or fail.
- **Ours.** The same script sent to duck-policy's `set_command` on the Viam machine. Trunk velocity
  is logged from `io`.
- **Metrics.** Falls per 20 episodes; velocity-tracking RMSE over the commanded segments (body-frame
  vx and wz); tick-rate statistics from `status`.
- **Pass if all hold:**
  - falls ≤ the reference's falls + 1;
  - our tracking RMSE ≤ 1.25 × the reference's RMSE;
  - mean tick rate ≥ 49.5 Hz;
  - p99 tick time < 20 ms;
  - < 1 % of ticks late.
- **Report.** Written to `docs/superpowers/specs/<date>-microduck-parity-results.md`.

### 6. Duck scene package

Pollen's walk MJCF (`robot_walk.xml`, `scene_walk.xml`) and meshes, with a `robot` config, published
as a **private** Viam archive package. The meshes are NC-licensed, so they are never bundled in the
public module. The config references it as `${packages.archive.<name>}`, as `rl-insert-sim` does.

### 7. Training job (milestone 3)

- `microduck_rl`'s `Mjlab-Velocity-Flat-MicroDuck` runs as a Viam training job.
- It uses a separate training image: mjlab 1.3.0, warp 1.12, torch 2.9.1, Python 3.12, BAM. Our
  rl-insert image pins different versions, so the two stay separate.
- The harness pattern from this repo carries over: image and sdist split, probe mode, video upload,
  package written before uploads.
- Output: `policy.onnx`, exported through `microduck_rl`'s `scripts/export.py` so the normalizer is
  built in, plus a `config.json` carrying the manifest fields duck-policy reads.
- Then parity runs again with this policy in place of Pollen's.

## Milestones

Each milestone gets its own implementation plan.

1. **mujoco-sim: floating base, BAM, `io`.** Units 1–3 and 6. Released upstream.
2. **duck-policy and walk parity.** Units 4–5, with Pollen's walk ONNX served by `onnx-cpu` on a Viam
   cloud simulation.
3. **Training job.** Unit 7, then parity with our policy. It can start in parallel with 1 and 2.
4. **Out of scope.** `BusIO` on the real duck: a later spec, once it arrives.

## Testing

- **mujoco-sim (CPU unit tests):**
  - a free-joint robot loads, and its limbs have trunk-relative frames;
  - a free joint on a non-root body, or on a robot riding a base, is rejected;
  - after the BAM spec rewrite, the compiled model's actuators match `load_mujoco_with_bam`'s
    (motor type, `forcerange`, `gear`, joint `damping`, `frictionloss`, friction solref and solimp)
    for `robot_walk.xml`;
  - a few hundred steps of mujoco-sim with BAM, from the same state and targets, track
    `load_mujoco_with_bam` plus `MujocoController` run directly, to float tolerance;
  - adding a limb arm after the robot, which forces a rebuild, leaves BAM bound to the new
    `model`/`data` (armature set, nonzero `ctrl` under load) with `q_target` equal to the post-`init`
    pose;
  - `io` returns state from one `t`, and `io` sent after a rebuild but before the next step acts on
    the new model (it binds first);
  - torque off makes the duck collapse. BAM's friction still acts on the back-driven motors, so the
    test expects a slowed collapse, not a free fall;
  - with `actuators` set, arm motion errors and no arm writes `ctrl`;
  - the existing ALOHA and TIAGo tests pass unchanged.
- **duck-policy:**
  - the `obs.py` golden test: a recorded `infer_policy.py` step reproduces its 61 inputs exactly;
  - the `joints` order from `io` is asserted to be the policy's order (left leg 5, head 4, right
    leg 5), with the mouth absent;
  - each stop condition fires under a fake `DuckIO`;
  - targets are clipped and rate-capped.
- **Integration:** the parity harness (unit 5) is the milestone 2 gate.

## Risks and open questions

- **MuJoCo version.** mujoco-sim runs MuJoCo 4.0, while `microduck_rl` and the BAM fit use 3.10. If
  parity fails on tracking rather than on timing, check this first. The reference runner can run on
  4.0 to separate the two.
- **Reference vs. our scene.** Two differences found while planning milestone 1:
  - `infer_policy.py` loads `scene.xml` (`robot_groundcontact.xml`) with the scene's integrator,
    which defaults to Euler.
  - mujoco-sim compiles with `implicitfast`, and our package uses `robot_walk.xml`, the velocity
    task's training model.

  The parity reference runner must load the same `duck.xml` and set `implicitfast`, so the only
  difference left is the stack under test.
- **Gravity compensation.** mujoco-sim's `load_mjcf` gravity-compensates every body of a robot
  file. Milestone 1 turns that off for floating robots; a weightless duck would pass every check.
- **Loop timing.** Python and two gRPC hops per tick against a 20 ms budget. The timing bar measures
  it in sim. On the RK3566 it is re-measured in the `BusIO` spec.
- **`onnx-cpu` on arm64.** Confirm that `viam:mlmodel:onnx-cpu` publishes linux/arm64 before
  milestone 2 ends. The robot needs it.
- **Pollen's walk policy.** Which published ONNX, and its license. The milestone 2 plan picks it from
  `pollen-robotics/microduck-policies` and records the revision.
- **viam-mujoco-sim branch.** The local checkout is on `diff-ik-policy-sim`. Milestone 1 branches from
  `main`.
