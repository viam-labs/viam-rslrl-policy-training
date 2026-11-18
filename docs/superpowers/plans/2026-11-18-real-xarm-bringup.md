# Real xArm Bring-up Implementation Plan (plan 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the `rl-insert` v6 policy through `devrel:rl-policy` on the real xArm6 + G2 at **armfarm15** (hackathons org) and get real insertions, without ever surprising the arm or the person next to it.

**Architecture:** Nothing new in the loop: the same controller (0.1.6+) that went 20/20 nominal and 18/20 off-nominal on the sim. This plan is about what differs on hardware: the arm's own kinematics and streaming driver, latency, the hole's true pose (no vision: calibrated by touch-off), the grip, and safety layers outside our code. Each task is read-only or moves the arm a little more than the last.

**Tech stack:** `viam:ufactory:xArm6` 1.1.38 (Modbus, host 192.168.10.241), `viam:ufactory:gripper` (G2), `viam:mlmodel:onnx-cpu`, `devrel:rl-policy`, the Python SDK from the laptop for probes.

---

## What armfarm15 is today (read 2026-11-18, config revision 2026-11-16T19:39:08Z)

- Org **hackathons** (`6d6c7293-…`), location Arm Farm, machine `13044ee6-198e-4759-8236-49d6842cc86d`, part `armfarm15-main`. Online.
- Arm `arm` from fragment `xarm-realsense-gripper2`: `speed_degs_per_sec` 120 (machine override), `acceleration_degs_per_sec_per_sec` 300, **`collision_sensitivity` 0 (off)**, frame = world.
- Gripper `gripper` (`viam:ufactory:gripper`, G2), frame z 105 mm, yaw 180. Wrapped by `gripper-wrapper`. `gripper_force` unset (module default 50 %).
- Wrist RealSense `cam` on the arm at (-83, 14, 18) mm: a body on link 6 the training never saw (collision check near the fixture).
- Obstacles fragment (disabled): table box centre z -123, 200 tall, so the **table top is ~23 mm below the arm base** (verify); front wall face at x 690, side wall face at y 450.
- Also on the machine and able to command the arm: `vla-controller` (streams to `arm`), `teleop*`, `servo` (devrel:diff-ik), an SO-101 leader arm. Only one thing may drive `arm` at a time.
- Home pose (`home-pose` switch): (0, -15.5, -68.5, 0, 84.0, 0) deg. The policy's start is near (0, -4.2, -39.1, 0, 43.3, 0) deg with the tip 10-20 mm above a hole at (400, 0, 50) mm.

## Ground rules (every task)
- Any task that moves the arm needs a person at the arm's e-stop, and the user's go-ahead for that session. I ask; I don't assume a previous yes carries over.
- Before moving: nothing else running on `arm` (`vla-controller`, teleop, servo idle).
- The controller's caps stay at run 6's measured defaults or tighter; `jam_s` short (1 s) until contact behaviour is known.
- Config changes on armfarm15 (another team's machine, another org) are listed and approved before they're made; each is reversible and noted here.

## Open questions for the user
1. **Cross-org access.** `devrel:rl-policy` and the `rl-insert` ML model package are private to DevRelDemos; armfarm15 is in hackathons. Options: (a) make the module `public_unlisted` and copy the model package into hackathons, (b) upload both into hackathons, (c) move the work to a DevRelDemos machine. Recommended: (a).
2. **Force/torque sensor.** Is UFactory's wrist F/T sensor fitted? The module has `viam:ufactory:ft_sensor`; a real force reading would replace the jam heuristic with training's own 40 N limit.
3. **Fixture.** Where can the hole block go, and can it be clamped? The policy expects the mouth near (400, 0, 50) mm in the arm base frame, i.e. ~73 mm above the table: a riser under the printed block.
4. **Collision sensitivity.** OK to set it from 0 to 3 for our sessions (restore 0 after)?

---

### Task 1: Get the controller onto armfarm15 (no motion)
**Done 2026-11-18:** module `public_unlisted`; `rl-insert` v6 copied to hackathons; package, `policy-model` and `policy` (0.1.7) on armfarm15, idle; `pre_insert_pose` answers. Answers: (a), no F/T sensor, collision sensitivity 3 OK.
- [x] Resolve open question 1. For (a): `meta.json` visibility `public_unlisted`, upload; `viam packages export` rl-insert v6 then `viam packages upload` it into hackathons as `rl-insert` v6.
- [ ] With approval, add to armfarm15: the `rl-insert` package, `viam:onnx-cpu` + `policy-model` (`model_path` `${packages.ml_model.rl-insert}/policy.onnx`), `devrel:rl-policy` + `policy` (`arm`, `gripper`, `mlmodel`, `package_dir`, a placeholder `hole_mouth_mm` [400, 0, 50], `jam_s` 1, `timeout_s` 5). Not started.
- [ ] Check: `status` idle; `pre_insert_pose` answers.

### Task 2: Kinematics parity with the real arm (read-only)
Our `kinematics.xml` is training's xArm6 (viam-ufactory-xarm's URDF). The real arm may be a variant (the module picks `xarm6_1305.urdf` for arm-type 1305).
**Result 2026-11-18** at joints (-19.9, -35.2, -69.9, 5.6, 114.7, -1.9) deg: our flange FK vs the driver's `GetEndPosition` 0.001 mm, 0.000 deg. Both are built from the same URDF, so this checks the model, not the hardware; Task 6 calibrates the hole in our FK at the insertion pose, so model error mostly cancels there.
- [x] Read joints and `get_end_position` (flange pose, mm) at the current pose; compare with `Kinematics.fk` of the same joints, minus the TCP offset. Pass: < 0.5 mm, < 0.2 deg. A mismatch means a variant: get the arm-type, rebuild `kinematics.xml` from the matching URDF, and say whether training must be rerun (it must if the link lengths differ).
- [ ] Repeat at 3 more poses the arm already holds during normal use (ask the user to leave it at them, or use teleop), not ones we command.

### Task 3: Latency (read-only)
**Result 2026-11-18** (0.1.7, n 200): joint read p50 2.5 / p99 4.0 / max 5.3 ms; infer p50 2.1 / p99 3.5 / max 5.2 ms. Pass. Staleness check still to do (needs manual mode, Task 6).
- [x] Add a `benchmark` DoCommand to the controller: N joint reads + N infers with a zero observation, returns p50/p99/max ms. No motion.
- [ ] Run N = 200. Pass: joint read p99 < 20 ms, read + infer + act p99 < 35 ms (the sim's typical was 8 ms with 80 ms outliers). The driver says its status report is every 200 ms; check the joint read isn't that stale (read twice 10 ms apart while the arm is moved by hand in manual mode: values must change).

### Task 4: Streaming probe on the real arm (moves, ±2 deg on joint 6)
**Result 2026-11-18** (armfarm15, collision sensitivity 3, person at e-stop; joint 6 at the arm's resting pose, probe 1 +/-2 deg, probe 2 a 0 -> 3 -> 0 deg ramp, scratchpad `stream_probe.py`, `stream_probe2.py`; no faults).
- **Timing is stream-absolute** (viam-ufactory-xarm `MoveThroughJointPositionsStreamed`, commit 73551ea): it skips the t=0 point, then for each point *sends it, then waits until anchor + its time* before taking the next, so every setpoint goes out one point spacing early (10 ms for our 5 points a tick). A batch arriving while the driver is idle (starved) has its first point sent on arrival: both probes' "immediate" moves. Streaming one tick ahead never starves it, so the steady state is a uniform 10 ms lead: no change to `point_times`. The validator only checks *declared* velocities and accelerations (we declare none), so speed 120 / accel 300 don't reject our points.
- **Joint reads are the commanded target, not the measured position.** Probe 2's 10 ms samples were exactly +0.250, +0.500 ... +3.000 deg, the streamed values. The driver's `JointPositions` reads register 0x2A with no argument; UFactory's SDK uses that call for the *target* angle (`get_servo_angle(is_real=False)`), and gets the real angle with the same register plus one data byte (`get_joint_states(num=1)`, firmware >= 1.9.110; num 3 adds velocity and effort). The policy needs measured joints (tip position, `joint_vel`, and `ik_error`, its contact signal): with targets, contact is invisible and the success, jam and slip checks would read the commanded depth. **Blocks Task 8 (contact).** Fix: the driver reads real angles (a small change in `arm/comm.go` `JointPositions`).
- **Fix tested on armfarm15 2026-11-18:** `../viam-ufactory-xarm` branch `fix/real-joint-positions` (commit 2c70325 on v1.1.38): `JointPositions` sends 0x2A with a joint-states count of 1 on firmware >= 1.9.110 (armfarm15: 2.8.2, xArm6 1305 variant, which shares the xArm6 kinematics). Built CGO-free for linux/amd64 and installed with `viam module reload-local --file` (`viam module reload`, the cloud variant, only toggled the reload flags and never built). Ramp probe rerun: reads now track the joint (0.996 deg at 0.30 s, 1.834 at 0.46, 2.460 at 0.59, settling 2.996 at 0.86 s while the stream commanded 0 -> 3 deg over 0.6 s), ~90-100 ms behind the stream's timestamps, roughly within training's 0-2 step action delay plus servo lag.
- [x] Person at e-stop. Arm at home-pose, clear of everything. Run scratchpad `stream_probe.py` against armfarm15: joint 6 +2 deg at 0/50/100 ms, back at 150/200 (sent at 300 ms), +2 at 1500/1600 (sent at 1 s).
- [ ] Record, as for the sim (plan 3 Task 7): absolute from the stream anchor or per batch, what a late batch does, whether the first point is skipped, and whether `speed_degs_per_sec` 120 / `acceleration_degs_per_sec_per_sec` 300 limit streamed points (run 6's policy peaks near 100 deg/s with sharp reversals). If acceleration limits bind, measure the lag with a 20 Hz ±1 deg square wave.
- [ ] Update `point_times` only if the real driver differs, behind a config attribute so the sim keeps working.

### Task 5: Fixture, peg and grip (hardware, the user)
- [ ] Print `hardware/stl/hole_block_c0_35.stl` (policy's clearance) and `hole_block_c0_5.stl` (easier first contact), `peg.stl`; a riser so the mouth sits ~73 mm above the table (measure the table-to-base offset first). Fit-test with `fit_coupon.stl`.
- [ ] Clamp the block near (400, 0) mm in the base frame, upright.
- [ ] Grip: set G2 `gripper_force` so the pads squeeze ~30 N (training's `grip_force`); check the peg protrudes 40 mm past the pad centre (`protrusion_m`) and doesn't slip under a firm hand push along its axis.

### Task 6: Calibrate the hole by touch-off (arm moved by hand)
- [ ] With the peg gripped, manual mode, the user guides the peg into the hole until it seats on the floor (depth 20 mm). Read joints; `hole_mouth = nominal_tip + (0, 0, depth)`. Three times, re-gripping between; spread ≤ 0.3 mm or repeat. Also the tip's tilt from vertical at the seated pose (training's hole is upright; > 1 deg is out of distribution).
- [ ] Set `hole_mouth_mm` to the mean.

### Task 7: Air runs (moves, no contact)
- [ ] Start the controller with `hole_mouth_mm` 40 mm **above** the calibrated mouth (`start` takes it per episode): the policy "inserts" into air above the real hole. 5 episodes from `pre_insert_pose` for that virtual hole. Watch: smoothness, step sizes against the caps, overruns, `ticks_ms`, that it ends `succeeded` or `jam` (no contact is possible).

### Task 8: First contact (moves, contact)
- [ ] Collision sensitivity on (open question 4), `jam_s` 1, `timeout_s` 5, c0.5 block first. 5 nominal episodes from `pre_insert_pose`; then 5 on c0.35. Record outcome, steps, overruns, and the arm's joint torques (`load` DoCommand) at the end of each.
- [ ] Stop criteria for the session: any collision trip, any `slip`, or two `jam`s in a row: stop and look before continuing.

### Task 9: Off-nominal and results
- [ ] 20 episodes from training's start distribution (as `sim_offnominal.py`, with the real arm and no peg reset: the user re-seats the peg after a failure).
- [ ] Write `docs/superpowers/specs/2026-11-xx-real-xarm-results.md`; commit.

## Not in this plan
- Vision for the hole pose (hole_mouth_mm on `start` is the hook).
- Run 7 (grasp slip in training): only if Task 8/9 show jams or slip.
- Releasing and retreating after success: the caller's job.
