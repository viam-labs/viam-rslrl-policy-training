# Sim-to-sim results: policy controller on rl-insert (plan 3, Task 10)

**Date:** 2026-11-18. **Result: 20/20 succeeded** (pass bar: within 20 points of training's success).

## Setup
- Policy: `rl-insert` ML model package v6 (run 6, script v7, best checkpoint at iteration 850, training success-rate EMA 0.98 at full difficulty). First policy trained with the once-per-step IK.
- Controller: `devrel:rl-policy` 0.1.3 (`devrel:rl-policy:controller`), with `viam:mlmodel:onnx-cpu` serving `policy.onnx`. Attributes: `hole_mouth_mm: [400, 0, 50]`, safety caps at their defaults (2x run 6's p99.9: action 9, 19 mm and 10 deg per step, 41 mm workspace).
- Cell: rl-insert on `viam-labs:mujoco-sim` 0.16.1 (archive `rl-insert-sim` 0.2.0), arm `max_joint_deg_s: 60`, `max_joint_deg_s2: 120`.
- Servo: the cell's arm actuators match `config.json` `control.joint_servo_sim` (kp 2000, dampratio 1, gravity compensated). The cell has no gain randomisation (training: +/-20%), so this tests the nominal servo only.
- Episode reset: world `reset`, gripper open, `pause`, peg to (0.4, 0, 0.069), gripper grab, `play`, 1 s settle, then controller `start`; poll `status`. The start is the nominal one (tip ~17 mm above the mouth, no lateral offset): training's difficulty-0 case.

## Results
| | |
|---|---|
| Succeeded | 20 / 20 |
| Steps to success | 27-31 (median 30; 1.5 s at 20 Hz). Training's episodes at full difficulty averaged ~26. |
| Overruns | 0 in all 20 |
| Peg tip depth below the mouth after success (sim's peg pose) | 14.7 mm typical (14.1-15.0); success is judged on the arm's nominal tip at 15 mm, so the peg sits ~0.3 mm higher in the grasp than nominal |
| Peg lateral offset from the hole axis | ~0.4 mm (radial clearance 0.35 mm): resting against the wall |

Tick timing (0.1.3 `status.ticks_ms`, one episode): start late ~1 ms, joint read 3-6 ms, inference ~2.3 ms, IK + safety ~0.6 ms: ~8 ms of the 50 ms budget.

## Notes
- The first episode after a module (re)start failed on overruns (4 in 7 ticks); every later one ran with none. Likely first-call warm-up (ONNX session, gRPC channels). Fix before hardware: one warm-up `get_joint_positions` + `infer` in `start` before the loop's clock starts.
- The sim's streaming semantics (plan Task 7) hold for this loop: one tick ahead never starved the stream.
- The 60 deg/s joint cap did not stop success; whether it slows the policy (30 vs ~26 steps) is untested. Raise it and compare if it matters.
- Not tested here: off-nominal starts (training's lateral offset and hole bias), servo gain variation, the real xArm's streaming driver (anchor + time, late points sent at once).
