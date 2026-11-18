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

## Off-nominal starts (training's full-difficulty distribution)
Script: scratchpad `sim_offnominal.py`, seed 0. Arm joints nominal +/-0.02 rad (1-2.4 deg TCP tilt), the nominal tip then
corrected (position only) to lateral +/-3 mm and 10-20 mm above the mouth, peg in hand; `start` with the hole estimate
off by +/-(2, 2, 1) mm (`hole_mouth_mm` on start, new in 0.1.4). Not reproduced: grasp offset/tilt in the hand, gain randomisation.

| Controller | Succeeded | Failures |
|---|---|---|
| 0.1.4 (training's success test against the *estimated* hole; > 3 overruns per episode fails) | 6/20 | 11 timeouts with the peg 15-19 mm in (1 mm centring can't pass against a 2 mm-off estimate), 3 short timeouts, 4 overruns (sporadic 20-80 ms joint reads at a steady real-time factor of 1) |
| 0.1.5 (success tolerances widened by `robustness.hole_bias_m`; fail on > 3 *consecutive* overruns) | 15/20 | 5 timeouts, 0 overruns |
| 0.1.6 + sim cell 0.2.1 (slip/jam guard, `jam_s` 2 s; G2 pads at 30 N each, was 15) | 18/20 | 2 jams, stopped after ~2.6 s: the peg on the mouth's chamfer 1.2-1.5 mm off the axis while the arm's tip read 6 mm deep (it slid ~5.5 mm up the fingers) |

Steps to success on 0.1.5: 21-57, median 25.

**The remaining failures are grasp slip.** Rerunning the five failed starts gave 3/5 (contact is not deterministic here). In the two that failed again, the arm's nominal tip was 21-22 mm below the estimated mouth and 5.5-7.7 mm off its axis, while the sim's peg was only 11-16 mm in and 0.4 mm off the axis: the peg had slid and pivoted in the fingers while the policy kept pushing for the full 10 s. Training holds the peg with a compliant weld (`grasp_model`, "real grasp slip is not modelled"); the sim cell's G2 holds it by friction, as the real one will.

**Grip in the sim cell (measured offline on the exported `g2.xml`).** One actuator closes both coupled fingers, so its force limit was shared: 15 N per pad at `grip_force` 30 N. Fixed in `add_gripper` (force limit 2 x grip_force, kp 2e4 so it saturates on a 10 mm peg), shipped as `rl-insert-sim` 0.2.1. The remaining slip is MuJoCo's soft friction: under a steady push the held peg creeps (~1.3 mm/s at 10 N, proportional to force) with no static hold. `noslip_iterations` 10 removes it (no creep to 40 N axial, 30 N sideways at the tip), but mujoco-sim keeps the world's own `<option>` (only `cone` and `impratio` come from model files), so the cell can't set it: a mujoco-sim feature request. The real G2's pads should hold statically, so the sim's slip is pessimistic; the guard is for the real arm's jams either way.

Steps to success on 0.1.6: 20-53, median 26.
