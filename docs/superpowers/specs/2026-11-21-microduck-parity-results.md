# MicroDuck walk parity: results (2026-11-21)

## The comparison

- **Ours:** Pollen's `velstand` walk policy, driven by our stack on the `microduck-preview` machine:
  - the duck-policy module (50 Hz read-infer-write);
  - `viam:mlmodel:onnx-cpu`;
  - viam-labs:mujoco-sim with the floating base, BAM and `io` (PR #7, released in 0.21.0);
  - duck-sim 0.2.0 (the ground-contact model).
- **Reference:** Pollen's own CPU reference (`infer_policy.py`'s code, run headless by
  `scripts/duck_reference.py`) on the same `duck.xml` with the same MuJoCo build.
- **Command script:** stand 2 s; vx 0.4 for 8 s; vx 0.4 with wz +0.6 for 5 s; vx 0.4 with wz −0.6
  for 5 s; stop 4 s; vx −0.4 for 6 s.
- **Episodes:** 20 on our side.
- **Tools:** `scripts/duck_parity.py` (`ours`, then `report`).

## Verdict: strict FAIL, on two items that aren't stack mismatches

| Check | Result |
|---|---|
| slip-free ≥ 18/20 | pass (20/20, real-time factor ≈ 0.998) |
| falls ≤ ref + 1 | pass (1, episode 15; see below) |
| RMSE vs command ≤ 1.25 × ref (vx / wz) | pass (0.235 vs 0.223 / 0.594 vs 0.583) |
| tick rate ≥ 49.5 Hz | pass (50.0 Hz) |
| p99 tick < 20 ms | pass (13.65 ms worst episode; about 6 ms typical) |
| late < 1 % | pass (17 / 29,154) |
| walker ran every episode | **FAIL**: episode 15 stopped with `overruns` |
| segments 0–4, mean vx and wz within 25 % of ref, ≤ 2 outliers | pass |
| segment 5 (backward) vx | **FAIL**: 6/20 outliers (episodes 5, 7, 12, 18, 19 stood still; 15 had fallen) |

### Per-segment agreement (ours = mean of 20 episodes)

| Segment | Reference vx / wz | Ours vx / wz |
|---|---|---|
| 0 stand | 0.000 / 0.002 | 0.000 / 0.003 |
| 1 forward 0.4 | 0.156 / −0.035 | 0.161 / −0.011 |
| 2 forward + turn left | 0.150 / 0.399 | 0.153 / 0.397 |
| 3 forward + turn right | 0.167 / −0.376 | 0.159 / −0.352 |
| 4 stop | −0.001 / −0.015 | 0.000 / −0.011 |
| 5 backward 0.4 (the 14 that walked) | −0.139 / −0.082 | −0.134 / −0.074 |

## Failure 1: the backward start is a basin of attraction in velstand, not a stack difference

In 5 of 19 episodes the duck stood still for the whole backward segment. In each, the walker was
running and the command was −0.4, unclipped. To test whether that is our stack or the policy, the
reference was perturbed at t = 24 s, the moment the backward command arrives, over 20 seeds each:

| Perturbation at t = 24 s | Reference stood still |
|---|---|
| ±0.01 rad uniform noise on the 14 joints | **3 / 20** |
| the command switch delayed by a random 0–80 ms | 0 / 20 |
| the start pose perturbed instead (±0.01 rad at t = 0; 6 seeds) | 0 / 6 |

- **It's the policy's behaviour.** From some stances after the 4 s stop, velstand at vx −0.4
  never starts walking backward. −0.4 is the edge of both the trained range and velstand's backward
  dead zone: −0.3 already stands still.
- **The rates are consistent.** Our real-time jitter perturbs the stance naturally, and the
  unperturbed reference is deterministic. 5/19 against 3/20 is within binomial noise.
- **Command timing isn't the cause.** The reference walks back every time when only the switch
  timing varies.
- **The speed matches.** Episodes that did start walking backward track the reference within 4 % on
  vx.

**Conclusion: not a stack mismatch.** The segment-5 check compares a deterministic reference with
a stochastic one. A fairer gate would compare the stand-still rate against the perturbed
reference's, or only score episodes that start walking. Before M3 or a real-duck demo, either take
the backward start as known velstand behaviour or test it with Pollen's daemon settings (action
scale 0.9, low-pass filters).

## Failure 2: one overrun stop (episode 15)

At about 12.9 s the loop had 3 late ticks in a row (worst tick that episode 13 ms at p99), so it
stopped as designed (`overruns`). It held its last targets, and the duck fell within ~1.5 s.
Across the other 19 episodes there were 14 late ticks in all, and none in a row of 3. That makes it
a rare scheduling stall on the development Mac (sim real-time factor 0.998, no slips), not
sustained slowness.

**Open decision:** the overrun rule trades hardware safety, by stopping a stale loop, against
robustness to rare stalls; a stopped loop on a biped means a fall. Options:
- keep it, and accept a stall-induced stop about once per 20 × 30 s;
- raise `late_streak` from 3 to 5 (100 ms) and keep the 5 %-of-50 rule.

On the real duck the loop runs on the robot itself, a different scheduler, so measure there before
settling it.

## What this shows

- **The stack reproduces Pollen's reference** for standing, forward walking and turning: segment
  means within 2–7 %, and the tracking RMSE ratio is 1.05 on vx and 1.02 on wz.
- **Timing has a large margin.** Read, infer and write take about 6 ms at p99, and the ticks
  average 50.0 Hz.
- **The observation matches exactly.** `tests/test_duck_obs.py`'s golden test checks it against
  `infer_policy` to 1e-6, standing and turning.
- **The remaining failures** are policy behaviour at the edge of its command range, and one
  scheduling stall that the safety rule caught.

Data (not committed): `dist/duck-reference.json`, `dist/duck-ours.json`, `dist/duck-parity.md`.
