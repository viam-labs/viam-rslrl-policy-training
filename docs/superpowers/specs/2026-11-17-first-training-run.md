# First training run: results

Date: 2026-11-17. Plan 2, Task 8.

## IDs

| What | Value |
|---|---|
| Job | `6afbf65d0858c3bb65334cbd`, status COMPLETED |
| Model | `rl-insert` v1 (ML model package, DevRelDemos) |
| Container and script | container `35da1a78-…`, `rl-insert-train` v1 (commit `a5d99b6`) |
| Arguments | `clearance_mm=1.0 num_envs=4096 max_iterations=1200 video_every_iters=100 run_name=c1.0-run1 upload_part_id=cab989eb-…` |
| Time | 4.2 h of training (15,153 s), about 4.3 h end to end. About 12.6 s per iteration, compared with 9.05 s in the smoke job. The difference is most likely video recording and longer episodes; it hasn't been profiled. |

## Results (`model_metrics.json`, 101 samples over 1200 iterations)

| Iteration (approx.) | Success | Excess force | Timeout | Slip | Mean reward | Episode length (steps) | Difficulty |
|---|---|---|---|---|---|---|---|
| 0 | 0.0% | 6.4% | 93.6% | 0 | -0.92 | 22.3 | 0.30 |
| 150 | 74.4% | 25.6% | 0 | 0 | 27.8 | 11.9 | 0.70 |
| 300 | 77.0% | 23.0% | 0 | 0 | 31.5 | 12.5 | 1.00 |
| 600 | 81.1% | 18.9% | 0 | 0 | 31.9 | 12.5 | 1.00 |
| 900 | 83.3% | 16.7% | 0 | 0 | 34.2 | 13.0 | 1.00 |
| 1199 | **85.7%** | 14.3% | 0 | 0 | 35.5 | 14.1 | 1.00 |

Trends:
- Success reaches about 74% within 150 iterations. It holds steady while the curriculum ramps from 0.3 to 1.0 (reached by iteration 300), then climbs slowly to 86% at full difficulty. It was still rising at the end, so the run isn't converged.
- Excess force (above 40 N) is the only failure mode: 26% at iteration 150, 14% at the end. Timeouts are gone after the first few iterations, and the peg never slipped from the grasp.
- Episodes get slightly longer late in the run (12 to 14 steps, about 0.7 s at 20 Hz). The policy appears to be trading speed for gentler contact; nothing beyond the force rate falling at the same time was checked.

## Package check

- **Contents:** `policy.onnx` (203 KB), `config.json`, `checkpoint.pt` (1.3 MB) and `model_metrics.json`.
- **onnxruntime:** input `obs` has shape [1,33] and output `actions` has shape [1,6]. The output is finite.
- **All-zero observation:** gives actions of up to about ±18.7 (37 mm and 18.7° per step after scaling). An all-zero observation is out of distribution, but `config.json` has `action.clip: null`. **The plan 3 controller must clip actions and cap per-step motion; it must not trust the policy output.**

## Videos (Viam Data, location `sr89pn2pma`, part `rl-insert-main`)

There are 12 clips of 10 s each, at env steps 0, 2400, …, 26400 (every 100 iterations: 0, 100, …, 1100). They're tagged `rl-insert`, `c1.0-run1` and `iter-N`. Only the first and last clips were downloaded and looked at, as contact sheets at 2 fps:

- **Step 0 (untrained):** the gripper drifts above the hole and tilts sideways. It never descends.
- **Step 26400 (iteration 1100):** the policy descends straight onto the hole and inserts again and again. Episodes reset roughly every second. Approach and insertion look vertical and centred. Contact force can't be judged from the video.

## Decisions and next steps

- **Release candidate:** `rl-insert` v1 is the first candidate for the sim-to-sim cell (plan 4). It was trained at 1.0 mm clearance. The 0.2–0.5 mm target fit still needs a tighter-clearance run, either fine-tuned from `checkpoint.pt` or trained from scratch.
- **Force:** excess force is the main thing to fix. Options:
  - raise the force penalty;
  - lower `pos_scale_m` near contact;
  - train longer, since the force rate was still falling at the end.
- **Deployment safety:** the action clip and per-step limit above belong in the plan 3 controller.
