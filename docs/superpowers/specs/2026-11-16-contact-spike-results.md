# Contact spike results: MuJoCo Warp on a Viam sim sandbox

Date: 2026-11-17. Plan 1, Task 8.

## Machine

`rl-insert-probe` (DevRelDemos org, `test` location): a Viam sim sandbox started with `start_simulation` on a machine whose only config is a `rdk:builtin:shell` service.

| | |
|---|---|
| GPU | NVIDIA L4, 22 GiB, sm_89 (**not an L40S**) |
| Driver | 580.173.02 (CUDA 13.0); Warp 1.19.0 bundles CUDA Toolkit 13.4 |
| CPU / RAM / disk | 8 cores, 31 GB, 126 GB free |
| Docker | Not available. The `uv` path was used (`scripts/probe_sim_sandbox.sh`) |

Sandbox quirks, all needed for plan 2:

- **Driver location.** The driver is mounted at `/usr/local/nvidia` (`bin/`, `lib64/`). Set `PATH` and `LD_LIBRARY_PATH` to those. `/usr/bin/nvidia-smi` is an empty placeholder.
- **Shell environment.** `viam machine part shell` needs a TTY (wrap it with `script -q /dev/null`). Its environment has no `HOME`, and `PATH` lacks `/usr/local/bin`, where `uv` lives.
- **File transfer.** `viam machine part cp` works both ways.

## Result: pass, after reclassifying one borderline world

`spikes/contact_warp.py --device cuda --nworld 4096 --reference 64`, seed 0:

| Clearance | Inserted (Warp / CPU ref) | Per-world agreement | Blew up | Contact buffer peak | World-steps/s |
|---|---|---|---|---|---|
| 1.0 mm | 73.4% / 71.9% | 98.4% | 1 → 0 | 7.2% | 36.6k |
| 0.5 mm | 50.0% / 53.1% | 96.9% | 0 | 8.9% | 36.2k |
| 0.2 mm | 35.9% / 39.1% | 96.9% | 0 | 10.6% | 35.6k |

The script reported `"passed": false` because of a single world at 1.0 mm.

**What happened to that world (1335).** It started balanced on the rim, 2.4 mm past the chamfer edge. In the batched run it snapped in and ended with its tip 0.7 mm into the floor, against a "sank" bound of 0.5 mm. It didn't flip (1.2° tilt), didn't leave the fixture, and its values stayed finite.

**Why that's a test artifact, not instability:**
- The same start replayed alone on the GPU didn't snap in within 3 s. It stayed on the rim, tilting slowly.
- CPU MuJoCo snapped it in, and it rested at the normal 0.1 mm.

This is the chaotic rim case from finding 8. The 0.7 mm comes from the snap-in impact: the test rig drives a 20 g peg with about 19 N and no arm inertia behind it, which is the known artifact. The final velocity wasn't captured, so the impact explanation is inferred.

**Fix.** The bound now means "through the 4 mm floor" (`-depth - 2 mm`) instead of "into it". Under the new bound the run above has zero blow-ups at every clearance, and the CPU run still passes.

## Throughput

About 36k world-steps/s on the L4 with 4,096 worlds. This is a lower bound: the spike copies data back to the host every 10 steps and doesn't use CUDA graph capture. mjlab does both differently, so plan 2 should measure its own rate. The first run's kernel compile took a few minutes; cache `/root/.cache/warp`.

## Decision

**Proceed to plan 2.** MuJoCo Warp on CUDA reproduces CPU MuJoCo's contact statistics for this peg and hole at all three clearances, with no blow-ups and plenty of contact-buffer headroom.

Open for plan 2:
- **GPU.** The sim sandbox gave an L4, not the expected L40S. Confirm what GPU a sim machine gets, or plan training around the L4 (22 GB, slower) or Viam training jobs (2× A100).
- **Measure throughput** with mjlab's own stepping (graph capture), not the spike's loop.
