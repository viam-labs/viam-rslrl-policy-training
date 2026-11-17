# Vertex probe and GPU smoke job: results

Date: 2026-11-17. Plan 2, Task 7.

## IDs

| What | Value |
|---|---|
| Image | `docker.io/hipsterbrown/viam-rl-insert-train:0.1.0-1` (public), digest `sha256:8e35726d…` |
| Container ID (DevRelDemos) | `35da1a78-361d-4fba-abe8-95d2d156b0ee` |
| Training script | `16518049-9dd3-479f-9644-c5d112aa42d8:rl-insert-train` v1, built from commit `a5d99b6` |
| Probe job | `6afbedbfdfffabc7948ae040` (model `rl-insert-probe` v1): completed in about 4.5 min of training time, after about 10 min of provisioning |
| Smoke job | `6afbf2b8d3cf807ab57b2ce7` (model `rl-insert-smoke` v1): completed in 5 min |
| First training job | `6afbf65d0858c3bb65334cbd` (model `rl-insert` v1). See `2026-11-17-first-training-run.md` |

## Probe (`probe.json`, 4096 envs)

| Check | Result |
|---|---|
| GPUs | 2× NVIDIA A100-SXM4-80GB (not 40 GB as documented) |
| Driver | 580.178.04, CUDA 13.0. The default cu130 image works, so no cu126/warp-1.17 fallback is needed |
| torch / Warp | 2.14.1+cu130, CUDA available. Warp 1.19.0 sees 2 devices. The run used `cuda:0` |
| GL | `egl` on the **GPU** (renderer "NVIDIA Corporation \| NVIDIA A100-SXM4-80GB"), not llvmpipe |
| Grasp hold (CUDA, float32, 64 envs) | settle 0.085 mm, creep 0.064 mm: **pass** |
| Throughput | 10.4k policy steps/s, 522k physics steps/s (random actions, no PPO update) |
| Video upload | uploaded to Viam Data under `rl-insert` (file id `…/sr89pn2pma/94FJUU…`) |

The package installed from the sdist without PyPI. That's the build-isolation fix in `a5d99b6`, found in review.

## Smoke training job (3 iterations, 4096 envs, clearance 1.0 mm)

- **Iteration time:** 9.05–9.09 s in steady state, 15.3 s for the first. Collection is 8.9 s of that, so the PPO update is about 0.1 s.
- **Videos:** 3 clips (env steps 0, 24, 48) uploaded to Viam Data with tags `rl-insert`, `smoke-1` and `iter-N`.
- **Package** (`viam packages export` gives `<dest>/1/<name>.tar.gz`): `policy.onnx` (onnxruntime check: `[1,33]` → `[1,6]`, finite), `config.json`, `checkpoint.pt` and `model_metrics.json`, with rates in [0, 1].
- **Early behaviour:** success 0, timeouts about 93%, excess force about 6%, mean episode length growing. That's expected after 3 iterations.

## Decisions

- **Settings for the first real job:** `max_iterations=1200` (about 3 h at 9.05 s), `video_every_iters=100` (about 12 clips), clearance 1.0 mm, 4096 envs.
- **No image or driver changes are needed.** The second A100 sits idle; multi-GPU stays out of scope until single-GPU learning is shown to work.
