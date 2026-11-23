# viam-rslrl-policy-training

Train robot control policies in simulation as Viam custom training jobs, then run them on Viam machines.
Training uses [mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp, RSL-RL PPO). Machines run the
policy through `viam:mlmodel:onnx-cpu` and a small policy-loop module.

The repo holds two robots that share one pipeline:

| Robot | Task | Training | Deploy |
|---|---|---|---|
| xArm6 + UFactory G2 | peg-in-hole insertion (`PegInsert-xArm6`) | `rl_insert/`, `model/` | `policy_controller/` (`devrel:rl-policy`) |
| Pollen MicroDuck (biped) | walk (`Mjlab-Velocity-Flat-MicroDuck`) and dance (`Mjlab-Dance-Flat-MicroDuck`) | `duck_train/`, `model/harness.py` | `duck_policy/` (`devrel:duck-policy`) |

## The pipeline

```
 sim scene / choreography        Viam training job (GPU)            Viam machine
 ────────────────────────  ──►  ─────────────────────────────  ──►  ──────────────────────────────
 mjlab task + assets            image + training-script sdist       ml_model package (policy.onnx,
 (dance: recorded on a          → policy.onnx, config.json,         config.json) → onnx-cpu →
  mujoco-sim machine,             checkpoint.pt, metrics              policy-loop service → robot
  exported as a dataset)                                              (viam-labs:mujoco-sim or hardware)
```

- **`config.json` is the contract** between training and deploy. It carries the joint order, default
  pose, action scale, joint limits, control rate, command mode, the observation layout and safety caps.
  The deploy modules read it and nothing else about the model.
- **Each job has a `probe` mode**: a few minutes on the GPU to check drivers, GL, throughput, ONNX
  export and video upload before you spend hours on a real run.
- **Sim-to-sim before hardware**: each policy runs on a `viam-labs:mujoco-sim` machine and is compared
  with a plain-MuJoCo reference (`scripts/duck_parity.py`, the controller parity tests).

## Layout

| Path | What it is |
|---|---|
| `rl_insert/` | xArm6 + G2 + peg/hole scene (MjSpec) and the mjlab task: env config, MDP, IK action, PPO runner |
| `model/` | Viam training-script entry for peg insertion (`model/training.py`), plus `harness.py` and `viam_upload.py`, the job helpers both robots share |
| `duck_train/` | MicroDuck training entry (`train.py`), dance task and rewards, packaging |
| `policy_controller/` | `devrel:rl-policy` module: 20 Hz insertion loop on an arm. See [its README](policy_controller/README.md) |
| `duck_policy/` | `devrel:duck-policy` module: 50 Hz walk/dance loop and gamepad teleop. See [its README](duck_policy/README.md) |
| `dances/` | Dance keyframe files (`bob-sway.json`) |
| `scripts/` | Asset export, policy fetch, smoke tests, parity and reference runs (each script's docstring has its usage) |
| `container/` | Training images: `Dockerfile.train` (peg insertion), `Dockerfile.duck` (MicroDuck) |
| `hardware/` | Printable peg, hole blocks and fit coupon for the real-arm test ([nurb](hardware/AGENTS.md) parts) |
| `spikes/` | `contact_warp.py`: the MuJoCo Warp vs CPU contact gate run before training |
| `tests/` | pytest suite (CPU only; one GPU test is skipped) |
| `docs/superpowers/specs/` | Designs and results for each milestone, plus example machine configs (`duck-sim-machine.json`, `duck-choreo-machine.json`) |
| `docs/superpowers/plans/` | The implementation plan for each milestone |

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # mjlab (CPU), viam-sdk, onnxruntime, dev tools
uv run pytest      # the whole suite runs on a laptop
```

The MicroDuck scripts that drive Pollen's reference need a
[microduck_rl](https://github.com/pollen-robotics/microduck_rl) checkout at the pinned commit
(`0daab16`) and a viam-mujoco-sim environment. The docstring of each `scripts/duck_*.py` gives the
exact `uv run` line.

## Train a policy

1. **Build and push the training image** (linux/amd64):
   ```bash
   docker build --platform linux/amd64 -f container/Dockerfile.train -t <registry>/rl-insert-train:<tag> .
   docker build --platform linux/amd64 -f container/Dockerfile.duck  -t <registry>/microduck-train:<tag> .
   ```
2. **Build the training-script sdist** into `dist/`:
   ```bash
   ./build.sh        # dist/rl-insert-train.tar.gz   (peg insertion)
   ./build_duck.sh   # dist/microduck-train.tar.gz   (MicroDuck)
   ```
3. **Upload the script** with `viam training-script upload`, register the image as a training
   container, and submit a job. Viam runs `python3 -m model.training --model_output_directory=...
   --key=value ...`. Pass `mode=probe` first on any new image or hardware.
4. **The output directory becomes an ML model package**: `policy.onnx`, `config.json`,
   `checkpoint.pt`, `model_metrics.json` (plus `kinematics.xml` for the arm). Training videos go to
   Viam Data as they're written.

Version rule: never re-push an image tag or re-upload a script or package version. Bump it.

## Run it on a machine

1. Add the ML model package and an `viam:mlmodel:onnx-cpu` service pointing at
   `${packages.ml_model.<name>}/policy.onnx`.
2. Add the policy service (`devrel:rl-policy:controller` or `devrel:duck-policy:controller`) with
   `package_dir: ${packages.ml_model.<name>}`.
3. Start it with the DoCommand `{"command": "start"}` and watch `{"command": "status"}`.

`docs/superpowers/specs/duck-sim-machine.json` is a complete MicroDuck sim machine: world, duck,
walker, dancer and teleop.

Scripts that talk to a machine take a `cell.json` with the machine address and API key. Keep it, and
`viam.json`, out of git.

## A new dance

1. Pose keyframes on the choreo machine (`docs/superpowers/specs/duck-choreo-machine.json`, the duck
   pinned in the air): `scripts/duck_choreo.py pose|preview|check`.
2. `scripts/duck_choreo.py record` captures 3 cycles at 50 Hz as a sequence. Add it to a dataset.
3. Train `Mjlab-Dance-Flat-MicroDuck` on the dataset export (about 80 min on one A100).
4. `scripts/duck_reference.py --stats` fills the package's safety caps. Then deploy it as a
   `dancer` and run `{"command": "start", "cycles": 3}`.

## Status

| Milestone | Result | Write-up |
|---|---|---|
| Peg insertion, trained | 85.7 % success on the first run; later runs ~98 % at full difficulty | `specs/2026-11-17-first-training-run.md` |
| Peg insertion, sim-to-sim | 20/20 on mujoco-sim | `specs/2026-11-18-sim-to-sim-results.md` |
| MicroDuck walk parity (Pollen's policy) | matches the reference on tracking and timing | `specs/2026-11-21-microduck-parity-results.md` |
| MicroDuck walk, our own training job | walks and turns on the preview machine | `specs/2026-11-21-microduck-training-results.md` |
| MicroDuck dance | passes parity when the host isn't stalling; walker → dancer handover works | `specs/2026-11-21-microduck-dance-results.md` |
| Gamepad teleop | drives the walker from a web gamepad | `specs/2026-11-22-microduck-teleop-design.md` |

Known follow-ups: one controller per robot (a walker and a dancer can both drive the duck today), and
the `late_streak` overrun rule, which waits for timing from the real duck.

## Licences

The MicroDuck meshes and Pollen's policies are CC BY-NC-SA 4.0 (microduck_rl's `LICENSE-MODELS.md`).
The duck-sim archive and duck model packages stay private to the org and are never bundled into a
public module.
