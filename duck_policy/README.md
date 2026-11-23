# devrel:duck-policy

A Viam module that runs Pollen MicroDuck policies (walk or dance) and drives them from a gamepad.

| Model | API | What it does |
|---|---|---|
| `devrel:duck-policy:controller` | `rdk:service:generic` | 50 Hz loop: read the duck, build the observation, infer through an mlmodel service, write joint targets |
| `devrel:duck-policy:teleop` | `rdk:service:generic` | Turns an input controller (gamepad) into `start`, `stop` and `set_command` on a controller |

## controller

```json
{"name": "walker", "api": "rdk:service:generic", "model": "devrel:duck-policy:controller",
 "attributes": {"duck": "duck", "mlmodel": "walk-model", "world": "sim-world",
                "package_dir": "${packages.ml_model.microduck-velocity}"}}
```

| Attribute | Required | Meaning |
|---|---|---|
| `duck` | yes | A generic component that serves the `io` DoCommand (`viam-labs:mujoco-sim:robot` today) |
| `mlmodel` | yes | An mlmodel service serving the package's `policy.onnx` (`viam:mlmodel:onnx-cpu`) |
| `package_dir` | yes | The ML model package directory; its `config.json` drives everything |
| `world` | no | The mujoco-sim world. With it, `start` resets the world when the duck has fallen or is more than 0.15 rad off its default pose |
| `fall_gravity_z`, `call_timeout_s`, `ramp_s` | no | Loop limits |

DoCommands:

| Command | Effect |
|---|---|
| `{"command": "start"}` | Start the loop. A dance package takes `"cycles": N` (0 = until stopped), then holds the default pose |
| `{"command": "stop"}` | Stop; returns the stop reason |
| `{"command": "status"}` | State, tick stats, resets, stop reason; for dances `cycles_done` and `holding` |
| `{"command": "set_command", "vx", "vy", "wz"}` | Velocity command, clipped to the package's twist range (walk packages only) |

**Stop reasons.** `fall` and `bad_action` turn torque off (limp). `stale_state`, `hung_call` and
`overruns` (3 late ticks in a row) hold the last targets.

**Starting near the pose.** An upright duck within 0.15 rad of its default pose goes straight to the
policy, so a walker → dancer handover needs no reset. Further off, the loop ramps to the default pose
(hardware) or resets the world (sim).

### config.json

Training writes it (`duck_train/package.py`); `scripts/fetch_walk_policy.py` writes one for Pollen's
`velstand`. The fields the loop reads:

- `joints`, `default_pose`, `action_scale`, `joint_limits`, `control_hz`
- `model.input`, `model.output`, `model.obs_len`
- `command`: `{"twist_min", "twist_max"}` for walking, or `{"mode": "phase", "period_s"}` for a dance
- `observation`: `{"dim": 61, "terms": [{"name": "gyro", "dim": 3}, ...]}`. Known terms are `gyro`,
  `gravity`, `q_rel`, `dq`, `last_action` and `command`. A package without it (or with a string) uses
  the 61-value layout Pollen's policies take.
- `safety`: `action_bound`, `max_step_rad`

## teleop

```json
{"name": "teleop", "api": "rdk:service:generic", "model": "devrel:duck-policy:teleop",
 "attributes": {"input_controller": "WebGamepad", "policy": "walker",
                "mapping": {"hold": "AbsoluteZ", "vx": "AbsoluteRX", "vy": "AbsoluteY", "wz": "AbsoluteRY"}}}
```

Hold the `hold` control (`ButtonLT` by default) to drive: left stick sets vx/vy, right stick sets wz.
Start starts the policy and Select stops it. `mapping` remaps any of `hold`, `vx`, `vy`, `wz`,
`start` and `stop` to Viam Control names. A controller error or disconnect sends one zero twist.

## Run it

`run.sh` builds a venv in this directory with uv (it fetches uv if it isn't on PATH) and puts the
repo root on `PYTHONPATH`. Use it as a local module:

```json
{"type": "local", "name": "duck-policy", "executable_path": "<repo>/duck_policy/run.sh"}
```

Tests: `uv run pytest tests/test_duck_*.py` from the repo root.

Design and results: `docs/superpowers/specs/2026-11-21-microduck-sim-to-sim-design.md` (the loop),
`2026-11-21-microduck-dance-design.md`, `2026-11-22-microduck-teleop-design.md`.
