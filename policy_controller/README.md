# devrel:rl-policy

A Viam module that runs the `rl-insert` peg-insertion policy on an arm. It streams joint points at
20 Hz, solving damped least-squares IK once per step, the same way training does.

| Model | API |
|---|---|
| `devrel:rl-policy:controller` | `rdk:service:generic` |

```json
{"name": "inserter", "api": "rdk:service:generic", "model": "devrel:rl-policy:controller",
 "attributes": {"arm": "arm", "gripper": "gripper", "mlmodel": "policy",
                "package_dir": "${packages.ml_model.rl-insert}", "hole_mouth_mm": [400, 0, 50]}}
```

| Attribute | Required | Meaning |
|---|---|---|
| `arm`, `gripper` | yes | The xArm6 and its gripper (the caller grips the peg; the loop doesn't use the gripper) |
| `mlmodel` | yes | An mlmodel service serving the package's `policy.onnx` |
| `package_dir` | yes | The `rl-insert` ML model package (`config.json`, `kinematics.xml`) |
| `hole_mouth_mm` | yes | `[x, y, z]` of the hole mouth in the arm base frame |
| `action_clip`, `max_step_mm`, `max_joint_step_deg`, `workspace_mm` | no | Safety caps; the defaults are 2× the policy's measured 99.9th percentile (`scripts/policy_envelope.py`) |
| `timeout_s`, `max_overruns`, `jam_s`, `call_timeout_s`, `stop_timeout_s` | no | Loop limits |

DoCommands: `start`, `stop`, `status`, `benchmark` (joint-read and inference timing) and
`pre_insert_pose` (where to put the peg before `start`).

## Build and upload

```bash
./build_controller.sh      # from the repo root: dist/rl-policy-module.tar.gz
```

Then run `viam module upload` from `policy_controller/`, where `meta.json` is.

Tests: `uv run pytest tests/test_controller_*.py`. Design: the "Deploy controller" section of
`docs/superpowers/specs/2026-11-16-peg-insert-rl-design.md`. Results:
`2026-11-18-sim-to-sim-results.md`.
