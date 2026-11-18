# Policy Controller Implementation Plan (plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Viam service that runs the trained `rl-insert` policy on an xArm6, first against the rl-insert sim cell, with training changed so the sim moves the arm exactly the way the controller will.

**Architecture:** Training switches from "re-solve IK every 1 ms" to "solve IK once per 50 ms policy step at the measured joints, ramp the joint target there from the previous commanded target", which is what a 20 Hz controller streaming joint points can reproduce. The training package then ships a mesh-free `kinematics.xml` (the same MuJoCo model training used), so the controller's FK and IK are the training's, not a reimplementation. The controller is a Python generic service: a pure core (`policy_controller/core.py`: FK, IK, observation, safety) that is tested for parity against the training env, and a thin Viam layer (`policy_controller/service.py`) that reads joints, calls the onnx-cpu mlmodel service, and streams joint points with `arm.move_through_joint_positions_streamed`.

**Tech Stack:** mjlab (training, CPU warp for tests), MuJoCo Python (controller FK and Jacobian), NumPy, viam-sdk (Python, `>=0.85`), `viam:mlmodel:onnx-cpu`, `viam-labs:mujoco-sim` 0.16.x for sim-to-sim.

**Design references:** `docs/superpowers/specs/2026-11-16-peg-insert-rl-design.md` ("Deploy controller": config, DoCommands, loop, stop conditions) and `docs/superpowers/specs/2026-11-17-reusable-rl-pipeline-design.md` ("Policy controller", "The hard part: reproducing the sim's IK"). This plan picks option (b) for the IK question: change training to match deployment.

**Decisions made here (deviations from the specs):**
- The hole pose is a service attribute, `hole_mouth_mm` in the arm base frame, not a frame-system lookup. ponytail: add `hole_frame` when a camera or fixture frame actually supplies it.
- The kinematics hash check becomes "use the package's `kinematics.xml`": there is nothing to drift.
- Module namespace `devrel` (the DevRelDemos org), module `rl-policy`, model `devrel:rl-policy:controller`, API `rdk:service:generic`.

---

## File structure

| File | Responsibility |
|---|---|
| `rl_insert/task/actions.py` (new) | `StepIKAction`: one DLS solve per policy step, joint target ramped over the step |
| `rl_insert/task/env_cfg.py` (modify) | use `StepIKActionCfg` |
| `rl_insert/robot.py` (modify) | `kinematics_spec()`: arm + gripper + `tcp` site, no meshes |
| `model/package.py`, `model/training.py` (modify) | write `kinematics.xml` into the package; `control.ik` gains `solve: "once_per_policy_step"`, `interpolation`, `posture_target`, `joint_limits` |
| `policy_controller/core.py` (new) | `Kinematics` (FK, DLS step), `Policy` state (observation, action to joint goal, safety). No Viam imports. |
| `policy_controller/service.py` (new) | the Viam generic service: config, DoCommands, 20 Hz loop, streaming |
| `policy_controller/main.py`, `policy_controller/meta.json`, `policy_controller/run.sh`, `policy_controller/requirements.txt` (new) | module packaging |
| `tests/test_step_ik.py`, `tests/test_controller_parity.py`, `tests/test_controller_safety.py` (new) | tests |

The controller lives in this repo so the parity test can import both sides. `policy_controller/` must not import `rl_insert` or `mjlab` at runtime (the module ships without them); only the tests do.

---

### Task 1: Solve IK once per policy step in training

**Files:**
- Create: `rl_insert/task/actions.py`
- Modify: `rl_insert/task/env_cfg.py` (the `actions={"ee_delta": DifferentialIKActionCfg(...)}` block, around line 297)
- Test: `tests/test_step_ik.py`

Background: mjlab's `DifferentialIKAction` (`.venv/lib/python3.12/site-packages/mjlab/envs/mdp/actions/differential_ik.py`) sets the desired pose in `process_actions` (once per `env.step`) and runs `compute_dq()` + `set_joint_position_target` in `apply_actions`, which mjlab calls every physics substep (`decimation` = 50 times). We keep `process_actions` and `compute_dq` and replace `apply_actions`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_step_ik.py
"""IK is solved once per policy step and the joint target ramps to that solution over the step."""
import torch

from rl_insert.task.checks import make_env


def test_ik_solved_once_and_target_reaches_goal():
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    term = env.action_manager.get_term("ee_delta")
    calls = 0
    solve = term.compute_dq

    def counted():
        nonlocal calls
        calls += 1
        return solve()

    term.compute_dq = counted
    env.step(torch.full((4, 6), 0.5))
    assert calls == 1
    q_goal = term._q_to.clone()
    target = env.scene["robot"].data.joint_pos_target[:, term._joint_ids]
    torch.testing.assert_close(target, q_goal)  # ramp ends on the goal
    assert (q_goal - term._q_from).abs().max() > 1e-4  # it moved


def test_ramp_continues_from_previous_target():
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    term = env.action_manager.get_term("ee_delta")
    env.step(torch.full((4, 6), 0.5))
    prev_goal = term._q_to.clone()
    env.step(torch.full((4, 6), 0.5))
    torch.testing.assert_close(term._q_from, prev_goal)  # no jump back to the lagging measured joints
```

(If the entity's target attribute is not `joint_pos_target`, find its name in `mjlab/entity/data.py` and use that.)

- [ ] **Step 2: Run it, expect FAIL** (`AttributeError: ... '_q_to'`): `uv run pytest tests/test_step_ik.py -v`

- [ ] **Step 3: Implement**

```python
# rl_insert/task/actions.py
"""The IK action as the deploy controller can run it: one solve per policy step, not per 1 ms.

A 20 Hz controller reads joints, solves damped least squares once, and streams joint points that
ramp to the solution over the 50 ms step. Training does the same, so the policy is trained on the
motion the real arm gets. config.json's control.ik documents it for the controller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs.mdp.actions.differential_ik import DifferentialIKAction, DifferentialIKActionCfg


class StepIKAction(DifferentialIKAction):
    def __init__(self, cfg: StepIKActionCfg, env):
        super().__init__(cfg, env)
        self._env = env
        shape = (env.num_envs, len(self._joint_ids))
        self._q_from = torch.zeros(shape, device=env.device)
        self._q_to = torch.zeros(shape, device=env.device)
        self._substeps = env.cfg.decimation
        self._k = 0

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)  # desired pose from the measured TCP
        q = self._entity.data.joint_pos[:, self._joint_ids]
        # Ramp from where the target already is, not from the measured joints: those lag the target
        # (PD lag, 0-2 step delay), and restarting there would command a small retreat every step.
        # First step of an episode: nothing commanded yet, so start at the measured joints.
        fresh = self._env.episode_length_buf == 0
        self._q_from[:] = torch.where(fresh.unsqueeze(-1), q, self._q_to)
        self._q_to[:] = q + self.compute_dq()  # IK linearised at the measured joints
        self._k = 0

    def apply_actions(self) -> None:
        self._k += 1
        a = min(self._k / self._substeps, 1.0)
        self._entity.set_joint_position_target(
            self._q_from + a * (self._q_to - self._q_from), joint_ids=self._joint_ids
        )


@dataclass(kw_only=True)
class StepIKActionCfg(DifferentialIKActionCfg):
    def build(self, env) -> StepIKAction:
        return StepIKAction(self, env)
```

In `env_cfg.py`, import `StepIKActionCfg` from `rl_insert.task.actions` and replace `DifferentialIKActionCfg(` with `StepIKActionCfg(` in the `ee_delta` action. Keep every argument.

If `episode_length_buf` is not yet 0 for a just-reset env when `process_actions` runs, check the order in `mjlab/envs/manager_based_rl_env.py` `step()` and key `fresh` on whatever marks the first step (the IK action's own `reset()` can set a per-env flag instead).

Note `max_dq` (0.5 rad default) now clamps the whole step's move instead of each 1 ms substep. A 2 mm / 1 degree step needs far less, so leave it.

- [ ] **Step 4: Run the new test and the whole suite**: `uv run pytest -q`. Expected: all pass. If `test_grasp_hold` or `test_peg_slipped` now fail, report it rather than loosening them: the grasp behaves differently under a ramped target and that needs a look.

- [ ] **Step 5: Measure throughput, before and after** (CPU, so only the ratio means anything):

```bash
git stash && uv run python -c "from rl_insert.task.checks import make_env, throughput; e=make_env(64,'cpu'); e.reset(); print(throughput(e))" && git stash pop
uv run python -c "from rl_insert.task.checks import make_env, throughput; e=make_env(64,'cpu'); e.reset(); print(throughput(e))"
```

Expected: `policy_steps_per_s` after >= before (49 fewer Jacobian + 6x6 solves per policy step). If it is slower, stop and report.

- [ ] **Step 6: Commit** `feat: solve the IK action once per policy step, ramping the joint target (what the controller streams)`

---

### Task 2: Ship the kinematics and the new IK contract in the package

**Files:**
- Modify: `rl_insert/robot.py` (add `kinematics_spec`), `model/package.py` (the `control.ik` dict around line 100), `model/training.py` (`write_package`)
- Test: add to `tests/test_robot.py`

- [ ] **Step 1: Failing test**

```python
# tests/test_robot.py (append)
def test_kinematics_spec_matches_training_fk():
    import mujoco
    import numpy as np
    from rl_insert.robot import add_gripper, kinematics_spec, load_xarm6

    full = load_xarm6(); add_gripper(full); full = full.compile()
    kin = mujoco.MjModel.from_xml_string(kinematics_spec().to_xml())
    assert kin.nmesh == 0
    q = np.array([0.1, -0.3, -0.6, 0.2, 0.9, -0.4])
    out = []
    for m in (full, kin):
        d = mujoco.MjData(m)
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        out.append((d.site("tcp").xpos.copy(), d.site("tcp").xmat.copy()))
    np.testing.assert_allclose(out[0][0], out[1][0], atol=1e-9)
    np.testing.assert_allclose(out[0][1], out[1][1], atol=1e-9)
```

(The arm's joints are the first 6 qpos entries in both models; if not, index by `ARM_JOINTS` via `m.joint(name).qposadr`.)

- [ ] **Step 2: Run, expect FAIL** (`ImportError: kinematics_spec`).

- [ ] **Step 3: Implement** `kinematics_spec()` in `rl_insert/robot.py`: `load_xarm6()`, `add_gripper(spec)`, then delete every mesh geom and every mesh, iterating over copies (`for g in list(spec.geoms)`: `spec.delete(g)` if `g.type == mjGEOM_MESH`; then `for m in list(spec.meshes)`: `spec.delete(m)`). Return the spec. If compiling fails because a body is left without mass, give that body an explicit tiny inertial (`body.mass = 1e-3; body.inertia = [1e-6] * 3; body.explicitinertial = True`); FK and Jacobians don't depend on mass.

- [ ] **Step 4: Package it.** In `model/training.py`'s `write_package`, write `kinematics_spec().to_xml()` to `out / "kinematics.xml"`. In `model/package.py`'s `control.ik`, replace `"solved_every_physics_step": True` with:

```python
"solve": "once per policy step, from the joints measured at the step",
"interpolation": "joint target ramps linearly from the previous step's target (the measured joints on an episode's first step) to measured q + dq over policy_dt; IK is linearised at the measured joints",
"kinematics_file": "kinematics.xml",
"site": ik.frame_name,
"posture_target": [default_q[j] for j in ARM_JOINTS],  # mjlab's default when posture_target is None
"joint_limits_rad": ...,  # [[lo, hi]] per arm joint: the robot entity's soft_joint_pos_limits (factor 1.0 = jnt_range)
```

Remove the `robustness.not_modelled` entry "joint-target streaming: sim re-solves IK toward the desired pose every 1 ms" (no longer true). Also add `"kinematics.xml"` to whatever list of package files `write_package` or its test checks (grep `policy.onnx` in `model/` and `tests/`).

- [ ] **Step 5:** `uv run pytest -q`, all pass. **Commit** `feat: package kinematics.xml and the once-per-step IK contract`.

---

### Task 3: Run 5 (needs the user's go-ahead)

Run 4 (`6afd2524538676b8a59d06cf`) used per-substep IK. Its policy can't be deployed by this controller. Once Tasks 1 and 2 are in:

- [ ] Show the user run 4's results and the throughput numbers from Task 1, then ask to submit.
- [ ] `./build.sh`, upload as the next `rl-insert-train` version, submit with run 4's args, `run_name=c0.35-run5`, next model version. Commands: see the run 4 submission in `docs/superpowers/specs/` or the memory file `viam-ids.md`.

---

### Task 4: Controller kinematics and IK, with parity against training

**Files:**
- Create: `policy_controller/__init__.py` (empty), `policy_controller/core.py`
- Test: `tests/test_controller_parity.py`

`core.py` uses only `numpy` and `mujoco`. Port the math the training uses verbatim, as NumPy: `compute_pose_error` and `apply_delta_pose` from `.venv/lib/python3.12/site-packages/mjlab/utils/lab_api/math.py` (lines ~869 and ~922), and `DifferentialIKAction.compute_dq` (normal equations, joint-limit and posture terms, damping, `max_dq` clamp). Read those functions before writing; the parity test is the referee.

- [ ] **Step 1: Failing parity test**

```python
# tests/test_controller_parity.py
"""The controller's IK reproduces the training env's, step for step."""
import json

import numpy as np
import torch

from model.package import control_contract
from policy_controller.core import Kinematics
from rl_insert.robot import kinematics_spec
from rl_insert.task.checks import make_env


def test_ik_goal_matches_training(tmp_path):
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    (tmp_path / "kinematics.xml").write_text(kinematics_spec().to_xml())
    cfg = control_contract(env.cfg)  # {"control": ..., "action": ..., "robot": ..., "task": ...}
    kin = Kinematics(tmp_path / "kinematics.xml", cfg["control"])
    term = env.action_manager.get_term("ee_delta")
    gen = torch.Generator().manual_seed(0)
    for _ in range(10):
        a = torch.randn(4, 6, generator=gen)
        q = term._entity.data.joint_pos[:, term._joint_ids].numpy().copy()
        env.step(a)
        for i in range(4):
            q_goal = kin.step_goal(q[i], a[i].numpy(), cfg["action"])
            np.testing.assert_allclose(q_goal, term._q_to[i].numpy(), atol=1e-5)
```

`build_config` needs an exported ONNX (it reads `observation_names` metadata), so first split it: move the `action`, `control`, `robot` and `task` dicts into `control_contract(env_cfg, p)` in `model/package.py`, which `build_config` calls. No behaviour change; the existing package test must still pass.

- [ ] **Step 2: Run, expect FAIL** (no `policy_controller.core`).

- [ ] **Step 3: Implement** `Kinematics(xml_path, control)`:
  - loads the model, finds `control["ik"]["site"]` and the arm joints by `control["joint_names"]`;
  - `fk(q) -> (pos (3,), rot (3,3))` via `mj_kinematics` + `mj_comPos` (needed before `mj_jacSite`);
  - `jac(q) -> (jacp, jacr)` (3x6 each) via `mj_jacSite`;
  - `step_goal(q, a, action_cfg) -> q + dq`: desired pose from `fk(q)` and `a` scaled by `pos_scale_m` / `rot_scale_rad` (`apply_delta_pose`), then one DLS step at `q`.
  - It also returns or stores the desired pose `(p_des, quat_des)`; Task 5's `ik_error` needs it.
  Quaternions are `w, x, y, z`, as in mjlab.

- [ ] **Step 4:** test passes; **commit** `feat: controller FK and IK, matching training's to 1e-5`.

---

### Task 5: Controller observation, with parity

**Files:** modify `policy_controller/core.py`; extend `tests/test_controller_parity.py`.

`Policy(config, kinematics)` holds the episode state: previous joints, last action, desired pose. `observe(q, hole_mouth) -> np.ndarray (obs_dim,)` concatenates `config["observation"]["terms"]` in order, each from its recipe in `config.json` (`env_cfg.OBS_DOC`):
- `tip_rel_est_hole`: `p + R[:, 2] * protrusion - hole_mouth` (`protrusion` = `robot.nominal_tcp.protrusion_m`);
- `ee_rot6d`: `R[:, :2].T.reshape(6)`;
- `joint_pos`: `q - default_joint_pos`;
- `joint_vel`: `(q - q_prev) / policy_dt`, 0 on the first step;
- `last_action`: the previous raw action (0 at start);
- `ik_error`: `[p_des - p, axis_angle(q_des * q_m^-1)]`, 0 on the first step.
An unknown term name raises `ValueError(f"no recipe for observation term {name}")`: that is the shared-vocabulary contract.
`act(q, hole_mouth, action) -> q_goal` records the action, desired pose and `q` for the next step.

- [ ] **Step 1: Failing test**: run the env 10 steps with random actions; before each `env.step`, compare `policy.observe(q_i, hole_est_i)` with `env.observation_manager.compute()["actor"][i]` (atol 1e-5), where `hole_est = mdp.hole_mouth(env) + mdp.state(env).hole_bias`; then `policy.act(...)` with the same action. `make_env` has observation noise off. If env i was reset by the last step (`env.episode_length_buf[i] == 0`), call `policy.reset()` for it first (envs auto-reset mid-run; stale `q_prev` and `last_action` would fail the comparison for the wrong reason).
- [ ] **Step 2:** FAIL. **Step 3:** implement. **Step 4:** PASS. **Step 5: commit** `feat: controller observation, matching training's actor observation`.

---

### Task 6: Safety

**Files:** modify `policy_controller/core.py`; create `tests/test_controller_safety.py`.

Training never clips actions (`config.json` `action.clip: null`), so the caps must never bind on what the policy actually does, or they change its behaviour. Set the defaults from data: run the trained policy (run 5's `policy.onnx`, via onnxruntime) in the training env on CPU (64 envs, 20 episodes each, `make_env`) and log per step |a| per element, the TCP move and the largest joint step. Defaults = 2x the 99.9th percentile of each, rounded up; write the measured percentiles in a comment next to them. `workspace_mm` 30 (box half-size around the hole mouth; the nominal tip must stay inside), `joint_limits` from `config.json`.

`last_action` records the raw policy output, before the clip, as training does (the action manager stores the raw action). Document that in the code.
`Policy.act` applies, in order: clip the action; after IK, scale `dq` down so the TCP moves at most `max_step_mm` and no joint more than `max_joint_step_deg`; clamp to joint limits; raise `SafetyStop(reason)` if the goal's nominal tip leaves the workspace box.

- [ ] Tests, one each: an action of 100 is clipped; a goal past the per-step cap is scaled to exactly the cap; a hole 100 mm away raises `SafetyStop("workspace")`; replaying the logged rollout's actions, no cap binds on more than 0.1% of steps.
- [ ] Implement, pass, **commit** `feat: controller safety limits`.

---

### Task 7: Probe the streaming semantics on the sim arm (needs the sim running)

The SDK contract: `TrajectoryPoint.time` is from the start of the motion, strictly increasing, first point 0, positions in degrees. mujoco-sim's docs say "follows timestamped points in sim time; each batch plays straight after the last", which could mean per-batch times. The xArm module sends each point at anchor + time and skips the first point.

- [ ] Ask the user to start the rl-insert simulation (it's billed). Then a scratch script (scratchpad, not the repo) connects with the machine's API key, resets the cell (world `reset`), opens one `move_through_joint_positions_streamed` stream, sends batch 1 = points at 0, 50, 100 ms moving joint 6 by +2 degrees, waits 300 ms, sends batch 2 = points at 150, 200 ms (absolute) moving it back, and samples `get_joint_positions` every 20 ms.
- [x] Record in this plan file which interpretation holds (absolute from stream start, or relative per batch), and how late batches behave. Task 8 implements the one that holds. Ask before stopping the sim.

**Result (2026-11-18, mujoco-sim 0.16.1, scratchpad `stream_probe.py`).** Neither pure reading holds. Times are one timeline across batches (the sim interpolates linearly from each point to the next, the step taking the difference of their times), but that timeline does not run while the stream is starved: a batch that arrives after the last point has played starts *now*, from the last point, and plays at its own time offsets from that point.
- Batch 1 (0/50/100 ms, j6 +2 deg): j6 reached +2.00 by 0.178 s (latency included).
- Batch 2 sent at 0.301 s, times 150/200: back to 0 by 0.385 s. Absolute would also play it at once; per-batch-relative from its arrival would wait until ~0.45 s.
- Batch 3 sent at 1.001 s, times 1500/1600: moved *at once* and slowly, 0.08 deg at 1.055 s to 1.97 deg at 2.285 s, i.e. over the 1.3 s from the previous last point (200 ms) to 1500 ms. Absolute would wait until 1.5 s; per-batch would start at 2.5 s.
- One empty `TrajectoryUpdate` per batch, on arrival.

For the controller: streaming one tick ahead never starves the queue, so stream-absolute times (Task 8's `point_times`) are what the sim plays, one tick behind the tick that sent them. A late tick that starves the queue does not jump: the next batch plays from the last point over the time gap, i.e. the ramp just stretches, and the timeline after that runs late by the starved time (the stream-absolute anchor drifts). Overruns therefore cost latency on the sim rather than a jump; keep `max_overruns` small. The xArm's driver (anchor + time, late points sent at once) is a different behaviour; re-probe on the real arm.

---

### Task 8: The Viam service

**Files:** create `policy_controller/service.py`, `policy_controller/main.py`; test the pure parts in `tests/test_controller_safety.py`.

Config attributes: `arm`, `gripper`, `mlmodel` (required dependency names); `package_dir` (the ML model package path, e.g. `${packages.ml_model.rl-insert}`); `hole_mouth_mm` ([x, y, z], arm base frame); optional `action_clip`, `max_step_mm`, `max_joint_step_deg`, `workspace_mm`, `timeout_s` (default 10, the training episode), `max_overruns` (default 3). `validate_config` returns the three dependencies and rejects a missing `package_dir` or `hole_mouth_mm`.

DoCommand:
- `{"command": "start"}`: requires the loop idle; resets `Policy`; starts the loop task. Returns `{"status": "running"}`.
- `{"command": "stop"}`: cancels the loop, calls `arm.stop()`.
- `{"command": "status"}`: `{"state": "idle|running|succeeded|failed", "reason": ..., "steps": n, "overruns": n, "last_obs": [...]}`.
- `{"command": "pre_insert_pose"}`: the flange pose (mm, degrees, OV) that puts the nominal tip 15 mm above `hole_mouth_mm`, TCP straight down, from `Kinematics` inverse of the TCP offset. ponytail: compute by IK on `kinematics.xml` from `default_joint_pos`.

Loop (one async task, 20 Hz on a monotonic clock):
1. `q = radians(await arm.get_joint_positions())`.
2. `obs = policy.observe(q, hole_m)`; `out = await mlmodel.infer({"obs": obs[None].astype(float32)})`; `a = out["actions"][0]` (names from `config["model"]["inputs"/"outputs"]`).
3. `q_goal = policy.act(q, hole_m, a)`; on `SafetyStop`, fail with its reason.
4. Five points ramping from the previous commanded goal (the measured `q` on the first tick) to `q_goal`, at +10..+50 ms with fractions 0.2, 0.4 ... 1.0 (training's ramp reaches k/50 at substep k), in degrees, timed per Task 7's finding (one step ahead of the tick, as the design spec's loop describes), into an `asyncio.Queue` that feeds the one long-lived `move_through_joint_positions_streamed` call. The first batch starts with the current joints at time 0.
5. Success: the nominal tip is at least `task.success_depth_m` below the mouth and within `task.success_centred_m` laterally (check the exact keys in `config.json`). On success, end the batch iterator (half-close) and report `succeeded`. On failure (timeout, overruns, inference error, stream error, safety), cancel the stream task, `await arm.stop()`, report `failed` with the reason.
6. Late tick (its window already started): skip ahead, count an overrun.

`main.py`: `Module.run_from_registry()` with the model registered via `EasyResource` (see the viam-python skill for the pattern).

- [ ] Unit-test `pre_insert_pose` (FK of its joints puts the tip 15 mm above the mouth) and the point timing function (pure: tick index -> list of times). The loop itself is verified in Task 10.
- [ ] **Commit** `feat: policy controller service`.

---

### Task 9: Package and upload the module

**Files:** `policy_controller/meta.json`, `run.sh`, `requirements.txt` (`viam-sdk>=0.85`, `mujoco==4.0.1` (the version that writes `kinematics.xml`), `numpy`), `build_controller.sh` at the repo root.

- [ ] `meta.json`: module id `devrel:rl-policy`, visibility private, one model `devrel:rl-policy:controller` (API `rdk:service:generic`), entrypoint `run.sh`. `run.sh` does `cd "$(dirname "$0")/.."` (so the package imports), creates a venv on first run, installs `requirements.txt`, then `exec python -m policy_controller.main "$@"`.
- [ ] `build_controller.sh` tars `policy_controller/` + `meta.json` + `run.sh` + `requirements.txt` into `dist/rl-policy-module.tar.gz`.
- [ ] Ask the user before the first registry upload (it creates a module in DevRelDemos). Then `viam module create --name rl-policy --org-id 16518049-9dd3-479f-9644-c5d112aa42d8` if needed, and `viam module upload --version 0.1.0 --platform linux/amd64 dist/rl-policy-module.tar.gz` (check the sim sandbox's platform first with `read_part_status` or the part's platform field).
- [ ] **Commit** `build: package the policy controller module`.

---

### Task 10: Sim-to-sim on rl-insert (needs the sim running)

- [ ] Check the servo first: the cell's arm comes from `scripts/export_sim_scene.py` `arm_spec`, which uses training's actuators (kp, dampratio from `ArmParams`). Confirm the exported `xarm6.xml` actuators match `config.json` `control.joint_servo_sim`, and note that the sim cell has no gain randomisation (training's is +/-20%), so it tests the nominal servo only.
- [ ] Add to the rl-insert machine (ask first; it changes the machine's config): the `rl-insert` ML model package (run 5's version), a `viam:mlmodel:onnx-cpu` service `policy-model` with `model_path` `${packages.ml_model.rl-insert}/policy.onnx`, and the controller service with `arm`, `gripper`, `mlmodel`, `package_dir: ${packages.ml_model.rl-insert}`, `hole_mouth_mm: [400, 0, 50]`.
- [ ] Episode script (scratchpad): the reset recipe in `scripts/export_sim_scene.py`'s docstring (world `reset`, gripper Open, `pause`, `set_prop_pose` peg at (0.4, 0, 0.069), gripper Grab, `play`), then controller `start`, poll `status` until done. 20 episodes; record succeeded / failed with reasons and steps.
- [ ] Pass bar: success within 20 points of run 5's training success at difficulty 0.3 (the cell's start is the nominal one). Below that, compare the controller's first few observations with the training env's for the same start (log them) before changing anything.
- [ ] Write the results to `docs/superpowers/specs/2026-11-xx-sim-to-sim-results.md`; **commit**.

---

## Not in this plan

- The real arm (validation ladder rungs 3 and 4).
- A generic, task-agnostic controller and a shared vocabulary package: after the reach tutorial (design doc, order of work).
- `hole_frame` from the frame system, and hole-pose estimation from a camera.
