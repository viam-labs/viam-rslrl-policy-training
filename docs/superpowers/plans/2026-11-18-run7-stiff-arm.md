# Run 7: stiff arm, pivoting grasp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrain the insertion policy against a sim that matches what armfarm15 showed in first contact (plan 4 Task 8): a stiff arm, give that the joints don't see, a grasp that pivots under side load, and a lower force budget.

**Architecture:** Four changes to the training task, all parameters on existing code paths: (1) arm servo kp 2000 → 1e4 with armature and a wider per-episode gain range; (2) the grasp weld's rotational weight (`torquescale`) sampled per episode from [3, 10], so the peg pivots in the hand under side load like the real G2's friction pinch; (3) side-load budget cut: force limit 40 → 20 N, penalty from 15 → 8 N; (4) slip termination 10 → 3 mm, so pivoting the peg is failure. The policy's observations are unchanged (the controller needs no change). Values are estimates (no dial indicator: real stiffness unmeasured); the ranges are wide on purpose.

**Tech Stack:** mjlab (MuJoCo Warp), RSL-RL, Viam training jobs; pytest.

**Why (evidence, plan 4 Task 8 and the 2026-11-18 stiffness study):**
- Sim tip stiffness at kp 2000: x 5.5, y 11.0, z 6.7 N/mm. Real xArm6 gear-only upper bound: x 10-30, y 45-100, z 17-59 N/mm (Harmonic Drive catalogue, gear sizes inferred). Matching x/z needs kp ~3.7e3-1.8e4.
- Episode 1: with the peg 12 mm in a 0.25 mm-clearance hole, the arm's nominal tip moved 5.6 mm in -x. The give was in the grasp (the peg ended skewed in the pads), invisible to the joints. In sim the weld pins the tip, so the policy learned that leaning is cheap.
- At kp 2000 with no armature, J6 limit-cycles (8.5 mrad p-p) under Euler at 1 ms; armature 0.05-0.2 makes kp 2e3-2e4 converge in 9-76 ms.

**Out of scope:** measured effort / F/T in observations; a grasp breakaway (frictionloss) model; the -x approach bias as such (the force and slip costs should remove it; check in Task 6).

---

## File map
- `rl_insert/robot.py` — `ArmParams` (kp, armature), `load_xarm6` applies armature.
- `rl_insert/task/env_cfg.py` — `TaskParams`: `pd_gain_scale`, `grasp_torquescale`, `force_limit`, `force_penalty_from`, `max_slip`.
- `rl_insert/task/mdp.py` — `ResetPegInHand` samples torquescale per episode.
- `model/package.py` — `config.json`: `joint_servo_sim` gains armature; `robustness` lists the torquescale range; `not_modelled` updated.
- Tests: `tests/test_robot.py`, `tests/test_grasp_hold.py`, new `tests/test_grasp_pivot.py`.

---

### Task 1: Stiff arm servo with armature

**Files:** Modify `rl_insert/robot.py:19-50`; Test `tests/test_robot.py`

- [ ] **Step 1: Failing test** (append to `tests/test_robot.py`)

```python
def test_stiff_servo_settles_without_limit_cycle():
    """kp 1e4 x [0.5, 2] with armature: every joint settles from a 10 mrad step (Euler, 1 ms).
    Without armature J6 limit-cycles even at kp 2000 (2026-11-18 stiffness study)."""
    from rl_insert.robot import ArmParams, GripperParams

    for scale in (0.5, 1.0, 2.0):
        spec = load_xarm6(ArmParams(kp=ArmParams().kp * scale))
        add_gripper(spec, GripperParams())
        m = spec.compile()
        m.opt.timestep, m.opt.integrator = 0.001, mujoco.mjtIntegrator.mjINT_EULER
        d = mujoco.MjData(m)
        q0 = np.array([0, -0.0728, -0.6831, 0, 0.7560, 0])
        d.qpos[:6] = q0
        d.ctrl[:6] = q0 + 0.01
        for _ in range(300):
            mujoco.mj_step(m, d)
        tail = []
        for _ in range(200):
            mujoco.mj_step(m, d)
            tail.append(d.qpos[:6].copy())
        tail = np.array(tail)
        assert np.abs(tail - (q0 + 0.01)).max() < 1e-3, scale  # within 1 mrad
        assert np.ptp(tail, axis=0).max() < 2e-4, scale  # no limit cycle
```

- [ ] **Step 2:** `uv run pytest tests/test_robot.py::test_stiff_servo_settles_without_limit_cycle -v` — expect FAIL (kp 2000, no armature: J6 cycles).

- [ ] **Step 3: Implement** in `rl_insert/robot.py`

```python
@dataclass(frozen=True)
class ArmParams:
    # ponytail: estimate. Real xArm6 tip stiffness is unmeasured (no dial indicator, 2026-11-18); gear-only
    # bounds need kp ~3.7e3-1.8e4 for x/z. Measure with plan 4's push-gauge procedure and retune.
    kp: float = 1e4
    dampratio: float = 1.0
    # Reflected rotor inertia guess; also keeps the explicit (Euler) servo stable at high kp.
    armature: tuple[float, ...] = (0.1, 0.1, 0.1, 0.05, 0.05, 0.05)
```

and in `load_xarm6`, before the actuator loop:

```python
    for name, a in zip(ARM_JOINTS, arm.armature):
        spec.joint(name).armature = a
```

- [ ] **Step 4:** run the test — PASS. Then `uv run pytest tests/ -q` — note any contact/grasp test that moved (Task 3 revisits grasp tests).

- [ ] **Step 5: Commit** `feat(sim): stiff arm servo (kp 1e4) with armature`

### Task 2: Wider servo gain range

**Files:** Modify `rl_insert/task/env_cfg.py:75`

- [ ] **Step 1:** `pd_gain_scale: tuple[float, float] = (0.5, 2.0)  # kp 5e3-2e4: covers the real arm's estimated range`
- [ ] **Step 2:** `uv run pytest tests/ -q` — PASS (no test pins the range; `tests/test_curriculum.py` and `test_scene.py` exercise the event).
- [ ] **Step 3: Commit** `feat(task): servo gains x[0.5, 2]`

### Task 3: Grasp pivots under side load

The weld's `torquescale` sets its rotational stiffness relative to translation. At 10 the peg can't pivot (p95 0.23 mm tip swing); at 1, pad torque alone swung the tip 7 mm. The real pinch held in free air (air runs) but pivoted ~5 mm under contact side load. Sample it per episode.

**Files:** Modify `rl_insert/task/env_cfg.py` (`TaskParams`), `rl_insert/task/mdp.py:31,168`; Create `tests/test_grasp_pivot.py`; Modify `tests/test_grasp_hold.py`

- [ ] **Step 1: Failing tests**

`tests/test_grasp_pivot.py`:

```python
"""The grasp's pivot stiffness spans the real G2's: holds in free air, pivots mm under side load."""

from dataclasses import replace

import torch

from rl_insert.task.checks import make_env, tip_lateral_under_load
from rl_insert.task.env_cfg import TaskParams


def _env(ts):
    # max_slip off: a soft grasp under 20 N would otherwise reset mid-measurement
    return make_env(num_envs=16, device="cpu", p=replace(TaskParams(), grasp_torquescale=(ts, ts), max_slip=1.0))


def test_soft_grasp_pivots_mm_under_20N():
    assert tip_lateral_under_load(_env(TaskParams().grasp_torquescale[0]), force_n=20.0) > 2.0


def test_stiff_grasp_barely_moves_under_20N():
    assert tip_lateral_under_load(_env(TaskParams().grasp_torquescale[1]), force_n=20.0) < 1.0


def test_torquescale_sampled_per_episode():
    env = make_env(num_envs=64, device="cpu")
    env.reset()
    ts = env.sim.model.eq_data[:, env.insert_state.weld_id, 10]
    lo, hi = TaskParams().grasp_torquescale
    assert ts.min() >= lo and ts.max() <= hi and torch.unique(ts).numel() > 32
```

In `tests/test_grasp_hold.py`, run the hold check at the soft end:

```python
from dataclasses import replace
from rl_insert.task.env_cfg import TaskParams

def test_grasp_holds_for_2s():
    lo = TaskParams().grasp_torquescale[0]
    env = make_env(num_envs=64, device="cpu", p=replace(TaskParams(), grasp_torquescale=(lo, lo)))
    ...  # rest unchanged
```

- [ ] **Step 2:** run them — FAIL (`grasp_torquescale` and `tip_lateral_under_load` don't exist).

- [ ] **Step 3: Implement**

`TaskParams` (after `grasp_solref`):

```python
    # ponytail: estimate. Weld rotation weight, per episode: low = the peg pivots in the pads under side
    # load (real G2 pinch, plan 4 Task 8), high = rigid. Retune from a measured pivot force.
    # Plan review (CPU, kp 2000, 20 N at the tip): ts 10 -> 0.56 mm, ts 2 -> 8.8 mm in the hand; real ~5 mm.
    grasp_torquescale: tuple[float, float] = (3.0, 10.0)
```

`mdp.py`: keep `GRASP_TORQUESCALE = 10.0` only as the placeholder default in `_add_grasp_weld`; in `ResetPegInHand.__call__` replace the `tail` line:

```python
        ts = _uniform(n, *p["grasp_torquescale"], dev)
        tail = torch.stack([ts, torch.zeros(n, device=dev)], dim=-1)
```

`checks.py` (next to `grasp_hold_slip`):

```python
def tip_lateral_under_load(env: ManagerBasedRlEnv, force_n: float, seconds: float = 1.0) -> float:
    """Max over envs of the peg tip's move in the hand (mm, TCP-frame x) with `force_n` pushed on the tip
    along TCP x: the grasp's pivot + weld compliance. Measured in the TCP frame because a zero StepIK
    action re-targets from the measured joints each step, so under load the arm itself creeps."""
    env.reset()
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    for _ in range(10):
        env.step(zero)  # settle
    peg = env.scene["peg"]
    bid = int(peg.indexing.body_ids[0])
    before = mdp.peg_in_hand(env)[:, 0].clone()
    for _ in range(int(seconds / env.step_dt)):
        _, rot = mdp.tcp_pose(env)
        f = rot[:, :, 0] * force_n  # TCP x in world
        # xfrc acts at the body's COM (~30 mm up the peg); add the moment that puts it at the tip
        tau = torch.cross(mdp.true_tip(env) - env.sim.data.xipos[:, bid], f, dim=-1)
        peg.write_external_wrench_to_sim(f.unsqueeze(1), tau.unsqueeze(1))
        _, _, terminated, truncated, _ = env.step(zero)
        assert not (terminated | truncated).any(), "an env reset mid-measurement"
    zero3 = torch.zeros(env.num_envs, 1, 3, device=env.device)
    peg.write_external_wrench_to_sim(zero3, zero3)  # the wrench persists until overwritten
    return float((mdp.peg_in_hand(env)[:, 0] - before).abs().max()) * 1000
```

(API checked in review: `Entity.write_external_wrench_to_sim(forces, torques, env_ids=None, body_ids=None)`, shapes (n, bodies, 3); `peg_in_hand` is the critic term `[tip pos in TCP frame (3), rot6d]`. Match `env.step`'s return tuple to mjlab's; `grasp_hold_slip` does the same no-reset check.)

- [ ] **Step 4:** run the three new tests and `test_grasp_hold.py` — PASS. If the soft end fails the hold test (creep > 0.5 mm in free air), raise the low end (3 → 4) until it holds; if `test_soft_grasp_pivots_mm_under_20N` then fails, report both numbers instead of forcing it: the range is the decision.

- [ ] **Step 5: Commit** `feat(task): grasp pivot stiffness sampled per episode`

### Task 4: Force and slip budgets

**Files:** Modify `rl_insert/task/env_cfg.py:66-68`

- [ ] **Step 1:**

```python
    force_limit: float = 20.0  # N, terminate. Was 40: armfarm15 tripped C31 (sensitivity 3) under side load
    force_penalty_from: float = 8.0  # N
    max_slip: float = 0.003  # m: pivoting the peg in the hand is failure (was 0.01)
```

- [ ] **Step 2:** `uv run pytest tests/ -q` — PASS (no test pins these; `test_peg_slipped.py` passes its own `max_slip`). Expect: with the soft grasp, ~15-20 N of side load ends the episode by slip before the force limit. That is the intent (pivoting is failure); Task 6 Step 4 is the fallback if it stalls learning.
- [ ] **Step 3: Commit** `feat(task): 20 N force limit, 3 mm slip termination`

### Task 5: Contract in config.json

**Files:** Modify `model/package.py:95` (`joint_servo_sim`), `155-173` (`robustness`), `174-181` (`grasp_model`); Test: existing package tests (`uv run pytest tests/ -k package -q`, or `tests/test_training_runtime.py`)

- [ ] **Step 1:**
  - `joint_servo_sim`: `{"kp": arm.kp, "dampratio": arm.dampratio, "armature": list(arm.armature), "gravity_compensated": True}`
  - `robustness`: add `"grasp_torquescale": list(p.grasp_torquescale)`.
  - `not_modelled`: replace `"real grasp slip (the grasp is a compliant weld)"` with `"grasp slip along the peg axis and breakaway (the grasp is a compliant weld; it pivots, it doesn't slide)"`.
  - `grasp_model` text: mention the per-episode torquescale.
- [ ] **Step 2:** tests PASS; the controller reads none of these keys (check: `grep -n joint_servo_sim\|grasp_torquescale policy_controller/`), so no controller change.
- [ ] **Step 3: Commit** `feat(model): config.json records run 7's servo and grasp model`

### Task 6: Probe, then run 7
**Progress 2026-11-18:** Tasks 1-5 committed (ea9dff5, 0f668ec, cb5c7b4, 0f7c74a, 75dcf36; suite 85 passed). Measured grasp pivot under 20 N at the tip: torquescale 3 -> 4.87 mm, 10 -> 0.63 mm (CPU). Script v8 uploaded. Probe job `6afe620aa2b9fca699211449` (A100): grasp hold settle 0.12 / creep 0.02 mm (pass), 13.2k policy steps/s at 4096 envs, video uploaded. Run 7 job `6afe6593b2c0e1b5b559b180`: rl-insert v7, clearance 0.35 mm, 1000 iterations, run name `c0.35-run7`, container `35da1a78`.

Follow `docs/superpowers/plans/2026-11-17-peg-insert-training.md` Tasks 6-8 (container, script upload, probe, run), DevRelDemos org. Never re-push a registered tag; bump the script version (run 6 was script v7, so v8).

- [ ] **Step 1:** local smoke: `uv run python -m model.training --mode probe` on CPU (or the probe job) — check throughput didn't collapse (armature and kp change nothing per step, but report it), grasp hold, and that fail-fast passes.
- [ ] **Step 2:** upload script v8; submit the probe job; read `probe.json`.
- [ ] **Step 3:** submit run 7 with run 6's `max_iterations` and the same curriculum. Watch for: success rate (expect slower to climb than run 6: less force budget), `peg_slipped` and `excess_force` termination rates, and the -x lean (mean tip x minus hole x below the mouth in rollouts; run 6/air runs: about -1.6 mm at the mouth, -10 mm unconstrained).
**Run 7 result 2026-11-18: failed fast** (no success in 300 iterations, 6.3 s/iteration). Terminations per iteration: it 0 excess_force 76.6, peg_slipped 70.8, timeout 10.3; it 50: 9.8 / 0 / 16.2; it 299: 1.8 / 0 / 20.1, episode length 198/200, keypoint_fine 0: the policy learned to hover short of the hole (runs 4-5's trap, now from the force termination). Scripted straight descent at 1 mm/step onto the mouth (CPU, 16 envs, difficulty 0, no termination; scratchpad `straight_insert.py`): peak peg-hole force p50/max 18.9/32.6 N at kp 2000 vs 65.6/160.4 N at kp 1e4. A blocked step now costs ~3.5x the force, and the limit halved: any rim contact at exploring speeds terminates. The slip termination only mattered in the first iterations.
- [ ] **Step 4:** if run 7 stalls (success < 50 % after the iteration where run 6 passed 80 %), first suspect: the 3 mm slip termination with the soft grasp. Widen `grasp_torquescale` low end to 3 or `max_slip` to 5 mm, one at a time, as run 7b.

### Task 7: Back to the controller and the real arm

- [ ] Recompute the controller's safety caps from run 7 rollouts (as `1ab3a80` did from run 6: 2x p99.9 of the envelope), update `policy_controller/core.py` `DEFAULTS`, bump the module.
- [ ] Sim-to-sim on rl-insert (plan 3 Task 10 scripts: `sim_episodes.py`, `sim_offnominal.py`): the sim cell's servo should take run 7's `joint_servo_sim` (kp, armature).
- [ ] armfarm15: recalibrate by the grip-in-hole method (plan 4 Task 6 result), then plan 4 Task 8 again. Stop on any C31 or `slip`.
- [ ] When a dial indicator is available: plan 4's push-gauge measurement, then retune `ArmParams.kp` and `grasp_torquescale`.
