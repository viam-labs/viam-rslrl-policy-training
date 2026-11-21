"""Pollen's own CPU reference for the MicroDuck walk policy, headless: microduck_rl's scripts/infer_policy.py
code (load_mujoco_with_bam, PolicyInference), its reset sequence and its 4 x 5 ms steps per action,
driven by a command script instead of the keyboard and run as fast as it can.

    uv run --project <mujoco-sim worktree> --with onnxruntime python scripts/duck_reference.py \
        <microduck_rl checkout> <duck-sim dir> <policy.onnx> <out.json> [--golden tests/data/duck_golden.json] [--stats <walk config.json>] [--script script.json]

The scene is the duck-sim package's duck.xml, so both sides of the parity check simulate the same file
with the same MuJoCo build. Records, every 0.1 s of sim time: t, trunk position and quaternion, gravity.
--golden saves ticks 0-24 and 600-624 (the turning segment): raw inputs and the observation infer_policy built, for the obs test.
--stats writes action_bound (2 x p99.9 |action|) and max_step_rad (2 x p99.9 |target change per tick|)
into the walk package's config.json safety block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# The command script: (seconds, vx, vy, wz). velstand has a dead zone (Pollen's docs/velstand_policy.md,
# microduck_velstand_env_cfg.py:344-380): measured on infer_policy's own scene, vx 0.15-0.2, |vx| <= 0.3
# backwards, and wz +-0.5 all stand still; vx 0.4 -> ~0.17 m/s, vx -0.4 -> ~-0.14, vx 0.4 + wz +-0.6 ->
# ~+-0.38 rad/s. So every moving segment uses values that walk (spec, milestone 2 decisions).
SCRIPT = [(2.0, 0.0, 0.0, 0.0), (8.0, 0.4, 0.0, 0.0), (5.0, 0.4, 0.0, 0.6), (5.0, 0.4, 0.0, -0.6),
          (4.0, 0.0, 0.0, 0.0), (6.0, -0.4, 0.0, 0.0)]
DEFAULT_SCRIPT = list(SCRIPT)
DURATION = sum(seg[0] for seg in SCRIPT)
TRUNK_Z = 0.125  # infer_policy.py:1376-1383


def opt_summary(model) -> dict:
    o = model.opt
    return {"integrator": int(o.integrator), "cone": int(o.cone), "impratio": float(o.impratio),
            "iterations": int(o.iterations), "tolerance": float(o.tolerance), "timestep": float(o.timestep)}


def set_script(script) -> None:
    """Replace SCRIPT in place (importers hold the list) and recompute DURATION."""
    global DURATION
    SCRIPT[:] = [tuple(map(float, seg)) for seg in script]
    DURATION = sum(seg[0] for seg in SCRIPT)


def command_at(t: float) -> tuple[float, float, float]:
    for seconds, vx, vy, wz in SCRIPT:
        if t < seconds:
            return vx, vy, wz
        t -= seconds
    return SCRIPT[-1][1:]


def run(checkout: Path, duck_xml: Path, onnx: Path, golden: Path | None = None) -> dict:
    sys.path.insert(0, str(checkout / "scripts"))
    import mujoco
    from infer_policy import BAM_VIN_MIN, PolicyInference, load_bam_model, load_mujoco_with_bam

    bam_model = load_bam_model(200.0, 7.4, None)
    # duck.xml has no floor (mujoco-sim adds one when it builds the scene): wrap it with the same plane,
    # next to duck.xml so the include and meshdir resolve, and remove the wrapper so it never ships.
    scene = duck_xml.parent / "_reference_scene.xml"
    scene.write_text('<mujoco model="reference">\n  <include file="duck.xml"/>\n  <worldbody>\n'
                     '    <geom name="floor" type="plane" size="0 0 0.05"/>\n  </worldbody>\n</mujoco>\n')
    try:
        model, data, bam_ctrl, _ = load_mujoco_with_bam(str(scene), bam_model, 0.005, 0.1, BAM_VIN_MIN)
    finally:
        scene.unlink()
    print("reference opt (as compiled):", opt_summary(model))
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # mujoco-sim forces this
    policy = PolicyInference(model, data, walking_onnx_path=str(onnx), bam_ctrl=bam_ctrl,
                             use_projected_gravity=True, new_cmd_obs=True)
    free = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
    qa = model.jnt_qposadr[free]
    data.qpos[qa:qa + 3] = (0.0, 0.0, TRUNK_Z)
    data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
    for i, idx in enumerate(policy.joint_qpos_indices):
        data.qpos[idx] = policy.default_pose[i]
    bam_ctrl.reset(data.qpos)
    policy.set_position_targets(policy.default_pose)
    mujoco.mj_forward(model, data)

    samples, ticks, actions, steps = [], [], [], []
    prev_target = np.array(policy.default_pose, float)
    tick = 0
    while data.time < DURATION - 1e-9:
        policy.vel_cmd[:] = command_at(data.time)
        policy._update_command()
        if golden is not None and (tick < 25 or 600 <= tick < 625):  # rest, then vx 0.4 + wz 0.6 at t = 12 s
            ticks.append({
                "gyro": policy.get_base_ang_vel().tolist(), "gravity": policy.get_projected_gravity().tolist(),
                "q": data.qpos[policy.joint_qpos_indices].tolist(), "dq": data.qvel[policy.joint_qvel_indices].tolist(),
                "last_action": policy.last_action.tolist(), "command": policy.vel_cmd.tolist(),
                "obs": policy.get_observations().tolist(),
            })
        action = policy.infer()
        policy.apply_action(action)
        target = np.array(policy.default_pose) + action * policy.action_scale
        actions.append(np.abs(action))
        steps.append(np.abs(target - prev_target))
        prev_target = target
        for _ in range(4):
            bam_ctrl.update()
            mujoco.mj_step(model, data)
        tick += 1
        if tick % 5 == 0:  # every 0.1 s
            quat = data.qpos[qa + 3:qa + 7].copy()
            samples.append({"t": round(float(data.time), 4), "trunk_pos": data.qpos[qa:qa + 3].tolist(),
                            "trunk_quat_wxyz": quat.tolist(), "gravity": policy.get_projected_gravity().tolist()})
    if golden is not None:
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_text(json.dumps({"default_pose": list(map(float, policy.default_pose)), "ticks": ticks}, indent=1))
    return {
        "script": [list(seg) for seg in SCRIPT],
        "samples": samples,
        "action_p999": float(np.percentile(np.concatenate(actions), 99.9)),
        "step_p999": float(np.percentile(np.concatenate(steps), 99.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkout", type=Path)
    ap.add_argument("duck_sim", type=Path)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--golden", type=Path)
    ap.add_argument("--stats", type=Path, help="walk package config.json to write the safety block into")
    ap.add_argument("--script", type=Path, help="JSON list of [seconds, vx, vy, wz] replacing the default command script")
    args = ap.parse_args()
    if args.script:
        set_script(json.loads(args.script.read_text()))
    result = run(args.checkout.resolve(), (args.duck_sim / "duck.xml").resolve(), args.onnx.resolve(), args.golden)
    args.out.write_text(json.dumps(result))
    if args.stats:
        cfg = json.loads(args.stats.read_text())
        cfg["safety"] = {"action_bound": round(2 * result["action_p999"], 3), "max_step_rad": round(2 * result["step_p999"], 3)}
        args.stats.write_text(json.dumps(cfg, indent=2) + "\n")
    last = result["samples"][-1]
    print(f"ran {last['t']:.1f} s sim; final trunk z {last['trunk_pos'][2]:.3f}, gravity z {last['gravity'][2]:.3f}; "
          f"action p99.9 {result['action_p999']:.3f}, step p99.9 {result['step_p999']:.3f}")


if __name__ == "__main__":
    main()
