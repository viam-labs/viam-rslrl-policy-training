"""Measure what a trained policy actually does, to set policy_controller's safety caps (they must never bind).

uv run python -m scripts.policy_envelope <package_dir> [out.npz] [num_envs=64] [episodes_per_env=20]

Runs package_dir/policy.onnx (onnxruntime, CPU) in the training env. Per step it logs |a| per element, the TCP move
(m) and the largest joint step (rad) of the controller's own unsafe IK goal at each env's measured q, prints the
99.9th percentiles and suggested caps (2x, rounded up), and saves raw action / q / hole / episode-start for
tests/test_controller_safety.py (POLICY_ENVELOPE_NPZ).
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

from model.package import control_contract
from policy_controller.core import Kinematics
from rl_insert.robot import kinematics_spec
from rl_insert.task import mdp
from rl_insert.task.checks import make_env


def main(package_dir: str, out: str | None = None, num_envs: int = 64, episodes: int = 20) -> None:
    pkg = Path(package_dir)
    sess = ort.InferenceSession(str(pkg / "policy.onnx"), providers=["CPUExecutionProvider"])
    env = make_env(num_envs=num_envs, device="cpu")
    config = json.loads((pkg / "config.json").read_text())
    contract = control_contract(env.cfg)
    for key in ("action", "control", "robot"):
        if key not in config or "joint_limits_rad" not in config.get("control", {}).get("ik", {}):
            print(f"config.json lacks the current {key} contract; using the repo's control_contract")
            config = config | contract
            break
    xml = Path(tempfile.mkdtemp()) / "kinematics.xml"
    xml.write_text(kinematics_spec().to_xml())
    kin = Kinematics(xml, config["control"])
    term = env.action_manager.get_term("ee_delta")

    env.reset()
    done = np.zeros(num_envs, int)
    rec = {k: [] for k in ("action", "q", "hole", "first")}
    moves, jsteps = [], []
    while done.min() < episodes:
        obs = env.observation_manager.compute()["actor"].numpy().astype(np.float32)
        a = np.concatenate([sess.run(None, {"obs": o[None]})[0] for o in obs])  # the export has batch size 1
        q = term._entity.data.joint_pos[:, term._joint_ids].numpy().copy()
        hole = (mdp.hole_mouth(env) + mdp.state(env).hole_bias - env.scene.env_origins).numpy()
        first = (env.episode_length_buf == 0).numpy()
        for i in range(num_envs):
            goal = kin.step_goal(q[i], a[i], config["action"])
            moves.append(np.linalg.norm(kin.fk(goal)[0] - kin.fk(q[i])[0]))
            jsteps.append(np.abs(goal - q[i]).max())
        for k, v in zip(rec, (a, q, hole, first)):
            rec[k].append(v)
        _, _, term_, trunc, _ = env.step(torch.as_tensor(a))
        done += (term_ | trunc).numpy()

    absa = np.abs(np.concatenate(rec["action"]))
    p = lambda x: np.percentile(x, 99.9)
    ceil2 = lambda x: float(np.ceil(2 * x))
    steps = len(moves)
    print(f"{steps} steps ({num_envs} envs, >= {episodes} episodes each)")
    print("|a| p99.9 per element:", np.percentile(absa, 99.9, axis=0).round(3), f"-> action_clip {ceil2(p(absa)):.0f}")
    print(f"TCP move p99.9 {p(moves) * 1e3:.3f} mm -> max_step_mm {ceil2(p(moves) * 1e3):.0f}")
    print(f"joint step p99.9 {np.rad2deg(p(jsteps)):.3f} deg -> max_joint_step_deg {ceil2(np.rad2deg(p(jsteps))):.0f}")
    if out:
        np.savez_compressed(out, **{k: np.concatenate(v) for k, v in rec.items()})
        print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or [None]), *map(int, sys.argv[3:5]))
