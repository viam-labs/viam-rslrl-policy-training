"""Check a model package's policy.onnx against its checkpoint.pt on one real observation.

uv run python -m scripts.onnx_check <package_dir>    # model.training's --model_output_directory
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from rl_insert.task import PegInsertRunner, TaskParams, peg_insert_env_cfg, peg_insert_ppo_cfg


def main(package_dir: str) -> None:
    pkg = Path(package_dir)
    ckpt, onnx_path = pkg / "checkpoint.pt", pkg / "policy.onnx"
    clearance = json.loads((pkg / "config.json").read_text())["task"]["clearance_m"]

    model = onnx.load(onnx_path)
    print(
        "inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input]
    )
    print(
        "outputs:",
        [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in model.graph.output],
    )
    ops = [n.op_type for n in model.graph.node]
    print("first ops (normalizer = Sub/Div before the first Gemm):", ops[:6])
    print("metadata:", {p.key: p.value[:80] for p in model.metadata_props})

    env_cfg = peg_insert_env_cfg(TaskParams(clearance=clearance), play=True)
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device="cpu"))
    runner = PegInsertRunner(env, asdict(peg_insert_ppo_cfg()), log_dir=None, device="cpu")  # no logs written
    runner.load(str(ckpt))
    policy = runner.get_inference_policy("cpu")

    for _ in range(3):  # a few random steps, so velocities, last action and IK error are non-zero
        obs, *_ = env.step(torch.rand(1, 6) * 2 - 1)
    with torch.no_grad():
        torch_act = policy(obs).numpy()
    sess = ort.InferenceSession(str(onnx_path))
    onnx_act = sess.run(None, {"obs": obs["actor"].numpy().astype(np.float32)})[0]
    print("torch:", torch_act.round(5), "\nonnx: ", onnx_act.round(5))
    err = np.abs(torch_act - onnx_act).max()
    print(f"max abs diff: {err:.2e}")
    assert err < 1e-5

    # The normalizer must matter: running the MLP on raw obs (normalizer bypassed) differs.
    raw = policy.mlp(obs["actor"]).detach().numpy()
    print("diff vs un-normalized MLP:", np.abs(raw - onnx_act).max())


if __name__ == "__main__":
    main(sys.argv[1])
