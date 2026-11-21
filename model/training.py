"""Viam custom training script for PegInsert-xArm6 (mjlab + RSL-RL PPO on MuJoCo Warp).

Viam runs `python3 -m model.training --model_output_directory=... <dataset args> --key=value ...`.
The dataset is a required placeholder and is ignored. Everything in the output directory except
tmp/ becomes the model package: policy.onnx, config.json, checkpoint.pt, model_metrics.json, kinematics.xml.

Modes:
- train (default): train on one GPU (CPU when there is none), upload training videos to Viam Data
  as they are written, then write the package.
- probe: a cheap first job on new hardware; writes probe.json (GPU, GL, grasp hold, throughput,
  video upload). Every check is independent and the job always exits 0.

mujoco reads MUJOCO_GL once, when its GL module loads, and mjlab defaults it to egl on Linux, so
the GL backend is picked (in a subprocess) before anything imports mujoco or mjlab.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

from model import harness
from model.harness import GL_TEST, check, device, select_gl, set_visible_devices, unquote  # noqa: F401

logger = logging.getLogger("rl-insert")
_device = device
_check = check

TAGS = ["rl-insert"]
UPLOAD_JOIN_TIMEOUT = 60.0  # s to wait for the uploader thread at the end of training


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse strictly: an unknown or misspelled custom arg exits 2."""
    p = argparse.ArgumentParser(prog="model.training", allow_abbrev=False)
    p.add_argument("--model_output_directory", type=lambda s: Path(unquote(s)), required=True)
    # Dataset arguments Viam passes. Accepted and ignored: this task trains in simulation.
    for name in ("dataset_file", "binary_data_file", "tabular_data_file", "sequences_file"):
        p.add_argument(f"--{name}", type=unquote, default=None, help="ignored (placeholder dataset)")
    p.add_argument("--mode", type=unquote, choices=("train", "probe"), default="train")
    p.add_argument("--clearance_mm", type=lambda s: float(unquote(s)), default=0.5)
    p.add_argument(
        "--max_iterations",
        type=lambda s: int(unquote(s)),
        default=None,
        help="PPO iterations (default: the task's PPO config)",
    )
    p.add_argument("--num_envs", type=lambda s: int(unquote(s)), default=4096)
    p.add_argument("--seed", type=lambda s: int(unquote(s)), default=42)
    p.add_argument(
        "--video_every_iters",
        type=lambda s: int(unquote(s)),
        default=250,
        help="record a training clip every N PPO iterations (0: no video)",
    )
    p.add_argument(
        "--video_length",
        type=lambda s: int(unquote(s)),
        default=200,
        help="clip length in policy steps (20 Hz)",
    )
    p.add_argument(
        "--upload_part_id",
        type=unquote,
        default=None,
        help="machine part the videos are uploaded under (no upload without it)",
    )
    p.add_argument("--upload_dataset_id", type=unquote, default=None)
    p.add_argument(
        "--fail_fast_iters",
        type=lambda s: int(unquote(s)),
        default=300,
        help="fail the job if no episode has succeeded by this iteration (0: never)",
    )
    p.add_argument("--run_name", type=unquote, default=None, help="default: peg-insert-<timestamp>")
    args = p.parse_args(argv)
    args.run_name = args.run_name or time.strftime("peg-insert-%Y%m%d-%H%M%S")
    return args


class VideoUploader(harness.VideoUploader):
    """Peg-insert's signature (args Namespace) over the shared uploader."""

    def __init__(self, folder, steps_per_iter, args, poll_s: float = 15.0):
        super().__init__(
            folder,
            steps_per_iter,
            tags=TAGS,
            run_name=args.run_name,
            upload_part_id=args.upload_part_id,
            upload_dataset_id=args.upload_dataset_id,
            poll_s=poll_s,
        )


def write_package(out: Path, log_dir: Path, env_cfg, p, run_name: str) -> None:
    from model.package import build_config, build_metrics, write_json
    from rl_insert.robot import kinematics_spec

    # The best checkpoint at full difficulty (PegInsertRunner.save), else the last one.
    best = log_dir / "best.json"
    if best.exists():
        ckpt, onnx_path = log_dir / "best.pt", log_dir / "best.onnx"
        picked = {"why": "best success-rate EMA at full difficulty", **json.loads(best.read_text())}
    else:
        ckpt = max(log_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
        onnx_path = log_dir / f"{log_dir.name}.onnx"
        picked = {"iteration": int(ckpt.stem.split("_")[1]), "success_rate_ema": None, "why": "last (never saved at full difficulty)"}
    shutil.copy(onnx_path, out / "policy.onnx")
    (out / "kinematics.xml").write_text(kinematics_spec().to_xml())  # the model the IK contract refers to
    shutil.copy(ckpt, out / "checkpoint.pt")
    config = build_config(out / "policy.onnx", env_cfg, p, run_name)
    config["checkpoint"] = picked
    write_json(out / "config.json", config)
    write_json(out / "model_metrics.json", build_metrics(log_dir))
    logger.info("wrote %s (checkpoint %s)", sorted(f.name for f in out.iterdir()), config["checkpoint"])


def train(args: argparse.Namespace, gl: dict) -> None:
    from mjlab.scripts.train import TrainConfig, run_train

    from rl_insert.task import TASK_ID, TaskParams, peg_insert_env_cfg, peg_insert_ppo_cfg
    from rl_insert.task.runner import PegInsertRunner

    out = args.model_output_directory
    # Train on local disk: the output dir may be a FUSE mount that re-uploads on every flush.
    # Copied to <out>/tmp/logs at the end.
    log_dir = Path(tempfile.mkdtemp(prefix="rl-insert-")) / "logs" / args.run_name
    p = TaskParams(clearance=args.clearance_mm / 1000)
    env_cfg = peg_insert_env_cfg(p)
    env_cfg.scene.num_envs = args.num_envs
    agent = peg_insert_ppo_cfg()
    agent.seed = args.seed
    agent.run_name = args.run_name
    if args.max_iterations is not None:
        agent.max_iterations = args.max_iterations
    PegInsertRunner.fail_fast_iters = args.fail_fast_iters  # run_train builds the runner itself
    video = gl["backend"] is not None and args.video_every_iters > 0
    cfg = TrainConfig(
        env=env_cfg,
        agent=agent,
        video=video,
        video_length=args.video_length,
        # VideoRecorder counts env.step() calls; one PPO iteration is num_steps_per_env of them.
        video_interval=max(1, args.video_every_iters) * agent.num_steps_per_env,
        log_root=str(log_dir.parent),
        gpu_ids=None,  # unused: run_train is called directly, on one device
    )
    # run_train picks its device from CUDA_VISIBLE_DEVICES: "" is CPU, otherwise cuda:LOCAL_RANK.
    set_visible_devices(_device().startswith("cuda"))

    uploader = VideoUploader(log_dir / "videos" / "train", agent.num_steps_per_env, args)
    if video:
        uploader.folder.mkdir(parents=True, exist_ok=True)
        uploader.start()
    try:
        try:
            run_train(TASK_ID, cfg, log_dir)
        except BaseException:
            try:  # keep whatever the run produced
                write_package(out, log_dir, env_cfg, p, args.run_name)
            except Exception:
                logger.error("no package after failed run:\n%s", traceback.format_exc())
            raise
        write_package(out, log_dir, env_cfg, p, args.run_name)  # before any upload can stall
    finally:
        uploader.stop.set()
        if video:
            uploader.join(timeout=UPLOAD_JOIN_TIMEOUT)
            uploader.sweep(final=True)
        try:
            shutil.copytree(log_dir.parent, out / "tmp" / "logs", dirs_exist_ok=True)
        except Exception:  # don't mask an exception from run_train
            logger.error("copying logs to %s failed:\n%s", out, traceback.format_exc())
        logger.info("videos: %s", uploader.done)


def probe(args: argparse.Namespace, gl: dict) -> None:
    out = args.model_output_directory
    results: dict = {"gl": gl}

    def nvidia_smi() -> dict:
        exe = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        return {"path": exe, "returncode": r.returncode, "output": (r.stdout + r.stderr)[-4000:]}

    def cuda() -> dict:
        import torch
        import warp as wp

        drv = wp.get_cuda_driver_version() if wp.is_cuda_available() else None
        return {
            "torch": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "warp": wp.__version__,
            "warp_cuda_available": wp.is_cuda_available(),
            "warp_cuda_devices": wp.get_cuda_device_count(),
            "cuda_driver_version": drv,
            "device_used": _device(),
        }

    def grasp_hold() -> dict:
        from rl_insert.task.checks import grasp_hold_slip, make_env

        env = make_env(64, _device())
        try:
            slip = grasp_hold_slip(env, seconds=2.0)
        finally:
            env.close()
        return {"num_envs": 64, "hold_s": 2.0, **slip, "pass": max(slip.values()) < 0.5}

    def speed() -> dict:
        from rl_insert.task.checks import make_env, throughput

        env = make_env(args.num_envs, _device())
        try:
            return {"num_envs": args.num_envs, **throughput(env)}
        finally:
            env.close()

    def video() -> dict:
        if gl["backend"] is None:
            return {"skipped": "no GL backend"}
        import mediapy
        import torch

        from rl_insert.task.checks import make_env
        from rl_insert.task.viam_upload import upload_video

        env = make_env(1, _device(), render=True)
        try:
            env.reset()
            frames = []
            for _ in range(40):  # 2 s at 20 Hz
                env.step(torch.zeros(1, 6, device=env.device))
                f = env.render()
                frames.append(f[0] if f.ndim == 4 else f)
        finally:
            env.close()
        path = out / "tmp" / "probe.mp4"
        mediapy.write_video(path, frames, fps=20)
        file_id = upload_video(
            str(path),
            args.upload_part_id,
            [*TAGS, "probe"],
            [args.upload_dataset_id] if args.upload_dataset_id else None,
        )
        return {"path": str(path), "frames": len(frames), "file_id": file_id, "uploaded": file_id is not None}

    for name, fn in (
        ("nvidia_smi", nvidia_smi),
        ("cuda", cuda),
        ("grasp_hold", grasp_hold),
        ("throughput", speed),
        ("video", video),
    ):
        _check(results, name, fn, out)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_output_directory = args.model_output_directory.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    out = args.model_output_directory
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    os.chdir(out / "tmp")  # anything a library writes relative to cwd stays out of the package
    gl = select_gl()
    if args.mode == "probe":
        probe(args, gl)
    else:
        train(args, gl)


if __name__ == "__main__":
    main()
