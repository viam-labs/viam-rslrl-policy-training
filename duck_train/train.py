"""Viam custom training script for the MicroDuck walk (Mjlab-Velocity-Flat-MicroDuck, pollen-robotics/microduck_rl
@0daab16: mjlab 1.3.0 + RSL-RL PPO on MuJoCo Warp, BAM XL330 actuators). Viam runs `python3 -m model.training`;
this sdist's model/training.py calls main() here.

Modes:
- train (default): train on one GPU, upload training videos as they're written, then write the package:
  policy.onnx (exported from the last checkpoint, normalizer included), config.json, checkpoint.pt,
  model_metrics.json.
- probe: a cheap first job; writes probe.json. Each check is independent and the job always exits 0. The
  checks: nvidia-smi, versions, a short training run (iterations, compile time, s/iteration), export and
  verify an ONNX, upload one video.

wandb is installed (mjlab imports it) but never used: logger=tensorboard and WANDB_MODE=disabled.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path

from model.harness import VideoUploader, check, device, select_gl, set_visible_devices, unquote

from .package import build_config, scalar_metrics, verify_onnx, write_json

logger = logging.getLogger("microduck-train")
TASK = "Mjlab-Velocity-Flat-MicroDuck"
TAGS = ["microduck"]
JOINTS = ["left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
          "neck_pitch", "head_pitch", "head_yaw", "head_roll",
          "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle"]
DEFAULT_POSE = [0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
                0.0, 0.0873, 0.4579, 0.0049, -0.4530]  # infer_policy.py:126-141
UPLOAD_JOIN_TIMEOUT = 60.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="model.training", allow_abbrev=False)
    p.add_argument("--model_output_directory", type=lambda s: Path(unquote(s)), required=True)
    for name in ("dataset_file", "binary_data_file", "tabular_data_file", "sequences_file"):
        p.add_argument(f"--{name}", type=unquote, default=None, help="ignored (placeholder dataset)")
    p.add_argument("--mode", type=unquote, choices=("train", "probe"), default="train")
    p.add_argument("--max_iterations", type=lambda s: int(unquote(s)), default=4000)
    p.add_argument("--num_envs", type=lambda s: int(unquote(s)), default=4096)
    p.add_argument("--seed", type=lambda s: int(unquote(s)), default=42)
    p.add_argument("--save_interval", type=lambda s: int(unquote(s)), default=250)
    p.add_argument("--video_every_iters", type=lambda s: int(unquote(s)), default=500)
    p.add_argument("--video_length", type=lambda s: int(unquote(s)), default=200)
    p.add_argument("--run_name", type=unquote, default=time.strftime("microduck-%Y%m%d-%H%M%S"))
    p.add_argument("--upload_part_id", type=unquote, default=None)
    p.add_argument("--upload_dataset_id", type=unquote, default=None)
    return p.parse_args(argv)


def _cfgs(args):
    import mjlab.tasks  # noqa: F401  (loads the mjlab.tasks entry points: registers microduck's tasks)
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg  # verified in mjlab 1.3.0 (return deep copies)

    env_cfg = load_env_cfg(TASK)
    env_cfg.scene.num_envs = args.num_envs
    agent = load_rl_cfg(TASK)
    agent.max_iterations = args.max_iterations
    agent.save_interval = args.save_interval
    agent.seed = args.seed
    agent.run_name = args.run_name
    agent.logger = "tensorboard"
    agent.upload_model = False
    return env_cfg, agent


def _last_checkpoint(log_dir: Path) -> Path:
    ckpts = sorted(log_dir.rglob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no checkpoint under {log_dir}")
    return ckpts[-1]


def export_and_verify(ckpt: Path, onnx_out: Path) -> dict:
    from mjlab_microduck.export import ExportConfig, run_export  # verified at MD/.../export.py:90

    run_export(TASK, ExportConfig(checkpoint_file=str(ckpt), num_envs=1, onnx_file=str(onnx_out)))
    return verify_onnx(onnx_out, JOINTS)


def write_package(out: Path, log_dir: Path, env_cfg, agent) -> None:
    import mujoco

    ckpt = _last_checkpoint(log_dir)
    meta = export_and_verify(ckpt, out / "policy.onnx")
    # The ONNX metadata rounds the pose to 3 decimals; config.json carries DEFAULT_POSE at full precision.
    meta_pose = [float(v) for v in meta["default_joint_pos"].split(",")]
    if any(abs(a - b) > 6e-4 for a, b in zip(meta_pose, DEFAULT_POSE)):
        raise ValueError(f"ONNX default_joint_pos {meta_pose} != DEFAULT_POSE")
    # Joint limits come from microduck's robot_walk.xml compiled directly (the training robot's model).
    xml = Path(__import__("mjlab_microduck").__file__).parent / "robot" / "microduck" / "robot_walk.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    limits = [list(model.joint(n).range) for n in JOINTS]
    twist = env_cfg.commands["twist"].ranges
    config = build_config(
        joints=JOINTS, default_pose=DEFAULT_POSE, action_scale=float(meta["action_scale"]), joint_limits=limits,
        twist_min=[twist.lin_vel_x[0], twist.lin_vel_y[0], twist.ang_vel_z[0]],
        twist_max=[twist.lin_vel_x[1], twist.lin_vel_y[1], twist.ang_vel_z[1]],
        control_hz=round(1 / (env_cfg.sim.mujoco.timestep * env_cfg.decimation)),
        source={"task": TASK, "microduck_rl": "0daab160d3fb1c2cbaba16a4a19052f36709dd9e", "checkpoint": ckpt.name,
                "iterations": agent.max_iterations, "num_envs": env_cfg.scene.num_envs, "seed": agent.seed},
    )
    write_json(out / "config.json", config)
    shutil.copy(ckpt, out / "checkpoint.pt")
    write_json(out / "model_metrics.json", scalar_metrics(log_dir))
    logger.info("wrote %s from %s", sorted(p.name for p in out.iterdir()), ckpt.name)


def train(args, gl: dict) -> None:
    from mjlab.scripts.train import TrainConfig, run_train

    out = args.model_output_directory
    log_dir = Path(tempfile.mkdtemp(prefix="microduck-")) / "logs" / args.run_name
    env_cfg, agent = _cfgs(args)
    video = gl["backend"] is not None and args.video_every_iters > 0
    cfg = TrainConfig(env=env_cfg, agent=agent, video=video, video_length=args.video_length,
                      video_interval=max(1, args.video_every_iters) * agent.num_steps_per_env,
                      gpu_ids=None)  # mjlab 1.3.0's TrainConfig has no log_root: run_train writes into log_dir
    set_visible_devices(device().startswith("cuda"))
    uploader = VideoUploader(log_dir / "videos" / "train", agent.num_steps_per_env, tags=TAGS, run_name=args.run_name,
                             upload_part_id=args.upload_part_id, upload_dataset_id=args.upload_dataset_id)
    if video:
        uploader.folder.mkdir(parents=True, exist_ok=True)
        uploader.start()
    try:
        try:
            run_train(TASK, cfg, log_dir)
        except BaseException:
            try:
                write_package(out, log_dir, env_cfg, agent)
            except Exception:
                logger.error("no package after failed run:\n%s", traceback.format_exc())
            raise
        write_package(out, log_dir, env_cfg, agent)
    finally:
        uploader.stop.set()
        if video:
            uploader.join(timeout=UPLOAD_JOIN_TIMEOUT)
            uploader.sweep(final=True)
        logger.info("videos: %s", uploader.done)
        try:
            shutil.copytree(log_dir.parent, out / "tmp" / "logs", dirs_exist_ok=True)
        except Exception:
            logger.error("copying logs failed:\n%s", traceback.format_exc())


def probe(args, gl: dict) -> None:
    out = args.model_output_directory
    results: dict = {"gl": gl}
    def nvidia_smi() -> dict:
        exe = shutil.which("nvidia-smi") or "/usr/local/nvidia/bin/nvidia-smi"  # not on PATH on Vertex
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
        return {"rc": r.returncode, "out": r.stdout[-3000:], "err": r.stderr[-1000:]}

    check(results, "nvidia_smi", nvidia_smi, out)

    def versions() -> dict:
        import mjlab, mujoco, mujoco_warp, torch, warp  # noqa: E401
        return {"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_ok": torch.cuda.is_available(),
                "warp": warp.__version__, "mujoco": mujoco.__version__, "mujoco_warp": getattr(mujoco_warp, "__version__", "?"),
                "mjlab": importlib.metadata.version("mjlab")}  # mjlab 1.3.0 has no __version__

    check(results, "versions", versions, out)
    log_dir = Path(tempfile.mkdtemp(prefix="microduck-probe-")) / "logs" / args.run_name

    def short_train() -> dict:
        from mjlab.scripts.train import TrainConfig, run_train

        args.max_iterations, args.save_interval = 20, 10  # probe: a short run, a checkpoint every 10
        env_cfg, agent = _cfgs(args)
        set_visible_devices(device().startswith("cuda"))
        video = gl["backend"] is not None
        t0 = time.monotonic()
        run_train(TASK, TrainConfig(env=env_cfg, agent=agent, video=video, video_length=100,
                                    video_interval=10 * agent.num_steps_per_env, gpu_ids=None), log_dir)
        return {"iterations": agent.max_iterations, "num_envs": env_cfg.scene.num_envs,
                "wall_s": round(time.monotonic() - t0, 1), "checkpoints": sorted(p.name for p in log_dir.rglob("model_*.pt")),
                "metrics": scalar_metrics(log_dir)}

    check(results, "short_train", short_train, out)
    check(results, "export", lambda: {"meta": export_and_verify(_last_checkpoint(log_dir), out / "probe_policy.onnx")}, out)

    def upload() -> dict:
        vids = sorted((log_dir / "videos" / "train").glob("*.mp4"))
        if not vids:
            return {"skipped": "no video"}
        from model.viam_upload import upload_video
        return {"file_id": upload_video(str(vids[0]), args.upload_part_id, [*TAGS, args.run_name, "probe"], None)}

    check(results, "video_upload", upload, out)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_output_directory = args.model_output_directory.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    os.environ.setdefault("WANDB_MODE", "disabled")
    out = args.model_output_directory
    (out / "tmp").mkdir(parents=True, exist_ok=True)
    os.chdir(out / "tmp")
    gl = select_gl()
    if args.mode == "probe":
        probe(args, gl)
    else:
        train(args, gl)
