"""PPO config and runner for PegInsert-xArm6."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.rl.runner import MjlabOnPolicyRunner

from .env_cfg import PPO_GAMMA


def peg_insert_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            obs_normalization=True,  # exported into the ONNX graph
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        ),
        critic=RslRlModelCfg(hidden_dims=(256, 128, 64), obs_normalization=True),
        algorithm=RslRlPpoAlgorithmCfg(entropy_coef=0.005, learning_rate=3e-4, gamma=PPO_GAMMA),
        experiment_name="peg_insert_xarm6",
        logger="tensorboard",
        num_steps_per_env=24,
        save_interval=50,
        max_iterations=1500,
    )


def success_rate(ep_extras: list[dict]) -> float | None:
    """Share of the iteration's episode ends that were successes; None when nothing ended.

    ponytail: counts are per term, so an episode ending on two terms at once (success and excess
    force on one step) counts twice and the rate reads a little low; fine for ranking checkpoints.
    """
    sums: dict[str, float] = {}
    for d in ep_extras:
        for k, v in d.items():
            if k.startswith("Episode_Termination/"):
                sums[k] = sums.get(k, 0.0) + float(torch.as_tensor(v, dtype=torch.float32).sum())
    total = sum(sums.values())
    return sums.get("Episode_Termination/success", 0.0) / total if total > 0 else None


class PegInsertRunner(MjlabOnPolicyRunner):
    """Exports ONNX on every save, keeps the best checkpoint, and fails fast on a dead run.

    mjlab's get_base_metadata asserts a "joint_pos" JointPositionAction (joint stiffness, action
    scale...), which this task doesn't have, so the metadata is replaced with the IK action's.

    Best: the success-rate EMA only runs while the curriculum is at full difficulty (it restarts
    each time difficulty reaches 1.0, so easier levels can't inflate it). At each save, if it beats
    the best so far and that save's ONNX export worked, the checkpoint and its ONNX are copied to best.pt / best.onnx (+ best.json).
    Fail fast: if no episode has succeeded by iteration fail_fast_iters, raise (0 disables). Runs 1
    and 3 first succeeded at about 24 and 40. rsl_rl already raises on NaN observations or rewards.
    """

    export_onnx_on_save = True
    fail_fast_iters = 300  # set from model.training's --fail_fast_iters
    ema_alpha = 0.05  # success-rate smoothing per iteration (~20-iteration window)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ema: float | None = None
        self.ever_succeeded = False
        self.best = -1.0
        log = self.logger.log

        def watched_log(**kw) -> None:  # ep_extras is cleared inside log, so read it first
            self._watch(kw["it"], success_rate(self.logger.ep_extras), self.env.unwrapped.insert_state.difficulty)
            log(**kw)

        self.logger.log = watched_log

    def _watch(self, it: int, rate: float | None, difficulty: float) -> None:
        if rate is not None:
            self.ever_succeeded |= rate > 0
        if difficulty < 1.0:
            self.ema = None
        elif rate is not None:
            self.ema = rate if self.ema is None else self.ema + self.ema_alpha * (rate - self.ema)
        if self.fail_fast_iters and it >= self.fail_fast_iters and not self.ever_succeeded:
            raise RuntimeError(f"fail fast: no successful episode in {it} iterations")

    def save(self, path: str, infos=None) -> None:
        # Every save overwrites one ONNX file and a failed export only warns, so remove it first:
        # if it's missing afterwards, this save's export failed and best.onnx must not get a stale one.
        _, _, onnx_path = self._get_export_paths(path)
        onnx_path.unlink(missing_ok=True)
        super().save(path, infos)
        if self.ema is None or self.ema <= self.best or not onnx_path.exists():
            return
        self.best = self.ema
        ckpt = Path(path)
        shutil.copy(ckpt, ckpt.parent / "best.pt")
        shutil.copy(onnx_path, ckpt.parent / "best.onnx")
        info = {"iteration": self.current_learning_iteration, "success_rate_ema": round(self.ema, 4)}
        (ckpt.parent / "best.json").write_text(json.dumps(info) + "\n")

    def learn(self, *args, **kwargs) -> None:
        try:
            super().learn(*args, **kwargs)
        finally:  # fail fast raises out of learn; its package wants the scalars too
            # rsl_rl never closes a plain tensorboard SummaryWriter (flush_secs=10), so the last
            # iterations' scalars may still be buffered when model_metrics.json reads the event file.
            writer = getattr(self.logger, "writer", None)
            if writer is not None and hasattr(writer, "flush"):
                writer.flush()

    def get_export_metadata(self, run_name: str) -> dict[str, list | str | float]:
        env = self.env.unwrapped
        ik = env.action_manager.get_term("ee_delta").cfg
        obs = env.observation_manager
        return {
            "run_path": run_name,
            "action_type": "ee_delta_hole_frame",
            "action_pos_scale": ik.delta_pos_scale,
            "action_rot_scale": ik.delta_ori_scale,
            "policy_dt": env.step_dt,
            "joint_names": list(env.scene["robot"].joint_names),
            "observation_names": obs.active_terms["actor"],
            "observation_dims": [d[-1] for d in obs.group_obs_term_dim["actor"]],
        }
