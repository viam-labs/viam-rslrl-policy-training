"""PPO config and runner for PegInsert-xArm6."""

from __future__ import annotations

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


class PegInsertRunner(MjlabOnPolicyRunner):
    """Exports ONNX on every save.

    mjlab's get_base_metadata asserts a "joint_pos" JointPositionAction (joint stiffness, action
    scale...), which this task doesn't have, so the metadata is replaced with the IK action's.
    """

    export_onnx_on_save = True

    def learn(self, *args, **kwargs) -> None:
        super().learn(*args, **kwargs)
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
