"""Mjlab-Dance-Flat-MicroDuck (spec Part 2): Pollen's flat velocity task (MD microduck_velocity_env_cfg) with the
velocity terms swapped for phase-indexed pose tracking against the dance's reference table."""

from __future__ import annotations

import dataclasses
from copy import deepcopy

import torch
from mjlab.managers import RewardTermCfg
from mjlab.tasks.registry import register_mjlab_task
from mjlab_microduck.tasks import MicroduckOnPolicyRunner
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import MicroduckRlCfg, make_microduck_velocity_env_cfg

from . import dance_rewards

DANCE_TASK = "Mjlab-Dance-Flat-MicroDuck"
REMOVED_REWARDS = ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance",
                   "foot_swing_height", "pose", "head_pose_tracking", "head_pose_bias", "body_pose_tracking")
REMOVED_CURRICULA = ("action_rate_weight", "standing_envs", "head_pose_range", "body_pose_range", "head_pose_bias_weight")
EPISODE_S = 20.0
REL_FROZEN = 0.2


class DancePhaseCommand(microduck_mdp.GroundPickPhaseCommand):
    """GroundPickPhaseCommand with a fraction of envs frozen at phase 0 (STAND hold)."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rel_frozen = float(cfg.rel_frozen_envs)

    def reset(self, env_ids):
        out = super().reset(env_ids)
        if env_ids is not None and len(env_ids) > 0:
            self._frozen[env_ids] = torch.rand(len(env_ids), device=self.device) < self._rel_frozen
            self._gp_phase[env_ids] = torch.where(self._frozen[env_ids], torch.zeros_like(self._gp_phase[env_ids]),
                                                  self._gp_phase[env_ids])
        return out

    def compute(self, dt):
        super().compute(dt)
        self._gp_phase[self._frozen] = 0.0
        self.vel_command_b[self._frozen, 0] = 1.0
        self.vel_command_b[self._frozen, 1] = 0.0


@dataclasses.dataclass(kw_only=True)
class DancePhaseCommandCfg(microduck_mdp.GroundPickPhaseCommandCfg):
    class_type: type = DancePhaseCommand
    rel_frozen_envs: float = REL_FROZEN

    def build(self, env):
        return DancePhaseCommand(self, env)


def make_dance_env_cfg(table, period_s: float, play: bool = False):
    cfg = make_microduck_velocity_env_cfg(play=play)
    twist = cfg.commands["twist"]
    fields = {f.name for f in dataclasses.fields(DancePhaseCommandCfg)}
    cfg.commands["twist"] = DancePhaseCommandCfg(**{**{k: v for k, v in vars(twist).items() if k in fields},
                                                    "class_type": DancePhaseCommand, "period": float(period_s),
                                                    "randomize_phase": True})
    for name in ("head_pose", "body_pose"):
        ranges = cfg.commands[name].ranges
        cfg.commands[name].ranges = tuple((0.0, 0.0) for _ in ranges)
    for name in REMOVED_REWARDS:
        del cfg.rewards[name]
    for name in REMOVED_CURRICULA:
        del cfg.curriculum[name]
    table = [list(map(float, row)) for row in table]
    cfg.rewards["dance_pose"] = RewardTermCfg(func=dance_rewards.dance_pose, weight=4.0, params={"table": table, "std": 0.3})
    cfg.rewards["dance_pose_l1"] = RewardTermCfg(func=dance_rewards.dance_pose_l1, weight=1.0, params={"table": table})
    cfg.rewards["feet_contact"] = RewardTermCfg(func=dance_rewards.feet_contact, weight=0.5)
    cfg.rewards["action_rate_l2"].weight = -1.0  # the removed action_rate_weight curriculum's final value
    cfg.episode_length_s = EPISODE_S
    return cfg


def register_dance_task(table, period_s: float) -> str:
    rl = deepcopy(MicroduckRlCfg)
    rl.experiment_name = "microduck_dance"
    register_mjlab_task(task_id=DANCE_TASK, env_cfg=make_dance_env_cfg(table, period_s),
                        play_env_cfg=make_dance_env_cfg(table, period_s, play=True), rl_cfg=rl,
                        runner_cls=MicroduckOnPolicyRunner)
    return DANCE_TASK
