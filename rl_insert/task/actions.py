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
        # First step of an episode (mjlab zeroes episode_length_buf on reset): nothing commanded
        # yet, so start at the measured joints.
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
