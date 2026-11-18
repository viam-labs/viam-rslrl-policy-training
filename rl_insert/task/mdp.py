"""MDP terms for PegInsert-xArm6: the batched reset, observations, rewards, terminations, curriculum.

Frames: the robot base is the world frame (one robot per Warp world, all at the origin). The hole
is upright with zero yaw, so the hole frame is the world frame translated to the mouth: vectors
"in the hole frame" here are world vectors. Only if the hole is ever not upright/zero-yaw,
which training does not cover, would the deploy side need to rotate actions and observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.event_manager import EventTermCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_conjugate,
    quat_from_matrix,
    quat_mul,
)

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

GRASP_WELD = "grasp"  # the peg-to-gripper weld equality added by env_cfg._add_grasp_weld
# Weld weight on rotation vs translation. At 1 the 20 g peg's tiny inertia leaves the weld soft in
# rotation: pad torque about the TCP swung the tip (40 mm out) p95 7 mm, max 10, under random
# actions on CPU, 64 envs. At 10: p95 0.23 mm, max 0.6 (100 gives the same).
GRASP_TORQUESCALE = 10.0
FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))  # peg +z points back toward the gripper


def _rot_y(theta: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    z, o = torch.zeros_like(theta), torch.ones_like(theta)
    return torch.stack([c, z, s, z, o, z, -s, z, c], dim=-1).view(-1, 3, 3)


def _uniform(n: int, lo: float, hi: float, device) -> torch.Tensor:
    return torch.rand(n, device=device) * (hi - lo) + lo


def rot6d(mat: torch.Tensor) -> torch.Tensor:
    """First two columns of a rotation matrix, flattened: continuous, unlike quaternions."""
    return mat[..., :, :2].transpose(-1, -2).reshape(*mat.shape[:-2], 6)


class InsertState:
    """Per-env episode state the terms share: the reset event owns it, the rest read it.

    Lives at env.insert_state (set by ResetPegInHand.__init__). Holds the per-episode hole-pose
    bias (the actor's *estimated* hole = true + bias) and the curriculum difficulty.
    """

    def __init__(self, env: ManagerBasedRlEnv, p: dict):
        self.p = p
        n, dev = env.num_envs, env.device
        self.hole_bias = torch.zeros(n, 3, device=dev)
        self.grasp_tip = torch.zeros(n, 3, device=dev)  # sampled tip in the TCP frame (the weld target)
        self.difficulty = p["difficulty_init"]  # scales lateral start offset and hole bias, 0..1
        self.succ = torch.zeros((), device=dev)  # successes among resets in the current curriculum window
        self.done = 0  # resets in the current window
        self.rate = 0.0  # success rate of the last completed window (scales the excess-force penalty)
        robot = env.scene["robot"]
        self.tcp_gid = int(robot.indexing.site_ids[robot.find_sites("tcp")[0][0]])
        self.flip = FLIP.to(dev)
        mj = env.sim.mj_model
        self.weld_id = mj.equality(GRASP_WELD).id
        self.gripper_bid = mj.body("robot/gripper").id


def state(env: ManagerBasedRlEnv) -> InsertState:
    return env.insert_state  # ty: ignore[unresolved-attribute]


# Geometry helpers (all world frame).


def tcp_pose(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    s = state(env)
    return env.sim.data.site_xpos[:, s.tcp_gid], env.sim.data.site_xmat[:, s.tcp_gid].view(-1, 3, 3)


def nominal_tip(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Peg tip assuming the nominal grasp: what the real arm can compute from FK."""
    pos, rot = tcp_pose(env)
    return pos + rot[:, :, 2] * state(env).p["protrusion"]


def true_tip(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["peg"].data.root_link_pose_w[:, :3]  # peg frame origin is its tip


def hole_mouth(env: ManagerBasedRlEnv) -> torch.Tensor:
    return env.scene["hole"].data.root_link_pose_w[:, :3]  # mocap root = hole mouth


def tip_in_hole(env: ManagerBasedRlEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(lateral distance from the hole axis, depth below the mouth), both m, true tip."""
    rel = true_tip(env) - hole_mouth(env)
    return rel[:, :2].norm(dim=-1), -rel[:, 2]


def peg_hole_force(env: ManagerBasedRlEnv, sensor_name: str = "peg_hole_contact") -> torch.Tensor:
    """Net peg-hole contact force, world frame (N, 3)."""
    return env.scene[sensor_name].data.force[:, 0, :]


# Reset.


class ResetPegInHand:
    """Batched equivalent of rl_insert.scene.place_peg_in_hand, plus hole placement.

    Runs after reset_scene_to_default has put the arm at its nominal joints with the fingers at
    the peg radius. Then:
    1. forward kinematics, so the TCP pose is current (the arm was just teleported);
    2. the peg goes in hand: tip `protrusion` past the TCP along TCP z, grasp offset along TCP x and
       z only (a y offset is re-centred by the coupled fingers), tilt about TCP y (the pad normal);
    3. the hole goes under the true tip: mouth = tip - (dx, dy, height), so the tip starts
       height (10-20 mm) above the mouth with a lateral offset;
    4. the per-episode hole-pose bias for the actor's estimated hole is sampled.

    The grasp is a weld (env_cfg._add_grasp_weld), so step 2 also writes that world's weld pose:
    anchored at the TCP (the grasp point), relative pose = the sampled grasp. Since the peg starts
    exactly at the weld pose, nothing settles after reset.
    """

    # eq_data gets per-world memory; after the event mjlab reruns set_const KIN0, which leaves
    # welds whose eq_data[11] mark is 0 (ours) untouched.
    model_fields = ("eq_data",)

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRlEnv):
        env.insert_state = InsertState(env, cfg.params["p"])  # ty: ignore[unresolved-attribute]

    def __call__(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, p: dict) -> None:
        s = state(env)
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n, dev = len(env_ids), env.device
        env.sim.forward()  # the arm was just written; site_xpos is stale until FK runs
        pos, rot = tcp_pose(env)
        pos, rot = pos[env_ids], rot[env_ids]

        g = p["grasp_offset"]
        offset = torch.stack(
            [_uniform(n, -g, g, dev), torch.zeros(n, device=dev), _uniform(n, -g, g, dev)], dim=-1
        )
        tilt = _rot_y(_uniform(n, -p["grasp_tilt"], p["grasp_tilt"], dev))
        along = tilt @ torch.tensor([0.0, 0.0, p["protrusion"]], device=dev)
        s.grasp_tip[env_ids] = offset + along
        tip = pos + (rot @ s.grasp_tip[env_ids].unsqueeze(-1)).squeeze(-1)
        peg_quat = quat_from_matrix(rot @ tilt @ s.flip)
        peg_state = torch.cat([tip, peg_quat, torch.zeros(n, 6, device=dev)], dim=-1)
        env.scene["peg"].write_root_state_to_sim(peg_state, env_ids=env_ids)

        # Weld data: [anchor in peg frame (3), anchor in gripper frame (3), quat gripper->peg (4),
        # torquescale, compute-from-qpos0 mark]. Anchor = the TCP.
        peg_rot = rot @ tilt @ s.flip
        g_pos = env.sim.data.xpos[env_ids, s.gripper_bid]
        g_quat = env.sim.data.xquat[env_ids, s.gripper_bid]
        g_rot = matrix_from_quat(g_quat)
        in_peg = (peg_rot.transpose(-1, -2) @ (pos - tip).unsqueeze(-1)).squeeze(-1)
        in_grip = (g_rot.transpose(-1, -2) @ (pos - g_pos).unsqueeze(-1)).squeeze(-1)
        rel = quat_mul(quat_conjugate(g_quat), peg_quat)
        tail = torch.tensor([GRASP_TORQUESCALE, 0.0], device=dev).expand(n, 2)
        env.sim.model.eq_data[env_ids, s.weld_id] = torch.cat([in_peg, in_grip, rel, tail], dim=-1)

        lat = p["start_lateral"] * s.difficulty
        h = _uniform(n, *p["start_height"], dev)
        mouth = tip - torch.stack([_uniform(n, -lat, lat, dev), _uniform(n, -lat, lat, dev), h], -1)
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
        env.scene["hole"].write_mocap_pose_to_sim(torch.cat([mouth, ident], -1), env_ids=env_ids)

        b = torch.tensor(p["hole_bias"], device=dev) * s.difficulty
        s.hole_bias[env_ids] = (torch.rand(n, 3, device=dev) * 2 - 1) * b


# Observations.


def tip_rel_est_hole(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Actor: nominal peg tip relative to the estimated hole (true + episode bias), hole frame."""
    return nominal_tip(env) - (hole_mouth(env) + state(env).hole_bias)


def ee_rot6d(env: ManagerBasedRlEnv) -> torch.Tensor:
    """TCP orientation in the hole frame (= world, see module docstring), rot6d."""
    return rot6d(tcp_pose(env)[1])


def ik_tracking_error(env: ManagerBasedRlEnv, action_name: str = "ee_delta") -> torch.Tensor:
    """Commanded minus measured TCP pose (pos m, axis-angle rad), world axes: the contact proxy.

    Commanded = the IK action's desired pose (private _desired_pos/_desired_quat: mjlab has no
    public accessor), set when the last action was processed: p_des = p + pos_scale * a[:3],
    R_des = R_delta(rot_scale * a[3:]) R, from the TCP pose measured at that policy step. The
    rotation part is the axis-angle of q_des * q_meas^-1. The action's reset() zeroes that buffer,
    so the first step after reset reports zero instead of a bogus error. env_cfg.OBS_DOC holds
    the deploy-side recipe.
    """
    term = env.action_manager.get_term(action_name)
    pos, rot = tcp_pose(env)
    quat = quat_from_matrix(rot)
    dq = quat_mul(term._desired_quat, quat_conjugate(quat))
    err = torch.cat([term._desired_pos - pos, axis_angle_from_quat(dq)], dim=-1)
    return err * (env.episode_length_buf > 0).unsqueeze(-1)


class JointVelFiniteDiff:
    """(q_t - q_{t-1}) / policy_dt from joint positions at consecutive policy steps; 0 on the first
    step after reset. What the deploy side computes from measured joints (the arm API has no
    velocity), unlike qvel.

    Keyed on env.common_step_counter, so extra compute() calls within a step change nothing.
    Envs on their first step restart the difference from the current joints, which also covers a
    manual env.reset() (it doesn't advance the counter).
    """

    def __init__(self, cfg, env: ManagerBasedRlEnv):
        self.prev = self.cur = None
        self.step = -1

    def __call__(self, env: ManagerBasedRlEnv, asset_cfg) -> torch.Tensor:
        q = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
        if env.common_step_counter != self.step:
            self.step = env.common_step_counter
            self.prev, self.cur = (q.clone() if self.cur is None else self.cur), q.clone()
        fresh = env.episode_length_buf == 0
        self.cur[fresh] = q[fresh]
        self.prev[fresh] = q[fresh]  # so the first step reads 0
        return (self.cur - self.prev) / env.step_dt


def hole_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true hole mouth in the robot base frame."""
    return hole_mouth(env)


def tip_rel_hole(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true peg tip relative to the true hole."""
    return true_tip(env) - hole_mouth(env)


def contact_force(env: ManagerBasedRlEnv) -> torch.Tensor:
    return peg_hole_force(env)


def peg_in_hand(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Critic: true peg pose in the TCP frame (tip position 3 + rot6d 6)."""
    pos, rot = tcp_pose(env)
    peg = env.scene["peg"].data.root_link_pose_w
    rt = rot.transpose(-1, -2)
    rel_pos = (rt @ (peg[:, :3] - pos).unsqueeze(-1)).squeeze(-1)
    return torch.cat([rel_pos, rot6d(rt @ matrix_from_quat(peg[:, 3:7]))], dim=-1)


# Rewards.


def keypoint_tanh(env: ManagerBasedRlEnv, std: float, depth: float) -> torch.Tensor:
    """1 - tanh(d / std), d = true tip to the point `depth` down the hole axis."""
    rel = true_tip(env) - hole_mouth(env)
    rel[:, 2] += depth
    return 1.0 - torch.tanh(rel.norm(dim=-1) / std)


def insertion_depth(env: ManagerBasedRlEnv, depth: float) -> torch.Tensor:
    """Depth below the mouth as a fraction of the success depth, counted only near the axis."""
    lateral, d = tip_in_hole(env)
    inside = lateral < state(env).p["peg_radius"]  # the tip can only be below the mouth in the bore
    return (d.clamp(0.0, depth) / depth) * inside


def success(env: ManagerBasedRlEnv, depth: float, centred: float) -> torch.Tensor:
    lateral, d = tip_in_hole(env)
    return (d >= depth) & (lateral <= centred)


def force_over(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    return (peg_hole_force(env).norm(dim=-1) - threshold).clamp(min=0.0)


# Terminations.


def peg_slipped(env: ManagerBasedRlEnv, max_slip: float) -> torch.Tensor:
    """Peg tip moved more than max_slip from where this episode's grasp put it (or fell).

    Against the sampled grasp, not the nominal one: the grasp randomisation alone puts the tip up
    to ~6 mm off nominal, which used to eat most of the budget.
    """
    pos, rot = tcp_pose(env)
    tip = pos + (rot @ state(env).grasp_tip.unsqueeze(-1)).squeeze(-1)
    return (true_tip(env) - tip).norm(dim=-1) > max_slip


def excess_force(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    return peg_hole_force(env).norm(dim=-1) > threshold


def excess_force_penalty(env: ManagerBasedRlEnv, threshold: float) -> torch.Tensor:
    """excess_force scaled by the recent success rate (InsertState.rate, 0 until a window completes).

    At full weight from the start, runs 4 and 5 learned to hover at the mouth and never found
    success: before any success, pushing was all risk. Scaled, the penalty only bites once success
    is something the policy is giving up by ramming.
    """
    return excess_force(env, threshold).float() * state(env).rate


def gripper_hit(env: ManagerBasedRlEnv, sensor_name: str = "gripper_hole_contact") -> torch.Tensor:
    """Palm or a finger touched the fixture."""
    return (env.scene[sensor_name].data.found > 0).any(dim=-1)


# Curriculum.


def widen_on_success(env: ManagerBasedRlEnv, env_ids, up: float, down: float, step: float) -> torch.Tensor:
    """Widen the start offset and hole bias (difficulty 0..1) when resetting envs mostly succeed.

    Runs at the start of every reset on the envs about to reset, while the termination manager
    still holds why they ended. Outcomes accumulate over a window of num_envs resets (about one
    episode per env, spread out because run_train uses init_at_random_ep_len=True); then difficulty
    moves by at most one `step` and the window restarts. One .item() per window. Clearance can't be
    in this loop: it is compiled mesh geometry, so it is fixed per run (TaskParams.clearance).
    """
    s = state(env)
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0 and env.common_step_counter > 0:
        s.succ += env.termination_manager.get_term("success")[env_ids].float().sum()
        s.done += len(env_ids)
        if s.done >= env.num_envs:
            rate = s.rate = s.succ.item() / s.done
            if rate > up:
                s.difficulty = min(1.0, s.difficulty + step)
            elif rate < down:
                s.difficulty = max(s.p["difficulty_init"], s.difficulty - step)
            s.succ.zero_()
            s.done = 0
    return torch.tensor(s.difficulty, device=env.device)
