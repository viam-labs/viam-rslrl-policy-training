"""PegInsert-xArm6: the mjlab env config.

Entities (mjlab allows one freejoint per entity, so the peg is its own entity):
- robot: xArm6 + G2 gripper from rl_insert.robot, keeping its MjSpec actuators (XmlActuatorCfg).
  Fixed base; mjlab wraps it in a mocap body at the origin. The gripper is commanded closed
  (ctrl 0, squeezing at grip_force) and is not part of the action.
- peg: free body from rl_insert.peg_hole.add_peg (frame = tip), held by a compliant weld to the
  gripper body (_add_grasp_weld) whose pose is the per-env sampled grasp.
- hole: from add_hole; fixed, so mjlab wraps it in a mocap body and the reset moves it per env.

Physics keeps what the foundations plan validated: mesh peg/hole, solref 0.002 on peg/hole/pads
and the finger equality, 1 ms timestep, elliptic cone, impratio 10, gravcomp, and the Euler
integrator the contact tests ran with (mjlab defaults to implicitfast).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp import events as mj_events
from mjlab.envs.mdp import observations as mj_obs
from mjlab.envs.mdp import rewards as mj_rew
from mjlab.envs.mdp import terminations as mj_term
from mjlab.envs.mdp.actions import DifferentialIKActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from rl_insert.peg_hole import TIMESTEP, PegHoleParams, add_hole, add_peg
from rl_insert.robot import ARM_JOINTS, GripperParams, add_gripper, load_xarm6

from . import mdp


@dataclass(frozen=True)
class TaskParams:
    """Everything a run varies. Becomes the training task config / config.json `task` section."""

    clearance: float = 0.0005  # fixed per run: compiled mesh geometry, see mdp.widen_on_success
    hole_mouth: tuple[float, float, float] = (0.4, 0.0, 0.05)  # nominal, robot base frame
    protrusion: float = 0.02  # peg tip past the TCP (pad centre)
    start_height: tuple[float, float] = (0.010, 0.020)  # tip above the mouth at reset
    start_lateral: float = 0.003  # +/- m at full difficulty
    hole_bias: tuple[float, float, float] = (0.002, 0.002, 0.001)  # +/- m at full difficulty
    difficulty_init: float = 0.3
    grasp_offset: float = 0.003  # +/- m along TCP x and z
    grasp_tilt: float = float(np.deg2rad(3.0))  # +/- rad about TCP y (the pad normal)
    success_depth: float = 0.015  # of the 20 mm bore
    success_centred: float = 0.001
    force_limit: float = 40.0  # N, terminate; keep below the xArm's collision trip
    force_penalty_from: float = 15.0  # N
    max_slip: float = 0.01  # m
    action_pos_scale: float = 0.002  # m per unit action per policy step
    action_rot_scale: float = float(np.deg2rad(1.0))  # rad per unit action per policy step
    # The peg is held by a compliant weld (see _add_grasp_weld), so grip force is fixed at
    # GripperParams.grip_force and no longer randomized: it only presses the pads on a welded peg.
    grasp_solref: float = 0.002  # s, weld time constant; see _add_grasp_weld for the measured compliance
    start_joint_offset: float = 0.02  # +/- rad per arm joint at reset (TCP tilt up to a few degrees)
    pd_gain_scale: tuple[float, float] = (0.8, 1.2)  # arm servo kp and kd, scaled per episode
    action_delay_steps: tuple[int, int] = (0, 2)  # policy steps; constant within an episode
    episode_s: float = 10.0
    decimation: int = 50  # 1 ms physics, 20 Hz policy


PPO_GAMMA = 0.99  # discount, used here to size the success bonus (runner.py uses it for PPO)

# Units and frame of each actor observation term, for config.json. Keep in step with `actor` below.
OBS_DOC: dict[str, dict[str, str]] = {
    "tip_rel_est_hole": {
        "units": "m",
        "frame": "hole",
        "doc": "nominal peg tip (TCP + protrusion along TCP z) minus estimated hole mouth",
    },
    "ee_rot6d": {
        "units": "rot6d",
        "frame": "hole",
        "doc": "TCP rotation in the hole frame, columns x then y, 3 each",
    },
    "joint_pos": {"units": "rad", "frame": "joint", "doc": "arm joints minus control.default_joint_pos"},
    "joint_vel": {
        "units": "rad/s",
        "frame": "joint",
        "doc": "(q_t - q_{t-1}) / policy_dt from measured arm joint positions at consecutive policy "
        "steps; 0 on the first step after reset",
    },
    "last_action": {"units": "action", "frame": "hole", "doc": "previous raw policy output, unscaled"},
    "ik_error": {
        "units": "m (0:3), rad axis-angle (3:6)",
        "frame": "hole",
        "doc": "the IK action's desired TCP pose minus the measured TCP pose, world (= hole) axes. "
        "At the previous policy step the action a (6, unscaled) set, from the TCP pose (p, R) "
        "measured then: p_des = p + pos_scale * a[0:3], R_des = R_delta R with R_delta the "
        "rotation of axis-angle rot_scale * a[3:6]. Now, with the measured TCP pose (p_m, R_m): "
        "[0:3] = p_des - p_m, [3:6] = axis-angle (rotation vector, |.| <= pi) of q_des * q_m^-1. "
        "0 on the first step after reset",
    },
}


def nominal_joint_pos(p: TaskParams, standoff: float | None = None) -> np.ndarray:
    """Arm joints putting the nominal peg tip `standoff` above the nominal hole mouth, TCP pointing
    straight down with the same yaw as the xArm's zero pose (TCP x = -base x).

    Damped least squares on CPU MuJoCo; deterministic, ~ms, so it runs at config time instead of
    being a stale constant.
    """
    standoff = sum(p.start_height) / 2 if standoff is None else standoff
    spec = load_xarm6()
    add_gripper(spec)
    m = spec.compile()
    d = mujoco.MjData(m)
    site = m.site("tcp").id
    target_pos = np.array(p.hole_mouth) + [0.0, 0.0, p.protrusion + standoff]
    target_rot = np.diag([-1.0, 1.0, -1.0])
    q = np.array([0.0, 0.3, -1.0, 0.0, 0.7, 0.0])  # elbow up, wrist down
    arm = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(200):
        d.qpos[arm] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        rot = d.site_xmat[site].reshape(3, 3)
        err_rot = 0.5 * sum(np.cross(rot[:, i], target_rot[:, i]) for i in range(3))
        err = np.concatenate([target_pos - d.site_xpos[site], err_rot])
        if np.linalg.norm(err) < 1e-9:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, site)
        jac = np.vstack([jacp, jacr])[:, arm]
        q = q + jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(6), err)
    assert np.linalg.norm(err) < 1e-6, f"nominal IK did not converge: {err}"
    return q


def _robot_spec(g: GripperParams) -> mujoco.MjSpec:
    spec = load_xarm6()
    add_gripper(spec, g)
    # The arm-link meshes only matter for collisions the task never has; skip their narrowphase.
    for body in spec.bodies[1:]:
        if body.name.startswith("link"):
            for geom in body.geoms:
                geom.contype = geom.conaffinity = 0
    # Pads win friction mixing against the peg (MuJoCo takes the max otherwise), so finger-peg
    # friction is exactly the pads' value and can be randomized on its own.
    for name in ("pad_left", "pad_right"):
        spec.geom(name).priority = 1
    return spec


def _hole_spec(ph: PegHoleParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = add_hole(spec, spec.worldbody, ph)
    for i, geom in enumerate(body.geoms):
        geom.name = geom.name or f"hole_piece_{i}"  # add_hole leaves the ring pieces unnamed
        geom.priority = 1  # hole friction wins against the peg, as for the pads
    return spec


def _peg_spec(ph: PegHoleParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    add_peg(spec, spec.worldbody, ph)
    return spec


def _add_grasp_weld(spec: mujoco.MjSpec, timeconst: float) -> None:
    """Weld the peg to the gripper body: the grasp model.

    On MuJoCo Warp the pads alone let the peg creep (up to ~0.4 mm/s) and get knocked up to 5 mm
    along its axis in the first 0.4 s after reset, whatever the grip force or finger preload.
    The real G2 holds a 20 g peg rigidly at these loads, so the weld carries it; the fingers still
    close on it for contact and looks. The pose here is a placeholder: ResetPegInHand writes each
    world's grasp pose into eq_data. An explicit identity quat leaves eq_data[11] (the "compute
    from qpos0" mark) at 0, so set_const never overwrites the per-world pose.

    Measured on CPU Warp (64 envs): timeconst 0.002 s gives settle 0.14 mm / creep 0.07 mm over
    2 s, and 20 N moves the peg in the hand ~0.2 mm axially / ~0.5 mm laterally (max 0.7 / 1.2).
    0.005 s: settle 0.4, creep 0.28, lateral 20 N up to 2.5 mm; 0.01 s fails the 0.5 mm gates.
    """
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name=mdp.GRASP_WELD,
        objtype=mujoco.mjtObj.mjOBJ_BODY,
        name1="robot/gripper",
        name2="peg/peg",
        solref=[timeconst, 1.0],
        data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    )


def peg_insert_env_cfg(p: TaskParams = TaskParams(), play: bool = False) -> ManagerBasedRlEnvCfg:
    ph = PegHoleParams(clearance=p.clearance)
    g = GripperParams()
    q0 = nominal_joint_pos(p)
    pd = asdict(p) | {"peg_radius": ph.peg_radius}  # what the terms read via env.insert_state

    robot = EntityCfg(
        spec_fn=lambda: _robot_spec(g),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={**dict(zip(ARM_JOINTS, q0.tolist())), "finger_.*": ph.peg_radius},
        ),
        articulation=EntityArticulationInfoCfg(
            actuators=(
                # Action latency: the arm's joint targets reach the servos 0-2 policy steps late,
                # one lag per episode (hold_prob 1: sampled at reset, then held).
                XmlActuatorCfg(
                    target_names_expr=ARM_JOINTS,
                    delay_min_lag=p.action_delay_steps[0] * p.decimation,
                    delay_max_lag=p.action_delay_steps[1] * p.decimation,
                    delay_hold_prob=1.0,
                ),
                XmlActuatorCfg(target_names_expr=("finger_left",)),  # the gripper; target stays 0 = closed
            ),
        ),
    )
    scene = SceneCfg(
        num_envs=4,
        env_spacing=1.0,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={
            "robot": robot,
            "peg": EntityCfg(spec_fn=lambda: _peg_spec(ph)),
            "hole": EntityCfg(
                spec_fn=lambda: _hole_spec(ph),
                init_state=EntityCfg.InitialStateCfg(pos=p.hole_mouth),
            ),
        },
        spec_fn=lambda spec: _add_grasp_weld(spec, p.grasp_solref),
        sensors=(
            ContactSensorCfg(
                name="peg_hole_contact",
                primary=ContactMatch(mode="body", pattern="peg", entity="peg"),
                secondary=ContactMatch(mode="body", pattern="hole", entity="hole"),
                fields=("found", "force"),
                reduce="netforce",  # one net wrench, world frame
            ),
        ),
    )

    arm = SceneEntityCfg("robot", joint_names=ARM_JOINTS)
    actor = {
        "tip_rel_est_hole": ObservationTermCfg(
            func=mdp.tip_rel_est_hole, noise=Unoise(n_min=-5e-4, n_max=5e-4)
        ),
        "ee_rot6d": ObservationTermCfg(func=mdp.ee_rot6d, noise=Unoise(n_min=-5e-3, n_max=5e-3)),
        "joint_pos": ObservationTermCfg(
            func=mj_obs.joint_pos_rel, params={"asset_cfg": arm}, noise=Unoise(n_min=-1e-3, n_max=1e-3)
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.JointVelFiniteDiff, params={"asset_cfg": arm}, noise=Unoise(n_min=-0.02, n_max=0.02)
        ),
        "last_action": ObservationTermCfg(func=mj_obs.last_action),
        "ik_error": ObservationTermCfg(func=mdp.ik_tracking_error, noise=Unoise(n_min=-2e-4, n_max=2e-4)),
    }
    critic = {
        **actor,
        "hole_pos": ObservationTermCfg(func=mdp.hole_pos),
        "tip_rel_hole": ObservationTermCfg(func=mdp.tip_rel_hole),
        "contact_force": ObservationTermCfg(func=mdp.contact_force, scale=0.05),
        "peg_in_hand": ObservationTermCfg(func=mdp.peg_in_hand),
    }

    depth = p.success_depth
    dense = {"keypoint_coarse": 1.0, "keypoint_fine": 1.0, "depth": 2.0}  # each term is in [0, 1]
    # Rewards are scaled by step_dt, so hovering just short of success can earn at most
    # sum(dense) * dt per step, sum(dense) * dt / (1 - gamma) = 20 discounted. Success ends the
    # episode, so its one-off bonus (weight * dt) must beat that: twice the bound, weight 800 (40).
    success_weight = 2 * sum(dense.values()) / (1 - PPO_GAMMA)
    cfg = ManagerBasedRlEnvCfg(
        scene=scene,
        observations={
            "actor": ObservationGroupCfg(actor, enable_corruption=not play),
            "critic": ObservationGroupCfg(critic, enable_corruption=False),
        },
        actions={
            "ee_delta": DifferentialIKActionCfg(
                entity_name="robot",
                actuator_names=ARM_JOINTS,
                frame_type="site",
                frame_name="tcp",
                use_relative_mode=True,  # world-frame delta from the *measured* TCP pose each step
                delta_pos_scale=p.action_pos_scale,
                delta_ori_scale=p.action_rot_scale,
            ),
        },
        events={
            "reset_scene": EventTermCfg(func=mj_events.reset_scene_to_default, mode="reset"),
            # Before reset_peg_hole: it places the peg and hole from the TCP pose these joints give.
            "start_pose": EventTermCfg(
                func=mj_events.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-p.start_joint_offset, p.start_joint_offset),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": arm,
                },
            ),
            "reset_peg_hole": EventTermCfg(func=mdp.ResetPegInHand, mode="reset", params={"p": pd}),
            "arm_pd_gains": EventTermCfg(
                func=dr.pd_gains,
                mode="reset",
                params={
                    # actuator_ids index entity.actuators: [0] is the arm XmlActuatorCfg group
                    "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0]),
                    "kp_range": p.pd_gain_scale,
                    "kd_range": p.pd_gain_scale,
                },
            ),
            "friction_peg_hole": EventTermCfg(
                func=dr.geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("hole", geom_names=(".*",)),
                    "ranges": (0.3, 0.8),
                    "shared_random": True,
                },
            ),
            "friction_finger_peg": EventTermCfg(
                func=dr.geom_friction,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", geom_names=("pad_.*",)),
                    "ranges": (0.7, 1.3),
                    "shared_random": True,
                },
            ),
        },
        rewards={
            "keypoint_coarse": RewardTermCfg(
                func=mdp.keypoint_tanh, weight=dense["keypoint_coarse"], params={"std": 0.02, "depth": depth}
            ),
            "keypoint_fine": RewardTermCfg(
                func=mdp.keypoint_tanh, weight=dense["keypoint_fine"], params={"std": 0.003, "depth": depth}
            ),
            "depth": RewardTermCfg(func=mdp.insertion_depth, weight=dense["depth"], params={"depth": depth}),
            "success": RewardTermCfg(
                func=mdp.success, weight=success_weight, params={"depth": depth, "centred": p.success_centred}
            ),
            "action_rate": RewardTermCfg(func=mj_rew.action_rate_l2, weight=-0.01),
            "joint_vel": RewardTermCfg(func=mj_rew.joint_vel_l2, weight=-0.01, params={"asset_cfg": arm}),
            "force": RewardTermCfg(
                func=mdp.force_over, weight=-0.05, params={"threshold": p.force_penalty_from}
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mj_term.time_out, time_out=True),
            "success": TerminationTermCfg(
                func=mdp.success, params={"depth": depth, "centred": p.success_centred}
            ),
            "peg_slipped": TerminationTermCfg(func=mdp.peg_slipped, params={"max_slip": p.max_slip}),
            "excess_force": TerminationTermCfg(func=mdp.excess_force, params={"threshold": p.force_limit}),
        },
        curriculum={
            "difficulty": CurriculumTermCfg(
                func=mdp.widen_on_success, params={"up": 0.8, "down": 0.3, "step": 0.05}
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.ASSET_BODY,
            entity_name="hole",
            body_name="hole",
            distance=0.35,
            elevation=-20.0,
            azimuth=135.0,
        ),
        sim=SimulationCfg(
            nconmax=64,  # per world; see scripts/smoke.py for the peak seen
            njmax=400,
            mujoco=MujocoCfg(timestep=TIMESTEP, integrator="euler", cone="elliptic", impratio=10),
        ),
        decimation=p.decimation,
        episode_length_s=p.episode_s,
    )
    if play:
        cfg.curriculum = {}
    return cfg
