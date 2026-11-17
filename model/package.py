"""The model package files beyond policy.onnx: config.json (the deploy contract) and model_metrics.json."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import onnx
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjlab.envs import ManagerBasedRlEnvCfg
from rl_insert.peg_hole import PegHoleParams
from rl_insert.robot import ARM_JOINTS, FLANGE, XARM6_URDF, ArmParams, GripperParams
from rl_insert.task.env_cfg import OBS_DOC, TaskParams

SCHEMA_VERSION = 1

# model_metrics.json name -> tensorboard scalar tag. mjlab logs Episode_Termination/<k> as the number
# of envs that ended for reason k per reset (count_nonzero), averaged by rsl_rl over the iteration,
# so build_metrics turns those into rates: each count over the sum of all of them at that iteration.
METRIC_TAGS = {
    "success_rate": "Episode_Termination/success",
    "slip_rate": "Episode_Termination/peg_slipped",
    "excess_force_rate": "Episode_Termination/excess_force",
    "timeout_rate": "Episode_Termination/time_out",
    "gripper_hit_rate": "Episode_Termination/gripper_hit",
    "mean_reward": "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "curriculum_difficulty": "Curriculum/difficulty",
}
MAX_METRIC_POINTS = 100


def _tensor_contract(value_info) -> dict:
    return {
        "name": value_info.name,
        "dtype": "float32",
        "shape": [d.dim_value for d in value_info.type.tensor_type.shape.dim],
    }


def build_config(onnx_path: Path, env_cfg: ManagerBasedRlEnvCfg, p: TaskParams, run_name: str) -> dict:
    model = onnx.load(onnx_path)
    meta = {e.key: e.value for e in model.metadata_props}
    names = meta["observation_names"].split(",")
    dims = [round(float(d)) for d in meta["observation_dims"].split(",")]
    actor = env_cfg.observations["actor"].terms
    terms, offset = [], 0
    for name, dim in zip(names, dims, strict=True):
        noise = actor[name].noise
        terms.append(
            {
                "name": name,
                "offset": offset,
                "dim": dim,
                **OBS_DOC[name],
                "train_noise": None if noise is None else [noise.n_min, noise.n_max],
            }
        )
        offset += dim

    ik = env_cfg.actions["ee_delta"]
    robot = env_cfg.scene.entities["robot"]
    default_q = robot.init_state.joint_pos
    g, ph, arm = GripperParams(), PegHoleParams(clearance=p.clearance), ArmParams()
    events = env_cfg.events
    return {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "model": {
            "format": "onnx",
            "opset": model.opset_import[0].version,
            "inputs": [_tensor_contract(i) for i in model.graph.input],
            "outputs": [_tensor_contract(o) for o in model.graph.output],
            "obs_normalization": "inside the graph",
        },
        "observation": {"dim": offset, "terms": terms},
        "action": {
            "type": "ee_delta_hole_frame",
            "dim": 6,
            "layout": "[dx, dy, dz, rx, ry, rz]: translation and axis-angle rotation of the TCP",
            "frame": "hole frame axes (origin at the mouth, +z out of the surface, insertion is -z). "
            "Rotate into the arm base frame before use. Applied to the measured TCP pose each "
            "step (target = FK(measured) + delta), rotation pre-multiplied: R_target = R_delta R_tcp.",
            "pos_scale_m": ik.delta_pos_scale,
            "rot_scale_rad": ik.delta_ori_scale,
            "clip": None,
        },
        "control": {
            "policy_dt": env_cfg.sim.mujoco.timestep * env_cfg.decimation,
            "physics_dt": env_cfg.sim.mujoco.timestep,
            "joint_names": list(ARM_JOINTS),
            "default_joint_pos": [default_q[j] for j in ARM_JOINTS],
            "ik": {
                "method": "damped least squares, joint space: (J^T W J + damping^2 I) dq = J^T W e",
                "frame": f"{ik.frame_type} {ik.frame_name}",
                "solved_every_physics_step": True,
                "damping": ik.damping,
                "max_dq_rad": ik.max_dq,
                "position_weight": ik.position_weight,
                "orientation_weight": ik.orientation_weight,
                "joint_limit_weight": ik.joint_limit_weight,
                "posture_weight": ik.posture_weight,
            },
            "joint_servo_sim": {"kp": arm.kp, "dampratio": arm.dampratio, "gravity_compensated": True},
            "gripper": {"close_command": 0.0, **asdict(g)},
        },
        "robot": {
            "model_id": "ufactory-xarm6",
            "urdf": "rl_insert/assets/xarm6/xarm6.urdf",
            "urdf_sha256": hashlib.sha256(XARM6_URDF.read_bytes()).hexdigest(),
            "flange": FLANGE,
            "nominal_tcp": {
                "flange_to_pad_centre_m": g.tcp,
                "protrusion_m": p.protrusion,
                "flange_to_peg_tip_m": [0.0, 0.0, g.tcp + p.protrusion],
            },
        },
        "task": {
            "peg_hole": asdict(ph) | {"hole_radius": ph.hole_radius},
            "clearance_m": p.clearance,
            "nominal_hole_mouth_base_frame": list(p.hole_mouth),
            "pre_insert_tip_height_m": list(p.start_height),
            "success_depth_m": p.success_depth,
            "success_centred_m": p.success_centred,
            "time_limit_s": env_cfg.episode_length_s,
            "force_limit_n": p.force_limit,
        },
        "robustness": {
            "start_lateral_m": p.start_lateral,
            "hole_bias_m": list(p.hole_bias),
            "grasp_offset_m": p.grasp_offset,
            "grasp_tilt_rad": p.grasp_tilt,
            "difficulty_init": p.difficulty_init,
            "hole_friction": list(events["friction_peg_hole"].params["ranges"]),
            "pad_friction": list(events["friction_finger_peg"].params["ranges"]),
            "start_joint_offset_rad": p.start_joint_offset,
            "arm_pd_gain_scale": list(p.pd_gain_scale),
            "action_delay_policy_steps": list(p.action_delay_steps),
            "observation_noise": "per term: observation.terms[].train_noise (uniform)",
            "not_modelled": [
                "force/torque sensing (none on the real arm either)",
                "hole orientation error (the hole is upright, zero yaw; bias is position only)",
                "part dimension tolerances (clearance is fixed per run)",
                "real grasp slip (the grasp is a compliant weld)",
                "joint-target streaming: sim re-solves IK toward the desired pose every 1 ms",
            ],
        },
        "grasp_model": {
            "type": "weld",
            "doc": "sim holds the peg with a compliant weld (gripper body to peg, anchored at the TCP) at "
            "the sampled grasp offset; pads still close on it at a fixed grip force. Real grasp "
            "slip is not modelled: at <= 40 N the G2 (10-50 N grip) holds a 20 g peg.",
            "weld_solref_timeconst_s": p.grasp_solref,
            "grip_force_n": g.grip_force,
        },
    }


def build_metrics(event_dir: Path) -> dict[str, list[float]]:
    ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags()["scalars"])
    series = {tag: {e.step: e.value for e in ea.Scalars(tag)} for tag in METRIC_TAGS.values() if tag in have}
    ends = [v for tag, v in series.items() if tag.startswith("Episode_Termination/")]
    total = {step: sum(e.get(step, 0.0) for e in ends) for step in set().union(*ends)}
    metrics = {}
    for name, tag in METRIC_TAGS.items():
        if tag in series:
            points = sorted(series[tag].items())
            if tag.startswith("Episode_Termination/"):  # count -> rate; skip iterations where none ended
                values = [v / total[step] for step, v in points if total[step] > 0]
            else:
                values = [v for _, v in points]
            stride = max(1, -(-len(values) // MAX_METRIC_POINTS))
            kept = values[::stride]
            if (len(values) - 1) % stride:
                kept.append(values[-1])  # always end on the final iteration
            metrics[name] = kept
    return metrics


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")
