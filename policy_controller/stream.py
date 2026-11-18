"""Pure pieces of the controller loop: stream timing, ramp, pre-insert pose, success. No viam-sdk."""

import numpy as np

from policy_controller.core import pose_error, quat_from_mat

SUBSTEPS = 5


def point_times(tick, policy_dt):
    """Seconds for tick's SUBSTEPS points. SDK contract: absolute from the stream start, one tick ahead, so tick k
    fills [(k+1)dt, (k+2)dt] at +dt/5 steps. Task 7 (sim probe) confirmed it: mujoco-sim plays one timeline across
    batches but pauses it while starved, so stream-absolute times hold when streaming one tick ahead. If a module
    took per-batch-relative times instead, change this to `[(j + 1) * policy_dt / SUBSTEPS for j in range(SUBSTEPS)]`
    (the first batch's t=0 point stays)."""
    return [(tick + 1) * policy_dt + (j + 1) * policy_dt / SUBSTEPS for j in range(SUBSTEPS)]


def ramp(q_from, q_goal):
    """SUBSTEPS joint vectors at fractions 0.2 .. 1.0 from q_from to q_goal (training's ramp)."""
    q_from, q_goal = np.asarray(q_from, float), np.asarray(q_goal, float)
    return [q_from + (q_goal - q_from) * (j + 1) / SUBSTEPS for j in range(SUBSTEPS)]


def nominal_tip(kin, cfg, q):
    pos, rot = kin.fk(q)
    return pos + rot[:, 2] * cfg["robot"]["nominal_tcp"]["protrusion_m"]


def succeeded(kin, cfg, q, hole_m):
    """Training's success (tip success_depth_m deep, within success_centred_m of the axis) judged against the
    *estimated* hole, so both tolerances widen by the hole-estimate error training covered
    (robustness.hole_bias_m). Worst case that is still 13 mm into the hole: it can't be anywhere else."""
    rel = nominal_tip(kin, cfg, q) - hole_m
    task, bias = cfg["task"], np.asarray(cfg["robustness"]["hole_bias_m"])
    return (-rel[2] >= task["success_depth_m"] - bias[2]
            and np.linalg.norm(rel[:2]) <= task["success_centred_m"] + np.linalg.norm(bias[:2]))


def orientation_vector(rot):
    """(o, theta rad) of a rotation matrix in Viam's orientation vector convention: o is the z axis; theta spins
    about o from the x axis of Rz(lon) Ry(lat), lon/lat being o's azimuth and polar angle (pole: lon = 0)."""
    o = rot[:, 2]
    az, el = np.arctan2(o[1], o[0]) if 1 - abs(o[2]) > 1e-4 else 0.0, np.arccos(np.clip(o[2], -1, 1))
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    ry = np.array([[np.cos(el), 0, np.sin(el)], [0, 1, 0], [-np.sin(el), 0, np.cos(el)]])
    x_ref = rz @ ry @ [1.0, 0, 0]
    theta = np.arctan2(o @ np.cross(x_ref, rot[:, 0]), x_ref @ rot[:, 0])
    return o, theta


def pre_insert(kin, cfg, hole_m, height_m=0.015):
    """(q, flange pose) with the nominal tip height_m above hole_m and the TCP pointing as at default_joint_pos
    (down). Pose: x, y, z in mm; o_x, o_y, o_z, theta in degrees (Viam orientation vector), arm base frame."""
    ctl = cfg["control"]
    q = np.array(ctl["default_joint_pos"], float)
    _, rot_t = kin.fk(q)
    quat_t = quat_from_mat(rot_t)
    p_t = np.asarray(hole_m, float) + [0, 0, height_m] - rot_t[:, 2] * cfg["robot"]["nominal_tcp"]["protrusion_m"]
    for _ in range(100):
        pos, rot = kin.fk(q)
        jp, jr = kin.jac()
        e = np.concatenate(pose_error(pos, quat_from_mat(rot), p_t, quat_t))
        q = q + np.linalg.solve(np.vstack([jp, jr]).T @ np.vstack([jp, jr]) + 1e-6 * np.eye(len(q)), np.vstack([jp, jr]).T @ e)
    pos, rot = kin.fk(q)
    flange = pos - rot[:, 2] * ctl["gripper"]["tcp"]
    o, theta = orientation_vector(rot)
    pose = dict(zip(["x", "y", "z"], (flange * 1000).tolist())) | dict(
        zip(["o_x", "o_y", "o_z"], o.tolist()), theta=float(np.degrees(theta))
    )
    return q, pose


def slip_or_jam(kin, cfg, q, hole_m, in_hole_s, jam_s):
    """Why to stop pushing, or None. The controller can't see the peg, only the arm's nominal tip:
    - "slip": the nominal tip is deeper than the peg could be (hole depth + the hole-estimate error +
      training's grasp offset along the tool), so the peg has slid up in the fingers;
    - "jam": the tip has been in the hole (deeper than the estimate error) for jam_s without succeeding.
    Returns (reason or None, in_hole): in_hole tells the caller whether to keep the in-hole clock running."""
    rel = nominal_tip(kin, cfg, q) - hole_m
    bias, grasp = np.asarray(cfg["robustness"]["hole_bias_m"]), cfg["robustness"]["grasp_offset_m"]
    depth = -rel[2]
    if depth > cfg["task"]["peg_hole"]["depth"] + bias[2] + grasp:
        return "slip", True
    in_hole = depth > bias[2] + 0.002
    return ("jam" if in_hole and in_hole_s > jam_s else None), in_hole
