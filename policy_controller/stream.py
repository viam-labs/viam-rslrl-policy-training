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
    rel = nominal_tip(kin, cfg, q) - hole_m
    task = cfg["task"]
    return -rel[2] >= task["success_depth_m"] and np.linalg.norm(rel[:2]) <= task["success_centred_m"]


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
