"""The controller's kinematics: FK and the once-per-step DLS IK, ported from training (mjlab). numpy and mujoco only."""

import mujoco
import numpy as np


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_from_mat(rot):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot).reshape(9))
    return quat


def axis_angle_from_quat(quat, eps=1e-6):
    """mjlab's axis_angle_from_quat: shortest-way rotation vector."""
    quat = quat * (-1.0 if quat[0] < 0 else 1.0)
    half = np.arctan2(np.linalg.norm(quat[1:]), quat[0])
    angle = 2.0 * half
    k = np.sin(half) / angle if abs(angle) > eps else 0.5 - angle * angle / 48
    return quat[1:] / k


def apply_delta_pose(pos, quat, delta, eps=1e-6):
    """mjlab's apply_delta_pose: translate, and pre-multiply the rotation by the axis-angle delta."""
    angle = np.linalg.norm(delta[3:])
    if angle > eps:
        axis = delta[3:] / angle
        dq = np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])
        dq /= np.linalg.norm(dq)
    else:
        dq = np.array([1.0, 0, 0, 0])
    return pos + delta[:3], quat_mul(dq, quat)


def pose_error(pos, quat, pos_des, quat_des):
    """mjlab's compute_pose_error: position difference and axis-angle of q_des * q^-1."""
    quat_inv = quat * np.array([1.0, -1, -1, -1]) / (quat @ quat)
    return pos_des - pos, axis_angle_from_quat(quat_mul(quat_des, quat_inv))


class Kinematics:
    def __init__(self, xml_path, control: dict):
        self.ctl = control
        ik = control["ik"]
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.site = self.model.site(ik["site"]).id
        joints = [self.model.joint(n) for n in control["joint_names"]]
        self.qadr = [j.qposadr[0] for j in joints]
        self.dofadr = [j.dofadr[0] for j in joints]
        self.p_des = self.quat_des = None  # the last step_goal's desired pose (the ik_error observation)

    def fk(self, q):
        """TCP (pos (3,), rot (3, 3)) in the arm base frame."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)  # mj_jacSite needs it
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3).copy()

    def jac(self):
        """Site Jacobian (3x6 each) at the q of the last fk()."""
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site)
        return jacp[:, self.dofadr], jacr[:, self.dofadr]

    def step_goal(self, q, a, action_cfg):
        """Joint goal q + dq: one DLS step at the measured q towards the pose the action asks for."""
        ik = self.ctl["ik"]
        q = np.asarray(q, float)
        pos, rot = self.fk(q)
        quat = quat_from_mat(rot)
        a = np.asarray(a, float)
        delta = np.concatenate([a[:3] * action_cfg["pos_scale_m"], a[3:] * action_cfg["rot_scale_rad"]])
        self.p_des, self.quat_des = apply_delta_pose(pos, quat, delta)
        pos_err, rot_err = pose_error(pos, quat, self.p_des, self.quat_des)
        jacp, jacr = self.jac()
        wp2, wo2 = ik["position_weight"] ** 2, ik["orientation_weight"] ** 2
        jtj = wp2 * jacp.T @ jacp + wo2 * jacr.T @ jacr
        jtdx = wp2 * jacp.T @ pos_err + wo2 * jacr.T @ rot_err
        lower, upper = np.array(ik["joint_limits_rad"]).T
        r_limit = np.minimum(upper - q, 0) + np.maximum(lower - q, 0)
        wl2 = ik["joint_limit_weight"] ** 2 * (r_limit != 0)
        wpost2 = ik["posture_weight"] ** 2
        jtj += np.diag(wl2 + wpost2 + max(ik["damping"], 1e-6) ** 2)
        jtdx += wl2 * r_limit + wpost2 * (np.array(ik["posture_target"]) - q)
        dq = np.linalg.solve(jtj, jtdx)
        return q + np.clip(dq, -ik["max_dq_rad"], ik["max_dq_rad"])


class Policy:
    """Episode state and observation recipes (config.json observation.terms, env_cfg.OBS_DOC). One per episode stream."""

    def __init__(self, config: dict, kinematics: Kinematics):
        self.cfg, self.kin = config, kinematics
        self.reset()

    def reset(self):
        self.q_prev = self.p_des = self.quat_des = None
        self.last_action = np.zeros(self.cfg["action"]["dim"])

    def observe(self, q, hole_mouth):
        q = np.asarray(q, float)
        ctl, dt = self.cfg["control"], self.cfg["control"]["policy_dt"]
        pos, rot = self.kin.fk(q)
        recipes = {
            "tip_rel_est_hole": lambda: pos + rot[:, 2] * self.cfg["robot"]["nominal_tcp"]["protrusion_m"] - hole_mouth,
            "ee_rot6d": lambda: rot[:, :2].T.reshape(6),
            "joint_pos": lambda: q - ctl["default_joint_pos"],
            "joint_vel": lambda: np.zeros(len(q)) if self.q_prev is None else (q - self.q_prev) / dt,
            "last_action": lambda: self.last_action,
            "ik_error": lambda: np.zeros(6)
            if self.p_des is None
            else np.concatenate(pose_error(pos, quat_from_mat(rot), self.p_des, self.quat_des)),
        }
        out = []
        for term in self.cfg["observation"]["terms"]:
            if term["name"] not in recipes:
                raise ValueError(f"no recipe for observation term {term['name']}")
            out.append(np.asarray(recipes[term["name"]](), float))
        return np.concatenate(out)

    def act(self, q, hole_mouth, action):
        """q_goal for this step's raw action; records what the next observe() needs."""
        q_goal = self.kin.step_goal(q, action, self.cfg["action"])
        self.p_des, self.quat_des = self.kin.p_des, self.kin.quat_des
        self.q_prev, self.last_action = np.asarray(q, float), np.asarray(action, float)
        return q_goal
