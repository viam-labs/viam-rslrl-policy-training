"""IK is solved once per policy step and the joint target ramps to that solution over the step."""
import torch

from rl_insert.task.checks import make_env


def test_ik_solved_once_and_target_reaches_goal():
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    term = env.action_manager.get_term("ee_delta")
    calls = 0
    solve = term.compute_dq

    def counted():
        nonlocal calls
        calls += 1
        return solve()

    term.compute_dq = counted
    env.step(torch.full((4, 6), 0.5))
    assert calls == 1
    q_goal = term._q_to.clone()
    target = env.scene["robot"].data.joint_pos_target[:, term._joint_ids]
    torch.testing.assert_close(target, q_goal)  # ramp ends on the goal
    assert (q_goal - term._q_from).abs().max() > 1e-4  # it moved


def test_ramp_continues_from_previous_target():
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    term = env.action_manager.get_term("ee_delta")
    env.step(torch.full((4, 6), 0.5))
    prev_goal = term._q_to.clone()
    env.step(torch.full((4, 6), 0.5))
    torch.testing.assert_close(term._q_from, prev_goal)  # no jump back to the lagging measured joints


def test_first_step_after_reset_starts_at_measured_joints():
    env = make_env(num_envs=4, device="cpu")
    env.reset()
    term = env.action_manager.get_term("ee_delta")
    env.step(torch.full((4, 6), 0.5))
    env.reset()
    q = env.scene["robot"].data.joint_pos[:, term._joint_ids].clone()
    env.step(torch.zeros(4, 6))
    torch.testing.assert_close(term._q_from, q)
