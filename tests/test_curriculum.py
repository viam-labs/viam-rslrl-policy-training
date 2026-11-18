"""widen_on_success moves difficulty one step per num_envs resets, not per reset call."""

from types import SimpleNamespace

import torch

from rl_insert.task import mdp

KW = dict(up=0.8, down=0.3, step=0.05)


def make(n=10, difficulty=0.5):
    succ = torch.zeros(n, dtype=torch.bool)
    s = mdp.InsertState.__new__(mdp.InsertState)
    s.p = {"difficulty_init": 0.3}
    s.difficulty = difficulty
    s.succ = torch.zeros((), device="cpu")
    s.done = 0
    s.rate = 0.0
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        common_step_counter=1,
        insert_state=s,
        termination_manager=SimpleNamespace(get_term=lambda name: succ),
    )
    return env, s, succ


def call(env, ids):
    return mdp.widen_on_success(env, torch.tensor(ids), **KW)


def test_no_change_before_window():
    env, s, succ = make()
    succ[:] = True
    call(env, [0, 1, 2, 3, 4])
    assert s.difficulty == 0.5
    out = call(env, [5, 6, 7, 8])
    assert s.difficulty == 0.5
    assert isinstance(out, torch.Tensor) and out.device.type == "cpu" and float(out) == 0.5


def test_rises_once_per_window():
    env, s, succ = make()
    succ[:] = True
    call(env, [0, 1, 2, 3, 4])
    call(env, [5, 6, 7, 8, 9])
    assert abs(s.difficulty - 0.55) < 1e-9
    assert s.done == 0 and float(s.succ) == 0


def test_falls_but_not_below_init():
    env, s, succ = make(difficulty=0.32)
    call(env, list(range(10)))
    assert abs(s.difficulty - 0.3) < 1e-9
    call(env, list(range(10)))
    assert abs(s.difficulty - 0.3) < 1e-9


def test_window_rate_scales_excess_force_penalty(monkeypatch):
    env, s, succ = make()
    succ[:4] = True
    call(env, list(range(10)))
    assert abs(s.rate - 0.4) < 1e-9
    monkeypatch.setattr(mdp, "excess_force", lambda env, threshold: torch.tensor([True, False]))
    assert torch.allclose(mdp.excess_force_penalty(env, 40.0), torch.tensor([0.4, 0.0]))
