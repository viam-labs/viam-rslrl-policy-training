import asyncio

import numpy as np
import pytest

from duck_policy.io import Snapshot
from duck_policy.loop import Limits, PolicyLoop
from duck_policy.obs import Command

DEFAULT = [0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0, 0, 0, 0.0873, 0.4579, 0.0049, -0.4530]
CFG = {"default_pose": DEFAULT, "action_scale": 1.0, "joint_limits": [[-1.6, 1.6]] * 14,
       "control_hz": 50, "command": {"twist_min": [-0.4, -0.3, -1.0], "twist_max": [0.4, 0.3, 1.0]},
       "safety": {"action_bound": 3.0, "max_step_rad": 0.5}}


class FakeIO:
    """Sim time advances 20 ms per read; gravity and generation can be scripted per read."""

    def __init__(self, gravity=lambda i: [0, 0, -1], generation=lambda i: 1, frozen=False, hang_at=None):
        self.i, self.writes, self.torques = 0, [], []
        self.gravity, self.generation, self.frozen, self.hang_at = gravity, generation, frozen, hang_at

    def _snap(self):
        return Snapshot(t=0.0 if self.frozen else 0.02 * self.i, q=list(DEFAULT), dq=[0.0] * 14, gyro=[0, 0, 0],
                        gravity=self.gravity(self.i), generation=self.generation(self.i), torque=True)

    async def read(self):
        self.i += 1
        if self.hang_at is not None and self.i >= self.hang_at:
            await asyncio.sleep(1.0)
        return self._snap()

    async def write(self, targets):
        self.writes.append(list(targets))

    async def torque(self, on):
        self.torques.append(on)
        return self._snap()


def zero_policy(obs):
    return np.zeros(14, np.float32)


def loop(io, infer=zero_policy, ticks=200, **limits):
    lp = PolicyLoop(io, infer, CFG, Limits(ramp_s=0.1, **limits))
    lp.command = Command(0.1, 0.0, 0.0)
    return lp, asyncio.run(lp.run(max_ticks=ticks))


def test_a_healthy_run_writes_default_plus_action_and_counts_ticks():
    io = FakeIO()
    lp, reason = loop(io, ticks=20)
    assert reason == "max_ticks" and lp.ticks == 20
    assert io.writes[-1] == pytest.approx(DEFAULT, abs=1e-6)  # zero action: the default pose
    assert lp.last_obs[48:51] == pytest.approx([0.1, 0.0, 0.0])


def test_fall_goes_limp_after_three_ticks():
    io = FakeIO(gravity=lambda i: [0.9, 0, -0.3] if i > 10 else [0, 0, -1])
    lp, reason = loop(io)
    assert reason == "fall" and io.torques == [True, False]


def test_bad_action_goes_limp():
    io = FakeIO()
    lp, reason = loop(io, infer=lambda obs: np.full(14, np.nan, np.float32))
    assert reason == "bad_action" and io.torques[-1] is False
    io2 = FakeIO()
    _, reason2 = loop(io2, infer=lambda obs: np.full(14, 9.0, np.float32))
    assert reason2 == "bad_action"


def test_stale_state_and_rebuild_stop_and_hold():
    io = FakeIO(frozen=True)
    _, reason = loop(io)
    assert reason == "stale_state" and io.torques == [True]  # held, not limp
    io2 = FakeIO(generation=lambda i: 1 if i < 20 else 2)
    _, reason2 = loop(io2)
    assert reason2 == "rebuilt" and io2.torques == [True]


def test_hung_call_stops():
    io = FakeIO(hang_at=15)
    _, reason = loop(io, call_timeout_s=0.05)
    assert reason == "hung_call"


def test_targets_are_clipped_and_rate_capped():
    io = FakeIO()
    lp, _ = loop(io, infer=lambda obs: np.full(14, 2.5, np.float32), ticks=60)
    w = np.array(io.writes)
    assert (w <= 1.6 + 1e-9).all()  # joint limit
    assert np.abs(np.diff(w, axis=0)).max() <= 0.5 + 1e-9  # max_step_rad


def test_set_command_is_clipped():
    lp = PolicyLoop(FakeIO(), zero_policy, CFG, Limits())
    lp.set_command(Command(1.0, 0.0, -5.0))
    assert (lp.command.vx, lp.command.wz) == (0.4, -1.0) and lp.command_clipped
