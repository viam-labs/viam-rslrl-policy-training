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

    def __init__(self, gravity=lambda i: [0, 0, -1], generation=lambda i: 1, frozen=False, hang_at=None,
                 q=None, slow_at=None, repeat_at=None, torque_fails=False, reset_at=None):
        self.i, self.writes, self.torques = 0, [], []
        self.q, self.slow_at, self.repeat_at, self.torque_fails = q or DEFAULT, slow_at, repeat_at, torque_fails
        self.gravity, self.generation, self.frozen, self.hang_at = gravity, generation, frozen, hang_at
        self.reset_at = reset_at  # after this read index, t counts from 0 again and generation is 2

    def _snap(self):
        after = self.reset_at is not None and self.i > self.reset_at
        i = self.i - self.reset_at if after else self.i
        return Snapshot(t=0.0 if self.frozen else 0.02 * (i - (self.i == self.repeat_at)), q=list(self.q), dq=[0.0] * 14, gyro=[0, 0, 0],
                        gravity=self.gravity(self.i), generation=2 if after else self.generation(self.i), torque=True)

    async def read(self):
        self.i += 1
        if self.hang_at is not None and self.i >= self.hang_at:
            await asyncio.sleep(1.0)
        if self.i == self.slow_at:
            await asyncio.sleep(0.06)
        return self._snap()

    async def write(self, targets):
        self.writes.append(list(targets))

    async def torque(self, on):
        if self.torque_fails and not on:
            raise RuntimeError("grpc down")
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


def test_stale_state_stops_and_holds():
    io = FakeIO(frozen=True)
    _, reason = loop(io)
    assert reason == "stale_state" and io.torques == [True]  # held, not limp


def test_world_reset_resyncs_and_keeps_running():
    io, seen = FakeIO(reset_at=20), []

    def infer(obs):
        seen.append(obs.copy())
        return np.full(14, 2.0, np.float32)

    lp, reason = loop(io, infer=infer, ticks=40)
    assert reason == "max_ticks" and lp.status()["resets"] == 1 and io.torques == [True]
    assert seen[20][34:48].any() and not seen[21][34:48].any()  # seen[0] is the warmup call; reset tick is seen[21]
    assert np.abs(np.array(io.writes[20]) - DEFAULT).max() <= 0.5 + 1e-9  # rate cap from snap.q


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


def test_warmup_infer_runs_first_and_is_reported():
    io, seen = FakeIO(), []

    def infer(obs):
        seen.append((len(io.torques), obs.shape))
        return np.zeros(14, np.float32)

    lp, _ = loop(io, infer=infer, ticks=3)
    assert seen[0] == (0, (61,))  # before torque(True)
    assert lp.status()["warmup_ms"] is not None


def test_hung_warmup_stops():
    async def slow(obs):
        await asyncio.sleep(1.0)

    lp = PolicyLoop(FakeIO(), slow, CFG, Limits(warmup_timeout_s=0.05))
    assert asyncio.run(lp.run(max_ticks=5)) == "hung_call"


def test_overrun_is_counted_not_caught_up():
    lp, reason = loop(FakeIO(slow_at=15), ticks=40, call_timeout_s=0.2)
    assert reason == "max_ticks" and lp.late == 1


def test_one_repeated_t_continues_three_stop():
    _, reason = loop(FakeIO(repeat_at=12), ticks=30)
    assert reason == "max_ticks"


def test_limp_survives_a_failed_torque_off():
    _, reason = loop(FakeIO(gravity=lambda i: [0.9, 0, -0.3], torque_fails=True))
    assert reason.startswith("fall") and "torque off failed" in reason


def test_ramp_writes_before_policy_when_away_from_default():
    io = FakeIO(q=[0.0] * 14)
    lp, _ = loop(io, ticks=3)
    w = np.array(io.writes)
    assert len(w) == 5 + 3 and w[4] == pytest.approx(DEFAULT) and w[0] == pytest.approx(np.array(DEFAULT) / 5)
