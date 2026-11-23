"""duck-policy's 50 Hz loop: ramp to the default pose, then each tick read the duck, build the
observation, infer, and write default_pose + action * scale (clipped to joint limits, rate-capped).
Read-infer-write in that order matches infer_policy.py (observe, infer, apply, then step). Every stop
condition ends the loop with a reason; a fall or a bad action also turns torque off (limp), the rest hold
the last targets. See spec unit 4."""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from .io import DuckIO, Snapshot
from .obs import Command, build_obs, obs_layout, phase_command


@dataclass(frozen=True)
class Limits:
    fall_gravity_z: float = -0.5  # trunk tilted ~60 degrees
    fall_ticks: int = 3
    late_window: int = 50
    late_fraction: float = 0.05
    late_streak: int = 3
    call_timeout_s: float = 0.05  # one call; 15 ms killed a live episode on a rare stall (p99 ~6 ms). The duck tips in ~1.5 s.
    ramp_s: float = 1.0
    warmup_timeout_s: float = 1.0
    stale_ticks: int = 3


RAMP_THRESHOLD = 0.15  # rad: a start pose farther than this is ramped to default; nearer, the policy starts from it (the walker stands 0.063 off)


class Stop(Exception):
    def __init__(self, reason: str, limp: bool = False) -> None:
        super().__init__(reason)
        self.reason, self.limp = reason, limp


class PolicyLoop:
    def __init__(self, io: DuckIO, infer: Callable[[np.ndarray], np.ndarray | Awaitable[np.ndarray]],
                 cfg: dict, limits: Limits, cycles: int = 0) -> None:
        self.io, self.infer, self.limits = io, infer, limits
        self.default = np.asarray(cfg["default_pose"], float)
        self.scale = float(cfg["action_scale"])
        lim = np.asarray(cfg["joint_limits"], float)
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.dt = 1.0 / float(cfg.get("control_hz", 50))
        self.layout = obs_layout(cfg)
        self.obs_len, self.cmd_size = sum(n for _, n in self.layout), dict(self.layout).get("command", 3)
        cmd = cfg["command"]
        self.period = float(cmd["period_s"]) if cmd.get("mode") == "phase" else None
        if self.period is None:
            self.twist_lo, self.twist_hi = cmd["twist_min"], cmd["twist_max"]
        self.cycles, self.cycles_done, self.phase, self.holding = int(cycles), 0, 0.0, False
        safety = cfg.get("safety") or {}
        self.action_bound = float(safety.get("action_bound", 5.0))
        self.max_step = float(safety.get("max_step_rad", 0.5))
        self.command, self.command_clipped = Command(), False
        self.ticks, self.late, self.reason, self.last_obs, self.t0, self.t_end = 0, 0, None, None, None, None
        self.warmup_ms, self.resets = None, 0
        self.tick_ms: deque[float] = deque(maxlen=3000)

    def set_command(self, command: Command) -> None:
        if self.period is not None:
            raise ValueError("set_command isn't available in phase mode")
        clipped = command.clipped(self.twist_lo, self.twist_hi)
        self.command_clipped, self.command = clipped != command, clipped

    async def _call(self, coro):
        try:
            return await asyncio.wait_for(coro, self.limits.call_timeout_s)
        except asyncio.TimeoutError:
            raise Stop("hung_call") from None

    async def _infer(self, obs: np.ndarray) -> np.ndarray:
        out = self.infer(obs)
        if asyncio.iscoroutine(out) or isinstance(out, Awaitable):
            out = await self._call(out)
        return np.asarray(out, float).reshape(14)

    def _target(self, action: np.ndarray, prev: np.ndarray) -> np.ndarray:
        target = np.clip(self.default + action * self.scale, self.lo, self.hi)
        return prev + np.clip(target - prev, -self.max_step, self.max_step)

    async def run(self, max_ticks: int | None = None) -> str:
        try:
            await self._run_timed(max_ticks)
            self.reason = "max_ticks"
        except Stop as stop:
            self.reason = stop.reason
            if stop.limp:
                try:  # shielded: a concurrent stop can't cancel it; a gRPC error can't escape
                    await asyncio.shield(self.io.torque(False))
                except Exception as error:
                    self.reason += f" (torque off failed: {error})"
        except asyncio.CancelledError:
            self.reason = "stopped"
            raise
        except Exception as error:  # e.g. a gRPC error from the duck or the mlmodel service
            self.reason = f"error: {error}"
        return self.reason

    async def _run_timed(self, max_ticks: int | None) -> None:
        try:
            await self._run(max_ticks)
        finally:
            self.t_end = time.monotonic()

    async def _run(self, max_ticks: int | None) -> None:
        lim = self.limits
        # Warm-up: the first onnx-cpu infer over gRPC is slow. Not counted, not under call_timeout_s.
        warm = time.monotonic()
        out = self.infer(np.zeros(self.obs_len, np.float32))
        if asyncio.iscoroutine(out) or isinstance(out, Awaitable):
            try:
                await asyncio.wait_for(out, lim.warmup_timeout_s)
            except asyncio.TimeoutError:
                raise Stop("hung_call") from None
        self.warmup_ms = round(1000 * (time.monotonic() - warm), 1)
        snap: Snapshot = await self._call(self.io.torque(True))
        generation, prev_t = snap.generation, snap.t
        # Ramp from where the duck is to the default pose. After a reset at STAND it's already there, and
        # the ramp is skipped: holding the pose with no policy, the duck tips over within ~1.5 s (plan review).
        start = np.asarray(snap.q, float)
        n = 0 if np.abs(start - self.default).max() < RAMP_THRESHOLD else max(1, round(lim.ramp_s / self.dt))
        for i in range(1, n + 1):
            await self._call(self.io.write((start + (self.default - start) * i / n).tolist()))
            await asyncio.sleep(self.dt)
        if n == 0:
            # Let sim time move past the torque snapshot's t. mujoco-sim steps in 5 ms, so a read 2 ms
            # later often shows the same t, which would stop with "stale_state" on tick 1.
            await asyncio.sleep(self.dt)
        prev_target, last_action = self.default.copy(), np.zeros(14)
        fall, late_hist, late_streak, stale, slot = 0, deque(maxlen=lim.late_window), 0, 0, 1
        t0 = self.t0 = time.monotonic()
        while max_ticks is None or self.ticks < max_ticks:
            deadline = t0 + slot * self.dt
            tick_start = time.monotonic()
            snap = await self._call(self.io.read())
            if snap.generation != generation:
                # World reset: duck back at STAND, sim time restarts, torque stays on. Re-sync and carry on.
                generation, prev_t, stale, fall = snap.generation, snap.t, 0, 0
                last_action, prev_target = np.zeros(14), np.asarray(snap.q, float)
                self.resets += 1
            else:
                stale = stale + 1 if snap.t <= prev_t else 0  # a repeated observation is harmless; N in a row isn't
            if stale >= lim.stale_ticks:
                raise Stop("stale_state")
            prev_t = max(prev_t, snap.t)
            fall = fall + 1 if snap.gravity[2] > lim.fall_gravity_z else 0
            if fall >= lim.fall_ticks:
                raise Stop("fall", limp=True)
            if self.period is None:
                command = self.command.vector(self.cmd_size)
            else:
                s = (slot - 1) * self.dt  # this tick's scheduled time since t0: skipped slots advance it like real time
                done = int(s // self.period)
                self.holding = bool(self.cycles) and done >= self.cycles
                self.cycles_done = min(done, self.cycles) if self.cycles else done
                self.phase = 0.0 if self.holding else (s / self.period) % 1.0
                command = phase_command(self.phase, self.cmd_size)
            obs = build_obs(gyro=snap.gyro, gravity=snap.gravity, q=snap.q, dq=snap.dq,
                            default_pose=self.default, last_action=last_action, command=command,
                            layout=self.layout)
            self.last_obs = obs
            action = await self._infer(obs)
            if not np.all(np.isfinite(action)) or np.abs(action).max() > self.action_bound:
                raise Stop("bad_action", limp=True)
            target = self._target(action, prev_target)
            await self._call(self.io.write(target.tolist()))
            prev_target, last_action = target, action
            self.ticks += 1
            elapsed = time.monotonic() - tick_start
            self.tick_ms.append(1000 * elapsed)
            now = time.monotonic()
            is_late = now > deadline
            # A missed deadline is counted, not caught up: skip to the next grid slot.
            slot = math.floor((now - t0) / self.dt) + 1 if is_late else slot + 1
            self.late += is_late
            late_hist.append(is_late)
            late_streak = late_streak + 1 if is_late else 0
            if late_streak >= lim.late_streak or (len(late_hist) == lim.late_window
                                                  and sum(late_hist) > lim.late_fraction * lim.late_window):
                raise Stop("overruns")
            await asyncio.sleep(max(0.0, t0 + (slot - 1) * self.dt - time.monotonic()))

    def status(self) -> dict:
        ms = sorted(self.tick_ms)
        pct = (lambda p: round(ms[min(len(ms) - 1, int(p * len(ms)))], 2)) if ms else (lambda p: None)
        run_s = None if self.t0 is None else round((self.t_end or time.monotonic()) - self.t0, 3)
        phase = {"phase": self.phase, "period_s": self.period, "cycles_done": self.cycles_done, "holding": self.holding} if self.period else {}
        return {**phase, "ticks": self.ticks, "late": self.late, "resets": self.resets, "run_s": run_s, "warmup_ms": self.warmup_ms,  # run_s: since the loop's t0 (after any ramp)
                "tick_ms_p50": pct(0.5), "tick_ms_p99": pct(0.99),
                "reason": self.reason, "command": [self.command.vx, self.command.vy, self.command.wz],
                "command_clipped": self.command_clipped}
