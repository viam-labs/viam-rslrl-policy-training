"""The Viam generic service: 20 Hz policy loop streaming joint goals to an arm. See the design spec, Deploy controller."""

import asyncio
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.mlmodel import MLModel
from viam.utils import struct_to_dict

from policy_controller.core import Kinematics, Policy, SafetyStop
from policy_controller.stream import point_times, pre_insert, ramp, succeeded

LIMIT_KEYS = ("action_clip", "max_step_mm", "max_joint_step_deg", "workspace_mm")


class PolicyController(Generic, EasyResource):
    MODEL = Model(ModelFamily("devrel", "rl-policy"), "controller")

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        for key in ("arm", "gripper", "mlmodel", "package_dir", "hole_mouth_mm"):
            if not attrs.get(key):
                raise ValueError(f"attribute {key} is required")
        if len(attrs["hole_mouth_mm"]) != 3:
            raise ValueError("hole_mouth_mm must be [x, y, z]")
        return [attrs["arm"], attrs["gripper"], attrs["mlmodel"]], []

    @classmethod
    def new(cls, config, dependencies):
        # viam-sdk 0.85's EasyResource.new only sets the name; the module's reconfigure is remove + new.
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    async def _halt(self):
        """Cancel a running loop (run() cancels the stream and stops the arm) and wait for it."""
        if self.task and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def close(self):
        await self._halt()

    def reconfigure(self, config, dependencies):
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()  # run()'s cancel path stops the stream and the arm
        attrs = struct_to_dict(config.attributes)
        pkg = Path(attrs["package_dir"])
        self.cfg = json.loads((pkg / "config.json").read_text())
        self.arm = dependencies[Arm.get_resource_name(attrs["arm"])]
        self.gripper = dependencies[Gripper.get_resource_name(attrs["gripper"])]  # held by the caller, unused here
        self.mlmodel = dependencies[MLModel.get_resource_name(attrs["mlmodel"])]
        self.hole_cfg = self.hole = np.array(attrs["hole_mouth_mm"], float) / 1000
        self.timeout_s, self.max_overruns = float(attrs.get("timeout_s", 10)), int(attrs.get("max_overruns", 3))
        self.policy = Policy(
            self.cfg,
            Kinematics(pkg / "kinematics.xml", self.cfg["control"]),
            {k: attrs[k] for k in LIMIT_KEYS if k in attrs},
        )
        self.task = None
        self.state, self.reason, self.steps, self.overruns, self.last_obs = "idle", None, 0, 0, None
        self.ticks = []  # per tick [start late, read, infer, act] ms, for status

    async def do_command(self, command, *, timeout=None, **kwargs):
        match command.get("command"):
            case "start":
                # TODO (plan decision: not now): require the gripper closed on the peg, and reject start states
                # outside the trained start distribution (offset and tilt from the pre-insert pose).
                if self.state == "running":
                    raise RuntimeError("already running")
                # Optional per-episode hole estimate (mm, arm base frame), e.g. from vision; else the configured one.
                hole = command.get("hole_mouth_mm")
                self.hole = self.hole_cfg if hole is None else np.array(hole, float) / 1000
                self.policy.reset()
                self.state, self.reason, self.steps, self.overruns, self.last_obs = "running", None, 0, 0, None
                self.ticks = []
                self.task = asyncio.create_task(self.run())
                return {"status": "running"}
            case "stop":
                was_running = self.state == "running"
                await self._halt()
                await self.arm.stop()
                if was_running:
                    self.state, self.reason = "idle", "stopped"
                return {"status": "stopped"}
            case "status":
                return {
                    "state": self.state,
                    "reason": self.reason,
                    "steps": self.steps,
                    "overruns": self.overruns,
                    "last_obs": self.last_obs,
                    "ticks_ms": self.ticks[-40:],
                }
            case "pre_insert_pose":
                # Flange pose in the ARM BASE frame: the caller builds a PoseInFrame with the arm's frame name.
                lo, hi = self.cfg["task"]["pre_insert_tip_height_m"]
                q, pose = pre_insert(self.policy.kin, self.cfg, self.hole_cfg, (lo + hi) / 2)
                return pose | {"frame": "arm base", "joints_deg": np.degrees(q).tolist()}
        raise ValueError(f"unknown command {command.get('command')}")

    async def run(self):
        """Fails the episode with a reason (stream cancelled, arm stopped) or succeeds (stream half-closed)."""
        dt, io = self.cfg["control"]["policy_dt"], self.cfg["model"]
        self.ticks = []
        queue: asyncio.Queue = asyncio.Queue()

        async def batches():
            while (batch := await queue.get()) is not None:
                yield batch

        async def consume():
            async for _ in self.arm.move_through_joint_positions_streamed(batches()):
                pass

        stream = None
        try:
            q = np.radians((await self.arm.get_joint_positions()).values)
            prev_goal = q
            # Warm-up: the first infer after a (re)start is slow enough to overrun ticks (sim-to-sim, 2026-11-18).
            await self.mlmodel.infer({io["inputs"][0]["name"]: self.policy.observe(q, self.hole)[None].astype(np.float32)})
            self.policy.reset()  # observe() above recorded nothing, but start the episode clean
            queue.put_nowait([Arm.TrajectoryPoint(timedelta(0), np.degrees(q).tolist())])
            stream = asyncio.create_task(consume())
            t0 = time.monotonic()  # the stream anchor is set server-side a little later: our times run early, the safe side
            tick = 0
            while True:
                await asyncio.sleep(max(0.0, t0 + tick * dt - time.monotonic()))
                if stream.done():
                    stream.result()  # raises the stream error
                    raise RuntimeError("stream ended")
                if time.monotonic() - t0 > self.timeout_s:
                    raise SafetyStop("timeout")
                ts = time.monotonic()
                q = np.radians((await self.arm.get_joint_positions()).values)
                tr = time.monotonic()
                if succeeded(self.policy.kin, self.cfg, q, self.hole):
                    queue.put_nowait(None)  # half-close; the arm settles on the last goal
                    try:
                        await asyncio.wait_for(stream, 5)
                    except asyncio.TimeoutError:
                        raise SafetyStop("settle_timeout")
                    self.state = "succeeded"
                    return
                obs = self.policy.observe(q, self.hole)
                self.last_obs = obs.tolist()
                out = await self.mlmodel.infer({io["inputs"][0]["name"]: obs[None].astype(np.float32)})
                ti = time.monotonic()
                self.ticks.append([round(1000 * x, 1) for x in (ts - t0 - tick * dt, tr - ts, ti - tr)])
                if time.monotonic() - t0 >= (tick + 1) * dt:  # this tick's window has started: drop it, skip ahead
                    self.overruns += 1
                    self.policy.q_prev = q  # joint_vel stays a one-tick difference
                    if self.overruns > self.max_overruns:
                        raise SafetyStop("overruns")
                    tick = int((time.monotonic() - t0) / dt) + 1
                    continue
                goal = self.policy.act(q, self.hole, np.asarray(out[io["outputs"][0]["name"]])[0])
                self.ticks[-1].append(round(1000 * (time.monotonic() - ti), 1))
                times = point_times(tick, dt)
                queue.put_nowait(
                    [Arm.TrajectoryPoint(timedelta(seconds=t), np.degrees(p).tolist()) for t, p in zip(times, ramp(prev_goal, goal))]
                )
                prev_goal, self.steps, tick = goal, self.steps + 1, tick + 1
        except BaseException as e:  # failure or cancellation: kill the stream first so no queued points play out
            if stream:
                stream.cancel()
                await asyncio.gather(stream, return_exceptions=True)
            await self.arm.stop()
            if isinstance(e, asyncio.CancelledError):
                raise
            self.state, self.reason = "failed", e.reason if isinstance(e, SafetyStop) else repr(e)
