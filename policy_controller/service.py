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

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        pkg = Path(attrs["package_dir"])
        self.cfg = json.loads((pkg / "config.json").read_text())
        self.arm = dependencies[Arm.get_resource_name(attrs["arm"])]
        self.gripper = dependencies[Gripper.get_resource_name(attrs["gripper"])]  # held by the caller, unused here
        self.mlmodel = dependencies[MLModel.get_resource_name(attrs["mlmodel"])]
        self.hole = np.array(attrs["hole_mouth_mm"], float) / 1000
        self.timeout_s, self.max_overruns = float(attrs.get("timeout_s", 10)), int(attrs.get("max_overruns", 3))
        self.policy = Policy(
            self.cfg,
            Kinematics(pkg / "kinematics.xml", self.cfg["control"]),
            {k: attrs[k] for k in LIMIT_KEYS if k in attrs},
        )
        self.task = None
        self.state, self.reason, self.steps, self.overruns, self.last_obs = "idle", None, 0, 0, None

    async def do_command(self, command, *, timeout=None, **kwargs):
        match command.get("command"):
            case "start":
                if self.state == "running":
                    raise RuntimeError("already running")
                self.policy.reset()
                self.state, self.reason, self.steps, self.overruns, self.last_obs = "running", None, 0, 0, None
                self.task = asyncio.create_task(self.run())
                return {"status": "running"}
            case "stop":
                if self.task:
                    self.task.cancel()
                    await asyncio.gather(self.task, return_exceptions=True)
                await self.arm.stop()
                self.state, self.reason = "idle", "stopped"
                return {"status": "stopped"}
            case "status":
                return {
                    "state": self.state,
                    "reason": self.reason,
                    "steps": self.steps,
                    "overruns": self.overruns,
                    "last_obs": self.last_obs,
                }
            case "pre_insert_pose":
                q, pose = pre_insert(self.policy.kin, self.cfg, self.hole)
                return pose | {"joints_deg": np.degrees(q).tolist()}
        raise ValueError(f"unknown command {command.get('command')}")

    async def run(self):
        """Fails the episode with a reason (stream cancelled, arm stopped) or succeeds (stream half-closed)."""
        dt, io = self.cfg["control"]["policy_dt"], self.cfg["model"]
        queue: asyncio.Queue = asyncio.Queue()

        async def batches():
            while (batch := await queue.get()) is not None:
                yield batch

        async def consume():
            async for _ in self.arm.move_through_joint_positions_streamed(batches()):
                pass

        stream = None
        try:
            q = np.radians(await self.arm.get_joint_positions())
            prev_goal = q
            queue.put_nowait([Arm.TrajectoryPoint(timedelta(0), np.degrees(q).tolist())])
            stream = asyncio.create_task(consume())
            t0 = time.monotonic()
            tick = 0
            while True:
                await asyncio.sleep(max(0.0, t0 + tick * dt - time.monotonic()))
                if stream.done():
                    stream.result()  # raises the stream error
                    raise RuntimeError("stream ended")
                if time.monotonic() - t0 > self.timeout_s:
                    raise SafetyStop("timeout")
                q = np.radians(await self.arm.get_joint_positions())
                if succeeded(self.policy.kin, self.cfg, q, self.hole):
                    queue.put_nowait(None)  # half-close; the arm settles on the last goal
                    await stream
                    self.state = "succeeded"
                    return
                obs = self.policy.observe(q, self.hole)
                self.last_obs = obs.tolist()
                out = await self.mlmodel.infer({io["inputs"][0]["name"]: obs[None].astype(np.float32)})
                if time.monotonic() - t0 >= (tick + 1) * dt:  # this tick's window has started: drop it, skip ahead
                    self.overruns += 1
                    if self.overruns > self.max_overruns:
                        raise SafetyStop("overruns")
                    tick = int((time.monotonic() - t0) / dt) + 1
                    continue
                goal = self.policy.act(q, self.hole, np.asarray(out[io["outputs"][0]["name"]])[0])
                times = point_times(tick, dt)
                queue.put_nowait(
                    [Arm.TrajectoryPoint(timedelta(seconds=t), np.degrees(p).tolist()) for t, p in zip(times, ramp(prev_goal, goal))]
                )
                prev_goal, self.steps, tick = goal, self.steps + 1, tick + 1
        except asyncio.CancelledError:
            if stream:
                stream.cancel()
            raise
        except Exception as e:
            self.state, self.reason = "failed", e.reason if isinstance(e, SafetyStop) else repr(e)
            if stream:
                stream.cancel()
            await self.arm.stop()
