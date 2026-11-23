"""Viam generic service devrel:duck-policy:controller: runs the MicroDuck walk policy at 50 Hz against a
duck that serves the io DoCommand (viam-labs:mujoco-sim's robot now; a hardware module later), with
inference through an mlmodel service (viam:mlmodel:onnx-cpu). DoCommand: start, stop, status,
set_command {vx, vy, wz} (velocity packages only). A package whose command mode is "phase" is a dance:
start {cycles: N} runs N phase cycles, then holds STAND. Optional attribute world (the mujoco-sim world): start then recovers a fallen duck
by pause/reset/torque-on/start/play (this unpauses a paused world). In sim a start whose pose is more than
RAMP_THRESHOLD (0.15 rad) off STAND also resets the world (the duck returns to its spawn point); nearer, the
new policy takes over from where the duck stands, so a walker/dancer handover needs no reset. See spec unit 4."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
from viam.components.generic import Generic as GenericComponent
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.services.mlmodel import MLModel
from viam.utils import struct_to_dict

from .io import SimIO
from .loop import RAMP_THRESHOLD, Limits, PolicyLoop
from .obs import Command

UPRIGHT_GRAVITY_Z = -0.9  # ~25 degrees of tilt
LIMIT_KEYS = ("fall_gravity_z", "call_timeout_s", "ramp_s")


class DuckPolicy(Generic, EasyResource):
    MODEL = Model(ModelFamily("devrel", "duck-policy"), "controller")

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        for key in ("duck", "mlmodel", "package_dir"):
            if not attrs.get(key):
                raise ValueError(f"attribute {key} is required")
        return [attrs["duck"], attrs["mlmodel"], *([attrs["world"]] if attrs.get("world") else [])], []

    @classmethod
    def new(cls, config, dependencies):
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        duck = dependencies[GenericComponent.get_resource_name(attrs["duck"])]
        mlmodel = dependencies[MLModel.get_resource_name(attrs["mlmodel"])]
        world = dependencies[GenericComponent.get_resource_name(attrs["world"])] if attrs.get("world") else None
        self._configure(attrs, duck=duck, mlmodel=mlmodel, world=world)

    def _configure(self, attrs: dict, *, duck: Any, mlmodel: Any, world: Any = None) -> None:
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
        self.cfg = json.loads((Path(attrs["package_dir"]) / "config.json").read_text())
        self.duck, self.mlmodel, self.world = duck, mlmodel, world
        self.limits = Limits(**{k: float(attrs[k]) for k in LIMIT_KEYS if k in attrs})
        self.input_name = self.cfg["model"]["input"]
        self.output_name = self.cfg["model"]["output"]
        self.task, self.loop, self.command = None, None, Command()
        self._start_lock = getattr(self, "_start_lock", None) or asyncio.Lock()  # kept across reconfigures

    async def _infer(self, obs: np.ndarray) -> np.ndarray:
        out = await self.mlmodel.infer({self.input_name: obs[None].astype(np.float32)})
        return np.asarray(out[self.output_name])[0]

    def _state(self) -> str:
        if self.task is None:
            return "idle"
        return "running" if not self.task.done() else "stopped"

    async def _start(self, cycles: int = 0) -> dict:
        if self._state() == "running":
            raise ValueError("already running")
        io = SimIO(self.duck, self.cfg["joints"])
        recover: str | bool = False
        if self.world is not None:
            # a hung duck must not hold the start lock forever
            snap = await asyncio.wait_for(io.read(), self.limits.call_timeout_s * 10)
            far = np.abs(np.asarray(snap.q) - np.asarray(self.cfg["default_pose"])).max() >= RAMP_THRESHOLD
            # off_pose: the loop would ramp to default with no policy; in sim the reset replaces the ramp
            recover = "fallen" if snap.gravity[2] > UPRIGHT_GRAVITY_Z else "off_pose" if far else False
        try:
            if recover:
                await self.world.do_command({"command": "pause"})
                await self.world.do_command({"command": "reset"})
                await asyncio.wait_for(io.torque(True), self.limits.call_timeout_s * 10)  # stiff before physics resumes
            self.loop = PolicyLoop(io, self._infer, self.cfg, self.limits, cycles=cycles)
            if self.cfg["command"].get("mode") != "phase":
                self.loop.set_command(self.command)
            self.task = asyncio.create_task(self.loop.run())
        finally:
            if recover:  # never leave the world paused
                await self.world.do_command({"command": "play"})
        return {"state": "running", "recovered": recover}

    async def do_command(self, command, *, timeout=None, **kwargs):
        match command.get("command"):
            case "start":
                cycles = int(command.get("cycles", 0))
                if cycles < 0:
                    raise ValueError("cycles must be >= 0 (0 = dance until stopped)")
                async with self._start_lock:
                    return await self._start(cycles)
            case "stop":
                if self.task and not self.task.done():
                    self.task.cancel()
                    await asyncio.gather(self.task, return_exceptions=True)
                return {"state": self._state(), "reason": self.loop.reason if self.loop else None}
            case "set_command":
                if self.cfg["command"].get("mode") == "phase":
                    raise ValueError("set_command isn't available in phase mode")
                c = Command(float(command.get("vx", 0)), float(command.get("vy", 0)), float(command.get("wz", 0)))
                clipped = c.clipped(self.cfg["command"]["twist_min"], self.cfg["command"]["twist_max"])
                self.command = clipped
                if self.loop:
                    self.loop.set_command(c)
                return {"command": [clipped.vx, clipped.vy, clipped.wz], "command_clipped": clipped != c}
            case "status":
                status = self.loop.status() if self.loop else {}
                return {"state": self._state(), **status}
            case other:
                raise ValueError(f"unknown command {other!r}; known: start, stop, status, set_command")

    async def close(self):
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
