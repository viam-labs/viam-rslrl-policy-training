"""Viam generic service devrel:duck-policy:controller: runs the MicroDuck walk policy at 50 Hz against a
duck that serves the io DoCommand (viam-labs:mujoco-sim's robot now; a hardware module later), with
inference through an mlmodel service (viam:mlmodel:onnx-cpu). DoCommand: start, stop, status,
set_command {vx, vy, wz}. See spec unit 4."""

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
from .loop import Limits, PolicyLoop
from .obs import Command

LIMIT_KEYS = ("fall_gravity_z", "call_timeout_s", "ramp_s")


class DuckPolicy(Generic, EasyResource):
    MODEL = Model(ModelFamily("devrel", "duck-policy"), "controller")

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        for key in ("duck", "mlmodel", "package_dir"):
            if not attrs.get(key):
                raise ValueError(f"attribute {key} is required")
        return [attrs["duck"], attrs["mlmodel"]], []

    @classmethod
    def new(cls, config, dependencies):
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        duck = dependencies[GenericComponent.get_resource_name(attrs["duck"])]
        mlmodel = dependencies[MLModel.get_resource_name(attrs["mlmodel"])]
        self._configure(attrs, duck=duck, mlmodel=mlmodel)

    def _configure(self, attrs: dict, *, duck: Any, mlmodel: Any) -> None:
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
        self.cfg = json.loads((Path(attrs["package_dir"]) / "config.json").read_text())
        self.duck, self.mlmodel = duck, mlmodel
        self.limits = Limits(**{k: float(attrs[k]) for k in LIMIT_KEYS if k in attrs})
        self.input_name = self.cfg["model"]["input"]
        self.output_name = self.cfg["model"]["output"]
        self.task, self.loop, self.command = None, None, Command()

    async def _infer(self, obs: np.ndarray) -> np.ndarray:
        out = await self.mlmodel.infer({self.input_name: obs[None].astype(np.float32)})
        return np.asarray(out[self.output_name])[0]

    def _state(self) -> str:
        if self.task is None:
            return "idle"
        return "running" if not self.task.done() else "stopped"

    async def do_command(self, command, *, timeout=None, **kwargs):
        match command.get("command"):
            case "start":
                if self._state() == "running":
                    raise ValueError("already running")
                self.loop = PolicyLoop(SimIO(self.duck, self.cfg["joints"]), self._infer, self.cfg, self.limits)
                self.loop.set_command(self.command)
                self.task = asyncio.create_task(self.loop.run())
                return {"state": "running"}
            case "stop":
                if self.task and not self.task.done():
                    self.task.cancel()
                    await asyncio.gather(self.task, return_exceptions=True)
                return {"state": self._state(), "reason": self.loop.reason if self.loop else None}
            case "set_command":
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
