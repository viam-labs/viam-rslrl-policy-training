"""Viam generic service devrel:duck-policy:teleop: drives a policy service (the walker's DoCommands
start, stop, set_command {vx, vy, wz}) from any input controller. Hold LB to drive: left stick
vx/vy, right stick wz. Start starts the policy, Select stops it. A controller error or Disconnect
sends one zero twist. Spec: docs/superpowers/specs/2026-11-22-microduck-teleop-design.md."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from viam.components.input import Control, Controller, EventType
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

DEADZONE = 0.1
# (axis, scale): gamepads read stick-up and stick-left as negative, so every axis is negated.
AXES = ((Control.ABSOLUTE_Y, 0.4), (Control.ABSOLUTE_X, 0.3), (Control.ABSOLUTE_RX, 1.0))
HOLD = Control.BUTTON_LT
HELD = {EventType.BUTTON_PRESS, EventType.BUTTON_HOLD}  # the Linux gamepad repeats a held key as ButtonHold
BUTTONS = ((Control.BUTTON_START, "start"), (Control.BUTTON_SELECT, "stop"))
ZERO = (0.0, 0.0, 0.0)


@dataclass
class TeleopState:
    handled: dict = field(default_factory=dict)  # Control -> time of the last press acted on
    primed: bool = False


def map_events(events: dict, state: TeleopState) -> tuple[tuple[float, float, float] | None, list[str]]:
    """The controller's latest event per control -> (twist, actions). twist is None while any control's
    latest event is Disconnect. The first call only records the button times: webgamepad keeps its last
    events across teleop restarts, so an old press must not fire."""
    actions = []
    for control, action in BUTTONS:
        e = events.get(control)
        if not state.primed:
            state.handled[control] = e.time if e else 0.0
        elif e and e.event == EventType.BUTTON_PRESS and e.time > state.handled.get(control, 0.0):
            state.handled[control] = e.time
            actions.append(action)
    first = not state.primed
    state.primed = True
    if any(e.event == EventType.DISCONNECT for e in events.values()):
        return None, []
    if first:
        actions = []
    hold = events.get(HOLD)
    if hold is None or hold.event not in HELD:
        return ZERO, actions

    def axis(control: Control, scale: float) -> float:
        e = events.get(control)
        v = e.value if e else 0.0
        return 0.0 if abs(v) < DEADZONE else -v * scale

    return tuple(axis(c, s) for c, s in AXES), actions


HZ = 20.0
REFRESH_S = 0.5  # resend an unchanged twist this often while the controller is healthy
CALL_TIMEOUT_S = 0.5  # a hung get_events takes the error path (zero twist)
WARN_EVERY_S = 1.0


class Teleop(Generic, EasyResource):
    MODEL = Model(ModelFamily("devrel", "duck-policy"), "teleop")

    @classmethod
    def validate_config(cls, config: ComponentConfig):
        attrs = struct_to_dict(config.attributes)
        for key in ("input_controller", "policy"):
            if not attrs.get(key):
                raise ValueError(f"attribute {key} is required")
        return [attrs["input_controller"], attrs["policy"]], []

    @classmethod
    def new(cls, config, dependencies):
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        self._configure(dependencies[Controller.get_resource_name(attrs["input_controller"])],
                        dependencies[Generic.get_resource_name(attrs["policy"])])
        self.task = asyncio.get_running_loop().create_task(self._run())

    def _configure(self, controller: Any, policy: Any) -> None:
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
        self.task = None
        self.controller, self.policy = controller, policy
        self.state = TeleopState()  # re-primed: presses from before a reconfigure are stale
        self.sent, self.sent_at, self.warned_at = None, -math.inf, -math.inf

    def _warn(self, now: float, msg: str) -> None:
        if now - self.warned_at >= WARN_EVERY_S:
            self.logger.warning(msg)
            self.warned_at = now

    async def _call(self, command: dict, now: float) -> None:
        try:
            await self.policy.do_command(command)
        except Exception as e:  # keep polling: the policy may come back
            self._warn(now, f"policy {command.get('command')}: {e}")

    async def tick(self, now: float) -> None:
        try:
            events = await asyncio.wait_for(self.controller.get_events(), CALL_TIMEOUT_S)
            twist, actions = map_events(events, self.state)
        except Exception as e:
            self._warn(now, f"input controller: {e}")
            twist, actions = None, []
        for action in actions:
            await self._call({"command": action}, now)
        healthy = twist is not None
        twist = twist if healthy else ZERO
        if twist != self.sent or (healthy and now - self.sent_at >= REFRESH_S):
            vx, vy, wz = twist
            await self._call({"command": "set_command", "vx": vx, "vy": vy, "wz": wz}, now)
            self.sent, self.sent_at = twist, now

    async def _run(self) -> None:
        while True:
            await self.tick(time.monotonic())
            await asyncio.sleep(1 / HZ)

    async def do_command(self, command, *, timeout=None, **kwargs):
        raise ValueError("teleop takes no commands")

    async def close(self):
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        if getattr(self, "policy", None) is not None:  # removed while LB held: don't leave the walker walking
            await self._call({"command": "set_command", "vx": 0.0, "vy": 0.0, "wz": 0.0}, time.monotonic())
