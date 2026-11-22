"""Viam generic service devrel:duck-policy:teleop: drives a policy service (the walker's DoCommands
start, stop, set_command {vx, vy, wz}) from any input controller. Hold LB to drive: left stick
vx/vy, right stick wz (optional `mapping` {hold, vx, vy, wz, start, stop} -> Control names). Start starts the policy, Select stops it. A controller error or Disconnect
sends one zero twist. Spec: docs/superpowers/specs/2026-11-22-microduck-teleop-design.md."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field, fields
from typing import Any

from viam.components.input import Control, Controller, EventType
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

DEADZONE = 0.1
SCALES = (0.4, 0.3, 1.0)  # vx, vy, wz; gamepads read stick-up and stick-left as negative, so every axis is negated
HELD = {EventType.BUTTON_PRESS, EventType.BUTTON_HOLD}  # the Linux gamepad repeats a held key as ButtonHold
HOLD_AXIS_MIN = 0.5  # an axis used as the hold control counts as held above this
ZERO = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Mapping:
    hold: Control = Control.BUTTON_LT
    vx: Control = Control.ABSOLUTE_Y
    vy: Control = Control.ABSOLUTE_X
    wz: Control = Control.ABSOLUTE_RX
    start: Control = Control.BUTTON_START
    stop: Control = Control.BUTTON_SELECT

    def __post_init__(self):  # accept Control names too
        for f in fields(self):
            name = getattr(self, f.name)
            try:
                object.__setattr__(self, f.name, Control(name))
            except ValueError:
                raise ValueError(f"mapping.{f.name}: {name!r} is not a Control name") from None

    @classmethod
    def from_attrs(cls, attrs) -> "Mapping":
        if attrs is None:
            return cls()
        if not isinstance(attrs, dict):
            raise ValueError("attribute mapping must be an object")
        known = [f.name for f in fields(cls)]
        unknown = set(attrs) - set(known)
        if unknown:
            raise ValueError(f"mapping: unknown key {sorted(unknown)}; known keys: {known}")
        return cls(**attrs)


@dataclass
class TeleopState:
    handled: dict = field(default_factory=dict)  # Control -> time of the last press acted on
    primed: bool = False


def map_events(events: dict, state: TeleopState, m: Mapping = Mapping()) -> tuple[tuple[float, float, float] | None, list[str]]:
    """The controller's latest event per control -> (twist, actions). twist is None while any control's
    latest event is Disconnect. The first call only records the button times: webgamepad keeps its last
    events across teleop restarts, so an old press must not fire."""
    actions = []
    for control, action in ((m.start, "start"), (m.stop, "stop")):
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
    hold = events.get(m.hold)
    if hold is None or not (hold.event in HELD or (hold.event == EventType.POSITION_CHANGE_ABSOLUTE
                                                   and hold.value > HOLD_AXIS_MIN)):
        return ZERO, actions

    def axis(control: Control, scale: float) -> float:
        e = events.get(control)
        v = e.value if e else 0.0
        return 0.0 if abs(v) < DEADZONE else -v * scale

    return tuple(axis(c, s) for c, s in zip((m.vx, m.vy, m.wz), SCALES)), actions


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
        Mapping.from_attrs(attrs.get("mapping"))
        return [attrs["input_controller"], attrs["policy"]], []

    @classmethod
    def new(cls, config, dependencies):
        self = super().new(config, dependencies)
        self.reconfigure(config, dependencies)
        return self

    def reconfigure(self, config, dependencies):
        attrs = struct_to_dict(config.attributes)
        self._configure(dependencies[Controller.get_resource_name(attrs["input_controller"])],
                        dependencies[Generic.get_resource_name(attrs["policy"])],
                        Mapping.from_attrs(attrs.get("mapping")))
        self.task = asyncio.get_running_loop().create_task(self._run())

    def _configure(self, controller: Any, policy: Any, mapping: Mapping) -> None:
        if getattr(self, "task", None) and not self.task.done():
            self.task.cancel()
        self.task = None
        self.controller, self.policy, self.mapping = controller, policy, mapping
        self.state = TeleopState()  # re-primed: presses from before a reconfigure are stale
        self.sent, self.sent_at, self.warned_at = None, -math.inf, -math.inf

    def _warn(self, now: float, msg: str) -> None:
        if now - self.warned_at >= WARN_EVERY_S:
            self.logger.warning(msg)
            self.warned_at = now

    async def _call(self, command: dict, now: float) -> bool:
        try:
            await asyncio.wait_for(self.policy.do_command(command), CALL_TIMEOUT_S)
            return True
        except Exception as e:  # keep polling: the policy may come back (TimeoutError included)
            self._warn(now, f"policy {command.get('command')}: {e!r}")
            return False

    async def tick(self, now: float) -> None:
        try:
            events = await asyncio.wait_for(self.controller.get_events(), CALL_TIMEOUT_S)
            twist, actions = map_events(events, self.state, self.mapping)
        except Exception as e:
            self._warn(now, f"input controller: {e}")
            twist, actions = None, []
        for action in actions:
            await self._call({"command": action}, now)
        healthy = twist is not None
        twist = twist if healthy else ZERO
        if twist != self.sent or (healthy and now - self.sent_at >= REFRESH_S):
            vx, vy, wz = twist
            if await self._call({"command": "set_command", "vx": vx, "vy": vy, "wz": wz}, now):
                self.sent, self.sent_at = twist, now  # on failure, retry next tick

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
