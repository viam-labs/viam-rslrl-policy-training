"""Viam generic service devrel:duck-policy:teleop: drives a policy service (the walker's DoCommands
start, stop, set_command {vx, vy, wz}) from any input controller. Hold LB to drive: left stick
vx/vy, right stick wz. Start starts the policy, Select stops it. A controller error or Disconnect
sends one zero twist. Spec: docs/superpowers/specs/2026-11-22-microduck-teleop-design.md."""

from __future__ import annotations

from dataclasses import dataclass, field

from viam.components.input import Control, EventType

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
