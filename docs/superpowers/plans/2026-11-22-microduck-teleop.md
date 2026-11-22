# MicroDuck Gamepad Teleop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A `devrel:duck-policy:teleop` service that drives the MicroDuck walk policy from any Viam
input controller: the sticks set the twist, hold-to-drive is on LB, Start and Select start and stop
the policy.

**Architecture:** One new file, `duck_policy/teleop.py`, in the existing duck-policy module. It has:
- a pure `map_events(events, state)`, which turns the controller's latest events into a twist and
  button actions;
- a `Teleop` generic service whose 20 Hz background task polls `get_events()`, calls `map_events`,
  and sends DoCommands to the `policy` dependency (the walker).

The walker is unchanged. Spec: `docs/superpowers/specs/2026-11-22-microduck-teleop-design.md`.

**Tech Stack:** Python 3.12, viam-sdk ≥ 0.85 (`viam.components.input`, `EasyResource`), pytest
(no pytest-asyncio: tests use `asyncio.run`), uv.

---

## Facts the implementer needs

- **Running tests:** `uv run pytest -q` from the repo root (`/Users/nick.hehr/src/viam-rslrl-policy-training`).
  The baseline is 113 passed, 1 skipped.
- **The SDK's input API** (checked against the installed SDK):
  - `from viam.components.input import Control, Controller, Event, EventType`;
  - `Event(time: float, event: EventType, control: Control, value: float)`, where `time` is epoch seconds;
  - `await controller.get_events()` returns `Dict[Control, Event]`, the latest event per control.
    It only includes controls that have had an event;
  - the names used here are `Control.ABSOLUTE_X`, `ABSOLUTE_Y`, `ABSOLUTE_RX`, `BUTTON_LT`,
    `BUTTON_START` and `BUTTON_SELECT`, and `EventType.BUTTON_PRESS`, `BUTTON_HOLD`, `DISCONNECT`
    and `CONNECT`;
  - `Controller.get_resource_name("WebGamepad")` gives `rdk:component:input_controller/WebGamepad`,
    and `viam.services.generic.Generic.get_resource_name("walker")` gives `rdk:service:generic/walker`.
- **The walker's DoCommands** (`duck_policy/service.py`): `{"command": "start"}`,
  `{"command": "stop"}` and `{"command": "set_command", "vx", "vy", "wz"}`. It clips to
  ±0.4 / ±0.3 / ±1.0.
- **Pattern to follow:** `duck_policy/service.py`: `EasyResource`, a `validate_config` that returns
  `(deps, [])`, `new` calling `reconfigure`, and `struct_to_dict(config.attributes)`.
- **Registration:** the model registers when its module is imported. `duck_policy/main.py` imports
  each model module, and `duck_policy/meta.json` lists each model.

## File structure

| File | Change | Responsibility |
|---|---|---|
| `duck_policy/teleop.py` | create | `TeleopState`, `map_events` (pure), and the `Teleop` service (`validate_config`, `reconfigure`, `tick`, loop, `close`) |
| `duck_policy/main.py` | modify | import `duck_policy.teleop` so the model registers |
| `duck_policy/meta.json` | modify | add the teleop model entry |
| `tests/test_duck_teleop.py` | create | `map_events` cases, `validate_config`, `tick` with stub controller and policy |
| `docs/superpowers/specs/duck-sim-machine.json` | modify (Task 3) | add the `WebGamepad` component and the `teleop` service |

---

### Task 1: `map_events`, the pure mapping

**Files:**
- Create: `duck_policy/teleop.py`
- Test: `tests/test_duck_teleop.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_duck_teleop.py`:

```python
import asyncio

import pytest
from viam.components.input import Control, Event, EventType
from viam.proto.app.robot import ComponentConfig
from viam.utils import dict_to_struct

from duck_policy.teleop import TeleopState, map_events

P, H, R = EventType.BUTTON_PRESS, EventType.BUTTON_HOLD, EventType.BUTTON_RELEASE
ABS = EventType.POSITION_CHANGE_ABSOLUTE


def ev(control, event, value=0.0, t=1.0):
    return Event(time=t, event=event, control=control, value=value)


def events(*evs):
    return {e.control: e for e in evs}


def primed():
    state = TeleopState()
    map_events({}, state)  # the first poll only records button times
    return state


def held(*more):
    return events(ev(Control.BUTTON_LT, P, 1.0), *more)


def test_no_twist_without_lb():
    twist, actions = map_events(events(ev(Control.ABSOLUTE_Y, ABS, -1.0)), primed())
    assert twist == (0.0, 0.0, 0.0) and actions == []


def test_axes_sign_and_scale_while_held():
    twist, _ = map_events(held(ev(Control.ABSOLUTE_Y, ABS, -1.0), ev(Control.ABSOLUTE_X, ABS, -1.0),
                               ev(Control.ABSOLUTE_RX, ABS, -1.0)), primed())
    assert twist == pytest.approx((0.4, 0.3, 1.0))  # stick up, left, right-stick left: forward, left, CCW
    twist, _ = map_events(held(ev(Control.ABSOLUTE_Y, ABS, 0.5)), primed())
    assert twist == pytest.approx((-0.2, 0.0, 0.0))


def test_button_hold_counts_as_held():
    twist, _ = map_events(events(ev(Control.BUTTON_LT, H, 2.0), ev(Control.ABSOLUTE_Y, ABS, -1.0)), primed())
    assert twist == pytest.approx((0.4, 0.0, 0.0))


def test_released_or_connect_lb_is_zero():
    for lt in (ev(Control.BUTTON_LT, R, 0.0), ev(Control.BUTTON_LT, EventType.CONNECT, 0.0)):
        twist, _ = map_events(events(lt, ev(Control.ABSOLUTE_Y, ABS, -1.0)), primed())
        assert twist == (0.0, 0.0, 0.0)


def test_dead_zone_and_missing_axes():
    twist, _ = map_events(held(ev(Control.ABSOLUTE_Y, ABS, 0.09), ev(Control.ABSOLUTE_X, ABS, -0.09)), primed())
    assert twist == (0.0, 0.0, 0.0)  # RX missing reads as 0 too


def test_disconnect_on_any_control_gives_none():
    twist, actions = map_events(held(ev(Control.ABSOLUTE_Y, ABS, -1.0), ev(Control.ABSOLUTE_X, EventType.DISCONNECT)),
                                primed())
    assert twist is None and actions == []


def test_start_and_select_fire_once_per_press():
    state = primed()
    _, actions = map_events(events(ev(Control.BUTTON_START, P, 1.0, t=5.0)), state)
    assert actions == ["start"]
    _, actions = map_events(events(ev(Control.BUTTON_START, P, 1.0, t=5.0)), state)
    assert actions == []  # same event again
    _, actions = map_events(events(ev(Control.BUTTON_START, R, 0.0, t=5.5),
                                   ev(Control.BUTTON_SELECT, P, 1.0, t=6.0)), state)
    assert actions == ["stop"]


def test_presses_present_on_the_first_poll_are_stale():
    state = TeleopState()
    old = events(ev(Control.BUTTON_START, P, 1.0, t=3.0), ev(Control.BUTTON_SELECT, P, 1.0, t=4.0))
    assert map_events(old, state)[1] == []
    assert map_events(old, state)[1] == []
    assert map_events(events(ev(Control.BUTTON_START, P, 1.0, t=7.0)), state)[1] == ["start"]
```

- [ ] **Step 2: Run the tests and check that they fail**

Run: `uv run pytest tests/test_duck_teleop.py -q`
Expected: collection error, `ModuleNotFoundError: No module named 'duck_policy.teleop'`.

- [ ] **Step 3: Write `map_events`**

`duck_policy/teleop.py`:

```python
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
```

(On the first call the loop never appends an action, because `primed` is False. `actions = []`
under `if first` keeps that explicit.)

- [ ] **Step 4: Run the tests and check that they pass**

Run: `uv run pytest tests/test_duck_teleop.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add duck_policy/teleop.py tests/test_duck_teleop.py
git commit -m "feat: teleop map_events (hold-to-drive, sticks to twist, Start/Select)"
```

---

### Task 2: The `Teleop` service and its registration

**Files:**
- Modify: `duck_policy/teleop.py` (append the service)
- Modify: `duck_policy/main.py`, `duck_policy/meta.json`
- Test: `tests/test_duck_teleop.py` (append)

- [ ] **Step 1: Write the failing tests** (append to `tests/test_duck_teleop.py`)

```python
from duck_policy.teleop import REFRESH_S, Teleop  # noqa: E402


def tcfg(attrs):
    return ComponentConfig(name="teleop", api="rdk:service:generic", model="devrel:duck-policy:teleop",
                           attributes=dict_to_struct(attrs))


def test_validate_requires_input_controller_and_policy():
    with pytest.raises(ValueError, match="input_controller"):
        Teleop.validate_config(tcfg({"policy": "walker"}))
    with pytest.raises(ValueError, match="policy"):
        Teleop.validate_config(tcfg({"input_controller": "WebGamepad"}))
    deps, _ = Teleop.validate_config(tcfg({"input_controller": "WebGamepad", "policy": "walker"}))
    assert sorted(deps) == ["WebGamepad", "walker"]


class StubController:
    def __init__(self):
        self.events, self.fail = {}, False

    async def get_events(self, **kwargs):
        if self.fail:
            raise RuntimeError("controller gone")
        return self.events


class StubPolicy:
    def __init__(self):
        self.calls = []

    async def do_command(self, command, **kwargs):
        self.calls.append(command)
        return {}


def make():
    t = Teleop("teleop")
    ctl, pol = StubController(), StubPolicy()
    t._configure(ctl, pol)
    return t, ctl, pol


def twists(pol):
    return [(c["vx"], c["vy"], c["wz"]) for c in pol.calls if c["command"] == "set_command"]


def test_tick_sends_on_change_and_refreshes():
    t, ctl, pol = make()
    asyncio.run(t.tick(0.0))  # first poll: zero twist sent once
    asyncio.run(t.tick(0.05))  # unchanged, not yet due for a refresh
    assert twists(pol) == [(0.0, 0.0, 0.0)]
    ctl.events = held(ev(Control.ABSOLUTE_Y, ABS, -1.0))
    asyncio.run(t.tick(0.1))
    assert twists(pol)[-1] == pytest.approx((0.4, 0.0, 0.0))
    asyncio.run(t.tick(0.1 + REFRESH_S))  # unchanged but due
    assert len(twists(pol)) == 3


def test_tick_sends_button_actions():
    t, ctl, pol = make()
    asyncio.run(t.tick(0.0))
    ctl.events = events(ev(Control.BUTTON_START, P, 1.0, t=9.0))
    asyncio.run(t.tick(0.05))
    assert {"command": "start"} in pol.calls


def test_controller_error_sends_zero_once():
    t, ctl, pol = make()
    ctl.events = held(ev(Control.ABSOLUTE_Y, ABS, -1.0))
    asyncio.run(t.tick(0.0))
    ctl.fail = True
    for i in range(1, 30):  # 1.45 s of errors: one zero, no refreshes
        asyncio.run(t.tick(i * 0.05))
    assert twists(pol) == [pytest.approx((0.4, 0.0, 0.0)), (0.0, 0.0, 0.0)]
    assert all(c["command"] == "set_command" for c in pol.calls)  # never stops the policy


def test_disconnect_sends_zero_once():
    t, ctl, pol = make()
    ctl.events = held(ev(Control.ABSOLUTE_Y, ABS, -1.0))
    asyncio.run(t.tick(0.0))
    ctl.events = held(ev(Control.ABSOLUTE_Y, EventType.DISCONNECT))
    for i in range(1, 30):
        asyncio.run(t.tick(i * 0.05))
    assert twists(pol)[-1] == (0.0, 0.0, 0.0) and len(twists(pol)) == 2


def test_policy_errors_dont_stop_the_loop():
    t, ctl, pol = make()

    async def boom(command, **kwargs):
        raise RuntimeError("walker down")

    pol.do_command = boom
    asyncio.run(t.tick(0.0))  # must not raise
    asyncio.run(t.tick(0.05))


def test_close_sends_zero():
    t, ctl, pol = make()
    ctl.events = held(ev(Control.ABSOLUTE_Y, ABS, -1.0))
    asyncio.run(t.tick(0.0))
    asyncio.run(t.close())
    assert twists(pol)[-1] == (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run the tests and check that they fail**

Run: `uv run pytest tests/test_duck_teleop.py -q`
Expected: `ImportError: cannot import name 'REFRESH_S'`.

- [ ] **Step 3: Write the service** (append to `duck_policy/teleop.py`, and merge the imports into
  the top of the file)

```python
import asyncio
import math
import time
from typing import Any

from viam.components.input import Controller
from viam.proto.app.robot import ComponentConfig
from viam.resource.easy_resource import EasyResource
from viam.resource.types import Model, ModelFamily
from viam.services.generic import Generic
from viam.utils import struct_to_dict

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
```

Keep `from __future__ import annotations` as the first statement of the file when merging the imports.

Notes:
- **Zero once on errors:** while the controller is unhealthy, the twist is ZERO and refreshes are off.
  So zero goes out once, on the change, and a run of errors sends nothing more.
- **Recovery:** when the controller comes back healthy and LB isn't held, ZERO equals `self.sent`,
  so nothing is sent until the refresh is due.
- **Lifecycle:** on a config change this SDK doesn't call `reconfigure` in place. It awaits
  `close()` on the old instance and calls `new()` again (which calls `reconfigure`). Both run on the
  module's event loop, so `get_running_loop()` works. The new instance starts unprimed and sends a
  zero twist on its first tick. The tests call `_configure` directly and drive `tick`, so they
  never start the background task.

- [ ] **Step 4: Register the model**

`duck_policy/main.py`, after the service import:

```python
from duck_policy.teleop import Teleop  # noqa: F401  (registers the model)
```

`duck_policy/meta.json`, append to `models`:

```json
    {
      "api": "rdk:service:generic",
      "model": "devrel:duck-policy:teleop",
      "short_description": "Drives a policy service from an input controller: hold LB, sticks to twist, Start/Select to start/stop."
    }
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_duck_teleop.py -q && uv run pytest -q`
Expected: 15 passed in the teleop file. The whole suite: 128 passed, 1 skipped (113 + 15).

Check that the module still imports: `uv run python -c "import duck_policy.main"` (no output, exit 0).

- [ ] **Step 6: Commit**

```bash
git add duck_policy/teleop.py duck_policy/main.py duck_policy/meta.json tests/test_duck_teleop.py
git commit -m "feat: devrel:duck-policy:teleop service (input controller -> policy DoCommands)"
```

---

### Task 3 (controller, with the user): the preview machine and a manual test

The controller does this task (it needs the Viam MCP tools and the user's browser gamepad), not an
implementer.

- [ ] **Step 1: Machine config.** Add to `docs/superpowers/specs/duck-sim-machine.json` and to the
  live `microduck-preview` config (read the revision with `read_machine_config`, then write it with
  `set_machine_config`):

```json
{"name": "WebGamepad", "api": "rdk:component:input_controller", "model": "webgamepad", "attributes": {}}
```

in `components`, and

```json
{"name": "teleop", "api": "rdk:service:generic", "model": "devrel:duck-policy:teleop",
 "attributes": {"input_controller": "WebGamepad", "policy": "walker"}}
```

in `services`.

- [ ] **Step 2: Restart the duck-policy module** so it registers the new model. It's a local module,
  and the running process predates `teleop.py`. Restart the background viam-server task (TaskStop,
  then start it again with the same command and log path). Run `uv run /Users/nick.hehr/.claude/plugins/cache/viam-mujoco-sim/mujoco-sim/0.1.6/skills/mujoco-sim-cell/scripts/cell.py wait --cell <scratchpad>/microduck-preview/cell.json`
  (the mujoco-sim-cell skill's tool), then check
  `read_machine_logs` for `teleop` errors.
- [ ] **Step 3: Manual test with the user.** The user opens the machine's Control tab with a gamepad
  connected to the browser.
  1. Start → the walker reports `running` (`run_docommand` status).
  2. Hold LB and push the left stick fully forward → the duck walks (watch `trunk_pos` x through the
     duck's `io`, or the overview camera).
  3. Release LB → the duck stops and stands.
  4. Select → the walker reports `stopped`.
  5. Start, hold LB with the stick forward, then close the tab: record whether the duck keeps
     walking. This is the spec's known limit; just record the result.
- [ ] **Step 4: Commit** the machine config, and add a short "Teleop" note with the step 3 results to
  `docs/superpowers/specs/2026-11-22-microduck-teleop-design.md`.

```bash
git add docs/superpowers/specs/duck-sim-machine.json docs/superpowers/specs/2026-11-22-microduck-teleop-design.md
git commit -m "feat: preview machine gets WebGamepad + teleop; manual test notes"
```

## Done when

- `uv run pytest -q` shows 128 passed, 1 skipped.
- The preview machine has `WebGamepad` and `teleop`, and they're ready.
- The user has driven the duck with LB and the sticks, and the close-the-tab result is recorded.
