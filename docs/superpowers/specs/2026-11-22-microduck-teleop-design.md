# MicroDuck gamepad teleop: design (2026-11-22)

## Goal

Drive the MicroDuck's walk policy from any Viam input controller (a gamepad). The sticks set the
twist command, hold-to-drive gates it, and buttons start and stop the policy. The first target is
the `microduck-preview` sim machine, using `webgamepad` (a gamepad connected to the browser,
through the Viam app's Control tab). The same code later works with a gamepad plugged into the
duck's Pi (the Linux `gamepad` model), with no code change: only the `input_controller` name in the config changes.

Out of scope:
- head pose and body pose commands (the walker doesn't expose them yet);
- configurable button and axis mapping (fixed defaults until a real controller needs otherwise);
- driving the dance service.

## The `teleop` service

A new model, `devrel:duck-policy:teleop` (`rdk:service:generic`), in the existing duck-policy
module (`duck_policy/teleop.py`, registered in `duck_policy/main.py` and `meta.json`).

```json
{"name": "teleop", "api": "rdk:service:generic", "model": "devrel:duck-policy:teleop",
 "attributes": {"input_controller": "WebGamepad", "policy": "walker"}}
```

- **`input_controller`** (required): any `rdk:component:input_controller`.
- **`policy`** (required): any generic service that accepts the walker's DoCommands:
  `set_command {vx, vy, wz}`, `start` and `stop`. It's named `policy`, not `walker`, so later
  policy services can be driven the same way.
- **Validation:** both are returned as required dependencies from `validate_config`. A missing
  attribute is a config error.
- **No scaling attributes:** teleop maps the sticks to the policy's trained range, and the policy
  service clips as it does today.

### Loop

A background asyncio task, started in `new`/`reconfigure` and cancelled in `close` and before a
reconfigure, runs at 20 Hz:

1. **Read** `get_events()` from the controller: the latest event per control (`Dict[Control, Event]`,
   `Event.time` in epoch seconds). The controller only returns controls that have had an event, so a
   missing control means not held, or axis 0.
2. **Map** the events to an action with a pure function, `map_events(events, state) -> (twist, action)`:
   - **Hold-to-drive:** the twist is non-zero only while `ButtonLT` (the left bumper) is held:
     its latest event is `ButtonPress` or `ButtonHold` (the Linux gamepad sends `ButtonHold` with
     value 2 on key-repeat). Anything else (`ButtonRelease`, `Connect`, missing) is released.
     Releasing it gives a zero twist at once.
   - **Axes:** values are in −1..1. A dead zone of 0.1 maps |v| < 0.1 to 0. Gamepads read
     stick-up and stick-left as negative, so:
     - `vx = −AbsoluteY × 0.4`
     - `vy = −AbsoluteX × 0.3`
     - `wz = −AbsoluteRX × 1.0`
   - **Buttons:** a new `ButtonPress` on `ButtonStart` (one newer than the last handled, by event
     time) sends `start`. A new press on `ButtonSelect` sends `stop`.
   - **No stale presses:** webgamepad keeps its last events for the life of the component, across
     teleop restarts. So the first poll after `new` or `reconfigure` only records the current
     Start and Select event times as handled; only presses after that count.
3. **Send:** call the policy's `set_command` when the twist differs from the last one sent, and
   every 0.5 s regardless, so the policy keeps a current command.

The policy stands still below about vx 0.2 and wz 0.6 (the trained dead zone, in the milestone 3
results), so small stick movements don't move the duck. That's the policy, not teleop.

### Safety

- **The controller fails or disconnects** (`get_events` raises, or any control's latest event is
  `Disconnect`): send a zero twist once, then keep polling. Teleop never stops the policy on its own; the
  duck keeps balancing in place.
- **Releasing hold-to-drive** zeroes the twist immediately.
- **Known limit:** webgamepad keeps the last event it received, with no timeout. If the browser
  freezes, the tab closes, or the network drops while LB is held, it may keep reporting LB held (unless
  the app sends a release or `Disconnect`, which isn't confirmed). Hold-to-drive narrows this, but
  doesn't close it. The real duck's local `gamepad` model
  doesn't have this problem.
- **A failing policy call is logged at most once per second,** and the loop continues.

## Machine setup

On the preview machine and in `docs/superpowers/specs/duck-sim-machine.json`, add:
- a `WebGamepad` component: `rdk:component:input_controller`, model `webgamepad`. The name must be
  exactly `WebGamepad`, or the Control tab won't send browser input to it;
- the `teleop` service above.

To drive: open the machine's Control tab with a gamepad connected to the browser. Press Start,
hold LB and use the sticks. Because of the policy's dead zone, the duck walks only past about half
stick, and turns past about 60 %.

## Testing

- **Unit (`tests/test_duck_teleop.py`):** `map_events` with stub events covers:
  - the dead zone;
  - sign and scale per axis;
  - no twist without LB, and zero after LB is released;
  - Start and Select fire once per press, not again on the same event;
  - a press that was already there on the first poll doesn't fire;
  - missing controls and `Connect` events read as released and axis 0;
  - `Disconnect` on any control gives a zero twist.

  Plus one loop test: a stub controller and a stub policy over a few ticks check send-on-change,
  the 0.5 s refresh, and the zero-once on a controller error.
- **Manual (on the preview machine):**
  - stand, hold LB and push forward → the duck walks;
  - release LB → it stops and stands;
  - Select stops the walker; Start starts it;
  - hold LB, close the tab, and record whether the duck keeps walking (it tests the known limit).
