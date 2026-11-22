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
