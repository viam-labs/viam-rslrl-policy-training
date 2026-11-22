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
