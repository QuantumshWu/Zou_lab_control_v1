"""Pulse parameter addressing remains pure; Pulse Scan execution is covered separately.

These tests pin immutable field addressing shared by API-slot resolution and other timing callers.
The generic scan's explicit scan-slot and API-slot execution strategies live in
``test_pulse_scan_execution.py``.
"""

from __future__ import annotations
from Zou_lab_control.neutral_atom.ports import PortCatalog

import pytest

from Zou_lab_control.neutral_atom.timing import (
    PulseParam,
    PulseTableState,
    default_imaging_template,
    enumerate_pulse_params,
    period_index_by_name,
)


def _template() -> PulseTableState:
    return default_imaging_template()


def test_period_index_by_name_resolves_named_period():
    state = _template()
    idx = period_index_by_name(state, "image_0")
    assert idx is not None
    assert str(state.periods[idx].name).strip().lower() == "image_0"
    assert period_index_by_name(state, "no-such-period") is None
    assert period_index_by_name(state, "  IMAGE_0 ") == idx


def test_pulse_param_apply_duration_returns_new_state_and_sets_value():
    state = _template()
    idx = period_index_by_name(state, "image_0")
    before = state.periods[idx].duration
    out = PulseParam("duration", str(idx), unit="us").apply(state, 12.0)
    assert out is not state
    assert state.periods[idx].duration == before
    assert out.periods[idx].duration == pytest.approx(12.0)
    assert out.periods[idx].unit == "us"


def test_pulse_param_apply_delay_sets_channel_delay():
    state = _template()
    channel = state.port_catalog.raw_lanes[0]
    out = PulseParam("delay", channel, unit="us").apply(state, 3.0)
    assert out is not state
    assert out.delays.get(channel) == 3.0
    assert channel not in state.delays


def test_pulse_param_apply_dac_is_unitless_signed_code():
    state = PulseTableState(port_catalog=PortCatalog.from_channels(["da[0]", "da[1]", "trig"]))
    state.add_period(50.0, name="hold", unit="us")
    bus = next(iter(state.bus_channels(min_width=1)))
    out = PulseParam("dac", f"{bus}@0").apply(state, -2.0)
    assert out is not state
    assert out.analog_bus_value_at_period_start(0, bus) == -2


def test_pulse_param_apply_unknown_kind_raises():
    with pytest.raises(ValueError):
        PulseParam("frequency", "0").apply(_template(), 1.0)


def test_enumerate_pulse_params_covers_durations_delays_and_dac():
    state = PulseTableState(port_catalog=PortCatalog.from_channels(["da[0]", "da[1]", "probe", "trig"]))
    state.add_period(50.0, name="cool", unit="us")
    state.add_period(2.0, name="image", unit="us")
    params = enumerate_pulse_params(state)
    kinds = {kind for kind, _, _ in params}
    assert {"duration", "delay", "dac"} <= kinds
    assert [target for kind, target, _ in params if kind == "duration"] == [
        str(i) for i in range(len(state.periods))]
    buses = list(state.bus_channels(min_width=1))
    members = {channel for channels in state.bus_channels(min_width=1).values() for channel in channels}
    assert [target for kind, target, _ in params if kind == "delay"] == (
        buses + [
            channel for channel in state.port_catalog.raw_lanes
            if channel not in members
        ])
    assert [target for kind, target, _ in params if kind == "dac"] == [
        f"{bus}@{i}" for bus in buses for i in range(len(state.periods))]
