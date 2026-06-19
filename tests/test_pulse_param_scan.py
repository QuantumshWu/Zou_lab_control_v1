"""Contract tests for the software-scan addressing layer (timing.PulseParam et al.).

A "pulse scan" sweeps a parameter by, PER POINT, editing a loaded template and recompiling --
the NON-slot software scan (so it can sweep a period duration, a channel delay, OR a DAC bus
level of any program, no hardware scan slot required).  These pin the single-source addressing:

  * ``period_index_by_name`` resolves a named period (and ``with_imaging_exposure`` still uses it);
  * ``PulseParam.apply`` returns a NEW state (input untouched) with the named parameter set,
    delegating to the real setters (so it inherits their guards) -- for duration / delay / dac;
  * ``enumerate_pulse_params`` lists exactly the addressable params (durations + delays + per-bus
    DAC), the one source the GUI dropdown and a notebook share.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.timing import (
    PulseParam,
    PulseTableState,
    default_imaging_template,
    enumerate_pulse_params,
    period_index_by_name,
)


def _template() -> PulseTableState:
    # the shipped imaging template has a named 'image' period (case-insensitive resolve target)
    return default_imaging_template()


def test_period_index_by_name_resolves_named_period():
    state = _template()
    idx = period_index_by_name(state, "image")
    assert idx is not None
    assert str(state.periods[idx].name).strip().lower() == "image"
    assert period_index_by_name(state, "no-such-period") is None
    # case / whitespace insensitive
    assert period_index_by_name(state, "  IMAGE ") == idx


def test_with_imaging_exposure_still_uses_the_shared_resolver():
    from Zou_lab_control.neutral_atom.timing import exposure_from_sequence

    state = _template()
    idx = period_index_by_name(state, "image")
    out = state.with_imaging_exposure(0.007)
    assert out is not state                                    # returns a copy
    seq = out.to_sequence(name="t")
    assert exposure_from_sequence(seq, default=0.05) == pytest.approx(0.007)
    assert period_index_by_name(out, "image") == idx           # structure preserved


def test_pulse_param_apply_duration_returns_new_state_and_sets_value():
    state = _template()
    idx = period_index_by_name(state, "image")
    before = state.periods[idx].duration
    param = PulseParam("duration", str(idx), unit="us")
    out = param.apply(state, 12.0)
    assert out is not state                                    # input never mutated
    assert state.periods[idx].duration == before               # original unchanged
    # the edited period's compiled exposure is the set value (12 us)
    from Zou_lab_control.neutral_atom.timing import exposure_from_sequence
    assert exposure_from_sequence(out.to_sequence(name="t"), default=0.05) == pytest.approx(12e-6)


def test_pulse_param_apply_delay_sets_channel_delay():
    state = _template()
    channel = state.channels[0]
    param = PulseParam("delay", channel, unit="us")
    out = param.apply(state, 3.0)
    assert out is not state
    assert out.delays.get(channel) == 3.0                      # set via set_channel_delay
    assert channel not in state.delays                         # original untouched


def test_pulse_param_apply_dac_is_unitless_signed_code():
    # build a 2-channel DAC bus template so a bus exists to address
    state = PulseTableState(channels=["da[0]", "da[1]", "trig"])
    state.add_period(50.0, name="hold", unit="us")
    bus = next(iter(state.bus_channels(min_width=1)))
    param = PulseParam("dac", f"{bus}@0")                      # no unit -> unitless signed LSB
    out = param.apply(state, -2.0)                             # signed code (2-bit bus range -2..1)
    assert out is not state
    # the bus carries the SIGNED value we set (0 = 0 V on the offset-binary driver)
    assert out.analog_bus_value_at_period_start(0, bus) == -2


def test_pulse_param_apply_unknown_kind_raises():
    with pytest.raises(ValueError):
        PulseParam("frequency", "0").apply(_template(), 1.0)


def test_enumerate_pulse_params_covers_durations_delays_and_dac():
    state = PulseTableState(channels=["da[0]", "da[1]", "probe", "trig"])
    state.add_period(50.0, name="cool", unit="us")
    state.add_period(2.0, name="image", unit="us")
    params = enumerate_pulse_params(state)
    kinds = {kind for kind, _, _ in params}
    assert {"duration", "delay", "dac"} <= kinds
    # one duration per period
    durations = [t for k, t, _ in params if k == "duration"]
    assert durations == [str(i) for i in range(len(state.periods))]
    # one delay per channel
    delays = [t for k, t, _ in params if k == "delay"]
    assert delays == list(state.channels)
    # one dac entry per (bus, period)
    dac = [t for k, t, _ in params if k == "dac"]
    assert dac == [f"{bus}@{i}" for bus in state.bus_channels(min_width=1)
                   for i in range(len(state.periods))]
    # every enumerated (kind, target) is actually applyable (or raises a CLEAR error for
    # scan-bound durations) -- they all round-trip through the real setters
    for kind, target, _label in params:
        PulseParam(kind, target, unit="us").apply(state, 1.0)
