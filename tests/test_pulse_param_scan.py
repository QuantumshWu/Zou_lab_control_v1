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


# --------------------------------------------------------------------------- Pulse scan measurement
def _calibrated():
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


def _run_pulse_scan(exp, **build_kwargs):
    """Build the Pulse-scan measurement and drive it to completion through the SAME live node
    the task console uses; return (x, y) published on the hub."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode

    spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
    measurement = spec.build(**build_kwargs)
    hub = SignalHub()
    node = ScannedMeasurementNode(hub, measurement, x_key=spec.x_key, y_key=spec.y_key,
                                  prefix=f"{spec.key}_")
    node.run_to_completion()
    return (np.asarray(hub.latest(f"{spec.key}_{spec.x_key}"), dtype=float),
            np.asarray(hub.latest(f"{spec.key}_{spec.y_key}"), dtype=float),
            measurement)


def test_pulse_scan_spec_discovered_with_unique_keys():
    """The generic Pulse-scan measurement is auto-discovered into the Add-Panel catalog with
    unique x/y signal keys (the registry raises on a collision with temperature/readout)."""
    exp = _calibrated()
    try:
        specs = {s.name: s for s in exp.readout.measurement_specs()}
        assert "Pulse scan" in specs
        spec = specs["Pulse scan"]
        assert (spec.x_key, spec.y_key) == ("param", "signal")
        # the scan-target field is the dynamic pulse_param control bound to the template field
        target = next(p for p in spec.params if p.kind == "pulse_param")
        assert target.depends_on == "template"
    finally:
        exp.close()


@pytest.mark.parametrize("reduce", ["counts", "loading", "fidelity"])
def test_pulse_scan_runs_each_reduce_mode_headless(reduce):
    """Each reduce mode sweeps the template's first duration and yields a finite curve over the
    SAME camera/calibration contract (virtual==real; only frames are simulated)."""
    exp = _calibrated()
    try:
        x, y, m = _run_pulse_scan(exp, template="imaging_template.json", scan_target="duration:0",
                                  scan=(10.0, 40.0, 5), shots=4, reduce=reduce)
        assert x.shape == (5,) and y.shape == (5,)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert m.reducer.labels[0].endswith("(us)")            # x-axis labelled with the param + unit
    finally:
        exp.close()


def test_pulse_scan_delay_is_software_non_slot():
    """A channel DELAY scan runs -- proving the NON-slot software path (delay is NOT a hardware
    scan slot; it is set per point by editing the template), the capability the user asked for."""
    from Zou_lab_control.neutral_atom.operations.logic import CalibrateReadoutTask
    from Zou_lab_control.neutral_atom.timing import enumerate_pulse_params

    exp = _calibrated()
    try:
        state = CalibrateReadoutTask._resolve_template("imaging_template.json")
        kind, target, _ = next((p for p in enumerate_pulse_params(state) if p[0] == "delay"))
        x, y, _m = _run_pulse_scan(exp, template="imaging_template.json",
                                   scan_target=f"{kind}:{target}", scan=(0.0, 5.0, 3),
                                   shots=3, reduce="counts")
        assert x.shape == (3,) and np.isfinite(y).all()
    finally:
        exp.close()


def test_pulse_scan_missing_target_raises():
    """An empty scan target fails LOUD (the GUI also marks it required), never silently scans
    nothing."""
    exp = _calibrated()
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        with pytest.raises(ValueError):
            spec.build(template="imaging_template.json", scan_target="", scan=(0.0, 1.0, 3))
    finally:
        exp.close()
