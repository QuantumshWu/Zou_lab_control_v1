"""Contract tests for the software-scan addressing layer + the redesigned ``Pulse scan``
measurement.

The redesigned ``Pulse scan`` is parameterised purely by the loaded template's named slots:
  * EVERY API slot (a1, a2, ...) gets ONE numeric value set ONCE at build time;
  * EVERY scan slot (s0, s1, ...) gets ONE points expression (linspace / list) the run
    evaluates to a 1-D sweep; the run advances every scan slot in lockstep, one row at a time.

No hardcoded slot names; no single "scan target" anymore.  The PulseParam / enumerate path
remains for the GUI's editor introspection (it is the single source for the auto-generated
UI's per-slot rows).
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
    return default_imaging_template()


def _safe_unlink(path) -> None:
    """Best-effort delete of a gitignored temp pulse file, tolerant of a transient Windows
    file-handle race (AV / GC) -- a lingering temp file under pulses/ is harmless."""
    import gc
    p = Path(path)
    for _ in range(3):
        try:
            p.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()


# --------------------------------------------------------------------------- name/value plumbing
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
    param = PulseParam("duration", str(idx), unit="us")
    out = param.apply(state, 12.0)
    assert out is not state
    assert state.periods[idx].duration == before
    assert out.periods[idx].duration == pytest.approx(12.0)
    assert out.periods[idx].unit == "us"


def test_pulse_param_apply_delay_sets_channel_delay():
    state = _template()
    channel = state.channels[0]
    param = PulseParam("delay", channel, unit="us")
    out = param.apply(state, 3.0)
    assert out is not state
    assert out.delays.get(channel) == 3.0
    assert channel not in state.delays


def test_pulse_param_apply_dac_is_unitless_signed_code():
    state = PulseTableState(channels=["da[0]", "da[1]", "trig"])
    state.add_period(50.0, name="hold", unit="us")
    bus = next(iter(state.bus_channels(min_width=1)))
    param = PulseParam("dac", f"{bus}@0")
    out = param.apply(state, -2.0)
    assert out is not state
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
    durations = [t for k, t, _ in params if k == "duration"]
    assert durations == [str(i) for i in range(len(state.periods))]
    delays = [t for k, t, _ in params if k == "delay"]
    assert delays == list(state.channels)
    dac = [t for k, t, _ in params if k == "dac"]
    assert dac == [f"{bus}@{i}" for bus in state.bus_channels(min_width=1)
                   for i in range(len(state.periods))]
    for kind, target, _label in params:
        PulseParam(kind, target, unit="us").apply(state, 1.0)


# --------------------------------------------------------------------------- Pulse scan measurement
def _calibrated():
    import Zou_lab_control.neutral_atom as na

    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4), "image_shape": (48, 60)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    return exp


def _bound_probe_with_scan(exp) -> str:
    """Write a temporary probe pulse with ONE scan slot bound to its only duration, return
    the JSON path.  Lets the scan tests exercise the slot-driven path without depending on
    whatever the shipped probe_template carries (and without single_imaging_template's
    pre-bound a1 on the same period -- a clash with a scan slot on that field)."""
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod

    state = PulseTableState(channels=["trap", "cooling", "probe", "emCCD"],
                            visible_channels=["trap", "cooling", "probe", "emCCD"])
    states_load = (1, 1, 0, 0)
    states_image = (1, 0, 1, 1)
    state.periods.append(PulsePeriod(0.002, states_load, unit="s", name="load"))
    state.periods.append(PulsePeriod(0.005, states_image, unit="s", name="image"))
    state.bind_field("duration", "1", label="probe", unit="us")     # scan the image duration
    state.validate()
    out = Path("pulses") / "_test_probe_scan.json"
    state.save(str(out))
    return str(out)


def _run_pulse_scan(exp, *, y=None, **build_kwargs):
    """Build the Pulse-scan PLAN, wire a Judge-occupancy processor as the DECOUPLED y source,
    and drive the node to completion -- the SAME path the task console uses, but single-threaded
    (the occupancy is settled inline per point for deterministic headless timing).  Returns
    (x, y, node, hub)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor, PulseScanNode

    spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
    plan = spec.build(y=y, **build_kwargs)
    hub = SignalHub()
    # The y producer: a real Judge-occupancy node consuming the frame pulse-scan publishes
    # (publishes bare 'occupied'/'rate'/... since prefix="").  Not started on its own thread;
    # pulse-scan settles it INLINE per point (single-threaded determinism, same evaluate path).
    occ = OccupancyProcessor(hub, calibration=exp.readout.require(thresholds=True),
                             source_expr={"inputs": ["frame"], "source": "value = signal"}, grid_shape=(3, 4))
    plan.settle = occ.step
    node = PulseScanNode(hub, plan, x_key=spec.x_key, y_key=spec.y_key, prefix=f"{spec.key}_")
    node.run_to_completion()
    return (np.asarray(hub.latest(f"{spec.key}_{spec.x_key}"), dtype=float),
            np.asarray(hub.latest(f"{spec.key}_{spec.y_key}"), dtype=float),
            node, hub)


def test_pulse_scan_spec_carries_pulse_slots_and_signal_expr_y():
    """The decoupled spec exposes ONE pulse_slots auto-form (api + scan slots bound to the
    template) and a signal_expr y (the subscribed source) -- no reduce/shots, and it is tagged
    as a device-driving pulse_scan node."""
    exp = _calibrated()
    try:
        specs = {s.name: s for s in exp.readout.measurement_specs()}
        assert "Pulse scan" in specs
        spec = specs["Pulse scan"]
        assert (spec.x_key, spec.y_key) == ("param", "signal")
        assert spec.metadata.get("node") == "pulse_scan"
        slots_decl = next(p for p in spec.params if p.kind == "pulse_slots")
        assert slots_decl.depends_on == "template"
        # y is a multi-source signal expression (the decoupled subscription), not a reducer choice
        y_decl = next(p for p in spec.params if p.kind == "signal_expr")
        assert "inputs" in y_decl.default and "source" in y_decl.default
        keys = {p.key for p in spec.params}
        assert "reduce" not in keys and "shots" not in keys      # old reducer params are gone
        assert "n_frames" in keys
        assert not any(p.kind == "pulse_param" for p in spec.params)
    finally:
        exp.close()


def test_pulse_scan_y_tracks_subscribed_occupancy_signal():
    """The node sweeps ONE bound scan slot (x = the points) and y = the occupancy signal it
    subscribes to: after each point publishes a frame + the occupancy is settled, y equals the
    consumer's just-published value (the decoupled-source + settle contract)."""
    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=probe,
            pulse_slots={"api": {}, "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(10000, 40000, 5).reshape(-1, 1)"},
            n_frames=1, y={"inputs": ["rate"], "source": "value = signal"})
        assert x.shape == (5,) and y.shape == (5,)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        # y is the loading rate (a fraction) the occupancy node published, not a frame reduction
        assert np.all((y >= 0.0) & (y <= 1.0))
        # the final y point equals the occupancy 'rate' currently on the hub (read AFTER settle)
        assert y[-1] == pytest.approx(float(hub.latest("rate")))
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_y_expression_over_multiple_signals():
    """The y source is a multi-slot expression: it can combine several subscribed signals
    (e.g. mean of the per-site occupancy vector)."""
    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=probe,
            pulse_slots={"api": {}, "scan_code": "import numpy as np\n"
                         "scan_table = np.array([10000, 20000, 30000]).reshape(-1, 1)"},
            n_frames=1, y={"inputs": ["occupied"], "source": "value = np.mean(signal)"})
        assert x.shape == (3,) and y.shape == (3,)
        assert np.isfinite(y).all() and np.all((y >= 0.0) & (y <= 1.0))
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_api_value_is_applied_once_at_build():
    """An api-slot value the operator sets in the form is APPLIED ONCE to the loaded template
    (set_api on a known handle); a typo raises (template exposes no such handle)."""
    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        # set the probe template's first delay channel as an api slot a1, then drive it
        from Zou_lab_control.neutral_atom.timing import PulseTableState
        st = PulseTableState.load(probe)
        ch = st.channels[0]
        st.bind_api_field("delay", ch, unit="us")
        st.save(probe)
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        scan_code = "import numpy as np\nscan_table = np.linspace(1000, 3000, 3).reshape(-1, 1)"
        plan = spec.build(template=probe,
                          pulse_slots={"api": {"a1": 1.5}, "scan_code": scan_code})
        assert plan.base_state.delays.get(ch) == pytest.approx(1.5)
        with pytest.raises(ValueError):
            spec.build(template=probe,
                       pulse_slots={"api": {"a9": 1.0}, "scan_code": scan_code})
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_empty_program_uses_column_stack_default():
    """An empty scan program is not an error -- it falls back to the column_stack template for
    the bound slot count (the scan model is ONE table; the default is a usable starting table)."""
    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        plan = spec.build(template=probe, pulse_slots={"api": {}, "scan_code": ""})
        assert plan.scan_names == ["s0"]
        assert plan.scan_arrays[0].size >= 1            # the default column_stack sweep ran
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_table_column_count_must_match_slots():
    """The scan table is ONE (N_points x n_slots) array -- the slots advance in lockstep, so the
    program must give ONE column per bound scan slot.  A wrong column count fails LOUD (a 1-column
    table for a TWO-slot template is rejected, not silently padded)."""
    exp = _calibrated()
    from Zou_lab_control.neutral_atom.timing import single_imaging_template
    st = single_imaging_template()
    st.add_period(20.0, name="probe2", unit="us")
    st.bind_field("duration", "1", unit="us")
    st.bind_field("duration", "2", unit="us")
    probe = Path("pulses") / "_test_probe_2scan.json"
    st.save(str(probe))
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        with pytest.raises(ValueError, match=r"(?i)column.*2 scan slot"):
            spec.build(template=str(probe),
                       pulse_slots={"api": {}, "scan_code": "import numpy as np\n"
                                    "scan_table = np.linspace(0, 5000, 3).reshape(-1, 1)"})  # only 1 column
    finally:
        _safe_unlink(probe)
        exp.close()
