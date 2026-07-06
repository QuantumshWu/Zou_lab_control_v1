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
    # A DAC bus OWNS its delay (it fans out to its members), so the bus name -- NOT each
    # member channel (da[0]/da[1]) -- is the delay target; plain channels list themselves.
    delays = [t for k, t, _ in params if k == "delay"]
    buses = list(state.bus_channels(min_width=1))
    bus_members = {ch for members in state.bus_channels(min_width=1).values() for ch in members}
    expected_delays = buses + [c for c in state.channels if c not in bus_members]
    assert delays == expected_delays
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
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"}, grid_shape=(3, 4))
    plan.settle = occ.step
    node = PulseScanNode(hub, plan, x_key=spec.x_key, y_key=spec.y_key, prefix=f"{spec.key}_")
    node.run_to_completion()
    # The node publishes the RAW (repeat, points, 1) block; a plot REDUCES the repeat axis (the
    # node never combines).  These tests check the displayed curve, so reduce it here the same way a
    # panel does -- with the ONE owned helper -- to the (points,) curve (repeat=1 -> a single pass).
    from Zou_lab_control.frontend.live import reduce_repeat
    y = np.asarray(reduce_repeat(np.asarray(hub.latest(node.y_signal), dtype=float), "replace"),
                   dtype=float).reshape(-1)
    return (np.asarray(hub.latest(node.x_signal), dtype=float), y, node, hub)


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
        assert "n_frames" not in keys                            # decoupled from the camera (#6)
        assert "y_name" in keys                                  # settable output-signal name (#7)
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
            y={"inputs": ["rate"], "source": "value = signal"})
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
            y={"inputs": ["occupied"], "source": "value = np.mean(signal)"})
        assert x.shape == (3,) and y.shape == (3,)
        assert np.isfinite(y).all() and np.all((y >= 0.0) & (y <= 1.0))
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_output_signal_name_is_settable(tmp_path):
    """#7: the y_name param renames the OUTPUT (y) signal the scan publishes, so a plot picks a
    meaningful name (e.g. 'loading_rate') instead of a generic 'signal'."""
    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=probe, y_name="loading_rate",
            pulse_slots={"api": {}, "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(10000, 40000, 4).reshape(-1, 1)"},
            y={"inputs": ["rate"], "source": "value = signal"})
        assert node.y_signal == "pulse_scan_loading_rate"
        assert "pulse_scan_loading_rate" in hub.names()       # published under the chosen name
        assert y.shape == (4,)
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_2d_grid_publishes_a_2d_map():
    """#5b: a 2-D (grid) scan -- the program declares scan_shape (n0, n1) -- publishes the per-point
    y reshaped into an (n0, n1) map (``<y>_grid``), so a 2D image panel shows the 2D scan."""
    from Zou_lab_control.neutral_atom.timing import PulseTableState, single_imaging_template
    exp = _calibrated()
    st = single_imaging_template()
    st.add_period(20.0, name="probe2", unit="us")
    st.bind_field("duration", "1", unit="us")
    st.bind_field("duration", "2", unit="us")
    probe = Path("pulses") / "_test_probe_2d.json"
    st.save(str(probe))
    grid = ("import numpy as np\n"
            "a = np.linspace(10000, 30000, 3); b = np.linspace(10000, 20000, 2)\n"
            "A, B = np.meshgrid(a, b, indexing='ij')\n"
            "scan_table = np.column_stack([A.ravel(), B.ravel()])\n"
            "scan_shape = (len(a), len(b))\n")
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=str(probe), pulse_slots={"api": {}, "scan_code": grid},
            y={"inputs": ["rate"], "source": "value = signal"})
        assert node.scan_shape == (3, 2)
        grid_name = node.y_signal + "_grid"
        assert grid_name in hub.names()
        # the grid is the RAW (repeat, n0, n1) block; a 2-D panel reduces the repeat axis to the
        # (n0, n1) map (the node never combines).  repeat=1 -> a single (1, 3, 2) slice.
        from Zou_lab_control.frontend.live import reduce_repeat
        raw_grid = np.asarray(hub.latest(grid_name))
        assert raw_grid.shape == (1, 3, 2)
        assert np.asarray(reduce_repeat(raw_grid, "replace")).shape == (3, 2)   # the 2-D scan map
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


def _scan_node(exp, *, repeat=1, tag="a"):
    """Build a PulseScanNode over a 3-point software sweep with the given ``repeat`` (0 = ∞, the ONE
    knob -- a pulse-scan pass IS a whole sweep, so ``repeat`` IS the whole-sweep count), settled INLINE
    per point (deterministic headless run).  ``tag`` keeps each probe file UNIQUE (a shared path races a
    still-open Windows handle)."""
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor, PulseScanNode
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod

    state = PulseTableState(channels=["trap", "cooling", "probe", "emCCD"],
                            visible_channels=["trap", "cooling", "probe", "emCCD"])
    state.periods.append(PulsePeriod(0.002, (1, 1, 0, 0), unit="s", name="load"))
    state.periods.append(PulsePeriod(0.005, (1, 0, 1, 1), unit="s", name="image"))
    state.bind_field("duration", "1", label="probe", unit="us")
    state.validate()
    probe = str(Path("pulses") / f"_test_probe_repeats_{tag}.json")
    state.save(probe)
    spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
    plan = spec.build(
        template=probe,
        pulse_slots={"api": {}, "scan_code": "import numpy as np\n"
                     "scan_table = np.linspace(10000, 30000, 3).reshape(-1, 1)"},
        y={"inputs": ["rate"], "source": "value = signal"})
    hub = SignalHub()
    occ = OccupancyProcessor(hub, calibration=exp.readout.require(thresholds=True),
                             source_expr={"inputs": ["frame_0"], "source": "value = signal"}, grid_shape=(3, 4))
    plan.settle = occ.step
    node = PulseScanNode(hub, plan, x_key=spec.x_key, y_key=spec.y_key, prefix=f"{spec.key}_",
                         repeat=repeat)
    return node, probe


def test_pulse_scan_no_sequencer_freezes_cleanly_instead_of_wedging():
    """A None sequencer is a LEGAL config: ``triggered_frames`` then fires nothing, so the
    camera sees no trigger and the scan point yields NO frame -- the shot returns empty
    (freeze) instead of fabricating data, and NONE of the sequencer-owned actions (fire,
    inter-point settle) may crash on ``None`` (the ``None.settle`` AttributeError wedge this
    originally guarded)."""
    exp = _calibrated()
    node = probe = None
    try:
        node, probe = _scan_node(exp, repeat=1, tag="nosettle")
        node.sequencer = None                 # no bound streamer -> nothing ever fires
        node.extra_delay_s = 1e-4             # a positive inter-point settle -> would hit None.settle
        out = node.shot()                     # must NOT raise; no trigger -> no frame -> freeze
        assert out == {}
        assert node.points_done == 0          # honestly frozen: no fabricated advance
        assert not node.finished
    finally:
        if probe:
            _safe_unlink(probe)
        exp.close()


def test_pulse_scan_repeat_is_the_whole_sweep_count():
    """#H3u-2: ``repeat`` (the ONE knob the console injects) IS the whole-sweep count for a pulse-scan
    (a pass IS a sweep).  ``repeat=K`` reaches ``node.repeat`` and means K whole sweeps."""
    exp = _calibrated()
    node = probe = None
    try:
        node, probe = _scan_node(exp, repeat=2, tag="thread")
        assert node.repeat == 2
        assert node.total_points == 6                        # 3 points x 2 whole sweeps
        # The plan no longer carries a separate sweep count (#H3u-2: subsumed by ``repeat``).
        assert not hasattr(node.plan, "scan_repeats")
        assert not hasattr(node, "free_run")
    finally:
        if probe:
            _safe_unlink(probe)
        exp.close()


def test_pulse_scan_finite_repeats_stops_after_k_whole_sweeps():
    """#H3u-2: repeat=K runs K WHOLE point-by-point sweeps then stops (a finite software sweep)."""
    exp = _calibrated()
    node = probe = None
    try:
        node, probe = _scan_node(exp, repeat=2, tag="finite")        # 3 points x 2 sweeps = 6 points
        assert node.total_points == 6
        node.run_to_completion()
        assert node.finished
        assert node.points_done == 6
    finally:
        if probe:
            _safe_unlink(probe)
        exp.close()


def test_pulse_scan_repeat_zero_is_infinite():
    """#H3u-2: repeat=0 = ∞ (the ONLY infinite knob, no free-run toggle) -- the sweep rolls forever."""
    exp = _calibrated()
    node0 = probe0 = None
    try:
        node0, probe0 = _scan_node(exp, repeat=0, tag="inf")
        assert node0.total_points == 0                       # unbounded
        assert not node0.finished
        node0.step(); node0.step(); node0.step(); node0.step()   # past one whole sweep (3 points)
        assert not node0.finished                            # repeat=0 rolls on
    finally:
        if probe0:
            _safe_unlink(probe0)
        exp.close()


def test_pulse_scan_repeats_keeps_all_k_sweeps_for_averaging():
    """#H3u-2 REGRESSION: repeat=K must KEEP all K sweeps in the published block (the repeat-axis depth
    == K), so the plot's repeat_mode=average actually averages K sweeps.  The historical bug (#H3t): the
    ring used a separate camera-frame repeat (default 1), so all K sweeps overwrote ONE slot and every
    sweep but the LAST was silently dropped.  Unifying to the ONE ``repeat`` knob makes the ring = K."""
    import numpy as np
    exp = _calibrated()
    node = probe = None
    try:
        node, probe = _scan_node(exp, repeat=3, tag="depth")
        assert node._ring == 3                               # ring holds all K sweeps
        node.run_to_completion()
        block = np.asarray(node._publish_raw())
        assert block.shape[0] == 3, f"published depth must be K=3 (all sweeps kept), got {block.shape}"
        # no sweep slot was left unwritten (i.e. none was overwritten/dropped)
        assert not np.isnan(block).all(axis=tuple(range(1, block.ndim))).any()
    finally:
        if probe:
            _safe_unlink(probe)
        exp.close()


def _bound_probe_with_api() -> str:
    """A probe with ONE API slot (a duration handle) and NO scan slot -- the api-only sweep case."""
    from Zou_lab_control.neutral_atom.timing import PulseTableState
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod

    state = PulseTableState(channels=["trap", "cooling", "probe", "emCCD"],
                            visible_channels=["trap", "cooling", "probe", "emCCD"])
    state.periods.append(PulsePeriod(0.002, (1, 1, 0, 0), unit="s", name="load"))
    state.periods.append(PulsePeriod(0.005, (1, 0, 1, 1), unit="s", name="image"))
    state.bind_api_field("duration", "1", unit="us")     # a1 = the image duration (API handle)
    state.validate()
    out = Path("pulses") / "_test_probe_api.json"
    state.save(str(out))
    return str(out)


def test_pulse_scan_api_sweep_software_drives_each_point():
    """#H3s-F7: in API mode, a pulse scan sweeps the api slots in SOFTWARE -- per point it set_api's
    that row's value, fires, and reads y.  x = the api sweep points (the swept API slot's value),
    exactly like the hardware scan table but loaded in software (the SAME single scan_code program,
    selected by scan_mode='api')."""
    exp = _calibrated()
    probe = _bound_probe_with_api()
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=probe,
            pulse_slots={"api": {}, "scan_mode": "api", "scan_code": "import numpy as np\n"
                         "scan_table = np.linspace(2.0, 8.0, 4).reshape(-1, 1)"},
            y={"inputs": ["rate"], "source": "value = signal"})
        assert node.scan_names == []                       # no hardware scan slot
        assert node.api_names == ["a1"]                    # the api slot is the swept dimension
        assert x.shape == (4,) and y.shape == (4,)
        assert np.allclose(x, [2.0, 4.0, 6.0, 8.0])        # x = the api sweep points
        assert np.isfinite(y).all() and np.all((y >= 0.0) & (y <= 1.0))
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_extra_delay_settles_via_device():
    """#H3-3: the inter-point wait (load -> on_pulse -> wait pulse done -> settle -> next) is owned
    by the DEVICE: the node calls sequencer.settle(extra_delay) once per point, never a hand-rolled
    sleep in the analysis layer."""
    exp = _calibrated()
    probe = _bound_probe_with_api()
    try:
        seen: list[float] = []
        seq = exp.devices.sequencer
        orig = seq.settle
        seq.settle = lambda s, **k: (seen.append(float(s)), orig(s, **k))[1]   # record + still call real
        try:
            x, y, node, hub = _run_pulse_scan(
                exp, template=probe,
                pulse_slots={"api": {}, "scan_mode": "api", "extra_delay": 0.05,
                             "scan_code": "import numpy as np\n"
                             "scan_table = np.linspace(2.0, 8.0, 3).reshape(-1, 1)"},
                y={"inputs": ["rate"], "source": "value = signal"})
        finally:
            seq.settle = orig
        assert len(seen) == 3                               # one settle per scan point
        assert all(abs(s - 0.05) < 1e-9 for s in seen)      # the adjustable extra delay
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


def test_pulse_scan_3_level_grid_publishes_the_nested_block_for_a_facet_grid():
    """A 3-LEVEL grid scan -- the program declares scan_shape (n0, n1, n2) -- publishes the per-point
    y reshaped into the (repeat, n0, n1, n2) block, and the ONE facet rule (live.facet_cells) expands
    its OUTER axis into n0 cells of (n1, n2) maps: the 3-D pulse scan displayed as a 2d facet grid.
    (The old scan_shape validator hard-rejected anything but a 2-tuple; a nested scan is now a
    first-class declaration -- the streamed table stays flat, only the display un-flattens.)"""
    from Zou_lab_control.neutral_atom.timing import PulseTableState, single_imaging_template
    exp = _calibrated()
    st = single_imaging_template()
    st.add_period(20.0, name="probe2", unit="us")
    st.add_period(20.0, name="probe3", unit="us")
    st.bind_field("duration", "1", unit="us")
    st.bind_field("duration", "2", unit="us")
    st.bind_field("duration", "3", unit="us")
    probe = Path("pulses") / "_test_probe_3d.json"
    st.save(str(probe))
    prog = ("import numpy as np\n"
            "a = np.linspace(10000, 30000, 3)\n"
            "b = np.linspace(10000, 20000, 2)\n"
            "c = np.linspace(15000, 25000, 2)\n"
            "A, B, C = np.meshgrid(a, b, c, indexing='ij')\n"
            "scan_table = np.column_stack([A.ravel(), B.ravel(), C.ravel()])\n"
            "scan_shape = (len(a), len(b), len(c))\n")
    try:
        x, y, node, hub = _run_pulse_scan(
            exp, template=str(probe), pulse_slots={"api": {}, "scan_code": prog},
            y={"inputs": ["rate"], "source": "value = signal"})
        assert node.scan_shape == (3, 2, 2)
        assert node.grid_shape == (3, 2, 2)
        grid_name = node.y_signal + "_grid"
        assert grid_name in hub.names()
        raw_grid = np.asarray(hub.latest(grid_name))
        assert raw_grid.shape == (1, 3, 2, 2), raw_grid.shape
        # the flat y block + the declared points shape feed the facet slicer: outer axis -> 3 cells,
        # each a (2, 2) frame -- exactly what a console grid panel bound to <y> with facet=points:0 shows
        from Zou_lab_control.frontend.live import facet_cells
        block = np.asarray(hub.latest(node.y_signal))          # (repeat, 12, 1)
        cells = facet_cells(block, "points:0", sub_plot_kind="2d", points_shape=(3, 2, 2))
        assert len(cells) == 3 and cells[0].shape == (2, 2)
        np.testing.assert_allclose(np.stack(cells).reshape(3, 2, 2), raw_grid[0])
    finally:
        _safe_unlink(probe)
        exp.close()


def test_pulse_scan_stale_y_inputs_record_nan_not_the_previous_shot(monkeypatch, caplog):
    """#C1: a y input that never refreshes within the settle window (its producer is NOT running)
    must be recorded as NaN -- never the hub's PREVIOUS-shot value.  The old behaviour read
    whatever was in the hub after the timeout, silently filling the whole curve with one stale
    number (plus a 5 s stall per point).  The never-wedge contract stays: the scan completes."""
    import logging

    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import PulseScanNode

    exp = _calibrated()
    probe = _bound_probe_with_scan(exp)
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        plan = spec.build(template=probe,
                          pulse_slots={"api": {}, "scan_mode": "table",
                                       "scan_code": "import numpy as np\n"
                                       "scan_table = np.linspace(2.0, 8.0, 3).reshape(-1, 1)"},
                          y={"inputs": ["rate"], "source": "value = signal"})
        hub = SignalHub()
        hub.publish({"rate": 0.73})                      # a STALE value from a previous run
        plan.settle = None                               # no inline consumer -- nothing refreshes 'rate'
        monkeypatch.setattr(PulseScanNode, "SETTLE_TIMEOUT_S", 0.05)   # fast timeout for the test
        node = PulseScanNode(hub, plan, x_key=spec.x_key, y_key=spec.y_key, prefix=f"{spec.key}_")
        with caplog.at_level(logging.WARNING):
            node.run_to_completion()
        block = np.asarray(hub.latest(node.y_signal), dtype=float)
        assert np.all(np.isnan(block)), block            # every stale point is NaN, never 0.73
        assert any("did not refresh" in r.message for r in caplog.records)   # visible, once
        assert sum("did not refresh" in r.message for r in caplog.records) == 1
    finally:
        _safe_unlink(probe)
        exp.close()
