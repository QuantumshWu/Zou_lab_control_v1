"""MOT coil-field optimisation chain -- MECHANICAL contracts for every link.

The chain: a pulse template with THREE named DAC scan slots -> one hardware table drives the coil
buses as bit-channel pulses -> the monitor-camera measurement observes the fired levels -> a
2d-panel box drag creates a generic ROI crop row whose scalar y Pulse Scan consumes -> the 3-D
tensor facets by its declared point shape.  The Optimize-MOT-field task uses the same scan-slot
pulse semantics to find the optimum.  Each contract pins one link so the chain cannot diverge:

* ``decode_analog_bus`` is the scan-slot resolver's inverse THROUGH the compiled artefact: resolving
  a DAC slot must bake the value into period states, or the virtual/real sequence would ignore it;
  the decoder reads the
  DELAYED base-cycle timeline -- the one ``edges()`` actually streams -- never the raw pulse list;
* the virtual monitor camera senses ONLY the sequence FIRED over its wired streamer (the
  ``sequencer`` construction parameter -- the simulated trigger cable); free-running (its
  ``Software`` default) it images the held/safe output state, never anyone's set-points;
* the intensity primitive (disc minus annulus -- the task's own math);
* the hardware scan carries its declared ``scan_shape`` for generic tensor faceting;
* the task converges on the SAME model the frames obey (no ground-truth peeking: the test talks
  to the public device model ``mot_efficiency``/``b0``, the task only ever sees frames).

Expected values are DERIVED from the device object (``cam.b0`` etc.) -- never re-typed literals.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.devices.registry import load_devices
from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames
from Zou_lab_control.neutral_atom.operations.tasks.mot_field import (
    DEFAULT_MOT_TEMPLATE, OptimizeMotFieldTask, mot_roi_intensity)
from Zou_lab_control._clock import DEFAULT_CLOCK_HZ
from Zou_lab_control.neutral_atom.timing import (
    BUS_SAFE_SIGNED_LEVEL, PulseTableState, resolve_fireable_template, single_imaging_template)
from Zou_lab_control.neutral_atom.timing.sequence import decode_analog_bus

MOT_TEMPLATE = str(REPO_ROOT / DEFAULT_MOT_TEMPLATE)


def _mot_state():
    # The ONE timing-layer fireable loader the MOT task itself uses (resolve + hardware tick).
    return resolve_fireable_template(MOT_TEMPLATE, default_name=DEFAULT_MOT_TEMPLATE,
                                     default_factory=single_imaging_template)


def test_mot_template_projects_semantic_slots_to_real_board_catalog():
    """The bundled recipe is reusable, but the connected sequencer owns all raw lanes,
    bus widths/indices and clocks.  Projection preserves public da_x/y/z names while
    binding them to the real 10-bit da_bias buses and the real emCCD trigger."""

    hardware = PulseTableState.load(
        REPO_ROOT / "pulses" / "camera_imaging_address_switch.json").port_catalog
    aligned = _mot_state().aligned_to_catalog(hardware)
    assert aligned.scan_names == ["da_x", "da_y", "da_z"]
    assert [slot.dac_bus for slot in aligned.scan_slots] == [
        "da_bias_x", "da_bias_y", "da_bias_z"]
    assert {name: hardware.by_key[name].width for name in aligned.analog_bus_modes} == {
        "da_bias_x": 10, "da_bias_y": 10, "da_bias_z": 10}
    trig = hardware.by_key["ch11"].lanes[0]
    bit = hardware.raw_lanes.index(trig)
    assert [period.states[bit] for period in aligned.periods] == [0, 1, 0]
    aligned.validate()


def _mid_frame_time(sequence) -> float:
    # Mid of the DELAYED base-cycle timeline (base_duration is delay-inclusive) -- the same
    # sense-time rule VirtualMotCamera._render_frames applies, so a .delay() on the coil bit
    # channels moves this sense point with the hardware output.
    return 0.5 * sequence.base_duration


def _edges_level(sequence, members, at_time: float) -> int:
    """Read a bus level straight off the streamed ``edges()`` timeline -- the independent ground
    truth for what the hardware plays at ``at_time`` (decoder and streamer cannot drift).  The DA
    pins add ONE rule on top of the raw edge masks: each DA bit's delay scheduler idles at its
    ``SAFE_BIT`` -- the bits of the mid-scale ``BUS_SAFE_VALUE`` code -- until the bus's first
    scheduled change, so before ANY member bit has gone high the pins play the SAFE level
    (signed 0 = 0 V), never the all-bits-low word (``engine_model.bus_delay_line_reference``)."""
    ticks, masks, channels = sequence.edges()
    tick = at_time * DEFAULT_CLOCK_HZ
    # a member with no pulses at all (an un-set bit -- the encoder emits no pulse for it) never
    # appears in the edge channel list: it is low for the whole sequence
    idx = [(bit, channels.index(ch)) for bit, ch in enumerate(members) if ch in channels]
    mask = 0
    window_started = False   # a member bit high at/before the sample = the delayed window began
    for t, m in zip(ticks, masks):
        if t > tick:
            break
        mask = m
        if any((m >> i) & 1 for _, i in idx):
            window_started = True
    if not window_started:
        return BUS_SAFE_SIGNED_LEVEL
    word = sum(1 << bit for bit, i in idx if (mask >> i) & 1)
    return word - (1 << (len(members) - 1))


# --------------------------------------------------------------------------- encode <-> decode
@pytest.mark.parametrize("values", [(7, -5, 11), (0, 0, 0), (-31, 31, 1)])
def test_decode_analog_bus_inverts_scan_slot_resolve_through_compiled_sequence(values):
    """semantic scan-slot resolve -> to_sequence -> decode_analog_bus round-trips EXACTLY (incl. the signed
    range of the template's 10-bit buses -- except the full-negative code -2^(B-1), whose
    all-bits-low projection is bit-identical to an UNDRIVEN bus and so decodes as the safe level;
    that corner is pinned in test_virtual_da_safe_state.py).  This pins BOTH directions: the
    encoder must bake each resolved DAC value into the compiled artefact, and the decoder must be
    its exact inverse."""
    state = _mot_state()
    slots = state.scan_names
    resolved = state.with_slots_resolved(dict(zip(slots, values)))
    sequence = resolved.to_sequence()
    t = _mid_frame_time(sequence)
    decoded = [decode_analog_bus(sequence, members, t)
               for members in state.port_catalog.analog_buses.values()]
    assert decoded == list(values)


def test_decode_analog_bus_reads_the_delayed_timeline():
    """``.delay()`` on the bus bit channels shifts the window the hardware DRIVES (``edges()``
    streams the shifted pulses), so the decoder must read the SAME delayed base-cycle timeline:
    BEFORE the delayed window the DA pins rest at the SAFE level (each bit's delay scheduler
    idles at its ``SAFE_BIT`` -- the mid-scale ``BUS_SAFE_VALUE`` code = 0 V -- never the
    all-bits-low full-negative word), AFTER it the programmed level appears -- both
    cross-checked against ``edges()`` itself.  (Regressions: decode used to read the RAW pulse
    list, returning the programmed level at a time the hardware had not started driving yet;
    then it read the delayed timeline but decoded the pre-window idle as the all-low word
    -2^(B-1) while the hardware idles at the mid code.)"""
    state = _mot_state()
    slots = state.scan_names
    values = (7, -5, 11)
    sequence = state.with_slots_resolved(dict(zip(slots, values))).to_sequence()
    dt = 20e-6
    for members in state.port_catalog.analog_buses.values():
        for channel in members:
            sequence = sequence.delay(channel, dt)
    t_after = _mid_frame_time(sequence)                # mid of the DELAYED base cycle
    t_before = 0.25 * dt                               # the delayed bits have not started driving
    for members, value in zip(state.port_catalog.analog_buses.values(), values):
        assert decode_analog_bus(sequence, members, t_after) == value \
            == _edges_level(sequence, members, t_after)
        assert decode_analog_bus(sequence, members, t_before) == BUS_SAFE_SIGNED_LEVEL \
            == _edges_level(sequence, members, t_before)


# --------------------------------------------------------------------------- virtual sensing
def test_virtual_mot_camera_senses_only_the_fired_sequence():
    """The virtual monitor camera senses ONLY what the wired streamer drives: free-running
    (``Software``, the real monitor's discovery default) it images the SAFE state before anything
    fires -- one dark frame at all-zero coil levels, never a fabricated bright MOT; a sequence
    FIRED on its wired streamer reconstructs each coil level from the compiled bit-channel pulses
    (never from anyone's set-points), and the frame brightness follows its own public MOT model."""
    ds = load_devices("virtual", open_devices=False)
    cam, seqr = ds.devices["monitor_camera"], ds.devices["sequencer"]
    state = _mot_state()
    slots = state.scan_names

    # Nothing fired: the free-running sensor still delivers a frame (a real Software Basler never
    # freezes), imaging the SAFE state -- every coil DAC parked at signed level 0.
    idle = cam.acquire(1)
    assert len(idle) == 1 and idle[0].shape == tuple(cam.sensor_shape)
    assert cam.last_levels == {bus: 0.0 for bus in cam.coil_buses}

    at_peak = {name: int(cam.b0[bus]) for name, bus in zip(slots, cam.coil_buses)}
    seq = state.with_slots_resolved(at_peak).to_sequence()
    frame_peak = triggered_frames(cam, seqr, seq, 1)[0]
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}

    far = {name: int(cam.b0[bus] - 4 * cam.b_sigma[bus]) for name, bus in zip(slots, cam.coil_buses)}
    frame_far = triggered_frames(cam, seqr, state.with_slots_resolved(far).to_sequence(), 1)[0]
    # brightness ratio follows the SAME public model the optimum test uses.  The faithful sensor
    # is full-frame (the spot is a tiny fraction of 2.3 MP), so brightness is read AT the spot
    # centre and the threshold derives from the camera's own model, never a re-typed constant.
    assert cam.mot_efficiency(cam.last_levels) < 1e-3
    cy, cx = cam.height // 2, cam.width // 2
    peak_spot = float(frame_peak[cy - 3:cy + 4, cx - 3:cx + 4].mean())
    far_spot = float(frame_far[cy - 3:cy + 4, cx - 3:cx + 4].mean())
    assert peak_spot > far_spot + 0.5 * cam.peak_counts


def test_virtual_mot_camera_senses_the_delayed_coil_timeline():
    """The monitor camera's sense time and decode live on the SAME delayed timeline the hardware
    plays: a moderately delayed coil bus still senses the programmed levels (the mid-cycle point
    sits inside the shifted driven window), and a bus delayed beyond the un-delayed span senses
    the SAFE level (mid of the DELAYED cycle precedes the driven window, where the DA delay
    scheduler idles at the mid code = 0 V) -- never the raw view's set-points.  (Regression: the
    raw-pulse t_end + raw-pulse decode sensed the pre-delay levels no matter what delay the fired
    sequence carried.)"""
    ds = load_devices("virtual", open_devices=False)
    cam, seqr = ds.devices["monitor_camera"], ds.devices["sequencer"]
    state = _mot_state()
    slots = state.scan_names
    at_peak = {name: int(cam.b0[bus]) for name, bus in zip(slots, cam.coil_buses)}
    seq = state.with_slots_resolved(at_peak).to_sequence()

    def with_coil_delay(dt: float):
        out = seq
        for members in cam.coil_buses.values():
            for channel in members:
                out = out.delay(channel, dt)
        return out

    # moderate delay: the mid-cycle sense point sits inside the shifted driven window
    triggered_frames(cam, seqr, with_coil_delay(20e-6), 1)
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}

    # delay beyond the un-delayed span (derived, never a re-typed template length): mid of the
    # DELAYED cycle now precedes the driven window, where the hardware's DA delay scheduler still
    # idles at the SAFE mid code (0 V) -- the camera must sense exactly that, not the raw view's
    # programmed set-points (and never the all-bits-low full-negative word)
    triggered_frames(cam, seqr, with_coil_delay(2.0 * seq.base_duration), 1)
    assert cam.last_levels == {bus: BUS_SAFE_SIGNED_LEVEL for bus in cam.coil_buses}


# --------------------------------------------------------------------------- intensity primitive
def test_mot_roi_intensity_is_disc_minus_annulus():
    """The ONE intensity rule: mean(disc) - mean(annulus r..2r).  A flat background with a
    uniform disc of +A on top reads exactly A (background cancels)."""
    frame = np.full((64, 64), 100.0)
    yy, xx = np.mgrid[0:64, 0:64]
    disc = (xx - 32) ** 2 + (yy - 32) ** 2 <= 8 ** 2
    frame[disc] += 25.0
    assert mot_roi_intensity(frame, 32, 32, 8) == pytest.approx(25.0)
    with pytest.raises(ValueError):
        mot_roi_intensity(frame, 500, 500, 8)          # ROI outside the frame
    with pytest.raises(ValueError):
        mot_roi_intensity(np.zeros((2, 3, 4)), 1, 1, 1)  # not one (H, W) frame


# --------------------------------------------------------------------------- hardware scan grid geometry
def test_hardware_scan_carries_declared_scan_shape():
    """The semantic x/y/z hardware table carries its declared grid shape."""
    from Zou_lab_control import neutral_atom as na
    from Zou_lab_control.neutral_atom.operations.measurement import SWEEP_SCAN_SLOT
    exp = na.connect("virtual")
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        prog = ("import numpy as np\n"
                "a = np.array([0, 4]); b = np.array([-4, 0]); c = np.array([8, 12])\n"
                "A, B, C = np.meshgrid(a, b, c, indexing='ij')\n"
                "scan_table = np.column_stack([A.ravel(), B.ravel(), C.ravel()])\n"
                "scan_shape = (len(a), len(b), len(c))\n")
        plan = spec.build(template=MOT_TEMPLATE,
                          pulse_slots={"api": {}, "sweep_kind": SWEEP_SCAN_SLOT, "program": prog})
        assert plan.sweep_kind == SWEEP_SCAN_SLOT
        assert plan.pulse_state.api_slots == []
        assert plan.scan_shape == (2, 2, 2)
        assert plan.scan_names == ["da_x", "da_y", "da_z"]
        assert not hasattr(plan, "camera")
    finally:
        exp.close()


# --------------------------------------------------------------------------- optimum refinement
def test_refine_optimum_centroid_beats_bare_argmax():
    """The 3^n-neighbourhood centre-of-mass lands closer to an off-grid peak than the argmax
    cell centre; a flat block degrades gracefully to the argmax coordinates."""
    axes = [np.arange(0, 10, 2.0)] * 3                 # grid step 2, peak off-grid
    true = (5.0, 4.6, 3.2)
    grids = np.meshgrid(*axes, indexing="ij")
    block = np.exp(-0.5 * sum((g - t) ** 2 for g, t in zip(grids, true)) / 3.0**2)
    best, peak = OptimizeMotFieldTask.refine_optimum(block, axes)
    idx = np.unravel_index(int(np.argmax(block)), block.shape)
    argmax_coords = [float(ax[i]) for ax, i in zip(axes, idx)]
    for b, a, t in zip(best, argmax_coords, true):
        assert abs(b - t) <= abs(a - t) + 1e-9
    assert peak == pytest.approx(float(block.max()))
    flat_best, _ = OptimizeMotFieldTask.refine_optimum(np.zeros((3, 3, 3)), [np.arange(3.0)] * 3)
    assert flat_best == (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- end-to-end optimum
def test_optimize_mot_field_task_converges_on_the_device_model(tmp_path):
    """The one-click task, fed ONLY frames, lands on the virtual device's own optimum (cam.b0)
    within one grid step per axis -- asserted against the device's public model, never a
    re-typed constant."""
    ds = load_devices("virtual", open_devices=False)
    cam = ds.devices["monitor_camera"]
    b0 = [cam.b0[bus] for bus in cam.coil_buses]       # template bus order == coil_buses order
    hub = SignalHub()
    task = OptimizeMotFieldTask(
        hub, cam, ds.devices["sequencer"], template=MOT_TEMPLATE,
        center_x=b0[0] - 2, center_y=b0[1] + 2, center_z=b0[2] - 2,   # off-centre on purpose
        span=6, points=5, folder=str(tmp_path / "mot_report"))
    task.run_to_completion()
    r = task.result
    step = 2 * 6 / (5 - 1)                             # one grid step (span/points of this run)
    for key, expect in zip(("best_x", "best_y", "best_z"), b0):
        assert abs(r[key] - expect) <= step, (key, r[key], expect)
    assert r["best_intensity"] > 0
    report = Path(r["report_dir"])
    saved = np.load(report / "mot_field_scan.npz")
    assert saved["intensity"].shape == (5, 5, 5)
    assert np.isfinite(saved["intensity"]).all()


def test_report_saves_the_bz_plane_grid_figure(tmp_path):
    """The final report is BOTH the npz block AND the Bz-plane facet-grid figure
    (``mot_field_planes.png``, one (Bx,By) map per Bz plane).  The figure goes through the ONE
    ``frontend.grid`` factory, exposed on the registered plotter; a wrong ``plotter.plot(kind="grid")``
    call used to raise and be SWALLOWED by ``except: pass``, silently dropping the figure.  Importing
    the frontend registers the plotter, so this pins that the report image is actually written."""
    import Zou_lab_control.frontend  # noqa: F401 -- registers the neutral-atom plotter (with .grid)

    ds = load_devices("virtual", open_devices=False)
    cam = ds.devices["monitor_camera"]
    b0 = [cam.b0[bus] for bus in cam.coil_buses]
    task = OptimizeMotFieldTask(
        SignalHub(), cam, ds.devices["sequencer"], template=MOT_TEMPLATE,
        center_x=b0[0], center_y=b0[1], center_z=b0[2], span=6, points=3,
        folder=str(tmp_path / "rep"))
    task.run_to_completion()
    report = Path(task.result["report_dir"])
    assert (report / "mot_field_scan.npz").exists()
    assert (report / "mot_field_planes.png").exists(), "the report Bz-plane grid figure is missing"


# --------------------------------------------------------------------------- LIVE mid-run 3-D grid
def test_task_mid_run_channel_is_the_live_3d_grid(tmp_path):
    """The one-click task's PRIMARY mid-run channel is the accumulating 3-D scan GRID, not just the
    current camera frame: ``grid`` is declared FIRST in ``mid_run`` (so a single mid-run panel shows
    the grid), the task announces its ``grid_shape`` up front (before firing a point, so the console
    can lay out the empty facet grid immediately), and the published (repeat, n_points, 1) block --
    reshaped by that shape -- IS the exact scan the report saves.  This pins that the live panel is
    the real 3-D scan filling in, never a lone 2-D image (the exact thing the operator watched)."""
    ds = load_devices("virtual", open_devices=False)
    cam = ds.devices["monitor_camera"]
    b0 = [cam.b0[bus] for bus in cam.coil_buses]
    task = OptimizeMotFieldTask(
        SignalHub(), cam, ds.devices["sequencer"], template=MOT_TEMPLATE,
        center_x=b0[0], center_y=b0[1], center_z=b0[2], span=6, points=3,
        folder=str(tmp_path / "rep"))
    assert task.mid_run[0] == "grid"                   # grid is the panel's DEFAULT (first) mid-run key
    assert task.grid_shape == (3, 3, 3)                # deterministic from the sweep params, pre-run
    assert len(task.grid_shape) >= 2                   # -> the console facets it (a 1-D shape would not)
    task.run_to_completion()
    block = np.asarray(task.output.latest("grid"), dtype=float)
    assert block.shape == (1, int(np.prod(task.grid_shape)), 1)   # the (repeat, n_points, 1) facet block
    saved = np.load(Path(task.result["report_dir"]) / "mot_field_scan.npz")["intensity"]
    # the LIVE grid IS the report scan (same numbers), reshaped by the DECLARED grid_shape
    np.testing.assert_allclose(block.reshape(task.grid_shape), saved, equal_nan=True)


def test_console_task_panel_is_a_live_facet_grid():
    """Driving the REAL console's task-run path for the one-click MOT task opens its mid-run panel as
    a FACET GRID bound to the reserved ``__task_frame__`` (one (Bx,By) map per Bz plane) -- NOT a lone
    2-D frame -- built through the SAME PanelConfig path a manual panel uses, with every knob from a
    SINGLE source (no magic constants in ``_set_task_running``):

    * size == the ONE ``recommended_grid_size`` rule (a 3-cell grid -> ``2x2``, never a hardcoded 4x4);
    * ``sub_plot_kind`` is NOT hand-set (it auto-derives from the remaining axes);
    * the live grid's cmap == the Setting popup's default == ``PALETTE`` (``_resolved_cmap`` -- render
      and Setting can never diverge into the grey-vs-inferno bug again).
    """
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    from Zou_lab_control.frontend.task_console import (
        TaskConsole, LogicNodeConfig, default_console_state, PANEL_PARAMS, _resolved_cmap)
    from Zou_lab_control.frontend.style import PALETTE
    from Zou_lab_control.frontend.live import recommended_grid_size
    from Zou_lab_control import neutral_atom as na

    ensure_qt_app()
    exp = na.connect("virtual")
    try:
        con = TaskConsole(hub=SignalHub(), state=default_console_state(), session=exp,
                          measurements=exp.readout.measurement_specs(),
                          processors=exp.readout.processor_specs(),
                          tasks=exp.readout.task_specs(), window_px=(1000, 700))
        cam = exp.devices["monitor_camera"]
        b0 = [cam.b0[bus] for bus in cam.coil_buses]
        con._add_logic_node(LogicNodeConfig(
            kind="task", name="Optimize MOT field", title="Optimize MOT field",
            values={"center_x": b0[0], "center_y": b0[1], "center_z": b0[2], "span": 6, "points": 3}))
        row = con.logic_nodes[-1]
        node = con._build_logic_node(row.node, dict(row.node.values))   # build only -- do NOT run the daemon
        con._set_task_running(row, node)                                # the panel-open path
        cfg = con._task_card.config
        assert cfg.kind == "grid"
        assert list(node.grid_shape) == [3, 3, 3]
        assert "points_shape" not in cfg.params       # geometry comes only from TaskOutput's schema
        assert cfg.params["facet"] == f"points:{len(node.grid_shape) - 1}"
        assert "__task_frame__" in cfg.source
        # size from the ONE recommendation rule, not a magic constant
        assert cfg.size == recommended_grid_size(node.grid_shape[-1]) == "2x2"
        # sub_plot_kind auto-derives (it is not hand-set in the task panel config)
        assert "sub_plot_kind" not in cfg.params
        # the live grid render cmap == the Setting default == the ONE PALETTE source
        setting_default = next(d.default for d in PANEL_PARAMS["2d"] if d.key == "cmap")
        assert _resolved_cmap("2d", cfg.params) == setting_default == PALETTE["cmap_scan"]

        # Publish one typed canonical block with a nontrivial validity mask.  The reserved task source
        # carries all three pieces -- data, schema and validity -- through the same generic card path as
        # a Hub signal; faceting the final point axis yields one (Bx,By) image per Bz value.
        from Zou_lab_control.frontend.task_console import SIG_VALID_KEY, TASK_FRAME_KEY
        from Zou_lab_control.neutral_atom.core.signal_tensor import SignalTensor
        output_spec = next(spec for spec in node.output_specs() if spec.name == "grid")
        schema = output_spec.to_schema()
        values = np.arange(np.prod(node.grid_shape), dtype=float).reshape(1, -1, 1)
        valid = np.ones(values.shape[:2], dtype=bool)
        valid[0, -1] = False
        node.output.publish(grid=SignalTensor(values, schema, valid=valid))
        con._refresh_task_panel()

        structure = con._signal_structure(TASK_FRAME_KEY)
        assert structure["points_shape"] == node.grid_shape
        assert structure["data_shape"] == (1,)
        assert structure["grid_shape"] == node.grid_shape
        assert structure["param_names"] == node.scan_names
        namespace = con._expression_namespace()
        np.testing.assert_array_equal(namespace[SIG_VALID_KEY][TASK_FRAME_KEY], valid)
        block = con._task_card._signal_then_repeat(namespace)
        cells = con._task_card._facet_cells(block)
        assert len(cells) == node.grid_shape[-1]
        assert all(cell.shape == node.grid_shape[:2] for cell in cells)
        assert con._task_card.plotter is not None
        con.shutdown()
    finally:
        exp.close()


# --------------------------------------------------------------------------- discovery
def test_registries_discover_the_mot_chain():
    """Console visibility: the task is auto-discovered, while generic Pulse Scan exposes the
    external y selector without owning a camera role.  The bespoke MOT-intensity processor is
    GONE (its live-scalar role is the generic ROI crop chain; the optimiser's math lives on the
    task) -- pinned so a stale copy cannot silently reappear in the catalog."""
    from Zou_lab_control import neutral_atom as na
    exp = na.connect("virtual")
    try:
        assert "Optimize MOT field" in {s.name for s in exp.readout.task_specs()}
        assert "MOT intensity" not in {s.name for s in exp.readout.processor_specs()}
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        assert not any(p.key == "camera" for p in spec.params)
        assert any(p.key == "y" for p in spec.params)
    finally:
        exp.close()


def test_task_requires_exactly_three_dac_slots():
    """A template without the three coil handles fails LOUDLY with guidance (not a silent
    mis-scan of whatever slots happen to exist)."""
    from Zou_lab_control.neutral_atom.timing import single_imaging_template
    ds = load_devices("virtual", open_devices=False)
    task = OptimizeMotFieldTask(SignalHub(), ds.devices["monitor_camera"], ds.devices["sequencer"])
    with pytest.raises(ValueError, match="da_x.*da_y.*da_z"):
        task._coil_slots(single_imaging_template())
