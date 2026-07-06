"""MOT coil-field optimisation chain -- MECHANICAL contracts for every link.

The chain (W round): pulse template with THREE dac api slots -> fired sequence encodes the coil
buses as bit-channel pulses -> the virtual MOT monitor camera SENSES those levels back through
``decode_analog_bus`` (the encoder's exact inverse) -> ``mot_roi_intensity`` reads the MOT spot ->
the manual pulse-scan (api mode, camera=monitor_camera) facets the 3-D block / the
Optimize-MOT-field task finds the optimum.  Each contract here pins one link so the whole chain
can never silently rot:

* decode_analog_bus really is set_api's inverse THROUGH the compiled artefact (this also pins the
  ``_set_api_field`` dac fix: set_api must bake the plan into period states, else the sequence a
  virtual/real machine plays would ignore software-set DAC values entirely), and it reads the
  DELAYED base-cycle timeline -- the one ``edges()`` actually streams -- never the raw pulse list;
* the virtual monitor camera senses ONLY the sequence FIRED over its wired streamer (the
  ``sequencer`` construction parameter -- the simulated trigger cable); free-running (its
  ``Software`` default) it images the held/safe output state, never anyone's set-points;
* the intensity primitive + processor shapes;
* the api sweep carries its declared scan_shape (grid facet parity with the hardware scan);
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
from Zou_lab_control.neutral_atom.operations.measurements.pulse_scan import _resolve_probe_template
from Zou_lab_control.neutral_atom.operations.processors.mot_intensity import (
    MotIntensityProcessor, mot_roi_intensity)
from Zou_lab_control.neutral_atom.operations.measurement import triggered_frames
from Zou_lab_control.neutral_atom.operations.tasks.mot_field import (
    DEFAULT_MOT_TEMPLATE, OptimizeMotFieldTask)
from Zou_lab_control._clock import DEFAULT_CLOCK_HZ
from Zou_lab_control.neutral_atom.timing.sequence import decode_analog_bus

MOT_TEMPLATE = str(REPO_ROOT / DEFAULT_MOT_TEMPLATE)


def _mot_state():
    return _resolve_probe_template(MOT_TEMPLATE)


def _mid_frame_time(sequence) -> float:
    # Mid of the DELAYED base-cycle timeline (base_duration is delay-inclusive) -- the same
    # sense-time rule VirtualMotCamera._render_frames applies, so a .delay() on the coil bit
    # channels moves this sense point with the hardware output.
    return 0.5 * sequence.base_duration


def _edges_level(sequence, members, at_time: float) -> int:
    """Read a bus level straight off the streamed ``edges()`` timeline -- the independent ground
    truth for what the hardware plays at ``at_time`` (decoder and streamer cannot drift)."""
    ticks, masks, channels = sequence.edges()
    tick = at_time * DEFAULT_CLOCK_HZ
    mask = 0
    for t, m in zip(ticks, masks):
        if t > tick:
            break
        mask = m
    # a member with no pulses at all (an un-set bit -- the encoder emits no pulse for it) never
    # appears in the edge channel list: it is low for the whole sequence
    word = sum(1 << bit for bit, ch in enumerate(members)
               if ch in channels and (mask >> channels.index(ch)) & 1)
    return word - (1 << (len(members) - 1))


# --------------------------------------------------------------------------- encode <-> decode
@pytest.mark.parametrize("values", [(7, -5, 11), (0, 0, 0), (-32, 31, 1)])
def test_decode_analog_bus_inverts_set_api_through_compiled_sequence(values):
    """set_api(dac) -> to_sequence -> decode_analog_bus round-trips EXACTLY (incl. the signed
    range edges of the template's 6-bit buses).  This pins BOTH directions: the encoder must bake
    api-set DAC values into the compiled artefact (the _set_api_field fix -- before it, a software
    set_api was silently absent from the sequence), and the decoder must be its exact inverse."""
    state = _mot_state()
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    resolved = state.with_api_resolved(dict(zip(slots, values)))
    sequence = resolved.to_sequence()
    t = _mid_frame_time(sequence)
    decoded = [decode_analog_bus(sequence, members, t)
               for members in state.analog_buses.values()]
    assert decoded == list(values)


def test_decode_analog_bus_reads_the_delayed_timeline():
    """``.delay()`` on the bus bit channels shifts the window the hardware DRIVES (``edges()``
    streams the shifted pulses), so the decoder must read the SAME delayed base-cycle timeline:
    BEFORE the delay the bits are still un-driven (the all-low word), AFTER it the programmed
    level appears -- both cross-checked against ``edges()`` itself.  (Regression: decode used to
    read the RAW pulse list, so with a delayed coil bus it returned the programmed level at a
    time the hardware output had not started driving yet.)"""
    state = _mot_state()
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    values = (7, -5, 11)
    sequence = state.with_api_resolved(dict(zip(slots, values))).to_sequence()
    dt = 20e-6
    for members in state.analog_buses.values():
        for channel in members:
            sequence = sequence.delay(channel, dt)
    t_after = _mid_frame_time(sequence)                # mid of the DELAYED base cycle
    t_before = 0.25 * dt                               # the delayed bits have not started driving
    for members, value in zip(state.analog_buses.values(), values):
        assert decode_analog_bus(sequence, members, t_after) == value \
            == _edges_level(sequence, members, t_after)
        undriven = -(1 << (len(members) - 1))          # all bits low -> the signed minimum word
        assert decode_analog_bus(sequence, members, t_before) == undriven \
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
    slots = [s.name for s in state.api_slots if s.kind == "dac"]

    # Nothing fired: the free-running sensor still delivers a frame (a real Software Basler never
    # freezes), imaging the SAFE state -- every coil DAC parked at signed level 0.
    idle = cam.acquire(1)
    assert len(idle) == 1 and idle[0].shape == tuple(cam.sensor_shape)
    assert cam.last_levels == {bus: 0.0 for bus in cam.coil_buses}

    at_peak = {name: int(cam.b0[bus]) for name, bus in zip(slots, cam.coil_buses)}
    seq = state.with_api_resolved(at_peak).to_sequence()
    frame_peak = triggered_frames(cam, seqr, seq, 1)[0]
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}

    far = {name: int(cam.b0[bus] - 4 * cam.b_sigma[bus]) for name, bus in zip(slots, cam.coil_buses)}
    frame_far = triggered_frames(cam, seqr, state.with_api_resolved(far).to_sequence(), 1)[0]
    # brightness ratio follows the SAME public model the optimum test uses
    assert cam.mot_efficiency(cam.last_levels) < 1e-3
    assert float(frame_peak.mean()) > float(frame_far.mean()) + 10


def test_virtual_mot_camera_senses_the_delayed_coil_timeline():
    """The monitor camera's sense time and decode live on the SAME delayed timeline the hardware
    plays: a moderately delayed coil bus still senses the programmed levels (the mid-cycle point
    sits inside the shifted driven window), and a bus delayed beyond the un-delayed span senses
    the un-driven word (mid of the DELAYED cycle precedes the driven window) -- never the raw
    view's set-points.  (Regression: the raw-pulse t_end + raw-pulse decode sensed the pre-delay
    levels no matter what delay the fired sequence carried.)"""
    ds = load_devices("virtual", open_devices=False)
    cam, seqr = ds.devices["monitor_camera"], ds.devices["sequencer"]
    state = _mot_state()
    slots = [s.name for s in state.api_slots if s.kind == "dac"]
    at_peak = {name: int(cam.b0[bus]) for name, bus in zip(slots, cam.coil_buses)}
    seq = state.with_api_resolved(at_peak).to_sequence()

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
    # DELAYED cycle now precedes the driven window, so the hardware plays nothing there yet --
    # the camera must sense exactly that, not the raw view's programmed set-points
    triggered_frames(cam, seqr, with_coil_delay(2.0 * seq.base_duration), 1)
    assert cam.last_levels == {bus: -(1 << (len(members) - 1))
                               for bus, members in cam.coil_buses.items()}


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


def test_mot_intensity_processor_accepts_bare_and_block_frames():
    """The processor consumes ONE (H, W) frame or the camera's uniform (repeat, 1, H, W) block,
    measures each repeat slice through the same primitive, then reduces (its declared contract)."""
    assert MotIntensityProcessor.repeat_contract == "reduce"
    assert MotIntensityProcessor.provides == ("mot_intensity",)
    hub = SignalHub()
    proc = MotIntensityProcessor(hub, source_expr={"inputs": ["frame_0"], "source": "value = signal"},
                                 roi_radius=6.0)
    frame = np.full((32, 32), 10.0)
    bare = proc.transform({"frame_0": frame})["mot_intensity"]
    block = proc.transform({"frame_0": np.stack([frame, frame])[:, None]})["mot_intensity"]
    assert bare == pytest.approx(0.0)                  # flat frame: disc == annulus
    assert block == pytest.approx(bare)
    with pytest.raises(ValueError):
        proc.transform({"frame_0": np.zeros((2, 3, 4, 5, 6))})


# --------------------------------------------------------------------------- api sweep grid parity
def test_api_sweep_carries_declared_scan_shape():
    """A multi-axis SOFTWARE api sweep declares scan_shape exactly like the hardware table -- the
    x/y/z coil grid facets into the same grid display either way.  (Regression: the api branch
    used to discard the declaration, so a 3-D coil sweep could never facet.)"""
    from Zou_lab_control import neutral_atom as na
    exp = na.connect("virtual")
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        prog = ("import numpy as np\n"
                "a = np.array([0, 4]); b = np.array([-4, 0]); c = np.array([8, 12])\n"
                "A, B, C = np.meshgrid(a, b, c, indexing='ij')\n"
                "scan_table = np.column_stack([A.ravel(), B.ravel(), C.ravel()])\n"
                "scan_shape = (len(a), len(b), len(c))\n")
        plan = spec.build(template=MOT_TEMPLATE, camera="monitor_camera",
                          pulse_slots={"api": {}, "scan_mode": "api", "scan_code": prog})
        assert plan.scan_shape == (2, 2, 2)
        assert plan.api_names == [s.name for s in plan.base_state.api_slots]
        assert plan.camera is exp.devices["monitor_camera"]
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
        assert cfg.params["points_shape"] == list(node.grid_shape)
        assert cfg.params["facet"] == f"points:{len(node.grid_shape) - 1}"
        assert "__task_frame__" in cfg.source
        # size from the ONE recommendation rule, not a magic constant
        assert cfg.size == recommended_grid_size(node.grid_shape[-1]) == "2x2"
        # sub_plot_kind auto-derives (it is not hand-set in the task panel config)
        assert "sub_plot_kind" not in cfg.params
        # the live grid render cmap == the Setting default == the ONE PALETTE source
        setting_default = next(d.default for d in PANEL_PARAMS["2d"] if d.key == "cmap")
        assert _resolved_cmap("2d", cfg.params) == setting_default == PALETTE["cmap_scan"]
        con.shutdown()
    finally:
        exp.close()


# --------------------------------------------------------------------------- discovery
def test_registries_discover_the_mot_chain():
    """Console visibility: the task, the processor and the monitor camera choice are all
    auto-discovered (no hand-wiring anywhere)."""
    from Zou_lab_control import neutral_atom as na
    exp = na.connect("virtual")
    try:
        assert "Optimize MOT field" in {s.name for s in exp.readout.task_specs()}
        assert "MOT intensity" in {s.name for s in exp.readout.processor_specs()}
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        camera_param = next(p for p in spec.params if p.key == "camera")
        assert "monitor_camera" in camera_param.choices
    finally:
        exp.close()


def test_task_requires_exactly_three_dac_slots():
    """A template without the three coil handles fails LOUDLY with guidance (not a silent
    mis-scan of whatever slots happen to exist)."""
    from Zou_lab_control.neutral_atom.timing import single_imaging_template
    ds = load_devices("virtual", open_devices=False)
    task = OptimizeMotFieldTask(SignalHub(), ds.devices["monitor_camera"], ds.devices["sequencer"])
    with pytest.raises(ValueError, match="THREE dac api slots"):
        task._coil_slots(single_imaging_template())
