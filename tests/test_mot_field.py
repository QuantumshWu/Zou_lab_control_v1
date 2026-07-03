"""MOT coil-field optimisation chain -- MECHANICAL contracts for every link.

The chain (W round): pulse template with THREE dac api slots -> fired sequence encodes the coil
buses as bit-channel pulses -> the virtual MOT monitor camera SENSES those levels back through
``decode_analog_bus`` (the encoder's exact inverse) -> ``mot_roi_intensity`` reads the MOT spot ->
the manual pulse-scan (api mode, camera=monitor_camera) facets the 3-D block / the
Optimize-MOT-field task finds the optimum.  Each contract here pins one link so the whole chain
can never silently rot:

* decode_analog_bus really is set_api's inverse THROUGH the compiled artefact (this also pins the
  ``_set_api_field`` dac fix: set_api must bake the plan into period states, else the sequence a
  virtual/real machine plays would ignore software-set DAC values entirely);
* the virtual monitor camera is a PURE triggered grabber sensing only the sequence FIRED over
  its trigger wire (the ``sequencer`` construction parameter -- the simulated trigger cable);
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
from Zou_lab_control.neutral_atom.timing.sequence import decode_analog_bus

MOT_TEMPLATE = str(REPO_ROOT / DEFAULT_MOT_TEMPLATE)


def _mot_state():
    return _resolve_probe_template(MOT_TEMPLATE)


def _mid_frame_time(sequence) -> float:
    t_end = max(p.start + p.duration for p in sequence.pulses)
    return 0.5 * t_end


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


# --------------------------------------------------------------------------- virtual sensing
def test_virtual_mot_camera_senses_only_the_fired_sequence():
    """The virtual monitor camera is a PURE triggered grabber: nothing fired -> no frame; a
    sequence FIRED on its wired streamer reconstructs each coil level from the compiled
    bit-channel pulses (never from anyone's set-points), and the frame brightness follows its
    own public MOT model."""
    ds = load_devices("virtual", open_devices=False)
    cam, seqr = ds.devices["monitor_camera"], ds.devices["sequencer"]
    state = _mot_state()
    slots = [s.name for s in state.api_slots if s.kind == "dac"]

    assert cam.acquire(1) == []                        # nothing fired -> no trigger -> no frame

    at_peak = {name: int(cam.b0[bus]) for name, bus in zip(slots, cam.coil_buses)}
    seq = state.with_api_resolved(at_peak).to_sequence()
    frame_peak = triggered_frames(cam, seqr, seq, 1)[0]
    assert cam.last_levels == {bus: int(cam.b0[bus]) for bus in cam.coil_buses}

    far = {name: int(cam.b0[bus] - 4 * cam.b_sigma[bus]) for name, bus in zip(slots, cam.coil_buses)}
    frame_far = triggered_frames(cam, seqr, state.with_api_resolved(far).to_sequence(), 1)[0]
    # brightness ratio follows the SAME public model the optimum test uses
    assert cam.mot_efficiency(cam.last_levels) < 1e-3
    assert float(frame_peak.mean()) > float(frame_far.mean()) + 10


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
