"""Backend Measurement abstraction + release-recapture thermometry.

Scoped to the new ``operations/measurement.py`` + ``operations/temperature.py``
backend and the ``ReadoutSubsystem.temperature`` entry point.  The virtual e2e
runs the SAME contract path real hardware runs (only the camera frames are fake),
per the AGENTS.md virtual==real rule -- guarded structurally by
``test_virtual_equals_real_contract``.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
import Zou_lab_control.frontend  # noqa: F401  -- self-registers the live-plot viewer
from Zou_lab_control.neutral_atom.core.calibration import TrapCalibration
from Zou_lab_control.neutral_atom.operations.measurement import (
    NFramePlan,
    OtsuFidelityReducer,
    ScanAxis,
    ScanResult,
    ScannedMeasurement,
)
from Zou_lab_control.neutral_atom.operations.temperature import (
    K_B,
    RB87_MASS,
    ReleaseRecapturePlan,
    SurvivalReducer,
    build_release_recapture_pulse,
    fit_temperature,
    release_recapture_survival,
)


def wait_until_done(task, *, timeout=10.0):
    measurement = task.measurement
    assert measurement is not None
    deadline = time.perf_counter() + timeout
    while not measurement.done and time.perf_counter() < deadline:
        time.sleep(0.01)
    measurement.refresh(draw=False)
    if measurement.error is not None:
        raise measurement.error
    assert measurement.done
    return task


# ----------------------------------------------------------- SurvivalReducer unit


def _two_site_calibration():
    """Box calibration with two sites at distinct pixel centers, threshold 50."""

    centers = np.array([[3.0, 3.0], [10.0, 3.0]], dtype=float)
    return TrapCalibration(centers, thresholds=np.array([50.0, 50.0]), roi_radius=1, method="box")


def _frame_with_bright(bright_sites):
    """16x16 frame; chosen site ROIs set bright (above threshold), others dark."""

    img = np.full((16, 16), 5.0, dtype=float)
    centers = {0: (3, 3), 1: (10, 3)}
    for site in bright_sites:
        cx, cy = centers[site]
        img[cy - 1:cy + 2, cx - 1:cx + 2] = 200.0
    return img


def test_survival_reducer_scalar_counts_both_frames_occupied():
    cal = _two_site_calibration()
    reducer = SurvivalReducer(per_site=False)

    # Both sites occupied in frame 1; only site 0 still occupied in frame 2.
    f0 = _frame_with_bright([0, 1])
    f1 = _frame_with_bright([0])
    # survival = (occ1 & occ2).sum / occ1.sum = 1 / 2
    assert reducer.reduce([f0, f1], cal) == pytest.approx(0.5)

    # No survivors -> 0; both survive -> 1.
    assert reducer.reduce([_frame_with_bright([0, 1]), _frame_with_bright([])], cal) == pytest.approx(0.0)
    assert reducer.reduce([_frame_with_bright([0, 1]), _frame_with_bright([0, 1])], cal) == pytest.approx(1.0)
    # Denominator floors at 1 when frame 1 had no atoms (no divide-by-zero).
    assert reducer.reduce([_frame_with_bright([]), _frame_with_bright([])], cal) == pytest.approx(0.0)


def test_survival_reducer_per_site_marks_lost_survived_and_nan():
    cal = _two_site_calibration()
    reducer = SurvivalReducer(per_site=True).bind_calibration(cal)
    assert reducer.n_series == 2

    f0 = _frame_with_bright([0, 1])   # both occupied in frame 1
    f1 = _frame_with_bright([0])      # site 0 survives, site 1 lost
    out = reducer.reduce([f0, f1], cal)
    assert out[0] == pytest.approx(1.0)   # survived
    assert out[1] == pytest.approx(0.0)   # lost

    # A site empty in frame 1 carries no survival information -> nan.
    out2 = reducer.reduce([_frame_with_bright([0]), _frame_with_bright([0])], cal)
    assert out2[0] == pytest.approx(1.0)
    assert np.isnan(out2[1])


def test_survival_reducer_requires_two_frames():
    cal = _two_site_calibration()
    with pytest.raises(ValueError):
        SurvivalReducer().reduce([_frame_with_bright([0])], cal)


# ------------------------------------------------------------ fit_temperature


@pytest.mark.parametrize("T_true", [10e-6, 30e-6, 100e-6])
def test_fit_temperature_round_trips_known_temperature(T_true):
    rc = 1.5e-6
    sigma_v = np.sqrt(K_B * T_true / RB87_MASS)   # derive from the source constants (single source)
    t = np.linspace(0.0, 4.0 * rc / sigma_v, 25)
    survival = release_recapture_survival(t, T_true, capture_radius=rc, baseline=0.95)
    rng = np.random.default_rng(0)
    noisy = np.clip(survival + rng.normal(0.0, 0.01, size=survival.shape), 0.0, 1.0)

    fit = fit_temperature(t, noisy, capture_radius=rc)
    assert fit.success
    assert fit.temperature_K == pytest.approx(T_true, rel=0.1)
    assert fit.baseline == pytest.approx(0.95, abs=0.05)
    assert fit.model.shape == t.shape


def test_fit_temperature_requires_capture_radius_and_enough_points():
    t = np.array([0.0, 1e-5, 2e-5])
    y = np.array([0.9, 0.6, 0.3])
    with pytest.raises(ValueError):
        fit_temperature(t, y, capture_radius=0.0)
    with pytest.raises(ValueError):
        fit_temperature(t[:2], y[:2], capture_radius=1e-6)


# ------------------------------------------------- release-recapture pulse shape


def test_release_recapture_pulse_has_two_triggers_and_bound_t_off():
    state = build_release_recapture_pulse(
        channels=["trap", "probe", "emCCD"], exposure=2e-3, settle=2e-4, recapture=2e-4
    )
    assert state.primary_time_slot() == "s0"
    seq = state.to_sequence(slots=state.reference_slots(), time_step_ns=state.time_step_ns, expand_repeat=False)
    assert na.count_trigger_pulses(seq, trigger_channels=["emCCD"]) == 2


def test_release_recapture_plan_rejects_non_two_frames():
    with pytest.raises(ValueError):
        ReleaseRecapturePlan(n_frames=3)


def test_temperature_spec_is_a_selectable_pulse_scan_special_case():
    """#H3u-4: the Temperature SPEC exposes a SELECTABLE pulse ``template`` (the release-recapture
    program, shipped as pulses/release_recapture.json) -- the pulse-selection step that makes it visibly
    a pulse-scan whose swept axis is the template's trap-off duration slot -- while STAYING the coupled
    tier (``scan_tier='coupled'``, NOT routed through the decoupled PulseScanNode)."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Temperature"]
        decls = {p.key: p for p in spec.params}
        assert "template" in decls and decls["template"].kind == "path"           # a SELECTABLE pulse
        assert decls["template"].default == "pulses/release_recapture.json"        # the shipped default
        # the COUPLED tier marker (single source for docs / boundary test) -- NOT the decoupled node
        assert spec.metadata.get("scan_tier") == "coupled"
        assert spec.metadata.get("node") != "pulse_scan"
        # building from the DEFAULT (selectable) template runs the coupled survival scan end-to-end
        scan = spec.build(template="pulses/release_recapture.json",
                          t_off=(0.0, 40.0, 4), shots=2).run(live=False, display=False)
        assert scan.data_y.shape == (4, 1)
        assert np.all((scan.y >= 0.0) & (scan.y <= 1.0))
    finally:
        exp.close()


def test_temperature_template_rejects_a_missing_file():
    """A bogus template path fails LOUD (the operator picked a file that is not there), never silently
    falling back -- the pulse-selection step is real."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    try:
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Temperature"]
        with pytest.raises(FileNotFoundError):
            spec.build(template="pulses/does_not_exist_rr.json", t_off=(0.0, 40.0, 3), shots=2)
    finally:
        exp.close()


def test_temperature_survival_reaches_zero_at_large_trap_off():
    """#H3v-2: survival DROPS TO ~0 at large trap-off (the atoms physically fly away) -- the model is
    REAL ballistic loss, NOT a faked saturating curve.  The floor the user saw came from imaging the
    survival frames at a different exposure than the thresholds were calibrated at: a threshold is
    exposure-specific, so empty sites crossed a mismatched threshold = a readout false-positive floor
    that never reached 0.  The Temperature SPEC build now images at the SAME exposure the thresholds were
    learnt at (recorded on the calibration metadata), so the readout self-matches and survival -> 0."""
    exp = _calibrated_virtual_session(grid=(5, 7), readout_exposure=20e-3)
    try:
        # the calibration records the exposure its thresholds were learnt at (so the readout self-matches)
        assert exp.readout.current.metadata.get("threshold_exposure") == pytest.approx(20e-3)
        spec = {s.name: s for s in exp.readout.measurement_specs()}["Temperature"]
        scan = spec.build(template="pulses/release_recapture.json",
                          t_off=(0.0, 2000.0, 6), shots=40).run(live=False, display=False)
        y = np.asarray(scan.y)
        assert y[0] >= 0.9                          # nearly all atoms recaptured at t_off = 0
        assert y[-1] <= 0.02                        # survival reaches ~0 at large trap-off (atoms gone)
        assert y[0] - y[-1] >= 0.85                 # a full ballistic fall-off, not a high floor

        # the PHYSICAL model itself drops to 0 occupancy (proves real dynamics, not a faked floor): at a
        # large trap-off EVERY atom ballistically escapes the capture radius before the recapture.
        ta = exp.devices.trap_array
        ta.occupancy = np.ones(ta.n_sites, dtype=bool)
        assert int(ta.apply_recapture_loss(5e-3).sum()) == 0
    finally:
        exp.close()


def test_virtual_parses_trap_off_from_release_recapture_sequence():
    """The virtual data source recovers ``t_off`` from the SAME fired sequence the
    analysis layer built: it is the trap-channel OFF gap between the two camera
    triggers.  Frame 0 has no preceding trap-off; frame 1 carries the bound t_off."""

    from Zou_lab_control.neutral_atom.devices.virtual import trap_off_durations_per_frame
    from Zou_lab_control.neutral_atom.devices import bind_pulse

    state = build_release_recapture_pulse(
        channels=["trap", "probe", "emCCD"], exposure=2e-3, settle=2e-4, recapture=2e-4
    )
    sequencer = na.RuntimeSequencer(channels=["trap", "probe", "emCCD"], trigger_channels=["emCCD"])
    pulse = bind_pulse(sequencer, state)

    for t_off in (0.0, 2e-5, 7.5e-5, 1.5e-4):
        seq = pulse.frame_sequence(2, time_ns=t_off * 1e9)
        per_frame = trap_off_durations_per_frame(seq, 2, trigger_channels=["emCCD"])
        assert per_frame[0] == 0.0                      # nothing precedes frame 1
        # frame 2's preceding trap-off equals the requested t_off (>= 1 tick = 20 ns
        # after clock-grid snapping, so a 0 request reads ~one tick, not exactly 0).
        assert per_frame[1] == pytest.approx(max(t_off, 2e-8), abs=2.1e-8)

    # An ordinary imaging sequence (trap held ON the whole time) reports no trap-off.
    from Zou_lab_control.neutral_atom.timing import imaging_sequence
    img = imaging_sequence(exposure=2e-3, load=True)
    assert trap_off_durations_per_frame(img, 1, trigger_channels=["emCCD"]) == [0.0]


# ----------------------------------------------------- virtual e2e (contract path)


def _calibrated_virtual_session(grid=(3, 4), readout_exposure=2e-3):
    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    # Calibrate the per-site threshold AT the exposure the release-recapture readout
    # uses (a few ms): a threshold is exposure-specific (bright counts scale with the
    # imaging time), so it must be learnt under the same readout conditions it will be
    # applied to -- exactly as on real hardware.
    exp.readout.thresholds(frames=40, exposure=readout_exposure, display=False)
    return exp


def test_temperature_scan_virtual_end_to_end_headless():
    exp = _calibrated_virtual_session()
    state = build_release_recapture_pulse(
        channels=list(exp.devices.sequencer.channels), exposure=2e-3, settle=2e-4, recapture=2e-4
    )
    pulse = na.bind_pulse(exp.devices.sequencer, state)

    t_off = np.array([0.0, 5e-6, 1e-5, 2e-5, 4e-5, 8e-5])
    scan = exp.readout.temperature(t_off, pulse=pulse, shots=3, live=False, display=False)

    # ScanResult contract fields.
    assert isinstance(scan, ScanResult)
    assert isinstance(scan, na.MeasurementTaskResult)
    assert scan.x.shape == t_off.shape
    assert scan.data_y.shape == (len(t_off), 1)
    assert scan.labels[0] == "Trap-off time (s)"
    assert scan.summary()["finished"] is True
    assert scan.points_done == len(t_off)
    # Survival is a fraction in [0, 1] at every point, and high near t_off=0 (atoms
    # are still there).  The virtual data source models ballistic release-recapture
    # loss (see test_temperature_scan_virtual_survival_decays_with_t_off), so survival
    # falls with t_off; here we assert the shape/validity contract this path produces.
    y = scan.y
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert y[0] >= 0.5


def test_temperature_scan_virtual_survival_decays_with_t_off():
    """The virtual backend's release-recapture loss model yields a NON-FLAT survival
    curve: ~1 near t_off=0, falling as t_off grows.  This runs the SAME contract path
    real hardware runs -- ``exp.readout.temperature`` only ever touches the camera
    via ``devices/base.py`` (acquire) and ``calibration.detect``; it reads no
    simulation ground truth.  The decay therefore emerges purely from the fake camera
    FRAMES, exactly as it would from real frames (only who produced the frames differs).

    The fit then recovers a temperature close to the model's truth, confirming the
    data-source physics matches the analysis-layer model without the analysis layer
    knowing the truth."""

    # Read SURVIVAL at a high-fidelity exposure (20 ms) so y reflects the ATOM survival physics, not the
    # readout error: with the v16-aligned photon budget a 3 ms readout is genuinely ~0.85 fidelity, which
    # would floor y[0] below the true 1.0 (a real lab reads survival at high fidelity for the same reason).
    exp = _calibrated_virtual_session(grid=(5, 7), readout_exposure=20e-3)
    # The default loss model (50 uK, 6 um capture radius) has its half-survival near
    # ~75 us, so sweep 0..300 us to see the full fall-off.
    state = build_release_recapture_pulse(
        channels=list(exp.devices.sequencer.channels), exposure=20e-3, settle=2e-4, recapture=2e-4
    )
    pulse = na.bind_pulse(exp.devices.sequencer, state)

    t_off = np.linspace(0.0, 300e-6, 13)
    # Many shots per point so the survival FRACTION is a low-noise estimate of the
    # underlying probability (each shot is a fresh loading on the virtual backend).
    scan = exp.readout.temperature(t_off, pulse=pulse, shots=24, live=False, display=False)
    y = scan.y

    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    # High at t_off=0 (atoms cannot move), low at the longest t_off, big overall drop.
    assert y[0] >= 0.85
    assert y[-1] <= 0.25
    assert y[0] - y[-1] >= 0.5
    # Coarse monotonicity: the first third of the curve sits well above the last third
    # (point-to-point dips from shot noise are allowed; the TREND must be a decay).
    third = max(1, len(y) // 3)
    assert np.mean(y[:third]) > np.mean(y[-third:]) + 0.4

    # The fit recovers a temperature in the right ballpark of the model truth (50 uK),
    # proving the data-source physics and the analysis model agree -- with the analysis
    # layer never reading that truth.
    fit = fit_temperature(t_off, y, capture_radius=6.0e-6)
    assert fit.success
    assert 25e-6 <= fit.temperature_K <= 100e-6


def test_temperature_scan_virtual_per_site_returns_one_column_per_site():
    exp = _calibrated_virtual_session(grid=(2, 3))
    n_sites = exp.devices.trap_array.n_sites
    state = build_release_recapture_pulse(
        channels=list(exp.devices.sequencer.channels), exposure=2e-3, settle=2e-4, recapture=2e-4
    )
    pulse = na.bind_pulse(exp.devices.sequencer, state)

    t_off = np.array([0.0, 1e-5, 2e-5, 4e-5])
    scan = exp.readout.temperature(t_off, pulse=pulse, shots=2, per_site=True, live=False, display=False)
    assert scan.data_y.shape == (len(t_off), n_sites)
    finite = scan.data_y[np.isfinite(scan.data_y)]
    assert finite.size > 0
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_temperature_requires_thresholds():
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3), "image_shape": (48, 64)})
    exp.readout.sitemap(method="box", frames=4, display=False)   # no thresholds yet
    state = build_release_recapture_pulse(channels=list(exp.devices.sequencer.channels))
    pulse = na.bind_pulse(exp.devices.sequencer, state)
    with pytest.raises(RuntimeError):
        exp.readout.temperature([0.0, 1e-5], pulse=pulse, live=False, display=False)


def test_temperature_rejects_pulse_without_t_off_slot():
    exp = _calibrated_virtual_session(grid=(2, 3))
    # An imaging pulse table with no bound duration slot -> no t_off axis.
    state = na.PulseTableState(
        channels=["trap", "probe", "emCCD"],
        periods=[
            na.PulsePeriod(2e6, (1, 1, 1), unit="ns", name="image"),
            na.PulsePeriod(1e5, (1, 0, 0), unit="ns", name="settle"),
        ],
        repeat_forever=False,
    )
    pulse = na.bind_pulse(exp.devices.sequencer, state)
    with pytest.raises(ValueError):
        exp.readout.temperature([0.0, 1e-5], pulse=pulse, live=False, display=False)


def test_temperature_scan_virtual_live_path_uses_frontend_session():
    exp = _calibrated_virtual_session(grid=(2, 3))
    state = build_release_recapture_pulse(
        channels=list(exp.devices.sequencer.channels), exposure=2e-3, settle=2e-4, recapture=2e-4
    )
    pulse = na.bind_pulse(exp.devices.sequencer, state)

    t_off = np.array([0.0, 1e-5, 2e-5, 4e-5])
    scan = exp.readout.temperature(t_off, pulse=pulse, shots=2, live=True, display=False, update_time=0.01)
    assert scan.measurement is not None
    assert scan.plot is scan.measurement.plot
    wait_until_done(scan)
    assert np.all(np.isfinite(scan.y))
    assert scan.summary()["finished"] is True


# ------------------------------------------------- generic ScannedMeasurement reuse


def test_generic_scanned_measurement_drives_detection_axis_directly():
    """The detection-time scan is just a ScannedMeasurement (the abstraction is
    not temperature-specific): build one directly with a fidelity reducer."""

    exp = na.connect("virtual", sitemap={"grid_shape": (1, 2), "image_shape": (32, 48)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    cal = exp.readout.current

    hardware_channels = [f"ch{i:02d}" for i in range(8)]
    sequencer = na.RuntimeSequencer(channels=hardware_channels, clock_hz=100_000_000, trigger_channels=["ch03"])
    state = na.PulseTableState(
        channels=["ch00", "ch03"],
        periods=[
            na.PulsePeriod("s0", (1, 1), unit="str (ns)"),
            na.PulsePeriod(100, (0, 0), unit="ns"),
        ],
        scan_slots=[{"kind": "duration", "target": "0", "unit": "ns", "nominal": 1000.0}],
        time_step_ns=10,
        repeat_forever=True,
    )
    pulse = na.bind_pulse(sequencer, state)

    times = np.array([2e-6, 4e-6, 8e-6])
    axis = ScanAxis(slot="s0", values=times, label="Detection time (s)", unit="s", kind="duration")
    scan = ScannedMeasurement(
        pulse, exp.devices.camera, sequencer, cal, axis,
        NFramePlan(n_frames=4), OtsuFidelityReducer(site=None), shots_per_point=1,
    )
    result = scan.run(live=False, display=False)
    assert result.data_y.shape == (3, 1)
    assert np.all(np.isfinite(result.y))
    assert np.all((result.y >= 0.0) & (result.y <= 1.0))


def test_readout_duration_fidelity_is_detection_time_alias():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)
    scan = exp.readout.readout_duration_fidelity([5e-6, 2e-5], shots=4, live=False, display=False)
    assert isinstance(scan, na.DetectionTimeScanResult)
    assert scan.summary()["finished"] is True


def test_duration_axis_is_snapped_to_clock_grid():
    """A continuous / off-grid duration sweep is quantized to whole clock ticks at the
    scan engine, so the plotted x equals the exposures actually run and no built sequence
    trips the clock-grid validator (the "repeat period not on the 5e7 clock grid" failure).
    The engine is the SINGLE snap point shared by readout-fidelity and any duration scan."""

    exp = na.connect("virtual", sitemap={"grid_shape": (1, 2), "image_shape": (32, 48)})
    exp.readout.sitemap(method="box", frames=4, display=False)
    clock = float(exp.devices.sequencer.clock_hz)        # 50 MHz -> 20 ns tick

    # Deliberately off-grid times: a linspace whose points are not whole ticks.
    times = np.linspace(2.0e-6, 5000.0e-6, 12)
    raw_ticks = times * clock
    assert np.any(np.abs(raw_ticks - np.round(raw_ticks)) > 1e-6), "fixture must be off-grid"

    scan = exp.readout.build_detection_scan(times, shots=4)
    snapped_ticks = np.asarray(scan.axis.values) * clock
    # every swept value now lands on a whole tick ...
    assert np.allclose(snapped_ticks, np.round(snapped_ticks), atol=1e-6)
    # ... and stays within one tick of the requested value (an honest, minimal nudge).
    assert np.all(np.abs(np.asarray(scan.axis.values) - times) <= 1.0 / clock + 1e-15)
    # ... and the scan runs end-to-end without the validator rejecting an off-grid period.
    result = scan.run(live=False, display=False)
    assert result.summary()["finished"] is True
