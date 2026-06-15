"""Declarative measurement specs + ScannedMeasurementFeed (P5 backend wiring).

Scoped to the new ``ParamDecl``/``MeasurementSpec`` descriptors, the
``ScannedMeasurementFeed`` console adapter, and ``ReadoutSubsystem``'s builders +
``measurement_specs``.  The feed is exercised on the virtual backend end-to-end:
it must run the SAME contract path real hardware runs (only the camera frames are
fake) -- guarded structurally by ``test_virtual_equals_real_contract`` -- and
publish a survival curve that DECAYS with trap-off time (the virtual loss model
is on).
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
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.operations.measurement import (
    MeasurementSpec,
    ParamDecl,
    ScannedMeasurement,
)
from Zou_lab_control.neutral_atom.operations.feeds import ScannedMeasurementFeed
from Zou_lab_control.neutral_atom.operations.temperature import (
    build_release_recapture_pulse,
    fit_temperature,
)


# ------------------------------------------------------------ descriptor unit


def test_param_decl_validates_kind_and_normalizes():
    p = ParamDecl("t_off", "Trap-off", "AXIS_RANGE", default=(0.0, 300.0, 13), unit="us")
    assert p.kind == "axis_range"           # normalized to lower-case
    assert p.key == "t_off"
    assert p.choices == ()                   # coerced to a tuple
    with pytest.raises(ValueError):
        ParamDecl("bad", "Bad", "nonsense", default=0)


def test_measurement_spec_lookup_and_defaults():
    decls = (
        ParamDecl("a", "A", "float", default=1.0),
        ParamDecl("b", "B", "int", default=3),
    )
    spec = MeasurementSpec(
        name="demo", params=decls, result_labels=("x", "y"),
        x_key="xx", y_key="yy", build=lambda **kw: None,
    )
    assert spec.param("a").label == "A"
    assert spec.defaults() == {"a": 1.0, "b": 3}
    with pytest.raises(KeyError):
        spec.param("missing")


# ---------------------------------------------------------- spec.build contract


def _calibrated_virtual_session(grid=(2, 3)):
    exp = na.connect("virtual", sitemap={"grid_shape": grid, "image_shape": (64, 80)})
    exp.readout.sitemap(method="box", frames=6, display=False)
    exp.readout.thresholds(frames=40, display=False)
    return exp


def test_builtin_specs_build_unrun_scanned_measurements():
    exp = _calibrated_virtual_session()
    specs = exp.readout.measurement_specs()
    assert [s.name for s in specs] == [
        "Temperature (release-recapture)",
        "Readout duration -> fidelity",
    ]
    temp, dur = specs
    assert temp.x_key == "rr_t_off" and temp.y_key == "rr_survival"
    assert temp.result_labels == ("Trap-off time (s)", "Survival")
    # capture_radius -> metres conversion for fit_temperature is carried in metadata.
    assert temp.metadata["fit"] == "fit_temperature"
    assert temp.metadata["fit_param"] == "capture_radius"
    assert temp.metadata["fit_param_scale"] == pytest.approx(1e-6)
    assert dur.x_key == "dur_detection_time" and dur.y_key == "dur_fidelity"
    assert dur.result_labels == ("Detection time (s)", "Fidelity")

    m_temp = temp.build(**temp.defaults())
    assert isinstance(m_temp, ScannedMeasurement)
    assert m_temp.axis.values.size == 13          # default points
    m_dur = dur.build(**dur.defaults())
    assert isinstance(m_dur, ScannedMeasurement)
    assert m_dur.axis.values.size == 11


# ------------------------------------------------- ScannedMeasurementFeed (sync)


def _temperature_feed(exp, hub, *, points, shots, t_max_us=300.0):
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, t_max_us, points), shots=shots, capture_radius=6.0)
    return ScannedMeasurementFeed(
        hub, measurement, x_key=spec.x_key, y_key=spec.y_key, grid_shape=spec.grid_shape,
    ), spec


def test_temperature_feed_run_to_completion_publishes_full_decaying_curve():
    exp = _calibrated_virtual_session(grid=(5, 7))
    hub = SignalHub()
    feed, spec = _temperature_feed(exp, hub, points=13, shots=24)

    feed.run_to_completion()

    assert feed.finished
    x = hub.latest(spec.x_key)
    y = hub.latest(spec.y_key)
    # Cumulative curve has exactly ``points`` entries when complete.
    assert x.shape == (13,)
    assert y.shape == (13,)
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    # The virtual loss model is on: survival is high near t_off=0, low at the end,
    # with a clear overall decay (this is the SAME contract path real hardware runs).
    assert y[0] >= 0.85
    assert y[-1] <= 0.25
    third = max(1, len(y) // 3)
    assert np.mean(y[:third]) > np.mean(y[-third:]) + 0.4
    # scan_done flips to 1 on the final publish.
    assert float(hub.latest("scan_done")) == pytest.approx(1.0)

    # The fit recovers a temperature in the model's ballpark (50 uK truth), using the
    # capture-radius conversion the spec metadata declares (um -> m).
    scale = spec.metadata["fit_param_scale"]
    fit = fit_temperature(x, y, capture_radius=6.0 * scale)
    assert fit.success
    assert 25e-6 <= fit.temperature_K <= 100e-6


def test_temperature_feed_per_site_publishes_latest_site_vector_and_grid():
    exp = _calibrated_virtual_session(grid=(2, 3))
    n_sites = exp.devices.trap_array.n_sites
    hub = SignalHub()
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, 80.0, 4), shots=2, capture_radius=6.0, per_site=True)
    feed = ScannedMeasurementFeed(
        hub, measurement, x_key=spec.x_key, y_key=spec.y_key, grid_shape=spec.grid_shape,
    )
    feed.run_to_completion()

    sites = hub.latest(spec.y_key + "_sites")
    grid = hub.latest(spec.y_key + "_grid")
    assert sites.shape == (n_sites,)
    assert grid.shape == spec.grid_shape
    finite = sites[np.isfinite(sites)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_readout_duration_feed_runs_out_a_fidelity_curve():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)
    hub = SignalHub()
    spec = exp.readout.measurement_specs()[1]
    measurement = spec.build(duration=(5.0, 50.0, 4), shots=6)
    feed = ScannedMeasurementFeed(hub, measurement, x_key=spec.x_key, y_key=spec.y_key)

    feed.run_to_completion()

    x = hub.latest(spec.x_key)
    y = hub.latest(spec.y_key)
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))


# ------------------------------------------------ ScannedMeasurementFeed (thread)


def test_temperature_feed_start_thread_auto_stops_when_scan_completes():
    exp = _calibrated_virtual_session(grid=(2, 3))
    hub = SignalHub()
    feed, spec = _temperature_feed(exp, hub, points=4, shots=2, t_max_us=80.0)

    feed.start(rate_hz=50.0)
    deadline = time.perf_counter() + 10.0
    while feed.running and time.perf_counter() < deadline:
        time.sleep(0.02)

    assert feed.finished
    assert feed.running is False        # finite scan stopped its own daemon thread
    y = hub.latest(spec.y_key)
    assert y.shape == (4,)
    assert np.all(np.isfinite(y))


# ------------------------------------------------ acquisition-parameter protocol


def test_camera_frame_feed_exposes_camera_params_and_applies_them_live():
    """A panel is a VIEW; the feed's SOURCE owns the editable params.  A raw-frame
    feed's source is the camera, so acquisition_parameters() reports the camera's
    exposure and set_acquisition_parameters() reconfigures the camera in place --
    no rebuild, same feed keeps publishing 'frame'."""
    exp = na.connect("virtual")
    cam = exp.devices.camera
    hub = SignalHub()
    feed = na.CameraFrameFeed(hub, cam)

    assert feed.published_signals() == frozenset({"frame"})
    params = feed.acquisition_parameters()
    assert "exposure" in params and params["exposure"] == float(cam.exposure)

    feed.set_acquisition_parameters(exposure=0.05)
    assert float(cam.exposure) == 0.05                      # applied to the camera in place
    assert feed.acquisition_parameters()["exposure"] == 0.05

    feed.step()
    frame = hub.latest("frame")
    assert np.ndim(frame) == 2                              # a real 2-D camera frame


def test_loading_feed_acquisition_params_reapply_in_place():
    """LoadingFeed's source is its own analysis; editing a param re-calibrates the
    SAME running feed (no instance swap)."""
    exp = na.connect("virtual", sitemap={"grid_shape": (5, 7)})
    hub = SignalHub()
    feed = na.LoadingFeed(hub, exp.devices.camera, sequencer=exp.devices.sequencer, grid_shape=(5, 7))
    params = feed.acquisition_parameters()
    for key in ("exposure", "roi_radius", "grid_shape", "ema", "calibration_frames", "threshold_frames"):
        assert key in params

    same = feed
    feed.set_acquisition_parameters(roi_radius=2)
    assert feed is same                                     # in place, not a new instance
    assert feed.roi_radius == 2
    assert feed.acquisition_parameters()["roi_radius"] == 2
    assert len(feed.centers) > 0 and feed.thresholds.shape[0] == len(feed.centers)  # re-calibrated
