"""Declarative measurement specs + ScannedMeasurementNode (P5 backend wiring).

Scoped to the new ``ParamDecl``/``MeasurementSpec`` descriptors, the
``ScannedMeasurementNode`` console adapter, and ``ReadoutSubsystem``'s builders +
``measurement_specs``.  The node is exercised on the virtual backend end-to-end:
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
from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode
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
    by_name = {s.name: s for s in specs}
    # The two purpose-built SCANNED readout measurements are auto-discovered alongside the
    # generic "Pulse scan"; select them by name rather than position so adding a measurement
    # never silently breaks this (the registry order/count is not the invariant here).
    assert {"Temperature", "Fidelity vs duration"} <= set(by_name)
    temp = by_name["Temperature"]
    dur = by_name["Fidelity vs duration"]
    # x_key/y_key are the BARE quantity tokens; the slug (key) is the single source the
    # node prefixes every signal with -> temperature_t_off / temperature_survival.
    assert temp.key == "temperature"
    assert temp.x_key == "t_off" and temp.y_key == "survival"
    assert temp.result_labels == ("Trap-off time", "Survival")
    # capture_radius -> metres conversion for fit_temperature is carried in metadata.
    assert temp.metadata["fit"] == "fit_temperature"
    assert temp.metadata["fit_param"] == "capture_radius"
    assert temp.metadata["fit_param_scale"] == pytest.approx(1e-6)
    assert dur.key == "readout"
    assert dur.x_key == "detection_time" and dur.y_key == "fidelity"
    assert dur.result_labels == ("Detection time", "Fidelity")

    m_temp = temp.build(**temp.defaults())
    assert isinstance(m_temp, ScannedMeasurement)
    assert m_temp.axis.values.size == 13          # default points
    m_dur = dur.build(**dur.defaults())
    assert isinstance(m_dur, ScannedMeasurement)
    assert m_dur.axis.values.size == 11


# ------------------------------------------------- ScannedMeasurementNode (sync)


def _temperature_node(exp, hub, *, points, shots, t_max_us=300.0):
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, t_max_us, points), shots=shots, capture_radius=6.0)
    return ScannedMeasurementNode(
        hub, measurement, x_key=spec.x_key, y_key=spec.y_key,
    ), spec


def test_temperature_node_run_to_completion_publishes_full_decaying_curve():
    exp = _calibrated_virtual_session(grid=(5, 7))
    hub = SignalHub()
    node, spec = _temperature_node(exp, hub, points=13, shots=24)

    node.run_to_completion()

    assert node.finished
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


def test_temperature_node_per_site_publishes_latest_site_vector_only():
    """A per-site scan publishes ONLY the flat per-site vector (``<y>_sites``); a site-grid
    view is a reshape EXPRESSION on it, not a separate published signal (no ``<y>_grid``)."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    n_sites = exp.devices.trap_array.n_sites
    hub = SignalHub()
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, 80.0, 4), shots=2, capture_radius=6.0, per_site=True)
    node = ScannedMeasurementNode(hub, measurement, x_key=spec.x_key, y_key=spec.y_key)
    node.run_to_completion()

    sites = hub.latest(spec.y_key + "_sites")
    assert sites.shape == (n_sites,)
    finite = sites[np.isfinite(sites)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    # No redundant signals: the grid + per-node shot counter are NOT published (the grid is
    # a reshape expression; the panel namespace already carries a global ``shot``).
    published = node.published_signals()
    assert (spec.y_key + "_grid") not in published
    assert "shot" not in published and (spec.y_key + "_shot") not in published
    # ... but the per-site vector reshapes cleanly to the grid via an expression, using
    # the trap array's grid shape (the source of truth -- not a stored spec field).
    grid = exp.devices.trap_array.grid_shape
    assert sites.reshape(grid).shape == grid


def test_readout_duration_node_runs_out_a_fidelity_curve():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)
    hub = SignalHub()
    spec = exp.readout.measurement_specs()[1]
    measurement = spec.build(duration=(5.0, 50.0, 4), shots=6)
    node = ScannedMeasurementNode(hub, measurement, x_key=spec.x_key, y_key=spec.y_key)

    node.run_to_completion()

    x = hub.latest(spec.x_key)
    y = hub.latest(spec.y_key)
    assert x.shape == (4,)
    assert y.shape == (4,)
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))


# ------------------------------------------------ ScannedMeasurementNode (thread)


def test_temperature_node_start_thread_auto_stops_when_scan_completes():
    exp = _calibrated_virtual_session(grid=(2, 3))
    hub = SignalHub()
    node, spec = _temperature_node(exp, hub, points=4, shots=2, t_max_us=80.0)

    node.start(rate_hz=50.0)
    deadline = time.perf_counter() + 10.0
    while node.running and time.perf_counter() < deadline:
        time.sleep(0.02)

    assert node.finished
    assert node.running is False        # finite scan stopped its own daemon thread
    y = hub.latest(spec.y_key)
    assert y.shape == (4,)
    assert np.all(np.isfinite(y))


# ------------------------------------------------ ROI: plot-coord -> device grid


def test_snap_subarray_rounds_to_grid_and_clamps():
    """The single source of truth for the plot-coord -> device sub-array adaptation:
    a requested (x, w, y, h) is rounded to the camera's step grid (the qCMOS step
    is 4) and clamped inside the sensor, so a raw plot selection can never ask for
    an illegal window the hardware would silently clamp."""
    from Zou_lab_control.neutral_atom.devices.base import snap_subarray
    assert snap_subarray([10, 20, 6, 16], step=4, max_w=80, max_h=64) == (8, 20, 8, 16)
    assert snap_subarray([1670, 20, 1166, 20], step=4, max_w=2304, max_h=2304) == (1672, 20, 1168, 20)
    # oversize window clamps so x+w <= max_w and y+h <= max_h, staying on the grid
    x, w, y, h = snap_subarray([70, 40, 60, 40], step=4, max_w=80, max_h=64)
    assert (x, w, y, h) == (40, 40, 24, 40)
    assert x + w <= 80 and y + h <= 64 and all(v % 4 == 0 for v in (x, w, y, h))


def test_virtual_camera_honors_roi_snaps_and_crops():
    """The virtual camera mirrors the real sub-array: configure(roi=...) snaps the
    request to the step grid, reports the ACTUALLY-applied window via .roi, and
    acquire() CROPS the frame to it -- so the virtual path exercises the SAME ROI
    contract a real qCMOS does (a ROI bug shows up in a virtual test; switching to
    real changes only connect())."""
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3), "image_shape": (64, 80)})
    cam = exp.devices.camera
    assert cam.roi is None                                  # full frame by default (unchanged)
    cam.configure(roi=[10, 20, 6, 16])                      # x=10, y=6 are NOT multiples of 4
    assert cam.roi == (8, 20, 8, 16)                        # snapped to the grid, reported back
    frame = cam.acquire(1)[-1]
    assert frame.shape == (16, 20)                          # actually CROPPED to (h, w) of the ROI


def test_qcmos_subarray_snaps_writes_and_reads_back():
    """The real qCMOS adapter snaps the requested ROI to the hardware grid (queried
    via prop_getattr step/max), writes the sub-array in the safe order, and reports
    the value the camera ACTUALLY applied (prop_setgetvalue read-back) via .roi --
    so a raw, non-aligned plot selection images the snapped region, not whatever the
    hardware silently clamps to.  Only the lowest layer (the DCAM device) is faked."""
    import sys
    import types
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera, QCMOSConfig

    ids = ["EXPOSURETIME", "TRIGGERSOURCE", "TRIGGERACTIVE", "TRIGGERPOLARITY", "READOUTSPEED",
           "SENSORMODE", "TRIGGER_GLOBALEXPOSURE", "SUBARRAYMODE",
           "SUBARRAYHSIZE", "SUBARRAYHPOS", "SUBARRAYVSIZE", "SUBARRAYVPOS"]
    IDPROP = types.SimpleNamespace(**{name: i for i, name in enumerate(ids)})
    SUBARRAY_SIZE_IDS = {IDPROP.SUBARRAYHSIZE, IDPROP.SUBARRAYVSIZE}
    SUBARRAY_POS_IDS = {IDPROP.SUBARRAYHPOS, IDPROP.SUBARRAYVPOS}
    DCAMPROP = types.SimpleNamespace(
        MODE=types.SimpleNamespace(ON=2, OFF=1),
        TRIGGERSOURCE=types.SimpleNamespace(EXTERNAL=2),
        TRIGGERACTIVE=types.SimpleNamespace(EDGE=1),
        TRIGGERPOLARITY=types.SimpleNamespace(POSITIVE=2),
    )

    class _Attr:
        valuemin, valuemax, valuestep = 0.0, 2304.0, 4.0   # sensor 2304, step 4 (qCMOS)

    class _Err:
        def is_timeout(self):
            return False

    class _FakeDcam:
        def __init__(self, _index=0):
            self._store = {}

        def dev_open(self):
            return True

        def lasterr(self):
            return _Err()

        def prop_setvalue(self, idprop, value):
            self._store[idprop] = value
            return True

        def prop_getvalue(self, idprop):
            return self._store.get(idprop, 0.0)

        def prop_getattr(self, idprop):
            return _Attr()

        def prop_setgetvalue(self, idprop, value, option=0):
            # faithful hardware: snap sub-array writes to the step-4 grid
            if idprop in SUBARRAY_SIZE_IDS or idprop in SUBARRAY_POS_IDS:
                value = int(round(value / 4)) * 4
            self._store[idprop] = value
            return float(value)

        def dev_close(self):
            return None

    fake = types.ModuleType("zlc_fake_dcam")
    fake.Dcamapi = types.SimpleNamespace(init=lambda: True, uninit=lambda: None)
    fake.Dcam = _FakeDcam
    fake.DCAM_IDPROP = IDPROP
    fake.DCAMPROP = DCAMPROP
    sys.modules["zlc_fake_dcam"] = fake
    try:
        cam = QCMOSCamera(QCMOSConfig(exposure=0.02, roi=[1670, 20, 1166, 20]),
                          dcam_module="zlc_fake_dcam")
        cam.open()
        # requested x=1670, y=1166 are NOT multiples of 4 -> snapped + read back
        assert cam.roi == (1672, 20, 1168, 20)
        # a runtime change to another non-aligned window re-snaps + re-reports
        cam.configure(roi=[101, 33, 201, 17])
        assert cam.roi == (100, 32, 200, 16)
    finally:
        sys.modules.pop("zlc_fake_dcam", None)


def test_region_to_acquisition_parameters_is_owned_by_the_source():
    """The plot selector is a GENERIC interface -- it yields a rectangle as four
    endpoints (x_min, x_max, y_min, y_max) in plot coords and knows nothing about
    cameras.  Each SOURCE converts that rectangle to its own ACQUISITION params,
    which stay in PLOT coordinates: a camera measurement keeps it as ``region`` endpoints
    (the device-ROI conversion is hidden in set_acquisition_parameters); a source
    with no spatial region returns {} (the selection is a no-op for it).  So the
    frontend never encodes a device-specific shape."""
    exp = na.connect("virtual", sitemap={"grid_shape": (2, 3), "image_shape": (64, 80)})
    hub = SignalHub()
    cam_node = na.CameraMeasurement(hub, exp.devices.camera)
    # endpoints stay endpoints (NOT collapsed to position+size) -- plot format
    assert cam_node.region_to_acquisition_parameters(10, 30, 6, 22) == {"region": [10, 30, 6, 22]}
    # endpoints come in any order -> sorted endpoints
    assert cam_node.region_to_acquisition_parameters(30, 10, 22, 6) == {"region": [10, 30, 6, 22]}
    # the round trip through the camera: set region endpoints -> device ROI -> read
    # back as endpoints (full sensor 64x80, no snap needed for /4-aligned values)
    cam_node.set_acquisition_parameters(region=[8, 28, 12, 28])
    assert cam_node.acquisition_parameters()["region"] == [8, 28, 12, 28]
    # a non-spatial node (a Processor / scan measurement) has no rectangle param
    from Zou_lab_control.neutral_atom.operations.logic import OccupancyProcessor
    assert OccupancyProcessor(hub, calibration=None).region_to_acquisition_parameters(10, 30, 6, 22) == {}


# ------------------------------------------------ acquisition-parameter protocol


def test_camera_measurement_exposes_camera_params_and_applies_them_live():
    """A panel is a VIEW; the node's SOURCE owns the editable params.  A raw-frame
    camera measurement's source is the camera, so acquisition_parameters() reports the
    camera's exposure and set_acquisition_parameters() reconfigures the camera in place
    -- no rebuild, same measurement keeps publishing 'frame'."""
    exp = na.connect("virtual")
    cam = exp.devices.camera
    hub = SignalHub()
    cam_node = na.CameraMeasurement(hub, cam)

    # default frames_per_cycle=1 -> the first trigger as both 'frame' (back-compat /
    # default 2D panel) and 'frame_0' (the per-trigger name).
    assert cam_node.published_signals() == frozenset({"frame", "frame_0"})
    params = cam_node.acquisition_parameters()
    assert "exposure" in params and params["exposure"] == float(cam.exposure)
    assert params["frames_per_cycle"] == 1

    cam_node.set_acquisition_parameters(exposure=0.05)
    assert float(cam.exposure) == 0.05                      # applied to the camera in place
    assert cam_node.acquisition_parameters()["exposure"] == 0.05

    cam_node.step()
    frame = hub.latest("frame")
    assert np.ndim(frame) == 2                              # a real 2-D camera frame


def test_running_node_applies_params_in_owner_thread_no_concurrent_acquire():
    """ARCHITECTURE invariant: while a node's acquisition loop runs, it is the SOLE
    owner of the source.  An edit from another thread goes through
    ``apply_acquisition_parameters``, which QUEUES it; the loop applies it BETWEEN
    shots in its own thread.  So (1) the source is reconfigured with NO second
    ``acquire()`` ever running on it concurrently -- the deadlock/freeze that a
    GUI-thread stop/start would cause -- (2) the node is NOT restarted (same
    thread keeps streaming), and (3) the change lands within ~1 shot.

    Faked at the lowest level only: a camera whose ``acquire`` BLOCKS for the
    exposure (like a real qCMOS) and records the max concurrent acquire depth."""
    import threading
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice

    class _BlockingCam(CameraDevice):
        def __init__(self):
            self._exposure = 0.05
            self._roi = (1648, 64, 1144, 64)
            self._depth = 0
            self.max_depth = 0
            self._lock = threading.Lock()

        @property
        def exposure(self):
            return self._exposure

        @property
        def roi(self):
            return self._roi

        def configure(self, *, exposure=None, **kw):
            if exposure is not None:
                self._exposure = float(exposure)
            if kw.get("roi") is not None:
                self._roi = tuple(int(v) for v in kw["roi"])

        def acquire(self, frames=1, *, sequence=None, sequencer=None, stop=None, **kw):
            with self._lock:
                self._depth += 1
                self.max_depth = max(self.max_depth, self._depth)
            try:
                deadline = time.monotonic() + self._exposure
                while time.monotonic() < deadline:
                    if stop is not None and stop.is_set():
                        return [None]
                    time.sleep(0.005)
                w, h = self._roi[1], self._roi[3]
                return [np.zeros((h, w), dtype=float)]
            finally:
                with self._lock:
                    self._depth -= 1

    hub = SignalHub()
    cam = _BlockingCam()
    cam_node = na.CameraMeasurement(hub, cam)
    cam_node.start(rate_hz=40.0)
    try:
        deadline = time.monotonic() + 2.0
        while cam_node.shots < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
        thread_before = cam_node._thread
        # an edit from THIS (non-owner) thread -- must be queued, not applied here.
        # region endpoints [1664,1696,1160,1192] -> internal device ROI (1664,32,1160,32)
        cam_node.apply_acquisition_parameters(region=[1664, 1696, 1160, 1192])
        deadline = time.monotonic() + 2.0
        while cam.roi != (1664, 32, 1160, 32) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert cam.roi == (1664, 32, 1160, 32)           # applied by the loop
        assert cam_node.running                           # still streaming
        assert cam_node._thread is thread_before         # SAME thread -- no restart
        assert cam.max_depth == 1                         # never two acquire() at once
    finally:
        cam_node.stop()


# The old monolithic loading-readout in-place re-calibration test was removed: the
# loading readout is now COMPOSED by the user -- a CameraMeasurement publishing
# ``frame`` + an OccupancyProcessor turning ``frame`` into occupancy + a
# CalibrateReadoutTask producing the calibration -- each addressed independently.
# Composition + the real detect path are covered by
# tests/test_logic_node_split_contract.py.
