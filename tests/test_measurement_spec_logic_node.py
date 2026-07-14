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
from conftest import raw_device_set


REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.core.params import ParamDecl
from Zou_lab_control.neutral_atom.operations.measurement import (
    MeasurementSpec,
    ScannedMeasurement,
)
from Zou_lab_control.neutral_atom.operations.logic import ScannedMeasurementNode
from Zou_lab_control.neutral_atom.operations.temperature import (
    build_release_recapture_pulse,
    fit_temperature,
)

from conftest import fire_live_imaging   # the live "On Pulse" the trigger-driven camera needs


def _curve(hub, name):
    """The DISPLAYED curve for a scan signal: the node publishes the RAW (R,P,*data_shape)
    block, and a plot reduces the repeat axis.  These tests check the curve, so reduce it the same
    way a panel does (R=1 -> a single pass) -> (P,) / (P,D)."""
    from Zou_lab_control.frontend.live import reduce_repeat
    return np.asarray(reduce_repeat(np.asarray(hub.latest(name), dtype=float), "replace"), dtype=float)


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
    # capture_radius is an ANALYSIS/fit input (na.fit_temperature), NOT an acquisition param (#H3q):
    # the measurement acquires survival-vs-t_off; the metadata only points at the fit.
    assert "capture_radius" not in {d.key for d in temp.params}
    assert "fit_param" not in temp.metadata and "fit" not in temp.metadata
    assert "analysis_fit" in temp.metadata
    assert dur.key == "readout"
    assert dur.x_key == "detection_time" and dur.y_key == "fidelity"
    assert dur.result_labels == ("Detection time", "Fidelity")

    m_temp = temp.build(**temp.defaults())
    assert isinstance(m_temp, ScannedMeasurement)
    assert m_temp.axis.values.size == 13          # default points
    m_dur = dur.build(**dur.defaults())
    assert isinstance(m_dur, ScannedMeasurement)
    assert m_dur.axis.values.size == 11


# --------------------------------------------- spec.make_node (spec owns assembly)


def test_measurement_spec_make_node_builds_a_live_node_matching_its_declared_signals():
    """The ``ProcessorSpec.make_node`` counterpart: EVERY discovered measurement spec assembles its OWN
    live logic node via ``spec.make_node(hub, prefix=, repeat=)`` -- the console no longer imports a
    concrete na node class to pick one by a metadata string.  The node it returns is a real LogicNode
    that publishes EXACTLY the signals the spec implies behind the prefix (``<slug>_<x>``/``<slug>_<y>``),
    so the GUI's `_build_logic_node` reduces to one `make_node` call with no node-class knowledge."""
    from Zou_lab_control.neutral_atom.operations.logic import LogicNode

    exp = _calibrated_virtual_session(grid=(2, 3))
    try:
        hub = SignalHub()
        for spec in exp.readout.measurement_specs():
            node = spec.make_node(hub, prefix=f"{spec.key}_", repeat=1, **spec.defaults())
            assert isinstance(node, LogicNode)
            published = node.published_signals()
            # The node is the concrete source of its bindings. PulseScan replaces the generic
            # descriptor token with the selected pulse field's semantic coordinate name.
            assert node.x_signal in published
            assert node.y_signal in published
    finally:
        exp.close()


def test_make_node_routes_each_scan_tier_to_its_node_class():
    """make_node routes by the spec's scan TIER (its single source ``metadata['node']``), NOT by the
    caller knowing a class: the DECOUPLED ``"pulse_scan"`` tier -> a PulseScanNode (device driver whose
    y is a signal_expr off another node); the COUPLED temperature tier (no ``"pulse_scan"`` key) ->
    a frame-reducing ScannedMeasurementNode.  The tier's physics is pinned by
    test_scan_tier_boundary; here we pin that make_node honours it."""
    from Zou_lab_control.neutral_atom.operations.logic import PulseScanNode, ScannedMeasurementNode

    exp = _calibrated_virtual_session(grid=(2, 3))
    try:
        hub = SignalHub()
        by_name = {s.name: s for s in exp.readout.measurement_specs()}
        pscan = by_name["Pulse scan"]
        temp = by_name["Temperature"]
        assert pscan.metadata.get("node") == "pulse_scan"
        assert temp.metadata.get("node") != "pulse_scan"
        assert isinstance(pscan.make_node(hub, prefix="pulse_scan_", **pscan.defaults()), PulseScanNode)
        assert isinstance(temp.make_node(hub, prefix="temperature_", **temp.defaults()),
                          ScannedMeasurementNode)
    finally:
        exp.close()


def test_task_console_does_not_import_concrete_scan_node_classes_to_assemble_measurements():
    """DECOUPLING guard (#finding-14): the console's measurement-assembly path goes through
    ``spec.make_node``, so ``task_console.py`` must NOT import the concrete scan node classes
    (``PulseScanNode`` / ``ScannedMeasurementNode``) to pick one by ``metadata['node']``.  A regression
    that re-couples the GUI to a concrete na node class re-adds such an import and trips this."""
    from Zou_lab_control.frontend import task_console as tc_mod

    src = Path(tc_mod.__file__).read_text(encoding="utf-8")
    for cls in ("PulseScanNode", "ScannedMeasurementNode"):
        # the names may still appear in PROSE (docstrings/comments) but never in an import statement
        assert f"import {cls}" not in src and f"import ... {cls}" not in src
    import re
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") and "operations.logic" in stripped and " import " in stripped:
            imported = stripped.split(" import ", 1)[1]
            assert "PulseScanNode" not in imported and "ScannedMeasurementNode" not in imported, (
                f"task_console must not import a concrete scan node class: {stripped!r}")


# ------------------------------------------------- ScannedMeasurementNode (sync)


def _temperature_node(exp, hub, *, points, shots, t_max_us=300.0):
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, t_max_us, points), shots=shots)
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
    y = _curve(hub, spec.y_key).reshape(-1)
    # Cumulative curve has exactly ``points`` entries when complete.
    assert x.shape == (1, 13, 1)
    assert y.shape == (13,)
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    # The virtual loss model is on: survival is high near t_off=0, low at the end,
    # with a clear overall decay (this is the SAME contract path real hardware runs).
    assert y[0] >= 0.85
    assert y[-1] <= 0.25
    third = max(1, len(y) // 3)
    assert np.mean(y[:third]) > np.mean(y[-third:]) + 0.4
    # The fit recovers a temperature in the model's ballpark (50 uK truth).  capture_radius is an
    # ANALYSIS input supplied AT FIT TIME in metres (the trap geometry), not an acquisition param.
    fit = fit_temperature(x.reshape(-1), y, capture_radius=6.0e-6)
    assert fit.success
    assert 25e-6 <= fit.temperature_K <= 100e-6


def test_temperature_node_per_site_carries_the_per_site_dimension_in_the_raw_block():
    """A per-site scan declares ``data_shape=(n_sites,)``, so the ONE raw ``y`` block is
    ``(R,P,n_sites)`` -- the per-site vectors live in the data axis (a 1-D plot
    draws one line per site; a grid view reshapes a reduced point).  The node publishes only
    its coordinate and measured tensor; completion remains node control state."""
    exp = _calibrated_virtual_session(grid=(2, 3))
    n_sites = raw_device_set(exp).trap_array.n_sites
    hub = SignalHub()
    spec = exp.readout.measurement_specs()[0]
    measurement = spec.build(t_off=(0.0, 80.0, 4), shots=2, per_site=True)
    node = ScannedMeasurementNode(hub, measurement, x_key=spec.x_key, y_key=spec.y_key)
    node.run_to_completion()

    raw = np.asarray(hub.latest(spec.y_key))
    assert raw.shape == (1, 4, n_sites)              # (R,P,*data_shape), data_shape=(n_sites,)
    reduced = _curve(hub, spec.y_key)                # plot reduction -> (points, n_sites)
    assert reduced.shape == (4, n_sites)
    finite = reduced[np.isfinite(reduced)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert node.published_signals() == frozenset({spec.x_key, spec.y_key})
    # the latest point's per-site vector reshapes cleanly to the grid (the trap array owns the shape).
    grid = raw_device_set(exp).trap_array.grid_shape
    assert reduced[-1].reshape(grid).shape == grid


def test_readout_duration_node_runs_out_a_fidelity_curve():
    exp = na.connect("virtual")
    exp.readout.sitemap(frames=4, display=False)
    exp.readout.thresholds(frames=20, display=False)
    hub = SignalHub()
    spec = {item.name: item for item in exp.readout.measurement_specs()}["Fidelity vs duration"]
    measurement = spec.build(duration=(5.0, 50.0, 4), shots=6)
    node = ScannedMeasurementNode(hub, measurement, x_key=spec.x_key, y_key=spec.y_key)

    node.run_to_completion()

    x = hub.latest(spec.x_key)
    y = _curve(hub, spec.y_key).reshape(-1)
    assert x.shape == (1, 4, 1)
    assert y.shape == (4,)
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))


# ------------------------------------------------ ScannedMeasurementNode (thread)


def test_temperature_node_start_thread_auto_stops_when_scan_completes():
    exp = _calibrated_virtual_session(grid=(2, 3))
    hub = SignalHub()
    node, spec = _temperature_node(exp, hub, points=4, shots=2, t_max_us=80.0)

    runtime = exp._require_runtime_services()
    handle = runtime.fence.start(node)
    handle.wait_started(2.0)
    deadline = time.perf_counter() + 10.0
    while node.running and time.perf_counter() < deadline:
        time.sleep(0.02)

    assert node.finished
    assert node.running is False        # finite scan stopped its own daemon thread
    y = _curve(hub, spec.y_key).reshape(-1)
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
    cam = raw_device_set(exp).camera
    assert cam.roi is None                                  # full frame by default (unchanged)
    cam.configure(roi=[10, 20, 6, 16])                      # x=10, y=6 are NOT multiples of 4
    assert cam.roi == (8, 20, 8, 16)                        # snapped to the grid, reported back
    fire_live_imaging(exp)                                  # On Pulse: the trigger-driven camera streams
    frame = cam.acquire(1)[-1]                # the wired camera senses the firing itself
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
    cam_node = na.CameraMeasurement(hub, raw_device_set(exp).camera)
    # endpoints stay endpoints (NOT collapsed to position+size) -- plot format
    assert cam_node.region_to_acquisition_parameters(10, 30, 6, 22) == {"region": [10, 30, 6, 22]}
    # endpoints come in any order -> sorted endpoints
    assert cam_node.region_to_acquisition_parameters(30, 10, 22, 6) == {"region": [10, 30, 6, 22]}
    # the round trip through the camera: set region endpoints -> device ROI -> read
    # back as endpoints (full sensor 64x80, no snap needed for /4-aligned values)
    cam_node.set_acquisition_parameters(region=[8, 28, 12, 28])
    assert cam_node.acquisition_parameters()["region"] == [8, 28, 12, 28]
    # a non-spatial processor has no rectangle parameter.
    from Zou_lab_control.neutral_atom.operations.logic import Processor

    class _NonSpatialProcessor(Processor):
        provides = ("value",)

        def transform(self, inputs):
            return {"value": inputs["frame"]}

    processor = _NonSpatialProcessor(hub, consumes=("frame",))
    assert processor.region_to_acquisition_parameters(10, 30, 6, 22) == {}


# ------------------------------------------------ acquisition-parameter protocol


def test_camera_measurement_exposes_camera_params_and_applies_them_live():
    """A panel is a VIEW; the node's SOURCE owns the editable params.  A raw-frame
    camera measurement's source is the camera, so acquisition_parameters() reports the
    camera's exposure and set_acquisition_parameters() reconfigures the camera in place
    -- no rebuild, same measurement keeps publishing 'frame'."""
    exp = na.connect("virtual")
    cam = raw_device_set(exp).camera
    hub = SignalHub()
    cam_node = na.CameraMeasurement(hub, cam, sequencer=raw_device_set(exp).sequencer)
    fire_live_imaging(exp)                                  # On Pulse: the trigger-driven camera streams

    # default frames_per_cycle=1 -> 'frame' = the (repeat, H, W) data array (a 2D panel reduces its
    # event 0's repeat block).  ONE signal per emCCD event of the cycle; frames_per_cycle=1 -> just frame_0.
    assert cam_node.published_signals() == frozenset({"frame_0"})
    params = cam_node.acquisition_parameters()
    assert "exposure" in params and params["exposure"] == float(cam.exposure)
    assert params["frames_per_cycle"] == 1

    cam_node.set_acquisition_parameters(exposure=0.05)
    assert float(cam.exposure) == 0.05                      # applied to the camera in place
    assert cam_node.acquisition_parameters()["exposure"] == 0.05

    cam_node.step()
    frame = np.asarray(hub.latest("frame_0"))              # frame_0 IS event 0's (repeat,1,H,W) block
    assert frame.ndim == 4                                  # the (repeat, 1, H, W) data block
    assert frame.shape[1] == 1                              # one data point (a frame sweeps no param)


def test_camera_build_applies_repeat_through_every_entry_point():
    """The camera's ``repeat`` (0 = ∞; K = keep & average a K-deep frame block then STOP) is the
    acquisition knob the plot's repeat_mode then collapses (average / add / ...).  It MUST be applied
    identically by EVERY build entry point -- the notebook ``camera_spec().build(hub, repeat=K)``, the
    ``camera_measurement(hub, repeat=K)`` helper, and (via that helper) the console -- never silently
    dropped on one path (that left a notebook camera stuck at repeat=0 = a 1-deep ring = no repeat_mode
    effect, while the GUI worked).  ``set_repeat`` is the single source; every path routes K to it."""
    exp = na.connect("virtual")
    hub = SignalHub()
    spec = exp.readout.camera_spec()
    for K in (0, 1, 50):
        # notebook spec path (build kwargs) and the bare helper must agree, ring depth = max(1, K)
        node_spec = spec.build(hub, repeat=K)
        node_help = exp.readout.camera_measurement(hub, repeat=K)
        assert node_spec.repeat == K and node_help.repeat == K
        assert node_spec._ring == max(1, K) and node_help._ring == max(1, K)


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

        def acquire(self, frames=1, *, stop=None, **kw):
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
    from zlc_neutral_atom.runtime import (
        CleanupStepAck,
        DeviceBroker,
        DeviceIdentityAck,
        DeviceIdentityEvidenceKind,
        MemoryQuarantineJournal,
        ResourceArbiter,
        ResourceKey,
        RunController,
        SafeStateAck,
        SafetyOperation,
    )
    from zlc_workbench import (
        LegacyDeviceRegistration,
        LegacyDeviceRegistry,
        LegacyRuntimeFence,
    )

    broker = DeviceBroker()
    registry = LegacyDeviceRegistry(broker)
    registry.register(
        LegacyDeviceRegistration(
            device=cam,
            key=ResourceKey.parse("device/camera/blocking-fixture"),
            identity_probe=lambda: DeviceIdentityAck(
                "fixture:blocking-camera",
                DeviceIdentityEvidenceKind.INSTALLATION_ASSERTED_ENDPOINT,
                "fixture:blocking-camera:connection",
                "test-assets-v1",
            ),
            cleanup_operations={
                SafetyOperation.SAFE_STATE: lambda: CleanupStepAck(
                    SafetyOperation.SAFE_STATE, "fixture-safe-command"
                )
            },
            cleanup_order=(SafetyOperation.SAFE_STATE,),
            verify_safe_state=lambda: SafeStateAck("fixture-safe-readback"),
        )
    )
    fence = LegacyRuntimeFence(
        RunController(ResourceArbiter(MemoryQuarantineJournal())), registry
    )
    run_handle = fence.start(cam_node)
    run_handle.wait_started(2.0)
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
        assert fence.stop(cam_node, timeout=2.0).terminated
