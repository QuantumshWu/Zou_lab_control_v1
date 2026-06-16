"""Mechanical guards for the architecture-audit hardening pass.

Each test pins one correctness contract that, if broken, would silently mislead
a real run (the project rule: a guideline that can be machine-enforced is
written as a FAILING test, not just prose).  Everything fakes only the lowest
layer (a dcam handle / a camera / an RPC conn); the logic under test is the same
code a real run executes.
"""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from Zou_lab_control.neutral_atom.core.signals import SignalHub
from Zou_lab_control.neutral_atom.devices.base import AcquisitionCancelled
from Zou_lab_control.neutral_atom.operations.feeds import ExperimentFeed
from Zou_lab_control.neutral_atom.operations.measurement import (
    ScanAxis,
    ScannedMeasurement,
)
from Zou_lab_control.neutral_atom.timing import imaging_channel_kwargs


# --------------------------------------------------------------------------- #11
def test_hub_tracks_per_signal_version():
    """A rolling monitor needs to tell 'my signal got a new sample' from 'some
    other feed bumped the global version' -- per-signal counters provide that."""
    hub = SignalHub()
    hub.publish({"a": 1.0})
    hub.publish({"a": 2.0, "b": 9.0})
    versions = hub.signal_versions()
    assert versions["a"] == 2
    assert versions["b"] == 1
    # an unrelated publish does NOT bump a quiet signal's counter
    hub.publish({"b": 8.0})
    assert hub.signal_versions()["a"] == 2
    hub.clear()
    assert hub.signal_versions() == {}


# --------------------------------------------------------------------------- M7
def test_wedged_feed_surfaces_error_not_silent():
    """A feed whose source raises must record the error + publish a health signal,
    never freeze silently."""
    hub = SignalHub()

    class BoomFeed(ExperimentFeed):
        def shot(self):
            raise RuntimeError("trigger never arrived")

    feed = BoomFeed(hub).start(rate_hz=50)
    try:
        deadline = time.monotonic() + 2.0
        while feed.last_error is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert feed.last_error is not None and "trigger never arrived" in feed.last_error
        assert feed.consecutive_errors >= 1
        assert "feed_error" in hub.names()
    finally:
        feed.stop()


# --------------------------------------------------------------------------- M4
def test_loading_feed_channel_mapping_follows_sequencer():
    """imaging_channel_kwargs maps the conventional roles onto whatever channels
    the bound sequencer exposes -- a real chNN streamer must NOT get the
    trap/cooling/probe/emCCD placeholders."""
    ch_seq = types.SimpleNamespace(channels=["ch00", "ch03", "ch09", "ch11"], trigger_channels=["ch11"])
    assert imaging_channel_kwargs(ch_seq) == {
        "trap_channel": "ch09", "cooling_channel": "ch00",
        "probe_channel": "ch03", "trigger_channel": "ch11",
    }
    named = types.SimpleNamespace(channels=["trap", "cooling", "probe", "emCCD"], trigger_channels=["emCCD"])
    assert imaging_channel_kwargs(named)["trigger_channel"] == "emCCD"
    # virtual / unbound: fall back to imaging_sequence's own placeholder defaults
    assert imaging_channel_kwargs(None) == {}


# --------------------------------------------------------------------------- M3
class _PresetReducer:
    """Per-site reducer that returns a preset row per shot (a NaN = empty site)."""

    n_series = 2

    def __init__(self, rows):
        self._rows = list(rows)
        self._i = 0

    def reduce(self, frames, calibration):
        row = self._rows[self._i % len(self._rows)]
        self._i += 1
        return np.asarray(row, dtype=float)


class _FakePlan:
    n_frames = 1

    def sequence_for(self, pulse, axis, value):
        return "seq"


class _FakePulse:
    sequencer = "seq-dev"

    def frame_sequence(self, *a, **k):
        return "seq"

    def set_time(self, *a, **k):
        return self


class _CountCamera:
    last_stop = None

    def acquire(self, frames=1, *, sequence=None, sequencer=None, stop=None, **kw):
        self.last_stop = stop
        return [np.zeros((2, 2))]


def test_per_site_nan_does_not_poison_scan_point():
    """np.nanmean over shots: a site empty on ONE shot must not drag that site's
    whole scan point to NaN (plain np.mean would)."""
    meas = ScannedMeasurement(
        pulse=_FakePulse(), camera=_CountCamera(), sequencer="seq-dev",
        calibration=None,
        axis=ScanAxis(slot="s0", values=[1.0], kind="duration"),
        plan=_FakePlan(),
        reducer=_PresetReducer([[1.0, np.nan], [3.0, 5.0]]),
        shots_per_point=2,
    )
    out = meas.measure(1.0)
    assert out[0] == pytest.approx(2.0)      # both shots present -> mean
    assert out[1] == pytest.approx(5.0)      # NaN shot ignored, NOT poisoned to NaN


# --------------------------------------------------------------------------- #1
def test_dac_axis_rejects_unknown_slot_early():
    """A DAC scan whose slot the bound pulse does not have moves nothing on real
    hardware -- it must fail at construction, not silently run a null scan."""
    # Guard the production attribute name itself: the validation reads
    # ``scan_var_names`` off the bound pulse; if that property is renamed the
    # check would silently become dead code, so assert it still exists.
    from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState
    assert hasattr(PulseTableState, "scan_var_names")

    class _SlotPulse(_FakePulse):
        pulse = types.SimpleNamespace(scan_var_names=["s0", "s1"])

    with pytest.raises(ValueError, match="not a scan slot"):
        ScannedMeasurement(
            pulse=_SlotPulse(), camera=_CountCamera(), sequencer="seq-dev",
            calibration=None,
            axis=ScanAxis(slot="s5", values=[0.0], kind="dac"),  # s5 absent
            plan=_FakePlan(), reducer=_PresetReducer([[0.0, 0.0]]),
        )
    # positive control: a real slot constructs fine
    ScannedMeasurement(
        pulse=_SlotPulse(), camera=_CountCamera(), sequencer="seq-dev",
        calibration=None,
        axis=ScanAxis(slot="s0", values=[0.0], kind="dac"),
        plan=_FakePlan(), reducer=_PresetReducer([[0.0, 0.0]]),
    )


# --------------------------------------------------------------------------- M1/M6/M2
def _fake_dcam_module():
    ns = types.SimpleNamespace
    DCAM_IDPROP = ns(
        EXPOSURETIME=1, TRIGGERSOURCE=2, TRIGGERACTIVE=3, TRIGGERPOLARITY=4,
        READOUTSPEED=5, SENSORMODE=6, TRIGGER_GLOBALEXPOSURE=7,
        SUBARRAYMODE=8, SUBARRAYHSIZE=9, SUBARRAYHPOS=10, SUBARRAYVSIZE=11, SUBARRAYVPOS=12,
    )
    DCAMPROP = ns(
        TRIGGERSOURCE=ns(EXTERNAL=2), TRIGGERACTIVE=ns(EDGE=1),
        TRIGGERPOLARITY=ns(POSITIVE=1), MODE=ns(ON=1),
    )
    return ns(DCAM_IDPROP=DCAM_IDPROP, DCAMPROP=DCAMPROP)


class _FakeDcam:
    def __init__(self, reject=None, clamp=None):
        self._reject = reject          # idprop value that prop_setvalue rejects
        self._clamp = clamp            # idprop whose read-back differs
        self._vals = {}

    def prop_setvalue(self, idprop, value):
        if idprop == self._reject:
            return False
        self._vals[idprop] = value
        return True

    def prop_getvalue(self, idprop):
        if idprop == self._clamp:
            return self._vals.get(idprop, 0) + 1   # substituted/clamped value
        return self._vals.get(idprop, 0)

    def lasterr(self):
        return "DCAMERR_FAKE"


def _camera_with(fake):
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
    cam = QCMOSCamera({"exposure": 0.02})
    cam._module = _fake_dcam_module()
    cam._dcam = fake
    return cam


def test_qcmos_write_settings_raises_on_rejected_prop():
    """An unchecked prop_setvalue would leave the camera silently mis-set (e.g.
    internal trigger) and then time out confusingly; it must raise at config."""
    mod = _fake_dcam_module()
    cam = _camera_with(_FakeDcam(reject=mod.DCAM_IDPROP.TRIGGERSOURCE))
    with pytest.raises(RuntimeError, match="trigger_source"):
        cam._write_settings()


def test_qcmos_write_settings_raises_on_clamped_trigger():
    """A verified trigger prop whose read-back differs (camera clamped/substituted
    it) must fail loud, not at first light."""
    mod = _fake_dcam_module()
    cam = _camera_with(_FakeDcam(clamp=mod.DCAM_IDPROP.TRIGGERACTIVE))
    with pytest.raises(RuntimeError, match="read back"):
        cam._write_settings()


def test_qcmos_write_settings_ok_when_accepted():
    cam = _camera_with(_FakeDcam())
    cam._write_settings()   # no raise: all writes accepted + read back equal


# --------------------------------------------------------------------------- M5
def test_qcmos_acquire_cancels_promptly_on_stop():
    """A live feed's Stop must interrupt a wedged trigger wait (AcquisitionCancelled),
    not block for the whole timeout."""

    class _WedgedDcam(_FakeDcam):
        def buf_alloc(self, n):
            return True

        def cap_start(self, bSequence=True):
            return True

        def wait_capevent_frameready(self, timeout_ms):
            return False          # no frame ever -> would block to timeout

        def cap_stop(self):
            return True

        def buf_release(self):
            return True

    cam = _camera_with(_WedgedDcam())
    stop = threading.Event()
    stop.set()
    with pytest.raises(AcquisitionCancelled):
        cam.acquire(1, stop=stop)


def test_qcmos_acquire_times_out_without_stop():
    class _WedgedDcam(_FakeDcam):
        def buf_alloc(self, n):
            return True

        def cap_start(self, bSequence=True):
            return True

        def wait_capevent_frameready(self, timeout_ms):
            return False

        def cap_stop(self):
            return True

        def buf_release(self):
            return True

    cam = _camera_with(_WedgedDcam())
    with pytest.raises(TimeoutError):
        cam.acquire(1, timeout_ms=10)


# --------------------------------------------------------------------------- M8
def test_remote_sequencer_abort_reconnects():
    """abort() runs on the error/safing path; it must reconnect like the other
    RPC calls, not silently no-op on a dropped link and leave outputs running."""
    from Zou_lab_control.neutral_atom.devices.sequencer import RemoteSequencer

    seq = RemoteSequencer(host="127.0.0.1", port=18861, channels=["ch00", "ch01"])
    opened = {"n": 0}

    class _Root:
        def __init__(self):
            self.aborted = False

        def abort(self):
            self.aborted = True

    def fake_open():
        opened["n"] += 1
        seq._conn = types.SimpleNamespace(root=_Root(), closed=False)
        return seq

    seq.open = fake_open
    seq.abort()
    assert opened["n"] == 1                 # abort routed through open()
    assert seq._conn.root.aborted is True


# --------------------------------------------------------------------------- B1 (re-audit)
def test_exposure_inference_follows_probe_channel_mapping():
    """The camera must read exposure off the SAME channel the imaging sequence
    used.  On a real chNN streamer the probe pulse is on ch03; inferring from the
    placeholder 'probe' name silently returns the default -> a flat exposure scan.
    This was virtual-invisible, so it is pinned here."""
    from Zou_lab_control.neutral_atom.timing import exposure_from_sequence, imaging_sequence

    ch_seq = types.SimpleNamespace(channels=["ch00", "ch03", "ch09", "ch11"], trigger_channels=["ch11"])
    kw = imaging_channel_kwargs(ch_seq)
    seq = imaging_sequence(exposure=0.037, load=True, name="readout", **kw)
    # inferring with the mapped probe channel (ch03) recovers the real exposure
    assert exposure_from_sequence(seq, default=0.02, channel=kw["probe_channel"]) == pytest.approx(0.037)
    # the placeholder default ('probe') MISSES the ch03 pulse -> silently default (the old bug)
    assert exposure_from_sequence(seq, default=0.02) == pytest.approx(0.02)


# --------------------------------------------------------------------------- M-B (re-audit)
def test_scan_engine_threads_stop_into_acquire():
    """A Stop must interrupt a wedged trigger DURING a scan point, not only
    between points: the feed shares its stop event with the measurement, which
    passes it to camera.acquire."""
    from Zou_lab_control.neutral_atom.operations.feeds import ScannedMeasurementFeed

    hub = SignalHub()
    meas = ScannedMeasurement(
        pulse=_FakePulse(), camera=_CountCamera(), sequencer="seq-dev",
        calibration=None,
        axis=ScanAxis(slot="s0", values=[1.0, 2.0], kind="duration"),
        plan=_FakePlan(), reducer=_PresetReducer([[1.0, 2.0]]),
    )
    feed = ScannedMeasurementFeed(hub, meas, x_key="x", y_key="y")
    assert meas.stop_event is feed._stop           # feed shared its stop event
    meas.measure(1.0)
    assert meas.camera.last_stop is feed._stop      # ...and it reaches camera.acquire


# --------------------------------------------------------------------------- round-3 B1 (temperature chNN)
def test_release_recapture_builds_on_chNN_channels():
    """The temperature pulse must target the bound sequencer's real channels;
    on a chNN streamer the trap/probe/emCCD placeholder roles aren't present and
    the builder raises -- so the spec must remap via imaging_channel_kwargs."""
    from Zou_lab_control.neutral_atom.operations.temperature import build_release_recapture_pulse

    ch_seq = types.SimpleNamespace(channels=["ch00", "ch03", "ch09", "ch11"], trigger_channels=["ch11"])
    kw = imaging_channel_kwargs(ch_seq)
    roles = {k: kw[k] for k in ("trap_channel", "probe_channel", "trigger_channel")}
    state = build_release_recapture_pulse(channels=ch_seq.channels, **roles)   # mapped roles -> builds
    assert state is not None
    with pytest.raises(ValueError):           # placeholder defaults absent on chNN -> the old bug
        build_release_recapture_pulse(channels=ch_seq.channels)


# --------------------------------------------------------------------------- round-3 M2 (calibration ROI fingerprint)
def test_calibration_rejects_wrong_image_shape():
    """Centers are absolute pixels; a frame from a different (shifted/resized) ROI
    must fail loud, not silently extract the wrong pixels."""
    from Zou_lab_control.neutral_atom.core.calibration import TrapCalibration

    cal = TrapCalibration(centers=[[2, 2]], thresholds=[5.0], metadata={"image_shape": [8, 8]})
    cal.signals(np.zeros((8, 8)))             # matching shape -> fine
    with pytest.raises(ValueError, match="does not match"):
        cal.signals(np.zeros((6, 10)))        # different shape -> raise (recalibrate)
    # no fingerprint recorded -> no check (backward compatible with old saved calibrations)
    TrapCalibration(centers=[[2, 2]], thresholds=[5.0]).signals(np.zeros((6, 10)))


# --------------------------------------------------------------------------- round-3 plot-kind table guard
def test_panel_kind_tables_agree():
    """A panel kind is driven by several parallel tables (label / default source /
    params); a new kind half-registered in only some of them is the asymmetry the
    audit flagged.  Mechanically require every kind to appear in all of them so a
    half-registration fails here instead of silently misbehaving in the console."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.task_console import PANEL_KINDS, PANEL_PARAMS, _DEFAULT_SOURCES

    kinds = set(PANEL_KINDS)
    assert set(_DEFAULT_SOURCES) == kinds, "every panel kind needs a _DEFAULT_SOURCES entry"
    assert set(PANEL_PARAMS) == kinds, "every panel kind needs a PANEL_PARAMS entry (empty tuple allowed)"


# --------------------------------------------------------------------------- Edit snapshot not squished
def test_edit_snapshot_canvas_keeps_design_height():
    """The Edit-tab frozen snapshot must keep the figure's design height: it lives
    in a scroll area whose QVBoxLayout would otherwise SQUISH the canvas to ~half
    height, clipping the plot and the y-axis label (the snapshot then looks empty/
    broken).  Pinning the canvas minimum height makes the page scroll instead."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib
    matplotlib.use("Agg")
    from PyQt5 import QtWidgets
    import Zou_lab_control.frontend as zf
    from Zou_lab_control.frontend.task_console import PanelEditor, TaskConsole

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    hub = SignalHub()
    state = zf.TaskConsoleState(name="t", panels=[
        zf.PanelConfig(kind="monitor", title="loading rate", size="2x2", source="value = rate")])
    console = TaskConsole(hub=hub, state=state)
    for i in range(30):                       # roll real points like a live feed
        hub.publish({"rate": float(i % 5)})
        console.refresh_once()
    editor = PanelEditor(console.cards[0], console)
    assert editor._canvas is not None
    # the floor: the snapshot can never be shorter than the figure's design size
    assert editor._canvas.minimumHeight() >= editor._canvas.sizeHint().height() > 0


# --------------------------------------------------------------------------- round-3 save->load reproduce
def test_data_figure_save_load_round_trip(tmp_path):
    """A finished scan saved to .npz must reopen as a refittable static figure
    (the reference has this; the framework lacked the read-back).  Rebuilds via
    the same plot() renderer, so the reloaded figure matches the live one."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib
    matplotlib.use("Agg")
    import Zou_lab_control.frontend as zf

    x = np.linspace(0.0, 1.0, 11)
    y = x ** 2
    fig = zf.plot(x, y, kind="1d", labels=("t (s)", "signal", "z"), update=False, display=False)
    saved_unit = fig.data_figure.unit_original
    out = fig.save(str(tmp_path / "scan"))
    df = zf.load(str(out["data"]), display=False)
    assert np.allclose(df.data_x[:, 0], x)            # data round-trips
    assert np.allclose(df.data_y[:, 0], y)
    assert df.labels[0].startswith("t")               # labels preserved
    assert df.unit_original == saved_unit             # recorded unit faithfully restored
    df.save(str(tmp_path / "reloaded"))               # fit/save stack works on the reload


# --------------------------------------------------------------------------- round-3 relim hysteresis
def test_relim_deadband_no_clip_no_jitter():
    """Autoscale dead-band: a noisy trace inside the view must NOT rescale every
    frame (y-axis jitter), but a point must NEVER be clipped -- rescaling always
    happens when data would leave the view."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import matplotlib
    matplotlib.use("Agg")
    import Zou_lab_control.frontend as zf

    fig = zf.plot(np.arange(8.0), np.zeros(8), kind="1d", update=False, display=False)
    fig.relim_mode = "normal"
    fig.data_y[:] = 5.0
    fig.relim()                                   # establishes the band around 5
    changed = []
    for v in (5.1, 4.9, 5.05, 4.95, 5.0):         # bounded noise inside the band
        fig.data_y[:] = v
        changed.append(fig.relim())
        assert fig.ylim_max >= v                  # never clips
    assert sum(changed) == 0                      # no per-frame rescale within band
    fig.data_y[:] = 100.0                          # a real jump forces a rescale...
    assert fig.relim() is True
    assert fig.ylim_max >= 100.0                   # ...and still contains the data
