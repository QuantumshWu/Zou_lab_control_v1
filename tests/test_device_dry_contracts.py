"""Device-layer single-source contracts (the devices DRY audit batch).

Each test machine-enforces one "this fact lives in ONE place" rule so a re-typed copy
cannot silently drift back in:

* trigger-channel normalization (None -> default line, () -> no counting line -> zero
  edges) lives ONLY in ``devices.camera_trigger`` -- call sites pass values through;
* a rising-edge-only "does this pulse trigger the camera" judgement is the ONE
  ``base_cycle_trigger_pulses`` (an explicit value=0 entry gates nothing, virtual == real);
* the software-trigger predicate is the ONE ``devices.base.is_software_trigger``;
* an illegal exposure fails identically on every backend (``_validated_exposure``);
* ``sensor_shape`` of a CLOSED real camera is the honest ``None`` of the base contract;
* the snapshot ``type`` key has ONE producer (``BaseDevice.snapshot``);
* the "settle fast-forwards with sleep_scale" rule has ONE home (``SequencerService.settle``).
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import (
    BaseDevice, DeviceProperty, LaserDevice, RFSourceDevice, SOFTWARE_TRIGGER,
    collect_device_properties, is_software_trigger, read_only)
from Zou_lab_control.neutral_atom.devices.camera_trigger import (
    DEFAULT_CAMERA_TRIGGER_CHANNELS, base_cycle_camera_trigger_pulses,
    base_cycle_trigger_pulses, count_camera_trigger_pulses, count_trigger_pulses,
    normalize_trigger_channels, resolve_camera_trigger_channels)
from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera, QCMOSConfig
from Zou_lab_control.neutral_atom.devices.sequencer import (
    ManualSequencer, RemoteSequencer, SequencerService)
from Zou_lab_control.neutral_atom.devices.virtual import (
    VirtualCamera, VirtualLaser, VirtualMotCamera, VirtualRF, VirtualSequencer, VirtualTrapArray)
from Zou_lab_control.neutral_atom.timing import PulseSequence


def _trap():
    return VirtualTrapArray(grid_shape=(2, 3), image_shape=(16, 20), seed=1)


def _single_trigger_seq(*, trigger_value: int = 1) -> PulseSequence:
    """One probe window + one emCCD trigger entry whose digital value is configurable:
    value=1 is a rising edge (one frame), value=0 explicitly drives the line LOW."""
    return (PulseSequence(name="one_window")
            .pulse("probe", 0.0, 1e-3)
            .pulse("emCCD", 0.0, 1e-3, value=trigger_value))


# --------------------------------------------------------------- [0] normalization single source
def test_trigger_channel_normalization_none_defaults_and_empty_means_no_line():
    """None -> the default camera line (applied INSIDE camera_trigger, nowhere else);
    () -> "no counting line" -> zero edges, never silently collapsed back to the default."""
    seq = _single_trigger_seq()
    assert normalize_trigger_channels(None) == DEFAULT_CAMERA_TRIGGER_CHANNELS
    assert normalize_trigger_channels(()) == ()
    assert count_trigger_pulses(seq) == count_trigger_pulses(
        seq, trigger_channels=DEFAULT_CAMERA_TRIGGER_CHANNELS) == 1
    assert base_cycle_trigger_pulses(seq, trigger_channels=()) == 0
    assert count_trigger_pulses(seq, trigger_channels=()) == 0


def test_resolve_camera_trigger_channels_reads_the_effective_line():
    """The camera->channels resolution is ONE rule: a free-running camera resolves to ()
    (zero edges via the *_camera_* wrappers), a hardware-triggered camera to its wiring,
    and an object declaring nothing to the default line."""
    seq = _single_trigger_seq()
    free = VirtualMotCamera(trigger_source=SOFTWARE_TRIGGER)
    assert resolve_camera_trigger_channels(free) == ()
    assert count_camera_trigger_pulses(free, seq) == 0
    assert base_cycle_camera_trigger_pulses(free, seq) == 0

    wired = VirtualCamera(_trap(), exposure=1e-3)            # default emCCD wiring
    assert resolve_camera_trigger_channels(wired) == DEFAULT_CAMERA_TRIGGER_CHANNELS
    assert count_camera_trigger_pulses(wired, seq) == 1

    class _Bare:                                             # declares nothing -> the default line
        pass

    assert resolve_camera_trigger_channels(_Bare()) == DEFAULT_CAMERA_TRIGGER_CHANNELS


# --------------------------------------------------------------- [1] one trigger judgement (edges)
def test_off_value_trigger_pulses_gate_no_live_frames():
    """A continuous (repeat_forever) program whose emCCD channel carries ONLY an explicit
    value=0 entry drives the line LOW -- no rising edge, so the virtual camera renders NO
    frame (the live view freezes), exactly like real hardware.  The judgement is the ONE
    ``base_cycle_trigger_pulses`` (rising edges only); the old value-blind duplicate
    fabricated frames here."""
    dark = _single_trigger_seq(trigger_value=0).forever()
    lit = _single_trigger_seq(trigger_value=1).forever()
    assert base_cycle_trigger_pulses(dark) == 0 and base_cycle_trigger_pulses(lit) == 1

    # Device-layer truth: the atom device renders nothing for the edge-less program.
    atoms = _trap()
    assert atoms.image_frames(dark, 1, capture_trigger_channels=("emCCD",), exposure=1e-3) == []
    assert len(atoms.image_frames(lit, 1, capture_trigger_channels=("emCCD",), exposure=1e-3)) == 1

    # Same contract path a live monitor takes (camera.acquire over the firing wire).
    seqr = VirtualSequencer()
    cam = VirtualCamera(_trap(), exposure=1e-3, sequencer=seqr)
    seqr.prepare(dark)
    seqr.fire(dark)
    assert cam.acquire(1) == []                              # no edge -> frozen, never fabricated
    seqr.prepare(lit)
    seqr.fire(lit)
    assert len(cam.acquire(1)) == 1                          # a rising edge -> one frame


# --------------------------------------------------------------- [4] software-trigger predicate
def test_software_trigger_predicate_is_single_source_and_case_insensitive():
    for spelling in ("Software", "software", "SOFTWARE", "soFTware"):
        assert is_software_trigger(spelling)
        assert PylonCamera(trigger_source=spelling)._free_run
        assert VirtualMotCamera(trigger_source=spelling)._free_run
    for hard in ("Line1", "mot_trigger", "ch05", ""):
        assert not is_software_trigger(hard)
        assert not PylonCamera(trigger_source=hard)._free_run
        assert not VirtualMotCamera(trigger_source=hard)._free_run


def test_free_run_is_one_inherited_base_property():
    """``_free_run`` lives ONCE on ``CameraDevice`` -- both backends inherit it (was a byte-identical
    copy on each), so the "am I software-triggered?" rule cannot fork virtual vs real.  A camera with
    NO ``trigger_source`` knob (the readout camera / qCMOS) is always hardware-triggered."""
    from Zou_lab_control.neutral_atom.devices.base import CameraDevice
    assert "_free_run" not in PylonCamera.__dict__ and "_free_run" not in VirtualMotCamera.__dict__
    assert PylonCamera._free_run is CameraDevice._free_run          # the ONE property object, inherited
    assert VirtualMotCamera._free_run is CameraDevice._free_run
    assert VirtualCamera(_trap(), exposure=1e-3)._free_run is False  # no trigger_source knob -> never free-run


def test_roi_clear_predicate_is_single_source():
    """``is_roi_clear`` -- the ONE free-text ROI predicate the device viewer's setter uses -- accepts
    every canonical ``ROI_CLEAR_SENTINELS`` spelling PLUS the ``FULL_FRAME_DISPLAY`` read-back the
    getter shows, case-insensitively, so what the field displays round-trips.  A real (x,w,y,h) window
    is NOT a clear."""
    from Zou_lab_control.neutral_atom.devices.base import (
        FULL_FRAME_DISPLAY, ROI_CLEAR_SENTINELS, is_roi_clear)
    for spelling in (*ROI_CLEAR_SENTINELS, FULL_FRAME_DISPLAY, "FULL", "Full Frame", "  none  "):
        assert is_roi_clear(spelling), spelling
    for window in ("(0, 64, 0, 64)", "10,20,30,40"):
        assert not is_roi_clear(window), window


# --------------------------------------------------------------- [3] exposure validation
def test_invalid_exposure_raises_identically_on_every_backend():
    """cam.exposure = -1 (or an illegal constructor/configure value) raises ValueError on EVERY
    backend -- the validation is the one base-class ``_validated_exposure``, so the real pylon
    can no longer silently store a negative exposure the virtual camera would reject."""
    cameras = [
        VirtualCamera(_trap(), exposure=1e-3),
        VirtualMotCamera(),
        PylonCamera(trigger_source=SOFTWARE_TRIGGER),
        QCMOSCamera({"exposure": 1e-3}),
    ]
    for cam in cameras:
        for bad in (0, -1.0, float("nan")):
            with pytest.raises(ValueError):
                cam.exposure = bad
            with pytest.raises(ValueError):
                cam.configure(exposure=bad)
    for bad_ctor in (
        lambda: VirtualCamera(_trap(), exposure=-1.0),
        lambda: VirtualMotCamera(exposure=0),
        lambda: PylonCamera(exposure=-1.0),
        lambda: QCMOSConfig(exposure=0),
        lambda: PylonCamera(timeout=0),                      # the sibling bare-float knob
    ):
        with pytest.raises(ValueError):
            bad_ctor()


# --------------------------------------------------------------- [2] closed sensor_shape is None
def test_closed_real_camera_sensor_shape_is_honest_none():
    """Both REAL adapters report the base contract's honest "unknown until opened": None --
    never a fabricated (0, 0) a consumer would treat as a known zero-size sensor.  A ROI
    configured before open is still stored (and re-snapped by open() against the live
    GenICam grid)."""
    pylon = PylonCamera(trigger_source=SOFTWARE_TRIGGER)
    qcmos = QCMOSCamera({"exposure": 1e-3})
    assert pylon.sensor_shape is None
    assert qcmos.sensor_shape is None
    assert pylon.frame_shape is None                         # derived contract stays honest too
    pylon.configure(roi=(100, 640, 80, 480))                 # configure-before-open is normal
    assert pylon.roi == (100, 640, 80, 480)


# --------------------------------------------------------------- [5] snapshot type single producer
def test_snapshot_type_key_is_produced_only_by_the_base_class(monkeypatch):
    """The ``type`` key of every device snapshot FLOWS from ``BaseDevice.snapshot`` (no subclass
    re-types the literal): stubbing the base method changes every subclass's reported type,
    which is only possible when they all call ``super().snapshot()``."""
    devices = [
        VirtualTrapArray(grid_shape=(2, 3), image_shape=(16, 20), seed=1),
        VirtualCamera(_trap(), exposure=1e-3),
        VirtualMotCamera(),
        VirtualSequencer(),
        PylonCamera(trigger_source=SOFTWARE_TRIGGER),
        QCMOSCamera({"exposure": 1e-3}),
        ManualSequencer(channels=["trap", "emCCD"]),
        RemoteSequencer(host="192.0.2.1", port=1, channels=["trap", "emCCD"]),
    ]
    for dev in devices:                                      # the normal reading first
        snap = dev.snapshot()
        assert snap["type"] == type(dev).__name__
        assert next(iter(snap)) == "type"                    # identity leads the snapshot
    monkeypatch.setattr(BaseDevice, "snapshot", lambda self: {"type": f"SENTINEL:{type(self).__name__}"})
    for dev in devices:
        assert dev.snapshot()["type"] == f"SENTINEL:{type(dev).__name__}", (
            f"{type(dev).__name__}.snapshot() does not source its 'type' from BaseDevice.snapshot")


# --------------------------------------------------------------- [5b] DeviceProperty is the ONE source
def test_snapshot_and_observe_are_derived_from_device_properties():
    """A device's LIVE knobs are declared ONCE as :class:`DeviceProperty`; both its ``snapshot`` dump
    AND its ``read_only`` observe surface DERIVE from that one declaration -- so a NEW knob needs no
    snapshot re-list and no OBSERVE_API entry (the duplication that used to be kept in sync by hand)."""
    class _Probe(BaseDevice):
        gain = DeviceProperty("float", lo=0.0, hi=10.0, default=2.5)
        mode = DeviceProperty("choice", choices=("a", "b"), default="a")

    dev = _Probe()
    snap = dev.snapshot()
    assert next(iter(snap)) == "type"                       # identity still leads the dump
    assert snap["gain"] == 2.5 and snap["mode"] == "a"      # every knob auto-dumped (no override)
    ro = read_only(dev)
    assert ro.gain == 2.5 and ro.mode == "a"                # observe view can READ every declared knob
    with pytest.raises(PermissionError):
        ro.gain = 9.0                                       # ...but never WRITE (the read-only proxy)


def test_laser_rf_do_not_re_list_their_knobs():
    """The laser / RF no longer hand-list their DeviceProperty knob names in a snapshot override OR an
    OBSERVE_API set -- both derive from the property declarations, so the drift trap (add a knob, forget
    to also add it to the snapshot AND the observe whitelist) is structurally gone."""
    assert "snapshot" not in VirtualLaser.__dict__ and "snapshot" not in VirtualRF.__dict__
    assert "OBSERVE_API" not in LaserDevice.__dict__ and "OBSERVE_API" not in RFSourceDevice.__dict__
    for dev in (VirtualLaser(), VirtualRF()):
        snap, ro = dev.snapshot(), read_only(dev)
        knobs = list(collect_device_properties(type(dev)))
        assert knobs, f"{type(dev).__name__} should declare DeviceProperty knobs"
        for knob in knobs:                                  # every declared knob: dumped AND observable
            assert knob in snap, f"{type(dev).__name__}.snapshot dropped {knob}"
            assert getattr(ro, knob) == snap[knob], f"{type(dev).__name__} observe/snapshot disagree on {knob}"


# --------------------------------------------------------------- [6] settle scaling single home
def test_settle_scaling_is_owned_by_the_sequencer_service():
    """Both composing backends DELEGATE settle to ``SequencerService.settle`` -- the one home of
    the sleep_scale fast-forward rule -- and the scale itself has one copy (the service's):
    flipping the VirtualSequencer's ``sleep_scale`` writes through, so settle can never read a
    stale adapter-side mirror."""
    for seqr in (VirtualSequencer(sleep_scale=0.0),
                 VirtualSequencer(channels=["trap", "emCCD"], sleep_scale=0.0)):
        calls: list = []
        original = seqr.service.settle
        seqr.service.settle = lambda seconds, *, stop=None: calls.append((seconds, stop))
        try:
            ev = threading.Event()
            seqr.settle(1.25, stop=ev)
            assert calls == [(1.25, ev)], f"{type(seqr).__name__}.settle does not delegate to the service"
        finally:
            seqr.service.settle = original

    # sleep_scale=0 fast-forwards a long settle to (essentially) nothing, cancellably.
    t0 = time.monotonic()
    SequencerService(channels=["trap"], sleep_scale=0.0).settle(100.0)
    assert time.monotonic() - t0 < 1.0

    seqr = VirtualSequencer(sleep_scale=0.25)
    assert seqr.sleep_scale == seqr.service.sleep_scale == 0.25
    seqr.sleep_scale = 0.0                                   # the adapter view writes through
    assert seqr.service.sleep_scale == 0.0
    t0 = time.monotonic()
    seqr.settle(100.0)
    assert time.monotonic() - t0 < 1.0
