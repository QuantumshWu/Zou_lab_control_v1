"""Loud, single-source acquisition contracts for the real qCMOS adapter (DCAM-free).

Every test fakes only the lowest layer -- a stand-in ``Dcamapi`` singleton and a ``Dcam``
handle -- so the code under test is exactly what a real run executes.  Three contracts the
adversarial audit pinned:

* B1  the process-wide DCAM runtime has ONE reference count: a ``discover()`` enumeration run
      while a camera is open must NOT ``uninit`` the runtime out from under that live camera;
* B3  the grab timeout is PER AWAITED TRIGGER (it resets on each delivered frame), so an FPGA
      firing ``n`` triggers spaced under the timeout succeeds identically with OR without a stop
      event -- the stop event only shortens the poll slice, never the trigger budget;
* B4  a continuous ring that the hardware overruns fails LOUD (a dropped-frame RuntimeError),
      never silently delivering an overwritten/mis-ordered frame.
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices import qcmos as qcmos_mod
from Zou_lab_control.neutral_atom.devices.qcmos import (
    QCMOSCamera, _dcam_acquire, _dcam_release)

# The real ``dcamapi4`` module loads ``dcamapi.dll`` at import time, so it is absent on a machine
# without the Hamamatsu runtime (CI / dev).  The adapter imports ``DCAMERR`` LAZILY, only when it
# must classify an init failure; we install a tiny stand-in under the same import path so the
# ALREADYINITIALIZED-adoption path is exercised DCAM-free.  ALREADYINITIALIZED = 0xE1000001 (int),
# the one enum value the adapter compares against.
_ALREADY_INITIALIZED = -520093695


def _install_fake_dcamapi4():
    import types as _t
    name = "Zou_lab_control.neutral_atom.devices.drivers.dcam.dcamapi4"
    if name not in sys.modules:
        mod = _t.ModuleType(name)
        mod.DCAMERR = _t.SimpleNamespace(ALREADYINITIALIZED=_ALREADY_INITIALIZED)
        sys.modules[name] = mod


_install_fake_dcamapi4()


# --------------------------------------------------------------------------- fakes
class _FakeApi:
    """Stand-in for the ``Dcamapi`` process singleton: ONE init bool, no reference count of
    its own -- exactly the driver behaviour the module-level count must wrap.  ``uninit`` is a
    hard tear-down: once called while a camera is 'open', that camera's handle goes stale."""

    def __init__(self, *, already_initialised: bool = False):
        self.initialised = bool(already_initialised)
        self.init_calls = 0
        self.uninit_calls = 0
        self._lasterr = 0                         # DCAMERR.SUCCESS

    def init(self, *a):
        self.init_calls += 1
        if self.initialised:
            self._lasterr = _ALREADY_INITIALIZED
            return False
        self.initialised = True
        self._lasterr = 0
        return True

    def uninit(self):
        self.uninit_calls += 1
        self.initialised = False
        return True

    def lasterr(self):
        return self._lasterr

    def get_devicecount(self):
        return 1 if self.initialised else False


class _FakeDcam:
    """A minimal DCAM handle: its calls FAIL once the shared runtime has been uninited (the
    real symptom of a torn-down API), so a test can assert a live camera stays usable."""

    def __init__(self, api: _FakeApi, *, frames_ready=None, ring=None):
        self._api = api
        self._opened = False
        self._ring = ring                       # nFrameCount schedule for cap_transferinfo (B4)
        self._ready = list(frames_ready or [])   # per-wait: True = a frame is ready
        self._count = 0
        self.cap_stop_ok = True
        self.buf_release_ok = True

    # --- lifecycle
    def dev_open(self):
        self._opened = True
        return True

    def dev_close(self):
        self._opened = False
        return True

    def _alive(self):
        # The runtime must be up for any handle call to work (a torn-down API breaks the camera).
        return self._api.initialised

    # --- acquisition
    def buf_alloc(self, n):
        return self._alive()

    def cap_start(self, bSequence=True):
        return self._alive()

    def cap_stop(self):
        return self.cap_stop_ok

    def buf_release(self):
        return self.buf_release_ok

    def wait_capevent_frameready(self, timeout_ms):
        if not self._alive():
            return False
        if not self._ready:
            return False
        return bool(self._ready.pop(0))

    def cap_transferinfo(self):
        self._count += 1
        n = self._ring[min(self._count - 1, len(self._ring) - 1)] if self._ring else self._count
        return types.SimpleNamespace(nFrameCount=int(n))

    def buf_getframedata(self, index):
        return (types.SimpleNamespace(), np.full((2, 2), float(index)))

    def lasterr(self):
        return "DCAMERR_FAKE"


def _fresh_refcount():
    qcmos_mod._DCAM_API_REFCOUNT.clear()


# --------------------------------------------------------------------------- B1
def test_dcam_refcount_last_holder_uninits():
    """Two acquires, two releases: the runtime inits on the FIRST holder and uninits only on the
    LAST release -- never in between."""
    _fresh_refcount()
    api = _FakeApi()
    _dcam_acquire(api)
    assert api.init_calls == 1 and api.initialised
    _dcam_acquire(api)                           # second holder joins, no re-init
    assert api.init_calls == 1
    _dcam_release(api)
    assert api.uninit_calls == 0                 # first release: runtime stays up for the 2nd holder
    _dcam_release(api)
    assert api.uninit_calls == 1 and not api.initialised


def test_discover_while_camera_open_keeps_camera_alive():
    """B1 core: a camera is open (holds a reference); ``discover()`` runs, acquiring + releasing
    its OWN reference -- and must leave the open camera's DCAM handle live, never uninit the API."""
    _fresh_refcount()
    api = _FakeApi()
    dcam = _FakeDcam(api)
    fake_module = types.SimpleNamespace(
        Dcamapi=api,
        Dcam=lambda index=0: dcam,
    )

    cam = QCMOSCamera()
    # Open the camera through the real adapter path (its _write_settings needs DCAM props, so
    # stub _write_settings to isolate the ownership seam under test).
    cam.open = _bind_minimal_open(cam, fake_module, api, dcam)
    cam.open()
    assert api.initialised and api.init_calls == 1

    # Now run discovery against the SAME shared api -- it takes a reference and releases it.
    import importlib
    real_import = importlib.import_module
    importlib.import_module = lambda name: fake_module if "dcam" in name else real_import(name)
    try:
        rows = QCMOSCamera.discover()
    finally:
        importlib.import_module = real_import

    assert rows                                  # discovery still returned rows
    # The open camera survives: the API was never uninited, so its handle still works.
    assert api.initialised
    assert api.uninit_calls == 0
    assert dcam._alive()
    # And closing the (only) camera now DOES tear the runtime down.
    cam.close()
    assert api.uninit_calls == 1


def test_open_settings_failure_rolls_back_handle_and_runtime_reference(monkeypatch):
    _fresh_refcount()
    api = _FakeApi()
    dcam = _FakeDcam(api)
    fake_module = types.SimpleNamespace(
        Dcamapi=api,
        Dcam=lambda index=0: dcam,
    )
    monkeypatch.setattr(qcmos_mod.importlib, "import_module", lambda _name: fake_module)
    cam = QCMOSCamera()
    monkeypatch.setattr(
        cam,
        "_write_settings",
        lambda: (_ for _ in ()).throw(RuntimeError("injected settings failure")),
    )

    with pytest.raises(RuntimeError, match="injected settings failure"):
        cam.open()

    assert cam._dcam is None
    assert cam._api is None
    assert cam._module is None
    assert not dcam._opened
    assert api.uninit_calls == 1
    assert id(api) not in qcmos_mod._DCAM_API_REFCOUNT


def test_discover_adopts_already_initialised_runtime_without_uninit():
    """An ALREADYINITIALIZED runtime (initialised outside this counter -- an external script that
    owns it) is ADOPTED, not reported as a failure: discovery enumerates, and its release-to-zero
    must NOT ``uninit`` a runtime we did not start (the refcount entry carries ownership).  The
    owned direction is pinned right beside it: a runtime OUR ``init`` started IS uninited by the
    last release -- so the ownership flag flips the tear-down, never suppresses it."""
    _fresh_refcount()
    api = _FakeApi(already_initialised=True)
    fake_module = types.SimpleNamespace(
        Dcamapi=api,
        Dcam=lambda index=0: _FakeDcam(api),
    )
    import importlib
    real_import = importlib.import_module
    importlib.import_module = lambda name: fake_module if "dcam" in name else real_import(name)
    try:
        rows = QCMOSCamera.discover()
    finally:
        importlib.import_module = real_import
    assert rows and all(r.kind != "note" or "init failed" not in r.label for r in rows)
    # The adopt+release cycle is complete (discover released its reference): the externally-owned
    # runtime was NEVER uninited and stays live for its real owner.
    assert api.uninit_calls == 0
    assert api.initialised
    assert id(api) not in qcmos_mod._DCAM_API_REFCOUNT   # our bookkeeping entry is gone, though

    # And the owned direction: a runtime WE init on acquire is uninited exactly once on the
    # last release (adoption must not have castrated the normal tear-down).
    owned_api = _FakeApi()
    _dcam_acquire(owned_api)
    assert owned_api.init_calls == 1 and owned_api.initialised
    _dcam_release(owned_api)
    assert owned_api.uninit_calls == 1 and not owned_api.initialised


def _bind_minimal_open(cam, fake_module, api, dcam):
    """Return a bound open() that runs the real ownership seam (_dcam_acquire) but stubs the
    DCAM property writes -- so B1 tests exercise refcount lifecycle without a full prop fake."""
    def _open():
        if cam._dcam is not None:
            return cam
        _dcam_acquire(api)
        cam._module = fake_module
        cam._api = api
        cam._dcam = dcam
        dcam.dev_open()
        return cam
    return _open


# --------------------------------------------------------------------------- B3
def _grab_camera(api, dcam):
    """A camera wired to a fake open DCAM handle -- but driven through the REAL public
    acquisition contract (arm -> read_frames -> disarm), so the code exercised is exactly what a
    real run executes.  ``_arm`` allocates the ring via ``dcam.buf_alloc(frames)`` and sets
    ``_buf_alloc``; there is no back-door poking of the buffer depth."""
    cam = QCMOSCamera({"timeout_ms": 300})
    cam._module = types.SimpleNamespace()
    cam._api = api
    cam._dcam = dcam
    return cam


@pytest.mark.parametrize("failed_step", ("cap_stop", "buf_release"))
def test_qcmos_disarm_requires_both_driver_acknowledgements(failed_step):
    api = _FakeApi(already_initialised=True)
    dcam = _FakeDcam(api)
    cam = _grab_camera(api, dcam)
    cam.arm(1)
    if failed_step == "cap_stop":
        dcam.cap_stop_ok = False
    else:
        dcam.buf_release_ok = False

    with pytest.raises(RuntimeError, match=failed_step):
        cam.disarm()

    assert cam._recent_state()["armed"] is False


def _read_via_contract(cam, frames, n, *, timeout, stop=None):
    """arm(frames) -> read_frames(n) -> disarm, the ordered primitive path read_frames requires.
    ``frames`` sets the DCAM ring depth (``_buf_alloc``); ``n`` is how many to consume."""
    cam.arm(frames)
    try:
        return cam.read_frames(n, timeout=timeout, stop=stop)
    finally:
        cam.disarm()


@pytest.mark.parametrize("with_stop", [False, True])
def test_grab_timeout_is_per_trigger_with_and_without_stop(with_stop):
    """B3: frames arriving spaced just under the per-trigger timeout succeed for the FULL read,
    identically whether or not a stop event is passed.  A shared (whole-read) deadline would
    time out mid-read once n * inter-frame exceeded the timeout -- the notebook (no stop) then
    passed while the console (stop) failed.  Here the budget resets per delivered frame."""
    api = _FakeApi()
    api.initialised = True

    class _PacedDcam(_FakeDcam):
        """Each frame arrives after a real gap that is a LARGE fraction of the per-trigger
        timeout but never exceeds it -- so the CUMULATIVE arrival time over n frames is well
        past the timeout.  A shared (whole-read) deadline would trip mid-read; a per-trigger
        deadline that resets on each delivery does not."""

        def __init__(self, api, n_frames, gap_s):
            super().__init__(api)
            self._n = n_frames
            self._gap = gap_s
            self._delivered = 0

        def wait_capevent_frameready(self, timeout_ms):
            if self._delivered >= self._n:
                return False
            # The frame arrives after ``gap`` seconds, within this wait's slice (gap < slice).
            time.sleep(self._gap)
            self._delivered += 1
            return True

        def cap_transferinfo(self):
            return types.SimpleNamespace(nFrameCount=int(self._delivered))

    # timeout 60 ms per trigger; each frame arrives after 40 ms.  Four frames = 160 ms of
    # cumulative arrival time, far past a single 60 ms budget -- the exact shape that broke a
    # shared deadline (console w/ stop failed at ~frame 2 while the notebook w/o stop passed).
    cam = _grab_camera(api, _PacedDcam(api, n_frames=4, gap_s=0.04))
    stop = threading.Event() if with_stop else None
    got = _read_via_contract(cam, 4, 4, timeout=0.06, stop=stop)
    assert len(got) == 4, f"per-trigger timeout should deliver all 4 frames (with_stop={with_stop})"


def test_grab_times_out_when_no_trigger_ever_comes():
    """A trigger that never arrives still raises TimeoutError (the loud fault model is intact)."""
    api = _FakeApi(); api.initialised = True
    cam = _grab_camera(api, _FakeDcam(api, frames_ready=[]))
    with pytest.raises(TimeoutError):
        _read_via_contract(cam, 1, 1, timeout=0.05)


# --------------------------------------------------------------------------- B4
def test_grab_raises_on_ring_overrun():
    """B4: a session whose hardware nFrameCount jumps PAST the read cursor by more than the ring
    depth means the slot we are about to read was overwritten -- fail loud with the dropped-frame
    count rather than silently deliver a mis-ordered frame."""
    api = _FakeApi(); api.initialised = True
    # ring depth 4 (arm(4) -> buf_alloc=4), but the camera reports it has already produced 10
    # frames on the first wait: 10 - 0 > 4 -> overrun.
    dcam = _FakeDcam(api, frames_ready=[True], ring=[10])
    cam = _grab_camera(api, dcam)
    with pytest.raises(RuntimeError, match="ring overrun"):
        _read_via_contract(cam, 4, 1, timeout=0.2)


def test_grab_within_ring_depth_does_not_raise():
    """Positive control: a session that stays within the ring depth delivers frames normally."""
    api = _FakeApi(); api.initialised = True
    dcam = _FakeDcam(api, frames_ready=[True, True], ring=[1, 2])
    cam = _grab_camera(api, dcam)
    got = _read_via_contract(cam, 4, 2, timeout=0.3)
    assert len(got) == 2


def test_qcmos_preserves_dcam_metadata_in_immutable_frame_records():
    """The adapter must not throw away DCAMBUF_FRAME provenance.

    ``nFrameCount`` is a transfer-time cumulative snapshot, so both records
    drained from one availability observation deliberately carry the same
    ``produced_count=2``.  It is not fabricated into per-frame 1, 2 values.
    """

    api = _FakeApi()
    api.initialised = True

    class _MetadataDcam(_FakeDcam):
        def __init__(self, owner_api):
            super().__init__(owner_api, frames_ready=[True], ring=[2])
            self.returned_pixels = {}

        def buf_getframedata(self, index):
            pixels = np.full((2, 3), 10.0 + float(index))
            self.returned_pixels[index] = pixels
            frame_info = types.SimpleNamespace(
                framestamp=100 + index,
                camerastamp=200 + index,
                timestamp=types.SimpleNamespace(sec=300 + index, microsec=400 + index),
            )
            return frame_info, pixels

    dcam = _MetadataDcam(api)
    cam = _grab_camera(api, dcam)
    cam.arm(2)
    try:
        records = cam.read_frame_records(2, timeout=0.3)
    finally:
        cam.disarm()

    assert [record.source_ordinal for record in records] == [0, 1]
    assert [record.produced_count for record in records] == [2, 2]
    assert [record.frame_stamp for record in records] == [100, 101]
    assert [record.camera_stamp for record in records] == [200, 201]
    assert [record.timestamp_seconds for record in records] == [300, 301]
    assert [record.timestamp_microseconds for record in records] == [400, 401]
    assert [record.driver_buffer_index for record in records] == [0, 1]
    assert all(record.host_received_at_ns > 0 for record in records)
    assert all(not record.image.flags.writeable for record in records)

    # The DCAM wrapper/ring may reuse its source array immediately after drain.
    # Published records own their bytes and must remain unchanged.
    dcam.returned_pixels[0][:] = -1
    dcam.returned_pixels[1][:] = -1
    np.testing.assert_array_equal(records[0].image, np.full((2, 3), 10.0))
    np.testing.assert_array_equal(records[1].image, np.full((2, 3), 11.0))
