"""Contracts for the Basler pylon MONITOR camera adapter (no Basler runtime needed).

A fake ``pypylon.pylon`` module is installed for the duration of each test, so the code under
test is exactly what a real run executes -- only the SDK/hardware layer is faked.  Two audit
findings are pinned:

* B2  ``open()`` replays EVERY configured setting through one path: a ROI set BEFORE open (the
      device set opens cameras last, so configure-before-open is normal) must reach the sensor,
      not silently image the full frame while ``cam.roi`` reports the window;
* B5  a Stop during a blocking ``RetrieveResult`` is honoured within ONE poll slice, not after
      the full timeout: the retrieve is sliced (min(timeout, 200 ms)) with the stop re-checked
      between slices.
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

from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera


# --------------------------------------------------------------------------- fake pypylon
class _Node:
    """A GenICam node: SetValue/GetValue plus the Min/Max/Inc grid the ROI snap consults."""

    def __init__(self, value=0, lo=0, hi=10_000, inc=1):
        self.value = int(value)
        self._lo, self._hi, self._inc = int(lo), int(hi), int(inc)
        self.writes: list[int] = []

    def SetValue(self, v):
        self.value = int(v)
        self.writes.append(int(v))

    def GetValue(self):
        return self.value

    def GetMin(self):
        return self._lo

    def GetMax(self):
        return self._hi

    def GetInc(self):
        return self._inc


class _StrNode:
    def __init__(self, value=""):
        self.value = value
        self.writes: list[str] = []

    def SetValue(self, v):
        self.value = v
        self.writes.append(v)

    def GetValue(self):
        return self.value


class _FakeCamera:
    """A pypylon InstantCamera stand-in with the nodes/methods the adapter touches."""

    def __init__(self, *, retrieve=None):
        self.Width = _Node(1920, lo=16, hi=1920, inc=4)
        self.Height = _Node(1080, lo=16, hi=1080, inc=2)
        self.OffsetX = _Node(0, lo=0, hi=1904, inc=4)
        self.OffsetY = _Node(0, lo=0, hi=1064, inc=2)
        self.WidthMax = _Node(1920)
        self.HeightMax = _Node(1080)
        self.ExposureTime = _Node(0)
        self.PixelFormat = _StrNode()
        self.TriggerSelector = _StrNode()
        self.TriggerMode = _StrNode()
        self.TriggerSource = _StrNode()
        self._grabbing = False
        self._opened = False
        self._retrieve = retrieve            # callable(timeout_ms) -> result-or-None

    def Open(self):
        self._opened = True

    def Close(self):
        self._opened = False

    def IsGrabbing(self):
        return self._grabbing

    def StartGrabbing(self, *a):
        self._grabbing = True

    def StartGrabbingMax(self, *a):
        self._grabbing = True

    def StopGrabbing(self):
        self._grabbing = False

    def RetrieveResult(self, timeout_ms, handling):
        if self._retrieve is not None:
            return self._retrieve(timeout_ms)
        return None


def _install_fake_pylon(camera):
    """Build a SELF-CONTAINED fake ``pypylon`` package (its own module object, never the real
    one) exposing exactly the names the adapter imports.  Building a fresh package -- rather than
    mutating a real ``pypylon`` module's ``.pylon`` attribute -- means the fixture restores the
    world completely by swapping the two ``sys.modules`` keys back, with no attribute residue that
    would leak the fake into later tests (the ``from pypylon import pylon`` other tests run)."""
    factory = types.SimpleNamespace(CreateFirstDevice=lambda: object(),
                                    EnumerateDevices=lambda: [])
    pylon_ns = types.ModuleType("pypylon.pylon")
    pylon_ns.TlFactory = types.SimpleNamespace(GetInstance=lambda: factory)
    pylon_ns.InstantCamera = lambda _dev: camera
    pylon_ns.GrabStrategy_LatestImageOnly = "latest"
    pylon_ns.GrabStrategy_OneByOne = "onebyone"
    pylon_ns.TimeoutHandling_Return = "return"
    pkg = types.ModuleType("pypylon")           # a FRESH package, never the installed one
    pkg.pylon = pylon_ns
    return pylon_ns, pkg


@pytest.fixture
def fake_pylon():
    """Swap ``pypylon`` (and ``pypylon.pylon``) for a self-contained fake for the test, restoring
    the real modules afterward so no fake leaks into later tests."""
    saved = {k: sys.modules.get(k) for k in ("pypylon", "pypylon.pylon")}

    def factory(camera):
        pylon_ns, pkg = _install_fake_pylon(camera)
        sys.modules["pypylon"] = pkg
        sys.modules["pypylon.pylon"] = pylon_ns
        return pylon_ns

    try:
        yield factory
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# --------------------------------------------------------------------------- B2
def test_open_applies_roi_configured_before_open(fake_pylon):
    """B2: a ROI set while the camera is still closed (configure-before-open -- the device set
    opens cameras LAST) is pushed to the sensor by open(), through the same apply path a later
    reconfigure uses.  Without it the stream runs full-frame while ``cam.roi`` lies."""
    camera = _FakeCamera()
    fake_pylon(camera)

    cam = PylonCamera(serial="", trigger_source="Software")
    cam.configure(roi=(100, 640, 80, 480))       # BEFORE open: no camera yet, just stored
    assert not camera.Width.writes                # nothing reached the sensor yet
    cam.open()

    # open() must have pushed the window to the hardware (snapped to Width/Height Inc).
    assert camera.Width.writes, "open() never applied the pre-configured ROI to the sensor"
    assert camera.Width.value == 640              # 640 is a multiple of Width.Inc(4)
    assert camera.Height.value == 480             # 480 is a multiple of Height.Inc(2)
    assert camera.OffsetX.value == 100            # 100 is a multiple of OffsetX.Inc(4)
    assert camera.OffsetY.value == 80
    # and the adapter reports the granted window, consistent with the hardware.
    assert cam.roi == (100, 640, 80, 480)


def test_reopen_reapplies_roi(fake_pylon):
    """A close -> reopen must re-push the ROI: the second open goes through the same single
    apply-all path, so the reopened stream is NOT silently full-frame."""
    camera = _FakeCamera()
    fake_pylon(camera)
    cam = PylonCamera(trigger_source="Software")
    cam.open()
    cam.configure(roi=(100, 640, 80, 480))
    cam.close()

    camera2 = _FakeCamera()
    fake_pylon(camera2)
    cam._camera = None                            # simulate the device being reopened fresh
    cam.open()
    assert camera2.Width.value == 640 and camera2.Height.value == 480


# --------------------------------------------------------------------------- B5
def test_grab_honours_stop_within_one_slice(fake_pylon):
    """B5: a Stop pressed during a blocking RetrieveResult is honoured within ~one poll slice
    (<= 200 ms), NOT after the full (here 30 s) timeout -- the retrieve is sliced and the stop
    re-checked between slices.  Asserted by timing: the read returns promptly after the stop."""
    slice_calls: list[int] = []

    def retrieve(timeout_ms):
        # Model a wedged trigger: each slice waits its slice_ms and returns no frame.
        slice_calls.append(int(timeout_ms))
        time.sleep(min(int(timeout_ms), 200) / 1000.0)
        return None

    camera = _FakeCamera(retrieve=retrieve)
    fake_pylon(camera)
    cam = PylonCamera(trigger_source="Line1", timeout=30.0)
    cam.open()

    stop = threading.Event()
    cam.arm(1)
    result = {}

    def do_read():
        t0 = time.monotonic()
        frames = cam.read_frames(1, timeout=30.0, stop=stop)
        result["elapsed"] = time.monotonic() - t0
        result["frames"] = frames

    worker = threading.Thread(target=do_read)
    worker.start()
    time.sleep(0.05)
    stop.set()                                    # press Stop mid-wait
    worker.join(timeout=3.0)
    cam.disarm()

    assert not worker.is_alive(), "read_frames sat inside RetrieveResult past the stop"
    assert result["elapsed"] < 2.0, f"Stop took {result['elapsed']:.2f}s (should be ~one slice)"
    assert result["frames"] == []                 # no frame -> partial read (pylon soft fault)
    # Each slice was bounded (never the full 30 s), which is what makes the stop responsive.
    assert slice_calls and all(ms <= 200 for ms in slice_calls)
