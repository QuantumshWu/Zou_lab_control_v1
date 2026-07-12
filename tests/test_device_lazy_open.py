"""Device lifecycle single-source contracts (real-hardware bring-up root fixes):

* A camera opens LAZILY on first use (``arm``) via the ONE base ``ensure_open`` -- no backend
  raises "used before open" (the historical PylonCamera outlier); ``open()`` runs exactly once and
  is idempotent.
* ``DeviceSet.open()`` is FAULT-ISOLATED: a device whose ``open()`` fails no longer rolls back and
  denies the others theirs -- a dead remote sequencer must not starve an independent camera opened
  beside it (cameras open last).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import CameraDevice, SequencerDevice
from Zou_lab_control.neutral_atom.devices.registry import load_devices


class _LazyCam(CameraDevice):
    """A minimal camera whose open() acquires a 'handle' and counts calls -- to prove the base
    arm() opens it on first use and never again."""

    def __init__(self):
        self._handle = None
        self.opens = 0
        self._exposure = 1e-3

    def open(self):
        self.opens += 1
        self._handle = object()
        return self

    @property
    def is_open(self) -> bool:
        return self._handle is not None

    @property
    def exposure(self) -> float:
        return self._exposure

    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure"}, kwargs)
        if exposure is not None:
            self._exposure = float(exposure)


def test_camera_arm_lazily_opens_once_and_is_idempotent():
    cam = _LazyCam()
    assert not cam.is_open and cam.opens == 0
    cam.arm()                                    # base arm() must ensure_open FIRST -- no raise
    try:
        assert cam.is_open and cam.opens == 1, "first arm must open exactly once"
    finally:
        cam.disarm()
    cam.arm()                                    # already open -> ensure_open is a no-op
    try:
        assert cam.opens == 1, "a second arm must NOT re-open an already-open camera"
    finally:
        cam.disarm()


def test_ensure_open_is_the_single_source_no_backend_arm_raises_on_unopened():
    """The 'open before use' policy lives in ONE place (base ensure_open); a backend's _arm must
    NOT re-decide it by raising when the handle is missing (the removed pylon guard).  Pins pylon +
    qCMOS: they expose is_open and their _arm source carries no 'before open' raise."""
    from Zou_lab_control.neutral_atom.devices import pylon, qcmos
    import inspect

    for cls in [pylon.PylonCamera, qcmos.QCMOSCamera]:
        src = inspect.getsource(cls._arm)
        assert "before open" not in src, f"{cls.__name__}._arm must not raise 'before open' (base ensure_open owns it)"
    # is_open predicate is wired to the handle attribute (unopened -> False).
    cam = pylon.PylonCamera.__new__(pylon.PylonCamera)
    cam._camera = None
    cam._connection_token = None
    assert cam.is_open is False
    class _LivePylonHandle:
        def IsOpen(self):
            return True

        def IsCameraDeviceRemoved(self):
            return False

    cam._camera = _LivePylonHandle()
    assert cam.is_open is False
    cam._connection_token = object()
    assert cam.is_open is True


def test_device_set_open_isolates_failures_so_a_dead_sequencer_never_starves_the_camera():
    dev_set = load_devices("virtual", open_devices=False)
    try:
        seq_name = next(n for n, d in dev_set.devices.items() if isinstance(d, SequencerDevice))
        cam_name = next(n for n, d in dev_set.devices.items() if isinstance(d, CameraDevice))
        opened: list[str] = []
        # cameras open LAST; make the sequencer's open() fail like an unreachable RPyC server.
        dev_set.devices[seq_name].open = lambda: (_ for _ in ()).throw(RuntimeError("server unreachable"))
        cam = dev_set.devices[cam_name]
        cam.open = lambda: (opened.append(cam_name), cam)[1]
        with pytest.raises(RuntimeError) as exc:
            dev_set.open()
        assert seq_name in str(exc.value), "the combined error must name the failed device"
        assert opened == [cam_name], "the camera must still open even though the sequencer failed"
    finally:
        dev_set.close()
