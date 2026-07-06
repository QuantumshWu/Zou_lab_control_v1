"""FULL_FRAME is the ONE ROI-clear sentinel every camera ``configure()`` honours (B1).

``configure(roi=None)`` means "leave unchanged" (so a multi-field configure can skip the ROI),
which is exactly why the measurement layer -- mapping a blank region box to None -- could never
clear an ROI: the per-backend ``("", "None")`` string sentinels sat INSIDE the ``roi is not None``
branch, an unreachable chain.  ``devices.base.FULL_FRAME`` is now the canonical clear sentinel
(with the GUI's natural blank spellings accepted as equivalents -- ``ROI_CLEAR_SENTINELS`` is the
single source both backends and this test read; the literal is never re-typed here).

Pinned per backend: set an ROI, ``roi=None`` leaves it, the sentinel clears it back to the full
sensor.  The virtual cameras run the same contract path real hardware does (``load_devices``);
the qCMOS/pylon adapters are exercised closed (no runtime needed) -- their configure stores the
cleared ROI through the same code path an open camera pushes to hardware.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import FULL_FRAME, ROI_CLEAR_SENTINELS
from Zou_lab_control.neutral_atom.devices.registry import load_devices


def test_full_frame_is_a_canonical_clear_sentinel():
    """The canonical spelling is a member of the ONE accepted-spellings source -- a backend
    testing membership in ROI_CLEAR_SENTINELS therefore always honours FULL_FRAME."""
    assert FULL_FRAME in ROI_CLEAR_SENTINELS


@pytest.mark.parametrize("sentinel", ROI_CLEAR_SENTINELS)
@pytest.mark.parametrize("name", ["camera", "monitor_camera"])
def test_virtual_cameras_clear_roi_only_via_sentinel(name, sentinel):
    """VirtualCamera / VirtualMotCamera: a set ROI survives ``roi=None`` (leave unchanged) and
    clears back to the full sensor (``roi is None``) on every accepted sentinel spelling."""
    cam = load_devices("virtual", open_devices=False).devices[name]
    cam.configure(roi=(4, 16, 4, 16))
    applied = cam.roi
    assert applied is not None
    cam.configure(roi=None)                       # None = leave unchanged, never a clear
    assert cam.roi == applied
    cam.configure(roi=sentinel)                   # the sentinel clears to full frame
    assert cam.roi is None


def test_closed_qcmos_and_pylon_accept_full_frame():
    """The real adapters honour the same sentinel through their (hardware-free) configure path:
    a stored ROI survives ``roi=None`` and clears on FULL_FRAME -- so the measurement layer's
    one canonical spelling works identically on every backend."""
    from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    qcmos = QCMOSCamera({"roi": (0, 8, 0, 8)})
    qcmos.configure(roi=None)
    assert qcmos.roi == (0, 8, 0, 8)
    qcmos.configure(roi=FULL_FRAME)
    assert qcmos.roi is None

    basler = PylonCamera()
    basler.configure(roi=(0, 8, 0, 8))
    assert basler.roi is not None
    basler.configure(roi=None)
    assert basler.roi is not None
    basler.configure(roi=FULL_FRAME)
    assert basler.roi is None


def test_frame_shape_derives_from_roi_else_sensor():
    """``frame_shape`` is DERIVED, never stored: the ROI window's (height, width) when a
    sub-array is set, else the full sensor.  It is the BUILD-time fact a CameraMeasurement
    seeds its ``data_shape`` from -- so a panel can render a lingering hub block the moment
    the measurement (re)starts, before any trigger fires."""
    cam = load_devices("virtual", open_devices=False).devices["camera"]
    assert cam.frame_shape == cam.sensor_shape          # no ROI -> the full sensor (H, W)
    cam.configure(roi=(4, 16, 8, 32))                   # (x, width, y, height)
    _x, w, _y, h = cam.roi                              # the SNAPPED window the camera reports
    assert cam.frame_shape == (h, w)
    cam.configure(roi=FULL_FRAME)
    assert cam.frame_shape == cam.sensor_shape


def test_camera_measurement_declares_data_shape_at_build():
    """The node's structure contract is TRUE the moment it is built (no first-frame wait):
    ``data_shape`` seeds from the camera's own ``frame_shape``.  A restarted measurement
    whose trigger is not firing yet used to declare data_shape=() -- and every 2-D panel
    bound to frame_0 refused the hub's lingering block."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), exp.devices.camera)
        assert node.data_shape == exp.devices.camera.frame_shape   # declared BEFORE any frame
    finally:
        exp.close()


def test_measurement_layer_blank_region_clears_the_roi_end_to_end():
    """#B1 end-to-end: the GUI/measurement path -- ``CameraMeasurement.set_acquisition_parameters``
    with a BLANK region -- must actually clear the camera ROI back to the full sensor.  The old
    mapping sent ``roi=None``, which every backend reads as 'leave unchanged', so the clear was
    dead from this path; it now sends the one FULL_FRAME sentinel."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub
    from Zou_lab_control.neutral_atom.operations.logic import CameraMeasurement

    exp = na.connect("virtual")
    try:
        node = CameraMeasurement(SignalHub(), exp.devices.camera)
        node.set_acquisition_parameters(region=[8, 24, 8, 24])
        assert exp.devices.camera.roi is not None            # a real sub-array was applied
        node.set_acquisition_parameters(region="")           # blank box = back to full frame
        assert exp.devices.camera.roi is None
    finally:
        exp.close()
