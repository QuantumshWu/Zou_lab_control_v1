"""The device viewer lets an operator EDIT a device's basic runtime parameters live, like an API call.

Two invariants: (1) each device declares its WRITABLE runtime knobs (a camera's exposure / ROI, qCMOS
readout speed, the sequencer's sleep scale), and a write routes through the device's OWN validated setter
(configure/property) -- the GUI never re-implements validation; (2) ``DeviceViewerPanel(editable=True)``
renders those as editable Set/Apply rows (``editable=False`` is a read-only peek) -- the SAME
``_DeviceControlPage`` the manager's Control tab uses, so there is ONE device-control renderer.
"""

import os
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def qt():
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from Zou_lab_control.frontend.qt_fluent import ensure_qt_app
    ensure_qt_app()


def test_devices_declare_writable_runtime_controls():
    """Each device's ``runtime_controls`` marks the basic params a user edits like an API call as
    WRITABLE (a setter), and the write goes through the device's own validated path -- so a camera's
    ROI round-trips (set snaps via the device's normalize_roi; 'full' clears back to the whole sensor)."""
    import Zou_lab_control.neutral_atom as na
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        cam = exp.devices.camera
        cam_ctrls = {c.decl.key: c for c in cam.runtime_controls()}
        assert cam_ctrls["exposure"].writable and cam_ctrls["roi"].writable   # basic camera knobs edit like the API
        cam_ctrls["roi"].setter(cam, "10, 40, 12, 40")
        assert cam.roi is not None                                            # the write took (device validated it)
        cam_ctrls["roi"].setter(cam, "full")
        assert cam.roi is None                                                # 'full' clears -- not "leave unchanged"

        seq_ctrls = {c.decl.key: c for c in exp.devices.sequencer.runtime_controls()}
        assert seq_ctrls["sleep_scale"].writable       # the virtual sequencer's one live knob
    finally:
        exp.close()


def test_qcmos_adds_writable_readout_speed():
    """A backend with an extra live knob extends the shared camera controls: qCMOS adds a WRITABLE
    readout_speed whose setter routes through configure(readout_speed=...) (the device validates)."""
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera
    cam = QCMOSCamera(dcam_module="dcam_stub_absent")   # never opened -> configure just updates config
    ctrls = {c.decl.key: c for c in cam.runtime_controls()}
    assert ctrls["readout_speed"].writable and ctrls["readout_speed"].decl.kind == "choice"
    ctrls["readout_speed"].setter(cam, "2")
    assert ctrls["readout_speed"].getter(cam) == "2"


def test_device_viewer_editable_flag_toggles_write_rows(qt):
    """``DeviceViewerPanel(editable=True)`` (the default) builds editable pages + a status line for
    rejected writes; ``editable=False`` builds the read-only peek -- the ONE renderer, two modes."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import DeviceViewerPanel, _DeviceControlPage
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        editable = DeviceViewerPanel(exp.devices, editable=True)
        pages = editable._tabs.findChildren(_DeviceControlPage)
        assert pages and all(not p._read_only for p in pages)
        assert editable._status is not None                     # a status line to surface a rejected write

        readonly = DeviceViewerPanel(exp.devices, editable=False)
        ro_pages = readonly._tabs.findChildren(_DeviceControlPage)
        assert ro_pages and all(p._read_only for p in ro_pages)
        assert readonly._status is None
    finally:
        exp.close()
