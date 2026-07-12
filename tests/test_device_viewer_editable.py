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
from conftest import raw_device_set

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
        cam = raw_device_set(exp).camera
        cam_ctrls = {c.decl.key: c for c in cam.runtime_controls()}
        assert cam_ctrls["exposure"].writable and cam_ctrls["roi"].writable   # basic camera knobs edit like the API
        cam_ctrls["roi"].setter(cam, "10, 40, 12, 40")
        assert cam.roi is not None                                            # the write took (device validated it)
        cam_ctrls["roi"].setter(cam, "full")
        assert cam.roi is None                                                # 'full' clears -- not "leave unchanged"

        seq_ctrls = {c.decl.key: c for c in raw_device_set(exp).sequencer.runtime_controls()}
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


def test_live_switch_writes_on_edit_without_apply(qt):
    """The Live switch = confocal's scroll-to-set mode.  With Live OFF (default) an edit to the Set
    widget is BUFFERED (only Apply writes it); with Live ON, every edit -- a mouse-wheel tick on the
    spin box -- writes to the device immediately, and engaging Live commits the shown value.  The
    same widget both ways, so there is one write path (the gated ``instant_apply``)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import _DeviceControlPage
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        cam = raw_device_set(exp).camera
        page = _DeviceControlPage(cam, role="camera")
        widget, handler = page._editors["exposure"]
        switch = page._live_switches["exposure"]

        start = float(cam.exposure)
        handler.write(widget, start * 2)          # edit while Live is OFF -> buffered, not written
        assert float(cam.exposure) == start       # the device did not move

        switch.setChecked(True)                   # engage Live -> commits the value shown now
        assert float(cam.exposure) == start * 2
        handler.write(widget, start * 3)          # now each edit (e.g. a wheel tick) writes live
        page._flush_live()                        # drain the rate limiter's trailing edge (no loop pump)
        assert float(cam.exposure) == start * 3

        switch.setChecked(False)                  # leaving Live re-buffers
        handler.write(widget, start * 4)
        page._flush_live()
        assert float(cam.exposure) == start * 3   # unchanged until Apply
    finally:
        exp.close()


def test_live_writes_are_rate_limited_leading_plus_trailing(qt):
    """Live "scroll to set" writes are RATE-LIMITED so a fast mouse-wheel scroll can not flood a
    blocking device write: many rapid edits in one window produce FAR fewer device writes than edits,
    the FIRST edit applies immediately (leading -- responsive), and after the window flushes the device
    holds the FINAL scrolled value (trailing -- never lost)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import _DeviceControlPage
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        cam = raw_device_set(exp).camera
        page = _DeviceControlPage(cam, role="camera")
        widget, handler = page._editors["exposure"]
        page._live_switches["exposure"].setChecked(True)      # Live ON

        base = float(cam.exposure)
        rapid = [base * (1.0 + 0.01 * i) for i in range(1, 21)]   # 20 rapid "wheel ticks" in ONE window
        observed = []
        for v in rapid:
            handler.write(widget, v)                          # a wheel tick (no event-loop pump here)
            observed.append(float(cam.exposure))

        assert observed[0] == rapid[0]              # leading edge: the FIRST tick applied at once
        assert set(observed) == {rapid[0]}          # rate-limited: every later tick was coalesced (no write)
        page._flush_live()                          # close the window
        assert float(cam.exposure) == rapid[-1]     # trailing edge: the FINAL scrolled value lands
    finally:
        exp.close()


def test_current_and_set_rows_share_a_right_edge(qt):
    """A control card's Current read-back and its Set editor use the SAME ``FluentSettingRow``
    geometry, so the two value boxes line up (the alignment the inline Apply used to break -- Apply
    now lives on its own action row below)."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import _DeviceControlPage
    from Zou_lab_control.frontend.qt_fluent import FluentReadoutEdit, FluentDoubleSpinBox
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        page = _DeviceControlPage(raw_device_set(exp).camera, role="camera")
        page.resize(1000, 760)
        page.show()
        widget, _ = page._editors["exposure"]
        # the Current read-back for exposure is the first polled getter's edit
        current = next(edit for control, edit in page._getters if control.decl.key == "exposure")
        assert isinstance(current, FluentReadoutEdit) and isinstance(widget, FluentDoubleSpinBox)
        right = lambda w: w.mapToGlobal(w.rect().topRight()).x()
        assert right(current) == right(widget)    # one right edge -> aligned
    finally:
        exp.close()


def test_device_viewer_editable_flag_toggles_write_rows(qt):
    """``DeviceViewerPanel(editable=True)`` (the default) builds editable pages + a status line for
    rejected writes; ``editable=False`` builds the read-only peek -- the ONE renderer, two modes."""
    import Zou_lab_control.neutral_atom as na
    from Zou_lab_control.frontend.device_manager import DeviceViewerPanel, _DeviceControlPage
    exp = na.connect("virtual", sitemap={"grid_shape": (3, 4)})
    try:
        editable = DeviceViewerPanel(raw_device_set(exp), editable=True)
        pages = editable._tabs.findChildren(_DeviceControlPage)
        assert pages and all(not p._read_only for p in pages)
        assert editable._status is not None                     # a status line to surface a rejected write

        readonly = DeviceViewerPanel(raw_device_set(exp), editable=False)
        ro_pages = readonly._tabs.findChildren(_DeviceControlPage)
        assert ro_pages and all(p._read_only for p in ro_pages)
        assert readonly._status is None
    finally:
        exp.close()
