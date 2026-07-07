"""Runtime-control catalog + editable device viewer contracts (#4/#6).

A device DESCRIBES ITS OWN live, tunable-at-runtime attributes through
``runtime_controls()`` -- a distinct layer from the construction-time ``config_params``
(what the config editor writes into a JSON entry) and the read-only ``OBSERVE_API`` /
``snapshot``.  Each :class:`RuntimeControl` pairs a ``ParamDecl`` with a getter and an
optional setter, and the GUI renders it through the SAME ``PARAM_WIDGETS`` registry the
config form uses (typed widgets, never an ``eval`` of typed text).  These contracts pin:

1. every ``RuntimeControl`` a device declares uses a ParamDecl kind the shared
   ``PARAM_WIDGETS`` registry handles, and a read-only control declares ``setter=None``
   (``writable`` is False) -- so the panel builds every control and disables the read-only
   ones by construction;
2. a writable control's setter validates in the DEVICE (a camera's ``exposure`` setter
   rejects a non-positive value), not the GUI -- the ``_DeviceControlPage`` Apply path
   writes THROUGH the device setter and an out-of-range value is refused there;
3. the task-console "Devices" viewer is READ-ONLY: every ``_DeviceControlPage`` it builds
   has no editable controls and it cannot mutate ``session.devices`` (the full config
   editor stays the separate ``exp.device_manager()`` / ``na.device_manager()`` entry).
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt5")
from PyQt5 import QtWidgets

import Zou_lab_control.neutral_atom as na
from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS
from Zou_lab_control.neutral_atom.devices.base import (
    CameraDevice,
    RuntimeControl,
    SequencerDevice,
)


@pytest.fixture(scope="module", autouse=True)
def _app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture()
def devices():
    devs = na.load_devices("virtual")
    yield devs
    devs.close()


# --------------------------------------------------------------------- #1 catalog shape


def test_every_runtime_control_kind_is_registry_handled(devices):
    """Every ParamDecl a device's runtime-control catalog declares uses a kind the shared
    PARAM_WIDGETS registry can build -- so no control silently degrades to nothing."""
    for name, device in devices.devices.items():
        controls = device.runtime_controls()
        assert isinstance(controls, tuple), f"{name}: runtime_controls must return a tuple"
        for control in controls:
            assert isinstance(control, RuntimeControl), f"{name}: {control!r} not a RuntimeControl"
            assert control.decl.kind in PARAM_WIDGETS, (
                f"{name}.{control.decl.key}: kind {control.decl.kind!r} has no PARAM_WIDGETS handler")


def test_readonly_controls_declare_no_setter(devices):
    """A read-only read-back declares ``setter=None`` (``writable`` False); a control WITH a
    setter is writable -- so ``writable`` is the single fact a panel reads to enable/disable."""
    for name, device in devices.devices.items():
        for control in device.runtime_controls():
            if control.setter is None:
                assert not control.writable, f"{name}.{control.decl.key}: setter None but writable True"
            else:
                assert control.writable, f"{name}.{control.decl.key}: has setter but writable False"


def test_base_default_is_empty_catalog():
    """A device with no declared runtime controls returns an EMPTY catalog (the default),
    never raises -- so a new device type is safe before it declares any."""
    from Zou_lab_control.neutral_atom.devices.base import BaseDevice

    class _Bare(BaseDevice):
        pass

    assert _Bare().runtime_controls() == ()


def test_camera_and_sequencer_declare_their_standard_controls(devices):
    """The contract classes declare the standard role controls: a camera a WRITABLE exposure + a
    WRITABLE ROI (edited live like the API -- ``"full"`` clears back to the whole sensor) + a read-only
    frame-shape read-back; a sequencer read-only firing/scan read-backs (+ a writable sleep_scale on the
    virtual backend that carries the attribute)."""
    camera = devices["camera"]
    assert isinstance(camera, CameraDevice)
    cam_ctrls = {c.decl.key: c for c in camera.runtime_controls()}
    assert cam_ctrls["exposure"].writable and cam_ctrls["exposure"].decl.kind == "float"
    assert cam_ctrls["roi"].writable                           # ROI is editable like an API call
    assert not cam_ctrls["frame_shape"].writable               # derived geometry read-back

    seq = devices["sequencer"]
    assert isinstance(seq, SequencerDevice)
    seq_ctrls = {c.decl.key: c for c in seq.runtime_controls()}
    assert not seq_ctrls["firing"].writable
    assert not seq_ctrls["scan_progress"].writable
    # the virtual sequencer carries a plain settable sleep_scale -> a writable knob
    assert seq_ctrls["sleep_scale"].writable and seq_ctrls["sleep_scale"].decl.kind == "float"


# -------------------------------------------------- numeric control = scrollable spin box (base rule)


def test_every_numeric_device_control_renders_a_scrollable_spinbox(devices):
    """A DEVICE runtime control that is a NUMBER always renders as a scrollable spin box, never the
    blank "leave empty = default" line edit -- enforced AT THE BASE (``RuntimeControl`` pins its decl
    ``optional=False``), so a device author can never declare a numeric knob that degrades to a line
    edit and misses the mouse-wheel adjust (the confocal "every numeric is a spin box" rule).  A unit
    rides the spin box's read-back, not the label, so this is also where "µ/m/k/M/G" substitution lands."""
    from Zou_lab_control.frontend.param_widgets import PARAM_WIDGETS, ParamWidgetContext
    from Zou_lab_control.frontend.qt_fluent import FluentDoubleSpinBox

    ctx = ParamWidgetContext()
    checked = 0
    for name, device in devices.devices.items():
        for control in device.runtime_controls():
            if control.decl.kind not in ("float", "int"):
                continue
            assert control.decl.blank_allowed is False, (
                f"{name}.{control.decl.key}: a numeric device control must be non-blank (a spin box), "
                "not the optional line edit")
            widget = PARAM_WIDGETS[control.decl.kind].build(control.decl, control.getter(device), ctx)
            assert isinstance(widget, FluentDoubleSpinBox), (
                f"{name}.{control.decl.key}: a numeric control built a {type(widget).__name__}, not a spin box")
            checked += 1
    assert checked, "expected at least one numeric device control to verify"


def test_numeric_device_read_back_engineering_scales_its_unit(devices):
    """The device viewer's live read-back is engineering-scaled with its unit SI-prefixed (the ONE
    ``format_reading`` formatter): an RF drive of 6.8e9 Hz reads "6.8... GHz", not a raw "6800000000",
    so the "common unit substitution" is done at the base, not per read-back."""
    from Zou_lab_control.frontend.param_widgets import format_reading

    rf = devices["rf"]
    freq_ctrl = next(c for c in rf.runtime_controls() if c.decl.key == "frequency_hz")
    rf.frequency_hz = 6.834682e9
    text = format_reading(freq_ctrl.decl, freq_ctrl.getter(rf))
    assert text.endswith("GHz") and "6.83" in text and "6800000000" not in text, text


# ---------------------------------------------- DeviceProperty auto-injection (confocal ManagedProperty)


def test_runtime_controls_are_derived_from_device_properties(devices):
    """``runtime_controls`` is DERIVED from the :class:`DeviceProperty` a device declares (confocal
    ``ManagedProperty`` -> ``gui_dict``), never hand-typed: the catalog for the RF / laser is exactly
    the DeviceProperty descriptors on their class (MRO-merged), in declaration order.  So a device
    author declares a knob ONCE -- the property + its metadata -- and its control auto-injects."""
    from Zou_lab_control.neutral_atom.devices.base import collect_device_properties

    for name in ("rf", "laser"):
        device = devices[name]
        derived = [c.decl.key for c in device.runtime_controls()]
        declared = list(collect_device_properties(type(device)))
        assert derived == declared, (
            f"{name}: runtime_controls {derived} must be derived from its DeviceProperty declarations "
            f"{declared} -- not hand-written")
        assert declared, f"{name}: expected the concrete device to declare DeviceProperty knobs"


def test_device_property_covers_the_three_param_kinds(devices):
    """The three device-param kinds all auto-inject their control from a DeviceProperty declaration:
    FLOAT (a scrollable spin box), BOOL (a switch) and CHOICE (a combo).  Shown on the RF source -- a
    δ / frequency / power float, a ``drive_on`` bool, a ``waveform`` choice -- so every kind is exercised
    end to end through the ONE declare-once path (not just supported in the abstract)."""
    controls = {c.decl.key: c for c in devices["rf"].runtime_controls()}
    assert controls["two_photon_detuning_gamma"].decl.kind == "float" and controls["two_photon_detuning_gamma"].writable
    assert controls["drive_on"].decl.kind == "bool" and controls["drive_on"].writable
    wf = controls["waveform"]
    assert wf.decl.kind == "choice" and wf.decl.choices == ("sine", "square", "triangle")
    laser = {c.decl.key: c for c in devices["laser"].runtime_controls()}
    assert laser["beam_on"].decl.kind == "bool" and laser["beam_on"].writable       # a writable bool
    assert laser["on_d1"].decl.kind == "bool" and not laser["on_d1"].writable        # a derived read-back bool


def test_device_property_is_the_property_and_validates(devices):
    """A DeviceProperty IS the Python property (the descriptor unifies the two): reading / writing the
    attribute goes through it, a float clamps to its bounds, a choice rejects a value outside its
    options, and a derived read-back (no setter) refuses assignment -- so the property behaviour and its
    GUI can never drift apart."""
    rf = devices["rf"]
    rf.two_photon_detuning_gamma = -0.4                       # the bidirectional setter moves frequency
    assert abs(rf.two_photon_detuning_gamma - (-0.4)) < 1e-9
    rf.power_dbm = 999.0                                      # float clamps to its declared upper bound
    assert rf.power_dbm == 40.0
    with pytest.raises(ValueError):
        rf.waveform = "noise"                                # not one of the declared choices
    laser = devices["laser"]
    with pytest.raises(AttributeError):
        laser.on_d1 = True                                   # a derived read-back is read-only


# --------------------------------------------------------------------- #2 write-through


def test_control_page_apply_writes_through_the_device_setter(devices):
    """The ``_DeviceControlPage`` Apply path reads the widget and calls the control's setter --
    a valid value lands on the device, and an out-of-range value is refused by the DEVICE'S own
    setter (validation in the device, not the GUI)."""
    from Zou_lab_control.frontend.device_manager import _DeviceControlPage

    camera = devices["camera"]
    statuses: list = []
    page = _DeviceControlPage(camera, role="camera", on_status=statuses.append)
    assert "exposure" in page._editors, "a writable exposure control must have an editor"
    widget, handler = page._editors["exposure"]
    exposure_ctrl = next(c for c in camera.runtime_controls() if c.decl.key == "exposure")

    handler.write(widget, 0.004)
    page._apply_control(exposure_ctrl, widget, handler)
    assert camera.exposure == 0.004                            # wrote through the device setter

    handler.write(widget, 0.0)                                 # illegal: exposure must be > 0
    page._apply_control(exposure_ctrl, widget, handler)
    assert camera.exposure == 0.004                            # device setter refused; unchanged
    assert any("exposure" in s for s in statuses)              # the reason surfaced on status


# --------------------------------------------------------------------- #6 device viewer (editable)


def test_readonly_control_page_has_no_editors(devices):
    """The read-only MODE of ``_DeviceControlPage`` (``read_only=True``) builds NO editable controls --
    a pure peek that can never write, even where a control is writable (the viewer's ``editable=False``
    path)."""
    from Zou_lab_control.frontend.device_manager import _DeviceControlPage

    page = _DeviceControlPage(devices["camera"], role="camera", read_only=True)
    assert page._editors == {}, "a read-only page must have no editable controls"


def test_device_viewer_is_editable_but_not_a_config_editor():
    """The task-console device VIEWER edits each device's runtime params live (like the API) -- its
    pages are EDITABLE (writable controls get an editor), but it is NOT the config editor: it has no
    add / remove / working-config surface, and opening it never swaps the session's device set (the
    full config editor is the separate device_manager entry)."""
    from Zou_lab_control.frontend.device_manager import DeviceViewerPanel, _DeviceControlPage

    exp = na.connect("virtual")
    try:
        before = exp.devices
        panel = DeviceViewerPanel(devices_provider=lambda: exp.devices)   # editable=True default
        tab_pages = [panel._tabs.widget(i) for i in range(panel._tabs.count())]
        control_pages = [w for w in tab_pages if isinstance(w, _DeviceControlPage)]
        assert control_pages, "the viewer must render a device control page per device"
        assert all(not page._read_only for page in control_pages)        # EDITABLE like the API
        assert any(page._editors for page in control_pages)              # writable controls got editors
        # but it is a runtime-control view, NOT the config editor -- no add/remove/config surface
        for forbidden in ("_apply", "_add_entry", "_remove_entry", "working_config"):
            assert not hasattr(panel, forbidden), f"the viewer must not expose {forbidden}"
        assert exp.devices is before                           # opening the viewer swapped nothing
    finally:
        exp.close()


def test_task_console_devices_button_opens_the_device_viewer():
    """The console's Devices button routes to ``session.device_viewer`` (edit runtime params live),
    a DISTINCT window from the full ``device_manager`` config editor -- both notebook entries exist."""
    exp = na.connect("virtual")
    try:
        # both facades exist and are distinct entries
        assert callable(exp.device_viewer) and callable(exp.device_manager)
        viewer = exp.device_viewer()
        assert viewer is exp.device_viewer()                  # one-per-session singleton
        # the viewer window is NOT the manager window (distinct singletons)
        manager = exp.device_manager()
        assert viewer is not manager
        viewer.close()
        manager.close()
    finally:
        exp.close()
