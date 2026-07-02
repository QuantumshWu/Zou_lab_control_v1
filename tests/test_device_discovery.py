"""Device auto-discovery + named-camera selection -- the X-round contracts.

The chain: ``na.discover_devices()`` scans the buses (Basler / VISA, confocal-style: a missing
library or empty bus is a REPORTED ROW, never an exception) and each camera hit carries a READY
``{"type", "params"}`` config entry -> ``load_devices`` builds the device -> every "which
camera?" choice in the measurement layer derives from the ONE :meth:`DeviceSet.camera_names`
source -> the console's Camera measurement / the notebook's ``capture()`` show the image.

Pure row builders are tested with fake enumerations (no hardware); the IO shell is only
asserted never to raise.  Expected names derive from the live device set, never re-typed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.discovery import (
    DiscoveredDevice, basler_rows, discover_devices, visa_rows)
from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.registry import load_devices


class _FakeInfo:
    def __init__(self, model: str, serial: str):
        self._model, self._serial = model, serial

    def GetModelName(self):
        return self._model

    def GetSerialNumber(self):
        return self._serial


def test_basler_rows_yield_ready_pylon_configs():
    """Every enumerated Basler camera becomes a row whose config is a READY device entry:
    a PylonCamera pinned to that serial, free-running (no pulse wiring needed for first light)."""
    rows = basler_rows([_FakeInfo("acA1920-155um", "24012345"),
                        _FakeInfo("acA2440-20gm", "40098765")])
    assert [r.ident for r in rows] == ["24012345", "40098765"]
    assert rows[0].label == "acA1920-155um"
    assert rows[0].config == {"type": "PylonCamera",
                              "params": {"serial": "24012345", "trigger_source": "Software"}}


def test_discovered_config_loads_straight_into_a_device_set():
    """The discovery row's config is not a suggestion to retype -- load_devices consumes it
    verbatim and builds the pinned PylonCamera (not opened: no hardware in CI)."""
    row = basler_rows([_FakeInfo("acA1920-155um", "24012345")])[0]
    devices = load_devices({"monitor_camera": row.config}, open_devices=False)
    cam = devices["monitor_camera"]
    assert isinstance(cam, PylonCamera)
    assert cam.serial == "24012345"
    assert cam.trigger_source == "Software"


def test_visa_rows_are_informational_listings():
    """Bare VISA instruments have no driver class here -> rows list address + *IDN? identity
    with config=None (the operator wires the address into a custom class)."""
    rows = visa_rows(("GPIB0::17::INSTR",), lambda r: "Rigol,DSG836,...")
    assert rows[0].ident == "GPIB0::17::INSTR"
    assert rows[0].label.startswith("Rigol")
    assert rows[0].config is None


def test_discover_devices_never_raises_and_always_reports():
    """The confocal contract: a missing library / empty bus is a reported row, not an error.
    (On any machine -- with or without pypylon, cameras, or a VISA backend -- this returns
    at least one row per scanned bus and never throws.)"""
    rows = discover_devices(display=False)
    assert rows and all(isinstance(r, DiscoveredDevice) for r in rows)
    assert any(r.kind in ("basler", "note") for r in rows)
    assert all(str(r) for r in rows)                  # every row renders as a table line


def test_camera_names_is_the_one_choice_source():
    """DeviceSet.camera_names() lists every CameraDevice sorted -- and BOTH camera choices
    (the Camera measurement's and Pulse scan's) are exactly that tuple (single source)."""
    import matplotlib
    matplotlib.use("Agg")
    from Zou_lab_control import neutral_atom as na

    exp = na.connect("virtual")
    try:
        names = exp.devices.camera_names()
        assert names == ("camera", "monitor_camera")
        cam_spec = exp.readout.camera_spec()
        assert next(p for p in cam_spec.params if p.key == "camera").choices == names
        scan_spec = {s.name: s for s in exp.readout.measurement_specs()}["Pulse scan"]
        assert next(p for p in scan_spec.params if p.key == "camera").choices == names
    finally:
        exp.close()


def test_camera_spec_builds_the_named_camera():
    """spec.build(camera="monitor_camera") wires the measurement to THAT sensor -- the
    console's Add Panel path and the notebook's build are the same call."""
    import matplotlib
    matplotlib.use("Agg")
    from Zou_lab_control import neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = na.connect("virtual")
    try:
        node = exp.readout.camera_spec().build(SignalHub(), camera="monitor_camera")
        assert node.camera is exp.devices["monitor_camera"]
        default = exp.readout.camera_spec().build(SignalHub())
        assert default.camera is exp.devices.camera
    finally:
        exp.close()


def test_camera_spec_exposure_is_per_selected_camera():
    """Exposure is the SELECTED camera's own state: the form default is blank (None) and a
    build without an explicit exposure must NOT touch the chosen sensor's exposure.
    (Regression: the spec used to freeze the MAIN camera's exposure as the default and
    silently overwrite the monitor camera's when the choice changed.)"""
    import matplotlib
    matplotlib.use("Agg")
    from Zou_lab_control import neutral_atom as na
    from Zou_lab_control.neutral_atom.core.signals import SignalHub

    exp = na.connect("virtual")
    try:
        spec = exp.readout.camera_spec()
        assert next(p for p in spec.params if p.key == "exposure").default is None
        monitor = exp.devices["monitor_camera"]
        own = float(monitor.exposure)
        assert own != float(exp.devices.camera.exposure)   # the two sensors genuinely differ
        spec.build(SignalHub(), camera="monitor_camera")   # blank exposure
        assert float(monitor.exposure) == own              # untouched
        spec.build(SignalHub(), camera="monitor_camera", exposure=2 * own)
        assert float(monitor.exposure) == 2 * own          # explicit value applied to THAT sensor
    finally:
        exp.close()


def test_basler_monitor_config_mixes_real_monitor_into_virtual_roster():
    """The shipped ``basler_monitor`` config is the incremental bring-up step: virtual
    readout roster + a REAL free-running PylonCamera as monitor_camera (first attached
    camera; discovery prints serials to pin one)."""
    devices = load_devices("basler_monitor", open_devices=False)
    assert isinstance(devices["monitor_camera"], PylonCamera)
    assert devices["monitor_camera"].trigger_source == "Software"
    assert devices.camera_names() == ("camera", "monitor_camera")


def test_pylon_camera_construction_needs_no_pylon_runtime():
    """pypylon is imported inside open() only -- constructing / configuring the class (and
    therefore loading any config that names it) works on a machine without the Basler SDK."""
    cam = PylonCamera(serial="123", trigger_source="Line1", pixel_format="Mono12")
    snap = cam.snapshot()
    assert snap["pixel_format"] == "Mono12"
    assert snap["trigger_source"] == "Line1"
    with pytest.raises(RuntimeError, match="before open"):
        cam.acquire(1)
