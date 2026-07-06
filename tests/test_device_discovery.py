"""Device auto-discovery + named-camera selection + user-owned hardware contracts.

The chain: ``na.discover_devices()`` scans the buses (confocal-style: a missing library or
empty bus is a REPORTED ROW, never an exception) and each camera hit carries a READY
``{"type", "params"}`` config entry -> ``load_devices`` builds the device -> every "which
camera?" choice in the measurement layer derives from the ONE :meth:`DeviceSet.camera_names`
source -> the console's Camera measurement / the notebook's ``capture()`` show the image.

User-owned hardware is a first-class citizen on the same chain: a class registered via
``register_device_class`` (or resolved straight from the caller's namespace with
``load_devices(..., lookup=globals())``) is discovered/built exactly like a built-in, and a
class-less bus plugs in as a named DISCOVERY PROVIDER (VISA is the built-in one).

Pure row builders are tested with fake enumerations (no hardware); the IO shell is only
asserted never to raise.  Expected names derive from the live device set, never re-typed.
"""

from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if sys.path[0] != str(REPO_ROOT):
    sys.path.insert(0, str(REPO_ROOT))

from Zou_lab_control.neutral_atom.devices.base import CameraDevice
from Zou_lab_control.neutral_atom.devices.discovery import (
    DISCOVERY_PROVIDERS, DiscoveredDevice, discover_devices, discovery_note,
    probe_visa, register_discovery_provider, visa_rows)
from Zou_lab_control.neutral_atom.devices.pylon import PylonCamera
from Zou_lab_control.neutral_atom.devices.registry import (
    DEVICE_CLASSES, load_devices, register_device_class)


class _FakeInfo:
    def __init__(self, model: str, serial: str):
        self._model, self._serial = model, serial

    def GetModelName(self):
        return self._model

    def GetSerialNumber(self):
        return self._serial


class ToyCamera(CameraDevice):
    """A USER-authored camera (defined outside the framework): registered at runtime or
    resolved straight from the caller's namespace -- zero framework edits either way."""

    def __init__(self, address: str = "toy:0"):
        self.address = str(address)
        self._exposure = 1e-3

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self.configure(exposure=float(value))

    def configure(self, *, exposure: float | None = None, **kwargs) -> None:
        self._reject_unknown_configure_keys({"exposure"}, kwargs)
        if exposure is not None:
            self._exposure = float(exposure)

    def acquire(self, frames: int = 1, *, stop=None, **kwargs):
        return self._retain([np.zeros((4, 4)) for _ in range(int(frames))])

    @classmethod
    def discover(cls):
        return [DiscoveredDevice(
            kind="toy", ident="toy:0", label="Toy frame source",
            config={"type": cls.__name__, "params": {"address": "toy:0"}})]


@pytest.fixture
def discovery_sandbox():
    """Snapshot the shared class/provider registries so a test's registrations never leak."""
    from Zou_lab_control.neutral_atom.devices import discovery, registry

    classes = dict(registry.DEVICE_CLASSES)
    providers = dict(discovery.DISCOVERY_PROVIDERS)
    try:
        yield
    finally:
        registry.DEVICE_CLASSES.clear()
        registry.DEVICE_CLASSES.update(classes)
        discovery.DISCOVERY_PROVIDERS.clear()
        discovery.DISCOVERY_PROVIDERS.update(providers)


def test_pylon_discovery_rows_yield_ready_configs_for_the_class_itself():
    """The DEVICE CLASS owns its discovery: every enumerated Basler camera becomes a row whose
    config is a READY entry for PylonCamera itself (serial-pinned, free-running) -- the class
    name comes from the class, never a hand-written mapping in the aggregator."""
    from Zou_lab_control.neutral_atom.devices.base import SOFTWARE_TRIGGER
    from Zou_lab_control.neutral_atom.devices.camera_trigger import DEFAULT_CAMERA_TRIGGER_CHANNELS

    rows = PylonCamera.discovery_rows([_FakeInfo("acA1920-155um", "24012345"),
                                       _FakeInfo("acA2440-20gm", "40098765")])
    assert [r.ident for r in rows] == ["24012345", "40098765"]
    assert rows[0].label == "acA1920-155um"
    # capture_trigger_channels is spelled out in the ready config (inert while free-running --
    # effective_trigger_channels is then () and edge counts are zero -- but the knob the operator
    # MUST retarget when switching trigger_source to a hardware line).  The virtual monitor camera
    # declares the SAME inert default in Software mode (virtual == real, see
    # test_monitor_camera_free_run.test_free_run_counting_line_declaration_is_identical_on_both_backends).
    assert rows[0].config == {"type": PylonCamera.__name__,
                              "params": {"serial": "24012345",
                                         "trigger_source": SOFTWARE_TRIGGER,
                                         "capture_trigger_channels": list(DEFAULT_CAMERA_TRIGGER_CHANNELS)}}


def test_discovery_aggregates_registered_self_describing_classes():
    """discover_devices() walks the registry and asks each class with a ``discover()``
    classmethod -- adding a discoverable device never touches discovery.py.  PylonCamera is
    registered and self-describing, so its bus is scanned (a note row when nothing attached)."""
    from Zou_lab_control.neutral_atom.devices.discovery import _registered_discoverers

    discoverers, import_notes = _registered_discoverers()
    names = [name for name, _cls in discoverers]
    assert "PylonCamera" in names
    assert "QCMOSCamera" in names                        # BOTH real cameras self-describe
    assert all(n.kind == "note" for n in import_notes)   # broken drivers become rows, not holes
    rows = discover_devices(display=False)
    assert rows and all(isinstance(r, DiscoveredDevice) for r in rows)
    assert any(r.kind in ("basler", "note") for r in rows)


def test_discovered_config_loads_straight_into_a_device_set():
    """The discovery row's config is not a suggestion to retype -- load_devices consumes it
    verbatim and builds the pinned PylonCamera (not opened: no hardware in CI)."""
    row = PylonCamera.discovery_rows([_FakeInfo("acA1920-155um", "24012345")])[0]
    devices = load_devices({"monitor_camera": row.config}, open_devices=False)
    cam = devices["monitor_camera"]
    assert isinstance(cam, PylonCamera)
    assert cam.serial == "24012345"
    assert cam.trigger_source == "Software"


def test_qcmos_discovery_rows_yield_ready_configs_for_the_class_itself():
    """The qCMOS self-describes exactly like the Pylon camera: every DCAM index becomes a row
    whose config is a READY entry for QCMOSCamera (device_index pinned, nested under the
    class's own ``config`` param shape) -- and that config loads verbatim."""
    from Zou_lab_control.neutral_atom.devices.qcmos import QCMOSCamera

    rows = QCMOSCamera.discovery_rows(2, lambda i: f"C15550-20UP #{i}")
    assert [r.ident for r in rows] == ["0", "1"]
    assert rows[1].config == {"type": "QCMOSCamera", "params": {"config": {"device_index": 1}}}
    devices = load_devices({"camera": rows[1].config}, open_devices=False)
    assert isinstance(devices["camera"], QCMOSCamera)
    assert devices["camera"].config.device_index == 1
    assert QCMOSCamera.discovery_rows(0, lambda i: "") == []


def test_registered_custom_class_is_discovered_end_to_end(discovery_sandbox):
    """The user-extension contract: ``register_device_class(MyCam)`` is ALL it takes for
    ``discover_devices()`` to include MyCam's own rows, and the row's ready config builds
    the device through ``load_devices`` under the registered name -- same chain as a
    built-in camera, zero framework edits."""
    register_device_class("ToyCamera", ToyCamera)
    rows = discover_devices(display=False)
    toy = [r for r in rows if r.kind == "toy"]
    assert [r.ident for r in toy] == ["toy:0"]
    devices = load_devices({"monitor_camera": toy[0].config}, open_devices=False)
    cam = devices["monitor_camera"]
    assert isinstance(cam, ToyCamera)
    assert cam.address == "toy:0"


def test_load_devices_lookup_namespace_resolves_user_classes():
    """The confocal lookup_dict pattern: a config ``"type"`` resolves from the CALLER's
    namespace first -- non-class entries are ignored, names absent from the namespace fall
    back to the registry, and a lookup hit never leaks into the shared registry."""
    namespace = {"ToyCamera": ToyCamera, "np": np, "Path": Path, "answer": 42}
    devices = load_devices({"camera": {"type": "ToyCamera", "params": {"address": "toy:7"}},
                            "sequencer": {"type": "VirtualSequencer"}},
                           lookup=namespace, open_devices=False)
    assert isinstance(devices["camera"], ToyCamera)
    assert devices["camera"].address == "toy:7"
    assert type(devices["sequencer"]).__name__ == "VirtualSequencer"   # registry fallback
    assert "ToyCamera" not in DEVICE_CLASSES                           # per-call, no leak
    with pytest.raises(KeyError, match="unknown device class"):
        load_devices({"camera": {"type": "ToyCamera"}}, open_devices=False)


def test_fully_qualified_config_type_does_not_leak_into_registry(discovery_sandbox):
    """B6: a fully-qualified path typed straight into a config (``{"type": "pkg.mod.Cls"}``) is
    resolved per-call via importlib -- but NEVER written back into the shared DEVICE_CLASSES.
    Otherwise every later ``discover_devices`` would scan an arbitrary (possibly COM-port-opening)
    class whose presence depends on what was loaded earlier.  Only REGISTERED keys whose value is
    a lazy import-path string are cached (the built-ins); an ad-hoc fq path is not a key."""
    from Zou_lab_control.neutral_atom.devices.registry import resolve_class

    fq = "Zou_lab_control.neutral_atom.devices.virtual.VirtualSequencer"
    assert fq not in DEVICE_CLASSES
    cls = resolve_class(fq)                                            # resolves via importlib
    assert cls.__name__ == "VirtualSequencer"
    assert fq not in DEVICE_CLASSES, "an ad-hoc fq config type leaked into the shared registry"

    # A REGISTERED name whose value is a lazy import string IS materialised in place (a pure cache
    # hit): resolving a built-in turns its path string into the class, still under the same key.
    # (Force the lazy-string state -- the sandbox fixture restores it -- so this is order-robust.)
    DEVICE_CLASSES["PylonCamera"] = "Zou_lab_control.neutral_atom.devices.pylon.PylonCamera"
    resolved = resolve_class("PylonCamera")
    assert resolved is PylonCamera
    assert DEVICE_CLASSES["PylonCamera"] is PylonCamera               # cached back under its own key


def test_load_devices_lookup_accepts_module_globals():
    """``lookup=globals()`` works verbatim: the whole calling namespace (imports, helpers,
    fixtures, dunders) filters down to its device classes automatically."""
    devices = load_devices({"camera": {"type": "ToyCamera"}}, lookup=globals(),
                           open_devices=False)
    assert isinstance(devices["camera"], ToyCamera)


def test_visa_rows_are_informational_listings():
    """Bare VISA instruments have no driver class here -> rows list address + *IDN? identity
    with config=None (the operator wires the address into a custom class)."""
    rows = visa_rows(("GPIB0::17::INSTR",), lambda r: "Rigol,DSG836,...")
    assert rows[0].ident == "GPIB0::17::INSTR"
    assert rows[0].label.startswith("Rigol")
    assert rows[0].config is None


def test_visa_is_a_builtin_discovery_provider():
    """The class-less VISA bus is no aggregator special case: it is the first entry of the
    SAME provider registry operators extend, and its scan obeys the confocal contract
    (rows or notes, never an exception -- with or without pyvisa installed)."""
    assert DISCOVERY_PROVIDERS["visa"] is probe_visa
    rows = probe_visa()
    assert rows and all(isinstance(r, DiscoveredDevice) for r in rows)
    assert all(r.kind in ("visa", "note") for r in rows)


def test_registered_provider_scans_and_a_crash_becomes_a_note(discovery_sandbox):
    """``register_discovery_provider`` puts a user bus into the same scan; a provider that
    raises is reported as a note row, never an exception out of ``discover_devices()``."""
    register_discovery_provider(
        "toybus", lambda: [DiscoveredDevice(kind="toybus", ident="usb:9", label="Widget")])

    def broken():
        raise RuntimeError("bus exploded")

    register_discovery_provider("brokenbus", broken)
    rows = discover_devices(display=False)
    assert any(r.kind == "toybus" and r.ident == "usb:9" for r in rows)
    note = next(r for r in rows if r.kind == "note" and r.ident == "brokenbus")
    assert "bus exploded" in note.label


def test_register_discovery_provider_validates_inputs():
    with pytest.raises(ValueError, match="not be empty"):
        register_discovery_provider("  ", lambda: [])
    with pytest.raises(TypeError, match="callable"):
        register_discovery_provider("x", "not a callable")


def test_discovery_protocol_is_public_at_the_package_level():
    """Everything a user needs to wire their OWN hardware is importable from ``na``: the
    row/note builders, both registration calls, and the lookup-capable loader."""
    import matplotlib
    matplotlib.use("Agg")
    from Zou_lab_control import neutral_atom as na

    assert na.DiscoveredDevice is DiscoveredDevice
    assert na.discovery_note is discovery_note
    assert na.register_discovery_provider is register_discovery_provider
    assert na.register_device_class is register_device_class
    assert "lookup" in inspect.signature(na.load_devices).parameters


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


def test_basler_monitor_config_is_a_bare_monitor_rig():
    """The shipped ``basler_monitor`` config declares ONLY what physically exists: one real
    free-running PylonCamera.  No fabricated virtual placeholders pad the roster -- missing
    roles are simply absent, and the session tolerates that (the decoupling contract)."""
    devices = load_devices("basler_monitor", open_devices=False)
    assert set(devices.devices) == {"monitor_camera"}
    assert isinstance(devices["monitor_camera"], PylonCamera)
    assert devices["monitor_camera"].trigger_source == "Software"
    assert devices.camera_names() == ("monitor_camera",)
    assert devices.default_camera_name() == "monitor_camera"


def test_session_tolerates_missing_roles():
    """NO role is mandatory: a config with only a monitor camera (or with no camera at all)
    still builds a working session -- the roles that exist light up, the rest raise only when
    actually asked for.  This is what lets a config declare ONLY the real hardware."""
    import matplotlib
    matplotlib.use("Agg")
    from Zou_lab_control.neutral_atom.session import NeutralAtomSession

    # a bare monitor rig (the basler_monitor shape, camera not opened)
    session = NeutralAtomSession(load_devices("basler_monitor", open_devices=False))
    assert session.devices.default_camera_name() == "monitor_camera"
    assert session.sequence is not None                    # imaging sequence composes w/o a readout camera
    # readout's camera spec picks the only camera as its default
    spec = session.readout.camera_spec()
    camera_param = next(p for p in spec.params if p.key == "camera")
    assert camera_param.default == "monitor_camera"
    assert camera_param.choices == ("monitor_camera",)

    # a config with NO camera at all is still a legal session; camera asks raise with guidance
    empty = NeutralAtomSession(load_devices({"sequencer": {"type": "VirtualSequencer"}},
                                            open_devices=False))
    assert empty.devices.camera_names() == ()
    with pytest.raises(AttributeError, match="no camera"):
        empty.devices.default_camera_name()


def test_pylon_camera_construction_needs_no_pylon_runtime():
    """pypylon is imported inside open() only -- constructing / configuring the class (and
    therefore loading any config that names it) works on a machine without the Basler SDK."""
    cam = PylonCamera(serial="123", trigger_source="Line1", pixel_format="Mono12")
    snap = cam.snapshot()
    assert snap["pixel_format"] == "Mono12"
    assert snap["trigger_source"] == "Line1"
    with pytest.raises(RuntimeError, match="before open"):
        cam.acquire(1)
