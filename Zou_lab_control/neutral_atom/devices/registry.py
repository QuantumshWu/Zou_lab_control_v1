"""Device registry and JSON/dict config loader."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .virtual import virtual_config, virtual_config_with_overrides
from .base import BaseDevice, CameraDevice, SequencerDevice, TrapArrayDevice


BUILTIN_DEVICE_CLASS_PATHS = {
    "PylonCamera": "Zou_lab_control.neutral_atom.devices.pylon.PylonCamera",
    "QCMOSCamera": "Zou_lab_control.neutral_atom.devices.qcmos.QCMOSCamera",
    "ManualSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.ManualSequencer",
    "RemoteSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.RemoteSequencer",
    "RuntimeSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.RuntimeSequencer",
    "VerilogSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.VerilogSequencer",
    "VirtualCamera": "Zou_lab_control.neutral_atom.devices.virtual.VirtualCamera",
    "VirtualMotCamera": "Zou_lab_control.neutral_atom.devices.virtual.VirtualMotCamera",
    "VirtualSequencer": "Zou_lab_control.neutral_atom.devices.virtual.VirtualSequencer",
    "VirtualTrapArray": "Zou_lab_control.neutral_atom.devices.virtual.VirtualTrapArray",
}
DEVICE_CLASSES: dict[str, type | str] = dict(BUILTIN_DEVICE_CLASS_PATHS)


@dataclass(frozen=True)
class DeviceDomain:
    """One ROLE-TYPE of device (camera, sequencer, trap array, and future RF source / DAQ / ...).

    The single source that makes device SELECTION and the device-manager GUI fully type-generic:
    nothing names a concrete type like ``"camera"`` -- every measurement/task role and every GUI
    section iterates :data:`DEVICE_DOMAINS`, so adding an RF source is ONE
    :func:`register_device_domain` call and every selection dropdown + the device manager pick it
    up with no edits to specs or the GUI."""

    key: str            # the conventional config role name ("camera", "sequencer")
    base_type: type     # the domain base class every device of this role subclasses
    label: str          # human label for a GUI section ("Camera", "Sequencer")


#: The registered device role-types -- the ONE list device selection + the device manager read.
DEVICE_DOMAINS: dict[str, DeviceDomain] = {}


def register_device_domain(key: str, base_type: type, label: str | None = None) -> None:
    """Register a device ROLE-TYPE so selection dropdowns + the device manager list it.  Built-ins
    are camera / sequencer / trap_array; a lab adding an RF source registers ``RFSourceDevice`` here
    and every measurement/GUI picks it up automatically (no camera-special code anywhere)."""
    if not str(key).strip():
        raise ValueError("device domain key must not be empty.")
    if not isinstance(base_type, type):
        raise TypeError("device domain base_type must be a class.")
    DEVICE_DOMAINS[str(key)] = DeviceDomain(
        str(key), base_type, str(label or str(key).replace("_", " ").title()))


def device_domains() -> tuple["DeviceDomain", ...]:
    """The registered device role-types, sorted by key -- the ONE sequence the GUI + specs iterate."""
    return tuple(DEVICE_DOMAINS[k] for k in sorted(DEVICE_DOMAINS))


register_device_domain("camera", CameraDevice, "Camera")
register_device_domain("sequencer", SequencerDevice, "Sequencer")
register_device_domain("trap_array", TrapArrayDevice, "Trap array")


def validate_device_contract(name: str, device: Any) -> None:
    """Raise if a configured device does not inherit its role's required base class.

    The role->base relation is :data:`DEVICE_DOMAINS` -- the SAME table
    :func:`register_device_domain` extends -- so a lab-registered domain (RF source,
    DAQ, ...) is contract-checked at load time exactly like the built-ins.  A name
    with no registered domain need only be a :class:`BaseDevice`."""

    domain = DEVICE_DOMAINS.get(str(name))
    expected = domain.base_type if domain is not None else BaseDevice
    if not isinstance(device, expected):
        raise TypeError(
            f"device {name!r} ({type(device).__name__}) must inherit {expected.__name__}. "
            "Implement the appropriate BaseDevice subclass instead of relying on duck typing."
        )


@dataclass
class DeviceSet:
    """Attribute-access container returned by ``load_devices``."""

    devices: dict[str, Any]
    config: dict[str, Any]

    @property
    def camera(self) -> CameraDevice:
        return self.require("camera", CameraDevice)

    @property
    def sequencer(self) -> SequencerDevice:
        return self.require("sequencer", SequencerDevice)

    @property
    def trap_array(self) -> TrapArrayDevice:
        return self.require("trap_array", TrapArrayDevice)

    def __getattr__(self, name: str):
        try:
            return self.devices[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __contains__(self, name: str) -> bool:
        return name in self.devices

    def __getitem__(self, name: str):
        return self.devices[name]

    def device_names(self, base_type: type) -> tuple[str, ...]:
        """Every device of a given ROLE TYPE in the set, sorted by name -- THE source for any
        "which <role>?" choice (a measurement's ``camera`` / ``sequencer`` ParamDecl).  One
        role-agnostic reader so camera, sequencer, trap-array (and any future role) answer the
        same way.  Empty when the config has no such device -- callers decide what that means
        (hide the panel / raise with guidance), never a fabricated placeholder name."""
        return tuple(sorted(name for name, dev in self.devices.items()
                            if isinstance(dev, base_type)))

    def default_device_name(self, base_type: type, conventional: str | None = None) -> str:
        """The device a spec binds to a role when the operator names none: the CONVENTIONAL role
        name (e.g. ``"camera"`` / ``"sequencer"``) when present, else the only/first device of that
        type.  Raises with guidance when the config has no device of the role at all."""
        names = self.device_names(base_type)
        if not names:
            role = conventional or base_type.__name__      # the human role word ("camera") if known
            raise AttributeError(
                f"this device config has no {role} -- add one (e.g. via na.discover_devices(); "
                "each found device's row carries a ready config entry).")
        return conventional if (conventional and conventional in names) else names[0]

    def camera_names(self) -> tuple[str, ...]:
        """Every camera in the set (thin wrapper over :meth:`device_names` -- ONE role-agnostic core)."""
        return self.device_names(CameraDevice)

    def default_camera_name(self) -> str:
        """The default readout camera (``"camera"`` role, else the first) -- thin wrapper over
        :meth:`default_device_name`."""
        return self.default_device_name(CameraDevice, conventional="camera")

    def require(self, name: str, expected_type: type | tuple[type, ...] | None = None):
        if name not in self.devices:
            raise AttributeError(name)
        device = self.devices[name]
        validate_device_contract(name, device)
        if expected_type is not None and not isinstance(device, expected_type):
            if isinstance(expected_type, tuple):
                expected_name = " / ".join(cls.__name__ for cls in expected_type)
            else:
                expected_name = expected_type.__name__
            raise TypeError(f"device {name!r} ({type(device).__name__}) must inherit {expected_name}.")
        return device

    def open(self) -> "DeviceSet":
        opened: list[tuple[str, Any]] = []
        try:
            for name in self._open_order():
                device = self.require(name)
                device.open()
                opened.append((name, device))
        except Exception:
            for _, device in reversed(opened):
                try:
                    device.close()
                except Exception:
                    pass
            raise
        return self

    def close(self) -> None:
        errors: list[str] = []
        ordered = [(name, self.devices[name]) for name in self._open_order() if name in self.devices]
        for name, device in reversed(ordered):
            close = getattr(device, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    errors.append(f"{name}: {exc}")
        if errors:
            raise RuntimeError("Device close failed: " + "; ".join(errors))

    def snapshot(self) -> dict[str, Any]:
        out = {}
        for name, device in self.devices.items():
            snap = getattr(device, "snapshot", None)
            out[name] = snap() if callable(snap) else {"type": type(device).__name__}
        return out

    def to_config(self) -> dict[str, Any]:
        """The round-trippable device CONFIG that reproduces THIS set: the ``{role: {"type", "params"}}``
        dict ``load_devices`` built the set from (``$device:`` cross-references intact), deep-copied so a
        caller can serialize / mutate it freely.  ``na.connect(device_set.to_config())`` -- or writing it
        to JSON and ``na.connect("that.json")`` -- rebuilds an equivalent device set.  This is the ONE
        source the session's ``save_config`` / the device-manager "Save config" button writes (vs
        :meth:`snapshot`, which is a per-device STATE dump that does NOT round-trip through ``type`` /
        ``params``)."""
        return deepcopy(self.config)

    def _open_order(self) -> list[str]:
        # Cameras open LAST (they may bind to an already-open sequencer/trigger source) --
        # decided by TYPE, not by a hardcoded device name, so a monitor_camera or any other
        # camera role gets the same treatment.
        names = list(self.devices)
        return ([name for name in names if not isinstance(self.devices[name], CameraDevice)]
                + [name for name in names if isinstance(self.devices[name], CameraDevice)])


def device_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def load_devices(
    config: str | Path | Mapping[str, Any] = "virtual",
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    open_devices: bool = False,
    lookup: Mapping[str, Any] | None = None,
) -> DeviceSet:
    """Load a device graph from ``virtual``, JSON, or a Python dict.

    ``lookup`` (optional) is a caller namespace consulted FIRST when resolving each
    entry's ``"type"`` -- pass ``globals()`` and any device class defined in the calling
    notebook is usable in a config without registration (the confocal lookup_dict
    pattern).  See :func:`resolve_class` for the full resolution order."""

    cfg = read_config(config)
    apply_device_overrides(cfg, overrides)
    devices: dict[str, Any] = {}
    visiting: set[str] = set()

    def resolve(value):
        if isinstance(value, str) and value.startswith("$device:"):
            return build(value.split(":", 1)[1])
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        return value

    def build(name: str):
        if name in devices:
            return devices[name]
        if name in visiting:
            raise ValueError(f"cyclic device dependency involving {name!r}.")
        if name not in cfg:
            raise KeyError(f"device {name!r} is not present in config.")
        visiting.add(name)
        entry = cfg[name]
        cls = resolve_class(str(entry["type"]), lookup=lookup)
        params = resolve(dict(entry.get("params", {})))
        devices[name] = cls(**params)
        validate_device_contract(name, devices[name])
        visiting.remove(name)
        return devices[name]

    try:
        for name in cfg:
            build(name)
        device_set = DeviceSet(devices, cfg)
        if open_devices:
            device_set.open()
    except Exception:
        DeviceSet(devices, cfg).close()
        raise
    return device_set


def resolve_connect_config(
    config: str | Path | Mapping[str, Any],
    *,
    trap_array: Mapping[str, Any] | None = None,
    sitemap: Mapping[str, Any] | None = None,
    camera: Mapping[str, Any] | None = None,
    sequencer: Mapping[str, Any] | None = None,
    params: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, dict[str, Any]] | None, dict[str, Any]]:
    """Resolve a ``connect()`` request into ``(device_config, overrides, defaults)``.

    Backend-specific config shortcuts live with the BACKEND (the device layer),
    not the orchestration facade: the virtual backend owns its ``sitemap`` /
    ``loss_rate`` / alias translation.  A real config (named JSON / path / dict)
    takes only per-device override mappings and rejects the virtual-only
    shortcuts.  ``session.connect`` calls this so it never imports a concrete
    backend or reads a backend's internal fields (keeps virtual == real)."""

    params = dict(params or {})
    if isinstance(config, str) and config.lower() == "virtual":
        cfg, defaults = virtual_config_with_overrides(
            trap_array=dict(trap_array) if trap_array else None,
            sitemap=dict(sitemap) if sitemap else None,
            camera=dict(camera) if camera else None,
            sequencer=dict(sequencer) if sequencer else None,
            params=params,
        )
        return cfg, None, defaults
    if sitemap or params:
        raise ValueError("sitemap and virtual shortcut parameters are only supported with config='virtual'.")
    overrides: dict[str, dict[str, Any]] = {}
    for name, device_params in (("trap_array", trap_array), ("camera", camera), ("sequencer", sequencer)):
        if device_params:
            overrides[name] = dict(device_params)
    return config, (overrides or None), {}


def read_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    if str(config).lower() == "virtual":
        return virtual_config()
    path = Path(config)
    if not path.exists():
        name = str(config)
        # A BARE config name (no directory component) resolves from the bundled
        # configs/ dir, WITH or WITHOUT an explicit .json suffix -- so both
        # ``remote_template`` and ``remote_template.json`` find the bundled file
        # (a natural thing to type that would otherwise be a FileNotFoundError).
        # A real path (absolute or with a directory) is used verbatim.
        if Path(name).parent == Path("."):
            for candidate in (device_config_dir() / name, device_config_dir() / f"{name}.json"):
                if candidate.exists():
                    path = candidate
                    break
    if not path.exists():
        raise FileNotFoundError(f"device config not found: {config}")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_device_overrides(cfg: dict[str, Any], overrides: Mapping[str, Mapping[str, Any]] | None) -> None:
    if not overrides:
        return
    for name, params in overrides.items():
        if params is None:
            continue
        if name not in cfg:
            raise KeyError(f"device {name!r} is not present in config.")
        if not isinstance(params, Mapping):
            raise TypeError(f"device override for {name!r} must be a mapping.")
        target = cfg[name].setdefault("params", {})
        if not isinstance(target, dict):
            raise TypeError(f"device {name!r} params must be a mapping to apply overrides.")
        deep_update(target, params)


def deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = deepcopy(value)


def resolve_class(name: str, lookup: Mapping[str, Any] | None = None) -> type:
    """Resolve a config ``"type"`` name to a device class.

    Resolution order: the caller's ``lookup`` namespace first (only entries that ARE
    ``BaseDevice`` subclasses count, so ``globals()`` passes verbatim -- helpers, modules,
    and constants in the namespace are ignored), then the registry (built-ins plus
    :func:`register_device_class`), then a fully-qualified import path.  A ``lookup`` hit
    is per-call: it never writes into the shared registry."""

    if lookup is not None:
        candidate = lookup.get(str(name))
        if isinstance(candidate, type) and issubclass(candidate, BaseDevice):
            return candidate
    registered = name in DEVICE_CLASSES
    target = DEVICE_CLASSES.get(name, name)
    if isinstance(target, type):
        return target
    if "." not in str(target):
        raise KeyError(f"unknown device class {name!r}. Known: {sorted(DEVICE_CLASSES)}")
    module_name, class_name = str(target).rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(cls, type):
        raise TypeError(f"resolved device class {name!r} is not a class.")
    # Cache back ONLY when ``name`` is a REGISTERED key whose value is a lazy import-path string
    # (the built-ins, or a ``register_device_class(name, "pkg.Cls")`` entry): materialising it to
    # the class is a pure cache hit.  A fully-qualified path typed straight into a config
    # (``{"type": "my_lab.SerialCam"}``) is NOT a registry key -- writing it back would silently
    # pin an arbitrary class (even a non-device one) into the shared registry forever, so every
    # later ``discover_devices`` would scan it.  ``importlib`` already caches the module, so the
    # per-call re-resolve is free; the registry stays exactly what was registered.
    if registered:
        DEVICE_CLASSES[name] = cls
    return cls


def register_device_class(name: str, cls: type | str) -> None:
    """Register a device class or import path for future ``load_devices`` calls."""

    if not str(name).strip():
        raise ValueError("device class name must not be empty.")
    if not isinstance(cls, type) and "." not in str(cls):
        raise ValueError("device class registration must be a class or fully qualified import path.")
    DEVICE_CLASSES[str(name)] = cls


def device_class_registry() -> dict[str, str]:
    """Return the known device classes without forcing every hardware import."""

    out = {}
    for name, target in DEVICE_CLASSES.items():
        if isinstance(target, type):
            out[name] = f"{target.__module__}.{target.__qualname__}"
        else:
            out[name] = str(target)
    return dict(sorted(out.items()))


def available_device_configs() -> list[str]:
    names = ["virtual"]
    if device_config_dir().exists():
        names.extend(path.stem for path in device_config_dir().glob("*.json"))
    return sorted(set(names))


__all__ = [
    "DEVICE_CLASSES",
    "DEVICE_DOMAINS",
    "DeviceDomain",
    "DeviceSet",
    "device_domains",
    "register_device_domain",
    "apply_device_overrides",
    "available_device_configs",
    "device_class_registry",
    "device_config_dir",
    "load_devices",
    "read_config",
    "register_device_class",
    "resolve_class",
    "resolve_connect_config",
    "validate_device_contract",
]
