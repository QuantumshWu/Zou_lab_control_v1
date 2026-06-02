"""Device registry and JSON/dict config loader."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .virtual import virtual_config
from .base import CameraDevice, SequencerDevice, TrapArrayDevice, validate_device_contract


BUILTIN_DEVICE_CLASS_PATHS = {
    "QCMOSCamera": "Zou_lab_control.neutral_atom.devices.qcmos.QCMOSCamera",
    "ManualSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.ManualSequencer",
    "RemoteSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.RemoteSequencer",
    "RuntimeSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.RuntimeSequencer",
    "VerilogSequencer": "Zou_lab_control.neutral_atom.devices.sequencer.VerilogSequencer",
    "VirtualCamera": "Zou_lab_control.neutral_atom.devices.virtual.VirtualCamera",
    "VirtualSequencer": "Zou_lab_control.neutral_atom.devices.virtual.VirtualSequencer",
    "VirtualTrapArray": "Zou_lab_control.neutral_atom.devices.virtual.VirtualTrapArray",
}
DEVICE_CLASSES: dict[str, type | str] = dict(BUILTIN_DEVICE_CLASS_PATHS)


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

    def _open_order(self) -> list[str]:
        names = list(self.devices)
        return [name for name in names if name != "camera"] + [name for name in names if name == "camera"]


def device_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def load_devices(
    config: str | Path | Mapping[str, Any] = "virtual",
    *,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
    open_devices: bool = False,
) -> DeviceSet:
    """Load a device graph from ``virtual``, JSON, or a Python dict."""

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
        cls = resolve_class(str(entry["type"]))
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


def read_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return deepcopy(dict(config))
    if str(config).lower() == "virtual":
        return virtual_config()
    path = Path(config)
    if not path.exists() and path.suffix == "":
        path = device_config_dir() / f"{config}.json"
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


def resolve_class(name: str) -> type:
    target = DEVICE_CLASSES.get(name, name)
    if isinstance(target, type):
        return target
    if "." not in str(target):
        raise KeyError(f"unknown device class {name!r}. Known: {sorted(DEVICE_CLASSES)}")
    module_name, class_name = str(target).rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    if not isinstance(cls, type):
        raise TypeError(f"resolved device class {name!r} is not a class.")
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
    "DeviceSet",
    "apply_device_overrides",
    "available_device_configs",
    "device_class_registry",
    "device_config_dir",
    "load_devices",
    "read_config",
    "register_device_class",
    "resolve_class",
]
