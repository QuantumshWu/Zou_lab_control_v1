"""Device registry and JSON/dict config loader."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from .qcmos import QCMOSCamera
from .sequencer import ManualSequencer, RemoteSequencer, RuntimeSequencer, VerilogSequencer
from .virtual import VirtualCamera, VirtualSequencer, VirtualTrapArray, virtual_config
from .base import CameraDevice, SequencerDevice, TrapArrayDevice, validate_device_contract


DEVICE_CLASSES = {
    "QCMOSCamera": QCMOSCamera,
    "ManualSequencer": ManualSequencer,
    "RemoteSequencer": RemoteSequencer,
    "RuntimeSequencer": RuntimeSequencer,
    "VerilogSequencer": VerilogSequencer,
    "VirtualCamera": VirtualCamera,
    "VirtualSequencer": VirtualSequencer,
    "VirtualTrapArray": VirtualTrapArray,
}


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

    def close(self) -> None:
        errors: list[str] = []
        for name, device in reversed(list(self.devices.items())):
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


def device_config_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "configs"


def load_devices(config: str | Path | Mapping[str, Any] = "virtual") -> DeviceSet:
    """Load a device graph from ``virtual``, JSON, or a Python dict."""

    cfg = read_config(config)
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
    except Exception:
        DeviceSet(devices, cfg).close()
        raise
    return DeviceSet(devices, cfg)


def read_config(config: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, Mapping):
        return dict(config)
    if str(config).lower() == "virtual":
        return virtual_config()
    path = Path(config)
    if not path.exists() and path.suffix == "":
        path = device_config_dir() / f"{config}.json"
    if not path.exists():
        raise FileNotFoundError(f"device config not found: {config}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_class(name: str) -> type:
    if name in DEVICE_CLASSES:
        return DEVICE_CLASSES[name]
    if "." not in name:
        raise KeyError(f"unknown device class {name!r}. Known: {sorted(DEVICE_CLASSES)}")
    module_name, class_name = name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def available_device_configs() -> list[str]:
    names = ["virtual"]
    if device_config_dir().exists():
        names.extend(path.stem for path in device_config_dir().glob("*.json"))
    return sorted(set(names))


__all__ = ["DEVICE_CLASSES", "DeviceSet", "available_device_configs", "device_config_dir", "load_devices", "read_config", "resolve_class"]
