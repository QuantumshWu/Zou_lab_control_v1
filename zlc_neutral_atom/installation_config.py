"""Human-readable installation graph configuration and atomic persistence."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path
from pkgutil import walk_packages
from types import MappingProxyType

from zlc_storage import atomic_write_text, canonical_text


INSTALLATION_CONFIG_FORMAT = "zlc.installation"


def _json_value(value: object, field: str) -> object:
    """Copy one ordinary JSON value while rejecting ambiguous Python values."""

    if value is None or type(value) in (bool, int, str):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must not contain non-finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _json_value(item, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        copied: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{field} keys must be non-empty strings")
            if key in copied:
                raise ValueError(f"{field} contains duplicate key {key!r}")
            copied[key] = _json_value(item, f"{field}.{key}")
        return MappingProxyType(copied)
    raise TypeError(f"{field} must contain only ordinary JSON values")


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class DeviceInstanceConfig:
    """One stable, human-named device instance in an installation graph."""

    instance_id: str
    role: str
    type_id: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "instance_id",
            canonical_text(self.instance_id, "device instance_id"),
        )
        if "/" in self.instance_id:
            raise ValueError("device instance_id cannot contain '/'")
        object.__setattr__(self, "role", canonical_text(self.role, "device role"))
        object.__setattr__(
            self,
            "type_id",
            canonical_text(self.type_id, "device type_id"),
        )
        if not isinstance(self.parameters, Mapping):
            raise TypeError("device parameters must be a mapping")
        copied = _json_value(self.parameters, "device parameters")
        assert isinstance(copied, Mapping)
        object.__setattr__(self, "parameters", copied)

    def to_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "role": self.role,
            "type_id": self.type_id,
            "parameters": _plain(self.parameters),
        }

    @classmethod
    def from_dict(cls, value: object) -> "DeviceInstanceConfig":
        fields = {"instance_id", "role", "type_id", "parameters"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError(
                "device instance must contain exactly "
                "['instance_id', 'parameters', 'role', 'type_id']"
            )
        return cls(
            instance_id=value["instance_id"],
            role=value["role"],
            type_id=value["type_id"],
            parameters=value["parameters"],
        )


@dataclass(frozen=True, slots=True)
class InstallationConfigDocument:
    """An ordered installation graph; order is the user's presentation order."""

    devices: tuple[DeviceInstanceConfig, ...]

    def __post_init__(self) -> None:
        devices = tuple(self.devices)
        if any(not isinstance(item, DeviceInstanceConfig) for item in devices):
            raise TypeError("devices must contain DeviceInstanceConfig values")
        ids = tuple(item.instance_id for item in devices)
        roles = tuple(item.role for item in devices)
        if len(ids) != len(set(ids)):
            raise ValueError("device instance_id values must be unique")
        if len(roles) != len(set(roles)):
            raise ValueError("device roles must be unique")
        object.__setattr__(self, "devices", devices)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": INSTALLATION_CONFIG_FORMAT,
            "devices": [item.to_dict() for item in self.devices],
        }

    @classmethod
    def from_dict(cls, value: object) -> "InstallationConfigDocument":
        if not isinstance(value, Mapping) or set(value) != {"format", "devices"}:
            raise ValueError(
                "installation config must contain exactly ['devices', 'format']"
            )
        if value["format"] != INSTALLATION_CONFIG_FORMAT:
            raise ValueError(
                f"unsupported installation config format {value['format']!r}"
            )
        devices = value["devices"]
        if not isinstance(devices, list):
            raise TypeError("installation devices must be a JSON array")
        return cls(tuple(DeviceInstanceConfig.from_dict(item) for item in devices))


def load_installation_config(
    path: str | os.PathLike[str],
) -> InstallationConfigDocument:
    target = _config_path(path)
    try:
        value = json.loads(
            target.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except OSError as exc:
        raise RuntimeError(f"cannot read installation config {target}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"installation config {target} is not valid JSON") from exc
    document = InstallationConfigDocument.from_dict(value)
    from zlc_neutral_atom.device_types import validate_installation_graph

    validate_installation_graph(document)
    return document


def save_installation_config(
    path: str | os.PathLike[str],
    document: InstallationConfigDocument,
) -> None:
    """Validate and atomically replace one ordinary, readable JSON file."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    from zlc_neutral_atom.device_types import validate_installation_graph

    validate_installation_graph(document)
    target = _config_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"installation config parent does not exist: {target.parent}"
        )
    payload = json.dumps(
        document.to_dict(),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
    ) + "\n"
    atomic_write_text(target, payload)


@cache
def _installation_templates() -> Mapping[str, InstallationConfigDocument]:
    namespace = import_module("zlc_neutral_atom.devices")
    prefix = namespace.__name__ + "."
    module_names = sorted(
        item.name
        for item in walk_packages(namespace.__path__, prefix=prefix)
        if item.name.endswith(".templates")
    )
    templates: dict[str, InstallationConfigDocument] = {}
    for module_name in module_names:
        module = import_module(module_name)
        values = getattr(module, "INSTALLATION_TEMPLATES", None)
        if not isinstance(values, Mapping):
            raise TypeError(
                f"{module_name} must export INSTALLATION_TEMPLATES mapping"
            )
        for name, document in values.items():
            normalized = canonical_text(name, "installation template name")
            if normalized in templates:
                raise ValueError(f"duplicate installation template {normalized!r}")
            if not isinstance(document, InstallationConfigDocument):
                raise TypeError("installation templates must be config documents")
            templates[normalized] = document
    if not templates:
        raise RuntimeError("no built-in installation templates were discovered")
    return MappingProxyType(dict(sorted(templates.items())))


def discover_installation_templates() -> tuple[str, ...]:
    return tuple(_installation_templates())


def installation_template(
    name: str,
    **overrides: object,
) -> InstallationConfigDocument:
    """Return one template with generic schema-owned convenience overrides."""

    normalized = canonical_text(name, "installation template name")
    try:
        document = _installation_templates()[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown installation template {normalized!r}") from exc
    from zlc_neutral_atom.device_types import device_type, validate_installation_graph

    if overrides:
        unknown = set(overrides)
        devices: list[DeviceInstanceConfig] = []
        for instance in document.devices:
            descriptor = device_type(instance.type_id)
            matches = set(overrides).intersection(descriptor.authoring_schema.keys)
            parameters = dict(instance.parameters)
            for key in matches:
                parameters[key] = overrides[key]
                unknown.discard(key)
            parameters = descriptor.authoring_schema.freeze(parameters)
            devices.append(
                DeviceInstanceConfig(
                    instance.instance_id,
                    instance.role,
                    instance.type_id,
                    parameters,
                )
            )
        if unknown:
            raise ValueError(
                f"template {normalized!r} has no fields {tuple(sorted(unknown))}"
            )
        document = InstallationConfigDocument(tuple(devices))

    validate_installation_graph(document)
    return document


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _config_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("installation config path must be path-like")
    return Path(path).expanduser().resolve()


__all__ = [
    "INSTALLATION_CONFIG_FORMAT",
    "DeviceInstanceConfig",
    "InstallationConfigDocument",
    "discover_installation_templates",
    "installation_template",
    "load_installation_config",
    "save_installation_config",
]
