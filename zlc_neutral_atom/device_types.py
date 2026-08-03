"""Fixed built-in device types and pure installation-graph preflight."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pkgutil import walk_packages
from types import MappingProxyType

from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.installation_config import (
    DeviceInstanceConfig,
    InstallationConfigDocument,
)
from zlc_storage import canonical_text


CAPABILITY_CAMERA_CAPTURE = "camera.capture"
CAPABILITY_CAMERA_MONITOR = "camera.monitor"
CAPABILITY_MOT_FIELD_CAPTURE = "camera.mot_field_capture"
CAPABILITY_CAMERA_SIGNAL_ASSOCIATION = "camera.signal_association"
CAPABILITY_EXTERNAL_TRIGGER_CLIENT = "pulse.external_trigger_client"
CAPABILITY_PULSE_EXECUTE = "pulse.execute"
CAPABILITY_READOUT_BINDING = "readout.binding"
CAPABILITY_RF_TABLE = "rf.table"


DeviceFactory = Callable[
    [DeviceInstanceConfig, Mapping[str, object], object, object | None],
    tuple[Mapping[str, object], Callable[[], None]],
]
DeviceDiscovery = Callable[[], tuple[DeviceInstanceConfig, ...]]


@dataclass(frozen=True, slots=True)
class DeviceTypeDescriptor:
    """One fixed leaf type; schema and dependencies are its whole declaration."""

    type_id: str
    domain: str
    label: str
    authoring_schema: AuthoringSchema
    capabilities: tuple[str, ...]
    requirements: tuple[tuple[str, str], ...]
    factory: DeviceFactory
    discover: DeviceDiscovery | None = None

    def __post_init__(self) -> None:
        for field in ("type_id", "domain", "label"):
            object.__setattr__(
                self,
                field,
                canonical_text(getattr(self, field), f"device {field}"),
            )
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        capabilities = tuple(
            canonical_text(value, "device capability")
            for value in self.capabilities
        )
        if not capabilities or len(capabilities) != len(set(capabilities)):
            raise ValueError("device capabilities must be non-empty and unique")
        object.__setattr__(self, "capabilities", capabilities)
        requirements = tuple(
            (
                canonical_text(parameter_key, "dependency parameter key"),
                canonical_text(capability, "required capability"),
            )
            for parameter_key, capability in self.requirements
        )
        keys = tuple(key for key, _capability in requirements)
        if len(keys) != len(set(keys)):
            raise ValueError("dependency parameter keys must be unique")
        undeclared = set(keys).difference(self.authoring_schema.keys)
        if undeclared:
            raise ValueError(
                "device requirements reference undeclared parameters: "
                f"{tuple(sorted(undeclared))}"
            )
        object.__setattr__(self, "requirements", requirements)
        if not callable(self.factory):
            raise TypeError("device factory must be callable")
        if self.discover is not None and not callable(self.discover):
            raise TypeError("device discover must be callable or None")

    @property
    def defaults(self) -> dict[str, object]:
        """Return UI draft defaults derived only from the declared schema."""

        return {field.key: field.default for field in self.authoring_schema.fields}


@cache
def discover_device_types() -> tuple[DeviceTypeDescriptor, ...]:
    """Discover the deterministic checked-in leaf modules, never plugins."""

    namespace = import_module("zlc_neutral_atom.devices")
    prefix = namespace.__name__ + "."
    module_names = sorted(
        item.name
        for item in walk_packages(namespace.__path__, prefix=prefix)
        if item.name.endswith(".device_types")
    )
    descriptors: list[DeviceTypeDescriptor] = []
    for module_name in module_names:
        module = import_module(module_name)
        values = getattr(module, "DEVICE_TYPES", None)
        if not isinstance(values, tuple) or any(
            not isinstance(value, DeviceTypeDescriptor) for value in values
        ):
            raise TypeError(f"{module_name} must export a DEVICE_TYPES tuple")
        descriptors.extend(values)
    type_ids = tuple(item.type_id for item in descriptors)
    if not descriptors:
        raise RuntimeError("no built-in device types were discovered")
    if len(type_ids) != len(set(type_ids)):
        raise ValueError("built-in device type_id values must be unique")
    return tuple(sorted(descriptors, key=lambda item: item.type_id))


def device_type(type_id: str) -> DeviceTypeDescriptor:
    normalized = canonical_text(type_id, "device type_id")
    matches = tuple(
        descriptor
        for descriptor in discover_device_types()
        if descriptor.type_id == normalized
    )
    if len(matches) != 1:
        raise ValueError(f"unknown device type {normalized!r}")
    return matches[0]


def validate_installation_graph(
    document: InstallationConfigDocument,
) -> tuple[DeviceInstanceConfig, ...]:
    """Purely validate and return stable dependency order for one graph."""

    if not isinstance(document, InstallationConfigDocument):
        raise TypeError("document must be InstallationConfigDocument")
    devices = document.devices
    if not devices:
        raise ValueError("installation graph must contain at least one device")
    by_id = {item.instance_id: item for item in devices}
    descriptors = {item.instance_id: device_type(item.type_id) for item in devices}
    dependencies: dict[str, tuple[str, ...]] = {}
    for item in devices:
        descriptor = descriptors[item.instance_id]
        descriptor.authoring_schema.freeze(item.parameters)
        refs: list[str] = []
        for parameter_key, required_capability in descriptor.requirements:
            reference = item.parameters.get(parameter_key)
            if not isinstance(reference, str) or not reference.strip():
                raise TypeError(
                    f"{item.instance_id}.{parameter_key} must be a device instance_id"
                )
            if reference != reference.strip():
                raise ValueError(
                    f"{item.instance_id}.{parameter_key} must be canonical text"
                )
            dependency = by_id.get(reference)
            if dependency is None:
                raise ValueError(
                    f"{item.instance_id}.{parameter_key} references missing device "
                    f"{reference!r}"
                )
            offered = descriptors[reference].capabilities
            if required_capability not in offered:
                raise ValueError(
                    f"device {reference!r} does not provide required capability "
                    f"{required_capability!r}"
                )
            refs.append(reference)
        dependencies[item.instance_id] = tuple(refs)

    # Kahn ordering is stable by the user's document order.  No factory or
    # discovery callable is invoked during this preflight.
    remaining = set(by_id)
    ordered: list[DeviceInstanceConfig] = []
    while remaining:
        ready = next(
            (
                item
                for item in devices
                if item.instance_id in remaining
                and all(
                    reference not in remaining
                    for reference in dependencies[item.instance_id]
                )
            ),
            None,
        )
        if ready is None:
            cycle = tuple(item.instance_id for item in devices if item.instance_id in remaining)
            raise ValueError(f"installation dependency cycle involves {cycle}")
        ordered.append(ready)
        remaining.remove(ready.instance_id)
    return tuple(ordered)


def dependency_capabilities(
    instance: DeviceInstanceConfig,
    connected: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    """Private composition helper keyed by the descriptor's parameter names."""

    descriptor = device_type(instance.type_id)
    result: dict[str, object] = {}
    for parameter_key, capability in descriptor.requirements:
        reference = instance.parameters[parameter_key]
        assert isinstance(reference, str)
        try:
            result[parameter_key] = connected[reference][capability]
        except KeyError as exc:  # preflight and topological order make this internal
            raise RuntimeError(
                f"dependency {reference!r} did not publish {capability!r}"
            ) from exc
    return MappingProxyType(result)


__all__ = [
    "CAPABILITY_CAMERA_CAPTURE",
    "CAPABILITY_CAMERA_MONITOR",
    "CAPABILITY_MOT_FIELD_CAPTURE",
    "CAPABILITY_CAMERA_SIGNAL_ASSOCIATION",
    "CAPABILITY_EXTERNAL_TRIGGER_CLIENT",
    "CAPABILITY_PULSE_EXECUTE",
    "CAPABILITY_READOUT_BINDING",
    "CAPABILITY_RF_TABLE",
    "DeviceTypeDescriptor",
    "device_type",
    "discover_device_types",
    "validate_installation_graph",
]
