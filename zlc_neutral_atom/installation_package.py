"""Frozen package contract and discovery for built-in installation backends.

Executable backends live below the fixed :mod:`zlc_neutral_atom.devices`
namespace.  The discovery mechanism itself is installation framework
infrastructure; it exposes no mutable registry, entry point, replacement hook,
or fallback dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pkgutil import walk_packages

from zlc_neutral_atom.installation_runtime import _InstallationComposition
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.installation_plan import InstallationDevicePlan
from zlc_pulse import PulseDocument
from zlc_storage import canonical_text


@dataclass(frozen=True, slots=True)
class InstallationPackage:
    """Complete immutable product declaration for one built-in backend.

    The leaf owns its config value, codec, authoring schema, public topology,
    presentation label, and executable composition.  Generic config storage and
    DeviceManager project this value; neither carries a backend switch.
    """

    backend: str
    label: str
    config_type: type
    authoring_schema: Callable[[object | None], AuthoringSchema]
    config_from_parameters: Callable[[Mapping[str, object]], object]
    config_to_parameters: Callable[[object], Mapping[str, object]]
    device_plan: tuple[InstallationDevicePlan, ...]
    compose: Callable[[object, PulseDocument | None], _InstallationComposition]
    default: bool = False
    pulse_editor_mode: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "backend",
            canonical_text(self.backend, "installation backend"),
        )
        object.__setattr__(
            self,
            "label",
            canonical_text(self.label, "installation backend label"),
        )
        if not isinstance(self.config_type, type):
            raise TypeError("config_type must be a type")
        if type(self.default) is not bool:
            raise TypeError("default must be bool")
        if self.pulse_editor_mode is not None:
            object.__setattr__(
                self,
                "pulse_editor_mode",
                canonical_text(self.pulse_editor_mode, "pulse editor mode"),
            )
        for name in (
            "authoring_schema",
            "config_from_parameters",
            "config_to_parameters",
            "compose",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"{name} must be callable")
        plan = tuple(self.device_plan)
        if not plan or any(not isinstance(item, InstallationDevicePlan) for item in plan):
            raise TypeError("device_plan must contain InstallationDevicePlan values")
        roles = tuple(item.role for item in plan)
        if len(roles) != len(set(roles)):
            raise ValueError("installation device plan roles must be unique")
        object.__setattr__(self, "device_plan", plan)

        schema = self.authoring_schema(None)
        if not isinstance(schema, AuthoringSchema):
            raise TypeError("authoring_schema must return AuthoringSchema")
        referenced = frozenset(
            key for item in plan for key in item.configuration_keys
        )
        if not referenced <= frozenset(schema.keys):
            raise ValueError(
                "installation device plan references undeclared config fields"
            )

    def require_config(self, config: object) -> object:
        if type(config) is not self.config_type:
            raise TypeError(
                f"{self.backend} installation requires {self.config_type.__name__}"
            )
        return config

    def parameters(self, config: object) -> dict[str, object]:
        self.require_config(config)
        values = dict(self.config_to_parameters(config))
        schema = self.authoring_schema(config)
        if not isinstance(schema, AuthoringSchema):
            raise TypeError("authoring_schema must return AuthoringSchema")
        if set(values) != set(schema.keys):
            raise ValueError(
                f"{self.backend} config codec differs from its authoring schema"
            )
        return schema.freeze(values)


@cache
def discover_installation_packages() -> tuple[InstallationPackage, ...]:
    """Return the validated frozen set of built-in executable backends."""

    namespace = import_module("zlc_neutral_atom.devices")
    prefix = namespace.__name__ + "."
    module_names = sorted(
        info.name
        for info in walk_packages(namespace.__path__, prefix=prefix)
        if info.name.endswith(".package")
    )
    if not module_names:
        raise RuntimeError("no installation backend packages were discovered")

    packages: list[InstallationPackage] = []
    for module_name in module_names:
        module = import_module(module_name)
        package = getattr(module, "INSTALLATION_PACKAGE", None)
        if not isinstance(package, InstallationPackage):
            raise TypeError(
                f"{module_name} must export one InstallationPackage as "
                "INSTALLATION_PACKAGE"
            )
        packages.append(package)

    backends = tuple(package.backend for package in packages)
    if len(set(backends)) != len(backends):
        raise ValueError("installation package backend names must be unique")
    config_types = tuple(package.config_type for package in packages)
    if len(set(config_types)) != len(config_types):
        raise ValueError("installation package config types must be unique")
    if sum(package.default for package in packages) != 1:
        raise ValueError("exactly one installation package must declare default=True")
    return tuple(sorted(packages, key=lambda package: package.backend))


def installation_package(backend: str) -> InstallationPackage:
    """Resolve one declared backend from the frozen built-in namespace."""

    name = canonical_text(backend, "installation backend")
    matches = tuple(
        package
        for package in discover_installation_packages()
        if package.backend == name
    )
    if len(matches) != 1:
        raise ValueError(f"unsupported installation backend {name!r}")
    return matches[0]


def installation_package_for_config(config: object) -> InstallationPackage:
    """Resolve the sole leaf owning the exact config value type."""

    matches = tuple(
        package
        for package in discover_installation_packages()
        if type(config) is package.config_type
    )
    if len(matches) != 1:
        raise TypeError("config must have exactly one built-in installation owner")
    return matches[0]


def default_installation_package() -> InstallationPackage:
    """Return the sole leaf explicitly declaring the product default."""

    return next(
        package for package in discover_installation_packages() if package.default
    )


__all__ = [
    "InstallationPackage",
    "discover_installation_packages",
    "default_installation_package",
    "installation_package",
    "installation_package_for_config",
]
