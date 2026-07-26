"""Immutable public namespace projected from built-in Logic-node packages."""

from __future__ import annotations

from types import MappingProxyType

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability, ArtifactDispatch
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    discover_logic_node_packages,
)


class LogicNodeApis:
    """Concrete attributes such as ``exp.nodes.calibration``.

    Attributes are installed exactly once from the validated fixed namespace.
    The object exposes no registration, replacement, string-dispatch, or
    fallback lookup operation.
    """

    __slots__ = (
        "_apis",
        "_artifact_operations",
        "_construction_order",
        "_packages",
    )

    def __init__(
        self,
        packages: tuple[LogicNodePackage, ...],
        host: object,
        core_artifact_capabilities: tuple[ArtifactCapability, ...],
    ) -> None:
        if not packages:
            raise ValueError("LogicNodeApis requires at least one capability")
        object.__setattr__(self, "_packages", tuple(packages))
        pending = {package.api_name: package for package in packages}
        apis: dict[str, object] = {}
        construction_order: list[LogicNodePackage] = []
        try:
            while pending:
                ready = tuple(
                    package
                    for package in pending.values()
                    if all(name in apis for name in package.api_dependencies)
                )
                if not ready:
                    cycle = tuple(sorted(pending))
                    raise ValueError(
                        f"Logic-node API dependency cycle among {cycle!r}"
                    )
                for package in sorted(ready, key=lambda value: value.api_name):
                    dependencies = tuple(
                        apis[name] for name in package.api_dependencies
                    )
                    apis[package.api_name] = package.bind_api(host, dependencies)
                    construction_order.append(package)
                    pending.pop(package.api_name)
        except BaseException:
            for package in reversed(construction_order):
                if package.close_api is not None:
                    package.close_api(apis[package.api_name])
            raise
        try:
            artifact_capabilities = list(core_artifact_capabilities)
            for package in packages:
                binder = package.bind_artifact_capabilities
                if binder is None:
                    continue
                bound = tuple(binder(apis[package.api_name]))
                if any(not isinstance(item, ArtifactCapability) for item in bound):
                    raise TypeError(
                        f"Logic-node package {package.api_name!r} returned another "
                        "artifact capability type"
                    )
                artifact_capabilities.extend(bound)
            artifact_operations = ArtifactDispatch(tuple(artifact_capabilities))
        except BaseException:
            for package in reversed(construction_order):
                if package.close_api is not None:
                    package.close_api(apis[package.api_name])
            raise
        object.__setattr__(self, "_apis", MappingProxyType(apis))
        object.__setattr__(self, "_construction_order", tuple(construction_order))
        object.__setattr__(self, "_artifact_operations", artifact_operations)

    def __getattr__(self, name: str) -> object:
        try:
            return self._apis[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __dir__(self) -> list[str]:
        return sorted((*super().__dir__(), *self._apis))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("LogicNodeApis is frozen")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(package.api_name for package in self._packages)

    def close(self) -> tuple[Exception, ...]:
        failures: list[Exception] = []
        for package in reversed(self._construction_order):
            if package.close_api is None:
                continue
            failures.extend(package.close_api(self._apis[package.api_name]))
        return tuple(failures)


def compose_logic_node_apis(
    host: object,
    core_artifact_capabilities: tuple[ArtifactCapability, ...],
) -> LogicNodeApis:
    return LogicNodeApis(
        discover_logic_node_packages(),
        host,
        core_artifact_capabilities,
    )


def compose_task_console_attachments(
    nodes: LogicNodeApis,
    catalog: object,
    projection,
) -> tuple:
    if not isinstance(nodes, LogicNodeApis):
        raise TypeError("nodes must be LogicNodeApis")
    return tuple(
        package.bind_task_console(
            getattr(nodes, package.api_name),
            catalog,
            projection,
        )
        for package in sorted(
            nodes._packages,
            key=lambda package: package.task_console_order,
        )
    )


__all__ = [
    "LogicNodeApis",
    "compose_logic_node_apis",
    "compose_task_console_attachments",
]
