"""Frozen package contract and discovery for built-in Logic-node leaves.

The discovery mechanism is framework infrastructure, not experiment semantics,
so it lives above the fixed :mod:`zlc_neutral_atom.logic_nodes` namespace that
it inspects.  There is no registration API, entry point, mutable global catalog,
or replacement hook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pkgutil import walk_packages
from typing import Protocol

from zlc_storage import canonical_text

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_declaration import LogicNodeDeclaration


class TaskConsoleProjection(Protocol):
    """Domain-neutral Workbench projectors injected by application composition."""

    def run(self, declaration: LogicNodeDeclaration, **kwargs) -> object: ...

    def processor(self, declaration: LogicNodeDeclaration, **kwargs) -> object: ...

    def resolve_final_or_saved(self, binding: object, **kwargs) -> object: ...


@dataclass(frozen=True, slots=True)
class LogicNodePackage:
    """One leaf-owned declaration plus its two application projections."""

    api_name: str
    declaration: LogicNodeDeclaration
    bind_api: Callable[[object, tuple[object, ...]], object]
    bind_task_console: Callable[
        [object, object, TaskConsoleProjection],
        object,
    ]
    task_console_order: int
    api_dependencies: tuple[str, ...] = ()
    close_api: Callable[[object], tuple[Exception, ...]] | None = None
    bind_artifact_capabilities: (
        Callable[[object], tuple[ArtifactCapability, ...]] | None
    ) = None

    def __post_init__(self) -> None:
        canonical_text(self.api_name, "Logic-node API name")
        if not self.api_name.isidentifier() or self.api_name.startswith("_"):
            raise ValueError("Logic-node API name must be a public Python identifier")
        if not isinstance(self.declaration, LogicNodeDeclaration):
            raise TypeError("declaration must be LogicNodeDeclaration")
        if not callable(self.bind_api):
            raise TypeError("bind_api must be callable")
        if not callable(self.bind_task_console):
            raise TypeError("bind_task_console must be callable")
        if type(self.task_console_order) is not int or self.task_console_order < 0:
            raise ValueError("task_console_order must be a non-negative int")
        dependencies = tuple(self.api_dependencies)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            for name in dependencies
        ):
            raise ValueError("API dependencies must be public Python identifiers")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("API dependencies must be unique")
        if self.api_name in dependencies:
            raise ValueError("a Logic-node API cannot depend on itself")
        object.__setattr__(self, "api_dependencies", dependencies)
        if self.close_api is not None and not callable(self.close_api):
            raise TypeError("close_api must be callable or None")
        if self.bind_artifact_capabilities is not None and not callable(
            self.bind_artifact_capabilities
        ):
            raise TypeError("bind_artifact_capabilities must be callable or None")


def discover_logic_node_packages() -> tuple[LogicNodePackage, ...]:
    """Import every fixed-name leaf package and return a validated frozen tuple."""

    namespace = import_module("zlc_neutral_atom.logic_nodes")
    prefix = namespace.__name__ + "."
    module_names = sorted(
        info.name
        for info in walk_packages(namespace.__path__, prefix=prefix)
        if info.name.endswith(".package")
    )
    if not module_names:
        raise RuntimeError("no Logic-node package declarations were discovered")

    packages: list[LogicNodePackage] = []
    for module_name in module_names:
        module = import_module(module_name)
        package = getattr(module, "LOGIC_NODE_PACKAGE", None)
        if not isinstance(package, LogicNodePackage):
            raise TypeError(
                f"{module_name} must export one LogicNodePackage as "
                "LOGIC_NODE_PACKAGE"
            )
        packages.append(package)

    by_name = {package.api_name: package for package in packages}
    if len(by_name) != len(packages):
        raise ValueError("Logic-node package API names must be unique")
    definition_keys = tuple(package.declaration.definition.key for package in packages)
    if len(set(definition_keys)) != len(definition_keys):
        raise ValueError("Logic-node package Definition keys must be unique")
    orders = tuple(package.task_console_order for package in packages)
    if len(set(orders)) != len(orders):
        raise ValueError("Logic-node package TaskConsole orders must be unique")
    for package in packages:
        missing = tuple(
            name for name in package.api_dependencies if name not in by_name
        )
        if missing:
            raise ValueError(
                f"Logic-node package {package.api_name!r} has missing API "
                f"dependencies {missing!r}"
            )
    return tuple(sorted(packages, key=lambda package: package.api_name))


__all__ = [
    "LogicNodePackage",
    "TaskConsoleProjection",
    "discover_logic_node_packages",
]
