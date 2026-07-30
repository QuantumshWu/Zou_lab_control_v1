"""Frozen package contract and discovery for built-in Logic-node leaves.

The discovery mechanism is framework infrastructure, not experiment semantics,
so it lives above the fixed :mod:`zlc_neutral_atom.logic_nodes` namespace that
it inspects.  There is no registration API, entry point, mutable global catalog,
or replacement hook.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from pathlib import Path

from zlc_storage import canonical_text

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.logic_node_declaration import LogicNodeDeclaration


@dataclass(frozen=True, slots=True)
class UiContributionDescriptor:
    """One lazy optional-UI owner without importing its module headlessly."""

    purpose: str
    module: str
    symbol: str

    def __post_init__(self) -> None:
        for value, label in (
            (self.purpose, "UI contribution purpose"),
            (self.module, "UI contribution module"),
            (self.symbol, "UI contribution symbol"),
        ):
            canonical_text(value, label)
        if not self.purpose.isidentifier() or self.purpose.startswith("_"):
            raise ValueError("UI contribution purpose must be a public identifier")
        if not self.symbol.isidentifier() or self.symbol.startswith("_"):
            raise ValueError("UI contribution symbol must be a public identifier")


@dataclass(frozen=True, slots=True)
class LogicNodePackage:
    """One leaf declaration and the narrow facts needed to compose it."""

    api_name: str
    declaration: LogicNodeDeclaration
    api_requirements: tuple[str, ...]
    bind_api: Callable[[tuple[object, ...], tuple[object, ...]], object]
    prepare_hosted: Callable[[object, object, object | None], object]
    api_dependencies: tuple[str, ...] = ()
    availability: Callable[[object, tuple[object, ...]], str | None] | None = None
    dynamic_choice_fact: str | None = None
    bind_hosted_request: Callable[[object, object, object], object] | None = None
    start_prepared: Callable[[object, object, object], object] | None = None
    resolve_artifact_reference: (
        Callable[[object, object, Callable[..., object]], object] | None
    ) = None
    project_signal_presentation: (
        Callable[[object, str, object, tuple[object, ...]], object | None] | None
    ) = None
    ui_contributions: tuple[UiContributionDescriptor, ...] = ()
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
        requirements = tuple(self.api_requirements)
        if any(
            not isinstance(name, str)
            or not name.isidentifier()
            or name.startswith("_")
            for name in requirements
        ):
            raise ValueError("API requirements must be public Python identifiers")
        if len(set(requirements)) != len(requirements):
            raise ValueError("API requirements must be unique")
        object.__setattr__(self, "api_requirements", requirements)
        if not callable(self.bind_api):
            raise TypeError("bind_api must be callable")
        if not callable(self.prepare_hosted):
            raise TypeError("prepare_hosted must be callable")
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
        if self.availability is not None and not callable(self.availability):
            raise TypeError("availability must be callable or None")
        dynamic_fact = self.dynamic_choice_fact
        resolver = self.declaration.resolve_dynamic_choices
        if (dynamic_fact is None) != (resolver is None):
            raise ValueError(
                "dynamic_choice_fact must exactly match the declaration resolver"
            )
        for name in (
            "bind_hosted_request",
            "start_prepared",
            "resolve_artifact_reference",
            "project_signal_presentation",
        ):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable or None")
        if (self.declaration.bind_request is None) == (
            self.bind_hosted_request is None
        ):
            raise ValueError(
                "exactly one declaration or package request binder is required"
            )
        contributions = tuple(self.ui_contributions)
        if any(
            not isinstance(value, UiContributionDescriptor)
            for value in contributions
        ):
            raise TypeError(
                "ui_contributions must contain UiContributionDescriptor values"
            )
        purposes = tuple(value.purpose for value in contributions)
        if len(set(purposes)) != len(purposes):
            raise ValueError("UI contribution purposes must be unique within a leaf")
        object.__setattr__(self, "ui_contributions", contributions)
        if self.close_api is not None and not callable(self.close_api):
            raise TypeError("close_api must be callable or None")
        if self.bind_artifact_capabilities is not None and not callable(
            self.bind_artifact_capabilities
        ):
            raise TypeError("bind_artifact_capabilities must be callable or None")


@cache
def discover_logic_node_packages() -> tuple[LogicNodePackage, ...]:
    """Return validated fixed-name leaves in stable dependency order."""

    namespace = import_module("zlc_neutral_atom.logic_nodes")
    prefix = namespace.__name__ + "."
    module_names: set[str] = set()
    for root_value in namespace.__path__:
        root = Path(root_value)
        for path in root.rglob("package.py"):
            relative = path.relative_to(root).with_suffix("")
            if "ui" in relative.parts:
                continue
            module_names.add(prefix + ".".join(relative.parts))
    if not module_names:
        raise RuntimeError("no Logic-node package declarations were discovered")

    packages: list[LogicNodePackage] = []
    for module_name in sorted(module_names):
        module = import_module(module_name)
        package = getattr(module, "LOGIC_NODE_PACKAGE", None)
        if not isinstance(package, LogicNodePackage):
            raise TypeError(
                f"{module_name} must export one LogicNodePackage as "
                "LOGIC_NODE_PACKAGE"
            )
        expected_owner = module_name.removesuffix(".package")
        if package.declaration.definition.key.owner_package != expected_owner:
            raise ValueError(
                f"{module_name} Definition owner must be {expected_owner!r}"
            )
        expected_ui_prefix = expected_owner + ".ui."
        for contribution in package.ui_contributions:
            if not contribution.module.startswith(expected_ui_prefix):
                raise ValueError(
                    f"{module_name} UI contribution {contribution.purpose!r} "
                    f"must be owned below {expected_ui_prefix!r}"
                )
        packages.append(package)

    by_name = {package.api_name: package for package in packages}
    if len(by_name) != len(packages):
        raise ValueError("Logic-node package API names must be unique")
    definition_keys = tuple(package.declaration.definition.key for package in packages)
    if len(set(definition_keys)) != len(definition_keys):
        raise ValueError("Logic-node package Definition keys must be unique")
    for package in packages:
        missing = tuple(
            name for name in package.api_dependencies if name not in by_name
        )
        if missing:
            raise ValueError(
                f"Logic-node package {package.api_name!r} has missing API "
                f"dependencies {missing!r}"
            )
    pending = dict(by_name)
    resolved: set[str] = set()
    ordered: list[LogicNodePackage] = []
    while pending:
        ready = tuple(
            sorted(
                name
                for name, package in pending.items()
                if set(package.api_dependencies) <= resolved
            )
        )
        if not ready:
            raise ValueError(
                "Logic-node API dependency cycle among "
                f"{tuple(sorted(pending))!r}"
            )
        for name in ready:
            ordered.append(pending.pop(name))
            resolved.add(name)
    return tuple(ordered)


__all__ = [
    "LogicNodePackage",
    "UiContributionDescriptor",
    "discover_logic_node_packages",
]
