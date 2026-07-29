"""Immutable public namespace projected from built-in Logic-node packages."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from zlc_neutral_atom.artifact_dispatch import ArtifactCapability, ArtifactDispatch
from zlc_neutral_atom.logic_node_declaration import DynamicChoicePresentation
from zlc_neutral_atom.logic_node_package import (
    LogicNodePackage,
    discover_logic_node_packages,
)


def _ui_invoker(package: LogicNodePackage, open_workbench):
    """Bind one leaf's inert descriptors without importing optional UI now."""

    if not callable(open_workbench):
        raise TypeError("open_workbench must be callable")
    api_name = package.api_name
    contributions = tuple(package.ui_contributions)

    def invoke(purpose: str, *args, **kwargs):
        from Zou_lab_control.workbench import _invoke_logic_node_ui

        name = str(purpose).strip()
        if not name:
            raise ValueError("Logic-node UI purpose must be non-empty")
        return open_workbench(
            f"logic-node:{api_name}:{name}",
            lambda: _invoke_logic_node_ui(
                contributions,
                name,
                *args,
                **kwargs,
            ),
            existing_error=(
                f"Logic-node {api_name!r} already owns its {name!r} window"
            ),
        )

    return invoke


def _available_packages(
    packages: tuple[LogicNodePackage, ...],
    catalog: object,
    apparatus: tuple[object, ...],
) -> tuple[LogicNodePackage, ...]:
    """Evaluate immutable installation availability before constructing APIs."""

    unavailable: dict[str, str] = {}
    for package in packages:
        predicate = package.availability
        reason = None if predicate is None else predicate(catalog, apparatus)
        if reason is not None:
            if not isinstance(reason, str) or not reason.strip():
                raise TypeError(
                    f"Logic-node package {package.api_name!r} availability "
                    "must return a non-empty reason or None"
                )
            unavailable[package.api_name] = reason.strip()

    changed = True
    while changed:
        changed = False
        for package in packages:
            if package.api_name in unavailable:
                continue
            blocked = tuple(
                name for name in package.api_dependencies if name in unavailable
            )
            if blocked:
                unavailable[package.api_name] = (
                    "unavailable Logic-node dependencies: " + ", ".join(blocked)
                )
                changed = True
    return tuple(
        package for package in packages if package.api_name not in unavailable
    )


def _resolved_dynamic_choices(
    package: LogicNodePackage,
    facts: Mapping[str, object],
) -> tuple[DynamicChoicePresentation, ...]:
    fact_name = package.dynamic_choice_fact
    if fact_name is None:
        return ()
    try:
        context = facts[fact_name]
    except KeyError as error:
        raise ValueError(
            f"Logic-node package {package.api_name!r} has unresolved dynamic "
            f"choice fact {fact_name!r}"
        ) from error
    resolver = package.declaration.resolve_dynamic_choices
    if resolver is None:
        raise RuntimeError("dynamic choice fact lost its declaration resolver")
    values = tuple(resolver(context))
    if any(not isinstance(value, DynamicChoicePresentation) for value in values):
        raise TypeError("dynamic choice resolver returned another value type")
    expected = tuple(
        field.key
        for field in package.declaration.authoring_schema.fields
        if field.dynamic_choices
    )
    if tuple(value.field_key for value in values) != expected:
        raise ValueError("dynamic choice resolver changed its declared field order")
    return values


class LogicNodeApis:
    """Concrete frozen attributes such as ``exp.nodes.calibration``."""

    __slots__ = (
        "_apis",
        "_artifact_operations",
        "_construction_order",
        "_dynamic_choices",
        "_packages",
    )

    def __init__(
        self,
        packages: tuple[LogicNodePackage, ...],
        facts: Mapping[str, object],
        catalog: object,
        apparatus: tuple[object, ...],
        core_artifact_capabilities: tuple[ArtifactCapability, ...],
    ) -> None:
        all_packages = tuple(packages)
        available = _available_packages(
            all_packages,
            catalog,
            tuple(apparatus),
        )
        frozen_facts = dict(facts)
        requirements_by_name: dict[str, tuple[object, ...]] = {}
        dynamic_by_name: dict[str, tuple[DynamicChoicePresentation, ...]] = {}
        for package in available:
            missing = tuple(
                name for name in package.api_requirements if name not in frozen_facts
            )
            if missing:
                raise ValueError(
                    f"Logic-node package {package.api_name!r} has unresolved "
                    f"application facts {missing!r}"
                )
            values = tuple(
                _ui_invoker(package, frozen_facts[name])
                if name == "open_ui"
                else frozen_facts[name]
                for name in package.api_requirements
            )
            if "open_ui" in package.api_requirements and not package.ui_contributions:
                raise ValueError(
                    f"Logic-node package {package.api_name!r} requests UI without "
                    "declaring a contribution"
                )
            requirements_by_name[package.api_name] = values
            dynamic_by_name[package.api_name] = _resolved_dynamic_choices(
                package,
                frozen_facts,
            )

        apis: dict[str, object] = {}
        construction_order: list[LogicNodePackage] = []
        try:
            for package in available:
                dependencies = tuple(
                    apis[name] for name in package.api_dependencies
                )
                apis[package.api_name] = package.bind_api(
                    requirements_by_name[package.api_name],
                    dependencies,
                )
                construction_order.append(package)
        except BaseException:
            for package in reversed(construction_order):
                if package.close_api is not None:
                    package.close_api(apis[package.api_name])
            raise
        try:
            artifact_capabilities = list(core_artifact_capabilities)
            for package in available:
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
        object.__setattr__(self, "_packages", available)
        object.__setattr__(self, "_apis", MappingProxyType(apis))
        object.__setattr__(
            self,
            "_dynamic_choices",
            MappingProxyType(dynamic_by_name),
        )
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

    def _composition_entries(self):
        """Return frozen package/API/choice triples to the application root."""

        return tuple(
            (
                package,
                self._apis[package.api_name],
                self._dynamic_choices[package.api_name],
            )
            for package in self._packages
        )

    def close(self) -> tuple[Exception, ...]:
        failures: list[Exception] = []
        for package in reversed(self._construction_order):
            if package.close_api is None:
                continue
            failures.extend(package.close_api(self._apis[package.api_name]))
        return tuple(failures)


def _logic_node_composition_facts(readout):
    """Freeze the one local fact pool; no leaf receives this mapping."""

    from zlc_neutral_atom.installation import DeviceRef
    from zlc_pulse import PulseDocument, load_pulse_document
    from zlc_storage.paths import resolve_under

    from ._application_services import (
        application_operation_guard,
        application_start_run,
        open_workbench_handle,
        service_guard,
    )
    from ._readout_core import ReadoutFacade

    if not isinstance(readout, ReadoutFacade):
        raise TypeError("readout must be ReadoutFacade")
    services = readout._services
    with service_guard(services) as current:
        workspace = current.workspace_paths
        catalog = current.catalog
        apparatus = tuple(current.installation.readout_apparatus_facts)
        capture_repository = current.capture_repository
        runtime = current.runtime
        association_authorities = MappingProxyType(
            dict(current.installation.camera_signal_association_authorities)
        )
        camera_roles = tuple(catalog.roles("camera"))
        sequencer_roles = tuple(catalog.roles("sequencer"))
        rf_roles = tuple(catalog.roles("rf"))
        camera_refs = MappingProxyType(
            {role: catalog.require(role).ref for role in camera_roles}
        )
        sequencer_refs = MappingProxyType(
            {role: catalog.require(role).ref for role in sequencer_roles}
        )
        rf_refs = MappingProxyType(
            {role: catalog.require(role).ref for role in rf_roles}
        )
        camera_ports = MappingProxyType(
            {
                role: runtime.camera_port(reference)
                for role, reference in camera_refs.items()
            }
        )
        camera_monitor_ports = MappingProxyType(
            {
                role: runtime.camera_monitor_port(reference)
                for role, reference in camera_refs.items()
            }
        )
        pulse_ports = MappingProxyType(
            {
                role: runtime.pulse_port(reference)
                for role, reference in sequencer_refs.items()
            }
        )
        rf_ports = MappingProxyType(
            {
                role: runtime.rf_port(reference)
                for role, reference in rf_refs.items()
            }
        )
        readout_camera_roles = tuple(value.camera_role for value in apparatus)

    def resolve_reference(requested, references, preferred, domain):
        if requested is not None:
            try:
                return references[str(requested)]
            except KeyError as error:
                raise ValueError(
                    f"device role {requested!r} is not an installed {domain}"
                ) from error
        for role in preferred:
            if role in references:
                return references[role]
        if len(references) != 1:
            raise ValueError(
                f"installation has {len(references)} {domain} roles; "
                "choose one explicitly"
            )
        return next(iter(references.values()))

    def resolve_camera_ref(requested):
        return resolve_reference(
            requested,
            camera_refs,
            ("camera", "readout"),
            "Camera",
        )

    def resolve_sequencer_ref(requested):
        return resolve_reference(
            requested,
            sequencer_refs,
            ("sequencer",),
            "Sequencer",
        )

    def bound_port(reference, references, ports, domain):
        if not isinstance(reference, DeviceRef):
            raise TypeError(f"{domain} reference must be DeviceRef")
        expected = references.get(reference.role)
        if expected != reference:
            raise RuntimeError(f"{domain} reference belongs to another installation")
        return ports[reference.role]

    def load_pulse(value):
        if isinstance(value, PulseDocument):
            return value
        return load_pulse_document(resolve_under(workspace.pulses_root, value))

    def start_run(plan):
        return application_start_run(services, plan)

    def wait_run(handle):
        return runtime.wait(handle)

    def operation_guard():
        return application_operation_guard(services)

    def open_ui(key, compose, *, existing_error=None):
        return open_workbench_handle(
            services,
            key,
            compose,
            existing_error=existing_error,
        )

    facts = MappingProxyType(
        {
            "repository_root": workspace.repository_root,
            "pulses_root": workspace.pulses_root,
            "output_root": workspace.output_root,
            "capture_repository": capture_repository,
            "camera_signal_association_authorities": association_authorities,
            "readout_apparatus_facts": apparatus,
            "camera_roles": camera_roles,
            "rf_roles": rf_roles,
            "readout_camera_roles": readout_camera_roles,
            "resolve_camera_ref": resolve_camera_ref,
            "resolve_sequencer_ref": resolve_sequencer_ref,
            "load_pulse": load_pulse,
            "pulse_port": lambda reference: bound_port(
                reference,
                sequencer_refs,
                pulse_ports,
                "Sequencer",
            ),
            "camera_port": lambda reference: bound_port(
                reference,
                camera_refs,
                camera_ports,
                "Camera",
            ),
            "camera_monitor_port": lambda reference: bound_port(
                reference,
                camera_refs,
                camera_monitor_ports,
                "Camera monitor",
            ),
            "rf_ports": tuple(sorted(rf_ports.items())),
            "start_run": start_run,
            "wait_run": wait_run,
            "operation_guard": operation_guard,
            "open_ui": open_ui,
        }
    )
    return facts, catalog, apparatus


def compose_logic_node_apis(
    readout: object,
    core_artifact_capabilities: tuple[ArtifactCapability, ...],
) -> LogicNodeApis:
    facts, catalog, apparatus = _logic_node_composition_facts(readout)
    return LogicNodeApis(
        discover_logic_node_packages(),
        facts,
        catalog,
        apparatus,
        core_artifact_capabilities,
    )


__all__ = ["LogicNodeApis", "compose_logic_node_apis"]
