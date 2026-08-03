"""Catalog-driven public Logic-node API and application context."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from functools import partial
from pathlib import Path
import threading
import time
import warnings
from types import MappingProxyType
from uuid import uuid4

from zlc_neutral_atom.input_spec import ArtifactInputSpec, DatasetInputSpec, NodeInputSpec
from zlc_neutral_atom.logic_node import (
    LogicNodeApplicationContext,
    LogicNodeDescriptor,
    UiContribution,
    discover_logic_nodes,
)
from zlc_neutral_atom.runtime.hosted_run import LogicNodeHost
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_storage import canonical_text, resolve_under

from ._application_services import (
    ExperimentServices,
    application_operation_guard,
    application_start_run,
    open_workbench_handle,
    service_guard,
)


def _input_specs_by_key(
    descriptor: LogicNodeDescriptor,
) -> Mapping[str, NodeInputSpec]:
    return MappingProxyType({spec.key: spec for spec in descriptor.input_specs})


class _BoundLogicNodeContext(LogicNodeApplicationContext):
    """One invocation's resolved application facts, never a service locator."""

    __slots__ = (
        "_authored",
        "_descriptor",
        "_inputs",
        "_services",
    )

    def __init__(
        self,
        services: ExperimentServices,
        descriptor: LogicNodeDescriptor,
        authored: Mapping[str, object] | None = None,
        inputs: Mapping[str, object] | None = None,
    ) -> None:
        if not isinstance(services, ExperimentServices):
            raise TypeError("services must be ExperimentServices")
        if not isinstance(descriptor, LogicNodeDescriptor):
            raise TypeError("descriptor must be LogicNodeDescriptor")
        # Bind the immutable application owner before resolving optional
        # artifact inputs.  ``default_artifact`` is intentionally resolved
        # through this same context (not by TaskConsole or a leaf), so it
        # must be available while the input map is frozen.
        self._services = services
        self._descriptor = descriptor
        if (authored is None) != (inputs is None):
            raise ValueError("authored values and inputs must be bound together")
        frozen_authored: Mapping[str, object] | None = None
        resolved_inputs: Mapping[str, object] | None = None
        if authored is not None and inputs is not None:
            frozen_authored = descriptor.authoring_schema.freeze(authored)
            specs = _input_specs_by_key(descriptor)
            unknown = set(inputs).difference(specs)
            if unknown:
                raise ValueError(
                    "Logic-node inputs contain undeclared keys; "
                    f"unknown={tuple(sorted(unknown))}"
                )
            bound_inputs: dict[str, object] = {}
            for key, spec in specs.items():
                if isinstance(spec, DatasetInputSpec):
                    if key not in inputs:
                        raise ValueError(f"Dataset input {key!r} is unresolved")
                    value = inputs[key]
                    if not isinstance(value, str) or not value.strip():
                        raise TypeError(
                            f"Dataset input {key!r} must be a signal key"
                        )
                    bound_inputs[key] = value.strip()
                elif isinstance(spec, ArtifactInputSpec):
                    value = inputs.get(key)
                    if value is None and spec.allow_saved_reference:
                        value = self.default_artifact(spec.output_contract_id)
                        if value is None and spec.default_reference_path is not None:
                            value = resolve_under(
                                self.project_root,
                                spec.default_reference_path,
                            )
                    if value is None:
                        raise ValueError(f"Artifact input {key!r} is unresolved")
                    bound_inputs[key] = value
                else:  # pragma: no cover - NodeInputSpec vocabulary is closed
                    raise TypeError("descriptor contains another input type")
            resolved_inputs = MappingProxyType(bound_inputs)
        self._authored = (
            None
            if frozen_authored is None
            else MappingProxyType(dict(frozen_authored))
        )
        self._inputs = resolved_inputs

    @property
    def project_root(self) -> Path:
        return self._services.workspace_paths.project_root

    @property
    def pulses_root(self) -> Path:
        return self._services.workspace_paths.pulses_root

    @property
    def tasks_root(self) -> Path:
        return self._services.workspace_paths.tasks_root

    @property
    def runs_root(self) -> Path:
        return self._services.workspace_paths.runs_root

    @property
    def figures_root(self) -> Path:
        return self._services.workspace_paths.figures_root

    @property
    def device_catalog(self):
        return self._services.catalog

    @property
    def readout_bindings(self):
        return tuple(self._services.installation.readout_installation_bindings)

    @property
    def signal_plane(self):
        return self._services.signal_plane

    def input(self, spec: NodeInputSpec) -> object:
        expected = _input_specs_by_key(self._descriptor).get(spec.key)
        if expected != spec:
            raise ValueError("input spec is not owned by this Logic-node descriptor")
        if self._inputs is None:
            raise RuntimeError("this operation has no bound Logic-node inputs")
        return self._inputs[spec.key]

    def device(
        self,
        field_key: str,
        capability: str | None = None,
    ) -> object:
        key = canonical_text(field_key, "device authoring field")
        info, required_capabilities = self._selected_device(key)
        if capability is None:
            if len(required_capabilities) != 1:
                raise ValueError(
                    f"device field {key!r} declares several capabilities; "
                    "select one explicitly"
                )
            selected_capability = required_capabilities[0]
        else:
            selected_capability = canonical_text(capability, "device capability")
            if selected_capability not in required_capabilities:
                raise ValueError(
                    f"capability {selected_capability!r} is not declared for "
                    f"device field {key!r}"
                )
        return self._services.runtime.require_capability(
            info.ref,
            selected_capability,
        )

    def optional_device(
        self,
        field_key: str,
        capability: str,
    ) -> object | None:
        key = canonical_text(field_key, "device authoring field")
        selected_capability = canonical_text(
            capability,
            "optional device capability",
        )
        declared = dict(self._descriptor.optional_device_capabilities).get(key, ())
        if selected_capability not in declared:
            raise ValueError(
                f"capability {selected_capability!r} is not declared optional "
                f"for device field {key!r}"
            )
        info, _required = self._selected_device(key)
        if selected_capability not in info.capabilities:
            return None
        return self._services.runtime.require_capability(
            info.ref,
            selected_capability,
        )

    def _selected_device(self, key: str):
        requirements = dict(self._descriptor.device_requirements)
        try:
            required_capabilities = requirements[key]
        except KeyError as error:
            raise KeyError(
                f"{key!r} is not a descriptor-declared device field"
            ) from error
        if self._authored is None:
            raise RuntimeError("this operation has no bound Logic-node request")
        instance_id = self._authored[key]
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"select a device for {key!r}")
        info = self._services.catalog.require(instance_id.strip())
        missing = tuple(
            value
            for value in required_capabilities
            if value not in info.capabilities
        )
        if missing:
            raise ValueError(
                f"device {info.instance_id!r} does not provide {missing!r}"
            )
        return info, required_capabilities

    def start_run(self, plan: RunPlan) -> RunHandle:
        return application_start_run(self._services, plan)

    def wait_run(self, handle: RunHandle) -> object:
        return self._services.runtime.wait(handle)

    def operation_guard(self) -> AbstractContextManager[object]:
        return application_operation_guard(self._services)

    def default_artifact(self, contract_id: str) -> object | None:
        contract = canonical_text(contract_id, "artifact contract id")
        with self._services.operation_lock:
            if self._services.state != "OPEN":
                raise RuntimeError("Experiment is closing or closed")
            return self._services.default_artifacts.get(contract)

    def remember_artifact(self, contract_id: str, reference: object) -> None:
        contract = canonical_text(contract_id, "artifact contract id")
        if reference is None:
            raise ValueError("default artifact reference cannot be None")
        with self._services.operation_lock:
            if self._services.state != "OPEN":
                raise RuntimeError("Experiment is closing or closed")
            self._services.default_artifacts[contract] = reference

    def open_ui(
        self,
        contribution: UiContribution,
        *args,
        **kwargs,
    ) -> object:
        if contribution not in self._descriptor.ui_contributions:
            raise ValueError("UI contribution is not owned by this descriptor")

        def compose():
            from Zou_lab_control.workbench import _invoke_logic_node_ui

            return _invoke_logic_node_ui(contribution, *args, **kwargs)

        purpose = contribution.purpose
        return open_workbench_handle(
            self._services,
            f"logic-node:{self._descriptor.api_name}:{purpose}",
            compose,
            existing_error=(
                f"Logic-node {self._descriptor.api_name!r} already owns "
                f"its {purpose!r} window"
            ),
        )


class _HostRegistry:
    """Application-owned set used only to close public API hosts."""

    __slots__ = ("_hosts", "_lock")

    def __init__(self) -> None:
        self._hosts: set[LogicNodeHost] = set()
        self._lock = threading.RLock()

    def add(self, host: LogicNodeHost) -> None:
        if not isinstance(host, LogicNodeHost):
            raise TypeError("host must be LogicNodeHost")
        with self._lock:
            self._hosts.add(host)

    def discard(self, host: LogicNodeHost) -> None:
        with self._lock:
            self._hosts.discard(host)

    def close(self) -> tuple[Exception, ...]:
        with self._lock:
            hosts = tuple(self._hosts)
        failures: list[Exception] = []
        for host in hosts:
            try:
                host.cancel("Experiment is closing")
                while host.running:
                    host.poll()
                    time.sleep(0.005)
                host.shutdown()
            except Exception as error:
                failures.append(error)
            else:
                self.discard(host)
        return tuple(failures)


class NodeApi:
    """The same lightweight API projection for every discovered leaf."""

    __slots__ = ("_descriptor", "_registry", "_services")

    def __init__(
        self,
        descriptor: LogicNodeDescriptor,
        services: ExperimentServices,
        registry: _HostRegistry,
    ) -> None:
        self._descriptor = descriptor
        self._services = services
        self._registry = registry

    @property
    def descriptor(self) -> LogicNodeDescriptor:
        return self._descriptor

    @property
    def current_artifact(self) -> object | None:
        outputs = self._descriptor.artifact_outputs
        if len(outputs) != 1:
            raise AttributeError(
                "current_artifact requires exactly one declared Artifact output"
            )
        context = self._context()
        return context.default_artifact(outputs[0].contract_id)

    def build(
        self,
        values: Mapping[str, object] | None = None,
        /,
        **fields: object,
    ) -> object:
        authored = self._authored(values, fields)
        request = self._descriptor.build_request(authored)
        self._descriptor.outputs_for(request)
        return request

    def start(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        inputs: Mapping[str, object] | None = None,
        instance_id: str | None = None,
        request_owner_wake: Callable[[], None] | None = None,
        **fields: object,
    ) -> LogicNodeHost:
        with application_operation_guard(self._services):
            authored = self._authored(values, fields)
            request = self._descriptor.build_request(authored)
            context = _BoundLogicNodeContext(
                self._services,
                self._descriptor,
                authored,
                inputs or {},
            )
            host = LogicNodeHost.create(
                self._descriptor,
                request,
                context,
                instance_id or f"api_{self._descriptor.api_name}_{uuid4().hex}",
                request_owner_wake or (lambda: None),
            )
            try:
                host.start()
            except BaseException:
                host.shutdown()
                raise
            self._registry.add(host)
        return host

    def run(
        self,
        values: Mapping[str, object] | None = None,
        /,
        *,
        inputs: Mapping[str, object] | None = None,
        **fields: object,
    ) -> object:
        if self._descriptor.definition.kind == "processor":
            raise ValueError("continuous Processor nodes use start()/stop()")
        wake = threading.Event()
        host = self.start(
            values,
            inputs=inputs,
            request_owner_wake=wake.set,
            **fields,
        )
        while not host.terminal:
            host.poll()
            if not host.terminal:
                wake.wait(0.05)
                wake.clear()
        error = host.last_error
        result = host.final_result
        reported_warnings = host.observation.warnings
        try:
            host.shutdown()
        finally:
            self._registry.discard(host)
        if error is not None:
            raise RuntimeError(error)
        for message in reported_warnings:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        return result

    def stop(self, host: LogicNodeHost) -> None:
        if not isinstance(host, LogicNodeHost):
            raise TypeError("host must be LogicNodeHost")
        if host.definition_key != self._descriptor.definition.key:
            raise ValueError("host belongs to another Logic-node definition")
        host.cancel("NodeApi stop requested")
        host.poll()

    def open_ui(self, purpose: str, *args, **kwargs) -> object:
        name = canonical_text(purpose, "Logic-node UI purpose")
        contribution = next(
            (
                value
                for value in self._descriptor.ui_contributions
                if value.purpose == name
            ),
            None,
        )
        if contribution is None:
            raise KeyError(
                f"Logic-node {self._descriptor.api_name!r} has no {name!r} UI"
            )
        return self._context().open_ui(contribution, *args, **kwargs)

    def __getattr__(self, name: str):
        operation = self._descriptor.operations.get(name)
        if operation is None:
            raise AttributeError(name)
        return partial(operation, self._context())

    def __dir__(self) -> list[str]:
        return sorted((*super().__dir__(), *self._descriptor.operations))

    def _authored(
        self,
        values: Mapping[str, object] | None,
        fields: Mapping[str, object],
    ) -> dict[str, object]:
        if values is not None and fields:
            raise ValueError("pass a values mapping or keyword fields, not both")
        source = fields if values is None else values
        if not isinstance(source, Mapping):
            raise TypeError("Logic-node values must be a mapping")
        return self._descriptor.authoring_schema.freeze(source)

    def _context(
        self,
        authored: Mapping[str, object] | None = None,
        inputs: Mapping[str, object] | None = None,
    ) -> _BoundLogicNodeContext:
        with service_guard(self._services):
            return _BoundLogicNodeContext(
                self._services,
                self._descriptor,
                authored,
                inputs,
            )


class LogicNodeApis:
    """Frozen ``exp.nodes`` namespace projected from descriptor discovery."""

    __slots__ = ("_apis", "_by_key", "_descriptors", "_registry", "_services")

    def __init__(self, services: ExperimentServices) -> None:
        descriptors = discover_logic_nodes()
        registry = _HostRegistry()
        apis = {
            value.api_name: NodeApi(value, services, registry)
            for value in descriptors
        }
        self._services = services
        self._descriptors = descriptors
        self._registry = registry
        self._apis = MappingProxyType(apis)
        self._by_key = MappingProxyType(
            {value.definition.key: value for value in descriptors}
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(value.api_name for value in self._descriptors)

    @property
    def descriptors(self) -> tuple[LogicNodeDescriptor, ...]:
        return self._descriptors

    def descriptor_for(self, key) -> LogicNodeDescriptor | None:
        return self._by_key.get(key)

    def create_host(
        self,
        descriptor: LogicNodeDescriptor,
        authored: Mapping[str, object],
        inputs: Mapping[str, object],
        instance_id: str,
        request_owner_wake: Callable[[], None],
    ) -> LogicNodeHost:
        with application_operation_guard(self._services):
            current = self._by_key.get(descriptor.definition.key)
            if current is not descriptor:
                raise ValueError("descriptor does not belong to this Experiment")
            frozen = descriptor.authoring_schema.freeze(authored)
            request = descriptor.build_request(frozen)
            context = _BoundLogicNodeContext(
                self._services,
                descriptor,
                frozen,
                inputs,
            )
            return LogicNodeHost.create(
                descriptor,
                request,
                context,
                instance_id,
                request_owner_wake,
            )

    def __getattr__(self, name: str) -> NodeApi:
        try:
            return self._apis[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __dir__(self) -> list[str]:
        return sorted((*super().__dir__(), *self._apis))

    def close(self) -> tuple[Exception, ...]:
        return self._registry.close()


def compose_logic_node_apis(services: ExperimentServices) -> LogicNodeApis:
    return LogicNodeApis(services)


__all__ = ["LogicNodeApis", "NodeApi", "compose_logic_node_apis"]
