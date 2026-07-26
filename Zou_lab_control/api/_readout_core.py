"""Node-neutral public Readout API spine.

Only cross-capability capture operations and readout binding live here.
Concrete installation authority is supplied by the application composition;
this module imports no concrete Logic-node API.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.capture.application import (
    CaptureRequest,
    PreparedFiniteCapture,
    prepare_finite_capture,
)
from zlc_neutral_atom.capture.artifact import CaptureArtifact
from zlc_neutral_atom.capture.reference import (
    CAPTURE_ARTIFACT_REF_SCHEMA,
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument, PulseExecutionForm

from ._application_services import (
    ExperimentServices,
    load_project_pulse,
    resolve_role,
    service_guard,
)


class LogicNodeApplicationOperations:
    """Cohesive composition-only access to installed application operations.

    Logic-node package factories use this object once to freeze their exact
    callables and installation facts.  Node API instances retain only those
    selected dependencies, never this object as a queryable service graph.
    """

    __slots__ = ("_services",)

    def __init__(self, services: ExperimentServices) -> None:
        if not isinstance(services, ExperimentServices):
            raise TypeError("services must be ExperimentServices")
        self._services = services

    @property
    def repository_root(self) -> Path:
        with service_guard(self._services) as services:
            return services.repository_root

    @property
    def capture_repository(self):
        """Return the core Capture owner only to a package wiring closure."""

        with service_guard(self._services) as services:
            return services.capture_repository

    @property
    def camera_signal_association_authorities(self):
        with service_guard(self._services) as services:
            return MappingProxyType(
                dict(services.installation.camera_signal_association_authorities)
            )

    @property
    def readout_apparatus_facts(self) -> tuple:
        with service_guard(self._services) as services:
            return tuple(services.installation.readout_apparatus_facts)

    def roles(self, domain: str) -> tuple[str, ...]:
        with service_guard(self._services) as services:
            return services.catalog.roles(domain)

    def device_ref(self, role: str) -> DeviceRef:
        with service_guard(self._services) as services:
            return services.catalog.require(role).ref

    def device_domain(self, role: str) -> str | None:
        with service_guard(self._services) as services:
            info = services.catalog.find(role)
            return None if info is None else info.domain

    def resolve_role(
        self,
        requested: str | None,
        domain: str,
        preferred: tuple[str, ...],
    ) -> str:
        with service_guard(self._services) as services:
            return resolve_role(services.catalog, requested, domain, preferred)

    def pulse_port(self, reference: DeviceRef):
        with service_guard(self._services) as services:
            return services.runtime.pulse_port(reference)

    def camera_port(self, reference: DeviceRef):
        with service_guard(self._services) as services:
            return services.runtime.camera_port(reference)

    def camera_monitor_port(self, reference: DeviceRef):
        with service_guard(self._services) as services:
            return services.runtime.camera_monitor_port(reference)

    def rf_port(self, reference: DeviceRef):
        with service_guard(self._services) as services:
            return services.runtime.rf_port(reference)

    def start_run(self, plan):
        with service_guard(self._services) as services:
            return services.runtime.start(plan)

    def wait_run(self, handle: RunHandle):
        with service_guard(self._services) as services:
            runtime = services.runtime
        return runtime.wait(handle)


class ReadoutFacade:
    """Node-neutral capture API and installed primitives borrowed by node leaves."""

    __slots__ = ("_binding", "_services")

    def __init__(
        self,
        services: ExperimentServices,
        binding: ReadoutBindingKey | None = None,
    ) -> None:
        if not isinstance(services, ExperimentServices):
            raise TypeError("services must be ExperimentServices")
        if binding is not None and not isinstance(binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey or None")
        self._services = services
        self._binding = binding

    @property
    def readout_binding(self) -> ReadoutBindingKey | None:
        return self._binding

    def for_binding(
        self,
        binding: ReadoutBindingKey | str,
    ) -> "ReadoutFacade":
        key = (
            binding
            if isinstance(binding, ReadoutBindingKey)
            else ReadoutBindingKey(binding)
        )
        if self._binding is not None and key != self._binding:
            raise ValueError("a bound readout facade cannot switch bindings")
        with service_guard(self._services) as services:
            info = services.catalog.require(key.value)
            if info.domain != "camera":
                raise ValueError(f"readout binding {key.value!r} is not a camera")
        return ReadoutFacade(self._services, key)

    def _require_binding(self, actual: ReadoutBindingKey) -> None:
        if not isinstance(actual, ReadoutBindingKey):
            raise TypeError("actual readout binding must be ReadoutBindingKey")
        if self._binding is not None and actual != self._binding:
            raise ValueError(
                f"bound readout facade requires {self._binding.value!r}, "
                f"not {actual.value!r}"
            )

    def prepare_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        with service_guard(self._services) as services:
            return prepare_finite_capture(
                request,
                pulse_port=services.runtime.pulse_port(request.sequencer_ref),
                camera_port=services.runtime.camera_port(request.camera_ref),
                repository=services.capture_repository,
                start_run=services.runtime.start,
            )

    def capture_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        execution_form: PulseExecutionForm = PulseExecutionForm.STATIC_ONCE,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        repeat_count: int = 1,
        readout_events_per_repeat: int | None = None,
        within_point_grouping: tuple[tuple[int, int], ...] | None = None,
    ) -> CaptureRequest:
        return CaptureRequest(
            self.load_readout_pulse(pulse),
            execution_form,
            self.resolve_readout_camera_ref(camera_role),
            self.resolve_readout_sequencer_ref(sequencer_role),
            trigger_channel,
            repeat_count,
            readout_events_per_repeat,
            within_point_grouping,
        )

    def capture(self, pulse: PulseDocument | str | Path, **kwargs) -> CaptureArtifactRef:
        prepared = self.prepare_capture(self.capture_request(pulse, **kwargs))
        handle = prepared.start()
        with service_guard(self._services) as services:
            runtime = services.runtime
        return runtime.wait(handle)

    def start_capture(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return self.prepare_capture(self.capture_request(pulse, **kwargs)).start()

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        with service_guard(self._services) as services:
            return services.capture_repository.load(reference)

    def materialize_capture(self, reference: CaptureArtifactRef) -> OwnedSnapshot:
        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        with service_guard(self._services) as services:
            return services.capture_repository.materialize_final(reference)

    def _logic_node_operations(self) -> LogicNodeApplicationOperations:
        """Create the composition-only operation set consumed by leaf packages."""

        return LogicNodeApplicationOperations(self._services)

    def _project_capture_dataset(
        self,
        reference: CaptureArtifactRef,
        *,
        materialize: bool,
        abort_check=None,
    ):
        with service_guard(self._services) as services:
            return services.capture_repository.project_dataset_source(
                reference,
                materialize=materialize,
                abort_check=abort_check,
            )

    def _artifact_capabilities(self) -> tuple[ArtifactCapability, ...]:
        """Freeze the core Capture owner beside discovered Logic-node owners."""

        return (
            ArtifactCapability(
                format_id=CAPTURE_ARTIFACT_REF_SCHEMA,
                source_label="capture",
                reference_type=CaptureArtifactRef,
                project_dataset=self._project_capture_dataset,
                reference_to_tree=capture_artifact_ref_to_tree,
                reference_from_tree=capture_artifact_ref_from_tree,
            ),
        )

    def require_readout_binding(self, actual: ReadoutBindingKey) -> None:
        self._require_binding(actual)

    def _resolve_role(self, requested, domain, preferred):
        with service_guard(self._services) as services:
            return resolve_role(services.catalog, requested, domain, preferred)

    def _resolve_camera_role(self, requested):
        if self._binding is not None:
            if requested is not None and requested != self._binding.value:
                raise ValueError("bound readout API cannot target another camera")
            requested = self._binding.value
        return self._resolve_role(
            requested,
            "camera",
            ("camera", "readout"),
        )

    def load_readout_pulse(self, value):
        return load_project_pulse(value)

    def resolve_readout_camera_ref(self, requested):
        role = self._resolve_camera_role(requested)
        with service_guard(self._services) as services:
            return services.catalog.require(role).ref

    def resolve_readout_sequencer_ref(self, requested):
        role = self._resolve_role(requested, "sequencer", ("sequencer",))
        with service_guard(self._services) as services:
            return services.catalog.require(role).ref

__all__ = ["ReadoutFacade"]
