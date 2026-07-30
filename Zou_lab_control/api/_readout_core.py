"""Node-neutral public Readout API spine.

Only cross-capability capture operations and readout binding live here.
Concrete installation authority is supplied by the application composition;
this module imports no concrete Logic-node API.
"""

from __future__ import annotations

from pathlib import Path

from zlc_data import OwnedSnapshot
from zlc_neutral_atom.artifact_dataset_source import ArtifactDatasetSource
from zlc_neutral_atom.artifact_dispatch import ArtifactCapability
from zlc_neutral_atom.capture.application import (
    CaptureRequest,
    PreparedFiniteCapture,
    prepare_finite_capture,
)
from zlc_neutral_atom.capture.artifact import CaptureArtifact, load_capture_artifact
from zlc_neutral_atom.capture.reference import (
    CAPTURE_ARTIFACT_REF_SCHEMA,
    CaptureArtifactRef,
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import PulseDocument, PulseExecutionForm, load_pulse_document
from zlc_storage.paths import resolve_under

from ._application_services import (
    ExperimentServices,
    application_start_run,
    resolve_role,
    service_guard,
)


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
        with service_guard(self._services) as services:
            if not isinstance(request, CaptureRequest):
                raise TypeError("request must be CaptureRequest")
            return prepare_finite_capture(
                request,
                pulse_port=services.runtime.pulse_port(request.sequencer_ref),
                camera_port=services.runtime.camera_port(request.camera_ref),
                captures_root=services.captures_root,
                start_run=self._start_run,
            )

    def _start_run(self, plan):
        return application_start_run(self._services, plan)

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
        with service_guard(self._services) as services:
            if not isinstance(reference, CaptureArtifactRef):
                raise TypeError("reference must be CaptureArtifactRef")
            return load_capture_artifact(services.captures_root, reference)

    def materialize_capture(self, reference: CaptureArtifactRef) -> OwnedSnapshot:
        with service_guard(self._services) as services:
            if not isinstance(reference, CaptureArtifactRef):
                raise TypeError("reference must be CaptureArtifactRef")
            return load_capture_artifact(
                services.captures_root,
                reference,
                materialize=True,
            ).materialize_snapshot()

    def _project_capture_dataset(
        self,
        reference: CaptureArtifactRef,
        *,
        materialize: bool,
        abort_check=None,
    ):
        with service_guard(self._services) as services:
            artifact = load_capture_artifact(
                services.captures_root,
                reference,
                materialize=materialize,
            )
            if abort_check is not None:
                abort_check()
            source = artifact.frame_source
            snapshot = (
                artifact.materialize_snapshot(abort_check=abort_check)
                if materialize
                else None
            )
            return ArtifactDatasetSource(
                source.schema,
                source.ref(artifact.provenance.generation),
                snapshot,
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
        with service_guard(self._services) as services:
            pulses_root = services.workspace_paths.pulses_root
        if isinstance(value, PulseDocument):
            return value
        return load_pulse_document(resolve_under(pulses_root, value))

    def resolve_readout_camera_ref(self, requested):
        role = self._resolve_camera_role(requested)
        with service_guard(self._services) as services:
            return services.catalog.require(role).ref

    def resolve_readout_sequencer_ref(self, requested):
        role = self._resolve_role(requested, "sequencer", ("sequencer",))
        with service_guard(self._services) as services:
            return services.catalog.require(role).ref

__all__ = ["ReadoutFacade"]
