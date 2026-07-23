"""Notebook-first composition facade with no public raw hardware graph."""

from __future__ import annotations

import json
import threading
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping, TYPE_CHECKING
from uuid import uuid4

import numpy as np

from zlc_neutral_atom.installation import (
    DeviceRef,
    DeviceCatalogView,
)
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    load_installation_config,
)
from zlc_data import (
    READOUT_EVENT,
    REPEAT,
    MONITOR_HISTORY,
    SCAN_POINT,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisSpec,
    BlockId,
    CommittedTransform,
    ComponentValidity,
    DataTransformSpec,
    FitNumericPolicy,
    FitParameterConstraint,
    FitCancelled,
    FitResultBatch,
    FitSpec,
    OwnedSnapshot,
    Selection,
    bind_fit,
    commit_transform,
    expand_value_validity,
    expand_dataset_validity,
    fit_model_catalog,
    fit_spec_for,
    suggest_fit_draft,
    validate_fit_result_source_binding,
)
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    CaptureArtifact,
    CaptureArtifactRef,
    CaptureRepository,
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
)
from zlc_neutral_atom.capture_application import (
    CAPTURE_READOUT_EVENT_AXIS_ID,
    CaptureRequest,
    PlanDescriptor,
    PreparedFiniteCapture,
    PreparedFiniteCameraMeasurement,
    bind_finite_capture_request,
    bind_finite_capture_spec,
    prepare_finite_capture,
    prepare_finite_camera_measurement,
)
from zlc_neutral_atom.camera_measurement import CameraMeasurementRequest
from zlc_neutral_atom.capture_reference import (
    capture_artifact_ref_from_tree,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.monitor_application import (
    PreparedLiveCameraMeasurement,
    prepare_live_camera_measurement,
)
from zlc_neutral_atom.mot_field import (
    MotFieldRequest,
    MotFieldResult,
    build_mot_scan_program,
)
from zlc_neutral_atom.pulse_application import (
    AppliedPulseSnapshot,
    PreparedPulseExecution,
    PulseApplicationOwner,
    PulseRunDescriptor,
    PulseRunObservation,
    PulseRunRequest,
    PulseRunResult,
    PulseTargetDescriptor,
    prepare_pulse_execution,
)
from zlc_neutral_atom.pulse_programs import DEFAULT_MOT_FIELD_PULSE_PATH
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationAnalysisRequest,
    GridOrder,
    ReadoutModelKind,
    ResolvedCalibration,
    ThresholdMethod,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.coupled_measurements import (
    GreyMolassesDetuningRequest,
    ReadoutDurationFidelityRequest,
    TemperatureReleaseRecaptureRequest,
    bind_temperature_release_recapture,
    reject_grey_molasses_detuning,
    reject_readout_duration_fidelity,
)
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_neutral_atom.readout.contracts import (
    CalibrationCaptureLayout,
    ReadoutBindingKey,
)
from zlc_neutral_atom.readout.sitemap import (
    SitemapAcquisitionProfile,
    SitemapCalibrationRequest,
)
from zlc_neutral_atom.scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
    MaterializedScanData,
    PulseScanProgram,
    ScanPointTable,
    bind_scan_output_contract,
)
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_neutral_atom.scan.application import (
    PreparedOccupancyScan,
    compile_api_direct_scan_artifact_plan,
    compile_api_occupancy_scan_artifact_plan,
    compile_direct_scan_artifact_plan,
    compile_occupancy_scan_artifact_plan,
)
from zlc_neutral_atom.readout.occupancy import (
    OccupancyStreamProcessorSpec,
    resolve_occupancy_stream_schema,
)
from zlc_neutral_atom.readout.occupancy_pipeline import OccupancyPipelineSpec
from zlc_neutral_atom.release_recapture_application import (
    PreparedTemperatureReleaseRecapture,
    prepare_temperature_release_recapture,
)
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.runtime.pipeline import MinimalPipelineSpec
from zlc_neutral_atom.timing.occupancy import TriggeredOccupancySpec
from zlc_neutral_atom.timing.pulse import PulseScanProgress
from zlc_neutral_atom.timing.segmented import (
    ApiSlotSegmentedSpec,
)
from zlc_neutral_atom.runtime.run import CancelOutcome, RunHandle
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    bind_pulse_document_target,
    expand_autonomous_scan_repeats,
    require_autonomous_scan_resident_capacity,
    load_pulse_document,
)
from zlc_storage import canonical_digest
from zlc_storage import canonical_text as _text
from zlc_storage import durable_mkdir
from zlc_storage import positive_integer as _positive_int
from zlc_storage import positive_real as _positive_real

if TYPE_CHECKING:
    from zlc_frontend import DataFigure
    from zlc_frontend.figure import FigureDocument, ViewIntent, ViewPreferences
    from zlc_neutral_atom.readout.analysis import (
        CalibrationComputation,
        CalibrationReport,
    )
    from zlc_neutral_atom.readout.calibration_repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.readout.occupancy import ResolvedOccupancy
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository
    from zlc_neutral_atom.scan.repository import ScanArtifact, ScanRepository


_DEFAULT_CALIBRATION_TIMEOUT_SECONDS = 300.0
_DEFAULT_OCCUPANCY_TIMEOUT_SECONDS = 300.0
_DEFAULT_FIT_GUI_TIMEOUT_SECONDS = 30.0
_SCAN_REPEAT_AXIS_ID = AxisId("scan.repeat")
_SCAN_READOUT_EVENT_AXIS_ID = AxisId("scan.readout_event")


class _ResourceCleanupError(RuntimeError):
    """Report retaining every ordinary cleanup failure."""

    def __init__(
        self,
        message: str,
        failures: tuple[Exception, ...],
    ) -> None:
        if not failures:
            raise ValueError("cleanup error requires at least one failure")
        self.failures = failures
        details = "; ".join(
            f"{type(error).__name__}: {error}" for error in failures
        )
        super().__init__(f"{message}: {details}")


def _cleanup_failures(*actions) -> list[Exception]:
    failures: list[Exception] = []
    for action in actions:
        if action is None:
            continue
        try:
            action()
        except Exception as error:
            failures.append(error)
    return failures


def _require_runtime_shutdown(runtime, *, timeout: float) -> None:
    if not runtime.shutdown(timeout=timeout):
        diagnostics = tuple(getattr(runtime, "shutdown_diagnostics", ()))
        suffix = "" if not diagnostics else ": " + "; ".join(diagnostics)
        raise RuntimeError(
            "runtime did not terminate within the cleanup deadline" + suffix
        )


def _validate_scan_request_fields(
    program: PulseScanProgram,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    trigger_channel: str | None,
    output_transform_spec: DataTransformSpec | None,
) -> None:
    if not isinstance(
        program,
        (AutonomousScanSlotProgram, ApiSlotSegmentedProgram),
    ):
        raise TypeError("program must be a current pulse-scan program")
    if not isinstance(camera_ref, DeviceRef):
        raise TypeError("camera_ref must be DeviceRef")
    if not isinstance(sequencer_ref, DeviceRef):
        raise TypeError("sequencer_ref must be DeviceRef")
    if trigger_channel is not None:
        _text(trigger_channel, "trigger_channel")
    if output_transform_spec is not None:
        if not isinstance(output_transform_spec, DataTransformSpec):
            raise TypeError("output_transform_spec must be DataTransformSpec or None")
        if not output_transform_spec.operations:
            raise ValueError("empty output_transform_spec must be None")


def _resolve_scan_fixed_api(
    document: PulseDocument,
    values: Mapping[str, int | float] | None,
) -> AutonomousScanSlotProgram:
    """Freeze whole-run API constants before a SCAN_SLOT request exists.

    These values are constants for the complete autonomous table.  They are
    deliberately resolved out of the PulseDocument here; a future API-slot
    segmented sweep is a different request and cannot reuse this path.
    """

    supplied = {} if values is None else dict(values)
    expected = tuple(
        parameter.parameter_id for parameter in document.api_parameters
    )
    if set(supplied) != set(expected):
        missing = tuple(key for key in expected if key not in supplied)
        extra = tuple(key for key in supplied if key not in set(expected))
        raise ValueError(
            "SCAN_SLOT requires explicit whole-run values for every API parameter; "
            f"missing={missing}, extra={extra}"
        )
    return AutonomousScanSlotProgram(
        document,
        tuple((key, supplied[key]) for key in expected),
    )


@dataclass(frozen=True)
class ScanRequest:
    """Freeze one typed pulse-scan program and its direct-camera y intent."""

    program: PulseScanProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        _validate_scan_request_fields(
            self.program,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
        )


@dataclass(frozen=True)
class OccupancyScanRequest:
    """Freeze one camera→occupancy exact processor as multidimensional scan y."""

    program: PulseScanProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        _validate_scan_request_fields(
            self.program,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
        )

@dataclass(frozen=True)
class CalibrationArtifactRequest:
    """Freeze one committed capture, its binding, and calibration intent."""

    source_capture_ref: CaptureArtifactRef
    readout_binding: ReadoutBindingKey
    analysis: CalibrationAnalysisRequest
    timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_real(self.timeout_seconds, "timeout_seconds"),
        )


class SitemapCalibrationFailed(RuntimeError):
    """Calibration failed after its diagnostic raw capture became FINAL."""

    __slots__ = ("source_capture_ref",)

    def __init__(self, source_capture_ref: CaptureArtifactRef) -> None:
        if not isinstance(source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        self.source_capture_ref = source_capture_ref
        super().__init__(
            "sitemap calibration failed; the committed raw capture remains "
            f"available as {source_capture_ref!r}"
        )


class SitemapCalibrationInterrupted(KeyboardInterrupt):
    """Notebook interrupt after the sitemap raw capture became FINAL."""

    __slots__ = ("source_capture_ref",)

    def __init__(self, source_capture_ref: CaptureArtifactRef) -> None:
        if not isinstance(source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        self.source_capture_ref = source_capture_ref
        super().__init__(
            "sitemap calibration interrupted; the committed raw capture remains "
            f"available as {source_capture_ref!r}"
        )


@dataclass(frozen=True)
class DetectionRequest:
    """Freeze two committed inputs and one concrete occupancy model."""

    source_capture_ref: CaptureArtifactRef
    calibration_ref: CalibrationArtifactRef
    readout_binding: ReadoutBindingKey
    readout_event_axis_id: AxisId
    model_kind: ReadoutModelKind
    timeout_seconds: float = _DEFAULT_OCCUPANCY_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.source_capture_ref, CaptureArtifactRef):
            raise TypeError("source_capture_ref must be CaptureArtifactRef")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if not isinstance(self.readout_binding, ReadoutBindingKey):
            raise TypeError("readout_binding must be ReadoutBindingKey")
        if not isinstance(self.readout_event_axis_id, AxisId):
            raise TypeError("readout_event_axis_id must be AxisId")
        if not isinstance(self.model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be a concrete ReadoutModelKind")
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_real(self.timeout_seconds, "timeout_seconds"),
        )


@dataclass
class _ExperimentServices:
    runtime: object
    capture_repository: CaptureRepository
    scan_repository: "ScanRepository"
    calibration_repository_path: Path
    calibration_repository: "CalibrationRepository | None"
    occupancy_repository_path: Path
    occupancy_repository: "OccupancyRepository | None"
    fit_repository: FitResultRepository
    catalog: DeviceCatalogView
    installation_config: InstallationConfigDocument
    pulse_application: PulseApplicationOwner
    operation_lock: threading.RLock
    fit_operations_drained: threading.Event
    fit_operation_thread_counts: dict[int, int]
    gui_handles: dict[str, object]
    active_fit_operations: int = 0
    state: str = "OPEN"


_AUTHORITY_LOCK = threading.RLock()
_AUTHORITIES: dict[object, _ExperimentServices] = {}


def _register(token: object, services: _ExperimentServices) -> None:
    with _AUTHORITY_LOCK:
        if token in _AUTHORITIES:
            raise RuntimeError("Experiment authority token is already registered")
        _AUTHORITIES[token] = services


@contextmanager
def _service_guard(
    token: object,
) -> Iterator[_ExperimentServices]:
    with _AUTHORITY_LOCK:
        services = _AUTHORITIES.get(token)
    if services is None:
        raise RuntimeError("Experiment is closed")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        yield services


@contextmanager
def _fit_service_guard(
    token: object,
) -> Iterator[_ExperimentServices]:
    """Keep repositories alive for one long Fit without serializing figures."""

    with _AUTHORITY_LOCK:
        services = _AUTHORITIES.get(token)
    if services is None:
        raise RuntimeError("Experiment is closed")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        if services.active_fit_operations == 0:
            services.fit_operations_drained.clear()
        services.active_fit_operations += 1
        thread_id = threading.get_ident()
        services.fit_operation_thread_counts[thread_id] = (
            services.fit_operation_thread_counts.get(thread_id, 0) + 1
        )
    completed = False
    try:
        yield services
        completed = True
    finally:
        with services.operation_lock:
            if services.active_fit_operations <= 0:
                raise RuntimeError("Experiment Fit operation count underflow")
            services.active_fit_operations -= 1
            thread_count = services.fit_operation_thread_counts.get(thread_id, 0)
            if thread_count <= 0:
                raise RuntimeError("Experiment Fit thread count underflow")
            if thread_count == 1:
                services.fit_operation_thread_counts.pop(thread_id)
            else:
                services.fit_operation_thread_counts[thread_id] = thread_count - 1
            if services.active_fit_operations == 0:
                services.fit_operations_drained.set()
            remained_open = services.state == "OPEN"
    if completed and not remained_open:
        raise FitCancelled("Experiment began closing during Fit execution")


def _calibration_repository(
    services: _ExperimentServices,
) -> "CalibrationRepository":
    repository = services.calibration_repository
    if repository is None:
        from zlc_neutral_atom.readout.calibration_repository import (
            CalibrationRepository,
        )

        repository = CalibrationRepository(services.calibration_repository_path)
        services.calibration_repository = repository
    return repository


def _occupancy_repository(
    services: _ExperimentServices,
) -> "OccupancyRepository":
    repository = services.occupancy_repository
    if repository is None:
        from zlc_neutral_atom.readout.occupancy_repository import (
            OccupancyRepository,
        )

        repository = OccupancyRepository(services.occupancy_repository_path)
        services.occupancy_repository = repository
    return repository


def _occupancy_cell_navigation(
    reference,
    artifact,
):
    """Project one admitted occupancy artifact into frontend navigation."""

    from zlc_frontend.site_map_render import OccupancyCellNavigation

    schema = artifact.occupied.schema
    return OccupancyCellNavigation(
        artifact_identity=reference.target_ref,
        schema_fingerprint=schema.fingerprint,
        generation=artifact.generation,
        repeat_axis=schema.repeat_axis,
        point_axes=schema.point_axes,
        point_layout=schema.point_layout,
        cell_layout=schema.cell_layout,
    )


class PulseFacade:
    __slots__ = ("_token",)

    def __init__(self, token: object) -> None:
        self._token = token

    @property
    def target(self) -> PulseTargetDescriptor:
        with _service_guard(self._token) as services:
            role = _resolve_role(
                services.catalog,
                None,
                "sequencer",
                ("sequencer",),
            )
            reference = services.catalog.require(role).ref
            port = services.runtime.pulse_port(reference)
            capability = port.capability
            return PulseTargetDescriptor(
                reference,
                capability.manifest,
                capability.clock_hz,
                capability.geometry_fingerprint,
                capability.resident_scan_point_capacity,
            )

    def request(
        self,
        document: PulseDocument,
        execution_form: PulseExecutionForm = PulseExecutionForm.STATIC_ONCE,
        *,
        api_values: Mapping[str, int | float] | None = None,
        scan_sweep_count: int = 1,
        sequencer_role: str | None = None,
        timeout_seconds: float | None = None,
    ) -> PulseRunRequest:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        with _service_guard(self._token) as services:
            role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            reference = services.catalog.require(role).ref
        if api_values is not None and not isinstance(api_values, Mapping):
            raise TypeError("api_values must be a mapping or None")
        requested_api_values = {} if api_values is None else dict(api_values)
        expected_api_ids = tuple(
            parameter.parameter_id for parameter in document.api_parameters
        )
        expected_api_id_set = set(expected_api_ids)
        if set(requested_api_values) != expected_api_id_set:
            missing = tuple(
                parameter_id
                for parameter_id in expected_api_ids
                if parameter_id not in requested_api_values
            )
            unknown = tuple(
                parameter_id
                for parameter_id in requested_api_values
                if parameter_id not in expected_api_id_set
            )
            raise ValueError(
                "Pulse Run API values must exactly cover declared parameters; "
                f"missing={missing}, unknown={unknown}"
            )
        return PulseRunRequest(
            document=document,
            execution_form=execution_form,
            sequencer_ref=reference,
            timeout_seconds=timeout_seconds,
            api_values=tuple(
                (parameter_id, requested_api_values[parameter_id])
                for parameter_id in expected_api_ids
            ),
            scan_sweep_count=scan_sweep_count,
        )

    def snapshot(self) -> AppliedPulseSnapshot | None:
        with _service_guard(self._token) as services:
            return services.pulse_application.snapshot()

    def observe_active(self) -> PulseRunObservation | None:
        with _service_guard(self._token) as services:
            return services.pulse_application.observe_active()

    def observe_scan_progress(self) -> PulseScanProgress | None:
        with _service_guard(self._token) as services:
            return services.pulse_application.observe_scan_progress()

    def cancel_active(
        self,
        reason: str = "user requested pulse stop",
    ) -> CancelOutcome | None:
        with _service_guard(self._token) as services:
            return services.pulse_application.cancel_active(reason)

    def inspect(self, request: PulseRunRequest) -> PulseRunDescriptor:
        with _service_guard(self._token) as services:
            return _prepare_pulse_for_services(services, request).descriptor

    def start(self, request: PulseRunRequest) -> RunHandle:
        return _start_pulse(self._token, request)

    def run(self, request: PulseRunRequest) -> PulseRunResult:
        if request.requires_cancellation:
            raise ValueError("continuous pulse execution must be started and cancelled")
        return _run_pulse(self._token, request)


class ReadoutFacade:
    __slots__ = ("_token", "_binding")

    def __init__(
        self,
        token: object,
        binding: ReadoutBindingKey | None = None,
    ) -> None:
        self._token = token
        if binding is not None and not isinstance(binding, ReadoutBindingKey):
            raise TypeError("binding must be ReadoutBindingKey or None")
        self._binding = binding

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
        with _service_guard(self._token) as services:
            info = services.catalog.require(key.value)
            if info.domain != "camera":
                raise ValueError(f"readout binding {key.value!r} is not a camera")
        return ReadoutFacade(self._token, key)

    def _require_binding(self, actual: ReadoutBindingKey) -> None:
        if not isinstance(actual, ReadoutBindingKey):
            raise TypeError("actual readout binding must be ReadoutBindingKey")
        if self._binding is not None and actual != self._binding:
            raise ValueError(
                f"bound readout facade requires {self._binding.value!r}, "
                f"not {actual.value!r}"
            )

    def _resolve_camera_role(
        self,
        services: _ExperimentServices,
        requested: str | None,
    ) -> str:
        if self._binding is not None:
            if requested is not None and requested != self._binding.value:
                raise ValueError("bound readout facade cannot target another camera")
            requested = self._binding.value
        return _resolve_role(
            services.catalog,
            requested,
            "camera",
            ("readout", "camera"),
        )

    def camera_measurement_request(
        self,
        *,
        camera_role: str | None = None,
        repeat: int = 0,
        frames_per_cycle: int = 1,
    ) -> CameraMeasurementRequest:
        """Freeze Main's one Camera semantic: 0=live, K=finite."""

        if isinstance(repeat, bool) or not isinstance(repeat, int):
            raise TypeError("repeat must be an integer")
        if repeat < 0:
            raise ValueError("repeat must be non-negative")
        if (
            isinstance(frames_per_cycle, bool)
            or not isinstance(frames_per_cycle, int)
            or frames_per_cycle < 1
        ):
            raise ValueError("frames_per_cycle must be a positive integer")
        with _service_guard(self._token) as services:
            role = self._resolve_camera_role(services, camera_role)
            camera_ref = services.catalog.require(role).ref
            return CameraMeasurementRequest(
                camera_ref=camera_ref,
                repeat=repeat,
                frames_per_cycle=frames_per_cycle,
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
        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            camera_role = self._resolve_camera_role(services, camera_role)
            return CaptureRequest(
                document,
                execution_form,
                services.catalog.require(
                    camera_role
                ).ref,
                services.catalog.require(
                    _resolve_role(
                        services.catalog,
                        sequencer_role,
                        "sequencer",
                        ("sequencer",),
                    )
                ).ref,
                trigger_channel,
                repeat_count,
                readout_events_per_repeat,
                within_point_grouping,
            )

    def capture(self, pulse: PulseDocument | str | Path, **kwargs) -> CaptureArtifactRef:
        return _run(self._token, self.capture_request(pulse, **kwargs))

    def start_capture(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return _start(self._token, self.capture_request(pulse, **kwargs))

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        with _service_guard(self._token) as services:
            return services.capture_repository.load(reference)

    def temperature_release_recapture_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        trap_off_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        per_site: bool = False,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> TemperatureReleaseRecaptureRequest:
        """Freeze the autonomous two-readout temperature Measurement."""

        if not isinstance(calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            camera = self._resolve_camera_role(services, camera_role)
            sequencer = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return TemperatureReleaseRecaptureRequest(
                document,
                tuple(trap_off_seconds),
                shots,
                services.catalog.require(camera).ref,
                services.catalog.require(sequencer).ref,
                calibration_ref,
                model_kind,
                per_site,
                trigger_channel,
            )

    def start_temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ) -> RunHandle:
        with _service_guard(self._token) as services:
            return _prepare_temperature_release_recapture_for_services(
                services,
                request,
            ).start()

    def temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ):
        with _service_guard(self._token) as services:
            handle = _prepare_temperature_release_recapture_for_services(
                services,
                request,
            ).start()
            runtime = services.runtime
        return runtime.wait(handle)

    def readout_duration_fidelity_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        duration_seconds: tuple[float, ...],
        shots: int,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        site: int | None = None,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> ReadoutDurationFidelityRequest:
        """Freeze valid intent even when this installation lacks its Port."""

        if not isinstance(calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            camera = self._resolve_camera_role(services, camera_role)
            sequencer = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return ReadoutDurationFidelityRequest(
                document,
                tuple(duration_seconds),
                shots,
                services.catalog.require(camera).ref,
                services.catalog.require(sequencer).ref,
                calibration_ref,
                model_kind,
                site,
                trigger_channel,
            )

    def start_readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ) -> RunHandle:
        reject_readout_duration_fidelity(request)
        raise AssertionError("capability rejection did not raise")

    def grey_molasses_detuning_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        detuning_gamma: tuple[float, ...],
        trap_off_seconds: float,
        shots: int,
        rf_role: str,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        per_site: bool = False,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> GreyMolassesDetuningRequest:
        """Freeze grey-molasses intent without inventing a missing RF Port."""

        if not isinstance(calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            camera = self._resolve_camera_role(services, camera_role)
            sequencer = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return GreyMolassesDetuningRequest(
                document,
                tuple(detuning_gamma),
                trap_off_seconds,
                shots,
                services.catalog.require(camera).ref,
                services.catalog.require(sequencer).ref,
                rf_role,
                calibration_ref,
                model_kind,
                per_site,
                trigger_channel,
            )

    def start_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> RunHandle:
        reject_grey_molasses_detuning(request)
        raise AssertionError("capability rejection did not raise")

    def materialize_capture(self, reference: CaptureArtifactRef) -> OwnedSnapshot:
        """Load one FINAL capture as its canonical immutable dataset snapshot."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        with _service_guard(self._token) as services:
            return services.capture_repository.materialize_final(reference)

    def scan_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        api_values: Mapping[str, int | float] | None = None,
        output_transform_spec: DataTransformSpec | None = None,
    ) -> ScanRequest:
        """Build one direct-camera autonomous SCAN_SLOT request.

        The PulseDocument's frozen table is the only slot-value truth.  This
        first vertical slice deliberately has no API-slot or host-stepped
        fallback.
        """

        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            program = _resolve_scan_fixed_api(document, api_values)
            camera_role = self._resolve_camera_role(services, camera_role)
            return ScanRequest(
                program=program,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(
                    _resolve_role(
                        services.catalog,
                        sequencer_role,
                        "sequencer",
                        ("sequencer",),
                    )
                ).ref,
                trigger_channel=trigger_channel,
                output_transform_spec=output_transform_spec,
            )

    def scan(self, pulse: PulseDocument | str | Path, **kwargs) -> ScanArtifactRef:
        return _run_scan(self._token, self.scan_request(pulse, **kwargs))

    def start_scan(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return _start_scan(self._token, self.scan_request(pulse, **kwargs))

    def api_scan_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        api_table: ApiSegmentTable,
        segmentation_rationale: str,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        output_transform_spec: DataTransformSpec | None = None,
    ) -> ScanRequest:
        """Build the accepted finite API_SLOT segmented exception explicitly."""

        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            program = ApiSlotSegmentedProgram(
                document,
                api_table,
                segmentation_rationale,
            )
            camera_role = self._resolve_camera_role(services, camera_role)
            sequencer_role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return ScanRequest(
                program=program,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(sequencer_role).ref,
                trigger_channel=trigger_channel,
                output_transform_spec=output_transform_spec,
            )

    def api_scan(self, pulse: PulseDocument | str | Path, **kwargs) -> ScanArtifactRef:
        return _run_scan(self._token, self.api_scan_request(pulse, **kwargs))

    def start_api_scan(
        self,
        pulse: PulseDocument | str | Path,
        **kwargs,
    ) -> RunHandle:
        return _start_scan(self._token, self.api_scan_request(pulse, **kwargs))

    def occupancy_scan_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        api_values: Mapping[str, int | float] | None = None,
        output_transform_spec: DataTransformSpec | None = None,
    ) -> OccupancyScanRequest:
        """Build the first external Measurement→Processor SCAN_SLOT request."""

        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            program = _resolve_scan_fixed_api(document, api_values)
            camera_role = self._resolve_camera_role(services, camera_role)
            sequencer_role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return OccupancyScanRequest(
                program=program,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(sequencer_role).ref,
                calibration_ref=calibration_ref,
                model_kind=model_kind,
                trigger_channel=trigger_channel,
                output_transform_spec=output_transform_spec,
            )

    def occupancy_scan(
        self,
        pulse: PulseDocument | str | Path,
        **kwargs,
    ) -> ScanArtifactRef:
        return _run_scan(
            self._token,
            self.occupancy_scan_request(pulse, **kwargs),
        )

    def start_occupancy_scan(
        self,
        pulse: PulseDocument | str | Path,
        **kwargs,
    ) -> RunHandle:
        return _start_scan(
            self._token,
            self.occupancy_scan_request(pulse, **kwargs),
        )

    def api_occupancy_scan_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        api_table: ApiSegmentTable,
        segmentation_rationale: str,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        output_transform_spec: DataTransformSpec | None = None,
    ) -> OccupancyScanRequest:
        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            program = ApiSlotSegmentedProgram(
                document,
                api_table,
                segmentation_rationale,
            )
            camera_role = self._resolve_camera_role(services, camera_role)
            sequencer_role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return OccupancyScanRequest(
                program=program,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(sequencer_role).ref,
                calibration_ref=calibration_ref,
                model_kind=model_kind,
                trigger_channel=trigger_channel,
                output_transform_spec=output_transform_spec,
            )

    def api_occupancy_scan(
        self,
        pulse: PulseDocument | str | Path,
        **kwargs,
    ) -> ScanArtifactRef:
        return _run_scan(
            self._token,
            self.api_occupancy_scan_request(pulse, **kwargs),
        )

    def start_api_occupancy_scan(
        self,
        pulse: PulseDocument | str | Path,
        **kwargs,
    ) -> RunHandle:
        return _start_scan(
            self._token,
            self.api_occupancy_scan_request(pulse, **kwargs),
        )

    def load_scan(self, reference: ScanArtifactRef) -> "ScanArtifact":
        with _service_guard(self._token) as services:
            return services.scan_repository.admit(reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData:
        """Load the canonical scan dataset."""

        with _service_guard(self._token) as services:
            return services.scan_repository.materialize(reference)

    def mot_field_request(
        self,
        pulse: PulseDocument | str | Path = DEFAULT_MOT_FIELD_PULSE_PATH,
        *,
        center_x: float = 0.0,
        center_y: float = 0.0,
        center_z: float = 0.0,
        span: float = 12.0,
        points: int = 7,
        roi_cx: float | None = None,
        roi_cy: float | None = None,
        roi_radius: float = 8.0,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> MotFieldRequest:
        """Freeze one three-axis autonomous MOT scan and its ROI analysis."""

        document = (
            pulse
            if isinstance(pulse, PulseDocument)
            else load_pulse_document(pulse)
        )
        program = build_mot_scan_program(
            document,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            span=span,
            points=points,
        )
        with _service_guard(self._token) as services:
            camera_role = "mot_camera" if camera_role is None else camera_role
            camera_role = _resolve_role(
                services.catalog,
                camera_role,
                "camera",
                ("mot_camera",),
            )
            if camera_role != "mot_camera":
                raise ValueError(
                    "MOT field optimization requires the installation's "
                    "'mot_camera' role; an arbitrary camera is not a "
                    "coil-sensitive exact-scan sensor"
                )
            sequencer_role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return MotFieldRequest(
                program=program,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(sequencer_role).ref,
                roi_cx=roi_cx,
                roi_cy=roi_cy,
                roi_radius=roi_radius,
                trigger_channel=trigger_channel,
            )

    def _start_mot_field_scan(self, request: MotFieldRequest) -> RunHandle:
        """Workbench-only first stage: start the request's exact scan Run."""

        return _start_scan(self._token, _mot_field_scan_request(request))

    def _sitemap_profile(
        self,
        camera_role: str | None,
    ) -> tuple[str, SitemapAcquisitionProfile]:
        """Resolve the one installation-owned sitemap profile for a camera."""

        with _service_guard(self._token) as services:
            selected_camera = self._resolve_camera_role(services, camera_role)
            camera_ref = services.catalog.require(selected_camera).ref
            profile = services.runtime.sitemap_profile(camera_ref)
        if not isinstance(profile, SitemapAcquisitionProfile):
            raise TypeError("installation returned an invalid sitemap profile")
        if profile.readout_binding != ReadoutBindingKey(selected_camera):
            raise ValueError(
                "installation sitemap profile differs from the selected camera"
            )
        return selected_camera, profile

    def sitemap_camera_roles(self) -> tuple[str, ...]:
        """Installed camera roles with a complete live calibration profile."""

        with _service_guard(self._token) as services:
            roles = tuple(services.runtime.sitemap_camera_roles())
            cameras = set(services.catalog.roles("camera"))
        if any(role not in cameras for role in roles):
            raise RuntimeError(
                "installation sitemap capabilities differ from its camera catalog"
            )
        return roles

    @staticmethod
    def _sitemap_analysis(
        profile: SitemapAcquisitionProfile,
        *,
        threshold_method: ThresholdMethod | str,
        roi_radius: int | None,
    ) -> CalibrationAnalysisRequest:
        """Apply operator analysis intent to the installation's one profile."""

        if isinstance(threshold_method, str):
            try:
                threshold_method = ThresholdMethod(threshold_method.strip().lower())
            except ValueError as error:
                raise ValueError(
                    "threshold_method must be 'otsu' or 'bimodal'"
                ) from error
        if not isinstance(threshold_method, ThresholdMethod):
            raise TypeError("threshold_method must be ThresholdMethod or str")
        if roi_radius is not None:
            if isinstance(roi_radius, bool) or not isinstance(roi_radius, int):
                raise TypeError("roi_radius must be an integer or None")
            if roi_radius < 1:
                raise ValueError("roi_radius must be positive")
        return replace(
            profile.analysis_request(CAPTURE_READOUT_EVENT_AXIS_ID),
            threshold_method=threshold_method,
            **({} if roi_radius is None else {"box_radius": roi_radius}),
        )

    def sitemap_analysis_request(
        self,
        *,
        camera_role: str | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
    ) -> CalibrationAnalysisRequest:
        """Freeze sitemap analysis intent without manufacturing a live capture."""

        _selected_camera, profile = self._sitemap_profile(camera_role)
        return self._sitemap_analysis(
            profile,
            threshold_method=threshold_method,
            roi_radius=roi_radius,
        )

    def sitemap_request(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
        pulse: PulseDocument | str | Path | None = None,
        reference_exposure_s: float | None = None,
        readout_exposure_s: float | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
        calibration_timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ) -> SitemapCalibrationRequest:
        """Freeze one installation-qualified capture-then-calibration request.

        ``frames`` is the number of complete reference/readout/reference groups,
        not the total camera-frame count.  The hardware repeats each complete
        group.  The returned value contains a complete finite ``CaptureRequest``
        plus the immutable analysis intent for the second ordinary Run.  It
        contains no current/latest calibration pointer.
        """

        repeat_groups = _positive_int(frames, "frames")
        calibration_timeout = _positive_real(
            calibration_timeout_seconds,
            "calibration_timeout_seconds",
        )
        selected_camera, profile = self._sitemap_profile(camera_role)

        selected_pulse = (
            None
            if pulse is None
            else pulse
            if isinstance(pulse, PulseDocument)
            else load_pulse_document(Path(pulse).expanduser())
        )
        if (reference_exposure_s is None) != (readout_exposure_s is None):
            raise ValueError(
                "reference_exposure_s and readout_exposure_s must be set together"
            )
        if reference_exposure_s is None:
            selected_profile = (
                profile
                if selected_pulse is None
                else replace(profile, pulse_document=selected_pulse)
            )
            document = selected_profile.document_for_repeats(repeat_groups)
        else:
            document = profile.configured_document_for_repeats(
                repeat_groups,
                reference_exposure_s=reference_exposure_s,
                readout_exposure_s=readout_exposure_s,
                pulse_document=selected_pulse,
            )
        grouping = profile.repeat_major_grouping(repeat_groups)
        analysis = self._sitemap_analysis(
            profile,
            threshold_method=threshold_method,
            roi_radius=roi_radius,
        )
        capture_request = self.capture_request(
            document,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            camera_role=selected_camera,
            sequencer_role=profile.sequencer_role,
            trigger_channel=profile.trigger_channel,
            repeat_count=repeat_groups,
            readout_events_per_repeat=profile.event_count,
            within_point_grouping=grouping,
        )
        return SitemapCalibrationRequest(
            capture_request,
            analysis,
            calibration_timeout,
        )

    def sitemap(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
        calibration_timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ) -> CalibrationArtifactRef:
        """Capture and commit one installation-qualified site-map calibration.

        This convenience composes two ordinary Runs in order; it is not a
        child-plan engine or hidden current-calibration slot.  Advanced/custom
        calibration remains available through ``calibration_request``.
        """

        sequence = self.sitemap_request(
            frames=frames,
            camera_role=camera_role,
            calibration_timeout_seconds=calibration_timeout_seconds,
        )
        source = _run(self._token, sequence.capture_request)
        try:
            request = self.calibration_request(
                source,
                sequence.analysis,
                timeout_seconds=sequence.calibration_timeout_seconds,
            )
            return self.calibrate(request)
        except KeyboardInterrupt as error:
            raise SitemapCalibrationInterrupted(source) from error
        except Exception as error:
            raise SitemapCalibrationFailed(source) from error

    def _resolve_saved_calibration_capture(
        self,
        source_path: str | Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef:
        """Admit a task-exported raw capture bundle without inventing lineage.

        The raw arrays are an operator copy; the canonical capture in
        the CaptureRepository remains the authority for camera/pulse lineage.
        The bundle therefore resolves only while that repository artifact is
        available; copying anonymous arrays into another workspace cannot
        manufacture the missing physical evidence.
        A directory of anonymous ``.npy`` images is intentionally insufficient
        for a formal calibration because it cannot prove exposure, event order,
        camera binding, or the fired pulse.
        """

        folder = Path(source_path).expanduser().resolve()
        metadata_path = folder / "capture.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"saved calibration source has no {metadata_path.name}; "
                "plain frames have no camera/pulse lineage"
            )
        tree = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(tree, dict) or set(tree) != {
            "schema",
            "capture_ref",
            "dataset_schema_fingerprint",
            "values_file",
            "validity_file",
        }:
            raise ValueError("saved calibration capture metadata is not current")
        if tree["schema"] != "zlc.calibration-task.capture-export.v1":
            raise ValueError("saved calibration capture metadata has unknown schema")
        reference = capture_artifact_ref_from_tree(tree["capture_ref"])
        values_path = folder / str(tree["values_file"])
        validity_path = folder / str(tree["validity_file"])
        if not values_path.is_file() or not validity_path.is_file():
            raise FileNotFoundError("saved calibration raw arrays are incomplete")
        with _service_guard(self._token) as services:
            admitted = services.capture_repository.admit(reference)
            if admitted.artifact.camera_provenance.binding.value != expected_camera_role:
                raise ValueError(
                    "saved calibration capture belongs to camera role "
                    f"{admitted.artifact.camera_provenance.binding.value!r}, not "
                    f"{expected_camera_role!r}"
                )
            snapshot = admitted.materialize_snapshot()
        block = snapshot.block
        if tree["dataset_schema_fingerprint"] != block.schema.fingerprint:
            raise ValueError("saved calibration schema differs from its capture authority")
        values = np.load(values_path, allow_pickle=False)
        validity = np.load(validity_path, allow_pickle=False)
        expected_validity = expand_dataset_validity(block.validity, block.schema)
        if (
            values.dtype != block.values.dtype
            or values.shape != block.values.shape
            or not np.array_equal(values, block.values, equal_nan=True)
            or validity.dtype != np.dtype(bool)
            or validity.shape != expected_validity.shape
            or not np.array_equal(validity, expected_validity)
        ):
            raise ValueError(
                "saved calibration raw arrays differ from the lineage-bearing capture"
            )
        return reference

    def _write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str | Path,
        save_frames: bool,
    ) -> None:
        """Write task-owned pointers and optional raw export.

        CaptureRepository and CalibrationRepository continue to own canonical
        artifacts and the calibration report.  This task folder contains only
        explicit references to those artifacts plus, when requested, a checked
        raw-array export suitable for a later saved-source task.
        """

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if not isinstance(calibration, CalibrationArtifactRef):
            raise TypeError("calibration must be CalibrationArtifactRef")
        if type(save_frames) is not bool:
            raise TypeError("save_frames must be bool")
        root = Path(folder).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        (root / "calibration_ref.json").write_text(
            json.dumps(
                {
                    "schema": "zlc.calibration-task.result.v1",
                    "calibration_ref": calibration_artifact_ref_to_tree(calibration),
                    "source_capture_ref": capture_artifact_ref_to_tree(source),
                    "artifact_owner": "CalibrationRepository",
                    "report_owner": "CalibrationRepository",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if not save_frames:
            return
        frames = root / "frames"
        frames.mkdir(parents=True, exist_ok=True)
        with _service_guard(self._token) as services:
            admitted = services.capture_repository.admit(source)
            snapshot = admitted.materialize_snapshot()
        block = snapshot.block
        validity = np.asarray(
            expand_dataset_validity(block.validity, block.schema),
            dtype=bool,
        )
        np.save(frames / "values.npy", block.values, allow_pickle=False)
        np.save(frames / "validity.npy", validity, allow_pickle=False)
        (frames / "capture.json").write_text(
            json.dumps(
                {
                    "schema": "zlc.calibration-task.capture-export.v1",
                    "capture_ref": capture_artifact_ref_to_tree(source),
                    "dataset_schema_fingerprint": block.schema.fingerprint,
                    "values_file": "values.npy",
                    "validity_file": "validity.npy",
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
        *,
        timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ) -> CalibrationArtifactRequest:
        """Freeze explicit calibration intent from one FINAL capture inspection."""

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if not isinstance(analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        with _service_guard(self._token) as services:
            source_artifact = services.capture_repository.admit(source).artifact
            binding = source_artifact.camera_provenance.binding
            self._require_binding(binding)
            return CalibrationArtifactRequest(
                source,
                binding,
                analysis,
                timeout,
            )

    def start_calibration(
        self,
        request: CalibrationArtifactRequest,
    ) -> RunHandle:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return _start_calibration(self._token, request)

    def calibrate(
        self,
        request: CalibrationArtifactRequest,
    ) -> CalibrationArtifactRef:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return _run_calibration(self._token, request)

    def _start_calibration_from_editor(
        self,
        source_capture_ref: CaptureArtifactRef,
        readout_binding: ReadoutBindingKey,
        analysis: CalibrationAnalysisRequest,
        timeout_seconds: float,
    ) -> RunHandle:
        """Narrow Workbench command; request construction stays in this owner."""

        return self.start_calibration(
            CalibrationArtifactRequest(
                source_capture_ref,
                readout_binding,
                analysis,
                timeout_seconds,
            )
        )

    def calibration_gui(self, request: CalibrationArtifactRequest):
        """Edit one explicit request and commit each successful formal Run."""

        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        from zlc_workbench.calibration import CalibrationEditorSeed
        from Zou_lab_control.workbench import open_calibration_workbench

        seed = CalibrationEditorSeed(
            request.source_capture_ref,
            request.readout_binding,
            request.analysis,
            request.timeout_seconds,
        )
        return open_calibration_workbench(
            self._load_calibration_report_source,
            self._start_calibration_from_editor,
            seed=seed,
        )

    def calibration_edit_gui(
        self,
        reference: CalibrationArtifactRef,
        *,
        timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ):
        """Reopen an exact calibration and create a new immutable revision."""

        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        from Zou_lab_control.workbench import open_calibration_workbench

        return open_calibration_workbench(
            self._load_calibration_report_source,
            self._start_calibration_from_editor,
            reference=reference,
            timeout_seconds=timeout,
        )

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration:
        with _service_guard(self._token) as services:
            resolved = _calibration_repository(services).admit(
                reference,
                services.capture_repository,
            )
            self._require_binding(resolved.artifact.frame_contract.binding)
            return resolved

    def load_calibration_computation(
        self,
        reference: CalibrationArtifactRef,
    ) -> "CalibrationComputation":
        return self._load_calibration_report_source(reference)

    def _load_calibration_report_source(
        self,
        reference: CalibrationArtifactRef,
    ) -> "CalibrationComputation":
        """Load committed calibration diagnostics."""

        with _service_guard(self._token) as services:
            repository = _calibration_repository(services)
            computation = repository.load_computation(reference)
            self._require_binding(computation.artifact.frame_contract.binding)
            return computation

    def load_calibration_report(
        self,
        reference: CalibrationArtifactRef,
    ) -> "CalibrationReport":
        return self.load_calibration_computation(reference).report

    def calibration_report_gui(
        self,
        reference: CalibrationArtifactRef,
    ):
        """Open one committed calibration report without blocking the Qt owner."""

        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        from Zou_lab_control.workbench import open_calibration_report_workbench

        return open_calibration_report_workbench(
            self._load_calibration_report_source,
            reference,
        )

    def detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        model_kind: ReadoutModelKind | None = None,
        timeout_seconds: float = _DEFAULT_OCCUPANCY_TIMEOUT_SECONDS,
    ) -> DetectionRequest:
        """Freeze one committed single-event capture for occupancy analysis."""

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if not isinstance(calibration, CalibrationArtifactRef):
            raise TypeError("calibration must be CalibrationArtifactRef")
        if model_kind is not None and not isinstance(model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        with _service_guard(self._token) as services:
            source_artifact = services.capture_repository.admit(source).artifact
            calibration_artifact = _calibration_repository(services).load(calibration)
            binding = source_artifact.camera_provenance.binding
            if calibration_artifact.frame_contract.binding != binding:
                raise ValueError(
                    "capture and calibration name different readout bindings"
                )
            self._require_binding(binding)
            event_axes = tuple(
                axis
                for axis in source_artifact.frame_source.schema.point_axes
                if axis.role == READOUT_EVENT
            )
            if len(event_axes) != 1 or event_axes[0].size != 1:
                raise ValueError(
                    "detection requires exactly one singleton READOUT_EVENT axis"
                )
            selected_model = (
                calibration_artifact.default_model_kind
                if model_kind is None
                else model_kind
            )
            if selected_model not in tuple(
                model.kind for model in calibration_artifact.models
            ):
                raise KeyError(selected_model)
            return DetectionRequest(
                source,
                calibration,
                binding,
                event_axes[0].axis_id,
                selected_model,
                timeout,
            )

    def start_detection(self, request: DetectionRequest) -> RunHandle:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return _start_detection(self._token, request)

    def detect(self, request: DetectionRequest) -> OccupancyArtifactRef:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return _run_detection(self._token, request)

    def load_occupancy(
        self,
        reference: OccupancyArtifactRef,
    ) -> "ResolvedOccupancy":
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        with _service_guard(self._token) as services:
            resolved = _occupancy_repository(services).admit(
                reference,
                services.capture_repository,
                _calibration_repository(services),
            )
            self._require_binding(resolved.readout_binding)
            return resolved

    def _inspect_occupancy_cell_navigation(
        self,
        reference: OccupancyArtifactRef,
    ):
        """Resolve the committed outer axes needed by the navigator."""

        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        with _service_guard(self._token) as services:
            occupancy_repository = _occupancy_repository(services)
            calibration_repository = _calibration_repository(services)
            resolved = occupancy_repository.admit(
                reference,
                services.capture_repository,
                calibration_repository,
            )
            artifact = resolved.artifact
            source_artifact = services.capture_repository.admit(
                artifact.source_capture_ref
            ).artifact
            calibration_artifact = calibration_repository.load(
                artifact.calibration_reference
            )
            self._require_binding(resolved.readout_binding)
            if calibration_artifact.frame_contract.binding != resolved.readout_binding:
                raise ValueError("occupancy source and calibration bindings differ")
            source_schema = source_artifact.frame_source.schema
            occupied_schema = artifact.occupied.schema
            if (
                source_schema.repeat_axis != occupied_schema.repeat_axis
                or source_schema.point_axes != occupied_schema.point_axes
                or source_schema.point_layout != occupied_schema.point_layout
            ):
                raise ValueError("occupancy outer axes differ from the source capture")
            frame_schema = source_schema.cell_schema
            if len(frame_schema.data_shape) != 2:
                raise ValueError(
                    "physical occupancy map requires a two-dimensional camera frame"
                )
            frame_axes = frame_schema.data_axes
            x_axes = tuple(axis for axis in frame_axes if axis.role == SPATIAL_X)
            y_axes = tuple(axis for axis in frame_axes if axis.role == SPATIAL_Y)
            if len(x_axes) != 1 or len(y_axes) != 1:
                raise ValueError(
                    "physical occupancy map requires exactly one declared "
                    "SPATIAL_X and SPATIAL_Y frame axis"
                )
            site_count = occupied_schema.cell_schema.data_axes[0].size
            if calibration_artifact.site_map.site_axis.size != site_count:
                raise ValueError("occupancy SITE cardinality differs from calibration")
            return _occupancy_cell_navigation(reference, artifact)

    def _load_occupancy_cell_source(
        self,
        reference: OccupancyArtifactRef,
        selection: Selection | None,
        *,
        expected_navigation=None,
    ):
        """Compose one self-contained exact-cell view."""

        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        from zlc_frontend.site_map_render import (
            OccupancyCellNavigation,
            OccupancyCellView,
        )
        from zlc_frontend.site_map import site_ring_radius
        from zlc_frontend.figure import (
            DatasetId,
            EvaluatedAxis,
            EvaluatedImage,
            EvaluatedInput,
        )
        from zlc_frontend.image_view import ImageViewportTransform
        from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

        if expected_navigation is not None and not isinstance(
            expected_navigation,
            OccupancyCellNavigation,
        ):
            raise TypeError("expected_navigation must be OccupancyCellNavigation or None")

        with _service_guard(self._token) as services:
            occupancy_repository = _occupancy_repository(services)
            calibration_repository = _calibration_repository(services)
            resolved = occupancy_repository.admit(
                reference,
                services.capture_repository,
                calibration_repository,
            )
            artifact = resolved.artifact
            self._require_binding(resolved.readout_binding)
            source_ref = artifact.source_capture_ref
            calibration_ref = artifact.calibration_reference
            source = services.capture_repository.admit(source_ref)
            source_artifact = source.artifact
            frame_source = source_artifact.frame_source
            source_schema = frame_source.schema
            calibration = calibration_repository.load(calibration_ref)
            if calibration.frame_contract.binding != resolved.readout_binding:
                raise ValueError("occupancy source and calibration bindings differ")
            frame_schema = source_schema.cell_schema
            if len(frame_schema.data_shape) != 2:
                raise ValueError("physical occupancy map requires a two-dimensional camera frame")
            frame_axes = frame_schema.data_axes
            x_positions = tuple(
                index for index, axis in enumerate(frame_axes) if axis.role == SPATIAL_X
            )
            y_positions = tuple(
                index for index, axis in enumerate(frame_axes) if axis.role == SPATIAL_Y
            )
            if len(x_positions) != 1 or len(y_positions) != 1:
                raise ValueError(
                    "physical occupancy map requires exactly one declared "
                    "SPATIAL_X and SPATIAL_Y frame axis"
                )
            x_position, y_position = x_positions[0], y_positions[0]
            if x_position == y_position:
                raise ValueError("physical occupancy frame axes are not distinct")
            x_axis, y_axis = frame_axes[x_position], frame_axes[y_position]
            occupied_schema = artifact.occupied.schema
            site_axis = occupied_schema.cell_schema.data_axes[0]
            current_navigation = _occupancy_cell_navigation(reference, artifact)
            if (
                expected_navigation is not None
                and current_navigation.identity != expected_navigation.identity
            ):
                raise ValueError("occupancy artifact changed after navigation inspection")
            navigation = (
                current_navigation
                if expected_navigation is None
                else expected_navigation
            )
            repeat_index, point_storage_index, logical_point, _cell_label = (
                navigation.resolve_selection(selection)
            )
            address = DatasetCellAddress(
                repeat_index,
                point_storage_index,
            )

            r = address.repeat_index
            p = address.point_storage_index
            occupied = np.array(
                artifact.occupied.values[r, p, :],
                copy=True,
                order="C",
            )
            validity = artifact.occupied.validity
            if not isinstance(validity, ComponentValidity):
                raise TypeError("occupancy artifact lacks component validity")
            site_validity = np.array(validity.mask[r, p, :], copy=True, order="C")
            revision = artifact.occupied.revision
            generation = artifact.generation
            occupancy_input = EvaluatedInput(
                DatasetId(f"occupancy-cell-state-{reference.manifest_digest}"),
                artifact.occupied.ref(generation),
            )
            model_kind = artifact.model_kind
            del artifact, resolved, validity

            calibration_site_axis = calibration.site_map.site_axis
            if calibration_site_axis != site_axis:
                raise ValueError("occupancy SITE axis differs from its calibration")
            if np.any(site_validity & ~calibration.site_map.validity.mask):
                raise ValueError("occupancy marks a calibration-invalid site as valid")
            centers_xy = np.array(
                calibration.site_map.coordinates_xy,
                copy=True,
                order="C",
            )
            coordinate_frame = calibration.site_map.coordinate_frame
            if (
                x_axis.coordinate_frame != coordinate_frame
                or y_axis.coordinate_frame != coordinate_frame
            ):
                raise ValueError(
                    "camera spatial axes and calibration centers use different "
                    "coordinate frames"
                )
            del calibration, calibration_site_axis

            if source_schema.repeat_axis != occupied_schema.repeat_axis or (
                source_schema.point_axes != occupied_schema.point_axes
                or source_schema.point_layout != occupied_schema.point_layout
            ):
                raise ValueError("occupancy outer axes differ from the source capture")
            if frame_source.revision != revision:
                raise ValueError("occupancy revision differs from its source frame revision")
            source_generation = source_artifact.provenance.generation
            background_input = EvaluatedInput(
                DatasetId(f"occupancy-cell-frame-{source_ref.manifest_digest}"),
                frame_source.ref(source_generation),
            )
            sample = frame_source.read(address)
            if sample.image.schema != frame_schema:
                raise ValueError("exact source frame differs from the inspected frame schema")
            frame_validity = expand_value_validity(
                sample.image.validity,
                sample.image.schema,
            )
            frame_order_yx = (y_position, x_position)
            frame_values_yx = np.transpose(sample.image.values, frame_order_yx)
            frame_validity_yx = np.transpose(frame_validity, frame_order_yx)

            def evaluated_axis(axis: AxisSpec):
                indices = tuple(range(axis.size))
                return EvaluatedAxis(
                    axis.axis_id,
                    axis.name,
                    axis.role,
                    axis.unit,
                    indices,
                    tuple(axis.coordinate_at(index) for index in indices),
                )

            background = EvaluatedImage(
                evaluated_axis(x_axis),
                evaluated_axis(y_axis),
                frame_values_yx,
                frame_validity_yx,
            )
            home_viewport = ImageViewportTransform((y_axis, x_axis))
            metadata = sample.metadata
            timestamp = f"{metadata.captured_at:.9f}s"
            summary = (
                f"{reference.target_ref} | source={source_ref.target_ref} | "
                f"calibration={calibration_ref.target_ref}\n"
                f"model={model_kind.value} | revision={revision.value} | "
                f"generation={generation.value} | address=({r}, {p}) | "
                f"logical_point={logical_point}\n"
                f"frame ordinal={metadata.source_ordinal} | frame_stamp={metadata.frame_stamp} | "
                f"camera_stamp={metadata.camera_stamp} | captured_at={timestamp} | "
                f"correlation={metadata.correlation_id}"
            )
            cell_identity = canonical_digest(
                {
                    "schema": "zlc_frontend.ExactOccupancyCell",
                    "occupancy_artifact": reference.target_ref,
                    "source_capture": source_ref.target_ref,
                    "calibration": calibration_ref.target_ref,
                    "repeat_index": r,
                    "point_storage_index": p,
                    "logical_point": logical_point,
                }
            )
            view = OccupancyCellView(
                background=background,
                background_input=background_input,
                occupancy_input=occupancy_input,
                home_viewport=home_viewport,
                site_axis=site_axis,
                coordinate_frame=coordinate_frame,
                centers_xy=centers_xy,
                site_radius=site_ring_radius(centers_xy),
                occupied=occupied,
                site_validity=site_validity,
                calibration_identity=calibration_ref.target_ref,
                cell_identity=cell_identity,
                cell_selection=navigation.selection_for_indices(
                    repeat_index,
                    logical_point,
                ),
                run_id=source_artifact.run_id,
                provenance_epoch_id=source_generation.value,
                summary=summary,
            )
            del source, source_artifact, frame_source, sample, frame_validity
            return view

    def occupancy_cell_view(
        self,
        reference: OccupancyArtifactRef,
        *,
        selection: Selection | None = None,
    ):
        """Return the exact typed SiteMap presentation for one occupancy cell."""

        return self._load_occupancy_cell_source(reference, selection)

    def occupancy_cell_gui(
        self,
        reference: OccupancyArtifactRef,
        *,
        selection: Selection | None = None,
    ):
        """Open one exact same-shot camera/occupancy physical map."""

        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        from Zou_lab_control.workbench import open_occupancy_cell_workbench

        return open_occupancy_cell_workbench(
            self._inspect_occupancy_cell_navigation,
            self._load_occupancy_cell_source,
            reference,
            selection=selection,
        )


def _project_notebook_figure(
    services,
    source,
    *,
    intent,
    selection,
    preferences,
    occupancy_output,
    materialize: bool,
    draft_fit_result: FitResultBatch | None = None,
    preloaded_snapshot: OwnedSnapshot | None = None,
    preinspected_schema=None,
    preinspected_dataset_ref=None,
):
    """Composition-only ref dispatch; frontend never sees a neutral repository."""

    from zlc_frontend.figure import (
        DatasetDescriptor,
        DatasetId,
        FigureDocument,
        FigureLayer,
        ResolvedDataset,
        ResolvedDatasetMap,
        RepeatViewMode,
        SuggestionStatus,
        ViewIntent,
        ViewPreferences,
        suggest_fit_view,
        suggest_view,
    )

    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")
    if intent is not None and not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent or None")
    if preferences is not None and not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")
    is_occupancy = isinstance(source, OccupancyArtifactRef)
    if occupancy_output is not None and occupancy_output not in (
        "occupied",
        "counts",
    ):
        raise ValueError("occupancy_output must be 'occupied', 'counts', or None")
    if not is_occupancy and occupancy_output is not None:
        raise ValueError("occupancy_output is valid only for OccupancyArtifactRef")

    if draft_fit_result is not None and not isinstance(
        draft_fit_result,
        FitResultBatch,
    ):
        raise TypeError("draft_fit_result must be FitResultBatch or None")
    if preloaded_snapshot is not None and not isinstance(
        preloaded_snapshot,
        OwnedSnapshot,
    ):
        raise TypeError("preloaded_snapshot must be OwnedSnapshot or None")
    if preloaded_snapshot is not None and not isinstance(
        source,
        AdmittedFitResult,
    ):
        raise TypeError(
            "a preloaded figure snapshot is valid only for an admitted saved fit"
        )
    if (preinspected_schema is None) != (preinspected_dataset_ref is None):
        raise ValueError(
            "preinspected_schema and preinspected_dataset_ref must be supplied together"
        )
    if preinspected_schema is not None:
        if materialize or preloaded_snapshot is not None:
            raise ValueError(
                "preinspected metadata is valid only for document-only projection"
            )
        if not isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError(
                "preinspected metadata requires its capture or scan source"
            )
        if preinspected_dataset_ref.schema_fingerprint != preinspected_schema.fingerprint:
            raise ValueError("preinspected dataset ref differs from its schema")

    fit_result = draft_fit_result
    snapshot = preloaded_snapshot
    source_final = None
    selected_occupancy_output = None
    source_label = "capture"
    if draft_fit_result is not None:
        if not isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError(
                "a draft fit result requires its capture or scan source"
            )
        source_ref = source
    elif isinstance(source, ScanArtifactRef):
        source_label = "scan"
        source_ref = source
    elif is_occupancy:
        selected_occupancy_output = (
            "occupied" if occupancy_output is None else occupancy_output
        )
        repository = _occupancy_repository(services)
        resolved = repository.admit(
            source,
            services.capture_repository,
            _calibration_repository(services),
        )
        artifact = resolved.artifact
        selected_block = (
            artifact.occupied
            if selected_occupancy_output == "occupied"
            else artifact.counts
        )
        schema = selected_block.schema
        model_kind = artifact.model_kind
        if materialize:
            snapshot = (
                artifact.occupied_snapshot
                if selected_occupancy_output == "occupied"
                else artifact.counts_snapshot
            )
        del selected_block, artifact, resolved
        source_label = (
            f"occupancy {selected_occupancy_output} | {model_kind.value}"
        )
        source_ref = None
    elif isinstance(source, CaptureArtifactRef):
        source_ref = source
    elif isinstance(source, FitExecution):
        source_ref = source.source_artifact_ref
        fit_result = source.result
    elif isinstance(source, FitResultArtifactRef):
        admitted_fit = services.fit_repository.load(
            source,
            capture_repository=services.capture_repository,
            scan_repository=services.scan_repository,
        )
        source_ref = admitted_fit.source_artifact_ref
        fit_result = admitted_fit.result
    elif isinstance(source, AdmittedFitResult):
        source_ref = source.source_artifact_ref
        fit_result = source.result
    else:
        raise TypeError(
            "figure source must be ScanArtifactRef, OccupancyArtifactRef, "
            "CaptureArtifactRef, FitExecution, FitResultArtifactRef, "
            "or AdmittedFitResult"
        )

    if source_ref is not None:
        if isinstance(source_ref, CaptureArtifactRef):
            if snapshot is None:
                if preinspected_schema is None:
                    source_final = services.capture_repository.admit(source_ref)
                    capture = source_final.artifact
                    schema = capture.frame_source.schema
                    source_dataset_ref = capture.frame_source.ref(
                        capture.provenance.generation
                    )
                else:
                    schema = preinspected_schema
                    source_dataset_ref = preinspected_dataset_ref
            else:
                schema = snapshot.block.schema
                source_dataset_ref = snapshot.ref
        elif isinstance(source_ref, ScanArtifactRef):
            if snapshot is None:
                if preinspected_schema is None:
                    source_final = services.scan_repository.admit(source_ref)
                    scan = source_final
                    schema = scan.output_schema
                    source_dataset_ref = scan.output_dataset_ref
                else:
                    schema = preinspected_schema
                    source_dataset_ref = preinspected_dataset_ref
            else:
                schema = snapshot.block.schema
                source_dataset_ref = snapshot.ref
        else:
            raise TypeError("fit source artifact kind is not current")
    if fit_result is not None:
        validate_fit_result_source_binding(
            fit_result,
            source_dataset_ref,
            schema,
        )
    if fit_result is None:
        if intent is None:
            roles = {
                axis.role
                for axis in (
                    schema.repeat_axis,
                    *schema.point_axes,
                    *schema.cell_schema.data_axes,
                )
            }
            if selected_occupancy_output == "occupied":
                if roles.intersection((SCAN_POINT, SPECTRAL, MONITOR_HISTORY)):
                    resolved_intent = ViewIntent.CURVE
                else:
                    resolved_intent = ViewIntent.METER
            elif SPATIAL_X in roles and SPATIAL_Y in roles:
                resolved_intent = ViewIntent.IMAGE
            elif roles.intersection((SCAN_POINT, SPECTRAL, MONITOR_HISTORY)):
                resolved_intent = ViewIntent.CURVE
            else:
                resolved_intent = ViewIntent.HISTOGRAM
        else:
            resolved_intent = intent
        resolved_preferences = preferences
        if (
            selected_occupancy_output == "occupied"
            and resolved_intent is ViewIntent.METER
            and (preferences is None or preferences.repeat_mode is None)
        ):
            resolved_preferences = replace(
                ViewPreferences() if preferences is None else preferences,
                repeat_mode=RepeatViewMode.MEAN,
            )
        suggestion = suggest_view(
            schema,
            resolved_intent,
            selection,
            resolved_preferences,
        )
        label = source_label
    else:
        suggestion = suggest_fit_view(
            schema,
            fit_result,
            selection,
            preferences,
        )
        if suggestion.spec is not None and intent not in (None, suggestion.spec.intent):
            raise ValueError("requested figure intent is incompatible with the fitted axes")
        label = f"fit: {fit_result.spec.model_id}"
    if suggestion.status is SuggestionStatus.NEEDS_INPUT:
        details = "; ".join(reason.message for reason in suggestion.reasons)
        raise ValueError(f"figure view needs explicit input: {details}")
    assert suggestion.spec is not None

    dataset_id = DatasetId("source")
    document = FigureDocument(
        document_id=f"notebook-{uuid4().hex}",
        revision=0,
        datasets=(DatasetDescriptor(dataset_id, label, schema.fingerprint),),
        layers=(FigureLayer("data", dataset_id, suggestion.spec),),
    )
    if not materialize:
        return document, None, fit_result
    if snapshot is None and source_ref is not None:
        if source_final is None:
            raise RuntimeError("figure FINAL source is unavailable")
        del schema, suggestion
        if isinstance(source_ref, CaptureArtifactRef):
            snapshot = source_final.materialize_snapshot()
        elif isinstance(source_ref, ScanArtifactRef):
            snapshot = services.scan_repository.materialize(source_ref).snapshot
        else:  # pragma: no cover - source kind is closed above
            raise TypeError("fit source artifact kind is not current")
    if snapshot is None:
        raise RuntimeError("figure source was not materialized")
    return (
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_result,
    )


def _data_figure_for_services(
    services: _ExperimentServices,
    source,
    *,
    intent,
    selection,
    preferences,
    occupancy_output,
    draft_fit_result: FitResultBatch | None = None,
    preloaded_snapshot: OwnedSnapshot | None = None,
) -> "DataFigure":
    """Build one frozen DataFigure while repository authority stays private."""

    document, datasets, fit_result = _project_notebook_figure(
        services,
        source,
        intent=intent,
        selection=selection,
        preferences=preferences,
        occupancy_output=occupancy_output,
        materialize=True,
        draft_fit_result=draft_fit_result,
        preloaded_snapshot=preloaded_snapshot,
    )
    assert datasets is not None
    from zlc_frontend import DataFigure

    return DataFigure(
        document,
        datasets,
        fit_results=({"data": fit_result} if fit_result is not None else None),
    )


class Experiment:
    """Public notebook root containing values, requests, and narrow facades only."""

    __slots__ = (
        "_authority_token",
        "name",
        "device_catalog",
        "installation_config",
        "pulse",
        "readout",
    )

    def __init__(
        self,
        authority_token: object,
        *,
        name: str,
        device_catalog: DeviceCatalogView,
        installation_config: InstallationConfigDocument,
    ) -> None:
        self._authority_token = authority_token
        self.name = _text(name, "experiment name")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if not isinstance(installation_config, InstallationConfigDocument):
            raise TypeError(
                "installation_config must be InstallationConfigDocument"
            )
        self.device_catalog = device_catalog
        self.installation_config = installation_config
        self.readout = ReadoutFacade(authority_token)
        self.pulse = PulseFacade(authority_token)

    def pulse_gui(
        self,
        *,
        document: PulseDocument | None = None,
        path: str | Path | None = None,
    ):
        """Open the current PulseDocument editor on this exact installation."""

        from zlc_workbench.pulse_editor.app import open_pulse_editor

        key = "pulse-editor"
        with _service_guard(self._authority_token) as services:
            existing = services.gui_handles.get(key)
            if existing is not None:
                if document is not None or path is not None:
                    raise RuntimeError(
                        "this Experiment already owns a Pulse editor; use its Open "
                        "control instead of replacing unsaved state"
                    )
                if bool(getattr(existing, "permanently_closed", False)):
                    raise RuntimeError("this Experiment's Pulse editor is closing")
                existing.restore_window()
                return existing
            body = open_pulse_editor(self, document=document, path=path)
            services.gui_handles[key] = body
            return body

    def scan_gui(self, request: ScanRequest | OccupancyScanRequest):
        """Open the current typed SCAN_SLOT panel for a frozen request."""

        from Zou_lab_control.workbench import open_scan_workbench

        return open_scan_workbench(self, request)

    def task_console(self, *, task=None, state=None, **kwargs):
        """Lazily open the task console bound to this experiment.

        One composition root owns the window (``zlc_workbench.task_console.app``),
        so a notebook and the double-clickable launcher open the SAME console.
        """

        from zlc_workbench.task_console.app import open_task_console

        return open_task_console(self, task=task, state=state, **kwargs)

    def device_manager(self):
        """Open the one config/admin window bound to this installation."""

        from zlc_workbench.device_manager.app import open_device_manager

        key = "device-manager"
        with _service_guard(self._authority_token) as services:
            existing = services.gui_handles.get(key)
            if existing is not None:
                if bool(getattr(existing, "permanently_closed", False)):
                    raise RuntimeError("this Experiment's Device manager is closing")
                existing.restore_window()
                return existing
            body = open_device_manager(experiment=self)
            services.gui_handles[key] = body
            return body

    def start(self, request: CaptureRequest) -> RunHandle:
        return _start(self._authority_token, request)

    def run(self, request: CaptureRequest) -> CaptureArtifactRef:
        return _run(self._authority_token, request)

    def inspect(self, request: CaptureRequest) -> PlanDescriptor:
        with _service_guard(self._authority_token) as services:
            return _prepare_capture_for_services(services, request).descriptor

    def start_scan(
        self,
        request: ScanRequest | OccupancyScanRequest,
    ) -> RunHandle:
        return _start_scan(self._authority_token, request)

    def scan(
        self,
        request: ScanRequest | OccupancyScanRequest,
    ) -> ScanArtifactRef:
        return _run_scan(self._authority_token, request)

    def inspect_scan(self, request: ScanRequest) -> PlanDescriptor:
        with _service_guard(self._authority_token) as services:
            _plan, descriptor = _compile_direct_scan_for_services(services, request)
            return descriptor

    def fit(
        self,
        source: CaptureArtifactRef | ScanArtifactRef,
        spec: FitSpec | None = None,
        *,
        model: str | None = None,
        committed_transform: CommittedTransform | None = None,
        fit_axis_ids: tuple[AxisId, ...] | None = None,
        constraints: tuple[FitParameterConstraint, ...] = (),
        numeric_policy: FitNumericPolicy | None = None,
    ) -> FitExecution:
        """Fit one committed capture or FINAL scan without hidden reduction."""

        if not isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError("source must be CaptureArtifactRef or ScanArtifactRef")
        if (spec is None) == (model is None):
            raise ValueError("provide exactly one of spec or model")
        with _fit_service_guard(self._authority_token) as services:
            def closing_cancel_check() -> bool:
                with services.operation_lock:
                    return services.state != "OPEN"

            if spec is None:
                assert model is not None
                if isinstance(source, CaptureArtifactRef):
                    capture = services.capture_repository.admit(source).artifact
                    schema = capture.frame_source.schema
                else:
                    scan = services.scan_repository.admit(source)
                    schema = scan.output_schema
                spec = fit_spec_for(
                    schema,
                    model,
                    committed_transform=committed_transform,
                    fit_axis_ids=fit_axis_ids,
                    constraints=constraints,
                    numeric_policy=(
                        FitNumericPolicy()
                        if numeric_policy is None
                        else numeric_policy
                    ),
                )
                del schema
            elif any(
                value is not None
                for value in (
                    committed_transform,
                    fit_axis_ids,
                    numeric_policy,
                )
            ) or constraints:
                raise ValueError(
                    "spec cannot be combined with model convenience arguments"
                )
            if isinstance(source, CaptureArtifactRef):
                return services.fit_repository.execute_capture(
                    services.capture_repository,
                    source,
                    spec,
                    cancel_check=closing_cancel_check,
                )
            return services.fit_repository.execute_scan(
                services.scan_repository,
                source,
                spec,
                cancel_check=closing_cancel_check,
            )

    def load_fit(
        self,
        reference: FitResultArtifactRef,
    ) -> AdmittedFitResult:
        with _service_guard(self._authority_token) as services:
            return services.fit_repository.load(
                reference,
                capture_repository=services.capture_repository,
                scan_repository=services.scan_repository,
            )

    def _open_fit_capable_figure_gui(
        self,
        display_source,
        fit_source: CaptureArtifactRef | ScanArtifactRef,
        *,
        intent,
        selection: Selection | None,
        preferences,
        occupancy_output: str | None,
        selected_model: str | None = None,
        initial_fit_spec: FitSpec | None = None,
        initial_selection: Selection | None = None,
        open_analysis: bool = False,
        timeout_seconds: float = _DEFAULT_FIT_GUI_TIMEOUT_SECONDS,
        initial_fit_result_identity: str | None = None,
        direct_fit_single_panel: bool = False,
    ):
        """Compose the one Figure-owned Fit host without exposing repositories."""

        if not isinstance(fit_source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError("fit_source must be CaptureArtifactRef or ScanArtifactRef")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        chosen_model = (
            None if selected_model is None else _text(selected_model, "model")
        )
        if initial_fit_spec is not None and not isinstance(
            initial_fit_spec,
            FitSpec,
        ):
            raise TypeError("initial_fit_spec must be FitSpec or None")
        if initial_fit_spec is not None:
            if chosen_model is None:
                chosen_model = initial_fit_spec.model_id
            elif chosen_model != initial_fit_spec.model_id:
                raise ValueError("selected model differs from the initial FitSpec")

        def load_final_source(services):
            if isinstance(fit_source, CaptureArtifactRef):
                return services.capture_repository.admit(fit_source).artifact
            return services.scan_repository.admit(fit_source)

        def source_schema(services):
            artifact = load_final_source(services)
            if isinstance(fit_source, CaptureArtifactRef):
                return artifact.frame_source.schema
            return artifact.output_schema

        def prepare_fit(
            fit_axis_ids: tuple[AxisId, ...],
            authority_selection: Selection | None,
        ):
            if not fit_axis_ids or any(
                not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids
            ):
                raise ValueError("Figure Fit requires exact named display axes")
            if authority_selection is not None and not isinstance(
                authority_selection,
                Selection,
            ):
                raise TypeError("Figure Fit selection must be Selection or None")
            with _service_guard(self._authority_token) as services:
                schema = source_schema(services)
            seed_spec = initial_fit_spec
            if seed_spec is not None:
                if seed_spec.input_schema_fingerprint != schema.fingerprint:
                    raise ValueError("initial FitSpec belongs to another source schema")
                if seed_spec.fit_axis_ids != tuple(fit_axis_ids):
                    raise ValueError(
                        "displayed named Fit axes differ from the initial FitSpec"
                    )
            from zlc_frontend import fit_authoring_option

            options = []
            for definition in fit_model_catalog():
                try:
                    bound = suggest_fit_draft(
                        schema,
                        definition.model_id,
                        fit_axis_ids=tuple(fit_axis_ids),
                        selection=authority_selection,
                        constraints=(
                            seed_spec.constraints
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else ()
                        ),
                        numeric_policy=(
                            seed_spec.numeric_policy
                            if seed_spec is not None
                            and definition.model_id == seed_spec.model_id
                            else FitNumericPolicy()
                        ),
                    )
                    if (
                        seed_spec is not None
                        and definition.model_id == seed_spec.model_id
                        and bound.spec.committed_transform
                        == seed_spec.committed_transform
                    ):
                        del bound
                        bound = bind_fit(seed_spec, schema)
                except ValueError:
                    continue
                option = fit_authoring_option(bound)
                options.append(option)
                del bound
            if not options:
                raise ValueError(
                    "the displayed named axes and selection have no compatible Fit model"
                )
            if chosen_model is not None and chosen_model not in {
                option.spec.model_id for option in options
            }:
                raise ValueError(
                    f"Fit model {chosen_model!r} is not compatible with this panel"
                )
            return tuple(options)

        def execute_fit(
            spec: FitSpec,
            cancel_check,
            deadline_monotonic: float,
        ) -> FitExecution:
            if not isinstance(spec, FitSpec):
                raise TypeError("Figure Fit execution requires FitSpec")
            with _fit_service_guard(self._authority_token) as services:
                def combined_cancel_check() -> bool:
                    with services.operation_lock:
                        closing = services.state != "OPEN"
                    return closing or bool(cancel_check())

                if isinstance(fit_source, CaptureArtifactRef):
                    return services.fit_repository.execute_capture(
                        services.capture_repository,
                        fit_source,
                        spec,
                        cancel_check=combined_cancel_check,
                        deadline_monotonic=deadline_monotonic,
                    )
                return services.fit_repository.execute_scan(
                    services.scan_repository,
                    fit_source,
                    spec,
                    cancel_check=combined_cancel_check,
                    deadline_monotonic=deadline_monotonic,
                )

        def save_fit_execution(
            execution: FitExecution,
        ) -> FitResultArtifactRef:
            if not isinstance(execution, FitExecution):
                raise TypeError("Fit save requires FitExecution")
            if execution.source_artifact_ref != fit_source:
                raise ValueError("Fit execution belongs to another source artifact")
            with _service_guard(self._authority_token):
                return execution.save()

        def reload_fit_result(
            reference: FitResultArtifactRef,
        ) -> FitResultBatch:
            with _service_guard(self._authority_token) as services:
                admitted = services.fit_repository.load(
                    reference,
                    capture_repository=services.capture_repository,
                    scan_repository=services.scan_repository,
                )
            if admitted.source_artifact_ref != fit_source:
                raise ValueError("saved Fit reopened against another source artifact")
            return admitted.result

        figure_factory = self.figure
        if direct_fit_single_panel:
            if display_source != fit_source:
                raise ValueError(
                    "direct Fit single-panel display requires its exact source artifact"
                )

            def figure_factory(
                source,
                *,
                intent,
                selection,
                preferences,
                occupancy_output=None,
            ):
                """Resolve the labelled display cell on the Figure worker."""

                if source != fit_source:
                    raise ValueError("direct Fit Figure loader received another source")
                with _service_guard(self._authority_token) as services:
                    artifact = load_final_source(services)
                    if isinstance(fit_source, CaptureArtifactRef):
                        schema = artifact.frame_source.schema
                        dataset_ref = artifact.frame_source.ref(
                            artifact.provenance.generation
                        )
                    else:
                        schema = artifact.output_schema
                        dataset_ref = artifact.output_dataset_ref
                    del artifact
                    seed_document, _datasets, _fit_result = (
                        _project_notebook_figure(
                            services,
                            source,
                            intent=intent,
                            selection=selection,
                            preferences=preferences,
                            occupancy_output=occupancy_output,
                            materialize=False,
                            preinspected_schema=schema,
                            preinspected_dataset_ref=dataset_ref,
                        )
                    )
                    from zlc_frontend.figure import (
                        fit_single_panel_presentation,
                    )

                    display_selection, display_preferences = (
                        fit_single_panel_presentation(
                            schema,
                            seed_document.layers[0].view,
                            preferences,
                        )
                    )
                    del schema, dataset_ref, seed_document
                    return _data_figure_for_services(
                        services,
                        source,
                        intent=intent,
                        selection=display_selection,
                        preferences=display_preferences,
                        occupancy_output=occupancy_output,
                    )

        from Zou_lab_control.workbench import open_figure_workbench

        return open_figure_workbench(
            figure_factory,
            display_source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            occupancy_output=occupancy_output,
            fit_preparer=prepare_fit,
            fit_executor=execute_fit,
            fit_saver=save_fit_execution,
            fit_reloader=reload_fit_result,
            fit_selected_model=chosen_model,
            fit_initial_selection=initial_selection,
            open_fit_analysis=open_analysis,
            fit_timeout_seconds=timeout,
            initial_fit_result_identity=initial_fit_result_identity,
        )

    def fit_gui(
        self,
        source: CaptureArtifactRef | ScanArtifactRef,
        *,
        model: str | None = None,
        committed_transform: CommittedTransform | None = None,
        timeout_seconds: float = _DEFAULT_FIT_GUI_TIMEOUT_SECONDS,
    ):
        """Open the same DataFigure host with its Analysis tab selected."""

        if not isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            raise TypeError("source must be CaptureArtifactRef or ScanArtifactRef")
        initial_selection = None
        if committed_transform is not None:
            if not isinstance(committed_transform, CommittedTransform):
                raise TypeError(
                    "committed_transform must be CommittedTransform or None"
                )
            operations = committed_transform.spec.operations
            if len(operations) != 1 or not isinstance(operations[0], Selection):
                raise ValueError(
                    "fit_gui accepts only one range-preserving Selection transform"
                )
            initial_selection = operations[0]
        return self._open_fit_capable_figure_gui(
            source,
            source,
            intent=None,
            selection=None,
            preferences=None,
            occupancy_output=None,
            selected_model=model,
            initial_selection=initial_selection,
            open_analysis=True,
            timeout_seconds=timeout_seconds,
            direct_fit_single_panel=True,
        )

    def figure_document(
        self,
        source: (
            ScanArtifactRef
            | OccupancyArtifactRef
            | CaptureArtifactRef
            | FitExecution
            | FitResultArtifactRef
            | AdmittedFitResult
        ),
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        occupancy_output: str | None = None,
    ) -> "FigureDocument":
        """Project one committed source into a renderer-free document.

        Occupancy defaults to its classified ``occupied`` block; ``counts`` is
        an explicit presentation alternative.
        """

        with _service_guard(self._authority_token) as services:
            document, _datasets, _fit = _project_notebook_figure(
                services,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
                materialize=False,
            )
        return document

    def figure(
        self,
        source: (
            ScanArtifactRef
            | OccupancyArtifactRef
            | CaptureArtifactRef
            | FitExecution
            | FitResultArtifactRef
            | AdmittedFitResult
        ),
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        occupancy_output: str | None = None,
    ) -> "DataFigure":
        """Resolve one frozen source and return its optional-render DataFigure."""

        with _service_guard(self._authority_token) as services:
            return _data_figure_for_services(
                services,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
            )

    def figure_gui(
        self,
        source: (
            ScanArtifactRef
            | OccupancyArtifactRef
            | CaptureArtifactRef
            | FitExecution
            | FitResultArtifactRef
            | AdmittedFitResult
            | str
            | Path
            | None
        ) = None,
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        occupancy_output: str | None = None,
    ):
        """Open the saved-figure browser or show one typed frozen source.

        ``None`` opens the session-independent FigureViewer.  A filesystem path
        opens that viewer and commits the selected current archive; typed
        artifact/result inputs retain their existing interactive dispatch.
        """

        if source is None or isinstance(source, (str, Path)):
            supplied_overrides = tuple(
                name
                for name, value in (
                    ("intent", intent),
                    ("selection", selection),
                    ("preferences", preferences),
                    ("occupancy_output", occupancy_output),
                )
                if value is not None
            )
            if supplied_overrides:
                raise ValueError(
                    "saved FigureViewer does not accept typed-source view overrides: "
                    + ", ".join(supplied_overrides)
                )
            from zlc_workbench.figure_viewer.app import open_figure_viewer

            return open_figure_viewer(path=source)

        if (
            isinstance(source, FitResultArtifactRef)
            and intent is None
            and selection is None
            and preferences is None
            and occupancy_output is None
        ):
            authority_token = self._authority_token
            session_thread_id = None
            session_admitted = None
            session_snapshot = None
            session_model = None

            def load_saved_fit_grid_view(
                reference,
                *,
                page_address,
                cell_selection,
            ):
                nonlocal session_thread_id
                nonlocal session_admitted
                nonlocal session_snapshot
                nonlocal session_model
                if reference != source:
                    raise ValueError("saved-fit loader received another artifact ref")
                worker_thread_id = threading.get_ident()
                if session_thread_id is None:
                    session_thread_id = worker_thread_id
                elif session_thread_id != worker_thread_id:
                    raise RuntimeError(
                        "saved-fit view session changed worker thread"
                    )
                with _service_guard(authority_token) as services:
                    from zlc_frontend import FitGridModel

                    if session_admitted is None:
                        admitted = services.fit_repository.load(
                            reference,
                            capture_repository=services.capture_repository,
                            scan_repository=services.scan_repository,
                        )
                        model = FitGridModel.from_result(
                            reference.target_ref,
                            admitted.result,
                        )
                        source_ref = admitted.source_artifact_ref
                        if isinstance(source_ref, CaptureArtifactRef):
                            snapshot = services.capture_repository.materialize_final(
                                source_ref
                            )
                        elif isinstance(source_ref, ScanArtifactRef):
                            snapshot = services.scan_repository.materialize(
                                source_ref
                            ).snapshot
                        else:
                            raise TypeError("fit source artifact kind is not current")
                        session_admitted = admitted
                        session_snapshot = snapshot
                        session_model = model
                    else:
                        admitted = session_admitted
                        snapshot = session_snapshot
                        model = session_model
                        assert snapshot is not None and model is not None
                    if cell_selection is None:
                        page = model.page(page_address)
                        resolved_selection = page.selection
                        resolved_preferences = page.preferences
                        cell_summary = None
                    else:
                        if page_address is not None:
                            raise ValueError(
                                "saved-fit view cannot request a page and cell together"
                            )
                        model.resolve_selection(cell_selection)
                        page = None
                        resolved_selection = cell_selection
                        resolved_preferences = model.focus_preferences()
                        cell_summary = model.cell_summary(
                            admitted.result,
                            cell_selection,
                        )
                    figure = _data_figure_for_services(
                        services,
                        admitted,
                        intent=None,
                        selection=resolved_selection,
                        preferences=resolved_preferences,
                        occupancy_output=None,
                        preloaded_snapshot=snapshot,
                    )
                return figure, model, page, cell_summary

            def open_saved_fit_refit(reference, cell_selection):
                if reference != source:
                    raise ValueError("saved-fit refit received another artifact ref")
                admitted = session_admitted
                model = session_model
                if admitted is None or model is None:
                    raise RuntimeError("saved-fit refit requires an admitted grid session")
                model.resolve_selection(cell_selection)
                result = admitted.result
                fit_source = admitted.source_artifact_ref
                transform = result.spec.committed_transform
                authority_selection = None
                if transform is not None:
                    operations = transform.spec.operations
                    if len(operations) != 1 or not isinstance(
                        operations[0], Selection
                    ):
                        raise ValueError(
                            "saved Fit refit requires its exact range-preserving "
                            "Selection authority"
                        )
                    authority_selection = operations[0]
                return self._open_fit_capable_figure_gui(
                    fit_source,
                    fit_source,
                    intent=None,
                    # The focused batch cell chooses only the displayed source panel.
                    selection=cell_selection,
                    preferences=model.focus_preferences(),
                    occupancy_output=None,
                    selected_model=result.spec.model_id,
                    initial_fit_spec=result.spec,
                    initial_selection=authority_selection,
                    open_analysis=True,
                )

            from Zou_lab_control.workbench import open_saved_fit_grid_workbench

            return open_saved_fit_grid_workbench(
                load_saved_fit_grid_view,
                open_saved_fit_refit,
                source,
            )

        if isinstance(source, (CaptureArtifactRef, ScanArtifactRef)):
            return self._open_fit_capable_figure_gui(
                source,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
            )

        if isinstance(source, (FitExecution, AdmittedFitResult)):
            result = source.result
            transform = result.spec.committed_transform
            initial_authority_selection = None
            if transform is not None:
                operations = transform.spec.operations
                if len(operations) != 1 or not isinstance(
                    operations[0], Selection
                ):
                    raise ValueError(
                        "Fit result analysis requires its exact range-preserving "
                        "Selection authority"
                    )
                initial_authority_selection = operations[0]
            if isinstance(source, AdmittedFitResult):
                identity = (
                    f"{source.reference.repository_id}:"
                    f"{source.reference.manifest_digest}"
                )
                fit_source = source.source_artifact_ref
            else:
                identity = f"draft-execution:{id(source):x}"
                fit_source = source.source_artifact_ref
            return self._open_fit_capable_figure_gui(
                source,
                fit_source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
                selected_model=result.spec.model_id,
                initial_fit_spec=result.spec,
                initial_selection=initial_authority_selection,
                initial_fit_result_identity=identity,
            )

        from Zou_lab_control.workbench import open_figure_workbench

        initial_identity = (
            f"{source.repository_id}:{source.manifest_digest}"
            if isinstance(source, FitResultArtifactRef)
            else None
        )
        return open_figure_workbench(
            self.figure,
            source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            occupancy_output=occupancy_output,
            initial_fit_result_identity=initial_identity,
        )

    def close(self) -> None:
        with _AUTHORITY_LOCK:
            services = _AUTHORITIES.get(self._authority_token)
        if services is None:
            return
        with services.operation_lock:
            if services.state == "CLOSED":
                return
            if services.fit_operation_thread_counts.get(threading.get_ident(), 0):
                raise RuntimeError(
                    "Experiment cannot close reentrantly from its active Fit operation"
                )
            gui_handles = tuple(services.gui_handles.values())
            services.gui_handles.clear()
            for handle in gui_handles:
                retire = getattr(handle, "request_owner_close", None)
                if callable(retire):
                    retire()
            services.state = "CLOSING"
            fit_operations_drained = services.fit_operations_drained
        # A Figure Fit owns repository borrows outside the short composition
        # lock.  Closing flips the state first (its cancel check observes that
        # transition), then waits without holding the lock needed by the Fit's
        # finally block.  One context-manager exit therefore completes cleanup;
        # it never leaves a half-closed Experiment requiring a second close().
        fit_operations_drained.wait()
        with services.operation_lock:
            if services.state == "CLOSED":
                return
            if services.active_fit_operations:
                raise RuntimeError("Fit drain event disagrees with operation count")
            shutdown = services.runtime.shutdown(timeout=2.0)
            if not shutdown:
                diagnostics = tuple(
                    getattr(services.runtime, "shutdown_diagnostics", ())
                )
                suffix = (
                    ""
                    if not diagnostics
                    else ": " + "; ".join(diagnostics)
                )
                raise RuntimeError(
                    "Experiment close did not complete" + suffix
                )
            failures = _cleanup_failures(
                services.fit_repository.close,
                (
                    None
                    if services.occupancy_repository is None
                    else services.occupancy_repository.close
                ),
                (
                    None
                    if services.calibration_repository is None
                    else services.calibration_repository.close
                ),
                services.scan_repository.close,
                services.capture_repository.close,
            )
            if failures:
                raise _ResourceCleanupError(
                    "Experiment close failed",
                    tuple(failures),
                )
            services.state = "CLOSED"
        with _AUTHORITY_LOCK:
            if _AUTHORITIES.get(self._authority_token) is services:
                _AUTHORITIES.pop(self._authority_token, None)

    def __enter__(self) -> "Experiment":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()


def _resolve_role(
    catalog: DeviceCatalogView,
    requested: str | None,
    domain: str,
    preferred: tuple[str, ...],
) -> str:
    if requested is not None:
        info = catalog.require(requested)
        if info.domain != domain:
            raise ValueError(
                f"device role {requested!r} is {info.domain!r}, not {domain!r}"
            )
        return requested
    for role in preferred:
        info = catalog.find(role)
        if info is not None and info.domain == domain:
            return role
    candidates = catalog.roles(domain)
    if len(candidates) != 1:
        raise ValueError(
            f"installation has {len(candidates)} {domain} roles; choose one explicitly"
        )
    return candidates[0]


def _prepare_capture_for_services(
    services: _ExperimentServices,
    request: CaptureRequest,
) -> PreparedFiniteCapture:
    return prepare_finite_capture(
        request,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        camera_port=services.runtime.camera_port(request.camera_ref),
        repository=services.capture_repository,
        start_run=services.runtime.start,
    )


def _prepare_capture_for_workbench(
    experiment: Experiment,
    request: CaptureRequest,
) -> PreparedFiniteCapture:
    """Private friend seam; no notebook authority escapes to the Workbench."""

    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be Experiment")
    with _service_guard(experiment._authority_token) as services:
        return _prepare_capture_for_services(services, request)


def _prepare_temperature_release_recapture_for_services(
    services: _ExperimentServices,
    request: TemperatureReleaseRecaptureRequest,
) -> PreparedTemperatureReleaseRecapture:
    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError(
            "request must be TemperatureReleaseRecaptureRequest"
        )
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
    )
    bound = bind_temperature_release_recapture(
        request,
        calibration,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        camera_port=services.runtime.camera_port(request.camera_ref),
    )
    return prepare_temperature_release_recapture(
        bound,
        calibration,
        start_run=services.runtime.start,
    )


def _prepare_temperature_release_recapture_for_workbench(
    experiment: Experiment,
    request: TemperatureReleaseRecaptureRequest,
) -> PreparedTemperatureReleaseRecapture:
    """Private friend seam for the TaskConsole Measurement presenter."""

    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be Experiment")
    with _service_guard(experiment._authority_token) as services:
        return _prepare_temperature_release_recapture_for_services(
            services,
            request,
        )


def _prepare_camera_measurement_for_services(
    services: _ExperimentServices,
    request: CameraMeasurementRequest,
) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement:
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("request must be CameraMeasurementRequest")
    if request.repeat == 0:
        return prepare_live_camera_measurement(
            request,
            monitor_port=services.runtime.camera_monitor_port(request.camera_ref),
            start_run=services.runtime.start,
        )
    return prepare_finite_camera_measurement(
        request,
        camera_port=services.runtime.camera_port(request.camera_ref),
        repository=services.capture_repository,
        start_run=services.runtime.start,
    )


def _prepare_camera_measurement_for_workbench(
    experiment: Experiment,
    request: CameraMeasurementRequest,
) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement:
    """Private friend seam; no notebook authority escapes to the Workbench.

    The Workbench owns the optional live/preview slot factory, while the domain
    command owns whether this one request runs continuously or stops after K
    complete hardware-triggered cycles.
    """

    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be Experiment")
    with _service_guard(experiment._authority_token) as services:
        return _prepare_camera_measurement_for_services(services, request)


def _prepare_pulse_for_services(
    services: _ExperimentServices,
    request: PulseRunRequest,
) -> PreparedPulseExecution:
    return prepare_pulse_execution(
        request,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        start_run=services.runtime.start,
        on_applied=services.pulse_application._record_applied,
    )


def _start_pulse(token: object, request: PulseRunRequest) -> RunHandle:
    with _service_guard(token) as services:
        prepared = _prepare_pulse_for_services(services, request)
        return services.pulse_application.start(prepared)


def _run_pulse(token: object, request: PulseRunRequest) -> PulseRunResult:
    with _service_guard(token) as services:
        prepared = _prepare_pulse_for_services(services, request)
        handle = services.pulse_application.start(prepared)
        runtime = services.runtime
    result = runtime.wait(handle)
    if not isinstance(result, PulseRunResult):
        raise TypeError("pulse Run returned an unexpected result")
    return result


def _start(token: object, request: CaptureRequest) -> RunHandle:
    with _service_guard(token) as services:
        return _prepare_capture_for_services(services, request).start()


def _run(token: object, request: CaptureRequest) -> CaptureArtifactRef:
    with _service_guard(token) as services:
        handle = _prepare_capture_for_services(services, request).start()
        runtime = services.runtime
    return runtime.wait(handle)


def _bind_autonomous_scan_camera(
    services: _ExperimentServices,
    request: ScanRequest | OccupancyScanRequest,
):
    from zlc_neutral_atom.bootstrap._triggered_capture import (
        TriggeredCameraLayout,
        bind_triggered_camera_acquisition,
    )

    if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
        raise TypeError("request must be a current scan request")
    if not isinstance(request.program, AutonomousScanSlotProgram):
        raise TypeError("autonomous scan binding requires AutonomousScanSlotProgram")
    pulse_port = services.runtime.pulse_port(request.sequencer_ref)
    camera_port = services.runtime.camera_port(request.camera_ref)
    program = AutonomousScanSlotProgram(
        bind_pulse_document_target(
            request.program.document,
            pulse_port.capability.target,
        ),
        request.program.api_values,
    )
    logical_document = program.execution_document
    require_autonomous_scan_resident_capacity(
        logical_document,
        pulse_port.capability.resident_scan_point_capacity,
    )
    execution_document = expand_autonomous_scan_repeats(logical_document)
    point_table = program.point_table
    repeat_count = (
        1 if logical_document.repeat is None else logical_document.repeat.count
    )
    repeat_axis = AxisSpec(
        _SCAN_REPEAT_AXIS_ID,
        "repeat",
        REPEAT,
        repeat_count,
        tuple(range(repeat_count)),
    )
    scan_axes = point_table.point_axes
    binding = bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        pulse_document=execution_document,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channel=request.trigger_channel,
        layout=TriggeredCameraLayout(
            repeat_axis=repeat_axis,
            readout_event_axis_id=_SCAN_READOUT_EVENT_AXIS_ID,
            readout_events_per_repeat=1,
            scan_axes=scan_axes,
            scan_point_layout=point_table.point_layout,
        ),
    )
    if (
        binding.compiled_artifact.source_document_digest
        != execution_document.fingerprint
    ):
        raise RuntimeError(
            "compiled scan pulse differs from the repeat-major execution document"
        )
    return program, point_table, binding


def _bind_api_scan_camera(
    services: _ExperimentServices,
    request: ScanRequest | OccupancyScanRequest,
):
    from zlc_neutral_atom.bootstrap._triggered_capture import (
        bind_api_slot_segmented_camera_acquisition,
    )

    if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
        raise TypeError("request must be a current scan request")
    if not isinstance(request.program, ApiSlotSegmentedProgram):
        raise TypeError("API scan binding requires ApiSlotSegmentedProgram")
    binding = bind_api_slot_segmented_camera_acquisition(
        services.runtime.pulse_port(request.sequencer_ref),
        services.runtime.camera_port(request.camera_ref),
        program=request.program,
        trigger_channel=request.trigger_channel,
        repeat_axis_id=_SCAN_REPEAT_AXIS_ID,
        readout_event_axis_id=_SCAN_READOUT_EVENT_AXIS_ID,
    )
    return binding.program, binding.point_table, binding


def _compiled_scan_artifacts_digest(artifacts) -> str:
    return canonical_digest(
        {
            "owner": "Zou_lab_control.notebook.api-scan-compiled-lineage",
            "artifacts": [artifact.fingerprint for artifact in artifacts],
        }
    )


def _scan_transform(
    source_schema,
    point_table: ScanPointTable,
    requested: DataTransformSpec | None,
):
    requested_operations = () if requested is None else requested.operations
    committed = commit_transform(
        source_schema,
        DataTransformSpec(
            (
                Selection.index(_SCAN_READOUT_EVENT_AXIS_ID, 0),
                *requested_operations,
            )
        ),
    )
    return bind_scan_output_contract(source_schema, point_table, committed)


def _compile_direct_scan_for_services(
    services: _ExperimentServices,
    request: ScanRequest,
):
    if isinstance(request.program, AutonomousScanSlotProgram):
        program, point_table, binding = _bind_autonomous_scan_camera(
            services,
            request,
        )
    elif isinstance(request.program, ApiSlotSegmentedProgram):
        program, point_table, binding = _bind_api_scan_camera(services, request)
    else:
        raise TypeError("request has an unknown pulse-scan program")
    raw_schema = binding.measurement.capture_contract.dataset_schema
    output_contract = _scan_transform(
        raw_schema,
        point_table,
        request.output_transform_spec,
    )
    if isinstance(program, AutonomousScanSlotProgram):
        triggered, descriptor = bind_finite_capture_spec(
            binding=binding,
            block_id=BlockId(
                f"scan-camera-{binding.compiled_artifact.fingerprint[:20]}"
            ),
            camera_ref=request.camera_ref,
            sequencer_ref=request.sequencer_ref,
            execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
            name_prefix="Direct scan",
        )
        plan = compile_direct_scan_artifact_plan(
            triggered,
            services.scan_repository,
            program=program,
            output_contract=output_contract,
        )
        descriptor = replace(
            descriptor,
            output_shape=output_contract.output_dataset_schema.physical_shape,
            output_schema_fingerprint=output_contract.output_schema_fingerprint,
        )
        return plan, descriptor

    compiled_digest = _compiled_scan_artifacts_digest(binding.compiled_artifacts)
    capture = MinimalPipelineSpec(
        f"API segmented scan {program.document.name}",
        binding.measurement,
        BlockId(f"api-scan-camera-{compiled_digest[:20]}"),
    )
    segmented = ApiSlotSegmentedSpec(
        capture,
        binding.pulse_port,
        binding.point_descriptors,
        program.repeat_count,
    )
    plan = compile_api_direct_scan_artifact_plan(
        segmented,
        services.scan_repository,
        program=program,
        output_contract=output_contract,
    )
    descriptor = PlanDescriptor(
        capture.name,
        request.camera_ref.role,
        request.sequencer_ref.role,
        PulseExecutionForm.STATIC_ONCE,
        binding.trigger_channel,
        binding.expected_frames,
        output_contract.output_dataset_schema.physical_shape,
        output_contract.output_schema_fingerprint,
        compiled_digest,
        (
            str(binding.pulse_port.resource_claim.key),
            str(binding.measurement.capture_port.resource_claim.key),
        ),
    )
    return plan, descriptor


def _bind_occupancy_scan_for_services(
    services: _ExperimentServices,
    request: OccupancyScanRequest,
):
    if isinstance(request.program, AutonomousScanSlotProgram):
        program, point_table, binding = _bind_autonomous_scan_camera(
            services,
            request,
        )
        compiled_digest = binding.compiled_artifact.fingerprint
    elif isinstance(request.program, ApiSlotSegmentedProgram):
        program, point_table, binding = _bind_api_scan_camera(services, request)
        compiled_digest = _compiled_scan_artifacts_digest(
            binding.compiled_artifacts
        )
    else:
        raise TypeError("request has an unknown pulse-scan program")
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
    )
    selected_kind = calibration.artifact.select_model(request.model_kind).kind
    identity = canonical_digest(
        {
            "owner": "Zou_lab_control.notebook.occupancy-scan",
            "pulse_scan_program": program.fingerprint,
            "compiled_pulse_lineage": compiled_digest,
            "calibration": calibration_artifact_ref_to_tree(
                request.calibration_ref
            ),
            "model_kind": selected_kind.value,
        }
    )
    processor = OccupancyStreamProcessorSpec(
        calibration,
        StreamId(f"scan-occupancy-{identity}"),
        f"scan-occupancy-{identity}",
        selected_kind,
    )
    source_schema = resolve_occupancy_stream_schema(
        processor,
        binding.measurement.capture_contract.dataset_schema,
    ).counts_schema
    output_contract = _scan_transform(
        source_schema,
        point_table,
        request.output_transform_spec,
    )
    occupancy = OccupancyPipelineSpec(
        f"Occupancy scan {program.document.name}",
        binding.measurement,
        processor,
        BlockId(f"scan-counts-{identity}"),
        BlockId(f"scan-occupied-{identity}"),
    )
    if isinstance(program, AutonomousScanSlotProgram):
        scan_spec = TriggeredOccupancySpec(
            occupancy,
            binding.pulse_port,
            binding.pulse_request,
            binding.trigger_channel,
            binding.cell_plan,
        )
    else:
        scan_spec = ApiSlotSegmentedSpec(
            occupancy,
            binding.pulse_port,
            binding.point_descriptors,
            program.repeat_count,
        )
    return scan_spec, program, output_contract, source_schema


def _compile_occupancy_scan_for_services(
    services: _ExperimentServices,
    request: OccupancyScanRequest,
):
    scan_spec, program, output_contract, _source_schema = (
        _bind_occupancy_scan_for_services(services, request)
    )
    if isinstance(program, AutonomousScanSlotProgram):
        return compile_occupancy_scan_artifact_plan(
            scan_spec,
            services.scan_repository,
            program=program,
            output_contract=output_contract,
        )
    return compile_api_occupancy_scan_artifact_plan(
        scan_spec,
        services.scan_repository,
        program=program,
        output_contract=output_contract,
    )


def _prepare_occupancy_scan_for_workbench(
    experiment: Experiment,
    request: OccupancyScanRequest,
) -> PreparedOccupancyScan:
    """Private friend seam for the typed occupancy progressive panel."""

    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be Experiment")
    if not isinstance(request, OccupancyScanRequest):
        raise TypeError("request must be OccupancyScanRequest")
    with _service_guard(experiment._authority_token) as services:
        scan_spec, program, output_contract, source_schema = (
            _bind_occupancy_scan_for_services(services, request)
        )

    def start(preview):
        with _service_guard(experiment._authority_token) as services:
            if isinstance(program, AutonomousScanSlotProgram):
                plan = compile_occupancy_scan_artifact_plan(
                    scan_spec,
                    services.scan_repository,
                    program=program,
                    output_contract=output_contract,
                    preview=preview,
                )
            else:
                if preview is not None:
                    raise ValueError(
                        "API segmented occupancy is FINAL-only; preview is unsupported"
                    )
                plan = compile_api_occupancy_scan_artifact_plan(
                    scan_spec,
                    services.scan_repository,
                    program=program,
                    output_contract=output_contract,
                )
            return services.runtime.start(plan)

    return PreparedOccupancyScan(
        source_schema=source_schema,
        output_contract=output_contract,
        start=start,
    )


def _compile_scan_for_services(
    services: _ExperimentServices,
    request: ScanRequest | OccupancyScanRequest,
):
    if isinstance(request, ScanRequest):
        return _compile_direct_scan_for_services(services, request)[0]
    if isinstance(request, OccupancyScanRequest):
        return _compile_occupancy_scan_for_services(services, request)
    raise TypeError("request must be a current scan request")


def _start_scan(
    token: object,
    request: ScanRequest | OccupancyScanRequest,
) -> RunHandle:
    with _service_guard(token) as services:
        return services.runtime.start(_compile_scan_for_services(services, request))


def _run_scan(
    token: object,
    request: ScanRequest | OccupancyScanRequest,
) -> ScanArtifactRef:
    with _service_guard(token) as services:
        handle = services.runtime.start(_compile_scan_for_services(services, request))
        runtime = services.runtime
    result = runtime.wait(handle)
    if not isinstance(result, ScanArtifactRef):
        raise TypeError("scan Run returned a non-scan artifact ref")
    return result


def _mot_field_scan_request(request: MotFieldRequest) -> ScanRequest:
    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    return ScanRequest(
        program=request.program,
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        trigger_channel=request.trigger_channel,
        output_transform_spec=None,
    )


def _compile_calibration_for_services(
    services: _ExperimentServices,
    request: CalibrationArtifactRequest,
):
    if not isinstance(request, CalibrationArtifactRequest):
        raise TypeError("request must be CalibrationArtifactRequest")
    from zlc_neutral_atom.readout.calibration_repository import (
        compile_calibration_artifact_plan,
    )

    return compile_calibration_artifact_plan(
        request.source_capture_ref,
        services.capture_repository,
        _calibration_repository(services),
        request.analysis,
        expected_readout_binding=request.readout_binding,
        timeout_seconds=request.timeout_seconds,
    )


def _start_calibration(
    token: object,
    request: CalibrationArtifactRequest,
) -> RunHandle:
    with _service_guard(token) as services:
        plan = _compile_calibration_for_services(services, request)
        return services.runtime.start(plan)


def _run_calibration(
    token: object,
    request: CalibrationArtifactRequest,
) -> CalibrationArtifactRef:
    with _service_guard(token) as services:
        plan = _compile_calibration_for_services(services, request)
        handle = services.runtime.start(plan)
        runtime = services.runtime
    return runtime.wait(handle)


def _compile_detection_for_services(
    services: _ExperimentServices,
    request: DetectionRequest,
):
    if not isinstance(request, DetectionRequest):
        raise TypeError("request must be DetectionRequest")
    from zlc_neutral_atom.readout.occupancy_repository import (
        compile_occupancy_artifact_plan,
    )

    return compile_occupancy_artifact_plan(
        request.source_capture_ref,
        services.capture_repository,
        request.calibration_ref,
        _calibration_repository(services),
        _occupancy_repository(services),
        expected_readout_binding=request.readout_binding,
        readout_event_axis_id=request.readout_event_axis_id,
        model_kind=request.model_kind,
        timeout_seconds=request.timeout_seconds,
    )


def _start_detection(token: object, request: DetectionRequest) -> RunHandle:
    with _service_guard(token) as services:
        plan = _compile_detection_for_services(services, request)
        return services.runtime.start(plan)


def _run_detection(
    token: object,
    request: DetectionRequest,
) -> OccupancyArtifactRef:
    with _service_guard(token) as services:
        plan = _compile_detection_for_services(services, request)
        handle = services.runtime.start(plan)
        runtime = services.runtime
    return runtime.wait(handle)


_CONNECT_SEED_UNSET = object()


def _resolve_installation_document(
    config: str | Path | InstallationConfigDocument,
    seed: int | None | object,
) -> InstallationConfigDocument:
    if isinstance(config, InstallationConfigDocument):
        if seed is not _CONNECT_SEED_UNSET:
            raise ValueError(
                "seed belongs inside an InstallationConfigDocument"
            )
        return config
    if isinstance(config, Path):
        if seed is not _CONNECT_SEED_UNSET:
            raise ValueError("seed cannot override a saved installation config")
        return load_installation_config(config)
    if not isinstance(config, str):
        raise TypeError(
            "config must be 'virtual', a config path, or "
            "InstallationConfigDocument"
        )
    if config == "virtual":
        resolved_seed = 7 if seed is _CONNECT_SEED_UNSET else seed
        return InstallationConfigDocument.virtual(seed=resolved_seed)
    if config == "remote_pulse":
        raise ValueError(
            "remote_pulse requires InstallationConfigDocument.remote_pulse(...) "
            "or a saved installation config"
        )
    if seed is not _CONNECT_SEED_UNSET:
        raise ValueError("seed cannot override a saved installation config")
    return load_installation_config(config)


def connect(
    config: str | Path | InstallationConfigDocument = "virtual",
    *,
    repository: str | Path,
    name: str = "neutral_atom",
    seed: int | None | object = _CONNECT_SEED_UNSET,
    required_pulse_document: PulseDocument | None = None,
) -> Experiment:
    """Compose one notebook Experiment; raw devices remain authority-private."""

    installation_document = _resolve_installation_document(config, seed)
    if required_pulse_document is not None and not isinstance(
        required_pulse_document,
        PulseDocument,
    ):
        raise TypeError("required_pulse_document must be PulseDocument or None")
    if not isinstance(repository, (str, Path)):
        raise TypeError("repository must be an explicit experiment workspace root")
    canonical_name = _text(name, "experiment name")
    repository_root = Path(repository).expanduser().resolve()
    # The composition root owns the workspace hierarchy; each repository owns
    # exactly one child beneath it and never guesses missing ancestors.
    durable_mkdir(repository_root)
    capture_repository = None
    scan_repository = None
    fit_repository = None
    runtime = None
    try:
        capture_repository = CaptureRepository(repository_root / "captures")
        from zlc_neutral_atom.scan.repository import ScanRepository

        scan_repository = ScanRepository(repository_root / "scans")
        fit_repository = FitResultRepository(repository_root / "fits")
        from zlc_neutral_atom.bootstrap._installation import create_installation

        runtime = create_installation(
            installation_document,
            required_pulse_document=required_pulse_document,
        )
        catalog = runtime.device_catalog
        fit_operations_drained = threading.Event()
        fit_operations_drained.set()
        services = _ExperimentServices(
            runtime=runtime,
            capture_repository=capture_repository,
            scan_repository=scan_repository,
            calibration_repository_path=repository_root / "calibrations",
            calibration_repository=None,
            occupancy_repository_path=repository_root / "occupancy",
            occupancy_repository=None,
            fit_repository=fit_repository,
            catalog=catalog,
            installation_config=installation_document,
            pulse_application=PulseApplicationOwner(),
            operation_lock=threading.RLock(),
            fit_operations_drained=fit_operations_drained,
            fit_operation_thread_counts={},
            gui_handles={},
        )
        token = object()
        experiment = Experiment(
            token,
            name=canonical_name,
            device_catalog=catalog,
            installation_config=installation_document,
        )
        _register(token, services)
        return experiment
    except BaseException as error:
        failures = _cleanup_failures(
            (
                None
                if runtime is None
                else lambda: _require_runtime_shutdown(runtime, timeout=2.0)
            ),
            None if fit_repository is None else fit_repository.close,
            None if scan_repository is None else scan_repository.close,
            None if capture_repository is None else capture_repository.close,
        )
        if failures and isinstance(error, Exception):
            raise _ResourceCleanupError(
                "Experiment composition cleanup failed",
                tuple(failures),
            ) from error
        raise


def device_manager(
    config: str | Path | InstallationConfigDocument = "virtual",
    *,
    repository: str | Path | None = None,
    name: str = "neutral_atom",
    seed: int | None | object = _CONNECT_SEED_UNSET,
    on_initialized=None,
):
    """Open the standalone config editor before any installation is composed."""

    document = _resolve_installation_document(config, seed)
    config_path = None
    if isinstance(config, Path):
        config_path = config.expanduser().resolve()
    elif isinstance(config, str) and config not in {"virtual", "remote_pulse"}:
        config_path = Path(config).expanduser().resolve()
    from zlc_workbench.device_manager.app import open_device_manager

    return open_device_manager(
        document=document,
        config_path=config_path,
        repository=repository,
        name=name,
        on_initialized=on_initialized,
    )


__all__ = [
    "AdmittedFitResult",
    "AppliedPulseSnapshot",
    "BackgroundMode",
    "BoxReducer",
    "CalibrationAnalysisRequest",
    "CalibrationArtifactRequest",
    "CalibrationArtifactRef",
    "CalibrationCaptureLayout",
    "CaptureArtifactRef",
    "FitResultArtifactRef",
    "CaptureRequest",
    "CameraMeasurementRequest",
    "connect",
    "device_manager",
    "DetectionRequest",
    "Experiment",
    "FitExecution",
    "GridOrder",
    "GreyMolassesDetuningRequest",
    "InstallationConfigDocument",
    "MaterializedScanData",
    "MotFieldRequest",
    "MotFieldResult",
    "OccupancyScanRequest",
    "OccupancyArtifactRef",
    "PlanDescriptor",
    "PreparedPulseExecution",
    "PulseFacade",
    "PulseRunDescriptor",
    "PulseRunObservation",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "ReadoutFacade",
    "ReadoutDurationFidelityRequest",
    "ReadoutModelKind",
    "ScanArtifactRef",
    "ScanPointTable",
    "ScanRequest",
    "SitemapCalibrationRequest",
    "SitemapCalibrationFailed",
    "SitemapCalibrationInterrupted",
    "TemperatureReleaseRecaptureRequest",
]
