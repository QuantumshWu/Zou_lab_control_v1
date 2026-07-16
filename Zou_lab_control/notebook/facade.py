"""Notebook-first composition facade with no public raw hardware graph."""

from __future__ import annotations

import threading
import math
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Mapping, TYPE_CHECKING
from uuid import uuid4

from zlc_neutral_atom.installation import (
    DeviceRef,
    DeviceCatalogView,
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
    DataTransformSpec,
    FitNumericPolicy,
    FitParameterConstraint,
    FitSpec,
    Selection,
    commit_transform,
    fit_spec_for,
)
from zlc_neutral_atom.artifacts import (
    AdmittedCaptureFitResult,
    CaptureArtifact,
    CaptureArtifactRef,
    CaptureFitResultRepository,
    CaptureFitResultArtifactRef,
    CaptureRepository,
    FitExecution,
)
from zlc_neutral_atom.capture_application import (
    CAPTURE_READOUT_EVENT_AXIS_ID,
    CaptureRequest,
    PlanDescriptor,
    PreparedFiniteCapture,
    bind_finite_capture_spec,
    prepare_finite_capture,
)
from zlc_neutral_atom.pulse_application import (
    PreparedPulseExecution,
    PulseRunDescriptor,
    PulseRunRequest,
    PulseRunResult,
    PulseTargetDescriptor,
    prepare_pulse_execution,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationAnalysisRequest,
    GridOrder,
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
    calibration_artifact_ref_to_tree,
)
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_neutral_atom.readout.contracts import (
    CalibrationCaptureLayout,
    ReadoutBindingKey,
)
from zlc_neutral_atom.readout.sitemap import SitemapAcquisitionProfile
from zlc_neutral_atom.scan import (
    MaterializedScanData,
    ScanPointTable,
    bind_scan_output_contract,
)
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_neutral_atom.scan.application import (
    PreparedOccupancyScan,
    compile_direct_scan_artifact_plan,
    compile_occupancy_scan_artifact_plan,
)
from zlc_neutral_atom.readout.occupancy import (
    OccupancyStreamProcessorSpec,
    resolve_occupancy_stream_schema,
)
from zlc_neutral_atom.readout.occupancy_pipeline import OccupancyPipelineSpec
from zlc_neutral_atom.runtime.streams import StreamId
from zlc_neutral_atom.runtime.dataset import dataset_storage_nbytes
from zlc_neutral_atom.timing.occupancy import TriggeredOccupancySpec
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    bind_pulse_document_target,
    expand_autonomous_scan_repeats,
    require_autonomous_scan_resident_capacity,
    resolve_api_parameters,
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
    from zlc_neutral_atom.readout.analysis import CalibrationReport
    from zlc_neutral_atom.readout.calibration_repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.readout.occupancy import ResolvedOccupancy
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository
    from zlc_neutral_atom.scan.repository import ScanArtifact, ScanRepository


_DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES = 512 << 20
_DEFAULT_CALIBRATION_TIMEOUT_SECONDS = 300.0
_DEFAULT_OCCUPANCY_MEMORY_LIMIT_BYTES = 512 << 20
_DEFAULT_OCCUPANCY_TIMEOUT_SECONDS = 300.0
_DEFAULT_SCAN_MATERIALIZATION_MEMORY_LIMIT_BYTES = 512 << 20
_DEFAULT_FIGURE_MEMORY_LIMIT_BYTES = 512 << 20
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
        raise RuntimeError("runtime did not terminate within the cleanup deadline")


def _validate_scan_request_fields(
    pulse_document: PulseDocument,
    camera_ref: DeviceRef,
    sequencer_ref: DeviceRef,
    trigger_channel: str | None,
    output_transform_spec: DataTransformSpec | None,
    transport_memory_limit_bytes: int,
    memory_limit_bytes: int,
    timeout_seconds: float,
) -> tuple[int, int, float]:
    if not isinstance(pulse_document, PulseDocument):
        raise TypeError("pulse_document must be PulseDocument")
    if pulse_document.scan_table is None:
        raise ValueError("scan request requires a frozen PulseDocument scan table")
    if pulse_document.api_parameters:
        raise ValueError(
            "scan request requires a PulseDocument with every whole-run API "
            "parameter explicitly resolved"
        )
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
    return (
        _positive_int(
            transport_memory_limit_bytes,
            "transport_memory_limit_bytes",
        ),
        _positive_int(memory_limit_bytes, "memory_limit_bytes"),
        _positive_real(timeout_seconds, "timeout_seconds"),
    )


def _resolve_scan_fixed_api(
    document: PulseDocument,
    values: Mapping[str, int | float] | None,
) -> PulseDocument:
    """Freeze whole-run API constants before a SCAN_SLOT request exists.

    These values are constants for the complete autonomous table.  They are
    deliberately resolved out of the PulseDocument here; a future API-slot
    segmented sweep is a different request and cannot reuse this path.
    """

    supplied = {} if values is None else dict(values)
    resolved = resolve_api_parameters(document, supplied)
    if resolved.api_parameters:
        missing = tuple(item.parameter_id for item in resolved.api_parameters)
        raise ValueError(
            "SCAN_SLOT requires explicit whole-run values for every API parameter; "
            f"missing={missing}"
        )
    return resolved


@dataclass(frozen=True)
class ScanRequest:
    """Freeze one autonomous SCAN_SLOT table and its direct-camera y intent."""

    pulse_document: PulseDocument
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None
    transport_memory_limit_bytes: int = 64 << 20
    memory_limit_bytes: int = 512 << 20
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        transport, memory, timeout = _validate_scan_request_fields(
            self.pulse_document,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
            self.transport_memory_limit_bytes,
            self.memory_limit_bytes,
            self.timeout_seconds,
        )
        object.__setattr__(self, "transport_memory_limit_bytes", transport)
        object.__setattr__(self, "memory_limit_bytes", memory)
        object.__setattr__(self, "timeout_seconds", timeout)


@dataclass(frozen=True)
class OccupancyScanRequest:
    """Freeze one camera→occupancy exact processor as multidimensional scan y."""

    pulse_document: PulseDocument
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    trigger_channel: str | None = None
    output_transform_spec: DataTransformSpec | None = None
    transport_memory_limit_bytes: int = 64 << 20
    memory_limit_bytes: int = 512 << 20
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        transport, memory, timeout = _validate_scan_request_fields(
            self.pulse_document,
            self.camera_ref,
            self.sequencer_ref,
            self.trigger_channel,
            self.output_transform_spec,
            self.transport_memory_limit_bytes,
            self.memory_limit_bytes,
            self.timeout_seconds,
        )
        object.__setattr__(self, "transport_memory_limit_bytes", transport)
        object.__setattr__(self, "memory_limit_bytes", memory)
        object.__setattr__(self, "timeout_seconds", timeout)

@dataclass(frozen=True)
class CalibrationArtifactRequest:
    """Freeze one committed capture, its binding, and calibration intent."""

    source_capture_ref: CaptureArtifactRef
    readout_binding: ReadoutBindingKey
    analysis: CalibrationAnalysisRequest
    memory_limit_bytes: int = _DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES
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
            "memory_limit_bytes",
            _positive_int(self.memory_limit_bytes, "memory_limit_bytes"),
        )
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
    memory_limit_bytes: int = _DEFAULT_OCCUPANCY_MEMORY_LIMIT_BYTES
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
            "memory_limit_bytes",
            _positive_int(self.memory_limit_bytes, "memory_limit_bytes"),
        )
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
    fit_repository: CaptureFitResultRepository
    catalog: DeviceCatalogView
    operation_lock: threading.RLock
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
                capability.target,
                capability.clock_hz,
                capability.geometry_fingerprint,
                capability.resident_scan_point_capacity,
            )

    def request(
        self,
        document: PulseDocument,
        execution_form: PulseExecutionForm = PulseExecutionForm.STATIC_ONCE,
        *,
        sequencer_role: str | None = None,
        timeout_seconds: float | None = None,
    ) -> PulseRunRequest:
        with _service_guard(self._token) as services:
            role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            reference = services.catalog.require(role).ref
        timeout = (
            None
            if execution_form is PulseExecutionForm.CONTINUOUS_MONITOR
            else 30.0 if timeout_seconds is None else timeout_seconds
        )
        if (
            execution_form is PulseExecutionForm.CONTINUOUS_MONITOR
            and timeout_seconds is not None
        ):
            raise ValueError("continuous pulse execution does not accept a timeout")
        return PulseRunRequest(document, execution_form, reference, timeout)

    def inspect(self, request: PulseRunRequest) -> PulseRunDescriptor:
        with _service_guard(self._token) as services:
            return _prepare_pulse_for_services(services, request).descriptor

    def start(self, request: PulseRunRequest) -> RunHandle:
        return _start_pulse(self._token, request)

    def run(self, request: PulseRunRequest) -> PulseRunResult:
        if request.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
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
        transport_memory_limit_bytes: int = 64 << 20,
        pipeline_memory_limit_bytes: int = 256 << 20,
        timeout_seconds: float = 30.0,
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
                transport_memory_limit_bytes,
                pipeline_memory_limit_bytes,
                timeout_seconds,
            )

    def capture(self, pulse: PulseDocument | str | Path, **kwargs) -> CaptureArtifactRef:
        return _run(self._token, self.capture_request(pulse, **kwargs))

    def start_capture(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return _start(self._token, self.capture_request(pulse, **kwargs))

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        with _service_guard(self._token) as services:
            return services.capture_repository.load(reference)

    def scan_request(
        self,
        pulse: PulseDocument | str | Path,
        *,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
        api_values: Mapping[str, int | float] | None = None,
        output_transform_spec: DataTransformSpec | None = None,
        transport_memory_limit_bytes: int = 64 << 20,
        memory_limit_bytes: int = 512 << 20,
        timeout_seconds: float = 30.0,
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
            document = _resolve_scan_fixed_api(document, api_values)
            camera_role = self._resolve_camera_role(services, camera_role)
            return ScanRequest(
                pulse_document=document,
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
                transport_memory_limit_bytes=transport_memory_limit_bytes,
                memory_limit_bytes=memory_limit_bytes,
                timeout_seconds=timeout_seconds,
            )

    def scan(self, pulse: PulseDocument | str | Path, **kwargs) -> ScanArtifactRef:
        return _run_scan(self._token, self.scan_request(pulse, **kwargs))

    def start_scan(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return _start_scan(self._token, self.scan_request(pulse, **kwargs))

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
        transport_memory_limit_bytes: int = 64 << 20,
        memory_limit_bytes: int = 512 << 20,
        timeout_seconds: float = 30.0,
    ) -> OccupancyScanRequest:
        """Build the first external Measurement→Processor SCAN_SLOT request."""

        with _service_guard(self._token) as services:
            document = (
                pulse
                if isinstance(pulse, PulseDocument)
                else load_pulse_document(pulse)
            )
            document = _resolve_scan_fixed_api(document, api_values)
            camera_role = self._resolve_camera_role(services, camera_role)
            sequencer_role = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            return OccupancyScanRequest(
                pulse_document=document,
                camera_ref=services.catalog.require(camera_role).ref,
                sequencer_ref=services.catalog.require(sequencer_role).ref,
                calibration_ref=calibration_ref,
                model_kind=model_kind,
                trigger_channel=trigger_channel,
                output_transform_spec=output_transform_spec,
                transport_memory_limit_bytes=transport_memory_limit_bytes,
                memory_limit_bytes=memory_limit_bytes,
                timeout_seconds=timeout_seconds,
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

    def load_scan(self, reference: ScanArtifactRef) -> "ScanArtifact":
        with _service_guard(self._token) as services:
            return services.scan_repository.admit(reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
        *,
        memory_limit_bytes: int = (
            _DEFAULT_SCAN_MATERIALIZATION_MEMORY_LIMIT_BYTES
        ),
    ) -> MaterializedScanData:
        """Load the canonical scan dataset under one source/transform peak cap."""

        with _service_guard(self._token) as services:
            return services.scan_repository.materialize(
                reference,
                memory_limit_bytes=_positive_int(
                    memory_limit_bytes,
                    "memory_limit_bytes",
                ),
            )

    def sitemap(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
        transport_memory_limit_bytes: int = 64 << 20,
        capture_pipeline_memory_limit_bytes: int = 256 << 20,
        calibration_memory_limit_bytes: int = _DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES,
        capture_timeout_seconds: float = 30.0,
        calibration_timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ) -> CalibrationArtifactRef:
        """Capture and commit one installation-qualified site-map calibration.

        ``frames`` is the number of complete reference/readout/reference groups,
        not the total camera-frame count.  The hardware repeats each complete
        group.  This method composes two ordinary Runs in order; it is not a
        child-plan engine or hidden current-calibration slot.  Advanced/custom
        calibration remains available through ``calibration_request``.
        """

        repeat_groups = _positive_int(frames, "frames")
        transport_memory = _positive_int(
            transport_memory_limit_bytes,
            "transport_memory_limit_bytes",
        )
        capture_pipeline_memory = _positive_int(
            capture_pipeline_memory_limit_bytes,
            "capture_pipeline_memory_limit_bytes",
        )
        calibration_memory = _positive_int(
            calibration_memory_limit_bytes,
            "calibration_memory_limit_bytes",
        )
        capture_timeout = _positive_real(
            capture_timeout_seconds,
            "capture_timeout_seconds",
        )
        calibration_timeout = _positive_real(
            calibration_timeout_seconds,
            "calibration_timeout_seconds",
        )
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
        document = profile.document_for_repeats(repeat_groups)
        grouping = profile.repeat_major_grouping(repeat_groups)
        analysis = profile.analysis_request(CAPTURE_READOUT_EVENT_AXIS_ID)
        capture_request = self.capture_request(
            document,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            camera_role=selected_camera,
            sequencer_role=profile.sequencer_role,
            trigger_channel=profile.trigger_channel,
            repeat_count=repeat_groups,
            readout_events_per_repeat=profile.event_count,
            within_point_grouping=grouping,
            transport_memory_limit_bytes=transport_memory,
            pipeline_memory_limit_bytes=capture_pipeline_memory,
            timeout_seconds=capture_timeout,
        )
        source = _run(self._token, capture_request)
        try:
            request = self.calibration_request(
                source,
                analysis,
                memory_limit_bytes=calibration_memory,
                timeout_seconds=calibration_timeout,
            )
            return self.calibrate(request)
        except KeyboardInterrupt as error:
            raise SitemapCalibrationInterrupted(source) from error
        except Exception as error:
            raise SitemapCalibrationFailed(source) from error

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
        *,
        memory_limit_bytes: int = _DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES,
        timeout_seconds: float = _DEFAULT_CALIBRATION_TIMEOUT_SECONDS,
    ) -> CalibrationArtifactRequest:
        """Freeze explicit calibration intent from one FINAL capture inspection."""

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if not isinstance(analysis, CalibrationAnalysisRequest):
            raise TypeError("analysis must be CalibrationAnalysisRequest")
        memory_limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        with _service_guard(self._token) as services:
            inspected = services.capture_repository.inspect_final(
                source,
                memory_limit_bytes=memory_limit,
            )
            binding = inspected.readout_binding
            self._require_binding(binding)
            return CalibrationArtifactRequest(
                source,
                binding,
                analysis,
                memory_limit,
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

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
        *,
        memory_limit_bytes: int = _DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES,
    ) -> ResolvedCalibration:
        memory_limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        with _service_guard(self._token) as services:
            resolved = _calibration_repository(services).admit(
                reference,
                services.capture_repository,
                memory_limit_bytes=memory_limit,
            )
            self._require_binding(resolved.artifact.frame_contract.binding)
            return resolved

    def load_calibration_report(
        self,
        reference: CalibrationArtifactRef,
        *,
        memory_limit_bytes: int = _DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES,
    ) -> "CalibrationReport":
        memory_limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        with _service_guard(self._token) as services:
            repository = _calibration_repository(services)
            self._require_binding(
                repository.inspect_final(
                    reference,
                    memory_limit_bytes=memory_limit,
                ).readout_binding
            )
            return repository.load_report(
                reference,
                memory_limit_bytes=memory_limit,
            )

    def detection_request(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        model_kind: ReadoutModelKind | None = None,
        memory_limit_bytes: int = _DEFAULT_OCCUPANCY_MEMORY_LIMIT_BYTES,
        timeout_seconds: float = _DEFAULT_OCCUPANCY_TIMEOUT_SECONDS,
    ) -> DetectionRequest:
        """Freeze one committed single-event capture for occupancy analysis."""

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if not isinstance(calibration, CalibrationArtifactRef):
            raise TypeError("calibration must be CalibrationArtifactRef")
        if model_kind is not None and not isinstance(model_kind, ReadoutModelKind):
            raise TypeError("model_kind must be ReadoutModelKind or None")
        memory_limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        timeout = _positive_real(timeout_seconds, "timeout_seconds")
        with _service_guard(self._token) as services:
            inspected_source = services.capture_repository.inspect_final(
                source,
                memory_limit_bytes=memory_limit,
            )
            inspected_calibration = _calibration_repository(
                services
            ).inspect_final(
                calibration,
                memory_limit_bytes=memory_limit,
            )
            binding = inspected_source.readout_binding
            if inspected_calibration.readout_binding != binding:
                raise ValueError("capture and calibration name different readout bindings")
            self._require_binding(binding)
            event_axes = tuple(
                axis
                for axis in inspected_source.dataset_schema.point_axes
                if axis.role == READOUT_EVENT
            )
            if len(event_axes) != 1 or event_axes[0].size != 1:
                raise ValueError(
                    "detection requires exactly one singleton READOUT_EVENT axis"
                )
            selected_model = (
                inspected_calibration.default_model_kind
                if model_kind is None
                else model_kind
            )
            if selected_model not in inspected_calibration.model_kinds:
                raise KeyError(selected_model)
            return DetectionRequest(
                source,
                calibration,
                binding,
                event_axes[0].axis_id,
                selected_model,
                memory_limit,
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
        *,
        memory_limit_bytes: int = _DEFAULT_OCCUPANCY_MEMORY_LIMIT_BYTES,
    ) -> "ResolvedOccupancy":
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        memory_limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        with _service_guard(self._token) as services:
            resolved = _occupancy_repository(services).admit(
                reference,
                services.capture_repository,
                _calibration_repository(services),
                memory_limit_bytes=memory_limit,
            )
            self._require_binding(resolved.readout_binding)
            return resolved


def _project_notebook_figure(
    services,
    source,
    *,
    intent,
    selection,
    preferences,
    memory_limit_bytes: int | None,
):
    """Composition-only ref dispatch; frontend never sees a neutral repository."""

    from zlc_frontend.figure import (
        DatasetDescriptor,
        DatasetId,
        FigureDocument,
        FigureLayer,
        ResolvedDataset,
        ResolvedDatasetMap,
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

    fit_result = None
    snapshot = None
    admitted = None
    source_label = "capture"
    if isinstance(source, ScanArtifactRef):
        source_label = "scan"
        if memory_limit_bytes is None:
            schema = services.scan_repository.inspect_final(source).output_schema
        else:
            materialized = services.scan_repository.materialize(
                source,
                memory_limit_bytes=memory_limit_bytes,
            )
            schema = materialized.schema
            snapshot = materialized.snapshot
        source_ref = None
    elif isinstance(source, CaptureArtifactRef):
        source_ref = source
    elif isinstance(source, FitExecution):
        source_ref = source.source_capture_ref
        fit_result = source.result
    elif isinstance(source, CaptureFitResultArtifactRef):
        admitted_fit = services.fit_repository.load(
            source,
            services.capture_repository,
        )
        source_ref = admitted_fit.source_capture_ref
        fit_result = admitted_fit.result
    elif isinstance(source, AdmittedCaptureFitResult):
        source_ref = source.source_capture_ref
        fit_result = source.result
    else:
        raise TypeError(
            "figure source must be ScanArtifactRef, CaptureArtifactRef, "
            "FitExecution, CaptureFitResultArtifactRef, or "
            "AdmittedCaptureFitResult"
        )

    if source_ref is not None:
        admitted = services.capture_repository.admit(source_ref)
        schema = admitted.artifact.frame_source.schema
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
            if SPATIAL_X in roles and SPATIAL_Y in roles:
                resolved_intent = ViewIntent.IMAGE
            elif roles.intersection((SCAN_POINT, SPECTRAL, MONITOR_HISTORY)):
                resolved_intent = ViewIntent.CURVE
            else:
                resolved_intent = ViewIntent.HISTOGRAM
        else:
            resolved_intent = intent
        suggestion = suggest_view(
            schema,
            resolved_intent,
            selection,
            preferences,
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
    if memory_limit_bytes is None:
        return document, None, fit_result
    if snapshot is None:
        assert admitted is not None
        snapshot = admitted.materialize_snapshot(memory_limit_bytes=memory_limit_bytes)
    return (
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        fit_result,
    )


class Experiment:
    """Public notebook root containing values, requests, and narrow facades only."""

    __slots__ = ("_authority_token", "name", "device_catalog", "pulse", "readout")

    def __init__(
        self,
        authority_token: object,
        *,
        name: str,
        device_catalog: DeviceCatalogView,
    ) -> None:
        self._authority_token = authority_token
        self.name = _text(name, "experiment name")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        self.device_catalog = device_catalog
        self.readout = ReadoutFacade(authority_token)
        self.pulse = PulseFacade(authority_token)

    def pulse_gui(
        self,
        document: PulseDocument | None = None,
        *,
        path: str | Path | None = None,
    ):
        """Lazily open the current PulseWorkbench on this Experiment authority."""

        from Zou_lab_control.workbench import open_pulse_workbench

        return open_pulse_workbench(self, document, path=path)

    def scan_gui(self, request: ScanRequest | OccupancyScanRequest):
        """Open the current typed SCAN_SLOT panel for a frozen request."""

        from Zou_lab_control.workbench import open_scan_workbench

        return open_scan_workbench(self, request)

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
        source: CaptureArtifactRef,
        spec: FitSpec | None = None,
        *,
        model: str | None = None,
        committed_transform: CommittedTransform | None = None,
        fit_axis_ids: tuple[AxisId, ...] | None = None,
        constraints: tuple[FitParameterConstraint, ...] = (),
        numeric_policy: FitNumericPolicy | None = None,
    ) -> FitExecution:
        """Fit one committed capture without hiding any axis reduction."""

        if not isinstance(source, CaptureArtifactRef):
            raise TypeError("source must be CaptureArtifactRef")
        if (spec is None) == (model is None):
            raise ValueError("provide exactly one of spec or model")
        with _service_guard(self._authority_token) as services:
            admitted = services.capture_repository.admit(source)
            if spec is None:
                assert model is not None
                spec = fit_spec_for(
                    admitted.artifact.frame_source.schema,
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
            return services.fit_repository.execute(admitted, spec)

    def load_fit(
        self,
        reference: CaptureFitResultArtifactRef,
    ) -> AdmittedCaptureFitResult:
        with _service_guard(self._authority_token) as services:
            return services.fit_repository.load(
                reference,
                services.capture_repository,
            )

    def figure_document(
        self,
        source: (
            ScanArtifactRef
            | CaptureArtifactRef
            | FitExecution
            | CaptureFitResultArtifactRef
            | AdmittedCaptureFitResult
        ),
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
    ) -> "FigureDocument":
        """Project one committed scan/capture/result into a renderer-free document."""

        with _service_guard(self._authority_token) as services:
            document, _datasets, _fit = _project_notebook_figure(
                services,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                memory_limit_bytes=None,
            )
        return document

    def figure(
        self,
        source: (
            ScanArtifactRef
            | CaptureArtifactRef
            | FitExecution
            | CaptureFitResultArtifactRef
            | AdmittedCaptureFitResult
        ),
        *,
        intent: "ViewIntent | None" = None,
        selection: Selection | None = None,
        preferences: "ViewPreferences | None" = None,
        memory_limit_bytes: int = _DEFAULT_FIGURE_MEMORY_LIMIT_BYTES,
    ) -> "DataFigure":
        """Resolve one frozen source and return its optional-render DataFigure."""

        limit = _positive_int(memory_limit_bytes, "memory_limit_bytes")
        with _service_guard(self._authority_token) as services:
            document, datasets, fit_result = _project_notebook_figure(
                services,
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                memory_limit_bytes=limit,
            )
        assert datasets is not None
        from zlc_frontend import DataFigure

        retained_input_bytes = sum(
            dataset_storage_nbytes(entry.snapshot.block.schema)
            for entry in datasets.entries
        )
        evaluation_limit = limit - retained_input_bytes
        if evaluation_limit <= 0:
            raise MemoryError(
                "figure input snapshot leaves no memory for view evaluation"
            )

        return DataFigure(
            document,
            datasets,
            fit_results=({"data": fit_result} if fit_result is not None else None),
            evaluation_memory_limit_bytes=evaluation_limit,
        )

    def close(self) -> None:
        with _AUTHORITY_LOCK:
            services = _AUTHORITIES.get(self._authority_token)
        if services is None:
            return
        with services.operation_lock:
            if services.state == "CLOSED":
                return
            services.state = "CLOSING"
            shutdown = services.runtime.shutdown(timeout=2.0)
            if not shutdown:
                raise RuntimeError(
                    "Experiment close is waiting for an active Run to terminate"
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


def _prepare_pulse_for_services(
    services: _ExperimentServices,
    request: PulseRunRequest,
) -> PreparedPulseExecution:
    return prepare_pulse_execution(
        request,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        start_run=services.runtime.start,
    )


def _start_pulse(token: object, request: PulseRunRequest) -> RunHandle:
    with _service_guard(token) as services:
        return _prepare_pulse_for_services(services, request).start()


def _run_pulse(token: object, request: PulseRunRequest) -> PulseRunResult:
    with _service_guard(token) as services:
        handle = _prepare_pulse_for_services(services, request).start()
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


def _bind_scan_camera(
    services: _ExperimentServices,
    request: ScanRequest | OccupancyScanRequest,
):
    from zlc_neutral_atom.bootstrap._triggered_capture import (
        TriggeredCameraLayout,
        bind_triggered_camera_acquisition,
    )

    if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
        raise TypeError("request must be a current scan request")
    pulse_port = services.runtime.pulse_port(request.sequencer_ref)
    camera_port = services.runtime.camera_port(request.camera_ref)
    logical_document = bind_pulse_document_target(
        request.pulse_document,
        pulse_port.capability.target,
    )
    require_autonomous_scan_resident_capacity(
        logical_document,
        pulse_port.capability.resident_scan_point_capacity,
    )
    execution_document = expand_autonomous_scan_repeats(logical_document)
    point_table = ScanPointTable.from_pulse_document(logical_document)
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
        transport_memory_limit_bytes=request.transport_memory_limit_bytes,
    )
    if (
        binding.compiled_artifact.source_document_digest
        != execution_document.fingerprint
    ):
        raise RuntimeError(
            "compiled scan pulse differs from the repeat-major execution document"
        )
    return logical_document, point_table, binding


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
    logical_document, point_table, binding = _bind_scan_camera(services, request)
    raw_schema = binding.measurement.capture_contract.dataset_schema
    output_contract = _scan_transform(
        raw_schema,
        point_table,
        request.output_transform_spec,
    )
    triggered, descriptor = bind_finite_capture_spec(
        binding=binding,
        block_id=BlockId(
            f"scan-camera-{binding.compiled_artifact.fingerprint[:20]}"
        ),
        camera_ref=request.camera_ref,
        sequencer_ref=request.sequencer_ref,
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        pipeline_memory_limit_bytes=request.memory_limit_bytes,
        timeout_seconds=request.timeout_seconds,
        name_prefix="Direct scan",
    )
    plan = compile_direct_scan_artifact_plan(
        triggered,
        services.scan_repository,
        document=logical_document,
        output_contract=output_contract,
        memory_limit_bytes=request.memory_limit_bytes,
    )
    descriptor = replace(
        descriptor,
        output_shape=output_contract.output_dataset_schema.physical_shape,
        output_schema_fingerprint=output_contract.output_schema_fingerprint,
    )
    return plan, descriptor


def _bind_occupancy_scan_for_services(
    services: _ExperimentServices,
    request: OccupancyScanRequest,
):
    logical_document, point_table, binding = _bind_scan_camera(services, request)
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
        memory_limit_bytes=request.memory_limit_bytes,
    )
    selected_kind = calibration.artifact.select_model(request.model_kind).kind
    identity = canonical_digest(
        {
            "owner": "Zou_lab_control.notebook.occupancy-scan",
            "pulse_document": logical_document.fingerprint,
            "compiled_pulse": binding.compiled_artifact.fingerprint,
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
        f"Occupancy scan {logical_document.name}",
        binding.measurement,
        processor,
        BlockId(f"scan-counts-{identity}"),
        BlockId(f"scan-occupied-{identity}"),
        request.memory_limit_bytes,
        request.timeout_seconds,
    )
    triggered = TriggeredOccupancySpec(
        occupancy,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    return triggered, logical_document, output_contract, source_schema


def _compile_occupancy_scan_for_services(
    services: _ExperimentServices,
    request: OccupancyScanRequest,
):
    triggered, logical_document, output_contract, _source_schema = (
        _bind_occupancy_scan_for_services(services, request)
    )
    return compile_occupancy_scan_artifact_plan(
        triggered,
        services.scan_repository,
        document=logical_document,
        output_contract=output_contract,
        memory_limit_bytes=request.memory_limit_bytes,
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
        triggered, logical_document, output_contract, source_schema = (
            _bind_occupancy_scan_for_services(services, request)
        )

    def start(preview):
        with _service_guard(experiment._authority_token) as services:
            plan = compile_occupancy_scan_artifact_plan(
                triggered,
                services.scan_repository,
                document=logical_document,
                output_contract=output_contract,
                memory_limit_bytes=request.memory_limit_bytes,
                preview=preview,
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
        memory_limit_bytes=request.memory_limit_bytes,
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
        memory_limit_bytes=request.memory_limit_bytes,
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


def connect(
    config: str = "virtual",
    *,
    repository: str | Path,
    name: str = "neutral_atom",
    seed: int | None = 7,
) -> Experiment:
    """Compose one notebook Experiment; raw devices remain authority-private."""

    if not isinstance(config, str):
        raise TypeError("config must name an explicit target backend")
    if config != "virtual":
        raise ValueError(
            "the target composition currently accepts only the explicit "
            "'virtual' backend"
        )
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
        fit_repository = CaptureFitResultRepository(repository_root / "fits")
        from zlc_neutral_atom.bootstrap._installation import (
            create_virtual_installation,
        )

        runtime = create_virtual_installation(
            safety_journal_path=(
                repository_root / ".runtime" / "safety.journal"
            ),
            seed=seed,
        )
        catalog = runtime.device_catalog
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
            operation_lock=threading.RLock(),
        )
        token = object()
        experiment = Experiment(token, name=canonical_name, device_catalog=catalog)
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


__all__ = [
    "AdmittedCaptureFitResult",
    "BackgroundMode",
    "BoxReducer",
    "CalibrationAnalysisRequest",
    "CalibrationArtifactRequest",
    "CalibrationArtifactRef",
    "CalibrationCaptureLayout",
    "CaptureArtifactRef",
    "CaptureFitResultArtifactRef",
    "CaptureRequest",
    "connect",
    "DetectionRequest",
    "Experiment",
    "FitExecution",
    "GridOrder",
    "MaterializedScanData",
    "OccupancyScanRequest",
    "OccupancyArtifactRef",
    "PlanDescriptor",
    "PreparedPulseExecution",
    "PulseFacade",
    "PulseRunDescriptor",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "ReadoutFacade",
    "ReadoutModelKind",
    "ScanArtifactRef",
    "ScanPointTable",
    "ScanRequest",
    "SitemapCalibrationFailed",
    "SitemapCalibrationInterrupted",
]
