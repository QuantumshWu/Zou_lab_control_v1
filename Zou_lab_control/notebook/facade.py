"""Notebook-first composition facade with no public raw hardware graph."""

from __future__ import annotations

import threading
import math
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, TYPE_CHECKING

from zlc_neutral_atom.installation import (
    DeviceRef,
    DeviceCatalogView,
)
from zlc_data import (
    READOUT_EVENT,
    AxisId,
    BlockId,
    CommittedTransform,
    FitNumericPolicy,
    FitParameterConstraint,
    FitSpec,
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
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.readout.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationAnalysisRequest,
    GridOrder,
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef
from zlc_neutral_atom.readout.contracts import (
    CalibrationCaptureLayout,
    ReadoutBindingKey,
)
from zlc_neutral_atom.runtime.pipeline import (
    estimate_pipeline_peak_bytes,
    MinimalPipelineSpec,
)
from zlc_neutral_atom.runtime.run import RunHandle
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    load_pulse_document,
)
from zlc_storage import canonical_text as _text
from zlc_storage import durable_mkdir
from zlc_storage import positive_integer as _positive_int
from zlc_storage import positive_real as _positive_real

if TYPE_CHECKING:
    from zlc_neutral_atom.readout.analysis import CalibrationReport
    from zlc_neutral_atom.readout.calibration_repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.readout.occupancy import ResolvedOccupancy
    from zlc_neutral_atom.readout.occupancy_repository import OccupancyRepository


_DEFAULT_CALIBRATION_MEMORY_LIMIT_BYTES = 512 << 20
_DEFAULT_CALIBRATION_TIMEOUT_SECONDS = 300.0
_DEFAULT_OCCUPANCY_MEMORY_LIMIT_BYTES = 512 << 20
_DEFAULT_OCCUPANCY_TIMEOUT_SECONDS = 300.0


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


@dataclass(frozen=True)
class CaptureRequest:
    pulse_document: PulseDocument
    execution_form: PulseExecutionForm
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    trigger_channel: str | None = None
    repeat_count: int = 1
    readout_events_per_repeat: int | None = None
    within_point_grouping: tuple[tuple[int, int], ...] | None = None
    transport_memory_limit_bytes: int = 64 << 20
    pipeline_memory_limit_bytes: int = 256 << 20
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            raise ValueError("CaptureRequest requires a finite pulse execution form")
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if self.trigger_channel is not None:
            _text(self.trigger_channel, "trigger_channel")
        object.__setattr__(
            self,
            "repeat_count",
            _positive_int(self.repeat_count, "repeat_count"),
        )
        if self.readout_events_per_repeat is not None:
            object.__setattr__(
                self,
                "readout_events_per_repeat",
                _positive_int(
                    self.readout_events_per_repeat,
                    "readout_events_per_repeat",
                ),
            )
        if self.within_point_grouping is not None:
            try:
                grouping = tuple(
                    tuple(pair) for pair in self.within_point_grouping
                )
            except TypeError as exc:
                raise TypeError(
                    "within_point_grouping must be an iterable of pairs"
                ) from exc
            object.__setattr__(self, "within_point_grouping", grouping)
        object.__setattr__(
            self,
            "transport_memory_limit_bytes",
            _positive_int(
                self.transport_memory_limit_bytes,
                "transport_memory_limit_bytes",
            ),
        )
        object.__setattr__(
            self,
            "pipeline_memory_limit_bytes",
            _positive_int(
                self.pipeline_memory_limit_bytes,
                "pipeline_memory_limit_bytes",
            ),
        )
        object.__setattr__(
            self,
            "timeout_seconds",
            _positive_real(self.timeout_seconds, "timeout_seconds"),
        )


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


@dataclass(frozen=True)
class TimingTargetDescriptor:
    target: PulseTarget
    clock_hz: float
    geometry_fingerprint: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        if not math.isfinite(float(self.clock_hz)) or float(self.clock_hz) <= 0:
            raise ValueError("clock_hz must be finite and positive")
        object.__setattr__(self, "clock_hz", float(self.clock_hz))
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")

    @property
    def time_step_ns(self) -> float:
        return 1e9 / self.clock_hz


@dataclass(frozen=True)
class PlanDescriptor:
    name: str
    camera_role: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    trigger_channel: str
    expected_frames: int
    output_shape: tuple[int, ...]
    output_schema_fingerprint: str
    compiled_pulse_digest: str
    resource_claims: tuple[str, ...]
    estimated_peak_bytes: int


@dataclass
class _ExperimentServices:
    runtime: object
    capture_repository: CaptureRepository
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


class TimingFacade:
    __slots__ = ("_token",)

    def __init__(self, token: object) -> None:
        self._token = token

    @property
    def target(self) -> TimingTargetDescriptor:
        with _service_guard(self._token) as services:
            role = _resolve_role(
                services.catalog,
                None,
                "sequencer",
                ("sequencer",),
            )
            port = services.runtime.pulse_port(services.catalog.require(role).ref)
            capability = port.capability
            return TimingTargetDescriptor(
                capability.target,
                capability.clock_hz,
                capability.geometry_fingerprint,
            )


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
            if self._binding is not None:
                if camera_role is not None and camera_role != self._binding.value:
                    raise ValueError("bound readout facade cannot target another camera")
                camera_role = self._binding.value
            return CaptureRequest(
                document,
                execution_form,
                services.catalog.require(
                    _resolve_role(
                        services.catalog,
                        camera_role,
                        "camera",
                        ("readout", "camera"),
                    )
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


class Experiment:
    """Public notebook root containing values, requests, and narrow facades only."""

    __slots__ = ("_authority_token", "name", "device_catalog", "readout", "timing")

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
        self.timing = TimingFacade(authority_token)

    def start(self, request: CaptureRequest) -> RunHandle:
        return _start(self._authority_token, request)

    def run(self, request: CaptureRequest) -> CaptureArtifactRef:
        return _run(self._authority_token, request)

    def inspect(self, request: CaptureRequest) -> PlanDescriptor:
        _plan, descriptor = _compile(self._authority_token, request)
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


def _compile_for_services(
    services: _ExperimentServices,
    request: CaptureRequest,
):
    # Concrete binding is composition-private and imported only when a request
    # is compiled.  Importing the notebook value/facade surface never constructs
    # adapters or advertises a drive-capable Port.
    from zlc_neutral_atom.bootstrap._triggered_capture import (
        bind_triggered_camera_acquisition,
        TriggeredCameraLayout,
    )

    if not isinstance(request, CaptureRequest):
        raise TypeError("Experiment only accepts declarative CaptureRequest values")
    binding = bind_triggered_camera_acquisition(
        services.runtime.pulse_port(request.sequencer_ref),
        services.runtime.camera_port(request.camera_ref),
        pulse_document=request.pulse_document,
        execution_form=request.execution_form,
        trigger_channel=request.trigger_channel,
        layout=TriggeredCameraLayout(
            AxisId("capture.repeat"),
            AxisId("capture.scan_row_ordinal"),
            AxisId("capture.readout_event"),
            request.repeat_count,
            request.readout_events_per_repeat,
            request.within_point_grouping,
        ),
        transport_memory_limit_bytes=request.transport_memory_limit_bytes,
    )
    pipeline = MinimalPipelineSpec(
        f"Capture {binding.pulse_request.document.name}",
        binding.measurement,
        BlockId(f"capture-{binding.compiled_artifact.fingerprint[:20]}"),
        request.pipeline_memory_limit_bytes,
        timeout_seconds=request.timeout_seconds,
    )
    triggered = TriggeredCaptureSpec(
        pipeline,
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        binding.cell_plan,
    )
    plan = compile_capture_artifact_pipeline(
        triggered,
        services.capture_repository,
    )
    descriptor = PlanDescriptor(
        plan.name,
        request.camera_ref.role,
        request.sequencer_ref.role,
        request.execution_form,
        binding.trigger_channel,
        binding.expected_frames,
        binding.measurement.capture_contract.dataset_schema.physical_shape,
        binding.measurement.capture_contract.dataset_schema.fingerprint,
        binding.compiled_artifact.fingerprint,
        tuple(str(claim.key) for claim in plan.resource_claims),
        estimate_pipeline_peak_bytes(pipeline),
    )
    return plan, descriptor


def _compile(token: object, request: CaptureRequest):
    with _service_guard(token) as services:
        return _compile_for_services(services, request)


def _start(token: object, request: CaptureRequest) -> RunHandle:
    with _service_guard(token) as services:
        plan, _descriptor = _compile_for_services(services, request)
        return services.runtime.start(plan)


def _run(token: object, request: CaptureRequest) -> CaptureArtifactRef:
    with _service_guard(token) as services:
        plan, _descriptor = _compile_for_services(services, request)
        handle = services.runtime.start(plan)
        runtime = services.runtime
    return runtime.wait(handle)


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
    fit_repository = None
    runtime = None
    try:
        capture_repository = CaptureRepository(repository_root / "captures")
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
    "OccupancyArtifactRef",
    "PlanDescriptor",
    "ReadoutFacade",
    "ReadoutModelKind",
    "TimingFacade",
    "TimingTargetDescriptor",
]
