"""Notebook-first composition facade with no public raw hardware graph."""

from __future__ import annotations

import threading
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from Zou_lab_control.neutral_atom.device_catalog import (
    DeviceRef,
    DeviceCatalogView,
    _catalog_from_device_set,
)
from Zou_lab_control.neutral_atom.devices import (
    load_devices,
    resolve_connect_config,
)
from zlc_data import (
    AxisId,
    BlockId,
    CommittedTransform,
    FitNumericPolicy,
    FitParameterConstraint,
    FitResultArtifactRef,
    FitSpec,
    fit_spec_for,
)
from zlc_neutral_atom.artifacts import (
    AdmittedCaptureFitResult,
    CaptureArtifact,
    CaptureArtifactRef,
    CaptureFitResultRepository,
    CaptureRepository,
    FitExecution,
    compile_capture_artifact_pipeline,
)
from zlc_neutral_atom.readout.calibration import ResolvedCalibration
from zlc_neutral_atom.readout.calibration_reference import CalibrationArtifactRef
from zlc_neutral_atom.readout.calibration_repository import CalibrationRepository
from zlc_neutral_atom.readout.contracts import ReadoutBindingKey
from zlc_neutral_atom.runtime import (
    DatasetMaterializerSpec,
    MinimalPipelineSpec,
    PipelineMemoryProfile,
    RunHandle,
    estimate_pipeline_peak_bytes,
)
from zlc_neutral_atom.timing.capture import TriggeredCaptureSpec
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    load_pulse_document,
)
from zlc_storage import canonical_text as _text
from zlc_storage import positive_integer as _positive_int
from zlc_workbench._triggered_camera import (
    _TriggeredCameraLayout,
    _bind_triggered_camera_acquisition,
)
from zlc_workbench.legacy_neutral_atom import LegacyNeutralAtomRuntime

class _ResourceCleanupError(RuntimeError):
    """Python-3.9-compatible report retaining every ordinary cleanup failure."""

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
        canonical_layout = _TriggeredCameraLayout(
            AxisId("capture.repeat"),
            AxisId("capture.scan_row"),
            AxisId("capture.readout_event"),
            self.repeat_count,
            self.readout_events_per_repeat,
            self.within_point_grouping,
        )
        object.__setattr__(
            self,
            "within_point_grouping",
            canonical_layout.within_point_grouping,
        )
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
        timeout = float(self.timeout_seconds)
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_seconds must be finite and positive")
        object.__setattr__(self, "timeout_seconds", timeout)


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
    runtime: LegacyNeutralAtomRuntime
    devices: object
    capture_repository: CaptureRepository
    calibration_repository: CalibrationRepository
    fit_repository: CaptureFitResultRepository
    catalog: DeviceCatalogView
    operation_lock: threading.RLock
    closed: bool = False


_AUTHORITY_LOCK = threading.RLock()
_AUTHORITIES: dict[object, _ExperimentServices] = {}


def _register(services: _ExperimentServices) -> object:
    token = object()
    with _AUTHORITY_LOCK:
        _AUTHORITIES[token] = services
    return token


def _services(token: object) -> _ExperimentServices:
    with _AUTHORITY_LOCK:
        services = _AUTHORITIES.get(token)
    if services is None or services.closed:
        raise RuntimeError("Experiment is closed")
    return services


class TimingFacade:
    __slots__ = ("_token",)

    def __init__(self, token: object) -> None:
        self._token = token

    @property
    def target(self) -> TimingTargetDescriptor:
        services = _services(self._token)
        with services.operation_lock:
            port = services.runtime.bind_sequencer_port()
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
        services = _services(self._token)
        info = services.catalog.require(key.value)
        if info.domain != "camera":
            raise ValueError(f"readout binding {key.value!r} is not a camera")
        return ReadoutFacade(self._token, key)

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
        services = _services(self._token)
        document = pulse if isinstance(pulse, PulseDocument) else load_pulse_document(pulse)
        if self._binding is not None:
            if camera_role is not None and camera_role != self._binding.value:
                raise ValueError("bound readout facade cannot target another camera")
            camera_role = self._binding.value
        return CaptureRequest(
            document,
            execution_form,
            services.catalog.require(_resolve_role(
                services.catalog,
                camera_role,
                "camera",
                ("readout", "camera"),
            )).ref,
            services.catalog.require(_resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )).ref,
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
        return _services(self._token).capture_repository.load(reference)

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration:
        services = _services(self._token)
        return services.calibration_repository.admit(
            reference,
            services.capture_repository,
        )


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
        services = _services(self._authority_token)
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
        reference: FitResultArtifactRef,
    ) -> AdmittedCaptureFitResult:
        services = _services(self._authority_token)
        return services.fit_repository.load(
            reference,
            services.capture_repository,
        )

    def close(self) -> None:
        with _AUTHORITY_LOCK:
            services = _AUTHORITIES.get(self._authority_token)
        if services is None or services.closed:
            return
        failures: list[Exception] = []
        with services.operation_lock:
            if services.closed:
                return
            shutdown = services.runtime.shutdown(timeout=2.0)
            if not shutdown:
                raise RuntimeError(
                    "Experiment close is waiting for an active Run to terminate"
                )
            failures.extend(
                _cleanup_failures(
                    services.devices.close,
                    services.fit_repository.close,
                    services.calibration_repository.close,
                    services.capture_repository.close,
                )
            )
            services.closed = True
        with _AUTHORITY_LOCK:
            _AUTHORITIES.pop(self._authority_token, None)
        if failures:
            raise _ResourceCleanupError(
                "Experiment close failed",
                tuple(failures),
            )

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


def _compile(token: object, request: CaptureRequest):
    if not isinstance(request, CaptureRequest):
        raise TypeError("Experiment only accepts declarative CaptureRequest values")
    services = _services(token)
    with services.operation_lock:
        binding = _bind_triggered_camera_acquisition(
            services.runtime,
            services.catalog,
            pulse_document=request.pulse_document,
            execution_form=request.execution_form,
            camera_ref=request.camera_ref,
            sequencer_ref=request.sequencer_ref,
            trigger_channel=request.trigger_channel,
            layout=_TriggeredCameraLayout(
                AxisId("capture.repeat"),
                AxisId("capture.scan_row"),
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
            DatasetMaterializerSpec(
                BlockId(f"capture-{binding.compiled_artifact.fingerprint[:20]}"),
                PipelineMemoryProfile(
                    request.pipeline_memory_limit_bytes
                ),
            ),
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


def _start(token: object, request: CaptureRequest) -> RunHandle:
    services = _services(token)
    with services.operation_lock:
        plan, _descriptor = _compile(token, request)
        return services.runtime.controller.start(plan)


def _run(token: object, request: CaptureRequest) -> CaptureArtifactRef:
    return _start(token, request).result()


def connect(
    config: str | Path | Mapping[str, object] = "virtual",
    *,
    repository: str | Path,
    name: str = "neutral_atom",
    asset_map=None,
    safety_journal_path: str | Path | None = None,
    open_devices: bool = False,
    trap_array: dict | None = None,
    sitemap: dict | None = None,
    camera: dict | None = None,
    sequencer: dict | None = None,
    **virtual_params,
) -> Experiment:
    """Compose one notebook Experiment; raw devices remain authority-private."""

    if not isinstance(repository, (str, Path)):
        raise TypeError("repository must be an explicit experiment workspace root")
    repository_root = Path(repository).expanduser().resolve()
    capture_repository = None
    calibration_repository = None
    fit_repository = None
    devices = None
    runtime = None
    try:
        capture_repository = CaptureRepository(repository_root / "captures")
        calibration_repository = CalibrationRepository(
            repository_root / "calibrations"
        )
        fit_repository = CaptureFitResultRepository(repository_root / "fits")
        device_config, device_overrides, _defaults = resolve_connect_config(
            config,
            trap_array=trap_array,
            sitemap=sitemap,
            camera=camera,
            sequencer=sequencer,
            params=virtual_params,
        )
        devices = load_devices(
            device_config,
            overrides=device_overrides,
            open_devices=False,
        )
        runtime = LegacyNeutralAtomRuntime(
            devices,
            asset_map=asset_map,
            safety_journal_path=safety_journal_path,
        )
        if open_devices:
            runtime.ensure_device_set_connections(devices)
        installation_id = f"installation-{runtime.asset_map.revision[:20]}"
        catalog = _catalog_from_device_set(
            devices,
            installation_id=installation_id,
            installation_generation=1,
            installation_state_revision=1,
            revision=1,
        )
        services = _ExperimentServices(
            runtime,
            devices,
            capture_repository,
            calibration_repository,
            fit_repository,
            catalog,
            threading.RLock(),
        )
        token = _register(services)
        return Experiment(token, name=name, device_catalog=catalog)
    except BaseException as error:
        failures = _cleanup_failures(
            (
                None
                if runtime is None
                else lambda: _require_runtime_shutdown(runtime, timeout=2.0)
            ),
            None if devices is None else devices.close,
            None if fit_repository is None else fit_repository.close,
            None if calibration_repository is None else calibration_repository.close,
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
    "CaptureRequest",
    "connect",
    "Experiment",
    "FitExecution",
    "PlanDescriptor",
    "ReadoutFacade",
    "TimingFacade",
    "TimingTargetDescriptor",
]
