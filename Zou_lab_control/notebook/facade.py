"""Notebook-first composition facade with no public raw hardware graph."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterator, Mapping, TYPE_CHECKING

from zlc_neutral_atom.installation import (
    DeviceCatalogView,
)
from zlc_neutral_atom.installation_config import (
    InstallationConfigDocument,
    load_installation_config,
)
from zlc_data import (
    AxisId,
    CommittedTransform,
    DataTransformSpec,
    FitNumericPolicy,
    FitParameterConstraint,
    FitCancelled,
    FitResultBatch,
    FitSpec,
    OwnedSnapshot,
    Selection,
    bind_fit,
    fit_model_catalog,
    fit_spec_for,
    suggest_fit_draft,
)
from zlc_neutral_atom.artifacts import (
    AdmittedFitResult,
    FitExecution,
    FitResultArtifactRef,
    FitResultRepository,
)
from zlc_neutral_atom.capture.artifact import (
    CaptureArtifact,
    CaptureRepository,
)
from zlc_neutral_atom.capture.reference import CaptureArtifactRef
from zlc_neutral_atom.capture.application import (
    CaptureRequest,
    PlanDescriptor,
    PreparedFiniteCapture,
    prepare_finite_capture,
)
from zlc_neutral_atom.logic_nodes.camera_measurement import (
    CameraMeasurementRequest,
    DEFAULT_CAMERA_FRAMES_PER_CYCLE,
    DEFAULT_CAMERA_MEASUREMENT_REPEAT,
    DEFAULT_CAMERA_MEASUREMENT_ROLE,
    DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES,
    PreparedFiniteCameraMeasurement,
    PreparedLiveCameraMeasurement,
    prepare_finite_camera_measurement,
    prepare_live_camera_measurement,
)
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.logic_nodes.mot_field import (
    DEFAULT_MOT_FIELD_CAMERA_ROLE,
    DEFAULT_MOT_FIELD_CENTER_CODE,
    DEFAULT_MOT_FIELD_POINTS,
    DEFAULT_MOT_FIELD_PULSE_PATH,
    DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
    DEFAULT_MOT_FIELD_SPAN_CODE,
    MotFieldRequest,
    MotFieldResult,
    PreparedMotFieldAcquisition,
    build_mot_scan_program,
    prepare_mot_field_acquisition,
)
from zlc_neutral_atom.logic_nodes.mot_field import (
    MotFieldTaskIntent,
    PreparedMotFieldTask,
    prepare_mot_field_task as prepare_mot_field_task_application,
)
from zlc_neutral_atom.devices.sequencer.application import (
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
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    BackgroundMode,
    BoxReducer,
    CalibrationAnalysisRequest,
    GridOrder,
    ReadoutModelKind,
    ResolvedCalibration,
    ThresholdMethod,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.application import (
    CalibrationArtifactRequest,
    build_calibration_artifact_request,
    prepare_calibration_artifact_plan,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.task import (
    CalibrationTaskIntent,
    PreparedCalibrationTask,
    admit_calibration_capture_export,
    admit_calibration_task_output,
    prepare_calibration_task as prepare_calibration_task_application,
    write_calibration_task_outputs as write_calibration_application_outputs,
)
from zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning import (
    AutonomousMeasurementUnavailable,
    GREY_MOLASSES_CAPABILITY_GAP,
    GreyMolassesDetuningApplicationCommand,
    GreyMolassesDetuningIntent,
    GreyMolassesDetuningRequest,
    prepare_grey_molasses_detuning,
    prepare_grey_molasses_detuning_application,
)
from zlc_neutral_atom.logic_nodes.readout.duration_fidelity import (
    PreparedReadoutDurationFidelity,
    ReadoutDurationFidelityApplicationCommand,
    ReadoutDurationFidelityIntent,
    ReadoutDurationFidelityRequest,
    prepare_readout_duration_fidelity,
    prepare_readout_duration_fidelity_application,
)
from zlc_neutral_atom.logic_nodes.release_recapture.application import (
    PreparedReleaseRecapture,
)
from zlc_neutral_atom.logic_nodes.release_recapture.temperature import (
    TemperatureReleaseRecaptureApplicationCommand,
    TemperatureReleaseRecaptureIntent,
    TemperatureReleaseRecaptureRequest,
    prepare_temperature_release_recapture,
    prepare_temperature_release_recapture_application,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import OccupancyArtifactRef
from zlc_neutral_atom.logic_nodes.readout.occupancy.cell import (
    OccupancyCellDomain,
    inspect_occupancy_cell_domain,
    load_exact_occupancy_cell_source,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import CalibrationCaptureLayout
from zlc_neutral_atom.devices.camera.contract import ReadoutBindingKey
from zlc_neutral_atom.logic_nodes.readout.calibration.sitemap import (
    SitemapAcquisitionProfile,
    SitemapCalibrationRequest,
    build_sitemap_analysis_request,
    build_sitemap_calibration_request,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.installation import (
    build_sitemap_acquisition_profile,
)
from zlc_neutral_atom.logic_nodes.pulse_scan import (
    MaterializedScanData,
    ScanPointTable,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.source_binding import (
    PulseScanBoundRequest,
)
from zlc_neutral_atom.logic_nodes.pulse_scan.reference import ScanArtifactRef
from zlc_neutral_atom.logic_nodes.pulse_scan.application import (
    PreparedExactScan,
    prepare_exact_scan,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.application import (
    DetectionRequest,
    build_detection_request,
    prepare_detection_plan,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.processor_application import (
    PreparedOccupancyProcessor,
    OccupancyProcessorRequest,
    prepare_occupancy_processor,
)
from zlc_neutral_atom.devices.sequencer.port import PulseScanProgress
from zlc_neutral_atom.runtime.run import CancelOutcome, RunHandle
from zlc_neutral_atom.runtime.signal_source import SignalEventSource
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    load_pulse_document,
)
from zlc_storage import canonical_text as _text
from zlc_storage import durable_makedirs
from zlc_storage import positive_real as _positive_real
from zlc_storage.paths import resolve_under_project

if TYPE_CHECKING:
    from zlc_frontend import DataFigure
    from zlc_frontend.figure import FigureDocument, ViewIntent, ViewPreferences
    from zlc_neutral_atom.logic_nodes.readout.calibration.analysis import (
        CalibrationComputation,
        CalibrationReport,
    )
    from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
        CalibrationRepository,
    )
    from zlc_neutral_atom.logic_nodes.readout.occupancy.processor import ResolvedOccupancy
    from zlc_neutral_atom.logic_nodes.readout.occupancy.repository import OccupancyRepository
    from zlc_neutral_atom.logic_nodes.pulse_scan.repository import (
        ScanArtifact,
        ScanRepository,
    )


_DEFAULT_FIT_GUI_TIMEOUT_SECONDS = 30.0


def _load_project_pulse(value: PulseDocument | str | Path) -> PulseDocument:
    """Resolve one operator-authored pulse against the project catalog."""

    if isinstance(value, PulseDocument):
        return value
    return load_pulse_document(resolve_under_project(value))


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
    sitemap_profiles: Mapping[str, SitemapAcquisitionProfile]
    camera_signal_association_authorities: Mapping[str, object]
    installation_config: InstallationConfigDocument
    pulse_application: PulseApplicationOwner
    operation_lock: threading.RLock
    fit_operations_drained: threading.Event
    fit_operation_thread_counts: dict[int, int]
    gui_handles: dict[str, object]
    active_fit_operations: int = 0
    state: str = "OPEN"


@contextmanager
def _service_guard(
    services: _ExperimentServices,
) -> Iterator[_ExperimentServices]:
    """Borrow the one Experiment-owned service graph while it is open."""

    if not isinstance(services, _ExperimentServices):
        raise TypeError("services must be _ExperimentServices")
    with services.operation_lock:
        if services.state != "OPEN":
            raise RuntimeError("Experiment is closing or closed")
        yield services


@contextmanager
def _fit_service_guard(
    services: _ExperimentServices,
) -> Iterator[_ExperimentServices]:
    """Keep repositories alive for one long Fit without serializing figures."""

    if not isinstance(services, _ExperimentServices):
        raise TypeError("services must be _ExperimentServices")
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
        from zlc_neutral_atom.logic_nodes.readout.calibration.repository import (
            CalibrationRepository,
        )

        repository = CalibrationRepository(services.calibration_repository_path)
        services.calibration_repository = repository
    return repository


def _render_calibration_report(view):
    """Composition seam from Calibration's physical projection to frontend pixels."""

    from zlc_neutral_atom.logic_nodes.readout.calibration.ui.report_render import (
        render_calibration_report,
    )

    return render_calibration_report(view)


def _occupancy_repository(
    services: _ExperimentServices,
) -> "OccupancyRepository":
    repository = services.occupancy_repository
    if repository is None:
        from zlc_neutral_atom.logic_nodes.readout.occupancy.repository import (
            OccupancyRepository,
        )

        repository = OccupancyRepository(services.occupancy_repository_path)
        services.occupancy_repository = repository
    return repository


class PulseFacade:
    __slots__ = ("_services",)

    def __init__(self, services: _ExperimentServices) -> None:
        if not isinstance(services, _ExperimentServices):
            raise TypeError("services must be _ExperimentServices")
        self._services = services

    @property
    def target(self) -> PulseTargetDescriptor:
        with _service_guard(self._services) as services:
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
        with _service_guard(self._services) as services:
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
        with _service_guard(self._services) as services:
            return services.pulse_application.snapshot()

    def observe_active(self) -> PulseRunObservation | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.observe_active()

    def observe_scan_progress(self) -> PulseScanProgress | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.observe_scan_progress()

    def cancel_active(
        self,
        reason: str = "user requested pulse stop",
    ) -> CancelOutcome | None:
        with _service_guard(self._services) as services:
            return services.pulse_application.cancel_active(reason)

    def inspect(self, request: PulseRunRequest) -> PulseRunDescriptor:
        with _service_guard(self._services) as services:
            return _prepare_pulse_for_services(services, request).descriptor

    def start(self, request: PulseRunRequest) -> RunHandle:
        return _start_pulse(self._services, request)

    def run(self, request: PulseRunRequest) -> PulseRunResult:
        if request.requires_cancellation:
            raise ValueError("continuous pulse execution must be started and cancelled")
        return _run_pulse(self._services, request)


class ReadoutFacade:
    __slots__ = ("_services", "_binding")

    def __init__(
        self,
        services: _ExperimentServices,
        binding: ReadoutBindingKey | None = None,
    ) -> None:
        if not isinstance(services, _ExperimentServices):
            raise TypeError("services must be _ExperimentServices")
        self._services = services
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
        with _service_guard(self._services) as services:
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
            DEFAULT_CAMERA_MEASUREMENT_ROLE,
            ("readout", "camera"),
        )

    def camera_measurement_request(
        self,
        *,
        camera_role: str | None = None,
        repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT,
        history_cycles: int = DEFAULT_CAMERA_MONITOR_HISTORY_CYCLES,
        frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE,
        exposure: float | None = None,
    ) -> CameraMeasurementRequest:
        """Freeze Main's one Camera semantic: 0=live, K=finite."""

        with _service_guard(self._services) as services:
            role = self._resolve_camera_role(services, camera_role)
            camera_ref = services.catalog.require(role).ref
            return CameraMeasurementRequest(
                camera_ref=camera_ref,
                repeat=repeat,
                history_cycles=history_cycles,
                frames_per_cycle=frames_per_cycle,
                exposure_seconds=exposure,
            )

    def prepare_camera_measurement(
        self,
        request: CameraMeasurementRequest,
    ) -> PreparedLiveCameraMeasurement | PreparedFiniteCameraMeasurement:
        """Bind one typed Camera Measurement to the installed runtime.

        This is the public application boundary used by notebooks and GUI
        composition roots alike.  The returned command owns live-vs-finite
        acquisition semantics and its named outputs; callers do not receive
        device services or construct a projection callback.
        """

        with _service_guard(self._services) as services:
            return _prepare_camera_measurement_for_services(services, request)

    def prepare_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        """Bind one finite capture for an application-owned composite command."""

        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        with _service_guard(self._services) as services:
            return _prepare_capture_for_services(services, request)

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
        with _service_guard(self._services) as services:
            document = _load_project_pulse(pulse)
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
        return _run(self._services, self.capture_request(pulse, **kwargs))

    def start_capture(self, pulse: PulseDocument | str | Path, **kwargs) -> RunHandle:
        return _start(self._services, self.capture_request(pulse, **kwargs))

    def load_capture(self, reference: CaptureArtifactRef) -> CaptureArtifact:
        with _service_guard(self._services) as services:
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

        with _service_guard(self._services) as services:
            document = _load_project_pulse(pulse)
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
        with _service_guard(self._services) as services:
            return _prepare_temperature_release_recapture_for_services(
                services,
                request,
            ).start()

    def prepare_temperature_release_recapture_application(
        self,
        intent: TemperatureReleaseRecaptureIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> TemperatureReleaseRecaptureApplicationCommand:
        """Bind the complete Temperature Measurement behind one command."""

        return prepare_temperature_release_recapture_application(
            intent,
            calibration_ref,
            self,
        )

    def temperature_release_recapture(
        self,
        request: TemperatureReleaseRecaptureRequest,
    ):
        with _service_guard(self._services) as services:
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
        """Freeze one camera-rearmed API-slot duration sweep."""

        with _service_guard(self._services) as services:
            document = _load_project_pulse(pulse)
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
        with _service_guard(self._services) as services:
            return _prepare_readout_duration_fidelity_for_services(
                services,
                request,
            ).start()

    def prepare_readout_duration_fidelity_application(
        self,
        intent: ReadoutDurationFidelityIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> ReadoutDurationFidelityApplicationCommand:
        """Bind the complete readout-duration Measurement command."""

        return prepare_readout_duration_fidelity_application(
            intent,
            calibration_ref,
            self,
        )

    def readout_duration_fidelity(
        self,
        request: ReadoutDurationFidelityRequest,
    ):
        with _service_guard(self._services) as services:
            handle = _prepare_readout_duration_fidelity_for_services(
                services,
                request,
            ).start()
            runtime = services.runtime
        return runtime.wait(handle)

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

        with _service_guard(self._services) as services:
            document = _load_project_pulse(pulse)
            camera = self._resolve_camera_role(services, camera_role)
            sequencer = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            available_rf = services.catalog.roles("rf")
            resolved_rf_role = (
                _resolve_role(
                    services.catalog,
                    rf_role,
                    "rf",
                    ("rf",),
                )
                if available_rf
                else rf_role
            )
            return GreyMolassesDetuningRequest(
                document,
                tuple(detuning_gamma),
                trap_off_seconds,
                shots,
                services.catalog.require(camera).ref,
                services.catalog.require(sequencer).ref,
                resolved_rf_role,
                calibration_ref,
                model_kind,
                per_site,
                trigger_channel,
            )

    def start_grey_molasses_detuning(
        self,
        request: GreyMolassesDetuningRequest,
    ) -> RunHandle:
        with _service_guard(self._services) as services:
            return _prepare_grey_molasses_detuning_for_services(
                services,
                request,
            ).start()

    def prepare_grey_molasses_detuning_application(
        self,
        intent: GreyMolassesDetuningIntent,
        calibration_ref: CalibrationArtifactRef,
    ) -> GreyMolassesDetuningApplicationCommand:
        """Bind the complete Grey-molasses Measurement command."""

        return prepare_grey_molasses_detuning_application(
            intent,
            calibration_ref,
            self,
        )

    def materialize_capture(self, reference: CaptureArtifactRef) -> OwnedSnapshot:
        """Load one FINAL capture as its canonical immutable dataset snapshot."""

        if not isinstance(reference, CaptureArtifactRef):
            raise TypeError("reference must be CaptureArtifactRef")
        with _service_guard(self._services) as services:
            return services.capture_repository.materialize_final(reference)

    def prepare_scan_source(
        self,
        request: PulseScanBoundRequest,
        source: SignalEventSource,
        *,
        sequencer_role: str | None = None,
    ) -> PreparedExactScan:
        """Contribute only the installed sequencer to source-neutral PulseScan."""

        if not isinstance(request, PulseScanBoundRequest):
            raise TypeError("request must be PulseScanBoundRequest")
        if not isinstance(source, SignalEventSource):
            raise TypeError("source must implement SignalEventSource")
        with _service_guard(self._services) as services:
            sequencer = _resolve_role(
                services.catalog,
                sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            sequencer_ref = services.catalog.require(sequencer).ref
            return prepare_exact_scan(
                request,
                source,
                pulse_port=services.runtime.pulse_port(sequencer_ref),
                repository=services.scan_repository,
                start_run=services.runtime.start,
            )

    def load_scan(self, reference: ScanArtifactRef) -> "ScanArtifact":
        with _service_guard(self._services) as services:
            return services.scan_repository.admit(reference)

    def materialize_scan(
        self,
        reference: ScanArtifactRef,
    ) -> MaterializedScanData:
        """Load the canonical scan dataset."""

        with _service_guard(self._services) as services:
            return services.scan_repository.materialize(reference)

    def mot_field_request(
        self,
        pulse: PulseDocument | str | Path = DEFAULT_MOT_FIELD_PULSE_PATH,
        *,
        center_x: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        center_y: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        center_z: float = DEFAULT_MOT_FIELD_CENTER_CODE,
        span: float = DEFAULT_MOT_FIELD_SPAN_CODE,
        points: int = DEFAULT_MOT_FIELD_POINTS,
        roi_cx: float | None = None,
        roi_cy: float | None = None,
        roi_radius: float = DEFAULT_MOT_FIELD_ROI_RADIUS_PX,
        camera_role: str | None = None,
        sequencer_role: str | None = None,
        trigger_channel: str | None = None,
    ) -> MotFieldRequest:
        """Freeze one three-axis autonomous MOT scan and its ROI analysis."""

        document = _load_project_pulse(pulse)
        program = build_mot_scan_program(
            document,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
            span=span,
            points=points,
        )
        with _service_guard(self._services) as services:
            camera_role = (
                DEFAULT_MOT_FIELD_CAMERA_ROLE
                if camera_role is None
                else camera_role
            )
            camera_role = _resolve_role(
                services.catalog,
                camera_role,
                "camera",
                (DEFAULT_MOT_FIELD_CAMERA_ROLE,),
            )
            if camera_role != DEFAULT_MOT_FIELD_CAMERA_ROLE:
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

    def prepare_mot_field_acquisition(
        self,
        request: MotFieldRequest,
    ) -> PreparedMotFieldAcquisition:
        """Bind the MOT-owned coupled Camera + autonomous pulse acquisition."""

        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        with _service_guard(self._services) as services:
            return prepare_mot_field_acquisition(
                request,
                pulse_port=services.runtime.pulse_port(request.sequencer_ref),
                camera_port=services.runtime.camera_port(request.camera_ref),
                start_run=services.runtime.start,
            )

    def prepare_mot_field_task(
        self,
        intent: MotFieldTaskIntent,
    ) -> PreparedMotFieldTask:
        """Bind the complete MOT-field Task behind one public command."""

        return prepare_mot_field_task_application(intent, self)

    def prepare_calibration_task(
        self,
        intent: CalibrationTaskIntent,
    ) -> PreparedCalibrationTask:
        """Bind the complete Calibration Task behind one public command."""

        return prepare_calibration_task_application(intent, self)

    def _sitemap_profile(
        self,
        camera_role: str | None,
    ) -> tuple[str, SitemapAcquisitionProfile]:
        """Resolve the explicitly composed sitemap profile for a camera."""

        with _service_guard(self._services) as services:
            selected_camera = self._resolve_camera_role(services, camera_role)
            try:
                profile = services.sitemap_profiles[selected_camera]
            except KeyError as exc:
                raise ValueError(
                    f"camera role {selected_camera!r} has no sitemap profile"
                ) from exc
        if not isinstance(profile, SitemapAcquisitionProfile):
            raise TypeError("experiment composition contains an invalid sitemap profile")
        if profile.readout_binding != ReadoutBindingKey(selected_camera):
            raise ValueError(
                "composed sitemap profile differs from the selected camera"
            )
        return selected_camera, profile

    def sitemap_camera_roles(self) -> tuple[str, ...]:
        """Installed camera roles with a complete live calibration profile."""

        with _service_guard(self._services) as services:
            roles = tuple(services.sitemap_profiles)
            cameras = set(services.catalog.roles("camera"))
        if any(role not in cameras for role in roles):
            raise RuntimeError(
                "installation sitemap capabilities differ from its camera catalog"
            )
        return roles

    def sitemap_analysis_request(
        self,
        *,
        camera_role: str | None = None,
        threshold_method: ThresholdMethod | str = ThresholdMethod.OTSU,
        roi_radius: int | None = None,
    ) -> CalibrationAnalysisRequest:
        """Freeze sitemap analysis intent without manufacturing a live capture."""

        _selected_camera, profile = self._sitemap_profile(camera_role)
        return build_sitemap_analysis_request(
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
    ) -> SitemapCalibrationRequest:
        """Freeze one installation-qualified capture-then-calibration request.

        ``frames`` is the number of complete reference/readout/reference groups,
        not the total camera-frame count.  The hardware repeats each complete
        group.  The returned value contains a complete finite ``CaptureRequest``
        plus the immutable analysis intent for the second ordinary Run.  It
        contains no current/latest calibration pointer.
        """

        selected_camera, profile = self._sitemap_profile(camera_role)

        selected_pulse = None if pulse is None else _load_project_pulse(pulse)
        with _service_guard(self._services) as services:
            camera_ref = services.catalog.require(selected_camera).ref
            sequencer_role = _resolve_role(
                services.catalog,
                profile.sequencer_role,
                "sequencer",
                ("sequencer",),
            )
            sequencer_ref = services.catalog.require(sequencer_role).ref
        return build_sitemap_calibration_request(
            profile,
            camera_ref=camera_ref,
            sequencer_ref=sequencer_ref,
            repeat_groups=frames,
            pulse_document=selected_pulse,
            reference_exposure_s=reference_exposure_s,
            readout_exposure_s=readout_exposure_s,
            threshold_method=threshold_method,
            roi_radius=roi_radius,
        )

    def sitemap(
        self,
        *,
        frames: int = 20,
        camera_role: str | None = None,
    ) -> CalibrationArtifactRef:
        """Capture and commit one installation-qualified site-map calibration.

        This convenience composes two ordinary Runs in order; it is not a
        child-plan engine or hidden current-calibration slot.  Advanced/custom
        calibration remains available through ``calibration_request``.
        """

        sequence = self.sitemap_request(
            frames=frames,
            camera_role=camera_role,
        )
        source = _run(self._services, sequence.capture_request)
        try:
            request = self.calibration_request(
                source,
                sequence.analysis,
            )
            return self.calibrate(request)
        except KeyboardInterrupt as error:
            raise SitemapCalibrationInterrupted(source) from error
        except Exception as error:
            raise SitemapCalibrationFailed(source) from error

    def admit_saved_calibration_capture(
        self,
        source_path: str | Path,
        *,
        expected_camera_role: str,
    ) -> CaptureArtifactRef:
        """Admit one task export through the calibration application owner."""

        binding = ReadoutBindingKey(expected_camera_role)
        self._require_binding(binding)
        with _service_guard(self._services) as services:
            info = services.catalog.require(binding.value)
            if info.domain != "camera":
                raise ValueError(
                    f"saved calibration binding {binding.value!r} is not a camera"
                )
            return admit_calibration_capture_export(
                source_path,
                expected_camera_role=binding.value,
                capture_repository=services.capture_repository,
            )

    def write_calibration_task_outputs(
        self,
        source: CaptureArtifactRef,
        calibration: CalibrationArtifactRef,
        *,
        folder: str | Path,
        frame_export_policy: str,
        expected_camera_role: str | None = None,
    ) -> None:
        """Persist application-owned task outputs via the bound repositories."""

        binding = (
            self._binding
            if expected_camera_role is None
            else ReadoutBindingKey(expected_camera_role)
        )
        if binding is not None:
            self._require_binding(binding)
        with _service_guard(self._services) as services:
            if binding is not None:
                info = services.catalog.require(binding.value)
                if info.domain != "camera":
                    raise ValueError(
                        f"calibration binding {binding.value!r} is not a camera"
                    )
            write_calibration_application_outputs(
                source,
                calibration,
                folder=folder,
                frame_export_policy=frame_export_policy,
                capture_repository=services.capture_repository,
                calibration_repository=_calibration_repository(services),
                expected_camera_role=(
                    None if binding is None else binding.value
                ),
                render_report=_render_calibration_report,
            )

    def calibration_request(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> CalibrationArtifactRequest:
        """Freeze explicit calibration intent from one FINAL capture inspection."""

        with _service_guard(self._services) as services:
            request = build_calibration_artifact_request(
                services.capture_repository.admit(source),
                analysis,
            )
            self._require_binding(request.readout_binding)
            return request

    def start_calibration(
        self,
        request: CalibrationArtifactRequest,
    ) -> RunHandle:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return _start_calibration(self._services, request)

    def start_calibration_analysis(
        self,
        source: CaptureArtifactRef,
        analysis: CalibrationAnalysisRequest,
    ) -> RunHandle:
        """Start calibration from typed source and analysis application intent."""

        request = self.calibration_request(
            source,
            analysis,
        )
        return self.start_calibration(request)

    def calibrate(
        self,
        request: CalibrationArtifactRequest,
    ) -> CalibrationArtifactRef:
        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        return _run_calibration(self._services, request)

    def calibration_gui(self, request: CalibrationArtifactRequest):
        """Edit one explicit request and commit each successful formal Run."""

        if not isinstance(request, CalibrationArtifactRequest):
            raise TypeError("request must be CalibrationArtifactRequest")
        self._require_binding(request.readout_binding)
        from Zou_lab_control.workbench import open_calibration_workbench

        return open_calibration_workbench(
            self._load_calibration_report_source,
            self.start_calibration,
            request=request,
        )

    def calibration_edit_gui(
        self,
        reference: CalibrationArtifactRef,
    ):
        """Reopen an exact calibration and create a new immutable revision."""

        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        from Zou_lab_control.workbench import open_calibration_workbench

        return open_calibration_workbench(
            self._load_calibration_report_source,
            self.start_calibration,
            reference=reference,
        )

    def load_calibration(
        self,
        reference: CalibrationArtifactRef,
    ) -> ResolvedCalibration:
        with _service_guard(self._services) as services:
            resolved = _calibration_repository(services).admit(
                reference,
                services.capture_repository,
            )
            self._require_binding(resolved.artifact.frame_contract.binding)
            return resolved

    def load_saved_calibration(
        self,
        calibration_ref_file: str | Path,
    ) -> ResolvedCalibration:
        """Admit the exact calibration named by one task-output pointer.

        This is an explicit file selection, never a ``latest`` lookup.  The
        pointer's capture edge is checked against the admitted artifact so a
        copied or edited pointer cannot silently join unrelated provenance.
        """

        with _service_guard(self._services) as services:
            resolved = admit_calibration_task_output(
                calibration_ref_file,
                capture_repository=services.capture_repository,
                calibration_repository=_calibration_repository(services),
            )
            self._require_binding(resolved.artifact.frame_contract.binding)
        return resolved

    def prepare_occupancy_processor_request(
        self,
        request: OccupancyProcessorRequest,
    ) -> PreparedOccupancyProcessor:
        """Admit and prepare one complete typed Occupancy Processor request.

        The returned command owns model selection, schema validation, and the
        complete ``counts/occupied/rate`` output transaction.  A Workbench may
        host its pure ``evaluate`` operation, but does not receive a calibration
        loader or a physical projection callback.
        """

        if not isinstance(request, OccupancyProcessorRequest):
            raise TypeError("request must be OccupancyProcessorRequest")
        with _service_guard(self._services) as services:
            resolved = _calibration_repository(services).admit(
                request.calibration_ref,
                services.capture_repository,
            )
            self._require_binding(
                resolved.artifact.frame_contract.binding
            )
        return prepare_occupancy_processor(request, resolved)

    def prepare_occupancy_processor(
        self,
        camera_request: CameraMeasurementRequest,
        camera_output: DatasetOutputDeclaration,
        *,
        calibration_ref: CalibrationArtifactRef,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedOccupancyProcessor:
        """Notebook convenience for constructing one typed Processor request."""

        return self.prepare_occupancy_processor_request(
            OccupancyProcessorRequest(
                camera_request,
                camera_output,
                calibration_ref,
                model_kind,
            )
        )

    def prepare_saved_occupancy_processor(
        self,
        camera_request: CameraMeasurementRequest,
        camera_output: DatasetOutputDeclaration,
        *,
        calibration_ref_file: str | Path,
        model_kind: ReadoutModelKind | None = None,
    ) -> PreparedOccupancyProcessor:
        """Prepare Occupancy Processor from one saved calibration pointer."""

        resolved = self.load_saved_calibration(calibration_ref_file)
        request = OccupancyProcessorRequest(
            camera_request,
            camera_output,
            resolved.reference,
            model_kind,
        )
        return self.prepare_occupancy_processor_request(request)

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

        with _service_guard(self._services) as services:
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
    ) -> DetectionRequest:
        """Freeze one committed single-event capture for occupancy analysis."""

        with _service_guard(self._services) as services:
            request = build_detection_request(
                services.capture_repository.admit(source),
                _calibration_repository(services).admit(
                    calibration,
                    services.capture_repository,
                ),
                model_kind=model_kind,
            )
            self._require_binding(request.readout_binding)
            return request

    def start_detection(self, request: DetectionRequest) -> RunHandle:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return _start_detection(self._services, request)

    def detect(self, request: DetectionRequest) -> OccupancyArtifactRef:
        if not isinstance(request, DetectionRequest):
            raise TypeError("request must be DetectionRequest")
        self._require_binding(request.readout_binding)
        return _run_detection(self._services, request)

    def load_occupancy(
        self,
        reference: OccupancyArtifactRef,
    ) -> "ResolvedOccupancy":
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        with _service_guard(self._services) as services:
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
        with _service_guard(self._services) as services:
            domain = inspect_occupancy_cell_domain(
                reference,
                _occupancy_repository(services),
                services.capture_repository,
                _calibration_repository(services),
            )
            self._require_binding(domain.readout_binding)
            return domain

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
        if expected_navigation is not None and not isinstance(
            expected_navigation,
            OccupancyCellDomain,
        ):
            raise TypeError("expected_navigation must be OccupancyCellDomain or None")

        with _service_guard(self._services) as services:
            source = load_exact_occupancy_cell_source(
                reference,
                _occupancy_repository(services),
                services.capture_repository,
                _calibration_repository(services),
                selection,
                expected_domain_identity=(
                    None
                    if expected_navigation is None
                    else expected_navigation.identity
                ),
            )
            self._require_binding(source.domain.readout_binding)
            from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.view_projection import (
                build_exact_occupancy_cell_view,
            )

            return build_exact_occupancy_cell_view(source)

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
    """Resolve application artifacts, then delegate all Figure policy."""

    from zlc_frontend import (
        FrozenFigureSource,
        build_frozen_data_figure,
        build_frozen_figure_document,
    )

    is_occupancy = isinstance(source, OccupancyArtifactRef)
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
    occupancy_projection = None
    source_label = "capture"
    source_dataset_ref = None
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
        from zlc_neutral_atom.logic_nodes.readout.occupancy.ui.view_projection import (
            project_occupancy_figure,
        )

        resolved_occupancy = _occupancy_repository(services).admit(
            source,
            services.capture_repository,
            _calibration_repository(services),
        )
        occupancy_projection = project_occupancy_figure(
            resolved_occupancy,
            output=occupancy_output,
            materialize=materialize,
        )
        schema = occupancy_projection.schema
        snapshot = occupancy_projection.snapshot
        source_label = occupancy_projection.label
        selected_output = "occupied" if occupancy_output is None else occupancy_output
        selected_block = (
            resolved_occupancy.artifact.occupied
            if selected_output == "occupied"
            else resolved_occupancy.artifact.counts
        )
        source_dataset_ref = selected_block.ref(
            resolved_occupancy.artifact.generation
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
    if snapshot is None and source_ref is not None:
        if materialize:
            if source_final is None:
                raise RuntimeError("figure FINAL source is unavailable")
            if isinstance(source_ref, CaptureArtifactRef):
                snapshot = source_final.materialize_snapshot()
            elif isinstance(source_ref, ScanArtifactRef):
                snapshot = services.scan_repository.materialize(source_ref).snapshot
            else:  # pragma: no cover - source kind is closed above
                raise TypeError("fit source artifact kind is not current")
    if source_dataset_ref is None:
        raise RuntimeError("figure source Dataset identity is unavailable")

    resolved_intent = intent
    resolved_preferences = preferences
    if fit_result is None and occupancy_projection is not None:
        if resolved_intent is None:
            resolved_intent = occupancy_projection.default_intent
        resolved_preferences = occupancy_projection.resolve_preferences(
            resolved_intent,
            resolved_preferences,
        )
    frontend_source = FrozenFigureSource(
        label=(
            source_label
            if fit_result is None
            else f"fit: {fit_result.spec.model_id}"
        ),
        schema=schema,
        ref=source_dataset_ref,
        snapshot=snapshot,
        fit_result=fit_result,
    )
    if not materialize:
        document = build_frozen_figure_document(
            frontend_source,
            intent=resolved_intent,
            selection=selection,
            preferences=resolved_preferences,
        )
        return document, None, fit_result
    figure = build_frozen_data_figure(
        frontend_source,
        intent=resolved_intent,
        selection=selection,
        preferences=resolved_preferences,
    )
    return figure.document, figure, fit_result


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

    _document, figure, _fit_result = _project_notebook_figure(
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
    if figure is None:
        raise RuntimeError("frozen Figure source was not materialized")
    return figure


class Experiment:
    """Public notebook root containing values, requests, and narrow facades only."""

    __slots__ = (
        "_services",
        "name",
        "device_catalog",
        "installation_config",
        "pulse",
        "readout",
    )

    def __init__(
        self,
        services: _ExperimentServices,
        *,
        name: str,
        device_catalog: DeviceCatalogView,
        installation_config: InstallationConfigDocument,
    ) -> None:
        if not isinstance(services, _ExperimentServices):
            raise TypeError("services must be _ExperimentServices")
        self._services = services
        self.name = _text(name, "experiment name")
        if not isinstance(device_catalog, DeviceCatalogView):
            raise TypeError("device_catalog must be DeviceCatalogView")
        if not isinstance(installation_config, InstallationConfigDocument):
            raise TypeError(
                "installation_config must be InstallationConfigDocument"
            )
        self.device_catalog = device_catalog
        self.installation_config = installation_config
        self.readout = ReadoutFacade(services)
        self.pulse = PulseFacade(services)

    def pulse_gui(
        self,
        *,
        document: PulseDocument | None = None,
        path: str | Path | None = None,
    ):
        """Open the current PulseDocument editor on this exact installation."""

        from Zou_lab_control.workbench import open_pulse_editor

        return open_pulse_editor(self, document=document, path=path)

    def task_console(self, *, task=None, state=None, **kwargs):
        """Lazily open the task console bound to this experiment.

        One composition root owns the window (``zlc_workbench.task_console.app``),
        so a notebook and the double-clickable launcher open the SAME console.
        """

        from Zou_lab_control.workbench import open_task_console

        return open_task_console(self, task=task, state=state, **kwargs)

    def device_manager(self):
        """Open the one config/admin window bound to this installation."""

        from Zou_lab_control.workbench import open_device_manager

        return open_device_manager(self)

    def _open_workbench_handle(
        self,
        key: str,
        compose: Callable[[], object],
        *,
        existing_error: str | None = None,
    ):
        """Own one GUI handle without exposing the application service graph.

        Concrete Workbench construction lives in ``Zou_lab_control.workbench``;
        this method owns only Experiment-scoped reuse and retirement.
        """

        name = _text(key, "workbench key")
        if not callable(compose):
            raise TypeError("compose must be callable")
        with _service_guard(self._services) as services:
            existing = services.gui_handles.get(name)
            if existing is not None:
                if existing_error is not None:
                    raise RuntimeError(existing_error)
                if bool(getattr(existing, "permanently_closed", False)):
                    raise RuntimeError(f"this Experiment's {name} is closing")
                restore = getattr(existing, "restore_window", None)
                if not callable(restore):
                    raise RuntimeError(f"this Experiment's {name} cannot be restored")
                restore()
                return existing
            body = compose()
            services.gui_handles[name] = body
            return body

    def _close_for_device_restart(self) -> None:
        """Close this installation while its DeviceManager remains the reporter.

        The bound DeviceManager initiated the explicit restart transition, so
        it must survive long enough to receive and display the terminal admin
        state.  Every other registered Workbench is retired by ``close()``.
        """

        with _service_guard(self._services) as services:
            services.gui_handles.pop("device-manager", None)
        self.close()

    def start(self, request: CaptureRequest) -> RunHandle:
        return _start(self._services, request)

    def prepare_capture(self, request: CaptureRequest) -> PreparedFiniteCapture:
        """Bind one finite capture without exposing the runtime service graph."""

        if not isinstance(request, CaptureRequest):
            raise TypeError("request must be CaptureRequest")
        with _service_guard(self._services) as services:
            return _prepare_capture_for_services(services, request)

    def run(self, request: CaptureRequest) -> CaptureArtifactRef:
        return _run(self._services, request)

    def inspect(self, request: CaptureRequest) -> PlanDescriptor:
        with _service_guard(self._services) as services:
            return _prepare_capture_for_services(services, request).descriptor

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
        with _fit_service_guard(self._services) as services:
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
        with _service_guard(self._services) as services:
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
        open_fit: bool = False,
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
            with _service_guard(self._services) as services:
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
            with _fit_service_guard(self._services) as services:
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
            with _service_guard(self._services):
                return execution.save()

        def reload_fit_result(
            reference: FitResultArtifactRef,
        ) -> FitResultBatch:
            with _service_guard(self._services) as services:
                admitted = services.fit_repository.load(
                    reference,
                    capture_repository=services.capture_repository,
                    scan_repository=services.scan_repository,
                )
            if admitted.source_artifact_ref != fit_source:
                raise ValueError("saved Fit reopened against another source artifact")
            return admitted.result

        def figure_factory(source, *, intent, selection, preferences):
            return self.figure(
                source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
            )

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
            ):
                """Resolve the labelled display cell on the Figure worker."""

                if source != fit_source:
                    raise ValueError("direct Fit Figure loader received another source")
                with _service_guard(self._services) as services:
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
                            occupancy_output=None,
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
                        occupancy_output=None,
                    )

        from Zou_lab_control.workbench import open_figure_workbench

        return open_figure_workbench(
            figure_factory,
            display_source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            fit_preparer=prepare_fit,
            fit_executor=execute_fit,
            fit_saver=save_fit_execution,
            fit_reloader=reload_fit_result,
            fit_selected_model=chosen_model,
            fit_initial_selection=initial_selection,
            open_fit=open_fit,
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
        """Open the same DataFigure host with its Fit tab selected."""

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
            open_fit=True,
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

        with _service_guard(self._services) as services:
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

        with _service_guard(self._services) as services:
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
            experiment_services = self._services
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
                with _service_guard(experiment_services) as services:
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
                    open_fit=True,
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

        def figure_factory(current_source, *, intent, selection, preferences):
            return self.figure(
                current_source,
                intent=intent,
                selection=selection,
                preferences=preferences,
                occupancy_output=occupancy_output,
            )

        return open_figure_workbench(
            figure_factory,
            source,
            intent=intent,
            selection=selection,
            preferences=preferences,
            initial_fit_result_identity=initial_identity,
        )

    def close(self) -> None:
        services = self._services
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


def _prepare_temperature_release_recapture_for_services(
    services: _ExperimentServices,
    request: TemperatureReleaseRecaptureRequest,
) -> PreparedReleaseRecapture:
    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError(
            "request must be TemperatureReleaseRecaptureRequest"
        )
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
    )
    return prepare_temperature_release_recapture(
        request,
        calibration,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        camera_port=services.runtime.camera_port(request.camera_ref),
        start_run=services.runtime.start,
    )


def _prepare_readout_duration_fidelity_for_services(
    services: _ExperimentServices,
    request: ReadoutDurationFidelityRequest,
) -> PreparedReadoutDurationFidelity:
    if not isinstance(request, ReadoutDurationFidelityRequest):
        raise TypeError("request must be ReadoutDurationFidelityRequest")
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
    )
    return prepare_readout_duration_fidelity(
        request,
        calibration,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        camera_port=services.runtime.camera_port(request.camera_ref),
        start_run=services.runtime.start,
    )


def _prepare_grey_molasses_detuning_for_services(
    services: _ExperimentServices,
    request: GreyMolassesDetuningRequest,
) -> PreparedReleaseRecapture:
    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("request must be GreyMolassesDetuningRequest")
    rf_info = services.catalog.find(request.rf_role)
    if rf_info is None or rf_info.domain != "rf":
        raise AutonomousMeasurementUnavailable(
            GREY_MOLASSES_CAPABILITY_GAP
        )
    calibration = _calibration_repository(services).admit(
        request.calibration_ref,
        services.capture_repository,
    )
    return prepare_grey_molasses_detuning(
        request,
        calibration,
        pulse_port=services.runtime.pulse_port(request.sequencer_ref),
        camera_port=services.runtime.camera_port(request.camera_ref),
        rf_port=services.runtime.rf_port(rf_info.ref),
        start_run=services.runtime.start,
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
            association_authority=(
                services.camera_signal_association_authorities.get(
                    request.camera_ref.role
                )
            ),
        )
    return prepare_finite_camera_measurement(
        request,
        camera_port=services.runtime.camera_port(request.camera_ref),
        repository=services.capture_repository,
        start_run=services.runtime.start,
    )


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


def _start_pulse(services: _ExperimentServices, request: PulseRunRequest) -> RunHandle:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        return guarded.pulse_application.start(prepared)


def _run_pulse(services: _ExperimentServices, request: PulseRunRequest) -> PulseRunResult:
    with _service_guard(services) as guarded:
        prepared = _prepare_pulse_for_services(guarded, request)
        handle = guarded.pulse_application.start(prepared)
        runtime = guarded.runtime
    result = runtime.wait(handle)
    if not isinstance(result, PulseRunResult):
        raise TypeError("pulse Run returned an unexpected result")
    return result


def _start(services: _ExperimentServices, request: CaptureRequest) -> RunHandle:
    with _service_guard(services) as guarded:
        return _prepare_capture_for_services(guarded, request).start()


def _run(services: _ExperimentServices, request: CaptureRequest) -> CaptureArtifactRef:
    with _service_guard(services) as guarded:
        handle = _prepare_capture_for_services(guarded, request).start()
        runtime = guarded.runtime
    return runtime.wait(handle)


def _prepare_calibration_for_services(
    services: _ExperimentServices,
    request: CalibrationArtifactRequest,
):
    return prepare_calibration_artifact_plan(
        request,
        capture_repository=services.capture_repository,
        calibration_repository=_calibration_repository(services),
    )


def _start_calibration(
    services: _ExperimentServices,
    request: CalibrationArtifactRequest,
) -> RunHandle:
    with _service_guard(services) as guarded:
        plan = _prepare_calibration_for_services(guarded, request)
        return guarded.runtime.start(plan)


def _run_calibration(
    services: _ExperimentServices,
    request: CalibrationArtifactRequest,
) -> CalibrationArtifactRef:
    with _service_guard(services) as guarded:
        plan = _prepare_calibration_for_services(guarded, request)
        handle = guarded.runtime.start(plan)
        runtime = guarded.runtime
    return runtime.wait(handle)


def _prepare_detection_for_services(
    services: _ExperimentServices,
    request: DetectionRequest,
):
    return prepare_detection_plan(
        request,
        capture_repository=services.capture_repository,
        calibration_repository=_calibration_repository(services),
        occupancy_repository=_occupancy_repository(services),
    )


def _start_detection(services: _ExperimentServices, request: DetectionRequest) -> RunHandle:
    with _service_guard(services) as guarded:
        plan = _prepare_detection_for_services(guarded, request)
        return guarded.runtime.start(plan)


def _run_detection(
    services: _ExperimentServices,
    request: DetectionRequest,
) -> OccupancyArtifactRef:
    with _service_guard(services) as guarded:
        plan = _prepare_detection_for_services(guarded, request)
        handle = guarded.runtime.start(plan)
        runtime = guarded.runtime
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
    durable_makedirs(repository_root)
    capture_repository = None
    scan_repository = None
    fit_repository = None
    runtime = None
    try:
        capture_repository = CaptureRepository(repository_root / "captures")
        from zlc_neutral_atom.logic_nodes.pulse_scan.repository import ScanRepository

        scan_repository = ScanRepository(repository_root / "scans")
        fit_repository = FitResultRepository(repository_root / "fits")
        from zlc_neutral_atom.installation_dispatch import create_installation

        installation = create_installation(
            installation_document,
            required_pulse_document=required_pulse_document,
        )
        runtime = installation.runtime
        catalog = runtime.device_catalog
        sitemap_profiles: dict[str, SitemapAcquisitionProfile] = {}
        for apparatus in installation.readout_apparatus_facts:
            camera_info = catalog.require(apparatus.camera_role)
            sequencer_info = catalog.require(apparatus.sequencer_role)
            profile = build_sitemap_acquisition_profile(
                apparatus,
                camera_port=runtime.camera_port(camera_info.ref),
                pulse_port=runtime.pulse_port(sequencer_info.ref),
            )
            if profile.readout_binding.value in sitemap_profiles:
                raise ValueError(
                    "installation produced duplicate sitemap profile bindings"
                )
            sitemap_profiles[profile.readout_binding.value] = profile
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
            sitemap_profiles=MappingProxyType(sitemap_profiles),
            camera_signal_association_authorities=MappingProxyType(
                dict(installation.camera_signal_association_authorities)
            ),
            installation_config=installation_document,
            pulse_application=PulseApplicationOwner(),
            operation_lock=threading.RLock(),
            fit_operations_drained=fit_operations_drained,
            fit_operation_thread_counts={},
            gui_handles={},
        )
        experiment = Experiment(
            services,
            name=canonical_name,
            device_catalog=catalog,
            installation_config=installation_document,
        )
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
    from Zou_lab_control.workbench import open_device_manager

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
    "SitemapCalibrationRequest",
    "SitemapCalibrationFailed",
    "SitemapCalibrationInterrupted",
    "TemperatureReleaseRecaptureRequest",
]
