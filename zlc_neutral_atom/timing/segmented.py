"""Explicit API-slot segmentation over one armed exact camera transaction.

The FPGA still owns every edge inside a finite STATIC_ONCE segment.  The host
is present only at the explicitly-authorized segment boundaries: it waits for
one segment's camera event and physical pulse terminal before preparing the
next segment.  This is a concrete two-consumer coordinator, not a workflow
engine and not a fallback for autonomous SCAN_SLOT execution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Protocol, TypeVar

from zlc_data import READOUT_EVENT
from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.readout.occupancy_pipeline import (
    ExecutedOccupancy,
    ExactOccupancyTransaction,
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    _open_exact_occupancy,
    finalize_occupancy_result,
)
from zlc_neutral_atom.readout.physical_context import (
    derive_readout_physical_context,
)
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress, DatasetCellSchedule
from zlc_neutral_atom.runtime.pipeline import (
    BoundMeasurement,
    CapturePreviewPort,
    ExactCaptureTransaction,
    MinimalPipelineSpec,
    PipelineResult,
    _admit_capture_preview,
    _notify_preview_failure,
    _open_exact_capture_transaction,
    _settle_unbound_preview,
)
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
from zlc_pulse import PulseExecutionForm
from zlc_storage import nonnegative_integer, positive_integer

from ._coordination import run_cleanup_steps
from .lineage import (
    PulseCaptureBinding,
    PulseCaptureEvidence,
    PulseCaptureLineage,
)
from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
)


@dataclass(frozen=True, slots=True)
class ApiSlotPointDescriptor:
    """One unique API point and its fully-static finite pulse authority."""

    point_ordinal: int
    pulse_request: FinitePulseExecutionRequest
    pulse_binding: PulseCaptureBinding

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "point_ordinal",
            nonnegative_integer(self.point_ordinal, "point_ordinal"),
        )
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        if not isinstance(self.pulse_binding, PulseCaptureBinding):
            raise TypeError("pulse_binding must be PulseCaptureBinding")
        artifact = self.pulse_request.artifact
        if artifact.execution_form is not PulseExecutionForm.STATIC_ONCE:
            raise ValueError("API-slot segments require STATIC_ONCE pulse artifacts")
        if self.pulse_binding.compiled_artifact is not artifact:
            raise ValueError("segment request and capture binding differ")
        if self.pulse_binding.expected_trigger_count != 1:
            raise ValueError("each API-slot segment must emit exactly one camera trigger")
        plan = self.pulse_binding.cell_plan
        schema = plan.dataset_schema
        event_axes = tuple(
            axis for axis in schema.point_axes if axis.role == READOUT_EVENT
        )
        if (
            schema.repeat_axis.size != 1
            or len(event_axes) != 1
            or event_axes[0].size != 1
            or schema.point_layout.storage_size != 1
            or tuple(plan.cell_schedule) != (DatasetCellAddress(0, 0),)
            or plan.join_contract.within_point_grouping != ((0, 0),)
        ):
            raise ValueError("segment pulse binding must describe one local readout cell")


@dataclass(frozen=True, slots=True)
class ApiSlotSegmentLineage:
    """A segment address paired with its completed finite pulse receipt."""

    address: DatasetCellAddress
    point_descriptor: ApiSlotPointDescriptor
    lineage: PulseCaptureLineage

    def __post_init__(self) -> None:
        if not isinstance(self.address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress")
        if not isinstance(self.point_descriptor, ApiSlotPointDescriptor):
            raise TypeError("point_descriptor must be ApiSlotPointDescriptor")
        if not isinstance(self.lineage, PulseCaptureLineage):
            raise TypeError("lineage must be PulseCaptureLineage")
        if self.lineage.binding is not self.point_descriptor.pulse_binding:
            raise ValueError("segment lineage belongs to another point descriptor")

    @property
    def evidence(self) -> PulseCaptureEvidence:
        return self.lineage.evidence()


def _validated_point_descriptors(
    measurement: BoundMeasurement,
    pulse_port: BoundPulsePort,
    point_descriptors: tuple[ApiSlotPointDescriptor, ...],
    repeat_count: int,
) -> tuple[ApiSlotPointDescriptor, ...]:
    if not isinstance(measurement, BoundMeasurement):
        raise TypeError("measurement must be BoundMeasurement")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    repeats = positive_integer(repeat_count, "repeat_count")
    values = tuple(point_descriptors)
    if not values or any(
        not isinstance(value, ApiSlotPointDescriptor) for value in values
    ):
        raise ValueError("point_descriptors must contain API-slot point descriptors")
    if tuple(value.point_ordinal for value in values) != tuple(range(len(values))):
        raise ValueError("API point descriptors must be complete point-ordinal order")

    contract = measurement.capture_contract
    camera_spec = decode_camera_capture_spec(measurement.capture_spec)
    if camera_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError("API-slot segmented capture requires external triggering")
    camera_evidence = contract.capability.camera_capability_evidence
    if camera_evidence.exact_external_trigger_qualification_digest is None:
        raise ValueError(
            "API-slot segmented capture requires qualified ordered "
            "one-frame-per-trigger evidence"
        )
    required_interval = (
        camera_evidence.physical_facts.required_external_trigger_interval_seconds
    )
    if (
        required_interval is None
        or not math.isfinite(float(required_interval))
        or float(required_interval) < 0.0
    ):
        raise ValueError(
            "API-slot segmented capture requires a qualified external-trigger interval"
        )
    point_count = len(values)
    expected_segments = repeats * point_count
    if contract.dataset_schema.repeat_axis.size != repeats:
        raise ValueError("segment repeat count differs from the camera dataset")
    if contract.dataset_schema.point_layout.storage_size != point_count:
        raise ValueError("API point count differs from the camera point layout")
    if contract.total_events != expected_segments:
        raise ValueError("segment count differs from the expected camera event count")
    for ordinal, address in enumerate(contract.cell_schedule):
        repeat_index, point_ordinal = divmod(ordinal, point_count)
        if address != DatasetCellAddress(repeat_index, point_ordinal):
            raise ValueError(
                "camera cell schedule is not complete R-major/P-fast order"
            )

    trigger_channels = {value.pulse_binding.trigger_channel for value in values}
    if len(trigger_channels) != 1:
        raise ValueError("all API-slot segments must use one camera trigger channel")
    trigger_channel = next(iter(trigger_channels))
    camera_evidence.physical_facts.require_single_capture_trigger_channel(
        trigger_channel
    )
    capability = pulse_port.capability
    for value in values:
        artifact = value.pulse_request.artifact
        if artifact.target_abi_fingerprint != capability.target_abi_fingerprint:
            raise ValueError("segment pulse target differs from the live sequencer")
        if artifact.wire_image.geometry_fingerprint != capability.geometry_fingerprint:
            raise ValueError("segment wire geometry differs from the live sequencer")
    return values


def _required_external_trigger_interval_seconds(
    measurement: BoundMeasurement,
) -> float:
    value = (
        measurement.capture_contract.capability.camera_capability_evidence
        .physical_facts.required_external_trigger_interval_seconds
    )
    if value is None:
        raise RuntimeError("validated camera facts lost their trigger interval")
    return float(value)


def _host_boundary_delay_seconds(
    previous: ApiSlotPointDescriptor,
    following: ApiSlotPointDescriptor,
    required_interval_seconds: float,
) -> float:
    """Return only the host delay not already guaranteed by both pulse artifacts.

    Each STATIC_ONCE terminal proves that the prior program reached physical DONE
    and that its configured output-delay tail elapsed.  The following artifact's
    trigger prefix is also hardware-timed.  The host waits only for the remaining
    safety interval at the explicitly accepted API boundary; it never schedules a
    precision edge.
    """

    required = float(required_interval_seconds)
    previous_binding = previous.pulse_binding
    following_binding = following.pulse_binding
    previous_artifact = previous_binding.compiled_artifact
    following_artifact = following_binding.compiled_artifact
    previous_schedule = previous_binding.trigger_schedule
    following_schedule = following_binding.trigger_schedule
    if previous_schedule.total != 1 or following_schedule.total != 1:
        raise RuntimeError("API boundary timing requires one trigger per segment")
    previous_clock = previous_artifact.target_ir.clock_hz
    following_clock = following_artifact.target_ir.clock_hz
    previous_trigger_seconds = (
        int(previous_schedule.ticks_from_run_start[0]) / previous_clock
    )
    previous_after_trigger_seconds = (
        previous_artifact.target_ir.duration_seconds
        - previous_trigger_seconds
        + previous_artifact.max_configured_output_delay_ticks / previous_clock
    )
    following_before_trigger_seconds = (
        int(following_schedule.ticks_from_run_start[0]) / following_clock
    )
    hardware_guaranteed = (
        previous_after_trigger_seconds + following_before_trigger_seconds
    )
    return max(0.0, required - hardware_guaranteed)


def _wait_for_host_boundary(context: RunContext, not_before: float | None) -> None:
    if not_before is None:
        return
    while True:
        context.checkpoint()
        remaining = not_before - time.monotonic()
        if remaining <= 0.0:
            return
        if context.deadline is not None and not_before >= context.deadline:
            raise TimeoutError(
                "camera trigger safety interval exceeds the Run deadline"
            )
        time.sleep(min(0.01, remaining))


@dataclass(frozen=True, slots=True)
class ApiSlotSegmentedSpec:
    """One global exact pipeline driven by ordered finite API segments."""

    pipeline: MinimalPipelineSpec | OccupancyPipelineSpec
    pulse_port: BoundPulsePort
    point_descriptors: tuple[ApiSlotPointDescriptor, ...]
    repeat_count: int

    def __post_init__(self) -> None:
        if not isinstance(
            self.pipeline,
            (MinimalPipelineSpec, OccupancyPipelineSpec),
        ):
            raise TypeError("pipeline must be direct capture or occupancy")
        values = _validated_point_descriptors(
            self.pipeline.measurement,
            self.pulse_port,
            self.point_descriptors,
            self.repeat_count,
        )
        object.__setattr__(self, "point_descriptors", values)
        object.__setattr__(
            self,
            "repeat_count",
            positive_integer(self.repeat_count, "repeat_count"),
        )
        if not isinstance(self.pipeline, OccupancyPipelineSpec):
            return
        calibration = self.pipeline.processor.calibration.artifact
        contract = self.pipeline.measurement.capture_contract
        physical_facts = contract.capability.camera_physical_facts
        integration_offset = (
            physical_facts.external_trigger_integration_start_offset_seconds
        )
        for descriptor in values:
            current_context = derive_readout_physical_context(
                descriptor.pulse_binding,
                readout_event_index=0,
                integration_start_offset_seconds=integration_offset,
                integration_seconds=calibration.frame_contract.exposure_seconds,
            )
            if current_context != calibration.readout_physical_context:
                raise ValueError(
                    "one API-slot segment readout context differs from calibration"
                )


def _validated_segment_lineages(
    capture: PipelineResult,
    values: tuple[ApiSlotSegmentLineage, ...],
) -> tuple[ApiSlotSegmentLineage, ...]:
    if not isinstance(capture, PipelineResult):
        raise TypeError("capture must be PipelineResult")
    segments = tuple(values)
    if not segments or any(
        not isinstance(value, ApiSlotSegmentLineage) for value in segments
    ):
        raise ValueError("segments must contain completed API-slot lineages")
    addresses = tuple(value.address for value in segments)
    if addresses != tuple(capture.source_cell_schedule):
        raise ValueError("completed segment order differs from captured cell order")
    expected = len(segments)
    terminal = capture.capture_terminal
    if terminal.produced_count != expected or terminal.drained_count != expected:
        raise RuntimeError("camera terminal count differs from completed segments")
    trigger_channels = {value.lineage.trigger_channel for value in segments}
    if len(trigger_channels) != 1:
        raise RuntimeError("completed segments used different camera trigger channels")
    capture.camera_capability_evidence.physical_facts.require_single_capture_trigger_channel(
        next(iter(trigger_channels))
    )
    return segments


@dataclass(frozen=True, slots=True)
class ApiSlotSegmentedResult:
    """Validated direct or occupancy result plus exact segment evidence."""

    payload: PipelineResult | OccupancyPipelineResult
    segments: tuple[ApiSlotSegmentLineage, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (PipelineResult, OccupancyPipelineResult)):
            raise TypeError("payload must be direct capture or occupancy result")
        capture = (
            self.payload
            if isinstance(self.payload, PipelineResult)
            else self.payload.pipeline
        )
        values = _validated_segment_lineages(capture, self.segments)
        if isinstance(self.payload, OccupancyPipelineResult) and len(
            self.payload.dataset.event_metadata
        ) != len(values):
            raise RuntimeError("occupancy metadata count differs from pulse segments")
        if isinstance(self.payload, OccupancyPipelineResult) and not (
            self.payload.dataset.cell_schedule.same_order_as(
                self.payload.pipeline.source_cell_schedule
            )
        ):
            raise ValueError("occupancy and camera cell schedules differ")
        object.__setattr__(self, "segments", values)


_CompletedT = TypeVar("_CompletedT")


class _SegmentedExactCapture(Protocol[_CompletedT]):
    def start(self, context: RunContext) -> None: ...

    def capture_next(self, context: RunContext) -> None: ...

    def complete(self, context: RunContext) -> _CompletedT: ...

    def fail(self, error: BaseException) -> None: ...


@dataclass(slots=True)
class _PreparedSegmentedCapture:
    capture: _SegmentedExactCapture
    current_pulse: PulseSession | None = None


def _execute_segments(
    context: RunContext,
    *,
    pulse_port: BoundPulsePort,
    point_descriptors: tuple[ApiSlotPointDescriptor, ...],
    cell_schedule: DatasetCellSchedule,
    prepared: _PreparedSegmentedCapture,
    required_interval_seconds: float,
) -> tuple[object, tuple[ApiSlotSegmentLineage, ...]]:
    """Run one arm and the explicit terminal-gated segment sequence."""

    lineages: list[ApiSlotSegmentLineage] = []
    capture_started = False
    next_fire_not_before: float | None = None
    try:
        point_count = len(point_descriptors)
        for ordinal, address in enumerate(cell_schedule):
            descriptor = point_descriptors[ordinal % point_count]
            context.checkpoint()
            pulse = pulse_port.open_session(descriptor.pulse_request)
            prepared.current_pulse = pulse
            pulse.prepare(context)
            if not capture_started:
                prepared.capture.start(context)
                capture_started = True
            _wait_for_host_boundary(context, next_fire_not_before)
            pulse.fire(context)
            prepared.capture.capture_next(context)
            terminal = pulse.complete(context)
            if not pulse.owns_terminal(terminal):
                raise PermissionError(
                    "segment pulse terminal was not minted by its PulseSession"
                )
            lineages.append(
                ApiSlotSegmentLineage(
                    address,
                    descriptor,
                    PulseCaptureLineage(descriptor.pulse_binding, terminal),
                )
            )
            if ordinal + 1 < len(cell_schedule):
                following = point_descriptors[(ordinal + 1) % point_count]
                host_delay = _host_boundary_delay_seconds(
                    descriptor,
                    following,
                    required_interval_seconds,
                )
                next_fire_not_before = time.monotonic() + host_delay
        if len(lineages) != len(cell_schedule):
            raise RuntimeError("not every API-slot segment reached pulse terminal")
        completed = prepared.capture.complete(context)
        return completed, tuple(lineages)
    except BaseException as primary:
        try:
            prepared.capture.fail(primary)
        except BaseException as secondary:
            record_secondary_failure(
                primary,
                "aggregate capture poison also failed",
                secondary,
            )
        pulse = prepared.current_pulse
        if pulse is not None:
            try:
                pulse.fail()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "current segment pulse poison also failed",
                    secondary,
                )
        raise


def _cleanup_prepared(
    context: RunContext,
    *,
    pulse_port: BoundPulsePort,
    prepared: _PreparedSegmentedCapture,
    capture_cleanup,
) -> CleanupReport:
    pulse = prepared.current_pulse
    return run_cleanup_steps(
        (
            (lambda: pulse_port.verify_idle(context))
            if pulse is None
            else (lambda: pulse.cleanup(context))
        ),
        capture_cleanup,
    )


def compile_api_slot_segmented_pipeline(
    spec: ApiSlotSegmentedSpec,
    *,
    preview: CapturePreviewPort | None = None,
) -> RunPlan:
    """Compile one direct exact camera arm with ordered finite API segments."""

    if not isinstance(spec, ApiSlotSegmentedSpec) or not isinstance(
        spec.pipeline,
        MinimalPipelineSpec,
    ):
        raise TypeError("spec must contain a direct capture pipeline")
    capture_spec = spec.pipeline
    camera_port = capture_spec.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        error = ValueError("camera and sequencer must be distinct physical resources")
        _notify_preview_failure(preview, error)
        raise error
    preview_spec = _admit_capture_preview(capture_spec, preview)

    def preflight(context: RunContext) -> _PreparedSegmentedCapture:
        capture = _open_exact_capture_transaction(
            capture_spec,
            context,
            preview=preview,
            preview_spec=preview_spec,
        )
        return _PreparedSegmentedCapture(capture)

    def execute(
        context: RunContext,
        prepared: _PreparedSegmentedCapture,
    ) -> ApiSlotSegmentedResult:
        completed, segments = _execute_segments(
            context,
            pulse_port=pulse_port,
            point_descriptors=spec.point_descriptors,
            cell_schedule=spec.pipeline.measurement.capture_contract.cell_schedule,
            prepared=prepared,
            required_interval_seconds=_required_external_trigger_interval_seconds(
                spec.pipeline.measurement
            ),
        )
        if not isinstance(completed, PipelineResult):
            raise TypeError("direct segmented capture returned another result type")
        return ApiSlotSegmentedResult(completed, segments)

    def cleanup(
        context: RunContext,
        prepared: _PreparedSegmentedCapture | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            report = run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
            return _settle_unbound_preview(preview, report, primary)
        capture = prepared.capture
        if not isinstance(capture, ExactCaptureTransaction):
            raise TypeError("prepared direct capture has another type")
        report = _cleanup_prepared(
            context,
            pulse_port=pulse_port,
            prepared=prepared,
            capture_cleanup=lambda: capture.cleanup(context),
        )
        capture.settle_preview_after_cleanup(report, primary)
        return report

    def finalize(
        context: PostSafetyContext,
        result: ApiSlotSegmentedResult,
    ) -> ApiSlotSegmentedResult:
        if not isinstance(result, ApiSlotSegmentedResult) or not isinstance(
            result.payload,
            PipelineResult,
        ):
            raise TypeError("segmented direct finalize received another result")
        if result.payload.run_id != context.run_id.value:
            raise ValueError("segmented capture result belongs to another Run")
        context.checkpoint()
        return result

    return RunPlan(
        name=capture_spec.name,
        resource_claims=(pulse_port.resource_claim, camera_port.resource_claim),
        bound_devices=(pulse_port.device, camera_port.device),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            *pulse_port.interrupt_operations,
            *camera_port.interrupt_operations,
        ),
        requires_final_commit=False,
    )


def compile_api_slot_segmented_occupancy_pipeline(
    spec: ApiSlotSegmentedSpec,
) -> RunPlan:
    """Compile FINAL-only occupancy over one armed segmented capture."""

    if not isinstance(spec, ApiSlotSegmentedSpec) or not isinstance(
        spec.pipeline,
        OccupancyPipelineSpec,
    ):
        raise TypeError("spec must contain an occupancy pipeline")
    occupancy_spec = spec.pipeline
    camera_port = occupancy_spec.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(context: RunContext) -> _PreparedSegmentedCapture:
        occupancy = _open_exact_occupancy(
            occupancy_spec,
            context,
        )
        return _PreparedSegmentedCapture(occupancy)

    def execute(
        context: RunContext,
        prepared: _PreparedSegmentedCapture,
    ) -> tuple[ExecutedOccupancy, tuple[ApiSlotSegmentLineage, ...]]:
        completed, segments = _execute_segments(
            context,
            pulse_port=pulse_port,
            point_descriptors=spec.point_descriptors,
            cell_schedule=spec.pipeline.measurement.capture_contract.cell_schedule,
            prepared=prepared,
            required_interval_seconds=_required_external_trigger_interval_seconds(
                spec.pipeline.measurement
            ),
        )
        if not isinstance(completed, ExecutedOccupancy):
            raise TypeError("segmented occupancy returned another executed type")
        return completed, segments

    def cleanup(
        context: RunContext,
        prepared: _PreparedSegmentedCapture | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        occupancy = prepared.capture
        if not isinstance(occupancy, ExactOccupancyTransaction):
            raise TypeError("prepared occupancy capture has another type")
        return _cleanup_prepared(
            context,
            pulse_port=pulse_port,
            prepared=prepared,
            capture_cleanup=lambda: occupancy.cleanup(context),
        )

    def finalize(
        context: PostSafetyContext,
        executed: tuple[ExecutedOccupancy, tuple[ApiSlotSegmentLineage, ...]],
    ) -> ApiSlotSegmentedResult:
        if not isinstance(executed, tuple) or len(executed) != 2:
            raise TypeError("segmented occupancy finalize received another value")
        occupancy_body, segments = executed
        if not isinstance(occupancy_body, ExecutedOccupancy):
            raise TypeError("segmented occupancy finalize received another value")
        if occupancy_body.pipeline.run_id != context.run_id.value:
            raise ValueError("segmented occupancy result belongs to another Run")
        occupancy = finalize_occupancy_result(context, occupancy_body)
        context.checkpoint()
        return ApiSlotSegmentedResult(occupancy, segments)

    return RunPlan(
        name=occupancy_spec.name,
        resource_claims=(pulse_port.resource_claim, camera_port.resource_claim),
        bound_devices=(pulse_port.device, camera_port.device),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            *pulse_port.interrupt_operations,
            *camera_port.interrupt_operations,
        ),
        requires_final_commit=False,
    )


__all__ = [
    "ApiSlotPointDescriptor",
    "ApiSlotSegmentLineage",
    "ApiSlotSegmentedResult",
    "ApiSlotSegmentedSpec",
    "compile_api_slot_segmented_occupancy_pipeline",
    "compile_api_slot_segmented_pipeline",
]
