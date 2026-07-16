"""One autonomous FPGA-triggered camera-to-occupancy exact run."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from zlc_neutral_atom.readout.occupancy_pipeline import (
    ExecutedOccupancy,
    ExactOccupancyTransaction,
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    _finish_preview_after_post_safety,
    _occupancy_preview_spec,
    _open_exact_occupancy,
    _settle_unbound_preview,
    finalize_occupancy_result,
)
from zlc_neutral_atom.readout.physical_context import (
    derive_readout_physical_context,
)
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.pipeline import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
    _notify_preview_failure,
)
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from zlc_storage import canonical_text

from ._coordination import (
    execute_autonomous_single_fire,
    run_cleanup_steps,
    validate_single_trigger_capture_binding,
)
from .capture_plan import CompiledCaptureCellPlan
from .lineage import PulseCaptureBinding, PulseCaptureLineage
from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
)


@dataclass(frozen=True, slots=True)
class TriggeredOccupancySpec:
    occupancy: OccupancyPipelineSpec
    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: InitVar[str]
    cell_plan: InitVar[CompiledCaptureCellPlan]
    pulse_binding: PulseCaptureBinding = field(init=False)

    def __post_init__(
        self,
        trigger_channel: str,
        cell_plan: CompiledCaptureCellPlan,
    ) -> None:
        if not isinstance(self.occupancy, OccupancyPipelineSpec):
            raise TypeError("occupancy must be OccupancyPipelineSpec")
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        canonical_text(trigger_channel, "trigger_channel")
        if not isinstance(cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        pulse_binding = PulseCaptureBinding(
            self.pulse_request.artifact,
            trigger_channel,
            cell_plan,
        )
        contract = self.occupancy.measurement.capture_contract
        validate_single_trigger_capture_binding(
            capture_spec=self.occupancy.measurement.capture_spec,
            contract=contract,
            pulse_binding=pulse_binding,
        )
        object.__setattr__(self, "pulse_binding", pulse_binding)
        grouping = pulse_binding.cell_plan.join_contract.within_point_grouping
        event_indices = {event for _repeat, event in grouping}
        if len(event_indices) != 1:
            raise ValueError(
                "triggered occupancy requires one physical READOUT_EVENT"
            )
        calibration = self.occupancy.processor.calibration.artifact
        physical_facts = contract.capability.camera_physical_facts
        integration_offset = (
            physical_facts.external_trigger_integration_start_offset_seconds
        )
        current_context = derive_readout_physical_context(
            pulse_binding,
            readout_event_index=next(iter(event_indices)),
            integration_start_offset_seconds=integration_offset,
            integration_seconds=calibration.frame_contract.exposure_seconds,
        )
        if current_context != calibration.readout_physical_context:
            raise ValueError(
                "triggered occupancy pulse context differs from calibration"
            )


@dataclass(frozen=True, slots=True)
class TriggeredOccupancyPipelineResult:
    """Validated non-authoritative view of one triggered occupancy result.

    Durable repositories must admit their own closed result authority; this
    value only presents already-validated pulse, camera, and occupancy facts.
    """

    occupancy: OccupancyPipelineResult
    lineage: PulseCaptureLineage

    def __post_init__(self) -> None:
        if not isinstance(self.occupancy, OccupancyPipelineResult):
            raise TypeError("occupancy must be OccupancyPipelineResult")
        if not isinstance(self.lineage, PulseCaptureLineage):
            raise TypeError("lineage must be PulseCaptureLineage")

        pipeline = self.occupancy.pipeline
        self.lineage.cell_plan.validate_dataset_schema(
            pipeline.source_dataset_schema,
        )
        if (
            not self.occupancy.dataset.cell_schedule.same_order_as(
                self.lineage.cell_plan.cell_schedule
            )
        ):
            raise ValueError("occupancy event order differs from pulse cell plan")
        evidence = pipeline.camera_capability_evidence
        evidence.physical_facts.require_single_capture_trigger_channel(
            self.lineage.trigger_channel
        )
        expected = self.lineage.expected_trigger_count
        terminal = pipeline.capture_terminal
        if not (
            expected
            == terminal.produced_count
            == terminal.drained_count
            == len(self.occupancy.dataset.event_metadata)
        ):
            raise RuntimeError("pulse, camera, plan, and occupancy counts differ")


@dataclass(slots=True)
class _PreparedTriggeredOccupancy:
    occupancy: ExactOccupancyTransaction
    pulse: PulseSession


@dataclass(frozen=True, slots=True)
class _ExecutedTriggeredOccupancy:
    occupancy: ExecutedOccupancy
    lineage: PulseCaptureLineage
    preview: ExactDatasetPreviewPort | None


def _dispose_triggered_preview(value: _ExecutedTriggeredOccupancy) -> None:
    if type(value) is not _ExecutedTriggeredOccupancy:
        raise TypeError("unfinalized triggered occupancy has another type")
    _notify_preview_failure(
        value.preview,
        RuntimeError(
            "triggered occupancy preview rejected before post-safety "
            "finalization"
        ),
    )


def compile_triggered_occupancy_pipeline(
    spec: TriggeredOccupancySpec,
    *,
    preview: ExactDatasetPreviewPort | None = None,
    _admitted_preview_spec: ExactDatasetPreviewSpec | None = None,
    _retained_overhead_bytes: int = 0,
) -> RunPlan:
    """Compile ready-all -> arm camera -> one FIRE -> drain -> attest."""

    if not isinstance(spec, TriggeredOccupancySpec):
        raise TypeError("spec must be TriggeredOccupancySpec")
    camera_port = spec.occupancy.measurement.capture_port
    pulse_port = spec.pulse_port
    pulse_binding = spec.pulse_binding
    if camera_port.device.key == pulse_port.device.key:
        error = ValueError(
            "camera and sequencer must be distinct physical resources"
        )
        _notify_preview_failure(preview, error)
        raise error
    if _admitted_preview_spec is None:
        try:
            preview_spec = _occupancy_preview_spec(spec.occupancy, preview)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise
    else:
        try:
            if preview is None:
                raise ValueError(
                    "an admitted preview spec requires its preview port"
                )
            if not isinstance(_admitted_preview_spec, ExactDatasetPreviewSpec):
                raise TypeError("_admitted_preview_spec has the wrong type")
            preview_spec = _occupancy_preview_spec(spec.occupancy, preview)
            if preview_spec != _admitted_preview_spec:
                raise ValueError(
                    "occupancy preview budget changed after scan admission"
                )
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise

    def preflight(context: RunContext) -> _PreparedTriggeredOccupancy:
        try:
            pulse = pulse_port.open_session(spec.pulse_request)
        except BaseException as error:
            _notify_preview_failure(preview, error)
            raise
        try:
            occupancy = _open_exact_occupancy(
                spec.occupancy,
                context,
                preview=preview,
                preview_spec=preview_spec,
                retained_overhead_bytes=_retained_overhead_bytes,
            )
        except BaseException as primary:
            try:
                pulse.fail()
            except BaseException as secondary:
                record_secondary_failure(
                    primary,
                    "pulse preflight poison also failed",
                    secondary,
                )
            raise
        return _PreparedTriggeredOccupancy(occupancy, pulse)

    def execute(
        context: RunContext,
        prepared: _PreparedTriggeredOccupancy,
    ) -> _ExecutedTriggeredOccupancy:
        completed, terminal = execute_autonomous_single_fire(
            context,
            pulse=prepared.pulse,
            capture=prepared.occupancy,
        )
        if not isinstance(completed, ExecutedOccupancy):
            raise TypeError("occupancy capture returned another executed value")
        return _ExecutedTriggeredOccupancy(
            completed,
            PulseCaptureLineage(pulse_binding, terminal),
            prepared.occupancy.preview,
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredOccupancy | None,
        primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is not None:
            report = run_cleanup_steps(
                lambda: prepared.pulse.cleanup(context),
                lambda: prepared.occupancy.cleanup(context),
            )
            prepared.occupancy.settle_preview_after_cleanup(report, primary)
            return report
        report = run_cleanup_steps(
            lambda: pulse_port.verify_idle(context),
            lambda: camera_port.verify_idle(context),
        )
        return _settle_unbound_preview(preview, report, primary)

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedTriggeredOccupancy,
    ) -> TriggeredOccupancyPipelineResult:
        if type(executed) is not _ExecutedTriggeredOccupancy:
            raise TypeError("triggered occupancy finalize received another value")
        pipeline = executed.occupancy.pipeline
        try:
            if pipeline.run_id != context.run_id.value:
                raise ValueError("triggered occupancy result belongs to another Run")
            occupancy = finalize_occupancy_result(context, executed.occupancy)
            context.checkpoint()
            result = TriggeredOccupancyPipelineResult(occupancy, executed.lineage)
        except BaseException as error:
            _notify_preview_failure(executed.preview, error)
            raise
        _finish_preview_after_post_safety(executed.preview)
        return result

    return RunPlan(
        name=spec.occupancy.name,
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
        timeout_seconds=spec.occupancy.timeout_seconds,
        requires_final_commit=False,
        dispose_unfinalized=_dispose_triggered_preview,
    )


__all__ = [
    "compile_triggered_occupancy_pipeline",
    "TriggeredOccupancyPipelineResult",
    "TriggeredOccupancySpec",
]
