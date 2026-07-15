"""One autonomous FPGA-triggered camera-to-occupancy exact run."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

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
from zlc_neutral_atom.runtime.cleanup import CleanupReport
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


def compile_triggered_occupancy_pipeline(spec: TriggeredOccupancySpec) -> RunPlan:
    """Compile ready-all -> arm camera -> one FIRE -> drain -> attest."""

    if not isinstance(spec, TriggeredOccupancySpec):
        raise TypeError("spec must be TriggeredOccupancySpec")
    camera_port = spec.occupancy.measurement.capture_port
    pulse_port = spec.pulse_port
    pulse_binding = spec.pulse_binding
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(context: RunContext) -> _PreparedTriggeredOccupancy:
        pulse = pulse_port.open_session(spec.pulse_request)
        try:
            occupancy = _open_exact_occupancy(spec.occupancy, context)
        except BaseException:
            pulse.fail()
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
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredOccupancy | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is not None:
            return run_cleanup_steps(
                lambda: prepared.pulse.cleanup(context),
                lambda: prepared.occupancy.cleanup(context),
            )
        return run_cleanup_steps(
            lambda: pulse_port.verify_idle(context),
            lambda: camera_port.verify_idle(context),
        )

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedTriggeredOccupancy,
    ) -> TriggeredOccupancyPipelineResult:
        if type(executed) is not _ExecutedTriggeredOccupancy:
            raise TypeError("triggered occupancy finalize received another value")
        pipeline = executed.occupancy.pipeline
        if pipeline.run_id != context.run_id.value:
            raise ValueError("triggered occupancy result belongs to another Run")
        occupancy = finalize_occupancy_result(context, executed.occupancy)
        context.checkpoint()
        return TriggeredOccupancyPipelineResult(occupancy, executed.lineage)

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
    )


__all__ = [
    "compile_triggered_occupancy_pipeline",
    "TriggeredOccupancyPipelineResult",
    "TriggeredOccupancySpec",
]
