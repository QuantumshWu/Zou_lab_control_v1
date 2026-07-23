"""One autonomous FPGA-triggered two-readout survival run."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from zlc_neutral_atom.readout.release_recapture_pipeline import (
    ExactReleaseRecaptureTransaction,
    ReleaseRecapturePipelineResult,
    ReleaseRecapturePipelineSpec,
    open_exact_release_recapture,
)
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import PostSafetyContext, RunContext, RunPlan
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
class TriggeredReleaseRecaptureSpec:
    release_recapture: ReleaseRecapturePipelineSpec
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
        if not isinstance(
            self.release_recapture,
            ReleaseRecapturePipelineSpec,
        ):
            raise TypeError(
                "release_recapture must be ReleaseRecapturePipelineSpec"
            )
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        canonical_text(trigger_channel, "trigger_channel")
        if not isinstance(cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        binding = PulseCaptureBinding(
            self.pulse_request.artifact,
            trigger_channel,
            cell_plan,
        )
        validate_single_trigger_capture_binding(
            capture_spec=self.release_recapture.measurement.capture_spec,
            contract=(
                self.release_recapture.measurement.capture_contract
            ),
            pulse_binding=binding,
        )
        grouping = binding.cell_plan.join_contract.within_point_grouping
        event_indices = {event for _repeat, event in grouping}
        if event_indices != {0, 1}:
            raise ValueError(
                "release-recapture requires physical readout events 0 and 1"
            )
        object.__setattr__(self, "pulse_binding", binding)


@dataclass(frozen=True, slots=True)
class TriggeredReleaseRecaptureResult:
    release_recapture: ReleaseRecapturePipelineResult
    lineage: PulseCaptureLineage

    def __post_init__(self) -> None:
        if not isinstance(
            self.release_recapture,
            ReleaseRecapturePipelineResult,
        ):
            raise TypeError(
                "release_recapture must be ReleaseRecapturePipelineResult"
            )
        if not isinstance(self.lineage, PulseCaptureLineage):
            raise TypeError("lineage must be PulseCaptureLineage")
        pipeline = self.release_recapture.pipeline
        self.lineage.cell_plan.validate_dataset_schema(
            pipeline.source_dataset_schema
        )
        if not pipeline.source_cell_schedule.same_order_as(
            self.lineage.cell_plan.cell_schedule
        ):
            raise ValueError(
                "release-recapture camera order differs from pulse cell plan"
            )
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
            == pipeline.source_event_span.end_sequence
            - pipeline.source_event_span.start_sequence
        ):
            raise RuntimeError(
                "pulse, camera, and release-recapture source counts differ"
            )
        if pipeline.dataset.coverage.total_cells * 2 != expected:
            raise RuntimeError(
                "release-recapture output is not an exact two-frame reduction"
            )

    @property
    def survival(self):
        return self.release_recapture.survival


@dataclass(slots=True)
class _PreparedTriggeredReleaseRecapture:
    reducer: ExactReleaseRecaptureTransaction
    pulse: PulseSession


@dataclass(frozen=True, slots=True)
class _ExecutedTriggeredReleaseRecapture:
    release_recapture: ReleaseRecapturePipelineResult
    lineage: PulseCaptureLineage


def compile_triggered_release_recapture_pipeline(
    spec: TriggeredReleaseRecaptureSpec,
) -> RunPlan:
    """Compile ready-all -> arm camera -> one hardware FIRE -> exact pairs."""

    if not isinstance(spec, TriggeredReleaseRecaptureSpec):
        raise TypeError("spec must be TriggeredReleaseRecaptureSpec")
    camera_port = spec.release_recapture.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(context: RunContext) -> _PreparedTriggeredReleaseRecapture:
        pulse = pulse_port.open_session(spec.pulse_request)
        try:
            reducer = open_exact_release_recapture(
                spec.release_recapture,
                context,
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
        return _PreparedTriggeredReleaseRecapture(reducer, pulse)

    def execute(
        context: RunContext,
        prepared: _PreparedTriggeredReleaseRecapture,
    ) -> _ExecutedTriggeredReleaseRecapture:
        result, terminal = execute_autonomous_single_fire(
            context,
            pulse=prepared.pulse,
            capture=prepared.reducer,
        )
        if not isinstance(result, ReleaseRecapturePipelineResult):
            raise TypeError(
                "release-recapture capture returned another result type"
            )
        return _ExecutedTriggeredReleaseRecapture(
            result,
            PulseCaptureLineage(spec.pulse_binding, terminal),
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredReleaseRecapture | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        return run_cleanup_steps(
            lambda: prepared.pulse.cleanup(context),
            lambda: prepared.reducer.cleanup(context),
        )

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedTriggeredReleaseRecapture,
    ) -> TriggeredReleaseRecaptureResult:
        if not isinstance(
            executed,
            _ExecutedTriggeredReleaseRecapture,
        ):
            raise TypeError(
                "release-recapture finalize received another value"
            )
        if executed.release_recapture.pipeline.run_id != context.run_id.value:
            raise ValueError(
                "release-recapture result belongs to another Run"
            )
        context.checkpoint()
        return TriggeredReleaseRecaptureResult(
            executed.release_recapture,
            executed.lineage,
        )

    return RunPlan(
        name=spec.release_recapture.name,
        resource_claims=(
            pulse_port.resource_claim,
            camera_port.resource_claim,
        ),
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
    "TriggeredReleaseRecaptureResult",
    "TriggeredReleaseRecaptureSpec",
    "compile_triggered_release_recapture_pipeline",
]
