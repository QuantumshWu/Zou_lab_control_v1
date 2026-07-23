"""One autonomous FPGA-triggered two-readout survival run."""

from __future__ import annotations

import uuid
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
from zlc_neutral_atom.rf import BoundRfTablePort, RfDetuningTable, RfTableTerminal
from zlc_storage import canonical_text

from ._coordination import (
    execute_autonomous_single_fire,
    execute_autonomous_single_fire_with_rf_table,
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
    rf_port: BoundRfTablePort | None = None
    rf_table: RfDetuningTable | None = None
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
        if (self.rf_port is None) != (self.rf_table is None):
            raise ValueError("RF Port and table must be supplied together")
        if self.rf_port is not None:
            if not isinstance(self.rf_port, BoundRfTablePort):
                raise TypeError("rf_port must be BoundRfTablePort")
            if not isinstance(self.rf_table, RfDetuningTable):
                raise TypeError("rf_table must be RfDetuningTable")
            if self.rf_table.pulse_artifact_digest != self.pulse_request.artifact_digest:
                raise ValueError("RF table belongs to another pulse artifact")


@dataclass(frozen=True, slots=True)
class TriggeredReleaseRecaptureResult:
    release_recapture: ReleaseRecapturePipelineResult
    lineage: PulseCaptureLineage
    rf_terminal: RfTableTerminal | None = None

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
        if self.rf_terminal is not None and not isinstance(
            self.rf_terminal, RfTableTerminal
        ):
            raise TypeError("rf_terminal must be RfTableTerminal or None")
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
    rf_session_id: str | None


@dataclass(frozen=True, slots=True)
class _ExecutedTriggeredReleaseRecapture:
    release_recapture: ReleaseRecapturePipelineResult
    lineage: PulseCaptureLineage
    rf_terminal: RfTableTerminal | None


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
        return _PreparedTriggeredReleaseRecapture(
            reducer,
            pulse,
            None if spec.rf_port is None else uuid.uuid4().hex,
        )

    def execute(
        context: RunContext,
        prepared: _PreparedTriggeredReleaseRecapture,
    ) -> _ExecutedTriggeredReleaseRecapture:
        rf_terminal = None
        if spec.rf_port is None:
            result, terminal = execute_autonomous_single_fire(
                context,
                pulse=prepared.pulse,
                capture=prepared.reducer,
            )
        else:
            assert spec.rf_table is not None
            assert prepared.rf_session_id is not None
            result, terminal, rf_terminal = (
                execute_autonomous_single_fire_with_rf_table(
                    context,
                    pulse=prepared.pulse,
                    capture=prepared.reducer,
                    rf_port=spec.rf_port,
                    rf_session_id=prepared.rf_session_id,
                    rf_table=spec.rf_table,
                )
            )
        if not isinstance(result, ReleaseRecapturePipelineResult):
            raise TypeError(
                "release-recapture capture returned another result type"
            )
        return _ExecutedTriggeredReleaseRecapture(
            result,
            PulseCaptureLineage(spec.pulse_binding, terminal),
            rf_terminal,
        )

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredReleaseRecapture | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            steps = [
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            ]
            if spec.rf_port is not None:
                steps.append(lambda: spec.rf_port.verify_idle(context))
            return run_cleanup_steps(*steps)
        steps = [
            lambda: prepared.pulse.cleanup(context),
            lambda: prepared.reducer.cleanup(context),
        ]
        if spec.rf_port is not None:
            assert prepared.rf_session_id is not None
            steps.append(
                lambda: spec.rf_port.cleanup(context, prepared.rf_session_id)
            )
        return run_cleanup_steps(*steps)

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
            executed.rf_terminal,
        )

    rf_claims = () if spec.rf_port is None else (spec.rf_port.resource_claim,)
    rf_devices = () if spec.rf_port is None else (spec.rf_port.device,)
    rf_interrupts = () if spec.rf_port is None else spec.rf_port.interrupt_operations
    return RunPlan(
        name=spec.release_recapture.name,
        resource_claims=(
            pulse_port.resource_claim,
            camera_port.resource_claim,
            *rf_claims,
        ),
        bound_devices=(pulse_port.device, camera_port.device, *rf_devices),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=(
            *pulse_port.interrupt_operations,
            *camera_port.interrupt_operations,
            *rf_interrupts,
        ),
        requires_final_commit=False,
    )


__all__ = [
    "TriggeredReleaseRecaptureResult",
    "TriggeredReleaseRecaptureSpec",
    "compile_triggered_release_recapture_pipeline",
]
