"""One flat FPGA-triggered camera -> occupancy exact coordinator.

This is deliberately a concrete vertical slice, not a generic workflow engine:
one pulse session gates one already-bound camera/processor/materializer chain.
"""

from __future__ import annotations

from dataclasses import dataclass
from zlc_storage import canonical_text as _text

from zlc_neutral_atom.readout.occupancy_pipeline import (
    OccupancyPipelineResult,
    OccupancyPipelineSpec,
    _ExecutedOccupancyPipeline,
    _OccupancyTransaction,
    _finalize_occupancy_result,
    _open_occupancy_transaction,
)
from zlc_neutral_atom.runtime import (
    CleanupReport,
    PostSafetyContext,
    RunContext,
    RunMode,
    RunPlan,
    dataset_cell_permutation_digest,
)
from zlc_pulse import CompiledPulseArtifact, PulseExecutionForm

from ._coordination import (
    run_cleanup_steps,
    validate_single_trigger_capture_binding,
)
from .capture_plan import CompiledCaptureCellPlan
from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
    PulseTerminalAck,
    validate_pulse_terminal_for_artifact,
)


@dataclass(frozen=True)
class TriggeredOccupancySpec:
    occupancy: OccupancyPipelineSpec
    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str
    cell_plan: CompiledCaptureCellPlan

    def __post_init__(self) -> None:
        if not isinstance(self.occupancy, OccupancyPipelineSpec):
            raise TypeError("occupancy must be OccupancyPipelineSpec")
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        _text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        contract = self.occupancy.measurement.capture_contract
        schedule = validate_single_trigger_capture_binding(
            capture_spec=self.occupancy.measurement.capture_spec,
            contract=contract,
            artifact=self.pulse_request.artifact,
            trigger_channel=self.trigger_channel,
        )
        self.cell_plan.validate_against(
            self.pulse_request.artifact,
            contract.dataset_schema,
        )
        if self.cell_plan.trigger_channel != self.trigger_channel:
            raise ValueError("occupancy cell plan trigger channel differs")
        if self.cell_plan.expected_cells != contract.expected_cells:
            raise ValueError("occupancy cell plan differs from capture schedule")
        if schedule.total != contract.total_events:
            raise ValueError(
                "compiled trigger count differs from occupancy camera event budget"
            )


_TRIGGERED_OCCUPANCY_RESULT_TOKEN = object()


class TriggeredOccupancyPipelineResult:
    """Process-local occupancy result plus complete pulse/cell provenance."""

    __slots__ = (
        "_authority",
        "_occupancy",
        "_pulse_terminal",
        "_trigger_channel",
        "_compiled_artifact",
        "_cell_plan",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("TriggeredOccupancyPipelineResult is final")

    def __init__(
        self,
        authority: object,
        *,
        occupancy: OccupancyPipelineResult,
        pulse_terminal: PulseTerminalAck,
        trigger_channel: str,
        compiled_artifact: CompiledPulseArtifact,
        cell_plan: CompiledCaptureCellPlan,
    ) -> None:
        if authority is not _TRIGGERED_OCCUPANCY_RESULT_TOKEN:
            raise PermissionError(
                "TriggeredOccupancyPipelineResult can only be minted by its compiler"
            )
        if type(occupancy) is not OccupancyPipelineResult:
            raise TypeError("occupancy must be an exact OccupancyPipelineResult")
        if not isinstance(pulse_terminal, PulseTerminalAck):
            raise TypeError("pulse_terminal must be PulseTerminalAck")
        _text(trigger_channel, "trigger_channel")
        if not isinstance(compiled_artifact, CompiledPulseArtifact):
            raise TypeError("compiled_artifact must be CompiledPulseArtifact")
        if not isinstance(cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        if (
            cell_plan.compiled_pulse_artifact_digest
            != compiled_artifact.fingerprint
        ):
            raise ValueError("occupancy cell plan and compiled artifact differ")
        if cell_plan.execution_form is not compiled_artifact.execution_form:
            raise ValueError("occupancy cell plan and execution form differ")
        if cell_plan.trigger_channel != trigger_channel:
            raise ValueError("occupancy cell plan and trigger channel differ")
        source_schema = occupancy.source_dataset_schema
        physical_facts = occupancy.camera_capability_evidence.physical_facts
        physical_facts.require_single_capture_trigger_channel(trigger_channel)
        cell_plan.validate_against(compiled_artifact, source_schema)
        events = occupancy.dataset.events
        event_cells = tuple(cell for cell, _metadata in events)
        if event_cells != cell_plan.expected_cells:
            raise ValueError("occupancy event order differs from pulse cell plan")
        if (
            dataset_cell_permutation_digest(source_schema, event_cells)
            != cell_plan.cell_permutation_digest
        ):
            raise ValueError("occupancy event permutation digest differs from cell plan")
        validate_pulse_terminal_for_artifact(
            pulse_terminal,
            compiled_artifact,
        )
        counts = dict(
            pulse_terminal.expected_trigger_counts_from_completed_schedule
        )
        if trigger_channel not in counts:
            raise ValueError("pulse terminal omits the occupancy trigger channel")
        expected = counts[trigger_channel]
        terminal = occupancy.capture_terminal
        if not (
            expected
            == cell_plan.total_events
            == terminal.produced_count
            == terminal.drained_count
            == len(events)
        ):
            raise RuntimeError(
                "pulse, camera, cell-plan, and occupancy terminal counts differ"
            )
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_occupancy", occupancy)
        object.__setattr__(self, "_pulse_terminal", pulse_terminal)
        object.__setattr__(self, "_trigger_channel", trigger_channel)
        object.__setattr__(self, "_compiled_artifact", compiled_artifact)
        object.__setattr__(self, "_cell_plan", cell_plan)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TriggeredOccupancyPipelineResult is immutable")

    def __reduce__(self):
        raise TypeError("TriggeredOccupancyPipelineResult is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("TriggeredOccupancyPipelineResult is process-local")

    @property
    def occupancy(self) -> OccupancyPipelineResult:
        return self._occupancy

    @property
    def pulse_terminal(self) -> PulseTerminalAck:
        return self._pulse_terminal

    @property
    def trigger_channel(self) -> str:
        return self._trigger_channel

    @property
    def compiled_artifact(self) -> CompiledPulseArtifact:
        return self._compiled_artifact

    @property
    def cell_plan(self) -> CompiledCaptureCellPlan:
        return self._cell_plan

    @property
    def dataset(self):
        return self.occupancy.dataset

    @property
    def capture_terminal(self):
        return self.occupancy.capture_terminal

    @property
    def compiled_artifact_digest(self) -> str:
        return self.compiled_artifact.fingerprint

    @property
    def source_document_digest(self) -> str:
        return self.compiled_artifact.source_document_digest

    @property
    def execution_form(self) -> PulseExecutionForm:
        return self.compiled_artifact.execution_form


@dataclass
class _PreparedTriggeredOccupancy:
    occupancy: _OccupancyTransaction
    pulse: PulseSession


_EXECUTED_TRIGGERED_OCCUPANCY_TOKEN = object()


class _ExecutedTriggeredOccupancy:
    """Run-bound execute value that cannot splice pulse and camera epochs."""

    __slots__ = (
        "_authority",
        "_run_id",
        "_pulse_session_id",
        "_occupancy",
        "_pulse_terminal",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("_ExecutedTriggeredOccupancy is final")

    def __init__(
        self,
        authority: object,
        *,
        run_id: str,
        pulse_session_id: str,
        occupancy: _ExecutedOccupancyPipeline,
        pulse_terminal: PulseTerminalAck,
    ) -> None:
        if authority is not _EXECUTED_TRIGGERED_OCCUPANCY_TOKEN:
            raise PermissionError(
                "_ExecutedTriggeredOccupancy can only be minted by its coordinator"
            )
        _text(run_id, "run_id")
        _text(pulse_session_id, "pulse_session_id")
        if not isinstance(occupancy, _ExecutedOccupancyPipeline):
            raise TypeError("occupancy must be _ExecutedOccupancyPipeline")
        if not isinstance(pulse_terminal, PulseTerminalAck):
            raise TypeError("pulse_terminal must be PulseTerminalAck")
        if occupancy.pipeline.run_id != run_id:
            raise ValueError("executed occupancy belongs to another Run")
        if pulse_terminal.session_id != pulse_session_id:
            raise ValueError("pulse terminal belongs to another pulse session")
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_run_id", run_id)
        object.__setattr__(self, "_pulse_session_id", pulse_session_id)
        object.__setattr__(self, "_occupancy", occupancy)
        object.__setattr__(self, "_pulse_terminal", pulse_terminal)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("_ExecutedTriggeredOccupancy is immutable")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def pulse_session_id(self) -> str:
        return self._pulse_session_id

    @property
    def occupancy(self) -> _ExecutedOccupancyPipeline:
        return self._occupancy

    @property
    def pulse_terminal(self) -> PulseTerminalAck:
        return self._pulse_terminal


def compile_triggered_occupancy_pipeline(
    spec: TriggeredOccupancySpec,
) -> RunPlan:
    """Compile prepare -> arm full chain -> FIRE -> exact dual-field result."""

    if not isinstance(spec, TriggeredOccupancySpec):
        raise TypeError("spec must be TriggeredOccupancySpec")
    camera_port = spec.occupancy.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(context: RunContext) -> _PreparedTriggeredOccupancy:
        pulse = pulse_port.open_session(spec.pulse_request)
        try:
            occupancy = _open_occupancy_transaction(spec.occupancy, context)
        except BaseException:
            pulse.fail()
            raise
        return _PreparedTriggeredOccupancy(occupancy, pulse)

    def execute(
        context: RunContext,
        prepared: _PreparedTriggeredOccupancy,
    ) -> _ExecutedTriggeredOccupancy:
        try:
            # Preparing the pulse only loads frozen words.  The full camera ->
            # processor -> builder reservation is armed before the only FIRE.
            prepared.pulse.prepare(context)
            prepared.occupancy.start(context)
            prepared.pulse.fire(context)
            prepared.occupancy.capture_all(context)
            pulse_terminal = prepared.pulse.complete(context)
            if not prepared.pulse.owns_terminal(pulse_terminal):
                raise PermissionError(
                    "pulse terminal was not minted by the current pulse session"
                )
            occupancy = prepared.occupancy.complete(context)
            return _ExecutedTriggeredOccupancy(
                _EXECUTED_TRIGGERED_OCCUPANCY_TOKEN,
                run_id=context.run_id.value,
                pulse_session_id=prepared.pulse.session_id,
                occupancy=occupancy,
                pulse_terminal=pulse_terminal,
            )
        except BaseException as error:
            prepared.occupancy.fail(error)
            prepared.pulse.fail()
            raise

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredOccupancy | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        # Stop future trigger edges first.  The second step still runs if the
        # sequencer cleanup returns/throws an error.
        return run_cleanup_steps(
            lambda: prepared.pulse.cleanup(context),
            lambda: prepared.occupancy.cleanup(context),
        )

    def finalize(
        context: PostSafetyContext,
        executed: _ExecutedTriggeredOccupancy,
    ) -> TriggeredOccupancyPipelineResult:
        if type(executed) is not _ExecutedTriggeredOccupancy:
            raise TypeError("triggered occupancy finalize requires its executed value")
        if executed.run_id != context.run_id.value:
            raise ValueError("executed occupancy result belongs to another Run")
        context.checkpoint()
        occupancy = _finalize_occupancy_result(context, executed.occupancy)
        context.checkpoint()
        return TriggeredOccupancyPipelineResult(
            _TRIGGERED_OCCUPANCY_RESULT_TOKEN,
            occupancy=occupancy,
            pulse_terminal=executed.pulse_terminal,
            trigger_channel=spec.trigger_channel,
            compiled_artifact=spec.pulse_request.artifact,
            cell_plan=spec.cell_plan,
        )

    return RunPlan(
        name=spec.occupancy.name,
        mode=RunMode.FINITE_EXACT,
        resource_claims=(pulse_port.resource_claim, camera_port.resource_claim),
        hazard_claims=(pulse_port.hazard_claim, camera_port.hazard_claim),
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
