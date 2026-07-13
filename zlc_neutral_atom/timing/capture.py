"""One flat RunPlan coordinating finite FPGA execution and exact camera capture."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text

from zlc_neutral_atom.runtime import (
    CleanupReport,
    ExactCaptureTransaction,
    MinimalPipelineSpec,
    PipelineResult,
    PostSafetyContext,
    RunContext,
    RunMode,
    RunPlan,
    open_exact_capture,
)
from zlc_pulse import CompiledPulseArtifact, PulseExecutionForm

from .capture_plan import CompiledCaptureCellPlan
from ._coordination import (
    run_cleanup_steps,
    validate_single_trigger_capture_binding,
)

from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
    PulseTerminalAck,
    validate_pulse_terminal_for_artifact,
)


@dataclass(frozen=True)
class TriggeredCaptureSpec:
    capture: MinimalPipelineSpec
    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str
    cell_plan: CompiledCaptureCellPlan

    def __post_init__(self) -> None:
        if not isinstance(self.capture, MinimalPipelineSpec):
            raise TypeError("capture must be MinimalPipelineSpec")
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        canonical_text(self.trigger_channel, "trigger_channel")
        if not isinstance(self.cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        contract = self.capture.measurement.capture_contract
        schedule = validate_single_trigger_capture_binding(
            capture_spec=self.capture.measurement.capture_spec,
            contract=contract,
            artifact=self.pulse_request.artifact,
            trigger_channel=self.trigger_channel,
        )
        plan = self.cell_plan
        if plan.trigger_channel != self.trigger_channel:
            raise ValueError("capture cell plan trigger channel differs")
        plan.validate_against(
            self.pulse_request.artifact,
            contract.dataset_schema,
        )
        if plan.expected_cells != contract.expected_cells:
            raise ValueError("capture cell plan permutation differs from capture contract")
        expected = self.capture.measurement.capture_contract.total_events
        if schedule.total != expected or plan.total_events != expected:
            raise ValueError(
                f"compiled trigger count {schedule.total} differs from "
                f"camera event budget {expected}"
            )

    @property
    def trigger_total(self) -> int:
        return next(
            schedule.total
            for schedule in self.pulse_request.artifact.trigger_schedules
            if schedule.channel == self.trigger_channel
        )


_TRIGGERED_RESULT_TOKEN = object()


class TriggeredPipelineResult:
    __slots__ = (
        "_authority",
        "_capture",
        "_pulse_session_id",
        "_pulse_terminal",
        "_trigger_channel",
        "_compiled_artifact",
        "_cell_plan",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("TriggeredPipelineResult is final")

    def __init__(
        self,
        authority: object,
        *,
        capture: PipelineResult,
        pulse_session_id: str,
        pulse_terminal: PulseTerminalAck,
        trigger_channel: str,
        compiled_artifact: CompiledPulseArtifact,
        cell_plan: CompiledCaptureCellPlan,
    ) -> None:
        if authority is not _TRIGGERED_RESULT_TOKEN:
            raise PermissionError(
                "TriggeredPipelineResult can only be minted by its compiler"
            )
        if not isinstance(capture, PipelineResult):
            raise TypeError("capture must be PipelineResult")
        canonical_text(pulse_session_id, "pulse_session_id")
        if not isinstance(pulse_terminal, PulseTerminalAck):
            raise TypeError("pulse_terminal must be PulseTerminalAck")
        if pulse_terminal.session_id != pulse_session_id:
            raise ValueError("pulse terminal belongs to another pulse session")
        if not isinstance(compiled_artifact, CompiledPulseArtifact):
            raise TypeError("compiled_artifact must be CompiledPulseArtifact")
        if not isinstance(cell_plan, CompiledCaptureCellPlan):
            raise TypeError("cell_plan must be CompiledCaptureCellPlan")
        if (
            cell_plan.compiled_pulse_artifact_digest
            != compiled_artifact.fingerprint
        ):
            raise ValueError("cell plan and compiled artifact digest differ")
        if cell_plan.execution_form is not compiled_artifact.execution_form:
            raise ValueError("cell plan and execution form differ")
        if cell_plan.trigger_channel != trigger_channel:
            raise ValueError("cell plan and trigger channel differ")
        evidence = capture.camera_capability_evidence
        if evidence is None:
            raise ValueError(
                "triggered result requires broker-attested camera physical facts"
            )
        evidence.physical_facts.require_single_capture_trigger_channel(
            trigger_channel
        )
        cell_plan.validate_against(compiled_artifact, capture.dataset.block.schema)
        if capture.source_cell_schedule != cell_plan.expected_cells:
            raise ValueError("cell plan and captured source schedule differ")
        if (
            cell_plan.cell_permutation_digest
            != capture.dataset.provenance.join_plan_digest
        ):
            raise ValueError("cell plan and sealed dataset permutation differ")
        validate_pulse_terminal_for_artifact(
            pulse_terminal,
            compiled_artifact,
        )
        counts = dict(
            pulse_terminal.expected_trigger_counts_from_completed_schedule
        )
        if trigger_channel not in counts:
            raise ValueError("pulse terminal omits the bound camera trigger channel")
        expected = counts[trigger_channel]
        if (
            capture.capture_terminal.produced_count != expected
            or capture.capture_terminal.drained_count != expected
        ):
            raise RuntimeError(
                "camera terminal count differs from completed pulse schedule"
            )
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_capture", capture)
        object.__setattr__(self, "_pulse_session_id", pulse_session_id)
        object.__setattr__(self, "_pulse_terminal", pulse_terminal)
        object.__setattr__(self, "_trigger_channel", trigger_channel)
        object.__setattr__(self, "_compiled_artifact", compiled_artifact)
        object.__setattr__(self, "_cell_plan", cell_plan)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("TriggeredPipelineResult is immutable")

    def __reduce__(self):
        raise TypeError("TriggeredPipelineResult is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("TriggeredPipelineResult is process-local")

    @property
    def capture(self) -> PipelineResult:
        return self._capture

    @property
    def pulse_session_id(self) -> str:
        return self._pulse_session_id

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
        return self.capture.dataset

    @property
    def capture_terminal(self):
        return self.capture.capture_terminal

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
class _PreparedTriggeredCapture:
    capture: ExactCaptureTransaction
    pulse: PulseSession


def compile_triggered_pipeline(spec: TriggeredCaptureSpec) -> RunPlan:
    """Compile prepare→camera arm→one FPGA FIRE→drain→terminal into one Run."""

    if not isinstance(spec, TriggeredCaptureSpec):
        raise TypeError("spec must be TriggeredCaptureSpec")
    camera_port = spec.capture.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(context: RunContext) -> _PreparedTriggeredCapture:
        capture = open_exact_capture(spec.capture, context)
        try:
            pulse = pulse_port.open_session(spec.pulse_request)
            return _PreparedTriggeredCapture(capture, pulse)
        except BaseException as error:
            capture.abort_preflight(error)
            raise

    def execute(
        context: RunContext,
        prepared: _PreparedTriggeredCapture,
    ) -> TriggeredPipelineResult:
        try:
            # Loading the frozen table cannot emit an edge.  The camera is armed
            # only after prepare succeeds, and FIRE is the next hardware action.
            prepared.pulse.prepare(context)
            prepared.capture.start(context)
            prepared.pulse.fire(context)
            prepared.capture.capture_all(context)
            pulse_terminal = prepared.pulse.complete(context)
            if not prepared.pulse.owns_terminal(pulse_terminal):
                raise PermissionError(
                    "pulse terminal was not minted by the current pulse session"
                )
            capture_result = prepared.capture.complete(context)
            return TriggeredPipelineResult(
                _TRIGGERED_RESULT_TOKEN,
                capture=capture_result,
                pulse_session_id=prepared.pulse.session_id,
                pulse_terminal=pulse_terminal,
                trigger_channel=spec.trigger_channel,
                compiled_artifact=spec.pulse_request.artifact,
                cell_plan=spec.cell_plan,
            )
        except BaseException as error:
            prepared.capture.fail(error)
            prepared.pulse.fail()
            raise

    def cleanup(
        context: RunContext,
        prepared: _PreparedTriggeredCapture | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        # On failure/cancel, stop new hardware edges before terminating the
        # camera session.  On success both calls are idempotent terminal checks.
        return run_cleanup_steps(
            lambda: prepared.pulse.cleanup(context),
            lambda: prepared.capture.cleanup(context),
        )

    def finalize(
        context: PostSafetyContext,
        result: TriggeredPipelineResult,
    ) -> TriggeredPipelineResult:
        if type(result) is not TriggeredPipelineResult:
            raise TypeError("triggered capture finalize requires its compiler result")
        if result.capture.run_id != context.run_id.value:
            raise ValueError("triggered capture result belongs to another Run")
        context.checkpoint()
        return result

    return RunPlan(
        name=spec.capture.name,
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
        timeout_seconds=spec.capture.timeout_seconds,
        requires_final_commit=False,
    )


__all__ = [
    "compile_triggered_pipeline",
    "TriggeredCaptureSpec",
    "TriggeredPipelineResult",
]
