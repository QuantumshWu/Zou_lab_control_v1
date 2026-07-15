"""One flat RunPlan coordinating finite FPGA execution and exact camera capture."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field

from zlc_storage import canonical_text

from zlc_neutral_atom.runtime.pipeline import (
    ExactCaptureTransaction,
    MinimalPipelineSpec,
    PipelineResult,
    _open_exact_capture_transaction,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunPlan,
)
from .capture_plan import CompiledCaptureCellPlan
from ._coordination import (
    execute_autonomous_single_fire,
    run_cleanup_steps,
    validate_single_trigger_capture_binding,
)
from .lineage import PulseCaptureBinding, PulseCaptureLineage

from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
)


@dataclass(frozen=True)
class TriggeredCaptureSpec:
    capture: MinimalPipelineSpec
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
        if not isinstance(self.capture, MinimalPipelineSpec):
            raise TypeError("capture must be MinimalPipelineSpec")
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
        contract = self.capture.measurement.capture_contract
        validate_single_trigger_capture_binding(
            capture_spec=self.capture.measurement.capture_spec,
            contract=contract,
            pulse_binding=pulse_binding,
        )
        object.__setattr__(self, "pulse_binding", pulse_binding)


_TRIGGERED_RESULT_TOKEN = object()


class TriggeredPipelineResult:
    __slots__ = (
        "_authority",
        "_capture",
        "_lineage",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("TriggeredPipelineResult is final")

    def __init__(
        self,
        authority: object,
        *,
        capture: PipelineResult,
        lineage: PulseCaptureLineage,
    ) -> None:
        if authority is not _TRIGGERED_RESULT_TOKEN:
            raise PermissionError(
                "TriggeredPipelineResult can only be minted by its compiler"
            )
        if not isinstance(capture, PipelineResult):
            raise TypeError("capture must be PipelineResult")
        if not isinstance(lineage, PulseCaptureLineage):
            raise TypeError("lineage must be PulseCaptureLineage")
        evidence = capture.camera_capability_evidence
        evidence.physical_facts.require_single_capture_trigger_channel(
            lineage.trigger_channel
        )
        lineage.cell_plan.validate_dataset_schema(capture.dataset.block.schema)
        if not capture.source_cell_schedule.same_order_as(
            lineage.cell_plan.cell_schedule
        ):
            raise ValueError("cell plan and captured source schedule differ")
        expected = lineage.expected_trigger_count
        if (
            capture.capture_terminal.produced_count != expected
            or capture.capture_terminal.drained_count != expected
        ):
            raise RuntimeError(
                "camera terminal count differs from completed pulse schedule"
            )
        object.__setattr__(self, "_authority", authority)
        object.__setattr__(self, "_capture", capture)
        object.__setattr__(self, "_lineage", lineage)

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
        return self._lineage.terminal.session_id

    @property
    def lineage(self) -> PulseCaptureLineage:
        return self._lineage


def compile_triggered_pipeline(spec: TriggeredCaptureSpec) -> RunPlan:
    """Compile prepare→camera arm→one FPGA FIRE→drain→terminal into one Run."""

    if not isinstance(spec, TriggeredCaptureSpec):
        raise TypeError("spec must be TriggeredCaptureSpec")
    camera_port = spec.capture.measurement.capture_port
    pulse_port = spec.pulse_port
    if camera_port.device.key == pulse_port.device.key:
        raise ValueError("camera and sequencer must be distinct physical resources")

    def preflight(
        context: RunContext,
    ) -> tuple[ExactCaptureTransaction, PulseSession]:
        capture = _open_exact_capture_transaction(spec.capture, context)
        try:
            pulse = pulse_port.open_session(spec.pulse_request)
            return capture, pulse
        except BaseException as error:
            capture.abort_preflight(error)
            raise

    def execute(
        context: RunContext,
        prepared: tuple[ExactCaptureTransaction, PulseSession],
    ) -> TriggeredPipelineResult:
        capture, pulse = prepared
        capture_result, pulse_terminal = execute_autonomous_single_fire(
            context,
            pulse=pulse,
            capture=capture,
        )
        return TriggeredPipelineResult(
            _TRIGGERED_RESULT_TOKEN,
            capture=capture_result,
            lineage=PulseCaptureLineage(spec.pulse_binding, pulse_terminal),
        )

    def cleanup(
        context: RunContext,
        prepared: tuple[ExactCaptureTransaction, PulseSession] | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        if prepared is None:
            return run_cleanup_steps(
                lambda: pulse_port.verify_idle(context),
                lambda: camera_port.verify_idle(context),
            )
        # On failure/cancel, stop new hardware edges before terminating the
        # camera session.  On success both calls are idempotent terminal checks.
        capture, pulse = prepared
        return run_cleanup_steps(
            lambda: pulse.cleanup(context),
            lambda: capture.cleanup(context),
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
        timeout_seconds=spec.capture.timeout_seconds,
        requires_final_commit=False,
    )


__all__ = [
    "compile_triggered_pipeline",
    "TriggeredCaptureSpec",
    "TriggeredPipelineResult",
]
