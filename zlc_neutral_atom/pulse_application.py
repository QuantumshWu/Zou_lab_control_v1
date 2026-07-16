"""Declarative pulse-only application seam shared by notebook and Workbench."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
from typing import Callable

from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import (
    PostSafetyContext,
    RunContext,
    RunHandle,
    RunPlan,
)
from zlc_neutral_atom.timing.pulse import (
    BoundPulsePort,
    ContinuousPulseExecutionRequest,
    FinitePulseExecutionRequest,
    PulseSession,
    PulseTerminalEvidenceKind,
)
from zlc_pulse import (
    PulseDocument,
    PulseExecutionForm,
    PulseTarget,
    bind_pulse_document_target,
    compile_pulse_artifact,
)
from zlc_storage import canonical_text, positive_real


@dataclass(frozen=True, slots=True)
class PulseTargetDescriptor:
    """Read-only target facts; it contains no hardware drive capability."""

    sequencer_ref: DeviceRef
    target: PulseTarget
    clock_hz: float
    geometry_fingerprint: int
    resident_scan_point_capacity: int

    def __post_init__(self) -> None:
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if not isinstance(self.target, PulseTarget):
            raise TypeError("target must be PulseTarget")
        clock_hz = positive_real(self.clock_hz, "clock_hz")
        object.__setattr__(self, "clock_hz", clock_hz)
        if (
            isinstance(self.geometry_fingerprint, bool)
            or not isinstance(self.geometry_fingerprint, int)
            or not 0 <= self.geometry_fingerprint <= 0xFFFFFFFF
        ):
            raise ValueError("geometry_fingerprint must be an unsigned 32-bit integer")
        if (
            isinstance(self.resident_scan_point_capacity, bool)
            or not isinstance(self.resident_scan_point_capacity, int)
            or self.resident_scan_point_capacity < 1
        ):
            raise ValueError("resident_scan_point_capacity must be positive")

    @property
    def time_step_ns(self) -> float:
        return 1e9 / self.clock_hz


@dataclass(frozen=True, slots=True)
class PulseRunRequest:
    """One immutable run intent; editable state is frozen before construction."""

    document: PulseDocument
    execution_form: PulseExecutionForm
    sequencer_ref: DeviceRef
    timeout_seconds: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if self.document.api_parameters:
            raise ValueError(
                "hardware Pulse Run has unresolved API parameters; "
                "call resolve_api_parameters with explicit values first"
            )
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.execution_form is PulseExecutionForm.STATIC_REFERENCE_POINT:
            raise ValueError("STATIC_REFERENCE_POINT is a preview form, not a hardware Run")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if self.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
            if self.timeout_seconds is not None:
                raise ValueError("continuous pulse execution ends by cancellation, not timeout")
        else:
            if self.timeout_seconds is None:
                raise ValueError("finite pulse execution requires a timeout")
            object.__setattr__(
                self,
                "timeout_seconds",
                positive_real(self.timeout_seconds, "timeout_seconds"),
            )


@dataclass(frozen=True, slots=True)
class PulseRunDescriptor:
    name: str
    sequencer_role: str
    execution_form: PulseExecutionForm
    artifact_digest: str
    logical_duration_seconds: float
    scan_point_count: int
    resource_claim: str

    def __post_init__(self) -> None:
        canonical_text(self.name, "pulse run name")
        canonical_text(self.sequencer_role, "sequencer_role")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        canonical_text(self.artifact_digest, "artifact_digest")
        duration = float(self.logical_duration_seconds)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("logical_duration_seconds must be finite and positive")
        object.__setattr__(self, "logical_duration_seconds", duration)
        if (
            isinstance(self.scan_point_count, bool)
            or not isinstance(self.scan_point_count, int)
            or self.scan_point_count < 0
        ):
            raise ValueError("scan_point_count must be a non-negative integer")
        canonical_text(self.resource_claim, "resource_claim")


@dataclass(frozen=True, slots=True)
class PulseRunResult:
    run_id: str
    execution_form: PulseExecutionForm
    artifact_digest: str
    terminal_evidence_kind: PulseTerminalEvidenceKind

    def __post_init__(self) -> None:
        canonical_text(self.run_id, "run_id")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        canonical_text(self.artifact_digest, "artifact_digest")
        if not isinstance(self.terminal_evidence_kind, PulseTerminalEvidenceKind):
            raise TypeError("terminal_evidence_kind must be PulseTerminalEvidenceKind")


class PreparedPulseExecution:
    """One-shot command hiding the generation-bound Port and RunPlan."""

    __slots__ = ("_descriptor", "_lock", "_plan", "_start_run", "_started")

    def __init__(
        self,
        plan: RunPlan,
        start_run: Callable[[RunPlan], RunHandle],
        descriptor: PulseRunDescriptor,
    ) -> None:
        if not isinstance(plan, RunPlan):
            raise TypeError("plan must be RunPlan")
        if not callable(start_run):
            raise TypeError("start_run must be callable")
        if not isinstance(descriptor, PulseRunDescriptor):
            raise TypeError("descriptor must be PulseRunDescriptor")
        self._plan = plan
        self._start_run = start_run
        self._descriptor = descriptor
        self._lock = threading.Lock()
        self._started = False

    @property
    def descriptor(self) -> PulseRunDescriptor:
        return self._descriptor

    def start(self) -> RunHandle:
        with self._lock:
            if self._started:
                raise RuntimeError("PreparedPulseExecution is one-shot")
            self._started = True
        return self._start_run(self._plan)


def prepare_pulse_execution(
    request: PulseRunRequest,
    *,
    pulse_port: BoundPulsePort,
    start_run: Callable[[RunPlan], RunHandle],
) -> PreparedPulseExecution:
    """Bind, compile, and freeze one pulse-only Run without touching hardware."""

    if not isinstance(request, PulseRunRequest):
        raise TypeError("request must be PulseRunRequest")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    document = bind_pulse_document_target(
        request.document,
        pulse_port.capability.target,
    )
    artifact = compile_pulse_artifact(
        document,
        clock_hz=pulse_port.capability.clock_hz,
        execution_form=request.execution_form,
        live_target=pulse_port.capability.target,
    )
    if request.execution_form is PulseExecutionForm.CONTINUOUS_MONITOR:
        execution = ContinuousPulseExecutionRequest(document, artifact)
    else:
        execution = FinitePulseExecutionRequest(document, artifact)
    plan = _compile_pulse_run_plan(
        request,
        pulse_port=pulse_port,
        execution=execution,
    )
    descriptor = PulseRunDescriptor(
        f"Pulse {document.name}",
        request.sequencer_ref.role,
        request.execution_form,
        artifact.fingerprint,
        artifact.target_ir.duration_seconds,
        len(artifact.target_ir.scan_points),
        str(pulse_port.resource_claim.key),
    )
    return PreparedPulseExecution(plan, start_run, descriptor)


def _compile_pulse_run_plan(
    request: PulseRunRequest,
    *,
    pulse_port: BoundPulsePort,
    execution: FinitePulseExecutionRequest | ContinuousPulseExecutionRequest,
) -> RunPlan[PulseSession, PulseRunResult, PulseRunResult]:
    def preflight(_context: RunContext) -> PulseSession:
        return pulse_port.open_session(execution)

    def execute(context: RunContext, session: PulseSession) -> PulseRunResult:
        context.checkpoint()
        session.prepare(context)
        context.checkpoint()
        session.fire(context)
        if isinstance(execution, ContinuousPulseExecutionRequest):
            context.set_phase("holding-pulse")
            context.cancellation.wait_requested()
            raise RuntimeError("continuous cancellation wait returned without cancellation")
        terminal = session.complete(context)
        return PulseRunResult(
            context.run_id.value,
            request.execution_form,
            execution.artifact_digest,
            terminal.evidence_kind,
        )

    def cleanup(
        context: RunContext,
        session: PulseSession | None,
        _primary: BaseException | None,
    ) -> CleanupReport:
        return (
            pulse_port.verify_idle(context)
            if session is None
            else session.cleanup(context)
        )

    def finalize(
        context: PostSafetyContext,
        result: PulseRunResult,
    ) -> PulseRunResult:
        if not isinstance(result, PulseRunResult):
            raise TypeError("pulse Run finalized an unexpected result")
        if result.run_id != context.run_id.value:
            raise ValueError("pulse Run result belongs to another Run")
        context.checkpoint()
        return result

    return RunPlan(
        name=f"Pulse {request.document.name}",
        resource_claims=(pulse_port.resource_claim,),
        bound_devices=(pulse_port.device,),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=finalize,
        interrupt_operations=pulse_port.interrupt_operations,
        timeout_seconds=request.timeout_seconds,
        requires_final_commit=False,
    )


__all__ = [
    "PreparedPulseExecution",
    "PulseRunDescriptor",
    "PulseRunRequest",
    "PulseRunResult",
    "PulseTargetDescriptor",
    "prepare_pulse_execution",
]
