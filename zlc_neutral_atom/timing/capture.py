"""One flat RunPlan coordinating finite FPGA execution and exact camera capture."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.runtime import (
    CleanupReport,
    ExactCaptureTransaction,
    MinimalPipelineSpec,
    PipelineResult,
    RunContext,
    RunMode,
    RunPlan,
    open_exact_capture,
)
from zlc_pulse import PulseExecutionForm

from .pulse import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
    PulseSession,
    PulseTerminalAck,
)


@dataclass(frozen=True)
class TriggeredCaptureSpec:
    capture: MinimalPipelineSpec
    pulse_port: BoundPulsePort
    pulse_request: FinitePulseExecutionRequest
    trigger_channel: str

    def __post_init__(self) -> None:
        if not isinstance(self.capture, MinimalPipelineSpec):
            raise TypeError("capture must be MinimalPipelineSpec")
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.pulse_request, FinitePulseExecutionRequest):
            raise TypeError("pulse_request must be FinitePulseExecutionRequest")
        if (
            not isinstance(self.trigger_channel, str)
            or not self.trigger_channel
            or self.trigger_channel.strip() != self.trigger_channel
        ):
            raise ValueError("trigger_channel must be canonical non-empty text")
        camera_spec = decode_camera_capture_spec(
            self.capture.measurement.capture_spec
        )
        if camera_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
            raise ValueError("triggered capture requires an external-trigger camera")
        schedules = tuple(
            schedule
            for schedule in self.pulse_request.artifact.trigger_schedules
            if schedule.channel == self.trigger_channel
        )
        if len(schedules) != 1:
            raise ValueError("triggered capture requires exactly one compiled camera schedule")
        expected = self.capture.measurement.capture_contract.total_events
        if schedules[0].total != expected:
            raise ValueError(
                f"compiled trigger count {schedules[0].total} differs from "
                f"camera event budget {expected}"
            )

    @property
    def trigger_total(self) -> int:
        return next(
            schedule.total
            for schedule in self.pulse_request.artifact.trigger_schedules
            if schedule.channel == self.trigger_channel
        )


@dataclass(frozen=True)
class TriggeredPipelineResult:
    capture: PipelineResult
    pulse_terminal: PulseTerminalAck
    trigger_channel: str
    compiled_artifact_digest: str
    source_document_digest: str
    execution_form: PulseExecutionForm

    def __post_init__(self) -> None:
        if not isinstance(self.capture, PipelineResult):
            raise TypeError("capture must be PipelineResult")
        if not isinstance(self.pulse_terminal, PulseTerminalAck):
            raise TypeError("pulse_terminal must be PulseTerminalAck")
        for value, field in (
            (self.compiled_artifact_digest, "compiled_artifact_digest"),
            (self.source_document_digest, "source_document_digest"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{field} must be a lowercase SHA-256 digest")
        if not isinstance(self.execution_form, PulseExecutionForm):
            raise TypeError("execution_form must be PulseExecutionForm")
        if self.pulse_terminal.artifact_digest != self.compiled_artifact_digest:
            raise ValueError("pulse terminal and compiled artifact digest differ")
        counts = dict(self.pulse_terminal.completed_schedule_trigger_counts)
        if self.trigger_channel not in counts:
            raise ValueError("pulse terminal omits the bound camera trigger channel")
        expected = counts[self.trigger_channel]
        if (
            self.capture.capture_terminal.produced_count != expected
            or self.capture.capture_terminal.drained_count != expected
        ):
            raise RuntimeError("camera terminal count differs from completed pulse schedule")

    @property
    def dataset(self):
        return self.capture.dataset

    @property
    def capture_terminal(self):
        return self.capture.capture_terminal


@dataclass
class _PreparedTriggeredCapture:
    capture: ExactCaptureTransaction
    pulse: PulseSession


def _merge_cleanup(*reports: CleanupReport) -> CleanupReport:
    return CleanupReport(
        safety_proofs=tuple(
            proof for report in reports for proof in report.safety_proofs
        ),
        decisions=tuple(
            decision for report in reports for decision in report.decisions
        ),
        errors=tuple(error for report in reports for error in report.errors),
    )


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
            capture_result = prepared.capture.complete(context)
            return TriggeredPipelineResult(
                capture_result,
                pulse_terminal,
                spec.trigger_channel,
                spec.pulse_request.artifact_digest,
                spec.pulse_request.artifact.source_document_digest,
                spec.pulse_request.artifact.execution_form,
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
            return _merge_cleanup(
                pulse_port.verify_idle(context),
                camera_port.verify_idle(context),
            )
        # On failure/cancel, stop new hardware edges before terminating the
        # camera session.  On success both calls are idempotent terminal checks.
        pulse_report = prepared.pulse.cleanup(context)
        camera_report = prepared.capture.cleanup(context)
        return _merge_cleanup(pulse_report, camera_report)

    return RunPlan(
        name=spec.capture.name,
        mode=RunMode.FINITE_EXACT,
        resource_claims=(pulse_port.resource_claim, camera_port.resource_claim),
        hazard_claims=(pulse_port.hazard_claim, camera_port.hazard_claim),
        bound_devices=(pulse_port.device, camera_port.device),
        preflight=preflight,
        execute=execute,
        cleanup=cleanup,
        finalize=lambda _context, result: result,
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
