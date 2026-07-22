"""Small safety helpers shared by flat hardware coordinators."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.runtime.capture import (
    CameraCaptureContract,
    FrozenCaptureSpec,
)
from zlc_neutral_atom.runtime.cleanup import CleanupReport
from zlc_neutral_atom.runtime.run import RunContext
from zlc_neutral_atom.runtime._failure import record_secondary_failure
from zlc_pulse import DigitalTriggerSchedule

from .lineage import PulseCaptureBinding
from .pulse import PulseSession, PulseTerminalAck


_CaptureCompletionT = TypeVar("_CaptureCompletionT")


class _AutonomousExactCapture(Protocol[_CaptureCompletionT]):
    def start(self, context: RunContext) -> None: ...

    def capture_all(self, context: RunContext) -> None: ...

    def complete(self, context: RunContext) -> _CaptureCompletionT: ...

    def fail(self, error: BaseException) -> None: ...


def execute_autonomous_single_fire(
    context: RunContext,
    *,
    pulse: PulseSession,
    capture: _AutonomousExactCapture[_CaptureCompletionT],
) -> tuple[_CaptureCompletionT, PulseTerminalAck]:
    """Run the one shared hardware sequence for an autonomous exact capture."""

    try:
        pulse.prepare(context)
        capture.start(context)
        pulse.fire(context)
        capture.capture_all(context)
        terminal = pulse.complete(context)
        if not pulse.owns_terminal(terminal):
            raise PermissionError("pulse terminal was not minted by this session")
        return capture.complete(context), terminal
    except BaseException as primary:
        for label, fail in (
            ("capture poison", lambda: capture.fail(primary)),
            ("pulse poison", pulse.fail),
        ):
            try:
                fail()
            except BaseException as secondary:
                record_secondary_failure(primary, f"{label} also failed", secondary)
        raise


def validate_single_trigger_capture_binding(
    *,
    capture_spec: FrozenCaptureSpec,
    contract: CameraCaptureContract,
    pulse_binding: PulseCaptureBinding,
) -> DigitalTriggerSchedule:
    """Validate the exact single-wire camera/pulse join shared by coordinators."""

    if not isinstance(capture_spec, FrozenCaptureSpec):
        raise TypeError("capture_spec must be FrozenCaptureSpec")
    if not isinstance(contract, CameraCaptureContract):
        raise TypeError("contract must be CameraCaptureContract")
    if not isinstance(pulse_binding, PulseCaptureBinding):
        raise TypeError("pulse_binding must be PulseCaptureBinding")
    artifact = pulse_binding.compiled_artifact
    trigger_channel = pulse_binding.trigger_channel
    cell_plan = pulse_binding.cell_plan
    camera_spec = decode_camera_capture_spec(capture_spec)
    if camera_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError("exact triggered capture requires an external-trigger camera")
    evidence = contract.capability.camera_capability_evidence
    evidence.physical_facts.require_single_capture_trigger_channel(
        trigger_channel
    )
    if evidence.exact_external_trigger_qualification_digest is None:
        raise ValueError(
            "exact triggered capture requires E0-qualified ordered one-frame-per-trigger evidence"
        )
    schedule = pulse_binding.trigger_schedule
    minimum_interval_ticks = schedule.minimum_interval_ticks
    required_interval = (
        evidence.physical_facts.required_external_trigger_interval_seconds
    )
    if required_interval is None:
        raise ValueError(
            "exact triggered capture requires a qualified safe external-trigger interval"
        )
    if minimum_interval_ticks is not None:
        actual_interval = minimum_interval_ticks / artifact.target_ir.clock_hz
        if actual_interval < required_interval:
            raise ValueError(
                "compiled camera trigger interval "
                f"{actual_interval:.12g} s is shorter than the broker-attested "
                "required external trigger interval "
                f"{required_interval:.12g} s"
            )
    cell_plan.validate_dataset_schema(contract.dataset_schema)
    if not cell_plan.cell_schedule.same_order_as(contract.cell_schedule):
        raise ValueError("capture cell plan permutation differs from capture contract")
    expected = contract.total_events
    if pulse_binding.expected_trigger_count != expected:
        raise ValueError(
            f"pulse capture binding count differs from expected camera event count {expected}"
        )
    return schedule


def run_cleanup_steps(
    *steps: Callable[[], CleanupReport],
) -> CleanupReport:
    """Run every cleanup step in order, even when an earlier step fails.

    A thrown cleanup exception does not permit skipping the next physical
    resource.  The controller performs the authoritative safe-state readback;
    this helper only aggregates cleanup execution errors.
    """

    errors: list[BaseException] = []
    for step in steps:
        try:
            report = step()
            if not isinstance(report, CleanupReport):
                raise TypeError("hardware cleanup step must return CleanupReport")
        except BaseException as error:
            errors.append(error)
            continue
        errors.extend(report.errors)
    return CleanupReport.complete(errors=tuple(errors))


__all__: list[str] = []
