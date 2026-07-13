"""Small safety helpers shared by flat hardware coordinators."""

from __future__ import annotations

from collections.abc import Callable

from zlc_neutral_atom.acquisition import (
    CameraAcquisitionMode,
    decode_camera_capture_spec,
)
from zlc_neutral_atom.runtime import (
    CaptureStreamContract,
    CleanupReport,
    FrozenCaptureSpec,
)
from zlc_pulse import CompiledPulseArtifact, DigitalTriggerSchedule


def validate_single_trigger_capture_binding(
    *,
    capture_spec: FrozenCaptureSpec,
    contract: CaptureStreamContract,
    artifact: CompiledPulseArtifact,
    trigger_channel: str,
) -> DigitalTriggerSchedule:
    """Validate the exact single-wire camera/pulse join shared by coordinators."""

    if not isinstance(capture_spec, FrozenCaptureSpec):
        raise TypeError("capture_spec must be FrozenCaptureSpec")
    if not isinstance(contract, CaptureStreamContract):
        raise TypeError("contract must be CaptureStreamContract")
    if not isinstance(artifact, CompiledPulseArtifact):
        raise TypeError("artifact must be CompiledPulseArtifact")
    if not isinstance(trigger_channel, str) or not trigger_channel:
        raise ValueError("trigger_channel must be non-empty text")
    camera_spec = decode_camera_capture_spec(capture_spec)
    if camera_spec.mode is not CameraAcquisitionMode.EXTERNAL_TRIGGERED:
        raise ValueError("exact triggered capture requires an external-trigger camera")
    schedules = tuple(
        schedule
        for schedule in artifact.trigger_schedules
        if schedule.channel == trigger_channel
    )
    if len(schedules) != 1:
        raise ValueError(
            "exact triggered capture requires exactly one compiled camera schedule"
        )
    evidence = contract.capability.camera_capability_evidence
    if evidence is None:
        raise ValueError(
            "exact triggered capture requires broker-attested camera physical facts"
        )
    evidence.physical_facts.require_single_capture_trigger_channel(
        trigger_channel
    )
    return schedules[0]


def run_cleanup_steps(
    *steps: Callable[[], CleanupReport],
) -> CleanupReport:
    """Run every cleanup step in order, even when an earlier step fails.

    A thrown cleanup exception is evidence, not permission to skip the next
    physical resource.  Converting it into ``CleanupReport.errors`` lets the
    Run safety gate retain proofs from the other steps while still failing
    closed for the resource whose cleanup could not be proved.
    """

    safety_proofs = []
    decisions = []
    errors: list[BaseException] = []
    for step in steps:
        try:
            report = step()
            if not isinstance(report, CleanupReport):
                raise TypeError("hardware cleanup step must return CleanupReport")
        except BaseException as error:
            errors.append(error)
            continue
        safety_proofs.extend(report.safety_proofs)
        decisions.extend(report.decisions)
        errors.extend(report.errors)
    return CleanupReport(
        safety_proofs=tuple(safety_proofs),
        decisions=tuple(decisions),
        errors=tuple(errors),
    )


__all__: list[str] = []
