"""Physical pulse and camera binding shared only by two-readout Measurements."""

from __future__ import annotations

from dataclasses import replace

from zlc_data import AxisId, AxisSpec, PointLayout
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import CalibrationArtifact, ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.measurement_values import scale_authored_value
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    derive_readout_physical_context,
    digital_outputs_falling_after_period,
)
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.capture.binding import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_pulse import (
    FIELD_DURATION, TIME_UNIT_TO_NS, PulseDocument, PulseExecutionForm,
    PulseFieldRef, RepeatRegion, bind_pulse_document_target,
    expand_autonomous_scan_repeats, freeze_scan_table, replace_pulse_field,
)


def calibrated_probe_seconds(
    document: PulseDocument,
    calibration: CalibrationArtifact,
) -> float:
    """Read the probe-light window from the calibration's physical trace.

    A duration cannot be guessed from array shape, a presentation label, or the
    earliest unrelated transition.  Both named exposure windows must end the
    same digital output; the calibration context excludes the Camera trigger and
    therefore selects that output by its stable key.
    """

    context = calibration.readout_physical_context
    if context.target_abi_fingerprint != document.target.abi_fingerprint:
        raise ValueError(
            "release-recapture pulse target differs from the calibration target"
        )
    periods = {period.name: period for period in document.periods}
    try:
        first_exposure = periods["image1_expose"]
        second_exposure = periods["image2_expose"]
    except KeyError as error:
        raise ValueError(
            "release-recapture template omits a named exposure period"
        ) from error
    first = set(
        digital_outputs_falling_after_period(
            document,
            first_exposure.period_id,
        )
    )
    second = set(
        digital_outputs_falling_after_period(
            document,
            second_exposure.period_id,
        )
    )
    calibrated = {trace.output_key for trace in context.digital}
    probe_keys = tuple(sorted(first & second & calibrated))
    if len(probe_keys) != 1:
        raise ValueError(
            "release-recapture exposure waveform does not identify exactly one "
            "calibrated readout-light output"
        )
    traces = tuple(
        trace
        for trace in context.digital
        if trace.output_key == probe_keys[0]
    )
    if len(traces) != 1 or not traces[0].high_at_window_start:
        raise ValueError(
            "calibration readout context does not start with the declared probe on"
        )
    falling_ticks = tuple(
        tick for tick, high in traces[0].transitions if not high
    )
    if not falling_ticks:
        return context.integration_seconds
    duration = min(falling_ticks) / context.clock_hz
    if not 0.0 < duration <= context.integration_seconds:
        raise ValueError("calibration contains an invalid readout-light duration")
    return duration


def release_recapture_template(
    document: PulseDocument,
    calibration: CalibrationArtifact,
) -> PulseDocument:
    if tuple(parameter.parameter_id for parameter in document.scan_parameters) != (
        "t_off",
    ):
        raise ValueError(
            "release-recapture template must declare exactly one t_off SCAN_SLOT"
        )
    parameter = document.scan_parameters[0]
    if parameter.field.kind != FIELD_DURATION:
        raise ValueError("t_off must bind a pulse-period duration")
    if document.api_parameters:
        raise ValueError(
            "release-recapture autonomous Measurement has no unresolved API slots"
        )
    periods = {period.name: period for period in document.periods}
    expected = {
        "image1_expose",
        "image1_settle",
        "trap_off",
        "trap_recapture",
        "image2_expose",
        "image2_settle",
    }
    if not expected.issubset(periods):
        raise ValueError(
            "release-recapture template must contain the six named physical periods"
        )
    positions = {
        period.name: index for index, period in enumerate(document.periods)
    }
    if not (
        positions["image1_expose"]
        < positions["image1_settle"]
        < positions["trap_off"]
        < positions["trap_recapture"]
        < positions["image2_expose"]
        < positions["image2_settle"]
    ):
        raise ValueError(
            "release-recapture physical periods are not in acquisition order"
        )
    if parameter.field.period_id != periods["trap_off"].period_id:
        raise ValueError("t_off SCAN_SLOT must bind the named trap_off period")

    probe_seconds = calibrated_probe_seconds(document, calibration)
    integration = calibration.frame_contract.exposure_seconds
    # Keep the trap on until the sensor integration is complete.  One hardware
    # tick beyond the boundary prevents an equal-endpoint rounding ambiguity.
    settle_seconds = max(
        integration - probe_seconds + document.time_step_ns * 1e-9,
        0.0,
    )
    result = document
    for name in ("image1_expose", "image2_expose"):
        result = replace_pulse_field(
            result,
            PulseFieldRef(FIELD_DURATION, periods[name].period_id),
            probe_seconds,
            unit="s",
        )
    for name in ("image1_settle", "image2_settle"):
        current = periods[name]
        current_seconds = float(current.duration) * TIME_UNIT_TO_NS[current.unit] * 1e-9
        result = replace_pulse_field(
            result,
            PulseFieldRef(FIELD_DURATION, current.period_id),
            max(current_seconds, settle_seconds),
            unit="s",
        )
    return result


def freeze_release_recapture_rows(
    document: PulseDocument,
    calibration: CalibrationArtifact,
    trap_off_seconds: tuple[float, ...],
    shots: int,
) -> PulseDocument:
    """Freeze the physical pulse rows shared by Temperature and Grey molasses."""

    document = release_recapture_template(document, calibration)
    parameter = document.scan_parameters[0]
    unit_ns = TIME_UNIT_TO_NS[parameter.unit]
    frozen, normalization = freeze_scan_table(
        document,
        ("t_off",),
        tuple(
            (
                scale_authored_value(
                    seconds,
                    1e9 / unit_ns,
                    "trap_off_seconds",
                ),
            )
            for seconds in trap_off_seconds
        ),
    )
    if normalization.adjusted_cells:
        raise ValueError(
            "trap_off values must already lie on the selected pulse clock grid"
        )
    periods = document.periods
    repeat = (
        None
        if shots == 1
        else RepeatRegion(periods[0].period_id, periods[-1].period_id, shots)
    )
    return replace(document, scan_table=frozen, scan_recipe=None, repeat=repeat)


def bind_release_recapture_camera(
    document: PulseDocument,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    trigger_channel: str | None,
    repeat_axis: AxisSpec,
    readout_event_axis_id: AxisId,
    scan_axes: tuple[AxisSpec, ...],
    point_layout: PointLayout,
    calibration: ResolvedCalibration,
) -> tuple[PulseDocument, TriggeredCameraBinding]:
    """Bind the shared two-image physical acquisition, once, for both domains."""

    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    logical_document = bind_pulse_document_target(
        document,
        pulse_port.capability.target,
    )
    binding = bind_triggered_camera_acquisition(
        pulse_port,
        camera_port,
        pulse_document=expand_autonomous_scan_repeats(logical_document),
        execution_form=PulseExecutionForm.AUTONOMOUS_SCAN_ONCE,
        trigger_channel=trigger_channel,
        layout=TriggeredCameraLayout(
            repeat_axis=repeat_axis,
            readout_event_axis_id=readout_event_axis_id,
            readout_events_per_repeat=2,
            scan_axes=scan_axes,
            scan_point_layout=point_layout,
        ),
    )
    validate_live_release_recapture_calibration(binding, calibration)
    return logical_document, binding


def validate_live_release_recapture_calibration(
    binding: TriggeredCameraBinding,
    calibration: ResolvedCalibration,
) -> None:
    contract = binding.capture.capture_contract
    provenance = contract.camera_provenance
    frame_contract = calibration.artifact.frame_contract
    for event in (0, 1):
        frame_contract.assert_compatible(
            frame_contract.binding,
            provenance.descriptor,
            contract.dataset_schema,
            readout_event_index=event,
        )
    facts = contract.capability.camera_physical_facts
    offset = facts.external_trigger_integration_start_offset_seconds
    if offset is None:
        raise ValueError(
            "release-recapture requires a qualified camera integration offset"
        )
    pulse_binding = PulseCaptureBinding(
        binding.compiled_artifact,
        binding.trigger_channel,
        binding.cell_plan,
    )
    for event in (0, 1):
        observed = derive_readout_physical_context(
            pulse_binding,
            readout_event_index=event,
            integration_start_offset_seconds=offset,
            integration_seconds=frame_contract.exposure_seconds,
        )
        if observed != calibration.artifact.readout_physical_context:
            raise ValueError(
                f"release-recapture readout event {event} differs from calibration"
            )


__all__ = ["bind_release_recapture_camera", "freeze_release_recapture_rows"]
