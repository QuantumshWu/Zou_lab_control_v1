"""Coupled readout Measurements with explicit autonomous-hardware boundaries.

The old implementation put scan control, camera acquisition and a point reducer
in one ``ScannedMeasurementNode``.  This module keeps the useful
physics while making the ownership explicit:

* release-recapture acquisition is one autonomous SCAN_SLOT camera Measurement;
* its adjacent event-0/event-1 frames are reduced by one fixed 2:1
  StreamReducer in the same exact Run;
* readout-duration uses the explicitly segmented API boundary: the camera is
  configured and read back once per point, while every shot remains a frozen
  hardware-timed pulse;
* grey-molasses remains capability-gated by its synchronized RF Port.

No public request contains a timeout.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from numbers import Integral, Real

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    PointLayout,
)
from zlc_neutral_atom.acquisition.camera import (
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)
from zlc_neutral_atom.bootstrap._triggered_capture import (
    TriggeredCameraBinding,
    TriggeredCameraLayout,
    bind_triggered_camera_acquisition,
)
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.readout.calibration import (
    CalibrationArtifact,
    ReadoutModelKind,
    ResolvedCalibration,
)
from zlc_neutral_atom.readout.calibration_reference import (
    CalibrationArtifactRef,
)
from zlc_neutral_atom.readout.physical_context import (
    derive_readout_physical_context,
)
from zlc_neutral_atom.runtime.capture import BoundCapturePort
from zlc_neutral_atom.runtime.pipeline import BoundMeasurement
from zlc_neutral_atom.rf import BoundRfTablePort, RfDetuningTable
from zlc_neutral_atom.scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
    AutonomousScanSlotProgram,
)
from zlc_neutral_atom.timing.lineage import PulseCaptureBinding
from zlc_neutral_atom.timing.pulse import BoundPulsePort
from zlc_neutral_atom.timing.pulse import FinitePulseExecutionRequest
from zlc_pulse import (
    FIELD_DURATION,
    TIME_UNIT_TO_NS,
    PulseDocument,
    PulseExecutionForm,
    PulseFieldRef,
    RepeatRegion,
    bind_pulse_document_target,
    build_pulse_playback,
    compile_pulse_artifact,
    expand_autonomous_scan_repeats,
    freeze_scan_table,
    replace_pulse_field,
    resolve_api_parameters,
    require_autonomous_scan_resident_capacity,
)
from zlc_storage import canonical_digest, canonical_text, positive_integer


TEMPERATURE_RELEASE_RECAPTURE_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "temperature-release-recapture",
)
READOUT_DURATION_FIDELITY_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "readout-duration-fidelity",
)
GREY_MOLASSES_DETUNING_KEY = DefinitionKey(
    "zlc_neutral_atom.readout",
    "grey-molasses-detuning",
)

TEMPERATURE_RELEASE_RECAPTURE_DEFINITION = MeasurementDefinition(
    TEMPERATURE_RELEASE_RECAPTURE_KEY,
    "Temperature",
    "zlc.temperature-release-recapture-request",
    "zlc.temperature-release-recapture-binding",
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)
READOUT_DURATION_FIDELITY_DEFINITION = MeasurementDefinition(
    READOUT_DURATION_FIDELITY_KEY,
    "Fidelity vs duration",
    "zlc.readout-duration-fidelity-request",
    "zlc.readout-duration-fidelity-binding",
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)
GREY_MOLASSES_DETUNING_DEFINITION = MeasurementDefinition(
    GREY_MOLASSES_DETUNING_KEY,
    "Grey molasses detuning",
    "zlc.grey-molasses-detuning-request",
    "zlc.grey-molasses-detuning-binding",
    CAMERA_CAPTURE_SPEC_OWNER_FINGERPRINT,
)

# Every current user-facing Measurement remains visible.  Runnable capability
# is a Start/preflight fact, not a catalog-discovery filter.
COUPLED_MEASUREMENT_DEFINITIONS = (
    TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
    READOUT_DURATION_FIDELITY_DEFINITION,
    GREY_MOLASSES_DETUNING_DEFINITION,
)


def _numeric_axis(
    values: object,
    name: str,
    *,
    positive: bool,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not raw:
        raise ValueError(f"{name} must contain at least one value")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        item = float(value)
        if not isfinite(item) or (positive and item <= 0.0) or (
            not positive and item < 0.0
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} values must be finite and {qualifier}")
        result.append(item)
    return tuple(result)


def _optional_trigger(value: str | None) -> str | None:
    return None if value is None else canonical_text(value, "trigger_channel")


def _duration_axis_for_document(
    values: object,
    name: str,
    document: PulseDocument,
) -> tuple[float, ...]:
    axis = _numeric_axis(values, name, positive=True)
    minimum_seconds = document.time_step_ns * 1e-9
    if any(value < minimum_seconds for value in axis):
        raise ValueError(
            f"{name} values must be at least one pulse target clock tick "
            f"({minimum_seconds:.12g} s)"
        )
    return axis


def _model_kind(value: ReadoutModelKind | None) -> ReadoutModelKind | None:
    if value is not None and not isinstance(value, ReadoutModelKind):
        raise TypeError("model_kind must be ReadoutModelKind or None")
    return value


@dataclass(frozen=True)
class TemperatureReleaseRecaptureRequest:
    pulse_document: PulseDocument
    trap_off_seconds: tuple[float, ...]
    shots: int
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    per_site: bool = False
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        object.__setattr__(
            self,
            "trap_off_seconds",
            _duration_axis_for_document(
                self.trap_off_seconds,
                "trap_off_seconds",
                self.pulse_document,
            ),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        object.__setattr__(self, "model_kind", _model_kind(self.model_kind))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        object.__setattr__(
            self,
            "trigger_channel",
            _optional_trigger(self.trigger_channel),
        )


@dataclass(frozen=True)
class ReadoutDurationFidelityRequest:
    pulse_document: PulseDocument
    duration_seconds: tuple[float, ...]
    shots: int
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    site: int | None = None
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        object.__setattr__(
            self,
            "duration_seconds",
            _duration_axis_for_document(
                self.duration_seconds,
                "duration_seconds",
                self.pulse_document,
            ),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        for name in ("camera_ref", "sequencer_ref"):
            if not isinstance(getattr(self, name), DeviceRef):
                raise TypeError(f"{name} must be DeviceRef")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        object.__setattr__(self, "model_kind", _model_kind(self.model_kind))
        if self.site is not None:
            if (
                isinstance(self.site, bool)
                or not isinstance(self.site, Integral)
                or int(self.site) < 0
            ):
                raise ValueError("site must be a non-negative integer or None")
            object.__setattr__(self, "site", int(self.site))
        object.__setattr__(
            self,
            "trigger_channel",
            _optional_trigger(self.trigger_channel),
        )


@dataclass(frozen=True)
class GreyMolassesDetuningRequest:
    pulse_document: PulseDocument
    detuning_gamma: tuple[float, ...]
    trap_off_seconds: float
    shots: int
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    rf_role: str
    calibration_ref: CalibrationArtifactRef
    model_kind: ReadoutModelKind | None = None
    per_site: bool = False
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pulse_document, PulseDocument):
            raise TypeError("pulse_document must be PulseDocument")
        object.__setattr__(
            self,
            "detuning_gamma",
            _finite_signed_axis(self.detuning_gamma, "detuning_gamma"),
        )
        value = _duration_axis_for_document(
            (self.trap_off_seconds,),
            "trap_off_seconds",
            self.pulse_document,
        )[0]
        object.__setattr__(self, "trap_off_seconds", value)
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        for name in ("camera_ref", "sequencer_ref"):
            if not isinstance(getattr(self, name), DeviceRef):
                raise TypeError(f"{name} must be DeviceRef")
        object.__setattr__(
            self,
            "rf_role",
            canonical_text(self.rf_role, "rf_role"),
        )
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")
        object.__setattr__(self, "model_kind", _model_kind(self.model_kind))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        object.__setattr__(
            self,
            "trigger_channel",
            _optional_trigger(self.trigger_channel),
        )


@dataclass(frozen=True, slots=True)
class _GreyMolassesDetuningProgram:
    """Fixed release-recapture pulse rows plus the RF-owned logical scan axis."""

    document: PulseDocument
    detuning_axis: AxisSpec
    shots: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if (
            not isinstance(self.detuning_axis, AxisSpec)
            or self.detuning_axis.role != SCAN_POINT
        ):
            raise ValueError("detuning_axis must have SCAN_POINT role")
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        table = self.document.scan_table
        if table is None or len(table.rows) != self.detuning_axis.size:
            raise ValueError("pulse scan rows must match the RF detuning axis")

    @property
    def point_layout(self) -> PointLayout:
        return PointLayout.rect_c((self.detuning_axis.size,))

    @property
    def physical_detuning_gamma(self) -> tuple[float, ...]:
        coordinates = self.detuning_axis.coordinates
        assert coordinates is not None
        return tuple(
            float(value)
            for _repeat in range(self.shots)
            for value in coordinates
        )

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.grey-molasses-detuning-program",
                "pulse_document": self.document.fingerprint,
                "detuning_gamma": self.detuning_axis.coordinates,
                "shots": self.shots,
            }
        )


def _finite_signed_axis(values: object, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a numeric sequence")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be a numeric sequence") from exc
    if not raw:
        raise ValueError(f"{name} must contain at least one value")
    result = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} values must be real numbers")
        item = float(value)
        if not isfinite(item):
            raise ValueError(f"{name} values must be finite")
        result.append(item)
    return tuple(result)


def _calibrated_probe_seconds(
    document: PulseDocument,
    calibration: CalibrationArtifact,
) -> float:
    """Read the probe-light window from the calibration's physical trace.

    A duration cannot be guessed from array shape or from the earliest unrelated
    digital transition.  The pulse target must identify one digital ``probe``
    port, and that same stable output key selects the calibrated trace.
    """

    context = calibration.readout_physical_context
    probe_ports = tuple(
        port
        for port in document.target.ports
        if port.kind == "digital"
        and (port.key.casefold() == "probe" or port.label.casefold() == "probe")
    )
    if len(probe_ports) != 1:
        raise ValueError(
            "release-recapture target must identify exactly one digital probe port"
        )
    traces = tuple(
        trace
        for trace in context.digital
        if trace.output_key == probe_ports[0].key
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


def _release_recapture_template(
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

    probe_seconds = _calibrated_probe_seconds(document, calibration)
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


def _freeze_release_recapture_rows(
    document: PulseDocument,
    calibration: CalibrationArtifact,
    trap_off_seconds: tuple[float, ...],
    shots: int,
) -> PulseDocument:
    """Freeze the physical pulse rows shared by Temperature and Grey molasses."""

    document = _release_recapture_template(document, calibration)
    parameter = document.scan_parameters[0]
    unit_ns = TIME_UNIT_TO_NS[parameter.unit]
    frozen, _normalization = freeze_scan_table(
        document,
        ("t_off",),
        tuple((seconds * 1e9 / unit_ns,) for seconds in trap_off_seconds),
    )
    periods = document.periods
    repeat = (
        None
        if shots == 1
        else RepeatRegion(periods[0].period_id, periods[-1].period_id, shots)
    )
    return replace(document, scan_table=frozen, scan_recipe=None, repeat=repeat)


def build_temperature_release_recapture_program(
    request: TemperatureReleaseRecaptureRequest,
    calibration: ResolvedCalibration,
) -> AutonomousScanSlotProgram:
    """Freeze t_off rows and shots without touching a Device Port."""

    if not isinstance(request, TemperatureReleaseRecaptureRequest):
        raise TypeError("request must be TemperatureReleaseRecaptureRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if calibration.reference != request.calibration_ref:
        raise ValueError("resolved calibration differs from the request")
    document = _freeze_release_recapture_rows(
        request.pulse_document,
        calibration.artifact,
        request.trap_off_seconds,
        request.shots,
    )
    return AutonomousScanSlotProgram(document)


def _readout_duration_point_groups(
    program: ApiSlotSegmentedProgram,
) -> tuple[PulseDocument, ...]:
    """Resolve owner-frozen rows while retaining the hardware shot repeat.

    Generic API scans expand repeat into host-visible dataset cells.  This
    coupled Measurement instead arms one point group and lets the sequencer
    execute its whole-document RepeatRegion under one FIRE.
    """

    return tuple(
        resolve_api_parameters(
            program.document,
            dict(zip(program.table.columns, row, strict=True)),
        )
        for row in program.table.rows
    )


@dataclass(frozen=True)
class BoundReadoutDurationFidelity:
    """Target-bound point pulses for the admitted API-slot exposure sweep."""

    request: ReadoutDurationFidelityRequest
    program: ApiSlotSegmentedProgram
    pulse_port: BoundPulsePort
    camera_port: BoundCapturePort
    trigger_channel: str
    point_requests: tuple[FinitePulseExecutionRequest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, ReadoutDurationFidelityRequest):
            raise TypeError("request has another type")
        if not isinstance(self.program, ApiSlotSegmentedProgram):
            raise TypeError("program must be ApiSlotSegmentedProgram")
        if not isinstance(self.pulse_port, BoundPulsePort):
            raise TypeError("pulse_port must be BoundPulsePort")
        if not isinstance(self.camera_port, BoundCapturePort):
            raise TypeError("camera_port must be BoundCapturePort")
        requests = tuple(self.point_requests)
        if (
            self.program.repeat_count != self.request.shots
            or self.program.point_count != len(self.request.duration_seconds)
        ):
            raise ValueError("API program cardinality differs from the request")
        if len(requests) != self.program.point_count or any(
            not isinstance(value, FinitePulseExecutionRequest)
            for value in requests
        ):
            raise ValueError("point_requests must cover every duration in order")
        group_documents = _readout_duration_point_groups(self.program)
        if tuple(value.document for value in requests) != group_documents:
            raise ValueError("point requests differ from the frozen API program")
        object.__setattr__(self, "point_requests", requests)


def _validate_readout_duration_calibration(
    request: ReadoutDurationFidelityRequest,
    calibration: ResolvedCalibration,
    camera_port: BoundCapturePort,
) -> None:
    if calibration.reference != request.calibration_ref:
        raise ValueError("resolved calibration differs from the request")
    frame = calibration.artifact.frame_contract
    facts = camera_port.capability.camera_physical_facts
    payload = camera_port.capability.payload_contract
    if frame.binding.value != request.camera_ref.role:
        raise ValueError("calibration belongs to another camera role")
    observed = (
        facts.camera_identity,
        facts.sensor_identity,
        facts.optical_path,
        facts.sensor_shape_yx,
        facts.roi_origin_yx,
        facts.roi_shape_yx,
        facts.binning_yx,
        facts.spatial_y_axis_id,
        facts.spatial_x_axis_id,
        facts.coordinate_frame,
        facts.dtype,
        facts.count_unit,
        facts.exposure_seconds,
        facts.gain,
        facts.readout_mode,
        facts.opaque_frame_settings_fingerprint,
        payload.value_schema,
    )
    expected = (
        frame.camera_identity,
        frame.sensor_identity,
        frame.optical_path,
        frame.sensor_shape_yx,
        frame.roi_origin_yx,
        frame.roi_shape_yx,
        frame.binning_yx,
        frame.spatial_y_axis_id,
        frame.spatial_x_axis_id,
        frame.coordinate_frame,
        frame.dtype,
        frame.count_unit,
        frame.exposure_seconds,
        frame.gain,
        frame.readout_mode,
        frame.opaque_frame_settings_fingerprint,
        frame.frame_schema,
    )
    if observed != expected:
        raise ValueError(
            "readout-duration camera working point differs from its calibration"
        )
    model = calibration.artifact.select_model(request.model_kind)
    if request.site is not None:
        if request.site >= model.feature.site_axis.size:
            raise ValueError("selected site is outside the calibration site axis")
        if not bool(model.usable_sites.mask[request.site]):
            raise ValueError("selected site is invalid in the calibration model")


def bind_readout_duration_fidelity(
    request: ReadoutDurationFidelityRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> BoundReadoutDurationFidelity:
    """Bind one camera-rearmed API duration sweep without touching hardware."""

    if not isinstance(request, ReadoutDurationFidelityRequest):
        raise TypeError("request must be ReadoutDurationFidelityRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if not isinstance(pulse_port, BoundPulsePort):
        raise TypeError("pulse_port must be BoundPulsePort")
    if not isinstance(camera_port, BoundCapturePort):
        raise TypeError("camera_port must be BoundCapturePort")
    _validate_readout_duration_calibration(request, calibration, camera_port)

    document = bind_pulse_document_target(
        request.pulse_document,
        pulse_port.capability.target,
    )
    if document.scan_parameters or document.scan_table is not None:
        raise ValueError(
            "readout-duration template uses one API duration, not SCAN_SLOT"
        )
    if document.repeat is not None:
        raise ValueError("readout-duration template must describe one shot")
    if len(document.api_parameters) != 1:
        raise ValueError(
            "readout-duration template must declare exactly one API parameter"
        )
    parameter = document.api_parameters[0]
    if parameter.field.kind != FIELD_DURATION:
        raise ValueError("readout-duration API parameter must bind a period duration")
    period = document.period_by_id[parameter.field.period_id]
    probe_ports = tuple(
        port
        for port in document.target.ports
        if port.kind == "digital" and port.label.casefold() == "probe"
    )
    if len(probe_ports) != 1 or len(probe_ports[0].lanes) != 1:
        raise ValueError(
            "readout-duration target must declare one single-lane probe port"
        )
    probe_lane = document.target.raw_lanes.index(probe_ports[0].lanes[0])
    if period.states[probe_lane] != 1:
        raise ValueError(
            "readout-duration API period must be the probe-light window"
        )

    facts = camera_port.capability.camera_physical_facts
    if request.trigger_channel is None:
        if len(facts.capture_trigger_channels) != 1:
            raise ValueError(
                "readout-duration capture requires one camera trigger channel"
            )
        trigger_channel = facts.capture_trigger_channels[0]
    else:
        trigger_channel = request.trigger_channel
    facts.require_single_capture_trigger_channel(trigger_channel)
    try:
        trigger_lane = document.target.raw_lanes.index(trigger_channel)
    except ValueError as exc:
        raise ValueError(
            "camera trigger channel is absent from the bound pulse target"
        ) from exc
    period_index = document.periods.index(period)
    previous_trigger_state = (
        0
        if period_index == 0
        else document.periods[period_index - 1].states[trigger_lane]
    )
    if period.states[trigger_lane] != 1 or previous_trigger_state != 0:
        raise ValueError(
            "readout-duration API period must begin with the camera trigger edge"
        )

    periods = document.periods
    execution_document = (
        document
        if request.shots == 1
        else replace(
            document,
            repeat=RepeatRegion(
                periods[0].period_id,
                periods[-1].period_id,
                request.shots,
            ),
        )
    )
    scale = 1e9 / TIME_UNIT_TO_NS[parameter.unit]
    program = ApiSlotSegmentedProgram(
        execution_document,
        ApiSegmentTable(
            (parameter.parameter_id,),
            tuple((seconds * scale,) for seconds in request.duration_seconds),
        ),
        "camera integration time must be configured and read back at each API point",
    )
    point_requests = []
    for point_document in _readout_duration_point_groups(program):
        artifact = compile_pulse_artifact(
            point_document,
            clock_hz=pulse_port.capability.clock_hz,
            execution_form=PulseExecutionForm.STATIC_ONCE,
            trigger_channels=(trigger_channel,),
            live_target=pulse_port.capability.target,
        )
        schedules = tuple(
            value
            for value in artifact.trigger_schedules
            if value.channel == trigger_channel
        )
        if len(schedules) != 1 or schedules[0].total != request.shots:
            raise ValueError(
                "each readout-duration point must emit exactly one camera trigger per shot"
            )
        point_requests.append(
            FinitePulseExecutionRequest(point_document, artifact)
        )
    return BoundReadoutDurationFidelity(
        request,
        program,
        pulse_port,
        camera_port,
        trigger_channel,
        tuple(point_requests),
    )


def _build_grey_molasses_detuning_program(
    request: GreyMolassesDetuningRequest,
    calibration: ResolvedCalibration,
) -> _GreyMolassesDetuningProgram:
    """Freeze one fixed-t_off pulse row per RF point without touching a Port."""

    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("request must be GreyMolassesDetuningRequest")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be an admitted ResolvedCalibration")
    if calibration.reference != request.calibration_ref:
        raise ValueError("resolved calibration differs from the request")
    document = _freeze_release_recapture_rows(
        request.pulse_document,
        calibration.artifact,
        tuple(request.trap_off_seconds for _value in request.detuning_gamma),
        request.shots,
    )
    axis = AxisSpec(
        AxisId("grey_molasses.detuning"),
        "Two-photon detuning",
        SCAN_POINT,
        len(request.detuning_gamma),
        request.detuning_gamma,
        "Γ",
    )
    return _GreyMolassesDetuningProgram(
        document,
        axis,
        request.shots,
    )


def _bind_release_recapture_camera(
    document: PulseDocument,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    trigger_channel: str | None,
    repeat_axis: AxisSpec,
    readout_event_axis_id: AxisId,
    scan_axes: tuple[AxisSpec, ...],
    point_layout: PointLayout,
    definition: MeasurementDefinition,
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
    require_autonomous_scan_resident_capacity(
        logical_document,
        pulse_port.capability.resident_scan_point_capacity,
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
    base = binding.measurement
    binding = TriggeredCameraBinding(
        binding.pulse_port,
        binding.pulse_request,
        binding.trigger_channel,
        BoundMeasurement(
            definition,
            base.capture_port,
            base.capture_contract,
            base.capture_spec,
        ),
        binding.cell_plan,
    )
    _validate_live_release_recapture_calibration(binding, calibration)
    return logical_document, binding


@dataclass(frozen=True)
class BoundTemperatureReleaseRecapture:
    request: TemperatureReleaseRecaptureRequest
    program: AutonomousScanSlotProgram
    camera_binding: TriggeredCameraBinding

    def __post_init__(self) -> None:
        if not isinstance(self.request, TemperatureReleaseRecaptureRequest):
            raise TypeError("request has another type")
        if not isinstance(self.program, AutonomousScanSlotProgram):
            raise TypeError("program must be AutonomousScanSlotProgram")
        if not isinstance(self.camera_binding, TriggeredCameraBinding):
            raise TypeError("camera_binding must be TriggeredCameraBinding")
        if (
            self.camera_binding.measurement.definition
            != TEMPERATURE_RELEASE_RECAPTURE_DEFINITION
        ):
            raise ValueError("bound Measurement uses another definition")

def bind_temperature_release_recapture(
    request: TemperatureReleaseRecaptureRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> BoundTemperatureReleaseRecapture:
    """Bind the one honest current autonomous coupled Measurement."""

    program = build_temperature_release_recapture_program(request, calibration)
    point_table = program.point_table
    logical_document, binding = _bind_release_recapture_camera(
        program.execution_document,
        pulse_port=pulse_port,
        camera_port=camera_port,
        trigger_channel=request.trigger_channel,
        repeat_axis=AxisSpec(
            AxisId("temperature.repeat"),
            "repeat",
            REPEAT,
            request.shots,
            tuple(range(request.shots)),
        ),
        readout_event_axis_id=AxisId("temperature.readout_event"),
        # The scan table's physical rows remain in the pulse parameter's
        # authoring unit.  The Measurement contract exposes the operator-facing
        # physical quantity in SI, matching Main's ``Trap-off time (s)`` output.
        scan_axes=(
            AxisSpec(
                AxisId("temperature.t_off"),
                "Trap-off time",
                SCAN_POINT,
                len(request.trap_off_seconds),
                request.trap_off_seconds,
                "s",
            ),
        ),
        point_layout=point_table.point_layout,
        definition=TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
        calibration=calibration,
    )
    program = AutonomousScanSlotProgram(logical_document)
    return BoundTemperatureReleaseRecapture(request, program, binding)


@dataclass(frozen=True)
class BoundGreyMolassesDetuning:
    request: GreyMolassesDetuningRequest
    program: _GreyMolassesDetuningProgram
    camera_binding: TriggeredCameraBinding
    rf_port: BoundRfTablePort
    rf_table: RfDetuningTable

    def __post_init__(self) -> None:
        if not isinstance(self.request, GreyMolassesDetuningRequest):
            raise TypeError("request has another type")
        if not isinstance(self.program, _GreyMolassesDetuningProgram):
            raise TypeError("program has another type")
        if not isinstance(self.camera_binding, TriggeredCameraBinding):
            raise TypeError("camera_binding has another type")
        if not isinstance(self.rf_port, BoundRfTablePort):
            raise TypeError("rf_port has another type")
        if not isinstance(self.rf_table, RfDetuningTable):
            raise TypeError("rf_table has another type")
        if (
            self.camera_binding.measurement.definition
            != GREY_MOLASSES_DETUNING_DEFINITION
        ):
            raise ValueError("bound Measurement uses another definition")
        if (
            self.rf_table.pulse_artifact_digest
            != self.camera_binding.compiled_artifact.fingerprint
        ):
            raise ValueError("RF table belongs to another compiled pulse")


def bind_grey_molasses_detuning(
    request: GreyMolassesDetuningRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
    rf_port: BoundRfTablePort,
) -> BoundGreyMolassesDetuning:
    if not isinstance(rf_port, BoundRfTablePort):
        raise TypeError("rf_port must be BoundRfTablePort")
    program = _build_grey_molasses_detuning_program(request, calibration)
    logical_document, binding = _bind_release_recapture_camera(
        program.document,
        pulse_port=pulse_port,
        camera_port=camera_port,
        trigger_channel=request.trigger_channel,
        repeat_axis=AxisSpec(
            AxisId("grey_molasses.repeat"),
            "repeat",
            REPEAT,
            request.shots,
            tuple(range(request.shots)),
        ),
        readout_event_axis_id=AxisId("grey_molasses.readout_event"),
        scan_axes=(program.detuning_axis,),
        point_layout=program.point_layout,
        definition=GREY_MOLASSES_DETUNING_DEFINITION,
        calibration=calibration,
    )
    program = _GreyMolassesDetuningProgram(
        logical_document,
        program.detuning_axis,
        program.shots,
    )
    table = RfDetuningTable(
        binding.compiled_artifact.fingerprint,
        program.physical_detuning_gamma,
    )
    playback = build_pulse_playback(binding.compiled_artifact)
    physical_point_indices = tuple(
        group.point_index for group in playback.trigger_groups
    )
    if physical_point_indices != tuple(range(len(table.detuning_gamma))):
        raise RuntimeError(
            "compiled trigger groups differ from the RF physical table order"
        )
    logical_values = tuple(float(value) for value in request.detuning_gamma)
    for physical_index, value in enumerate(table.detuning_gamma):
        repeat_index, point_index = divmod(physical_index, len(logical_values))
        if repeat_index >= request.shots or value != logical_values[point_index]:
            raise RuntimeError("RF table is not R-major/P-fast")
    return BoundGreyMolassesDetuning(
        request,
        program,
        binding,
        rf_port,
        table,
    )


def _validate_live_release_recapture_calibration(
    binding: TriggeredCameraBinding,
    calibration: ResolvedCalibration,
) -> None:
    contract = binding.measurement.capture_contract
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


class AutonomousMeasurementUnavailable(RuntimeError):
    """The typed request is valid but the installed synchronous capability is absent."""


GREY_MOLASSES_CAPABILITY_GAP = (
    "grey-molasses detuning requires an RF Port that can preload and advance the "
    "complete two-photon-detuning table from the same hardware scan clock; the "
    "selected installation exposes no such RF Port"
)


def reject_grey_molasses_detuning(
    request: GreyMolassesDetuningRequest,
) -> None:
    if not isinstance(request, GreyMolassesDetuningRequest):
        raise TypeError("request must be GreyMolassesDetuningRequest")
    raise AutonomousMeasurementUnavailable(GREY_MOLASSES_CAPABILITY_GAP)


__all__ = [
    "AutonomousMeasurementUnavailable",
    "BoundGreyMolassesDetuning",
    "BoundReadoutDurationFidelity",
    "BoundTemperatureReleaseRecapture",
    "COUPLED_MEASUREMENT_DEFINITIONS",
    "GREY_MOLASSES_CAPABILITY_GAP",
    "GREY_MOLASSES_DETUNING_DEFINITION",
    "GREY_MOLASSES_DETUNING_KEY",
    "GreyMolassesDetuningRequest",
    "READOUT_DURATION_FIDELITY_DEFINITION",
    "READOUT_DURATION_FIDELITY_KEY",
    "ReadoutDurationFidelityRequest",
    "TEMPERATURE_RELEASE_RECAPTURE_DEFINITION",
    "TEMPERATURE_RELEASE_RECAPTURE_KEY",
    "TemperatureReleaseRecaptureRequest",
    "bind_grey_molasses_detuning",
    "bind_readout_duration_fidelity",
    "bind_temperature_release_recapture",
    "build_temperature_release_recapture_program",
    "reject_grey_molasses_detuning",
]
