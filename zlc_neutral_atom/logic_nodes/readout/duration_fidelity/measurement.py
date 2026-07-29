"""Readout-duration fidelity Measurement, from authoring through binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral

from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema, MINIMUM_POSITIVE_FLOAT
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_node_declaration import (
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.timing.pulse_parameter_scan import (
    ApiSegmentTable,
    ApiSlotSegmentedProgram,
)
from zlc_neutral_atom.logic_nodes.readout.calibration_input import (
    calibration_input_specs,
    calibration_reference,
)
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (duration_axis_for_document, linear_axis_from_range, numeric_axis, optional_trigger, readout_model_kind, scale_authored_value)
from zlc_neutral_atom.logic_nodes.readout.physical_context import (
    digital_outputs_falling_after_period,
)
from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import (
    BoundPulsePort,
    FinitePulseExecutionRequest,
)
from zlc_pulse import FIELD_DURATION, TIME_UNIT_TO_NS, PulseDocument, PulseExecutionForm, RepeatRegion, bind_pulse_document_target, compile_pulse_artifact, resolve_api_parameters
from zlc_storage import integer, normalized_text, positive_integer


READOUT_DURATION_FIDELITY_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.readout.duration_fidelity",
    "readout-duration-fidelity",
)


READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "fidelity",
        "zlc_neutral_atom.logic_nodes.readout-duration-fidelity.fidelity",
    ),
)


DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE = (2.0, 20_000.0, 11)


DEFAULT_READOUT_DURATION_SHOTS = 60


DEFAULT_READOUT_DURATION_SITE = None


DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH = "probe_template.json"


_MINIMUM_SHOTS = 1


_MINIMUM_SITE_INDEX = 0


_READOUT_DURATION_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH,
            required=True,
        ),
        AuthoringField(
            "duration",
            "axis_range",
            "Detection time",
            default=DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE,
            unit="us",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
        ),
        AuthoringField(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_READOUT_DURATION_SHOTS,
            minimum=_MINIMUM_SHOTS,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "site",
            "int",
            "Site (optional)",
            default=DEFAULT_READOUT_DURATION_SITE,
            minimum=_MINIMUM_SITE_INDEX,
            required=False,
            allow_blank=True,
        ),
    )
)


def readout_duration_fidelity_authoring_schema() -> AuthoringSchema:
    return _READOUT_DURATION_AUTHORING_SCHEMA


READOUT_DURATION_FIDELITY_DEFINITION = MeasurementDefinition(
    READOUT_DURATION_FIDELITY_KEY,
    "Fidelity vs duration",
    "zlc.readout-duration-fidelity-request",
    "zlc.readout-duration-fidelity-binding",
)


@dataclass(frozen=True, slots=True)
class ReadoutDurationFidelityIntent:
    """Device-independent physical input for a readout-duration Measurement."""

    pulse: str
    duration_seconds: tuple[float, ...]
    shots: int
    site: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pulse", normalized_text(self.pulse, "pulse"))
        object.__setattr__(
            self,
            "duration_seconds",
            numeric_axis(self.duration_seconds, "duration_seconds", positive=True),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        object.__setattr__(
            self,
            "site",
            integer(
                self.site,
                "site",
                optional=True,
                minimum=_MINIMUM_SITE_INDEX,
            ),
        )


def build_readout_duration_fidelity_intent(
    *,
    pulse: str,
    duration_microseconds: object,
    shots: object,
    site: object,
) -> ReadoutDurationFidelityIntent:
    """Convert an authored microsecond range into one physical intent."""

    return ReadoutDurationFidelityIntent(
        pulse,
        linear_axis_from_range(
            duration_microseconds,
            "duration",
            scale=1e-6,
            positive=True,
        ),
        shots,  # type: ignore[arg-type] - validated by the intent owner
        site,  # type: ignore[arg-type] - validated by the intent owner
    )


def build_readout_duration_intent_from_authoring(
    values: Mapping[str, object],
) -> ReadoutDurationFidelityIntent:
    authored = readout_duration_fidelity_authoring_schema().freeze(values)
    site = authored["site"]
    return build_readout_duration_fidelity_intent(
        pulse=authored["pulse"],  # type: ignore[arg-type]
        duration_microseconds=authored["duration"],
        shots=authored["shots"],
        site=site,
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
            duration_axis_for_document(
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
        object.__setattr__(self, "model_kind", readout_model_kind(self.model_kind))
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
            optional_trigger(self.trigger_channel),
        )


def _readout_duration_point_groups(
    program: ApiSlotSegmentedProgram,
) -> tuple[PulseDocument, ...]:
    """Resolve owner-frozen rows while retaining the hardware shot repeat.

    Generic API PulseScan repeats point segments through ``scan_sweep_count``.
    This coupled Measurement has one host sweep per duration and lets the
    sequencer execute its whole-document RepeatRegion under one FIRE.
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
            self.program.sweep_count != 1
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
    falling_outputs = tuple(
        key
        for key in digital_outputs_falling_after_period(
            document,
            period.period_id,
        )
        if document.target.by_key[key].lanes[0] != trigger_channel
    )
    if len(falling_outputs) != 1:
        raise ValueError(
            "readout-duration API waveform must end exactly one non-trigger "
            "digital readout-light output"
        )

    periods = document.periods
    execution_document = replace(
        document,
        scan_sweep_count=1,
        repeat=(
            None
            if request.shots == 1
            else RepeatRegion(
                periods[0].period_id,
                periods[-1].period_id,
                request.shots,
            )
        ),
    )
    scale = 1e9 / TIME_UNIT_TO_NS[parameter.unit]
    program = ApiSlotSegmentedProgram(
        execution_document,
        ApiSegmentTable(
            (parameter.parameter_id,),
            tuple(
                (
                    scale_authored_value(
                        seconds,
                        scale,
                        "duration_seconds",
                    ),
                )
                for seconds in request.duration_seconds
            ),
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


@dataclass(frozen=True, slots=True)
class CalibratedReadoutDurationFidelityIntent:
    intent: ReadoutDurationFidelityIntent
    calibration_ref: CalibrationArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ReadoutDurationFidelityIntent):
            raise TypeError("intent must be ReadoutDurationFidelityIntent")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")


def bind_readout_duration_fidelity_inputs(intent: ReadoutDurationFidelityIntent, inputs: BoundNodeInputs) -> CalibratedReadoutDurationFidelityIntent:
    return CalibratedReadoutDurationFidelityIntent(intent, calibration_reference(inputs))


READOUT_DURATION_FIDELITY_LOGIC_NODE = LogicNodeDeclaration(
    definition=READOUT_DURATION_FIDELITY_DEFINITION,
    description=(
        "Apply and read back camera integration time at each point, then "
        "publish calibrated readout fidelity"
    ),
    authoring_schema=_READOUT_DURATION_AUTHORING_SCHEMA,
    input_specs=calibration_input_specs(),
    outputs=(
        OutputPresentation(
            READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS[0],
            "fidelity",
            "Fidelity",
            "readout fidelity",
        ),
    ),
    build_request=build_readout_duration_intent_from_authoring,
    bind_request=bind_readout_duration_fidelity_inputs,
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
)



__all__ = ["BoundReadoutDurationFidelity", "CalibratedReadoutDurationFidelityIntent", "DEFAULT_READOUT_DURATION_FIDELITY_PULSE_PATH", "DEFAULT_READOUT_DURATION_MICROSECONDS_RANGE", "DEFAULT_READOUT_DURATION_SHOTS", "DEFAULT_READOUT_DURATION_SITE", "READOUT_DURATION_FIDELITY_DEFINITION", "READOUT_DURATION_FIDELITY_KEY", "READOUT_DURATION_FIDELITY_LOGIC_NODE", "READOUT_DURATION_FIDELITY_OUTPUT_DECLARATIONS", "ReadoutDurationFidelityIntent", "ReadoutDurationFidelityRequest", "bind_readout_duration_fidelity", "bind_readout_duration_fidelity_inputs", "build_readout_duration_fidelity_intent", "build_readout_duration_intent_from_authoring", "readout_duration_fidelity_authoring_schema"]
