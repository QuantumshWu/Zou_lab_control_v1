"""Temperature release-recapture Measurement, from authoring through binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    AxisId,
    AxisSpec,
    GridTopology,
    PointColumn,
    PointTable,
)
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
from zlc_neutral_atom.timing.pulse_parameter_scan import AutonomousScanSlotProgram
from zlc_neutral_atom.logic_nodes.readout.calibration_input import (
    calibration_input_specs,
    calibration_reference,
)
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (duration_axis_for_document, linear_axis_from_range, numeric_axis, optional_trigger, readout_model_kind)
from zlc_neutral_atom.logic_nodes.release_recapture.binding import bind_release_recapture_camera, freeze_release_recapture_rows
from zlc_neutral_atom.node_input import BoundNodeInputs
from .. import DEFAULT_RELEASE_RECAPTURE_PULSE_PATH
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.capture.binding import TriggeredCameraBinding
from zlc_pulse import PulseDocument
from zlc_storage import normalized_text, positive_integer


TEMPERATURE_RELEASE_RECAPTURE_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.release_recapture.temperature",
    "temperature-release-recapture",
)


TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "survival",
        "zlc_neutral_atom.temperature-release-recapture.survival",
    ),
)


DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE = (0.02, 300.02, 13)


DEFAULT_TEMPERATURE_SHOTS = 16


DEFAULT_TEMPERATURE_PER_SITE = False


_MINIMUM_SHOTS = 1


_TEMPERATURE_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
            description=(
                "Autonomous two-readout pulse with the declared t_off SCAN_SLOT"
            ),
        ),
        AuthoringField(
            "t_off",
            "axis_range",
            "Trap-off time",
            default=DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE,
            unit="us",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
            description=(
                "Positive durations; the selected PulseDocument validates its "
                "exact target clock grid when the request is frozen"
            ),
        ),
        AuthoringField(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_TEMPERATURE_SHOTS,
            minimum=_MINIMUM_SHOTS,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "per_site",
            "bool",
            "Per-site survival",
            default=DEFAULT_TEMPERATURE_PER_SITE,
        ),
    )
)


def temperature_release_recapture_authoring_schema() -> AuthoringSchema:
    return _TEMPERATURE_AUTHORING_SCHEMA


TEMPERATURE_RELEASE_RECAPTURE_DEFINITION = MeasurementDefinition(
    TEMPERATURE_RELEASE_RECAPTURE_KEY,
    "Temperature",
    "zlc.temperature-release-recapture-request",
    "zlc.temperature-release-recapture-binding",
)


@dataclass(frozen=True, slots=True)
class TemperatureReleaseRecaptureIntent:
    """Device-independent physical input for a temperature Measurement."""

    pulse: str
    trap_off_seconds: tuple[float, ...]
    shots: int
    per_site: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "pulse", normalized_text(self.pulse, "pulse"))
        object.__setattr__(
            self,
            "trap_off_seconds",
            numeric_axis(self.trap_off_seconds, "trap_off_seconds", positive=True),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")


def build_temperature_release_recapture_intent(
    *,
    pulse: str,
    trap_off_microseconds: object,
    shots: object,
    per_site: object,
) -> TemperatureReleaseRecaptureIntent:
    """Convert an authored microsecond range into one physical intent."""

    return TemperatureReleaseRecaptureIntent(
        pulse,
        linear_axis_from_range(
            trap_off_microseconds,
            "trap_off",
            scale=1e-6,
            positive=True,
        ),
        shots,  # type: ignore[arg-type] - validated by the intent owner
        per_site,
    )


def build_temperature_intent_from_authoring(
    values: Mapping[str, object],
) -> TemperatureReleaseRecaptureIntent:
    authored = temperature_release_recapture_authoring_schema().freeze(values)
    return build_temperature_release_recapture_intent(
        pulse=authored["pulse"],  # type: ignore[arg-type]
        trap_off_microseconds=authored["t_off"],
        shots=authored["shots"],
        per_site=authored["per_site"],
    )


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
            duration_axis_for_document(
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
        object.__setattr__(self, "model_kind", readout_model_kind(self.model_kind))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        object.__setattr__(
            self,
            "trigger_channel",
            optional_trigger(self.trigger_channel),
        )


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
    document = freeze_release_recapture_rows(
        request.pulse_document,
        calibration.artifact,
        request.trap_off_seconds,
        request.shots,
    )
    return AutonomousScanSlotProgram(document)


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


def bind_temperature_release_recapture(
    request: TemperatureReleaseRecaptureRequest,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> BoundTemperatureReleaseRecapture:
    """Bind the one honest current autonomous coupled Measurement."""

    program = build_temperature_release_recapture_program(request, calibration)
    coordinate_id = AxisId("temperature.t_off")
    coordinate = PointColumn(
        coordinate_id,
        "Trap-off time",
        SCAN_POINT,
        PointColumn.NUMERIC,
        request.trap_off_seconds,
        "s",
    )
    point_table = PointTable(len(coordinate.values), (coordinate,))
    grid_topology = (
        GridTopology(
            (coordinate_id,),
            (coordinate.values,),
            tuple((ordinal,) for ordinal in range(point_table.row_count)),
        )
        if len(set(coordinate.values)) == point_table.row_count
        else None
    )
    logical_document, binding = bind_release_recapture_camera(
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
        # Pulse rows keep the parameter's authored unit.  Dataset point rows
        # expose the operator-facing physical quantity in SI without changing
        # row order or expanding correlated coordinates.
        scan_point_table=point_table,
        scan_grid_topology=grid_topology,
        calibration=calibration,
    )
    program = AutonomousScanSlotProgram(logical_document)
    return BoundTemperatureReleaseRecapture(request, program, binding)


@dataclass(frozen=True, slots=True)
class CalibratedTemperatureReleaseRecaptureIntent:
    intent: TemperatureReleaseRecaptureIntent
    calibration_ref: CalibrationArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.intent, TemperatureReleaseRecaptureIntent):
            raise TypeError("intent must be TemperatureReleaseRecaptureIntent")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")


def bind_temperature_release_recapture_inputs(intent: TemperatureReleaseRecaptureIntent, inputs: BoundNodeInputs) -> CalibratedTemperatureReleaseRecaptureIntent:
    return CalibratedTemperatureReleaseRecaptureIntent(intent, calibration_reference(inputs))


TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE = LogicNodeDeclaration(
    definition=TEMPERATURE_RELEASE_RECAPTURE_DEFINITION,
    description=(
        "Autonomous hardware scan with two exact camera events per cell; "
        "publishes calibrated survival without dropping repeat/scan axes"
    ),
    authoring_schema=_TEMPERATURE_AUTHORING_SCHEMA,
    input_specs=calibration_input_specs(),
    outputs=(
        OutputPresentation(
            TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS[0],
            "survival",
            "Survival",
            "release-recapture survival",
        ),
    ),
    build_request=build_temperature_intent_from_authoring,
    bind_request=bind_temperature_release_recapture_inputs,
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
)



__all__ = ["BoundTemperatureReleaseRecapture", "CalibratedTemperatureReleaseRecaptureIntent", "DEFAULT_TEMPERATURE_PER_SITE", "DEFAULT_TEMPERATURE_SHOTS", "DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE", "TEMPERATURE_RELEASE_RECAPTURE_DEFINITION", "TEMPERATURE_RELEASE_RECAPTURE_KEY", "TEMPERATURE_RELEASE_RECAPTURE_LOGIC_NODE", "TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATIONS", "TemperatureReleaseRecaptureIntent", "TemperatureReleaseRecaptureRequest", "bind_temperature_release_recapture", "bind_temperature_release_recapture_inputs", "build_temperature_intent_from_authoring", "build_temperature_release_recapture_intent", "build_temperature_release_recapture_program", "temperature_release_recapture_authoring_schema"]
