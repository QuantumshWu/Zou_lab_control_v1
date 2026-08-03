"""Temperature release-recapture request and physical binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_data import REPEAT, SCAN_POINT, AxisId, AxisSpec, GridTopology, PointColumn, PointTable
from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema, MINIMUM_POSITIVE_FLOAT
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (
    duration_axis_for_document,
    linear_axis_from_range,
    numeric_axis,
)
from zlc_neutral_atom.logic_nodes.release_recapture.binding import (
    bind_release_recapture_camera,
    freeze_release_recapture_rows,
)
from zlc_neutral_atom.timing.pulse_parameter_scan import AutonomousScanSlotProgram
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_pulse import PulseDocument
from zlc_storage import canonical_text, positive_integer

from .. import DEFAULT_RELEASE_RECAPTURE_PULSE_PATH


DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE = (0.02, 300.02, 13)
DEFAULT_TEMPERATURE_SHOTS = 16
DEFAULT_TEMPERATURE_PER_SITE = False

TEMPERATURE_RELEASE_RECAPTURE_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.release_recapture.temperature",
    "temperature-release-recapture",
)
TEMPERATURE_RELEASE_RECAPTURE_DEFINITION = LogicNodeDefinition(
    TEMPERATURE_RELEASE_RECAPTURE_KEY,
    "Temperature",
    "measurement",
)
TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "survival",
    "zlc_neutral_atom.temperature-release-recapture.survival",
)

_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "camera_instance_id",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
        ),
        AuthoringField(
            "sequencer_instance_id",
            "choice",
            "Sequencer",
            required=True,
            dynamic_choices=True,
        ),
        AuthoringField(
            "pulse",
            "path",
            "Pulse template",
            default=DEFAULT_RELEASE_RECAPTURE_PULSE_PATH,
            required=True,
        ),
        AuthoringField(
            "t_off",
            "axis_range",
            "Trap-off time",
            default=DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE,
            unit="us",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
        ),
        AuthoringField(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_TEMPERATURE_SHOTS,
            minimum=1,
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
    return _AUTHORING_SCHEMA


@dataclass(frozen=True, slots=True)
class TemperatureReleaseRecaptureRequest:
    camera_instance_id: str
    sequencer_instance_id: str
    pulse: str
    trap_off_seconds: tuple[float, ...]
    shots: int
    per_site: bool

    def __post_init__(self) -> None:
        for field in ("camera_instance_id", "sequencer_instance_id", "pulse"):
            object.__setattr__(self, field, canonical_text(getattr(self, field), field))
        object.__setattr__(
            self,
            "trap_off_seconds",
            numeric_axis(self.trap_off_seconds, "trap_off_seconds", positive=True),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")


def build_temperature_release_recapture_request(
    values: Mapping[str, object],
) -> TemperatureReleaseRecaptureRequest:
    authored = _AUTHORING_SCHEMA.freeze(values)
    return TemperatureReleaseRecaptureRequest(
        authored["camera_instance_id"],
        authored["sequencer_instance_id"],
        authored["pulse"],
        linear_axis_from_range(
            authored["t_off"],
            "t_off",
            scale=1e-6,
            positive=True,
        ),
        authored["shots"],
        authored["per_site"],
    )


def bind_temperature_release_recapture(
    request: TemperatureReleaseRecaptureRequest,
    document: PulseDocument,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
):
    """Bind the temperature rows to the shared hardware pair pipeline."""

    durations = duration_axis_for_document(
        request.trap_off_seconds,
        "trap_off_seconds",
        document,
    )
    frozen = freeze_release_recapture_rows(
        document,
        calibration.artifact,
        durations,
        request.shots,
    )
    program = AutonomousScanSlotProgram(frozen)
    coordinate_id = AxisId("temperature.t_off")
    coordinate = PointColumn(
        coordinate_id,
        "Trap-off time",
        SCAN_POINT,
        PointColumn.NUMERIC,
        durations,
        "s",
    )
    point_table = PointTable(len(durations), (coordinate,))
    topology = (
        GridTopology(
            (coordinate_id,),
            (coordinate.values,),
            tuple((index,) for index in range(point_table.row_count)),
        )
        if len(set(coordinate.values)) == point_table.row_count
        else None
    )
    _logical_document, binding = bind_release_recapture_camera(
        program.execution_document,
        pulse_port=pulse_port,
        camera_port=camera_port,
        trigger_channel=None,
        repeat_axis=AxisSpec(
            AxisId("temperature.repeat"),
            "repeat",
            REPEAT,
            request.shots,
            tuple(range(request.shots)),
        ),
        readout_event_axis_id=AxisId("temperature.readout_event"),
        scan_point_table=point_table,
        scan_grid_topology=topology,
        calibration=calibration,
        camera_instance_id=request.camera_instance_id,
    )
    return binding


__all__ = [
    "DEFAULT_TEMPERATURE_PER_SITE",
    "DEFAULT_TEMPERATURE_SHOTS",
    "DEFAULT_TEMPERATURE_TRAP_OFF_MICROSECONDS_RANGE",
    "TEMPERATURE_RELEASE_RECAPTURE_DEFINITION",
    "TEMPERATURE_RELEASE_RECAPTURE_KEY",
    "TEMPERATURE_RELEASE_RECAPTURE_OUTPUT_DECLARATION",
    "TemperatureReleaseRecaptureRequest",
    "bind_temperature_release_recapture",
    "build_temperature_release_recapture_request",
    "temperature_release_recapture_authoring_schema",
]
