"""Grey-molasses detuning request and hardware-synchronized binding."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_data import REPEAT, SCAN_POINT, AxisId, AxisSpec, GridTopology, PointColumn, PointTable
from zlc_neutral_atom.authoring import AuthoringField, AuthoringSchema, MINIMUM_POSITIVE_FLOAT
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.capture.binding import TriggeredCameraBinding
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.rf import RfDetuningTable
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (
    duration_axis_for_document,
    finite_signed_axis,
    linear_axis_from_range,
    scale_authored_value,
)
from zlc_neutral_atom.logic_nodes.release_recapture.binding import (
    bind_release_recapture_camera,
    freeze_release_recapture_rows,
)
from zlc_pulse import PulseDocument, build_pulse_playback
from zlc_storage import canonical_text, finite_real, positive_integer

from .. import DEFAULT_RELEASE_RECAPTURE_PULSE_PATH


DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE = (-0.4, 0.4, 21)
DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS = 20.0
DEFAULT_GREY_MOLASSES_SHOTS = 16
DEFAULT_GREY_MOLASSES_PER_SITE = False

GREY_MOLASSES_DETUNING_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning",
    "grey-molasses-detuning",
)
GREY_MOLASSES_DETUNING_DEFINITION = LogicNodeDefinition(
    GREY_MOLASSES_DETUNING_KEY,
    "Grey molasses detuning",
    "measurement",
)
GREY_MOLASSES_DETUNING_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "recapture",
    "zlc_neutral_atom.grey-molasses-detuning.recapture",
)

_DETUNING_COORDINATE_ID = AxisId("grey_molasses.detuning")
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
            "rf_instance_id",
            "choice",
            "RF",
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
            "detuning",
            "axis_range",
            "Two-photon detuning",
            default=DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE,
            unit="Γ",
            required=True,
        ),
        AuthoringField(
            "t_off",
            "float",
            "Trap-off time",
            default=DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS,
            unit="us",
            minimum=MINIMUM_POSITIVE_FLOAT,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "shots",
            "int",
            "Shots / point",
            default=DEFAULT_GREY_MOLASSES_SHOTS,
            minimum=1,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "per_site",
            "bool",
            "Per-site survival",
            default=DEFAULT_GREY_MOLASSES_PER_SITE,
        ),
    )
)


def grey_molasses_detuning_authoring_schema() -> AuthoringSchema:
    return _AUTHORING_SCHEMA


@dataclass(frozen=True, slots=True)
class GreyMolassesDetuningRequest:
    camera_instance_id: str
    sequencer_instance_id: str
    rf_instance_id: str
    pulse: str
    detuning_gamma: tuple[float, ...]
    trap_off_seconds: float
    shots: int
    per_site: bool

    def __post_init__(self) -> None:
        for field in (
            "camera_instance_id",
            "sequencer_instance_id",
            "rf_instance_id",
            "pulse",
        ):
            object.__setattr__(self, field, canonical_text(getattr(self, field), field))
        object.__setattr__(
            self,
            "detuning_gamma",
            finite_signed_axis(self.detuning_gamma, "detuning_gamma"),
        )
        object.__setattr__(
            self,
            "trap_off_seconds",
            finite_real(self.trap_off_seconds, "trap_off_seconds", positive=True),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")


def build_grey_molasses_detuning_request(
    values: Mapping[str, object],
) -> GreyMolassesDetuningRequest:
    authored = _AUTHORING_SCHEMA.freeze(values)
    return GreyMolassesDetuningRequest(
        authored["camera_instance_id"],
        authored["sequencer_instance_id"],
        authored["rf_instance_id"],
        authored["pulse"],
        linear_axis_from_range(
            authored["detuning"],
            "detuning",
            scale=1.0,
            positive=False,
        ),
        scale_authored_value(authored["t_off"], 1e-6, "t_off"),
        authored["shots"],
        authored["per_site"],
    )


def bind_grey_molasses_detuning(
    request: GreyMolassesDetuningRequest,
    document: PulseDocument,
    calibration: ResolvedCalibration,
    *,
    pulse_port: BoundPulsePort,
    camera_port: BoundCapturePort,
) -> tuple[TriggeredCameraBinding, RfDetuningTable]:
    """Bind pulse/camera facts and the RF table to one common scan clock."""

    trap_off = duration_axis_for_document(
        (request.trap_off_seconds,),
        "trap_off_seconds",
        document,
    )[0]
    pulse_document = freeze_release_recapture_rows(
        document,
        calibration.artifact,
        tuple(trap_off for _ in request.detuning_gamma),
        request.shots,
    )
    coordinate = PointColumn(
        _DETUNING_COORDINATE_ID,
        "Two-photon detuning",
        SCAN_POINT,
        PointColumn.NUMERIC,
        request.detuning_gamma,
        "Γ",
    )
    point_table = PointTable(len(coordinate.values), (coordinate,))
    topology = (
        GridTopology(
            (_DETUNING_COORDINATE_ID,),
            (coordinate.values,),
            tuple((index,) for index in range(point_table.row_count)),
        )
        if len(set(coordinate.values)) == point_table.row_count
        else None
    )
    _logical_document, binding = bind_release_recapture_camera(
        pulse_document,
        pulse_port=pulse_port,
        camera_port=camera_port,
        trigger_channel=None,
        repeat_axis=AxisSpec(
            AxisId("grey_molasses.repeat"),
            "repeat",
            REPEAT,
            request.shots,
            tuple(range(request.shots)),
        ),
        readout_event_axis_id=AxisId("grey_molasses.readout_event"),
        scan_point_table=point_table,
        scan_grid_topology=topology,
        calibration=calibration,
        camera_instance_id=request.camera_instance_id,
    )
    physical_values = tuple(
        value
        for _repeat in range(request.shots)
        for value in request.detuning_gamma
    )
    table = RfDetuningTable(binding.compiled_artifact.fingerprint, physical_values)
    playback = build_pulse_playback(binding.compiled_artifact)
    if tuple(group.point_index for group in playback.trigger_groups) != tuple(
        range(len(table.detuning_gamma))
    ):
        raise RuntimeError("compiled trigger groups differ from the RF table order")
    return binding, table


__all__ = [
    "DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE",
    "DEFAULT_GREY_MOLASSES_PER_SITE",
    "DEFAULT_GREY_MOLASSES_SHOTS",
    "DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS",
    "GREY_MOLASSES_DETUNING_DEFINITION",
    "GREY_MOLASSES_DETUNING_KEY",
    "GREY_MOLASSES_DETUNING_OUTPUT_DECLARATION",
    "GreyMolassesDetuningRequest",
    "bind_grey_molasses_detuning",
    "build_grey_molasses_detuning_request",
    "grey_molasses_detuning_authoring_schema",
]
