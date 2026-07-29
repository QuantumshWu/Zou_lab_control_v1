"""Grey-molasses detuning Measurement, from authoring through binding."""

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
from zlc_data.codec import grid_topology_to_tree, point_table_to_tree
from zlc_neutral_atom.authoring import AuthoringChoice, AuthoringField, AuthoringSchema, MINIMUM_POSITIVE_FLOAT
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.devices.rf import BoundRfTablePort, RfDetuningTable
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
    PathPresentationHint,
)
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import ResolvedCalibration
from zlc_neutral_atom.logic_nodes.readout.calibration.reference import CalibrationArtifactRef
from zlc_neutral_atom.logic_nodes.readout.model_contract import ReadoutModelKind
from zlc_neutral_atom.logic_nodes.readout.calibration_input import (
    calibration_input_specs,
    calibration_reference,
)
from zlc_neutral_atom.logic_nodes.readout.measurement_values import (duration_axis_for_document, finite_signed_axis, linear_axis_from_range, optional_trigger, readout_model_kind, scale_authored_value)
from zlc_neutral_atom.logic_nodes.release_recapture.binding import bind_release_recapture_camera, freeze_release_recapture_rows
from zlc_neutral_atom.node_input import BoundNodeInputs
from .. import DEFAULT_RELEASE_RECAPTURE_PULSE_PATH
from zlc_neutral_atom.devices.camera.capture_port import BoundCapturePort
from zlc_neutral_atom.devices.sequencer.port import BoundPulsePort
from zlc_neutral_atom.capture.binding import TriggeredCameraBinding
from zlc_pulse import PulseDocument, build_pulse_playback
from zlc_storage import canonical_digest, canonical_text, finite_real, normalized_text, positive_integer


GREY_MOLASSES_DETUNING_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.release_recapture.grey_molasses_detuning",
    "grey-molasses-detuning",
)


GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration(
        "recapture",
        "zlc_neutral_atom.grey-molasses-detuning.recapture",
    ),
)


DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE = (-0.4, 0.4, 21)


DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS = 20.0


DEFAULT_GREY_MOLASSES_SHOTS = 16


DEFAULT_GREY_MOLASSES_PER_SITE = False


DEFAULT_GREY_MOLASSES_RF_ROLE = "rf"


_MINIMUM_SHOTS = 1


_DETUNING_COORDINATE_ID = AxisId("grey_molasses.detuning")


_GREY_MOLASSES_AUTHORING_SCHEMA = AuthoringSchema(
    (
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
            minimum=_MINIMUM_SHOTS,
            required=True,
            allow_blank=False,
        ),
        AuthoringField(
            "per_site",
            "bool",
            "Per-site survival",
            default=DEFAULT_GREY_MOLASSES_PER_SITE,
        ),
        AuthoringField(
            "rf_role",
            "choice",
            "RF role",
            required=True,
            dynamic_choices=True,
            description=(
                "Hardware-synchronized RF table Port advanced by the scan clock"
            ),
        ),
    )
)


def grey_molasses_detuning_authoring_schema() -> AuthoringSchema:
    return _GREY_MOLASSES_AUTHORING_SCHEMA


def grey_molasses_default_rf_role(available_roles) -> str | None:
    roles = tuple(available_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("RF roles must be unique")
    for role in roles:
        canonical_text(role, "RF role")
    if DEFAULT_GREY_MOLASSES_RF_ROLE in roles:
        return DEFAULT_GREY_MOLASSES_RF_ROLE
    return roles[0] if roles else None


GREY_MOLASSES_DETUNING_DEFINITION = MeasurementDefinition(
    GREY_MOLASSES_DETUNING_KEY,
    "Grey molasses detuning",
    "zlc.grey-molasses-detuning-request",
    "zlc.grey-molasses-detuning-binding",
)


@dataclass(frozen=True, slots=True)
class GreyMolassesDetuningIntent:
    """Device-independent physical input for a Grey-molasses Measurement."""

    pulse: str
    detuning_gamma: tuple[float, ...]
    trap_off_seconds: float
    shots: int
    rf_role: str
    per_site: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "pulse", normalized_text(self.pulse, "pulse"))
        object.__setattr__(
            self,
            "detuning_gamma",
            finite_signed_axis(self.detuning_gamma, "detuning_gamma"),
        )
        object.__setattr__(
            self,
            "trap_off_seconds",
            finite_real(
                self.trap_off_seconds,
                "trap_off_seconds",
                positive=True,
            ),
        )
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        object.__setattr__(
            self,
            "rf_role",
            normalized_text(self.rf_role, "rf_role"),
        )
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")


def build_grey_molasses_detuning_intent(
    *,
    pulse: str,
    detuning_gamma_range: object,
    trap_off_microseconds: object,
    shots: object,
    rf_role: object,
    per_site: object,
) -> GreyMolassesDetuningIntent:
    """Convert authored UI units into one physical Grey-molasses intent."""

    return GreyMolassesDetuningIntent(
        pulse,
        linear_axis_from_range(
            detuning_gamma_range,
            "detuning",
            scale=1.0,
            positive=False,
        ),
        scale_authored_value(
            finite_real(
                trap_off_microseconds,
                "t_off",
                positive=True,
            ),
            1e-6,
            "t_off",
        ),
        shots,  # type: ignore[arg-type] - validated by the intent owner
        rf_role,  # type: ignore[arg-type] - validated by the intent owner
        per_site,
    )


def build_grey_molasses_intent_from_authoring(
    values: Mapping[str, object],
) -> GreyMolassesDetuningIntent:
    authored = grey_molasses_detuning_authoring_schema().freeze(values)
    rf_role = authored["rf_role"]
    if rf_role is None:
        raise ValueError("select an installed synchronized RF role")
    return build_grey_molasses_detuning_intent(
        pulse=authored["pulse"],  # type: ignore[arg-type]
        detuning_gamma_range=authored["detuning"],
        trap_off_microseconds=authored["t_off"],
        shots=authored["shots"],
        rf_role=rf_role,
        per_site=authored["per_site"],
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
            finite_signed_axis(self.detuning_gamma, "detuning_gamma"),
        )
        value = duration_axis_for_document(
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
        object.__setattr__(self, "model_kind", readout_model_kind(self.model_kind))
        if type(self.per_site) is not bool:
            raise TypeError("per_site must be bool")
        object.__setattr__(
            self,
            "trigger_channel",
            optional_trigger(self.trigger_channel),
        )


@dataclass(frozen=True, slots=True)
class _GreyMolassesDetuningProgram:
    """Fixed pulse rows plus the RF-owned authored point-row coordinates."""

    document: PulseDocument
    point_table: PointTable
    grid_topology: GridTopology | None
    shots: int

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        columns = self.point_table.columns
        if (
            len(columns) != 1
            or columns[0].coordinate_id != _DETUNING_COORDINATE_ID
            or columns[0].role != SCAN_POINT
            or columns[0].value_kind != PointColumn.NUMERIC
            or any(value is None for value in columns[0].values)
        ):
            raise ValueError("point_table must contain the numeric detuning column")
        if self.grid_topology is not None and not isinstance(
            self.grid_topology,
            GridTopology,
        ):
            raise TypeError("grid_topology must be GridTopology or None")
        object.__setattr__(self, "shots", positive_integer(self.shots, "shots"))
        table = self.document.scan_table
        if table is None or len(table.rows) != self.point_table.row_count:
            raise ValueError("pulse scan rows must match the RF detuning rows")

    @property
    def physical_detuning_gamma(self) -> tuple[float, ...]:
        coordinates = self.point_table.column(
            _DETUNING_COORDINATE_ID
        ).values
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
                "point_table": point_table_to_tree(self.point_table),
                "grid_topology": (
                    None
                    if self.grid_topology is None
                    else grid_topology_to_tree(self.grid_topology)
                ),
                "shots": self.shots,
            }
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
    document = freeze_release_recapture_rows(
        request.pulse_document,
        calibration.artifact,
        tuple(request.trap_off_seconds for _value in request.detuning_gamma),
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
    grid_topology = (
        GridTopology(
            (_DETUNING_COORDINATE_ID,),
            (coordinate.values,),
            tuple((ordinal,) for ordinal in range(point_table.row_count)),
        )
        if len(set(coordinate.values)) == point_table.row_count
        else None
    )
    return _GreyMolassesDetuningProgram(
        document,
        point_table,
        grid_topology,
        request.shots,
    )


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
    logical_document, binding = bind_release_recapture_camera(
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
        scan_point_table=program.point_table,
        scan_grid_topology=program.grid_topology,
        calibration=calibration,
    )
    program = _GreyMolassesDetuningProgram(
        logical_document,
        program.point_table,
        program.grid_topology,
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


class AutonomousMeasurementUnavailable(RuntimeError):
    """The typed request is valid but the installed synchronous capability is absent."""


GREY_MOLASSES_CAPABILITY_GAP = (
    "grey-molasses detuning requires an RF Port that can preload and advance the "
    "complete two-photon-detuning table from the same hardware scan clock; the "
    "selected installation exposes no such RF Port"
)


@dataclass(frozen=True, slots=True)
class CalibratedGreyMolassesDetuningIntent:
    intent: GreyMolassesDetuningIntent
    calibration_ref: CalibrationArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.intent, GreyMolassesDetuningIntent):
            raise TypeError("intent must be GreyMolassesDetuningIntent")
        if not isinstance(self.calibration_ref, CalibrationArtifactRef):
            raise TypeError("calibration_ref must be CalibrationArtifactRef")


def bind_grey_molasses_detuning_inputs(intent: GreyMolassesDetuningIntent, inputs: BoundNodeInputs) -> CalibratedGreyMolassesDetuningIntent:
    return CalibratedGreyMolassesDetuningIntent(intent, calibration_reference(inputs))


def _grey_rf_choices(context: object) -> tuple[DynamicChoicePresentation, ...]:
    if not isinstance(context, tuple):
        raise TypeError("Grey-molasses dynamic choice context must be a role tuple")
    roles = tuple(context)
    return (
        DynamicChoicePresentation(
            "rf_role",
            tuple(AuthoringChoice(role, role) for role in roles),
            grey_molasses_default_rf_role(roles),
            GREY_MOLASSES_CAPABILITY_GAP if not roles else "",
        ),
    )


GREY_MOLASSES_DETUNING_LOGIC_NODE = LogicNodeDeclaration(
    definition=GREY_MOLASSES_DETUNING_DEFINITION,
    description=(
        "Autonomous release-recapture whose synchronized RF table advances "
        "from the hardware scan clock"
    ),
    authoring_schema=_GREY_MOLASSES_AUTHORING_SCHEMA,
    input_specs=calibration_input_specs(),
    outputs=(
        OutputPresentation(
            GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS[0],
            "recapture",
            "Recapture rate",
            "grey-molasses recapture rate",
        ),
    ),
    build_request=build_grey_molasses_intent_from_authoring,
    bind_request=bind_grey_molasses_detuning_inputs,
    path_presentations=(
        PathPresentationHint(
            "pulse",
            file_filter="Pulse program (*.json);;All files (*)",
            base_dir="pulses",
        ),
    ),
    resolve_dynamic_choices=_grey_rf_choices,
)



__all__ = ["AutonomousMeasurementUnavailable", "BoundGreyMolassesDetuning", "CalibratedGreyMolassesDetuningIntent", "DEFAULT_GREY_MOLASSES_DETUNING_GAMMA_RANGE", "DEFAULT_GREY_MOLASSES_PER_SITE", "DEFAULT_GREY_MOLASSES_RF_ROLE", "DEFAULT_GREY_MOLASSES_SHOTS", "DEFAULT_GREY_MOLASSES_TRAP_OFF_MICROSECONDS", "GREY_MOLASSES_CAPABILITY_GAP", "GREY_MOLASSES_DETUNING_DEFINITION", "GREY_MOLASSES_DETUNING_KEY", "GREY_MOLASSES_DETUNING_LOGIC_NODE", "GREY_MOLASSES_DETUNING_OUTPUT_DECLARATIONS", "GreyMolassesDetuningIntent", "GreyMolassesDetuningRequest", "bind_grey_molasses_detuning", "bind_grey_molasses_detuning_inputs", "build_grey_molasses_detuning_intent", "build_grey_molasses_intent_from_authoring", "grey_molasses_default_rf_role", "grey_molasses_detuning_authoring_schema"]
