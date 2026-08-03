"""Definition, request, and output semantics for Camera Measurement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zlc_data import (
    READOUT_EVENT,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    ValidityContract,
    ValueSchema,
)
from zlc_data.codec import dataset_revision_ref_to_tree
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.catalog import DefinitionKey, MeasurementDefinition
from zlc_neutral_atom.logic_node_declaration import (
    DynamicChoicePresentation,
    LogicNodeDeclaration,
    OutputPresentation,
)
from zlc_neutral_atom.capture.reference import (
    CaptureArtifactRef,
    capture_artifact_ref_to_tree,
)
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import (
    MonitorCoverage,
    MonitorDatasetSnapshot,
)
from zlc_neutral_atom.runtime.streams import event_ref_to_tree
from zlc_storage import canonical_text, positive_integer
from zlc_storage import positive_real

from zlc_neutral_atom.authoring import (
    MINIMUM_POSITIVE_FLOAT,
    AuthoringChoice,
    AuthoringField,
    AuthoringSchema,
)


DEFAULT_CAMERA_MEASUREMENT_ROLE = "camera"
DEFAULT_CAMERA_MEASUREMENT_REPEAT = 0
DEFAULT_CAMERA_FRAMES_PER_CYCLE = 1
MINIMUM_CAMERA_MEASUREMENT_REPEAT = 0
MINIMUM_CAMERA_FRAMES_PER_CYCLE = 1
CAMERA_FRAME_OUTPUT_CONTRACT_ID = (
    "zlc_neutral_atom.camera-measurement.current-frame"
)
CAMERA_MEASUREMENT_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.camera_measurement",
    "camera-measurement",
)
CAMERA_MEASUREMENT_DEFINITION = MeasurementDefinition(
    CAMERA_MEASUREMENT_KEY,
    "Camera",
    "zlc.camera-measurement-request",
    "zlc.camera-measurement-binding",
)


_CAMERA_MEASUREMENT_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "camera_role",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
            description="Frozen camera role from the current installation",
        ),
        AuthoringField(
            "frames_per_cycle",
            "int",
            "Frames per cycle",
            default=DEFAULT_CAMERA_FRAMES_PER_CYCLE,
            required=True,
            minimum=MINIMUM_CAMERA_FRAMES_PER_CYCLE,
            allow_blank=False,
            description=(
                "Ordered camera frames retained on an explicit READOUT_EVENT axis"
            ),
        ),
        AuthoringField(
            "exposure",
            "float",
            "Exposure",
            default=None,
            required=False,
            unit="s",
            minimum=MINIMUM_POSITIVE_FLOAT,
            allow_blank=True,
            description=(
                "Exposure applied and read back for this Camera run; blank keeps "
                "the selected camera's installed working point"
            ),
        ),
        AuthoringField(
            "repeat",
            "int",
            "Repeat",
            default=DEFAULT_CAMERA_MEASUREMENT_REPEAT,
            required=True,
            minimum=MINIMUM_CAMERA_MEASUREMENT_REPEAT,
            allow_blank=False,
            description=(
                "0 keeps this installed camera live; a positive value performs "
                "that many exact finite capture cycles"
            ),
        ),
    )
)


def camera_measurement_authoring_schema() -> AuthoringSchema:
    """Return the one scalar authoring declaration beside the typed request."""

    return _CAMERA_MEASUREMENT_AUTHORING_SCHEMA


def camera_measurement_roles(installed_roles) -> tuple[str, ...]:
    """Preserve every installed Camera role in catalog order."""

    roles = tuple(installed_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("installed camera roles must be unique")
    for role in roles:
        canonical_text(role, "installed camera role")
    return roles


def camera_measurement_default_role(available_roles) -> str | None:
    """Select the request owner's default from one frozen role projection."""

    roles = tuple(available_roles)
    if len(set(roles)) != len(roles):
        raise ValueError("available camera roles must be unique")
    for role in roles:
        canonical_text(role, "available camera role")
    if DEFAULT_CAMERA_MEASUREMENT_ROLE in roles:
        return DEFAULT_CAMERA_MEASUREMENT_ROLE
    return roles[0] if roles else None


@dataclass(frozen=True, slots=True)
class CameraMeasurementIntent:
    """Device-independent Camera authoring frozen before installation binding."""

    camera_role: str
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE
    exposure_seconds: float | None = None
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT

    def __post_init__(self) -> None:
        canonical_text(self.camera_role, "camera_role")
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        if self.exposure_seconds is not None:
            object.__setattr__(
                self,
                "exposure_seconds",
                positive_real(self.exposure_seconds, "exposure_seconds"),
            )

    @property
    def output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)


def build_camera_measurement_intent_from_authoring(
    values: Mapping[str, object],
) -> CameraMeasurementIntent:
    """Freeze Camera leaves without resolving an installed DeviceRef."""

    authored = camera_measurement_authoring_schema().freeze(values)
    return CameraMeasurementIntent(
        camera_role=authored["camera_role"],
        frames_per_cycle=authored["frames_per_cycle"],
        exposure_seconds=authored["exposure"],
        repeat=authored["repeat"],
    )


def camera_frame_output_index(output_name: str) -> int:
    """Parse one canonical ``frame_i`` name without accepting aliases."""

    name = canonical_text(output_name, "camera frame output name")
    prefix = "frame_"
    if not name.startswith(prefix):
        raise ValueError("camera frame output name must start with 'frame_'")
    token = name[len(prefix) :]
    if not token.isascii() or not token.isdecimal():
        raise ValueError("camera frame output index must be a decimal integer")
    index = int(token)
    if token != str(index):
        raise ValueError("camera frame output index must use canonical decimal text")
    return index


def camera_frame_output_declarations(
    frames_per_cycle: int,
) -> tuple[DatasetOutputDeclaration, ...]:
    """Return owner-paired public outputs for one camera cycle.

    A camera cycle is stored atomically as one Dataset whose named
    ``READOUT_EVENT`` axis has ``frames_per_cycle`` cells.  Presentation may
    expose those cells independently, but their names are owned here beside
    the request that defines the cycle -- never guessed from an ndarray shape
    and never supplemented with a lossy ``frame`` alias.
    """

    count = positive_integer(frames_per_cycle, "frames_per_cycle")
    return tuple(
        DatasetOutputDeclaration(
            f"frame_{index}",
            CAMERA_FRAME_OUTPUT_CONTRACT_ID,
        )
        for index in range(count)
    )


def _camera_frame_ref(
    source: OwnedSnapshot,
    event_axis_id: str,
    event_index: int,
    output_schema: DatasetSchema,
) -> DatasetRevisionRef:
    return DatasetRevisionRef(
        BlockId(f"camera-frame/{event_axis_id}/{event_index}"),
        source.ref.stream_generation,
        output_schema.fingerprint,
        source.ref.revision,
    )


def _materialize_camera_frame(
    source: OwnedSnapshot,
    event_column: PointColumn,
    event_index: int,
    point_ordinal: int,
) -> OwnedSnapshot:
    """Select one physical frame row and remove private cycle coordinates."""

    source_schema = source.block.schema
    selected_values = source.block.values[:, point_ordinal : point_ordinal + 1, ...]
    selected_validity = source.block.validity
    if isinstance(selected_validity, CellValidity):
        selected_validity = CellValidity(
            selected_validity.mask[:, point_ordinal : point_ordinal + 1]
        )
    elif isinstance(selected_validity, DatasetComponentValidity):
        selected_validity = DatasetComponentValidity(
            selected_validity.axis_ids,
            selected_validity.mask[:, point_ordinal : point_ordinal + 1, ...],
        )
    output_schema = DatasetSchema(
        source_schema.repeat_axis,
        PointTable(1),
        None,
        source_schema.cell_schema,
    )
    ref = _camera_frame_ref(
        source,
        event_column.coordinate_id.value,
        event_index,
        output_schema,
    )
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            selected_values,
            selected_validity,
            output_schema,
        ),
    )


def _unique_camera_point_ordinal(
    schema: DatasetSchema,
    **coordinates: tuple[PointColumn, int],
) -> int:
    matches = tuple(
        ordinal
        for ordinal in range(schema.point_table.row_count)
        if all(column.values[ordinal] == value for column, value in coordinates.values())
    )
    if len(matches) != 1:
        names = ", ".join(coordinates)
        raise ValueError(f"Camera Dataset requires one point row for {names}")
    return matches[0]


def _finite_camera_event_column(
    schema: DatasetSchema,
    frames_per_cycle: int,
) -> PointColumn:
    """Validate and return the sole canonical finite Camera event column."""

    event_columns = tuple(
        column
        for column in schema.point_table.columns
        if column.role == READOUT_EVENT
    )
    if (
        len(event_columns) != 1
        or schema.point_table.columns != event_columns
        or event_columns[0].values != tuple(range(frames_per_cycle))
    ):
        raise ValueError(
            "finite Camera source must contain only its canonical readout events"
        )
    if any(axis.role == READOUT_EVENT for axis in schema.cell_schema.data_axes):
        raise ValueError("READOUT_EVENT cannot be a trailing camera data axis")
    return event_columns[0]


def project_camera_measurement_outputs(
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, OwnedSnapshot]:
    """Materialize the Camera Measurement's declared ``frame_i`` outputs.

    The measurement contract, rather than a GUI, owns how one atomic camera
    cycle maps its named ``READOUT_EVENT`` cells to public outputs.  Repeat,
    the singleton public point carrier, trailing data axes, component validity,
    revision and stream generation are retained exactly.
    """

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("camera frame source must be OwnedSnapshot")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera output projection requires CameraMeasurementRequest")
    schema = source.block.schema
    event_column = _finite_camera_event_column(
        schema,
        request.frames_per_cycle,
    )

    projected: dict[str, OwnedSnapshot] = {}
    for event_index, output_name in enumerate(request.output_names):
        projected[output_name] = _materialize_camera_frame(
            source,
            event_column,
            event_index,
            event_index,
        )
    return projected


def camera_measurement_final_outputs(
    reference: CaptureArtifactRef,
    source: OwnedSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, FinalDatasetOutput]:
    """Publish every declared Camera frame from one admitted FINAL capture."""

    if not isinstance(reference, CaptureArtifactRef):
        raise TypeError("Camera FINAL reference must be CaptureArtifactRef")
    snapshots = project_camera_measurement_outputs(source, request)
    source_identity = capture_artifact_ref_to_tree(reference)
    declarations = request.output_declarations
    return {
        declaration.name: FinalDatasetOutput(
            declaration,
            snapshot,
        )
        for declaration, snapshot in zip(
            declarations,
            snapshots.values(),
            strict=True,
        )
    }


def project_camera_monitor_outputs(
    source: MonitorDatasetSnapshot,
    request: "CameraMeasurementRequest",
) -> dict[str, LiveDatasetOutput]:
    """Publish one latest atomic Camera cycle as declared ``frame_i`` values.

    The monitor owns one physical Dataset cell.  Its private value begins with
    the declared ``READOUT_EVENT`` component axis, while every public output
    removes that component and retains the canonical
    ``(R=1, P=1, *frame_shape)`` contract in the camera's native dtype.
    """

    if not isinstance(source, MonitorDatasetSnapshot):
        raise TypeError("source must be MonitorDatasetSnapshot")
    source_schema = source.snapshot.block.schema
    if (
        source_schema.repeat_axis.size != 1
        or source_schema.point_table != PointTable(1)
        or source_schema.grid_topology is not None
    ):
        raise ValueError("Camera monitor source must be one canonical latest cell")
    data_axes = source_schema.cell_schema.data_axes
    if not data_axes or data_axes[0].role != READOUT_EVENT:
        raise ValueError("Camera monitor value must begin with READOUT_EVENT")
    event_axis = data_axes[0]
    if (
        event_axis.size != request.frames_per_cycle
        or event_axis.coordinates != tuple(range(request.frames_per_cycle))
    ):
        raise ValueError(
            "Camera monitor READOUT_EVENT components differ from frames_per_cycle"
        )

    projected: dict[str, LiveDatasetOutput] = {}
    for declaration in request.output_declarations:
        output_name = declaration.name
        event_index = camera_frame_output_index(output_name)
        snapshot = _materialize_camera_monitor_frame(
            source.snapshot,
            event_axis,
            event_index,
        )
        output_schema = snapshot.block.schema
        if output_schema.point_table != PointTable(1):
            raise RuntimeError("Camera monitor storage axes leaked into a public frame")
        total = output_schema.repeat_axis.size * output_schema.point_table.row_count
        coverage = MonitorCoverage(
            written_cells=source.coverage.written_cells,
            total_cells=total,
            missed_events=source.coverage.missed_events,
            current_gap=source.coverage.current_gap,
        )
        projected[output_name] = LiveDatasetOutput(
            declaration,
            snapshot,
            coverage,
        )
    return projected


def _materialize_camera_monitor_frame(
    source: OwnedSnapshot,
    event_axis: AxisSpec,
    event_index: int,
) -> OwnedSnapshot:
    """Select one private cycle component without copying or changing dtype."""

    cycle_schema = source.block.schema.cell_schema
    if cycle_schema.data_axes[0] != event_axis:
        raise ValueError("Camera monitor event axis differs from its cycle schema")
    component_ids = cycle_schema.validity_contract.component_axis_ids
    if not component_ids or component_ids[0] != event_axis.axis_id:
        raise ValueError("Camera monitor cycle validity must begin with READOUT_EVENT")
    frame_component_ids = component_ids[1:]
    frame_validity = (
        ValidityContract.value()
        if not frame_component_ids
        else ValidityContract.components(*frame_component_ids)
    )
    frame_schema = ValueSchema(
        cycle_schema.data_axes[1:],
        frame_validity,
        cycle_schema.dtype,
        cycle_schema.value_unit,
    )
    source_validity = source.block.validity
    if not isinstance(source_validity, DatasetComponentValidity):
        raise TypeError("Camera monitor cycle requires component validity")
    selected_mask = source_validity.mask[:, :, event_index, ...]
    selected_validity = (
        CellValidity(selected_mask)
        if not frame_component_ids
        else DatasetComponentValidity(frame_component_ids, selected_mask)
    )
    output_schema = DatasetSchema(
        source.block.schema.repeat_axis,
        PointTable(1),
        None,
        frame_schema,
    )
    ref = _camera_frame_ref(
        source,
        event_axis.axis_id.value,
        event_index,
        output_schema,
    )
    return OwnedSnapshot(
        ref,
        DataBlock(
            ref.block_id,
            ref.revision,
            source.block.values[:, :, event_index, ...],
            selected_validity,
            output_schema,
        ),
    )


@dataclass(frozen=True)
class CameraMeasurementRequest:
    """Read raw camera cycles: ``repeat=0`` live, ``repeat=K`` finite.

    The request owns only the selected camera.  Trigger timing belongs to
    independently running hardware; Camera Measurement never acquires pulse or
    sequencer authority.
    """

    camera_ref: DeviceRef
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE
    exposure_seconds: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )
        exposure = self.exposure_seconds
        if exposure is not None:
            object.__setattr__(
                self,
                "exposure_seconds",
                positive_real(exposure, "exposure_seconds"),
            )

    @property
    def output_names(self) -> tuple[str, ...]:
        """One signal per declared readout event, in cycle order."""

        return tuple(output.name for output in self.output_declarations)

    @property
    def output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)


def _camera_request_outputs(
    request: object,
) -> tuple[OutputPresentation, ...]:
    if not isinstance(request, (CameraMeasurementIntent, CameraMeasurementRequest)):
        raise TypeError("Camera output owner requires an authored or bound request")
    return tuple(
        OutputPresentation(
            declaration,
            declaration.name,
            "Counts",
            "ordered camera readout event; repeat, point, and trailing data axes "
            "are preserved",
        )
        for declaration in request.output_declarations
    )


def _camera_role_choices(context: object) -> tuple[DynamicChoicePresentation, ...]:
    if not isinstance(context, tuple):
        raise TypeError("Camera dynamic choice context must be a role tuple")
    roles = camera_measurement_roles(context)
    return (
        DynamicChoicePresentation(
            "camera_role",
            tuple(AuthoringChoice(role, role) for role in roles),
            camera_measurement_default_role(roles),
            "Camera Measurement requires an installed camera role" if not roles else "",
        ),
    )


CAMERA_MEASUREMENT_LOGIC_NODE = LogicNodeDeclaration(
    definition=CAMERA_MEASUREMENT_DEFINITION,
    description="Acquire camera frames as a live or finite Measurement",
    authoring_schema=_CAMERA_MEASUREMENT_AUTHORING_SCHEMA,
    input_specs=(),
    outputs=(),
    build_request=build_camera_measurement_intent_from_authoring,
    bind_request=None,
    resolve_outputs=_camera_request_outputs,
    resolve_dynamic_choices=_camera_role_choices,
)


__all__ = [
    "CAMERA_MEASUREMENT_DEFINITION",
    "CAMERA_MEASUREMENT_KEY",
    "CAMERA_MEASUREMENT_LOGIC_NODE",
    "CameraMeasurementIntent",
    "CameraMeasurementRequest",
    "CAMERA_FRAME_OUTPUT_CONTRACT_ID",
    "DEFAULT_CAMERA_FRAMES_PER_CYCLE",
    "DEFAULT_CAMERA_MEASUREMENT_REPEAT",
    "DEFAULT_CAMERA_MEASUREMENT_ROLE",
    "MINIMUM_CAMERA_FRAMES_PER_CYCLE",
    "MINIMUM_CAMERA_MEASUREMENT_REPEAT",
    "camera_measurement_authoring_schema",
    "build_camera_measurement_intent_from_authoring",
    "camera_measurement_default_role",
    "camera_measurement_roles",
    "camera_frame_output_index",
    "camera_frame_output_declarations",
    "project_camera_measurement_outputs",
    "camera_measurement_final_outputs",
    "project_camera_monitor_outputs",
]
