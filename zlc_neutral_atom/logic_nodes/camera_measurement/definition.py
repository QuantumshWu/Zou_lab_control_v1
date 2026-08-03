"""Camera Measurement request and lossless public output contract."""

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
from zlc_neutral_atom.authoring import (
    MINIMUM_POSITIVE_FLOAT,
    AuthoringField,
    AuthoringSchema,
)
from zlc_neutral_atom.catalog import DefinitionKey, LogicNodeDefinition
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    LiveDatasetOutput,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage, MonitorDatasetSnapshot
from zlc_storage import canonical_text, positive_integer, positive_real


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
CAMERA_MEASUREMENT_DEFINITION = LogicNodeDefinition(
    CAMERA_MEASUREMENT_KEY,
    "Camera",
    "measurement",
)


_CAMERA_MEASUREMENT_AUTHORING_SCHEMA = AuthoringSchema(
    (
        AuthoringField(
            "camera_instance_id",
            "choice",
            "Camera",
            required=True,
            dynamic_choices=True,
            description="Installed Camera instance",
        ),
        AuthoringField(
            "frames_per_cycle",
            "int",
            "Frames per cycle",
            default=DEFAULT_CAMERA_FRAMES_PER_CYCLE,
            required=True,
            minimum=MINIMUM_CAMERA_FRAMES_PER_CYCLE,
            allow_blank=False,
            description="Ordered sibling frame outputs in each camera cycle",
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
            description="Blank keeps the installed camera working point",
        ),
        AuthoringField(
            "repeat",
            "int",
            "Repeat",
            default=DEFAULT_CAMERA_MEASUREMENT_REPEAT,
            required=True,
            minimum=MINIMUM_CAMERA_MEASUREMENT_REPEAT,
            allow_blank=False,
            description="0 is live; a positive value captures that many cycles",
        ),
    )
)


def camera_measurement_authoring_schema() -> AuthoringSchema:
    return _CAMERA_MEASUREMENT_AUTHORING_SCHEMA


@dataclass(frozen=True, slots=True)
class CameraMeasurementRequest:
    """One Camera acquisition; only completion policy differs for live/finite."""

    camera_instance_id: str
    repeat: int = DEFAULT_CAMERA_MEASUREMENT_REPEAT
    frames_per_cycle: int = DEFAULT_CAMERA_FRAMES_PER_CYCLE
    exposure_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "camera_instance_id",
            canonical_text(self.camera_instance_id, "camera_instance_id"),
        )
        if isinstance(self.repeat, bool) or not isinstance(self.repeat, int):
            raise TypeError("repeat must be an integer")
        if self.repeat < MINIMUM_CAMERA_MEASUREMENT_REPEAT:
            raise ValueError("repeat must be non-negative")
        object.__setattr__(
            self,
            "frames_per_cycle",
            positive_integer(self.frames_per_cycle, "frames_per_cycle"),
        )
        if self.exposure_seconds is not None:
            object.__setattr__(
                self,
                "exposure_seconds",
                positive_real(self.exposure_seconds, "exposure_seconds"),
            )

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(output.name for output in self.output_declarations)

    @property
    def output_declarations(self) -> tuple[DatasetOutputDeclaration, ...]:
        return camera_frame_output_declarations(self.frames_per_cycle)


def build_camera_measurement_request(
    values: Mapping[str, object],
) -> CameraMeasurementRequest:
    authored = camera_measurement_authoring_schema().freeze(values)
    return CameraMeasurementRequest(
        camera_instance_id=authored["camera_instance_id"],
        repeat=authored["repeat"],
        frames_per_cycle=authored["frames_per_cycle"],
        exposure_seconds=authored["exposure"],
    )


def camera_frame_output_index(output_name: str) -> int:
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


def _finite_camera_event_column(
    schema: DatasetSchema,
    frames_per_cycle: int,
) -> PointColumn:
    event_columns = tuple(
        column for column in schema.point_table.columns if column.role == READOUT_EVENT
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
    request: CameraMeasurementRequest,
) -> dict[str, OwnedSnapshot]:
    """Split an exact R×E×frame capture into atomic R×1×frame siblings."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("camera frame source must be OwnedSnapshot")
    if not isinstance(request, CameraMeasurementRequest):
        raise TypeError("camera output projection requires CameraMeasurementRequest")
    event_column = _finite_camera_event_column(
        source.block.schema,
        request.frames_per_cycle,
    )
    return {
        output_name: _materialize_camera_frame(
            source,
            event_column,
            event_index,
            event_index,
        )
        for event_index, output_name in enumerate(request.output_names)
    }


def camera_measurement_final_outputs(
    source: OwnedSnapshot,
    request: CameraMeasurementRequest,
) -> dict[str, FinalDatasetOutput]:
    snapshots = project_camera_measurement_outputs(source, request)
    return {
        declaration.name: FinalDatasetOutput(
            declaration,
            snapshots[declaration.name],
        )
        for declaration in request.output_declarations
    }


def _materialize_camera_monitor_frame(
    source: OwnedSnapshot,
    event_axis: AxisSpec,
    event_index: int,
) -> OwnedSnapshot:
    cycle_schema = source.block.schema.cell_schema
    if cycle_schema.data_axes[0] != event_axis:
        raise ValueError("Camera monitor event axis differs from its cycle schema")
    component_ids = cycle_schema.validity_contract.component_axis_ids
    if not component_ids or component_ids[0] != event_axis.axis_id:
        raise ValueError("Camera monitor cycle validity must begin with READOUT_EVENT")
    frame_component_ids = component_ids[1:]
    frame_schema = ValueSchema(
        cycle_schema.data_axes[1:],
        (
            ValidityContract.value()
            if not frame_component_ids
            else ValidityContract.components(*frame_component_ids)
        ),
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


def project_camera_monitor_outputs(
    source: MonitorDatasetSnapshot,
    request: CameraMeasurementRequest,
) -> dict[str, LiveDatasetOutput]:
    """Split one latest complete cycle into `(1,1,*frame)` siblings."""

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
        event_index = camera_frame_output_index(declaration.name)
        snapshot = _materialize_camera_monitor_frame(
            source.snapshot,
            event_axis,
            event_index,
        )
        projected[declaration.name] = LiveDatasetOutput(
            declaration,
            snapshot,
            MonitorCoverage(
                written_cells=source.coverage.written_cells,
                total_cells=1,
                missed_events=source.coverage.missed_events,
                current_gap=source.coverage.current_gap,
            ),
        )
    return projected


__all__ = [
    "CAMERA_FRAME_OUTPUT_CONTRACT_ID",
    "CAMERA_MEASUREMENT_DEFINITION",
    "CAMERA_MEASUREMENT_KEY",
    "CameraMeasurementRequest",
    "DEFAULT_CAMERA_FRAMES_PER_CYCLE",
    "DEFAULT_CAMERA_MEASUREMENT_REPEAT",
    "MINIMUM_CAMERA_FRAMES_PER_CYCLE",
    "MINIMUM_CAMERA_MEASUREMENT_REPEAT",
    "build_camera_measurement_request",
    "camera_frame_output_declarations",
    "camera_frame_output_index",
    "camera_measurement_authoring_schema",
    "camera_measurement_final_outputs",
    "project_camera_measurement_outputs",
    "project_camera_monitor_outputs",
]
