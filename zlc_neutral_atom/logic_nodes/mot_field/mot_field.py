"""MOT coil-field optimization over one autonomous SCAN_SLOT camera acquisition.

This capability owns the coupled Camera + Sequencer experiment:

* freeze the three semantic coil DAC axes into one autonomous pulse table;
* reduce each camera frame with the experiment's circular ROI-minus-annulus
  fluorescence rule; and
* refine the grid argmax by a local centre of mass.

Autonomous SCAN_SLOT execution is the only path.  Generic capture/pulse owners
provide exact transport and hardware commands; generic PulseScan is not part of
this task's application or result model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from zlc_data import (
    REPEAT,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    GridTopology,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    READOUT_EVENT,
    SCAN_POINT,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    expand_dataset_validity,
    grid_topology_to_tree,
    immutable_array,
    point_table_to_tree,
)
from zlc_neutral_atom.catalog import DefinitionKey, TaskDefinition
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    FinalDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.runtime.dataset import DatasetSealProvenance
from zlc_pulse import FrozenScanTable, PulseDocument, freeze_scan_table
from zlc_pulse.document import FIELD_DAC
from zlc_storage import canonical_digest, canonical_text, finite_real, positive_real

MOT_SCAN_PARAMETER_IDS = ("da_x", "da_y", "da_z")
MOT_FIELD_FINAL_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("mot_field", "zlc_neutral_atom.mot-field.result"),
    DatasetOutputDeclaration("scan", "zlc_neutral_atom.mot-field.source-scan"),
)
MOT_FIELD_REQUEST_SCHEMA = "zlc_neutral_atom.MotFieldRequest"
MOT_FIELD_TASK_KEY = DefinitionKey(
    "zlc_neutral_atom.logic_nodes.mot_field",
    "optimize-mot-field",
)
MOT_FIELD_TASK_DEFINITION = TaskDefinition(
    MOT_FIELD_TASK_KEY,
    "Optimize MOT field",
    MOT_FIELD_REQUEST_SCHEMA,
)
DEFAULT_MOT_FIELD_CAMERA_ROLE = "mot_camera"
DEFAULT_MOT_FIELD_CENTER_CODE = 0.0
DEFAULT_MOT_FIELD_SPAN_CODE = 12.0
DEFAULT_MOT_FIELD_POINTS = 7
MINIMUM_MOT_FIELD_POINTS = 2
DEFAULT_MOT_FIELD_ROI_CENTER_PX = 0.0
DEFAULT_MOT_FIELD_ROI_RADIUS_PX = 8.0


def _axis_codes(center: float, span: float, points: int) -> tuple[int, ...]:
    center = finite_real(center, "MOT axis centre")
    span = finite_real(span, "MOT axis span", minimum=0.0)
    if isinstance(points, bool) or not isinstance(points, Integral):
        raise TypeError("MOT points must be an integer")
    points = int(points)
    if points < MINIMUM_MOT_FIELD_POINTS:
        raise ValueError(
            "MOT points must be at least "
            f"{MINIMUM_MOT_FIELD_POINTS}"
        )
    values = np.unique(
        np.round(np.linspace(center - span, center + span, points)).astype(int)
    )
    if values.size < 2:
        raise ValueError(
            "MOT span and point count collapse to fewer than two DAC codes"
        )
    return tuple(int(value) for value in values)


def build_mot_scan_program(
    document: PulseDocument,
    *,
    center_x: float,
    center_y: float,
    center_z: float,
    span: float,
    points: int,
) -> "MotFieldProgram":
    """Freeze the complete x/y/z grid into the current pulse document.

    The semantic parameter ids are the experiment API.  Their target DAC ports
    remain owned by the PulseDocument/connected target binding.
    """

    if not isinstance(document, PulseDocument):
        raise TypeError("document must be PulseDocument")
    parameters = document.scan_parameter_by_id
    if tuple(parameter.parameter_id for parameter in document.scan_parameters) != (
        MOT_SCAN_PARAMETER_IDS
    ):
        raise ValueError(
            "MOT pulse template must declare exactly da_x, da_y, da_z in that order"
        )
    fields = tuple(parameters[key].field for key in MOT_SCAN_PARAMETER_IDS)
    if any(field.kind != FIELD_DAC for field in fields):
        raise ValueError("MOT scan parameters must all bind DAC fields")
    ports = tuple(field.port for field in fields)
    if any(port is None for port in ports) or len(set(ports)) != 3:
        raise ValueError("MOT scan parameters must bind three distinct DAC ports")
    if document.api_parameters:
        raise ValueError(
            "MOT autonomous scan template cannot retain unresolved API parameters"
        )

    axes = (
        _axis_codes(center_x, span, points),
        _axis_codes(center_y, span, points),
        _axis_codes(center_z, span, points),
    )
    rows = tuple(product(*axes))
    frozen, _normalization = freeze_scan_table(
        document,
        MOT_SCAN_PARAMETER_IDS,
        rows,
    )
    if not isinstance(frozen, FrozenScanTable):
        raise TypeError("pulse owner returned a non-FrozenScanTable")
    committed = replace(document, scan_table=frozen, scan_recipe=None)
    coordinate_ids = tuple(
        AxisId(f"mot-field.{parameter_id}")
        for parameter_id in MOT_SCAN_PARAMETER_IDS
    )
    point_table = PointTable(
        len(rows),
        tuple(
            PointColumn(
                coordinate_id,
                parameters[parameter_id].label or parameter_id,
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(row[position] for row in rows),
                parameters[parameter_id].unit,
            )
            for position, (parameter_id, coordinate_id) in enumerate(
                zip(MOT_SCAN_PARAMETER_IDS, coordinate_ids, strict=True)
            )
        ),
    )
    topology = GridTopology(
        coordinate_ids,
        axes,
        tuple(product(*(range(len(axis)) for axis in axes))),
    )
    return MotFieldProgram(
        committed,
        point_table,
        topology,
    )


@dataclass(frozen=True)
class MotFieldProgram:
    """The complete three-axis pulse table owned by the MOT task."""

    document: PulseDocument
    point_table: PointTable
    grid_topology: GridTopology

    def __post_init__(self) -> None:
        if not isinstance(self.document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        if not isinstance(self.grid_topology, GridTopology):
            raise TypeError("grid_topology must be GridTopology")
        columns = self.point_table.columns
        if len(columns) != 3 or any(column.role != SCAN_POINT for column in columns):
            raise ValueError("MOT program requires three SCAN_POINT columns")
        if tuple(column.coordinate_id.value for column in columns) != tuple(
            f"mot-field.{parameter_id}" for parameter_id in MOT_SCAN_PARAMETER_IDS
        ):
            raise ValueError("MOT coordinate identities differ from da_x/da_y/da_z")
        if self.grid_topology.dimension_ids != tuple(
            column.coordinate_id for column in columns
        ):
            raise ValueError("MOT topology dimensions differ from its PointTable")
        table = self.document.scan_table
        if table is None or table.columns != MOT_SCAN_PARAMETER_IDS:
            raise ValueError("MOT program must freeze da_x, da_y, da_z")
        reconstructed = tuple(
            tuple(column.values[row] for column in columns)
            for row in range(self.point_table.row_count)
        )
        if reconstructed != table.rows:
            raise ValueError("MOT PointTable does not reproduce the frozen pulse table")
        if self.point_table.row_count != math.prod(self.grid_topology.logical_shape):
            raise ValueError("MOT scan requires the complete Cartesian coil grid")
        if len(self.grid_topology.row_to_cell) != self.point_table.row_count:
            raise ValueError("MOT topology must map every frozen point row")
        for ordinal, cell in enumerate(self.grid_topology.row_to_cell):
            if any(
                self.point_table.column(dimension_id).values[ordinal]
                != self.grid_topology.coordinate_domains[position][cell[position]]
                for position, dimension_id in enumerate(
                    self.grid_topology.dimension_ids
                )
            ):
                raise ValueError("MOT topology coordinates differ from its PointTable")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(
            {
                "owner": "zlc_neutral_atom.mot-field-program",
                "pulse_document": self.document.fingerprint,
                "point_table": point_table_to_tree(self.point_table),
                "grid_topology": grid_topology_to_tree(self.grid_topology),
            }
        )


@dataclass(frozen=True)
class MotFieldRequest:
    """One immutable MOT optimization intent over an autonomous scan."""

    program: MotFieldProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    roi_cx: float | None = None
    roi_cy: float | None = None
    roi_radius: float = DEFAULT_MOT_FIELD_ROI_RADIUS_PX
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.program, MotFieldProgram):
            raise TypeError("program must be MotFieldProgram")
        columns = self.program.document.scan_table
        if columns is None or columns.columns != MOT_SCAN_PARAMETER_IDS:
            raise ValueError("MOT program must freeze da_x, da_y, da_z")
        if not isinstance(self.camera_ref, DeviceRef):
            raise TypeError("camera_ref must be DeviceRef")
        if not isinstance(self.sequencer_ref, DeviceRef):
            raise TypeError("sequencer_ref must be DeviceRef")
        for field in ("roi_cx", "roi_cy"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, finite_real(value, field))
        object.__setattr__(
            self,
            "roi_radius",
            positive_real(self.roi_radius, "roi_radius"),
        )
        if self.trigger_channel is not None:
            object.__setattr__(
                self,
                "trigger_channel",
                canonical_text(self.trigger_channel, "trigger_channel"),
            )



def mot_field_source_identity(
    snapshot: OwnedSnapshot,
    provenance: DatasetSealProvenance,
) -> str:
    """Digest the exact source facts used by one MOT result."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if not isinstance(provenance, DatasetSealProvenance):
        raise TypeError("provenance must be DatasetSealProvenance")
    ref = snapshot.ref
    trace = provenance.trace_binding
    return canonical_digest(
        {
            "owner": "zlc_neutral_atom.mot-field-acquisition",
            "block_id": ref.block_id.value,
            "generation": ref.stream_generation.value,
            "schema": ref.schema_fingerprint,
            "revision": ref.revision.value,
            "stream_id": provenance.stream_id.value,
            "start_sequence": provenance.start_sequence,
            "end_sequence": provenance.end_sequence,
            "join_plan": provenance.join_plan_digest,
            "ordered_metadata": provenance.ordered_metadata_digest,
            "metadata_contract": provenance.metadata_contract_fingerprint,
            "run_id": trace.run_id,
            "source_id": trace.source_id,
        }
    )


@dataclass(frozen=True)
class MotFieldAcquisitionResult:
    """Exact raw Camera dataset from one autonomous MOT hardware run."""

    snapshot: OwnedSnapshot
    provenance: DatasetSealProvenance
    source_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.provenance, DatasetSealProvenance):
            raise TypeError("provenance must be DatasetSealProvenance")
        if self.snapshot.ref.stream_generation != self.provenance.generation:
            raise ValueError("MOT source snapshot and provenance generations differ")
        schema = self.snapshot.block.schema
        expected = schema.repeat_axis.size * schema.point_table.row_count
        if self.provenance.end_sequence - self.provenance.start_sequence != expected:
            raise ValueError("MOT source provenance does not cover every dataset cell")
        if self.snapshot.ref.revision.value != expected:
            raise ValueError("MOT source snapshot is not the complete exact revision")
        if self.source_identity != mot_field_source_identity(
            self.snapshot,
            self.provenance,
        ):
            raise ValueError("MOT source identity differs from its exact dataset")


@dataclass(frozen=True, eq=False)
class MotFieldResult:
    """Computed MOT optimum retaining the exact source scan and full 3-D block."""

    source_identity: str
    point_table: PointTable
    grid_topology: GridTopology
    intensity: np.ndarray
    best_field: tuple[float, float, float]
    best_intensity: float
    __hash__ = None

    def __post_init__(self) -> None:
        canonical_text(self.source_identity, "source_identity")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        if not isinstance(self.grid_topology, GridTopology):
            raise TypeError("grid_topology must be GridTopology")
        if (
            len(self.point_table.columns) != 3
            or self.grid_topology.dimension_ids
            != tuple(column.coordinate_id for column in self.point_table.columns)
            or len(self.grid_topology.row_to_cell) != self.point_table.row_count
            or self.point_table.row_count != math.prod(self.grid_topology.logical_shape)
        ):
            raise ValueError("MOT result requires one complete three-dimensional grid")
        shape = self.grid_topology.logical_shape
        block = immutable_array(self.intensity, dtype=np.float64, shape=shape)
        if not np.isfinite(block).all():
            raise ValueError("MOT intensity block must be fully finite")
        best = tuple(float(value) for value in self.best_field)
        if len(best) != 3 or not all(math.isfinite(value) for value in best):
            raise ValueError("best_field must contain three finite values")
        peak = float(self.best_intensity)
        if not math.isfinite(peak):
            raise ValueError("best_intensity must be finite")
        object.__setattr__(self, "intensity", block)
        object.__setattr__(self, "best_field", best)
        object.__setattr__(self, "best_intensity", peak)

    @property
    def best_x(self) -> float:
        return self.best_field[0]

    @property
    def best_y(self) -> float:
        return self.best_field[1]

    @property
    def best_z(self) -> float:
        return self.best_field[2]


class MotRoiProjector:
    """Reusable circular ROI geometry for one camera frame shape."""

    __slots__ = (
        "_frame_shape",
        "_cx",
        "_cy",
        "_radius",
        "_y_slice",
        "_x_slice",
        "_disc",
        "_ring",
    )

    def __init__(
        self,
        frame_shape: tuple[int, int],
        cx: float,
        cy: float,
        radius: float,
    ) -> None:
        shape = tuple(frame_shape)
        if (
            len(shape) != 2
            or any(
                isinstance(size, bool)
                or not isinstance(size, Integral)
                or size <= 0
                for size in shape
            )
        ):
            raise ValueError("frame_shape must contain positive (height, width)")
        height, width = (int(shape[0]), int(shape[1]))
        cx = finite_real(cx, "roi centre x")
        cy = finite_real(cy, "roi centre y")
        radius = positive_real(radius, "roi radius")
        extent = 2.0 * radius
        x0 = min(width, max(0, int(math.floor(cx - extent))))
        x1 = min(width, max(0, int(math.ceil(cx + extent)) + 1))
        y0 = min(height, max(0, int(math.floor(cy - extent))))
        y1 = min(height, max(0, int(math.ceil(cy + extent)) + 1))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        radius_squared = (xx - cx) ** 2 + (yy - cy) ** 2
        disc = np.asarray(radius_squared <= radius**2, dtype=bool)
        ring = np.asarray(
            (radius_squared > radius**2)
            & (radius_squared <= extent**2),
            dtype=bool,
        )
        disc.setflags(write=False)
        ring.setflags(write=False)
        self._frame_shape = (height, width)
        self._cx = cx
        self._cy = cy
        self._radius = radius
        self._y_slice = slice(y0, y1)
        self._x_slice = slice(x0, x1)
        self._disc = disc
        self._ring = ring

    @property
    def frame_shape(self) -> tuple[int, int]:
        return self._frame_shape

    def intensity(
        self,
        frame: np.ndarray,
        *,
        validity: np.ndarray | None = None,
    ) -> float:
        """Apply the frozen disc/annulus geometry to one frame."""

        source = np.asarray(frame)
        if source.shape != self._frame_shape:
            raise ValueError(
                f"MOT ROI expects frame shape {self._frame_shape}; got {source.shape}"
            )
        region = np.asarray(
            source[self._y_slice, self._x_slice],
            dtype=float,
        )
        if validity is not None:
            valid = np.asarray(validity, dtype=bool)
            if valid.shape != self._frame_shape:
                raise ValueError("ROI validity shape differs from the frame")
            region_valid = (
                valid[self._y_slice, self._x_slice] & np.isfinite(region)
            )
        else:
            region_valid = np.isfinite(region)
        disc_valid = self._disc & region_valid
        height, width = self._frame_shape
        if not disc_valid.any():
            raise ValueError(
                f"MOT ROI (cx={self._cx}, cy={self._cy}, r={self._radius}) "
                f"has no valid pixels in the {height}x{width} frame"
            )
        if self._ring.any():
            ring_valid = self._ring & region_valid
            if not ring_valid.any():
                raise ValueError("MOT background annulus has no valid pixels")
            background = float(np.mean(region[ring_valid]))
        else:
            background = 0.0
        return float(np.mean(region[disc_valid]) - background)


def build_mot_intensity_projector(
    request: MotFieldRequest,
    source_schema: DatasetSchema,
) -> MotRoiProjector:
    """Freeze one ROI geometry shared by live and FINAL MOT projection."""

    _validate_mot_source_schema(request, source_schema)
    height, width = source_schema.cell_schema.data_shape
    return MotRoiProjector(
        (height, width),
        width / 2.0 if request.roi_cx is None else request.roi_cx,
        height / 2.0 if request.roi_cy is None else request.roi_cy,
        request.roi_radius,
    )


def mot_intensity_schema(
    request: MotFieldRequest,
    source_schema: DatasetSchema,
) -> DatasetSchema:
    """Return the scalar Bx/By/Bz schema shared by live and FINAL analysis."""

    _validate_mot_source_schema(request, source_schema)
    return DatasetSchema(
        source_schema.repeat_axis,
        request.program.point_table,
        request.program.grid_topology,
        ValueSchema.scalar(
            np.dtype("<f8"),
            source_schema.cell_schema.value_unit,
        ),
    )


def _validate_mot_source_schema(
    request: MotFieldRequest,
    schema: DatasetSchema,
) -> None:
    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    program = request.program
    columns = schema.point_table.columns
    if (
        schema.point_table.row_count != program.point_table.row_count
        or len(columns) != len(program.point_table.columns) + 1
        or columns[:-1] != program.point_table.columns
    ):
        raise ValueError("MOT source rows differ from the frozen MOT program")
    event_column = columns[-1]
    if (
        event_column.role != READOUT_EVENT
        or event_column.value_kind != PointColumn.NUMERIC
        or event_column.values != (0,) * schema.point_table.row_count
    ):
        raise ValueError("MOT camera acquisition requires one singleton readout event")
    topology = schema.grid_topology
    if topology is None or (
        topology.dimension_ids
        != (*program.grid_topology.dimension_ids, event_column.coordinate_id)
        or topology.coordinate_domains
        != (*program.grid_topology.coordinate_domains, (0,))
        or topology.row_to_cell
        != tuple((*cell, 0) for cell in program.grid_topology.row_to_cell)
    ):
        raise ValueError("MOT source topology differs from the frozen grid")
    if schema.repeat_axis.size != 1:
        raise ValueError(
            "MOT optimization requires exactly one repeat; it never auto-reduces repeat"
        )
    data_axes = schema.cell_schema.data_axes
    if tuple(axis.role for axis in data_axes) != (SPATIAL_Y, SPATIAL_X):
        raise ValueError("MOT scan output must preserve one spatial-y/x camera frame")


def _mot_storage_intensities(
    request: MotFieldRequest,
    values: np.ndarray,
    validity,
    schema: DatasetSchema,
    *,
    written_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    _validate_mot_source_schema(request, schema)
    if (
        isinstance(written_cells, bool)
        or not isinstance(written_cells, Integral)
        or written_cells < 0
        or written_cells > schema.point_table.row_count
    ):
        raise ValueError("written_cells differs from the MOT PointTable")
    array = np.asarray(values)
    expanded_validity = expand_dataset_validity(validity, schema)
    if array.shape != schema.physical_shape or expanded_validity.shape != array.shape:
        raise ValueError("MOT values/validity differ from their schema")
    projector = build_mot_intensity_projector(request, schema)
    intensities = np.zeros(schema.point_table.row_count, dtype=np.float64)
    present = np.zeros(schema.point_table.row_count, dtype=bool)
    for point_ordinal in range(int(written_cells)):
        frame = array[0, point_ordinal]
        frame_validity = expanded_validity[0, point_ordinal]
        intensities[point_ordinal] = projector.intensity(
            frame,
            validity=frame_validity,
        )
        present[point_ordinal] = True
    return intensities, present


def refine_mot_optimum(
    block: np.ndarray,
    axes: Sequence[Sequence[int | float]],
) -> tuple[tuple[float, float, float], float]:
    """Refine the grid argmax using a local 3^n centre-of-mass rule."""

    values = np.asarray(block, dtype=float)
    coordinates = tuple(np.asarray(axis, dtype=float) for axis in axes)
    if values.ndim != 3 or len(coordinates) != 3:
        raise ValueError("MOT optimum requires one three-dimensional grid")
    if values.shape != tuple(axis.size for axis in coordinates):
        raise ValueError("MOT grid shape differs from its coordinate axes")
    if not np.isfinite(values).all():
        raise ValueError("MOT optimum cannot consume missing/non-finite grid cells")

    index = np.unravel_index(int(np.argmax(values)), values.shape)
    low = tuple(max(0, item - 1) for item in index)
    high = tuple(min(size, item + 2) for item, size in zip(index, values.shape))
    region = values[tuple(slice(start, stop) for start, stop in zip(low, high))]
    weights = np.clip(region - float(np.min(region)), 0.0, None)
    total = float(np.sum(weights))
    best: list[float] = []
    for dimension, axis in enumerate(coordinates):
        local = axis[low[dimension] : high[dimension]]
        shape = [1] * region.ndim
        shape[dimension] = region.shape[dimension]
        if total > 0.0:
            best.append(float(np.sum(weights * local.reshape(shape)) / total))
        else:
            best.append(float(axis[index[dimension]]))
    return (best[0], best[1], best[2]), float(values[index])


def analyze_mot_scan(
    request: MotFieldRequest,
    acquisition: MotFieldAcquisitionResult,
) -> MotFieldResult:
    """Analyze the exact FINAL scan without flattening or implicit reduction."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(acquisition, MotFieldAcquisitionResult):
        raise TypeError("acquisition must be MotFieldAcquisitionResult")
    source = acquisition.snapshot.block
    schema = source.schema
    program = request.program
    storage_values, present = _mot_storage_intensities(
        request,
        source.values,
        source.validity,
        schema,
        written_cells=schema.point_table.row_count,
    )
    if not present.all():
        raise RuntimeError("FINAL MOT scan is missing intensity cells")
    logical = np.empty(program.grid_topology.logical_shape, dtype=np.float64)
    for point_ordinal, cell in enumerate(program.grid_topology.row_to_cell):
        logical[cell] = storage_values[point_ordinal]

    axes = program.grid_topology.coordinate_domains
    best, peak = refine_mot_optimum(logical, axes)
    return MotFieldResult(
        acquisition.source_identity,
        program.point_table,
        program.grid_topology,
        logical,
        best,
        peak,
    )


def materialize_mot_field_snapshot(result: MotFieldResult) -> OwnedSnapshot:
    """Express a typed logical 3-D MOT result in Dataset storage order."""

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    schema = DatasetSchema(
        AxisSpec(
            AxisId("mot-field.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        result.point_table,
        result.grid_topology,
        ValueSchema.scalar(np.dtype("<f8"), "counts"),
    )
    physical = np.empty(schema.physical_shape, dtype="<f8")
    for point_ordinal, cell in enumerate(result.grid_topology.row_to_cell):
        physical[0, point_ordinal, 0] = result.intensity[cell]
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.mot-field-result",
            "source_identity": result.source_identity,
            "schema": schema.fingerprint,
            "intensity": np.asarray(result.intensity).tolist(),
            "best_field": result.best_field,
            "best_intensity": result.best_intensity,
        }
    )
    block = DataBlock(
        BlockId(f"mot-field-intensity-{identity[:20]}"),
        DatasetRevision(0),
        physical,
        VALID,
        schema,
    )
    generation = StreamGenerationId(f"mot-field-result-{identity}")
    return OwnedSnapshot(block.ref(generation), block)


def mot_field_final_outputs(
    result: MotFieldResult,
    source_scan: MotFieldAcquisitionResult,
) -> dict[str, FinalDatasetOutput]:
    """Publish the analyzed MOT grid and its exact source scan together."""

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    if not isinstance(source_scan, MotFieldAcquisitionResult):
        raise TypeError("source_scan must be MotFieldAcquisitionResult")
    if source_scan.source_identity != result.source_identity:
        raise ValueError("MOT result and materialized source name different scans")
    source_identity = {"mot_field_source_identity": result.source_identity}
    snapshots = (
        materialize_mot_field_snapshot(result),
        source_scan.snapshot,
    )
    return {
        declaration.name: FinalDatasetOutput(
            declaration,
            snapshot,
            final_dataset_join_digest(
                owner="mot-field",
                declaration=declaration,
                source_identity=source_identity,
                snapshot=snapshot,
            ),
        )
        for declaration, snapshot in zip(
            MOT_FIELD_FINAL_OUTPUT_DECLARATIONS,
            snapshots,
            strict=True,
        )
    }


__all__ = [
    "DEFAULT_MOT_FIELD_CAMERA_ROLE",
    "DEFAULT_MOT_FIELD_CENTER_CODE",
    "DEFAULT_MOT_FIELD_POINTS",
    "DEFAULT_MOT_FIELD_ROI_CENTER_PX",
    "DEFAULT_MOT_FIELD_ROI_RADIUS_PX",
    "DEFAULT_MOT_FIELD_SPAN_CODE",
    "MOT_FIELD_REQUEST_SCHEMA",
    "MOT_FIELD_FINAL_OUTPUT_DECLARATIONS",
    "MOT_FIELD_TASK_DEFINITION",
    "MOT_FIELD_TASK_KEY",
    "MOT_SCAN_PARAMETER_IDS",
    "MINIMUM_MOT_FIELD_POINTS",
    "MotFieldRequest",
    "MotFieldProgram",
    "MotFieldAcquisitionResult",
    "MotFieldResult",
    "MotRoiProjector",
    "analyze_mot_scan",
    "build_mot_intensity_projector",
    "build_mot_scan_program",
    "mot_intensity_schema",
    "materialize_mot_field_snapshot",
    "mot_field_final_outputs",
    "mot_field_source_identity",
    "refine_mot_optimum",
]
