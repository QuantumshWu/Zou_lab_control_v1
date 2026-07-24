"""MOT coil-field optimization over one autonomous SCAN_SLOT camera scan.

The exact scan remains the sole hardware and persistence owner.  This module
adds only the neutral-atom physics that consumes its FINAL artifact:

* freeze the three semantic coil DAC axes into one autonomous pulse table;
* reduce each camera frame with the experiment's circular ROI-minus-annulus
  fluorescence rule; and
* refine the grid argmax by a local centre of mass.

Autonomous SCAN_SLOT execution is the normal path; this module contains only
the concrete MOT scan physics and result materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral, Real
from typing import TYPE_CHECKING, Sequence

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
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
    expand_dataset_validity,
    immutable_array,
)
from zlc_neutral_atom.catalog import DefinitionKey, TaskDefinition
from zlc_neutral_atom.dataset_output import (
    FinalDatasetOutput,
    final_dataset_join_digest,
)
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.scan import AutonomousScanSlotProgram, ScanArtifactRef
from zlc_neutral_atom.scan.reference import scan_artifact_ref_to_tree
from zlc_neutral_atom.scan.repository import MaterializedScanData
from zlc_pulse import FrozenScanTable, PulseDocument, freeze_scan_table
from zlc_pulse.document import FIELD_DAC
from zlc_storage import canonical_digest, canonical_text, finite_real, positive_real

if TYPE_CHECKING:
    from zlc_neutral_atom.scan.source_binding import ScanRequest


MOT_SCAN_PARAMETER_IDS = ("da_x", "da_y", "da_z")
MOT_FIELD_FINAL_OUTPUT_NAMES = ("mot_field", "scan")
MOT_FIELD_REQUEST_SCHEMA = "zlc_neutral_atom.MotFieldRequest"
MOT_FIELD_TASK_KEY = DefinitionKey(
    "zlc_neutral_atom.mot_field",
    "optimize-mot-field",
)
MOT_FIELD_TASK_DEFINITION = TaskDefinition(
    MOT_FIELD_TASK_KEY,
    "Optimize MOT field",
    MOT_FIELD_REQUEST_SCHEMA,
)
MOT_FIELD_TASK_DEFINITIONS = (MOT_FIELD_TASK_DEFINITION,)
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
) -> AutonomousScanSlotProgram:
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
    return AutonomousScanSlotProgram(committed)


@dataclass(frozen=True)
class MotFieldRequest:
    """One immutable MOT optimization intent over an autonomous scan."""

    program: AutonomousScanSlotProgram
    camera_ref: DeviceRef
    sequencer_ref: DeviceRef
    roi_cx: float | None = None
    roi_cy: float | None = None
    roi_radius: float = DEFAULT_MOT_FIELD_ROI_RADIUS_PX
    trigger_channel: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.program, AutonomousScanSlotProgram):
            raise TypeError("program must be AutonomousScanSlotProgram")
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

    def as_scan_request(self) -> "ScanRequest":
        """Expose the exact Camera scan owned by this MOT request.

        The notebook supplies installation identities when constructing this
        request; the MOT domain, not that facade, owns how its physics intent
        maps onto the generic scan application.
        """

        from zlc_neutral_atom.scan.source_binding import ScanRequest

        return ScanRequest(
            program=self.program,
            camera_ref=self.camera_ref,
            sequencer_ref=self.sequencer_ref,
            trigger_channel=self.trigger_channel,
            output_transform_spec=None,
        )


@dataclass(frozen=True, eq=False)
class MotFieldResult:
    """Computed MOT optimum retaining the exact source scan and full 3-D block."""

    scan_ref: ScanArtifactRef
    point_axes: tuple[AxisSpec, AxisSpec, AxisSpec]
    intensity: np.ndarray
    best_field: tuple[float, float, float]
    best_intensity: float
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.scan_ref, ScanArtifactRef):
            raise TypeError("scan_ref must be ScanArtifactRef")
        axes = tuple(self.point_axes)
        if len(axes) != 3 or any(not isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("point_axes must contain the three MOT AxisSpec values")
        shape = tuple(axis.size for axis in axes)
        block = immutable_array(self.intensity, dtype=np.float64, shape=shape)
        if not np.isfinite(block).all():
            raise ValueError("MOT intensity block must be fully finite")
        best = tuple(float(value) for value in self.best_field)
        if len(best) != 3 or not all(math.isfinite(value) for value in best):
            raise ValueError("best_field must contain three finite values")
        peak = float(self.best_intensity)
        if not math.isfinite(peak):
            raise ValueError("best_intensity must be finite")
        object.__setattr__(self, "point_axes", axes)
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


def mot_roi_intensity(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    validity: np.ndarray | None = None,
) -> float:
    """Mean circular-ROI counts minus the surrounding annulus mean."""

    data = np.asarray(frame)
    if data.ndim != 2:
        raise ValueError(
            f"mot_roi_intensity takes one (H, W) frame; got shape {data.shape}"
        )
    return MotRoiProjector(data.shape, cx, cy, radius).intensity(
        data,
        validity=validity,
    )


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
        source_schema.point_axes,
        source_schema.point_layout,
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
    table = request.program.point_table
    if schema.point_axes != table.point_axes:
        raise ValueError("MOT scan axes differ from the frozen MOT program")
    if schema.point_layout != table.point_layout:
        raise ValueError("MOT point layout differs from the frozen MOT program")
    if schema.repeat_axis.size != 1:
        raise ValueError(
            "MOT optimization requires exactly one repeat; it never auto-reduces repeat"
        )
    data_axes = schema.cell_schema.data_axes
    if tuple(axis.role for axis in data_axes) != (SPATIAL_Y, SPATIAL_X):
        raise ValueError("MOT scan output must preserve one spatial-y/x camera frame")
    if schema.point_layout.storage_size != math.prod(schema.point_layout.logical_shape):
        raise ValueError("MOT scan requires the complete Cartesian coil grid")


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
        or written_cells > schema.point_layout.storage_size
    ):
        raise ValueError("written_cells differs from the MOT point layout")
    array = np.asarray(values)
    expanded_validity = expand_dataset_validity(validity, schema)
    if array.shape != schema.physical_shape or expanded_validity.shape != array.shape:
        raise ValueError("MOT values/validity differ from their schema")
    projector = build_mot_intensity_projector(request, schema)
    intensities = np.zeros(schema.point_layout.storage_size, dtype=np.float64)
    present = np.zeros(schema.point_layout.storage_size, dtype=bool)
    for storage_index in range(int(written_cells)):
        frame = array[0, storage_index]
        frame_validity = expanded_validity[0, storage_index]
        intensities[storage_index] = projector.intensity(
            frame,
            validity=frame_validity,
        )
        present[storage_index] = True
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
    materialized: MaterializedScanData,
) -> MotFieldResult:
    """Analyze the exact FINAL scan without flattening or implicit reduction."""

    if not isinstance(request, MotFieldRequest):
        raise TypeError("request must be MotFieldRequest")
    if not isinstance(materialized, MaterializedScanData):
        raise TypeError("materialized must be MaterializedScanData")
    schema = materialized.schema
    table = request.program.point_table
    storage_values, present = _mot_storage_intensities(
        request,
        materialized.values,
        materialized.validity,
        schema,
        written_cells=schema.point_layout.storage_size,
    )
    if not present.all():
        raise RuntimeError("FINAL MOT scan is missing intensity cells")
    logical = np.empty(schema.point_layout.logical_shape, dtype=np.float64)
    for storage_index in range(schema.point_layout.storage_size):
        logical[schema.point_layout.multi_index(storage_index)] = storage_values[
            storage_index
        ]

    axes = tuple(
        tuple(axis.coordinates or tuple(range(axis.size)))
        for axis in table.point_axes
    )
    best, peak = refine_mot_optimum(logical, axes)
    return MotFieldResult(
        materialized.artifact_ref,
        table.point_axes,
        logical,
        best,
        peak,
    )


def materialize_mot_field_snapshot(result: MotFieldResult) -> OwnedSnapshot:
    """Express a typed logical 3-D MOT result in Dataset storage order."""

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    axes = tuple(result.point_axes)
    layout = PointLayout.rect_c(tuple(axis.size for axis in axes))
    physical = np.empty((1, layout.storage_size, 1), dtype="<f8")
    for storage_index in range(layout.storage_size):
        physical[0, storage_index, 0] = result.intensity[
            layout.multi_index(storage_index)
        ]
    identity = canonical_digest(
        {
            "owner": "zlc_neutral_atom.mot-field-result",
            "repository_id": result.scan_ref.repository_id,
            "scan_manifest": result.scan_ref.manifest_digest,
            "axes": tuple(
                tuple(axis.coordinate_at(index) for index in range(axis.size))
                for axis in result.point_axes
            ),
            "intensity": np.asarray(result.intensity).tolist(),
            "best_field": result.best_field,
            "best_intensity": result.best_intensity,
        }
    )
    schema = DatasetSchema(
        AxisSpec(
            AxisId("mot-field.repeat"),
            "repeat",
            REPEAT,
            1,
            (0,),
        ),
        axes,
        layout,
        ValueSchema.scalar(np.dtype("<f8"), "counts"),
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
    source_scan: MaterializedScanData,
) -> dict[str, FinalDatasetOutput]:
    """Publish the analyzed MOT grid and its exact source scan together."""

    if not isinstance(result, MotFieldResult):
        raise TypeError("result must be MotFieldResult")
    if not isinstance(source_scan, MaterializedScanData):
        raise TypeError("source_scan must be MaterializedScanData")
    if source_scan.artifact_ref != result.scan_ref:
        raise ValueError("MOT result and materialized source name different scans")
    source_identity = scan_artifact_ref_to_tree(result.scan_ref)
    snapshots = dict(
        zip(
            MOT_FIELD_FINAL_OUTPUT_NAMES,
            (materialize_mot_field_snapshot(result), source_scan.snapshot),
            strict=True,
        )
    )
    return {
        name: FinalDatasetOutput(
            name,
            snapshot,
            final_dataset_join_digest(
                owner="mot-field",
                output_name=name,
                source_identity=source_identity,
                snapshot=snapshot,
            ),
        )
        for name, snapshot in snapshots.items()
    }


__all__ = [
    "DEFAULT_MOT_FIELD_CAMERA_ROLE",
    "DEFAULT_MOT_FIELD_CENTER_CODE",
    "DEFAULT_MOT_FIELD_POINTS",
    "DEFAULT_MOT_FIELD_ROI_CENTER_PX",
    "DEFAULT_MOT_FIELD_ROI_RADIUS_PX",
    "DEFAULT_MOT_FIELD_SPAN_CODE",
    "MOT_FIELD_REQUEST_SCHEMA",
    "MOT_FIELD_FINAL_OUTPUT_NAMES",
    "MOT_FIELD_TASK_DEFINITION",
    "MOT_FIELD_TASK_DEFINITIONS",
    "MOT_FIELD_TASK_KEY",
    "MOT_SCAN_PARAMETER_IDS",
    "MINIMUM_MOT_FIELD_POINTS",
    "MotFieldRequest",
    "MotFieldResult",
    "MotRoiProjector",
    "analyze_mot_scan",
    "build_mot_intensity_projector",
    "build_mot_scan_program",
    "mot_intensity_schema",
    "materialize_mot_field_snapshot",
    "mot_field_final_outputs",
    "mot_roi_intensity",
    "refine_mot_optimum",
]
