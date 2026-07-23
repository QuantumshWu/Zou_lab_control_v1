"""MOT coil-field optimization over one autonomous SCAN_SLOT camera scan.

The exact scan remains the sole hardware and persistence owner.  This module
adds only the neutral-atom physics that consumes its FINAL artifact:

* freeze the three semantic coil DAC axes into one autonomous pulse table;
* reduce each camera frame with the experiment's circular ROI-minus-annulus
  fluorescence rule; and
* refine the grid argmax by a local centre of mass.

There is no host-stepped fallback and no child-plan/workflow engine.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import product
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from zlc_data import (
    SPATIAL_X,
    SPATIAL_Y,
    AxisSpec,
    expand_dataset_validity,
    immutable_array,
)
from zlc_neutral_atom.catalog import DefinitionKey, TaskDefinition
from zlc_neutral_atom.installation import DeviceRef
from zlc_neutral_atom.scan import AutonomousScanSlotProgram, ScanArtifactRef
from zlc_neutral_atom.scan.repository import MaterializedScanData
from zlc_pulse import FrozenScanTable, PulseDocument, freeze_scan_table
from zlc_pulse.document import FIELD_DAC
from zlc_storage import canonical_text, finite_real, positive_real


MOT_SCAN_PARAMETER_IDS = ("da_x", "da_y", "da_z")
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


def _axis_codes(center: float, span: float, points: int) -> tuple[int, ...]:
    center = finite_real(center, "MOT axis centre")
    span = finite_real(span, "MOT axis span", minimum=0.0)
    if isinstance(points, bool) or not isinstance(points, Integral):
        raise TypeError("MOT points must be an integer")
    points = int(points)
    if points < 2:
        raise ValueError("MOT points must be at least 2")
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
    roi_radius: float = 8.0
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


def mot_roi_intensity(
    frame: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    *,
    validity: np.ndarray | None = None,
) -> float:
    """Mean circular-ROI counts minus the surrounding annulus mean.

    This is the physical rule from ``main``.  Current component validity is
    consumed explicitly: invalid pixels do not enter either mean, and a missing
    ROI/background support fails instead of manufacturing a scalar.
    """

    data = np.asarray(frame, dtype=float)
    if data.ndim != 2:
        raise ValueError(
            f"mot_roi_intensity takes one (H, W) frame; got shape {data.shape}"
        )
    cx = finite_real(cx, "roi centre x")
    cy = finite_real(cy, "roi centre y")
    radius = positive_real(radius, "roi radius")
    if validity is None:
        valid = np.ones(data.shape, dtype=bool)
    else:
        valid = np.asarray(validity, dtype=bool)
        if valid.shape != data.shape:
            raise ValueError("ROI validity shape differs from the frame")
    valid = valid & np.isfinite(data)

    height, width = data.shape
    yy, xx = np.mgrid[0:height, 0:width]
    radius_squared = (xx - cx) ** 2 + (yy - cy) ** 2
    disc = radius_squared <= radius**2
    ring = (radius_squared > radius**2) & (radius_squared <= (2.0 * radius) ** 2)
    disc_valid = disc & valid
    if not disc_valid.any():
        raise ValueError(
            f"MOT ROI (cx={cx}, cy={cy}, r={radius}) has no valid pixels "
            f"in the {height}x{width} frame"
        )
    if ring.any():
        ring_valid = ring & valid
        if not ring_valid.any():
            raise ValueError("MOT background annulus has no valid pixels")
        background = float(np.mean(data[ring_valid]))
    else:
        background = 0.0
    return float(np.mean(data[disc_valid]) - background)


def refine_mot_optimum(
    block: np.ndarray,
    axes: Sequence[Sequence[int | float]],
) -> tuple[tuple[float, float, float], float]:
    """Refine the grid argmax using main's local 3^n centre-of-mass rule."""

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
    if schema.point_axes != table.point_axes:
        raise ValueError("FINAL scan axes differ from the frozen MOT program")
    if schema.point_layout != table.point_layout:
        raise ValueError("FINAL scan point layout differs from the frozen MOT program")
    if schema.repeat_axis.size != 1:
        raise ValueError(
            "MOT optimization requires exactly one repeat; it never auto-reduces repeat"
        )
    data_axes = schema.cell_schema.data_axes
    if tuple(axis.role for axis in data_axes) != (SPATIAL_Y, SPATIAL_X):
        raise ValueError("MOT scan output must preserve one spatial-y/x camera frame")
    if schema.point_layout.storage_size != math.prod(schema.point_layout.logical_shape):
        raise ValueError("MOT scan requires the complete Cartesian coil grid")

    values = np.asarray(materialized.values)
    validity = expand_dataset_validity(materialized.validity, schema)
    expected_shape = schema.physical_shape
    if values.shape != expected_shape or validity.shape != expected_shape:
        raise ValueError("materialized MOT values/validity differ from their schema")
    logical = np.empty(schema.point_layout.logical_shape, dtype=np.float64)
    for storage_index in range(schema.point_layout.storage_size):
        frame = values[0, storage_index]
        frame_validity = validity[0, storage_index]
        height, width = frame.shape
        cx = width / 2.0 if request.roi_cx is None else request.roi_cx
        cy = height / 2.0 if request.roi_cy is None else request.roi_cy
        logical[schema.point_layout.multi_index(storage_index)] = mot_roi_intensity(
            frame,
            cx,
            cy,
            request.roi_radius,
            validity=frame_validity,
        )

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


__all__ = [
    "MOT_FIELD_REQUEST_SCHEMA",
    "MOT_FIELD_TASK_DEFINITION",
    "MOT_FIELD_TASK_DEFINITIONS",
    "MOT_FIELD_TASK_KEY",
    "MOT_SCAN_PARAMETER_IDS",
    "MotFieldRequest",
    "MotFieldResult",
    "analyze_mot_scan",
    "build_mot_scan_program",
    "mot_roi_intensity",
    "refine_mot_optimum",
]
