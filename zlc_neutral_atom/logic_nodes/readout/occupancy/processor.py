"""Camera-frame to neutral-atom occupancy product.

The physical operation is deliberately small: bind one loaded calibration,
apply its selected readout model to every ``(R, P)`` camera cell, and preserve
the model's SITE axis and component validity.  Persistence, cursor ownership,
publication, and lifecycle belong to the generic Logic-node host.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from zlc_data import (
    SITE,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetSchema,
    OwnedSnapshot,
    ValidityContract,
    ValueSchema,
)
from zlc_data.value import dataset_cell_value, expand_dataset_validity
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
)
from zlc_neutral_atom.input_spec import DatasetInputSpec
from zlc_neutral_atom.logic_nodes.readout.calibration.calibration import (
    ResolvedCalibration,
    apply_readout_model,
)
from zlc_neutral_atom.logic_nodes.readout.contracts import FrameContract
from zlc_neutral_atom.logic_nodes.readout.model_contract import (
    ReadoutModelKind,
    readout_model_authoring_schema,
    readout_model_kind_from_authoring,
)
from zlc_neutral_atom.runtime.dataset import MonitorCoverage

_COUNTS_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "counts",
    "zlc_neutral_atom.occupancy.counts",
)
_OCCUPIED_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "occupied",
    "zlc_neutral_atom.occupancy.occupied",
)
_RATE_OUTPUT_DECLARATION = DatasetOutputDeclaration(
    "rate",
    "zlc_neutral_atom.occupancy.rate",
)
OCCUPANCY_LIVE_OUTPUT_DECLARATIONS = (
    _COUNTS_OUTPUT_DECLARATION,
    _OCCUPIED_OUTPUT_DECLARATION,
    _RATE_OUTPUT_DECLARATION,
)


_COUNTS_BLOCK_ID = BlockId("occupancy-counts")
_OCCUPIED_BLOCK_ID = BlockId("occupancy-occupied")
_RATE_BLOCK_ID = BlockId("occupancy-rate")

OCCUPANCY_CAMERA_INPUT_SPEC = DatasetInputSpec(
    "camera_frame",
    "Frame source",
    None,
    description=(
        "Current Dataset whose declared cell schema matches the selected "
        "Calibration frame schema; Occupancy never starts a device"
    ),
)
@dataclass(frozen=True, slots=True)
class OccupancyProcessorConfig:
    """Operator-authored model choice before input binding."""

    model_kind: ReadoutModelKind | None = None

    def __post_init__(self) -> None:
        if self.model_kind is not None and not isinstance(
            self.model_kind,
            ReadoutModelKind,
        ):
            raise TypeError("model_kind must be ReadoutModelKind or None")


def occupancy_authoring_schema():
    return readout_model_authoring_schema()


def build_occupancy_processor_config(
    values: Mapping[str, object],
) -> OccupancyProcessorConfig:
    authored = occupancy_authoring_schema().freeze(values)
    return OccupancyProcessorConfig(
        readout_model_kind_from_authoring(authored["model_kind"])
    )


def _output_schemas(
    frame_contract: FrameContract,
    site_axis: AxisSpec,
) -> tuple[ValueSchema, ValueSchema]:
    validity = ValidityContract.components(site_axis.axis_id)
    return (
        ValueSchema(
            (site_axis,),
            validity,
            np.dtype("<f8"),
            frame_contract.count_unit,
        ),
        ValueSchema((site_axis,), validity, np.dtype(bool), "occupation"),
    )


def _apply_occupancy_snapshot(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> tuple[OwnedSnapshot, OwnedSnapshot]:
    """Classify an immutable Camera dataset without reacquiring it."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be ResolvedCalibration")
    schema = source.block.schema
    artifact = calibration.artifact
    if schema.cell_schema != artifact.frame_contract.frame_schema:
        raise ValueError("source schema differs from the calibration FrameContract")
    model = artifact.select_model(model_kind)
    site_axis = model.feature.site_axis
    counts_value, occupied_value = _output_schemas(
        artifact.frame_contract,
        site_axis,
    )
    outer = (schema.repeat_axis, schema.point_table, schema.grid_topology)
    counts_schema = DatasetSchema(*outer, counts_value)
    occupied_schema = DatasetSchema(*outer, occupied_value)
    counts_values = np.zeros(counts_schema.physical_shape, dtype="<f8")
    occupied_values = np.zeros(occupied_schema.physical_shape, dtype=bool)
    valid_values = np.zeros(counts_schema.physical_shape, dtype=bool)
    for repeat_index in range(schema.repeat_axis.size):
        for point_index in range(schema.point_table.row_count):
            result = apply_readout_model(
                model,
                dataset_cell_value(source.block, repeat_index, point_index),
                expected_frame_schema=artifact.frame_contract.frame_schema,
            )
            validity = result.occupied.validity
            if not isinstance(validity, ComponentValidity):
                raise TypeError("readout result requires ComponentValidity")
            location = (repeat_index, point_index)
            counts_values[location] = result.signals.values
            occupied_values[location] = result.occupied.values
            valid_values[location] = validity.mask
    validity = DatasetComponentValidity((site_axis.axis_id,), valid_values)
    revision = source.block.revision
    counts = DataBlock(
        _COUNTS_BLOCK_ID,
        revision,
        counts_values,
        validity,
        counts_schema,
    )
    occupied = DataBlock(
        _OCCUPIED_BLOCK_ID,
        revision,
        occupied_values,
        validity,
        occupied_schema,
    )
    generation = source.ref.stream_generation
    return (
        OwnedSnapshot(counts.ref(generation), counts),
        OwnedSnapshot(occupied.ref(generation), occupied),
    )


def _occupancy_rate_snapshot(occupied: OwnedSnapshot) -> OwnedSnapshot:
    """Reduce the declared SITE axis into a validity-aware occupancy rate."""

    if not isinstance(occupied, OwnedSnapshot):
        raise TypeError("occupied must be an OwnedSnapshot")
    schema = occupied.block.schema
    axes = schema.cell_schema.data_axes
    if len(axes) != 1 or axes[0].role != SITE:
        raise ValueError("occupancy rate requires one declared SITE axis")
    validity = np.asarray(
        expand_dataset_validity(occupied.block.validity, schema),
        dtype=bool,
    )
    values = np.asarray(occupied.block.values, dtype=bool)
    denominator = np.count_nonzero(validity, axis=2)
    cell_validity = denominator > 0
    rate = np.zeros(cell_validity.shape, dtype="<f8")
    np.divide(
        np.count_nonzero(values & validity, axis=2),
        denominator,
        out=rate,
        where=cell_validity,
    )
    rate_schema = DatasetSchema(
        schema.repeat_axis,
        schema.point_table,
        schema.grid_topology,
        ValueSchema.scalar(np.dtype("<f8"), None),
    )
    block = DataBlock(
        _RATE_BLOCK_ID,
        occupied.block.revision,
        rate[..., np.newaxis],
        CellValidity(cell_validity),
        rate_schema,
    )
    return OwnedSnapshot(block.ref(occupied.ref.stream_generation), block)


def _evaluate_occupancy_processor(
    source: OwnedSnapshot,
    calibration: ResolvedCalibration,
    coverage: MonitorCoverage,
    *,
    model_kind: ReadoutModelKind | None = None,
) -> Mapping[str, LiveDatasetOutput]:
    """Classify one current Camera revision and select its current display cell."""

    if not isinstance(source, OwnedSnapshot):
        raise TypeError("source must be OwnedSnapshot")
    if type(calibration) is not ResolvedCalibration:
        raise TypeError("calibration must be ResolvedCalibration")
    if not isinstance(coverage, MonitorCoverage):
        raise TypeError("coverage must be MonitorCoverage")
    model = calibration.artifact.select_model(model_kind)
    counts, occupied = _apply_occupancy_snapshot(
        source,
        calibration,
        model_kind=model.kind,
    )
    rate = _occupancy_rate_snapshot(occupied)
    source_schema = source.block.schema
    if coverage.written_cells != 1 or coverage.total_cells != 1:
        raise ValueError("Occupancy requires one committed public Camera frame cell")
    if source_schema.repeat_axis.size != 1 or (
        source_schema.point_table.row_count != 1
    ):
        raise ValueError("Occupancy requires a public Camera frame with R=1 and P=1")
    outputs = {
        declaration.name: LiveDatasetOutput(
            declaration,
            snapshot,
            coverage,
        )
        for declaration, snapshot in zip(
            OCCUPANCY_LIVE_OUTPUT_DECLARATIONS,
            (counts, occupied, rate),
            strict=True,
        )
    }
    return outputs


__all__ = [
    "OCCUPANCY_CAMERA_INPUT_SPEC",
    "OCCUPANCY_LIVE_OUTPUT_DECLARATIONS",
    "OccupancyProcessorConfig",
    "build_occupancy_processor_config",
    "occupancy_authoring_schema",
]
