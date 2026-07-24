"""Progressive MOT-field data projection over one exact camera scan.

This module owns the neutral-atom meaning of the live MOT grid: selecting the
scan output geometry, reducing each camera frame with the frozen MOT ROI,
preserving value validity, and materializing an exact ``R x P x (1)`` Dataset
front.  The task's run-scoped reader/notification lifetime is assembled by
``mot_field_task_live``; no desktop package owns either concern.
"""

from __future__ import annotations

import numpy as np

from zlc_data import (
    BlockId,
    CellValidity,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    expand_value_validity,
)
from .dataset_output import (
    LiveDatasetOutput,
    single_live_dataset_output,
)

from .mot_field import (
    MotFieldRequest,
    MotRoiProjector,
    build_mot_intensity_projector,
    mot_intensity_schema,
)
from .runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
)
from .scan import ScanOutputContract


MOT_FIELD_LIVE_OUTPUT_NAMES = ("grid",)


def _source_to_output_points(
    source: DatasetSchema,
    output: DatasetSchema,
) -> tuple[int, ...]:
    """Map a singleton readout selection without guessing or flattening axes."""

    if source.repeat_axis != output.repeat_axis:
        raise ValueError("MOT preview transform changed the repeat axis")
    if source.cell_schema != output.cell_schema:
        raise ValueError("MOT preview transform changed camera frame values")
    source_ids = tuple(axis.axis_id for axis in source.point_axes)
    output_ids = tuple(axis.axis_id for axis in output.point_axes)
    try:
        selected_positions = tuple(source_ids.index(axis_id) for axis_id in output_ids)
    except ValueError as error:
        raise ValueError("MOT output introduced a point axis") from error
    if tuple(source.point_axes[position] for position in selected_positions) != (
        output.point_axes
    ):
        raise ValueError("MOT output point axes differ from their source axes")
    omitted = tuple(
        position
        for position in range(len(source.point_axes))
        if position not in selected_positions
    )
    if not omitted or any(
        source.point_axes[position].size != 1 for position in omitted
    ):
        raise ValueError(
            "MOT live projection requires only singleton source point selections"
        )

    mapping: list[int] = []
    for storage_index in range(source.point_layout.storage_size):
        source_multi = source.point_layout.multi_index(storage_index)
        if any(source_multi[position] != 0 for position in omitted):
            raise ValueError(
                "MOT selected source point is not the singleton coordinate"
            )
        output_multi = tuple(source_multi[position] for position in selected_positions)
        mapping.append(output.point_layout.storage_index(output_multi))
    result = tuple(mapping)
    if len(result) != output.point_layout.storage_size or set(result) != set(
        range(output.point_layout.storage_size)
    ):
        raise ValueError("MOT source-to-output point mapping is not lossless")
    return result


class MotFieldLiveProjection:
    """Atomically accumulate exact camera deltas into one scalar MOT grid.

    The instance is single-consumer by design.  Each accepted delta produces a
complete immutable provisional front.  A rejected delta leaves the prior
front unchanged, so the application live-output owner can publish only data whose revision,
    validity, metadata, and coverage were sealed together by this owner.
    """

    def __init__(
        self,
        request: MotFieldRequest,
        source_schema: DatasetSchema,
        output_contract: ScanOutputContract,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_contract, ScanOutputContract):
            raise TypeError("output_contract must be ScanOutputContract")
        if (
            output_contract.committed_transform.input_schema_fingerprint
            != source_schema.fingerprint
        ):
            raise ValueError("MOT output contract belongs to another source schema")

        transformed = output_contract.output_dataset_schema
        self._source_schema = source_schema
        self._output_schema = mot_intensity_schema(request, transformed)
        self._output_block_id = BlockId("mot-field-live-grid")
        self._source_to_output_points = _source_to_output_points(
            source_schema,
            transformed,
        )
        self._projector: MotRoiProjector = build_mot_intensity_projector(
            request,
            transformed,
        )
        self._values = np.zeros(self._output_schema.physical_shape, dtype=np.float64)
        self._valid = np.zeros(self._output_schema.physical_shape[:2], dtype=bool)
        self._metadata: tuple[object | None, ...] = (None,) * self._valid.size
        self._revision = DatasetRevision(0)
        self._generation = None

    def consume(self, delta: DatasetPreviewDelta) -> DatasetPreviewSnapshot:
        """Project one contiguous exact delta and atomically publish its front."""

        if not isinstance(delta, DatasetPreviewDelta):
            raise TypeError("delta must be DatasetPreviewDelta")
        if delta.ref.schema_fingerprint != self._source_schema.fingerprint:
            raise ValueError("MOT delta belongs to another source schema")
        if delta.after != self._revision:
            raise RuntimeError(
                "MOT delta is not contiguous with the projected revision"
            )
        if delta.coverage.total_cells != self._valid.size:
            raise ValueError("MOT delta total coverage differs from output schema")
        if (
            self._generation is not None
            and delta.ref.stream_generation != self._generation
        ):
            raise RuntimeError("MOT live projection changed stream generation")

        projected: list[tuple[tuple[int, int], float, object | None]] = []
        for cell in delta.cells:
            address = cell.address
            if address.repeat_index >= self._valid.shape[0]:
                raise ValueError("MOT delta repeat address exceeds output schema")
            try:
                output_point = self._source_to_output_points[
                    address.point_storage_index
                ]
            except IndexError as error:
                raise ValueError(
                    "MOT delta point address exceeds the source schema"
                ) from error
            projected.append(
                (
                    (address.repeat_index, output_point),
                    self._projector.intensity(
                        cell.value.values,
                        validity=expand_value_validity(
                            cell.value.validity,
                            cell.value.schema,
                        ),
                    ),
                    cell.metadata,
                )
            )

        output_cells = tuple(item[0] for item in projected)
        if len(set(output_cells)) != len(output_cells) or any(
            self._valid[output_cell] for output_cell in output_cells
        ):
            raise RuntimeError("MOT delta attempted to rewrite a committed grid cell")

        values = np.array(self._values, copy=True)
        valid = np.array(self._valid, copy=True)
        metadata = list(self._metadata)
        for output_cell, intensity, cell_metadata in projected:
            values[(*output_cell, 0)] = intensity
            valid[output_cell] = True
            flat = output_cell[0] * valid.shape[1] + output_cell[1]
            metadata[flat] = cell_metadata
        written = int(np.count_nonzero(valid))
        if written != delta.coverage.written_cells:
            raise RuntimeError("MOT delta coverage differs from accumulated grid cells")

        ref = DatasetRevisionRef(
            self._output_block_id,
            delta.ref.stream_generation,
            self._output_schema.fingerprint,
            delta.ref.revision,
        )
        block = DataBlock(
            ref.block_id,
            ref.revision,
            values,
            CellValidity(valid),
            self._output_schema,
        )
        front = DatasetPreviewSnapshot(
            OwnedSnapshot(ref, block),
            DatasetCoverage(written, valid.size),
            tuple(metadata),
        )

        self._values = values
        self._valid = valid
        self._metadata = tuple(metadata)
        self._revision = delta.ref.revision
        self._generation = delta.ref.stream_generation
        return front

    def live_dataset_outputs(
        self,
        frozen: DatasetPreviewSnapshot,
    ) -> dict[str, LiveDatasetOutput]:
        """Publish the provisional scalar grid under the MOT task's name."""

        if not isinstance(frozen, DatasetPreviewSnapshot):
            raise TypeError("MOT live output requires DatasetPreviewSnapshot")
        if frozen.snapshot.block.schema != self._output_schema:
            raise ValueError("MOT live output differs from its frozen grid schema")
        output = single_live_dataset_output(
            MOT_FIELD_LIVE_OUTPUT_NAMES[0],
            frozen,
        )
        return {output.name: output}


__all__ = [
    "MOT_FIELD_LIVE_OUTPUT_NAMES",
    "MotFieldLiveProjection",
]
