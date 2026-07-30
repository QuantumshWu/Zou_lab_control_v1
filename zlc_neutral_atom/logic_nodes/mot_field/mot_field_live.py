"""Progressive MOT-field data projection over one exact camera scan.

This module owns the neutral-atom meaning of the live MOT grid: selecting the
scan output geometry, reducing each camera frame with the frozen MOT ROI,
preserving value validity, and materializing an exact ``R x P x (1)`` Dataset
front.  The generic runtime exact-delta port owns reader/notification lifetime;
no desktop package owns either concern.
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
)
from zlc_data.value import expand_value_validity
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    single_live_dataset_output,
)

from .mot_field import (
    MotFieldRequest,
    MotRoiProjector,
    build_mot_intensity_projector,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
)


MOT_FIELD_LIVE_OUTPUT_DECLARATIONS = (
    DatasetOutputDeclaration("grid", "zlc_neutral_atom.mot-field.live-grid"),
)


class MotFieldLiveProjection:
    """Atomically accumulate exact camera deltas into one scalar MOT grid.

    The instance is single-consumer by design.  Ingestion validates a complete
    delta before mutating its fixed scalar buffers, so per-cell bookkeeping is
    O(1).  Immutable Dataset materialization occurs only when SignalPlane asks
    for a front; a rejected delta leaves the prior front unchanged.
    """

    def __init__(
        self,
        request: MotFieldRequest,
        source_schema: DatasetSchema,
        output_schema: DatasetSchema,
    ) -> None:
        if not isinstance(request, MotFieldRequest):
            raise TypeError("request must be MotFieldRequest")
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output_schema, DatasetSchema):
            raise TypeError("output_schema must be DatasetSchema")
        self._source_schema = source_schema
        self._output_schema = output_schema
        self._output_block_id = BlockId("mot-field-live-grid")
        self._projector: MotRoiProjector = build_mot_intensity_projector(
            request,
            source_schema,
        )
        self._values = np.zeros(self._output_schema.physical_shape, dtype=np.float64)
        self._valid = np.zeros(self._output_schema.physical_shape[:2], dtype=bool)
        self._metadata: list[object | None] = [None] * self._valid.size
        self._written_count = 0
        self._revision = DatasetRevision(0)
        self._generation = None

    def consume(
        self,
        delta: DatasetPreviewDelta,
    ) -> bool:
        """Validate then apply one contiguous exact delta in place."""

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
            output_point = address.point_ordinal
            if output_point >= self._output_schema.point_table.row_count:
                raise ValueError(
                    "MOT delta point address exceeds the source schema"
                )
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
            raise RuntimeError("MOT delta attempted to rewrite a written grid cell")

        expected_written = self._written_count + len(projected)
        if expected_written != delta.coverage.written_cells:
            raise RuntimeError("MOT delta coverage differs from accumulated grid cells")
        for output_cell, intensity, cell_metadata in projected:
            self._values[(*output_cell, 0)] = intensity
            self._valid[output_cell] = True
            flat = output_cell[0] * self._valid.shape[1] + output_cell[1]
            self._metadata[flat] = cell_metadata

        self._written_count = expected_written
        self._revision = delta.ref.revision
        self._generation = delta.ref.stream_generation
        return True

    def freeze_live_outputs(self) -> dict[str, LiveDatasetOutput]:
        """Materialize the current scalar front only on consumer demand."""

        if self._generation is None or self._revision.value < 1:
            raise RuntimeError("MOT live projection has no written cell")

        ref = DatasetRevisionRef(
            self._output_block_id,
            self._generation,
            self._output_schema.fingerprint,
            self._revision,
        )
        block = DataBlock(
            ref.block_id,
            ref.revision,
            self._values,
            CellValidity(self._valid),
            self._output_schema,
        )
        front = DatasetPreviewSnapshot(
            OwnedSnapshot(ref, block),
            DatasetCoverage(self._written_count, self._valid.size),
            tuple(self._metadata),
        )
        output = single_live_dataset_output(
            MOT_FIELD_LIVE_OUTPUT_DECLARATIONS[0],
            front,
        )
        return {output.name: output}


__all__ = [
    "MOT_FIELD_LIVE_OUTPUT_DECLARATIONS",
    "MotFieldLiveProjection",
]
