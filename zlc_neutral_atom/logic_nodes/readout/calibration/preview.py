"""Calibration's typed live capture preview.

This module is deliberately a leaf-owned adapter: the capture pipeline owns the
exact reader and :mod:`zlc_plot` owns all rendering.  The adapter only projects
the latest committed camera cell into the declared ``site_map`` output so the
generic TaskPreview path can display it while the calibration capture is still
running.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from zlc_data import (
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    REPEAT,
    StreamGenerationId,
    Value,
    Valid,
    ValidityMode,
)
from zlc_data.value import expand_component_validity
from zlc_neutral_atom.dataset_output import (
    DatasetOutputDeclaration,
    LiveDatasetOutput,
    single_live_dataset_output,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
)


_PREVIEW_REPEAT_AXIS = AxisId("calibration.preview.repeat")
_PREVIEW_BLOCK = BlockId("calibration.preview")
_PREVIEW_GENERATION = StreamGenerationId("calibration.preview")


class CalibrationCapturePreview:
    """Project one complete source cell at a time as a live ``site_map``.

    The source schedule may contain repeats, readout events, and scan/context
    rows.  Those are capture provenance, not image data dimensions; this
    preview intentionally exposes one canonical ``R=1, P=1`` image cell and
    never adds a buffer/history axis to the public signal.
    """

    __slots__ = (
        "_source_schema",
        "_output_schema",
        "_output",
        "_after",
        "_source_identity",
        "_values",
        "_validity",
    )

    def __init__(
        self,
        source_schema: DatasetSchema,
        output: DatasetOutputDeclaration,
    ) -> None:
        if not isinstance(source_schema, DatasetSchema):
            raise TypeError("source_schema must be DatasetSchema")
        if not isinstance(output, DatasetOutputDeclaration):
            raise TypeError("output must be DatasetOutputDeclaration")
        self._source_schema = source_schema
        self._output = output
        self._output_schema = DatasetSchema(
            AxisSpec(_PREVIEW_REPEAT_AXIS, "repeat", REPEAT, 1, (0,)),
            PointTable(1),
            None,
            source_schema.cell_schema,
        )
        self._after = DatasetRevision(0)
        self._source_identity: tuple[BlockId, StreamGenerationId] | None = None
        self._values: np.ndarray | None = None
        self._validity: object | None = None

    def consume(self, delta: DatasetPreviewDelta) -> bool:
        if not isinstance(delta, DatasetPreviewDelta):
            raise TypeError("delta must be DatasetPreviewDelta")
        if delta.after != self._after:
            raise RuntimeError("calibration preview delta is not contiguous")
        if delta.ref.schema_fingerprint != self._source_schema.fingerprint:
            raise ValueError("calibration preview changed source schema")
        identity = (delta.ref.block_id, delta.ref.stream_generation)
        if self._source_identity is not None and identity != self._source_identity:
            raise RuntimeError("calibration preview changed source identity")
        for cell in delta.cells:
            value = cell.value
            if not isinstance(value, Value) or value.schema is not self._source_schema.cell_schema:
                raise ValueError("calibration preview cell schema changed")
            self._values = value.values
            contract = value.schema.validity_contract
            if contract.mode is ValidityMode.VALUE:
                self._validity = CellValidity(
                    np.asarray(isinstance(value.validity, Valid), dtype=bool).reshape((1, 1))
                )
            else:
                component_shape = tuple(
                    value.schema.axis(axis_id).size
                    for axis_id in contract.component_axis_ids
                )
                self._validity = DatasetComponentValidity(
                    contract.component_axis_ids,
                    np.asarray(
                        expand_component_validity(value.validity, value.schema),
                        dtype=bool,
                    ).reshape((1, 1, *component_shape)),
                )
        self._after = delta.ref.revision
        self._source_identity = identity
        return bool(delta.cells)

    def freeze_live_outputs(self) -> Mapping[str, LiveDatasetOutput]:
        if self._values is None or self._validity is None:
            raise RuntimeError("calibration preview has no committed frame")
        identity = self._source_identity
        if identity is None:
            raise RuntimeError("calibration preview has no source identity")
        block = DataBlock(
            _PREVIEW_BLOCK,
            self._after,
            np.asarray(self._values).reshape(self._output_schema.physical_shape),
            self._validity,
            self._output_schema,
        )
        snapshot = OwnedSnapshot(
            DatasetRevisionRef(
                _PREVIEW_BLOCK,
                _PREVIEW_GENERATION,
                self._output_schema.fingerprint,
                self._after,
            ),
            block,
        )
        frozen = DatasetPreviewSnapshot(snapshot, DatasetCoverage(1, 1), (None,))
        return {self._output.name: single_live_dataset_output(self._output, frozen)}


__all__ = ["CalibrationCapturePreview"]
