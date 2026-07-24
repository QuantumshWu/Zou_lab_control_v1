"""Incremental live MOT intensity grid over one exact camera scan."""

from __future__ import annotations

import threading
from typing import Callable

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
from zlc_neutral_atom.mot_field import (
    MotFieldRequest,
    MotRoiProjector,
    build_mot_intensity_projector,
    mot_intensity_schema,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCoverage,
    DatasetPreviewDelta,
    DatasetPreviewSnapshot,
)
from zlc_neutral_atom.runtime.pipeline import (
    ExactDatasetPreviewPort,
    ExactDatasetPreviewSpec,
)
from zlc_neutral_atom.scan import ScanOutputContract

from .exact_live_slot import ExactDatasetLiveSlot


def _source_to_output_points(
    source: DatasetSchema,
    output: DatasetSchema,
) -> tuple[int, ...]:
    """Map the selected singleton readout axis without materializing frames."""

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
    if not omitted or any(source.point_axes[position].size != 1 for position in omitted):
        raise ValueError(
            "MOT live projection requires only singleton source point selections"
        )
    mapping: list[int] = []
    for storage_index in range(source.point_layout.storage_size):
        source_multi = source.point_layout.multi_index(storage_index)
        if any(source_multi[position] != 0 for position in omitted):
            raise ValueError("MOT selected source point is not the singleton coordinate")
        output_multi = tuple(source_multi[position] for position in selected_positions)
        mapping.append(output.point_layout.storage_index(output_multi))
    result = tuple(mapping)
    if len(result) != output.point_layout.storage_size or set(result) != set(
        range(output.point_layout.storage_size)
    ):
        raise ValueError("MOT source-to-output point mapping is not lossless")
    return result


class MotFieldGridLiveSlot:
    """Project exact cell deltas while retaining only one small scalar grid."""

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
        self._source = ExactDatasetLiveSlot(
            ExactDatasetPreviewSpec(source_schema.fingerprint)
        )
        self._projector: MotRoiProjector = build_mot_intensity_projector(
            request,
            transformed,
        )
        self._condition = threading.Condition(threading.Lock())
        self._listener: Callable[[], None] | None = None
        self._pending_change = False
        self._source_changed_flag = False
        self._run_id: str | None = None
        self._causation_domain_id: str | None = None
        self._revision_ref: DatasetRevisionRef | None = None
        self._coverage: DatasetCoverage | None = None
        self._failure: str | None = None
        self._notification_failure: str | None = None
        self._terminal = False
        self._closed = False
        self._values = np.zeros(
            self._output_schema.physical_shape,
            dtype=np.float64,
        )
        self._valid = np.zeros(self._output_schema.physical_shape[:2], dtype=bool)
        self._metadata: list[object | None] = [None] * self._valid.size
        self._source.set_change_listener(self._source_changed)
        self._worker = threading.Thread(
            target=self._watch,
            name="zlc-mot-field-live-grid",
            daemon=True,
        )
        self._worker.start()

    @property
    def preview_port(self) -> ExactDatasetPreviewPort:
        return self._source

    @property
    def terminal(self) -> bool:
        with self._condition:
            return self._terminal

    @property
    def notification_failure(self) -> str | None:
        with self._condition:
            return self._notification_failure

    def set_change_listener(self, listener: Callable[[], None]) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        replay = False
        with self._condition:
            if self._listener is not None:
                raise RuntimeError("MOT grid slot already has a listener")
            if self._closed:
                raise RuntimeError("MOT grid slot is closed")
            self._listener = listener
            replay, self._pending_change = self._pending_change, False
        if replay:
            self._call_listener(listener)

    def freeze_current(self) -> tuple[str, str, DatasetPreviewSnapshot]:
        with self._condition:
            if self._closed:
                raise RuntimeError("MOT grid slot is closed")
            if self._failure is not None:
                raise RuntimeError(self._failure)
            if (
                self._run_id is None
                or self._causation_domain_id is None
                or self._revision_ref is None
                or self._coverage is None
            ):
                raise RuntimeError("MOT grid has no projected revision yet")
            ref = self._revision_ref
            block = DataBlock(
                ref.block_id,
                ref.revision,
                self._values,
                CellValidity(self._valid),
                self._output_schema,
            )
            return (
                self._run_id,
                self._causation_domain_id,
                DatasetPreviewSnapshot(
                    OwnedSnapshot(ref, block),
                    self._coverage,
                    tuple(self._metadata),
                ),
            )

    def close(self) -> None:
        """Withdraw immediately; no GUI caller joins the projection worker."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._terminal = True
            self._listener = None
            self._condition.notify_all()
        self._source.close()

    def _source_changed(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._source_changed_flag = True
            self._condition.notify_all()

    def _watch(self) -> None:
        after = DatasetRevision(0)
        try:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._source_changed_flag
                )
                if self._closed:
                    return
            while True:
                with self._condition:
                    if self._closed:
                        return
                frozen = self._source.wait_and_freeze_delta(after, timeout=None)
                if frozen is not None:
                    run_id, causation, delta = frozen
                    if delta.cells:
                        after = delta.ref.revision
                        self._consume_delta(run_id, causation, delta)
                    continue
                failure = self._source.failure
                if failure is not None:
                    self._finish(failure=failure)
                    return
                if self._source.terminal:
                    self._finish()
                    return
        except BaseException as error:
            message = f"{type(error).__name__}: {error}"
            try:
                self._source.fail(message)
            except BaseException:
                pass
            self._finish(failure=message)

    def _consume_delta(
        self,
        run_id: str,
        causation_domain_id: str,
        delta: DatasetPreviewDelta,
    ) -> None:
        if delta.ref.schema_fingerprint != self._source_schema.fingerprint:
            raise ValueError("MOT delta belongs to another source schema")
        if delta.coverage.total_cells != self._valid.size:
            raise ValueError("MOT delta total coverage differs from output schema")
        projected: list[tuple[tuple[int, int], float, object | None]] = []
        for cell in delta.cells:
            address = cell.address
            if address.repeat_index >= self._valid.shape[0]:
                raise ValueError("MOT delta repeat address exceeds output schema")
            output_point = self._source_to_output_points[
                address.point_storage_index
            ]
            output_cell = (address.repeat_index, output_point)
            projected.append(
                (
                    output_cell,
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
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            if self._run_id is not None and self._run_id != run_id:
                raise RuntimeError("MOT live grid changed run identity")
            if (
                self._causation_domain_id is not None
                and self._causation_domain_id != causation_domain_id
            ):
                raise RuntimeError("MOT live grid changed causation identity")
            output_cells = tuple(item[0] for item in projected)
            if len(set(output_cells)) != len(output_cells) or any(
                self._valid[output_cell] for output_cell in output_cells
            ):
                raise RuntimeError(
                    "MOT delta attempted to rewrite a committed grid cell"
                )
            for output_cell, intensity, metadata in projected:
                self._values[(*output_cell, 0)] = intensity
                self._valid[output_cell] = True
                flat = output_cell[0] * self._valid.shape[1] + output_cell[1]
                self._metadata[flat] = metadata
            written = int(np.count_nonzero(self._valid))
            if written != delta.coverage.written_cells:
                raise RuntimeError(
                    "MOT delta coverage differs from accumulated grid cells"
                )
            self._run_id = run_id
            self._causation_domain_id = causation_domain_id
            self._revision_ref = DatasetRevisionRef(
                self._output_block_id,
                delta.ref.stream_generation,
                self._output_schema.fingerprint,
                delta.ref.revision,
            )
            self._coverage = DatasetCoverage(written, self._valid.size)
            listener = self._notify_locked()
        self._call_listener(listener)

    def _finish(self, *, failure: str | None = None) -> None:
        listener = None
        with self._condition:
            if self._closed or self._terminal:
                return
            self._failure = failure
            self._terminal = True
            listener = self._notify_locked()
            self._condition.notify_all()
        self._call_listener(listener)

    def _notify_locked(self) -> Callable[[], None] | None:
        if self._listener is None:
            self._pending_change = True
        return self._listener

    def _call_listener(self, listener: Callable[[], None] | None) -> None:
        if listener is None:
            return
        try:
            listener()
        except BaseException as error:
            with self._condition:
                if self._notification_failure is None:
                    self._notification_failure = f"{type(error).__name__}: {error}"


__all__ = [
    "MotFieldGridLiveSlot",
]
