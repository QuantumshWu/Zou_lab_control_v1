"""Single-owner event-to-dataset materialization with revisioned snapshots."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from enum import Enum
from numbers import Integral

import numpy as np

from zlc_data import (
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DataPatch,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    INVALID,
    Invalid,
    OwnedSnapshot,
    StreamGenerationId,
    VALID,
    Valid,
    ValidityMode,
    Value,
)

from .streams import (
    AcquisitionStream,
    Delivery,
    EndOfStream,
    Envelope,
    ExactReservation,
    MonitorTap,
    MonitorUpdate,
    ReservationState,
    StreamId,
    TraceBinding,
)


class DatasetMode(str, Enum):
    FINITE_EXACT = "FINITE_EXACT"
    ROLLING_MONITOR = "ROLLING_MONITOR"


class DatasetError(RuntimeError):
    pass


class DuplicateDatasetCell(DatasetError):
    pass


class MissingDatasetCells(DatasetError):
    pass


class SnapshotExpired(DatasetError):
    pass


_SEALED_TOKEN = object()


def dataset_cell_key_fingerprint(schema: DatasetSchema) -> str:
    """Bind repeat/point storage keys to one frozen DatasetSchema."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    contract = f"zlc.dataset-cell-address.v1:{schema.fingerprint}".encode("ascii")
    return hashlib.sha256(contract).hexdigest()


@dataclass(frozen=True, order=True)
class DatasetCellAddress:
    repeat_index: int
    point_storage_index: int

    def __post_init__(self) -> None:
        for field in ("repeat_index", "point_storage_index"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, int(value))


@dataclass(frozen=True)
class DatasetCellKeyContract:
    """Immutable join-key owner bound to one DatasetSchema storage domain."""

    schema: DatasetSchema

    @property
    def fingerprint(self) -> str:
        return dataset_cell_key_fingerprint(self.schema)

    def snapshot(self, key: object) -> DatasetCellAddress:
        self.validate(key)
        return key

    def validate(self, key: object) -> None:
        if not isinstance(key, DatasetCellAddress):
            raise TypeError("join key must be DatasetCellAddress")
        if key.repeat_index >= self.schema.repeat_axis.size:
            raise IndexError("join key repeat index is outside DatasetSchema")
        if key.point_storage_index >= self.schema.point_layout.storage_size:
            raise IndexError("join key point index is outside PointLayout")


@dataclass(frozen=True)
class DatasetCoverage:
    written_cells: int
    total_cells: int
    missed_events: int

    def __post_init__(self) -> None:
        for field in ("written_cells", "total_cells", "missed_events"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, int(value))
        if self.written_cells > self.total_cells:
            raise ValueError("written_cells cannot exceed total_cells")

    @property
    def complete(self) -> bool:
        return self.written_cells == self.total_cells and self.missed_events == 0


@dataclass(frozen=True)
class DatasetProgress:
    ref: DatasetRevisionRef
    dirty_cells: tuple[DatasetCellAddress, ...]
    coverage: DatasetCoverage

    def __post_init__(self) -> None:
        if not isinstance(self.ref, DatasetRevisionRef):
            raise TypeError("ref must be DatasetRevisionRef")
        cells = tuple(self.dirty_cells)
        if any(not isinstance(cell, DatasetCellAddress) for cell in cells):
            raise TypeError("dirty_cells must contain DatasetCellAddress values")
        if not isinstance(self.coverage, DatasetCoverage):
            raise TypeError("coverage must be DatasetCoverage")
        object.__setattr__(self, "dirty_cells", cells)


@dataclass(frozen=True)
class DatasetPreviewSnapshot:
    """Provisional materialization for display; never a formal storage input."""

    snapshot: OwnedSnapshot
    coverage: DatasetCoverage
    mode: DatasetMode

    @property
    def ref(self) -> DatasetRevisionRef:
        return self.snapshot.ref

    @property
    def block(self) -> DataBlock:
        return self.snapshot.block


@dataclass(frozen=True)
class DatasetSealProvenance:
    stream_id: StreamId
    generation: StreamGenerationId
    start_sequence: int
    end_sequence: int
    join_plan_digest: str
    ordered_event_digest: str
    trace_binding: TraceBinding


class SealedDatasetArtifact:
    """Opaque formal dataset capability minted only after exact terminal validation."""

    __slots__ = (
        "_snapshot",
        "_coverage",
        "_provenance",
    )

    def __init__(
        self,
        authority: object,
        *,
        snapshot: OwnedSnapshot,
        coverage: DatasetCoverage,
        stream_id,
        generation,
        start_sequence: int,
        end_sequence: int,
        join_plan_digest: str,
        ordered_event_digest: str,
        trace_binding: TraceBinding,
    ) -> None:
        if authority is not _SEALED_TOKEN:
            raise PermissionError("SealedDatasetArtifact can only be minted by DatasetBuilder")
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_coverage", coverage)
        object.__setattr__(
            self,
            "_provenance",
            DatasetSealProvenance(
                stream_id=stream_id,
                generation=generation,
                start_sequence=start_sequence,
                end_sequence=end_sequence,
                join_plan_digest=join_plan_digest,
                ordered_event_digest=ordered_event_digest,
                trace_binding=trace_binding,
            ),
        )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("SealedDatasetArtifact is immutable")

    @property
    def snapshot(self) -> OwnedSnapshot:
        return self._snapshot

    @property
    def ref(self) -> DatasetRevisionRef:
        return self._snapshot.ref

    @property
    def block(self) -> DataBlock:
        return self._snapshot.block

    @property
    def coverage(self) -> DatasetCoverage:
        return self._coverage

    @property
    def provenance(self) -> DatasetSealProvenance:
        return self._provenance


class DatasetBuilder:
    """Private mutable materializer; public reads are immutable owned snapshots."""

    def __init__(
        self,
        block_id: BlockId,
        source: ExactReservation[Value] | MonitorTap[Value],
        schema: DatasetSchema,
        mode: DatasetMode,
        *,
        expected_cells: tuple[DatasetCellAddress, ...] | None = None,
    ) -> None:
        if not isinstance(block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if not isinstance(mode, DatasetMode):
            raise TypeError("mode must be DatasetMode")
        if mode is DatasetMode.FINITE_EXACT and not isinstance(source, ExactReservation):
            raise TypeError("FINITE_EXACT DatasetBuilder must bind an ExactReservation")
        if mode is DatasetMode.ROLLING_MONITOR and not isinstance(source, MonitorTap):
            raise TypeError("ROLLING_MONITOR DatasetBuilder must bind a MonitorTap")
        self.block_id = block_id
        self._reservation = source if isinstance(source, ExactReservation) else None
        self._monitor = source if isinstance(source, MonitorTap) else None
        self._source: AcquisitionStream[Value] = source._stream
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        self.schema = schema
        self.mode = mode
        expected_key_fingerprint = dataset_cell_key_fingerprint(schema)
        if not isinstance(self._source._join_key_contract, DatasetCellKeyContract):
            raise DatasetError("dataset source must declare DatasetCellKeyContract")
        if self._source._join_key_contract.fingerprint != expected_key_fingerprint:
            raise DatasetError("dataset source join-key contract differs from DatasetSchema")
        total_cells = schema.repeat_axis.size * schema.point_layout.storage_size
        if self._reservation is not None:
            reserved_events = self._reservation.end_sequence - self._reservation.start_sequence
            if reserved_events != total_cells:
                raise DatasetError("exact reservation length must equal DatasetSchema cell count")
            if not isinstance(expected_cells, tuple) or len(expected_cells) != total_cells:
                raise DatasetError("exact materialization requires one frozen cell key per event")
            if any(not isinstance(cell, DatasetCellAddress) for cell in expected_cells):
                raise TypeError("expected_cells must contain DatasetCellAddress values")
            expected_domain = {
                DatasetCellAddress(repeat, point)
                for repeat in range(schema.repeat_axis.size)
                for point in range(schema.point_layout.storage_size)
            }
            if set(expected_cells) != expected_domain:
                raise DatasetError("expected_cells must cover every dataset cell exactly once")
        elif expected_cells is not None:
            raise DatasetError("rolling monitor materialization has no formal cell schedule")
        self._expected_cells = expected_cells
        join_hasher = hashlib.sha256()
        join_hasher.update(f"{schema.fingerprint}:".encode("ascii"))
        for cell in expected_cells or ():
            join_hasher.update(f"{cell.repeat_index},{cell.point_storage_index};".encode("ascii"))
        self._join_plan_digest = join_hasher.hexdigest()
        self._ordered_event_hasher = hashlib.sha256()
        self._lock = threading.RLock()
        self._values = np.zeros(schema.physical_shape, dtype=schema.cell_schema.dtype)
        self._written = np.zeros(schema.physical_shape[:2], dtype=bool)
        self._written_count = 0
        self._validity = self._new_validity_storage()
        self._revision = 0
        self._expected_sequence = (
            self._reservation.start_sequence if self._reservation is not None else 0
        )
        self._last_monitor_sequence: int | None = None
        self._missed_events = 0
        self._sealed = False
        self._aborted = False
        if self._reservation is not None:
            self._source._claim_materializer(self._reservation, self)

    @property
    def revision(self) -> DatasetRevision:
        with self._lock:
            return DatasetRevision(self._revision)

    @property
    def retained_patch_count(self) -> int:
        return 0

    def current_ref(self) -> DatasetRevisionRef:
        with self._lock:
            return self._ref_locked(self._revision)

    def consume(
        self,
        delivery: Delivery[Value],
    ) -> DatasetProgress:
        if self.mode is not DatasetMode.FINITE_EXACT:
            raise DatasetError("exact cursor consumption requires FINITE_EXACT mode")
        if not isinstance(delivery, Delivery) or not delivery.is_exact:
            raise TypeError("FINITE_EXACT DatasetBuilder requires an exact Delivery capability")
        if delivery.acknowledged:
            raise DatasetError("delivery was already acknowledged")
        if self._reservation is None:
            raise DatasetError("exact DatasetBuilder has no bound reservation")
        return self._source._consume_exact(
            self._reservation,
            delivery,
            self,
            lambda envelope: self._ingest(envelope, additional_missed=0),
        )

    def ingest_monitor(
        self,
        update: MonitorUpdate[Value],
    ) -> DatasetProgress:
        if self.mode is not DatasetMode.ROLLING_MONITOR:
            raise DatasetError("monitor updates require ROLLING_MONITOR mode")
        if not isinstance(update, MonitorUpdate):
            raise TypeError("update must be MonitorUpdate")
        if self._monitor is None or not self._monitor._owns_update(update):
            raise PermissionError("MonitorUpdate belongs to another monitor authority")
        return self._ingest(update.envelope, additional_missed=update.missed)

    def _ingest(
        self,
        envelope: Envelope[Value],
        *,
        additional_missed: int,
    ) -> DatasetProgress:
        if not isinstance(envelope, Envelope):
            raise TypeError("envelope must be Envelope")
        if not isinstance(envelope.payload, Value):
            raise TypeError("DatasetBuilder currently materializes Value payloads")
        address = envelope.join_key
        if not isinstance(address, DatasetCellAddress):
            raise DatasetError("dataset event is missing its typed DatasetCellAddress")
        with self._lock:
            self._ensure_writable_locked()
            self._validate_envelope_locked(envelope)
            self._validate_address_locked(address)
            cell = (address.repeat_index, address.point_storage_index)
            was_written = bool(self._written[cell])
            if self.mode is DatasetMode.FINITE_EXACT and was_written:
                raise DuplicateDatasetCell(f"dataset cell {cell} was already written")
            if self.mode is DatasetMode.FINITE_EXACT:
                if envelope.sequence != self._expected_sequence:
                    raise DatasetError(
                        f"exact dataset expected sequence {self._expected_sequence}, "
                        f"got {envelope.sequence}"
                    )
                assert self._expected_cells is not None
                schedule_index = envelope.sequence - self._reservation.start_sequence
                expected_address = self._expected_cells[schedule_index]
                if address != expected_address:
                    raise DatasetError(
                        f"event join key {address} differs from frozen plan key "
                        f"{expected_address} at sequence {envelope.sequence}"
                    )
            elif self._last_monitor_sequence is not None and envelope.sequence <= self._last_monitor_sequence:
                raise DatasetError("monitor dataset events must remain strictly ordered")

            base = DatasetRevision(self._revision)
            result = DatasetRevision(self._revision + 1)
            patch = DataPatch(
                block_id=self.block_id,
                base_revision=base,
                result_revision=result,
                target_cells=(cell,),
                values=envelope.payload.values.reshape((1, *self.schema.cell_schema.data_shape)),
                validity_patch=(envelope.payload.validity,),
                schema_fingerprint=self.schema.fingerprint,
            )
            self._apply_patch_locked(
                patch,
                self._values,
                self._written,
                self._validity,
            )
            self._revision += 1
            if not was_written:
                self._written_count += 1
            self._missed_events += additional_missed
            if self.mode is DatasetMode.FINITE_EXACT:
                self._expected_sequence += 1
                self._ordered_event_hasher.update(
                    f"{envelope.sequence}:{envelope.event_id.value};".encode("utf-8")
                )
            else:
                self._last_monitor_sequence = envelope.sequence
            return DatasetProgress(
                ref=self._ref_locked(self._revision),
                dirty_cells=(address,),
                coverage=self._coverage_locked(),
            )

    def materialize(self, ref: DatasetRevisionRef | None = None) -> DatasetPreviewSnapshot:
        with self._lock:
            selected = self._ref_locked(self._revision) if ref is None else ref
            self._validate_ref_locked(selected)
            target_revision = selected.revision.value
            if target_revision > self._revision:
                raise KeyError(f"dataset revision {target_revision} has not been committed")
            if target_revision != self._revision:
                raise SnapshotExpired(
                    "DatasetBuilder retains only the current revision; "
                    "callers retain OwnedSnapshot values, not mutable history"
                )
            block = DataBlock(
                block_id=self.block_id,
                revision=DatasetRevision(target_revision),
                values=self._values,
                validity=self._materialized_validity(self._validity),
                schema=self.schema,
            )
            return DatasetPreviewSnapshot(
                snapshot=OwnedSnapshot(selected, block),
                coverage=self._coverage_locked(),
                mode=self.mode,
            )

    def seal(self, eos: EndOfStream) -> SealedDatasetArtifact:
        if self.mode is not DatasetMode.FINITE_EXACT or self._reservation is None:
            raise DatasetError("rolling monitor datasets cannot become formal sealed datasets")
        self._source._seal_exact(self._reservation, eos, self, self._seal_locked)
        preview = self.materialize()
        return SealedDatasetArtifact(
            _SEALED_TOKEN,
            snapshot=preview.snapshot,
            coverage=preview.coverage,
            stream_id=self.stream_id,
            generation=self.generation,
            start_sequence=self._reservation.start_sequence,
            end_sequence=self._reservation.end_sequence,
            join_plan_digest=self._join_plan_digest,
            ordered_event_digest=self._ordered_event_hasher.copy().hexdigest(),
            trace_binding=self._reservation.trace_binding,
        )

    def _seal_locked(self) -> None:
        with self._lock:
            self._ensure_writable_locked()
            missing = np.argwhere(~self._written)
            if missing.size:
                cells = tuple(tuple(int(index) for index in row) for row in missing[:8])
                raise MissingDatasetCells(
                    f"dataset is missing {len(missing)} cells; first missing cells: {cells}"
                )
            if not self._coverage_locked().complete:
                raise DatasetError("formal dataset coverage is incomplete")
            self._sealed = True

    def abort(self) -> None:
        if self._reservation is not None:
            self._source._abort_materializer(
                self._reservation,
                self,
                self._mark_aborted_locked,
            )
            return
        self._mark_aborted_locked()

    def __enter__(self) -> "DatasetBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._reservation is None:
            return False
        cleanup_error: BaseException | None = None
        try:
            if not self._sealed and not self._aborted:
                self.abort()
            if self._reservation.state in (
                ReservationState.COMPLETED,
                ReservationState.FAILED,
                ReservationState.CANCELLED,
            ):
                self._reservation.release()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            if exc is None:
                raise cleanup_error
            if hasattr(exc, "add_note"):
                exc.add_note(f"DatasetBuilder teardown also failed: {cleanup_error!r}")
        return False

    def _mark_aborted_locked(self) -> None:
        with self._lock:
            if self._sealed:
                raise DatasetError("sealed dataset cannot be aborted")
            self._aborted = True

    def _new_validity_storage(self) -> np.ndarray:
        contract = self.schema.cell_schema.validity_contract
        if contract.mode is ValidityMode.VALUE:
            return np.zeros(self.schema.physical_shape[:2], dtype=bool)
        axes = tuple(self.schema.cell_schema.axis(axis_id) for axis_id in contract.component_axis_ids)
        return np.zeros(
            (*self.schema.physical_shape[:2], *(axis.size for axis in axes)),
            dtype=bool,
        )

    def _ensure_writable_locked(self) -> None:
        if self._sealed:
            raise DatasetError("dataset is sealed")
        if self._aborted:
            raise DatasetError("dataset is aborted")

    def _validate_envelope_locked(self, envelope: Envelope[Value]) -> None:
        if envelope.stream_generation != self.generation:
            raise DatasetError("envelope stream generation differs from DatasetBuilder")
        if envelope.stream_id != self.stream_id:
            raise DatasetError("envelope stream id differs from DatasetBuilder")
        if envelope.payload.schema.fingerprint != self.schema.cell_schema.fingerprint:
            raise DatasetError("ValueSchema fingerprint differs from DatasetSchema cell contract")

    def _validate_address_locked(self, address: DatasetCellAddress) -> None:
        if address.repeat_index >= self.schema.repeat_axis.size:
            raise IndexError("repeat index is outside DatasetSchema")
        if address.point_storage_index >= self.schema.point_layout.storage_size:
            raise IndexError("point storage index is outside PointLayout")

    def _value_validity_mask(self, validity) -> np.ndarray | bool:
        contract = self.schema.cell_schema.validity_contract
        if contract.mode is ValidityMode.VALUE:
            if isinstance(validity, Valid):
                return True
            if isinstance(validity, Invalid):
                return False
            raise ValueError("component validity cannot enter a VALUE dataset contract")
        declared = contract.component_axis_ids
        declared_shape = tuple(self.schema.cell_schema.axis(axis_id).size for axis_id in declared)
        if isinstance(validity, Valid):
            return np.ones(declared_shape, dtype=bool)
        if isinstance(validity, Invalid):
            return np.zeros(declared_shape, dtype=bool)
        if not isinstance(validity, ComponentValidity):
            raise TypeError("unsupported Value validity")
        positions = tuple(declared.index(axis_id) for axis_id in validity.axis_ids)
        shape = [1] * len(declared)
        for mask_axis, declared_axis in enumerate(positions):
            shape[declared_axis] = validity.mask.shape[mask_axis]
        return np.broadcast_to(validity.mask.reshape(tuple(shape)), declared_shape)

    def _apply_patch_locked(
        self,
        patch: DataPatch,
        values: np.ndarray,
        written: np.ndarray,
        validity: np.ndarray,
    ) -> None:
        if patch.block_id != self.block_id or patch.schema_fingerprint != self.schema.fingerprint:
            raise DatasetError("DataPatch targets another dataset contract")
        prepared = tuple(
            (cell, patch.values[index], self._value_validity_mask(patch.validity_patch[index]))
            for index, cell in enumerate(patch.target_cells)
        )
        for cell, cell_values, validity_mask in prepared:
            values[cell] = cell_values
            written[cell] = True
            validity[cell] = validity_mask

    def _materialized_validity(self, validity: np.ndarray):
        contract = self.schema.cell_schema.validity_contract
        if contract.mode is ValidityMode.VALUE:
            return CellValidity(validity)
        return ComponentValidity(contract.component_axis_ids, validity)

    def _coverage_locked(self) -> DatasetCoverage:
        return DatasetCoverage(
            written_cells=self._written_count,
            total_cells=int(self._written.size),
            missed_events=self._missed_events,
        )

    def _ref_locked(self, revision: int) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            block_id=self.block_id,
            stream_generation=self.generation,
            schema_fingerprint=self.schema.fingerprint,
            revision=DatasetRevision(revision),
        )

    def _validate_ref_locked(self, ref: DatasetRevisionRef) -> None:
        if not isinstance(ref, DatasetRevisionRef):
            raise TypeError("ref must be DatasetRevisionRef")
        if ref.block_id != self.block_id:
            raise ValueError("snapshot ref belongs to another block")
        if ref.stream_generation != self.generation:
            raise ValueError("snapshot ref belongs to another stream generation")
        if ref.schema_fingerprint != self.schema.fingerprint:
            raise ValueError("snapshot ref schema fingerprint differs")


__all__ = [
    "DatasetBuilder",
    "DatasetCellAddress",
    "DatasetCellKeyContract",
    "DatasetCoverage",
    "DatasetError",
    "DatasetMode",
    "DatasetProgress",
    "DatasetPreviewSnapshot",
    "DatasetSealProvenance",
    "DuplicateDatasetCell",
    "MissingDatasetCells",
    "SnapshotExpired",
    "SealedDatasetArtifact",
    "dataset_cell_key_fingerprint",
]
