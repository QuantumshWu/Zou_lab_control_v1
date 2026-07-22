"""Chunked binary frame storage for committed raw camera captures."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from zlc_data import (
    INVALID,
    VALID,
    AxisId,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    StreamGenerationId,
    Valid,
    Value,
    canonical_value_array,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
)
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalListEvent,
    ContentRef,
    ContentStoreAuthority,
    RepositoryRootLease,
    content_ref_from_tree,
    content_ref_to_tree,
    decode,
    encode,
    exact_mapping,
    nonnegative_integer,
    positive_integer,
    sha256_text,
)

from zlc_neutral_atom.acquisition import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    camera_frame_metadata_from_tree,
    camera_frame_metadata_to_tree,
)
from zlc_neutral_atom.runtime.dataset import (
    DatasetCellAddress,
    DatasetCellSchedule,
    OrderedDatasetMetadataHasher,
    dataset_cell_permutation_digest,
)


_FRAME_CHUNK_EVENTS = 64
_FRAME_INDEX_SCHEMA = "zlc_neutral_atom.CaptureFrameIndex"
_FRAME_SCHEMA_SCHEMA = "zlc_neutral_atom.CaptureFrameDatasetSchema"
_FRAME_EVENT_CHUNK_SCHEMA = "zlc_neutral_atom.CaptureFrameEventChunk"
_FRAME_EVENT_CHUNK_MAX_EVENTS = 256
_VALIDITY_KINDS = frozenset({"valid", "invalid", "cell", "component"})


@dataclass(frozen=True, slots=True)
class _FrameRecordGeometry:
    """One canonical frame-record and chunk layout calculation."""

    frame_nbytes: int
    validity_nbytes: int
    record_nbytes: int
    frames_per_chunk: int
    expected_chunks: int


@dataclass(frozen=True, slots=True)
class _CaptureFrameSourceInspection:
    """Root/schema facts shared with the full lazy-source load."""

    dataset_schema: DatasetSchema
    block_id: BlockId
    revision: DatasetRevision
    event_count: int
    event_chunk_refs: tuple[ContentRef, ...]
    frame_chunk_refs: tuple[ContentRef, ...]
    validity_kind: str
    validity_axis_ids: tuple[AxisId, ...]
    geometry: _FrameRecordGeometry


def _capture_frame_record_geometry(
    schema: DatasetSchema,
    validity_kind: str,
    validity_axis_ids: tuple[AxisId, ...],
    cell_count: int,
) -> _FrameRecordGeometry:
    """Return the sole binary layout used by staging and reading."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(validity_kind, str) or validity_kind not in _VALIDITY_KINDS:
        raise ValueError("frame source validity kind is unknown")
    axis_ids = tuple(validity_axis_ids)
    if any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
        raise TypeError("validity_axis_ids must contain AxisId values")
    if validity_kind == "component":
        if not axis_ids:
            raise ValueError("component validity requires named axes")
        declared = schema.cell_schema.validity_contract.component_axis_ids
        if any(axis_id not in declared for axis_id in axis_ids):
            raise ValueError(
                "component validity axis is absent from the schema contract"
            )
        selected = set(axis_ids)
        ordered = tuple(
            axis.axis_id
            for axis in schema.cell_schema.data_axes
            if axis.axis_id in selected
        )
        if axis_ids != ordered:
            raise ValueError(
                "component validity axes must be unique and follow data-axis order"
            )
        validity_nbytes = math.prod(
            schema.cell_schema.axis(axis_id).size for axis_id in axis_ids
        )
    else:
        if axis_ids:
            raise ValueError("only component validity may name axes")
        validity_nbytes = 1 if validity_kind == "cell" else 0

    cell_count = nonnegative_integer(cell_count, "cell_count")
    frame_nbytes = (
        math.prod(schema.cell_schema.data_shape) * schema.cell_schema.dtype.itemsize
    )
    record_nbytes = frame_nbytes + validity_nbytes
    if record_nbytes <= 0:
        raise ValueError("capture frame record must contain data")
    frames_per_chunk = _FRAME_CHUNK_EVENTS
    expected_chunks = (cell_count + frames_per_chunk - 1) // frames_per_chunk
    return _FrameRecordGeometry(
        frame_nbytes=frame_nbytes,
        validity_nbytes=validity_nbytes,
        record_nbytes=record_nbytes,
        frames_per_chunk=frames_per_chunk,
        expected_chunks=expected_chunks,
    )


def _cell_address_to_tree(cell: DatasetCellAddress) -> list[int]:
    if not isinstance(cell, DatasetCellAddress):
        raise TypeError("cell must be DatasetCellAddress")
    return [cell.repeat_index, cell.point_storage_index]


def _cell_address_from_tree(tree: object) -> DatasetCellAddress:
    if not isinstance(tree, list) or len(tree) != 2:
        raise ValueError("frame-event cell address must contain two integers")
    return DatasetCellAddress(
        nonnegative_integer(tree[0], "repeat_index"),
        nonnegative_integer(tree[1], "point_storage_index"),
    )


def _inverse_ordinal_width(event_count: int) -> int:
    count = positive_integer(event_count, "event_count")
    if count <= 0xFFFFFFFF:
        return 4
    if count <= 0xFFFFFFFFFFFFFFFF:
        return 8
    raise OverflowError("capture event count exceeds compact-index representation")


def _write_inverse_ordinal(
    packed: bytearray,
    width: int,
    linear_cell: int,
    ordinal: int,
) -> None:
    format_code = "<I" if width == 4 else "<Q"
    struct.pack_into(format_code, packed, linear_cell * width, ordinal + 1)


def _read_inverse_ordinal(packed: bytes, width: int, linear_cell: int) -> int:
    format_code = "<I" if width == 4 else "<Q"
    encoded = struct.unpack_from(format_code, packed, linear_cell * width)[0]
    if encoded == 0:
        raise RuntimeError("capture inverse index omits a validated dataset cell")
    return encoded - 1


def _validate_frame_events(
    schema: DatasetSchema,
    schedule: DatasetCellSchedule,
    metadata: tuple[CameraFrameMetadata, ...],
) -> tuple[bytes, int, str, str]:
    """Validate staging events and derive the compact immutable load receipts."""

    total = schema.repeat_axis.size * schema.point_layout.storage_size
    if len(schedule) != total or len(metadata) != total:
        raise ValueError("frame source schedule/metadata do not cover the dataset")
    join_plan_digest = schedule.digest_for_schema(schema)
    inverse_width = _inverse_ordinal_width(total)
    ordinal_by_linear_cell = bytearray(total * inverse_width)
    metadata_contract = CameraFrameMetadataContract()
    metadata_hasher = OrderedDatasetMetadataHasher(metadata_contract.fingerprint)

    for ordinal, (cell, item) in enumerate(zip(schedule, metadata, strict=True)):
        if not isinstance(item, CameraFrameMetadata):
            raise TypeError("metadata must contain CameraFrameMetadata")
        if item.source_ordinal != ordinal:
            raise ValueError("frame source metadata ordinals are not contiguous")
        metadata_hasher.update(metadata_contract.digest(item))
        linear_cell = (
            cell.repeat_index * schema.point_layout.storage_size
            + cell.point_storage_index
        )
        _write_inverse_ordinal(
            ordinal_by_linear_cell,
            inverse_width,
            linear_cell,
            ordinal,
        )

    return (
        bytes(ordinal_by_linear_cell),
        inverse_width,
        join_plan_digest,
        metadata_hasher.digest(),
    )


def _validity_descriptor(validity: object) -> tuple[str, tuple[object, ...]]:
    if isinstance(validity, Valid):
        return "valid", ()
    if isinstance(validity, Invalid):
        return "invalid", ()
    if isinstance(validity, CellValidity):
        return "cell", ()
    if isinstance(validity, ComponentValidity):
        return "component", validity.axis_ids
    raise TypeError("capture DataBlock has an unsupported validity type")


def _canonical_frame_bytes(
    values: np.ndarray,
    validity: Valid | Invalid | ComponentValidity,
    schema,
) -> bytes:
    """Materialize the zlc_data-owned canonical value at the CAS boundary."""

    canonical = canonical_value_array(values, validity, schema)
    if canonical is None:
        return bytes(np.asarray(values).nbytes)
    return canonical.tobytes(order="C")


def _frame_validity(
    block: DataBlock,
    cell: DatasetCellAddress,
) -> tuple[Valid | Invalid | ComponentValidity, bytes]:
    validity = block.validity
    location = (cell.repeat_index, cell.point_storage_index)
    if isinstance(validity, (Valid, Invalid)):
        return validity, b""
    if isinstance(validity, CellValidity):
        admitted = bool(validity.mask[location])
        return (VALID if admitted else INVALID), bytes((int(admitted),))
    assert isinstance(validity, ComponentValidity)
    mask = np.asarray(validity.mask[location], dtype=bool)
    frame_validity = ComponentValidity(validity.axis_ids, mask)
    return frame_validity, np.ascontiguousarray(mask).tobytes(order="C")


_CAPTURE_FRAME_SOURCE_TOKEN = object()


def _decode_frame_event_chunk(
    *,
    reference: ContentRef,
    chunk_index: int,
    event_count: int,
    schema: DatasetSchema,
    store_authority: ContentStoreAuthority,
) -> tuple[tuple[DatasetCellAddress, CameraFrameMetadata], ...]:
    """Decode and structurally validate one immutable event chunk."""

    start = nonnegative_integer(chunk_index, "chunk_index") * _FRAME_EVENT_CHUNK_MAX_EVENTS
    if start >= event_count:
        raise IndexError("capture frame-event chunk is outside the event domain")
    expected_count = min(_FRAME_EVENT_CHUNK_MAX_EVENTS, event_count - start)

    def admit_event_chunk(events) -> None:
        for event in events:
            if isinstance(event, CanonicalArrayEvent):
                raise ValueError(
                    "capture frame-event chunk cannot embed ndarrays"
                )
            if (
                isinstance(event, CanonicalListEvent)
                and event.path == ("events",)
                and event.length > _FRAME_EVENT_CHUNK_MAX_EVENTS
            ):
                raise ValueError(
                    "capture frame-event chunk exceeds event-count policy"
                )

    event_payload = store_authority.read_blob(reference)
    event_tree = exact_mapping(
        decode(
            event_payload,
            admit_structure=admit_event_chunk,
        ),
        {"schema", "start_ordinal", "events"},
        _FRAME_EVENT_CHUNK_SCHEMA,
    )
    if nonnegative_integer(event_tree["start_ordinal"], "start_ordinal") != start:
        raise ValueError("frame-event chunks are not contiguous")
    events = event_tree["events"]
    if not isinstance(events, list) or len(events) != expected_count:
        raise ValueError("frame-event chunk cardinality differs")
    records: list[tuple[DatasetCellAddress, CameraFrameMetadata]] = []
    for offset, event in enumerate(events):
        row = exact_mapping(
            event,
            {"cell", "metadata"},
            "frame event",
            discriminator=None,
        )
        metadata = camera_frame_metadata_from_tree(row["metadata"])
        if metadata.source_ordinal != start + offset:
            raise ValueError("frame-event metadata ordinal differs from chunk order")
        cell = _cell_address_from_tree(row["cell"])
        if (
            cell.repeat_index >= schema.repeat_axis.size
            or cell.point_storage_index >= schema.point_layout.storage_size
        ):
            raise ValueError("frame-event cell is outside DatasetSchema")
        records.append((cell, metadata))
    return tuple(records)


class CaptureFrameSource:
    """Repository-bound lazy access to immutable frame and event chunks."""

    __slots__ = (
        "_schema",
        "_block_id",
        "_revision",
        "_event_count",
        "_event_chunk_refs",
        "_ordered_metadata_digest",
        "_join_plan_digest",
        "_validity_kind",
        "_validity_axis_ids",
        "_chunk_refs",
        "_store",
        "_root_lease",
        "_ordinal_by_linear_cell",
        "_inverse_ordinal_width",
        "_geometry",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureFrameSource is final")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureFrameSource is immutable")

    def __init__(
        self,
        authority: object,
        *,
        schema: DatasetSchema,
        block_id: BlockId,
        revision: DatasetRevision,
        event_count: int,
        event_chunk_refs: tuple[ContentRef, ...],
        ordered_metadata_digest: str,
        join_plan_digest: str,
        ordinal_by_linear_cell: bytes,
        inverse_ordinal_width: int,
        validity_kind: str,
        validity_axis_ids: tuple[AxisId, ...],
        chunk_refs: tuple[ContentRef, ...],
        store_authority: ContentStoreAuthority,
        root_lease: RepositoryRootLease,
    ) -> None:
        if authority is not _CAPTURE_FRAME_SOURCE_TOKEN:
            raise PermissionError(
                "CaptureFrameSource can only be minted by its persistence owner"
            )
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if not isinstance(block_id, BlockId) or not isinstance(revision, DatasetRevision):
            raise TypeError("frame source requires BlockId and DatasetRevision")
        if not isinstance(store_authority, ContentStoreAuthority):
            raise TypeError("store_authority must be ContentStoreAuthority")
        if not isinstance(root_lease, RepositoryRootLease):
            raise TypeError("root_lease must be RepositoryRootLease")
        root_lease.require_active()
        if store_authority.root != root_lease.root / "content":
            raise ValueError("frame source content store differs from repository lease")
        event_count = positive_integer(event_count, "event_count")
        physical_cells = schema.repeat_axis.size * schema.point_layout.storage_size
        if event_count != physical_cells:
            raise ValueError("frame source event count differs from DatasetSchema")
        event_refs = tuple(event_chunk_refs)
        if any(not isinstance(item, ContentRef) for item in event_refs):
            raise TypeError("event_chunk_refs must contain ContentRef")
        expected_event_chunks = (
            event_count + _FRAME_EVENT_CHUNK_MAX_EVENTS - 1
        ) // _FRAME_EVENT_CHUNK_MAX_EVENTS
        if len(event_refs) != expected_event_chunks:
            raise ValueError("frame-event chunk count differs from event cardinality")
        sha256_text(ordered_metadata_digest, "ordered_metadata_digest")
        sha256_text(join_plan_digest, "join_plan_digest")
        if type(ordinal_by_linear_cell) is not bytes:
            raise TypeError("ordinal_by_linear_cell must be immutable bytes")
        if inverse_ordinal_width not in (4, 8):
            raise ValueError("inverse_ordinal_width must be four or eight bytes")
        if inverse_ordinal_width != _inverse_ordinal_width(event_count):
            raise ValueError("inverse ordinal width is not canonical for event count")
        if len(ordinal_by_linear_cell) != event_count * inverse_ordinal_width:
            raise ValueError("compact inverse index differs from event cardinality")
        refs = tuple(chunk_refs)
        if any(not isinstance(item, ContentRef) for item in refs):
            raise TypeError("chunk_refs must contain ContentRef")
        axis_ids = tuple(validity_axis_ids)
        geometry = _capture_frame_record_geometry(
            schema,
            validity_kind,
            axis_ids,
            event_count,
        )
        if len(refs) != geometry.expected_chunks:
            raise ValueError("frame chunk count differs from canonical chunking")
        for index, reference in enumerate(refs):
            count = min(
                geometry.frames_per_chunk,
                event_count - index * geometry.frames_per_chunk,
            )
            if reference.size != count * geometry.record_nbytes:
                raise ValueError("frame chunk size differs from canonical record layout")
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_block_id", block_id)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_event_count", event_count)
        object.__setattr__(self, "_event_chunk_refs", event_refs)
        object.__setattr__(self, "_ordered_metadata_digest", ordered_metadata_digest)
        object.__setattr__(self, "_join_plan_digest", join_plan_digest)
        object.__setattr__(self, "_validity_kind", validity_kind)
        object.__setattr__(self, "_validity_axis_ids", axis_ids)
        object.__setattr__(self, "_chunk_refs", refs)
        object.__setattr__(self, "_store", store_authority)
        object.__setattr__(self, "_root_lease", root_lease)
        object.__setattr__(
            self,
            "_ordinal_by_linear_cell",
            ordinal_by_linear_cell,
        )
        object.__setattr__(self, "_inverse_ordinal_width", inverse_ordinal_width)
        object.__setattr__(self, "_geometry", geometry)

    @property
    def schema(self) -> DatasetSchema:
        return self._schema

    @property
    def block_id(self) -> BlockId:
        return self._block_id

    @property
    def revision(self) -> DatasetRevision:
        return self._revision

    def ref(self, generation: StreamGenerationId) -> DatasetRevisionRef:
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        return DatasetRevisionRef(
            self._block_id,
            generation,
            self._schema.fingerprint,
            self._revision,
        )

    @property
    def event_count(self) -> int:
        return self._event_count

    @property
    def ordered_metadata_digest(self) -> str:
        return self._ordered_metadata_digest

    @property
    def join_plan_digest(self) -> str:
        return self._join_plan_digest

    def _read_event_chunk(
        self,
        chunk_index: int,
    ) -> tuple[tuple[DatasetCellAddress, CameraFrameMetadata], ...]:
        self._root_lease.require_active()
        records = _decode_frame_event_chunk(
            reference=self._event_chunk_refs[chunk_index],
            chunk_index=chunk_index,
            event_count=self._event_count,
            schema=self._schema,
            store_authority=self._store,
        )
        return records

    def iter_event_records(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraFrameMetadata]]:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            for chunk_index in range(len(self._event_chunk_refs)):
                records = self._read_event_chunk(chunk_index)
                yield from records

    def iter_cell_schedule(self) -> Iterator[DatasetCellAddress]:
        for cell, _metadata in self.iter_event_records():
            yield cell

    def _sample(
        self,
        ordinal: int,
        payload: bytes,
        metadata: CameraFrameMetadata,
    ) -> CameraSample:
        local = ordinal % self._geometry.frames_per_chunk
        offset = local * self._geometry.record_nbytes
        frame = np.frombuffer(
            payload,
            dtype=self._schema.cell_schema.dtype,
            count=math.prod(self._schema.cell_schema.data_shape),
            offset=offset,
        ).reshape(self._schema.cell_schema.data_shape, order="C")
        validity_offset = offset + self._geometry.frame_nbytes
        if self._validity_kind == "valid":
            validity = VALID
        elif self._validity_kind == "invalid":
            validity = INVALID
        elif self._validity_kind == "cell":
            encoded = payload[validity_offset]
            if encoded not in (0, 1):
                raise ValueError("cell validity byte is not canonical bool")
            validity = VALID if encoded else INVALID
        else:
            encoded = np.frombuffer(
                payload,
                dtype=np.uint8,
                count=self._geometry.validity_nbytes,
                offset=validity_offset,
            )
            if np.any(encoded > 1):
                raise ValueError("component validity bytes are not canonical bool")
            shape = tuple(
                self._schema.cell_schema.axis(axis_id).size
                for axis_id in self._validity_axis_ids
            )
            validity = ComponentValidity(
                self._validity_axis_ids,
                encoded.view(bool).reshape(shape, order="C"),
            )
        canonical = canonical_value_array(frame, validity, self._schema.cell_schema)
        raw = memoryview(payload)[offset : offset + self._geometry.frame_nbytes]
        if canonical is None:
            matches = not any(raw)
        else:
            matches = memoryview(canonical).cast("B") == raw
        if not matches:
            raise ValueError("frame chunk contains non-canonical value bytes")
        del canonical, raw
        return CameraSample(
            Value(frame, validity, self._schema.cell_schema),
            metadata,
        )

    def read(self, cell: DatasetCellAddress) -> CameraSample:
        iterator = self.iter_cells((cell,))
        try:
            return next(iterator)[1]
        finally:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()

    def iter_cells(
        self,
        cells: Iterable[DatasetCellAddress],
    ) -> Iterator[tuple[DatasetCellAddress, CameraSample]]:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            active_chunk = -1
            payload: bytes | None = None
            active_event_chunk = -1
            event_records: tuple[
                tuple[DatasetCellAddress, CameraFrameMetadata], ...
            ] | None = None
            for cell in cells:
                if not isinstance(cell, DatasetCellAddress):
                    raise TypeError("iter_cells requires DatasetCellAddress values")
                if (
                    cell.repeat_index >= self._schema.repeat_axis.size
                    or cell.point_storage_index >= self._schema.point_layout.storage_size
                ):
                    raise KeyError("cell is outside this capture frame source")
                linear_cell = (
                    cell.repeat_index * self._schema.point_layout.storage_size
                    + cell.point_storage_index
                )
                ordinal = _read_inverse_ordinal(
                    self._ordinal_by_linear_cell,
                    self._inverse_ordinal_width,
                    linear_cell,
                )
                event_chunk = ordinal // _FRAME_EVENT_CHUNK_MAX_EVENTS
                if event_chunk != active_event_chunk:
                    event_records = self._read_event_chunk(event_chunk)
                    active_event_chunk = event_chunk
                assert event_records is not None
                persisted_cell, metadata = event_records[
                    ordinal % _FRAME_EVENT_CHUNK_MAX_EVENTS
                ]
                if persisted_cell != cell:
                    raise RuntimeError("capture inverse index differs from event chunk")
                chunk = ordinal // self._geometry.frames_per_chunk
                if chunk != active_chunk:
                    payload = None
                    payload = self._store.read_blob(self._chunk_refs[chunk])
                    active_chunk = chunk
                assert payload is not None
                yield cell, self._sample(ordinal, payload, metadata)

    def iter_event_order(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraSample]]:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            active_chunk = -1
            payload: bytes | None = None
            for ordinal, (cell, metadata) in enumerate(self.iter_event_records()):
                chunk = ordinal // self._geometry.frames_per_chunk
                if chunk != active_chunk:
                    payload = self._store.read_blob(self._chunk_refs[chunk])
                    active_chunk = chunk
                assert payload is not None
                yield cell, self._sample(ordinal, payload, metadata)

    def materialize(
        self,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> DataBlock:
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            if abort_check is not None:
                abort_check()
            values = np.empty(
                self._schema.physical_shape,
                dtype=self._schema.cell_schema.dtype,
            )
            validity_values = (
                None
                if self._validity_kind in {"valid", "invalid"}
                else np.empty(
                    (
                        self._schema.repeat_axis.size,
                        self._schema.point_layout.storage_size,
                    )
                    + (
                        ()
                        if self._validity_kind == "cell"
                        else tuple(
                            self._schema.cell_schema.axis(axis_id).size
                            for axis_id in self._validity_axis_ids
                        )
                    ),
                    dtype=bool,
                )
            )
            for cell, sample in self.iter_event_order():
                if abort_check is not None:
                    abort_check()
                location = (cell.repeat_index, cell.point_storage_index)
                values[location] = sample.image.values
                if self._validity_kind == "cell":
                    assert validity_values is not None
                    validity_values[location] = isinstance(
                        sample.image.validity,
                        Valid,
                    )
                elif self._validity_kind == "component":
                    assert validity_values is not None
                    assert isinstance(sample.image.validity, ComponentValidity)
                    validity_values[location] = sample.image.validity.mask
            if self._validity_kind == "valid":
                validity = VALID
            elif self._validity_kind == "invalid":
                validity = INVALID
            elif self._validity_kind == "cell":
                assert validity_values is not None
                validity = CellValidity(validity_values)
            else:
                assert validity_values is not None
                validity = ComponentValidity(
                    self._validity_axis_ids,
                    validity_values,
                )
            if abort_check is not None:
                abort_check()
            return DataBlock(
                self._block_id,
                self._revision,
                values,
                validity,
                self._schema,
            )

    def _verify_all_frame_chunks(self) -> None:
        with self._root_lease.borrow() as read_borrow:
            read_borrow.require_active()
            for reference in self._chunk_refs:
                self._store.verify_blob(reference)


def _frame_schema_tree(schema: DatasetSchema) -> dict[str, object]:
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    return {
        "schema": _FRAME_SCHEMA_SCHEMA,
        "dataset_schema": dataset_schema_to_tree(schema),
    }


def _frame_event_chunk_tree(
    schedule: DatasetCellSchedule,
    metadata: tuple[CameraFrameMetadata, ...],
    start_ordinal: int,
    stop_ordinal: int,
) -> dict[str, object]:
    start = nonnegative_integer(start_ordinal, "start_ordinal")
    stop = nonnegative_integer(stop_ordinal, "stop_ordinal")
    if len(schedule) != len(metadata):
        raise ValueError("frame-event schedule and metadata cardinality differ")
    if stop <= start or stop > len(schedule):
        raise ValueError("frame-event chunk bounds are invalid")
    if stop - start > _FRAME_EVENT_CHUNK_MAX_EVENTS:
        raise ValueError("frame-event chunk exceeds its event-count bound")
    return {
        "schema": _FRAME_EVENT_CHUNK_SCHEMA,
        "start_ordinal": start,
        "events": [
            {
                "cell": _cell_address_to_tree(schedule.cell_at(ordinal)),
                "metadata": camera_frame_metadata_to_tree(
                    metadata[ordinal]
                ),
            }
            for ordinal in range(start, stop)
        ],
    }


def _frame_index_tree(
    source: CaptureFrameSource,
    dataset_schema_ref: ContentRef,
) -> dict[str, object]:
    if not isinstance(dataset_schema_ref, ContentRef):
        raise TypeError("dataset_schema_ref must be ContentRef")
    return {
        "schema": _FRAME_INDEX_SCHEMA,
        "block_id": source.block_id.value,
        "revision": source.revision.value,
        "dataset_schema_blob": content_ref_to_tree(dataset_schema_ref),
        "validity": {
            "kind": source._validity_kind,
            "axis_ids": [axis_id.value for axis_id in source._validity_axis_ids],
        },
        "event_count": source.event_count,
        "event_chunks": [
            content_ref_to_tree(item) for item in source._event_chunk_refs
        ],
        "frame_chunks": [content_ref_to_tree(item) for item in source._chunk_refs],
    }


def _stage_capture_frame_source(
    *,
    block: DataBlock,
    event_metadata: tuple[CameraFrameMetadata, ...],
    cell_schedule: DatasetCellSchedule,
    store_authority: ContentStoreAuthority,
    root_lease: RepositoryRootLease,
) -> tuple[CaptureFrameSource, ContentRef]:
    if not isinstance(block, DataBlock):
        raise TypeError("block must be DataBlock")
    if not isinstance(cell_schedule, DatasetCellSchedule):
        raise TypeError("cell_schedule must be DatasetCellSchedule")
    schedule, metadata = cell_schedule, tuple(event_metadata)
    validity_kind, validity_axis_ids = _validity_descriptor(block.validity)
    geometry = _capture_frame_record_geometry(
        block.schema,
        validity_kind,
        validity_axis_ids,
        len(schedule),
    )
    (
        ordinal_by_linear_cell,
        inverse_ordinal_width,
        join_plan_digest,
        ordered_metadata_digest,
    ) = _validate_frame_events(
        block.schema,
        schedule,
        metadata,
    )

    schema_tree = _frame_schema_tree(block.schema)
    schema_payload = encode(schema_tree)
    dataset_schema_ref = store_authority.put_blob(schema_payload)
    del schema_payload, schema_tree

    chunks: list[ContentRef] = []
    buffer = bytearray()
    for ordinal, cell in enumerate(schedule):
        frame_validity, validity_bytes = _frame_validity(block, cell)
        frame = block.values[cell.repeat_index, cell.point_storage_index]
        buffer.extend(_canonical_frame_bytes(frame, frame_validity, block.schema.cell_schema))
        buffer.extend(validity_bytes)
        if (ordinal + 1) % geometry.frames_per_chunk == 0:
            chunks.append(store_authority.put_blob(buffer))
            buffer.clear()
    if buffer:
        chunks.append(store_authority.put_blob(buffer))
    event_chunk_refs: list[ContentRef] = []
    for start in range(0, len(schedule), _FRAME_EVENT_CHUNK_MAX_EVENTS):
        stop = min(len(schedule), start + _FRAME_EVENT_CHUNK_MAX_EVENTS)
        event_tree = _frame_event_chunk_tree(schedule, metadata, start, stop)
        event_payload = encode(event_tree)
        event_ref = store_authority.put_blob(event_payload)
        event_chunk_refs.append(event_ref)
        del event_payload, event_tree

    source = CaptureFrameSource(
        _CAPTURE_FRAME_SOURCE_TOKEN,
        schema=block.schema,
        block_id=block.block_id,
        revision=block.revision,
        event_count=len(schedule),
        event_chunk_refs=tuple(event_chunk_refs),
        ordered_metadata_digest=ordered_metadata_digest,
        join_plan_digest=join_plan_digest,
        ordinal_by_linear_cell=ordinal_by_linear_cell,
        inverse_ordinal_width=inverse_ordinal_width,
        validity_kind=validity_kind,
        validity_axis_ids=validity_axis_ids,
        chunk_refs=tuple(chunks),
        store_authority=store_authority,
        root_lease=root_lease,
    )

    root_tree = _frame_index_tree(
        source,
        dataset_schema_ref,
    )
    root_payload = encode(root_tree)
    return source, store_authority.put_blob(root_payload)


def _load_capture_frame_source(
    reference: ContentRef,
    *,
    store_authority: ContentStoreAuthority,
    root_lease: RepositoryRootLease,
    abort_check: Callable[[], None] | None = None,
) -> CaptureFrameSource:
    payload = store_authority.read_blob(reference)
    return _capture_frame_source_from_payload(
        payload,
        store_authority=store_authority,
        root_lease=root_lease,
        abort_check=abort_check,
    )


def _inspect_capture_frame_source(
    reference_or_payload: ContentRef | bytes,
    *,
    store_authority: ContentStoreAuthority,
) -> _CaptureFrameSourceInspection:
    """Read root, schema, references, and canonical frame geometry.

    Passing a ``ContentRef`` reads that root blob; passing already-read ``bytes``
    avoids a second root read.  Neither form reads an event metadata chunk or a
    raw frame chunk.
    """

    if isinstance(reference_or_payload, ContentRef):
        payload = store_authority.read_blob(reference_or_payload)
    elif type(reference_or_payload) is bytes:
        payload = reference_or_payload
    else:
        raise TypeError("reference_or_payload must be ContentRef or bytes")
    def admit_root(events) -> None:
        for event in events:
            if isinstance(event, CanonicalArrayEvent):
                raise ValueError(
                    "capture frame-index root cannot embed ndarrays"
                )

    tree = exact_mapping(
        decode(
            payload,
            admit_structure=admit_root,
        ),
        {
            "schema",
            "block_id",
            "revision",
            "dataset_schema_blob",
            "validity",
            "event_count",
            "event_chunks",
            "frame_chunks",
        },
        _FRAME_INDEX_SCHEMA,
    )
    validity = exact_mapping(
        tree["validity"],
        {"kind", "axis_ids"},
        "frame validity",
        discriminator=None,
    )
    axis_ids = validity["axis_ids"]
    if not isinstance(axis_ids, list):
        raise ValueError("frame validity axis_ids must be a list")
    event_count = positive_integer(tree["event_count"], "event_count")
    schema_ref = content_ref_from_tree(tree["dataset_schema_blob"])
    event_chunk_tree = tree["event_chunks"]
    frame_chunk_tree = tree["frame_chunks"]
    if not isinstance(event_chunk_tree, list) or not isinstance(frame_chunk_tree, list):
        raise ValueError("frame index event and frame chunks must be lists")
    expected_event_chunks = (
        event_count + _FRAME_EVENT_CHUNK_MAX_EVENTS - 1
    ) // _FRAME_EVENT_CHUNK_MAX_EVENTS
    if len(event_chunk_tree) != expected_event_chunks:
        raise ValueError("frame-event chunk count differs from event cardinality")
    frame_chunk_refs = tuple(
        content_ref_from_tree(item) for item in frame_chunk_tree
    )
    event_chunk_refs = tuple(
        content_ref_from_tree(item) for item in event_chunk_tree
    )
    def admit_schema(events) -> None:
        if any(isinstance(event, CanonicalArrayEvent) for event in events):
            raise ValueError(
                "capture dataset-schema index component cannot embed ndarrays"
            )
    schema_payload = store_authority.read_blob(schema_ref)
    schema_tree = exact_mapping(
        decode(
            schema_payload,
            admit_structure=admit_schema,
        ),
        {"schema", "dataset_schema"},
        _FRAME_SCHEMA_SCHEMA,
    )
    schema = dataset_schema_from_tree(schema_tree["dataset_schema"])
    physical_cells = schema.repeat_axis.size * schema.point_layout.storage_size
    if event_count != physical_cells:
        raise ValueError("frame event count differs from DatasetSchema storage")
    validity_kind = validity["kind"]
    validity_axis_ids = tuple(AxisId(item) for item in axis_ids)
    geometry = _capture_frame_record_geometry(
        schema,
        validity_kind,
        validity_axis_ids,
        event_count,
    )
    if len(frame_chunk_refs) != geometry.expected_chunks:
        raise ValueError("frame chunk count differs from canonical chunking")
    for index, reference in enumerate(frame_chunk_refs):
        count = min(
            geometry.frames_per_chunk,
            event_count - index * geometry.frames_per_chunk,
        )
        if reference.size != count * geometry.record_nbytes:
            raise ValueError("frame chunk size differs from canonical record layout")

    block_id = BlockId(tree["block_id"])
    revision = DatasetRevision(nonnegative_integer(tree["revision"], "revision"))
    return _CaptureFrameSourceInspection(
        dataset_schema=schema,
        block_id=block_id,
        revision=revision,
        event_count=event_count,
        event_chunk_refs=event_chunk_refs,
        frame_chunk_refs=frame_chunk_refs,
        validity_kind=validity_kind,
        validity_axis_ids=validity_axis_ids,
        geometry=geometry,
    )


def _capture_frame_source_from_payload(
    payload: bytes,
    *,
    store_authority: ContentStoreAuthority,
    root_lease: RepositoryRootLease,
    abort_check: Callable[[], None] | None = None,
) -> CaptureFrameSource:
    if abort_check is not None and not callable(abort_check):
        raise TypeError("abort_check must be callable or None")
    if abort_check is not None:
        abort_check()
    inspection = _inspect_capture_frame_source(
        payload,
        store_authority=store_authority,
    )
    schema = inspection.dataset_schema
    event_count = inspection.event_count
    event_chunk_refs = inspection.event_chunk_refs
    frame_chunk_refs = inspection.frame_chunk_refs

    metadata_contract = CameraFrameMetadataContract()
    metadata_hasher = OrderedDatasetMetadataHasher(metadata_contract.fingerprint)
    inverse_ordinal_width = _inverse_ordinal_width(event_count)
    ordinal_by_linear_cell = bytearray(event_count * inverse_ordinal_width)

    def streamed_cells() -> Iterator[DatasetCellAddress]:
        for chunk_index, reference in enumerate(event_chunk_refs):
            if abort_check is not None:
                abort_check()
            records = _decode_frame_event_chunk(
                reference=reference,
                chunk_index=chunk_index,
                event_count=event_count,
                schema=schema,
                store_authority=store_authority,
            )
            start = chunk_index * _FRAME_EVENT_CHUNK_MAX_EVENTS
            for offset, (cell, item) in enumerate(records):
                if abort_check is not None and offset % 1024 == 0:
                    abort_check()
                ordinal = start + offset
                metadata_hasher.update(metadata_contract.digest(item))
                linear_cell = (
                    cell.repeat_index * schema.point_layout.storage_size
                    + cell.point_storage_index
                )
                _write_inverse_ordinal(
                    ordinal_by_linear_cell,
                    inverse_ordinal_width,
                    linear_cell,
                    ordinal,
                )
                yield cell

    join_plan_digest = dataset_cell_permutation_digest(schema, streamed_cells())
    ordered_metadata_digest = metadata_hasher.digest()
    if abort_check is not None:
        abort_check()

    source = CaptureFrameSource(
        _CAPTURE_FRAME_SOURCE_TOKEN,
        schema=schema,
        block_id=inspection.block_id,
        revision=inspection.revision,
        event_count=event_count,
        event_chunk_refs=event_chunk_refs,
        ordered_metadata_digest=ordered_metadata_digest,
        join_plan_digest=join_plan_digest,
        ordinal_by_linear_cell=bytes(ordinal_by_linear_cell),
        inverse_ordinal_width=inverse_ordinal_width,
        validity_kind=inspection.validity_kind,
        validity_axis_ids=inspection.validity_axis_ids,
        chunk_refs=frame_chunk_refs,
        store_authority=store_authority,
        root_lease=root_lease,
    )
    return source


__all__ = [
    "CaptureFrameSource",
]
