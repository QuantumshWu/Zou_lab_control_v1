"""Bounded binary frame storage for committed raw camera captures."""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Iterator
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
    CanonicalDecodeLimits,
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


_FRAME_CHUNK_TARGET_BYTES = 64 * 1024 * 1024
_FRAME_INDEX_SCHEMA = "zlc_neutral_atom.CaptureFrameIndex"
_FRAME_SCHEMA_SCHEMA = "zlc_neutral_atom.CaptureFrameDatasetSchema"
_FRAME_EVENT_CHUNK_SCHEMA = "zlc_neutral_atom.CaptureFrameEventChunk"
_FRAME_INDEX_ROOT_MAX_BYTES = 4 * 1024 * 1024
_FRAME_SCHEMA_MAX_BYTES = 16 * 1024 * 1024
_FRAME_EVENT_CHUNK_MAX_BYTES = 1 * 1024 * 1024
_FRAME_EVENT_CHUNK_MAX_EVENTS = 256
_VALIDITY_KINDS = frozenset({"valid", "invalid", "cell", "component"})


class _FrameResourceExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _FrameRecordGeometry:
    """One canonical frame-record and chunk layout calculation."""

    frame_nbytes: int
    validity_nbytes: int
    record_nbytes: int
    frames_per_chunk: int
    expected_chunks: int
    largest_chunk_nbytes: int


def _capture_frame_record_geometry(
    schema: DatasetSchema,
    validity_kind: str,
    validity_axis_ids: tuple[AxisId, ...],
    cell_count: int,
    max_chunk_blob_bytes: int,
) -> _FrameRecordGeometry:
    """Return the sole binary layout used by admission, staging, and reading."""

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
    max_chunk_blob_bytes = positive_integer(
        max_chunk_blob_bytes, "max_chunk_blob_bytes"
    )
    frame_nbytes = (
        math.prod(schema.cell_schema.data_shape) * schema.cell_schema.dtype.itemsize
    )
    record_nbytes = frame_nbytes + validity_nbytes
    if record_nbytes <= 0 or record_nbytes > max_chunk_blob_bytes:
        raise _FrameResourceExceeded("one capture frame record exceeds chunk policy")
    chunk_capacity = min(_FRAME_CHUNK_TARGET_BYTES, max_chunk_blob_bytes)
    frames_per_chunk = max(1, chunk_capacity // record_nbytes)
    expected_chunks = (cell_count + frames_per_chunk - 1) // frames_per_chunk
    largest_chunk_nbytes = min(cell_count, frames_per_chunk) * record_nbytes
    return _FrameRecordGeometry(
        frame_nbytes=frame_nbytes,
        validity_nbytes=validity_nbytes,
        record_nbytes=record_nbytes,
        frames_per_chunk=frames_per_chunk,
        expected_chunks=expected_chunks,
        largest_chunk_nbytes=largest_chunk_nbytes,
    )


def _capture_frame_read_scratch_bytes(
    schema: DatasetSchema,
    geometry: _FrameRecordGeometry,
    largest_chunk_bytes: int,
) -> int:
    """Peak for one resolved chunk layout and validity representation."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(geometry, _FrameRecordGeometry):
        raise TypeError("geometry must be _FrameRecordGeometry")
    largest_chunk_bytes = positive_integer(
        largest_chunk_bytes, "largest_chunk_bytes"
    )
    nan_mask_nbytes = (
        math.prod(schema.cell_schema.data_shape)
        if schema.cell_schema.dtype.kind in "fc"
        else 0
    )
    # While a generator resumes, its caller may still hold the prior immutable
    # frame and component mask.  The next Value/Validity construction owns one
    # more copy while the newly read chunk remains live.  Float/complex
    # canonicalization may additionally hold one dense np.isnan mask.
    return (
        largest_chunk_bytes
        + 2 * geometry.frame_nbytes
        + 2 * geometry.validity_nbytes
        + nan_mask_nbytes
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


def _admit_index_tree(
    tree: object,
    used_nodes: int,
    used_container_entries: int,
    max_canonical_nodes: int,
    max_canonical_container_entries: int,
) -> tuple[int, int]:
    """Apply the reader's structural limits without serializing twice."""

    nodes = used_nodes
    entries = used_container_entries

    def visit(value: object, depth: int) -> None:
        nonlocal nodes, entries
        nodes += 1
        if depth > 128 or nodes > max_canonical_nodes:
            raise _FrameResourceExceeded("capture frame index exceeds canonical policy")
        if isinstance(value, np.ndarray):
            raise _FrameResourceExceeded("capture frame index cannot embed ndarrays")
        if isinstance(value, dict):
            children = value.values()
        elif isinstance(value, (list, tuple)):
            children = value
        else:
            return
        entries += len(value)
        if entries > max_canonical_container_entries:
            raise _FrameResourceExceeded("capture frame index exceeds canonical policy")
        for child in children:
            visit(child, depth + 1)

    visit(tree, 0)
    return nodes, entries


def _index_tree_encoding_upper_bound(tree: object) -> int:
    """Conservative pre-encode bound for one ndarray-free index component."""

    def size(value: object) -> int:
        if isinstance(value, np.ndarray):
            raise _FrameResourceExceeded("capture frame index cannot embed ndarrays")
        if isinstance(value, dict):
            total = 96
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError("capture frame index mapping keys must be text")
                total += 32 + 8 * len(key.encode("utf-8")) + size(child)
            return total
        if isinstance(value, (list, tuple)):
            return 64 + sum(size(child) for child in value)
        if isinstance(value, str):
            return 32 + 8 * len(value.encode("utf-8"))
        if isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError("capture frame index does not support bytes-like leaves")
        if value is None or isinstance(value, (bool, float)):
            return 64
        if isinstance(value, int):
            return 32 + len(str(value))
        raise TypeError(
            f"capture frame index contains unsupported {type(value).__name__}"
        )

    return 32 + size(tree)


def _inverse_ordinal_width(event_count: int) -> int:
    count = positive_integer(event_count, "event_count")
    if count <= 0xFFFFFFFF:
        return 4
    if count <= 0xFFFFFFFFFFFFFFFF:
        return 8
    raise _FrameResourceExceeded("capture event count exceeds compact-index range")


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
    max_event_chunk_blob_bytes: int,
    max_canonical_nodes: int,
    max_canonical_container_entries: int,
) -> tuple[tuple[tuple[DatasetCellAddress, CameraFrameMetadata], ...], int, int]:
    """Decode and structurally validate one bounded immutable event chunk."""

    start = nonnegative_integer(chunk_index, "chunk_index") * _FRAME_EVENT_CHUNK_MAX_EVENTS
    if start >= event_count:
        raise IndexError("capture frame-event chunk is outside the event domain")
    expected_count = min(_FRAME_EVENT_CHUNK_MAX_EVENTS, event_count - start)

    def admit_event_chunk(events) -> None:
        for event in events:
            if isinstance(event, CanonicalArrayEvent):
                raise _FrameResourceExceeded(
                    "capture frame-event chunk cannot embed ndarrays"
                )
            if (
                isinstance(event, CanonicalListEvent)
                and event.path == ("events",)
                and event.length > _FRAME_EVENT_CHUNK_MAX_EVENTS
            ):
                raise _FrameResourceExceeded(
                    "capture frame-event chunk exceeds event-count policy"
                )

    event_payload = store_authority.read_blob(
        reference,
        max_bytes=max_event_chunk_blob_bytes,
    )
    event_tree = exact_mapping(
        decode(
            event_payload,
            admit_structure=admit_event_chunk,
            limits=CanonicalDecodeLimits(
                max_depth=128,
                max_nodes=max_canonical_nodes,
                max_container_entries=max_canonical_container_entries,
                max_arrays=0,
                max_total_array_bytes=0,
            ),
        ),
        {"schema", "start_ordinal", "events"},
        _FRAME_EVENT_CHUNK_SCHEMA,
    )
    used_nodes, used_container_entries = _admit_index_tree(
        event_tree,
        0,
        0,
        max_canonical_nodes,
        max_canonical_container_entries,
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
    return tuple(records), used_nodes, used_container_entries


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
        "_max_chunk_blob_bytes",
        "_max_event_chunk_blob_bytes",
        "_max_canonical_nodes",
        "_max_canonical_container_entries",
        "_ordinal_by_linear_cell",
        "_inverse_ordinal_width",
        "_geometry",
        "_max_read_scratch_bytes",
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
        max_total_frame_bytes: int,
        max_chunk_blob_bytes: int,
        max_event_chunk_blob_bytes: int,
        max_canonical_nodes: int,
        max_canonical_container_entries: int,
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
        max_chunk_blob_bytes = positive_integer(
            max_chunk_blob_bytes, "max_chunk_blob_bytes"
        )
        max_event_chunk_blob_bytes = positive_integer(
            max_event_chunk_blob_bytes,
            "max_event_chunk_blob_bytes",
        )
        max_canonical_nodes = positive_integer(
            max_canonical_nodes,
            "max_canonical_nodes",
        )
        max_canonical_container_entries = positive_integer(
            max_canonical_container_entries,
            "max_canonical_container_entries",
        )
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
        if any(
            item.size > max_event_chunk_blob_bytes
            for item in event_refs
        ):
            raise _FrameResourceExceeded("capture frame-event chunk exceeds policy")
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
            max_chunk_blob_bytes,
        )
        if event_count * geometry.record_nbytes > positive_integer(
            max_total_frame_bytes, "max_total_frame_bytes"
        ):
            raise _FrameResourceExceeded("capture frame bytes exceed repository policy")
        if len(refs) != geometry.expected_chunks:
            raise ValueError("frame chunk count differs from canonical chunking")
        for index, reference in enumerate(refs):
            count = min(
                geometry.frames_per_chunk,
                event_count - index * geometry.frames_per_chunk,
            )
            if reference.size != count * geometry.record_nbytes:
                raise ValueError("frame chunk size differs from canonical record layout")
            if reference.size > max_chunk_blob_bytes:
                raise _FrameResourceExceeded("frame chunk exceeds repository policy")
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
        object.__setattr__(self, "_max_chunk_blob_bytes", max_chunk_blob_bytes)
        object.__setattr__(
            self,
            "_max_event_chunk_blob_bytes",
            max_event_chunk_blob_bytes,
        )
        object.__setattr__(self, "_max_canonical_nodes", max_canonical_nodes)
        object.__setattr__(
            self,
            "_max_canonical_container_entries",
            max_canonical_container_entries,
        )
        object.__setattr__(
            self,
            "_ordinal_by_linear_cell",
            ordinal_by_linear_cell,
        )
        object.__setattr__(self, "_inverse_ordinal_width", inverse_ordinal_width)
        object.__setattr__(self, "_geometry", geometry)
        event_decode_scratch = (
            8 * max(item.size for item in event_refs)
            + _FRAME_EVENT_CHUNK_MAX_EVENTS
            * (CameraFrameMetadataContract().max_retained_nbytes + 128)
        )
        object.__setattr__(
            self,
            "_max_read_scratch_bytes",
            _capture_frame_read_scratch_bytes(
                schema,
                geometry,
                geometry.largest_chunk_nbytes,
            )
            + event_decode_scratch,
        )

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

    @property
    def max_read_scratch_bytes(self) -> int:
        return self._max_read_scratch_bytes

    def _read_event_chunk(
        self,
        chunk_index: int,
    ) -> tuple[tuple[DatasetCellAddress, CameraFrameMetadata], ...]:
        self._root_lease.require_active()
        records, _, _ = _decode_frame_event_chunk(
            reference=self._event_chunk_refs[chunk_index],
            chunk_index=chunk_index,
            event_count=self._event_count,
            schema=self._schema,
            store_authority=self._store,
            max_event_chunk_blob_bytes=self._max_event_chunk_blob_bytes,
            max_canonical_nodes=self._max_canonical_nodes,
            max_canonical_container_entries=self._max_canonical_container_entries,
        )
        return records

    def iter_event_records(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraFrameMetadata]]:
        self._root_lease.require_active()
        for chunk_index in range(len(self._event_chunk_refs)):
            self._root_lease.require_active()
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
        self._root_lease.require_active()
        active_chunk = -1
        payload: bytes | None = None
        active_event_chunk = -1
        event_records: tuple[
            tuple[DatasetCellAddress, CameraFrameMetadata], ...
        ] | None = None
        for cell in cells:
            self._root_lease.require_active()
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
                payload = self._store.read_blob(
                    self._chunk_refs[chunk],
                    max_bytes=self._max_chunk_blob_bytes,
                )
                active_chunk = chunk
            assert payload is not None
            yield cell, self._sample(ordinal, payload, metadata)

    def iter_event_order(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraSample]]:
        self._root_lease.require_active()
        active_chunk = -1
        payload: bytes | None = None
        for ordinal, (cell, metadata) in enumerate(self.iter_event_records()):
            chunk = ordinal // self._geometry.frames_per_chunk
            if chunk != active_chunk:
                payload = self._store.read_blob(
                    self._chunk_refs[chunk],
                    max_bytes=self._max_chunk_blob_bytes,
                )
                active_chunk = chunk
            assert payload is not None
            yield cell, self._sample(ordinal, payload, metadata)

    def materialize(self, *, memory_limit_bytes: int) -> DataBlock:
        self._root_lease.require_active()
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        cell_count = self._event_count
        values_nbytes = cell_count * self._geometry.frame_nbytes
        validity_nbytes = cell_count * self._geometry.validity_nbytes
        required = 2 * values_nbytes + 2 * validity_nbytes + self.max_read_scratch_bytes
        if required > limit:
            raise MemoryError(
                f"capture materialization peak {required} exceeds limit {limit}"
            )
        values = np.empty(self._schema.physical_shape, dtype=self._schema.cell_schema.dtype)
        validity_values = (
            None
            if self._validity_kind in {"valid", "invalid"}
            else np.empty(
                (self._schema.repeat_axis.size, self._schema.point_layout.storage_size)
                + (() if self._validity_kind == "cell" else tuple(
                    self._schema.cell_schema.axis(axis_id).size
                    for axis_id in self._validity_axis_ids
                )),
                dtype=bool,
            )
        )
        for cell, sample in self.iter_event_order():
            location = (cell.repeat_index, cell.point_storage_index)
            values[location] = sample.image.values
            if self._validity_kind == "cell":
                assert validity_values is not None
                validity_values[location] = isinstance(sample.image.validity, Valid)
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
            validity = ComponentValidity(self._validity_axis_ids, validity_values)
        return DataBlock(self._block_id, self._revision, values, validity, self._schema)

    def _verify_all_frame_chunks(self) -> None:
        self._root_lease.require_active()
        for reference in self._chunk_refs:
            self._root_lease.require_active()
            self._store.verify_blob(reference, max_bytes=self._max_chunk_blob_bytes)


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
    max_cells: int,
    max_total_frame_bytes: int,
    max_chunk_blob_bytes: int,
    max_frame_index_blob_bytes: int,
    max_canonical_nodes: int,
    max_canonical_container_entries: int,
) -> tuple[CaptureFrameSource, ContentRef]:
    if not isinstance(block, DataBlock):
        raise TypeError("block must be DataBlock")
    if not isinstance(cell_schedule, DatasetCellSchedule):
        raise TypeError("cell_schedule must be DatasetCellSchedule")
    schedule, metadata = cell_schedule, tuple(event_metadata)
    max_cells = positive_integer(max_cells, "max_cells")
    max_frame_index_blob_bytes = positive_integer(
        max_frame_index_blob_bytes,
        "max_frame_index_blob_bytes",
    )
    max_canonical_nodes = positive_integer(
        max_canonical_nodes,
        "max_canonical_nodes",
    )
    max_canonical_container_entries = positive_integer(
        max_canonical_container_entries,
        "max_canonical_container_entries",
    )
    if len(schedule) > max_cells or len(metadata) > max_cells:
        raise _FrameResourceExceeded("capture frame count exceeds repository policy")
    validity_kind, validity_axis_ids = _validity_descriptor(block.validity)
    max_chunk_blob_bytes = positive_integer(
        max_chunk_blob_bytes, "max_chunk_blob_bytes"
    )
    geometry = _capture_frame_record_geometry(
        block.schema,
        validity_kind,
        validity_axis_ids,
        len(schedule),
        max_chunk_blob_bytes,
    )
    total_bytes = len(schedule) * geometry.record_nbytes
    if total_bytes > positive_integer(max_total_frame_bytes, "max_total_frame_bytes"):
        raise _FrameResourceExceeded("capture frame bytes exceed repository policy")
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

    used_nodes = used_container_entries = 0
    schema_tree = _frame_schema_tree(block.schema)
    used_nodes, used_container_entries = _admit_index_tree(
        schema_tree,
        used_nodes,
        used_container_entries,
        max_canonical_nodes,
        max_canonical_container_entries,
    )
    schema_limit = min(_FRAME_SCHEMA_MAX_BYTES, max_frame_index_blob_bytes)
    if _index_tree_encoding_upper_bound(schema_tree) > schema_limit:
        raise _FrameResourceExceeded(
            "capture dataset-schema index component exceeds repository policy"
        )
    schema_payload = encode(schema_tree)
    if len(schema_payload) > schema_limit:
        raise _FrameResourceExceeded(
            "capture dataset-schema index component exceeds repository policy"
        )
    dataset_schema_ref = store_authority.put_blob(schema_payload)
    index_component_bytes = dataset_schema_ref.size
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
        used_nodes, used_container_entries = _admit_index_tree(
            event_tree,
            used_nodes,
            used_container_entries,
            max_canonical_nodes,
            max_canonical_container_entries,
        )
        event_limit = min(
            _FRAME_EVENT_CHUNK_MAX_BYTES,
            max_frame_index_blob_bytes,
        )
        if _index_tree_encoding_upper_bound(event_tree) > event_limit:
            raise _FrameResourceExceeded(
                "capture frame-event chunk exceeds its byte policy"
            )
        event_payload = encode(event_tree)
        if len(event_payload) > event_limit:
            raise _FrameResourceExceeded(
                "capture frame-event chunk exceeds its byte policy"
            )
        if index_component_bytes + len(event_payload) > max_frame_index_blob_bytes:
            raise _FrameResourceExceeded(
                "capture frame index exceeds repository resource policy"
            )
        event_ref = store_authority.put_blob(event_payload)
        event_chunk_refs.append(event_ref)
        index_component_bytes += event_ref.size
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
        max_total_frame_bytes=max_total_frame_bytes,
        max_chunk_blob_bytes=max_chunk_blob_bytes,
        max_event_chunk_blob_bytes=min(
            _FRAME_EVENT_CHUNK_MAX_BYTES,
            max_frame_index_blob_bytes,
        ),
        max_canonical_nodes=max_canonical_nodes,
        max_canonical_container_entries=max_canonical_container_entries,
    )

    root_tree = _frame_index_tree(
        source,
        dataset_schema_ref,
    )
    _admit_index_tree(
        root_tree,
        used_nodes,
        used_container_entries,
        max_canonical_nodes,
        max_canonical_container_entries,
    )
    root_limit = min(_FRAME_INDEX_ROOT_MAX_BYTES, max_frame_index_blob_bytes)
    if _index_tree_encoding_upper_bound(root_tree) > root_limit:
        raise _FrameResourceExceeded("capture frame-index root exceeds its byte policy")
    root_payload = encode(root_tree)
    if len(root_payload) > root_limit:
        raise _FrameResourceExceeded("capture frame-index root exceeds its byte policy")
    if index_component_bytes + len(root_payload) > max_frame_index_blob_bytes:
        raise _FrameResourceExceeded(
            "capture frame index exceeds repository resource policy"
        )
    return source, store_authority.put_blob(root_payload)


def _load_capture_frame_source(
    reference: ContentRef,
    *,
    store_authority: ContentStoreAuthority,
    root_lease: RepositoryRootLease,
    max_cells: int,
    max_total_frame_bytes: int,
    max_chunk_blob_bytes: int,
    max_frame_index_blob_bytes: int,
    max_canonical_nodes: int,
    max_canonical_container_entries: int,
) -> CaptureFrameSource:
    root_limit = min(
        _FRAME_INDEX_ROOT_MAX_BYTES,
        positive_integer(max_frame_index_blob_bytes, "max_frame_index_blob_bytes"),
    )
    if reference.size > root_limit:
        raise _FrameResourceExceeded("capture frame-index root exceeds its byte policy")
    payload = store_authority.read_blob(reference, max_bytes=root_limit)
    return _capture_frame_source_from_payload(
        payload,
        store_authority=store_authority,
        root_lease=root_lease,
        max_cells=max_cells,
        max_total_frame_bytes=max_total_frame_bytes,
        max_chunk_blob_bytes=max_chunk_blob_bytes,
        max_frame_index_blob_bytes=max_frame_index_blob_bytes,
        max_canonical_nodes=max_canonical_nodes,
        max_canonical_container_entries=max_canonical_container_entries,
    )


def _capture_frame_source_from_payload(
    payload: bytes,
    *,
    store_authority: ContentStoreAuthority,
    root_lease: RepositoryRootLease,
    max_cells: int,
    max_total_frame_bytes: int,
    max_chunk_blob_bytes: int,
    max_frame_index_blob_bytes: int,
    max_canonical_nodes: int,
    max_canonical_container_entries: int,
) -> CaptureFrameSource:
    max_cells = positive_integer(max_cells, "max_cells")
    max_total_frame_bytes = positive_integer(
        max_total_frame_bytes,
        "max_total_frame_bytes",
    )
    max_chunk_blob_bytes = positive_integer(
        max_chunk_blob_bytes,
        "max_chunk_blob_bytes",
    )
    max_frame_index_blob_bytes = positive_integer(
        max_frame_index_blob_bytes,
        "max_frame_index_blob_bytes",
    )
    max_canonical_nodes = positive_integer(
        max_canonical_nodes,
        "max_canonical_nodes",
    )
    max_canonical_container_entries = positive_integer(
        max_canonical_container_entries,
        "max_canonical_container_entries",
    )
    root_limit = min(_FRAME_INDEX_ROOT_MAX_BYTES, max_frame_index_blob_bytes)
    if len(payload) > root_limit:
        raise _FrameResourceExceeded("capture frame-index root exceeds its byte policy")
    max_event_chunks = (
        max_cells + _FRAME_EVENT_CHUNK_MAX_EVENTS - 1
    ) // _FRAME_EVENT_CHUNK_MAX_EVENTS

    def admit_root(events) -> None:
        for event in events:
            if isinstance(event, CanonicalArrayEvent):
                raise _FrameResourceExceeded(
                    "capture frame-index root cannot embed ndarrays"
                )
            if (
                isinstance(event, CanonicalListEvent)
                and (
                    (event.path == ("event_chunks",) and event.length > max_event_chunks)
                    or (event.path == ("frame_chunks",) and event.length > max_cells)
                )
            ):
                raise _FrameResourceExceeded(
                    "capture frame-index root exceeds reference-count policy"
                )

    tree = exact_mapping(
        decode(
            payload,
            admit_structure=admit_root,
            limits=CanonicalDecodeLimits(
                max_depth=128,
                max_nodes=max_canonical_nodes,
                max_container_entries=max_canonical_container_entries,
                max_arrays=0,
                max_total_array_bytes=0,
            ),
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
    used_nodes, used_container_entries = _admit_index_tree(
        tree,
        0,
        0,
        max_canonical_nodes,
        max_canonical_container_entries,
    )
    validity = exact_mapping(
        tree["validity"], {"kind", "axis_ids"}, "frame validity", discriminator=None
    )
    axis_ids = validity["axis_ids"]
    if not isinstance(axis_ids, list):
        raise ValueError("frame validity axis_ids must be a list")
    event_count = positive_integer(tree["event_count"], "event_count")
    if event_count > max_cells:
        raise _FrameResourceExceeded("capture frame count exceeds repository policy")
    schema_ref = content_ref_from_tree(tree["dataset_schema_blob"])
    schema_limit = min(_FRAME_SCHEMA_MAX_BYTES, max_frame_index_blob_bytes)
    if schema_ref.size > schema_limit:
        raise _FrameResourceExceeded(
            "capture dataset-schema index component exceeds byte policy"
        )
    event_chunk_tree = tree["event_chunks"]
    chunk_tree = tree["frame_chunks"]
    if not isinstance(event_chunk_tree, list) or not isinstance(chunk_tree, list):
        raise ValueError("frame index event and frame chunks must be lists")
    expected_event_chunks = (
        event_count + _FRAME_EVENT_CHUNK_MAX_EVENTS - 1
    ) // _FRAME_EVENT_CHUNK_MAX_EVENTS
    if len(event_chunk_tree) != expected_event_chunks:
        raise ValueError("frame-event chunk count differs from event cardinality")
    if len(chunk_tree) > max_cells:
        raise _FrameResourceExceeded("frame chunk count exceeds repository policy")
    frame_chunk_refs = tuple(content_ref_from_tree(item) for item in chunk_tree)
    event_chunk_refs = tuple(
        content_ref_from_tree(item) for item in event_chunk_tree
    )
    event_chunk_limit = min(
        _FRAME_EVENT_CHUNK_MAX_BYTES,
        max_frame_index_blob_bytes,
    )
    if any(reference.size > event_chunk_limit for reference in event_chunk_refs):
        raise _FrameResourceExceeded("capture frame-event chunk exceeds byte policy")
    index_bytes = (
        len(payload)
        + schema_ref.size
        + sum(reference.size for reference in event_chunk_refs)
    )
    if index_bytes > max_frame_index_blob_bytes:
        raise _FrameResourceExceeded(
            "capture frame index exceeds repository resource policy"
        )

    def admit_schema(events) -> None:
        if any(isinstance(event, CanonicalArrayEvent) for event in events):
            raise _FrameResourceExceeded(
                "capture dataset-schema index component cannot embed ndarrays"
            )

    schema_payload = store_authority.read_blob(
        schema_ref,
        max_bytes=schema_limit,
    )
    schema_tree = exact_mapping(
        decode(
            schema_payload,
            admit_structure=admit_schema,
            limits=CanonicalDecodeLimits(
                max_depth=128,
                max_nodes=max_canonical_nodes,
                max_container_entries=max_canonical_container_entries,
                max_arrays=0,
                max_total_array_bytes=0,
            ),
        ),
        {"schema", "dataset_schema"},
        _FRAME_SCHEMA_SCHEMA,
    )
    used_nodes, used_container_entries = _admit_index_tree(
        schema_tree,
        used_nodes,
        used_container_entries,
        max_canonical_nodes,
        max_canonical_container_entries,
    )
    schema = dataset_schema_from_tree(schema_tree["dataset_schema"])
    del schema_payload, schema_tree
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
        max_chunk_blob_bytes,
    )
    if event_count * geometry.record_nbytes > max_total_frame_bytes:
        raise _FrameResourceExceeded("capture frame bytes exceed repository policy")
    if len(frame_chunk_refs) != geometry.expected_chunks:
        raise ValueError("frame chunk count differs from canonical chunking")
    for index, reference in enumerate(frame_chunk_refs):
        count = min(
            geometry.frames_per_chunk,
            event_count - index * geometry.frames_per_chunk,
        )
        if reference.size != count * geometry.record_nbytes:
            raise ValueError("frame chunk size differs from canonical record layout")
        if reference.size > max_chunk_blob_bytes:
            raise _FrameResourceExceeded("frame chunk exceeds repository policy")
    block_id = BlockId(tree["block_id"])
    revision = DatasetRevision(nonnegative_integer(tree["revision"], "revision"))

    metadata_contract = CameraFrameMetadataContract()
    metadata_hasher = OrderedDatasetMetadataHasher(metadata_contract.fingerprint)
    inverse_ordinal_width = _inverse_ordinal_width(event_count)
    ordinal_by_linear_cell = bytearray(event_count * inverse_ordinal_width)

    def streamed_cells() -> Iterator[DatasetCellAddress]:
        nonlocal used_nodes, used_container_entries
        for chunk_index, reference in enumerate(event_chunk_refs):
            records, chunk_nodes, chunk_entries = _decode_frame_event_chunk(
                reference=reference,
                chunk_index=chunk_index,
                event_count=event_count,
                schema=schema,
                store_authority=store_authority,
                max_event_chunk_blob_bytes=event_chunk_limit,
                max_canonical_nodes=max_canonical_nodes,
                max_canonical_container_entries=max_canonical_container_entries,
            )
            used_nodes += chunk_nodes
            used_container_entries += chunk_entries
            if (
                used_nodes > max_canonical_nodes
                or used_container_entries > max_canonical_container_entries
            ):
                raise _FrameResourceExceeded(
                    "capture frame index exceeds canonical policy"
                )
            start = chunk_index * _FRAME_EVENT_CHUNK_MAX_EVENTS
            for offset, (cell, item) in enumerate(records):
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

    source = CaptureFrameSource(
        _CAPTURE_FRAME_SOURCE_TOKEN,
        schema=schema,
        block_id=block_id,
        revision=revision,
        event_count=event_count,
        event_chunk_refs=event_chunk_refs,
        ordered_metadata_digest=ordered_metadata_digest,
        join_plan_digest=join_plan_digest,
        ordinal_by_linear_cell=bytes(ordinal_by_linear_cell),
        inverse_ordinal_width=inverse_ordinal_width,
        validity_kind=validity_kind,
        validity_axis_ids=validity_axis_ids,
        chunk_refs=frame_chunk_refs,
        store_authority=store_authority,
        root_lease=root_lease,
        max_total_frame_bytes=max_total_frame_bytes,
        max_chunk_blob_bytes=max_chunk_blob_bytes,
        max_event_chunk_blob_bytes=event_chunk_limit,
        max_canonical_nodes=max_canonical_nodes,
        max_canonical_container_entries=max_canonical_container_entries,
    )
    return source


__all__ = [
    "CaptureFrameSource",
]
