"""Bounded binary frame storage for committed raw camera captures."""

from __future__ import annotations

import math
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
    DatasetSchema,
    Invalid,
    Valid,
    ValidityMode,
    Value,
    canonical_value_array,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
)
from zlc_storage import (
    CanonicalArrayEvent,
    CanonicalDecodeLimits,
    CanonicalEncodingError,
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
)

from zlc_neutral_atom.acquisition import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    camera_frame_metadata_from_tree,
    camera_frame_metadata_to_tree,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress


_FRAME_CHUNK_TARGET_BYTES = 64 * 1024 * 1024
_FRAME_INDEX_SCHEMA = "zlc_neutral_atom.CaptureFrameIndex"
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
    scratch_chunk_upper_nbytes: int


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
    # Compile-time admission historically reserves the whole policy capacity,
    # including any trailing bytes that cannot hold another complete record.
    # Preserve that conservative bound while deriving it from the same layout.
    scratch_chunk_upper_nbytes = min(
        cell_count * record_nbytes,
        max(record_nbytes, chunk_capacity),
    )
    return _FrameRecordGeometry(
        frame_nbytes=frame_nbytes,
        validity_nbytes=validity_nbytes,
        record_nbytes=record_nbytes,
        frames_per_chunk=frames_per_chunk,
        expected_chunks=expected_chunks,
        largest_chunk_nbytes=largest_chunk_nbytes,
        scratch_chunk_upper_nbytes=scratch_chunk_upper_nbytes,
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


def _capture_frame_source_scratch_upper_bound(
    schema: DatasetSchema,
    cell_count: int,
    max_chunk_blob_bytes: int,
) -> int:
    """Compile-time peak covering every valid dataset validity representation."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    cell_count = positive_integer(cell_count, "cell_count")
    max_chunk_blob_bytes = positive_integer(
        max_chunk_blob_bytes, "max_chunk_blob_bytes"
    )
    contract = schema.cell_schema.validity_contract
    if contract.mode is ValidityMode.COMPONENTS:
        validity_kind = "component"
        validity_axis_ids = contract.component_axis_ids
    else:
        # CellValidity is the largest remaining legal representation.
        validity_kind = "cell"
        validity_axis_ids = ()
    geometry = _capture_frame_record_geometry(
        schema,
        validity_kind,
        validity_axis_ids,
        cell_count,
        max_chunk_blob_bytes,
    )
    return _capture_frame_read_scratch_bytes(
        schema,
        geometry,
        geometry.scratch_chunk_upper_nbytes,
    )


def _cell_schedule_to_tree(
    schedule: tuple[DatasetCellAddress, ...],
) -> list[list[int]]:
    return [[cell.repeat_index, cell.point_storage_index] for cell in schedule]


def _cell_schedule_from_tree(tree: object) -> tuple[DatasetCellAddress, ...]:
    if not isinstance(tree, list):
        raise ValueError("frame-index cell_schedule must be a list")
    result = []
    for item in tree:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("frame-index cell address must contain two integers")
        result.append(
            DatasetCellAddress(
                nonnegative_integer(item[0], "repeat_index"),
                nonnegative_integer(item[1], "point_storage_index"),
            )
        )
    return tuple(result)


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


class CaptureFrameSource:
    """Repository-bound lazy access to one immutable exact frame sequence."""

    __slots__ = (
        "_schema",
        "_block_id",
        "_revision",
        "_schedule",
        "_metadata",
        "_validity_kind",
        "_validity_axis_ids",
        "_chunk_refs",
        "_store",
        "_root_lease",
        "_max_chunk_blob_bytes",
        "_cell_to_ordinal",
        "_geometry",
        "_max_read_scratch_bytes",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("CaptureFrameSource is final")

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureFrameSource is immutable")

    def __init__(
        self,
        *,
        schema: DatasetSchema,
        block_id: BlockId,
        revision: DatasetRevision,
        cell_schedule: tuple[DatasetCellAddress, ...],
        metadata_in_event_order: tuple[CameraFrameMetadata, ...],
        validity_kind: str,
        validity_axis_ids: tuple[AxisId, ...],
        chunk_refs: tuple[ContentRef, ...],
        store_authority: ContentStoreAuthority,
        root_lease: RepositoryRootLease,
        max_total_frame_bytes: int,
        max_chunk_blob_bytes: int,
    ) -> None:
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
        schedule = tuple(cell_schedule)
        metadata = tuple(metadata_in_event_order)
        refs = tuple(chunk_refs)
        if any(not isinstance(cell, DatasetCellAddress) for cell in schedule):
            raise TypeError("cell_schedule must contain DatasetCellAddress")
        if any(not isinstance(item, CameraFrameMetadata) for item in metadata):
            raise TypeError("metadata must contain CameraFrameMetadata")
        if any(not isinstance(item, ContentRef) for item in refs):
            raise TypeError("chunk_refs must contain ContentRef")
        total = schema.repeat_axis.size * schema.point_layout.storage_size
        if len(schedule) != total or len(metadata) != total:
            raise ValueError("frame source schedule/metadata do not cover the dataset")
        seen: set[DatasetCellAddress] = set()
        for cell in schedule:
            if (
                cell.repeat_index >= schema.repeat_axis.size
                or cell.point_storage_index >= schema.point_layout.storage_size
            ):
                raise ValueError("frame source schedule contains an out-of-domain cell")
            if cell in seen:
                raise ValueError("frame source schedule repeats a dataset cell")
            seen.add(cell)
        metadata_contract = CameraFrameMetadataContract()
        for ordinal, item in enumerate(metadata):
            metadata_contract.validate(item)
            if item.source_ordinal != ordinal:
                raise ValueError("frame source metadata ordinals are not contiguous")
        axis_ids = tuple(validity_axis_ids)
        geometry = _capture_frame_record_geometry(
            schema,
            validity_kind,
            axis_ids,
            total,
            max_chunk_blob_bytes,
        )
        if total * geometry.record_nbytes > positive_integer(
            max_total_frame_bytes, "max_total_frame_bytes"
        ):
            raise _FrameResourceExceeded("capture frame bytes exceed repository policy")
        if len(refs) != geometry.expected_chunks:
            raise ValueError("frame chunk count differs from canonical chunking")
        for index, reference in enumerate(refs):
            count = min(
                geometry.frames_per_chunk,
                total - index * geometry.frames_per_chunk,
            )
            if reference.size != count * geometry.record_nbytes:
                raise ValueError("frame chunk size differs from canonical record layout")
            if reference.size > max_chunk_blob_bytes:
                raise _FrameResourceExceeded("frame chunk exceeds repository policy")
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_block_id", block_id)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_schedule", schedule)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_validity_kind", validity_kind)
        object.__setattr__(self, "_validity_axis_ids", axis_ids)
        object.__setattr__(self, "_chunk_refs", refs)
        object.__setattr__(self, "_store", store_authority)
        object.__setattr__(self, "_root_lease", root_lease)
        object.__setattr__(self, "_max_chunk_blob_bytes", max_chunk_blob_bytes)
        object.__setattr__(self, "_cell_to_ordinal", {cell: index for index, cell in enumerate(schedule)})
        object.__setattr__(self, "_geometry", geometry)
        object.__setattr__(
            self,
            "_max_read_scratch_bytes",
            _capture_frame_read_scratch_bytes(
                schema,
                geometry,
                geometry.largest_chunk_nbytes,
            ),
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

    @property
    def cell_schedule(self) -> tuple[DatasetCellAddress, ...]:
        return self._schedule

    @property
    def metadata_in_event_order(self) -> tuple[CameraFrameMetadata, ...]:
        return self._metadata

    @property
    def max_read_scratch_bytes(self) -> int:
        return self._max_read_scratch_bytes

    def _sample(self, ordinal: int, payload: bytes) -> CameraSample:
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
            self._metadata[ordinal],
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
        for cell in cells:
            self._root_lease.require_active()
            if not isinstance(cell, DatasetCellAddress):
                raise TypeError("iter_cells requires DatasetCellAddress values")
            try:
                ordinal = self._cell_to_ordinal[cell]
            except KeyError as exc:
                raise KeyError("cell is outside this capture frame source") from exc
            chunk = ordinal // self._geometry.frames_per_chunk
            if chunk != active_chunk:
                payload = None
                payload = self._store.read_blob(
                    self._chunk_refs[chunk],
                    max_bytes=self._max_chunk_blob_bytes,
                )
                active_chunk = chunk
            assert payload is not None
            yield cell, self._sample(ordinal, payload)

    def iter_event_order(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraSample]]:
        return self.iter_cells(self._schedule)

    def materialize(self, *, memory_limit_bytes: int) -> DataBlock:
        self._root_lease.require_active()
        limit = positive_integer(memory_limit_bytes, "memory_limit_bytes")
        cell_count = len(self._schedule)
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

    def _verify_all_chunks(self) -> None:
        self._root_lease.require_active()
        for reference in self._chunk_refs:
            self._root_lease.require_active()
            self._store.verify_blob(reference, max_bytes=self._max_chunk_blob_bytes)


def _frame_index_tree(source: CaptureFrameSource) -> dict[str, object]:
    return {
        "schema": _FRAME_INDEX_SCHEMA,
        "block_id": source.block_id.value,
        "revision": source.revision.value,
        "dataset_schema": dataset_schema_to_tree(source.schema),
        "validity": {
            "kind": source._validity_kind,
            "axis_ids": [axis_id.value for axis_id in source._validity_axis_ids],
        },
        "cell_schedule": _cell_schedule_to_tree(source.cell_schedule),
        "event_metadata": [
            camera_frame_metadata_to_tree(item)
            for item in source.metadata_in_event_order
        ],
        "frame_chunks": [content_ref_to_tree(item) for item in source._chunk_refs],
    }


def _stage_capture_frame_source(
    *,
    block: DataBlock,
    event_metadata: tuple[CameraFrameMetadata, ...],
    cell_schedule: tuple[DatasetCellAddress, ...],
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
    schedule, metadata = tuple(cell_schedule), tuple(event_metadata)
    max_cells = positive_integer(max_cells, "max_cells")
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
    source = CaptureFrameSource(
        schema=block.schema,
        block_id=block.block_id,
        revision=block.revision,
        cell_schedule=schedule,
        metadata_in_event_order=metadata,
        validity_kind=validity_kind,
        validity_axis_ids=validity_axis_ids,
        chunk_refs=tuple(chunks),
        store_authority=store_authority,
        root_lease=root_lease,
        max_total_frame_bytes=max_total_frame_bytes,
        max_chunk_blob_bytes=max_chunk_blob_bytes,
    )
    index_payload = encode(_frame_index_tree(source))
    # The encoded index is not publishable merely because it fits in its blob
    # budget.  Reconstruct it through the exact reader boundary before the
    # manifest can name it, so canonical structure limits and every typed
    # invariant have one owner on both sides of persistence.
    try:
        admitted_source = _capture_frame_source_from_payload(
            index_payload,
            store_authority=store_authority,
            root_lease=root_lease,
            max_cells=max_cells,
            max_total_frame_bytes=max_total_frame_bytes,
            max_chunk_blob_bytes=max_chunk_blob_bytes,
            max_frame_index_blob_bytes=max_frame_index_blob_bytes,
            max_canonical_nodes=max_canonical_nodes,
            max_canonical_container_entries=max_canonical_container_entries,
        )
    except CanonicalEncodingError as exc:
        # encode() produced canonical bytes, so a decoder rejection here is a
        # reader resource-limit rejection rather than malformed external input.
        raise _FrameResourceExceeded(
            "capture frame index exceeds canonical structure policy"
        ) from exc
    return admitted_source, store_authority.put_blob(index_payload)


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
    if reference.size > max_frame_index_blob_bytes:
        raise _FrameResourceExceeded("capture frame index exceeds repository policy")
    payload = store_authority.read_blob(reference, max_bytes=max_frame_index_blob_bytes)
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
    if len(payload) > positive_integer(
        max_frame_index_blob_bytes, "max_frame_index_blob_bytes"
    ):
        raise _FrameResourceExceeded("capture frame index exceeds repository policy")

    def admit(events) -> None:
        for event in events:
            if isinstance(event, CanonicalArrayEvent):
                raise _FrameResourceExceeded("capture frame index cannot embed ndarrays")
            if (
                isinstance(event, CanonicalListEvent)
                and event.path in {
                    ("cell_schedule",),
                    ("event_metadata",),
                    ("frame_chunks",),
                }
                and event.length > max_cells
            ):
                raise _FrameResourceExceeded("capture frame index exceeds cell policy")

    tree = exact_mapping(
        decode(
            payload,
            admit_structure=admit,
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
            "dataset_schema",
            "validity",
            "cell_schedule",
            "event_metadata",
            "frame_chunks",
        },
        _FRAME_INDEX_SCHEMA,
    )
    validity = exact_mapping(
        tree["validity"], {"kind", "axis_ids"}, "frame validity", discriminator=None
    )
    axis_ids = validity["axis_ids"]
    if not isinstance(axis_ids, list):
        raise ValueError("frame validity axis_ids must be a list")
    schedule = _cell_schedule_from_tree(tree["cell_schedule"])
    metadata_tree = tree["event_metadata"]
    chunk_tree = tree["frame_chunks"]
    if not isinstance(metadata_tree, list) or not isinstance(chunk_tree, list):
        raise ValueError("frame index metadata and chunks must be lists")
    if len(schedule) > max_cells or len(metadata_tree) > max_cells:
        raise _FrameResourceExceeded("capture frame count exceeds repository policy")
    source = CaptureFrameSource(
        schema=dataset_schema_from_tree(tree["dataset_schema"]),
        block_id=BlockId(tree["block_id"]),
        revision=DatasetRevision(nonnegative_integer(tree["revision"], "revision")),
        cell_schedule=schedule,
        metadata_in_event_order=tuple(
            camera_frame_metadata_from_tree(item) for item in metadata_tree
        ),
        validity_kind=validity["kind"],
        validity_axis_ids=tuple(AxisId(item) for item in axis_ids),
        chunk_refs=tuple(content_ref_from_tree(item) for item in chunk_tree),
        store_authority=store_authority,
        root_lease=root_lease,
        max_total_frame_bytes=max_total_frame_bytes,
        max_chunk_blob_bytes=max_chunk_blob_bytes,
    )
    if encode(_frame_index_tree(source)) != payload:
        raise ValueError("capture frame index is not canonical")
    return source


__all__ = [
    "CaptureFrameSource",
]
