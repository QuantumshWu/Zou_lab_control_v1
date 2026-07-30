"""Direct, lazy frame storage for one durable camera capture."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

import numpy as np

from zlc_data import (
    INVALID,
    VALID,
    AxisId,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    StreamGenerationId,
    Valid,
    Value,
)
from zlc_data.codec import dataset_schema_from_tree, dataset_schema_to_tree
from zlc_neutral_atom.devices.camera.contract import (
    CameraFrameMetadata,
    CameraFrameMetadataContract,
    CameraSample,
    camera_frame_metadata_from_tree,
    camera_frame_metadata_to_tree,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress, DatasetCellSchedule
from zlc_storage.durability import atomic_write_file
from zlc_storage.paths import resolve_under


_FRAME_SOURCE_SCHEMA = "zlc_neutral_atom.capture-frame-source"
_VALUES_FILE = "frames.npy"
_VALIDITY_FILE = "validity.npy"
_VALIDITY_KINDS = frozenset({"valid", "invalid", "cell", "component"})


def _cell_to_tree(cell: DatasetCellAddress) -> list[int]:
    if not isinstance(cell, DatasetCellAddress):
        raise TypeError("cell must be DatasetCellAddress")
    return [cell.repeat_index, cell.point_ordinal]


def _cell_from_tree(tree: object) -> DatasetCellAddress:
    if (
        not isinstance(tree, list)
        or len(tree) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in tree)
    ):
        raise ValueError("capture cell must contain two integers")
    if tree[0] < 0 or tree[1] < 0:
        raise ValueError("capture cell indices must be nonnegative")
    return DatasetCellAddress(tree[0], tree[1])


def _relative_leaf(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty str")
    path = Path(value)
    if path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError(f"{field} must be one relative file name")
    return value


def _validity_descriptor(
    validity: object,
) -> tuple[str, tuple[AxisId, ...], np.ndarray | None]:
    if isinstance(validity, Valid):
        return "valid", (), None
    if isinstance(validity, Invalid):
        return "invalid", (), None
    if isinstance(validity, CellValidity):
        return "cell", (), validity.mask
    if isinstance(validity, DatasetComponentValidity):
        return "component", validity.axis_ids, validity.mask
    raise TypeError("capture DataBlock has an unsupported validity type")


def _write_npy(path: Path, array: np.ndarray) -> None:
    source = np.asarray(array)
    atomic_write_file(
        path,
        lambda stream: np.save(stream, source, allow_pickle=False),
    )


def _close_memmap(array: np.ndarray | None) -> None:
    if isinstance(array, np.memmap):
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def _inspect_npy(
    path: Path,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    field: str,
) -> None:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if array.shape != shape:
            raise ValueError(f"{field} shape differs from capture record")
        if array.dtype != dtype:
            raise TypeError(f"{field} dtype differs from capture record")
    finally:
        _close_memmap(array)


def _validate_events(
    schema: DatasetSchema,
    schedule: DatasetCellSchedule,
    metadata: tuple[CameraFrameMetadata, ...],
) -> tuple[int, ...]:
    total = schema.repeat_axis.size * schema.point_table.row_count
    schedule.validate_schema(schema)
    if len(schedule) != total or len(metadata) != total:
        raise ValueError("capture schedule and metadata must cover the Dataset")
    contract = CameraFrameMetadataContract()
    inverse = [-1] * total
    point_count = schema.point_table.row_count
    for ordinal, (cell, item) in enumerate(zip(schedule, metadata, strict=True)):
        contract.validate(item)
        if item.source_ordinal != ordinal:
            raise ValueError("capture metadata source ordinals must be contiguous")
        linear = cell.repeat_index * point_count + cell.point_ordinal
        if inverse[linear] != -1:
            raise ValueError("capture schedule repeats a Dataset cell")
        inverse[linear] = ordinal
    if any(value < 0 for value in inverse):
        raise ValueError("capture schedule omits a Dataset cell")
    return tuple(inverse)


class CaptureFrameSource:
    """Lazy access to arrays owned by one visible ``capture.json`` record."""

    __slots__ = (
        "_capture_dir",
        "_schema",
        "_block_id",
        "_revision",
        "_cell_schedule",
        "_metadata",
        "_ordinal_by_cell",
        "_values_path",
        "_validity_kind",
        "_validity_axis_ids",
        "_validity_path",
    )

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("CaptureFrameSource is immutable")

    def __init__(
        self,
        *,
        capture_dir: Path,
        schema: DatasetSchema,
        block_id: BlockId,
        revision: DatasetRevision,
        cell_schedule: DatasetCellSchedule,
        metadata: tuple[CameraFrameMetadata, ...],
        values_file: str,
        validity_kind: str,
        validity_axis_ids: tuple[AxisId, ...],
        validity_file: str | None,
    ) -> None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if not isinstance(block_id, BlockId) or not isinstance(revision, DatasetRevision):
            raise TypeError("frame source requires BlockId and DatasetRevision")
        directory = Path(capture_dir).expanduser()
        if not directory.is_absolute():
            raise ValueError("capture_dir must be absolute")
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        values_name = _relative_leaf(values_file, "values_file")
        values_path = resolve_under(directory, values_name)
        kind = str(validity_kind)
        if kind not in _VALIDITY_KINDS:
            raise ValueError("capture validity kind is unknown")
        axis_ids = tuple(validity_axis_ids)
        if any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
            raise TypeError("validity_axis_ids must contain AxisId values")
        if kind == "component":
            if axis_ids != schema.cell_schema.validity_contract.component_axis_ids:
                raise ValueError("component validity axes differ from DatasetSchema")
        elif axis_ids:
            raise ValueError("only component validity may name axes")
        if kind in {"cell", "component"}:
            if validity_file is None:
                raise ValueError("capture validity mask file is missing")
            validity_name = _relative_leaf(validity_file, "validity_file")
            validity_path = resolve_under(directory, validity_name)
        else:
            if validity_file is not None:
                raise ValueError("uniform validity cannot name a mask file")
            validity_path = None
        events = tuple(metadata)
        inverse = _validate_events(schema, cell_schedule, events)
        _inspect_npy(
            values_path,
            shape=schema.physical_shape,
            dtype=schema.cell_schema.dtype,
            field="capture values",
        )
        if validity_path is not None:
            shape = (
                schema.physical_shape[:2]
                if kind == "cell"
                else (
                    *schema.physical_shape[:2],
                    *(
                        schema.cell_schema.axis(axis_id).size
                        for axis_id in axis_ids
                    ),
                )
            )
            _inspect_npy(
                validity_path,
                shape=tuple(shape),
                dtype=np.dtype(bool),
                field="capture validity",
            )
        object.__setattr__(self, "_capture_dir", directory)
        object.__setattr__(self, "_schema", schema)
        object.__setattr__(self, "_block_id", block_id)
        object.__setattr__(self, "_revision", revision)
        object.__setattr__(self, "_cell_schedule", cell_schedule)
        object.__setattr__(self, "_metadata", events)
        object.__setattr__(self, "_ordinal_by_cell", inverse)
        object.__setattr__(self, "_values_path", values_path)
        object.__setattr__(self, "_validity_kind", kind)
        object.__setattr__(self, "_validity_axis_ids", axis_ids)
        object.__setattr__(self, "_validity_path", validity_path)

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
    def event_count(self) -> int:
        return len(self._metadata)

    @property
    def cell_schedule(self) -> DatasetCellSchedule:
        return self._cell_schedule

    def ref(self, generation: StreamGenerationId) -> DatasetRevisionRef:
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        return DatasetRevisionRef(
            self._block_id,
            generation,
            self._schema.fingerprint,
            self._revision,
        )

    def iter_event_records(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraFrameMetadata]]:
        yield from zip(self._cell_schedule, self._metadata, strict=True)

    def iter_cell_schedule(self) -> Iterator[DatasetCellAddress]:
        yield from self._cell_schedule

    def _cell_validity(
        self,
        mask: np.ndarray | None,
        cell: DatasetCellAddress,
    ) -> Valid | Invalid | ComponentValidity:
        if self._validity_kind == "valid":
            return VALID
        if self._validity_kind == "invalid":
            return INVALID
        assert mask is not None
        location = (cell.repeat_index, cell.point_ordinal)
        if self._validity_kind == "cell":
            return VALID if bool(mask[location]) else INVALID
        return ComponentValidity(self._validity_axis_ids, mask[location])

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
        values = np.load(self._values_path, mmap_mode="r", allow_pickle=False)
        mask = (
            None
            if self._validity_path is None
            else np.load(self._validity_path, mmap_mode="r", allow_pickle=False)
        )
        try:
            point_count = self._schema.point_table.row_count
            for cell in cells:
                if not isinstance(cell, DatasetCellAddress):
                    raise TypeError("iter_cells requires DatasetCellAddress values")
                if (
                    cell.repeat_index >= self._schema.repeat_axis.size
                    or cell.point_ordinal >= point_count
                ):
                    raise KeyError("cell is outside this capture frame source")
                linear = cell.repeat_index * point_count + cell.point_ordinal
                ordinal = self._ordinal_by_cell[linear]
                yield cell, CameraSample(
                    Value(
                        values[cell.repeat_index, cell.point_ordinal],
                        self._cell_validity(mask, cell),
                        self._schema.cell_schema,
                    ),
                    self._metadata[ordinal],
                )
        finally:
            _close_memmap(mask)
            _close_memmap(values)

    def iter_event_order(
        self,
    ) -> Iterator[tuple[DatasetCellAddress, CameraSample]]:
        yield from self.iter_cells(self._cell_schedule)

    def materialize(
        self,
        *,
        abort_check: Callable[[], None] | None = None,
    ) -> DataBlock:
        if abort_check is not None and not callable(abort_check):
            raise TypeError("abort_check must be callable or None")
        if abort_check is not None:
            abort_check()
        values = np.load(self._values_path, allow_pickle=False)
        mask = (
            None
            if self._validity_path is None
            else np.load(self._validity_path, allow_pickle=False)
        )
        if abort_check is not None:
            abort_check()
        if self._validity_kind == "valid":
            validity = VALID
        elif self._validity_kind == "invalid":
            validity = INVALID
        elif self._validity_kind == "cell":
            assert mask is not None
            validity = CellValidity(mask)
        else:
            assert mask is not None
            validity = DatasetComponentValidity(self._validity_axis_ids, mask)
        block = DataBlock(
            self._block_id,
            self._revision,
            values,
            validity,
            self._schema,
        )
        if abort_check is not None:
            abort_check()
        return block


def _write_capture_frame_source(
    capture_dir: Path,
    *,
    block: DataBlock,
    event_metadata: tuple[CameraFrameMetadata, ...],
    cell_schedule: DatasetCellSchedule,
) -> tuple[CaptureFrameSource, dict[str, object]]:
    """Write raw arrays first and return their record subtree."""

    if not isinstance(block, DataBlock):
        raise TypeError("block must be DataBlock")
    directory = Path(capture_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    metadata = tuple(event_metadata)
    _validate_events(block.schema, cell_schedule, metadata)
    kind, axis_ids, mask = _validity_descriptor(block.validity)
    _write_npy(resolve_under(directory, _VALUES_FILE), block.values)
    validity_file = None
    if mask is not None:
        _write_npy(resolve_under(directory, _VALIDITY_FILE), mask)
        validity_file = _VALIDITY_FILE
    tree: dict[str, object] = {
        "schema": _FRAME_SOURCE_SCHEMA,
        "dataset_schema": dataset_schema_to_tree(block.schema),
        "block_id": block.block_id.value,
        "revision": block.revision.value,
        "values_file": _VALUES_FILE,
        "validity": {
            "kind": kind,
            "axis_ids": [axis_id.value for axis_id in axis_ids],
            "file": validity_file,
        },
        "events": [
            {
                "cell": _cell_to_tree(cell),
                "metadata": camera_frame_metadata_to_tree(item),
            }
            for cell, item in zip(cell_schedule, metadata, strict=True)
        ],
    }
    return _capture_frame_source_from_tree(directory, tree), tree


def _capture_frame_source_from_tree(
    capture_dir: Path,
    tree: object,
) -> CaptureFrameSource:
    fields = {
        "schema",
        "dataset_schema",
        "block_id",
        "revision",
        "values_file",
        "validity",
        "events",
    }
    if not isinstance(tree, dict) or set(tree) != fields:
        raise ValueError("capture frame source has an unknown field set")
    if tree["schema"] != _FRAME_SOURCE_SCHEMA:
        raise ValueError("capture frame source schema is not current")
    schema = dataset_schema_from_tree(tree["dataset_schema"])
    revision = tree["revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("capture revision must be a nonnegative integer")
    validity = tree["validity"]
    if not isinstance(validity, dict) or set(validity) != {"kind", "axis_ids", "file"}:
        raise ValueError("capture validity has an unknown field set")
    axis_values = validity["axis_ids"]
    if not isinstance(axis_values, list):
        raise TypeError("capture validity axis_ids must be a list")
    events = tree["events"]
    if not isinstance(events, list):
        raise TypeError("capture events must be a list")
    cells: list[DatasetCellAddress] = []
    metadata: list[CameraFrameMetadata] = []
    for event in events:
        if not isinstance(event, dict) or set(event) != {"cell", "metadata"}:
            raise ValueError("capture event has an unknown field set")
        cells.append(_cell_from_tree(event["cell"]))
        metadata.append(camera_frame_metadata_from_tree(event["metadata"]))
    schedule = DatasetCellSchedule.from_cells(schema, cells)
    return CaptureFrameSource(
        capture_dir=Path(capture_dir),
        schema=schema,
        block_id=BlockId(tree["block_id"]),
        revision=DatasetRevision(revision),
        cell_schedule=schedule,
        metadata=tuple(metadata),
        values_file=tree["values_file"],
        validity_kind=validity["kind"],
        validity_axis_ids=tuple(AxisId(value) for value in axis_values),
        validity_file=validity["file"],
    )


__all__ = ["CaptureFrameSource"]
