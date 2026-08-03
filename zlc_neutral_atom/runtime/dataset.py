"""Single-owner event-to-dataset materialization with revisioned snapshots."""

from __future__ import annotations

import math
import struct
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from numbers import Integral
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
from zlc_storage import (
    exact_mapping as _exact_mapping,
    nonnegative_integer as _nonnegative_integer,
)

from zlc_data import (
    BlockId,
    AxisSpec,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetComponentValidity,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    REPEAT,
    Valid,
    ValidityMode,
    Value,
    ValueSchema,
)
from zlc_data.value import expand_component_validity

from ._failure import record_secondary_failure

from .streams import (
    AcquisitionStream,
    Delivery,
    EndOfStream,
    Envelope,
    EventRef,
    EventSpanRef,
    event_span_ref_from_tree,
    event_span_ref_to_tree,
    ExactConsumerReadiness,
    ExactReservation,
    MonitorTap,
    MonitorUpdate,
    ReservationState,
    StreamId,
)


PayloadT = TypeVar("PayloadT")
_DATASET_SEAL_PROVENANCE_SCHEMA = "zlc_neutral_atom.DatasetSealProvenance"


class DatasetEventAdapter(Protocol[PayloadT]):
    """Typed projection from one immutable stream payload into one dataset cell."""

    payload_contract: object
    value_schema: ValueSchema
    metadata_contract: "DatasetMetadataContract[PayloadT]"

    def value(self, payload: PayloadT) -> Value: ...



class DatasetMetadataContract(Protocol[PayloadT]):
    def snapshot(self, payload: PayloadT) -> object | None: ...

    def validate(self, metadata: object | None) -> None: ...


class DatasetError(RuntimeError):
    pass


class MissingDatasetCells(DatasetError):
    pass


class SnapshotExpired(DatasetError):
    pass


_SEALED_TOKEN = object()


def _intrinsically_bytes_backed_array(value: object) -> bool:
    if (
        type(value) is not np.ndarray
        or value.dtype.hasobject
        or value.dtype.fields is not None
        or value.flags.writeable
        or not value.flags.c_contiguous
    ):
        return False
    current: object = value
    seen: set[int] = set()
    while isinstance(current, np.ndarray):
        identity = id(current)
        if identity in seen or current.flags.writeable or current.flags.owndata:
            return False
        seen.add(identity)
        current = current.base
    if isinstance(current, bytes):
        return True
    return isinstance(current, memoryview) and current.readonly and isinstance(
        current.obj,
        bytes,
    )


def _is_deeply_immutable(
    value: object,
    active: set[int] | None = None,
    validated: set[int] | None = None,
) -> bool:
    """Accept immutable runtime snapshots, including bytes-owned numeric values."""

    if value is None or type(value) in (bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is Value:
        return True
    if isinstance(value, np.dtype):
        return True
    if isinstance(value, np.ndarray):
        return _intrinsically_bytes_backed_array(value)
    identity = id(value)
    active = set() if active is None else active
    validated = set() if validated is None else validated
    if identity in active:
        return False
    if identity in validated:
        return True
    active.add(identity)
    if isinstance(value, Enum):
        result = _is_deeply_immutable(value.value, active, validated)
    elif isinstance(value, (tuple, frozenset)):
        result = all(
            _is_deeply_immutable(item, active, validated) for item in value
        )
    elif is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        result = bool(parameters and parameters.frozen) and all(
            _is_deeply_immutable(
                getattr(value, field.name),
                active,
                validated,
            )
            for field in fields(value)
        )
    else:
        result = False
    active.remove(identity)
    if result:
        validated.add(identity)
    return result


@dataclass(frozen=True, order=True, slots=True)
class DatasetCellAddress:
    repeat_index: int
    point_ordinal: int

    def __post_init__(self) -> None:
        for field in ("repeat_index", "point_ordinal"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")
            object.__setattr__(self, field, int(value))


def _validated_cell_permutation(
    schema: DatasetSchema,
    cells: Iterable[DatasetCellAddress],
) -> Iterator[tuple[int, DatasetCellAddress, int]]:
    """Yield one complete typed permutation while holding only a one-byte seen map."""

    total = schema.repeat_axis.size * schema.point_table.row_count
    point_count = schema.point_table.row_count
    seen = bytearray(total)
    count = 0
    for cell in cells:
        if not isinstance(cell, DatasetCellAddress):
            raise TypeError("cell permutation must contain DatasetCellAddress values")
        if count >= total:
            raise ValueError("cell permutation length differs from DatasetSchema")
        if (
            cell.repeat_index >= schema.repeat_axis.size
            or cell.point_ordinal >= point_count
        ):
            raise ValueError("cell permutation contains an out-of-domain cell")
        linear = cell.repeat_index * point_count + cell.point_ordinal
        if seen[linear]:
            raise ValueError("cell permutation repeats a dataset cell")
        seen[linear] = 1
        yield count, cell, linear
        count += 1
    if count != total:
        raise ValueError("cell permutation length differs from DatasetSchema")


@dataclass(frozen=True)
class DatasetCellKeyContract:
    """Join-key domain containing repeat and point rows, never cell values."""

    repeat_axis: AxisSpec
    point_table: PointTable

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must be an AxisSpec with role 'repeat'")
        if not isinstance(self.point_table, PointTable):
            raise TypeError("point_table must be PointTable")
        if self.repeat_axis.axis_id in {
            column.coordinate_id for column in self.point_table.columns
        }:
            raise ValueError("repeat and point coordinate ids must be distinct")

    @classmethod
    def from_schema(cls, schema: DatasetSchema) -> "DatasetCellKeyContract":
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        return cls(schema.repeat_axis, schema.point_table)

    def snapshot(self, key: object) -> DatasetCellAddress:
        if not isinstance(key, DatasetCellAddress):
            raise TypeError("join key must be DatasetCellAddress")
        if key.repeat_index >= self.repeat_axis.size:
            raise IndexError("join key repeat index is outside DatasetSchema")
        if key.point_ordinal >= self.point_table.row_count:
            raise IndexError("join key point ordinal is outside PointTable")
        return DatasetCellAddress(key.repeat_index, key.point_ordinal)


_DATASET_CELL_SCHEDULE_TOKEN = object()


class DatasetCellSchedule:
    """Immutable packed owner of one complete ordinal-to-cell permutation."""

    __slots__ = (
        "_cell_count",
        "_integer_width",
        "_key_contract",
        "_packed",
        "_point_count",
        "_repeat_count",
    )

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("DatasetCellSchedule is sealed")

    def __init__(
        self,
        token: object,
        *,
        key_contract: DatasetCellKeyContract,
        repeat_count: int,
        point_count: int,
        integer_width: int,
        packed: bytes,
    ) -> None:
        if token is not _DATASET_CELL_SCHEDULE_TOKEN:
            raise TypeError("DatasetCellSchedule must be built with from_cells()")
        if not isinstance(key_contract, DatasetCellKeyContract):
            raise TypeError("key_contract must be DatasetCellKeyContract")
        object.__setattr__(self, "_key_contract", key_contract)
        object.__setattr__(self, "_repeat_count", repeat_count)
        object.__setattr__(self, "_point_count", point_count)
        object.__setattr__(self, "_cell_count", repeat_count * point_count)
        object.__setattr__(self, "_integer_width", integer_width)
        object.__setattr__(self, "_packed", packed)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("DatasetCellSchedule is immutable")

    @classmethod
    def from_cells(
        cls,
        schema: DatasetSchema,
        cells: Iterable[DatasetCellAddress],
    ) -> "DatasetCellSchedule":
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        key_contract = DatasetCellKeyContract.from_schema(schema)
        repeat_count = schema.repeat_axis.size
        point_count = schema.point_table.row_count
        total = repeat_count * point_count
        if total <= 0x1_0000_0000:
            integer_width = 4
        elif total <= 0x1_0000_0000_0000_0000:
            integer_width = 8
        else:
            raise DatasetError("cell schedule exceeds the packed u64 range")
        pack_format = "<I" if integer_width == 4 else "<Q"
        packed = bytearray(total * integer_width)
        for ordinal, cell, linear in _validated_cell_permutation(schema, cells):
            struct.pack_into(pack_format, packed, ordinal * integer_width, linear)
        return cls(
            _DATASET_CELL_SCHEDULE_TOKEN,
            key_contract=key_contract,
            repeat_count=repeat_count,
            point_count=point_count,
            integer_width=integer_width,
            packed=bytes(packed),
        )

    @property
    def key_contract(self) -> DatasetCellKeyContract:
        return self._key_contract

    def __len__(self) -> int:
        return self._cell_count

    def __iter__(self) -> Iterator[DatasetCellAddress]:
        unpack_format = "<I" if self._integer_width == 4 else "<Q"
        for offset in range(0, len(self._packed), self._integer_width):
            linear = struct.unpack_from(unpack_format, self._packed, offset)[0]
            repeat, point = divmod(linear, self._point_count)
            yield DatasetCellAddress(repeat, point)

    def cell_at(self, ordinal: int) -> DatasetCellAddress:
        if isinstance(ordinal, bool) or not isinstance(ordinal, Integral):
            raise TypeError("ordinal must be an integer")
        ordinal = int(ordinal)
        if ordinal < 0 or ordinal >= self._cell_count:
            raise IndexError("cell ordinal is outside DatasetCellSchedule")
        unpack_format = "<I" if self._integer_width == 4 else "<Q"
        linear = struct.unpack_from(
            unpack_format,
            self._packed,
            ordinal * self._integer_width,
        )[0]
        repeat, point = divmod(linear, self._point_count)
        return DatasetCellAddress(repeat, point)

    def validate_schema(self, schema: DatasetSchema) -> None:
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        if (
            DatasetCellKeyContract.from_schema(schema) != self._key_contract
            or schema.repeat_axis.size != self._repeat_count
            or schema.point_table.row_count != self._point_count
        ):
            raise DatasetError("cell schedule belongs to a different DatasetSchema")

    def same_order_as(self, other: object) -> bool:
        return (
            isinstance(other, DatasetCellSchedule)
            and self._key_contract == other._key_contract
            and self._cell_count == other._cell_count
            and self._integer_width == other._integer_width
            and self._packed == other._packed
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        return self.same_order_as(other)

    def __repr__(self) -> str:
        return (
            "DatasetCellSchedule("
            f"cells={self._cell_count}, packed_nbytes={len(self._packed)})"
        )


@dataclass(frozen=True)
class FrozenDatasetEdge(Generic[PayloadT]):
    """Single owner for a payload-to-dataset projection and exact schedule.

    ``DatasetSchema``, value projection, metadata projection, and the optional
    exact ordinal schedule are one structural contract.
    """

    schema: DatasetSchema
    event_adapter: DatasetEventAdapter[PayloadT]
    cell_schedule: DatasetCellSchedule | None = None
    _payload_contract: object = field(init=False, repr=False, compare=False)
    _metadata_contract: DatasetMetadataContract = field(
        init=False,
        repr=False,
        compare=False,
    )
    _value_schema: ValueSchema = field(init=False, repr=False, compare=False)
    _key_contract: DatasetCellKeyContract = field(
        init=False,
        repr=False,
        compare=False,
    )
    _value_operator: Callable[[PayloadT], Value] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        schema = self.schema
        adapter = self.event_adapter
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        parameters = getattr(type(adapter), "__dataclass_params__", None)
        if not is_dataclass(adapter) or not parameters or not parameters.frozen:
            raise TypeError("DatasetEventAdapter must be a frozen dataclass value")
        try:
            payload_contract = adapter.payload_contract
            value_schema = adapter.value_schema
            metadata = adapter.metadata_contract
            value_operator = adapter.value
        except AttributeError as error:
            raise TypeError(
                "event_adapter does not implement DatasetEventAdapter"
            ) from error
        if not callable(value_operator):
            raise TypeError("event_adapter.value must be callable")
        if value_schema is not schema.cell_schema:
            raise DatasetError("DatasetEventAdapter must share DatasetSchema.cell_schema")
        if not _is_deeply_immutable(adapter):
            raise TypeError(
                "DatasetEventAdapter fields must contain only intrinsically immutable values"
            )
        for name, owner in (
            ("payload contract", payload_contract),
            ("metadata contract", metadata),
        ):
            owner_parameters = getattr(type(owner), "__dataclass_params__", None)
            if (
                not is_dataclass(owner)
                or not owner_parameters
                or not owner_parameters.frozen
            ):
                raise TypeError(f"{name} must be a frozen dataclass value")
            if not _is_deeply_immutable(owner):
                raise TypeError(
                    f"{name} fields must contain only intrinsically immutable values"
                )
        for member in ("snapshot", "validate"):
            if not callable(getattr(payload_contract, member, None)):
                raise TypeError(f"event_adapter.payload_contract.{member} must be callable")
        for member in ("snapshot", "validate"):
            if not callable(getattr(metadata, member, None)):
                raise TypeError(f"metadata_contract.{member} must be callable")
        object.__setattr__(self, "_payload_contract", payload_contract)
        object.__setattr__(self, "_metadata_contract", metadata)
        object.__setattr__(self, "_value_schema", value_schema)
        object.__setattr__(self, "_value_operator", value_operator)
        schedule = self.cell_schedule
        key_contract = DatasetCellKeyContract.from_schema(schema)
        if schedule is None:
            pass
        else:
            if not isinstance(schedule, DatasetCellSchedule):
                raise TypeError("cell_schedule must be DatasetCellSchedule or None")
            schedule.validate_schema(schema)
            key_contract = schedule.key_contract
        object.__setattr__(
            self,
            "_key_contract",
            key_contract,
        )

    @property
    def payload_contract(self) -> object:
        return self._payload_contract

    @property
    def metadata_contract(self) -> DatasetMetadataContract:
        return self._metadata_contract

    @property
    def value_schema(self) -> ValueSchema:
        return self._value_schema

    def project_value(self, payload: PayloadT) -> Value:
        return self._value_operator(payload)

    @property
    def key_contract(self) -> DatasetCellKeyContract:
        return self._key_contract

    def validate_payload_stream(self, stream: AcquisitionStream[PayloadT]) -> None:
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("stream must be AcquisitionStream")
        if stream._payload_contract is not self.payload_contract:
            raise DatasetError("dataset edge must share the stream PayloadContract owner")

    def validate_stream(self, stream: AcquisitionStream[PayloadT]) -> None:
        self.validate_payload_stream(stream)
        key_contract = stream._join_key_contract
        if not isinstance(key_contract, DatasetCellKeyContract):
            raise DatasetError("dataset source must declare DatasetCellKeyContract")
        if key_contract != self.key_contract:
            raise DatasetError("dataset source join-key contract differs from edge schema")


@dataclass(frozen=True)
class DatasetCoverage:
    written_cells: int
    total_cells: int

    def __post_init__(self) -> None:
        _validate_cell_counts(self)

    @property
    def complete(self) -> bool:
        return self.written_cells == self.total_cells


@dataclass(frozen=True)
class MonitorCoverage:
    """Visible-window completeness plus lifetime monitor loss telemetry."""

    written_cells: int
    total_cells: int
    missed_events: int
    current_gap: bool

    def __post_init__(self) -> None:
        _validate_cell_counts(self)
        if (
            isinstance(self.missed_events, bool)
            or not isinstance(self.missed_events, Integral)
            or self.missed_events < 0
        ):
            raise ValueError("missed_events must be a non-negative integer")
        object.__setattr__(self, "missed_events", int(self.missed_events))
        if not isinstance(self.current_gap, bool):
            raise TypeError("current_gap must be bool")

    @property
    def complete(self) -> bool:
        return self.written_cells == self.total_cells and not self.current_gap


def _validate_cell_counts(coverage: DatasetCoverage | MonitorCoverage) -> None:
    for name in ("written_cells", "total_cells"):
        value = getattr(coverage, name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        object.__setattr__(coverage, name, int(value))
    if coverage.written_cells > coverage.total_cells:
        raise ValueError("written_cells cannot exceed total_cells")


@dataclass(frozen=True)
class DatasetPreviewSnapshot:
    """Provisional materialization for display; never a formal storage input."""

    snapshot: OwnedSnapshot
    coverage: DatasetCoverage
    cell_metadata: tuple[object | None, ...]

    @property
    def ref(self) -> DatasetRevisionRef:
        return self.snapshot.ref

    @property
    def block(self) -> DataBlock:
        return self.snapshot.block


@dataclass(frozen=True, slots=True)
class DatasetPreviewCell:
    """One immutable exact cell in its frozen commit order."""

    ordinal: int
    address: DatasetCellAddress
    value: Value
    metadata: object | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, Integral)
            or self.ordinal < 0
        ):
            raise ValueError("ordinal must be a non-negative integer")
        object.__setattr__(self, "ordinal", int(self.ordinal))
        if not isinstance(self.address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress")
        if not isinstance(self.value, Value):
            raise TypeError("value must be Value")
        if not _is_deeply_immutable(self.metadata):
            raise TypeError("metadata must be a deeply immutable snapshot")


@dataclass(frozen=True, slots=True)
class DatasetPreviewDelta:
    """New exact cells committed after one caller-owned revision cursor."""

    after: DatasetRevision
    ref: DatasetRevisionRef
    cells: tuple[DatasetPreviewCell, ...]
    coverage: DatasetCoverage

    def __post_init__(self) -> None:
        if not isinstance(self.after, DatasetRevision):
            raise TypeError("after must be DatasetRevision")
        if not isinstance(self.ref, DatasetRevisionRef):
            raise TypeError("ref must be DatasetRevisionRef")
        if not isinstance(self.coverage, DatasetCoverage):
            raise TypeError("coverage must be DatasetCoverage")
        cells = tuple(self.cells)
        if any(not isinstance(cell, DatasetPreviewCell) for cell in cells):
            raise TypeError("cells must contain DatasetPreviewCell values")
        start = self.after.value
        stop = self.ref.revision.value
        if stop < start:
            raise ValueError("delta ref cannot precede its revision cursor")
        if len(cells) != stop - start or any(
            cell.ordinal != start + offset
            for offset, cell in enumerate(cells)
        ):
            raise ValueError(
                "delta cells must exactly cover the committed revision interval"
            )
        if self.coverage.written_cells != stop:
            raise ValueError("delta revision differs from exact dataset coverage")
        object.__setattr__(self, "cells", cells)


@dataclass(frozen=True)
class MonitorDatasetSnapshot:
    """One atomically frozen live view with its aligned event identities."""

    snapshot: OwnedSnapshot
    coverage: MonitorCoverage
    cell_metadata: tuple[object | None, ...]
    event_refs: tuple[EventRef | None, ...]
    head: EventRef | None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, OwnedSnapshot):
            raise TypeError("snapshot must be OwnedSnapshot")
        if not isinstance(self.coverage, MonitorCoverage):
            raise TypeError("coverage must be MonitorCoverage")
        total = (
            self.snapshot.block.schema.repeat_axis.size
            * self.snapshot.block.schema.point_table.row_count
        )
        metadata = tuple(self.cell_metadata)
        references = tuple(self.event_refs)
        if len(metadata) != total or len(references) != total:
            raise ValueError("monitor metadata and event refs must align to dataset cells")
        if any(
            reference is not None and not isinstance(reference, EventRef)
            for reference in references
        ):
            raise TypeError("event_refs must contain EventRef or None")
        if self.head is not None:
            if not isinstance(self.head, EventRef):
                raise TypeError("head must be EventRef or None")
            if self.head not in references:
                raise ValueError("head must be present in the aligned event refs")
        object.__setattr__(self, "cell_metadata", metadata)
        object.__setattr__(self, "event_refs", references)

    @property
    def ref(self) -> DatasetRevisionRef:
        return self.snapshot.ref

    @property
    def block(self) -> DataBlock:
        return self.snapshot.block


@dataclass(frozen=True)
class DatasetSealProvenance:
    output_span: EventSpanRef
    direct_parent_span: EventSpanRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_span, EventSpanRef):
            raise TypeError("output_span must be EventSpanRef")
        if self.direct_parent_span is not None and not isinstance(
            self.direct_parent_span,
            EventSpanRef,
        ):
            raise TypeError("direct_parent_span must be EventSpanRef or None")

    @property
    def stream_id(self) -> StreamId:
        return self.output_span.stream_id

    @property
    def generation(self) -> StreamGenerationId:
        return self.output_span.generation

    @property
    def start_sequence(self) -> int:
        return self.output_span.start_sequence

    @property
    def end_sequence(self) -> int:
        return self.output_span.end_sequence


def dataset_seal_provenance_to_tree(
    value: DatasetSealProvenance,
) -> dict[str, object]:
    """Project raw or processed exact-dataset provenance without information loss."""

    if not isinstance(value, DatasetSealProvenance):
        raise TypeError("value must be DatasetSealProvenance")
    return {
        "schema": _DATASET_SEAL_PROVENANCE_SCHEMA,
        "output_span": event_span_ref_to_tree(value.output_span),
        "direct_parent_span": (
            None
            if value.direct_parent_span is None
            else event_span_ref_to_tree(value.direct_parent_span)
        ),
    }


def dataset_seal_provenance_from_tree(tree: object) -> DatasetSealProvenance:
    """Decode only the current complete DatasetSealProvenance representation."""

    data = _exact_mapping(
        tree,
        {
            "schema",
            "output_span",
            "direct_parent_span",
        },
        _DATASET_SEAL_PROVENANCE_SCHEMA,
    )
    direct_parent_span = data["direct_parent_span"]
    value = DatasetSealProvenance(
        output_span=event_span_ref_from_tree(data["output_span"]),
        direct_parent_span=(
            None
            if direct_parent_span is None
            else event_span_ref_from_tree(direct_parent_span)
        ),
    )
    if dataset_seal_provenance_to_tree(value) != tree:
        raise ValueError("DatasetSealProvenance tree is typed but non-canonical")
    return value


def raw_dataset_seal_provenance_to_tree(
    value: DatasetSealProvenance,
) -> dict[str, object]:
    """Encode the raw, non-derived seal projection owned by this module."""

    if not isinstance(value, DatasetSealProvenance):
        raise TypeError("value must be DatasetSealProvenance")
    if value.direct_parent_span is not None:
        raise ValueError("raw dataset provenance cannot contain a parent span")
    return {
        "output_span": event_span_ref_to_tree(value.output_span),
    }


def raw_dataset_seal_provenance_from_tree(tree: object) -> DatasetSealProvenance:
    data = _exact_mapping(
        tree,
        {"output_span"},
        "raw dataset seal provenance",
        discriminator=None,
    )
    value = DatasetSealProvenance(
        event_span_ref_from_tree(data["output_span"]),
    )
    if raw_dataset_seal_provenance_to_tree(value) != tree:
        raise ValueError("raw DatasetSealProvenance tree is non-canonical")
    return value


class SealedDatasetArtifact:
    """Opaque exact-transport capability; physical scans may require outer attestation."""

    __slots__ = (
        "_snapshot",
        "_coverage",
        "_provenance",
        "_cell_schedule",
        "_event_metadata",
        "_terminal_reservation",
    )

    def __init__(
        self,
        authority: object,
        *,
        snapshot: OwnedSnapshot,
        coverage: DatasetCoverage,
        output_span: EventSpanRef,
        cell_schedule: DatasetCellSchedule,
        event_metadata: tuple[object | None, ...],
        terminal_reservation: ExactReservation,
        direct_parent_span: EventSpanRef | None = None,
    ) -> None:
        if authority is not _SEALED_TOKEN:
            raise PermissionError("SealedDatasetArtifact can only be minted by DatasetBuilder")
        object.__setattr__(self, "_snapshot", snapshot)
        object.__setattr__(self, "_coverage", coverage)
        if not isinstance(cell_schedule, DatasetCellSchedule):
            raise TypeError("cell_schedule must be DatasetCellSchedule")
        cell_schedule.validate_schema(snapshot.block.schema)
        if not isinstance(output_span, EventSpanRef):
            raise TypeError("output_span must be EventSpanRef")
        if len(cell_schedule) != output_span.count:
            raise ValueError("cell_schedule cardinality differs from event interval")
        object.__setattr__(self, "_cell_schedule", cell_schedule)
        object.__setattr__(
            self,
            "_provenance",
            DatasetSealProvenance(
                output_span=output_span,
                direct_parent_span=direct_parent_span,
            ),
        )
        object.__setattr__(self, "_event_metadata", tuple(event_metadata))
        object.__setattr__(self, "_terminal_reservation", terminal_reservation)

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

    @property
    def event_metadata(self) -> tuple[object | None, ...]:
        return self._event_metadata

    @property
    def cell_schedule(self) -> DatasetCellSchedule:
        return self._cell_schedule

    def _belongs_to_terminal_reservation(self, reservation: object) -> bool:
        """Process-local ownership proof used only by PipelineResult minting."""

        return reservation is not None and self._terminal_reservation is reservation

    def _with_direct_parent_span(
        self,
        readiness: ExactConsumerReadiness,
        direct_parent_span: EventSpanRef,
    ) -> "SealedDatasetArtifact":
        """Bind a processed result to its exact immediate input interval."""

        if not isinstance(readiness, ExactConsumerReadiness):
            raise TypeError("readiness must be ExactConsumerReadiness")
        if not isinstance(direct_parent_span, EventSpanRef):
            raise TypeError("direct_parent_span must be EventSpanRef")
        reservation = readiness._source_reservation
        source = reservation._stream
        if (
            direct_parent_span.stream_id != source.stream_id
            or direct_parent_span.generation != source.generation
            or direct_parent_span.start_sequence != reservation.start_sequence
            or direct_parent_span.end_sequence != reservation.end_sequence
        ):
            raise DatasetError("direct parent span differs from readiness source interval")
        if self._terminal_reservation is not readiness._terminal_reservation:
            raise DatasetError("sealed dataset belongs to another terminal readiness")
        if readiness._source_reservation is readiness._terminal_reservation:
            raise DatasetError("direct datasets do not need a root input span")
        provenance = self._provenance
        return SealedDatasetArtifact(
            _SEALED_TOKEN,
            snapshot=self._snapshot,
            coverage=self._coverage,
            output_span=provenance.output_span,
            cell_schedule=self._cell_schedule,
            event_metadata=self._event_metadata,
            terminal_reservation=self._terminal_reservation,
            direct_parent_span=direct_parent_span,
        )


def _new_validity_storage(schema: DatasetSchema) -> np.ndarray:
    contract = schema.cell_schema.validity_contract
    if contract.mode is ValidityMode.VALUE:
        return np.zeros(schema.physical_shape[:2], dtype=bool)
    axes = tuple(schema.cell_schema.axis(axis_id) for axis_id in contract.component_axis_ids)
    return np.zeros(
        (*schema.physical_shape[:2], *(axis.size for axis in axes)),
        dtype=bool,
    )


def _value_validity_mask(schema: DatasetSchema, validity: object) -> np.ndarray | bool:
    contract = schema.cell_schema.validity_contract
    if contract.mode is ValidityMode.VALUE:
        if isinstance(validity, Valid):
            return True
        if isinstance(validity, Invalid):
            return False
        raise ValueError("component validity cannot enter a VALUE dataset contract")
    return expand_component_validity(validity, schema.cell_schema)


def _materialized_validity(schema: DatasetSchema, validity: np.ndarray):
    contract = schema.cell_schema.validity_contract
    if contract.mode is ValidityMode.VALUE:
        return CellValidity(validity)
    return DatasetComponentValidity(contract.component_axis_ids, validity)


def _project_payload(
    edge: FrozenDatasetEdge[PayloadT],
    payload: PayloadT,
) -> tuple[Value, object | None]:
    value = edge.project_value(payload)
    if not isinstance(value, Value):
        raise TypeError("DatasetEventAdapter.value must return Value")
    if value.schema is not edge.schema.cell_schema:
        raise DatasetError("event ValueSchema differs from DatasetSchema cell contract")
    contract = edge.metadata_contract
    metadata = contract.snapshot(payload)
    contract.validate(metadata)
    if not _is_deeply_immutable(metadata):
        raise TypeError("metadata contract must return a deeply immutable snapshot")
    return value, metadata


def _write_cell(
    cell: tuple[int, int],
    value: Value,
    validity_mask: np.ndarray | bool,
    values: np.ndarray,
    written: np.ndarray,
    validity: np.ndarray,
) -> None:
    values[cell] = value.values
    written[cell] = True
    validity[cell] = validity_mask


class DatasetBuilder(Generic[PayloadT]):
    """Finite exact event-to-dataset materializer and formal seal owner."""

    def __init__(
        self,
        block_id: BlockId,
        source: ExactReservation[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> None:
        if not isinstance(block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(source, ExactReservation):
            raise TypeError("DatasetBuilder must bind an ExactReservation")
        if not isinstance(edge, FrozenDatasetEdge):
            raise TypeError("edge must be FrozenDatasetEdge")
        if edge.cell_schedule is None:
            raise DatasetError("DatasetBuilder requires a frozen exact cell schedule")
        self.block_id = block_id
        self._reservation = source
        self._source: AcquisitionStream[PayloadT] = source._stream
        edge.validate_stream(self._source)
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        self.edge = edge
        self.schema = edge.schema
        self._cell_schedule = edge.cell_schedule
        total_cells = self.schema.repeat_axis.size * self.schema.point_table.row_count
        reserved_events = source.end_sequence - source.start_sequence
        if reserved_events != total_cells:
            raise DatasetError("exact reservation length must equal DatasetSchema cell count")
        self._lock = threading.RLock()
        self._preview_reader_minted = False
        self._values = np.zeros(
            self.schema.physical_shape,
            dtype=self.schema.cell_schema.dtype,
        )
        self._written = np.zeros(self.schema.physical_shape[:2], dtype=bool)
        self._written_count = 0
        self._validity = _new_validity_storage(self.schema)
        self._revision = 0
        self._expected_sequence = source.start_sequence
        self._cell_metadata: list[object | None] = [None] * total_cells
        self._ordered_event_metadata: list[object | None] = [None] * total_cells
        self._sealed = False
        self._aborted = False
        self._exact_readiness = self._source._claim_consumer(
            source,
            self,
            terminal=True,
        )

    @property
    def revision(self) -> DatasetRevision:
        with self._lock:
            return DatasetRevision(self._revision)

    def current_ref(self) -> DatasetRevisionRef:
        with self._lock:
            return self._ref_locked(self._revision)

    def consume(self, delivery: Delivery[PayloadT]) -> None:
        if not isinstance(delivery, Delivery) or not delivery.is_exact:
            raise TypeError("DatasetBuilder requires an exact Delivery capability")
        if delivery.acknowledged:
            raise DatasetError("delivery was already acknowledged")
        projected = _project_payload(self.edge, delivery.envelope.payload)
        self._source._consume_exact(
            self._reservation,
            delivery,
            self,
            lambda envelope: self._ingest(envelope, projected),
        )

    def _ingest(
        self,
        envelope: Envelope[PayloadT],
        projected: tuple[Value, object | None],
    ) -> None:
        address = envelope.join_key
        if not isinstance(address, DatasetCellAddress):
            raise DatasetError("dataset event is missing its typed DatasetCellAddress")
        value, metadata = projected
        validity_mask = _value_validity_mask(self.schema, value.validity)
        with self._lock:
            self._ensure_writable_locked()
            self._validate_envelope_identity_locked(envelope)
            if envelope.sequence != self._expected_sequence:
                raise DatasetError(
                    f"exact dataset expected sequence {self._expected_sequence}, "
                    f"got {envelope.sequence}"
                )
            schedule_index = envelope.sequence - self._reservation.start_sequence
            expected_address = self._cell_schedule.cell_at(schedule_index)
            if address != expected_address:
                raise DatasetError(
                    f"event join key {address} differs from frozen plan key "
                    f"{expected_address} at sequence {envelope.sequence}"
                )
            cell = (address.repeat_index, address.point_ordinal)
            _write_cell(
                cell,
                value,
                validity_mask,
                self._values,
                self._written,
                self._validity,
            )
            self._revision += 1
            self._written_count += 1
            self._expected_sequence += 1
            self._ordered_event_metadata[schedule_index] = metadata
            flat_cell = (
                address.repeat_index * self.schema.point_table.row_count
                + address.point_ordinal
            )
            self._cell_metadata[flat_cell] = metadata

    def materialize(self, ref: DatasetRevisionRef | None = None) -> DatasetPreviewSnapshot:
        with self._lock:
            selected = self._select_current_ref_locked(ref)
            block = DataBlock(
                block_id=self.block_id,
                revision=selected.revision,
                values=self._values,
                validity=_materialized_validity(self.schema, self._validity),
                schema=self.schema,
            )
            return DatasetPreviewSnapshot(
                snapshot=OwnedSnapshot(selected, block),
                coverage=self._coverage_locked(),
                cell_metadata=tuple(self._cell_metadata),
            )

    def materialize_delta(self, after: DatasetRevision) -> DatasetPreviewDelta:
        """Freeze at most one append-only cell, copying it outside the lock."""

        if not isinstance(after, DatasetRevision):
            raise TypeError("after must be DatasetRevision")
        with self._lock:
            if self._aborted:
                raise DatasetError("aborted exact dataset cannot be previewed")
            if after.value > self._revision:
                raise KeyError(
                    f"dataset revision {after.value} has not been committed"
                )
            stop = min(self._revision, after.value + 1)
            selected = self._ref_locked(stop)
            total = (
                self.schema.repeat_axis.size
                * self.schema.point_table.row_count
            )
            coverage = DatasetCoverage(stop, total)
            source = None
            contract = self.schema.cell_schema.validity_contract
            if stop > after.value:
                ordinal = after.value
                address = self._cell_schedule.cell_at(ordinal)
                cell = (address.repeat_index, address.point_ordinal)
                source = (
                    ordinal,
                    address,
                    self._values[cell],
                    (
                        bool(self._validity[cell])
                        if contract.mode is ValidityMode.VALUE
                        else self._validity[cell]
                    ),
                    self._ordered_event_metadata[ordinal],
                )

        cells = ()
        if source is not None:
            ordinal, address, values, validity_source, metadata = source
            validity = (
                (Valid() if validity_source else Invalid())
                if contract.mode is ValidityMode.VALUE
                else ComponentValidity(
                    contract.component_axis_ids,
                    validity_source,
                )
            )
            cells = (
                DatasetPreviewCell(
                    ordinal,
                    address,
                    Value(values, validity, self.schema.cell_schema),
                    metadata,
                ),
            )
        return DatasetPreviewDelta(
            after,
            selected,
            cells,
            coverage,
        )

    def seal(self, eos: EndOfStream) -> SealedDatasetArtifact:
        self._source._complete_consumer(self._reservation, eos, self, self._seal_locked)
        preview = self.materialize()
        return SealedDatasetArtifact(
            _SEALED_TOKEN,
            snapshot=preview.snapshot,
            coverage=preview.coverage,
            output_span=EventSpanRef(
                self.stream_id,
                self.generation,
                self._reservation.start_sequence,
                self._reservation.end_sequence,
            ),
            cell_schedule=self._cell_schedule,
            event_metadata=tuple(self._ordered_event_metadata),
            terminal_reservation=self._reservation,
        )

    def _seal_locked(self) -> None:
        with self._lock:
            self._ensure_writable_locked()
            missing = np.argwhere(~self._written)
            if missing.size:
                cells = tuple(tuple(int(index) for index in row) for row in missing)
                raise MissingDatasetCells(
                    f"dataset is missing {len(missing)} cells: {cells}"
                )
            if not self._coverage_locked().complete:
                raise DatasetError("formal dataset coverage is incomplete")
            self._sealed = True

    def abort(self) -> None:
        self._source._abort_consumer(
            self._reservation,
            self,
            self._mark_aborted_locked,
        )

    def exact_readiness(self) -> ExactConsumerReadiness:
        with self._lock:
            self._ensure_writable_locked()
            readiness = self._exact_readiness
        readiness._validate_terminal_sink()
        return readiness

    def open_preview_reader(self) -> "ExactDatasetPreviewReader":
        """Mint a read-only progressive view without exposing write authority."""

        with self._lock:
            self._ensure_writable_locked()
            if self._preview_reader_minted:
                raise RuntimeError("exact dataset already has a preview reader")
            self._preview_reader_minted = True
        return ExactDatasetPreviewReader(_PREVIEW_READER_TOKEN, self)

    def close(self) -> None:
        """Idempotently abort if needed and release the exact reservation."""

        if self._reservation.state is ReservationState.RELEASED:
            return
        with self._lock:
            needs_abort = not self._sealed and not self._aborted
        if needs_abort:
            self.abort()
        if self._reservation.state in (
            ReservationState.COMPLETED,
            ReservationState.FAILED,
            ReservationState.CANCELLED,
        ):
            self._reservation.release()

    def __enter__(self) -> "DatasetBuilder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _close_preserving_body_error(self.close, exc, "DatasetBuilder teardown also failed")
        return False

    def _mark_aborted_locked(self) -> None:
        with self._lock:
            if self._sealed:
                raise DatasetError("sealed dataset cannot be aborted")
            self._aborted = True

    def _ensure_writable_locked(self) -> None:
        if self._sealed:
            raise DatasetError("dataset is sealed")
        if self._aborted:
            raise DatasetError("dataset is aborted")

    def _validate_envelope_identity_locked(self, envelope: Envelope[PayloadT]) -> None:
        if envelope.stream_generation != self.generation:
            raise DatasetError("envelope stream generation differs from DatasetBuilder")
        if envelope.stream_id != self.stream_id:
            raise DatasetError("envelope stream id differs from DatasetBuilder")

    def _coverage_locked(self) -> DatasetCoverage:
        return DatasetCoverage(self._written_count, int(self._written.size))

    def _ref_locked(self, revision: int) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            block_id=self.block_id,
            stream_generation=self.generation,
            schema_fingerprint=self.schema.fingerprint,
            revision=DatasetRevision(revision),
        )

    def _select_current_ref_locked(
        self,
        ref: DatasetRevisionRef | None,
    ) -> DatasetRevisionRef:
        selected = self._ref_locked(self._revision) if ref is None else ref
        _validate_dataset_ref(selected, self.block_id, self.generation, self.schema)
        target_revision = selected.revision.value
        if target_revision > self._revision:
            raise KeyError(f"dataset revision {target_revision} has not been committed")
        if target_revision != self._revision:
            raise SnapshotExpired(
                "materializers retain only the current revision; callers retain snapshots"
            )
        return selected


_PREVIEW_READER_TOKEN = object()


class ExactDatasetPreviewReader:
    """Process-local read-only revision surface over one exact materializer.

    An attached exact-live port can freeze committed deltas after the producer
    explicitly announces an update.  The reader cannot wait, consume, seal,
    abort, release, or obtain the exact reservation/cursor, so DatasetBuilder
    remains the sole authority and the port owns its own wake/close lifetime.
    """

    __slots__ = ("__builder",)

    def __init_subclass__(cls, **_kwargs) -> None:
        raise TypeError("ExactDatasetPreviewReader is final")

    def __init__(self, authority: object, builder: DatasetBuilder) -> None:
        if authority is not _PREVIEW_READER_TOKEN:
            raise PermissionError(
                "ExactDatasetPreviewReader can only be minted by DatasetBuilder"
            )
        if not isinstance(builder, DatasetBuilder):
            raise TypeError("builder must be DatasetBuilder")
        self.__builder = builder

    def __reduce__(self):
        raise TypeError("ExactDatasetPreviewReader is process-local")

    def __reduce_ex__(self, _protocol: int):
        raise TypeError("ExactDatasetPreviewReader is process-local")

    @property
    def schema(self) -> DatasetSchema:
        return self.__builder.schema

    @property
    def stream_generation(self) -> StreamGenerationId:
        return self.__builder.generation

    @property
    def coverage(self) -> DatasetCoverage:
        builder = self.__builder
        with builder._lock:
            return builder._coverage_locked()

    @property
    def terminal(self) -> bool:
        builder = self.__builder
        with builder._lock:
            return builder._sealed or builder._aborted

    @property
    def failed(self) -> bool:
        builder = self.__builder
        with builder._lock:
            return builder._aborted

    def freeze_delta(self, after: DatasetRevision) -> DatasetPreviewDelta:
        """Freeze each newly committed cell once without copying prior cells."""

        if not isinstance(after, DatasetRevision):
            raise TypeError("after must be DatasetRevision")
        return self.__builder.materialize_delta(after)


@dataclass(frozen=True, slots=True, eq=False)
class _LatestCellReplacement(Generic[PayloadT]):
    """Fully written latest-cell shadow awaiting one authoritative envelope."""

    payload: PayloadT
    base_revision: int
    expected_sequence: int
    values: np.ndarray
    written: np.ndarray
    validity: np.ndarray
    cell_metadata: list[object | None]
    event_refs: list[EventRef | None]


class MonitorDataset(Generic[PayloadT]):
    """Sequence-owned live materializer; never a formal artifact authority."""

    @classmethod
    def keyed_cycle(
        cls,
        block_id: BlockId,
        source: MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> "MonitorDataset[PayloadT]":
        if edge.cell_schedule is None:
            raise DatasetError("keyed_cycle requires a frozen complete cell schedule")
        return cls(block_id, source, edge)

    @classmethod
    def latest_cell(
        cls,
        block_id: BlockId,
        source: MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> "MonitorDataset[PayloadT]":
        if edge.cell_schedule is not None:
            raise DatasetError("latest_cell requires a schedule-free dataset edge")
        schema = edge.schema
        if (
            schema.repeat_axis.size != 1
            or schema.point_table != PointTable(1)
            or schema.grid_topology is not None
        ):
            raise DatasetError(
                "latest_cell requires canonical single-cell storage "
                "(R=1, P=1, no point columns or grid topology)"
            )
        return cls(block_id, source, edge)

    def __init__(
        self,
        block_id: BlockId,
        source: MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> None:
        if not isinstance(block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(source, MonitorTap):
            raise TypeError("MonitorDataset must bind a MonitorTap")
        if not isinstance(edge, FrozenDatasetEdge):
            raise TypeError("edge must be FrozenDatasetEdge")
        self.block_id = block_id
        self._monitor = source
        self._source: AcquisitionStream[PayloadT] = source._stream
        self.edge = edge
        self.schema = edge.schema
        self._cycle_schedule = edge.cell_schedule
        if self._cycle_schedule is None:
            edge.validate_payload_stream(self._source)
            if (
                self.schema.repeat_axis.size != 1
                or self.schema.point_table != PointTable(1)
                or self.schema.grid_topology is not None
            ):
                raise DatasetError(
                    "schedule-free MonitorDataset requires latest-cell geometry"
                )
        else:
            edge.validate_stream(self._source)
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        total_cells = self.schema.repeat_axis.size * self.schema.point_table.row_count
        source._claim_consumer(self)
        try:
            self._lock = threading.RLock()
            self._consume_lock = threading.Lock()
            self._values = np.zeros(
                self.schema.physical_shape,
                dtype=self.schema.cell_schema.dtype,
            )
            self._written = np.zeros(self.schema.physical_shape[:2], dtype=bool)
            self._validity = _new_validity_storage(self.schema)
            self._cell_metadata: list[object | None] = [None] * total_cells
            self._event_refs: list[EventRef | None] = [None] * total_cells
            self._revision = 0
            self._last_sequence: int | None = None
            self._missed_events = 0
            self._current_gap = False
            self._head: EventRef | None = None
            self._latest_replacement: _LatestCellReplacement[PayloadT] | None = None
            self._aborted = False
        except BaseException:
            source.close()
            raise

    @property
    def revision(self) -> DatasetRevision:
        with self._lock:
            return DatasetRevision(self._revision)

    def current_ref(self) -> DatasetRevisionRef:
        with self._lock:
            return self._ref_locked(self._revision)

    def ingest_next(self, timeout: float | None = None) -> DatasetRevisionRef:
        with self._consume_lock:
            with self._lock:
                if self._latest_replacement is not None:
                    raise DatasetError(
                        "ordinary monitor ingest cannot consume a staged latest-cell replacement"
                    )
            update = self._monitor._next_for(self, timeout)
            return self._ingest(update)

    def ingest_latest(
        self,
        *,
        account_skipped_events: bool = True,
        expected_event_ref: EventRef | None = None,
    ) -> DatasetRevisionRef:
        """Ingest newest, optionally admitting an authored sparse source view.

        ``expected_event_ref`` binds the materialized head to the producer's
        already accepted exact envelope; it is checked before and after the
        atomic write.  ``account_skipped_events=False`` is legal only for a
        single-cell latest view whose omitted events were explicitly selected
        out by that same producer.
        """

        if not isinstance(account_skipped_events, bool):
            raise TypeError("account_skipped_events must be bool")
        if expected_event_ref is not None and not isinstance(
            expected_event_ref,
            EventRef,
        ):
            raise TypeError("expected_event_ref must be EventRef or None")
        with self._consume_lock:
            with self._lock:
                if self._latest_replacement is not None:
                    raise DatasetError(
                        "ordinary monitor ingest cannot consume a staged latest-cell replacement"
                    )
                if not account_skipped_events and self._cycle_schedule is not None:
                    raise DatasetError(
                        "intentional source selection requires a latest-cell monitor"
                    )
            update = self._monitor._latest_for(self)
            if (
                expected_event_ref is not None
                and update.envelope.ref != expected_event_ref
            ):
                raise DatasetError(
                    "monitor tap delivered another event than the selected exact envelope"
                )
            revision = self._ingest(
                update,
                account_skipped_events=account_skipped_events,
            )
            if expected_event_ref is not None:
                with self._lock:
                    if self._head != expected_event_ref:
                        raise DatasetError(
                            "monitor head differs from the selected exact envelope"
                        )
            return revision

    def prepare_latest_cell_replacement(
        self,
        payload: PayloadT,
    ) -> _LatestCellReplacement[PayloadT]:
        """Write every fallible part of one latest cell before stream publish.

        The returned private token is not a stream event.  Until a matching
        authoritative envelope is committed, the existing materialization and
        its sequence watermark remain untouched.
        """

        with self._consume_lock:
            with self._lock:
                self._ensure_writable_locked()
                if self._cycle_schedule is not None:
                    raise DatasetError(
                        "prepare_latest_cell_replacement requires latest-cell storage"
                    )
                if self._latest_replacement is not None:
                    raise DatasetError("latest-cell replacement is already pending")

                contract = self.edge.payload_contract
                owned_payload = contract.snapshot(payload)
                contract.validate(owned_payload)
                value, metadata = _project_payload(self.edge, owned_payload)
                validity_mask = _value_validity_mask(self.schema, value.validity)
                total_cells = (
                    self.schema.repeat_axis.size
                    * self.schema.point_table.row_count
                )
                values = np.zeros(
                    self.schema.physical_shape,
                    dtype=self.schema.cell_schema.dtype,
                )
                written = np.zeros(self.schema.physical_shape[:2], dtype=bool)
                validity = _new_validity_storage(self.schema)
                cell_metadata: list[object | None] = [None] * total_cells
                event_refs: list[EventRef | None] = [None] * total_cells
                _write_cell(
                    (0, 0),
                    value,
                    validity_mask,
                    values,
                    written,
                    validity,
                )
                cell_metadata[0] = metadata
                replacement = _LatestCellReplacement(
                    owned_payload,
                    self._revision,
                    0 if self._last_sequence is None else self._last_sequence + 1,
                    values,
                    written,
                    validity,
                    cell_metadata,
                    event_refs,
                )
                self._latest_replacement = replacement
                return replacement

    def abort_latest_cell_replacement(
        self,
        replacement: _LatestCellReplacement[PayloadT],
    ) -> None:
        """Discard one unpublished replacement without touching the current cell."""

        with self._consume_lock:
            with self._lock:
                self._ensure_writable_locked()
                if self._latest_replacement is not replacement:
                    raise DatasetError("latest-cell replacement token is not pending")
                self._latest_replacement = None

    def commit_latest_cell_replacement(
        self,
        replacement: _LatestCellReplacement[PayloadT],
        envelope: Envelope[PayloadT],
        *,
        timeout: float | None = 0.0,
    ) -> None:
        """Bind a published envelope to a fully written shadow and swap owners.

        No projector, allocator, reducer, or cell writer is called after the
        stream publication boundary.  A failure here is therefore an internal
        stream/tap identity violation; callers must retire the generation rather
        than resume its old binding.
        """

        if not isinstance(envelope, Envelope):
            raise TypeError("latest-cell replacement envelope must be Envelope")
        with self._consume_lock:
            with self._lock:
                self._ensure_writable_locked()
                if self._latest_replacement is not replacement:
                    raise DatasetError("latest-cell replacement token is not pending")
            try:
                update = self._monitor._next_for(self, timeout)
                with self._lock:
                    self._ensure_writable_locked()
                    if self._latest_replacement is not replacement:
                        raise DatasetError("latest-cell replacement changed during commit")
                    if update.envelope is not envelope:
                        raise DatasetError(
                            "latest-cell replacement consumed another stream envelope"
                        )
                    self._validate_envelope_identity_locked(envelope)
                    if envelope.payload is not replacement.payload:
                        raise DatasetError(
                            "latest-cell replacement published another owned payload"
                        )
                    if envelope.sequence != replacement.expected_sequence:
                        raise DatasetError(
                            "latest-cell replacement sequence differs from its staged watermark"
                        )
                    if update.missed != 0:
                        raise DatasetError(
                            "latest-cell replacement requires ordered monitor delivery"
                        )
                    if self._revision != replacement.base_revision:
                        raise DatasetError(
                            "latest-cell replacement base revision changed before commit"
                        )

                    replacement.event_refs[0] = envelope.ref
                    self._values = replacement.values
                    self._written = replacement.written
                    self._validity = replacement.validity
                    self._cell_metadata = replacement.cell_metadata
                    self._event_refs = replacement.event_refs
                    self._missed_events = 0
                    self._current_gap = False
                    self._last_sequence = envelope.sequence
                    self._head = envelope.ref
                    self._revision += 1
                    self._latest_replacement = None
            except BaseException:
                with self._lock:
                    if self._latest_replacement is replacement:
                        self._latest_replacement = None
                raise

    def _ingest(
        self,
        update: MonitorUpdate[PayloadT],
        *,
        account_skipped_events: bool = True,
    ) -> DatasetRevisionRef:
        envelope = update.envelope
        value, metadata = _project_payload(self.edge, envelope.payload)
        validity_mask = _value_validity_mask(self.schema, value.validity)
        latest_state = None
        if self._cycle_schedule is None:
            values = np.zeros(
                self.schema.physical_shape,
                dtype=self.schema.cell_schema.dtype,
            )
            written = np.zeros(self.schema.physical_shape[:2], dtype=bool)
            validity = _new_validity_storage(self.schema)
            _write_cell(
                (0, 0),
                value,
                validity_mask,
                values,
                written,
                validity,
            )
            latest_state = (values, written, validity)
        with self._lock:
            self._ensure_writable_locked()
            self._validate_envelope_identity_locked(envelope)
            expected_sequence = (
                0 if self._last_sequence is None else self._last_sequence + 1
            )
            if envelope.sequence < expected_sequence:
                raise DatasetError(
                    "monitor dataset events must remain strictly ordered"
                )
            sequence_gap = envelope.sequence - expected_sequence
            if self._cycle_schedule is None:
                assert latest_state is not None
                values, written, validity = latest_state
                cell_metadata = [metadata]
                event_refs = [envelope.ref]
                missed_now = (
                    max(update.missed, sequence_gap)
                    if account_skipped_events
                    else 0
                )
                missed_events = self._missed_events + missed_now
                current_gap = missed_now > 0
            else:
                offset = envelope.sequence % len(self._cycle_schedule)
                expected_address = self._cycle_schedule.cell_at(offset)
                if envelope.join_key != expected_address:
                    raise DatasetError(
                        f"monitor cycle key {envelope.join_key!r} differs from "
                        f"frozen key {expected_address!r} at sequence "
                        f"{envelope.sequence}"
                    )
                if offset == 0 or sequence_gap > 0 or update.missed > 0:
                    self._clear_locked()
                cell = (
                    expected_address.repeat_index,
                    expected_address.point_ordinal,
                )
                values = self._values
                written = self._written
                validity = self._validity
                cell_metadata = self._cell_metadata
                event_refs = self._event_refs
                missed_events = self._missed_events + max(
                    update.missed,
                    sequence_gap,
                )
                _write_cell(
                    cell,
                    value,
                    validity_mask,
                    values,
                    written,
                    validity,
                )
                flat_cell = cell[0] * self.schema.point_table.row_count + cell[1]
                cell_metadata[flat_cell] = metadata
                event_refs[flat_cell] = envelope.ref
            if self._cycle_schedule is None:
                self._values = values
                self._written = written
                self._validity = validity
                self._cell_metadata = cell_metadata
                self._event_refs = event_refs
                self._current_gap = current_gap
            self._missed_events = missed_events
            self._last_sequence = envelope.sequence
            self._head = envelope.ref
            self._revision += 1
            return self._ref_locked(self._revision)

    def freeze_current(self) -> MonitorDatasetSnapshot:
        """Freeze the current live revision through the desktop read seam."""

        return self.materialize(None)

    def materialize(
        self,
        ref: DatasetRevisionRef | None = None,
    ) -> MonitorDatasetSnapshot:
        """Freeze the current revision, optionally asserting its exact ref."""

        with self._lock:
            selected = self._select_current_ref_locked(ref)
            values = self._values
            written = self._written
            validity = self._validity
            metadata = tuple(self._cell_metadata)
            event_refs = tuple(self._event_refs)
            block = DataBlock(
                block_id=self.block_id,
                revision=selected.revision,
                values=values,
                validity=_materialized_validity(self.schema, validity),
                schema=self.schema,
            )
            return MonitorDatasetSnapshot(
                snapshot=OwnedSnapshot(selected, block),
                coverage=self._coverage_locked(written),
                cell_metadata=metadata,
                event_refs=event_refs,
                head=self._head,
            )

    def close(self) -> None:
        with self._lock:
            self._aborted = True
            self._latest_replacement = None
        self._monitor.close()

    def __enter__(self) -> "MonitorDataset":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        _close_preserving_body_error(
            self.close,
            exc,
            "MonitorDataset teardown also failed",
        )
        return False

    def _clear_locked(self) -> None:
        self._values.fill(0)
        self._written.fill(False)
        self._validity.fill(False)
        self._cell_metadata[:] = [None] * len(self._cell_metadata)
        self._event_refs[:] = [None] * len(self._event_refs)

    def _coverage_locked(
        self,
        written: np.ndarray,
    ) -> MonitorCoverage:
        return MonitorCoverage(
            written_cells=int(np.count_nonzero(written)),
            total_cells=int(written.size),
            missed_events=self._missed_events,
            current_gap=(
                self._current_gap if self._cycle_schedule is None else False
            ),
        )

    def _ensure_writable_locked(self) -> None:
        if self._aborted:
            raise DatasetError("monitor dataset is closed")

    def _validate_envelope_identity_locked(self, envelope: Envelope[PayloadT]) -> None:
        if envelope.stream_generation != self.generation:
            raise DatasetError("envelope stream generation differs from MonitorDataset")
        if envelope.stream_id != self.stream_id:
            raise DatasetError("envelope stream id differs from MonitorDataset")

    def _ref_locked(self, revision: int) -> DatasetRevisionRef:
        return DatasetRevisionRef(
            block_id=self.block_id,
            stream_generation=self.generation,
            schema_fingerprint=self.schema.fingerprint,
            revision=DatasetRevision(revision),
        )

    def _select_current_ref_locked(
        self,
        ref: DatasetRevisionRef | None,
    ) -> DatasetRevisionRef:
        selected = self._ref_locked(self._revision) if ref is None else ref
        _validate_dataset_ref(selected, self.block_id, self.generation, self.schema)
        target_revision = selected.revision.value
        if target_revision > self._revision:
            raise KeyError(f"dataset revision {target_revision} has not been committed")
        if target_revision != self._revision:
            raise SnapshotExpired(
                "materializers retain only the current revision; callers retain snapshots"
            )
        return selected


def _validate_dataset_ref(
    ref: DatasetRevisionRef,
    block_id: BlockId,
    generation: StreamGenerationId,
    schema: DatasetSchema,
) -> None:
    if not isinstance(ref, DatasetRevisionRef):
        raise TypeError("ref must be DatasetRevisionRef")
    if ref.block_id != block_id:
        raise ValueError("snapshot ref belongs to another block")
    if ref.stream_generation != generation:
        raise ValueError("snapshot ref belongs to another stream generation")
    if ref.schema_fingerprint != schema.fingerprint:
        raise ValueError("snapshot ref schema fingerprint differs")


def _close_preserving_body_error(
    close: Callable[[], None],
    body_error: BaseException | None,
    message: str,
) -> None:
    try:
        close()
    except BaseException as cleanup_error:
        if body_error is None:
            raise
        record_secondary_failure(body_error, message, cleanup_error)


__all__ = [
    "DatasetBuilder",
    "DatasetCellAddress",
    "DatasetCellSchedule",
    "DatasetCellKeyContract",
    "DatasetCoverage",
    "DatasetEventAdapter",
    "DatasetMetadataContract",
    "DatasetError",
    "DatasetPreviewCell",
    "DatasetPreviewDelta",
    "DatasetPreviewSnapshot",
    "DatasetSealProvenance",
    "ExactDatasetPreviewReader",
    "FrozenDatasetEdge",
    "MissingDatasetCells",
    "MonitorCoverage",
    "MonitorDataset",
    "MonitorDatasetSnapshot",
    "SnapshotExpired",
    "SealedDatasetArtifact",
    "dataset_seal_provenance_from_tree",
    "dataset_seal_provenance_to_tree",
    "raw_dataset_seal_provenance_from_tree",
    "raw_dataset_seal_provenance_to_tree",
]
