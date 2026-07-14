"""Single-owner event-to-dataset materialization with revisioned snapshots."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from numbers import Integral
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
from zlc_storage import canonical_digest, encode, sha256_text as _sha256_digest

from zlc_data import (
    BlockId,
    AxisLayoutMode,
    AxisSpec,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
    MONITOR_HISTORY,
    StreamGenerationId,
    PointLayout,
    REPEAT,
    Valid,
    ValidityMode,
    Value,
    ValuePayloadContract,
    ValueSchema,
    axis_to_tree,
    expand_component_validity,
    point_layout_to_tree,
)

from ._failure import record_secondary_failure

from .streams import (
    AcquisitionStream,
    ArtifactInputRef,
    Delivery,
    EndOfStream,
    Envelope,
    EventRef,
    EventSpanRef,
    ExactConsumerReadiness,
    ExactReservation,
    MonitorTap,
    MonitorUpdate,
    ReservationState,
    ProcessorStageProvenance,
    StreamId,
    TraceBinding,
    event_ref_to_tree,
)


PayloadT = TypeVar("PayloadT")


class DatasetEventAdapter(Protocol[PayloadT]):
    """Typed projection from one immutable stream payload into one dataset cell."""

    payload_contract: object
    value_schema: ValueSchema
    metadata_contract: "DatasetMetadataContract[PayloadT]"
    operator_fingerprint: str

    def value(self, payload: PayloadT) -> Value: ...



class DatasetMetadataContract(Protocol[PayloadT]):
    fingerprint: str
    max_retained_nbytes: int

    def snapshot(self, payload: PayloadT) -> object | None: ...

    def validate(self, metadata: object | None) -> None: ...

    def retained_nbytes(self, metadata: object | None) -> int: ...

    def digest(self, metadata: object | None) -> str: ...


@dataclass(frozen=True)
class NoDatasetMetadataContract:
    fingerprint: str = hashlib.sha256(b"zlc.dataset-metadata.none").hexdigest()
    max_retained_nbytes: int = 0

    @staticmethod
    def snapshot(_payload: object) -> None:
        return None

    @staticmethod
    def validate(metadata: object | None) -> None:
        if metadata is not None:
            raise TypeError("no-metadata contract accepts only None")

    @staticmethod
    def retained_nbytes(metadata: object | None) -> int:
        NoDatasetMetadataContract.validate(metadata)
        return 0

    @staticmethod
    def digest(metadata: object | None) -> str:
        NoDatasetMetadataContract.validate(metadata)
        return hashlib.sha256(b"null").hexdigest()


@dataclass(frozen=True)
class ValueDatasetEventAdapter:
    payload_contract: ValuePayloadContract
    metadata_contract: NoDatasetMetadataContract = NoDatasetMetadataContract()
    operator_fingerprint: str = canonical_digest(
        {
            "owner": "zlc_neutral_atom.runtime.dataset.ValueDatasetEventAdapter",
            "operator": "identity-value",
        }
    )

    @property
    def value_schema(self) -> ValueSchema:
        return self.payload_contract.schema

    def value(self, payload: Value) -> Value:
        return payload



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


@dataclass(frozen=True)
class DatasetCellDomain:
    """Sampling-cell identity independent of the value stored in each cell."""

    repeat_axis: AxisSpec
    point_axes: tuple[AxisSpec, ...]
    point_layout: PointLayout
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.repeat_axis, AxisSpec) or self.repeat_axis.role != REPEAT:
            raise ValueError("repeat_axis must be an AxisSpec with role 'repeat'")
        axes = tuple(self.point_axes)
        if not axes or any(not isinstance(axis, AxisSpec) for axis in axes):
            raise ValueError("point_axes must contain AxisSpec values")
        if len({axis.axis_id for axis in axes}) != len(axes):
            raise ValueError("point_axes must have unique AxisId values")
        if self.repeat_axis.axis_id in {axis.axis_id for axis in axes}:
            raise ValueError("repeat and point axes must be distinct")
        if not isinstance(self.point_layout, PointLayout):
            raise TypeError("point_layout must be PointLayout")
        if self.point_layout.logical_shape != tuple(axis.size for axis in axes):
            raise ValueError("point_layout shape differs from point_axes")
        object.__setattr__(self, "point_axes", axes)
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(
                {
                    "contract": "zlc_neutral_atom.DatasetCellDomain",
                    "repeat_axis": axis_to_tree(self.repeat_axis),
                    "point_axes": [axis_to_tree(axis) for axis in axes],
                    "point_layout": point_layout_to_tree(self.point_layout),
                }
            ),
        )

    @classmethod
    def from_schema(cls, schema: DatasetSchema) -> "DatasetCellDomain":
        if not isinstance(schema, DatasetSchema):
            raise TypeError("schema must be DatasetSchema")
        return cls(schema.repeat_axis, schema.point_axes, schema.point_layout)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


def dataset_cell_key_fingerprint(
    schema_or_domain: DatasetSchema | DatasetCellDomain,
) -> str:
    """Bind cell keys to sampling axes/layout, never the cell ValueSchema."""

    if isinstance(schema_or_domain, DatasetSchema):
        domain = DatasetCellDomain.from_schema(schema_or_domain)
    elif isinstance(schema_or_domain, DatasetCellDomain):
        domain = schema_or_domain
    else:
        raise TypeError("value must be DatasetSchema or DatasetCellDomain")
    return domain.fingerprint


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


def dataset_cell_permutation_digest(
    schema: DatasetSchema,
    cells: tuple[DatasetCellAddress, ...],
) -> str:
    """Canonical identity of one complete event-ordinal to dataset-cell plan."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    ordered = tuple(cells)
    total = schema.repeat_axis.size * schema.point_layout.storage_size
    if len(ordered) != total:
        raise ValueError("cell permutation length differs from DatasetSchema")
    if any(not isinstance(cell, DatasetCellAddress) for cell in ordered):
        raise TypeError("cell permutation must contain DatasetCellAddress values")
    expected = {
        DatasetCellAddress(repeat, point)
        for repeat in range(schema.repeat_axis.size)
        for point in range(schema.point_layout.storage_size)
    }
    if set(ordered) != expected:
        raise ValueError("cell permutation must cover every dataset cell exactly once")
    return dataset_cell_permutation_fingerprint(schema.fingerprint, ordered)


def dataset_cell_permutation_fingerprint(
    dataset_schema_fingerprint: str,
    cells: tuple[DatasetCellAddress, ...],
) -> str:
    _sha256_digest(dataset_schema_fingerprint, "dataset_schema_fingerprint")
    ordered = tuple(cells)
    if any(not isinstance(cell, DatasetCellAddress) for cell in ordered):
        raise TypeError("cells must contain DatasetCellAddress values")
    return canonical_digest(
        {
            "contract": "zlc_neutral_atom.DatasetCellPermutation",
            "dataset_schema_fingerprint": dataset_schema_fingerprint,
            "cells": [
                [cell.repeat_index, cell.point_storage_index]
                for cell in ordered
            ],
        }
    )


def _dataset_key_sequence_digest_from_rows(
    key_contract_fingerprint: str,
    cells: tuple[DatasetCellAddress, ...],
) -> str:
    return canonical_digest(
        {
            "contract": "zlc_neutral_atom.DatasetKeySequence",
            "key_contract_fingerprint": key_contract_fingerprint,
            "cells": [
                [cell.repeat_index, cell.point_storage_index]
                for cell in cells
            ],
        }
    )


def dataset_key_sequence_digest(
    schema: DatasetSchema,
    cells: tuple[DatasetCellAddress, ...],
) -> str:
    """Identity of ordinal keys independent of the per-cell ValueSchema."""

    ordered = tuple(cells)
    dataset_cell_permutation_digest(schema, ordered)
    return _dataset_key_sequence_digest_from_rows(
        dataset_cell_key_fingerprint(schema),
        ordered,
    )


def _dataset_consumer_contract_digest_from_schedule(
    dataset_schema_fingerprint: str,
    schedule_digest: str,
    metadata_contract_fingerprint: str,
    event_adapter_operator_fingerprint: str,
) -> str:
    return canonical_digest(
        {
            "contract": "zlc_neutral_atom.DatasetConsumerContract",
            "dataset_schema_fingerprint": dataset_schema_fingerprint,
            "join_plan_digest": schedule_digest,
            "metadata_contract_fingerprint": metadata_contract_fingerprint,
            "event_adapter_operator_fingerprint": (
                event_adapter_operator_fingerprint
            ),
        }
    )


def dataset_consumer_contract_digest(
    schema: DatasetSchema,
    cells: tuple[DatasetCellAddress, ...],
    metadata_contract_fingerprint: str,
    event_adapter_operator_fingerprint: str,
) -> str:
    """Bind a sink to schema, ordering, metadata, and value-projection semantics."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    _sha256_digest(metadata_contract_fingerprint, "metadata contract fingerprint")
    _sha256_digest(
        event_adapter_operator_fingerprint,
        "event adapter operator fingerprint",
    )
    schedule_digest = dataset_cell_permutation_digest(schema, cells)
    return _dataset_consumer_contract_digest_from_schedule(
        schema.fingerprint,
        schedule_digest,
        metadata_contract_fingerprint,
        event_adapter_operator_fingerprint,
    )


@dataclass(frozen=True)
class DatasetCellKeyContract:
    """Immutable join-key owner bound to one DatasetSchema storage domain."""

    domain: DatasetCellDomain

    def __post_init__(self) -> None:
        if isinstance(self.domain, DatasetSchema):
            object.__setattr__(self, "domain", DatasetCellDomain.from_schema(self.domain))
        elif not isinstance(self.domain, DatasetCellDomain):
            raise TypeError("domain must be DatasetCellDomain or DatasetSchema")

    @property
    def fingerprint(self) -> str:
        return self.domain.fingerprint

    def snapshot(self, key: object) -> DatasetCellAddress:
        if not isinstance(key, DatasetCellAddress):
            raise TypeError("join key must be DatasetCellAddress")
        if key.repeat_index >= self.domain.repeat_axis.size:
            raise IndexError("join key repeat index is outside DatasetSchema")
        if key.point_storage_index >= self.domain.point_layout.storage_size:
            raise IndexError("join key point index is outside PointLayout")
        return DatasetCellAddress(key.repeat_index, key.point_storage_index)


@dataclass(frozen=True)
class FrozenDatasetEdge(Generic[PayloadT]):
    """Single owner for a payload-to-dataset edge and every derived digest.

    ``DatasetSchema``, value projection, metadata projection, and the optional
    exact ordinal schedule are one contract.  Callers cannot independently
    report digests that describe a different cell schema or projection.
    """

    schema: DatasetSchema
    event_adapter: DatasetEventAdapter[PayloadT]
    expected_cells: tuple[DatasetCellAddress, ...] | None = None
    schedule_digest: str | None = field(init=False)
    key_sequence_digest: str | None = field(init=False)
    consumer_contract_digest: str | None = field(init=False)
    _payload_contract: object = field(init=False, repr=False, compare=False)
    _payload_contract_fingerprint: str = field(init=False, repr=False, compare=False)
    _payload_max_retained_nbytes: int = field(init=False, repr=False, compare=False)
    _metadata_contract: DatasetMetadataContract = field(
        init=False,
        repr=False,
        compare=False,
    )
    _metadata_contract_fingerprint: str = field(
        init=False,
        repr=False,
        compare=False,
    )
    _metadata_max_retained_nbytes: int = field(
        init=False,
        repr=False,
        compare=False,
    )
    _value_schema: ValueSchema = field(init=False, repr=False, compare=False)
    _operator_fingerprint: str = field(init=False, repr=False, compare=False)
    _key_contract_fingerprint: str = field(
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
            operator_fingerprint = adapter.operator_fingerprint
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
        payload_fingerprint = _sha256_digest(
            payload_contract.fingerprint,
            "payload contract fingerprint",
        )
        operator_fingerprint = _sha256_digest(
            operator_fingerprint,
            "event adapter operator fingerprint",
        )
        for member in ("snapshot", "retained_nbytes", "digest"):
            if not callable(getattr(payload_contract, member, None)):
                raise TypeError(f"event_adapter.payload_contract.{member} must be callable")
        metadata_fingerprint = _sha256_digest(
            metadata.fingerprint,
            "metadata contract fingerprint",
        )
        payload_max_bytes = payload_contract.max_retained_nbytes
        if (
            isinstance(payload_max_bytes, bool)
            or not isinstance(payload_max_bytes, Integral)
            or payload_max_bytes <= 0
        ):
            raise ValueError("payload contract max_retained_nbytes must be positive")
        payload_max_bytes = int(payload_max_bytes)
        metadata_max_bytes = metadata.max_retained_nbytes
        if (
            isinstance(metadata_max_bytes, bool)
            or not isinstance(metadata_max_bytes, Integral)
            or metadata_max_bytes < 0
        ):
            raise ValueError(
                "metadata contract max_retained_nbytes must be non-negative"
            )
        metadata_max_bytes = int(metadata_max_bytes)
        for member in ("snapshot", "validate", "retained_nbytes", "digest"):
            if not callable(getattr(metadata, member, None)):
                raise TypeError(f"metadata_contract.{member} must be callable")
        value_max_bytes = ValuePayloadContract(schema.cell_schema).max_retained_nbytes
        if value_max_bytes + metadata_max_bytes > payload_max_bytes:
            raise DatasetError(
                "Value plus metadata worst-case bytes exceed the PayloadContract"
            )
        object.__setattr__(self, "_payload_contract", payload_contract)
        object.__setattr__(
            self,
            "_payload_contract_fingerprint",
            payload_fingerprint,
        )
        object.__setattr__(self, "_payload_max_retained_nbytes", payload_max_bytes)
        object.__setattr__(self, "_metadata_contract", metadata)
        object.__setattr__(
            self,
            "_metadata_contract_fingerprint",
            metadata_fingerprint,
        )
        object.__setattr__(
            self,
            "_metadata_max_retained_nbytes",
            metadata_max_bytes,
        )
        object.__setattr__(self, "_value_schema", value_schema)
        object.__setattr__(self, "_operator_fingerprint", operator_fingerprint)
        object.__setattr__(self, "_value_operator", value_operator)
        key_contract_fingerprint = dataset_cell_key_fingerprint(schema)
        object.__setattr__(
            self,
            "_key_contract_fingerprint",
            key_contract_fingerprint,
        )

        cells = self.expected_cells
        if cells is None:
            schedule_digest = None
            consumer_digest = None
            key_sequence_digest = None
        else:
            if not isinstance(cells, tuple):
                raise TypeError("expected_cells must be a tuple or None")
            if any(not isinstance(cell, DatasetCellAddress) for cell in cells):
                raise TypeError("expected_cells must contain DatasetCellAddress values")
            cells = tuple(
                DatasetCellAddress(
                    cell.repeat_index,
                    cell.point_storage_index,
                )
                for cell in cells
            )
            schedule_digest = dataset_cell_permutation_digest(schema, cells)
            consumer_digest = _dataset_consumer_contract_digest_from_schedule(
                schema.fingerprint,
                schedule_digest,
                metadata_fingerprint,
                operator_fingerprint,
            )
            key_sequence_digest = _dataset_key_sequence_digest_from_rows(
                key_contract_fingerprint,
                cells,
            )
            object.__setattr__(self, "expected_cells", cells)
        object.__setattr__(self, "schedule_digest", schedule_digest)
        object.__setattr__(self, "key_sequence_digest", key_sequence_digest)
        object.__setattr__(self, "consumer_contract_digest", consumer_digest)

    @property
    def payload_contract(self) -> object:
        return self._payload_contract

    @property
    def source_payload_contract(self) -> object:
        return self._payload_contract

    @property
    def payload_contract_fingerprint(self) -> str:
        return self._payload_contract_fingerprint

    @property
    def payload_max_retained_nbytes(self) -> int:
        return self._payload_max_retained_nbytes

    @property
    def metadata_contract(self) -> DatasetMetadataContract:
        return self._metadata_contract

    @property
    def metadata_contract_fingerprint(self) -> str:
        return self._metadata_contract_fingerprint

    @property
    def metadata_max_retained_nbytes(self) -> int:
        return self._metadata_max_retained_nbytes

    @property
    def value_schema(self) -> ValueSchema:
        return self._value_schema

    @property
    def operator_fingerprint(self) -> str:
        return self._operator_fingerprint

    def project_value(self, payload: PayloadT) -> Value:
        return self._value_operator(payload)

    @property
    def key_contract_fingerprint(self) -> str:
        return self._key_contract_fingerprint

    @property
    def exact_key_sequence_digest(self) -> str:
        if self.expected_cells is None or self.key_sequence_digest is None:
            raise DatasetError("rolling dataset edge has no exact key sequence")
        return self.key_sequence_digest

    def validate_payload_stream(self, stream: AcquisitionStream[PayloadT]) -> None:
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("stream must be AcquisitionStream")
        if stream._payload_contract is not self.payload_contract:
            raise DatasetError("dataset edge must share the stream PayloadContract owner")
        if stream.payload_contract_fingerprint != self.payload_contract_fingerprint:
            raise DatasetError("dataset edge payload fingerprint differs from stream")
        if stream.max_payload_bytes != self.payload_max_retained_nbytes:
            raise DatasetError("dataset edge payload byte bound differs from stream")

    def validate_stream(self, stream: AcquisitionStream[PayloadT]) -> None:
        self.validate_payload_stream(stream)
        key_contract = stream._join_key_contract
        if not isinstance(key_contract, DatasetCellKeyContract):
            raise DatasetError("dataset source must declare DatasetCellKeyContract")
        if key_contract.fingerprint != self.key_contract_fingerprint:
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
    cell_metadata: tuple[object | None, ...]

    @property
    def ref(self) -> DatasetRevisionRef:
        return self.snapshot.ref

    @property
    def block(self) -> DataBlock:
        return self.snapshot.block


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
            * self.snapshot.block.schema.point_layout.storage_size
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
class DatasetDerivationProvenance:
    """Bounded root-to-terminal provenance for a processed exact dataset."""

    chain_contract_digest: str
    root_input_span: EventSpanRef
    stages: tuple[ProcessorStageProvenance, ...]

    def __post_init__(self) -> None:
        _sha256_digest(self.chain_contract_digest, "chain_contract_digest")
        if not isinstance(self.root_input_span, EventSpanRef):
            raise TypeError("root_input_span must be EventSpanRef")
        stages = tuple(self.stages)
        if not stages or any(
            not isinstance(stage, ProcessorStageProvenance) for stage in stages
        ):
            raise TypeError("stages must contain ProcessorStageProvenance values")
        object.__setattr__(self, "stages", stages)

    @property
    def artifact_inputs(self) -> tuple[ArtifactInputRef, ...]:
        ordered: list[ArtifactInputRef] = []
        seen: set[str] = set()
        for stage in self.stages:
            for reference in stage.direct_artifact_inputs:
                identity = reference.fingerprint
                if identity not in seen:
                    seen.add(identity)
                    ordered.append(reference)
        return tuple(ordered)


@dataclass(frozen=True)
class DatasetSealProvenance:
    stream_id: StreamId
    generation: StreamGenerationId
    start_sequence: int
    end_sequence: int
    join_plan_digest: str
    ordered_event_digest: str
    ordered_metadata_digest: str
    metadata_contract_fingerprint: str
    trace_binding: TraceBinding
    derivation: DatasetDerivationProvenance | None = None

    def __post_init__(self) -> None:
        if self.derivation is not None and not isinstance(
            self.derivation,
            DatasetDerivationProvenance,
        ):
            raise TypeError("derivation must be DatasetDerivationProvenance or None")


class SealedDatasetArtifact:
    """Opaque exact-transport capability; physical scans may require outer attestation."""

    __slots__ = (
        "_snapshot",
        "_coverage",
        "_provenance",
        "_event_metadata",
        "_terminal_reservation",
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
        ordered_metadata_digest: str,
        metadata_contract_fingerprint: str,
        trace_binding: TraceBinding,
        event_metadata: tuple[object | None, ...],
        terminal_reservation: ExactReservation,
        derivation: DatasetDerivationProvenance | None = None,
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
                ordered_metadata_digest=ordered_metadata_digest,
                metadata_contract_fingerprint=metadata_contract_fingerprint,
                trace_binding=trace_binding,
                derivation=derivation,
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

    def _belongs_to_terminal_reservation(self, reservation: object) -> bool:
        """Process-local ownership proof used only by PipelineResult minting."""

        return reservation is not None and self._terminal_reservation is reservation

    def _with_derivation(
        self,
        readiness: ExactConsumerReadiness,
        root_input_span: EventSpanRef,
    ) -> "SealedDatasetArtifact":
        """Return an enriched immutable result authorized by one exact chain."""

        if not isinstance(readiness, ExactConsumerReadiness):
            raise TypeError("readiness must be ExactConsumerReadiness")
        if not isinstance(root_input_span, EventSpanRef):
            raise TypeError("root_input_span must be EventSpanRef")
        reservation = readiness._source_reservation
        source = reservation._stream
        if (
            root_input_span.stream_id != source.stream_id
            or root_input_span.generation != source.generation
            or root_input_span.start_sequence != reservation.start_sequence
            or root_input_span.end_sequence != reservation.end_sequence
        ):
            raise DatasetError("root input span differs from readiness source interval")
        if self._terminal_reservation is not readiness._terminal_reservation:
            raise DatasetError("sealed dataset belongs to another terminal readiness")
        stages = readiness.processor_stages
        if not stages:
            raise DatasetError("processed derivation requires at least one stage")
        derivation = DatasetDerivationProvenance(
            readiness.chain_contract_digest,
            root_input_span,
            stages,
        )
        provenance = self._provenance
        return SealedDatasetArtifact(
            _SEALED_TOKEN,
            snapshot=self._snapshot,
            coverage=self._coverage,
            stream_id=provenance.stream_id,
            generation=provenance.generation,
            start_sequence=provenance.start_sequence,
            end_sequence=provenance.end_sequence,
            join_plan_digest=provenance.join_plan_digest,
            ordered_event_digest=provenance.ordered_event_digest,
            ordered_metadata_digest=provenance.ordered_metadata_digest,
            metadata_contract_fingerprint=provenance.metadata_contract_fingerprint,
            trace_binding=provenance.trace_binding,
            event_metadata=self._event_metadata,
            terminal_reservation=self._terminal_reservation,
            derivation=derivation,
        )


class OrderedDatasetEventHasher:
    """Streaming owner for exact materialization content plus metadata identity."""

    __slots__ = (
        "_stream_id",
        "_generation",
        "_start_sequence",
        "_next_sequence",
        "_hasher",
    )

    def __init__(
        self,
        stream_id: StreamId,
        generation: StreamGenerationId,
        start_sequence: int,
    ) -> None:
        if not isinstance(stream_id, StreamId):
            raise TypeError("stream_id must be StreamId")
        if not isinstance(generation, StreamGenerationId):
            raise TypeError("generation must be StreamGenerationId")
        if (
            isinstance(start_sequence, bool)
            or not isinstance(start_sequence, Integral)
            or start_sequence < 0
        ):
            raise ValueError("start_sequence must be a non-negative integer")
        self._stream_id = stream_id
        self._generation = generation
        self._start_sequence = int(start_sequence)
        self._next_sequence = int(start_sequence)
        self._hasher = hashlib.sha256()
        self._hasher.update(
            b"zlc_neutral_atom.DatasetOrderedPayloadEvents\x00"
        )

    def update(self, reference: EventRef, metadata_digest: str) -> None:
        if not isinstance(reference, EventRef):
            raise TypeError("reference must be EventRef")
        if (
            reference.stream_id != self._stream_id
            or reference.generation != self._generation
            or reference.sequence != self._next_sequence
        ):
            raise ValueError(
                "EventRef differs from the ordered dataset stream/generation/sequence"
            )
        metadata_digest = _sha256_digest(metadata_digest, "metadata_digest")
        encoded = encode(
            {
                "event_ref": event_ref_to_tree(reference),
                "metadata_digest": metadata_digest,
            }
        )
        self._hasher.update(len(encoded).to_bytes(8, "big"))
        self._hasher.update(encoded)
        self._next_sequence += 1

    def digest(self, end_sequence: int) -> str:
        if (
            isinstance(end_sequence, bool)
            or not isinstance(end_sequence, Integral)
            or int(end_sequence) != self._next_sequence
        ):
            raise ValueError("ordered dataset event digest has incomplete coverage")
        return self._hasher.copy().hexdigest()


class OrderedDatasetMetadataHasher:
    """Single owner of the metadata-order digest used by exact datasets."""

    __slots__ = ("_hasher",)

    def __init__(self, metadata_contract_fingerprint: str) -> None:
        fingerprint = _sha256_digest(
            metadata_contract_fingerprint,
            "metadata_contract_fingerprint",
        )
        self._hasher = hashlib.sha256()
        self._hasher.update(fingerprint.encode("ascii"))

    def update(self, metadata_digest: str) -> None:
        digest = _sha256_digest(metadata_digest, "metadata_digest")
        self._hasher.update(digest.encode("ascii"))

    def digest(self) -> str:
        return self._hasher.copy().hexdigest()


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
    return ComponentValidity(contract.component_axis_ids, validity)


def _project_payload(
    edge: FrozenDatasetEdge[PayloadT],
    payload: PayloadT,
    *,
    include_metadata_digest: bool,
) -> tuple[Value, object | None, str | None]:
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
    retained = contract.retained_nbytes(metadata)
    if (
        isinstance(retained, bool)
        or not isinstance(retained, Integral)
        or retained < 0
        or retained > edge.metadata_max_retained_nbytes
    ):
        raise ValueError("metadata retained bytes exceed the declared metadata bound")
    digest = (
        _sha256_digest(contract.digest(metadata), "metadata digest")
        if include_metadata_digest
        else None
    )
    return value, metadata, digest


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
        if edge.expected_cells is None:
            raise DatasetError("DatasetBuilder requires a frozen exact cell schedule")
        if edge.schedule_digest is None or edge.consumer_contract_digest is None:
            raise DatasetError("exact dataset edge is missing its formal digests")
        self.block_id = block_id
        self._reservation = source
        self._source: AcquisitionStream[PayloadT] = source._stream
        edge.validate_stream(self._source)
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        self.edge = edge
        self.schema = edge.schema
        self._expected_cells = edge.expected_cells
        self._join_plan_digest = edge.schedule_digest
        self._metadata_contract_fingerprint = edge.metadata_contract_fingerprint
        total_cells = self.schema.repeat_axis.size * self.schema.point_layout.storage_size
        reserved_events = source.end_sequence - source.start_sequence
        if reserved_events != total_cells:
            raise DatasetError("exact reservation length must equal DatasetSchema cell count")
        self._ordered_event_hasher = OrderedDatasetEventHasher(
            self.stream_id,
            self.generation,
            source.start_sequence,
        )
        self._ordered_metadata_hasher = OrderedDatasetMetadataHasher(
            self._metadata_contract_fingerprint
        )
        self._lock = threading.RLock()
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
            source_contract_digest=edge.consumer_contract_digest,
            source_schedule_digest=edge.schedule_digest,
            source_key_sequence_digest=edge.exact_key_sequence_digest,
            chain_contract_digest=edge.consumer_contract_digest,
            terminal=True,
        )

    @property
    def revision(self) -> DatasetRevision:
        with self._lock:
            return DatasetRevision(self._revision)

    def current_ref(self) -> DatasetRevisionRef:
        with self._lock:
            return self._ref_locked(self._revision)

    def consume(self, delivery: Delivery[PayloadT]) -> DatasetProgress:
        if not isinstance(delivery, Delivery) or not delivery.is_exact:
            raise TypeError("DatasetBuilder requires an exact Delivery capability")
        if delivery.acknowledged:
            raise DatasetError("delivery was already acknowledged")
        projected = _project_payload(
            self.edge,
            delivery.envelope.payload,
            include_metadata_digest=True,
        )
        return self._source._consume_exact(
            self._reservation,
            delivery,
            self,
            lambda envelope: self._ingest(envelope, projected),
        )

    def _ingest(
        self,
        envelope: Envelope[PayloadT],
        projected: tuple[Value, object | None, str | None],
    ) -> DatasetProgress:
        address = envelope.join_key
        if not isinstance(address, DatasetCellAddress):
            raise DatasetError("dataset event is missing its typed DatasetCellAddress")
        value, metadata, metadata_digest = projected
        assert metadata_digest is not None
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
            expected_address = self._expected_cells[schedule_index]
            if address != expected_address:
                raise DatasetError(
                    f"event join key {address} differs from frozen plan key "
                    f"{expected_address} at sequence {envelope.sequence}"
                )
            cell = (address.repeat_index, address.point_storage_index)
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
            self._ordered_metadata_hasher.update(metadata_digest)
            self._ordered_event_hasher.update(envelope.ref, metadata_digest)
            flat_cell = (
                address.repeat_index * self.schema.point_layout.storage_size
                + address.point_storage_index
            )
            self._cell_metadata[flat_cell] = metadata
            return DatasetProgress(
                ref=self._ref_locked(self._revision),
                dirty_cells=(address,),
                coverage=self._coverage_locked(),
            )

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

    def seal(self, eos: EndOfStream) -> SealedDatasetArtifact:
        self._source._complete_consumer(self._reservation, eos, self, self._seal_locked)
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
            ordered_event_digest=self._ordered_event_hasher.digest(
                self._reservation.end_sequence
            ),
            ordered_metadata_digest=self._ordered_metadata_hasher.digest(),
            metadata_contract_fingerprint=self._metadata_contract_fingerprint,
            trace_binding=self._reservation.trace_binding,
            event_metadata=tuple(self._ordered_event_metadata),
            terminal_reservation=self._reservation,
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


class MonitorDataset(Generic[PayloadT]):
    """Sequence-owned live materializer; never a formal artifact authority."""

    @classmethod
    def keyed_cycle(
        cls,
        block_id: BlockId,
        source: MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> "MonitorDataset[PayloadT]":
        if edge.expected_cells is None:
            raise DatasetError("keyed_cycle requires a frozen complete cell schedule")
        return cls(block_id, source, edge)

    @classmethod
    def append_window(
        cls,
        block_id: BlockId,
        source: MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
    ) -> "MonitorDataset[PayloadT]":
        if edge.expected_cells is not None:
            raise DatasetError("append_window requires a schedule-free dataset edge")
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
        self._cycle_cells = edge.expected_cells
        if self._cycle_cells is None:
            edge.validate_payload_stream(self._source)
            if self.schema.repeat_axis.size != 1:
                raise DatasetError("append_window requires a single repeat storage row")
            if (
                len(self.schema.point_axes) != 1
                or self.schema.point_axes[0].role != MONITOR_HISTORY
                or self.schema.point_layout.mode is not AxisLayoutMode.RECT_C
                or self.schema.point_axes[0].coordinates
                != tuple(range(self.schema.point_axes[0].size))
            ):
                raise DatasetError(
                    "append_window requires one dense MONITOR_HISTORY axis with "
                    "newest-first slot coordinates 0..history-1"
                )
        else:
            edge.validate_stream(self._source)
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        total_cells = self.schema.repeat_axis.size * self.schema.point_layout.storage_size
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
            self._head: EventRef | None = None
            self._next_slot = 0
            self._count = 0
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
            update = self._monitor._next_for(self, timeout)
            return self._ingest(update)

    def ingest_latest(self) -> DatasetRevisionRef:
        with self._consume_lock:
            update = self._monitor._latest_for(self)
            return self._ingest(update)

    def _ingest(self, update: MonitorUpdate[PayloadT]) -> DatasetRevisionRef:
        envelope = update.envelope
        value, metadata, _digest = _project_payload(
            self.edge,
            envelope.payload,
            include_metadata_digest=False,
        )
        validity_mask = _value_validity_mask(self.schema, value.validity)
        with self._lock:
            self._ensure_writable_locked()
            self._validate_envelope_identity_locked(envelope)
            expected_sequence = 0 if self._last_sequence is None else self._last_sequence + 1
            if envelope.sequence < expected_sequence:
                raise DatasetError("monitor dataset events must remain strictly ordered")
            sequence_gap = envelope.sequence - expected_sequence
            self._missed_events += max(update.missed, sequence_gap)
            if self._cycle_cells is None:
                cell = (0, self._next_slot)
            else:
                offset = envelope.sequence % len(self._cycle_cells)
                expected_address = self._cycle_cells[offset]
                if envelope.join_key != expected_address:
                    raise DatasetError(
                        f"monitor cycle key {envelope.join_key!r} differs from "
                        f"frozen key {expected_address!r} at sequence {envelope.sequence}"
                    )
                if offset == 0 or sequence_gap > 0 or update.missed > 0:
                    self._clear_locked()
                cell = (
                    expected_address.repeat_index,
                    expected_address.point_storage_index,
                )
            _write_cell(
                cell,
                value,
                validity_mask,
                self._values,
                self._written,
                self._validity,
            )
            flat_cell = cell[0] * self.schema.point_layout.storage_size + cell[1]
            self._cell_metadata[flat_cell] = metadata
            self._event_refs[flat_cell] = envelope.ref
            if self._cycle_cells is None:
                capacity = self.schema.point_layout.storage_size
                self._next_slot = (self._next_slot + 1) % capacity
                self._count = min(self._count + 1, capacity)
            self._last_sequence = envelope.sequence
            self._head = envelope.ref
            self._revision += 1
            return self._ref_locked(self._revision)

    def materialize(
        self,
        ref: DatasetRevisionRef | None = None,
    ) -> MonitorDatasetSnapshot:
        with self._lock:
            selected = self._select_current_ref_locked(ref)
            if self._cycle_cells is None:
                order = self._append_order_locked()
                values = self._values[:, order, ...]
                written = self._written[:, order]
                validity = self._validity[:, order, ...]
                metadata = tuple(self._cell_metadata[index] for index in order)
                event_refs = tuple(self._event_refs[index] for index in order)
            else:
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
                coverage=self._coverage_locked(written, event_refs),
                cell_metadata=metadata,
                event_refs=event_refs,
                head=self._head,
            )

    def close(self) -> None:
        with self._lock:
            self._aborted = True
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

    def _append_order_locked(self) -> tuple[int, ...]:
        capacity = self.schema.point_layout.storage_size
        used = tuple((self._next_slot - 1 - age) % capacity for age in range(self._count))
        used_set = set(used)
        return used + tuple(slot for slot in range(capacity) if slot not in used_set)

    def _clear_locked(self) -> None:
        self._values.fill(0)
        self._written.fill(False)
        self._validity.fill(False)
        self._cell_metadata[:] = [None] * len(self._cell_metadata)
        self._event_refs[:] = [None] * len(self._event_refs)

    def _coverage_locked(
        self,
        written: np.ndarray,
        event_refs: tuple[EventRef | None, ...],
    ) -> MonitorCoverage:
        current_gap = False
        if self._cycle_cells is None:
            retained = tuple(reference for reference in event_refs if reference is not None)
            current_gap = any(
                newer.sequence != older.sequence + 1
                for newer, older in zip(retained, retained[1:])
            )
        return MonitorCoverage(
            written_cells=int(np.count_nonzero(written)),
            total_cells=int(written.size),
            missed_events=self._missed_events,
            current_gap=current_gap,
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
    "DatasetCellDomain",
    "DatasetCellKeyContract",
    "DatasetCoverage",
    "DatasetDerivationProvenance",
    "DatasetEventAdapter",
    "DatasetMetadataContract",
    "DatasetError",
    "DatasetProgress",
    "DatasetPreviewSnapshot",
    "DatasetSealProvenance",
    "FrozenDatasetEdge",
    "MissingDatasetCells",
    "MonitorCoverage",
    "MonitorDataset",
    "MonitorDatasetSnapshot",
    "NoDatasetMetadataContract",
    "OrderedDatasetEventHasher",
    "OrderedDatasetMetadataHasher",
    "SnapshotExpired",
    "SealedDatasetArtifact",
    "ValueDatasetEventAdapter",
    "dataset_cell_key_fingerprint",
    "dataset_consumer_contract_digest",
    "dataset_cell_permutation_digest",
    "dataset_cell_permutation_fingerprint",
    "dataset_key_sequence_digest",
]
