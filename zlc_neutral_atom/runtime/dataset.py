"""Single-owner event-to-dataset materialization with revisioned snapshots."""

from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from numbers import Integral
from typing import Callable, Generic, Protocol, TypeVar

import numpy as np
from zlc_storage import canonical_digest, encode, sha256_text as _sha256_digest

from zlc_data import (
    BlockId,
    AxisSpec,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DataPatch,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    Invalid,
    OwnedSnapshot,
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


def _deeply_immutable_metadata(
    value: object,
    active: set[int] | None = None,
    validated: set[int] | None = None,
) -> bool:
    """Accept immutable runtime snapshots, including bytes-owned numeric values."""

    if value is None or type(value) in (bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
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
        result = _deeply_immutable_metadata(value.value, active, validated)
    elif isinstance(value, (tuple, frozenset)):
        result = all(
            _deeply_immutable_metadata(item, active, validated) for item in value
        )
    elif is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        result = bool(parameters and parameters.frozen) and all(
            _deeply_immutable_metadata(
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


def _intrinsically_immutable_contract_value(
    value: object,
    *,
    active: set[int] | None = None,
    validated: set[int] | None = None,
    budget: list[int] | None = None,
    depth: int = 0,
) -> bool:
    """Validate retained contract definitions, not runtime metadata values.

    Contract owners may contain only immutable declarative structure.  In
    particular, mapping proxies and read-only arrays are still aliases to an
    external owner and are intentionally rejected here.  Shared immutable
    subgraphs are accepted once, while an active-path identity is a cycle.
    """

    if value is None or type(value) in (bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, np.dtype):
        return True
    if depth > 64:
        return False
    identity = id(value)
    active = set() if active is None else active
    validated = set() if validated is None else validated
    budget = [0] if budget is None else budget
    if identity in active:
        return False
    if identity in validated:
        return True
    budget[0] += 1
    if budget[0] > 262_144:
        return False
    active.add(identity)
    if isinstance(value, Enum):
        result = _intrinsically_immutable_contract_value(
            value.value,
            active=active,
            validated=validated,
            budget=budget,
            depth=depth + 1,
        )
    elif isinstance(value, (tuple, frozenset)):
        result = all(
            _intrinsically_immutable_contract_value(
                item,
                active=active,
                validated=validated,
                budget=budget,
                depth=depth + 1,
            )
            for item in value
        )
    elif is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        result = bool(parameters and parameters.frozen) and all(
            _intrinsically_immutable_contract_value(
                getattr(value, item.name),
                active=active,
                validated=validated,
                budget=budget,
                depth=depth + 1,
            )
            for item in fields(value)
        )
    else:
        result = False
    active.remove(identity)
    if result:
        validated.add(identity)
    return result


def _snapshot_intrinsically_immutable_contract_value(
    value: object,
    *,
    preserve: dict[int, object] | None = None,
    memo: dict[int, object] | None = None,
    active: set[int] | None = None,
) -> object:
    """Rebuild an immutable owner graph so reflective caller mutation cannot drift it."""

    preserve = {} if preserve is None else preserve
    memo = {} if memo is None else memo
    active = set() if active is None else active
    identity = id(value)
    if identity in preserve:
        return preserve[identity]
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, (np.dtype, Enum)):
        return value
    if identity in active:
        raise TypeError("contract owner graph contains a recursive cycle")
    if identity in memo:
        return memo[identity]
    active.add(identity)
    if isinstance(value, tuple):
        result = tuple(
            _snapshot_intrinsically_immutable_contract_value(
                item,
                preserve=preserve,
                memo=memo,
                active=active,
            )
            for item in value
        )
    elif isinstance(value, frozenset):
        result = frozenset(
            _snapshot_intrinsically_immutable_contract_value(
                item,
                preserve=preserve,
                memo=memo,
                active=active,
            )
            for item in value
        )
    elif is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if not parameters or not parameters.frozen:
            active.remove(identity)
            raise TypeError("contract owner must be a frozen dataclass value")
        updates = {
            item.name: _snapshot_intrinsically_immutable_contract_value(
                getattr(value, item.name),
                preserve=preserve,
                memo=memo,
                active=active,
            )
            for item in fields(value)
            if item.init
        }
        try:
            result = replace(value, **updates)
        except (TypeError, ValueError) as error:
            active.remove(identity)
            raise TypeError(
                f"cannot reconstruct immutable contract owner {type(value).__name__}"
            ) from error
    else:
        active.remove(identity)
        raise TypeError(
            f"unsupported immutable contract owner value {type(value).__name__}"
        )
    active.remove(identity)
    memo[identity] = result
    if not _intrinsically_immutable_contract_value(result):
        raise TypeError("reconstructed contract owner is not intrinsically immutable")
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
    if (
        not isinstance(dataset_schema_fingerprint, str)
        or len(dataset_schema_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in dataset_schema_fingerprint)
    ):
        raise ValueError("dataset_schema_fingerprint must be a lowercase SHA-256 digest")
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


def dataset_key_sequence_digest(
    schema: DatasetSchema,
    cells: tuple[DatasetCellAddress, ...],
) -> str:
    """Identity of ordinal keys independent of the per-cell ValueSchema."""

    ordered = tuple(cells)
    # Reuse the complete-permutation validation, then deliberately bind only
    # the key domain and address order.
    dataset_cell_permutation_digest(schema, ordered)
    return canonical_digest(
        {
            "contract": "zlc_neutral_atom.DatasetKeySequence",
            "key_contract_fingerprint": dataset_cell_key_fingerprint(schema),
            "cells": [
                [cell.repeat_index, cell.point_storage_index]
                for cell in ordered
            ],
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
    return canonical_digest(
        {
            "contract": "zlc_neutral_atom.DatasetConsumerContract",
            "dataset_schema_fingerprint": schema.fingerprint,
            "join_plan_digest": dataset_cell_permutation_digest(schema, cells),
            "metadata_contract_fingerprint": metadata_contract_fingerprint,
            "event_adapter_operator_fingerprint": (
                event_adapter_operator_fingerprint
            ),
        }
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
        self.validate(key)
        return key

    def validate(self, key: object) -> None:
        if not isinstance(key, DatasetCellAddress):
            raise TypeError("join key must be DatasetCellAddress")
        if key.repeat_index >= self.domain.repeat_axis.size:
            raise IndexError("join key repeat index is outside DatasetSchema")
        if key.point_storage_index >= self.domain.point_layout.storage_size:
            raise IndexError("join key point index is outside PointLayout")


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
    schedule_digest: str = field(init=False)
    key_sequence_digest: str | None = field(init=False)
    consumer_contract_digest: str = field(init=False)
    _source_payload_contract: object = field(init=False, repr=False, compare=False)
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
        if not _intrinsically_immutable_contract_value(adapter):
            raise TypeError(
                "DatasetEventAdapter fields must be recursively intrinsically immutable"
            )
        try:
            source_payload_contract = adapter.payload_contract
        except AttributeError as error:
            raise TypeError(
                "event_adapter does not implement DatasetEventAdapter"
            ) from error
        adapter = _snapshot_intrinsically_immutable_contract_value(
            adapter,
            preserve={id(schema.cell_schema): schema.cell_schema},
        )
        object.__setattr__(self, "event_adapter", adapter)
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
        for name, owner in (
            ("payload contract", payload_contract),
            ("metadata contract", metadata),
        ):
            owner_parameters = getattr(type(owner), "__dataclass_params__", None)
            if (
                not is_dataclass(owner)
                or not owner_parameters
                or not owner_parameters.frozen
                or not _intrinsically_immutable_contract_value(owner)
            ):
                raise TypeError(
                    f"{name} must be a recursively intrinsically immutable "
                    "frozen dataclass value"
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
            "_source_payload_contract",
            source_payload_contract,
        )
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

        cells = self.expected_cells
        if cells is None:
            schedule_digest = canonical_digest(
                {
                    "contract": "zlc_neutral_atom.RollingDatasetJoin",
                    "dataset_schema_fingerprint": schema.fingerprint,
                }
            )
            consumer_digest = canonical_digest(
                {
                    "contract": "zlc_neutral_atom.RollingDatasetConsumer",
                    "dataset_schema_fingerprint": schema.fingerprint,
                    "metadata_contract_fingerprint": metadata_fingerprint,
                    "event_adapter_operator_fingerprint": operator_fingerprint,
                }
            )
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
            consumer_digest = dataset_consumer_contract_digest(
                schema,
                cells,
                metadata_fingerprint,
                operator_fingerprint,
            )
            key_sequence_digest = dataset_key_sequence_digest(schema, cells)
            object.__setattr__(self, "expected_cells", cells)
        object.__setattr__(self, "schedule_digest", schedule_digest)
        object.__setattr__(self, "key_sequence_digest", key_sequence_digest)
        object.__setattr__(self, "consumer_contract_digest", consumer_digest)

    @property
    def payload_contract(self) -> object:
        self._validate_projection_binding()
        return self._payload_contract

    @property
    def source_payload_contract(self) -> object:
        return self._source_payload_contract

    @property
    def payload_contract_fingerprint(self) -> str:
        return self._payload_contract_fingerprint

    @property
    def payload_max_retained_nbytes(self) -> int:
        return self._payload_max_retained_nbytes

    @property
    def metadata_contract(self) -> DatasetMetadataContract:
        self._validate_projection_binding()
        return self._metadata_contract

    @property
    def metadata_contract_fingerprint(self) -> str:
        return self._metadata_contract_fingerprint

    @property
    def metadata_max_retained_nbytes(self) -> int:
        return self._metadata_max_retained_nbytes

    @property
    def value_schema(self) -> ValueSchema:
        self._validate_projection_binding()
        return self._value_schema

    @property
    def operator_fingerprint(self) -> str:
        self._validate_projection_binding()
        return self._operator_fingerprint

    def project_value(self, payload: PayloadT) -> Value:
        self._validate_projection_binding()
        return self._value_operator(payload)

    def _validate_projection_binding(self) -> None:
        """Reject reflective drift between the adapter and cached hot-path owners."""

        adapter = self.event_adapter
        try:
            payload_contract = adapter.payload_contract
            metadata_contract = adapter.metadata_contract
            value_schema = adapter.value_schema
            operator_fingerprint = adapter.operator_fingerprint
            current_operator = adapter.value
        except AttributeError as error:
            raise DatasetError("dataset event adapter binding changed") from error
        if (
            payload_contract is not self._payload_contract
            or metadata_contract is not self._metadata_contract
            or value_schema is not self._value_schema
            or operator_fingerprint != self._operator_fingerprint
        ):
            raise DatasetError("dataset event adapter binding changed")
        cached_operator = self._value_operator
        cached_owner = getattr(cached_operator, "__self__", None)
        current_owner = getattr(current_operator, "__self__", None)
        cached_function = getattr(cached_operator, "__func__", cached_operator)
        current_function = getattr(current_operator, "__func__", current_operator)
        if cached_owner is not current_owner or cached_function is not current_function:
            raise DatasetError("dataset value projection binding changed")

    @property
    def key_contract_fingerprint(self) -> str:
        return dataset_cell_key_fingerprint(self.schema)

    @property
    def exact_key_sequence_digest(self) -> str:
        if self.expected_cells is None or self.key_sequence_digest is None:
            raise DatasetError("rolling dataset edge has no exact key sequence")
        return self.key_sequence_digest

    def validate_stream(self, stream: AcquisitionStream[PayloadT]) -> None:
        self._validate_projection_binding()
        if not isinstance(stream, AcquisitionStream):
            raise TypeError("stream must be AcquisitionStream")
        if (
            stream._payload_contract is not self.source_payload_contract
            and stream._payload_contract is not self.payload_contract
        ):
            raise DatasetError("dataset edge must share the stream PayloadContract owner")
        if stream.payload_contract_fingerprint != self.payload_contract_fingerprint:
            raise DatasetError("dataset edge payload fingerprint differs from stream")
        if stream.max_payload_bytes != self.payload_max_retained_nbytes:
            raise DatasetError("dataset edge payload byte bound differs from stream")
        key_contract = stream._join_key_contract
        if not isinstance(key_contract, DatasetCellKeyContract):
            raise DatasetError("dataset source must declare DatasetCellKeyContract")
        if key_contract.fingerprint != self.key_contract_fingerprint:
            raise DatasetError("dataset source join-key contract differs from edge schema")


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
    cell_metadata: tuple[object | None, ...]

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


class DatasetBuilder(Generic[PayloadT]):
    """Private mutable materializer; public reads are immutable owned snapshots."""

    def __init__(
        self,
        block_id: BlockId,
        source: ExactReservation[PayloadT] | MonitorTap[PayloadT],
        edge: FrozenDatasetEdge[PayloadT],
        mode: DatasetMode,
    ) -> None:
        if not isinstance(block_id, BlockId):
            raise TypeError("block_id must be BlockId")
        if not isinstance(edge, FrozenDatasetEdge):
            raise TypeError("edge must be FrozenDatasetEdge")
        if not isinstance(mode, DatasetMode):
            raise TypeError("mode must be DatasetMode")
        if mode is DatasetMode.FINITE_EXACT and not isinstance(source, ExactReservation):
            raise TypeError("FINITE_EXACT DatasetBuilder must bind an ExactReservation")
        if mode is DatasetMode.ROLLING_MONITOR and not isinstance(source, MonitorTap):
            raise TypeError("ROLLING_MONITOR DatasetBuilder must bind a MonitorTap")
        if mode is DatasetMode.FINITE_EXACT and edge.expected_cells is None:
            raise DatasetError("FINITE_EXACT DatasetBuilder requires an exact edge schedule")
        if mode is DatasetMode.ROLLING_MONITOR and edge.expected_cells is not None:
            raise DatasetError("ROLLING_MONITOR DatasetBuilder requires a schedule-free edge")
        self.block_id = block_id
        self._reservation = source if isinstance(source, ExactReservation) else None
        self._monitor = source if isinstance(source, MonitorTap) else None
        self._source: AcquisitionStream[PayloadT] = source._stream
        edge.validate_stream(self._source)
        self.stream_id = self._source.stream_id
        self.generation = self._source.generation
        self.edge = edge
        self.schema = edge.schema
        self.mode = mode
        metadata_contract = edge.metadata_contract
        self._metadata_contract = metadata_contract
        self._metadata_contract_fingerprint = edge.metadata_contract_fingerprint
        self._metadata_max_retained_nbytes = edge.metadata_max_retained_nbytes
        schema = edge.schema
        expected_cells = edge.expected_cells
        total_cells = schema.repeat_axis.size * schema.point_layout.storage_size
        if self._reservation is not None:
            reserved_events = self._reservation.end_sequence - self._reservation.start_sequence
            if reserved_events != total_cells:
                raise DatasetError("exact reservation length must equal DatasetSchema cell count")
        self._expected_cells = expected_cells
        self._join_plan_digest = edge.schedule_digest
        self._ordered_event_hasher = (
            OrderedDatasetEventHasher(
                self.stream_id,
                self.generation,
                self._reservation.start_sequence,
            )
            if self._reservation is not None
            else None
        )
        self._ordered_metadata_hasher = OrderedDatasetMetadataHasher(
            self._metadata_contract_fingerprint
        )
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
        self._cell_metadata: list[object | None] = [None] * total_cells
        self._ordered_event_metadata: list[object | None] = [None] * total_cells
        self._sealed = False
        self._aborted = False
        self._exact_readiness: ExactConsumerReadiness | None = None
        if self._reservation is not None:
            self._exact_readiness = self._source._claim_consumer(
                self._reservation,
                self,
                source_contract_digest=edge.consumer_contract_digest,
                source_schedule_digest=self._join_plan_digest,
                source_key_sequence_digest=edge.exact_key_sequence_digest,
                chain_contract_digest=edge.consumer_contract_digest,
                terminal=True,
            )

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
        delivery: Delivery[PayloadT],
    ) -> DatasetProgress:
        if self.mode is not DatasetMode.FINITE_EXACT:
            raise DatasetError("exact cursor consumption requires FINITE_EXACT mode")
        if not isinstance(delivery, Delivery) or not delivery.is_exact:
            raise TypeError("FINITE_EXACT DatasetBuilder requires an exact Delivery capability")
        if delivery.acknowledged:
            raise DatasetError("delivery was already acknowledged")
        if self._reservation is None:
            raise DatasetError("exact DatasetBuilder has no bound reservation")
        projected = self._project_payload(delivery.envelope.payload)
        return self._source._consume_exact(
            self._reservation,
            delivery,
            self,
            lambda envelope: self._ingest(
                envelope,
                projected=projected,
                additional_missed=0,
            ),
        )

    def ingest_monitor(
        self,
        update: MonitorUpdate[PayloadT],
    ) -> DatasetProgress:
        if self.mode is not DatasetMode.ROLLING_MONITOR:
            raise DatasetError("monitor updates require ROLLING_MONITOR mode")
        if not isinstance(update, MonitorUpdate):
            raise TypeError("update must be MonitorUpdate")
        if self._monitor is None or not self._monitor._owns_update(update):
            raise PermissionError("MonitorUpdate belongs to another monitor authority")
        projected = self._project_payload(update.envelope.payload)
        return self._ingest(
            update.envelope,
            projected=projected,
            additional_missed=update.missed,
        )

    def _ingest(
        self,
        envelope: Envelope[PayloadT],
        *,
        projected: tuple[Value, object | None, str],
        additional_missed: int,
    ) -> DatasetProgress:
        if not isinstance(envelope, Envelope):
            raise TypeError("envelope must be Envelope")
        address = envelope.join_key
        if not isinstance(address, DatasetCellAddress):
            raise DatasetError("dataset event is missing its typed DatasetCellAddress")
        with self._lock:
            self._ensure_writable_locked()
            self._validate_envelope_identity_locked(envelope)
            value, metadata, metadata_digest = projected
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
                values=value.values.reshape((1, *self.schema.cell_schema.data_shape)),
                validity_patch=(value.validity,),
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
                self._ordered_event_metadata[schedule_index] = metadata
                self._ordered_metadata_hasher.update(metadata_digest)
                assert self._ordered_event_hasher is not None
                self._ordered_event_hasher.update(envelope.ref, metadata_digest)
            else:
                self._last_monitor_sequence = envelope.sequence
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
                cell_metadata=tuple(self._cell_metadata),
            )

    def seal(self, eos: EndOfStream) -> SealedDatasetArtifact:
        if self.mode is not DatasetMode.FINITE_EXACT or self._reservation is None:
            raise DatasetError("rolling monitor datasets cannot become formal sealed datasets")
        self._source._complete_consumer(self._reservation, eos, self, self._seal_locked)
        preview = self.materialize()
        assert self._ordered_event_hasher is not None
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
        if self._reservation is not None:
            self._source._abort_consumer(
                self._reservation,
                self,
                self._mark_aborted_locked,
            )
            return
        self._mark_aborted_locked()

    def exact_readiness(self) -> ExactConsumerReadiness:
        if self.mode is not DatasetMode.FINITE_EXACT or self._reservation is None:
            raise DatasetError("only a finite exact DatasetBuilder can authorize capture")
        with self._lock:
            self._ensure_writable_locked()
            readiness = self._exact_readiness
        if readiness is None:
            raise DatasetError("finite exact DatasetBuilder lost its readiness proof")
        readiness._validate_terminal_sink()
        return readiness

    def close(self) -> None:
        """Idempotently abort if needed and release the exact reservation."""

        if self._reservation is None:
            with self._lock:
                if not self._sealed and not self._aborted:
                    self._mark_aborted_locked()
            return
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
        if self._reservation is None:
            return False
        cleanup_error: BaseException | None = None
        try:
            self.close()
        except BaseException as error:
            cleanup_error = error
        if cleanup_error is not None:
            if exc is None:
                raise cleanup_error
            record_secondary_failure(
                exc,
                "DatasetBuilder teardown also failed",
                cleanup_error,
            )
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

    def _validate_envelope_identity_locked(
        self,
        envelope: Envelope[PayloadT],
    ) -> None:
        if envelope.stream_generation != self.generation:
            raise DatasetError("envelope stream generation differs from DatasetBuilder")
        if envelope.stream_id != self.stream_id:
            raise DatasetError("envelope stream id differs from DatasetBuilder")

    def _project_payload(
        self,
        payload: PayloadT,
    ) -> tuple[Value, object | None, str]:
        value = self.edge.project_value(payload)
        if not isinstance(value, Value):
            raise TypeError("DatasetEventAdapter.value must return Value")
        if value.schema is not self.schema.cell_schema:
            raise DatasetError("event ValueSchema differs from DatasetSchema cell contract")
        contract = self._metadata_contract
        metadata = contract.snapshot(payload)
        contract.validate(metadata)
        if not _deeply_immutable_metadata(metadata):
            raise TypeError("metadata contract must return a deeply immutable snapshot")
        retained = contract.retained_nbytes(metadata)
        if (
            isinstance(retained, bool)
            or not isinstance(retained, Integral)
            or retained < 0
            or retained > self._metadata_max_retained_nbytes
        ):
            raise ValueError("metadata retained bytes exceed the declared metadata bound")
        digest = _sha256_digest(contract.digest(metadata), "metadata digest")
        return value, metadata, digest

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
        return expand_component_validity(validity, self.schema.cell_schema)

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
    "DatasetCellDomain",
    "DatasetCellKeyContract",
    "DatasetCoverage",
    "DatasetDerivationProvenance",
    "DatasetEventAdapter",
    "DatasetMetadataContract",
    "DatasetError",
    "DatasetMode",
    "DatasetProgress",
    "DatasetPreviewSnapshot",
    "DatasetSealProvenance",
    "DuplicateDatasetCell",
    "FrozenDatasetEdge",
    "MissingDatasetCells",
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
