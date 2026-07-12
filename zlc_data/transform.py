"""Axis-total, sparse-preserving authoritative data transforms."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral
import numpy as np

from zlc_storage.canonical import canonical_digest

from ._arrays import canonical_dtype, immutable_array
from .axis import AxisId, AxisSpec
from .layout import AxisLayout, AxisLayoutMode
from .schema import DatasetSchema
from .selection import (
    CoordinateRangeSelection,
    IndexRangeSelection,
    IndexSelection,
    Selection,
)
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    Invalid,
    RowComponentValidity,
    Valid,
    ValidityMode,
)
from .value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    OwnedSnapshot,
)


TRANSFORM_SPEC_SCHEMA = "zlc_data.DataTransformSpec/v1"
COMMITTED_TRANSFORM_SCHEMA = "zlc_data.CommittedTransform/v1"
TRANSFORMED_SCHEMA_SCHEMA = "zlc_data.TransformedSchema/v1"


class ReductionMethod(str, Enum):
    MEAN = "MEAN"
    SUM = "SUM"
    MIN = "MIN"
    MAX = "MAX"


class MissingPolicy(str, Enum):
    """How absent logical contributors affect a reduction output row."""

    REQUIRE_ALL = "REQUIRE_ALL"
    OMIT_MISSING = "OMIT_MISSING"


class ValidityPolicy(str, Enum):
    """How present-but-invalid components affect a reduction value."""

    REQUIRE_ALL = "REQUIRE_ALL"
    OMIT_INVALID = "OMIT_INVALID"
    MIN_COUNT = "MIN_COUNT"


class TransformOrigin(str, Enum):
    USER = "USER"
    ACCEPTED_SUGGESTION = "ACCEPTED_SUGGESTION"
    SAVED = "SAVED"


@dataclass(frozen=True, order=True)
class TransformRevision:
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, Integral) or self.value < 0:
            raise ValueError("TransformRevision must be a non-negative integer")
        object.__setattr__(self, "value", int(self.value))


@dataclass(frozen=True)
class ReductionSpec:
    axis_ids: tuple[AxisId, ...]
    method: ReductionMethod
    missing_policy: MissingPolicy = MissingPolicy.REQUIRE_ALL
    validity_policy: ValidityPolicy = ValidityPolicy.REQUIRE_ALL
    minimum_valid_count: int | None = None

    def __post_init__(self) -> None:
        axis_ids = tuple(self.axis_ids)
        if not axis_ids or any(not isinstance(axis_id, AxisId) for axis_id in axis_ids):
            raise ValueError("ReductionSpec axis_ids must contain at least one AxisId")
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("ReductionSpec axis_ids must be unique")
        if not isinstance(self.method, ReductionMethod):
            raise TypeError("method must be ReductionMethod")
        if not isinstance(self.missing_policy, MissingPolicy):
            raise TypeError("missing_policy must be MissingPolicy")
        if not isinstance(self.validity_policy, ValidityPolicy):
            raise TypeError("validity_policy must be ValidityPolicy")
        if self.validity_policy is ValidityPolicy.MIN_COUNT:
            if (
                isinstance(self.minimum_valid_count, bool)
                or not isinstance(self.minimum_valid_count, Integral)
                or self.minimum_valid_count <= 0
            ):
                raise ValueError("MIN_COUNT requires a positive minimum_valid_count")
            object.__setattr__(self, "minimum_valid_count", int(self.minimum_valid_count))
        elif self.minimum_valid_count is not None:
            raise ValueError("minimum_valid_count is only valid with MIN_COUNT")
        object.__setattr__(self, "axis_ids", tuple(sorted(axis_ids, key=lambda item: item.value)))


@dataclass(frozen=True)
class Select:
    selection: Selection

    def __post_init__(self) -> None:
        if not isinstance(self.selection, Selection):
            raise TypeError("selection must be Selection")


@dataclass(frozen=True)
class Reduce:
    reduction: ReductionSpec

    def __post_init__(self) -> None:
        if not isinstance(self.reduction, ReductionSpec):
            raise TypeError("reduction must be ReductionSpec")


TransformOperation = Select | Reduce


@dataclass(frozen=True)
class DataTransformSpec:
    operations: tuple[TransformOperation, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(not isinstance(operation, (Select, Reduce)) for operation in operations):
            raise TypeError("DataTransformSpec contains an unsupported operation")
        object.__setattr__(self, "operations", operations)


@dataclass(frozen=True)
class TransformedSchema:
    cell_axes: tuple[AxisSpec, ...]
    cell_layout: AxisLayout
    data_axes: tuple[AxisSpec, ...]
    validity_axis_ids: tuple[AxisId, ...]
    dtype: np.dtype
    value_unit: str | None

    def __post_init__(self) -> None:
        cell_axes = tuple(self.cell_axes)
        data_axes = tuple(self.data_axes)
        validity_axis_ids = tuple(self.validity_axis_ids)
        if len(set(validity_axis_ids)) != len(validity_axis_ids):
            raise ValueError("transformed validity axis ids must be unique")
        if any(not isinstance(axis, AxisSpec) for axis in cell_axes + data_axes):
            raise TypeError("transformed axes must contain AxisSpec values")
        axis_ids = tuple(axis.axis_id for axis in cell_axes + data_axes)
        if len(set(axis_ids)) != len(axis_ids):
            raise ValueError("transformed axis ids must be unique")
        if not isinstance(self.cell_layout, AxisLayout):
            raise TypeError("cell_layout must be AxisLayout")
        if self.cell_layout.logical_shape != tuple(axis.size for axis in cell_axes):
            raise ValueError("cell layout shape does not match cell axes")
        available_data_ids = tuple(axis.axis_id for axis in data_axes)
        try:
            validity_positions = tuple(
                available_data_ids.index(axis_id) for axis_id in validity_axis_ids
            )
        except ValueError as exc:
            raise ValueError("validity axis is absent from transformed data axes") from exc
        if validity_positions != tuple(sorted(validity_positions)):
            raise ValueError("validity axes must follow transformed data-axis order")
        if self.value_unit is not None and (
            not isinstance(self.value_unit, str)
            or not self.value_unit
            or self.value_unit.strip() != self.value_unit
        ):
            raise ValueError("value_unit must be non-empty text or None")
        object.__setattr__(self, "cell_axes", cell_axes)
        object.__setattr__(self, "data_axes", data_axes)
        object.__setattr__(self, "validity_axis_ids", validity_axis_ids)
        object.__setattr__(self, "dtype", canonical_dtype(self.dtype))

    @property
    def data_shape(self) -> tuple[int, ...]:
        return tuple(axis.size for axis in self.data_axes)

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return (self.cell_layout.storage_size, *self.data_shape)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        return self.cell_axes + self.data_axes

    @property
    def fingerprint(self) -> str:
        from .transform_codec import transformed_schema_to_tree

        return canonical_digest(transformed_schema_to_tree(self))

    def axis(self, axis_id: AxisId) -> AxisSpec:
        for axis in self.axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)


@dataclass(frozen=True)
class TransformRecord:
    operation_index: int
    operation_kind: str
    operation_digest: str
    input_schema_fingerprint: str
    output_schema_fingerprint: str
    input_present_cells: int
    output_present_cells: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.operation_index, bool)
            or not isinstance(self.operation_index, Integral)
            or self.operation_index < 0
        ):
            raise ValueError("operation_index must be a non-negative integer")
        if self.operation_kind not in {"SELECT", "REDUCE"}:
            raise ValueError("operation_kind is not a closed transform operation")
        for field, value in (
            ("operation_digest", self.operation_digest),
            ("input_schema_fingerprint", self.input_schema_fingerprint),
            ("output_schema_fingerprint", self.output_schema_fingerprint),
        ):
            _require_digest(value, field)
        for field, value in (
            ("input_present_cells", self.input_present_cells),
            ("output_present_cells", self.output_present_cells),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{field} must be a non-negative integer")


@dataclass(frozen=True, eq=False)
class TransformedData:
    source_ref: DatasetRevisionRef
    transform_digest: str
    transform_revision: TransformRevision
    transform_origin: TransformOrigin
    values: np.ndarray
    validity: Valid | Invalid | RowComponentValidity
    schema: TransformedSchema
    records: tuple[TransformRecord, ...]
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        _require_digest(self.transform_digest, "transform_digest")
        if not isinstance(self.transform_revision, TransformRevision):
            raise TypeError("transform_revision must be TransformRevision")
        if not isinstance(self.transform_origin, TransformOrigin):
            raise TypeError("transform_origin must be TransformOrigin")
        if not isinstance(self.schema, TransformedSchema):
            raise TypeError("schema must be TransformedSchema")
        records = tuple(self.records)
        if any(not isinstance(record, TransformRecord) for record in records):
            raise TypeError("records must contain TransformRecord values")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "values",
            immutable_array(self.values, dtype=self.schema.dtype, shape=self.schema.physical_shape),
        )
        _validate_transformed_validity(self.validity, self.schema)

    def expanded_validity(self) -> np.ndarray:
        """Return a read-only broadcast view aligned to ``values``."""

        return _expand_transformed_validity(self.validity, self.schema)


@dataclass(frozen=True, eq=False)
class PreviewTransformedData:
    """Non-authoritative transform evaluation; cannot enter fit/artifact APIs."""

    source_block_id: BlockId
    source_revision: DatasetRevision
    source_schema_fingerprint: str
    values: np.ndarray
    validity: Valid | Invalid | RowComponentValidity
    schema: TransformedSchema
    records: tuple[TransformRecord, ...]
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_block_id, BlockId):
            raise TypeError("source_block_id must be BlockId")
        if not isinstance(self.source_revision, DatasetRevision):
            raise TypeError("source_revision must be DatasetRevision")
        _require_digest(self.source_schema_fingerprint, "source_schema_fingerprint")
        if not isinstance(self.schema, TransformedSchema):
            raise TypeError("schema must be TransformedSchema")
        records = tuple(self.records)
        if any(not isinstance(record, TransformRecord) for record in records):
            raise TypeError("records must contain TransformRecord values")
        object.__setattr__(self, "records", records)
        object.__setattr__(
            self,
            "values",
            immutable_array(self.values, dtype=self.schema.dtype, shape=self.schema.physical_shape),
        )
        _validate_transformed_validity(self.validity, self.schema)

    def expanded_validity(self) -> np.ndarray:
        return _expand_transformed_validity(self.validity, self.schema)


@dataclass(frozen=True)
class CommittedTransform:
    input_schema_fingerprint: str
    spec: DataTransformSpec
    output_schema_fingerprint: str
    revision: TransformRevision
    origin: TransformOrigin
    transform_digest: str

    def __post_init__(self) -> None:
        _require_digest(self.input_schema_fingerprint, "input_schema_fingerprint")
        _require_digest(self.output_schema_fingerprint, "output_schema_fingerprint")
        _require_digest(self.transform_digest, "transform_digest")
        if not isinstance(self.revision, TransformRevision):
            raise TypeError("revision must be TransformRevision")
        if not isinstance(self.origin, TransformOrigin):
            raise TypeError("origin must be TransformOrigin")
        if not isinstance(self.spec, DataTransformSpec) or not self.spec.operations:
            raise ValueError("CommittedTransform requires a non-empty DataTransformSpec")
        from .transform_codec import committed_transform_payload_tree

        expected = canonical_digest(committed_transform_payload_tree(self))
        if self.transform_digest != expected:
            raise ValueError("transform_digest does not match committed transform content")


@dataclass
class _State:
    schema: TransformedSchema
    values: np.ndarray | None
    validity: Valid | Invalid | RowComponentValidity | None
    records: list[TransformRecord]


def _source_validity(
    block: DataBlock,
) -> Valid | Invalid | RowComponentValidity:
    validity = block.validity
    if isinstance(validity, (Valid, Invalid)):
        return validity
    if isinstance(validity, CellValidity):
        return RowComponentValidity((), validity.mask.reshape(-1))
    if isinstance(validity, ComponentValidity):
        return RowComponentValidity(
            validity.axis_ids,
            validity.mask.reshape((-1, *validity.mask.shape[2:])),
        )
    raise TypeError(f"unsupported dataset validity {type(validity).__name__}")


def _validate_transformed_validity(
    validity: Valid | Invalid | RowComponentValidity,
    schema: TransformedSchema,
) -> None:
    if isinstance(validity, (Valid, Invalid)):
        return
    if not isinstance(validity, RowComponentValidity):
        raise TypeError("transformed validity must be Valid, Invalid, or RowComponentValidity")
    available = tuple(axis.axis_id for axis in schema.data_axes)
    try:
        positions = tuple(available.index(axis_id) for axis_id in validity.axis_ids)
    except ValueError as exc:
        raise ValueError("row validity axis is absent from transformed schema") from exc
    if positions != tuple(sorted(positions)):
        raise ValueError("row validity axes must follow transformed data-axis order")
    if any(axis_id not in schema.validity_axis_ids for axis_id in validity.axis_ids):
        raise ValueError("row validity exceeds the transformed validity contract")
    expected = (schema.cell_layout.storage_size,) + tuple(
        schema.data_axes[position].size for position in positions
    )
    if validity.mask.shape != expected:
        raise ValueError(
            f"row validity shape {validity.mask.shape} does not match named axes {expected}"
        )


def _expand_transformed_validity(
    validity: Valid | Invalid | RowComponentValidity,
    schema: TransformedSchema,
) -> np.ndarray:
    _validate_transformed_validity(validity, schema)
    if isinstance(validity, (Valid, Invalid)):
        return np.broadcast_to(isinstance(validity, Valid), schema.physical_shape)
    available = tuple(axis.axis_id for axis in schema.data_axes)
    shape = [schema.cell_layout.storage_size] + [1] * len(schema.data_axes)
    for mask_position, axis_id in enumerate(validity.axis_ids):
        shape[1 + available.index(axis_id)] = validity.mask.shape[1 + mask_position]
    return np.broadcast_to(validity.mask.reshape(tuple(shape)), schema.physical_shape)


def _select_validity_rows(
    validity: Valid | Invalid | RowComponentValidity,
    selected: np.ndarray,
) -> Valid | Invalid | RowComponentValidity:
    if isinstance(validity, (Valid, Invalid)):
        return validity
    return RowComponentValidity(validity.axis_ids, validity.mask[selected])


def _select_validity_data(
    validity: Valid | Invalid | RowComponentValidity,
    axis_id: AxisId,
    indices: tuple[int, ...],
    drop: bool,
) -> Valid | Invalid | RowComponentValidity:
    if isinstance(validity, (Valid, Invalid)) or axis_id not in validity.axis_ids:
        return validity
    position = validity.axis_ids.index(axis_id)
    array_axis = 1 + position
    if drop:
        mask = np.take(validity.mask, indices[0], axis=array_axis)
        axis_ids = validity.axis_ids[:position] + validity.axis_ids[position + 1 :]
    else:
        mask = np.take(validity.mask, indices, axis=array_axis)
        axis_ids = validity.axis_ids
    return RowComponentValidity(axis_ids, mask)


def _compact_transformed_validity(
    mask: np.ndarray,
    schema: TransformedSchema,
) -> Valid | Invalid | RowComponentValidity:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != schema.physical_shape:
        raise ValueError("expanded transformed validity shape disagrees with schema")
    if mask.size and np.all(mask):
        return VALID
    if mask.size and not np.any(mask):
        return INVALID
    current_axis_ids = [axis.axis_id for axis in schema.data_axes]
    compact = mask
    contract = set(schema.validity_axis_ids)
    for position in range(len(current_axis_ids) - 1, -1, -1):
        axis_id = current_axis_ids[position]
        if axis_id in contract:
            continue
        array_axis = 1 + position
        first = np.take(compact, 0, axis=array_axis)
        if not np.array_equal(
            compact, np.broadcast_to(np.expand_dims(first, array_axis), compact.shape)
        ):
            raise RuntimeError("validity varies along an axis absent from its contract")
        compact = first
        current_axis_ids.pop(position)
    return RowComponentValidity(tuple(current_axis_ids), compact)


def commit_transform(
    schema: DatasetSchema,
    authoritative_spec: DataTransformSpec,
    *,
    revision: TransformRevision,
    origin: TransformOrigin,
) -> CommittedTransform:
    """Validate and freeze an authoritative transform without touching dataset values."""

    if not isinstance(authoritative_spec, DataTransformSpec):
        raise TypeError("authoritative_spec must be DataTransformSpec")
    if not authoritative_spec.operations:
        raise ValueError("identity input uses None; an empty transform cannot be committed")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    state = _source_state(schema, values=None, validity=None)
    if not isinstance(revision, TransformRevision):
        raise TypeError("revision must be TransformRevision")
    if not isinstance(origin, TransformOrigin):
        raise TypeError("origin must be TransformOrigin")
    state = _run_operations(state, authoritative_spec)
    provisional = object.__new__(CommittedTransform)
    object.__setattr__(provisional, "input_schema_fingerprint", schema.fingerprint)
    object.__setattr__(provisional, "spec", authoritative_spec)
    object.__setattr__(provisional, "output_schema_fingerprint", state.schema.fingerprint)
    object.__setattr__(provisional, "revision", revision)
    object.__setattr__(provisional, "origin", origin)
    object.__setattr__(provisional, "transform_digest", "0" * 64)
    from .transform_codec import committed_transform_payload_tree

    digest = canonical_digest(committed_transform_payload_tree(provisional))
    return CommittedTransform(
        schema.fingerprint,
        authoritative_spec,
        state.schema.fingerprint,
        revision,
        origin,
        digest,
    )


def apply_transform(
    snapshot: OwnedSnapshot,
    transform: CommittedTransform,
) -> TransformedData:
    """Execute one committed authority transform against an identified snapshot."""

    if not isinstance(snapshot, OwnedSnapshot):
        raise TypeError("snapshot must be OwnedSnapshot")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    block = snapshot.block
    if transform.input_schema_fingerprint != block.schema.fingerprint:
        raise ValueError("CommittedTransform input schema fingerprint is stale")
    state = _execute_transform(block, transform.spec)
    if state.schema.fingerprint != transform.output_schema_fingerprint:
        raise RuntimeError("transform execution disagrees with its committed output schema")
    assert state.values is not None and state.validity is not None
    return TransformedData(
        snapshot.ref,
        transform.transform_digest,
        transform.revision,
        transform.origin,
        state.values,
        state.validity,
        state.schema,
        tuple(state.records),
    )


def preview_transform(block: DataBlock, spec: DataTransformSpec) -> PreviewTransformedData:
    """Evaluate an uncommitted draft into a type that cannot masquerade as authority."""

    if not isinstance(block, DataBlock):
        raise TypeError("block must be DataBlock")
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec")
    state = _execute_transform(block, spec)
    assert state.values is not None and state.validity is not None
    return PreviewTransformedData(
        block.block_id,
        block.revision,
        block.schema.fingerprint,
        state.values,
        state.validity,
        state.schema,
        tuple(state.records),
    )


def _execute_transform(block: DataBlock, spec: DataTransformSpec) -> _State:
    state = _source_state(
        block.schema,
        values=block.values.reshape((-1, *block.schema.cell_schema.data_shape)),
        validity=_source_validity(block),
    )
    return _run_operations(state, spec)


def _source_state(
    schema: DatasetSchema,
    *,
    values: np.ndarray | None,
    validity: Valid | Invalid | RowComponentValidity | None,
) -> _State:
    cell_axes = (schema.repeat_axis, *schema.point_axes)
    if schema.point_layout.mode is AxisLayoutMode.RECT_C or (
        schema.point_layout.mode is AxisLayoutMode.RECT_F
        and len(schema.point_axes) <= 1
    ):
        cell_layout = AxisLayout.rect_c(tuple(axis.size for axis in cell_axes))
    else:
        point_layout = AxisLayout(
            schema.point_layout.logical_shape,
            schema.point_layout.mode,
            schema.point_layout.storage_size,
            schema.point_layout.storage_to_multi,
        )
        cell_layout = AxisLayout.product(
            AxisLayout.rect_c((schema.repeat_axis.size,)), point_layout
        )
    transformed_schema = TransformedSchema(
        cell_axes,
        cell_layout,
        schema.cell_schema.data_axes,
        schema.cell_schema.validity_contract.component_axis_ids
        if schema.cell_schema.validity_contract.mode is ValidityMode.COMPONENTS
        else (),
        schema.cell_schema.dtype,
        schema.cell_schema.value_unit,
    )
    return _State(transformed_schema, values, validity, [])


def _run_operations(state: _State, spec: DataTransformSpec) -> _State:
    for operation_index, operation in enumerate(spec.operations):
        before = state.schema
        if isinstance(operation, Select):
            state = _apply_selection(state, operation.selection)
            operation_kind = "SELECT"
        else:
            state = _apply_reduction(state, operation.reduction)
            operation_kind = "REDUCE"
        state.records.append(
            TransformRecord(
                operation_index,
                operation_kind,
                _operation_digest(operation),
                before.fingerprint,
                state.schema.fingerprint,
                before.cell_layout.storage_size,
                state.schema.cell_layout.storage_size,
            )
        )
    return state


def _operation_digest(operation: TransformOperation) -> str:
    from .transform_codec import data_transform_spec_to_tree

    return canonical_digest(data_transform_spec_to_tree(DataTransformSpec((operation,))))


def _apply_selection(state: _State, selection: Selection) -> _State:
    for term in selection.terms:
        cell_ids = tuple(axis.axis_id for axis in state.schema.cell_axes)
        data_ids = tuple(axis.axis_id for axis in state.schema.data_axes)
        if term.axis_id in cell_ids:
            state = _select_cell_axis(state, cell_ids.index(term.axis_id), term)
        elif term.axis_id in data_ids:
            state = _select_data_axis(state, data_ids.index(term.axis_id), term)
        else:
            raise KeyError(f"selection axis {term.axis_id} is absent from transformed schema")
    return state


def _selection_indices(axis: AxisSpec, term) -> tuple[tuple[int, ...], bool]:
    if isinstance(term, IndexSelection):
        if term.index >= axis.size:
            raise IndexError(f"selection index {term.index} is outside axis {axis.axis_id}")
        return (term.index,), True
    if isinstance(term, IndexRangeSelection):
        if term.stop > axis.size:
            raise IndexError(f"selection range stop {term.stop} is outside axis {axis.axis_id}")
        return tuple(range(term.start, term.stop)), False
    if not isinstance(term, CoordinateRangeSelection):
        raise TypeError(f"unsupported selection term {type(term).__name__}")
    if axis.coordinates is None:
        raise ValueError(f"axis {axis.axis_id} has no coordinates for coordinate selection")
    if axis.coordinate_frame != term.coordinate_frame:
        raise ValueError(f"coordinate frame mismatch for axis {axis.axis_id}")
    if any(
        value is None or isinstance(value, (bool, str))
        for value in axis.coordinates
    ):
        raise TypeError(f"axis {axis.axis_id} coordinates are not entirely numeric")
    indices = tuple(
        index
        for index, value in enumerate(axis.coordinates)
        if term.lower <= value <= term.upper
    )
    if not indices:
        raise ValueError(f"coordinate selection is empty on axis {axis.axis_id}")
    return indices, False


def _selected_axis(axis: AxisSpec, indices: tuple[int, ...]) -> AxisSpec:
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(indices),
        None if axis.coordinates is None else tuple(axis.coordinates[index] for index in indices),
        axis.unit,
        axis.coordinate_frame,
    )


def _select_cell_axis(state: _State, position: int, term) -> _State:
    axis = state.schema.cell_axes[position]
    indices, drop = _selection_indices(axis, term)
    axes = list(state.schema.cell_axes)
    if drop:
        axes.pop(position)
    else:
        axes[position] = _selected_axis(axis, indices)
    layout = _select_layout(state.schema.cell_layout, position, indices, drop)
    schema = TransformedSchema(
        tuple(axes),
        layout,
        state.schema.data_axes,
        state.schema.validity_axis_ids,
        state.schema.dtype,
        state.schema.value_unit,
    )
    if state.values is None:
        values = None
        validity = None
    else:
        selected = np.isin(_axis_index_vector(state.schema.cell_layout, position), indices)
        values = state.values[selected]
        assert state.validity is not None
        validity = _select_validity_rows(state.validity, selected)
        if values.shape[0] != layout.storage_size:
            raise RuntimeError("cell selection layout disagrees with selected physical rows")
    return _State(schema, values, validity, state.records)


def _select_data_axis(state: _State, position: int, term) -> _State:
    axis = state.schema.data_axes[position]
    indices, drop = _selection_indices(axis, term)
    array_axis = 1 + position
    axes = list(state.schema.data_axes)
    if drop:
        axes.pop(position)
        values = None if state.values is None else np.take(state.values, indices[0], axis=array_axis)
    else:
        axes[position] = _selected_axis(axis, indices)
        values = None if state.values is None else np.take(state.values, indices, axis=array_axis)
    validity_axis_ids = tuple(
        axis_id
        for axis_id in state.schema.validity_axis_ids
        if not (drop and axis_id == axis.axis_id)
    )
    validity = (
        None
        if state.validity is None
        else _select_validity_data(state.validity, axis.axis_id, indices, drop)
    )
    schema = TransformedSchema(
        state.schema.cell_axes,
        state.schema.cell_layout,
        tuple(axes),
        validity_axis_ids,
        state.schema.dtype,
        state.schema.value_unit,
    )
    return _State(schema, values, validity, state.records)


def _apply_reduction(state: _State, reduction: ReductionSpec) -> _State:
    cell_ids = tuple(axis.axis_id for axis in state.schema.cell_axes)
    data_ids = tuple(axis.axis_id for axis in state.schema.data_axes)
    unknown = tuple(axis_id for axis_id in reduction.axis_ids if axis_id not in cell_ids + data_ids)
    if unknown:
        raise KeyError(f"reduction axes are absent from transformed schema: {unknown}")
    cell_positions = tuple(index for index, axis_id in enumerate(cell_ids) if axis_id in reduction.axis_ids)
    data_positions = tuple(index for index, axis_id in enumerate(data_ids) if axis_id in reduction.axis_ids)
    maximum_valid_contributors = math.prod(
        state.schema.cell_axes[position].size for position in cell_positions
    ) * math.prod(state.schema.data_axes[position].size for position in data_positions)
    if (
        reduction.validity_policy is ValidityPolicy.MIN_COUNT
        and reduction.minimum_valid_count is not None
        and reduction.minimum_valid_count > maximum_valid_contributors
    ):
        raise ValueError(
            "minimum_valid_count exceeds the reduction's maximum contributor count"
        )
    if reduction.method in (ReductionMethod.MIN, ReductionMethod.MAX) and state.schema.dtype.kind == "c":
        raise TypeError("MIN/MAX reductions are undefined for complex values")

    remaining_cell_axes = tuple(
        axis for index, axis in enumerate(state.schema.cell_axes) if index not in cell_positions
    )
    remaining_data_axes = tuple(
        axis for index, axis in enumerate(state.schema.data_axes) if index not in data_positions
    )
    output_layout = _reduce_layout(
        state.schema.cell_layout, cell_positions, reduction.missing_policy
    )
    output_dtype = _reduction_dtype(state.schema.dtype, reduction.method)
    schema = TransformedSchema(
        remaining_cell_axes,
        output_layout,
        remaining_data_axes,
        tuple(
            axis_id
            for axis_id in state.schema.validity_axis_ids
            if axis_id not in reduction.axis_ids
        ),
        output_dtype,
        state.schema.value_unit,
    )
    if state.values is None:
        return _State(schema, None, None, state.records)
    assert state.validity is not None

    if not cell_positions:
        reduced_values, reduced_validity = _reduce_arrays(
            state.values,
            _expand_transformed_validity(state.validity, state.schema),
            tuple(1 + position for position in data_positions),
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        return _State(
            schema,
            reduced_values,
            _compact_transformed_validity(reduced_validity, schema),
            state.records,
        )

    if _is_full_layout(state.schema.cell_layout):
        logical_values = _full_rows_to_logical(state.values, state.schema.cell_layout)
        logical_validity = _full_rows_to_logical(
            _expand_transformed_validity(state.validity, state.schema),
            state.schema.cell_layout,
        )
        axes = tuple(cell_positions) + tuple(
            len(state.schema.cell_axes) + position for position in data_positions
        )
        reduced_values, reduced_validity = _reduce_arrays(
            logical_values,
            logical_validity,
            axes,
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        values_array = _logical_to_full_rows(reduced_values, output_layout)
        validity_array = _logical_to_full_rows(reduced_validity, output_layout)
        return _State(
            schema,
            values_array,
            _compact_transformed_validity(validity_array, schema),
            state.records,
        )

    factor_axes = _factor_aligned_reduction_axes(
        state.schema.cell_layout, cell_positions
    )
    if factor_axes is not None and state.schema.cell_layout.factors is not None and (
        output_layout.storage_size
        == math.prod(
            factor.storage_size
            for index, factor in enumerate(state.schema.cell_layout.factors)
            if index not in factor_axes
        )
    ):
        factor_shape = tuple(
            factor.storage_size for factor in state.schema.cell_layout.factors
        )
        physical_values = state.values.reshape(
            (*factor_shape, *state.schema.data_shape)
        )
        physical_validity = _expand_transformed_validity(
            state.validity, state.schema
        ).reshape((*factor_shape, *state.schema.data_shape))
        axes = tuple(factor_axes) + tuple(
            len(factor_shape) + position for position in data_positions
        )
        reduced_values, reduced_validity = _reduce_arrays(
            physical_values,
            physical_validity,
            axes,
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        values_array = reduced_values.reshape(schema.physical_shape)
        validity_array = reduced_validity.reshape(schema.physical_shape)
        return _State(
            schema,
            values_array,
            _compact_transformed_validity(validity_array, schema),
            state.records,
        )

    source_mapping = _layout_mapping(state.schema.cell_layout)
    groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row, multi in enumerate(source_mapping):
        key = tuple(value for index, value in enumerate(multi) if index not in cell_positions)
        groups.setdefault(key, []).append(row)
    expected_contributors = math.prod(
        state.schema.cell_axes[position].size for position in cell_positions
    )
    if not cell_positions:
        expected_contributors = 1

    output_mapping: list[tuple[int, ...]] = []
    output_values: list[np.ndarray] = []
    output_validity: list[np.ndarray] = []
    for key, rows in groups.items():
        if (
            reduction.missing_policy is MissingPolicy.REQUIRE_ALL
            and len(rows) != expected_contributors
        ):
            continue
        output_mapping.append(key)
        values = state.values[np.asarray(rows, dtype=np.intp)]
        validity = _expand_transformed_validity(state.validity, state.schema)[
            np.asarray(rows, dtype=np.intp)
        ]
        # ``values`` always has an artificial group-row dimension.  A data-only
        # reduction groups one physical cell at a time, so that singleton must
        # still be collapsed rather than leaking into the output shape.
        reduction_axes = tuple([0] + [1 + position for position in data_positions])
        if not reduction_axes:
            raise RuntimeError("ReductionSpec did not resolve to a physical reduction axis")
        reduced_values, reduced_validity = _reduce_arrays(
            values,
            validity,
            reduction_axes,
            reduction.method,
            reduction.validity_policy,
            reduction.minimum_valid_count,
            output_dtype,
        )
        output_values.append(reduced_values)
        output_validity.append(reduced_validity)

    runtime_layout = _layout_from_mapping(
        tuple(axis.size for axis in remaining_cell_axes), tuple(output_mapping)
    )
    if _layout_mapping(runtime_layout) != _layout_mapping(output_layout):
        raise RuntimeError("reduction runtime layout disagrees with schema compilation")
    if output_values:
        values_array = np.stack(output_values, axis=0).astype(output_dtype, copy=False)
        validity_array = np.stack(output_validity, axis=0).astype(bool, copy=False)
    else:
        values_array = np.empty(schema.physical_shape, dtype=output_dtype)
        validity_array = np.empty(schema.physical_shape, dtype=bool)
    return _State(
        schema,
        values_array,
        _compact_transformed_validity(validity_array, schema),
        state.records,
    )


def _rect_rows_to_logical(values: np.ndarray, layout: AxisLayout) -> np.ndarray:
    data_shape = values.shape[1:]
    if layout.mode is AxisLayoutMode.RECT_C:
        return values.reshape((*layout.logical_shape, *data_shape))
    flat_data = math.prod(data_shape)
    transposed = values.reshape(layout.storage_size, flat_data).T
    logical = transposed.reshape((flat_data, *layout.logical_shape), order="F")
    return np.moveaxis(logical, 0, -1).reshape((*layout.logical_shape, *data_shape))


def _logical_to_rect_rows(values: np.ndarray, layout: AxisLayout) -> np.ndarray:
    data_shape = values.shape[len(layout.logical_shape) :]
    if layout.mode is AxisLayoutMode.RECT_C:
        return values.reshape((layout.storage_size, *data_shape))
    flat_data = math.prod(data_shape)
    with_flat_data = values.reshape((*layout.logical_shape, flat_data))
    transposed = np.moveaxis(with_flat_data, -1, 0)
    rows = transposed.reshape((flat_data, layout.storage_size), order="F").T
    return rows.reshape((layout.storage_size, *data_shape))


def _is_full_layout(layout: AxisLayout) -> bool:
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        return True
    if layout.mode is AxisLayoutMode.EXPLICIT:
        return layout.storage_size == math.prod(layout.logical_shape)
    if layout.mode is AxisLayoutMode.PRODUCT:
        assert layout.factors is not None
        return all(_is_full_layout(factor) for factor in layout.factors)
    return False


def _factor_aligned_reduction_axes(
    layout: AxisLayout,
    cell_positions: tuple[int, ...],
) -> tuple[int, ...] | None:
    if layout.mode is not AxisLayoutMode.PRODUCT:
        return None
    assert layout.factors is not None
    selected = set(cell_positions)
    offset = 0
    factor_axes: list[int] = []
    for factor_index, factor in enumerate(layout.factors):
        positions = set(range(offset, offset + len(factor.logical_shape)))
        overlap = selected & positions
        if overlap and overlap != positions:
            return None
        if overlap:
            factor_axes.append(factor_index)
        offset += len(factor.logical_shape)
    return tuple(factor_axes) if factor_axes else None


def _logical_c_indices(layout: AxisLayout) -> np.ndarray:
    if layout.mode is AxisLayoutMode.RECT_C:
        return np.arange(layout.storage_size, dtype=np.int64)
    columns = tuple(
        _axis_index_vector(layout, position)
        for position in range(len(layout.logical_shape))
    )
    if not columns:
        return np.array([0], dtype=np.int64)
    return np.ravel_multi_index(columns, layout.logical_shape, order="C").astype(
        np.int64, copy=False
    )


def _full_rows_to_logical(values: np.ndarray, layout: AxisLayout) -> np.ndarray:
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        return _rect_rows_to_logical(values, layout)
    if not _is_full_layout(layout):
        raise ValueError("layout is not a full Cartesian layout")
    logical_rows = np.empty_like(values)
    logical_rows[_logical_c_indices(layout)] = values
    return logical_rows.reshape((*layout.logical_shape, *values.shape[1:]))


def _logical_to_full_rows(values: np.ndarray, layout: AxisLayout) -> np.ndarray:
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        return _logical_to_rect_rows(values, layout)
    if not _is_full_layout(layout):
        raise ValueError("layout is not a full Cartesian layout")
    data_shape = values.shape[len(layout.logical_shape) :]
    logical_rows = values.reshape((layout.storage_size, *data_shape))
    return logical_rows[_logical_c_indices(layout)]


def _reduce_arrays(
    values: np.ndarray,
    validity: np.ndarray,
    axes: tuple[int, ...],
    method: ReductionMethod,
    policy: ValidityPolicy,
    minimum_valid_count: int | None,
    output_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.sum(validity, axis=axes, dtype=np.int64)
    if policy is ValidityPolicy.REQUIRE_ALL:
        total = math.prod(values.shape[axis] for axis in axes)
        output_validity = counts == total
    elif policy is ValidityPolicy.MIN_COUNT:
        assert minimum_valid_count is not None
        output_validity = counts >= minimum_valid_count
    else:
        output_validity = counts > 0
    if method in (ReductionMethod.MEAN, ReductionMethod.SUM):
        safe = np.where(validity, values, 0)
        if method is ReductionMethod.SUM and values.dtype.kind in "biu":
            summed = _checked_integer_sum(safe, axes, output_dtype)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                summed = np.sum(safe, axis=axes, dtype=output_dtype)
        if values.dtype.kind in "fc" and np.all(np.isfinite(safe)):
            if np.any((counts > 0) & ~np.isfinite(summed)):
                raise OverflowError("floating reduction overflowed its canonical output dtype")
        if method is ReductionMethod.MEAN:
            result = np.zeros(np.shape(summed), dtype=output_dtype)
            np.divide(summed, counts, out=result, where=counts > 0)
        else:
            result = np.asarray(summed, dtype=output_dtype)
    else:
        masked = np.ma.array(values, mask=~validity)
        reduced = masked.min(axis=axes) if method is ReductionMethod.MIN else masked.max(axis=axes)
        result = np.asarray(np.ma.filled(reduced, 0), dtype=output_dtype)
    result = np.where(output_validity, result, np.zeros((), dtype=output_dtype))
    return np.asarray(result, dtype=output_dtype), np.asarray(output_validity, dtype=bool)


def _reduction_dtype(dtype: np.dtype, method: ReductionMethod) -> np.dtype:
    dtype = canonical_dtype(dtype)
    if method is ReductionMethod.MEAN:
        return np.dtype("<c16") if dtype.kind == "c" else np.dtype("<f8")
    if method is ReductionMethod.SUM:
        if dtype.kind == "b" or dtype.kind == "i":
            return np.dtype("<i8")
        if dtype.kind == "u":
            return np.dtype("<u8")
        if dtype.kind == "f":
            return np.dtype("<f8")
        if dtype.kind == "c":
            return np.dtype("<c16")
    return dtype


def _checked_integer_sum(
    safe_values: np.ndarray,
    axes: tuple[int, ...],
    output_dtype: np.dtype,
) -> np.ndarray:
    contributor_bound = math.prod(safe_values.shape[axis] for axis in axes)
    input_info = np.iinfo(safe_values.dtype) if safe_values.dtype.kind != "b" else None
    info = np.iinfo(output_dtype)
    statically_safe = safe_values.dtype.kind == "b" or (
        input_info is not None
        and input_info.max * contributor_bound <= info.max
        and (safe_values.dtype.kind != "i" or input_info.min * contributor_bound >= info.min)
    )
    if statically_safe:
        return np.sum(safe_values, axis=axes, dtype=output_dtype)
    exact = safe_values.astype(object)
    for axis in sorted(axes, reverse=True):
        exact = np.sum(exact, axis=axis, dtype=object)
    flat = np.asarray(exact, dtype=object).reshape(-1)
    if any(value < info.min or value > info.max for value in flat):
        raise OverflowError("integer SUM exceeds its canonical output dtype")
    return np.asarray(exact, dtype=output_dtype)


def _layout_mapping(layout: AxisLayout) -> tuple[tuple[int, ...], ...]:
    return tuple(layout.multi_index(index) for index in range(layout.storage_size))


def _axis_index_vector(layout: AxisLayout, position: int) -> np.ndarray:
    if not 0 <= position < len(layout.logical_shape):
        raise IndexError("layout axis position is out of range")
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        if layout.mode is AxisLayoutMode.RECT_C:
            stride = math.prod(layout.logical_shape[position + 1 :])
        else:
            stride = math.prod(layout.logical_shape[:position])
        return (
            np.arange(layout.storage_size, dtype=np.int64) // stride
        ) % layout.logical_shape[position]
    if layout.mode is AxisLayoutMode.EXPLICIT:
        assert layout.storage_to_multi is not None
        return np.fromiter(
            (multi[position] for multi in layout.storage_to_multi),
            dtype=np.int64,
            count=layout.storage_size,
        )
    assert layout.factors is not None
    axis_offset = 0
    for factor_index, factor in enumerate(layout.factors):
        next_offset = axis_offset + len(factor.logical_shape)
        if position < next_offset:
            child = _axis_index_vector(factor, position - axis_offset)
            after = math.prod(
                item.storage_size for item in layout.factors[factor_index + 1 :]
            )
            before = math.prod(
                item.storage_size for item in layout.factors[:factor_index]
            )
            physical = np.tile(
                np.repeat(np.arange(factor.storage_size, dtype=np.int64), after),
                before,
            )
            return child[physical]
        axis_offset = next_offset
    raise RuntimeError("PRODUCT axis resolution failed")


def _select_layout(
    layout: AxisLayout,
    position: int,
    indices: tuple[int, ...],
    drop: bool,
) -> AxisLayout:
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        shape = list(layout.logical_shape)
        if drop:
            shape.pop(position)
        else:
            shape[position] = len(indices)
        factory = AxisLayout.rect_c if layout.mode is AxisLayoutMode.RECT_C else AxisLayout.rect_f
        return factory(tuple(shape))
    if layout.mode is AxisLayoutMode.EXPLICIT:
        assert layout.storage_to_multi is not None
        remap = {old: new for new, old in enumerate(indices)}
        mapping: list[tuple[int, ...]] = []
        for multi in layout.storage_to_multi:
            if multi[position] not in remap:
                continue
            if drop:
                mapping.append(multi[:position] + multi[position + 1 :])
            else:
                mapping.append(
                    multi[:position]
                    + (remap[multi[position]],)
                    + multi[position + 1 :]
                )
        shape = list(layout.logical_shape)
        if drop:
            shape.pop(position)
        else:
            shape[position] = len(indices)
        return _layout_from_mapping(tuple(shape), tuple(mapping))
    assert layout.factors is not None
    axis_offset = 0
    factors = list(layout.factors)
    for factor_index, factor in enumerate(factors):
        next_offset = axis_offset + len(factor.logical_shape)
        if position < next_offset:
            factors[factor_index] = _select_layout(
                factor, position - axis_offset, indices, drop
            )
            return AxisLayout.product(*factors)
        axis_offset = next_offset
    raise RuntimeError("PRODUCT selection axis resolution failed")


def _reduce_layout(
    layout: AxisLayout,
    positions: tuple[int, ...],
    missing_policy: MissingPolicy,
) -> AxisLayout:
    if not positions:
        return layout
    position_set = set(positions)
    if layout.mode in (AxisLayoutMode.RECT_C, AxisLayoutMode.RECT_F):
        shape = tuple(
            size for index, size in enumerate(layout.logical_shape) if index not in position_set
        )
        factory = AxisLayout.rect_c if layout.mode is AxisLayoutMode.RECT_C else AxisLayout.rect_f
        return factory(shape)
    if layout.mode is AxisLayoutMode.EXPLICIT:
        assert layout.storage_to_multi is not None
        groups: OrderedDict[tuple[int, ...], int] = OrderedDict()
        for multi in layout.storage_to_multi:
            key = tuple(
                value for index, value in enumerate(multi) if index not in position_set
            )
            groups[key] = groups.get(key, 0) + 1
        expected = math.prod(layout.logical_shape[index] for index in positions)
        if not positions:
            expected = 1
        mapping = tuple(
            key
            for key, count in groups.items()
            if missing_policy is MissingPolicy.OMIT_MISSING or count == expected
        )
        shape = tuple(
            size for index, size in enumerate(layout.logical_shape) if index not in position_set
        )
        return _layout_from_mapping(shape, mapping)
    assert layout.factors is not None
    axis_offset = 0
    output_factors: list[AxisLayout] = []
    for factor in layout.factors:
        local_positions = tuple(
            position - axis_offset
            for position in positions
            if axis_offset <= position < axis_offset + len(factor.logical_shape)
        )
        output_factors.append(_reduce_layout(factor, local_positions, missing_policy))
        axis_offset += len(factor.logical_shape)
    return AxisLayout.product(*output_factors)


def _layout_from_mapping(
    logical_shape: tuple[int, ...],
    mapping: tuple[tuple[int, ...], ...],
) -> AxisLayout:
    logical_shape = tuple(logical_shape)
    mapping = tuple(tuple(index) for index in mapping)
    total = math.prod(logical_shape)
    if len(mapping) == total:
        rectangular_c = AxisLayout.rect_c(logical_shape)
        if mapping == _layout_mapping(rectangular_c):
            return rectangular_c
        rectangular_f = AxisLayout.rect_f(logical_shape)
        if mapping == _layout_mapping(rectangular_f):
            return rectangular_f
    return AxisLayout.explicit(logical_shape, mapping)


def _require_digest(value: str, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


__all__ = [
    "CommittedTransform",
    "DataTransformSpec",
    "MissingPolicy",
    "PreviewTransformedData",
    "Reduce",
    "ReductionMethod",
    "ReductionSpec",
    "Select",
    "TransformRecord",
    "TransformOrigin",
    "TransformRevision",
    "TransformedData",
    "TransformedSchema",
    "ValidityPolicy",
    "apply_transform",
    "commit_transform",
    "preview_transform",
]
