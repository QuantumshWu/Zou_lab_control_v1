"""Axis-total, sparse-preserving authoritative data transforms."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import math
from numbers import Integral
import numpy as np

from zlc_storage.canonical import canonical_digest, canonical_text, sha256_text

from ._arrays import canonical_dtype, immutable_array
from .axis import AxisId, AxisSpec
from .layout import AxisLayout, AxisLayoutMode
from .numeric import (
    _integer_sum_requires_object,
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
)
from .schema import DatasetSchema, ValueSchema
from .selection import (
    CoordinateRangeSelection,
    IndexRangeSelection,
    IndexSelection,
    Selection,
    resolve_selection_indices,
)
from .validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    Invalid,
    RowComponentValidity,
    Valid,
    ValidityContract,
    ValidityMode,
)
from .value import (
    DataBlock,
    DatasetRevisionRef,
    OwnedSnapshot,
    Value,
)


TRANSFORM_SPEC_SCHEMA = "zlc_data.DataTransformSpec"
COMMITTED_TRANSFORM_SCHEMA = "zlc_data.CommittedTransform"
TRANSFORMED_SCHEMA_SCHEMA = "zlc_data.TransformedSchema"


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
class DataTransformSpec:
    operations: tuple[Selection | ReductionSpec, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(
            not isinstance(operation, (Selection, ReductionSpec))
            for operation in operations
        ):
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
    _fingerprint: str | None = field(
        init=False,
        repr=False,
        compare=False,
        default=None,
    )

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
        if self.value_unit is not None:
            canonical_text(self.value_unit, "value_unit")
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
        fingerprint = self._fingerprint
        if fingerprint is None:
            from .transform_codec import transformed_schema_to_tree

            fingerprint = canonical_digest(transformed_schema_to_tree(self))
            object.__setattr__(self, "_fingerprint", fingerprint)
        return fingerprint

    def axis(self, axis_id: AxisId) -> AxisSpec:
        for axis in self.axes:
            if axis.axis_id == axis_id:
                return axis
        raise KeyError(axis_id)


@dataclass(frozen=True)
class CommittedTransform:
    input_schema_fingerprint: str
    spec: DataTransformSpec
    output_schema_fingerprint: str

    def __post_init__(self) -> None:
        sha256_text(self.input_schema_fingerprint, "input_schema_fingerprint")
        sha256_text(self.output_schema_fingerprint, "output_schema_fingerprint")
        if not isinstance(self.spec, DataTransformSpec) or not self.spec.operations:
            raise ValueError("CommittedTransform requires a non-empty DataTransformSpec")


@dataclass(frozen=True, eq=False)
class TransformedData:
    source_ref: DatasetRevisionRef
    transform: CommittedTransform
    values: np.ndarray
    validity: Valid | Invalid | RowComponentValidity
    schema: TransformedSchema
    __hash__ = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, DatasetRevisionRef):
            raise TypeError("source_ref must be DatasetRevisionRef")
        if not isinstance(self.transform, CommittedTransform):
            raise TypeError("transform must be CommittedTransform")
        if not isinstance(self.schema, TransformedSchema):
            raise TypeError("schema must be TransformedSchema")
        object.__setattr__(
            self,
            "values",
            immutable_array(self.values, dtype=self.schema.dtype, shape=self.schema.physical_shape),
        )
        _validate_transformed_validity(self.validity, self.schema)

    def expanded_validity(self) -> np.ndarray:
        """Return a read-only broadcast view aligned to ``values``."""

        return _expand_transformed_validity(self.validity, self.schema)


@dataclass
class _State:
    schema: TransformedSchema
    values: np.ndarray | None
    validity: Valid | Invalid | RowComponentValidity | None


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
    indices: range | tuple[int, ...],
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
        if isinstance(indices, range):
            selection = [slice(None)] * validity.mask.ndim
            selection[array_axis] = slice(indices.start, indices.stop)
            mask = validity.mask[tuple(selection)]
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
) -> CommittedTransform:
    """Validate and freeze an authoritative transform without touching dataset values."""

    if not isinstance(authoritative_spec, DataTransformSpec):
        raise TypeError("authoritative_spec must be DataTransformSpec")
    if not authoritative_spec.operations:
        raise ValueError("identity input uses None; an empty transform cannot be committed")
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    output_schema = _compile_transform_schema(schema, authoritative_spec)
    return CommittedTransform(
        schema.fingerprint,
        authoritative_spec,
        output_schema.fingerprint,
    )


def resolve_transformed_schema(
    schema: DatasetSchema,
    transform: CommittedTransform,
) -> TransformedSchema:
    """Resolve and verify a committed transform without allocating dataset values."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    if transform.input_schema_fingerprint != schema.fingerprint:
        raise ValueError("CommittedTransform input schema fingerprint is stale")
    output = _compile_transform_schema(schema, transform.spec)
    if output.fingerprint != transform.output_schema_fingerprint:
        raise ValueError("CommittedTransform output schema fingerprint is inconsistent")
    return output


def resolve_value_transform_schema(
    schema: ValueSchema,
    spec: DataTransformSpec,
) -> ValueSchema:
    """Resolve one named-axis transform over a single :class:`Value`.

    A ``Value`` has data axes only.  Consequently every operation must name an
    axis that is still present in ``schema.data_axes``; repeat/point cell-axis
    operations are rejected as absent rather than being guessed or fabricated.
    """

    if not isinstance(schema, ValueSchema):
        raise TypeError("schema must be ValueSchema")
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec")
    state = _run_operations(
        _value_source_state(schema, values=None, validity=None),
        spec,
    )
    return _value_schema_from_transformed(state.schema)


def apply_value_transform(
    value: Value,
    spec: DataTransformSpec,
) -> Value:
    """Apply one named-axis transform directly to an immutable ``Value``.

    Range selections retain their named axes, index selections explicitly drop
    only the selected axis, and reductions drop only their declared axes.  No
    trailing data axis is flattened, selected implicitly, or averaged.
    """

    if not isinstance(value, Value):
        raise TypeError("value must be Value")
    if not isinstance(spec, DataTransformSpec):
        raise TypeError("spec must be DataTransformSpec")
    state = _run_operations(
        _value_source_state(
            value.schema,
            values=value.values.reshape((1, *value.schema.data_shape)),
            validity=_value_source_validity(value),
        ),
        spec,
    )
    if state.values is None or state.validity is None:
        raise RuntimeError("materialized Value transform produced no data")
    schema = _value_schema_from_transformed(state.schema)
    return Value(
        state.values.reshape(schema.data_shape),
        _value_validity_from_transformed(state.validity),
        schema,
    )


def _compile_transform_schema(
    schema: DatasetSchema,
    spec: DataTransformSpec,
) -> TransformedSchema:
    return _run_operations(
        _source_state(schema, values=None, validity=None), spec
    ).schema


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
        transform,
        state.values,
        state.validity,
        state.schema,
    )


def _require_materialized_snapshot_step(schema: TransformedSchema, step) -> None:
    """Keep the DatasetSchema bridge within its declared cell semantics."""

    cell_axes = {axis.axis_id: axis for axis in schema.cell_axes}
    if isinstance(step, ReductionSpec):
        if any(axis_id in cell_axes for axis_id in step.axis_ids):
            raise ValueError(
                "bounded dataset snapshots do not reduce repeat/point axes"
            )
        return
    term = step.terms[0]
    axis = cell_axes.get(term.axis_id)
    if axis is None:
        return
    singleton_drop = (
        isinstance(term, IndexSelection) and axis.size == 1 and term.index == 0
    )
    full_noop = (
        isinstance(term, IndexRangeSelection)
        and term.start == 0
        and term.stop == axis.size
    )
    if not (singleton_drop or full_noop):
        raise ValueError(
            "materialized dataset snapshots only select a singleton cell axis or "
            "retain its full index range"
        )


def _validate_materialized_snapshot_transform(
    source_schema: DatasetSchema,
    transform: CommittedTransform,
) -> None:
    """Validate the cell-preserving transform used by snapshot materialization."""

    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    if transform.input_schema_fingerprint != source_schema.fingerprint:
        raise ValueError("CommittedTransform input schema fingerprint is stale")
    state = _State(_source_transformed_schema(source_schema), None, None)
    for operation in transform.spec.operations:
        steps = (
            tuple(Selection((term,)) for term in operation.terms)
            if isinstance(operation, Selection)
            else (operation,)
        )
        for step in steps:
            _require_materialized_snapshot_step(state.schema, step)
            state = (
                _apply_selection(state, step)
                if isinstance(step, Selection)
                else _apply_reduction(state, step)
            )
    if state.schema.fingerprint != transform.output_schema_fingerprint:
        raise RuntimeError("transform validation disagrees with committed output schema")


def materialize_transformed_snapshot(
    snapshot: OwnedSnapshot,
    transform: CommittedTransform,
    *,
    output_ref: DatasetRevisionRef,
    output_schema: DatasetSchema,
) -> OwnedSnapshot:
    """Execute one cell-preserving transform into its final DataBlock."""

    if not isinstance(output_ref, DatasetRevisionRef):
        raise TypeError("output_ref must be DatasetRevisionRef")
    if not isinstance(output_schema, DatasetSchema):
        raise TypeError("output_schema must be DatasetSchema")
    if output_ref.block_id == snapshot.ref.block_id:
        raise ValueError("a transformed snapshot cannot reuse its source BlockId")
    if output_ref.schema_fingerprint != output_schema.fingerprint:
        raise ValueError("output_ref schema fingerprint differs from output_schema")
    if _source_transformed_schema(output_schema).fingerprint != transform.output_schema_fingerprint:
        raise ValueError("output_schema differs from the committed transform")
    _validate_materialized_snapshot_transform(snapshot.block.schema, transform)
    state = _execute_transform(snapshot.block, transform.spec)
    if state.schema.fingerprint != transform.output_schema_fingerprint:
        raise RuntimeError("transform execution disagrees with its committed output schema")
    assert state.values is not None and state.validity is not None
    validity: Valid | Invalid | CellValidity | ComponentValidity
    if isinstance(state.validity, (Valid, Invalid)):
        validity = state.validity
    elif state.validity.axis_ids:
        validity = ComponentValidity(
            state.validity.axis_ids,
            state.validity.mask.reshape(
                output_schema.repeat_axis.size,
                output_schema.point_layout.storage_size,
                *state.validity.mask.shape[1:],
            ),
        )
    else:
        validity = CellValidity(
            state.validity.mask.reshape(
                output_schema.repeat_axis.size,
                output_schema.point_layout.storage_size,
            )
        )
    block = DataBlock(
        output_ref.block_id,
        output_ref.revision,
        state.values.reshape(output_schema.physical_shape),
        validity,
        output_schema,
    )
    return OwnedSnapshot(output_ref, block)


def _execute_transform(block: DataBlock, spec: DataTransformSpec) -> _State:
    state = _source_state(
        block.schema,
        values=block.values.reshape((-1, *block.schema.cell_schema.data_shape)),
        validity=_source_validity(block),
    )
    return _run_operations(state, spec)


def _value_source_state(
    schema: ValueSchema,
    *,
    values: np.ndarray | None,
    validity: Valid | Invalid | RowComponentValidity | None,
) -> _State:
    """Enter the shared transform engine without inventing a cell axis."""

    if not isinstance(schema, ValueSchema):
        raise TypeError("schema must be ValueSchema")
    transformed = TransformedSchema(
        (),
        AxisLayout.rect_c(()),
        schema.data_axes,
        (
            schema.validity_contract.component_axis_ids
            if schema.validity_contract.mode is ValidityMode.COMPONENTS
            else ()
        ),
        schema.dtype,
        schema.value_unit,
    )
    return _State(transformed, values, validity)


def _value_source_validity(
    value: Value,
) -> Valid | Invalid | RowComponentValidity:
    validity = value.validity
    if isinstance(validity, (Valid, Invalid)):
        return validity
    if isinstance(validity, ComponentValidity):
        return RowComponentValidity(
            validity.axis_ids,
            validity.mask.reshape((1, *validity.mask.shape)),
        )
    raise TypeError(f"unsupported Value validity {type(validity).__name__}")


def _value_schema_from_transformed(schema: TransformedSchema) -> ValueSchema:
    if (
        schema.cell_axes
        or schema.cell_layout.logical_shape
        or schema.cell_layout.storage_size != 1
    ):
        raise RuntimeError("Value transform cannot produce cell axes")
    validity_contract = (
        ValidityContract.components(*schema.validity_axis_ids)
        if schema.validity_axis_ids
        else ValidityContract.value()
    )
    return ValueSchema(
        schema.data_axes,
        validity_contract,
        schema.dtype,
        schema.value_unit,
    )


def _value_validity_from_transformed(
    validity: Valid | Invalid | RowComponentValidity,
) -> Valid | Invalid | ComponentValidity:
    if isinstance(validity, (Valid, Invalid)):
        return validity
    if not isinstance(validity, RowComponentValidity):
        raise TypeError(
            "transformed Value validity must be Valid, Invalid, or "
            "RowComponentValidity"
        )
    if validity.mask.shape[0] != 1:
        raise RuntimeError("Value transform produced more than one physical row")
    if validity.axis_ids:
        return ComponentValidity(validity.axis_ids, validity.mask[0])
    if validity.mask.shape != (1,):
        raise RuntimeError("scalar Value validity has an unexpected shape")
    return VALID if bool(validity.mask[0]) else INVALID


def _source_state(
    schema: DatasetSchema,
    *,
    values: np.ndarray | None,
    validity: Valid | Invalid | RowComponentValidity | None,
) -> _State:
    return _State(_source_transformed_schema(schema), values, validity)


def _source_transformed_schema(schema: DatasetSchema) -> TransformedSchema:
    """Project one DatasetSchema into the transform domain exactly once."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    cell_axes = (schema.repeat_axis, *schema.point_axes)
    transformed_schema = TransformedSchema(
        cell_axes,
        schema.cell_layout,
        schema.cell_schema.data_axes,
        schema.cell_schema.validity_contract.component_axis_ids
        if schema.cell_schema.validity_contract.mode is ValidityMode.COMPONENTS
        else (),
        schema.cell_schema.dtype,
        schema.cell_schema.value_unit,
    )
    return transformed_schema


def _run_operations(state: _State, spec: DataTransformSpec) -> _State:
    for operation in spec.operations:
        if isinstance(operation, Selection):
            state = _apply_selection(state, operation)
        else:
            state = _apply_reduction(state, operation)
    return state


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


def _selected_axis(
    axis: AxisSpec,
    indices: range | tuple[int, ...],
) -> AxisSpec:
    if isinstance(indices, range) and indices.start == 0 and indices.stop == axis.size:
        return axis
    if axis.coordinates is None:
        if not isinstance(indices, range):  # coordinate selections require declared coordinates
            raise RuntimeError("implicit-coordinate selection is not contiguous")
        coordinates = None
        index_origin = axis.index_origin + indices.start
    elif isinstance(indices, range):
        coordinates = axis.coordinates[indices.start : indices.stop]
        index_origin = 0
    else:
        coordinates = tuple(axis.coordinates[index] for index in indices)
        index_origin = 0
    return AxisSpec(
        axis.axis_id,
        axis.name,
        axis.role,
        len(indices),
        coordinates,
        axis.unit,
        axis.coordinate_frame,
        index_origin,
    )


def _select_cell_axis(state: _State, position: int, term) -> _State:
    axis = state.schema.cell_axes[position]
    indices, drop = resolve_selection_indices(axis, term)
    if (
        not drop
        and isinstance(indices, range)
        and indices.start == 0
        and indices.stop == axis.size
    ):
        return state
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
    elif drop and axis.size == 1:
        # Removing a singleton logical cell axis changes only the schema.  Every
        # physical row survives in the same order, so boolean indexing would be
        # a full-array copy with no data-semantic effect.
        values = state.values
        validity = state.validity
    else:
        logical_indices = state.schema.cell_layout.axis_indices(position)
        if isinstance(indices, range):
            selected = logical_indices >= indices.start
            np.logical_and(logical_indices < indices.stop, selected, out=selected)
        else:
            selected = np.isin(logical_indices, indices)
        values = state.values[selected]
        assert state.validity is not None
        validity = _select_validity_rows(state.validity, selected)
        if values.shape[0] != layout.storage_size:
            raise RuntimeError("cell selection layout disagrees with selected physical rows")
    return _State(schema, values, validity)


def _select_data_axis(state: _State, position: int, term) -> _State:
    axis = state.schema.data_axes[position]
    indices, drop = resolve_selection_indices(axis, term)
    if (
        not drop
        and isinstance(indices, range)
        and indices.start == 0
        and indices.stop == axis.size
    ):
        return state
    array_axis = 1 + position
    axes = list(state.schema.data_axes)
    if drop:
        axes.pop(position)
        values = None if state.values is None else np.take(state.values, indices[0], axis=array_axis)
    else:
        axes[position] = _selected_axis(axis, indices)
        if state.values is None:
            values = None
        elif isinstance(indices, range):
            selection = [slice(None)] * state.values.ndim
            selection[array_axis] = slice(indices.start, indices.stop)
            values = state.values[tuple(selection)]
        else:
            values = np.take(state.values, indices, axis=array_axis)
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
    return _State(schema, values, validity)


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
        return _State(schema, None, None)
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
        )

    groups: OrderedDict[tuple[int, ...], list[int]] = OrderedDict()
    for row in range(state.schema.cell_layout.storage_size):
        multi = state.schema.cell_layout.multi_index(row)
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

    if len(output_mapping) != output_layout.storage_size or any(
        multi != output_layout.multi_index(index)
        for index, multi in enumerate(output_mapping)
    ):
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
        layout.axis_indices(position)
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
    if values.dtype.kind in "fc" and np.any(
        np.logical_and(validity, ~np.isfinite(values))
    ):
        raise ValueError("reduction received a valid non-finite value")
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
            summed = checked_numeric_sum(
                safe,
                axes,
                output_dtype=output_dtype,
            )
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
        return canonical_mean_dtype(dtype)
    if method is ReductionMethod.SUM:
        return canonical_sum_dtype(dtype)
    return dtype


def _select_layout(
    layout: AxisLayout,
    position: int,
    indices: range | tuple[int, ...],
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
        remap = (
            None
            if isinstance(indices, range)
            else {old: new for new, old in enumerate(indices)}
        )
        mapping: list[tuple[int, ...]] = []
        for multi in layout.storage_to_multi:
            old = multi[position]
            if remap is None:
                if not indices.start <= old < indices.stop:
                    continue
                new = old - indices.start
            else:
                if old not in remap:
                    continue
                new = remap[old]
            if drop:
                mapping.append(multi[:position] + multi[position + 1 :])
            else:
                mapping.append(
                    multi[:position]
                    + (new,)
                    + multi[position + 1 :]
                )
        shape = list(layout.logical_shape)
        if drop:
            shape.pop(position)
        else:
            shape[position] = len(indices)
        return AxisLayout.from_mapping(tuple(shape), tuple(mapping))
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
        return AxisLayout.from_mapping(shape, mapping)
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


__all__ = [
    "CommittedTransform",
    "DataTransformSpec",
    "MissingPolicy",
    "ReductionMethod",
    "ReductionSpec",
    "TransformedData",
    "TransformedSchema",
    "ValidityPolicy",
    "apply_transform",
    "apply_value_transform",
    "commit_transform",
    "materialize_transformed_snapshot",
    "resolve_transformed_schema",
    "resolve_value_transform_schema",
]
