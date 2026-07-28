"""Declarative contracts and source-aware validation for headless views."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from zlc_data import (
    COMPONENT,
    MONITOR_HISTORY,
    READOUT_EVENT,
    REPEAT,
    SCALAR,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisId,
    AxisRoleId,
    AxisSourceRef,
    AxisSpec,
    DatasetSchema,
    FitResultBatch,
    PointColumn,
    ResolvedPointRows,
    Selection,
    resolve_point_rows,
    resolve_selection_indices,
)
from zlc_data.axis import point_ordinal_axis
from zlc_storage import canonical_text

from .model import (
    DATASET_VIEW_INTENTS,
    AxisAddress,
    AxisRolePolicy,
    AxisViewRole,
    DisplayReductionMethod,
    DisplaySlot,
    FixedIndex,
    LatestNonempty,
    SourceViewBinding,
    ViewContract,
    ViewIntent,
    ViewSpec,
)


IMAGE_CONTRACT = ViewContract(
    ViewIntent.IMAGE,
    (
        DisplaySlot(AxisViewRole.IMAGE_X, (SPATIAL_X, SCAN_POINT)),
        DisplaySlot(AxisViewRole.IMAGE_Y, (SPATIAL_Y, SCAN_POINT)),
    ),
    (
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SITE, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SPATIAL_X, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SPATIAL_Y, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
    ),
    (REPEAT,),
)


CURVE_CONTRACT = ViewContract(
    ViewIntent.CURVE,
    (
        DisplaySlot(
            AxisViewRole.X,
            (
                SPECTRAL,
                SCAN_POINT,
                MONITOR_HISTORY,
                SITE,
                SPATIAL_X,
                SPATIAL_Y,
            ),
        ),
    ),
    (
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SITE, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        AxisRolePolicy(SPATIAL_X, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SPATIAL_Y, (AxisViewRole.SELECTED, AxisViewRole.FACET, AxisViewRole.BATCH)),
    ),
    (REPEAT, SPATIAL_X, SPATIAL_Y),
)


HISTOGRAM_CONTRACT = ViewContract(
    ViewIntent.HISTOGRAM,
    (),
    (
        AxisRolePolicy(
            READOUT_EVENT,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            MONITOR_HISTORY,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            SITE,
            (AxisViewRole.SAMPLE, AxisViewRole.BATCH, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            COMPONENT,
            (AxisViewRole.SAMPLE, AxisViewRole.BATCH, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            SCAN_POINT,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            SPECTRAL,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            SPATIAL_X,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
        AxisRolePolicy(
            SPATIAL_Y,
            (AxisViewRole.SAMPLE, AxisViewRole.SELECTED, AxisViewRole.FACET),
        ),
    ),
    (REPEAT,),
)


METER_CONTRACT = ViewContract(
    ViewIntent.METER,
    (),
    (
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(SITE, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(SPATIAL_X, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
        AxisRolePolicy(SPATIAL_Y, (AxisViewRole.SELECTED, AxisViewRole.FACET)),
    ),
    (REPEAT, SPATIAL_X, SPATIAL_Y),
)


@dataclass(frozen=True, slots=True)
class DocumentViewContract:
    """A view fed by an authored document rather than a DatasetSchema."""

    intent: ViewIntent
    source_schema: str

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ViewIntent):
            raise TypeError("intent must be ViewIntent")
        if self.intent in DATASET_VIEW_INTENTS:
            raise ValueError("dataset-fed intents require ViewContract")
        source_schema = canonical_text(self.source_schema, "document source_schema")
        if "." not in source_schema:
            raise ValueError("document source_schema must be an owner-qualified name")
        object.__setattr__(self, "source_schema", source_schema)


PULSE_CONTRACT = DocumentViewContract(
    ViewIntent.PULSE,
    "zlc_pulse.PulseTimelineDocument",
)


VIEW_CONTRACTS: Mapping[
    ViewIntent,
    ViewContract | DocumentViewContract,
] = MappingProxyType(
    {
        ViewIntent.IMAGE: IMAGE_CONTRACT,
        ViewIntent.CURVE: CURVE_CONTRACT,
        ViewIntent.HISTOGRAM: HISTOGRAM_CONTRACT,
        ViewIntent.METER: METER_CONTRACT,
        ViewIntent.PULSE: PULSE_CONTRACT,
    }
)


def contract_for(intent: ViewIntent) -> ViewContract | DocumentViewContract:
    if not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent")
    return VIEW_CONTRACTS[intent]


def dataset_contract_for(intent: ViewIntent) -> ViewContract:
    contract = contract_for(intent)
    if not isinstance(contract, ViewContract):
        raise ValueError(
            f"{intent.value} is document-fed from {contract.source_schema}; "
            "it has no DataBlock ViewSpec/evaluator path"
        )
    return contract


def _source_key(source: AxisSourceRef) -> tuple[str, str]:
    return (
        source.kind,
        "" if source.axis_id is None else source.axis_id.value,
    )


def _tensor_axes(schema: DatasetSchema) -> tuple[AxisSpec, ...]:
    return (schema.repeat_axis, *schema.cell_schema.data_axes)


def _dataset_sources(schema: DatasetSchema) -> tuple[AxisSourceRef, ...]:
    """Return the declared source vocabulary in producer-owned order."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    result = [AxisSourceRef.tensor(schema.repeat_axis.axis_id)]
    result.append(AxisSourceRef.point_rows())
    result.extend(
        AxisSourceRef.point_coordinate(column.coordinate_id)
        for column in schema.point_table.columns
    )
    # The synthetic ordinal is a stable fallback after authored coordinates,
    # never a competitor that hides the producer's declared X column.
    result.append(AxisSourceRef.point_ordinal())
    if schema.grid_topology is not None:
        result.extend(
            AxisSourceRef.grid_dimension(dimension_id)
            for dimension_id in schema.grid_topology.dimension_ids
        )
    result.extend(
        AxisSourceRef.tensor(axis.axis_id)
        for axis in schema.cell_schema.data_axes
    )
    return tuple(result)


def _tensor_axis(schema: DatasetSchema, source: AxisSourceRef) -> AxisSpec:
    if source.kind != AxisSourceRef.TENSOR or source.axis_id is None:
        raise KeyError(source)
    for axis in _tensor_axes(schema):
        if axis.axis_id == source.axis_id:
            return axis
    raise KeyError(source)


def _fit_display_selection_indices(
    schema: DatasetSchema,
    result: FitResultBatch,
) -> tuple[tuple[AxisSourceRef, tuple[int, ...]], ...]:
    """Validate the sole range-preserving Fit selection against source axes.

    A saved Curve/Image Fit is displayed over the unchanged source Figure; the
    committed selection limits only its overlay.  This private contract owner
    resolves that authority once so suggestion, immutable Figure validation,
    and both render projections cannot drift into separate interpretations.
    Histogram transforms have their own exact sample/bin projection and never
    pass through this path.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    transform = result.spec.committed_transform
    if transform.source_schema_fingerprint != schema.fingerprint:
        raise ValueError("Fit transform belongs to another source schema")
    operations = tuple(transform.spec.operations)
    if not operations:
        return ()
    if len(operations) != 1 or not isinstance(operations[0], Selection):
        raise ValueError(
            "Curve/Image Fit display supports identity or one range-preserving "
            "Selection authority"
        )

    selection = operations[0]
    independent = tuple(result.spec.independent_sources)
    fit_axes = dict(zip(independent, result.fit_axis_specs, strict=True))
    terms = {term.axis_id: term for term in selection.terms}
    resolved = []
    for source in independent:
        if source.axis_id not in terms:
            continue
        if source.kind != AxisSourceRef.TENSOR:
            raise ValueError("Fit Selection may name only tensor independent sources")
        axis = _tensor_axis(schema, source)
        indices, drops_axis = resolve_selection_indices(axis, terms[source.axis_id])
        if drops_axis:
            raise ValueError("Fit display cannot replay an axis-collapsing Selection")
        exact_indices = tuple(indices)
        fit_axis = fit_axes[source]
        if len(exact_indices) != fit_axis.size or any(
            axis.coordinate_at(source_index) != fit_axis.coordinate_at(output_index)
            for output_index, source_index in enumerate(exact_indices)
        ):
            raise ValueError("Fit Selection coordinates differ from its fitted axis")
        resolved.append((source, exact_indices))
    if len(resolved) != len(selection.terms):
        raise ValueError("Fit Selection names a non-independent source")
    return tuple(resolved)


def _point_column(schema: DatasetSchema, source: AxisSourceRef) -> PointColumn:
    if source.kind not in {
        AxisSourceRef.POINT_COORDINATE,
        AxisSourceRef.GRID_DIMENSION,
    } or source.axis_id is None:
        raise KeyError(source)
    return schema.point_table.column(source.axis_id)


def _source_role(schema: DatasetSchema, source: AxisSourceRef) -> AxisRoleId:
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).role
    if source.kind == AxisSourceRef.POINT_ROWS:
        return SCAN_POINT
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).role
    return _point_column(schema, source).role


def _source_name(schema: DatasetSchema, source: AxisSourceRef) -> str:
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).name
    if source.kind == AxisSourceRef.POINT_ROWS:
        return "points"
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).name
    return _point_column(schema, source).name


def _source_unit(schema: DatasetSchema, source: AxisSourceRef) -> str | None:
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).unit
    if source.kind == AxisSourceRef.POINT_ROWS:
        return None
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).unit
    return _point_column(schema, source).unit


def _source_coordinate_frame(schema: DatasetSchema, source: AxisSourceRef):
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).coordinate_frame
    if source.kind == AxisSourceRef.POINT_ROWS:
        return None
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).coordinate_frame
    return _point_column(schema, source).coordinate_frame


def _source_cardinality(schema: DatasetSchema, source: AxisSourceRef) -> int:
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).size
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).size
    if source.kind in {
        AxisSourceRef.POINT_ROWS,
        AxisSourceRef.POINT_COORDINATE,
    }:
        return schema.point_table.row_count
    topology = schema.grid_topology
    if topology is None or source.axis_id not in topology.dimension_ids:
        raise KeyError(source)
    position = topology.dimension_ids.index(source.axis_id)
    return len(topology.coordinate_domains[position])


def _source_coordinate(
    schema: DatasetSchema,
    source: AxisSourceRef,
    index: int,
):
    if source.kind == AxisSourceRef.TENSOR:
        return _tensor_axis(schema, source).coordinate_at(index)
    if source.kind == AxisSourceRef.POINT_ROWS:
        if not 0 <= index < schema.point_table.row_count:
            raise IndexError("point ordinal is outside PointTable")
        return index
    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return point_ordinal_axis(schema.point_table.row_count).coordinate_at(index)
    if source.kind == AxisSourceRef.POINT_COORDINATE:
        return _point_column(schema, source).values[index]
    topology = schema.grid_topology
    if topology is None or source.axis_id not in topology.dimension_ids:
        raise KeyError(source)
    position = topology.dimension_ids.index(source.axis_id)
    return topology.coordinate_domains[position][index]


def _point_bindings(view: ViewSpec) -> tuple[SourceViewBinding, ...]:
    return tuple(
        binding
        for binding in view.source_bindings
        if binding.source.kind != AxisSourceRef.TENSOR
    )


def _resolve_view_point_rows(
    schema: DatasetSchema,
    view: ViewSpec,
) -> ResolvedPointRows:
    """Validate point-source roles and resolve their one shared row domain."""

    bindings = _point_bindings(view)
    raw = tuple(
        binding
        for binding in bindings
        if binding.source.kind
        in {
            AxisSourceRef.POINT_ROWS,
            AxisSourceRef.POINT_ORDINAL,
            AxisSourceRef.POINT_COORDINATE,
        }
    )
    topology_bindings = tuple(
        binding
        for binding in bindings
        if binding.source.kind == AxisSourceRef.GRID_DIMENSION
    )
    if raw and topology_bindings:
        raise ValueError("raw point sources and GridDimension sources cannot be mixed")

    point_rows_bindings = tuple(
        binding
        for binding in raw
        if binding.source.kind == AxisSourceRef.POINT_ROWS
    )
    if len(point_rows_bindings) > 1:
        raise ValueError("a ViewSpec may consume PointRows only once")

    raw_image = tuple(
        binding
        for binding in raw
        if binding.role in (AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y)
    )
    if len(raw_image) > 1:
        raise ValueError("two correlated raw point sources cannot form an image plane")

    point_facets = tuple(
        binding for binding in bindings if binding.role is AxisViewRole.FACET
    )
    if len(point_facets) > 1:
        raise ValueError("the point domain may have at most one FACET source")

    for binding in bindings:
        source = binding.source
        role = binding.role
        if source.kind == AxisSourceRef.POINT_ROWS:
            allowed = {
                AxisViewRole.SAMPLE,
                AxisViewRole.BATCH,
                AxisViewRole.FACET,
                AxisViewRole.REDUCED,
            }
        elif source.kind == AxisSourceRef.POINT_ORDINAL:
            allowed = {
                AxisViewRole.X,
                AxisViewRole.IMAGE_X,
                AxisViewRole.IMAGE_Y,
            }
        elif source.kind == AxisSourceRef.POINT_COORDINATE:
            column = _point_column(schema, source)
            allowed = {AxisViewRole.BATCH, AxisViewRole.FACET}
            if column.value_kind == PointColumn.NUMERIC:
                allowed |= {
                    AxisViewRole.X,
                    AxisViewRole.IMAGE_X,
                    AxisViewRole.IMAGE_Y,
                }
        else:
            topology = schema.grid_topology
            if topology is None or source.axis_id not in topology.dimension_ids:
                raise ValueError("GridDimension source is absent from GridTopology")
            allowed = {
                AxisViewRole.X,
                AxisViewRole.IMAGE_X,
                AxisViewRole.IMAGE_Y,
                AxisViewRole.BATCH,
                AxisViewRole.FACET,
                AxisViewRole.SELECTED,
                AxisViewRole.REDUCED,
            }
        if role not in allowed:
            raise ValueError(f"{source.kind} cannot use {role.value}")

    if point_rows_bindings:
        point_rows_role = point_rows_bindings[0].role
        coordinate_x = any(
            binding.source.kind == AxisSourceRef.POINT_COORDINATE
            and binding.role in (AxisViewRole.X, AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y)
            for binding in raw
        )
        if coordinate_x and point_rows_role in {
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.REDUCED,
        }:
            raise ValueError("PointRows consumption conflicts with a point coordinate axis")
        if point_rows_role is AxisViewRole.SAMPLE and view.intent is not ViewIntent.HISTOGRAM:
            raise ValueError("PointRows SAMPLE is valid only for HISTOGRAM")

    group_sources = tuple(
        binding.source
        for binding in bindings
        if binding.role in (AxisViewRole.BATCH, AxisViewRole.FACET)
    )
    selected_ordinals = _resolve_selected_point_ordinals(schema, view)
    resolved = resolve_point_rows(
        schema.point_table,
        schema.grid_topology,
        point_ordinals=selected_ordinals,
        group_sources=group_sources,
    )

    if topology_bindings:
        topology = schema.grid_topology
        assert topology is not None
        bound_ids = {binding.source.axis_id for binding in topology_bindings}
        if (
            len(resolved.surviving_ordinals) > 1
            and bound_ids != set(topology.dimension_ids)
        ):
            raise ValueError(
                "every GridTopology dimension must be bound unless one row survives"
            )

    if len(resolved.surviving_ordinals) > 1 and not bindings:
        raise ValueError("multiple point rows require an explicit point source binding")

    consumes_group_members = any(
        binding.role
        in {
            AxisViewRole.X,
            AxisViewRole.IMAGE_X,
            AxisViewRole.IMAGE_Y,
            AxisViewRole.SAMPLE,
            AxisViewRole.REDUCED,
        }
        for binding in bindings
    ) or bool(topology_bindings)
    if not consumes_group_members and any(
        len(members) > 1 for members in resolved.group_member_ordinals
    ):
        raise ValueError("point grouping leaves multiple rows unresolved")
    return resolved


def _resolve_selected_point_ordinals(
    schema: DatasetSchema,
    view: ViewSpec,
    *,
    ignore_selected_sources: tuple[AxisSourceRef, ...] = (),
) -> tuple[int, ...]:
    """Resolve the exact physical rows surviving every point-domain choice.

    ``resolve_point_rows`` owns the authored row filter and point grouping.
    Grid ``SELECTED`` bindings are a frontend view concern, so this existing
    Figure-contract owner applies them once for validation, evaluation, and
    editor candidate checks.  Keeping this here prevents each surface from
    inventing a subtly different sparse-grid rule.
    """

    ignored = tuple(ignore_selected_sources)
    if any(not isinstance(source, AxisSourceRef) for source in ignored):
        raise TypeError("ignore_selected_sources must contain AxisSourceRef values")
    if len(set(ignored)) != len(ignored):
        raise ValueError("ignore_selected_sources must be unique")
    ordinals = resolve_point_rows(
        schema.point_table,
        schema.grid_topology,
        point_ordinals=view.point_ordinals,
    ).surviving_ordinals
    topology = schema.grid_topology
    for binding in view.source_bindings:
        if (
            binding.source.kind != AxisSourceRef.GRID_DIMENSION
            or binding.role is not AxisViewRole.SELECTED
            or binding.source in ignored
        ):
            continue
        if not isinstance(binding.selector, FixedIndex):
            raise TypeError("GridDimension SELECTED requires FixedIndex")
        if topology is None or binding.source.axis_id not in topology.dimension_ids:
            raise ValueError("selected GridDimension is absent from GridTopology")
        position = topology.dimension_ids.index(binding.source.axis_id)
        ordinals = tuple(
            ordinal
            for ordinal in ordinals
            if topology.row_to_cell[ordinal][position] == binding.selector.index
        )
    if not ordinals:
        raise ValueError("point selection contains no physical row")
    return ordinals


def _resolved_point_group_records(
    schema: DatasetSchema,
    view: ViewSpec,
) -> tuple[
    tuple[
        tuple[AxisAddress, ...],
        tuple[AxisAddress, ...],
        tuple[int, ...],
        int,
    ],
    ...,
]:
    """Project canonical point groups into the addresses used by Figure cells.

    This is the sole owner of the logical per-source indices for correlated
    point groups.  Evaluation and focused-panel reconstruction must consume the
    same records; a logical group index is not a physical point ordinal.
    """

    resolved = _resolve_view_point_rows(schema, view)
    binding_by_source = {
        binding.source: binding for binding in view.source_bindings
    }
    value_indices: dict[AxisSourceRef, dict[object, int]] = {
        source: {} for source in resolved.group_sources
    }
    records = []
    for group_index, (address, values, members) in enumerate(
        zip(
            resolved.group_addresses,
            resolved.group_values,
            resolved.group_member_ordinals,
            strict=True,
        )
    ):
        facet = []
        batch = []
        for position, source in enumerate(resolved.group_sources):
            coordinate = values[position]
            indices = value_indices[source]
            logical_index = indices.setdefault(coordinate, len(indices))
            if source.kind == AxisSourceRef.GRID_DIMENSION:
                logical_index = int(address[position])
            item = AxisAddress(
                source,
                _source_name(schema, source),
                _source_role(schema, source),
                logical_index,
                coordinate,
            )
            role = binding_by_source[source].role
            if role is AxisViewRole.FACET:
                facet.append(item)
            elif role is AxisViewRole.BATCH:
                batch.append(item)
        records.append((tuple(facet), tuple(batch), members, group_index))
    return tuple(records)


def _repeat_binding_allowed(intent: ViewIntent, binding: SourceViewBinding) -> bool:
    role = binding.role
    if intent is ViewIntent.IMAGE:
        return role in {AxisViewRole.REDUCED, AxisViewRole.SELECTED, AxisViewRole.FACET}
    if intent is ViewIntent.CURVE:
        return role in {
            AxisViewRole.REDUCED,
            AxisViewRole.SELECTED,
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
        }
    if intent is ViewIntent.HISTOGRAM:
        return role in {
            AxisViewRole.SAMPLE,
            AxisViewRole.BATCH,
            AxisViewRole.REDUCED,
            AxisViewRole.SELECTED,
            AxisViewRole.FACET,
        }
    return role in {AxisViewRole.SELECTED, AxisViewRole.REDUCED}


def _display_dtype_issue(schema: DatasetSchema, intent: ViewIntent) -> str | None:
    if schema.cell_schema.dtype.kind == "c" and intent is not ViewIntent.METER:
        return (
            f"{intent.value} does not define a complex-value projection; "
            "select an explicit real, imaginary, magnitude, or phase transform first"
        )
    return None


def validate_view_spec(
    schema: DatasetSchema,
    spec: ViewSpec,
    contract: ViewContract | None = None,
) -> None:
    """Validate total tensor coverage and source-aware presentation safety."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(spec, ViewSpec):
        raise TypeError("spec must be ViewSpec")
    contract = dataset_contract_for(spec.intent) if contract is None else contract
    if not isinstance(contract, ViewContract) or contract.intent is not spec.intent:
        raise ValueError("view contract does not match ViewSpec intent")
    if spec.schema_fingerprint != schema.fingerprint:
        raise ValueError("ViewSpec schema fingerprint is stale")
    dtype_issue = _display_dtype_issue(schema, spec.intent)
    if dtype_issue is not None:
        raise ValueError(dtype_issue)

    declared = set(_dataset_sources(schema))
    actual = {binding.source for binding in spec.source_bindings}
    unknown = actual - declared
    if unknown:
        raise ValueError(f"ViewSpec references absent sources: {tuple(sorted(unknown))}")
    required_tensor = {
        AxisSourceRef.tensor(axis.axis_id) for axis in _tensor_axes(schema)
    }
    missing_tensor = required_tensor - actual
    if missing_tensor:
        raise ValueError(
            f"ViewSpec must bind every tensor source exactly once; missing={missing_tensor}"
        )

    resolved_points = _resolve_view_point_rows(schema, spec)
    roles = tuple(binding.role for binding in spec.source_bindings)
    expected_display = tuple(slot.binding_role for slot in contract.display_slots)
    actual_display = tuple(
        role
        for role in roles
        if role in {AxisViewRole.X, AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y}
    )
    if sorted(role.value for role in actual_display) != sorted(
        role.value for role in expected_display
    ):
        raise ValueError("ViewSpec display sources do not satisfy its ViewContract")
    for slot in contract.display_slots:
        binding = next(
            item for item in spec.source_bindings if item.role is slot.binding_role
        )
        if _source_role(schema, binding.source) not in slot.preferred_axis_roles:
            raise ValueError(
                f"source {_source_name(schema, binding.source)} cannot fill "
                f"{slot.binding_role.value}"
            )

    if spec.intent is ViewIntent.HISTOGRAM and AxisViewRole.SAMPLE not in roles:
        raise ValueError("HISTOGRAM ViewSpec requires at least one SAMPLE source")
    if spec.intent is not ViewIntent.HISTOGRAM and AxisViewRole.SAMPLE in roles:
        raise ValueError("SAMPLE sources are valid only for HISTOGRAM")
    if sum(role is AxisViewRole.FACET for role in roles) > 1:
        raise ValueError("a ViewSpec may contain at most one FACET source")

    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    repeat_binding = spec.binding(repeat_source)
    if not _repeat_binding_allowed(spec.intent, repeat_binding):
        raise ValueError("repeat source has an unsupported presentation binding")

    latest_count = 0
    for binding in spec.source_bindings:
        source = binding.source
        role = _source_role(schema, source)
        if role == SCALAR:
            if (
                binding.role is not AxisViewRole.SELECTED
                or not isinstance(binding.selector, FixedIndex)
                or binding.selector.index != 0
            ):
                raise ValueError(
                    "the scalar carrier must select its sole physical item"
                )
            continue
        if isinstance(binding.selector, LatestNonempty):
            latest_count += 1
            if source != repeat_source:
                raise ValueError("LatestNonempty is valid only for repeat")
        if isinstance(binding.selector, FixedIndex):
            if binding.selector.index >= _source_cardinality(schema, source):
                raise IndexError(f"selector index is outside source {source}")
        if binding.role is AxisViewRole.REDUCED:
            assert binding.reduction is not None
            if source.kind == AxisSourceRef.TENSOR and role not in contract.reducible_axis_roles:
                raise ValueError(f"source role {role} cannot be display-reduced")
            if binding.reduction.method not in {
                DisplayReductionMethod.MEAN,
                DisplayReductionMethod.SUM,
            }:
                raise ValueError("unsupported display reduction method")
        elif source.kind == AxisSourceRef.TENSOR and binding.role in {
            AxisViewRole.BATCH,
            AxisViewRole.FACET,
            AxisViewRole.SELECTED,
            AxisViewRole.SAMPLE,
        }:
            if role == REPEAT:
                continue
            policy = contract.policy_for(role)
            if policy is None or (
                binding.role is not AxisViewRole.SELECTED
                and binding.role not in policy.automatic_roles
            ):
                raise ValueError(
                    f"{binding.role.value} is not allowed for source role {role}"
                )
    if latest_count > 1:
        raise ValueError("a ViewSpec may contain only one LatestNonempty selector")

    methods = {
        binding.reduction.method
        for binding in spec.source_bindings
        if binding.role is AxisViewRole.REDUCED
    }
    if len(methods) > 1:
        raise ValueError("joint display reductions must use one common method")
    if spec.intent is ViewIntent.METER:
        unresolved = tuple(
            binding.source
            for binding in spec.source_bindings
            if binding.role
            not in {
                AxisViewRole.FACET,
                AxisViewRole.BATCH,
                AxisViewRole.SELECTED,
                AxisViewRole.REDUCED,
            }
        )
        if unresolved:
            raise ValueError(f"METER has unresolved sources: {unresolved}")

    if not resolved_points.group_member_ordinals:
        raise ValueError("point projection resolved no groups")


__all__ = [
    "CURVE_CONTRACT",
    "DocumentViewContract",
    "HISTOGRAM_CONTRACT",
    "IMAGE_CONTRACT",
    "METER_CONTRACT",
    "PULSE_CONTRACT",
    "VIEW_CONTRACTS",
    "contract_for",
    "dataset_contract_for",
    "validate_view_spec",
]
