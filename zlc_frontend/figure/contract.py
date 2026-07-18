"""Declarative contracts and validation for headless presentation views."""

from __future__ import annotations

from typing import Sequence

from zlc_data import (
    COMPONENT,
    CoordinateRangeSelection,
    FitResultBatch,
    IndexRangeSelection,
    MONITOR_HISTORY,
    READOUT_EVENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    SPECTRAL,
    AxisSpec,
    DatasetSchema,
    Selection,
    ValidityContract,
    ValueSchema,
    resolve_selection_indices,
    resolve_transformed_schema,
)

from .model import (
    AxisRolePolicy,
    AxisViewRole,
    DisplayReductionMethod,
    DisplaySlot,
    FixedIndex,
    LatestNonempty,
    RepeatViewMode,
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
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(SITE, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SPATIAL_X, ()),
        AxisRolePolicy(SPATIAL_Y, ()),
    ),
    (
        RepeatViewMode.MEAN,
        RepeatViewMode.LATEST,
        RepeatViewMode.SUM,
        RepeatViewMode.FACET,
    ),
    RepeatViewMode.MEAN,
    (REPEAT,),
    maximum_batch_series=1,
    maximum_facet_cells=36,
)


CURVE_CONTRACT = ViewContract(
    ViewIntent.CURVE,
    (DisplaySlot(AxisViewRole.X, (SPECTRAL, SCAN_POINT, MONITOR_HISTORY)),),
    (
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SITE, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.BATCH, AxisViewRole.FACET)),
        # A spatial curve requires an explicit pixel/ROI/page selection.  It
        # is never made scalar by an automatic mean or automatic gallery.
        AxisRolePolicy(SPATIAL_X, ()),
        AxisRolePolicy(SPATIAL_Y, ()),
    ),
    (
        RepeatViewMode.MEAN,
        RepeatViewMode.LATEST,
        RepeatViewMode.SUM,
        RepeatViewMode.BATCH,
        RepeatViewMode.FACET,
    ),
    RepeatViewMode.MEAN,
    (REPEAT, SPATIAL_X, SPATIAL_Y),
    maximum_batch_series=32,
    maximum_facet_cells=36,
)


HISTOGRAM_CONTRACT = ViewContract(
    ViewIntent.HISTOGRAM,
    (),
    (
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.SAMPLE,)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SAMPLE,)),
        AxisRolePolicy(SITE, (AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.FACET, AxisViewRole.BATCH)),
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SPATIAL_X, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
        AxisRolePolicy(SPATIAL_Y, (AxisViewRole.FACET, AxisViewRole.SLIDER)),
    ),
    (
        RepeatViewMode.SAMPLE,
        RepeatViewMode.BATCH,
        RepeatViewMode.MEAN,
        RepeatViewMode.SUM,
        RepeatViewMode.LATEST,
    ),
    RepeatViewMode.SAMPLE,
    (REPEAT,),
    maximum_batch_series=32,
    maximum_facet_cells=36,
)


METER_CONTRACT = ViewContract(
    ViewIntent.METER,
    (),
    (
        AxisRolePolicy(SCAN_POINT, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(SPECTRAL, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(READOUT_EVENT, (AxisViewRole.FACET,)),
        AxisRolePolicy(MONITOR_HISTORY, (AxisViewRole.SLIDER, AxisViewRole.FACET)),
        AxisRolePolicy(SITE, (AxisViewRole.FACET,)),
        AxisRolePolicy(COMPONENT, (AxisViewRole.FACET,)),
        AxisRolePolicy(SPATIAL_X, ()),
        AxisRolePolicy(SPATIAL_Y, ()),
    ),
    (RepeatViewMode.LATEST, RepeatViewMode.MEAN, RepeatViewMode.SUM),
    RepeatViewMode.LATEST,
    (REPEAT, SPATIAL_X, SPATIAL_Y),
    maximum_batch_series=1,
    maximum_facet_cells=36,
)


VIEW_CONTRACTS = {
    ViewIntent.IMAGE: IMAGE_CONTRACT,
    ViewIntent.CURVE: CURVE_CONTRACT,
    ViewIntent.HISTOGRAM: HISTOGRAM_CONTRACT,
    ViewIntent.METER: METER_CONTRACT,
}


def contract_for(intent: ViewIntent) -> ViewContract:
    if not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent")
    return VIEW_CONTRACTS[intent]


def dataset_axes(schema: DatasetSchema):
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    return (schema.repeat_axis, *schema.point_axes, *schema.cell_schema.data_axes)


def _selection_fit_projection(
    source_schema: DatasetSchema,
    result: FitResultBatch,
) -> tuple[DatasetSchema, Selection]:
    """Return the exact effective schema for one displayable Fit ROI.

    This is a narrow authority-to-presentation contract, not a second transform
    engine.  It accepts exactly one range-preserving Selection over spatial
    data axes that remain fitted axes.  FigureEvaluator can then read the raw
    snapshot through the identical selection without inventing a derived ref.
    """

    transform = result.spec.committed_transform
    if transform is None:
        raise ValueError("fit result has no committed transform")
    operations = transform.spec.operations
    if len(operations) != 1 or not isinstance(operations[0], Selection):
        raise ValueError(
            "transformed fit display requires exactly one spatial Selection"
        )
    authority_selection = operations[0]
    if any(
        not isinstance(term, (IndexRangeSelection, CoordinateRangeSelection))
        for term in authority_selection.terms
    ):
        raise ValueError(
            "transformed fit display supports only range-preserving selections"
        )
    source_data = {
        axis.axis_id: axis for axis in source_schema.cell_schema.data_axes
    }
    selected_ids = {term.axis_id for term in authority_selection.terms}
    if any(
        axis_id not in source_data
        or source_data[axis_id].role not in (SPATIAL_X, SPATIAL_Y)
        for axis_id in selected_ids
    ):
        raise ValueError(
            "transformed fit display selections must name spatial data axes"
        )
    if not selected_ids <= set(result.spec.fit_axis_ids):
        raise ValueError(
            "every selected spatial axis must remain an explicit fit axis"
        )

    resolved = resolve_transformed_schema(source_schema, transform)
    source_cell_axes = (source_schema.repeat_axis, *source_schema.point_axes)
    if (
        resolved.cell_axes != source_cell_axes
        or resolved.cell_layout != source_schema.cell_layout
    ):
        raise ValueError(
            "transformed fit display cannot select or reduce repeat/point axes"
        )
    if result.effective_schema_fingerprint != resolved.fingerprint:
        raise ValueError("fit result effective schema differs from its transform")
    if result.fit_axis_specs != tuple(
        resolved.axis(axis_id) for axis_id in result.spec.fit_axis_ids
    ) or result.batch_axis_specs != tuple(
        resolved.axis(axis_id) for axis_id in result.spec.batch_axis_ids
    ):
        raise ValueError("fit result axes differ from its transformed schema")
    validity_contract = (
        ValidityContract.components(*resolved.validity_axis_ids)
        if resolved.validity_axis_ids
        else ValidityContract.value()
    )
    effective_schema = DatasetSchema(
        source_schema.repeat_axis,
        source_schema.point_axes,
        source_schema.point_layout,
        ValueSchema(
            resolved.data_axes,
            validity_contract,
            resolved.dtype,
            resolved.value_unit,
        ),
    )
    return effective_schema, authority_selection


def display_axis_indices(
    axis: AxisSpec,
    selections: Sequence[Selection],
) -> Sequence[int]:
    """Resolve the one optional display selection for an axis."""

    if not isinstance(axis, AxisSpec):
        raise TypeError("axis must be zlc_data.AxisSpec")
    selections = tuple(selections)
    if any(not isinstance(selection, Selection) for selection in selections):
        raise TypeError("selections must contain zlc_data.Selection values")
    terms = tuple(
        term
        for selection in selections
        for term in selection.terms
        if term.axis_id == axis.axis_id
    )
    if len(terms) > 1:
        raise ValueError(f"axis {axis.axis_id} has multiple display selection terms")
    term = terms[0] if terms else None
    if term is None:
        return range(axis.size)
    indices, _drop = resolve_selection_indices(axis, term)
    return indices


def _first_visible_point_tuple(
    schema: DatasetSchema,
    allowed_indices,
    fixed_indices=None,
) -> tuple[int, ...] | None:
    """Return one physically present point tuple inside the visible selections."""

    fixed_indices = {} if fixed_indices is None else fixed_indices
    point_axes = schema.point_axes
    if not point_axes:
        return ()
    allowed_membership = {}
    for axis in point_axes:
        indices = allowed_indices[axis.axis_id]
        allowed_membership[axis.axis_id] = (
            indices if isinstance(indices, range) else frozenset(indices)
        )
    layout = schema.point_layout
    if layout.storage_to_multi is None:
        candidate = tuple(
            fixed_indices.get(axis.axis_id, allowed_indices[axis.axis_id][0])
            for axis in point_axes
        )
        if all(
            index in allowed_membership[axis.axis_id]
            for axis, index in zip(point_axes, candidate)
        ):
            return candidate
        return None
    for candidate in layout.storage_to_multi:
        if all(
            index in allowed_membership[axis.axis_id]
            and fixed_indices.get(axis.axis_id, index) == index
            for axis, index in zip(point_axes, candidate)
        ):
            return candidate
    return None


def _repeat_mode_for_binding(binding) -> RepeatViewMode:
    if binding.role is AxisViewRole.REDUCED:
        assert binding.reduction is not None
        return (
            RepeatViewMode.MEAN
            if binding.reduction.method is DisplayReductionMethod.MEAN
            else RepeatViewMode.SUM
        )
    if binding.role is AxisViewRole.BATCH:
        return RepeatViewMode.BATCH
    if binding.role is AxisViewRole.FACET:
        return RepeatViewMode.FACET
    if binding.role is AxisViewRole.SAMPLE:
        return RepeatViewMode.SAMPLE
    if binding.role is AxisViewRole.SELECTED:
        return RepeatViewMode.LATEST
    raise ValueError("repeat axis has an unsupported presentation binding")


def _display_dtype_issue(schema: DatasetSchema, intent: ViewIntent) -> str | None:
    """Return the one value-domain issue the current ViewSpec cannot express."""

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
    """Validate total axis coverage and intent-specific presentation safety."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(spec, ViewSpec):
        raise TypeError("spec must be ViewSpec")
    contract = contract_for(spec.intent) if contract is None else contract
    if not isinstance(contract, ViewContract) or contract.intent is not spec.intent:
        raise ValueError("view contract does not match ViewSpec intent")
    if spec.schema_fingerprint != schema.fingerprint:
        raise ValueError("ViewSpec schema fingerprint is stale")
    dtype_issue = _display_dtype_issue(schema, spec.intent)
    if dtype_issue is not None:
        raise ValueError(dtype_issue)
    axes = dataset_axes(schema)
    expected_ids = tuple(axis.axis_id for axis in axes)
    actual_ids = tuple(binding.axis_id for binding in spec.axis_bindings)
    if set(actual_ids) != set(expected_ids):
        missing = tuple(axis_id for axis_id in expected_ids if axis_id not in actual_ids)
        extra = tuple(axis_id for axis_id in actual_ids if axis_id not in expected_ids)
        raise ValueError(
            "ViewSpec must bind every dataset AxisId exactly once; "
            f"missing={missing}, extra={extra}"
        )
    axis_by_id = {axis.axis_id: axis for axis in axes}
    selected_axis_ids = {
        term.axis_id
        for selection in spec.display_selections
        for term in selection.terms
    }
    unknown_selection_axes = selected_axis_ids - set(axis_by_id)
    if unknown_selection_axes:
        raise ValueError(
            f"display selection references absent axes: {tuple(sorted(unknown_selection_axes))}"
        )
    allowed_indices = {
        axis.axis_id: display_axis_indices(axis, spec.display_selections)
        for axis in axes
    }
    fixed_indices = {
        binding.axis_id: binding.selector.index
        for binding in spec.axis_bindings
        if isinstance(binding.selector, FixedIndex)
    }
    for axis_id, index in fixed_indices.items():
        if index not in allowed_indices[axis_id]:
            raise IndexError(
                f"selector index is outside the display selection on axis {axis_id}"
            )
    if _first_visible_point_tuple(schema, allowed_indices, fixed_indices) is None:
        raise ValueError(
            "point selections and fixed selectors do not identify a physical point"
        )
    effective_cardinality = {
        axis_id: len(indices) for axis_id, indices in allowed_indices.items()
    }
    roles = tuple(binding.role for binding in spec.axis_bindings)
    expected_display = tuple(slot.binding_role for slot in contract.display_slots)
    actual_display = tuple(role for role in roles if role in {
        AxisViewRole.X, AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y
    })
    if sorted(role.value for role in actual_display) != sorted(role.value for role in expected_display):
        raise ValueError("ViewSpec display axes do not satisfy its ViewContract")
    for slot in contract.display_slots:
        binding = next(item for item in spec.axis_bindings if item.role is slot.binding_role)
        axis = axis_by_id[binding.axis_id]
        if axis.role not in slot.preferred_axis_roles:
            raise ValueError(
                f"axis {axis.axis_id} role {axis.role} is not allowed for {slot.binding_role.value}"
            )
    if spec.intent is ViewIntent.HISTOGRAM and AxisViewRole.SAMPLE not in roles:
        raise ValueError("HISTOGRAM ViewSpec requires at least one SAMPLE axis")
    if spec.intent is not ViewIntent.HISTOGRAM and AxisViewRole.SAMPLE in roles:
        raise ValueError("SAMPLE axes are valid only for HISTOGRAM")

    batch_size = 1
    facet_size = 1
    for binding in spec.axis_bindings:
        axis = axis_by_id[binding.axis_id]
        policy = contract.policy_for(axis.role)
        if axis.role == REPEAT:
            repeat_mode = _repeat_mode_for_binding(binding)
            if repeat_mode not in contract.repeat_modes:
                raise ValueError(
                    f"repeat mode {repeat_mode.value} is not allowed by this ViewContract"
                )
        if isinstance(binding.selector, LatestNonempty):
            if axis.axis_id != schema.repeat_axis.axis_id:
                raise ValueError(
                    "LatestNonempty is valid only for the logical repeat axis"
                )
        if binding.role is AxisViewRole.REDUCED:
            assert binding.reduction is not None
            if axis.role not in contract.reducible_axis_roles:
                raise ValueError(f"axis role {axis.role} cannot be display-reduced by this contract")
            if axis.role != REPEAT and axis.axis_id not in selected_axis_ids:
                raise ValueError(
                    f"axis {axis.axis_id} may be display-reduced only after an explicit selection"
                )
            if binding.reduction.method not in (
                DisplayReductionMethod.MEAN,
                DisplayReductionMethod.SUM,
            ):
                raise ValueError("unsupported display reduction method")
        elif binding.role in (AxisViewRole.BATCH, AxisViewRole.FACET, AxisViewRole.SLIDER, AxisViewRole.SAMPLE):
            if axis.role == REPEAT:
                pass
            elif policy is None or binding.role not in policy.automatic_roles:
                explicit_spatial_curve_page = (
                    spec.intent is ViewIntent.CURVE
                    and axis.role in (SPATIAL_X, SPATIAL_Y)
                    and binding.role is AxisViewRole.FACET
                    and axis.axis_id in selected_axis_ids
                )
                # Explicit SAMPLE is permitted for histogram axes only; it is
                # review-required at suggestion time but still a valid spec.
                if not (
                    spec.intent is ViewIntent.HISTOGRAM
                    and binding.role is AxisViewRole.SAMPLE
                ) and not explicit_spatial_curve_page:
                    raise ValueError(
                        f"{binding.role.value} is not allowed for axis role {axis.role}"
                    )
        if binding.role is AxisViewRole.BATCH:
            batch_size *= effective_cardinality[axis.axis_id]
        if binding.role is AxisViewRole.FACET:
            facet_size *= effective_cardinality[axis.axis_id]
    reduction_methods = {
        binding.reduction.method
        for binding in spec.axis_bindings
        if binding.role is AxisViewRole.REDUCED
    }
    if len(reduction_methods) > 1:
        raise ValueError("joint display reductions must use one common method")
    if batch_size > contract.maximum_batch_series:
        raise ValueError(
            f"batch product {batch_size} exceeds contract limit {contract.maximum_batch_series}"
        )
    if facet_size > contract.maximum_facet_cells:
        raise ValueError(
            f"facet product {facet_size} exceeds contract limit {contract.maximum_facet_cells}"
        )
    if spec.intent is ViewIntent.METER:
        unresolved = tuple(
            binding.axis_id
            for binding in spec.axis_bindings
            if binding.role not in (
                AxisViewRole.FACET,
                AxisViewRole.BATCH,
                AxisViewRole.SELECTED,
                AxisViewRole.SLIDER,
                AxisViewRole.REDUCED,
            )
        )
        if unresolved:
            raise ValueError(f"METER has unresolved axes: {unresolved}")


def _validate_selection_fit_view(
    schema: DatasetSchema,
    result: FitResultBatch,
    view: ViewSpec,
) -> None:
    """Prove that a raw view reproduces one selection-only Fit authority.

    Figure evaluation still reads the immutable raw snapshot.  Therefore the
    committed selection must appear byte-for-byte in the display selection,
    and the only additional selection terms may identify saved batch cells.
    Any display reduction would change the observations and is rejected.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if not isinstance(view, ViewSpec):
        raise TypeError("view must be ViewSpec")
    if result.spec.committed_transform is None:
        return

    _effective_schema, authority_selection = _selection_fit_projection(
        schema,
        result,
    )
    terms = tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
    )
    actual_terms = {term.axis_id: term for term in terms}
    if len(actual_terms) != len(terms):
        raise ValueError("figure selection repeats an axis")
    authority_terms = {
        term.axis_id: term for term in authority_selection.terms
    }
    if any(
        actual_terms.get(axis_id) != term
        for axis_id, term in authority_terms.items()
    ):
        raise ValueError("figure ROI differs from the fit committed transform")
    batch_ids = {axis.axis_id for axis in result.batch_axis_specs}
    if set(actual_terms) - set(authority_terms) - batch_ids:
        raise ValueError(
            "figure selection outside fit batch axes differs from the committed transform"
        )
    if any(
        binding.role is AxisViewRole.REDUCED
        for binding in view.axis_bindings
    ):
        raise ValueError("selection-only transformed fit display cannot reduce axes")
    validate_view_spec(schema, view, contract_for(view.intent))


__all__ = [
    "CURVE_CONTRACT",
    "HISTOGRAM_CONTRACT",
    "IMAGE_CONTRACT",
    "METER_CONTRACT",
    "VIEW_CONTRACTS",
    "contract_for",
    "dataset_axes",
    "display_axis_indices",
    "validate_view_spec",
]
