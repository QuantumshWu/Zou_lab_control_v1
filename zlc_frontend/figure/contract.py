"""Declarative contracts and validation for headless presentation views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Sequence

from zlc_data import (
    COMPONENT,
    AxisId,
    BoundFit,
    CommittedTransform,
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
    PointLayout,
    Selection,
    TransformedSchema,
    ValidityContract,
    ValueSchema,
    resolve_selection_indices,
    resolve_transformed_schema,
)
from zlc_storage import canonical_text

from .model import (
    DATASET_VIEW_INTENTS,
    AxisRolePolicy,
    AxisViewRole,
    DisplayReductionMethod,
    DisplaySlot,
    FixedIndex,
    LatestNonempty,
    RepeatViewMode,
    ViewContract,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
)


class _CoordinateSelectionIndices(Sequence[int]):
    """Lazy non-contiguous display indices; materialization belongs to evaluation."""

    __slots__ = ("_coordinates", "_lower", "_upper", "_count")

    def __init__(self, coordinates, lower: float, upper: float, count: int) -> None:
        self._coordinates = coordinates
        self._lower = lower
        self._upper = upper
        self._count = count

    def __len__(self) -> int:
        return self._count

    def __iter__(self):
        for index, value in enumerate(self._coordinates):
            if self._lower <= value <= self._upper:
                yield index

    def __reversed__(self):
        for index in range(len(self._coordinates) - 1, -1, -1):
            value = self._coordinates[index]
            if self._lower <= value <= self._upper:
                yield index

    def __contains__(self, item: object) -> bool:
        if isinstance(item, bool) or not isinstance(item, Integral):
            return False
        index = int(item)
        if index < 0 or index >= len(self._coordinates):
            return False
        value = self._coordinates[index]
        return bool(self._lower <= value <= self._upper)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return tuple(self)[item]
        index = int(item)
        if index < 0:
            index += self._count
        if index < 0 or index >= self._count:
            raise IndexError(index)
        for position, resolved in enumerate(self):
            if position == index:
                return resolved
        raise IndexError(index)


def _coordinate_display_indices(
    axis: AxisSpec,
    term: CoordinateRangeSelection,
) -> Sequence[int]:
    coordinates = axis.coordinates
    if coordinates is None:
        raise ValueError(f"axis {axis.axis_id} has no coordinates for coordinate selection")
    if axis.coordinate_frame != term.coordinate_frame:
        raise ValueError(f"coordinate frame mismatch for axis {axis.axis_id}")
    first = -1
    last = -1
    count = 0
    contiguous = True
    for index, value in enumerate(coordinates):
        if value is None or isinstance(value, (bool, str)) or not isinstance(value, Real):
            raise TypeError(f"axis {axis.axis_id} coordinates are not entirely numeric")
        if term.lower <= value <= term.upper:
            if first < 0:
                first = index
            elif index != last + 1:
                contiguous = False
            last = index
            count += 1
    if count == 0:
        raise ValueError(f"coordinate selection is empty on axis {axis.axis_id}")
    if contiguous:
        return range(first, last + 1)
    return _CoordinateSelectionIndices(coordinates, term.lower, term.upper, count)


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
                SPATIAL_X,
                SPATIAL_Y,
            ),
        ),
    ),
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
        RepeatViewMode.FACET,
    ),
    RepeatViewMode.SAMPLE,
    (REPEAT,),
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
)


@dataclass(frozen=True, slots=True)
class DocumentViewContract:
    """A view fed by an authored document rather than a DatasetSchema.

    The separate type is the firewall: document views have no repeat policy,
    axis-role suggestion, batch capacity, or facet capacity to fill with inert
    values. ``source_schema`` is an opaque owner-qualified identity; frontend
    neither imports that owner nor decodes its document.
    """

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


VIEW_CONTRACTS = {
    ViewIntent.IMAGE: IMAGE_CONTRACT,
    ViewIntent.CURVE: CURVE_CONTRACT,
    ViewIntent.HISTOGRAM: HISTOGRAM_CONTRACT,
    ViewIntent.METER: METER_CONTRACT,
    ViewIntent.PULSE: PULSE_CONTRACT,
}


def contract_for(intent: ViewIntent) -> ViewContract | DocumentViewContract:
    if not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent")
    return VIEW_CONTRACTS[intent]


def dataset_contract_for(intent: ViewIntent) -> ViewContract:
    """Return a DatasetSchema contract or reject a document-fed intent."""

    contract = contract_for(intent)
    if not isinstance(contract, ViewContract):
        raise ValueError(
            f"{intent.value} is document-fed from {contract.source_schema}; "
            "it has no DataBlock ViewSpec/evaluator path"
        )
    return contract


def dataset_axes(schema: DatasetSchema):
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    return (schema.repeat_axis, *schema.point_axes, *schema.cell_schema.data_axes)


def _selection_transform_projection(
    source_schema: DatasetSchema,
    transform: CommittedTransform,
    fit_axis_ids: tuple[AxisId, ...],
) -> tuple[TransformedSchema, DatasetSchema, Selection]:
    """Validate and project the one displayable committed Fit ROI.

    This is a narrow authority-to-presentation contract, not a second transform
    engine.  It accepts exactly one range-preserving Selection over scan or
    spectral point fit axes and/or spatial data fit axes.  FigureEvaluator can
    then read the raw snapshot through the identical selection without
    inventing a derived ref.
    """

    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(transform, CommittedTransform):
        raise TypeError("transform must be CommittedTransform")
    fit_axis_ids = tuple(fit_axis_ids)
    if any(not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids):
        raise TypeError("fit_axis_ids must contain AxisId values")
    operations = transform.spec.operations
    if len(operations) != 1 or not isinstance(operations[0], Selection):
        raise ValueError(
            "transformed fit display requires exactly one range Selection"
        )
    authority_selection = operations[0]
    if any(
        not isinstance(term, (IndexRangeSelection, CoordinateRangeSelection))
        for term in authority_selection.terms
    ):
        raise ValueError(
            "transformed fit display supports only range-preserving selections"
        )
    source_points = {
        axis.axis_id: axis for axis in source_schema.point_axes
    }
    source_data = {
        axis.axis_id: axis for axis in source_schema.cell_schema.data_axes
    }
    selected_ids = {term.axis_id for term in authority_selection.terms}
    if not selected_ids <= set(fit_axis_ids):
        raise ValueError(
            "every selected axis must remain an explicit fit axis"
        )
    for axis_id in selected_ids:
        point_axis = source_points.get(axis_id)
        data_axis = source_data.get(axis_id)
        if point_axis is not None and point_axis.role in (SCAN_POINT, SPECTRAL):
            continue
        if data_axis is not None and data_axis.role in (SPATIAL_X, SPATIAL_Y):
            continue
        raise ValueError(
            "transformed fit display range selections support only scan/spectral "
            "point fit axes and spatial data fit axes"
        )

    resolved = resolve_transformed_schema(source_schema, transform)
    source_cell_ids = tuple(
        axis.axis_id
        for axis in (source_schema.repeat_axis, *source_schema.point_axes)
    )
    if tuple(axis.axis_id for axis in resolved.cell_axes) != source_cell_ids:
        raise ValueError(
            "transformed fit display range selection changed cell-axis identity"
        )
    repeat_axis = resolved.cell_axes[0]
    point_axes = resolved.cell_axes[1:]
    if repeat_axis.role != REPEAT:
        raise ValueError("transformed fit display lost the logical repeat axis")
    if resolved.cell_layout.storage_size == 0:
        raise ValueError("transformed fit display selection contains no physical point")
    if resolved.cell_layout.storage_size % repeat_axis.size:
        raise ValueError("transformed fit display cell layout is not repeat-factorable")
    point_storage_size = resolved.cell_layout.storage_size // repeat_axis.size
    point_mapping = []
    for storage_index in range(point_storage_size):
        multi = resolved.cell_layout.multi_index(storage_index)
        if multi[0] != 0:
            raise ValueError(
                "transformed fit display cell layout is not repeat-major"
            )
        point_mapping.append(multi[1:])
    point_layout = PointLayout.from_mapping(
        tuple(axis.size for axis in point_axes),
        tuple(point_mapping),
    )
    validity_contract = (
        ValidityContract.components(*resolved.validity_axis_ids)
        if resolved.validity_axis_ids
        else ValidityContract.value()
    )
    effective_schema = DatasetSchema(
        repeat_axis,
        point_axes,
        point_layout,
        ValueSchema(
            resolved.data_axes,
            validity_contract,
            resolved.dtype,
            resolved.value_unit,
        ),
    )
    if effective_schema.cell_layout != resolved.cell_layout:
        raise ValueError(
            "transformed fit display cannot faithfully represent the resolved cell layout"
        )
    return resolved, effective_schema, authority_selection


def selection_fit_view_projection(
    bound: BoundFit,
) -> tuple[DatasetSchema, Selection]:
    """Return the exact raw-snapshot projection for one displayable bound Fit."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    resolved, effective_schema, authority_selection = (
        _selection_transform_projection(
            bound.expected_schema,
            bound.spec.committed_transform,
            bound.spec.fit_axis_ids,
        )
    )
    if bound.effective_schema != resolved:
        raise ValueError("bound Fit effective schema differs from its transform")
    return effective_schema, authority_selection


def _selection_fit_projection(
    source_schema: DatasetSchema,
    result: FitResultBatch,
) -> tuple[DatasetSchema, Selection]:
    """Return the exact effective schema for one displayable Fit result."""

    transform = result.spec.committed_transform
    if transform is None:
        raise ValueError("fit result has no committed transform")
    resolved, effective_schema, authority_selection = (
        _selection_transform_projection(
            source_schema,
            transform,
            result.spec.fit_axis_ids,
        )
    )
    if result.effective_schema_fingerprint != resolved.fingerprint:
        raise ValueError("fit result effective schema differs from its transform")
    if result.fit_axis_specs != tuple(
        resolved.axis(axis_id) for axis_id in result.spec.fit_axis_ids
    ) or result.batch_axis_specs != tuple(
        resolved.axis(axis_id) for axis_id in result.spec.batch_axis_ids
    ):
        raise ValueError("fit result axes differ from its transformed schema")
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
    if isinstance(term, CoordinateRangeSelection):
        return _coordinate_display_indices(axis, term)
    indices, _drop = resolve_selection_indices(axis, term)
    return indices


def fit_single_panel_presentation(
    schema: DatasetSchema,
    view: ViewSpec,
    preferences: ViewPreferences | None = None,
) -> tuple[Selection | None, ViewPreferences]:
    """Freeze one labelled display cell for direct Figure-owned Fit authoring.

    The returned ``Selection`` is presentation state only.  It collapses every
    visible FACET/BATCH axis to one explicit logical index and replaces a
    multi-element repeat display reduction with the same explicit cell.  It
    never invents an X/IMAGE-axis selection (an existing display range is
    preserved) and therefore must never be copied into a
    :class:`zlc_data.FitSpec`.

    Sparse point layouts are resolved as one physical tuple.  Choosing each
    point axis independently (or assuming logical index zero exists) would
    create a display cell that the source never published.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(view, ViewSpec):
        raise TypeError("view must be ViewSpec")
    if view.schema_fingerprint != schema.fingerprint:
        raise ValueError("ViewSpec schema fingerprint is stale")
    preferences = ViewPreferences() if preferences is None else preferences
    if not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")
    validate_view_spec(schema, view, dataset_contract_for(view.intent))

    axes = dataset_axes(schema)
    axis_by_id = {axis.axis_id: axis for axis in axes}
    terms = tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
    )
    existing_by_axis = {term.axis_id: term for term in terms}
    if len(existing_by_axis) != len(terms):
        raise ValueError("display selections constrain one axis more than once")
    allowed = {
        axis.axis_id: display_axis_indices(axis, view.display_selections)
        for axis in axes
    }
    if any(len(indices) == 0 for indices in allowed.values()):
        raise ValueError("display selection contains an empty axis")

    collapse_ids = {
        binding.axis_id
        for binding in view.axis_bindings
        if binding.role in (AxisViewRole.BATCH, AxisViewRole.FACET)
        or (
            binding.role is AxisViewRole.REDUCED
            and axis_by_id[binding.axis_id].size > 1
        )
    }
    reduced_ids = {
        binding.axis_id
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.REDUCED
        and axis_by_id[binding.axis_id].size > 1
    }
    nonrepeat_reduced = reduced_ids - {schema.repeat_axis.axis_id}
    if nonrepeat_reduced:
        raise ValueError(
            "direct Fit single-panel presentation cannot rewrite a non-repeat "
            "display reduction"
        )

    fixed_indices = {
        binding.axis_id: binding.selector.index
        for binding in view.axis_bindings
        if isinstance(binding.selector, FixedIndex)
    }
    point_tuple = _first_visible_point_tuple(schema, allowed, fixed_indices)
    if point_tuple is None:
        raise ValueError(
            "display selections and selectors contain no physical point tuple"
        )
    point_index = {
        axis.axis_id: index
        for axis, index in zip(schema.point_axes, point_tuple)
    }

    merged_terms = {
        axis_id: term
        for axis_id, term in existing_by_axis.items()
        if axis_id not in collapse_ids
    }
    for axis_id in sorted(collapse_ids, key=lambda value: value.value):
        index = point_index.get(axis_id, allowed[axis_id][0])
        if index not in allowed[axis_id]:
            raise ValueError(
                f"single-panel index {index} conflicts with display selection on {axis_id}"
            )
        merged_terms[axis_id] = Selection.index(axis_id, index).terms[0]

    selection = (
        None
        if not merged_terms
        else Selection(tuple(merged_terms.values()))
    )
    adjusted_preferences = replace(
        preferences,
        repeat_mode=(
            RepeatViewMode.LATEST
            if schema.repeat_axis.axis_id in reduced_ids
            else preferences.repeat_mode
        ),
        batch_axis_ids=tuple(
            axis_id
            for axis_id in preferences.batch_axis_ids
            if axis_id not in collapse_ids
        ),
        facet_axis_ids=tuple(
            axis_id
            for axis_id in preferences.facet_axis_ids
            if axis_id not in collapse_ids
        ),
        sample_axis_ids=tuple(
            axis_id
            for axis_id in preferences.sample_axis_ids
            if axis_id not in collapse_ids
        ),
    )
    return selection, adjusted_preferences


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
            indices
            if isinstance(indices, (range, _CoordinateSelectionIndices))
            else frozenset(indices)
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
    contract = dataset_contract_for(spec.intent) if contract is None else contract
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
    reduction_methods = {
        binding.reduction.method
        for binding in spec.axis_bindings
        if binding.role is AxisViewRole.REDUCED
    }
    if len(reduction_methods) > 1:
        raise ValueError("joint display reductions must use one common method")
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
    validate_view_spec(schema, view, dataset_contract_for(view.intent))


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
    "dataset_axes",
    "display_axis_indices",
    "fit_single_panel_presentation",
    "validate_view_spec",
]
