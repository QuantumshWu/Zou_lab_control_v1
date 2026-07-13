"""Deterministic, metadata-only ViewSpec suggestions."""

from __future__ import annotations

from zlc_data import (
    CoordinateRangeSelection,
    DatasetSchema,
    IndexRangeSelection,
    REPEAT,
    Selection,
)

from .contract import (
    _display_axis_cardinality,
    contract_for,
    dataset_axes,
    display_axis_indices,
    validate_view_spec,
)
from .model import (
    AxisViewBinding,
    AxisViewRole,
    DecisionReason,
    DisplayReduction,
    DisplayReductionMethod,
    FixedIndex,
    LatestNonempty,
    RepeatViewMode,
    SuggestionStatus,
    ViewAlternative,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    ViewSuggestion,
)


def _needs(
    code: str,
    message: str,
    *,
    axis_id=None,
    alternatives=(),
) -> ViewSuggestion:
    return ViewSuggestion(
        None,
        SuggestionStatus.NEEDS_INPUT,
        (DecisionReason(code, message, axis_id),),
        tuple(alternatives),
    )


def _selection_indices(schema: DatasetSchema, selection: Selection | None) -> dict:
    """Resolve only selections that name one unambiguous axis index.

    A range with multiple members remains a range and therefore cannot silently
    become a scalar display selection.  It remains an explicit display range.
    """

    if selection is None:
        return {}
    if not isinstance(selection, Selection):
        raise TypeError("selection must be zlc_data.Selection or None")
    axes = {axis.axis_id: axis for axis in dataset_axes(schema)}
    resolved = {}
    for term in selection.terms:
        try:
            axis = axes[term.axis_id]
        except KeyError as exc:
            raise ValueError(f"selection references absent axis {term.axis_id}") from exc
        indices = display_axis_indices(axis, (selection,))
        if len(indices) == 1:
            resolved[axis.axis_id] = indices[0]
    return resolved


def _preferred_display_axis(preferences: ViewPreferences, role: AxisViewRole):
    return {
        AxisViewRole.X: preferences.x_axis_id,
        AxisViewRole.IMAGE_X: preferences.image_x_axis_id,
        AxisViewRole.IMAGE_Y: preferences.image_y_axis_id,
    }[role]


def _repeat_binding(axis_id, mode: RepeatViewMode) -> AxisViewBinding:
    if mode is RepeatViewMode.MEAN:
        return AxisViewBinding(
            axis_id,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(DisplayReductionMethod.MEAN),
        )
    if mode is RepeatViewMode.SUM:
        return AxisViewBinding(
            axis_id,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(DisplayReductionMethod.SUM),
        )
    if mode is RepeatViewMode.LATEST:
        return AxisViewBinding(axis_id, AxisViewRole.SELECTED, selector=LatestNonempty())
    if mode is RepeatViewMode.BATCH:
        return AxisViewBinding(axis_id, AxisViewRole.BATCH)
    return AxisViewBinding(axis_id, AxisViewRole.SAMPLE)


def suggest_view(
    schema: DatasetSchema,
    intent: ViewIntent,
    selection: Selection | None = None,
    preferences: ViewPreferences | None = None,
) -> ViewSuggestion:
    """Suggest one safe presentation using schema metadata only.

    The function never receives or reads values.  Its output therefore cannot
    vary with signal magnitude, singleton accidents, NaNs, or array ordering.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent")
    preferences = ViewPreferences() if preferences is None else preferences
    if not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")
    contract = contract_for(intent)
    axes = dataset_axes(schema)
    axis_by_id = {axis.axis_id: axis for axis in axes}
    if selection is not None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        unknown_selection_axes = tuple(
            term.axis_id for term in selection.terms if term.axis_id not in axis_by_id
        )
        if unknown_selection_axes:
            return _needs(
                "STALE_SELECTION_AXIS",
                f"selection references absent axes: {unknown_selection_axes}",
            )
        for term in selection.terms:
            axis = axis_by_id[term.axis_id]
            try:
                display_axis_indices(axis, (selection,))
            except IndexError as exc:
                return _needs(
                    "SELECTION_OUT_OF_RANGE",
                    str(exc),
                    axis_id=term.axis_id,
                )
            except (TypeError, ValueError) as exc:
                return _needs(
                    "SELECTION_INVALID",
                    str(exc),
                    axis_id=term.axis_id,
                )
    range_selected_axes = {
        term.axis_id
        for term in (() if selection is None else selection.terms)
        if isinstance(term, (IndexRangeSelection, CoordinateRangeSelection))
    }
    display_selections = () if selection is None else (selection,)
    effective_cardinality = {
        axis.axis_id: _display_axis_cardinality(axis, display_selections)
        for axis in axes
    }
    unbound = set(axis_by_id)
    bindings: dict = {}
    reasons: list[DecisionReason] = []
    alternatives: list[ViewAlternative] = []
    review_required = False

    for slot in contract.display_slots:
        preferred = _preferred_display_axis(preferences, slot.binding_role)
        if preferred is not None:
            axis = axis_by_id.get(preferred)
            if axis is None or preferred not in unbound:
                return _needs(
                    "INVALID_DISPLAY_AXIS",
                    f"preferred {slot.binding_role.value} axis is absent or already bound",
                    axis_id=preferred,
                )
            if axis.role not in slot.preferred_axis_roles:
                return _needs(
                    "DISPLAY_ROLE_MISMATCH",
                    f"axis {axis.name} cannot fill {slot.binding_role.value}",
                    axis_id=preferred,
                )
            chosen = axis
        else:
            chosen = None
            for role in slot.preferred_axis_roles:
                candidates = tuple(
                    axis for axis in axes if axis.axis_id in unbound and axis.role == role
                )
                if len(candidates) == 1:
                    chosen = candidates[0]
                    break
                if len(candidates) > 1:
                    alternatives = [
                        ViewAlternative(axis.axis_id, slot.binding_role, axis.name)
                        for axis in sorted(candidates, key=lambda item: item.axis_id.value)
                    ]
                    return _needs(
                        "AMBIGUOUS_DISPLAY_AXIS",
                        f"multiple {role.value} axes can fill {slot.binding_role.value}",
                        alternatives=alternatives,
                    )
            if chosen is None:
                return _needs(
                    "MISSING_DISPLAY_AXIS",
                    f"no declared axis can fill {slot.binding_role.value}",
                )
        bindings[chosen.axis_id] = AxisViewBinding(chosen.axis_id, slot.binding_role)
        unbound.remove(chosen.axis_id)
        reasons.append(
            DecisionReason(
                "DISPLAY_AXIS",
                f"{chosen.name} is the {slot.binding_role.value} axis by declared role",
                chosen.axis_id,
            )
        )

    repeat_axis = schema.repeat_axis
    if repeat_axis.axis_id in unbound:
        repeat_mode = preferences.repeat_mode or contract.default_repeat_mode
        if repeat_mode not in contract.repeat_modes:
            return _needs(
                "REPEAT_MODE_NOT_ALLOWED",
                f"{repeat_mode.value} is not allowed for {intent.value}",
                axis_id=repeat_axis.axis_id,
            )
        bindings[repeat_axis.axis_id] = _repeat_binding(repeat_axis.axis_id, repeat_mode)
        unbound.remove(repeat_axis.axis_id)
        reasons.append(
            DecisionReason(
                "REPEAT_POLICY",
                f"repeat uses the {repeat_mode.value} presentation policy",
                repeat_axis.axis_id,
            )
        )

    selected = _selection_indices(schema, selection)
    for axis_id, index in sorted(selected.items(), key=lambda item: item[0].value):
        if axis_id not in unbound:
            reasons.append(
                DecisionReason(
                    "BOUND_AXIS_SELECTION",
                    f"{axis_by_id[axis_id].name} keeps its visible role inside the explicit selection",
                    axis_id,
                )
            )
        else:
            axis = axis_by_id[axis_id]
            bindings[axis_id] = AxisViewBinding(
                axis_id, AxisViewRole.SELECTED, selector=FixedIndex(index)
            )
            unbound.remove(axis_id)
            reasons.append(
                DecisionReason(
                    "EXPLICIT_SELECTION",
                    f"{axis.name} uses an explicit selection",
                    axis_id,
                )
            )

    explicit_roles = (
        (preferences.sample_axis_ids, AxisViewRole.SAMPLE),
        (preferences.batch_axis_ids, AxisViewRole.BATCH),
        (preferences.facet_axis_ids, AxisViewRole.FACET),
    )
    batch_size = 1
    facet_size = 1
    for axis_ids, view_role in explicit_roles:
        for axis_id in axis_ids:
            if axis_id not in unbound:
                return _needs(
                    "EXPLICIT_ROLE_CONFLICT",
                    f"axis cannot also be {view_role.value}",
                    axis_id=axis_id,
                )
            axis = axis_by_id.get(axis_id)
            if axis is None:
                return _needs("UNKNOWN_AXIS", "preferred axis is absent", axis_id=axis_id)
            if view_role is AxisViewRole.SAMPLE and intent is not ViewIntent.HISTOGRAM:
                return _needs("SAMPLE_REQUIRES_HISTOGRAM", "SAMPLE is histogram-only", axis_id=axis_id)
            if view_role is AxisViewRole.BATCH:
                batch_size *= effective_cardinality[axis.axis_id]
                if batch_size > contract.maximum_batch_series:
                    return _needs("BATCH_LIMIT", "explicit batch exceeds contract limit", axis_id=axis_id)
            if view_role is AxisViewRole.FACET:
                facet_size *= effective_cardinality[axis.axis_id]
                if facet_size > contract.maximum_facet_cells:
                    return _needs("FACET_LIMIT", "explicit facet exceeds contract limit", axis_id=axis_id)
            bindings[axis_id] = AxisViewBinding(axis_id, view_role)
            unbound.remove(axis_id)
            if view_role is AxisViewRole.SAMPLE and axis.role not in (REPEAT,):
                review_required = True
            reasons.append(
                DecisionReason("EXPLICIT_AXIS_ROLE", f"{axis.name} uses {view_role.value}", axis_id)
            )

    latest_used = any(
        isinstance(binding.selector, LatestNonempty) for binding in bindings.values()
    )
    for axis in axes:
        if axis.axis_id not in unbound:
            continue
        if axis.axis_id in range_selected_axes and axis.role in contract.reducible_axis_roles:
            bindings[axis.axis_id] = AxisViewBinding(
                axis.axis_id,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            )
            unbound.remove(axis.axis_id)
            review_required = True
            reasons.append(
                DecisionReason(
                    "SELECTED_DISPLAY_REDUCTION",
                    f"{axis.name} is mean-reduced only inside the explicit display selection",
                    axis.axis_id,
                )
            )
            continue
        policy = contract.policy_for(axis.role)
        if policy is None or not policy.automatic_roles:
            return _needs(
                "UNRESOLVED_INFORMATION_AXIS",
                f"{axis.name} requires an explicit selection, facet, batch, or reducer",
                axis_id=axis.axis_id,
            )
        chosen_role = None
        for candidate in policy.automatic_roles:
            if candidate is AxisViewRole.BATCH:
                cardinality = effective_cardinality[axis.axis_id]
                if batch_size * cardinality <= contract.maximum_batch_series:
                    batch_size *= cardinality
                    chosen_role = candidate
                    break
            elif candidate is AxisViewRole.FACET:
                cardinality = effective_cardinality[axis.axis_id]
                if facet_size * cardinality <= contract.maximum_facet_cells:
                    facet_size *= cardinality
                    chosen_role = candidate
                    break
            elif candidate is AxisViewRole.SLIDER:
                if not latest_used:
                    latest_used = True
                    chosen_role = candidate
                    break
            elif candidate is AxisViewRole.SAMPLE:
                chosen_role = candidate
                break
        if chosen_role is None:
            return _needs(
                "VIEW_CAPACITY_EXHAUSTED",
                f"no safe visible role remains for {axis.name}",
                axis_id=axis.axis_id,
            )
        if chosen_role is AxisViewRole.SLIDER:
            binding = AxisViewBinding(
                axis.axis_id, AxisViewRole.SLIDER, selector=LatestNonempty()
            )
        else:
            binding = AxisViewBinding(axis.axis_id, chosen_role)
        bindings[axis.axis_id] = binding
        unbound.remove(axis.axis_id)
        reasons.append(
            DecisionReason(
                "AUTOMATIC_VISIBLE_ROLE",
                f"{axis.name} remains visible as {chosen_role.value}",
                axis.axis_id,
            )
        )

    spec = ViewSpec(
        schema.fingerprint,
        intent,
        tuple(bindings.values()),
        display_selections,
    )
    try:
        validate_view_spec(schema, spec, contract)
    except (TypeError, ValueError, IndexError) as exc:
        return _needs("CONTRACT_REJECTED", str(exc))
    return ViewSuggestion(
        spec,
        SuggestionStatus.REVIEW_REQUIRED if review_required else SuggestionStatus.RESOLVED,
        tuple(reasons),
        tuple(alternatives),
    )


__all__ = ["suggest_view"]
