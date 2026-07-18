"""Deterministic, metadata-only ViewSpec suggestions."""

from __future__ import annotations

from dataclasses import replace
from math import prod

from zlc_data import (
    DatasetSchema,
    FitResultBatch,
    REPEAT,
    Selection,
)

from .contract import (
    _first_visible_point_tuple,
    _selection_fit_projection,
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


def _preferred_display_axis(preferences: ViewPreferences, role: AxisViewRole):
    return {
        AxisViewRole.X: preferences.x_axis_id,
        AxisViewRole.IMAGE_X: preferences.image_x_axis_id,
        AxisViewRole.IMAGE_Y: preferences.image_y_axis_id,
    }[role]


def _repeat_binding(axis_id, mode: RepeatViewMode, allowed_indices) -> AxisViewBinding:
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
        if len(allowed_indices) == 1:
            return AxisViewBinding(
                axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(allowed_indices[0]),
            )
        return AxisViewBinding(axis_id, AxisViewRole.SELECTED, selector=LatestNonempty())
    if mode is RepeatViewMode.BATCH:
        return AxisViewBinding(axis_id, AxisViewRole.BATCH)
    if mode is RepeatViewMode.FACET:
        return AxisViewBinding(axis_id, AxisViewRole.FACET)
    return AxisViewBinding(axis_id, AxisViewRole.SAMPLE)


def _plan_automatic_bindings(
    axes,
    contract,
    allowed,
    existing_bindings,
    point_defaults,
):
    """Find the best contract-ordered visible layout without using dataset order."""

    policy_order = {
        policy.axis_role: position
        for position, policy in enumerate(contract.role_policies)
    }
    ordered_axes = tuple(
        sorted(
            axes,
            key=lambda axis: (
                policy_order.get(axis.role, len(policy_order)),
                axis.axis_id.value,
            ),
        )
    )
    batch_product = prod(
        len(allowed[binding.axis_id])
        for binding in existing_bindings.values()
        if binding.role is AxisViewRole.BATCH
    )
    facet_product = prod(
        len(allowed[binding.axis_id])
        for binding in existing_bindings.values()
        if binding.role is AxisViewRole.FACET
    )
    if (
        batch_product > contract.maximum_batch_series
        or facet_product > contract.maximum_facet_cells
    ):
        return None

    # A state is bounded by the two contract products.  Its value keeps the
    # lexicographically best contract-rank plan reaching that state; future
    # feasibility depends only on the state.
    states = {
        (batch_product, facet_product): ((), ()),
    }
    for axis in ordered_axes:
        policy = contract.policy_for(axis.role)
        if policy is None or not policy.automatic_roles:
            return None
        candidates = []
        cardinality = len(allowed[axis.axis_id])
        for rank, role in enumerate(policy.automatic_roles):
            if role is AxisViewRole.SLIDER:
                fixed = AxisViewBinding(
                    axis.axis_id,
                    role,
                    selector=FixedIndex(
                        point_defaults.get(
                            axis.axis_id,
                            allowed[axis.axis_id][0],
                        )
                    ),
                )
                candidates.append((rank, fixed))
            else:
                candidates.append((rank, AxisViewBinding(axis.axis_id, role)))

        next_states = {}
        for (batch, facet), (score, plan) in states.items():
            for rank, binding in candidates:
                next_batch, next_facet = batch, facet
                if binding.role is AxisViewRole.BATCH:
                    next_batch *= cardinality
                    if next_batch > contract.maximum_batch_series:
                        continue
                elif binding.role is AxisViewRole.FACET:
                    next_facet *= cardinality
                    if next_facet > contract.maximum_facet_cells:
                        continue
                state = (next_batch, next_facet)
                candidate = (score + (rank,), plan + (binding,))
                incumbent = next_states.get(state)
                if incumbent is None or candidate[0] < incumbent[0]:
                    next_states[state] = candidate
        if not next_states:
            return None
        states = next_states

    _, (_, planned) = min(
        states.items(),
        key=lambda item: (item[1][0], item[0]),
    )
    return {binding.axis_id: binding for binding in planned}


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
    display_selections = () if selection is None else (selection,)
    allowed = {}
    for axis in axes:
        try:
            allowed[axis.axis_id] = display_axis_indices(axis, display_selections)
        except IndexError as exc:
            return _needs(
                "SELECTION_OUT_OF_RANGE",
                str(exc),
                axis_id=axis.axis_id,
            )
        except (TypeError, ValueError) as exc:
            return _needs(
                "SELECTION_INVALID",
                str(exc),
                axis_id=axis.axis_id,
            )
    selected_axis_ids = {
        term.axis_id for term in (() if selection is None else selection.terms)
    }
    selected = {
        axis_id: allowed[axis_id][0]
        for axis_id in selected_axis_ids
        if len(allowed[axis_id]) == 1
    }
    point_tuple = _first_visible_point_tuple(schema, allowed)
    if point_tuple is None:
        return _needs(
            "EMPTY_POINT_SELECTION",
            "the display selection contains no physical point-layout row",
        )
    point_defaults = {
        axis.axis_id: index
        for axis, index in zip(schema.point_axes, point_tuple)
    }
    unbound = set(axis_by_id)
    bindings: dict = {}
    reasons: list[DecisionReason] = []
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
            role_candidates = tuple(
                (
                    role,
                    tuple(
                        axis
                        for axis in axes
                        if axis.axis_id in unbound and axis.role == role
                    ),
                )
                for role in slot.preferred_axis_roles
            )
            unselected_candidates = tuple(
                (
                    role,
                    tuple(
                        axis
                        for axis in candidates
                        if axis.axis_id not in selected
                    ),
                )
                for role, candidates in role_candidates
            )
            candidate_groups = (
                unselected_candidates
                if any(candidates for _, candidates in unselected_candidates)
                else role_candidates
            )
            for role, candidates in candidate_groups:
                if len(candidates) == 1:
                    chosen = candidates[0]
                    break
                if len(candidates) > 1:
                    alternatives = tuple(
                        ViewAlternative(axis.axis_id, slot.binding_role, axis.name)
                        for axis in sorted(candidates, key=lambda item: item.axis_id.value)
                    )
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
        bindings[repeat_axis.axis_id] = _repeat_binding(
            repeat_axis.axis_id,
            repeat_mode,
            allowed[repeat_axis.axis_id],
        )
        unbound.remove(repeat_axis.axis_id)
        reasons.append(
            DecisionReason(
                "REPEAT_POLICY",
                f"repeat uses the {repeat_mode.value} presentation policy",
                repeat_axis.axis_id,
            )
        )

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
    batch_size = prod(
        len(allowed[binding.axis_id])
        for binding in bindings.values()
        if binding.role is AxisViewRole.BATCH
    )
    facet_size = prod(
        len(allowed[binding.axis_id])
        for binding in bindings.values()
        if binding.role is AxisViewRole.FACET
    )
    if batch_size > contract.maximum_batch_series:
        return _needs(
            "BATCH_LIMIT",
            "repeat batch exceeds contract limit",
            axis_id=repeat_axis.axis_id,
        )
    if facet_size > contract.maximum_facet_cells:
        return _needs("FACET_LIMIT", "visible facets exceed contract limit")
    for axis_ids, view_role in explicit_roles:
        for axis_id in axis_ids:
            axis = axis_by_id.get(axis_id)
            if axis is None:
                return _needs("UNKNOWN_AXIS", "preferred axis is absent", axis_id=axis_id)
            if axis_id not in unbound:
                return _needs(
                    "EXPLICIT_ROLE_CONFLICT",
                    f"axis cannot also be {view_role.value}",
                    axis_id=axis_id,
                )
            if view_role is AxisViewRole.SAMPLE and intent is not ViewIntent.HISTOGRAM:
                return _needs("SAMPLE_REQUIRES_HISTOGRAM", "SAMPLE is histogram-only", axis_id=axis_id)
            if view_role is AxisViewRole.BATCH:
                batch_size *= len(allowed[axis.axis_id])
                if batch_size > contract.maximum_batch_series:
                    return _needs("BATCH_LIMIT", "explicit batch exceeds contract limit", axis_id=axis_id)
            if view_role is AxisViewRole.FACET:
                facet_size *= len(allowed[axis.axis_id])
                if facet_size > contract.maximum_facet_cells:
                    return _needs("FACET_LIMIT", "explicit facet exceeds contract limit", axis_id=axis_id)
            bindings[axis_id] = AxisViewBinding(axis_id, view_role)
            unbound.remove(axis_id)
            if view_role is AxisViewRole.SAMPLE and axis.role not in (REPEAT,):
                review_required = True
            reasons.append(
                DecisionReason("EXPLICIT_AXIS_ROLE", f"{axis.name} uses {view_role.value}", axis_id)
            )

    automatic_axes = tuple(axis for axis in axes if axis.axis_id in unbound)
    for axis in automatic_axes:
        policy = contract.policy_for(axis.role)
        if policy is None or not policy.automatic_roles:
            return _needs(
                "UNRESOLVED_INFORMATION_AXIS",
                f"{axis.name} requires an explicit selection, facet, batch, or reducer",
                axis_id=axis.axis_id,
            )
    planned = _plan_automatic_bindings(
        automatic_axes,
        contract,
        allowed,
        bindings,
        point_defaults,
    )
    if planned is None:
        return _needs(
            "VIEW_CAPACITY_EXHAUSTED",
            "no contract-valid visible assignment fits the batch/facet limits",
        )
    for axis in automatic_axes:
        binding = planned[axis.axis_id]
        bindings[axis.axis_id] = binding
        unbound.remove(axis.axis_id)
        reasons.append(
            DecisionReason(
                "AUTOMATIC_VISIBLE_ROLE",
                f"{axis.name} remains visible as {binding.role.value}",
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
        (),
    )


def _suggest_fit_view_for_effective_schema(
    schema: DatasetSchema,
    result: FitResultBatch,
    selection: Selection | None = None,
    preferences: ViewPreferences | None = None,
) -> ViewSuggestion:
    """Suggest against the exact schema consumed by one fit."""

    arity = len(result.fit_axis_specs)
    if arity not in (1, 2):
        return _needs(
            "FIT_ARITY_UNSUPPORTED",
            "the current figure surface supports one- and two-axis fit models",
        )
    preferences = ViewPreferences() if preferences is None else preferences
    if not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")

    repeat_axis = schema.repeat_axis
    batch_ids = {axis.axis_id for axis in result.batch_axis_specs}
    selected = selection
    if repeat_axis.axis_id in batch_ids and preferences.repeat_mode is None:
        terms = () if selection is None else selection.terms
        if all(term.axis_id != repeat_axis.axis_id for term in terms):
            selected = Selection((*terms, *Selection.index(repeat_axis.axis_id, 0).terms))
        preferences = replace(preferences, repeat_mode=RepeatViewMode.LATEST)

    if arity == 1:
        fit_axis_id = result.fit_axis_specs[0].axis_id
        if preferences.x_axis_id not in (None, fit_axis_id):
            return _needs(
                "FIT_AXIS_VIEW_MISMATCH",
                "the requested x axis differs from the fitted axis",
                axis_id=preferences.x_axis_id,
            )
        preferences = replace(preferences, x_axis_id=fit_axis_id)
        intent = ViewIntent.CURVE
    else:
        x_axis_id = result.fit_axis_specs[0].axis_id
        y_axis_id = result.fit_axis_specs[1].axis_id
        if preferences.image_x_axis_id not in (None, x_axis_id):
            return _needs(
                "FIT_X_AXIS_VIEW_MISMATCH",
                "the requested image x axis differs from the first fitted axis",
                axis_id=preferences.image_x_axis_id,
            )
        if preferences.image_y_axis_id not in (None, y_axis_id):
            return _needs(
                "FIT_Y_AXIS_VIEW_MISMATCH",
                "the requested image y axis differs from the second fitted axis",
                axis_id=preferences.image_y_axis_id,
            )
        preferences = replace(
            preferences,
            image_x_axis_id=x_axis_id,
            image_y_axis_id=y_axis_id,
        )
        intent = ViewIntent.IMAGE

    suggestion = suggest_view(schema, intent, selected, preferences)
    if suggestion.spec is None:
        return suggestion
    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    for axis in result.batch_axis_specs:
        if suggestion.spec.binding(axis.axis_id).role not in allowed_batch_roles:
            return _needs(
                "FIT_BATCH_AXIS_NOT_IDENTIFIED",
                f"fit batch axis {axis.name} is not visible or explicitly selected",
                axis_id=axis.axis_id,
            )
    return suggestion


def suggest_fit_view(
    schema: DatasetSchema,
    result: FitResultBatch,
    selection: Selection | None = None,
    preferences: ViewPreferences | None = None,
) -> ViewSuggestion:
    """Suggest one faithful display for raw or explicitly transformed fit data."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if result.source_ref.schema_fingerprint != schema.fingerprint:
        return _needs(
            "FIT_SOURCE_SCHEMA_MISMATCH",
            "fit result and displayed source use different schemas",
        )
    if result.spec.committed_transform is None:
        return _suggest_fit_view_for_effective_schema(
            schema,
            result,
            selection,
            preferences,
        )
    if selection is not None:
        if not isinstance(selection, Selection):
            raise TypeError("selection must be zlc_data.Selection or None")
        batch_ids = {axis.axis_id for axis in result.batch_axis_specs}
        if any(term.axis_id not in batch_ids for term in selection.terms):
            return _needs(
                "TRANSFORMED_FIT_SELECTION_CONFLICT",
                "additional transformed-fit display selection may name batch axes only",
            )
    try:
        effective_schema, authority_selection = _selection_fit_projection(
            schema,
            result,
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _needs("TRANSFORMED_FIT_DISPLAY_UNAVAILABLE", str(exc))
    effective = _suggest_fit_view_for_effective_schema(
        effective_schema,
        result,
        selection,
        preferences,
    )
    if effective.spec is None:
        return effective
    try:
        display_selection = Selection(
            (
                *authority_selection.terms,
                *(
                    term
                    for display in effective.spec.display_selections
                    for term in display.terms
                ),
            )
        )
        lifted = ViewSpec(
            schema.fingerprint,
            effective.spec.intent,
            effective.spec.axis_bindings,
            (display_selection,),
        )
        validate_view_spec(schema, lifted, contract_for(lifted.intent))
    except (TypeError, ValueError, IndexError) as exc:
        return _needs(
            "TRANSFORMED_FIT_VIEW_REJECTED",
            f"committed ROI and display selection cannot be composed: {exc}",
        )
    return ViewSuggestion(
        lifted,
        effective.status,
        (
            DecisionReason(
                "COMMITTED_TRANSFORM_DISPLAY",
                "the visible ROI is copied exactly from the committed FitSpec",
            ),
            *effective.reasons,
        ),
        (),
    )


__all__ = ["suggest_fit_view", "suggest_view"]
