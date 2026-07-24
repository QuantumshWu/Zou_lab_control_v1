"""Pure named-axis resolver for a one-facet Grid Figure.

A Grid control authors exactly two facts: the cell ``ViewIntent`` and the one
declared axis expanded into cells.  Completing the remaining display roles is
frontend policy and belongs here, not in a Qt combo-box callback.  Resolution
uses axis declarations and schema order only; it never inspects array rank,
values, singleton accidents, or an ``AxisId`` spelling.
"""

from __future__ import annotations

from dataclasses import replace

from zlc_data import AxisId, AxisSpec, DatasetSchema, Selection

from .contract import dataset_axes, dataset_contract_for
from .model import (
    AxisViewRole,
    DecisionReason,
    RepeatViewMode,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    ViewSuggestion,
)
from .suggest import _suggest_view


_GRID_INTENTS = frozenset(
    (ViewIntent.CURVE, ViewIntent.HISTOGRAM, ViewIntent.IMAGE)
)


def _unresolved(code: str, message: str, axis_id=None) -> ViewSuggestion:
    return ViewSuggestion(
        None,
        SuggestionStatus.NEEDS_INPUT,
        (DecisionReason(code, message, axis_id),),
        (),
    )


def _grid_contract(intent: ViewIntent):
    """Prefer a single cell over silently creating additional facet axes."""

    contract = dataset_contract_for(intent)
    policies = tuple(
        replace(
            policy,
            automatic_roles=tuple(
                role
                for role in policy.automatic_roles
                if role is not AxisViewRole.FACET
            )
            + tuple(
                role
                for role in policy.automatic_roles
                if role is AxisViewRole.FACET
            ),
        )
        for policy in contract.role_policies
    )
    return replace(contract, role_policies=policies)


def _display_selection(view: ViewSpec | None) -> Selection | None:
    if view is None:
        return None
    terms = tuple(
        term
        for selection in view.display_selections
        for term in selection.terms
    )
    return None if not terms else Selection(terms)


def _display_preferences(
    view: ViewSpec | None,
    facet_axis_id: AxisId,
) -> dict[str, AxisId | None]:
    role_to_field = {
        AxisViewRole.X: "x_axis_id",
        AxisViewRole.IMAGE_X: "image_x_axis_id",
        AxisViewRole.IMAGE_Y: "image_y_axis_id",
    }
    prepared: dict[str, AxisId | None] = {
        field: None for field in role_to_field.values()
    }
    if view is None:
        return prepared
    for binding in view.axis_bindings:
        field = role_to_field.get(binding.role)
        if field is not None and binding.axis_id != facet_axis_id:
            prepared[field] = binding.axis_id
    return prepared


def grid_facet_axis(view: ViewSpec) -> AxisId:
    """Return the sole authored facet axis, rejecting multi-facet residue."""

    if not isinstance(view, ViewSpec):
        raise TypeError("view must be ViewSpec")
    facets = tuple(
        binding.axis_id
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.FACET
    )
    if len(facets) != 1:
        raise ValueError("a Grid ViewSpec must contain exactly one facet axis")
    return facets[0]


def resolve_grid_view(
    schema: DatasetSchema,
    intent: ViewIntent,
    facet_axis_id: AxisId,
    *,
    current_view: ViewSpec | None = None,
    repeat_mode: RepeatViewMode | None = None,
) -> ViewSuggestion:
    """Resolve one named facet into a complete, single-facet ``ViewSpec``.

    A previous view contributes only its display selection, explicit display
    axes, and repeat policy.  It cannot smuggle a second facet or a hidden Qt
    choice into the new request.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(intent, ViewIntent) or intent not in _GRID_INTENTS:
        raise ValueError("Grid cells require CURVE, HISTOGRAM, or IMAGE intent")
    if not isinstance(facet_axis_id, AxisId):
        raise TypeError("facet_axis_id must be AxisId")
    selection_view = current_view
    if current_view is not None:
        if not isinstance(current_view, ViewSpec):
            raise TypeError("current_view must be ViewSpec or None")
        if current_view.schema_fingerprint != schema.fingerprint:
            raise ValueError("current Grid view belongs to another schema")
        if current_view.intent is not intent:
            current_view = None
    if repeat_mode is not None and not isinstance(repeat_mode, RepeatViewMode):
        raise TypeError("repeat_mode must be RepeatViewMode or None")

    axes = dataset_axes(schema)
    axis_by_id = {axis.axis_id: axis for axis in axes}
    facet_axis = axis_by_id.get(facet_axis_id)
    if facet_axis is None:
        return _unresolved(
            "UNKNOWN_GRID_FACET",
            "the selected facet axis is absent from the dataset schema",
            facet_axis_id,
        )
    if facet_axis.size <= 1:
        return _unresolved(
            "SINGLETON_GRID_FACET",
            "a singleton axis cannot expand into Grid cells",
            facet_axis_id,
        )

    repeat_axis_id = schema.repeat_axis.axis_id
    contract = _grid_contract(intent)
    if facet_axis_id == repeat_axis_id:
        if repeat_mode not in (None, RepeatViewMode.FACET):
            return _unresolved(
                "FACET_REPEAT_CONFLICT",
                "the repeat axis is the Grid facet and must remain FACET",
                facet_axis_id,
            )
        resolved_repeat_mode = RepeatViewMode.FACET
        facet_ids = ()
    else:
        if repeat_mode is None and current_view is not None:
            from .contract import _repeat_mode_for_binding

            candidate_mode = _repeat_mode_for_binding(
                current_view.binding(repeat_axis_id)
            )
            repeat_mode = (
                None
                if candidate_mode is RepeatViewMode.FACET
                else candidate_mode
            )
        resolved_repeat_mode = repeat_mode or contract.default_repeat_mode
        if resolved_repeat_mode is RepeatViewMode.FACET:
            return _unresolved(
                "MULTIPLE_GRID_FACETS",
                "repeat FACET would create a second Grid facet axis",
                repeat_axis_id,
            )
        facet_ids = (facet_axis_id,)

    display_preferences = _display_preferences(current_view, facet_axis_id)
    preferences = ViewPreferences(
        repeat_mode=resolved_repeat_mode,
        facet_axis_ids=facet_ids,
        **display_preferences,
    )
    # A cell-family change must not silently erase the Figure's display ROI.
    # Display-axis preferences are intent-specific, but Selection is not.
    selection = _display_selection(selection_view)
    suggestion = _suggest_view(
        schema,
        intent,
        selection,
        preferences,
        contract=contract,
    )

    # Multiple same-role display candidates are a presentation choice, not a
    # second operator-facing Grid dimension.  Resolve them by declared schema
    # order, explicitly excluding the authored facet.  This policy is pure and
    # testable; the Qt shell never compares or sorts AxisId strings.
    axis_order = {axis.axis_id: index for index, axis in enumerate(axes)}
    preference_field = {
        AxisViewRole.X: "x_axis_id",
        AxisViewRole.IMAGE_X: "image_x_axis_id",
        AxisViewRole.IMAGE_Y: "image_y_axis_id",
    }
    filled_roles: set[AxisViewRole] = set()
    while suggestion.status is SuggestionStatus.NEEDS_INPUT:
        if not any(
            reason.code == "AMBIGUOUS_DISPLAY_AXIS"
            for reason in suggestion.reasons
        ):
            break
        alternatives = tuple(
            alternative
            for alternative in suggestion.alternatives
            if alternative.axis_id != facet_axis_id
        )
        roles = {alternative.binding_role for alternative in alternatives}
        if len(roles) != 1:
            break
        role = next(iter(roles))
        field = preference_field.get(role)
        if (
            field is None
            or role in filled_roles
            or getattr(preferences, field) is not None
            or not alternatives
        ):
            break
        chosen = min(
            alternatives,
            key=lambda alternative: axis_order[alternative.axis_id],
        )
        preferences = replace(preferences, **{field: chosen.axis_id})
        filled_roles.add(role)
        suggestion = _suggest_view(
            schema,
            intent,
            selection,
            preferences,
            contract=contract,
        )

    if suggestion.spec is None:
        return suggestion
    try:
        resolved_facet = grid_facet_axis(suggestion.spec)
    except ValueError:
        return _unresolved(
            "GRID_FACET_CARDINALITY",
            "Grid resolution did not produce exactly one facet axis",
            facet_axis_id,
        )
    if resolved_facet != facet_axis_id:
        return _unresolved(
            "GRID_FACET_MISMATCH",
            "Grid resolution changed the authored facet axis",
            facet_axis_id,
        )
    return suggestion


def grid_facet_axes(
    schema: DatasetSchema,
    intent: ViewIntent,
    *,
    current_view: ViewSpec | None = None,
) -> tuple[AxisSpec, ...]:
    """Return declared axes that can form a complete one-facet Grid view."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    return tuple(
        axis
        for axis in dataset_axes(schema)
        if axis.size > 1
        and resolve_grid_view(
            schema,
            intent,
            axis.axis_id,
            current_view=current_view,
        ).spec
        is not None
    )


__all__ = ["grid_facet_axes", "grid_facet_axis", "resolve_grid_view"]
