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


# Ordered product vocabulary.  The order is presentation policy as well as a
# validation boundary: Workbench labels these values but must not restate which
# Figure intents a Grid can contain.
GRID_INTENTS = (
    ViewIntent.CURVE,
    ViewIntent.HISTOGRAM,
    ViewIntent.IMAGE,
)


def _unresolved(code: str, message: str, axis_id=None) -> ViewSuggestion:
    return ViewSuggestion(
        None,
        SuggestionStatus.NEEDS_INPUT,
        (DecisionReason(code, message, axis_id),),
        (),
    )


def _grid_contract(intent: ViewIntent):
    """Allow one explicit Grid facet without auto-inventing another one.

    Ordinary contracts keep extra information axes visible as slider, batch,
    or histogram-sample bindings; they never create implicit multi-cell output.
    Grid is the explicit page control, so FACET remains legal for every role the
    intent already knows.  It is always placed last; the suggestion engine uses
    an available ordinary role for every *other* axis and can never manufacture
    a second facet merely because the schema gained another dimension.
    """

    contract = dataset_contract_for(intent)
    policies = tuple(
        replace(
            policy,
            automatic_roles=tuple(
                role
                for role in policy.automatic_roles
                if role is not AxisViewRole.FACET
            )
            + (AxisViewRole.FACET,),
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
    schema: DatasetSchema,
    contract,
    view: ViewSpec | None,
    facet_axis_id: AxisId,
) -> dict[str, AxisId | None]:
    """Reserve the facet, then fill display slots from declared axes.

    The generic suggestion engine normally chooses display axes before it
    applies explicit FACET/BATCH preferences.  That ordering is right for an
    ordinary one-panel suggestion, but wrong for Grid authoring: the operator
    has already said which axis becomes the cells.  Without reserving it here,
    a sole scan axis is first consumed as ``X`` and the later facet request
    conflicts with it; adding an unrelated second scan axis then makes the same
    facet suddenly legal.  Grid choices must not depend on that accident.

    Existing display bindings survive a facet edit when they still satisfy the
    typed contract.  Empty slots are filled by the contract's role preference,
    and within one role by :func:`dataset_axes` declaration order.  No ndarray
    rank, value, singleton heuristic, or AxisId spelling enters the decision.
    """

    role_to_field = {
        AxisViewRole.X: "x_axis_id",
        AxisViewRole.IMAGE_X: "image_x_axis_id",
        AxisViewRole.IMAGE_Y: "image_y_axis_id",
    }
    prepared: dict[str, AxisId | None] = {
        field: None for field in role_to_field.values()
    }
    axes = dataset_axes(schema)
    axis_by_id = {axis.axis_id: axis for axis in axes}
    slot_by_role = {
        slot.binding_role: slot for slot in contract.display_slots
    }
    used = {facet_axis_id}
    if view is not None:
        for binding in view.axis_bindings:
            field = role_to_field.get(binding.role)
            slot = slot_by_role.get(binding.role)
            axis = axis_by_id.get(binding.axis_id)
            if (
                field is not None
                and slot is not None
                and axis is not None
                and binding.axis_id not in used
                and axis.role in slot.preferred_axis_roles
            ):
                prepared[field] = binding.axis_id
                used.add(binding.axis_id)

    for slot in contract.display_slots:
        field = role_to_field[slot.binding_role]
        if prepared[field] is not None:
            continue
        chosen = next(
            (
                axis
                for preferred_role in slot.preferred_axis_roles
                for axis in axes
                if axis.axis_id not in used and axis.role == preferred_role
            ),
            None,
        )
        if chosen is not None:
            prepared[field] = chosen.axis_id
            used.add(chosen.axis_id)
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
    if not isinstance(intent, ViewIntent) or intent not in GRID_INTENTS:
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

    display_preferences = _display_preferences(
        schema,
        contract,
        current_view,
        facet_axis_id,
    )
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


def suggest_default_grid_view(schema: DatasetSchema) -> ViewSuggestion:
    """Suggest a stable, visible Grid view from declared axis semantics.

    Intent preference is stable frontend policy: an image cell is preferred
    when the schema can supply its plane, followed by a curve and a histogram.
    Within that intent the first legal declared axis is the default facet.  The
    authored ``ViewSpec`` still records that exact choice and Setting exposes
    every alternative, so this removes startup friction without introducing a
    hidden projection.  Point, scan, repeat, and trailing data axes all use the
    same resolver; ndarray rank, values, and AxisId spelling never participate.
    """

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    for intent in (
        ViewIntent.IMAGE,
        ViewIntent.CURVE,
        ViewIntent.HISTOGRAM,
    ):
        candidates = tuple(
            suggestion
            for axis in dataset_axes(schema)
            if axis.size > 1
            for suggestion in (
                resolve_grid_view(schema, intent, axis.axis_id),
            )
            if suggestion.spec is not None
        )
        if not candidates:
            continue
        return candidates[0]
    return _unresolved(
        "NO_DEFAULT_GRID_VIEW",
        "the DatasetSchema does not determine a complete Grid view",
    )


__all__ = [
    "GRID_INTENTS",
    "grid_facet_axes",
    "grid_facet_axis",
    "resolve_grid_view",
    "suggest_default_grid_view",
]
