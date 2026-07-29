"""Pure typed-source resolver for a one-facet Grid Figure."""

from __future__ import annotations

from dataclasses import replace

from zlc_data import AxisSourceRef, DatasetSchema
from zlc_data.schema import resolve_point_rows

from .contract import (
    _dataset_sources,
    _resolve_selected_point_ordinals,
    _source_cardinality,
    _source_role,
    dataset_contract_for,
)
from .model import (
    AxisViewRole,
    DecisionReason,
    SourceViewBinding,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    ViewSuggestion,
)
from .suggest import _suggest_view


GRID_INTENTS = (
    ViewIntent.CURVE,
    ViewIntent.HISTOGRAM,
    ViewIntent.IMAGE,
)


def _unresolved(
    code: str,
    message: str,
    source: AxisSourceRef | None = None,
) -> ViewSuggestion:
    return ViewSuggestion(
        None,
        SuggestionStatus.NEEDS_INPUT,
        (DecisionReason(code, message, source),),
    )


def _grid_contract(intent: ViewIntent):
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


def _display_preferences(
    current_view: ViewSpec | None,
    facet_source: AxisSourceRef,
) -> dict[str, AxisSourceRef | None]:
    fields = {
        AxisViewRole.X: "x_source",
        AxisViewRole.IMAGE_X: "image_x_source",
        AxisViewRole.IMAGE_Y: "image_y_source",
    }
    result = {field: None for field in fields.values()}
    if current_view is None:
        return result
    for binding in current_view.source_bindings:
        field = fields.get(binding.role)
        if field is not None and binding.source != facet_source:
            result[field] = binding.source
    return result


def _facet_cardinality(
    schema: DatasetSchema,
    source: AxisSourceRef,
    point_ordinals: tuple[int, ...] | None,
) -> int:
    if source.kind in {
        AxisSourceRef.POINT_ROWS,
        AxisSourceRef.POINT_COORDINATE,
        AxisSourceRef.GRID_DIMENSION,
    }:
        resolved = resolve_point_rows(
            schema.point_table,
            schema.grid_topology,
            point_ordinals=point_ordinals,
            group_sources=(source,),
        )
        return len(resolved.group_member_ordinals)
    return _source_cardinality(schema, source)


def _facet_source_rejection(
    schema: DatasetSchema,
    intent: ViewIntent,
    source: AxisSourceRef,
    point_ordinals: tuple[int, ...] | None,
) -> ViewSuggestion | None:
    """Return the sole semantic/cardinality rejection for one declared facet."""

    if source.kind == AxisSourceRef.POINT_ORDINAL:
        return _unresolved(
            "POINT_ORDINAL_GRID_FACET",
            "the synthetic point ordinal cannot replace an authored Grid facet",
            source,
        )
    if source.kind != AxisSourceRef.POINT_ROWS and source != AxisSourceRef.tensor(
        schema.repeat_axis.axis_id
    ):
        policy = _grid_contract(intent).policy_for(_source_role(schema, source))
        if policy is None or AxisViewRole.FACET not in policy.automatic_roles:
            return _unresolved(
                "GRID_FACET_ROLE_UNAVAILABLE",
                "the declared source role cannot expand into Grid cells",
                source,
            )
    try:
        cardinality = _facet_cardinality(schema, source, point_ordinals)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return _unresolved("GRID_FACET_UNAVAILABLE", str(error), source)
    if cardinality <= 1:
        return _unresolved(
            "SINGLETON_GRID_FACET",
            "a singleton source cannot expand into Grid cells",
            source,
        )
    return None


def grid_facet_source(view: ViewSpec) -> AxisSourceRef:
    """Return the sole authored facet source."""

    if not isinstance(view, ViewSpec):
        raise TypeError("view must be ViewSpec")
    facets = tuple(
        binding.source
        for binding in view.source_bindings
        if binding.role is AxisViewRole.FACET
    )
    if len(facets) != 1:
        raise ValueError("a Grid ViewSpec must contain exactly one facet source")
    return facets[0]


def resolve_grid_view(
    schema: DatasetSchema,
    intent: ViewIntent,
    facet_source: AxisSourceRef,
    *,
    current_view: ViewSpec | None = None,
    repeat_binding: SourceViewBinding | None = None,
) -> ViewSuggestion:
    """Resolve one typed facet into a complete single-facet ViewSpec."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(intent, ViewIntent) or intent not in GRID_INTENTS:
        raise ValueError("Grid cells require CURVE, HISTOGRAM, or IMAGE intent")
    if not isinstance(facet_source, AxisSourceRef):
        raise TypeError("facet_source must be AxisSourceRef")
    if facet_source not in set(_dataset_sources(schema)):
        return _unresolved(
            "UNKNOWN_GRID_FACET",
            "the selected facet source is absent from the DatasetSchema",
            facet_source,
        )
    if current_view is not None:
        if not isinstance(current_view, ViewSpec):
            raise TypeError("current_view must be ViewSpec or None")
        if current_view.schema_fingerprint != schema.fingerprint:
            raise ValueError("current Grid view belongs to another schema")
        if current_view.intent is not intent:
            current_view = None
    if repeat_binding is not None and not isinstance(
        repeat_binding, SourceViewBinding
    ):
        raise TypeError("repeat_binding must be SourceViewBinding or None")

    point_ordinals = None if current_view is None else current_view.point_ordinals
    cardinality_ordinals = point_ordinals
    if current_view is not None:
        if facet_source.kind == AxisSourceRef.GRID_DIMENSION:
            cardinality_ordinals = _resolve_selected_point_ordinals(
                schema,
                current_view,
                ignore_selected_sources=(facet_source,),
            )
        elif facet_source.kind in {
            AxisSourceRef.POINT_ROWS,
            AxisSourceRef.POINT_COORDINATE,
        } and any(
            binding.source.kind == AxisSourceRef.GRID_DIMENSION
            and binding.role is AxisViewRole.SELECTED
            for binding in current_view.source_bindings
        ):
            # Raw/topology bindings cannot coexist. Preserve the exact visible
            # topology slice by materializing it into the sole row-filter
            # authority before the suggestion drops GridDimension bindings.
            point_ordinals = _resolve_selected_point_ordinals(schema, current_view)
            cardinality_ordinals = point_ordinals
    rejection = _facet_source_rejection(
        schema,
        intent,
        facet_source,
        cardinality_ordinals,
    )
    if rejection is not None:
        return rejection

    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    if repeat_binding is None and current_view is not None:
        previous = current_view.binding(repeat_source)
        if previous.role is not AxisViewRole.FACET:
            repeat_binding = previous
    if facet_source == repeat_source:
        repeat_binding = SourceViewBinding(repeat_source, AxisViewRole.FACET)
        facet_sources = ()
    else:
        if repeat_binding is not None and repeat_binding.role is AxisViewRole.FACET:
            return _unresolved(
                "MULTIPLE_GRID_FACETS",
                "repeat FACET would create a second Grid facet source",
                repeat_source,
            )
        facet_sources = (facet_source,)

    preferences = ViewPreferences(
        repeat_binding=repeat_binding,
        facet_sources=facet_sources,
        **_display_preferences(current_view, facet_source),
    )
    suggestion = _suggest_view(
        schema,
        intent,
        point_ordinals,
        preferences,
        contract=_grid_contract(intent),
    )
    if suggestion.spec is None:
        return suggestion
    try:
        resolved = grid_facet_source(suggestion.spec)
    except ValueError:
        return _unresolved(
            "GRID_FACET_CARDINALITY",
            "Grid resolution did not produce exactly one facet source",
            facet_source,
        )
    if resolved != facet_source:
        return _unresolved(
            "GRID_FACET_MISMATCH",
            "Grid resolution changed the authored facet source",
            facet_source,
        )
    return suggestion


def grid_facet_sources(
    schema: DatasetSchema,
    intent: ViewIntent,
) -> tuple[AxisSourceRef, ...]:
    """Return every declared source that is legal as the sole Grid facet."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(intent, ViewIntent) or intent not in GRID_INTENTS:
        raise ValueError("Grid cells require CURVE, HISTOGRAM, or IMAGE intent")
    return tuple(
        source
        for source in _dataset_sources(schema)
        if _facet_source_rejection(schema, intent, source, None) is None
    )


def suggest_default_grid_view(schema: DatasetSchema) -> ViewSuggestion:
    """Resolve a Grid default only when one declared candidate is unique."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    sources = _dataset_sources(schema)
    topology_sources = tuple(
        source
        for source in sources
        if source.kind == AxisSourceRef.GRID_DIMENSION
    )
    source_tiers = (
        topology_sources,
        tuple(source for source in sources if source not in topology_sources),
    )
    for tier in source_tiers:
        eligible_sources = tuple(
            source
            for source in tier
            if any(
                _facet_source_rejection(schema, intent, source, None) is None
                for intent in GRID_INTENTS
            )
        )
        if not eligible_sources:
            continue
        if len(eligible_sources) > 1:
            return _unresolved(
                "AMBIGUOUS_DEFAULT_GRID_FACET",
                "multiple equally preferred Grid facet sources are declared",
            )
        source = eligible_sources[0]
        candidates = tuple(
            suggestion
            for intent in GRID_INTENTS
            if (suggestion := resolve_grid_view(schema, intent, source)).spec
            is not None
        )
        if len(candidates) == 1:
            return candidates[0]
        return _unresolved(
            "AMBIGUOUS_DEFAULT_GRID_VIEW",
            "the preferred Grid facet does not determine one cell intent",
            source,
        )
    return _unresolved(
        "NO_DEFAULT_GRID_VIEW",
        "the DatasetSchema does not determine a complete Grid view",
    )


__all__ = [
    "GRID_INTENTS",
    "grid_facet_source",
    "grid_facet_sources",
    "resolve_grid_view",
    "suggest_default_grid_view",
]
