"""Deterministic, metadata-only typed-source ViewSpec suggestions."""

from __future__ import annotations

from dataclasses import replace

from zlc_data import (
    SCALAR,
    AxisSourceRef,
    DatasetSchema,
    FitResultBatch,
    HistogramSpec,
    IndexSelection,
    MissingPolicy,
    PointColumn,
    ReductionMethod,
    ReductionSpec,
    Selection,
    ValidityPolicy,
)

from .contract import (
    CURVE_CONTRACT,
    HISTOGRAM_CONTRACT,
    IMAGE_CONTRACT,
    _source_cardinality,
    _source_name,
    _source_role,
    _tensor_axis,
    _dataset_sources,
    dataset_contract_for,
    validate_view_spec,
)
from .model import (
    AxisViewRole,
    DecisionReason,
    DisplayReduction,
    DisplayReductionMethod,
    FixedIndex,
    LatestNonempty,
    SourceViewBinding,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    ViewSuggestion,
)


def _needs(
    code: str,
    message: str,
    *,
    source: AxisSourceRef | None = None,
) -> ViewSuggestion:
    return ViewSuggestion(
        None,
        SuggestionStatus.NEEDS_INPUT,
        (DecisionReason(code, message, source),),
    )


def _preferred_display_source(
    preferences: ViewPreferences,
    role: AxisViewRole,
) -> AxisSourceRef | None:
    return {
        AxisViewRole.X: preferences.x_source,
        AxisViewRole.IMAGE_X: preferences.image_x_source,
        AxisViewRole.IMAGE_Y: preferences.image_y_source,
    }[role]


def _default_repeat_binding(
    schema: DatasetSchema,
    intent: ViewIntent,
) -> SourceViewBinding:
    source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    if intent in {ViewIntent.IMAGE, ViewIntent.CURVE}:
        return SourceViewBinding(
            source,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(DisplayReductionMethod.MEAN),
        )
    if intent is ViewIntent.HISTOGRAM:
        return SourceViewBinding(source, AxisViewRole.SAMPLE)
    selector = FixedIndex(0) if schema.repeat_axis.size == 1 else LatestNonempty()
    return SourceViewBinding(source, AxisViewRole.SELECTED, selector=selector)


def _display_source_allowed(
    schema: DatasetSchema,
    source: AxisSourceRef,
    role: AxisViewRole,
    chosen: tuple[AxisSourceRef, ...],
) -> bool:
    if source.kind == AxisSourceRef.POINT_ROWS:
        return False
    if source.kind == AxisSourceRef.POINT_COORDINATE:
        assert source.axis_id is not None
        if schema.point_table.column(source.axis_id).value_kind != PointColumn.NUMERIC:
            return False
    if role in {AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y}:
        raw = {
            AxisSourceRef.POINT_ORDINAL,
            AxisSourceRef.POINT_COORDINATE,
        }
        if source.kind in raw and any(item.kind in raw for item in chosen):
            return False
        if source.kind == AxisSourceRef.GRID_DIMENSION and any(
            item.kind in raw for item in chosen
        ):
            return False
        if source.kind in raw and any(
            item.kind == AxisSourceRef.GRID_DIMENSION for item in chosen
        ):
            return False
    return True


def _automatic_tensor_binding(schema, source, contract):
    axis = _tensor_axis(schema, source)
    policy = contract.policy_for(axis.role)
    if policy is None or not policy.automatic_roles:
        return None
    role = policy.automatic_roles[0]
    if role is AxisViewRole.SELECTED:
        return SourceViewBinding(source, role, selector=FixedIndex(0))
    return SourceViewBinding(source, role)


def _suggest_view(
    schema: DatasetSchema,
    intent: ViewIntent,
    point_ordinals: tuple[int, ...] | None = None,
    preferences: ViewPreferences | None = None,
    *,
    contract,
) -> ViewSuggestion:
    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(intent, ViewIntent):
        raise TypeError("intent must be ViewIntent")
    preferences = ViewPreferences() if preferences is None else preferences
    if not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")

    available = _dataset_sources(schema)
    preferred_sources = (
        preferences.x_source,
        preferences.image_x_source,
        preferences.image_y_source,
        *preferences.batch_sources,
        *preferences.facet_sources,
        *preferences.sample_sources,
    )
    topology_mode = any(
        source is not None and source.kind == AxisSourceRef.GRID_DIMENSION
        for source in preferred_sources
    )
    if topology_mode:
        available = tuple(
            source
            for source in available
            if source.kind
            not in {
                AxisSourceRef.POINT_ROWS,
                AxisSourceRef.POINT_ORDINAL,
                AxisSourceRef.POINT_COORDINATE,
            }
        )
    else:
        # A GridTopology is optional presentation metadata over the same
        # authored point rows.  Its dimensions must not compete with the
        # declared row coordinates unless the caller explicitly chooses a
        # grid-dimension source.  Otherwise a one-dimensional grid exposes the
        # same physical axis twice and makes an ordinary curve spuriously
        # ambiguous.
        available = tuple(
            source
            for source in available
            if source.kind != AxisSourceRef.GRID_DIMENSION
        )
    available_set = set(available)
    reserved = {
        *preferences.batch_sources,
        *preferences.facet_sources,
        *preferences.sample_sources,
    }
    bindings: dict[AxisSourceRef, SourceViewBinding] = {}
    reasons: list[DecisionReason] = []
    chosen_display: list[AxisSourceRef] = []

    for slot in contract.display_slots:
        preferred = _preferred_display_source(preferences, slot.binding_role)
        if preferred is not None:
            if (
                preferred not in available_set
                or preferred in bindings
                or preferred in reserved
            ):
                return _needs(
                    "INVALID_DISPLAY_SOURCE",
                    f"preferred {slot.binding_role.value} source is absent or already bound",
                    source=preferred,
                )
            if (
                _source_role(schema, preferred) not in slot.preferred_axis_roles
                or not _display_source_allowed(
                    schema, preferred, slot.binding_role, tuple(chosen_display)
                )
            ):
                return _needs(
                    "DISPLAY_ROLE_MISMATCH",
                    f"{_source_name(schema, preferred)} cannot fill {slot.binding_role.value}",
                    source=preferred,
                )
            chosen = preferred
        else:
            chosen = None
            for preferred_role in slot.preferred_axis_roles:
                role_candidates = tuple(
                    source
                    for source in available
                    if source not in bindings
                    and source not in reserved
                    and _source_role(schema, source) == preferred_role
                    and _display_source_allowed(
                        schema,
                        source,
                        slot.binding_role,
                        tuple(chosen_display),
                    )
                )
                # Point ordinal is a fallback for an otherwise unlabelled
                # authored row sequence.  It never competes with a declared
                # numeric coordinate of the same preferred role.
                declared = tuple(
                    source
                    for source in role_candidates
                    if source.kind != AxisSourceRef.POINT_ORDINAL
                )
                candidates = declared or role_candidates
                if not candidates:
                    continue
                if len(candidates) > 1:
                    return _needs(
                        "AMBIGUOUS_DISPLAY_SOURCE",
                        f"multiple declared sources can fill {slot.binding_role.value}: "
                        + ", ".join(_source_name(schema, source) for source in candidates),
                    )
                chosen = candidates[0]
                break
            if chosen is None:
                return _needs(
                    "MISSING_DISPLAY_SOURCE",
                    f"no declared source can fill {slot.binding_role.value}",
                )
        bindings[chosen] = SourceViewBinding(chosen, slot.binding_role)
        chosen_display.append(chosen)
        reasons.append(
            DecisionReason(
                "DISPLAY_SOURCE",
                f"{_source_name(schema, chosen)} is the {slot.binding_role.value} source",
                chosen,
            )
        )

    for axis in schema.cell_schema.data_axes:
        source = AxisSourceRef.tensor(axis.axis_id)
        if source in bindings or axis.role != SCALAR:
            continue
        bindings[source] = SourceViewBinding(
            source,
            AxisViewRole.SELECTED,
            selector=FixedIndex(0),
        )
        reasons.append(
            DecisionReason(
                "SCALAR_CARRIER",
                "the declared scalar carrier resolves to its sole physical item",
                source,
            )
        )

    repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    explicit_repeat_role = next(
        (
            role
            for sources, role in (
                (preferences.sample_sources, AxisViewRole.SAMPLE),
                (preferences.batch_sources, AxisViewRole.BATCH),
                (preferences.facet_sources, AxisViewRole.FACET),
            )
            if repeat_source in sources
        ),
        None,
    )
    if repeat_source not in bindings and explicit_repeat_role is not None:
        bindings[repeat_source] = SourceViewBinding(
            repeat_source,
            explicit_repeat_role,
        )
        reasons.append(
            DecisionReason(
                "EXPLICIT_REPEAT_ROLE",
                f"repeat uses {explicit_repeat_role.value}",
                repeat_source,
            )
        )
    elif repeat_source not in bindings:
        repeat_binding = preferences.repeat_binding or _default_repeat_binding(
            schema, intent
        )
        if repeat_binding.source != repeat_source:
            return _needs(
                "INVALID_REPEAT_SOURCE",
                "repeat_binding must bind the declared repeat tensor source",
                source=repeat_binding.source,
            )
        bindings[repeat_source] = repeat_binding
        reasons.append(
            DecisionReason(
                "REPEAT_POLICY",
                f"repeat uses {repeat_binding.role.value}",
                repeat_source,
            )
        )

    explicit_roles = (
        (preferences.sample_sources, AxisViewRole.SAMPLE),
        (preferences.batch_sources, AxisViewRole.BATCH),
        (preferences.facet_sources, AxisViewRole.FACET),
    )
    for sources, role in explicit_roles:
        for source in sources:
            if source == repeat_source and bindings.get(source) == SourceViewBinding(
                source,
                role,
            ):
                continue
            if source not in available_set:
                return _needs("UNKNOWN_SOURCE", "preferred source is absent", source=source)
            if source in bindings:
                return _needs(
                    "EXPLICIT_ROLE_CONFLICT",
                    f"source cannot also be {role.value}",
                    source=source,
                )
            if role is AxisViewRole.SAMPLE and intent is not ViewIntent.HISTOGRAM:
                return _needs(
                    "SAMPLE_REQUIRES_HISTOGRAM",
                    "SAMPLE is histogram-only",
                    source=source,
                )
            bindings[source] = SourceViewBinding(source, role)
            reasons.append(
                DecisionReason(
                    "EXPLICIT_SOURCE_ROLE",
                    f"{_source_name(schema, source)} uses {role.value}",
                    source,
                )
            )

    for axis in (schema.repeat_axis, *schema.cell_schema.data_axes):
        source = AxisSourceRef.tensor(axis.axis_id)
        if source in bindings:
            continue
        binding = _automatic_tensor_binding(schema, source, contract)
        if binding is None:
            return _needs(
                "UNRESOLVED_TENSOR_SOURCE",
                f"{axis.name} requires an explicit role",
                source=source,
            )
        bindings[source] = binding
        reasons.append(
            DecisionReason(
                "AUTOMATIC_VISIBLE_ROLE",
                f"{axis.name} remains visible as {binding.role.value}",
                source,
            )
        )

    spec = ViewSpec(
        schema.fingerprint,
        intent,
        tuple(bindings.values()),
        point_ordinals,
    )
    try:
        validate_view_spec(schema, spec, contract)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _needs("CONTRACT_REJECTED", str(exc))
    return ViewSuggestion(
        spec,
        SuggestionStatus.RESOLVED,
        tuple(reasons),
    )


def suggest_view(
    schema: DatasetSchema,
    intent: ViewIntent,
    point_ordinals: tuple[int, ...] | None = None,
    preferences: ViewPreferences | None = None,
) -> ViewSuggestion:
    """Suggest one safe presentation using the ordinary public contract."""

    return _suggest_view(
        schema,
        intent,
        point_ordinals,
        preferences,
        contract=dataset_contract_for(intent),
    )


_AUTHORING_ROLE_FIELDS = {
    AxisViewRole.X: "x_source",
    AxisViewRole.IMAGE_X: "image_x_source",
    AxisViewRole.IMAGE_Y: "image_y_source",
    AxisViewRole.SAMPLE: "sample_sources",
    AxisViewRole.BATCH: "batch_sources",
}
_AUTHORING_DISPLAY_ROLES = frozenset(
    {AxisViewRole.X, AxisViewRole.IMAGE_X, AxisViewRole.IMAGE_Y}
)


def _view_preferences_from_spec(
    schema: DatasetSchema,
    view: ViewSpec,
) -> ViewPreferences:
    """Project one valid view back to the preferences that reproduce it."""

    validate_view_spec(schema, view)
    repeat = view.binding(AxisSourceRef.tensor(schema.repeat_axis.axis_id))
    display = {
        binding.role: binding.source
        for binding in view.source_bindings
        if binding.role in _AUTHORING_DISPLAY_ROLES
    }
    grouped = {
        role: tuple(binding.source for binding in view.source_bindings
                    if binding.role is role)
        for role in (AxisViewRole.SAMPLE, AxisViewRole.BATCH, AxisViewRole.FACET)
    }
    return ViewPreferences(
        repeat_binding=(
            None if repeat.role in grouped else repeat
        ),
        x_source=display.get(AxisViewRole.X),
        image_x_source=display.get(AxisViewRole.IMAGE_X),
        image_y_source=display.get(AxisViewRole.IMAGE_Y),
        sample_sources=grouped[AxisViewRole.SAMPLE],
        batch_sources=grouped[AxisViewRole.BATCH],
        facet_sources=grouped[AxisViewRole.FACET],
    )


def _with_authoring_role(
    preferences: ViewPreferences,
    role: AxisViewRole,
    value: AxisSourceRef | tuple[AxisSourceRef, ...],
) -> ViewPreferences:
    fields = {
        field: getattr(preferences, field)
        for field in (*_AUTHORING_ROLE_FIELDS.values(), "facet_sources")
    }
    if role in _AUTHORING_DISPLAY_ROLES:
        for field, existing in tuple(fields.items()):
            fields[field] = (
                tuple(item for item in existing if item != value)
                if isinstance(existing, tuple)
                else None if existing == value else existing
            )
        fields[_AUTHORING_ROLE_FIELDS[role]] = value
    else:
        selected = tuple(value)
        target = _AUTHORING_ROLE_FIELDS[role]
        for field, existing in tuple(fields.items()):
            if field != target:
                fields[field] = (
                    tuple(item for item in existing if item not in selected)
                    if isinstance(existing, tuple)
                    else None if existing in selected else existing
                )
        fields[target] = selected
    return replace(preferences, **fields)


def _view_authoring_candidates(
    schema: DatasetSchema,
    intent: ViewIntent,
    *,
    current_view: ViewSpec | None = None,
    preferences: ViewPreferences | None = None,
    faceted: bool = False,
):
    """Return legal adjacent role edits using only existing Figure values."""

    if not isinstance(schema, DatasetSchema) or not isinstance(intent, ViewIntent):
        raise TypeError("view authoring requires DatasetSchema and ViewIntent")
    if not isinstance(faceted, bool):
        raise TypeError("faceted must be bool")
    if current_view is not None:
        if current_view.intent is not intent:
            raise ValueError("current view and authoring intent disagree")
        preferences = _view_preferences_from_spec(schema, current_view)
    else:
        preferences = ViewPreferences() if preferences is None else preferences
        if not isinstance(preferences, ViewPreferences):
            raise TypeError("preferences must be ViewPreferences or None")

    roles = []
    if current_view is not None or not faceted:
        roles.extend(slot.binding_role for slot in dataset_contract_for(intent).display_slots)
    roles.extend((AxisViewRole.SAMPLE, AxisViewRole.BATCH))
    repeat = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    protected = {
        source
        for source in (
            preferences.x_source,
            preferences.image_x_source,
            preferences.image_y_source,
            *preferences.facet_sources,
        )
        if source is not None
    }
    invalid = {
        "INVALID_DISPLAY_SOURCE",
        "DISPLAY_ROLE_MISMATCH",
        "UNKNOWN_SOURCE",
        "EXPLICIT_ROLE_CONFLICT",
    }
    exact = {} if current_view is None else {
        binding.source: binding
        for binding in current_view.source_bindings
        if binding.role in {AxisViewRole.SELECTED, AxisViewRole.REDUCED}
    }
    rows = []
    for role in roles:
        grouped = role in {AxisViewRole.SAMPLE, AxisViewRole.BATCH}
        authored = getattr(preferences, _AUTHORING_ROLE_FIELDS[role])
        current = tuple(source for source in authored if source != repeat) if grouped else authored
        sources = tuple(
            source for source in _dataset_sources(schema)
            if source != repeat and (not grouped or source not in protected)
        )
        values = (
            (current,) + tuple(
                tuple(item for item in current if item != source)
                if source in current else (*current, source)
                for source in sources
            ) if grouped else sources
        )
        choices = []
        for value in dict.fromkeys(values):
            requested = value
            if grouped and repeat in authored:
                requested = (repeat, *value)
            try:
                candidate_preferences = _with_authoring_role(
                    preferences, role, requested
                )
                suggestion = suggest_view(
                    schema,
                    intent,
                    None if current_view is None else current_view.point_ordinals,
                    candidate_preferences,
                )
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            candidate = suggestion.spec
            if candidate is None:
                if current_view is not None or role not in _AUTHORING_DISPLAY_ROLES:
                    continue
                if any(reason.code in invalid for reason in suggestion.reasons):
                    continue
            else:
                if sum(binding.role is AxisViewRole.FACET
                       for binding in candidate.source_bindings) != int(faceted):
                    continue
                if current_view is not None:
                    candidate = replace(
                        candidate,
                        source_bindings=tuple(
                            previous
                            if (previous := exact.get(binding.source)) is not None
                            and previous.role is binding.role else binding
                            for binding in candidate.source_bindings
                        ),
                    )
                    try:
                        validate_view_spec(schema, candidate)
                    except (KeyError, TypeError, ValueError, IndexError):
                        continue
                    if candidate == current_view and value != current:
                        continue
            choices.append((value, candidate_preferences, candidate))
        has_current = any(value == current for value, *_rest in choices)
        if choices and (
            not has_current
            or current
            or len(choices) > 1
            or role in _AUTHORING_DISPLAY_ROLES
        ):
            selected = current if has_current else None
            rows.append((role, selected, tuple(choices)))
    return tuple(rows)


def _repeat_authoring_candidates(
    schema: DatasetSchema,
    view: ViewSpec,
) -> tuple[SourceViewBinding, ...]:
    """Return every legal adjacent repeat binding for one valid view."""

    validate_view_spec(schema, view)
    source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
    current = view.binding(source)
    if current.role is AxisViewRole.FACET:
        return ()
    fixed = current.selector.index if isinstance(current.selector, FixedIndex) else 0
    candidates = (
        SourceViewBinding(
            source,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(DisplayReductionMethod.MEAN),
        ),
        SourceViewBinding(
            source,
            AxisViewRole.REDUCED,
            reduction=DisplayReduction(DisplayReductionMethod.SUM),
        ),
        SourceViewBinding(source, AxisViewRole.SELECTED, selector=LatestNonempty()),
        SourceViewBinding(source, AxisViewRole.SELECTED, selector=FixedIndex(fixed)),
        SourceViewBinding(source, AxisViewRole.BATCH),
        SourceViewBinding(source, AxisViewRole.SAMPLE),
    )
    legal = []
    for binding in candidates:
        candidate = replace(
            view,
            source_bindings=tuple(
                binding if item.source == source else item
                for item in view.source_bindings
            ),
        )
        try:
            validate_view_spec(schema, candidate)
        except (ValueError, IndexError):
            continue
        legal.append(binding)
    if current not in legal:
        raise ValueError("current repeat binding is absent from legal view choices")
    return tuple(legal)


def _suggest_histogram_fit_view(
    schema: DatasetSchema,
    result: FitResultBatch,
    preferences: ViewPreferences,
) -> ViewSuggestion:
    """Rebuild one raw HISTOGRAM view from its committed sample authority."""

    transform = result.spec.committed_transform
    operations = tuple(transform.spec.operations)
    histogram = operations[-1]
    assert isinstance(histogram, HistogramSpec)
    if (
        len(result.spec.independent_sources) != 1
        or result.spec.independent_sources[0]
        != AxisSourceRef.tensor(histogram.bin_axis_id)
        or len(result.fit_axis_specs) != 1
        or result.fit_axis_specs[0].axis_id != histogram.bin_axis_id
    ):
        return _needs(
            "HISTOGRAM_FIT_AXIS_MISMATCH",
            "Histogram Fit result does not use its committed bin axis",
        )

    selection = None
    reduction = None
    for operation in operations[:-1]:
        if isinstance(operation, Selection) and selection is None and reduction is None:
            selection = operation
        elif isinstance(operation, ReductionSpec) and reduction is None:
            reduction = operation
        else:
            return _needs(
                "HISTOGRAM_TRANSFORM_UNREPRESENTABLE",
                "Histogram Fit transform is outside the canonical Figure view",
            )

    bindings: dict[AxisSourceRef, SourceViewBinding] = {}
    if selection is not None:
        for term in selection.terms:
            if not isinstance(term, IndexSelection):
                return _needs(
                    "HISTOGRAM_SELECTION_UNREPRESENTABLE",
                    "Histogram Figure can replay only committed fixed-index selectors",
                )
            source = AxisSourceRef.tensor(term.axis_id)
            try:
                axis = _tensor_axis(schema, source)
            except KeyError:
                return _needs(
                    "HISTOGRAM_SELECTION_SOURCE_MISSING",
                    "Histogram Fit selected a source absent from the Dataset",
                    source=source,
                )
            if term.index >= axis.size:
                return _needs(
                    "HISTOGRAM_SELECTION_INDEX_INVALID",
                    "Histogram Fit selector lies outside the Dataset source",
                    source=source,
                )
            bindings[source] = SourceViewBinding(
                source,
                AxisViewRole.SELECTED,
                selector=FixedIndex(term.index),
            )

    if reduction is not None:
        if (
            reduction.method not in {ReductionMethod.MEAN, ReductionMethod.SUM}
            or reduction.missing_policy is not MissingPolicy.OMIT_MISSING
            or reduction.validity_policy is not ValidityPolicy.OMIT_INVALID
            or reduction.minimum_valid_count is not None
        ):
            return _needs(
                "HISTOGRAM_REDUCTION_UNREPRESENTABLE",
                "Histogram Fit reduction differs from the canonical display reduction",
            )
        display_method = (
            DisplayReductionMethod.MEAN
            if reduction.method is ReductionMethod.MEAN
            else DisplayReductionMethod.SUM
        )
        for source in reduction.sources:
            if source in bindings:
                return _needs(
                    "HISTOGRAM_TRANSFORM_SOURCE_CONFLICT",
                    "Histogram Fit selects and reduces the same source",
                    source=source,
                )
            bindings[source] = SourceViewBinding(
                source,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(display_method),
            )

    available = set(_dataset_sources(schema))
    for source in histogram.sources:
        if source not in available or source in bindings:
            return _needs(
                "HISTOGRAM_SAMPLE_SOURCE_INVALID",
                "Histogram Fit sample source is absent or already transformed",
                source=source,
            )
        bindings[source] = SourceViewBinding(source, AxisViewRole.SAMPLE)

    if any(
        value is not None
        for value in (
            preferences.x_source,
            preferences.image_x_source,
            preferences.image_y_source,
        )
    ) or (
        preferences.sample_sources
        and tuple(preferences.sample_sources) != tuple(sorted(histogram.sources))
    ):
        return _needs(
            "HISTOGRAM_VIEW_PREFERENCE_MISMATCH",
            "saved Histogram Fit fixes its sample sources",
        )

    group_sources = tuple(
        source for source in result.spec.batch_sources if source not in bindings
    )
    requested_groups = (
        *preferences.facet_sources,
        *preferences.batch_sources,
    )
    if requested_groups:
        if set(requested_groups) != set(group_sources):
            return _needs(
                "FIT_BATCH_VIEW_MISMATCH",
                "Fit batch presentation must cover every unfixed batch source",
            )
        facet_sources = preferences.facet_sources
        batch_sources = preferences.batch_sources
    elif group_sources:
        repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
        facet = repeat_source if repeat_source in group_sources else group_sources[0]
        facet_sources = (facet,)
        batch_sources = tuple(source for source in group_sources if source != facet)
    else:
        facet_sources = ()
        batch_sources = ()
    for source in facet_sources:
        bindings[source] = SourceViewBinding(source, AxisViewRole.FACET)
    for source in batch_sources:
        bindings[source] = SourceViewBinding(source, AxisViewRole.BATCH)

    if preferences.repeat_binding is not None:
        repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
        if bindings.get(repeat_source) != preferences.repeat_binding:
            return _needs(
                "HISTOGRAM_REPEAT_VIEW_MISMATCH",
                "saved Histogram Fit fixes the repeat source role",
                source=repeat_source,
            )

    for axis in (schema.repeat_axis, *schema.cell_schema.data_axes):
        source = AxisSourceRef.tensor(axis.axis_id)
        if source in bindings:
            continue
        if axis.role != SCALAR:
            return _needs(
                "HISTOGRAM_SOURCE_UNRESOLVED",
                "Histogram Fit left an informative tensor source unbound",
                source=source,
            )
        bindings[source] = SourceViewBinding(
            source,
            AxisViewRole.SELECTED,
            selector=FixedIndex(0),
        )

    spec = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        tuple(bindings.values()),
        transform.exact_point_ordinals,
    )
    try:
        validate_view_spec(schema, spec, HISTOGRAM_CONTRACT)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return _needs("CONTRACT_REJECTED", str(exc))
    return ViewSuggestion(
        spec,
        SuggestionStatus.RESOLVED,
        (
            DecisionReason(
                "FIT_HISTOGRAM_AUTHORITY",
                "Histogram view is derived from its committed samples and bins",
            ),
        ),
    )


def suggest_fit_view(
    schema: DatasetSchema,
    result: FitResultBatch,
    preferences: ViewPreferences | None = None,
) -> ViewSuggestion:
    """Suggest the raw Figure projection for an already bound Fit result."""

    if not isinstance(schema, DatasetSchema):
        raise TypeError("schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if result.source_ref.schema_fingerprint != schema.fingerprint:
        return _needs(
            "FIT_SOURCE_SCHEMA_MISMATCH",
            "fit result and displayed source use different schemas",
        )
    transform = result.spec.committed_transform
    preferences = ViewPreferences() if preferences is None else preferences
    if not isinstance(preferences, ViewPreferences):
        raise TypeError("preferences must be ViewPreferences or None")
    if transform.spec.operations and isinstance(
        transform.spec.operations[-1],
        HistogramSpec,
    ):
        return _suggest_histogram_fit_view(schema, result, preferences)
    available = set(_dataset_sources(schema))
    fit_sources = result.spec.independent_sources
    if any(source not in available for source in fit_sources):
        return _needs(
            "FIT_SOURCE_UNAVAILABLE",
            "a fitted source is absent from the DatasetSchema source vocabulary",
        )
    if len(fit_sources) not in (1, 2):
        return _needs(
            "FIT_ARITY_UNSUPPORTED",
            "the current figure surface supports one- and two-source fit models",
        )
    batch_sources = result.spec.batch_sources
    repeated_point_group = next(
        (
            source
            for source in batch_sources
            if source.kind == AxisSourceRef.POINT_COORDINATE
            and any(
                len(members) > 1
                for members in result.point_groups.group_member_ordinals
            )
        ),
        None,
    )
    if repeated_point_group is not None and any(
        source.kind == AxisSourceRef.TENSOR for source in fit_sources
    ):
        return _needs(
            "FIT_POINT_GROUP_REDUCTION_REQUIRED",
            "a repeated point-coordinate batch needs an explicit PointRows "
            "reduction before it can be shown as one Figure cell",
            source=repeated_point_group,
        )
    requested_groups = {
        *preferences.batch_sources,
        *preferences.facet_sources,
    }
    if requested_groups and requested_groups != set(batch_sources):
        return _needs(
            "FIT_BATCH_VIEW_MISMATCH",
            "Fit batch presentation must cover every authoritative batch source",
        )
    if not requested_groups and batch_sources:
        repeat_source = AxisSourceRef.tensor(schema.repeat_axis.axis_id)
        facet_source = (
            repeat_source if repeat_source in batch_sources else batch_sources[0]
        )
        preferences = replace(
            preferences,
            facet_sources=(facet_source,),
            batch_sources=tuple(
                source for source in batch_sources if source != facet_source
            ),
        )
    if len(fit_sources) == 1:
        if preferences.x_source not in (None, fit_sources[0]):
            return _needs(
                "FIT_SOURCE_VIEW_MISMATCH",
                "the requested X source differs from the fitted source",
                source=preferences.x_source,
            )
        preferences = replace(preferences, x_source=fit_sources[0])
        intent = ViewIntent.CURVE
        contract = CURVE_CONTRACT
    else:
        if preferences.image_x_source not in (None, fit_sources[0]):
            return _needs(
                "FIT_X_SOURCE_VIEW_MISMATCH",
                "the requested IMAGE_X source differs from the fitted source",
                source=preferences.image_x_source,
            )
        if preferences.image_y_source not in (None, fit_sources[1]):
            return _needs(
                "FIT_Y_SOURCE_VIEW_MISMATCH",
                "the requested IMAGE_Y source differs from the fitted source",
                source=preferences.image_y_source,
            )
        preferences = replace(
            preferences,
            image_x_source=fit_sources[0],
            image_y_source=fit_sources[1],
        )
        intent = ViewIntent.IMAGE
        contract = IMAGE_CONTRACT
    suggestion = _suggest_view(
        schema,
        intent,
        transform.exact_point_ordinals,
        preferences,
        contract=contract,
    )
    if suggestion.spec is None:
        return suggestion
    allowed_batch_roles = {AxisViewRole.BATCH, AxisViewRole.FACET}
    for source, axis in zip(
        result.spec.batch_sources,
        result.batch_axis_specs,
        strict=True,
    ):
        if suggestion.spec.binding(source).role not in allowed_batch_roles:
            return _needs(
                "FIT_BATCH_SOURCE_NOT_IDENTIFIED",
                f"fit batch source {axis.name} is not visible or selected",
                source=source,
            )
    return suggestion


__all__ = ["suggest_fit_view", "suggest_view"]
