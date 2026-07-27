"""Headless projection between bound Fit requests and their text editor.

``zlc_data`` remains the sole owner of model, axis, and constraint semantics.
This module exposes a small, reversible authoring value for the Figure UI and
rebuilds a validated ``FitSpec`` from the one visible arguments line.  It
contains no Qt, repository, execution, display selection, or persistence
authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import (
    AxisId,
    BoundFit,
    DataTransformSpec,
    FitNumericPolicy,
    FitSpec,
    HISTOGRAM_BIN_AXIS_ID,
    HistogramSpec,
    IndexSelection,
    MissingPolicy,
    ReductionMethod,
    ReductionSpec,
    Selection,
    ValidityPolicy,
    bind_fit,
    commit_transform,
    fit_model_catalog,
    fit_spec_for,
    suggest_fit_draft,
)

from ._fit_arguments import format_fit_arguments, parse_fit_arguments
from .authority import describe_authoritative_transform
from .data_figure import DataFigure
from .figure import (
    AxisViewRole,
    DisplayReductionMethod,
    FixedIndex,
    LatestNonempty,
    ViewIntent,
)
from .histogram_display import HistogramBinProjection


@dataclass(frozen=True, slots=True)
class FitAuthoringOption:
    """Small presentation projection of a worker-bound Fit request."""

    spec: FitSpec
    display_name: str
    parameter_names: tuple[str, ...]
    argument_text: str
    fit_axis_roles: tuple[object, ...]
    batch_axis_sizes: tuple[tuple[object, int], ...]
    axis_summary: str
    authority_summary: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, FitSpec):
            raise TypeError("Fit authoring option requires FitSpec")
        if not self.display_name:
            raise ValueError("Fit authoring display_name must be non-empty")
        if (
            not self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
            or any(
                not isinstance(name, str) or not name.isidentifier()
                for name in self.parameter_names
            )
        ):
            raise ValueError(
                "Fit authoring parameter names must be unique identifiers"
            )
        if not isinstance(self.argument_text, str):
            raise TypeError("Fit authoring argument_text must be text")
        # The prefilled text is part of the reversible presentation contract.
        # Validate it here so an invalid option never reaches either Qt host.
        parse_fit_arguments(self.argument_text, self.parameter_names)
        if len(self.fit_axis_roles) != len(self.spec.fit_axis_ids):
            raise ValueError("Fit authoring roles differ from its fit axes")
        if tuple(axis_id for axis_id, _size in self.batch_axis_sizes) != (
            self.spec.batch_axis_ids
        ) or any(size <= 0 for _axis_id, size in self.batch_axis_sizes):
            raise ValueError("Fit authoring batch sizes differ from its batch axes")
        if not self.axis_summary or not self.authority_summary:
            raise ValueError("Fit authoring summaries must be non-empty")


@dataclass(frozen=True, slots=True)
class FitAuthoringDraft:
    """One reversible model/arguments draft shared by every Figure surface."""

    selected_model_id: str
    arguments_by_model: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        selected = str(self.selected_model_id).strip()
        arguments = tuple(self.arguments_by_model)
        if not selected:
            raise ValueError("Fit draft selected_model_id must be non-empty")
        if not arguments or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or not isinstance(item[1], str)
            for item in arguments
        ):
            raise ValueError(
                "Fit draft arguments must contain model-id/text pairs"
            )
        model_ids = tuple(model_id for model_id, _text in arguments)
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("Fit draft model ids must be unique")
        if selected not in model_ids:
            raise ValueError("Fit draft selected model has no arguments entry")
        object.__setattr__(self, "selected_model_id", selected)
        object.__setattr__(self, "arguments_by_model", arguments)

    def arguments_for(self, model_id: str) -> str:
        identity = str(model_id)
        for candidate, arguments in self.arguments_by_model:
            if candidate == identity:
                return arguments
        raise KeyError(identity)


def reconcile_fit_authoring_draft(
    options: tuple[FitAuthoringOption, ...],
    previous: FitAuthoringDraft | None = None,
    *,
    selected_model: str | None = None,
) -> FitAuthoringDraft:
    """Reconcile one shared draft against the currently prepared models."""

    prepared = tuple(options)
    if not prepared or any(
        not isinstance(option, FitAuthoringOption) for option in prepared
    ):
        raise ValueError("Fit draft reconciliation requires authoring options")
    model_ids = tuple(option.spec.model_id for option in prepared)
    if len(model_ids) != len(set(model_ids)):
        raise ValueError("Fit authoring options contain duplicate model ids")
    if previous is not None and not isinstance(previous, FitAuthoringDraft):
        raise TypeError("previous must be FitAuthoringDraft or None")
    if selected_model is not None and selected_model not in model_ids:
        raise ValueError("selected_model is not present in Fit options")
    previous_arguments = (
        {} if previous is None else dict(previous.arguments_by_model)
    )
    selected = (
        selected_model
        if selected_model is not None
        else previous.selected_model_id
        if previous is not None and previous.selected_model_id in model_ids
        else model_ids[0]
    )
    return FitAuthoringDraft(
        selected,
        tuple(
            (
                option.spec.model_id,
                previous_arguments.get(
                    option.spec.model_id,
                    option.argument_text,
                ),
            )
            for option in prepared
        ),
    )


def fit_projection_metadata(
    figure: DataFigure,
    intent: ViewIntent,
) -> tuple[tuple[AxisId, ...], tuple[tuple[AxisId, AxisViewRole], ...]]:
    """Project one Figure's declared view roles into exact Fit axes.

    Both DataFigure windows and embedded TaskConsole panels call this owner;
    neither GUI shell is allowed to reinterpret rank, shape, X/Y, or batch
    roles for itself.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("fit projection requires DataFigure")
    if not isinstance(intent, ViewIntent):
        raise TypeError("fit projection requires ViewIntent")
    if len(figure.document.layers) != 1:
        raise ValueError("Fit projection requires exactly one Figure layer")
    layer = figure.document.layers[0]
    roles = tuple(
        sorted(
            (
                (binding.axis_id, binding.role)
                for binding in layer.view.axis_bindings
            ),
            key=lambda item: item[0].value,
        )
    )
    if intent is ViewIntent.CURVE:
        fit_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.X
        )
    elif intent is ViewIntent.IMAGE:
        x_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_X
        )
        y_axes = tuple(
            axis_id for axis_id, role in roles if role is AxisViewRole.IMAGE_Y
        )
        fit_axes = (*x_axes, *y_axes)
    elif intent is ViewIntent.HISTOGRAM:
        fit_axes = (HISTOGRAM_BIN_AXIS_ID,)
    else:
        fit_axes = ()
    expected = (
        1
        if intent is ViewIntent.CURVE
        else 2
        if intent is ViewIntent.IMAGE
        else 1
        if intent is ViewIntent.HISTOGRAM
        else 0
    )
    if len(fit_axes) != expected:
        raise ValueError("typed figure has ambiguous fitted display axes")
    return fit_axes, roles


def validate_fit_authoring_options(
    options: tuple[FitAuthoringOption, ...],
    *,
    fit_axis_ids: tuple[AxisId, ...],
    axis_roles: tuple[tuple[AxisId, AxisViewRole], ...],
    selection: Selection | None,
    allow_prepared_transform: bool = False,
) -> tuple[FitAuthoringOption, ...]:
    """Keep only results that can map back onto the exact visible Figure."""

    prepared_options = tuple(options)
    if not prepared_options or any(
        not isinstance(option, FitAuthoringOption)
        for option in prepared_options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    if any(not isinstance(axis_id, AxisId) for axis_id in fit_axis_ids):
        raise TypeError("fit_axis_ids must contain AxisId values")
    if any(
        not isinstance(axis_id, AxisId)
        or not isinstance(role, AxisViewRole)
        for axis_id, role in axis_roles
    ):
        raise TypeError("axis_roles must contain AxisId/AxisViewRole pairs")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("selection must be Selection or None")

    role_by_axis = dict(axis_roles)
    accepted_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    prepared = []
    for option in prepared_options:
        if option.spec.fit_axis_ids != fit_axis_ids:
            continue
        batch_sizes = dict(option.batch_axis_sizes)

        def batch_axis_is_replayable(axis_id: AxisId) -> bool:
            role = role_by_axis.get(axis_id)
            if role in accepted_batch_roles:
                return True
            return bool(
                role is AxisViewRole.REDUCED
                and batch_sizes[axis_id] == 1
            )

        if any(
            not batch_axis_is_replayable(axis_id)
            for axis_id in option.spec.batch_axis_ids
        ):
            continue
        transform = option.spec.committed_transform
        if selection is None:
            if transform is not None and not allow_prepared_transform:
                continue
        else:
            if transform is None:
                continue
            if tuple(transform.spec.operations) != (selection,):
                continue
        prepared.append(option)
    if not prepared:
        raise ValueError(
            "the visible panel cannot map an authoritative Fit result without "
            "reducing or guessing a named batch axis"
        )
    return tuple(prepared)


def histogram_fit_transform(
    figure: DataFigure,
    projection: HistogramBinProjection,
):
    """Commit the exact displayed samples/bins without promoting display state.

    The Figure view remains presentation-only.  This function copies only its
    explicit named selection/reduction intent and the already-painted bin
    edges into a new authoritative transform when the operator asks to Fit.
    """

    if not isinstance(projection, HistogramBinProjection):
        raise TypeError("Histogram Fit requires its exact visible bin projection")
    layer = figure.document.layers[0]
    evaluated_layer = figure.evaluated.layers[0]
    if layer.layer_id != evaluated_layer.layer_id:
        raise ValueError("Histogram Figure layer identity changed during Fit preparation")
    view = layer.view
    fixed_by_axis: dict[AxisId, int] = {}
    resolution_by_axis = {
        resolution.axis_id: resolution.index
        for resolution in evaluated_layer.resolutions
    }
    for binding in view.axis_bindings:
        if binding.role not in (AxisViewRole.SELECTED, AxisViewRole.SLIDER):
            continue
        selector = binding.selector
        if isinstance(selector, FixedIndex):
            fixed_by_axis[binding.axis_id] = selector.index
        elif isinstance(selector, LatestNonempty):
            try:
                fixed_by_axis[binding.axis_id] = resolution_by_axis[binding.axis_id]
            except KeyError as exc:
                raise ValueError(
                    f"Histogram Fit cannot resolve latest index for {binding.axis_id}"
                ) from exc
        else:  # pragma: no cover - ViewSpec owns the closed selector union.
            raise TypeError("Histogram Figure has an unsupported selector")

    operations: list[object] = []
    for display_selection in view.display_selections:
        retained = tuple(
            term
            for term in display_selection.terms
            if term.axis_id not in fixed_by_axis
        )
        if retained:
            operations.append(Selection(retained))
    if fixed_by_axis:
        operations.append(
            Selection(
                tuple(
                    IndexSelection(axis_id, index)
                    for axis_id, index in fixed_by_axis.items()
                )
            )
        )

    reduced = tuple(
        binding
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.REDUCED
    )
    if reduced:
        methods = {binding.reduction.method for binding in reduced}
        if len(methods) != 1:
            raise ValueError("Histogram Fit reductions must share one method")
        method = next(iter(methods))
        operations.append(
            ReductionSpec(
                tuple(binding.axis_id for binding in reduced),
                ReductionMethod.MEAN
                if method is DisplayReductionMethod.MEAN
                else ReductionMethod.SUM,
                missing_policy=MissingPolicy.OMIT_MISSING,
                validity_policy=ValidityPolicy.OMIT_INVALID,
            )
        )

    sample_axis_ids = tuple(
        binding.axis_id
        for binding in view.axis_bindings
        if binding.role is AxisViewRole.SAMPLE
    )
    operations.append(
        HistogramSpec(
            sample_axis_ids,
            tuple(float(value) for value in projection.bin_edges),
        )
    )
    snapshot = figure.datasets.resolve(layer.dataset_id)
    return commit_transform(
        snapshot.block.schema,
        DataTransformSpec(tuple(operations)),
    )


def prepare_fit_authoring_options(
    figure: DataFigure,
    selection: Selection | None,
    *,
    seed_spec: FitSpec | None = None,
    histogram_projection: HistogramBinProjection | None = None,
) -> tuple[FitAuthoringOption, ...]:
    """Prepare every compatible model for one exact authored Figure.

    This is the single Figure-to-Fit authoring seam used by embedded panels and
    standalone DataFigure windows.  It derives axes only from the authored
    ViewSpec, preserves an explicit compatible seed, and never consults rank,
    shape, viewport limits, or display reduction as analysis authority.
    """

    if not isinstance(figure, DataFigure):
        raise TypeError("fit preparation requires DataFigure")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("fit selection must be Selection or None")
    if seed_spec is not None and not isinstance(seed_spec, FitSpec):
        raise TypeError("seed_spec must be FitSpec or None")
    if len(figure.document.layers) != 1 or len(figure.datasets.entries) != 1:
        raise ValueError("Figure Fit requires exactly one dataset layer")
    layer = figure.document.layers[0]
    intent = layer.view.intent
    if intent not in (ViewIntent.CURVE, ViewIntent.IMAGE, ViewIntent.HISTOGRAM):
        raise ValueError("Fit is available only for curve, image, and histogram Figures")
    if intent is ViewIntent.HISTOGRAM:
        if selection is not None:
            raise ValueError("Histogram Fit authority comes from its named sample view")
        if histogram_projection is None:
            raise ValueError("Histogram Fit requires the exact painted bin projection")
    elif histogram_projection is not None:
        raise ValueError("only a Histogram Figure accepts a bin projection")

    fit_axis_ids, axis_roles = fit_projection_metadata(figure, intent)
    snapshot = figure.datasets.resolve(layer.dataset_id)
    schema = snapshot.block.schema
    seed_matches_schema = bool(
        seed_spec is not None
        and seed_spec.input_schema_fingerprint == schema.fingerprint
    )
    seed_matches_authority = False
    if seed_matches_schema and seed_spec is not None:
        transform = seed_spec.committed_transform
        if selection is None:
            # A saved non-Selection transform has no selector representation,
            # so opening the Figure must retain it exactly.  A single saved
            # Selection is different: once the author explicitly chooses full
            # range, ``None`` means remove that Selection rather than silently
            # keeping yesterday's authority behind an empty selector.
            operations = () if transform is None else tuple(transform.spec.operations)
            seed_matches_authority = not (
                len(operations) == 1
                and isinstance(operations[0], Selection)
            )
        elif transform is not None:
            seed_matches_authority = tuple(transform.spec.operations) == (selection,)

    options = []
    histogram_transform = (
        histogram_fit_transform(figure, histogram_projection)
        if intent is ViewIntent.HISTOGRAM
        else None
    )
    catalog = fit_model_catalog()
    if intent is ViewIntent.HISTOGRAM:
        preferred = ("bimodal_gaussian", "histogram_gaussian")
        catalog = tuple(
            sorted(
                catalog,
                key=lambda definition: (
                    preferred.index(definition.model_id)
                    if definition.model_id in preferred
                    else len(preferred),
                    definition.model_id,
                ),
            )
        )
    for definition in catalog:
        same_seed_model = bool(
            seed_spec is not None
            and seed_spec.model_id == definition.model_id
        )
        try:
            if intent is ViewIntent.HISTOGRAM:
                if (
                    same_seed_model
                    and seed_matches_schema
                    and seed_spec is not None
                    and seed_spec.committed_transform == histogram_transform
                ):
                    bound = bind_fit(seed_spec, schema)
                else:
                    bound = bind_fit(
                        fit_spec_for(
                            schema,
                            definition.model_id,
                            committed_transform=histogram_transform,
                            fit_axis_ids=(HISTOGRAM_BIN_AXIS_ID,),
                            constraints=(
                                seed_spec.constraints if same_seed_model else ()
                            ),
                            numeric_policy=(
                                seed_spec.numeric_policy
                                if same_seed_model
                                else FitNumericPolicy()
                            ),
                        ),
                        schema,
                    )
            elif same_seed_model and seed_matches_authority:
                bound = bind_fit(seed_spec, schema)
            else:
                bound = suggest_fit_draft(
                    schema,
                    definition.model_id,
                    fit_axis_ids=fit_axis_ids,
                    selection=selection,
                    constraints=(
                        seed_spec.constraints if same_seed_model else ()
                    ),
                    numeric_policy=(
                        seed_spec.numeric_policy
                        if same_seed_model
                        else FitNumericPolicy()
                    ),
                )
        except (TypeError, ValueError):
            continue
        options.append(fit_authoring_option(bound))
    if not options:
        raise ValueError("the Figure's declared axes admit no Fit model")
    return validate_fit_authoring_options(
        tuple(options),
        fit_axis_ids=fit_axis_ids,
        axis_roles=axis_roles,
        selection=selection,
        allow_prepared_transform=True,
    )


def fit_axis_summary(bound: BoundFit) -> str:
    """Describe the exact named fit/batch-axis split without reducing an axis."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")

    def describe(axis_id) -> str:
        axis = bound.effective_schema.axis(axis_id)
        unit = f" {axis.unit}" if axis.unit else ""
        return (
            f"{axis.name} ({axis.axis_id}) "
            f"[{axis.role.value}; size={axis.size}]{unit}"
        )

    fit_axes = ", ".join(describe(axis_id) for axis_id in bound.spec.fit_axis_ids)
    batch_axes = ", ".join(
        describe(axis_id) for axis_id in bound.spec.batch_axis_ids
    )
    return f"fit axes: {fit_axes} · batch axes: {batch_axes or 'none'}"


def fit_authority_summary(bound: BoundFit) -> str:
    """Describe the immutable transform authority carried by one bound Fit."""

    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    transform = bound.spec.committed_transform
    return describe_authoritative_transform(
        None if transform is None else transform.spec
    )


def fit_authoring_option(bound: BoundFit) -> FitAuthoringOption:
    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")
    axis_summary = fit_axis_summary(bound)
    authority_summary = fit_authority_summary(bound)
    parameter_names = tuple(
        parameter.name for parameter in bound.parameter_definitions
    )
    argument_text = format_fit_arguments(
        bound.spec.constraints,
        parameter_names,
    )
    return FitAuthoringOption(
        bound.spec,
        bound.model.display_name,
        parameter_names,
        argument_text,
        tuple(
            bound.effective_schema.axis(axis_id).role
            for axis_id in bound.spec.fit_axis_ids
        ),
        tuple(
            (axis_id, bound.effective_schema.axis(axis_id).size)
            for axis_id in bound.spec.batch_axis_ids
        ),
        axis_summary,
        authority_summary,
    )


def fit_spec_from_arguments(
    option: FitAuthoringOption,
    arguments: str,
) -> FitSpec:
    """Parse and domain-validate one authority-bearing Fit request.

    No current viewport, display reduction, or selector value is consulted.
    Empty text means automatic model initialization and domains.
    """

    if not isinstance(option, FitAuthoringOption):
        raise TypeError("option must be FitAuthoringOption")
    constraints = parse_fit_arguments(arguments, option.parameter_names)
    return replace(option.spec, constraints=constraints)


__all__ = [
    "FitAuthoringDraft",
    "FitAuthoringOption",
    "fit_authoring_option",
    "fit_authority_summary",
    "fit_axis_summary",
    "fit_projection_metadata",
    "prepare_fit_authoring_options",
    "reconcile_fit_authoring_draft",
    "fit_spec_from_arguments",
    "histogram_fit_transform",
    "validate_fit_authoring_options",
]
