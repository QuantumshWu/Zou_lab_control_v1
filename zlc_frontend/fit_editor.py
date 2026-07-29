"""Headless projection between bound Fit requests and their text editor.

``zlc_data`` remains the sole owner of model, axis, and constraint semantics.
This module exposes a small, reversible authoring value for the Figure UI and
rebuilds a validated ``FitSpec`` from the one visible arguments line.  It
contains no Qt, repository, execution, mutable display selection, or
persistence authority.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from zlc_data import (
    SCALAR,
    AxisSourceRef,
    CommittedTransform,
    FitResultBatch,
    FitSpec,
    HISTOGRAM_BIN_AXIS_ID,
    Selection,
)
from zlc_data.fit import (
    BoundFit,
    FitModelDefinition,
    bind_fit,
    fit_model_catalog,
    fit_spec_for,
)

from ._fit_arguments import format_fit_arguments, parse_fit_arguments
from .authority import describe_authoritative_transform
from .data_figure import DataFigure
from .figure.contract import (
    _dataset_sources,
    _fit_authority_selection,
    _fit_transform_from_view,
    _source_coordinate_frame,
    _source_role,
    _source_unit,
)
from .figure import (
    AxisViewRole,
    EvaluatedHistogram,
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


def _fit_sources_for_figure(figure: DataFigure) -> tuple[AxisSourceRef, ...]:
    """Return the exact Fit axes declared by one Figure view."""

    if not isinstance(figure, DataFigure):
        raise TypeError("fit projection requires DataFigure")
    if len(figure.document.layers) != 1:
        raise ValueError("Fit projection requires exactly one Figure layer")
    layer = figure.document.layers[0]
    intent = layer.view.intent
    if intent is ViewIntent.CURVE:
        fit_axes = tuple(
            binding.source
            for binding in layer.view.source_bindings
            if binding.role is AxisViewRole.X
        )
    elif intent is ViewIntent.IMAGE:
        x_axes = tuple(
            binding.source
            for binding in layer.view.source_bindings
            if binding.role is AxisViewRole.IMAGE_X
        )
        y_axes = tuple(
            binding.source
            for binding in layer.view.source_bindings
            if binding.role is AxisViewRole.IMAGE_Y
        )
        fit_axes = (*x_axes, *y_axes)
    elif intent is ViewIntent.HISTOGRAM:
        fit_axes = (AxisSourceRef.tensor(HISTOGRAM_BIN_AXIS_ID),)
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
    return fit_axes


def validate_fit_authoring_options(
    options: tuple[FitAuthoringOption, ...],
    *,
    figure: DataFigure,
    selection: Selection | None,
    histogram_projection: HistogramBinProjection | None = None,
) -> tuple[FitAuthoringOption, ...]:
    """Require every injected option to equal the visible Figure authority."""

    prepared_options = tuple(options)
    if not prepared_options or any(
        not isinstance(option, FitAuthoringOption)
        for option in prepared_options
    ):
        raise ValueError("Fit preparation produced no FitAuthoringOption")
    required = figure_fit_transform(
        figure,
        selection,
        histogram_projection=histogram_projection,
    )
    fit_sources = _fit_sources_for_figure(figure)
    batch_sources = _fit_batch_sources(figure, required, fit_sources)
    for option in prepared_options:
        if (
            option.spec.committed_transform != required
            or option.spec.independent_sources != fit_sources
            or option.spec.batch_sources != batch_sources
        ):
            raise ValueError(
                "Fit option differs from the exact visible Figure authority"
            )
    return prepared_options


def figure_fit_transform(
    figure: DataFigure,
    selection: Selection | None,
    *,
    histogram_projection: HistogramBinProjection | None = None,
) -> CommittedTransform:
    """Translate one evaluated Figure into its sole authoritative Fit transform."""

    if not isinstance(figure, DataFigure):
        raise TypeError("Fit authority requires DataFigure")
    if selection is not None and not isinstance(selection, Selection):
        raise TypeError("Fit selection must be Selection or None")
    if len(figure.document.layers) != 1 or len(figure.evaluated.layers) != 1:
        raise ValueError("Figure Fit requires exactly one evaluated layer")
    layer = figure.document.layers[0]
    evaluated = figure.evaluated.layers[0]
    if layer.layer_id != evaluated.layer_id:
        raise ValueError("Figure layer changed during Fit preparation")
    if layer.view.intent is ViewIntent.HISTOGRAM:
        if selection is not None:
            raise ValueError("Histogram Fit does not accept an independent ROI")
        if not isinstance(histogram_projection, HistogramBinProjection):
            raise TypeError("Histogram Fit requires its exact painted bins")
        samples = tuple(
            series.data.samples
            for cell in evaluated.cells
            for series in cell.series
            if isinstance(series.data, EvaluatedHistogram)
        )
        if (
            len(samples) != sum(len(cell.series) for cell in evaluated.cells)
            or len(samples) != len(histogram_projection.series_samples)
            or any(
                source is not projected
                for source, projected in zip(
                    samples,
                    histogram_projection.series_samples,
                    strict=True,
                )
            )
        ):
            raise ValueError("Histogram bins belong to another Figure projection")
        edges = tuple(float(value) for value in histogram_projection.bin_edges)
    else:
        if histogram_projection is not None:
            raise ValueError("only a Histogram Figure accepts painted bins")
        edges = None
    schema = figure.datasets.resolve(layer.dataset_id).block.schema
    return _fit_transform_from_view(
        schema,
        layer.view,
        evaluated.resolutions,
        independent_selection=selection,
        histogram_bin_edges=edges,
    )


def _fit_selection_from_result(
    figure: DataFigure,
    result: FitResultBatch,
) -> Selection | None:
    layer = figure.document.layers[0]
    evaluated = figure.evaluated.layers[0]
    schema = figure.datasets.resolve(layer.dataset_id).block.schema
    return _fit_authority_selection(
        schema,
        layer.view,
        evaluated.resolutions,
        result,
    )


def _batch_sources_for(
    schema,
    independent_sources: tuple[AxisSourceRef, ...],
    preferred_point_sources: tuple[AxisSourceRef, ...],
) -> tuple[AxisSourceRef, ...]:
    """Preserve every non-independent information source as Fit batch state."""

    independent = set(independent_sources)
    tensor = tuple(
        AxisSourceRef.tensor(axis.axis_id)
        for axis in (schema.repeat_axis, *schema.cell_schema.data_axes)
        if axis.role != SCALAR
        and axis.size > 1
        and AxisSourceRef.tensor(axis.axis_id) not in independent
    )
    available = _dataset_sources(schema)
    preferred = {
        source
        for source in preferred_point_sources
        if schema.point_table.row_count > 1
        and source.kind != AxisSourceRef.TENSOR
        and source not in independent
    }
    point = tuple(
        source
        for source in available
        if source in preferred
    )
    point_independent = any(
        source.kind != AxisSourceRef.TENSOR for source in independent_sources
    )
    if (
        schema.point_table.row_count > 1
        and not point_independent
        and not point
    ):
        point = (AxisSourceRef.point_rows(),)
    return (*tensor, *point)


def _fit_batch_sources(
    figure: DataFigure,
    transform: CommittedTransform,
    independent_sources: tuple[AxisSourceRef, ...],
) -> tuple[AxisSourceRef, ...]:
    view = figure.document.layers[0].view
    preferred_point_sources = tuple(
        binding.source
        for binding in view.source_bindings
        if binding.role in {AxisViewRole.BATCH, AxisViewRole.FACET}
        and binding.source.kind != AxisSourceRef.TENSOR
    )
    return _batch_sources_for(
        transform.effective_output_schema,
        independent_sources,
        preferred_point_sources,
    )


def _fit_model_accepts_declared_axes(
    definition: FitModelDefinition,
    schema,
    sources: tuple[AxisSourceRef, ...],
) -> bool:
    """Filter catalog entries by their complete static axis contract.

    Once this predicate accepts a model, request construction is expected to
    succeed.  Any later exception is therefore an implementation or authored
    request error and must reach the caller rather than masquerading as model
    incompatibility.
    """

    if definition.independent_arity != len(sources):
        return False
    roles = tuple(_source_role(schema, source) for source in sources)
    if any(
        role not in requirement
        for role, requirement in zip(
            roles,
            definition.axis_requirements,
            strict=True,
        )
    ):
        return False
    if definition.require_common_axis_unit and len(
        {_source_unit(schema, source) for source in sources}
    ) != 1:
        return False
    if definition.require_common_coordinate_frame and len(
        {_source_coordinate_frame(schema, source) for source in sources}
    ) != 1:
        return False
    return True


def prepare_fit_authoring_options(
    figure: DataFigure,
    selection: Selection | None,
    *,
    seed_spec: FitSpec | None = None,
    histogram_projection: HistogramBinProjection | None = None,
) -> tuple[FitAuthoringOption, ...]:
    """Prepare every compatible model for one exact authored Figure.

    This is the single Figure-to-Fit authoring seam used by embedded panels and
    standalone DataFigure windows.  At the operator's Fit action it freezes the
    exact named ViewSpec selection/reduction into a data-owned transform; rank,
    shape, viewport limits, and implicit display guesses are never authority.
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

    fit_sources = _fit_sources_for_figure(figure)
    snapshot = figure.datasets.resolve(layer.dataset_id)
    schema = snapshot.block.schema
    transform = figure_fit_transform(
        figure,
        selection,
        histogram_projection=histogram_projection,
    )
    batch_sources = _fit_batch_sources(figure, transform, fit_sources)
    options = []
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
    catalog = tuple(
        definition
        for definition in catalog
        if _fit_model_accepts_declared_axes(
            definition,
            transform.effective_output_schema,
            fit_sources,
        )
    )
    for definition in catalog:
        exact_seed = bool(
            seed_spec is not None
            and seed_spec.model_id == definition.model_id
            and seed_spec.committed_transform == transform
            and seed_spec.independent_sources == fit_sources
            and seed_spec.batch_sources == batch_sources
        )
        if exact_seed:
            assert seed_spec is not None
            bound = bind_fit(seed_spec, schema)
        else:
            bound = bind_fit(
                fit_spec_for(
                    schema,
                    definition.model_id,
                    committed_transform=transform,
                    independent_sources=fit_sources,
                    batch_sources=batch_sources,
                ),
                schema,
            )
        options.append(_fit_authoring_option(bound))
    if not options:
        raise ValueError("the Figure's declared axes admit no Fit model")
    return validate_fit_authoring_options(
        tuple(options),
        figure=figure,
        selection=selection,
        histogram_projection=histogram_projection,
    )


def _fit_authoring_option(bound: BoundFit) -> FitAuthoringOption:
    if not isinstance(bound, BoundFit):
        raise TypeError("bound must be BoundFit")

    def describe(source, axis) -> str:
        unit = f" {axis.unit}" if axis.unit else ""
        source_name = (
            source.kind.lower()
            if source.axis_id is None
            else f"{source.kind.lower()}:{source.axis_id.value}"
        )
        return (
            f"{axis.name} ({source_name}) "
            f"[{axis.role.value}; size={axis.size}]{unit}"
        )

    fit_axes = ", ".join(
        describe(source, axis)
        for source, axis in zip(
            bound.spec.independent_sources,
            bound.fit_axis_specs,
            strict=True,
        )
    )
    batch_axes = ", ".join(
        describe(source, axis)
        for source, axis in zip(
            bound.spec.batch_sources,
            bound.batch_axis_specs,
            strict=True,
        )
    )
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
        f"fit axes: {fit_axes} · batch axes: {batch_axes or 'none'}",
        describe_authoritative_transform(bound.spec.committed_transform.spec),
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
    "figure_fit_transform",
    "prepare_fit_authoring_options",
    "reconcile_fit_authoring_draft",
    "fit_spec_from_arguments",
    "validate_fit_authoring_options",
]
