"""Exact formal Fit projection for one frozen HISTOGRAM Figure."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from zlc_data import (
    AxisSourceRef,
    FitBatchStatus,
    FitResultBatch,
    HISTOGRAM_BIN,
    HistogramSpec,
    ReductionSpec,
    Selection,
)
from zlc_data.fit import (
    evaluate_fit_model_components,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_text

from .figure import AxisViewRole, EvaluatedHistogram, ViewIntent
from .figure.contract import _fit_authority_selection
from .fit_curve_projection import _panel_prediction_point_share
from .fit_projection import fit_batch_storage_index
from .histogram_display import (
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramDisplayState,
)
from .render import HistogramFitOverlay


def _histogram_fit_display_state(
    figure,
    state: HistogramDisplayState | FacetedHistogramDisplayState,
    result: FitResultBatch,
) -> tuple[
    HistogramDisplayState | FacetedHistogramDisplayState,
    tuple[float, float],
]:
    """Seed exact saved bins into the canonical Histogram display state."""

    from .data_figure import DataFigure

    if not isinstance(figure, DataFigure):
        raise TypeError("histogram Fit display requires DataFigure")
    if not isinstance(result, FitResultBatch):
        raise TypeError("histogram Fit display requires FitResultBatch")
    operations = tuple(result.spec.committed_transform.spec.operations)
    if not operations or not isinstance(operations[-1], HistogramSpec):
        raise ValueError("histogram Fit result lacks an authoritative bin projection")
    edges = np.asarray(operations[-1].bin_edges, dtype=np.dtype("<f8"))
    cells = figure.evaluated.layers[0].cells
    samples = tuple(
        series.data.samples
        for cell in cells
        for series in cell.series
        if isinstance(series.data, EvaluatedHistogram)
    )
    if not samples or len(samples) != sum(len(cell.series) for cell in cells):
        raise ValueError("histogram Fit Figure contains another evaluated data kind")
    all_boolean = all(values.dtype.kind == "b" for values in samples)
    if all_boolean:
        if not np.array_equal(
            edges,
            np.asarray((-0.5, 0.5, 1.5), dtype=np.dtype("<f8")),
        ):
            raise ValueError("boolean histogram Fit has noncanonical bin edges")
        bin_count = (
            state.display.bin_count
            if isinstance(state, FacetedHistogramDisplayState)
            else state.bin_count
        )
    else:
        bin_count = len(edges) - 1
    if isinstance(state, FacetedHistogramDisplayState):
        state = replace(state, display=replace(state.display, bin_count=bin_count))
    elif isinstance(state, HistogramDisplayState):
        state = replace(state, bin_count=bin_count)
    else:
        raise TypeError("histogram Fit requires HistogramDisplayState")
    return state, (float(edges[0]), float(edges[-1]))


def _histogram_fit_presentation(
    figure,
    result: FitResultBatch,
    *,
    result_identity: str,
    display_state: HistogramDisplayState | FacetedHistogramDisplayState,
    bin_projection: HistogramBinProjection | None = None,
    maximum_prediction_points: int | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[
    HistogramDisplayState | FacetedHistogramDisplayState,
    HistogramBinProjection,
    tuple[tuple[HistogramFitOverlay, ...], ...],
]:
    """Materialize exact saved bins and overlays for every canonical cell."""

    from .data_figure import DataFigure

    if not isinstance(figure, DataFigure):
        raise TypeError("histogram Fit projection requires DataFigure")
    if not isinstance(result, FitResultBatch):
        raise TypeError("histogram Fit projection requires FitResultBatch")
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError("check_cancelled must be callable or None")
    identity = canonical_text(result_identity, "histogram Fit result identity")
    document = figure.document
    evaluated = figure.evaluated
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
        or len(document.layers) != 1
        or len(evaluated.layers) != 1
        or len(evaluated.inputs) != 1
    ):
        raise ValueError("histogram Fit requires one exact document layer/input")
    document_layer = document.layers[0]
    layer = evaluated.layers[0]
    if (
        document_layer.view.intent is not ViewIntent.HISTOGRAM
        or layer.layer_id != document_layer.layer_id
        or layer.dataset_id != document_layer.dataset_id
        or not layer.cells
    ):
        raise ValueError("histogram Fit layer differs from its document")
    evaluated_input = evaluated.inputs[0]
    if (
        evaluated_input.dataset_id != layer.dataset_id
        or result.source_ref != evaluated_input.ref
    ):
        raise ValueError("histogram Fit result belongs to another source revision")
    source_schema = figure.datasets.resolve(document_layer.dataset_id).block.schema
    validate_fit_result_source_binding(result, evaluated_input.ref, source_schema)
    if len(result.fit_axis_specs) != 1 or result.fit_axis_specs[0].role != HISTOGRAM_BIN:
        raise ValueError("histogram Fit requires one declared HISTOGRAM_BIN axis")
    operations = tuple(result.spec.committed_transform.spec.operations)
    if not operations or not isinstance(operations[-1], HistogramSpec):
        raise ValueError("histogram Fit result lacks an authoritative bin projection")
    histogram = operations[-1]
    display_state, _home = _histogram_fit_display_state(
        figure,
        display_state,
        result,
    )
    display = (
        display_state.display
        if isinstance(display_state, FacetedHistogramDisplayState)
        else display_state
    )
    samples = tuple(
        series.data.samples
        for cell in layer.cells
        for series in cell.series
        if isinstance(series.data, EvaluatedHistogram)
    )
    if bin_projection is None:
        projection = HistogramBinProjection._from_committed_edges(
            samples,
            bins=display.bin_count,
            bin_edges=histogram.bin_edges,
        )
    else:
        if not isinstance(bin_projection, HistogramBinProjection):
            raise TypeError("bin_projection must be HistogramBinProjection or None")
        if (
            len(bin_projection.series_samples) != len(samples)
            or any(
                projected is not sample
                for projected, sample in zip(
                    bin_projection.series_samples,
                    samples,
                    strict=True,
                )
            )
            or bin_projection.requested_bin_count != display.bin_count
            or not np.array_equal(
                bin_projection.bin_edges,
                np.asarray(histogram.bin_edges, dtype=np.dtype("<f8")),
            )
        ):
            raise ValueError(
                "supplied histogram projection differs from the committed Fit bins"
            )
        projection = bin_projection
    if _fit_authority_selection(
        source_schema,
        document_layer.view,
        layer.resolutions,
        result,
    ) is not None:
        raise ValueError("histogram Fit cannot carry an independent range ROI")
    fit_axis = result.fit_axis_specs[0]
    if histogram.bin_axis_id != fit_axis.axis_id:
        raise ValueError("histogram Fit axis differs from its committed projection")
    centers = (projection.bin_edges[:-1] + projection.bin_edges[1:]) * 0.5
    if fit_axis.size != len(centers) or any(
        fit_axis.coordinate_at(index) != float(value)
        for index, value in enumerate(centers)
    ):
        raise ValueError("histogram Fit coordinates differ from painted bin centres")
    sample_sources = {
        binding.source
        for binding in document_layer.view.source_bindings
        if binding.role is AxisViewRole.SAMPLE
    }
    if set(histogram.sources) != sample_sources:
        raise ValueError("histogram Fit sample axes differ from the painted Figure")
    if sum(len(cell.series) for cell in layer.cells) != len(projection.bin_counts):
        raise ValueError("histogram Fit series differ from painted bars")

    projected_sources = set(histogram.sources)
    for operation in operations[:-1]:
        if isinstance(operation, Selection):
            projected_sources.update(
                AxisSourceRef.tensor(term.axis_id) for term in operation.terms
            )
        elif isinstance(operation, ReductionSpec):
            projected_sources.update(operation.sources)

    used_storage: set[int] = set()
    projected_cells = []
    for cell in layer.cells:
        projected_series = []
        for series in cell.series:
            if check_cancelled is not None:
                check_cancelled()
            if not isinstance(series.data, EvaluatedHistogram):
                raise ValueError("histogram Fit projection found a non-histogram series")
            storage_index = fit_batch_storage_index(
                result,
                layer,
                cell,
                series,
                projected_sources=tuple(sorted(projected_sources)),
            )
            if storage_index is None:
                status = None
                diagnostic = "NOT_PRESENT"
            else:
                if storage_index in used_storage:
                    raise ValueError("two histogram series map to one Fit storage row")
                used_storage.add(storage_index)
                status = result.statuses[storage_index]
                diagnostic = status.value
                if result.errors[storage_index]:
                    diagnostic = f"{diagnostic}: {result.errors[storage_index]}"
            projected_series.append((series, storage_index, status, diagnostic))
        projected_cells.append(tuple(projected_series))

    converged_count = sum(
        status is FitBatchStatus.CONVERGED
        for cell in projected_cells
        for _series, _storage, status, _diagnostic in cell
    )
    component_count = 0
    if converged_count:
        first_converged_storage = next(
            storage
            for cell in projected_cells
            for _series, storage, status, _diagnostic in cell
            if status is FitBatchStatus.CONVERGED
        )
        assert first_converged_storage is not None
        # Ask the model owner for its visual-component arity without creating
        # any prediction vertices.  This avoids copying a model-id/component
        # lookup into the presentation layer while still allocating the real
        # panel budget before evaluating a plotted coordinate vector.
        component_count = len(
            evaluate_fit_model_components(
                result.spec.model_id,
                (np.empty((0,), dtype=np.dtype("<f8")),),
                result.parameter_values[first_converged_storage],
            )
        )
        if component_count < 1:
            raise ValueError("converged histogram Fit exposed no visual components")
    point_share = _panel_prediction_point_share(
        maximum_prediction_points,
        converged_count * component_count,
    )
    point_count = (
        0
        if converged_count == 0
        else 400
        if point_share is None
        else min(400, point_share)
    )
    coordinates = (
        np.empty((0,), dtype=np.dtype("<f8"))
        if point_count == 0
        else np.linspace(
            float(projection.bin_edges[0]),
            float(projection.bin_edges[-1]),
            point_count,
            dtype=np.dtype("<f8"),
        )
    )

    overlays_by_cell = []
    for projected_series in projected_cells:
        cell_overlays = []
        for series, storage_index, status, diagnostic in projected_series:
            if check_cancelled is not None:
                check_cancelled()
            if status is FitBatchStatus.CONVERGED:
                assert storage_index is not None
                overlay_coordinates = coordinates
                components = evaluate_fit_model_components(
                    result.spec.model_id,
                    (coordinates,),
                    result.parameter_values[storage_index],
                )
                diagnostic = ""
            else:
                overlay_coordinates = np.empty((0,), dtype=np.dtype("<f8"))
                components = ()
            cell_overlays.append(
                HistogramFitOverlay(
                    result.source_ref,
                    identity,
                    series.batch_address,
                    storage_index,
                    status,
                    diagnostic,
                    overlay_coordinates,
                    components,
                )
            )
        overlays_by_cell.append(tuple(cell_overlays))
    return display_state, projection, tuple(overlays_by_cell)


__all__: list[str] = []
