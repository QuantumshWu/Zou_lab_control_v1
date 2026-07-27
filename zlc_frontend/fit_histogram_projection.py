"""Exact formal Fit projections for one frozen HISTOGRAM Figure."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from zlc_data import (
    FitBatchStatus,
    FitResultBatch,
    HISTOGRAM_BIN,
    HistogramSpec,
    ReductionSpec,
    Selection,
    evaluate_fit_model_components,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_text

from .figure import AxisViewRole, EvaluatedHistogram, ViewIntent
from .fit_projection import fit_batch_storage_index
from .histogram_display import HistogramBinProjection
from .fit_editor import histogram_fit_transform
from .render import HistogramFitOverlay


def transient_single_panel_histogram_fit_overlays(
    figure,
    projection: HistogramBinProjection,
    result: FitResultBatch,
    *,
    result_identity: str,
    check_cancelled: Callable[[], None] | None = None,
) -> tuple[HistogramFitOverlay, ...]:
    """Project a standard exact-source Fit result onto the painted histogram."""

    from .data_figure import DataFigure

    if not isinstance(figure, DataFigure):
        raise TypeError("histogram Fit projection requires DataFigure")
    document = figure.document
    evaluated = figure.evaluated
    if not isinstance(projection, HistogramBinProjection):
        raise TypeError("histogram Fit projection requires HistogramBinProjection")
    if not isinstance(result, FitResultBatch):
        raise TypeError("histogram Fit projection requires FitResultBatch")
    if check_cancelled is not None and not callable(check_cancelled):
        raise TypeError("check_cancelled must be callable or None")
    identity = canonical_text(result_identity, "histogram Fit result identity")
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
        or len(layer.cells) != 1
    ):
        raise ValueError("histogram Fit layer differs from its document")
    evaluated_input = evaluated.inputs[0]
    if evaluated_input.dataset_id != layer.dataset_id or result.source_ref != evaluated_input.ref:
        raise ValueError("histogram Fit result belongs to another source revision")
    source_schema = figure.datasets.resolve(document_layer.dataset_id).block.schema
    validate_fit_result_source_binding(result, evaluated_input.ref, source_schema)
    if len(result.fit_axis_specs) != 1 or result.fit_axis_specs[0].role != HISTOGRAM_BIN:
        raise ValueError("histogram Fit requires one declared HISTOGRAM_BIN axis")
    transform = result.spec.committed_transform
    operations = () if transform is None else tuple(transform.spec.operations)
    if not operations or not isinstance(operations[-1], HistogramSpec):
        raise ValueError("histogram Fit result lacks an authoritative bin projection")
    histogram = operations[-1]
    expected_transform = histogram_fit_transform(figure, projection)
    if result.spec.committed_transform != expected_transform:
        raise ValueError(
            "histogram Fit authority differs from the exact painted Figure view"
        )
    if histogram.bin_axis_id != result.fit_axis_specs[0].axis_id:
        raise ValueError("histogram Fit axis differs from its committed projection")
    if not np.array_equal(
        np.asarray(histogram.bin_edges, dtype=np.float64),
        np.asarray(projection.bin_edges, dtype=np.float64),
    ):
        raise ValueError("histogram Fit bin edges differ from the painted bars")
    sample_ids = {
        binding.axis_id
        for binding in document_layer.view.axis_bindings
        if binding.role is AxisViewRole.SAMPLE
    }
    if set(histogram.axis_ids) != sample_ids:
        raise ValueError("histogram Fit sample axes differ from the painted Figure")
    centers = (projection.bin_edges[:-1] + projection.bin_edges[1:]) * 0.5
    fit_axis = result.fit_axis_specs[0]
    if fit_axis.size != len(centers) or any(
        fit_axis.coordinate_at(index) != float(value)
        for index, value in enumerate(centers)
    ):
        raise ValueError("histogram Fit coordinates differ from painted bin centres")

    cell = layer.cells[0]
    if len(cell.series) != len(projection.bin_counts):
        raise ValueError("histogram Fit series differ from painted bars")
    coordinates = np.linspace(
        float(projection.bin_edges[0]),
        float(projection.bin_edges[-1]),
        400,
        dtype=np.dtype("<f8"),
    )
    overlays = []
    projected_axis_ids = set(histogram.axis_ids)
    for operation in operations[:-1]:
        if isinstance(operation, Selection):
            projected_axis_ids.update(term.axis_id for term in operation.terms)
        elif isinstance(operation, ReductionSpec):
            projected_axis_ids.update(operation.axis_ids)
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
            projected_axis_ids=tuple(
                sorted(projected_axis_ids, key=lambda item: item.value)
            ),
        )
        if storage_index is None:
            status = None
            diagnostic = "NOT_PRESENT"
            overlay_coordinates = np.empty((0,), dtype=np.dtype("<f8"))
            components = ()
        else:
            status = result.statuses[storage_index]
            diagnostic = status.value
            if result.errors[storage_index]:
                diagnostic = f"{diagnostic}: {result.errors[storage_index]}"
            if status is FitBatchStatus.CONVERGED:
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
        overlays.append(
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
    return tuple(overlays)


__all__ = ["transient_single_panel_histogram_fit_overlays"]
