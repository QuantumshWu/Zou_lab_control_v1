"""Lazy Matplotlib/Agg rendering for frozen :mod:`zlc_frontend` values."""

from __future__ import annotations

import math
from numbers import Number

import numpy as np

from zlc_data import FitBatchStatus, FitResultBatch

from .figure import (
    AxisAddress,
    AxisResolution,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)


def _address_label(items: tuple[AxisAddress, ...] | tuple[AxisResolution, ...]) -> str:
    return ", ".join(f"{item.axis_id.value}={item.coordinate}" for item in items)


def _reduction_label(reductions) -> str:
    labels = []
    for reduction in reductions:
        axes = ",".join(axis_id.value for axis_id in reduction.axis_ids)
        contributors = str(reduction.minimum_contributors)
        if reduction.minimum_contributors != reduction.maximum_contributors:
            contributors = (
                f"{reduction.minimum_contributors}..{reduction.maximum_contributors}"
            )
        labels.append(
            f"{reduction.method.value.lower()}({axes}, n={contributors})"
        )
    return "; ".join(labels)


def _series_label(series, *, include_reductions: bool) -> str | None:
    parts = [_address_label(series.batch_address)]
    if include_reductions:
        parts.append(_reduction_label(series.reductions))
    label = " · ".join(part for part in parts if part)
    return label or None


def _axis_label(axis) -> str:
    return axis.name if axis.unit is None else f"{axis.name} [{axis.unit}]"


def _numeric_centers(axis):
    values = tuple(axis.coordinates)
    if not values or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Number)
        for value in values
    ):
        return None
    centers = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(centers)):
        return None
    if len(centers) > 1:
        delta = np.diff(centers)
        if not (np.all(delta > 0) or np.all(delta < 0)):
            return None
    return centers


def _image_axis(axis):
    centers = _numeric_centers(axis)
    if centers is None:
        positions = np.arange(len(axis.coordinates), dtype=np.float64)
        return (
            np.arange(len(axis.coordinates) + 1, dtype=np.float64) - 0.5,
            positions,
            tuple(str(value) for value in axis.coordinates),
        )
    if len(centers) == 1:
        edges = np.asarray((centers[0] - 0.5, centers[0] + 0.5))
    else:
        middle = (centers[:-1] + centers[1:]) / 2.0
        edges = np.concatenate(
            ((centers[0] - (middle[0] - centers[0]),), middle,
             (centers[-1] + (centers[-1] - middle[-1]),))
        )
    return edges, centers, None


def _batch_storage_index(result, layer, cell, series) -> int | None:
    addresses = (*cell.facet_address, *series.batch_address)
    by_axis = {item.axis_id: item.index for item in addresses}
    if len(by_axis) != len(addresses):
        raise RuntimeError("figure batch/facet addresses contain a duplicate axis")
    for resolution in layer.resolutions:
        incumbent = by_axis.setdefault(resolution.axis_id, resolution.index)
        if incumbent != resolution.index:
            raise RuntimeError("figure address and resolution disagree")
    expected = {axis.axis_id for axis in result.batch_axis_specs}
    extras = set(by_axis) - expected
    if extras:
        raise RuntimeError(f"figure resolved non-batch fit axes: {sorted(map(str, extras))}")
    multi = []
    for axis in result.batch_axis_specs:
        if axis.axis_id in by_axis:
            multi.append(by_axis[axis.axis_id])
        elif axis.size == 1:
            multi.append(0)
        else:
            raise RuntimeError(f"figure does not identify fit batch axis {axis.axis_id}")
    try:
        return result.batch_layout.storage_index(tuple(multi))
    except KeyError:
        # A sparse source may expose a logical gallery hole that has no
        # authoritative fit row.  Keep later storage rows aligned and mark the
        # hole; never substitute a neighbouring fit.
        return None


def _fit_status(axis, result: FitResultBatch, index: int | None) -> bool:
    if index is None:
        message = "NOT_PRESENT"
    else:
        status = result.statuses[index]
        if status is FitBatchStatus.CONVERGED:
            return True
        message = status.value
        if result.errors[index]:
            message = f"{message}: {result.errors[index]}"
    axis.text(
        0.02,
        0.98,
        f"fit {message}",
        transform=axis.transAxes,
        va="top",
        color="crimson",
        fontsize="small",
    )
    return False


def _curve(axis, layer, cell, series_group, fit_result):
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedCurve)
        if np.iscomplexobj(data.values):
            raise ValueError("complex curves require an explicit real-valued display transform")
        x = np.asarray(data.x_axis.coordinates)
        values = np.ma.array(data.values, mask=~data.validity)
        label = _series_label(series, include_reductions=multiple_series)
        axis.plot(x, values, marker="o", linestyle="-", label=label)
        if fit_result is not None:
            index = _batch_storage_index(fit_result, layer, cell, series)
            if _fit_status(axis, fit_result, index):
                coordinates = np.asarray(data.x_axis.coordinates, dtype=np.float64)
                predicted = fit_result.evaluate_batch(index, (coordinates,))
                axis.plot(
                    coordinates,
                    predicted,
                    linestyle="--",
                    label=("fit" if label is None else f"fit {label}"),
                )
    data = series_group[0].data
    axis.set_xlabel(_axis_label(data.x_axis))
    if len(series_group) > 1 or any(series.batch_address for series in series_group):
        axis.legend(fontsize="small")


def _image(axis, figure, layer, cell, series, fit_result):
    data = series.data
    assert isinstance(data, EvaluatedImage)
    if np.iscomplexobj(data.values):
        raise ValueError("complex images require an explicit real-valued display transform")
    x_edges, x_centers, x_labels = _image_axis(data.x_axis)
    y_edges, y_centers, y_labels = _image_axis(data.y_axis)
    values = np.ma.array(data.values, mask=~data.validity)
    mesh = axis.pcolormesh(x_edges, y_edges, values, shading="flat")
    figure.colorbar(mesh, ax=axis)
    if x_labels is not None:
        axis.set_xticks(x_centers, x_labels)
    if y_labels is not None:
        axis.set_yticks(y_centers, y_labels)
    axis.set_xlabel(_axis_label(data.x_axis))
    axis.set_ylabel(_axis_label(data.y_axis))
    if fit_result is not None:
        index = _batch_storage_index(fit_result, layer, cell, series)
        if _fit_status(axis, fit_result, index):
            x_grid, y_grid = np.meshgrid(
                np.asarray(data.x_axis.coordinates, dtype=np.float64),
                np.asarray(data.y_axis.coordinates, dtype=np.float64),
            )
            grids = {
                data.x_axis.axis_id: x_grid,
                data.y_axis.axis_id: y_grid,
            }
            coordinates = tuple(grids[item.axis_id] for item in fit_result.fit_axis_specs)
            predicted = fit_result.evaluate_batch(index, coordinates)
            if np.ptp(predicted) > 0:
                axis.contour(x_grid, y_grid, predicted, colors="white", linewidths=0.8)


def _histogram(axis, series_group):
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedHistogram)
        label = _series_label(series, include_reductions=multiple_series)
        axis.hist(data.samples, bins="auto", histtype="step", label=label)
    if len(series_group) > 1 or any(series.batch_address for series in series_group):
        axis.legend(fontsize="small")


def _meter(axis, series_group):
    axis.set_axis_off()
    lines = []
    multiple_series = len(series_group) > 1
    for series in series_group:
        data = series.data
        assert isinstance(data, EvaluatedMeter)
        label = _series_label(series, include_reductions=multiple_series) or ""
        value = str(data.value) if data.valid else "invalid"
        lines.append(f"{label}: {value}" if label else value)
    axis.text(0.5, 0.5, "\n".join(lines), ha="center", va="center")


def _panels(evaluated: EvaluatedFigureData):
    panels = []
    for layer in evaluated.layers:
        for cell in layer.cells:
            if all(isinstance(series.data, EvaluatedImage) for series in cell.series):
                panels.extend((layer, cell, (series,)) for series in cell.series)
            else:
                panels.append((layer, cell, cell.series))
    return panels


def render_evaluated_figure(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: dict[str, FitResultBatch],
):
    """Render immutable DTOs without pyplot, rcParams, or shared Figures."""

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")
    layer_docs = {layer.layer_id: layer for layer in document.layers}
    descriptors = {item.dataset_id: item for item in document.datasets}
    panels = _panels(evaluated)
    columns = min(3, max(1, len(panels)))
    rows = math.ceil(len(panels) / columns)
    figure = Figure(figsize=(5.0 * columns, 4.0 * rows), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns, squeeze=False).reshape(-1)

    for target, (layer, cell, series_group) in zip(axes, panels):
        layer_doc = layer_docs[layer.layer_id]
        fit_result = fit_results.get(layer.layer_id)
        kind = series_group[0].data
        if isinstance(kind, EvaluatedCurve):
            _curve(target, layer, cell, series_group, fit_result)
        elif isinstance(kind, EvaluatedImage):
            _image(target, figure, layer, cell, series_group[0], fit_result)
        elif isinstance(kind, EvaluatedHistogram):
            if fit_result is not None:
                raise ValueError("fit overlays require a curve or image view")
            _histogram(target, series_group)
        elif isinstance(kind, EvaluatedMeter):
            if fit_result is not None:
                raise ValueError("fit overlays require a curve or image view")
            _meter(target, series_group)
        else:  # pragma: no cover - closed EvaluatedLayerData union
            raise TypeError(f"unsupported evaluated data {type(kind).__name__}")
        title = descriptors[layer.dataset_id].label
        # A multi-series curve is already labelled per batch in its legend;
        # naming the whole panel after the first series would misdescribe it.
        addresses = cell.facet_address
        if len(series_group) == 1:
            addresses = (*addresses, *series_group[0].batch_address)
        details = _address_label(addresses)
        resolved = _address_label(layer.resolutions)
        if details:
            title = f"{title} — {details}"
        if resolved:
            title = f"{title}\nview: {resolved}"
        if len(series_group) == 1:
            reduced = _reduction_label(series_group[0].reductions)
            if reduced:
                title = f"{title}\nreduce: {reduced}"
        target.set_title(title)
    for unused in axes[len(panels):]:
        unused.set_visible(False)
    return figure


__all__ = ["render_evaluated_figure"]
