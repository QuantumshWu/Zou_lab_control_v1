"""Renderer-neutral Figure panel identity and Fit batch projection.

This module owns the generic mapping from an evaluated Figure to logical
panels, stable focus selections, titles, and sparse Fit rows.  Image-specific
geometry remains in :mod:`fit_image_projection`; renderers consume this owner
instead of importing an image module for generic Figure semantics.
"""

from __future__ import annotations

from zlc_data import AxisId, FitResultBatch, IndexSelection, SCALAR_AXIS, Selection

from .figure import (
    AxisAddress,
    AxisResolution,
    EvaluatedCell,
    EvaluatedFigureData,
    EvaluatedImage,
    EvaluatedLayer,
    EvaluatedSeries,
    FigureDocument,
)
from .fit_grid import coordinate_label


def address_label(
    items: tuple[AxisAddress, ...] | tuple[AxisResolution, ...],
) -> str:
    labels = []
    for item in items:
        coordinate = coordinate_label(item.coordinate)
        if isinstance(item, AxisAddress):
            labels.append(f"{item.axis_name}={coordinate}")
            continue
        labels.append(f"{item.axis_id.value}={coordinate}")
    return ", ".join(labels)


def reduction_label(reductions) -> str:
    labels = []
    for reduction in reductions:
        axes = ",".join(axis_id.value for axis_id in reduction.axis_ids)
        contributors = coordinate_label(reduction.minimum_contributors)
        if reduction.minimum_contributors != reduction.maximum_contributors:
            contributors = (
                f"{coordinate_label(reduction.minimum_contributors)}.."
                f"{coordinate_label(reduction.maximum_contributors)}"
            )
        labels.append(
            f"{reduction.method.value.lower()}({axes}, n={contributors})"
        )
    return "; ".join(labels)


def iter_evaluated_figure_panels(evaluated: EvaluatedFigureData):
    """Yield panels in the one canonical renderer-independent order."""

    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    for layer in evaluated.layers:
        for cell in layer.cells:
            if all(isinstance(series.data, EvaluatedImage) for series in cell.series):
                for series in cell.series:
                    yield layer, cell, (series,)
            else:
                yield layer, cell, cell.series


def evaluated_figure_panels(evaluated: EvaluatedFigureData):
    """Return the canonical display-panel order without importing a renderer."""

    return tuple(iter_evaluated_figure_panels(evaluated))


def fit_panel_selection(
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series_group: tuple[EvaluatedSeries, ...],
    result: FitResultBatch | None,
) -> Selection | None:
    expected = (
        set()
        if result is None
        else {axis.axis_id for axis in result.batch_axis_specs}
    )
    addresses = [*cell.facet_address]
    if len(series_group) == 1:
        addresses.extend(series_group[0].batch_address)
    addresses.extend(layer.resolutions)
    by_axis = {}
    for address in addresses:
        if expected and address.axis_id not in expected:
            continue
        incumbent = by_axis.setdefault(address.axis_id, address.index)
        if incumbent != address.index:
            raise RuntimeError("figure panel addresses disagree")
    terms = tuple(
        IndexSelection(axis_id, index)
        for axis_id, index in sorted(
            by_axis.items(),
            key=lambda item: item[0].value,
        )
    )
    return None if not terms else Selection(terms)


def panel_focus_selection(
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series_group: tuple[EvaluatedSeries, ...],
) -> Selection | None:
    """Return stable logical cell identity for live overview focus.

    Dynamic resolution facts identify the evaluated snapshot, not the cell;
    including them would make focus stale whenever a live source advanced.
    """

    if not isinstance(layer, EvaluatedLayer):
        raise TypeError("layer must be EvaluatedLayer")
    if not isinstance(cell, EvaluatedCell):
        raise TypeError("cell must be EvaluatedCell")
    if not isinstance(series_group, tuple) or any(
        not isinstance(series, EvaluatedSeries) for series in series_group
    ):
        raise TypeError("series_group must contain EvaluatedSeries values")
    addresses = [*cell.facet_address]
    if len(series_group) == 1:
        addresses.extend(series_group[0].batch_address)
    by_axis = {}
    for address in addresses:
        incumbent = by_axis.setdefault(address.axis_id, address.index)
        if incumbent != address.index:
            raise RuntimeError("figure panel addresses disagree")
    terms = tuple(
        IndexSelection(axis_id, index)
        for axis_id, index in sorted(
            by_axis.items(),
            key=lambda item: item[0].value,
        )
    )
    return None if not terms else Selection(terms)


def figure_panel_title(
    document: FigureDocument,
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series_group: tuple[EvaluatedSeries, ...],
) -> str:
    title = document.descriptor(layer.dataset_id).label
    addresses = cell.facet_address
    if len(series_group) == 1:
        addresses = (*addresses, *series_group[0].batch_address)
    details = address_label(addresses)
    resolved = address_label(layer.resolutions)
    if details:
        title = f"{title} — {details}"
    if resolved:
        title = f"{title}\nview: {resolved}"
    if len(series_group) == 1:
        reduced = reduction_label(series_group[0].reductions)
        if reduced:
            title = f"{title}\nreduce: {reduced}"
    return title


def fit_batch_multi_index(
    result: FitResultBatch,
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series: EvaluatedSeries,
    *,
    projected_axis_ids: tuple[AxisId, ...] = (),
) -> tuple[int, ...]:
    """Map one exact displayed series onto the Fit batch's logical address."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    addresses = (*cell.facet_address, *series.batch_address)
    by_axis = {item.axis_id: item.index for item in addresses}
    if len(by_axis) != len(addresses):
        raise RuntimeError("figure batch/facet addresses contain a duplicate axis")
    for resolution in layer.resolutions:
        incumbent = by_axis.setdefault(resolution.axis_id, resolution.index)
        if incumbent != resolution.index:
            raise RuntimeError("figure address and resolution disagree")
    expected = {axis.axis_id for axis in result.batch_axis_specs}
    projected_axis_ids = tuple(projected_axis_ids)
    projected = set(projected_axis_ids)
    if len(projected) != len(projected_axis_ids) or any(
        not isinstance(axis_id, AxisId) for axis_id in projected
    ):
        raise ValueError("projected_axis_ids must contain unique AxisId values")
    extras = set(by_axis) - expected - projected - {SCALAR_AXIS.axis_id}
    if extras:
        raise RuntimeError(
            f"figure resolved non-batch fit axes: {sorted(map(str, extras))}"
        )
    multi = []
    for axis in result.batch_axis_specs:
        if axis.axis_id in by_axis:
            multi.append(by_axis[axis.axis_id])
        elif axis.size == 1:
            multi.append(0)
        else:
            raise RuntimeError(
                f"figure does not identify fit batch axis {axis.axis_id}"
            )
    return tuple(multi)


def fit_batch_storage_index(
    result: FitResultBatch,
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series: EvaluatedSeries,
    *,
    projected_axis_ids: tuple[AxisId, ...] = (),
) -> int | None:
    """Resolve one displayed cell to its authoritative sparse Fit row."""

    multi = fit_batch_multi_index(
        result,
        layer,
        cell,
        series,
        projected_axis_ids=projected_axis_ids,
    )
    try:
        return result.batch_layout.storage_index(multi)
    except KeyError:
        return None


__all__ = [
    "address_label",
    "evaluated_figure_panels",
    "figure_panel_title",
    "fit_batch_multi_index",
    "fit_batch_storage_index",
    "fit_panel_selection",
    "iter_evaluated_figure_panels",
    "panel_focus_selection",
    "reduction_label",
]
