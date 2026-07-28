"""Renderer-neutral Figure panel identity and Fit batch projection.

This module owns the generic mapping from an evaluated Figure to logical
panels, stable source-aware focus addresses, titles, and sparse Fit rows.  Image-specific
geometry remains in :mod:`fit_image_projection`; renderers consume this owner
instead of importing an image module for generic Figure semantics.
"""

from __future__ import annotations

from numbers import Integral

from zlc_data import (
    AxisSourceRef,
    FitResultBatch,
    SCALAR_AXIS,
    exact_integer_text,
)

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

def coordinate_label(value: object) -> str:
    """Return one stable human-readable coordinate value."""

    if not isinstance(value, bool) and isinstance(value, Integral):
        return exact_integer_text(value)
    return str(value)


def address_label(
    items: tuple[AxisAddress, ...] | tuple[AxisResolution, ...],
) -> str:
    labels = []
    for item in items:
        coordinate = coordinate_label(item.coordinate)
        if isinstance(item, AxisAddress):
            labels.append(f"{item.axis_name}={coordinate}")
            continue
        source = item.source
        label = source.kind.lower() if source.axis_id is None else source.axis_id.value
        labels.append(f"{label}={coordinate}")
    return ", ".join(labels)


def reduction_label(reductions) -> str:
    labels = []
    for reduction in reductions:
        axes = ",".join(
            source.kind.lower()
            if source.axis_id is None
            else source.axis_id.value
            for source in reduction.sources
        )
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


def canonical_panel_focus_address(
    value: object,
) -> tuple[tuple[AxisSourceRef, int], ...]:
    """Validate a panel address; the empty tuple identifies the sole panel."""

    entries = tuple(value)
    prepared = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError("panel focus address entries must be (source, index) tuples")
        source, index = entry
        if not isinstance(source, AxisSourceRef):
            raise TypeError("panel focus address source must be AxisSourceRef")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError("panel focus address index must be non-negative")
        prepared.append((source, index))
    if len({source for source, _index in prepared}) != len(prepared):
        raise ValueError("panel focus address cannot repeat a source")
    return tuple(sorted(prepared, key=lambda item: item[0]))


def panel_focus_address(
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series_group: tuple[EvaluatedSeries, ...],
) -> tuple[tuple[AxisSourceRef, int], ...]:
    """Return one canonical source-aware logical panel address.

    Dynamic resolution facts identify the evaluated snapshot, not the cell;
    including them would make focus stale whenever a live source advanced. A
    source with no AxisId (notably POINT_ROWS) is still fully identified here;
    exact physical row membership is resolved later by the Figure contract.
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
    by_source: dict[AxisSourceRef, int] = {}
    for address in addresses:
        incumbent = by_source.setdefault(address.source, address.index)
        if incumbent != address.index:
            raise RuntimeError("figure panel addresses disagree")
    return canonical_panel_focus_address(by_source.items())


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
    projected_sources: tuple[AxisSourceRef, ...] = (),
) -> tuple[int, ...]:
    """Map one exact displayed series onto the Fit batch's logical address."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    addresses = (*cell.facet_address, *series.batch_address)
    by_source = {item.source: item.index for item in addresses}
    if len(by_source) != len(addresses):
        raise RuntimeError("figure batch/facet addresses contain a duplicate source")
    for resolution in layer.resolutions:
        incumbent = by_source.setdefault(resolution.source, resolution.index)
        if incumbent != resolution.index:
            raise RuntimeError("figure address and resolution disagree")
    expected = set(result.spec.batch_sources)
    projected_sources = tuple(projected_sources)
    projected = set(projected_sources)
    if len(projected) != len(projected_sources) or any(
        not isinstance(source, AxisSourceRef) for source in projected
    ):
        raise ValueError("projected_sources must contain unique AxisSourceRef values")
    extras = (
        set(by_source)
        - expected
        - projected
        - {AxisSourceRef.tensor(SCALAR_AXIS.axis_id)}
    )
    if extras:
        raise RuntimeError(
            f"figure resolved non-batch fit axes: {sorted(map(str, extras))}"
        )
    multi = []
    for source, axis in zip(
        result.spec.batch_sources,
        result.batch_axis_specs,
        strict=True,
    ):
        if source in by_source:
            multi.append(by_source[source])
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
    projected_sources: tuple[AxisSourceRef, ...] = (),
) -> int | None:
    """Resolve one displayed cell to its authoritative sparse Fit row."""

    multi = fit_batch_multi_index(
        result,
        layer,
        cell,
        series,
        projected_sources=projected_sources,
    )
    try:
        return result.batch_layout.storage_index(multi)
    except KeyError:
        return None


__all__ = [
    "address_label",
    "canonical_panel_focus_address",
    "coordinate_label",
    "evaluated_figure_panels",
    "figure_panel_title",
    "fit_batch_multi_index",
    "fit_batch_storage_index",
    "iter_evaluated_figure_panels",
    "panel_focus_address",
    "reduction_label",
]
