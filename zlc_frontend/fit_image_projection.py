"""Pure saved-fit projections shared by typed IMAGE boards and Agg export."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Mapping

from zlc_data import (
    FitBatchStatus,
    FitResultBatch,
    IndexSelection,
    Selection,
)
from zlc_storage import canonical_text

from .figure import (
    AxisAddress,
    AxisResolution,
    EvaluatedCell,
    EvaluatedFigureData,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedLayer,
    EvaluatedSeries,
    FigureDocument,
)
from .fit_grid import _fit_cell_address, _fit_cell_summary_text
from .image_view import ImageViewportTransform
from .render import RadialGaussianImageFitOverlay


_RADIAL_GAUSSIAN_MODEL_ID = "radial_gaussian_center"
_RADIAL_GAUSSIAN_PARAMETERS = frozenset(
    {
        "amplitude",
        "offset",
        "one_over_e_radius",
        "center_x",
        "center_y",
    }
)


@dataclass(frozen=True, slots=True, eq=False)
class RadialGaussianImageFitPanel:
    """One exact evaluated IMAGE cell plus its bounded saved-fit projection."""

    selection: Selection | None
    image: EvaluatedImage
    evaluated_input: EvaluatedInput
    home_viewport: ImageViewportTransform
    summary: str
    fit_overlay: RadialGaussianImageFitOverlay

    def __post_init__(self) -> None:
        if self.selection is not None and not isinstance(self.selection, Selection):
            raise TypeError("radial fit panel selection must be Selection or None")
        if not isinstance(self.image, EvaluatedImage):
            raise TypeError("radial fit panel image must be EvaluatedImage")
        if not isinstance(self.evaluated_input, EvaluatedInput):
            raise TypeError("radial fit panel input must be EvaluatedInput")
        if not isinstance(self.home_viewport, ImageViewportTransform):
            raise TypeError("radial fit panel home_viewport must be ImageViewportTransform")
        if self.home_viewport.viewport_revision != 0:
            raise ValueError("radial fit panel home_viewport must have revision zero")
        if self.home_viewport.raster_shape != self.image.values.shape:
            raise ValueError("radial fit panel home viewport differs from image geometry")
        if (
            self.home_viewport.x_axis.axis_id != self.image.x_axis.axis_id
            or self.home_viewport.y_axis.axis_id != self.image.y_axis.axis_id
        ):
            raise ValueError("radial fit panel home axes differ from evaluated image axes")
        canonical_text(self.summary, "radial fit panel summary")
        if len(self.summary) > 8192:
            raise ValueError("radial fit panel summary exceeds its display bound")
        if not isinstance(self.fit_overlay, RadialGaussianImageFitOverlay):
            raise TypeError("radial fit panel overlay has the wrong type")
        if self.fit_overlay.source_ref != self.evaluated_input.ref:
            raise ValueError("radial fit panel overlay belongs to another input")
        if self.fit_overlay.coordinate_frame != self.home_viewport.coordinate_frame:
            raise ValueError("radial fit panel overlay belongs to another coordinate frame")

    @property
    def fit_storage_index(self) -> int | None:
        return self.fit_overlay.batch_storage_index

    @property
    def caption(self) -> str:
        return self.fit_overlay.caption


def address_label(
    items: tuple[AxisAddress, ...] | tuple[AxisResolution, ...],
) -> str:
    return ", ".join(f"{item.axis_id.value}={item.coordinate}" for item in items)


def reduction_label(reductions) -> str:
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


def evaluated_figure_panels(evaluated: EvaluatedFigureData):
    """Return the canonical display-panel order without importing a renderer."""

    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    panels = []
    for layer in evaluated.layers:
        for cell in layer.cells:
            if all(isinstance(series.data, EvaluatedImage) for series in cell.series):
                panels.extend((layer, cell, (series,)) for series in cell.series)
            else:
                panels.append((layer, cell, cell.series))
    return tuple(panels)


def _fit_batch_multi_index(
    result: FitResultBatch,
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series: EvaluatedSeries,
) -> tuple[int, ...]:

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
    extras = set(by_axis) - expected
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
            raise RuntimeError(f"figure does not identify fit batch axis {axis.axis_id}")
    return tuple(multi)


def fit_batch_storage_index(
    result: FitResultBatch,
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series: EvaluatedSeries,
) -> int | None:
    """Resolve one displayed cell to its authoritative sparse fit row."""

    multi = _fit_batch_multi_index(result, layer, cell, series)
    try:
        return result.batch_layout.storage_index(multi)
    except KeyError:
        # Sparse logical galleries retain their holes.  Never shift a later
        # stored row into the missing cell.
        return None


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


def radial_gaussian_fit_geometry(
    result: FitResultBatch,
    storage_index: int,
) -> tuple[tuple[float, float], float]:
    """Project one converged row by parameter *name*, never catalog position."""

    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if result.spec.model_id != _RADIAL_GAUSSIAN_MODEL_ID:
        raise ValueError("radial image projection requires radial_gaussian_center")
    if (
        isinstance(storage_index, bool)
        or not isinstance(storage_index, Integral)
        or not 0 <= int(storage_index) < result.batch_layout.storage_size
    ):
        raise IndexError("fit storage index is outside the saved batch")
    index = int(storage_index)
    if result.statuses[index] is not FitBatchStatus.CONVERGED:
        raise ValueError("failed fit rows do not carry radial geometry")
    definitions = result.parameter_definitions
    names = tuple(definition.name for definition in definitions)
    if len(names) != len(set(names)) or frozenset(names) != _RADIAL_GAUSSIAN_PARAMETERS:
        raise ValueError("radial-Gaussian parameter definition names have drifted")
    values = {
        definition.name: float(value)
        for definition, value in zip(
            definitions,
            result.parameter_values[index],
            strict=True,
        )
    }
    # Amplitude and offset do not define the annotation geometry, but reading
    # them by name closes the entire model-to-projection contract and makes a
    # future catalog rename fail closed instead of silently shifting columns.
    amplitude = values["amplitude"]
    offset = values["offset"]
    radius = values["one_over_e_radius"]
    center = (values["center_x"], values["center_y"])
    if not all(math.isfinite(value) for value in (amplitude, offset, radius, *center)):
        raise ValueError("converged radial-Gaussian parameters must be finite")
    if radius <= 0.0:
        raise ValueError("converged radial-Gaussian radius must be positive")
    return center, radius


def radial_gaussian_fit_overlay(
    result: FitResultBatch,
    storage_index: int | None,
    *,
    artifact_identity: str,
    caption: str,
    evaluated_input: EvaluatedInput,
) -> RadialGaussianImageFitOverlay:
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    if not isinstance(evaluated_input, EvaluatedInput):
        raise TypeError("evaluated_input must be EvaluatedInput")
    if result.source_ref != evaluated_input.ref:
        raise ValueError("saved fit and evaluated image input revisions differ")
    if len(result.fit_axis_specs) != 2:
        raise ValueError("radial image projection requires two fitted axes")
    frames = tuple(axis.coordinate_frame for axis in result.fit_axis_specs)
    if frames[0] is None or frames[0] != frames[1]:
        raise ValueError("radial image fit axes require one declared coordinate frame")
    if storage_index is None:
        return RadialGaussianImageFitOverlay(
            result.source_ref,
            artifact_identity,
            None,
            None,
            frames[0],
            caption,
            "NOT_PRESENT",
        )
    if (
        isinstance(storage_index, bool)
        or not isinstance(storage_index, Integral)
        or not 0 <= int(storage_index) < result.batch_layout.storage_size
    ):
        raise IndexError("fit storage index is outside the saved batch")
    index = int(storage_index)
    status = result.statuses[index]
    diagnostic = ""
    center = None
    radius = None
    if status is FitBatchStatus.CONVERGED:
        center, radius = radial_gaussian_fit_geometry(result, index)
    else:
        diagnostic = status.value
        if result.errors[index]:
            diagnostic = f"{diagnostic}: {result.errors[index]}"
    return RadialGaussianImageFitOverlay(
        result.source_ref,
        artifact_identity,
        index,
        status,
        frames[0],
        caption,
        diagnostic,
        center,
        radius,
    )


def radial_gaussian_image_fit_panels(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    fit_results: Mapping[str, FitResultBatch],
    layer_id: str,
    *,
    artifact_identity: str,
) -> tuple[RadialGaussianImageFitPanel, ...]:
    """Project every logical IMAGE panel, including sparse fit holes."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    layer_id = canonical_text(layer_id, "fit image layer_id")
    canonical_text(artifact_identity, "fit artifact identity")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")
    try:
        result = fit_results[layer_id]
    except KeyError as exc:
        raise ValueError(f"layer {layer_id!r} has no saved fit result") from exc
    if not isinstance(result, FitResultBatch):
        raise TypeError("saved fit mapping values must be FitResultBatch")
    if result.spec.model_id != _RADIAL_GAUSSIAN_MODEL_ID:
        raise ValueError("typed radial image panels require radial_gaussian_center")
    document_layers = {layer.layer_id: layer for layer in document.layers}
    try:
        document_layer = document_layers[layer_id]
    except KeyError as exc:
        raise ValueError(f"unknown figure layer {layer_id!r}") from exc
    inputs = {item.dataset_id: item for item in evaluated.inputs}
    try:
        evaluated_input = inputs[document_layer.dataset_id]
    except KeyError as exc:
        raise ValueError("evaluated figure omitted the fitted layer input") from exc

    home_viewport = ImageViewportTransform(result.fit_axis_specs)
    projected = []
    for layer, cell, series_group in evaluated_figure_panels(evaluated):
        if layer.layer_id != layer_id:
            continue
        if layer.dataset_id != document_layer.dataset_id:
            raise ValueError("evaluated fit layer belongs to another dataset")
        if len(series_group) != 1 or not isinstance(
            series_group[0].data,
            EvaluatedImage,
        ):
            raise ValueError("radial saved-fit layer must contain only IMAGE panels")
        series = series_group[0]
        image = series.data
        if (
            home_viewport.x_axis.axis_id != image.x_axis.axis_id
            or home_viewport.y_axis.axis_id != image.y_axis.axis_id
        ):
            raise ValueError(
                "role-resolved radial fit axes differ from evaluated image x/y axes"
            )
        if home_viewport.raster_shape != image.values.shape:
            raise ValueError("radial fit axes differ from evaluated image geometry")
        multi_index = _fit_batch_multi_index(result, layer, cell, series)
        storage = fit_batch_storage_index(result, layer, cell, series)
        selection = fit_panel_selection(layer, cell, series_group, result)
        caption = figure_panel_title(document, layer, cell, series_group)
        address = _fit_cell_address(result.batch_axis_specs, multi_index)
        summary = (
            _fit_cell_summary_text(result, storage, address)
            if storage is not None
            else (
                f"{address}\n"
                "storage row NOT_PRESENT · status NOT_PRESENT\n"
                "observations present/valid/used N/A · evaluations N/A\n"
                "parameters: N/A\n"
                "Sparse logical cell has no saved fit row; no neighbouring row "
                "was substituted."
            )
        )
        overlay = radial_gaussian_fit_overlay(
            result,
            storage,
            artifact_identity=artifact_identity,
            caption=caption,
            evaluated_input=evaluated_input,
        )
        projected.append(
            RadialGaussianImageFitPanel(
                selection,
                image,
                evaluated_input,
                home_viewport,
                summary,
                overlay,
            )
        )
    if not projected:
        raise ValueError(f"layer {layer_id!r} produced no IMAGE panels")
    return tuple(projected)


__all__ = ["RadialGaussianImageFitPanel"]
