"""Pure saved-fit projections shared by typed IMAGE boards and Agg export."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
from typing import Callable

from zlc_data import (
    DatasetSchema,
    FitBatchStatus,
    FitResultBatch,
    IndexSelection,
    REPEAT,
    Selection,
    SITE,
    resolve_selection_indices,
    validate_fit_result_source_binding,
)
from zlc_storage import canonical_text

from .figure import (
    AxisAddress,
    AxisResolution,
    AxisViewRole,
    EvaluatedCell,
    EvaluatedFigureData,
    EvaluatedImage,
    EvaluatedInput,
    EvaluatedLayer,
    EvaluatedSeries,
    FigureDocument,
    FigureLayer,
    ViewIntent,
)
from .figure.contract import _selection_fit_projection, dataset_axes
from .fit_grid import (
    _fit_cell_address,
    _fit_cell_summary_text,
    coordinate_label,
)
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


def _sorted_indices_contain_all(
    available,
    required,
    *,
    check_cancelled: Callable[[], None] | None = None,
) -> bool:
    """Prove ordered-index containment without allocating integer sets."""

    if isinstance(available, range) and isinstance(required, range):
        if len(required) == 0:
            return True
        if required[0] not in available or required[-1] not in available:
            return False
        return len(required) == 1 or required.step % available.step == 0
    if isinstance(available, range):
        for position, index in enumerate(required):
            if check_cancelled is not None and position % 4096 == 0:
                check_cancelled()
            if index not in available:
                return False
        return True
    available_iter = iter(available)
    try:
        current = next(available_iter)
    except StopIteration:
        return len(required) == 0
    for position, target in enumerate(required):
        if check_cancelled is not None and position % 4096 == 0:
            check_cancelled()
        while current < target:
            try:
                current = next(available_iter)
            except StopIteration:
                return False
        if current != target:
            return False
    return True


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
    labels = []
    for item in items:
        coordinate = coordinate_label(item.coordinate)
        if isinstance(item, AxisAddress):
            if item.axis_role == SITE:
                labels.append(f"site {item.index}")
                continue
            if item.axis_role == REPEAT:
                labels.append(f"repeat {item.index}")
                continue
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


def _iter_evaluated_figure_panels(evaluated: EvaluatedFigureData):
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

    return tuple(_iter_evaluated_figure_panels(evaluated))


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


def panel_focus_selection(
    layer: EvaluatedLayer,
    cell: EvaluatedCell,
    series_group: tuple[EvaluatedSeries, ...],
) -> Selection | None:
    """Return the stable logical cell identity used by live overview focus.

    Facet and single-series batch addresses identify the cell.  Dynamic
    resolution facts such as ``LatestNonempty`` identify the snapshot used to
    evaluate it, not the cell itself; including them made a valid focus stale
    whenever the next live snapshot advanced.
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


def transient_single_panel_radial_fit_overlay(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    source_schema: DatasetSchema,
    result: FitResultBatch,
    *,
    result_identity: str,
    check_cancelled: Callable[[], None] | None = None,
) -> RadialGaussianImageFitOverlay:
    """Project one radial result over an unchanged cached full IMAGE view."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    if not isinstance(source_schema, DatasetSchema):
        raise TypeError("source_schema must be DatasetSchema")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    canonical_text(result_identity, "radial fit result identity")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")
    if (
        len(document.layers) != 1
        or len(evaluated.layers) != 1
        or len(evaluated.inputs) != 1
    ):
        raise ValueError("transient radial projection requires one layer and input")
    document_layer = document.layers[0]
    layer = evaluated.layers[0]
    evaluated_input = evaluated.inputs[0]
    if (
        layer.layer_id != document_layer.layer_id
        or layer.dataset_id != document_layer.dataset_id
        or evaluated_input.dataset_id != layer.dataset_id
    ):
        raise ValueError("transient radial source identities differ")
    validate_fit_result_source_binding(result, evaluated_input.ref, source_schema)
    if result.spec.model_id != _RADIAL_GAUSSIAN_MODEL_ID:
        raise ValueError("typed radial image projection requires radial_gaussian_center")
    if len(result.fit_axis_specs) != 2:
        raise ValueError("typed radial image projection requires two fitted axes")
    if (
        document_layer.view.intent is not ViewIntent.IMAGE
        or document_layer.view.binding(result.fit_axis_specs[0].axis_id).role
        is not AxisViewRole.IMAGE_X
        or document_layer.view.binding(result.fit_axis_specs[1].axis_id).role
        is not AxisViewRole.IMAGE_Y
    ):
        raise ValueError("cached IMAGE view does not bind the fitted x/y axes")
    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
        AxisViewRole.SLIDER,
    }
    if any(
        document_layer.view.binding(axis.axis_id).role not in allowed_batch_roles
        and not (
            axis.size == 1
            and document_layer.view.binding(axis.axis_id).role
            is AxisViewRole.REDUCED
        )
        for axis in result.batch_axis_specs
    ):
        raise ValueError("cached IMAGE view does not uniquely bind fit batch axes")
    if len(layer.cells) != 1 or len(layer.cells[0].series) != 1:
        raise ValueError("transient radial projection requires one IMAGE panel")
    cell = layer.cells[0]
    series = cell.series[0]
    image = series.data
    if not isinstance(image, EvaluatedImage):
        raise ValueError("transient radial projection requires EvaluatedImage")
    if (
        image.x_axis.axis_id != result.fit_axis_specs[0].axis_id
        or image.y_axis.axis_id != result.fit_axis_specs[1].axis_id
    ):
        raise ValueError("cached IMAGE axes differ from radial fit axes")

    if result.spec.committed_transform is not None:
        _effective, authority_selection = _selection_fit_projection(
            source_schema,
            result,
        )
        terms = {term.axis_id: term for term in authority_selection.terms}
        for fit_axis, evaluated_axis in zip(
            result.fit_axis_specs,
            (image.x_axis, image.y_axis),
            strict=True,
        ):
            try:
                term = terms[fit_axis.axis_id]
            except KeyError as exc:
                raise ValueError(
                    "transient radial fit requires an authority box on x/y"
                ) from exc
            selected, _drop = resolve_selection_indices(
                next(
                    axis
                    for axis in dataset_axes(source_schema)
                    if axis.axis_id == fit_axis.axis_id
                ),
                term,
            )
            if not _sorted_indices_contain_all(
                evaluated_axis.indices,
                selected,
                check_cancelled=check_cancelled,
            ):
                raise ValueError(
                    "cached IMAGE view does not contain the complete committed box"
                )

    storage = fit_batch_storage_index(result, layer, cell, series)
    return radial_gaussian_fit_overlay(
        result,
        storage,
        artifact_identity=result_identity,
        caption=figure_panel_title(document, layer, cell, (series,)),
        evaluated_input=evaluated_input,
    )


def _radial_image_projection_context(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    result: FitResultBatch,
    layer_id: str,
    *,
    artifact_identity: str,
) -> tuple[FigureLayer, EvaluatedInput]:
    """Validate and resolve the immutable owners shared by projection passes."""

    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    if not isinstance(result, FitResultBatch):
        raise TypeError("result must be FitResultBatch")
    layer_id = canonical_text(layer_id, "fit image layer_id")
    identity = canonical_text(artifact_identity, "fit artifact identity")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")
    if result.spec.model_id != _RADIAL_GAUSSIAN_MODEL_ID:
        raise ValueError("typed radial image panels require radial_gaussian_center")
    document_layer = next(
        (layer for layer in document.layers if layer.layer_id == layer_id),
        None,
    )
    if document_layer is None:
        raise ValueError(f"unknown figure layer {layer_id!r}")
    evaluated_input = next(
        (
            item
            for item in evaluated.inputs
            if item.dataset_id == document_layer.dataset_id
        ),
        None,
    )
    if evaluated_input is None:
        raise ValueError("evaluated figure omitted the fitted layer input")
    if result.source_ref != evaluated_input.ref:
        raise ValueError("saved fit and evaluated image input revisions differ")
    if len(result.fit_axis_specs) != 2:
        raise ValueError("radial image projection requires two fitted axes")
    frames = tuple(axis.coordinate_frame for axis in result.fit_axis_specs)
    if frames[0] is None or frames[0] != frames[1]:
        raise ValueError("radial image fit axes require one coordinate frame")
    return document_layer, evaluated_input


def radial_gaussian_image_fit_panels(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
    result: FitResultBatch,
    layer_id: str,
    *,
    artifact_identity: str,
) -> tuple[RadialGaussianImageFitPanel, ...]:
    """Project every logical IMAGE panel, including sparse fit holes."""

    document_layer, evaluated_input = _radial_image_projection_context(
        document,
        evaluated,
        result,
        layer_id,
        artifact_identity=artifact_identity,
    )

    home_viewport = ImageViewportTransform(result.fit_axis_specs)
    projected = []
    for layer, cell, series_group in _iter_evaluated_figure_panels(evaluated):
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


__all__ = [
    "RadialGaussianImageFitPanel",
    "panel_focus_selection",
]
