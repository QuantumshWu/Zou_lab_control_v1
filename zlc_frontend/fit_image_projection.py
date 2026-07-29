"""Pure saved-fit projections shared by typed IMAGE boards and Agg export."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Callable

from zlc_data import (
    DatasetSchema,
    FitBatchStatus,
    FitResultBatch,
)
from zlc_data.fit import validate_fit_result_source_binding
from zlc_storage import canonical_text

from .figure import (
    AxisViewRole,
    EvaluatedFigureData,
    EvaluatedImage,
    EvaluatedInput,
    FigureDocument,
    ViewIntent,
)
from .figure.contract import _fit_display_selection_indices
from .fit_projection import (
    figure_panel_title,
    fit_batch_multi_index,
    fit_batch_storage_index,
)
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
        or document_layer.view.binding(result.spec.independent_sources[0]).role
        is not AxisViewRole.IMAGE_X
        or document_layer.view.binding(result.spec.independent_sources[1]).role
        is not AxisViewRole.IMAGE_Y
    ):
        raise ValueError("cached IMAGE view does not bind the fitted x/y axes")
    allowed_batch_roles = {
        AxisViewRole.BATCH,
        AxisViewRole.FACET,
        AxisViewRole.SELECTED,
    }
    if any(
        document_layer.view.binding(source).role not in allowed_batch_roles
        for source in result.spec.batch_sources
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
        image.x_axis.source != result.spec.independent_sources[0]
        or image.y_axis.source != result.spec.independent_sources[1]
    ):
        raise ValueError("cached IMAGE axes differ from radial fit axes")

    selected_by_source = dict(
        _fit_display_selection_indices(
            source_schema,
            document_layer.view,
            layer.resolutions,
            result,
        )
    )
    for evaluated_axis, fit_axis, source in zip(
        (image.x_axis, image.y_axis),
        result.fit_axis_specs,
        result.spec.independent_sources,
        strict=True,
    ):
        selected = selected_by_source.get(source)
        if selected is None:
            coordinates = evaluated_axis.coordinates
        else:
            coordinate_by_index = dict(
                zip(evaluated_axis.indices, evaluated_axis.coordinates, strict=True)
            )
            try:
                coordinates = tuple(coordinate_by_index[index] for index in selected)
            except KeyError as exc:
                raise ValueError(
                    "IMAGE Figure does not contain every selected Fit sample"
                ) from exc
        if len(coordinates) != fit_axis.size or any(
            coordinate != fit_axis.coordinate_at(index)
            for index, coordinate in enumerate(coordinates)
        ):
            raise ValueError("IMAGE Fit coordinates differ from the displayed source")

    storage = fit_batch_storage_index(result, layer, cell, series)
    return radial_gaussian_fit_overlay(
        result,
        storage,
        artifact_identity=result_identity,
        caption=figure_panel_title(document, layer, cell, (series,)),
        evaluated_input=evaluated_input,
    )


__all__ = [
    "radial_gaussian_fit_geometry",
    "radial_gaussian_fit_overlay",
    "transient_single_panel_radial_fit_overlay",
]
