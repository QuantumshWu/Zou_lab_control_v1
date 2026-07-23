"""Headless typed preview projection for a finite camera capture."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_data import SPATIAL_X, SPATIAL_Y
from zlc_frontend.figure import (
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    SuggestionStatus,
    ViewIntent,
    suggest_view,
)
from zlc_frontend.image_display import (
    ImageDisplayState,
    image_viewport_for_display_state,
)
from zlc_frontend.image_view import ImageViewportTransform
from zlc_neutral_atom.capture_application import PreparedFiniteCapture


IMAGE_PANEL_ID = "capture-image"
PROJECTION_TEXT = "latest rendered raw frame · DISPLAY ONLY"


@dataclass(frozen=True, slots=True)
class CapturePreviewProjection:
    """One checked display product for the next finite-capture Run."""

    generation: int
    dataset_id: DatasetId
    document: FigureDocument
    image_viewport: ImageViewportTransform


def prepare_capture_preview_projection(
    command: PreparedFiniteCapture,
    generation: int,
    image_display: ImageDisplayState,
) -> CapturePreviewProjection:
    if not isinstance(command, PreparedFiniteCapture):
        raise TypeError("command must be PreparedFiniteCapture")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("capture preview generation must be a positive integer")
    if not isinstance(image_display, ImageDisplayState):
        raise TypeError("image_display must be ImageDisplayState")

    schema = command.preview_schema
    y_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_Y
    )
    x_axes = tuple(
        axis for axis in schema.cell_schema.data_axes if axis.role == SPATIAL_X
    )
    if (
        len(y_axes) != 1
        or len(x_axes) != 1
        or len(schema.cell_schema.data_axes) != 2
    ):
        raise ValueError(
            "finite capture image requires exactly one declared "
            "SPATIAL_Y and SPATIAL_X axis"
        )
    suggestion = suggest_view(schema, ViewIntent.IMAGE)
    if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
        raise ValueError("IMAGE view needs an explicit axis choice")

    dataset_id = DatasetId(f"capture-preview-{generation}")
    document = FigureDocument(
        f"capture-preview-{generation}",
        0,
        (
            DatasetDescriptor(
                dataset_id,
                "Raw camera frame",
                schema.fingerprint,
            ),
        ),
        (FigureLayer(IMAGE_PANEL_ID, dataset_id, suggestion.spec),),
    )
    image_viewport = image_viewport_for_display_state(
        image_display,
        ImageViewportTransform((y_axes[0], x_axes[0])),
    )
    return CapturePreviewProjection(
        generation,
        dataset_id,
        document,
        image_viewport,
    )


__all__ = [
    "CapturePreviewProjection",
    "IMAGE_PANEL_ID",
    "PROJECTION_TEXT",
    "prepare_capture_preview_projection",
]
