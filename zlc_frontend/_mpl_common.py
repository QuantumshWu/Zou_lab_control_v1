"""Internal Matplotlib implementation owner: common."""

from __future__ import annotations

import math
from numbers import Integral, Number

import numpy as np
from zlc_data import FitBatchStatus, FitResultBatch
from zlc_storage import positive_integer
from .figure import (
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)
from .fit_image_projection import (
    RadialGaussianImageFitPanel,
    address_label as _address_label,
    evaluated_figure_panels as _panels,
    figure_panel_title as _panel_title,
    fit_batch_storage_index as _batch_storage_index,
    fit_panel_selection as _fit_panel_selection,
    panel_focus_selection as _panel_focus_selection,
    radial_gaussian_fit_geometry,
    reduction_label as _reduction_label,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    CURVE_LINESTYLE,
    CURVE_MARKER,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_FAILURE_COLOR,
    FIT_LINESTYLE,
    HIST_FILL_ALPHA,
    LINE_CYCLE,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    axis_label_fontsize,
    bimodal_fit_line_specs,
    render_style_context,
    small_fontsize,
    threshold_line_kwargs,
    tick_fontsize,
)
from .render import RasterBuffer


def _render_dpi(value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Number)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("dpi must be a finite positive number")
    return float(value)


def raster_from_agg(
    figure,
    *,
    physical_size: tuple[int, int] | None = None,
) -> RasterBuffer:
    """Own one Agg front at an exact requested physical-pixel size.

    Named live panels keep Main's authored floating Figure size so their fixed
    data box does not move.  Fractional DPR can nevertheless make Agg's floor
    allocation differ by one trailing-margin pixel from Qt's rounded contract.
    Crop or pad only the right/bottom handoff edge after drawing; artist
    transforms and the visible data box remain untouched.
    """

    figure.canvas.draw()
    actual_width, actual_height = figure.canvas.get_width_height()
    if physical_size is None:
        return RasterBuffer.from_agg_rgba(
            actual_width,
            actual_height,
            figure.canvas.buffer_rgba(),
        )
    width = positive_integer(physical_size[0], "physical width")
    height = positive_integer(physical_size[1], "physical height")
    if (width, height) == (actual_width, actual_height):
        return RasterBuffer.from_agg_rgba(
            width,
            height,
            figure.canvas.buffer_rgba(),
        )

    source = np.asarray(figure.canvas.buffer_rgba(), dtype=np.uint8)
    pixels = np.empty((height, width, 4), dtype=np.uint8)
    background = np.rint(
        np.clip(np.asarray(figure.get_facecolor()), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
    pixels[...] = background
    copy_width = min(width, actual_width)
    copy_height = min(height, actual_height)
    pixels[:copy_height, :copy_width] = source[
        :copy_height,
        :copy_width,
    ]
    return RasterBuffer(width, height, pixels.tobytes(order="C"))

def _series_label(series, *, include_reductions: bool) -> str | None:
    parts = [_address_label(series.batch_address)]
    if include_reductions:
        parts.append(_reduction_label(series.reductions))
    label = " | ".join(part for part in parts if part)
    return label or None

def _display_series_label(
    layer_id: str,
    series,
    index: int,
    *,
    multiple_series: bool,
) -> str:
    """Return the nonempty label used by both artists and exact payloads."""

    label = _series_label(series, include_reductions=multiple_series)
    if label is not None:
        return label
    return layer_id if not multiple_series else f"{layer_id} {index + 1}"

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
        color=FIT_FAILURE_COLOR,
        fontsize=ANNOTATION_FONT_SIZE,
    )
    return False

def _require_evaluated_identity(
    document: FigureDocument,
    evaluated: EvaluatedFigureData,
) -> None:
    if not isinstance(document, FigureDocument):
        raise TypeError("document must be FigureDocument")
    if not isinstance(evaluated, EvaluatedFigureData):
        raise TypeError("evaluated must be EvaluatedFigureData")
    if (
        document.document_id != evaluated.document_id
        or document.revision != evaluated.document_revision
    ):
        raise ValueError("document and evaluated data identities differ")

def release_agg_figure(figure) -> None:
    """Clear artists and sever the Figure/Canvas ownership edge."""
    canvas = getattr(figure, "canvas", None)
    try:
        figure.clear()
    finally:
        try:
            figure.set_canvas(None)
            if getattr(canvas, "figure", None) is figure:
                canvas.figure = None
        finally:
            figure = canvas = None

__all__ = [
    "raster_from_agg",
    "release_agg_figure",
]
