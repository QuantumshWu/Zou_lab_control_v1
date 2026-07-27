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
from .fit_projection import (
    address_label as _address_label,
    reduction_label as _reduction_label,
)
from .render_style import (
    ANNOTATION_FONT_SIZE,
    CURVE_LINESTYLE,
    CURVE_MARKER,
    FIT_CONTOUR_COLOR,
    FIT_CONTOUR_LINEWIDTH,
    FIT_FAILURE_COLOR,
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
    return _raster_from_drawn_agg(
        figure,
        physical_size=physical_size,
    )


def _raster_from_drawn_agg(
    figure,
    *,
    physical_size: tuple[int, int] | None = None,
) -> RasterBuffer:
    """Copy the already-drawn Agg renderer into an immutable front."""

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


class _AggBlitCache:
    """Renderer-owned Agg blit cache for progressive live fronts.

    This is the same visual algorithm as Main: a chrome change gets one normal
    full draw; only a second frame with identical chrome captures a data-free
    background and enters the steady blit path.  A plot whose ticks/limits keep
    changing therefore pays no extra recapture work, while a stable camera or
    curve redraws only its data artists.  The cache never crosses the worker
    boundary and every result is copied into one immutable full front.
    """

    __slots__ = (
        "_artist_ids",
        "_background",
        "_primed",
        "_signature",
    )

    def __init__(self) -> None:
        self._artist_ids: tuple[int, ...] = ()
        self._background = None
        self._primed = False
        self._signature = None

    def clear(self) -> None:
        self._artist_ids = ()
        self._background = None
        self._primed = False
        self._signature = None

    def raster(
        self,
        figure,
        dynamic_artists,
        *,
        layout_key,
        chrome_key,
        physical_size: tuple[int, int],
    ) -> RasterBuffer:
        supplied = tuple(artist for artist in dynamic_artists if artist is not None)
        # Main's generic rule is intentionally broader than each plotter's
        # hand-written update list: every Axes data/annotation artist is absent
        # from the chrome background.  This prevents a colorbar collection,
        # threshold line, or newly added fit text from becoming a ghost merely
        # because one caller forgot to extend a tuple.
        generic = []
        for axis in figure.axes:
            generic.extend(axis.lines)
            generic.extend(axis.images)
            generic.extend(axis.collections)
            generic.extend(axis.patches)
            generic.extend(axis.texts)
        unique = []
        seen = set()
        for artist in (*generic, *supplied):
            if id(artist) in seen:
                continue
            seen.add(id(artist))
            unique.append(artist)
        artists = tuple(
            artist
            for _index, artist in sorted(
                enumerate(unique),
                key=lambda pair: (float(pair[1].get_zorder()), pair[0]),
            )
        )
        artist_ids = tuple(id(artist) for artist in artists)
        signature = (layout_key, chrome_key, artist_ids)
        try:
            if not self._primed or self._signature != signature:
                self._primed = True
                self._signature = signature
                self._artist_ids = artist_ids
                self._background = None
                return raster_from_agg(figure, physical_size=physical_size)

            if self._background is None:
                artist_visibility = tuple(
                    bool(artist.get_visible()) for artist in artists
                )
                try:
                    for artist in artists:
                        artist.set_visible(False)
                    figure.canvas.draw()
                    self._background = figure.canvas.copy_from_bbox(figure.bbox)
                finally:
                    for artist, was_visible in zip(
                        artists,
                        artist_visibility,
                        strict=True,
                    ):
                        artist.set_visible(was_visible)
            assert self._background is not None
            figure.canvas.restore_region(self._background)
            for artist in artists:
                if not artist.get_visible():
                    continue
                axes = getattr(artist, "axes", None)
                if axes is None:
                    figure.draw_artist(artist)
                else:
                    axes.draw_artist(artist)
            return _raster_from_drawn_agg(
                figure,
                physical_size=physical_size,
            )
        except BaseException:
            # Region operations are an optimisation boundary, not a product
            # failure.  Clear the possibly partial cache and return the exact
            # ordinary full render; the next stable revision may try again.
            self.clear()
            raster = raster_from_agg(figure, physical_size=physical_size)
            self._primed = True
            self._signature = signature
            self._artist_ids = artist_ids
            return raster


def _agg_layout_key(figure, *, extra=()) -> tuple:
    """Freeze facts that require rebuilding the outer Figure background."""

    return (
        tuple(float(value) for value in figure.get_size_inches()),
        float(figure.dpi),
        tuple(float(value) for value in figure.get_facecolor()),
        tuple(
            (
                id(axis),
                tuple(float(value) for value in axis.get_position().bounds),
                bool(axis.get_visible()),
            )
            for axis in figure.axes
        ),
        tuple(extra),
    )


def _agg_chrome_key(figure, *, extra=()) -> tuple:
    """Freeze the visible non-data facts that invalidate an Agg background."""

    axes = []
    for axis in figure.axes:
        axes.append(
            (
                tuple(float(value) for value in axis.get_position().bounds),
                tuple(float(value) for value in axis.get_xlim()),
                tuple(float(value) for value in axis.get_ylim()),
                str(axis.get_xscale()),
                str(axis.get_yscale()),
                str(axis.get_xlabel()),
                str(axis.get_ylabel()),
                str(axis.get_title()),
                tuple(float(value) for value in axis.get_xticks()),
                tuple(float(value) for value in axis.get_yticks()),
                tuple(label.get_text() for label in axis.get_xticklabels()),
                tuple(label.get_text() for label in axis.get_yticklabels()),
                bool(axis.get_visible()),
                bool(axis.axison),
                str(axis.get_aspect()),
                str(axis.get_anchor()),
            )
        )
    return (
        tuple(float(value) for value in figure.get_size_inches()),
        float(figure.dpi),
        tuple(float(value) for value in figure.get_facecolor()),
        tuple(axes),
        tuple(extra),
    )

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
