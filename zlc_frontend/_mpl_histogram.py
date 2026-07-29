"""Internal Matplotlib implementation owner: histogram."""

from __future__ import annotations

import math
import numpy as np
from zlc_data import FitBatchStatus
from zlc_data.fit import analyze_bimodal_distribution
from .figure import (
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedFigureData,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
)
from .histogram_display import (
    DEFAULT_HISTOGRAM_BINS,
    FacetedHistogramDisplayState,
    HistogramBinProjection,
    HistogramDisplayState,
    HistogramViewportTransform,
    histogram_count_limits,
    histogram_home_x_limits,
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

from ._mpl_common import (
    _series_label,
)

_AUTOMATIC_ANALYSIS_NOT_SUPPLIED = object()


def _histogram_analysis_cache(cache, counts_group, edges):
    """Return the exact observation-keyed automatic bimodal analysis cache."""

    primary_counts = np.asarray(counts_group[0])
    edge_values = np.asarray(edges)
    key = (
        edge_values.dtype.str,
        edge_values.tobytes(order="C"),
        primary_counts.dtype.str,
        primary_counts.tobytes(order="C"),
    )
    if cache is not None and cache[0] == key:
        return cache
    bin_centers = (edge_values[:-1] + edge_values[1:]) * 0.5
    return (
        key,
        analyze_bimodal_distribution(bin_centers, primary_counts),
    )

def _histogram_left_fraction(
    threshold: float,
    counts,
    edges,
) -> float:
    """Fraction of primary samples <= threshold from ALREADY-BINNED counts
    (linear interpolation inside the cut bin) -- the reference's
    ``_left_fraction`` verbatim: O(bins), never a per-sample comparison, with
    sub-bin error below one bin's mass."""

    edges = np.asarray(edges, dtype=float)
    n = np.asarray(counts, dtype=float)
    if n.shape != (max(len(edges) - 1, 0),):
        raise ValueError("histogram counts do not align with bin edges")
    total = float(np.sum(n))
    if n.size == 0 or total <= 0:
        return 0.0
    i = int(np.clip(np.searchsorted(edges, threshold, side="right") - 1, -1, len(n)))
    if i < 0:
        return 0.0
    if i >= len(n):
        return 1.0
    frac_in_bin = (threshold - edges[i]) / max(float(edges[i + 1] - edges[i]), 1e-300)
    return float((np.sum(n[:i]) + n[i] * np.clip(frac_in_bin, 0.0, 1.0)) / total)

def _histogram_projection(series_group, bins: int):
    return HistogramBinProjection(
        tuple(series.data.samples for series in series_group), bins=bins
    )

def _grid_histogram_value_range(panels) -> tuple[float, float]:
    """Return Main HistogramCell's one robust shared thumbnail domain."""

    finite_groups = []
    for _layer, _cell, series_group in panels:
        for series in series_group:
            data = series.data
            if not isinstance(data, EvaluatedHistogram):
                raise TypeError("grid histogram range requires histogram cells")
            values = np.asarray(data.samples)
            if values.size:
                finite = values[
                    np.isfinite(values)
                ] if values.dtype.kind not in "iub" else values
                if finite.size:
                    finite_groups.append(finite)
    pooled = (
        np.concatenate(finite_groups)
        if finite_groups
        else np.asarray((0.0, 1.0))
    )
    if pooled.size > 2:
        low, high = (
            float(value)
            for value in np.quantile(pooled, (0.002, 0.998))
        )
    else:
        low, high = float(np.min(pooled)), float(np.max(pooled))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(pooled))
        high = float(np.max(pooled)) + 1.0
    span = high - low
    return low - 0.04 * span, high + 0.04 * span

def _histogram(
    axis,
    series_group,
    *,
    bins: int = DEFAULT_HISTOGRAM_BINS,
    projection=None,
):
    """Draw filled, shared-edge distributions and return their exact bins."""

    bin_projection = (
        _histogram_projection(series_group, bins)
        if projection is None
        else projection
    )
    counts_group = bin_projection.bin_counts
    edges = bin_projection.bin_edges
    artists = _draw_histogram_projection(
        axis,
        series_group,
        counts_group,
        edges,
    )
    return artists, counts_group, edges


def _histogram_vertices(edges, counts) -> np.ndarray:
    """Return Main's exact four-corner filled-bar geometry."""

    edges = np.asarray(edges, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    if edges.ndim != 1 or counts.ndim != 1:
        raise ValueError("histogram edges and counts must be one-dimensional")
    if edges.size != counts.size + 1:
        raise ValueError("histogram counts do not align with bin edges")
    vertices = np.empty((counts.size, 4, 2), dtype=np.float64)
    left = edges[:-1]
    right = edges[1:]
    vertices[:, 0, 0] = left
    vertices[:, 0, 1] = 0.0
    vertices[:, 1, 0] = left
    vertices[:, 1, 1] = counts
    vertices[:, 2, 0] = right
    vertices[:, 2, 1] = counts
    vertices[:, 3, 0] = right
    vertices[:, 3, 1] = 0.0
    return vertices


def _update_histogram_artist(artist, counts, edges) -> None:
    """Update one persistent PolyCollection through the same geometry owner."""

    artist.set_verts(_histogram_vertices(edges, counts))

def _draw_histogram_projection(
    axis,
    series_group,
    counts_group,
    edges,
    *,
    fill_alpha: float = HIST_FILL_ALPHA,
) -> tuple[object, ...]:
    """Draw one panel from already-frozen shared histogram geometry."""

    from matplotlib.collections import PolyCollection

    multiple_series = len(series_group) > 1
    all_boolean = all(
        isinstance(series.data, EvaluatedHistogram)
        and np.issubdtype(series.data.samples.dtype, np.bool_)
        for series in series_group
    )
    artists = []
    for index, (series, counts) in enumerate(
        zip(series_group, counts_group, strict=True)
    ):
        data = series.data
        assert isinstance(data, EvaluatedHistogram)
        label = _series_label(series, include_reductions=multiple_series)
        artist = PolyCollection(
            _histogram_vertices(edges, counts),
            facecolors=(
                PALETTE["hist_fill"]
                if index == 0
                else LINE_CYCLE[index % len(LINE_CYCLE)]
            ),
            alpha=float(fill_alpha),
            label=label,
        )
        axis.add_collection(artist)
        artists.append(artist)
    if all_boolean:
        axis.set_xticks((0, 1), ("false", "true"))
    return tuple(artists)

def _update_histogram_presentation(
    axis,
    state: HistogramDisplayState,
    counts_group,
    edges,
    *,
    fit_overlays=(),
    fit_artists=(),
    threshold_artists=(),
    stats_text=None,
    show_stats: bool,
    threshold_linewidth: float,
    automatic_analysis=_AUTOMATIC_ANALYSIS_NOT_SUPPLIED,
):
    """Draw Distribution's one automatic analysis or a formal Fit projection.

    The bounded automatic two-Gaussian analysis is display-only and consumes
    the already-binned primary series on the render worker.  It never creates
    a Fit artifact or publishes parameters.  ``fit_overlays`` is the
    authoritative exact-source Figure Fit projection; its presence suppresses
    the automatic analysis completely.  Authored thresholds always override
    the automatic display cut without mutating ``state``.
    """

    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    if not isinstance(show_stats, bool):
        raise TypeError("show_stats must be bool")
    counts_group = tuple(np.asarray(item, dtype=np.float64) for item in counts_group)
    if not counts_group:
        raise ValueError("histogram presentation requires at least one count series")
    edges = np.asarray(edges, dtype=np.float64)
    if edges.shape != (counts_group[0].size + 1,) or any(
        counts.shape != counts_group[0].shape for counts in counts_group
    ):
        raise ValueError("histogram counts and edges must share one projection")

    from .render import HistogramFitOverlay

    fit_overlays = tuple(fit_overlays)
    if fit_overlays and (
        len(fit_overlays) != len(counts_group)
        or any(not isinstance(item, HistogramFitOverlay) for item in fit_overlays)
    ):
        raise ValueError("formal histogram Fit overlays must align with series")

    if fit_overlays:
        automatic_analysis = None
    elif automatic_analysis is _AUTOMATIC_ANALYSIS_NOT_SUPPLIED:
        automatic_analysis = _histogram_analysis_cache(
            None,
            counts_group,
            edges,
        )[1]

    fit_artists = tuple(fit_artists)
    required_artist_count = 3 * len(counts_group)
    if fit_artists and len(fit_artists) != required_artist_count:
        for artist in fit_artists:
            artist.remove()
        fit_artists = ()
    if not fit_artists:
        fit_artists = tuple(
            axis.plot((), (), **spec)[0]
            for _series in counts_group
            for spec in bimodal_fit_line_specs()
        )
    for artist in fit_artists:
        artist.set_data((), ())
    if fit_overlays:
        for series_index, overlay in enumerate(fit_overlays):
            if overlay.status is None or not overlay.component_predictions:
                continue
            artists = fit_artists[3 * series_index : 3 * series_index + 3]
            components = overlay.component_predictions
            if len(components) == 1:
                artists[2].set_data(overlay.coordinates, components[0])
            elif len(components) == 3:
                for artist, values in zip(artists, components, strict=True):
                    artist.set_data(overlay.coordinates, values)
            else:
                raise ValueError(
                    "histogram Fit model exposed an unsupported component count"
                )
    elif automatic_analysis.status is FitBatchStatus.CONVERGED:
        for artist, values in zip(
            fit_artists[:3],
            automatic_analysis.component_predictions,
            strict=True,
        ):
            artist.set_data(automatic_analysis.coordinates, values)

    thresholds = tuple(float(value) for value in state.thresholds)
    if (
        not thresholds
        and automatic_analysis is not None
        and automatic_analysis.threshold is not None
    ):
        thresholds = (float(automatic_analysis.threshold),)
    threshold_artists = list(threshold_artists)
    while len(threshold_artists) < len(thresholds):
        threshold_artists.append(
            axis.axvline(
                0.0,
                **threshold_line_kwargs(threshold_linewidth),
            )
        )
    while len(threshold_artists) > len(thresholds):
        threshold_artists.pop().remove()
    for artist, threshold in zip(
        threshold_artists,
        thresholds,
        strict=True,
    ):
        artist.set_xdata((threshold, threshold))

    if show_stats and stats_text is None:
        stats_text = axis.text(
            0.975,
            0.975,
            "",
            transform=axis.transAxes,
            ha="right",
            va="top",
            color=PALETTE["annotation"],
            fontsize=small_fontsize(),
        )
    if show_stats:
        assert stats_text is not None
        if thresholds:
            threshold = thresholds[0]
            left_fraction = _histogram_left_fraction(
                threshold,
                counts_group[0],
                edges,
            )
            stats_text.set_text(
                f"th={threshold:.4g}\n"
                f"L/R={100.0 * left_fraction:.1f}%/"
                f"{100.0 * (1.0 - left_fraction):.1f}%"
            )
        else:
            failed = next(
                (
                    overlay.diagnostic
                    for overlay in fit_overlays
                    if overlay.diagnostic
                ),
                "",
            )
            stats_text.set_text(failed)
    elif stats_text is not None:
        stats_text.set_text("")

    return (
        fit_artists,
        tuple(threshold_artists),
        stats_text,
        thresholds,
    )

__all__ = [
]
