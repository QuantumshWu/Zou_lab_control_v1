"""Internal Matplotlib implementation owner: histogram."""

from __future__ import annotations

import math
import numpy as np
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
    HistogramCountScale,
    HistogramDisplayState,
    HistogramFitMode,
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

from ._mpl_common import (
    _series_label,
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
    _draw_histogram_projection(axis, series_group, counts_group, edges)
    return counts_group, edges

def _draw_histogram_projection(
    axis,
    series_group,
    counts_group,
    edges,
    *,
    fill_alpha: float = HIST_FILL_ALPHA,
) -> None:
    """Draw one panel from already-frozen shared histogram geometry."""

    multiple_series = len(series_group) > 1
    all_boolean = all(
        isinstance(series.data, EvaluatedHistogram)
        and np.issubdtype(series.data.samples.dtype, np.bool_)
        for series in series_group
    )
    for index, (series, counts) in enumerate(
        zip(series_group, counts_group, strict=True)
    ):
        data = series.data
        assert isinstance(data, EvaluatedHistogram)
        label = _series_label(series, include_reductions=multiple_series)
        axis.stairs(
            counts,
            edges,
            fill=True,
            color=(
                PALETTE["hist_fill"]
                if index == 0
                else LINE_CYCLE[index % len(LINE_CYCLE)]
            ),
            alpha=float(fill_alpha),
            label=label,
        )
    if all_boolean:
        axis.set_xticks((0, 1), ("false", "true"))

def _update_histogram_presentation(
    axis,
    state: HistogramDisplayState,
    counts_group,
    edges,
    *,
    analysis_counts_group=None,
    analysis_edges=None,
    fit_artists=(),
    threshold_artists=(),
    stats_text=None,
    show_stats: bool,
    infer_fit_threshold: bool = True,
    threshold_linewidth: float,
):
    """Draw/update the one established histogram fit/cut presentation.

    A full Distribution and every compact Grid cell call this same owner.  The
    caller supplies only whether the tiny thumbnail has room for statistics
    and the established narrow threshold stroke; fit policy, colours, derived
    cut semantics, and text remain single-sourced here.
    """

    if not isinstance(state, HistogramDisplayState):
        raise TypeError("state must be HistogramDisplayState")
    if not isinstance(show_stats, bool):
        raise TypeError("show_stats must be bool")
    if not isinstance(infer_fit_threshold, bool):
        raise TypeError("infer_fit_threshold must be bool")

    from zlc_data.readout_math import confidence_weighted_fidelity

    from .histogram_fit import fit_histogram

    counts_group = tuple(np.asarray(item, dtype=np.float64) for item in counts_group)
    if not counts_group:
        raise ValueError("histogram presentation requires at least one count series")
    edges = np.asarray(edges, dtype=np.float64)
    analysis_counts_group = (
        counts_group
        if analysis_counts_group is None
        else tuple(
            np.asarray(item, dtype=np.float64)
            for item in analysis_counts_group
        )
    )
    if len(analysis_counts_group) != len(counts_group):
        raise ValueError(
            "histogram analysis and visible series counts must align"
        )
    analysis_edges = np.asarray(
        edges if analysis_edges is None else analysis_edges,
        dtype=np.float64,
    )
    counts = analysis_counts_group[0]
    fit = fit_histogram(analysis_edges, counts, state.fit_mode.value)

    fit_artists = tuple(fit_artists)
    if not fit_artists:
        fit_artists = tuple(
            axis.plot((), (), **spec)[0] for spec in bimodal_fit_line_specs()
        )
    if len(fit_artists) != 3:
        raise RuntimeError("histogram presentation requires three fit artists")
    left_artist, right_artist, total_artist = fit_artists
    for artist in fit_artists:
        artist.set_data((), ())
    if fit.valid:
        # The model is fitted from the complete sample projection but drawn
        # only across the robust visible thumbnail domain.
        coordinates = np.linspace(edges[0], edges[-1], 400)
        left, right, total = fit.curves(coordinates)
        visible_bin_width = float(edges[-1] - edges[0]) / max(
            len(edges) - 1,
            1,
        )
        analysis_bin_width = float(
            analysis_edges[-1] - analysis_edges[0]
        ) / max(len(analysis_edges) - 1, 1)
        presentation_scale = visible_bin_width / analysis_bin_width
        if left is not None:
            left_artist.set_data(coordinates, left * presentation_scale)
            right_artist.set_data(coordinates, right * presentation_scale)
        total_artist.set_data(coordinates, total * presentation_scale)

    thresholds = tuple(float(value) for value in state.thresholds)
    if (
        infer_fit_threshold
        and not thresholds
        and fit.separated
        and fit.threshold is not None
    ):
        thresholds = (float(fit.threshold),)
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
                counts,
                analysis_edges,
            )
            fidelity = None
            parameters = fit.bimodal_parameters
            if parameters is not None and fit.separated:
                amp0, mean0, sigma0, amp1, mean1, sigma1 = parameters
                _weighted, raw, _separation = confidence_weighted_fidelity(
                    threshold,
                    mean0,
                    sigma0,
                    abs(amp0 * sigma0),
                    mean1,
                    sigma1,
                    abs(amp1 * sigma1),
                )
                fidelity = float(raw)
            fidelity_text = (
                "fit F=N/A"
                if fidelity is None
                else f"fit F={100.0 * fidelity:.1f}%"
            )
            fit_cut = (
                ""
                if fit.threshold is None
                else f"\nfit cut={fit.threshold:.4g}"
            )
            stats_text.set_text(
                f"th={threshold:.4g}\n{fidelity_text}\n"
                f"L/R={100.0 * left_fraction:.1f}%/"
                f"{100.0 * (1.0 - left_fraction):.1f}%{fit_cut}"
            )
        elif fit.single_parameters is not None:
            _amplitude, mean, sigma = fit.single_parameters
            fitted = np.asarray(
                fit.evaluate(
                    (analysis_edges[:-1] + analysis_edges[1:]) / 2.0
                )
            )
            total_count = float(np.sum(counts))
            out_of_fit = (
                None
                if total_count <= 0.0
                else max(
                    0.0,
                    1.0
                    - float(
                        np.sum(
                            np.minimum(
                                counts,
                                np.clip(fitted, 0.0, None),
                            )
                        )
                    )
                    / total_count,
                )
            )
            suffix = (
                ""
                if out_of_fit is None
                else f"\nout-of-fit={100.0 * out_of_fit:.1f}%"
            )
            stats_text.set_text(
                f"gauss mean={mean:.4g}\nsd={abs(sigma):.3g}{suffix}"
            )
        elif state.fit_mode is HistogramFitMode.NONE:
            stats_text.set_text("")
        else:
            stats_text.set_text("single peak (no split)")
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
