"""Display-only Agg composition for a frozen readout calibration report."""

from __future__ import annotations

from collections.abc import Callable
import gc
from io import BytesIO
import math
from typing import Protocol, runtime_checkable

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from .encoded_raster import (
    EncodedRasterDocument,
    EncodedRasterPage,
    png_raster_size,
)
from .matplotlib_render import release_agg_figure
from .site_map import site_ring_radius
from .render_style import (
    FIT_FAILURE_COLOR,
    HIST_FILL_ALPHA,
    PALETTE,
    SERIES_COLORS,
    SITE_OCCUPANCY_STYLE,
    apply_title,
    render_style_context,
    threshold_line_kwargs,
)


_SCREEN_DPI = 150


@runtime_checkable
class CalibrationModelView(Protocol):
    """Structural renderer input; calibration owns every physical value."""

    label: str
    is_default: bool
    signals: np.ndarray
    signal_validity: np.ndarray
    bin_edges: np.ndarray
    quick_thresholds: np.ndarray
    formal_thresholds: np.ndarray
    runtime_thresholds: np.ndarray
    runtime_threshold_sources: tuple[str, ...]
    feature_validity: np.ndarray
    runtime_usable: np.ndarray
    bright_above: np.ndarray
    model_fidelity: np.ndarray
    heldout_fidelity: np.ndarray
    runtime_model_fidelity_mean: float
    aggregate_fidelity: float
    global_fidelity: float


@runtime_checkable
class CalibrationReportView(Protocol):
    """Structural presentation contract implemented by the domain projection."""

    reference_average: np.ndarray
    reference_average_validity: np.ndarray
    actual_centers_xy: np.ndarray
    expected_centers_xy: np.ndarray | None
    site_validity: np.ndarray
    default_boxes_xywh: np.ndarray
    grid_shape_yx: tuple[int, int]
    site_grid_positions_yx: tuple[tuple[int, int], ...]
    site_labels: tuple[str, ...]
    occupied_labels: np.ndarray
    dark_labels: np.ndarray
    label_validity: np.ndarray
    models: tuple[CalibrationModelView, ...]
    psf_kernels: np.ndarray | None = None
    psf_mode: str | None = None
    psf_fit_ok: np.ndarray | None = None
    psf_sigma_xy: np.ndarray | None = None

def _format_metric(value: float) -> str:
    return "N/A" if not math.isfinite(value) else f"{value:.4f}"


def _new_figure(width: int, height: int) -> Figure:
    figure = Figure(
        figsize=(width / _SCREEN_DPI, height / _SCREEN_DPI),
        dpi=_SCREEN_DPI,
        constrained_layout=True,
    )
    FigureCanvasAgg(figure)
    return figure


def _render_page(
    *,
    width: int,
    height: int,
    builder,
    checkpoint: Callable[[], None],
) -> bytes:
    figure = None
    with render_style_context():
        try:
            checkpoint()
            figure = _new_figure(width, height)
            builder(figure)
            target = BytesIO()
            figure.savefig(target, format="png", dpi=_SCREEN_DPI)
            payload = target.getvalue()
            checkpoint()
        finally:
            if figure is not None:
                release_agg_figure(figure)
            figure = None
            gc.collect()
    png_raster_size(payload)
    return payload


def _build_overview(view: CalibrationReportView, figure: Figure) -> None:
    layout = figure.add_gridspec(2, 2, width_ratios=(1.35, 1.0))
    site_axis = figure.add_subplot(layout[:, 0])
    fidelity_axis = figure.add_subplot(layout[0, 1])
    pooled_axis = figure.add_subplot(layout[1, 1])

    cmap = colormaps["gray"].copy()
    cmap.set_bad(FIT_FAILURE_COLOR)
    image = np.ma.array(
        view.reference_average,
        mask=~view.reference_average_validity,
    )
    site_axis.imshow(image, cmap=cmap, origin="upper", interpolation="nearest")
    radius = site_ring_radius(view.actual_centers_xy)
    valid_color = SITE_OCCUPANCY_STYLE["empty"]["color"]
    for site, ((x, y), valid, box) in enumerate(
        zip(
            view.actual_centers_xy,
            view.site_validity,
            view.default_boxes_xywh,
            strict=True,
        )
    ):
        color = valid_color if valid else FIT_FAILURE_COLOR
        site_axis.add_patch(Circle((x, y), radius, fill=False, color=color, linewidth=0.8))
        x0, y0, width, height = box
        site_axis.add_patch(
            Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=PALETTE["threshold"],
                linewidth=0.35,
                alpha=0.65,
            )
        )
        site_axis.text(x + radius, y - radius, view.site_labels[site], color=color, fontsize=5)
    if view.expected_centers_xy is not None:
        site_axis.scatter(
            view.expected_centers_xy[:, 0],
            view.expected_centers_xy[:, 1],
            marker="+",
            s=15,
            color=SERIES_COLORS[0],
            linewidths=0.6,
            label="declared centers",
        )
        site_axis.legend(loc="lower right")
    site_axis.set_xlabel("camera x [px]")
    site_axis.set_ylabel("camera y [px]")
    apply_title(site_axis, "Reference average | detected sites | runtime boxes")

    sites = np.arange(len(view.site_labels))
    for model, color in zip(view.models, SERIES_COLORS, strict=False):
        values = np.where(model.runtime_usable, model.model_fidelity, np.nan)
        fidelity_axis.plot(sites, values, marker="o", markersize=2.5, label=model.label, color=color)
    fidelity_axis.set_ylim(0.45, 1.01)
    fidelity_axis.set_xlabel("canonical site index")
    fidelity_axis.set_ylabel("Gaussian model-overlap fidelity")
    fidelity_axis.legend(loc="lower right")
    apply_title(fidelity_axis, "Per-site model fidelity (invalid/unused omitted)")

    model = next(item for item in view.models if item.is_default)
    for population, population_mask, color in (
        ("dark", view.dark_labels, PALETTE["dark"]),
        ("bright", view.occupied_labels, PALETTE["bright"]),
    ):
        values = []
        for site in range(len(view.site_labels)):
            if (
                not model.runtime_usable[site]
                or not math.isfinite(model.runtime_thresholds[site])
            ):
                continue
            mask = model.signal_validity[:, site] & view.label_validity[:, site] & population_mask[:, site]
            values.extend(
                (model.signals[mask, site] - model.runtime_thresholds[site]).tolist()
            )
        if values:
            pooled_axis.hist(values, bins=60, color=color, alpha=HIST_FILL_ALPHA, label=population)
    pooled_axis.axvline(0.0, **threshold_line_kwargs(1.2))
    pooled_axis.set_xlabel("signal - runtime threshold")
    pooled_axis.set_ylabel("valid group × site samples")
    pooled_axis.legend(loc="upper right")
    apply_title(pooled_axis, f"Default model pooled diagnostic | {model.label}")


def _build_histogram_grid(
    view: CalibrationReportView,
    model: CalibrationModelView,
    figure: Figure,
) -> None:
    rows, columns = view.grid_shape_yx
    axes = np.asarray(figure.subplots(rows, columns, squeeze=False), dtype=object)
    for site, (row, column) in enumerate(view.site_grid_positions_yx):
        axis = axes[row, column]
        valid = model.signal_validity[:, site] & view.label_validity[:, site]
        dark = model.signals[valid & view.dark_labels[:, site], site]
        bright = model.signals[valid & view.occupied_labels[:, site], site]
        if dark.size:
            axis.hist(dark, bins=model.bin_edges, color=PALETTE["dark"], alpha=HIST_FILL_ALPHA)
        if bright.size:
            axis.hist(bright, bins=model.bin_edges, color=PALETTE["bright"], alpha=HIST_FILL_ALPHA)
        runtime_threshold = model.runtime_thresholds[site]
        if math.isfinite(runtime_threshold):
            axis.axvline(runtime_threshold, **threshold_line_kwargs(0.7))
        model_fidelity = _format_metric(model.model_fidelity[site])
        heldout = _format_metric(model.heldout_fidelity[site])
        threshold_source = model.runtime_threshold_sources[site]
        flags = []
        if not model.feature_validity[site]:
            flags.append("feature-invalid")
        if not model.runtime_usable[site]:
            flags.append("runtime-unused")
        if not model.bright_above[site]:
            flags.append("bad-polarity")
        suffix = " | " + "/".join(flags) if flags else ""
        apply_title(
            axis,
            f"{view.site_labels[site]} | T="
            f"{'N/A' if not math.isfinite(runtime_threshold) else f'{runtime_threshold:.3g}'} "
            f"{threshold_source} | "
            f"M={model_fidelity} | H={heldout}{suffix}",
            size=5.2,
        )
        axis.tick_params(labelsize=4.5)
        if row == rows - 1:
            axis.set_xlabel("signal", fontsize=5)
        if column == 0:
            axis.set_ylabel("shots", fontsize=5)
    figure.suptitle(
        f"{model.label} | stored bins | runtime threshold | valid dark/bright labels",
        fontsize=8,
    )


def _build_psf_grid(view: CalibrationReportView, figure: Figure) -> None:
    assert view.psf_kernels is not None
    rows, columns = view.grid_shape_yx
    axes = np.asarray(figure.subplots(rows, columns, squeeze=False), dtype=object)
    vmax = float(np.max(view.psf_kernels))
    for site, (row, column) in enumerate(view.site_grid_positions_yx):
        axis = axes[row, column]
        axis.imshow(
            view.psf_kernels[site],
            cmap="inferno",
            origin="lower",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
        )
        sx, sy = view.psf_sigma_xy[site]
        state = "fit" if view.psf_fit_ok[site] else "fallback"
        apply_title(
            axis,
            f"{view.site_labels[site]} | sigma=({sx:.2f},{sy:.2f}) | {state}",
            size=5.2,
        )
        axis.set_xticks(())
        axis.set_yticks(())
    captions = {
        "per-site": "Empirical per-site PSF kernels (stored artifact values)",
        "uniform": "Empirical shared uniform PSF kernel (shown per site)",
    }
    try:
        caption = captions[view.psf_mode]
    except KeyError as error:
        raise ValueError("PSF report has an unknown physical feature mode") from error
    figure.suptitle(caption, fontsize=8)


def render_calibration_report(
    view: CalibrationReportView,
    *,
    checkpoint: Callable[[], None] | None = None,
) -> EncodedRasterDocument:
    """Render stored diagnostics without fitting, rethresholding, or reshaping sites."""

    if not isinstance(view, CalibrationReportView):
        raise TypeError("view must be CalibrationReportView")
    check = (lambda: None) if checkpoint is None else checkpoint
    if not callable(check):
        raise TypeError("checkpoint must be callable or None")
    pages = []
    payload = _render_page(
        width=1800,
        height=1100,
        builder=lambda figure: _build_overview(view, figure),
        checkpoint=check,
    )
    pages.append(EncodedRasterPage("overview", "Overview", payload))

    rows, columns = view.grid_shape_yx
    grid_width = max(1200, 260 * columns)
    grid_height = max(800, 220 * rows)
    for model in view.models:
        payload = _render_page(
            width=grid_width,
            height=grid_height,
            builder=lambda figure, selected=model: _build_histogram_grid(view, selected, figure),
            checkpoint=check,
        )
        pages.append(EncodedRasterPage(f"hist-{model.label}", model.label, payload))
    if view.psf_kernels is not None:
        payload = _render_page(
            width=grid_width,
            height=grid_height,
            builder=lambda figure: _build_psf_grid(view, figure),
            checkpoint=check,
        )
        pages.append(EncodedRasterPage("psf-kernels", "PSF kernels", payload))

    model_summaries = []
    for model in view.models:
        default = " default" if model.is_default else ""
        model_summaries.append(
            f"{model.label}{default}: model="
            f"{_format_metric(model.runtime_model_fidelity_mean)}, "
            f"held-out={_format_metric(model.aggregate_fidelity)}, "
            f"global={_format_metric(model.global_fidelity)}, "
            f"usable={int(np.count_nonzero(model.runtime_usable))}/{len(view.site_labels)}"
        )
    summary = (
        f"{len(view.site_labels)} sites · {len(view.models)} models\n"
        + "\n".join(model_summaries)
    )
    return EncodedRasterDocument(summary, tuple(pages))


__all__ = [
    "CalibrationModelView",
    "CalibrationReportView",
    "render_calibration_report",
]
