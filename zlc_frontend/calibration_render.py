"""Display-only Agg composition for a frozen readout calibration report."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import gc
from io import BytesIO
import math

import numpy as np
from matplotlib import colormaps
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from zlc_storage import canonical_text

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
def _array(value, dtype, shape, field_name, *, finite: bool = False):
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{field_name} must have shape {shape}, got {result.shape}")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{field_name} must be finite")
    if result.flags.writeable:
        result = np.array(result, copy=True)
        result.setflags(write=False)
    return result


def _metric(value: object, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{field_name} must be numeric")
    result = float(value)
    if math.isinf(result):
        raise ValueError(f"{field_name} must be finite or NaN")
    return result


@dataclass(frozen=True, eq=False)
class CalibrationModelView:
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
    aggregate_fidelity: float
    global_fidelity: float

    def __post_init__(self) -> None:
        canonical_text(self.label, "model label")
        if type(self.is_default) is not bool:
            raise TypeError("is_default must be bool")
        signals = np.asarray(self.signals, dtype="<f8")
        if signals.ndim != 2:
            raise ValueError("model signals must have shape (groups, sites)")
        shape = signals.shape
        sites = shape[1]
        object.__setattr__(self, "signals", _array(signals, "<f8", shape, "signals"))
        object.__setattr__(
            self,
            "signal_validity",
            _array(self.signal_validity, "bool", shape, "signal_validity"),
        )
        edges = np.asarray(self.bin_edges, dtype="<f8")
        if edges.ndim != 1 or edges.size < 3 or not np.all(np.diff(edges) > 0):
            raise ValueError("bin_edges must be one strictly increasing axis")
        object.__setattr__(self, "bin_edges", _array(edges, "<f8", edges.shape, "bin_edges", finite=True))
        for name, dtype in (
            ("quick_thresholds", "<f8"),
            ("formal_thresholds", "<f8"),
            ("runtime_thresholds", "<f8"),
            ("feature_validity", "bool"),
            ("runtime_usable", "bool"),
            ("bright_above", "bool"),
            ("model_fidelity", "<f8"),
            ("heldout_fidelity", "<f8"),
        ):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), dtype, (sites,), name),
            )
        sources = tuple(self.runtime_threshold_sources)
        if len(sources) != sites or any(
            source not in ("formal", "quick-fallback") for source in sources
        ):
            raise ValueError(
                "runtime_threshold_sources must identify every site as formal "
                "or quick-fallback"
            )
        if np.any(self.runtime_usable & ~np.isfinite(self.runtime_thresholds)):
            raise ValueError("runtime-usable sites require finite thresholds")
        if np.any(self.runtime_usable & ~self.feature_validity):
            raise ValueError("runtime-usable sites must be feature-valid")
        object.__setattr__(self, "runtime_threshold_sources", sources)
        object.__setattr__(
            self,
            "aggregate_fidelity",
            _metric(self.aggregate_fidelity, "aggregate_fidelity"),
        )
        object.__setattr__(
            self,
            "global_fidelity",
            _metric(self.global_fidelity, "global_fidelity"),
        )

@dataclass(frozen=True, eq=False)
class CalibrationReportView:
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
    psf_caption: str | None = None
    psf_fit_ok: np.ndarray | None = None
    psf_sigma_xy: np.ndarray | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.reference_average, dtype="<f8")
        if image.ndim != 2:
            raise ValueError("reference_average must be a two-dimensional image")
        image = _array(image, "<f8", image.shape, "reference_average", finite=True)
        validity = _array(
            self.reference_average_validity,
            "bool",
            image.shape,
            "reference_average_validity",
        )
        centers = np.asarray(self.actual_centers_xy, dtype="<f8")
        if centers.ndim != 2 or centers.shape[1:] != (2,):
            raise ValueError("actual_centers_xy must have shape (sites, 2)")
        centers = _array(centers, "<f8", centers.shape, "actual_centers_xy", finite=True)
        sites = centers.shape[0]
        raw_grid = tuple(self.grid_shape_yx)
        if (
            len(raw_grid) != 2
            or any(type(item) is not int or item <= 0 for item in raw_grid)
            or math.prod(raw_grid) != sites
        ):
            raise ValueError("grid_shape_yx must be two positive axes covering every site")
        labels = tuple(self.site_labels)
        if len(labels) != sites:
            raise ValueError("site_labels must contain one label per site")
        for label in labels:
            canonical_text(label, "site label")
        positions = tuple(tuple(position) for position in self.site_grid_positions_yx)
        if (
            len(positions) != sites
            or any(
                len(position) != 2
                or any(type(index) is not int for index in position)
                for position in positions
            )
            or set(positions)
            != {
                (row, column)
                for row in range(raw_grid[0])
                for column in range(raw_grid[1])
            }
        ):
            raise ValueError(
                "site_grid_positions_yx must be a bijection over the declared grid"
            )
        expected = self.expected_centers_xy
        if expected is not None:
            expected = _array(
                expected,
                "<f8",
                centers.shape,
                "expected_centers_xy",
                finite=True,
            )
        models = tuple(self.models)
        if (
            not models
            or any(not isinstance(model, CalibrationModelView) for model in models)
            or len({model.label for model in models}) != len(models)
            or sum(model.is_default for model in models) != 1
        ):
            raise ValueError("models must be unique and name exactly one default")
        group_shape = models[0].signals.shape
        if group_shape[1] != sites or any(model.signals.shape != group_shape for model in models):
            raise ValueError("every model must use the same declared group/site axes")
        for name in ("occupied_labels", "dark_labels", "label_validity"):
            object.__setattr__(
                self,
                name,
                _array(getattr(self, name), "bool", group_shape, name),
            )
        if np.any(self.occupied_labels & self.dark_labels) or np.any(
            (self.occupied_labels | self.dark_labels) & ~self.label_validity
        ):
            raise ValueError("reference population labels are inconsistent")
        kernels = self.psf_kernels
        fit_ok = self.psf_fit_ok
        sigma = self.psf_sigma_xy
        if kernels is None:
            if self.psf_caption is not None or fit_ok is not None or sigma is not None:
                raise ValueError("PSF fit diagnostics require empirical kernels")
        else:
            canonical_text(self.psf_caption, "PSF caption")
            kernels = np.asarray(kernels, dtype="<f8")
            if kernels.ndim != 3 or kernels.shape[0] != sites:
                raise ValueError("psf_kernels must have shape (sites, y, x)")
            kernels = _array(kernels, "<f8", kernels.shape, "psf_kernels", finite=True)
            fit_ok = _array(fit_ok, "bool", (sites,), "psf_fit_ok")
            sigma = _array(sigma, "<f8", (sites, 2), "psf_sigma_xy")
        object.__setattr__(self, "reference_average", image)
        object.__setattr__(self, "reference_average_validity", validity)
        object.__setattr__(self, "actual_centers_xy", centers)
        object.__setattr__(self, "expected_centers_xy", expected)
        object.__setattr__(self, "site_validity", _array(self.site_validity, "bool", (sites,), "site_validity"))
        object.__setattr__(self, "default_boxes_xywh", _array(self.default_boxes_xywh, "<i8", (sites, 4), "default_boxes_xywh"))
        object.__setattr__(self, "grid_shape_yx", raw_grid)
        object.__setattr__(self, "site_grid_positions_yx", positions)
        object.__setattr__(self, "site_labels", labels)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "psf_kernels", kernels)
        object.__setattr__(self, "psf_fit_ok", fit_ok)
        object.__setattr__(self, "psf_sigma_xy", sigma)

def _mean_metric(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


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
    figure.suptitle(view.psf_caption, fontsize=8)


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
            f"{_format_metric(_mean_metric(model.model_fidelity[model.runtime_usable]))}, "
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
