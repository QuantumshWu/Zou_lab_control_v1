"""Decoupled live/static plotting classes for Jupyter experiment front-ends."""

from __future__ import annotations

from math import ceil, erf, sqrt
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

from .canvas import FigureSpec, configure_canvas, create_axes_fixed, display_figure, new_figure, split_axes_horizontally
from .selectors import DragHLine, DragVLine, InteractionBundle, PlotState, attach_interaction
from .style import apply_style
from .ticks import apply_smart_ticks


DEFAULT_COLORS = ["grey", "skyblue", "tab:blue", "tab:orange"]
PULSE_COLORS = [
    "#5D7583",
    "#C37D5A",
    "#6F8D73",
    "#A66E87",
    "#7A6FA4",
    "#B5A262",
    "#5E9A9A",
    "#9A765E",
    "#7890B5",
    "#8B8B8B",
    "#B97878",
    "#679174",
]


def _as_data_x(data_x) -> np.ndarray:
    x = np.asarray(data_x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("data_x must be a 1D or 2D array.")
    return x


def _as_data_y(data_y, n: int) -> np.ndarray:
    if data_y is None:
        return np.full((n, 1), np.nan, dtype=float)
    y = np.asarray(data_y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    if y.ndim != 2:
        raise ValueError("data_y must be a 1D or 2D array.")
    if len(y) != n:
        raise ValueError("data_y and data_x must have the same length.")
    return y


def _square_extent(extent: Sequence[float]) -> list[float]:
    left, right, bottom, upper = extent
    width = right - left
    height = bottom - upper
    if width >= height:
        pad = (width - height) / 2
        bottom += pad
        upper -= pad
    else:
        pad = (height - width) / 2
        left -= pad
        right += pad
    return [left, right, bottom, upper]


def _float2str_eng(x: float, length: int = 5) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{float(x):.{max(0, length - 1)}g}"


def _positive_float(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be finite, not a boolean.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and > 0.")
    return result


def _with_title_margin(spec: FigureSpec, title: str, margins_supplied: bool) -> FigureSpec:
    if not title or margins_supplied:
        return spec
    left, right, bottom, top = spec.margins_px
    return FigureSpec(data_px=spec.data_px, margins_px=(left, right, bottom, max(top, 70)), dpi=spec.dpi)


def _update_verts(bins, counts, verts, mode: str = "horizontal") -> None:
    left = bins[:-1]
    right = bins[1:]
    if mode == "horizontal":
        verts[:, 0, 0] = 0
        verts[:, 0, 1] = left
        verts[:, 1, 0] = counts
        verts[:, 1, 1] = left
        verts[:, 2, 0] = counts
        verts[:, 2, 1] = right
        verts[:, 3, 0] = 0
        verts[:, 3, 1] = right
    else:
        verts[:, 0, 0] = left
        verts[:, 0, 1] = 0
        verts[:, 1, 0] = left
        verts[:, 1, 1] = counts
        verts[:, 2, 0] = right
        verts[:, 2, 1] = counts
        verts[:, 3, 0] = right
        verts[:, 3, 1] = 0


class BaseLivePlot:
    """Base class shared by notebook live plotters.

    The experiment side mutates ``data_y`` or calls ``update_point``/``roll``.
    The plotter side owns figure lifecycle, layout, selectors, and data handles.
    """

    plot_type = "base"

    def __init__(
        self,
        data_x=np.arange(100),
        data_y=None,
        *,
        labels: Sequence[str] = ("X", "Y", "Z"),
        update_time: float = 0.1,
        fig: plt.Figure | None = None,
        relim_mode: str = "normal",
        spec: FigureSpec | None = None,
        data_px: tuple[int, int] | None = None,
        margins_px: tuple[int, int, int, int] | None = None,
        smart_ticks: bool = True,
        interactions: bool = True,
        title: str | None = None,
        name: str = "figure",
        info: Mapping[str, Any] | None = None,
        unit: str | None = None,
    ):
        self.labels = list(labels)
        self.xlabel = self.labels[0]
        self.ylabel = self.labels[1] if len(self.labels) > 1 else "Y"
        self.zlabel = self.labels[-1]
        self.title = "" if title is None else str(title)
        self.data_x = _as_data_x(data_x)
        self.data_y = _as_data_y(data_y, len(self.data_x))
        self.points_total = len(self.data_x)
        self.points_done = self._infer_points_done()
        self.repeat_cur = 1
        self.repeat_label = 1
        self.update_time = _positive_float(update_time, "update_time")
        self.relim_mode = relim_mode
        self.fig = fig
        self.ax = None
        self.axes = None
        self.tools = InteractionBundle()
        self.area = None
        self.cross = None
        self.zoom = None
        self.drag = None
        self.lines = []
        self.smart_ticks = bool(smart_ticks)
        self.interactions = bool(interactions)
        self.name = name
        self.info = dict(info or {})
        self.unit = unit or self.info.get("unit") or "1"
        margins_supplied = margins_px is not None
        self.spec = spec or FigureSpec()
        if data_px is not None or margins_px is not None:
            self.spec = FigureSpec(data_px=data_px or self.spec.data_px, margins_px=margins_px or self.spec.margins_px, dpi=self.spec.dpi)
        self.spec = _with_title_margin(self.spec, self.title, margins_supplied)
        self.ylim_min = 0.0
        self.ylim_max = 1.0
        self._shown = False
        self.data_figure = None
        self._stopped = False

    def _infer_points_done(self) -> int:
        finite = np.isfinite(self.data_y[:, 0])
        return int(np.count_nonzero(finite))

    def show(self, *, display: bool = True):
        """Initialize and optionally display the figure."""
        apply_style({"figure.dpi": self.spec.dpi})
        if self.fig is None:
            self.fig = new_figure(spec=self.spec)
        else:
            configure_canvas(self.fig)
            self.fig.clear()
        self.ax = create_axes_fixed(self.fig, self.spec.data_px, self.spec.margins_px)
        self.axes = self.ax
        if self.smart_ticks:
            apply_smart_ticks(self.ax)
        self.init_core()
        self._apply_title()
        self._install_state()
        if self.interactions:
            self._attach_interactions()
        self._shown = True
        if display:
            display_figure(self.fig)
        else:
            self.fig.canvas.draw()
        return self

    def watch(
        self,
        *,
        interval: float | None = None,
        stop_when_full: bool = True,
        done=None,
        points_done=None,
        copy: bool = False,
        lock=None,
    ):
        """Start frontend-side refresh of the shared ``data_y`` array."""
        from ._watcher import ArrayWatcher, _strict_bool

        if not self._shown:
            self.show()
        if getattr(getattr(self, "_watcher", None), "running", False):
            self._watcher.stop()
        self._stopped = False
        static_done = None if callable(done) or done is None else _strict_bool(done, "done")

        def is_done():
            if callable(done):
                external_done = _strict_bool(done(), "done callback return")
            elif done is None:
                external_done = False
            else:
                external_done = static_done
            return self._stopped or external_done

        self._watcher = ArrayWatcher(
            self,
            self.data_y,
            interval=interval,
            done=is_done,
            points_done=points_done,
            stop_when_full=stop_when_full,
            auto_show=False,
            copy=copy,
            lock=lock,
        )
        self._watcher.start()
        return self

    def refresh(self, *, draw: bool = True):
        """Refresh artists from the currently shared arrays."""
        return self.update(draw=draw)

    def stop(self):
        """Stop frontend refresh and mark any watched stream as done."""
        self._stopped = True
        if getattr(self, "_watcher", None) is not None:
            self._watcher.stop()
        return self

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type=self.plot_type)

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def _apply_title(self) -> None:
        if self.title and self.ax is not None:
            self.ax.set_title(
                self.title,
                fontsize=matplotlib.rcParams["axes.labelsize"],
                pad=max(float(matplotlib.rcParams.get("axes.titlepad", 1.5)), 2.5),
            )

    def init_core(self) -> None:
        raise NotImplementedError

    def update_core(self) -> None:
        raise NotImplementedError

    def update(self, data_y=None, *, points_done: int | None = None, repeat_cur: int | None = None, draw: bool = True):
        """Refresh artists from current or newly supplied data."""
        if not self._shown:
            self.show(display=False)
        if data_y is not None:
            self.data_y = _as_data_y(data_y, len(self.data_x))
        self.points_done = self._infer_points_done() if points_done is None else int(points_done)
        self.repeat_cur = self.repeat_cur if repeat_cur is None else int(repeat_cur)
        self.update_core()
        self._install_state()
        if draw:
            self.draw()
        return self

    def draw(self) -> None:
        self.fig.canvas.draw_idle()
        try:
            self.fig.canvas.flush_events()
        except Exception:
            pass

    def update_point(self, index: int, value, *, mode: str = "replace", repeat_cur: int | None = None, draw: bool = True):
        """Update one point using a measurement-like update mode."""
        if repeat_cur is not None:
            self.repeat_cur = int(repeat_cur)
        value = np.asarray(value, dtype=float).reshape(-1)
        if mode == "replace":
            self.data_y[index, : len(value)] = value
        elif mode == "add":
            if np.isnan(self.data_y[index, 0]):
                self.data_y[index, : len(value)] = value
            else:
                self.data_y[index, : len(value)] += value
        elif mode == "create":
            start = (self.repeat_cur - 1) * len(value)
            stop = start + len(value)
            if stop > self.data_y.shape[1]:
                extra = stop - self.data_y.shape[1]
                self.data_y = np.pad(self.data_y, ((0, 0), (0, extra)), constant_values=np.nan)
            self.data_y[index, start:stop] = value
        else:
            raise ValueError("mode must be replace, add, or create.")
        self.points_done = max(self.points_done, int(index) + 1)
        return self.update(points_done=self.points_done, repeat_cur=self.repeat_cur, draw=draw)

    def roll(self, value, *, draw: bool = True):
        """Roll newest data to the left/front, matching the old live() mode."""
        value = np.asarray(value, dtype=float).reshape(-1)
        self.data_y[:] = np.roll(self.data_y, shift=1, axis=0)
        self.data_y[0, : len(value)] = value
        self.points_done = min(self.points_total, self.points_done + 1)
        return self.update(points_done=self.points_done, draw=draw)

    def relim(self, values=None) -> bool:
        vals = np.asarray(self.data_y[:, 0] if values is None else values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return False
        max_y = float(np.nanmax(vals))
        min_y = float(np.nanmin(vals))
        if min_y < 0:
            self.relim_mode = "tight"
        data_range = (max_y - min_y) if self.relim_mode == "tight" else (max_y - 0)
        if data_range == 0:
            data_range = abs(max_y) if max_y else 1.0

        old = (self.ylim_min, self.ylim_max)
        if self.relim_mode == "normal":
            self.ylim_min = 0
            self.ylim_max = max_y * 1.2 if max_y else 1
        else:
            self.ylim_min = min_y - 0.1 * data_range
            self.ylim_max = max_y + 0.1 * data_range
        return old != (self.ylim_min, self.ylim_max)

    def after_plot(self):
        """Create and attach a DataFigure handle."""
        return self.to_data_figure()

    def to_data_figure(self):
        from .data_figure import DataFigure

        self.data_figure = DataFigure(self)
        return self.data_figure

    def save(self, path: str = "", **kwargs):
        return self.to_data_figure().save(path, **kwargs)


class Live1D(BaseLivePlot):
    """Live 1D line plot with fixed-size notebook layout."""

    plot_type = "1D"

    def init_core(self) -> None:
        self.lines = self.ax.plot(self.data_x[:, 0], self.data_y, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(self.data_x[0, 0], self.data_x[-1, 0])
        self.relim()
        self.ax.set_ylim(self.ylim_min, self.ylim_max)

    def update_core(self) -> None:
        if self.repeat_label != self.repeat_cur:
            self.ylabel = f"{self.labels[1]} x{self.repeat_cur}" if self.repeat_cur != 1 else self.labels[1]
            self.repeat_label = self.repeat_cur
            self.ax.set_ylabel(self.ylabel)
        self.relim()
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        for i, line in enumerate(self.lines):
            if i < self.data_y.shape[1]:
                line.set_data(self.data_x[:, 0], self.data_y[:, i])

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="1D", x_array=self.data_x[:, 0], y_array=self.data_y)


class LiveLiveDis(Live1D):
    """Live rolling trace plus side distribution and Gaussian width monitor."""

    plot_type = "live-distribution"

    def init_core(self) -> None:
        self.ax, self.axdis = split_axes_horizontally(self.fig, self.ax, [0.825, 0.15], [0.025])
        self.axes = self.ax
        self.lines = self.ax.plot(self.data_x[:, 0], self.data_y, alpha=1)
        for i, line in enumerate(self.lines):
            line.set_color(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(np.nanmin(self.data_x[:, 0]), np.nanmax(self.data_x[:, 0]))
        self.relim()
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.axdis.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        self.axdis.tick_params(axis="both", which="both", bottom=False, top=False)
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.n_bins = int(max(3, min(self.points_total // 4, 50)))
        self.n, self.bins = self._hist()
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors="grey")
        self.axdis.add_collection(self.poly)
        self.counts_max = max(10, int(np.nanmax(self.n) + 5 if self.n.size else 10))
        self.axdis.set_xlim(0, self.counts_max)
        (self.gauss_line,) = self.axdis.plot([], [], color="orange", alpha=1)
        self.text = None
        self.fit_text = None

    def _hist(self):
        vals = self.data_y[: max(self.points_done, 1), 0]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.array([0.0])
        return np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

    @staticmethod
    def _gauss_func(x, amp, mu, sigma):
        return amp * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

    def _update_gauss_fit(self):
        mask = self.n > 0
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        x = centers[mask]
        y = self.n[mask]
        if len(x) < 3 or np.ptp(x) == 0:
            return
        try:
            popt, _ = curve_fit(
                self._gauss_func,
                x,
                y,
                p0=[np.max(y), np.mean(x), max(np.ptp(x) / 4, 1e-12)],
                bounds=([0, np.min(x), max(np.ptp(x) / 100, 1e-12)], [max(np.max(y) * 4, 1), np.max(x), max(np.ptp(x) * 10, 1e-12)]),
            )
        except Exception:
            return
        x_fit = np.linspace(self.ylim_min, self.ylim_max, 100)
        self.gauss_line.set_data(self._gauss_func(x_fit, *popt), x_fit)
        if popt[1] <= 0:
            label = r"$\sigma$=0"
        else:
            ratio = popt[2] / np.sqrt(popt[1])
            label = rf"$\sigma$={ratio:.2f}$\sqrt{{\mu}}$"
        if self.fit_text is None:
            self.fit_text = self.axdis.text(
                0.5,
                1.005,
                label,
                transform=self.axdis.transAxes,
                color="orange",
                ha="center",
                va="bottom",
                fontsize=matplotlib.rcParams["legend.fontsize"],
            )
        else:
            self.fit_text.set_text(label)

    def update_core(self) -> None:
        super().update_core()
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n, self.bins = self._hist()
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly.set_verts(self.verts)
        counts_max = max(10, int(max(np.nanmax(self.n) + 5, np.nanmax(self.n) * 1.5)))
        self.axdis.set_xlim(0, counts_max)
        newest = self.data_y[0, 0]
        if np.isfinite(newest):
            label = f"{newest:.6g}"
            if self.text is None:
                self.text = self.ax.text(
                    0.9,
                    1.005,
                    label,
                    transform=self.ax.transAxes,
                    color="grey",
                    ha="right",
                    va="bottom",
                    fontsize=matplotlib.rcParams["legend.fontsize"],
                )
            else:
                self.text.set_text(label)
        self._update_gauss_fit()

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="1D", x_array=self.data_x[:, 0], y_array=self.data_y, axdis=self.axdis)


class Live2DDis(BaseLivePlot):
    """Live 2D image with side distribution, colorbar, and draggable clim."""

    plot_type = "2D"

    def __init__(self, *args, cmap: str = "inferno", bad_color: str = "white", square: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_x.shape[1] != 2:
            raise ValueError("Live2DDis requires data_x with shape (N, 2).")
        self.cmap = cmap
        self.bad_color = bad_color
        self.square = square

    def fill_grid(self) -> np.ndarray:
        grid = np.full(self.data_shape, np.nan)
        for (x, y), z in zip(self.data_x, self.data_y[:, 0]):
            ix = int(np.searchsorted(self.x_array, x))
            iy = int(np.searchsorted(self.y_array, y))
            if 0 <= iy < grid.shape[0] and 0 <= ix < grid.shape[1]:
                grid[iy, ix] = z
        return grid

    def init_core(self) -> None:
        self.ax, self.axdis, self.cax = split_axes_horizontally(self.fig, self.ax, [0.75, 0.1, 0.1], [0.025, 0.025])
        self.axes = self.ax
        self.x_array = np.unique(self.data_x[:, 0])
        self.y_array = np.unique(self.data_x[:, 1])
        self.data_shape = (len(self.y_array), len(self.x_array))
        self.grid = self.fill_grid()
        try:
            cmap = matplotlib.colormaps[self.cmap].copy()
        except Exception:
            cmap = plt.get_cmap(self.cmap).copy()
        cmap.set_bad(self.bad_color)
        dx = 0.5 * (self.x_array[-1] - self.x_array[0]) / len(self.x_array) if len(self.x_array) > 1 else 0.5
        dy = 0.5 * (self.y_array[-1] - self.y_array[0]) / len(self.y_array) if len(self.y_array) > 1 else 0.5
        self.extent = [self.x_array[0] - dx, self.x_array[-1] + dx, self.y_array[-1] + dy, self.y_array[0] - dy]
        self.image = self.ax.imshow(self.grid, cmap=cmap, extent=self.extent, interpolation="none")
        self.lines = [self.image]
        self.ax.set_anchor("W")
        self.ax.set_aspect("equal", adjustable="box")
        self.extents_square = _square_extent(self.extent) if self.square else list(self.extent)
        self.ax.set_xlim(self.extents_square[0], self.extents_square[1])
        self.ax.set_ylim(self.extents_square[2], self.extents_square[3])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.cbar = self.fig.colorbar(self.image, cax=self.cax)
        self.cbar.set_label(self.zlabel)
        self._init_distribution()

    def _finite_values(self):
        vals = self.data_y[: self.points_done, 0]
        return vals[np.isfinite(vals)]

    def _init_distribution(self) -> None:
        vals = self._finite_values()
        if vals.size:
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
        else:
            y_min, y_max = 0.0, 1.0
        span = y_max - y_min
        if span == 0:
            span = abs(y_max) if y_max else 1.0
        self.ylim_min = y_min - 0.1 * span
        self.ylim_max = y_max + 0.1 * span
        self.image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n_bins = int(max(8, min(max(self.points_total, 1) // 4, 50)))
        self.n, self.bins = np.histogram(vals if vals.size else [0], bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors="grey")
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, max(10, int(np.max(self.n) + 5)))
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
        self.axdis.tick_params(axis="y", which="both", left=True, right=False, labelleft=False, labelright=False)
        self.line_min = self.axdis.axhline(y_min, color="grey", linewidth=matplotlib.rcParams["legend.fontsize"] / 2, alpha=0.3)
        self.line_max = self.axdis.axhline(y_max, color="grey", linewidth=matplotlib.rcParams["legend.fontsize"] / 2, alpha=0.3)
        cmap = self.image.get_cmap()
        self.line_l = self.axdis.axhline(self.ylim_min, color=cmap(0.0), linewidth=matplotlib.rcParams["legend.fontsize"] / 2)
        self.line_h = self.axdis.axhline(self.ylim_max, color=cmap(0.95), linewidth=matplotlib.rcParams["legend.fontsize"] / 2)
        self.cax.set_yticks([y_min, y_max])
        self.cax.set_yticklabels([_float2str_eng(v, length=5) for v in [y_min, y_max]])
        self.drag = DragHLine(self.line_l, self.line_h, self.update_clim, self.axdis)

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax, drag=self.drag, axdis=self.axdis, cax=self.cax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def update_clim(self) -> None:
        self.image.set_clim(float(self.line_l.get_ydata()[0]), float(self.line_h.get_ydata()[0]))

    def update_core(self) -> None:
        self.grid = self.fill_grid()
        self.image.set_array(self.grid)
        vals = self._finite_values()
        if vals.size:
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            span = y_max - y_min
            if span == 0:
                span = abs(y_max) if y_max else 1.0
            self.ylim_min = y_min - 0.1 * span
            self.ylim_max = y_max + 0.1 * span
            self.axdis.set_ylim(self.ylim_min, self.ylim_max)
            self.n, self.bins = np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
            _update_verts(self.bins, self.n, self.verts, mode="horizontal")
            self.poly.set_verts(self.verts)
            self.axdis.set_xlim(0, max(10, int(max(np.max(self.n) + 5, np.max(self.n) * 1.5))))
            self.line_min.set_ydata([y_min, y_min])
            self.line_max.set_ydata([y_max, y_max])
            if float(self.line_l.get_ydata()[0]) > y_min or float(self.line_h.get_ydata()[0]) < y_max:
                self.line_l.set_ydata([self.ylim_min, self.ylim_min])
                self.line_h.set_ydata([self.ylim_max, self.ylim_max])
                self.update_clim()
            self.cax.set_yticks([y_min, y_max])
            self.cax.set_yticklabels([_float2str_eng(v, length=5) for v in [y_min, y_max]])

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(
            plot_type="2D",
            x_array=self.x_array,
            y_array=self.y_array,
            grid=self.grid,
            axdis=self.axdis,
            cax=self.cax,
            extents_square=self.extents_square,
            bad_color=self.bad_color,
        )


def _pulse_attr(row, name: str, default=None):
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _pulse_rows(sequence) -> list[dict[str, Any]]:
    if hasattr(sequence, "effective_pulses") and callable(sequence.effective_pulses):
        raw = sequence.effective_pulses()
    elif hasattr(sequence, "table") and callable(sequence.table):
        raw = sequence.table()
    elif isinstance(sequence, Mapping):
        raw = sequence.get("pulses", [])
    else:
        raw = sequence
    rows: list[dict[str, Any]] = []
    for item in raw:
        channel = str(_pulse_attr(item, "channel", "pulse"))
        start = float(_pulse_attr(item, "start", 0.0))
        duration = float(_pulse_attr(item, "duration", 0.0))
        value = int(_pulse_attr(item, "value", 1))
        name = str(_pulse_attr(item, "name", ""))
        if not np.isfinite(start) or not np.isfinite(duration) or duration < 0:
            raise ValueError("pulse start and duration must be finite, with duration >= 0.")
        rows.append({"channel": channel, "start": start, "duration": duration, "stop": start + duration, "value": value, "name": name})
    rows.sort(key=lambda row: (row["start"], row["channel"]))
    return rows


def _pulse_time_unit(span_s: float) -> tuple[float, str]:
    span = abs(float(span_s))
    if span < 1e-6:
        return 1e-9, "ns"
    if span < 1e-3:
        return 1e-6, "us"
    if span < 1.0:
        return 1e-3, "ms"
    return 1.0, "s"


def _label_with_unit(label: str, unit: str) -> str:
    if label.endswith(")") and "(" in label:
        return f"{label[: label.rfind('(')].rstrip()} ({unit})"
    return f"{label} ({unit})"


def pulse_plot_channels(
    sequence,
    *,
    channels: Sequence[str] | None = None,
    include_always_off: bool = False,
    minimum: int = 1,
) -> list[str]:
    """Return the channels a pulse plot should display."""

    rows = _pulse_rows(sequence)
    if channels is None:
        if hasattr(sequence, "channels"):
            channels = list(getattr(sequence, "channels"))
        else:
            channels = sorted({row["channel"] for row in rows})
    ordered = [str(channel) for channel in (channels or [])]
    if include_always_off:
        return ordered or ["pulse"]
    active = {row["channel"] for row in rows if row["value"] and row["duration"] > 0}
    visible = [channel for channel in ordered if channel in active]
    if not visible and ordered:
        visible = ordered[: max(0, min(int(minimum), len(ordered)))]
    return visible or ["pulse"]


def pulse_plot_spec(
    channel_count: int,
    *,
    data_width_px: int = 520,
    rows_per_block: int = 10,
    block_height_px: int = 360,
    margins_px: tuple[int, int, int, int] = (110, 90, 100, 50),
    dpi: int = 300,
) -> FigureSpec:
    """FigureSpec sized so pulse rows stay legible beyond 10 channels."""

    rows_per_block = max(1, int(rows_per_block))
    chunks = max(1, ceil(max(1, int(channel_count)) / rows_per_block))
    return FigureSpec(data_px=(int(data_width_px), int(block_height_px) * chunks), margins_px=margins_px, dpi=int(dpi))


def pulse_repeat_notation(
    state_or_start=None,
    repeat_end: int | None = None,
    repeat_count: int | None = None,
    *,
    default_forever: bool = True,
) -> str:
    """Return a compact repeat label for pulse plots."""

    if hasattr(state_or_start, "repeat_start"):
        repeat_start = getattr(state_or_start, "repeat_start", None)
        repeat_end = getattr(state_or_start, "repeat_end", None)
        repeat_count = getattr(state_or_start, "repeat_count", None)
        periods = list(getattr(state_or_start, "periods", ()))
    else:
        repeat_start = state_or_start
        periods = []
    if repeat_start is None or repeat_end is None:
        return "repeat ∞" if default_forever else ""
    repeat_count = 1 if repeat_count is None else int(repeat_count)
    inner = f"P{int(repeat_start) + 1}-P{int(repeat_end) + 1} x{repeat_count}"
    if periods and (int(repeat_start) != 0 or int(repeat_end) != len(periods) - 1):
        return f"repeat ∞ + {inner}"
    return f"repeat {inner}"


def _pulse_period_starts_ns(periods, *, x_ns: float | None = None, time_step_ns: float | None = None) -> list[float]:
    starts_ns = [0.0]
    for period in periods:
        duration_ns = period.duration_ns(x_ns=0.0 if x_ns is None else x_ns, time_step_ns=time_step_ns)
        starts_ns.append(starts_ns[-1] + float(duration_ns))
    return starts_ns


def pulse_repeat_marker(
    state_or_periods=None,
    *,
    repeat_start: int | None = None,
    repeat_end: int | None = None,
    repeat_count: int | None = None,
    x_ns: float | None = None,
    time_step_ns: float | None = None,
    total_duration_s: float | None = None,
    default_forever: bool = True,
) -> tuple[float, float, str] | None:
    """Return ``(start_s, stop_s, label)`` for a pulse-plot repeat bracket."""

    periods = None
    if hasattr(state_or_periods, "periods"):
        periods = list(getattr(state_or_periods, "periods"))
        repeat_start = getattr(state_or_periods, "repeat_start", repeat_start)
        repeat_end = getattr(state_or_periods, "repeat_end", repeat_end)
        repeat_count = getattr(state_or_periods, "repeat_count", repeat_count)
        x_ns = getattr(state_or_periods, "x_ns", x_ns)
        time_step_ns = getattr(state_or_periods, "time_step_ns", time_step_ns)
    elif state_or_periods is not None:
        periods = list(state_or_periods)

    if periods is None:
        if total_duration_s is None or not default_forever:
            return None
        return (0.0, float(total_duration_s), "∞")

    starts_ns = _pulse_period_starts_ns(periods, x_ns=x_ns, time_step_ns=time_step_ns)
    if repeat_start is None or repeat_end is None:
        if not default_forever:
            return None
        return (0.0, starts_ns[-1] * 1e-9, "∞")
    repeat_start = int(repeat_start)
    repeat_end = int(repeat_end)
    if repeat_start < 0 or repeat_end < repeat_start or repeat_end + 1 >= len(starts_ns):
        return None
    repeat_count = 1 if repeat_count is None else int(repeat_count)
    return (starts_ns[repeat_start] * 1e-9, starts_ns[repeat_end + 1] * 1e-9, f"x{repeat_count}")


def pulse_repeat_markers(
    state_or_periods=None,
    *,
    repeat_start: int | None = None,
    repeat_end: int | None = None,
    repeat_count: int | None = None,
    x_ns: float | None = None,
    time_step_ns: float | None = None,
    total_duration_s: float | None = None,
    default_forever: bool = True,
) -> list[tuple[float, float, str]]:
    """Return all repeat brackets that should be drawn on a pulse plot."""

    periods = None
    if hasattr(state_or_periods, "periods"):
        periods = list(getattr(state_or_periods, "periods"))
        repeat_start = getattr(state_or_periods, "repeat_start", repeat_start)
        repeat_end = getattr(state_or_periods, "repeat_end", repeat_end)
        repeat_count = getattr(state_or_periods, "repeat_count", repeat_count)
        x_ns = getattr(state_or_periods, "x_ns", x_ns)
        time_step_ns = getattr(state_or_periods, "time_step_ns", time_step_ns)
    elif state_or_periods is not None:
        periods = list(state_or_periods)

    if periods is None:
        if total_duration_s is None or not default_forever:
            return []
        return [(0.0, float(total_duration_s), "∞")]

    starts_ns = _pulse_period_starts_ns(periods, x_ns=x_ns, time_step_ns=time_step_ns)
    total = starts_ns[-1] * 1e-9
    if repeat_start is None or repeat_end is None:
        return [(0.0, total, "∞")] if default_forever else []

    repeat_start = int(repeat_start)
    repeat_end = int(repeat_end)
    if repeat_start < 0 or repeat_end < repeat_start or repeat_end + 1 >= len(starts_ns):
        return []
    repeat_count = 1 if repeat_count is None else int(repeat_count)
    inner = (starts_ns[repeat_start] * 1e-9, starts_ns[repeat_end + 1] * 1e-9, f"x{repeat_count}")
    if repeat_start == 0 and repeat_end == len(periods) - 1:
        return [inner]
    return [(0.0, total, "∞"), inner] if default_forever else [inner]


class PulseSequenceFigure(BaseLivePlot):
    """Filled-rectangle pulse timeline for sequencer/verilog inspection."""

    plot_type = "pulse"

    def __init__(
        self,
        sequence,
        *,
        channels: Sequence[str] | None = None,
        channel_labels: Mapping[str, str] | None = None,
        colors: Sequence[str] | None = None,
        show_names: bool = False,
        include_always_off: bool = False,
        repeat_notation: str | None = None,
        repeat_bracket: tuple[float, float, str] | None = None,
        repeat_brackets: Sequence[tuple[float, float, str]] | None = None,
        auto_height: bool = True,
        labels: Sequence[str] = ("Time (s)", "Pulse", "State"),
        **kwargs,
    ):
        self.sequence = sequence
        self.pulses = _pulse_rows(sequence)
        self.show_names = bool(show_names)
        self.channel_labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
        self.repeat_notation = "" if repeat_notation is None else str(repeat_notation)
        if repeat_brackets is None:
            repeat_brackets = [repeat_bracket] if repeat_bracket is not None else []
        self.repeat_brackets = [tuple(item) for item in repeat_brackets if item is not None]
        self.repeat_bracket = self.repeat_brackets[0] if self.repeat_brackets else None
        if channels is None:
            if hasattr(sequence, "channels"):
                channels = list(getattr(sequence, "channels"))
            else:
                channels = []
        self.channels = pulse_plot_channels(
            sequence,
            channels=channels,
            include_always_off=include_always_off,
        )
        self.channel_colors = list(colors or PULSE_COLORS)
        dummy_n = max(1, len(self.pulses))
        if auto_height and not any(key in kwargs for key in ("spec", "data_px", "margins_px")):
            kwargs["spec"] = pulse_plot_spec(len(self.channels))
        super().__init__(np.arange(dummy_n, dtype=float), np.zeros((dummy_n, 1), dtype=float), labels=labels, relim_mode="tight", **kwargs)

    @property
    def duration(self) -> float:
        if not self.pulses:
            return 1.0
        return max(float(row["stop"]) for row in self.pulses)

    def _xlimits_for_timeline(self, start_min: float, stop_max: float, *, has_bracket: bool) -> tuple[float, float]:
        span = max(float(stop_max - start_min), 1e-12)
        margin_x = max(span * (0.065 if has_bracket else 0.025), 1e-12)
        left_limit = start_min - margin_x if has_bracket else max(0.0, start_min - margin_x)
        right_limit = stop_max + margin_x * (1.25 if has_bracket else 0.8)
        return left_limit, right_limit

    def init_core(self) -> None:
        self.ax.set_ylabel(self.ylabel)
        color_map = {channel: self.channel_colors[i % len(self.channel_colors)] for i, channel in enumerate(self.channels)}
        row_height = 0.64 if len(self.channels) <= 10 else max(0.42, 6.4 / len(self.channels))
        n_channels = len(self.channels)
        index_map = {channel: n_channels - 1 - i for i, channel in enumerate(self.channels)}
        start_min = min([0.0] + [float(row["start"]) for row in self.pulses])
        stop_max = max([1e-12] + [float(row["stop"]) for row in self.pulses])
        bracket_bounds: list[tuple[float, float]] = []
        for repeat_bracket in self.repeat_brackets:
            try:
                bracket_start = float(repeat_bracket[0])
                bracket_stop = float(repeat_bracket[1])
            except Exception:
                bracket_start = bracket_stop = float("nan")
            if np.isfinite(bracket_start) and np.isfinite(bracket_stop) and bracket_stop > bracket_start:
                bracket_bounds.append((bracket_start, bracket_stop))
                start_min = min(start_min, bracket_start)
                stop_max = max(stop_max, bracket_stop)
        span = max(stop_max - start_min, 1e-12)
        self.time_scale, self.time_unit = _pulse_time_unit(span)
        self.ax.set_xlabel(_label_with_unit(self.xlabel, self.time_unit))
        left_limit, right_limit = self._xlimits_for_timeline(start_min, stop_max, has_bracket=bool(bracket_bounds))
        self.off_lines = []
        baseline_offset = row_height / 2
        pulse_zorder = 3
        self._pulse_baseline_y = {}
        for channel, y in index_map.items():
            color = color_map[channel]
            baseline_y = y - baseline_offset
            self._pulse_baseline_y[channel] = baseline_y
            self.off_lines.append(
                self.ax.hlines(
                    baseline_y,
                    left_limit,
                    right_limit,
                    color=color,
                    linewidth=0.65,
                    alpha=1.0,
                    zorder=pulse_zorder,
                )
            )
        self.pulse_artists = []
        for row in self.pulses:
            if not row["value"] or row["channel"] not in index_map or row["duration"] <= 0:
                continue
            y = index_map[row["channel"]]
            color = color_map[row["channel"]]
            patch = Rectangle(
                (row["start"], self._pulse_baseline_y[row["channel"]]),
                row["duration"],
                row_height,
                facecolor=color,
                edgecolor="none",
                linewidth=0.0,
                alpha=1.0,
                zorder=pulse_zorder,
            )
            self.ax.add_patch(patch)
            self.pulse_artists.append(patch)
            if self.show_names and row["name"] and row["duration"] >= 0.09 * max(self.duration, 1e-12):
                self.ax.text(
                    row["start"] + row["duration"] / 2,
                    y,
                    row["name"],
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=max(4.8, matplotlib.rcParams["legend.fontsize"] - 1.2),
                    clip_on=True,
                    zorder=pulse_zorder + 1,
                )

        self.ax.set_xlim(left_limit, right_limit)
        ylim_top = n_channels - 0.38
        if self.repeat_brackets:
            ylim_top = n_channels + 0.45 + 0.16 * max(0, len(self.repeat_brackets) - 1)
        self.ax.set_ylim(-0.62, ylim_top)
        self.ax.set_yticks([index_map[channel] for channel in self.channels])
        self.ax.set_yticklabels([self.channel_labels.get(channel, channel) for channel in self.channels])
        self.ax.tick_params(axis="y", labelsize=max(4.8, matplotlib.rcParams["ytick.labelsize"] - 1.2))
        for tick, channel in zip(self.ax.get_yticklabels(), self.channels):
            tick.set_color(color_map[channel])
        if self.repeat_notation and not self.repeat_brackets:
            self.ax.text(
                0.995,
                1.012,
                self.repeat_notation,
                transform=self.ax.transAxes,
                ha="right",
                va="bottom",
                color="0.35",
                fontsize=max(5.5, matplotlib.rcParams["legend.fontsize"] - 1.0),
            )
        self._draw_repeat_bracket(n_channels)
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        self.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _pos: "" if value < 0 else _float2str_eng(value / self.time_scale, length=4)))
        self.ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False, pad=2)
        self.ax.set_axisbelow(True)
        self.ax.grid(axis="x", color="0.88", linewidth=0.35, zorder=0)
        for gridline in self.ax.get_xgridlines():
            gridline.set_zorder(0)
        self.ax.spines[["top", "right"]].set_visible(False)
        self.lines = [*self.off_lines, *self.pulse_artists]

    def update_core(self) -> None:
        pass

    def _draw_repeat_bracket(self, n_channels: int) -> None:
        if not self.repeat_brackets:
            return
        colors = ("#6A6A6A", "#C96F3D", "#4F7EA8", "#8B6BB8")
        self.repeat_bracket_artists = []
        self.repeat_bracket_labels = []
        xlim = self.ax.get_xlim()
        span = max(float(xlim[1] - xlim[0]), 1e-12)
        tick_base = span * 0.024
        bracket_count = max(1, len(self.repeat_brackets))
        for index, repeat_bracket in enumerate(self.repeat_brackets):
            try:
                start, stop, label = repeat_bracket
                start = float(start)
                stop = float(stop)
                label = str(label)
            except Exception:
                continue
            if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
                continue
            color = colors[index % len(colors)]
            alpha = 0.58
            outer_depth = max(0, bracket_count - 1 - index)
            y_low = -0.42 - 0.10 * outer_depth
            y_high = float(n_channels) - 0.12 + 0.14 * outer_depth
            tick = tick_base
            if stop > start:
                tick = min(tick, max(stop - start, 0.0) * 0.2)
            tick = max(tick, span * 0.006)
            left_artist = self.ax.plot(
                [start + tick, start, start, start + tick],
                [y_high, y_high, y_low, y_low],
                color=color,
                alpha=alpha,
                linewidth=1.05,
                solid_capstyle="round",
                clip_on=True,
                zorder=8 + index,
            )[0]
            right_artist = self.ax.plot(
                [stop - tick, stop, stop, stop - tick],
                [y_high, y_high, y_low, y_low],
                color=color,
                alpha=alpha,
                linewidth=1.05,
                solid_capstyle="round",
                clip_on=True,
                zorder=8 + index,
            )[0]
            self.repeat_bracket_artists.extend([left_artist, right_artist])
            label_artist = self.ax.text(
                stop + tick * 0.12,
                y_high + 0.035,
                label,
                ha="left",
                va="bottom",
                color=color,
                fontfamily="DejaVu Sans",
                alpha=alpha,
                fontsize=max(5.5, matplotlib.rcParams["legend.fontsize"] - 0.8),
                clip_on=False,
                zorder=9 + index,
            )
            self.repeat_bracket_labels.append(label_artist)
        if self.repeat_bracket_artists:
            self.repeat_bracket_artist = self.repeat_bracket_artists[0]
        if self.repeat_bracket_labels:
            self.repeat_bracket_label = self.repeat_bracket_labels[-1]

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax, area=False)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="pulse", x_array=None, y_array=None)


class HistogramFigure(BaseLivePlot):
    """Neutral-atom-friendly histogram with threshold classification tools."""

    plot_type = "hist"

    def __init__(
        self,
        values,
        *,
        bins: int | Sequence[float] = 50,
        thresholds: Sequence[float] | None = None,
        labels: Sequence[str] = ("Counts", "Shots", "Population"),
        **kwargs,
    ):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.bins_arg = bins
        self.thresholds = list(thresholds or [])
        super().__init__(np.arange(len(self.values)), self.values, labels=labels, relim_mode="tight", **kwargs)

    def init_core(self) -> None:
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        vals = self.values[np.isfinite(self.values)]
        if vals.size == 0:
            vals = np.array([0.0])
        self.n, self.bins = np.histogram(vals, bins=self.bins_arg)
        self.verts = np.empty((len(self.n), 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="vertical")
        self.poly = PolyCollection(self.verts, facecolors="grey")
        self.ax.add_collection(self.poly)
        (self.fit_line_left,) = self.ax.plot([], [], color="skyblue", linewidth=1, alpha=0.8)
        (self.fit_line_right,) = self.ax.plot([], [], color="orange", linewidth=1, alpha=0.8)
        (self.fit_line_total,) = self.ax.plot([], [], color="black", linewidth=1, alpha=0.35)
        self.bimodal_popt = None
        self.fit_threshold = None
        self._fit_bimodal()
        if not self.thresholds:
            self.thresholds = [self.fit_threshold if self.fit_threshold is not None else float(np.nanmedian(vals))]
        self.threshold_lines = []
        self.threshold_draggers = []
        for threshold in self.thresholds:
            line = self.ax.axvline(threshold, color="orange", linewidth=1.9, alpha=0.95, zorder=5)
            self.threshold_lines.append(line)
            self.threshold_draggers.append(DragVLine(line, self._on_threshold_drag, self.ax))
        self.stats_text = self.ax.text(
            0.975,
            0.975,
            "",
            transform=self.ax.transAxes,
            ha="right",
            va="top",
            color="black",
            fontsize=matplotlib.rcParams["legend.fontsize"],
        )
        self.ax.set_xlim(self.bins[0], self.bins[-1])
        self.ax.set_ylim(0, max(1, float(np.max(self.n) * 1.2)))
        self._update_hist_stats()

    def update(self, values=None, *, data_y=None, points_done: int | None = None, repeat_cur: int | None = None, draw: bool = True):
        if values is None and data_y is not None:
            values = data_y
        if values is not None:
            self.values = np.asarray(values, dtype=float).reshape(-1)
            self.data_x = _as_data_x(np.arange(len(self.values)))
            self.data_y = _as_data_y(self.values, len(self.values))
            self.points_total = len(self.values)
        return super().update(self.data_y, points_done=points_done or len(self.values), repeat_cur=repeat_cur, draw=draw)

    def update_core(self) -> None:
        vals = self.values[np.isfinite(self.values)]
        if vals.size == 0:
            vals = np.array([0.0])
        self.n, self.bins = np.histogram(vals, bins=self.bins_arg)
        if len(self.verts) != len(self.n):
            self.verts = np.empty((len(self.n), 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="vertical")
        self.poly.set_verts(self.verts)
        self.ax.set_xlim(self.bins[0], self.bins[-1])
        self.ax.set_ylim(0, max(1, float(np.max(self.n) * 1.2)))
        self._fit_bimodal()
        while len(self.threshold_lines) < len(self.thresholds):
            line = self.ax.axvline(self.thresholds[len(self.threshold_lines)], color="orange", linewidth=1.9, alpha=0.95, zorder=5)
            self.threshold_lines.append(line)
            self.threshold_draggers.append(DragVLine(line, self._on_threshold_drag, self.ax))
        for line, threshold in zip(self.threshold_lines, self.thresholds):
            line.set_xdata([threshold, threshold])
        self._update_hist_stats()

    def set_thresholds(self, thresholds: Sequence[float]):
        self.thresholds = list(thresholds)
        self.update_core()
        self.draw()
        return self

    def classify(self, values=None) -> np.ndarray:
        vals = self.values if values is None else np.asarray(values, dtype=float)
        return np.digitize(vals, np.sort(self.thresholds))

    def fractions(self, values=None) -> dict[int, float]:
        states = self.classify(values)
        if len(states) == 0:
            return {}
        return {int(state): float(np.mean(states == state)) for state in np.unique(states)}

    @staticmethod
    def _gauss(x, amp, mu, sigma):
        sigma = max(float(abs(sigma)), 1e-12)
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    @classmethod
    def _bimodal_model(cls, x, amp0, mu0, sigma0, amp1, mu1, sigma1):
        return cls._gauss(x, amp0, mu0, sigma0) + cls._gauss(x, amp1, mu1, sigma1)

    def _fit_bimodal(self) -> None:
        vals = self.values[np.isfinite(self.values)]
        if vals.size < 6 or np.ptp(vals) == 0:
            self.bimodal_popt = None
            self.fit_threshold = None
            return

        centers = (self.bins[:-1] + self.bins[1:]) / 2
        counts = self.n.astype(float)
        split = np.nanmedian(vals)
        left_vals = vals[vals <= split]
        right_vals = vals[vals > split]
        if left_vals.size < 2 or right_vals.size < 2:
            left_vals = vals[: vals.size // 2]
            right_vals = vals[vals.size // 2 :]

        mu0 = float(np.nanmean(left_vals))
        mu1 = float(np.nanmean(right_vals))
        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
        span = float(np.ptp(vals)) or 1.0
        sigma0 = max(float(np.nanstd(left_vals)), span / 20, 1e-9)
        sigma1 = max(float(np.nanstd(right_vals)), span / 20, 1e-9)
        amp = max(float(np.nanmax(counts)), 1.0)
        p0 = [amp, mu0, sigma0, amp, mu1, sigma1]
        bounds = (
            [0, float(np.nanmin(vals)), span / 200, 0, float(np.nanmin(vals)), span / 200],
            [max(amp * 5, 1), float(np.nanmax(vals)), span * 2, max(amp * 5, 1), float(np.nanmax(vals)), span * 2],
        )
        try:
            popt, _ = curve_fit(self._bimodal_model, centers, counts, p0=p0, bounds=bounds, maxfev=20000)
        except Exception:
            self.bimodal_popt = None
            self.fit_threshold = None
            return

        if popt[1] > popt[4]:
            popt = np.array([popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]], dtype=float)
        self.bimodal_popt = popt
        x_fit = np.linspace(self.bins[0], self.bins[-1], 400)
        y0 = self._gauss(x_fit, *popt[:3])
        y1 = self._gauss(x_fit, *popt[3:])
        self.fit_line_left.set_data(x_fit, y0)
        self.fit_line_right.set_data(x_fit, y1)
        self.fit_line_total.set_data(x_fit, y0 + y1)
        lo, hi = float(popt[1]), float(popt[4])
        x_mid = np.linspace(lo, hi, 400)
        diff = np.abs(self._gauss(x_mid, *popt[:3]) - self._gauss(x_mid, *popt[3:]))
        self.fit_threshold = float(x_mid[int(np.nanargmin(diff))])

    @staticmethod
    def _normal_cdf(x, mu, sigma):
        sigma = max(float(abs(sigma)), 1e-12)
        return 0.5 * (1 + erf((float(x) - float(mu)) / (sigma * sqrt(2))))

    def _fit_fidelity(self, threshold: float) -> float | None:
        if self.bimodal_popt is None:
            return None
        amp0, mu0, sigma0, amp1, mu1, sigma1 = self.bimodal_popt
        w0 = abs(amp0 * sigma0)
        w1 = abs(amp1 * sigma1)
        if (w0 + w1) == 0:
            return None
        left_ok = self._normal_cdf(threshold, mu0, sigma0)
        right_ok = 1 - self._normal_cdf(threshold, mu1, sigma1)
        raw = float((w0 * left_ok + w1 * right_ok) / (w0 + w1))
        separation = abs(float(mu1) - float(mu0)) / sqrt(float(sigma0) ** 2 + float(sigma1) ** 2)
        balance = 2.0 * min(w0, w1) / (w0 + w1)
        effective_separation = max(0.0, separation - 2.0)
        confidence = float(np.clip(balance * (1.0 - np.exp(-0.5 * effective_separation * effective_separation)), 0.0, 1.0))
        return float(0.5 + (raw - 0.5) * confidence)

    def _on_threshold_drag(self, x: float) -> None:
        if not self.thresholds:
            self.thresholds = [float(x)]
        else:
            self.thresholds[0] = float(x)
        self._update_hist_stats()

    def _update_hist_stats(self) -> None:
        if not self.thresholds:
            return
        threshold = float(self.thresholds[0])
        vals = self.values[np.isfinite(self.values)]
        if vals.size:
            left = float(np.mean(vals <= threshold))
            right = 1.0 - left
        else:
            left = right = 0.0
        fidelity = self._fit_fidelity(threshold)
        if fidelity is None:
            fidelity_text = "fit F=N/A"
        else:
            fidelity_text = f"fit F={100 * fidelity:.1f}%"
        fit_threshold = "" if self.fit_threshold is None else f"\nfit cut={self.fit_threshold:.4g}"
        self.stats_text.set_text(
            f"th={threshold:.4g}\n{fidelity_text}\nL/R={100 * left:.1f}%/{100 * right:.1f}%{fit_threshold}"
        )

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="hist", x_array=self.bins, y_array=self.n)


def _is_watch_update(update) -> bool:
    if isinstance(update, str):
        return update.lower() in {"watch", "timer", "live", "auto"}
    return bool(update)


def _normalize_kind(kind: str | None) -> str:
    if kind is None:
        return "auto"
    normalized = str(kind).lower().replace("_", "-")
    aliases = {
        "line": "1d",
        "trace": "1d",
        "image": "2d",
        "map": "2d",
        "histogram": "hist",
        "distribution": "hist",
        "pulses": "pulse",
        "pulse-sequence": "pulse",
        "sequence": "pulse",
        "timing": "pulse",
        "live-dis": "monitor",
        "live-distribution": "monitor",
        "rolling": "monitor",
    }
    return aliases.get(normalized, normalized)


def plot(
    data_x,
    data_y=None,
    *,
    kind: str | None = "auto",
    update: str | bool | None = "once",
    labels: Sequence[str] | None = None,
    display: bool = True,
    data_figure: bool = True,
    watch_interval: float | None = None,
    stop_when_full: bool = True,
    done=None,
    points_done=None,
    copy: bool = False,
    lock=None,
    **kwargs,
):
    """Create a static or live notebook plot from the same array contract.

    ``data_x`` is ``(N, coord_dim)`` and ``data_y`` is ``(N, channel_dim)``.
    ``coord_dim == 1`` creates a 1D line plot and ``coord_dim == 2`` creates a
    2D scan image. ``kind="hist"`` treats ``data_x`` as the values array. With
    ``update="watch"``, the returned object starts a frontend timer and refreshes
    from the same shared arrays while acquisition code mutates them.
    """
    normalized_kind = _normalize_kind(kind)
    should_watch = _is_watch_update(update)

    if normalized_kind == "hist":
        values = data_x if data_y is None else data_y
        labels = tuple(labels or ("Counts", "Shots", "Population"))
        plotter = HistogramFigure(values, labels=labels, **kwargs).show(display=display)
    elif normalized_kind == "pulse":
        if should_watch:
            raise ValueError("pulse plots are static timing diagrams; update='watch' is not supported.")
        labels = tuple(labels or ("Time (s)", "Pulse", "State"))
        plotter = PulseSequenceFigure(data_x, labels=labels, **kwargs).show(display=display)
    else:
        x = _as_data_x(data_x)
        y = _as_data_y(data_y, len(x))
        if normalized_kind == "auto":
            normalized_kind = "2d" if x.shape[1] == 2 else "1d"
        labels = tuple(labels or ("X", "Y", "Z"))
        if normalized_kind == "1d":
            plotter = Live1D(x, y, labels=labels, **kwargs).show(display=display)
        elif normalized_kind == "2d":
            if "square" in kwargs:
                square = kwargs.pop("square")
                if square is not True:
                    raise ValueError("frontend.plot 2D figures are always square; call Live2DDis directly for internal non-square experiments.")
            plotter = Live2DDis(x, y, labels=labels, square=True, **kwargs).show(display=display)
        elif normalized_kind == "monitor":
            plotter = LiveLiveDis(x, y, labels=labels, **kwargs).show(display=display)
        else:
            raise ValueError("kind must be auto, 1d, 2d, monitor, hist, or pulse.")

    if should_watch:
        plotter.watch(
            interval=watch_interval,
            stop_when_full=stop_when_full,
            done=done,
            points_done=points_done,
            copy=copy,
            lock=lock,
        )
    elif data_figure:
        plotter.to_data_figure()
    return plotter


__all__ = [
    "BaseLivePlot",
    "HistogramFigure",
    "Live1D",
    "Live2DDis",
    "LiveLiveDis",
    "PulseSequenceFigure",
    "plot",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_marker",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
]
