"""Post-processing, fitting, unit conversion, and saving for front-end figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import re
import time
from typing import Any, Callable, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.signal import find_peaks
import warnings

from .style import PALETTE, small_fontsize


VALID_FIT_FUNCS = ["lorent", "lorent_zeeman", "rabi", "decay", "center", "gaussian"]


@dataclass
class FitResult:
    """Structured fit result returned by DataFigure fit methods."""

    names: list[str]
    popt: np.ndarray | None
    pcov: np.ndarray | None
    function: str


def _as_2d_y(data_y) -> np.ndarray:
    y = np.asarray(data_y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    return y


def resolve_save_base(path, stem: str) -> Path:
    """Turn a save ``path`` + ``stem`` into the extension-less base path, and mkdir its parent --
    the ONE place a figure+npz pair's location is resolved (DataFigure.save and the grid save both
    call it).  Empty / ``"."`` -> the bare ``stem``; a directory or trailing-separator path -> the
    stem inside it; a path that already has a suffix -> that path sans suffix; anything else ->
    ``<path>_<stem>`` (#C4)."""
    p = Path(path)
    if str(p) in ("", "."):
        base = Path(stem)
    elif str(p).endswith(("/", "\\")) or p.is_dir():
        base = p / stem
    elif p.suffix:
        base = p.with_suffix("")
    else:
        base = Path(f"{p}_{stem}")
    base.parent.mkdir(parents=True, exist_ok=True)
    return base


class DataFigure:
    """Data and post-processing handle for a front-end figure.

    A DataFigure can be created from a ``Live1D``/``Live2DDis``/``HistogramFigure``
    object, or from explicit ``fig``, ``data_x`` and ``data_y`` handles.  Pass an
    explicit ``ax`` to bind the fitting/post-processing to ONE axes of a
    multi-axes figure (e.g. a single cell of a site-histogram grid), so the same
    reusable stack works per cell instead of only on ``fig.axes[0]``.
    """

    def __init__(
        self,
        live_plot=None,
        *,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        data_x=None,
        data_y=None,
        labels: Sequence[str] | None = None,
        tools=None,
        info: Mapping[str, Any] | None = None,
        name: str | None = None,
        unit: str | None = None,
    ):
        self.live_plot = live_plot
        if live_plot is not None:
            fig = live_plot.fig
            data_x = live_plot.data_x
            data_y = live_plot.data_y
            labels = getattr(live_plot, "labels", labels)
            tools = getattr(live_plot, "tools", tools)
            info = getattr(live_plot, "info", info)
            name = getattr(live_plot, "name", name)
            unit = getattr(live_plot, "unit", unit)

        if fig is None or data_x is None or data_y is None:
            raise ValueError("DataFigure needs either live_plot or fig/data_x/data_y.")

        self.fig = fig
        # The single axes this DataFigure draws on / reads limits from.  Defaults
        # to the figure's first axes (single-axes plots); an explicit ``ax`` binds
        # it to one cell of a multi-axes figure so the stack is reusable per cell.
        self._ax = ax if ax is not None else self.fig.axes[0]
        self.data_x = np.asarray(data_x, dtype=float)
        if self.data_x.ndim == 1:
            self.data_x = self.data_x[:, None]
        self.data_x_original = copy.deepcopy(self.data_x)
        self.data_y = _as_2d_y(data_y)
        self.labels = list(labels) if labels is not None else ["X", "Y", "Z"]
        self.info = dict(info or {})
        self.name = name or self.info.get("name") or self.info.get("class_name") or "figure"

        self.area = getattr(tools, "area", None)
        self.zoom = getattr(tools, "zoom", None)
        self.cross = getattr(tools, "cross", None)
        if live_plot is not None:
            self.area = getattr(live_plot, "area", self.area)
            self.zoom = getattr(live_plot, "zoom", self.zoom)
            self.cross = getattr(live_plot, "cross", self.cross)

        first_ax = self._ax
        # The fitting FAMILY ("1D" / "2D") comes from the plot's DECLARED
        # ``render_family`` (the single-source PLOT_KINDS table in live.py), not
        # re-derived from matplotlib artists.  Fall back to the legacy artist
        # heuristic (an image axes => 2D) when there is nothing to ask -- no
        # live_plot (a GridPlot per-cell DataFigure built with ``fig=``/``ax=``, or
        # a bare externally-constructed DataFigure) OR a plot that declares the
        # ``"auto"`` sentinel (the site map, whose family is image-only when a
        # background frame is supplied, so it stays artist-derived per figure).
        declared = getattr(live_plot, "render_family", None) if live_plot is not None else None
        self.plot_type = declared if declared in ("1D", "2D") else ("2D" if first_ax.images else "1D")
        self.ylabel_original = self.labels[1] if len(self.labels) > 1 else first_ax.get_ylabel()
        self.unit = unit or self._infer_unit(first_ax.get_xlabel())
        self.unit_original = self.info.get("unit", self.unit)
        self._load_unit_conversion()

        self.p0 = None
        self.popt = None
        self.fit = None
        self.fit_func = None
        self.text = None
        self._scatter_list = []
        # The SOURCE binding (hub + producing node + wired inputs), copied from the live plot when this
        # DataFigure wraps one, so ``save`` can write the RICH npz (``info['signals']`` +
        # ``info['provenance']``) through the ONE core capture -- the SAME logic the console panel uses.
        # ``None`` (a bare array DataFigure) => ``save`` writes the basic figure+npz (old behaviour).
        self._figure_source = getattr(live_plot, "_figure_source", None)
        warnings.filterwarnings("ignore", category=OptimizeWarning)

    @staticmethod
    def _infer_unit(label: str) -> str:
        match = re.search(r"\((.+)\)$", label or "")
        return match.group(1) if match else "1"

    def _load_unit_conversion(self) -> None:
        if self.unit in ["GHz", "nm", "MHz"]:
            spl = 299792458
            self.conversion_map = {
                "nm": ("GHz", lambda x: spl / x),
                "GHz": ("MHz", lambda x: x * 1e3),
                "MHz": ("nm", lambda x: spl / (x / 1e3)),
            }
        elif self.unit in ["ns", "us", "ms"]:
            self.conversion_map = {
                "ms": ("ns", lambda x: x * 1e6),
                "ns": ("us", lambda x: x / 1e3),
                "us": ("ms", lambda x: x / 1e3),
            }
        else:
            self.conversion_map = None
        if self.conversion_map is None or self.unit_original not in self.conversion_map:
            self.unit_original = self.unit
        self._update_transform_back()

    def xlim(self, x_min: float, x_max: float) -> None:
        self._ax.set_xlim(x_min, x_max)
        self.fig.canvas.draw_idle()

    def ylim(self, y_min: float, y_max: float) -> None:
        self._ax.set_ylim(y_min, y_max)
        self.fig.canvas.draw_idle()

    def bind_source(self, hub, node, *, inputs, resolve_node=None, session=None):
        """Stamp WHERE this figure's data came from so :meth:`save` writes the RICH npz -- the DataFigure
        counterpart of :meth:`live.BaseLivePlot.bind_source`, for when a caller holds the DataFigure
        directly.  Returns ``self`` for chaining."""
        self._figure_source = {"hub": hub, "node": node, "inputs": list(inputs or []),
                               "resolve_node": resolve_node, "session": session}
        return self

    def _rich_capture(self) -> dict[str, Any]:
        """The ``signals`` + ``provenance`` blocks a RICH save folds into ``info``, captured through the
        ONE frontend-neutral core (``operations.figure_capture``) -- the SAME logic the console panel's
        Save uses, so a notebook ``.save()`` and a GUI panel Save write byte-identical rich npz.  Empty
        when this figure carries no source binding (a bare array plot): then ``save`` writes the basic
        payload (old behaviour).  Never raises -- a capture that fails for one block simply omits it."""
        src = self._figure_source
        if not src:
            return {}
        from Zou_lab_control.neutral_atom.operations.figure_capture import (
            capture_figure_signals, capture_figure_provenance)
        out: dict[str, Any] = {}
        try:
            signals = capture_figure_signals(src.get("hub"), src.get("node"), src.get("inputs"))
        except Exception:
            signals = {}
        if signals:
            out["signals"] = signals
        try:
            prov = capture_figure_provenance(src.get("node"), resolve_node=src.get("resolve_node"),
                                             session=src.get("session"))
        except Exception:
            prov = None
        if prov is not None:
            out["provenance"] = prov
        return out

    def save(
        self,
        path: str | Path = "",
        *,
        extra_info: Mapping[str, Any] | None = None,
        image_ext: str = "png",
    ) -> dict[str, Path]:
        """Save the figure image and a matching ``.npz`` payload.

        When this figure carries a SOURCE binding (``bind_source`` / a plot created from live signals),
        the save is RICH: it folds ``info['signals']`` (raw hub blocks + roles, so a site map's underlay
        frame + centres round-trip) and ``info['provenance']`` (the device state the data was taken under)
        captured through the ONE frontend-neutral core -- identical to the console panel's Save.  An
        explicit ``extra_info`` key WINS over the auto-capture (the GUI passes its own richer blocks), and
        a bare array figure (no binding) writes just the basic figure+npz."""
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        stem = "_".join(p for p in (self.name, current_time) if p)
        base = resolve_save_base(path, stem)
        image_path = base.with_suffix(f".{image_ext}")
        data_path = base.with_suffix(".npz")
        info = {
            **self.info,
            **self._rich_capture(),          # auto signals/provenance (bound figure); {} for a bare plot
            **dict(extra_info or {}),         # an explicit caller block WINS over the auto-capture
            "labels": self.labels,
            "name": self.name,
            "unit": self.unit_original,
            "points_done": getattr(self.live_plot, "points_done", len(self.data_x)),
            "repeat_cur": getattr(self.live_plot, "repeat_cur", 1),
        }
        # Fold the CURRENT view unit + any applied fit into ``info`` so a reload
        # reproduces "the figure as saved" (not just the raw arrays): ``unit`` is
        # the display unit, and a fit -- if one has been drawn -- carries its
        # function name, coefficients and parameter names so a reader can see /
        # re-apply it.  These only APPEND keys; the data_x/data_y/info structure
        # is unchanged, so an older reader still reads the file.
        fit_info = self._saved_fit_info()
        if fit_info is not None:
            info.setdefault("fit", fit_info)
        self.fig.savefig(image_path, bbox_inches="tight")
        np.savez(data_path, data_x=self.data_x_original, data_y=self.data_y, info=info)
        return {"figure": image_path, "data": data_path}

    def _saved_fit_info(self) -> dict[str, Any] | None:
        """The applied fit as a JSON-friendly dict (``func`` + ``names`` + ``popt``) for the
        saved ``info``, or ``None`` when no fit has been drawn.  ``popt`` becomes a plain list
        (never a raw ndarray) so ``np.savez`` round-trips it cleanly and a reader can read it
        without unpacking an object array."""
        if not self.fit_func or self.popt is None:
            return None
        names = list(getattr(self, "popt_str", []) or [])
        return {"func": str(self.fit_func), "names": names,
                "popt": [float(v) for v in np.asarray(self.popt, dtype=float).ravel()]}

    def _align_to_grid(self, value: float, axis: str) -> float:
        if self.plot_type != "2D" or self.live_plot is None:
            return value
        if not hasattr(self, "grid_center"):
            self.grid_center = self.data_x[0]
            x_array = np.asarray(getattr(self.live_plot, "x_array"))
            y_array = np.asarray(getattr(self.live_plot, "y_array"))
            self.step_x = abs(x_array[1] - x_array[0]) if len(x_array) > 1 else 1
            self.step_y = abs(y_array[1] - y_array[0]) if len(y_array) > 1 else 1
        if axis == "x":
            return round((value - self.grid_center[0]) / self.step_x) * self.step_x + self.grid_center[0]
        return round((value - self.grid_center[1]) / self.step_y) * self.step_y + self.grid_center[1]

    def _valid_index(self) -> np.ndarray:
        return np.array([i for i, row in enumerate(self.data_y) if np.isfinite(row[0])], dtype=int)

    def _select_fit(self, min_num: int = 2):
        valid_index = self._valid_index()
        if valid_index.size == 0:
            raise ValueError("No finite data points are available for fitting.")

        if self.plot_type == "1D":
            x = self.data_x[valid_index, 0]
            y = self.data_y[valid_index, 0]
            area = getattr(self.area, "range", [None, None, None, None])
            if area[0] is None:
                xlim = self._ax.get_xlim()
                xl, xh = sorted(xlim)
            else:
                xl, xh = sorted(area[:2])
            mask = (x >= xl) & (x <= xh)
            if int(mask.sum()) <= min_num:
                return x, y
            return x[mask], y[mask]

        x_all = self.data_x[valid_index, 0]
        y_all = self.data_x[valid_index, 1]
        z_all = self.data_y[valid_index, 0]
        area = getattr(self.area, "range", [None, None, None, None])
        if area[0] is None:
            xl, xh = sorted(self._ax.get_xlim())
            yl, yh = sorted(self._ax.get_ylim())
        else:
            xl, xh, yl, yh = area
            xl, xh = sorted([xl, xh])
            yl, yh = sorted([yl, yh])
        xl, xh = [self._align_to_grid(v, "x") for v in (xl, xh)]
        yl, yh = [self._align_to_grid(v, "y") for v in (yl, yh)]
        mask = (x_all >= xl) & (x_all <= xh) & (y_all >= yl) & (y_all <= yh)
        if int(mask.sum()) <= min_num:
            return (x_all, y_all), z_all
        return (x_all[mask], y_all[mask]), z_all[mask]

    def _place_text(self, ax: plt.Axes, text) -> None:
        candidates = [
            (0.025, 0.85, "left", "top"),
            (0.975, 0.85, "right", "top"),
            (0.025, 0.025, "left", "bottom"),
            (0.975, 0.025, "right", "bottom"),
            (0.5, 0.025, "center", "bottom"),
            (0.5, 0.85, "center", "top"),
        ]
        renderer = ax.figure.canvas.get_renderer()
        best = candidates[0]
        best_overlap = float("inf")
        if self.plot_type == "1D":
            pts = np.column_stack([self.data_x[:, 0], self.data_y[:, 0]])
            pts = pts[np.isfinite(pts).all(axis=1)]
            pts_disp = ax.transData.transform(pts[:: max(1, len(pts) // 1000)]) if len(pts) else np.empty((0, 2))
        else:
            pts_disp = np.empty((0, 2))

        for cand in candidates:
            text.set_position(cand[:2])
            text.set_ha(cand[2])
            text.set_va(cand[3])
            ax.figure.canvas.draw()
            bbox = text.get_window_extent(renderer).expanded(1.05, 1.1)
            if len(pts_disp) == 0:
                overlap = 0
            else:
                overlap = int(
                    np.sum(
                        (pts_disp[:, 0] >= bbox.x0)
                        & (pts_disp[:, 0] <= bbox.x1)
                        & (pts_disp[:, 1] >= bbox.y0)
                        & (pts_disp[:, 1] <= bbox.y1)
                    )
                )
            if overlap < best_overlap:
                best_overlap = overlap
                best = cand
        text.set_position(best[:2])
        text.set_ha(best[2])
        text.set_va(best[3])

    def _display_popt(self, popt, names: Sequence[str], is_display: bool = True) -> None:
        formatted = []
        for name, value in zip(names, popt):
            formatted.append(f"{name}={float(value):.5g}")
        result = f"{self.formula_str}\n" + "\n".join(formatted)

        if is_display:
            if self.text is None:
                self.text = self._ax.text(
                    0.5,
                    0.5,
                    result,
                    transform=self._ax.transAxes,
                    color=PALETTE["fit_text"],
                    ha="center",
                    va="center",
                    fontsize=small_fontsize(),
                )
            else:
                self.text.set_text(result)
            self._place_text(self._ax, self.text)
        elif self.text is not None:
            self.text.remove()
            self.text = None

        lines = getattr(self.live_plot, "lines", None) if self.live_plot is not None else self._ax.lines
        for line in lines:
            if hasattr(line, "set_alpha"):
                line.set_alpha(0.5)
        if self.plot_type == "1D" and len(self.data_y) < 2000:
            self._line_to_scatter()
        self.fig.canvas.draw_idle()

    @staticmethod
    def _clean_param_name(name: str) -> str:
        return re.sub(r"[\$\\{}]", "", name)

    def _fit_and_draw(self, is_fit: bool, is_display: bool, kwargs: Mapping[str, Any]) -> tuple[np.ndarray | None, Any]:
        for idx, param in enumerate(self.popt_str):
            clean = self._clean_param_name(param)
            fixed = kwargs.get(clean, None)
            if fixed is None:
                continue
            low, high = np.sort([fixed * (1 - 1e-5), fixed * (1 + 1e-5)])
            if low == high:
                low, high = fixed - 1e-12, fixed + 1e-12
            self.bounds[0][idx], self.bounds[1][idx] = low, high
            for p0 in self.p0_list:
                p0[idx] = fixed

        if is_fit:
            loss_min = np.inf
            popt = None
            pcov = None
            for p0 in self.p0_list:
                try:
                    popt_cur, pcov_cur = curve_fit(self._fit_func, self.data_x_p, self.data_y_p, p0=p0, bounds=self.bounds)
                    loss_cur = np.sum((self._fit_func(self.data_x_p, *popt_cur) - self.data_y_p) ** 2)
                    if loss_cur < loss_min:
                        loss_min = loss_cur
                        popt = popt_cur
                        pcov = pcov_cur
                except Exception:
                    continue
            if popt is None:
                return None, None
        else:
            popt, pcov = np.asarray(self.p0_list[0], dtype=float), None

        self.popt = popt
        self._display_popt(popt, self.popt_str, is_display)
        ax = self._ax
        if self.plot_type == "1D":
            yfit = self._fit_func(self.data_x[:, 0], *popt)
            if self.fit is None:
                self.fit = ax.plot(self.data_x[:, 0], yfit, color=PALETTE["fit_right"], linestyle="-", linewidth=2, alpha=0.5)
            else:
                self.fit[0].set_data(self.data_x[:, 0], yfit)
        else:
            if self.fit is None:
                self.fit = [ax.scatter(popt[-2], popt[-1], color=PALETTE["fit_right"], s=50)]
                circle = matplotlib.patches.Circle(
                    (popt[-2], popt[-1]),
                    radius=abs(popt[-3]),
                    edgecolor=PALETTE["fit_right"],
                    facecolor="none",
                    linewidth=2,
                    alpha=0.5,
                )
                self.fit.append(circle)
                ax.add_patch(circle)
            else:
                self.fit[0].set_offsets((popt[-2], popt[-1]))
                self.fit[1].set_center((popt[-2], popt[-1]))
                self.fit[1].set_radius(abs(popt[-3]))
        self.fig.canvas.draw_idle()
        return popt, pcov

    def lorent(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        if self.plot_type == "2D":
            return FitResult(["x_0", "FWHM", "H", "B"], None, None, "lorent"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=4)
        self.formula_str = r"$f(x)=H\frac{(FWHM/2)^2}{(x-x_0)^2+(FWHM/2)^2}+B$"

        def _lorent(x, center, full_width, height, bg):
            return height * ((full_width / 2) ** 2) / ((x - center) ** 2 + (full_width / 2) ** 2) + bg

        self._fit_func = _lorent
        if p0 is None:
            span = abs(self.data_x_p[0] - self.data_x_p[-1]) or 1
            amp = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
            self.p0_list = [
                [self.data_x_p[np.nanargmax(self.data_y_p)], span / 4, amp, np.nanmin(self.data_y_p)],
                [self.data_x_p[np.nanargmin(self.data_y_p)], span / 4, -amp, np.nanmax(self.data_y_p)],
            ]
        else:
            self.p0_list = [list(p0)]
        width = abs(self.p0_list[0][1]) or 1
        yrange = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
        self.bounds = [
            [np.nanmin(self.data_x_p), width / 10, -10 * yrange, np.nanmin(self.data_y_p) - 10 * yrange],
            [np.nanmax(self.data_x_p), width * 10, 10 * yrange, np.nanmax(self.data_y_p) + 10 * yrange],
        ]
        self.popt_str = ["x_0", "FWHM", "H", "B"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "lorent"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def gaussian(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        # NOTE (DRY boundary, #H3w-5): this is the INTERACTIVE CURVE-FIT model for a 1-D plot -- a peak
        # on a BACKGROUND, so it carries an ``offset`` (B) term and amplitude may be negative (a dip).
        # It is deliberately DISTINCT from ``_readout_math.gaussian`` (a normalised, offset-free PEAK
        # used by the per-site readout fidelity math).  Different models for different jobs -- do NOT
        # "unify" them; the offset here would corrupt the readout overlap integral, and removing it
        # would break fitting a peak that sits on a pedestal.
        if self.plot_type == "2D":
            return FitResult(["A", "B", "sigma", "x_0"], None, None, "gaussian"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=4)
        self.formula_str = r"$f(x)=Ae^{-(x-x_0)^2/(2\sigma^2)}+B$"

        def _gaussian(x, amplitude, offset, sigma, x0):
            return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset

        self._fit_func = _gaussian
        if p0 is None:
            amp = np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)
            offset = np.nanmin(self.data_y_p)
            sigma = abs(self.data_x_p[-1] - self.data_x_p[0]) / 6 or 1
            x0 = self.data_x_p[np.nanargmax(self.data_y_p)]
            self.p0_list = [[amp, offset, sigma, x0], [-amp, np.nanmax(self.data_y_p), sigma, x0]]
        else:
            self.p0_list = [list(p0)]
        yrange = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
        sigma0 = abs(self.p0_list[0][2]) or 1
        self.bounds = [
            [-10 * yrange, np.nanmin(self.data_y_p) - 10 * yrange, sigma0 / 20, np.nanmin(self.data_x_p)],
            [10 * yrange, np.nanmax(self.data_y_p) + 10 * yrange, sigma0 * 20, np.nanmax(self.data_x_p)],
        ]
        self.popt_str = ["A", "B", "sigma", "x_0"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "gaussian"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def lorent_zeeman(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        if self.plot_type == "2D":
            return FitResult(["x_0", "FWHM", "H", "B", "delta"], None, None, "lorent_zeeman"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        self.formula_str = r"$f(x)=H(L(\delta/2)+L(-\delta/2))+B$"

        def _lorent_zeeman(x, center, full_width, height, bg, split):
            return height * ((full_width / 2) ** 2) / ((x - center - split / 2) ** 2 + (full_width / 2) ** 2) + height * (
                (full_width / 2) ** 2
            ) / ((x - center + split / 2) ** 2 + (full_width / 2) ** 2) + bg

        self._fit_func = _lorent_zeeman
        if p0 is None:
            amp = np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)
            peaks, props = find_peaks(self.data_y_p, width=1, prominence=abs(amp) / 8 if amp else None)
            if len(peaks) == 0:
                return FitResult([], None, None, "lorent_zeeman"), None
            largest = peaks[np.argsort(self.data_y_p[peaks])[::-1]]
            step = abs(self.data_x_p[1] - self.data_x_p[0]) if len(self.data_x_p) > 1 else 1
            width = float(props["widths"][np.argsort(self.data_y_p[peaks])[-1]] * step)
            self.p0_list = []
            for second_peak in largest[: min(4, len(largest))]:
                center = self.data_x_p[int(np.mean([largest[0], second_peak]))]
                split = abs((self.data_x_p[second_peak] - center) * 2)
                self.p0_list.append([center, width or step, amp, np.nanmin(self.data_y_p), split])
        else:
            self.p0_list = [list(p0)]
        width = abs(self.p0_list[0][1]) or 1
        yrange = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
        xrange = abs(self.data_x_p[-1] - self.data_x_p[0]) or 1
        self.bounds = [
            [np.nanmin(self.data_x_p), width / 10, -10 * yrange, np.nanmin(self.data_y_p) - 10 * yrange, 0],
            [np.nanmax(self.data_x_p), width * 10, 10 * yrange, np.nanmax(self.data_y_p) + 10 * yrange, 2 * xrange],
        ]
        self.popt_str = ["x_0", "FWHM", "H", "B", "delta"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "lorent_zeeman"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def rabi(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        if self.plot_type == "2D":
            return FitResult(["A", "B", "f", "tau", "phi"], None, None, "rabi"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        self.formula_str = r"$f(x)=A\sin(2{\pi}fx+\varphi)e^{-x/\tau}+B$"

        def _rabi(x, amplitude, offset, omega, decay, phi):
            return amplitude * np.sin(2 * np.pi * omega * x + phi) * np.exp(-x / decay) + offset

        self._fit_func = _rabi
        if p0 is None:
            amp = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) / 2 or 1
            offset = np.nanmean(self.data_y_p)
            delta_x = self.data_x_p[1] - self.data_x_p[0] if len(self.data_x_p) > 1 else 1
            y_detrended = self.data_y_p - offset
            freq = np.fft.fftfreq(len(y_detrended), d=delta_x)
            vals = np.fft.fft(y_detrended)
            mask = freq > 0
            omega = abs(freq[mask][np.argmax(np.abs(vals[mask]))]) if np.any(mask) else 1 / (abs(delta_x) * len(y_detrended))
            decay = abs(self.data_x_p[-1] - self.data_x_p[0]) or 1
            self.p0_list = [[amp, offset, omega, decay, np.pi / 2], [-amp, offset, omega, decay, np.pi / 2]]
        else:
            self.p0_list = [list(p0)]
        amp0, off0, om0, dec0, phi0 = self.p0_list[0]
        yrange = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
        self.bounds = [
            [-5 * abs(amp0), off0 - 2 * yrange, max(abs(om0) / 10, 1e-15), max(abs(dec0) / 20, 1e-15), phi0 - np.pi],
            [5 * abs(amp0), off0 + 2 * yrange, max(abs(om0) * 10, 1e-15), max(abs(dec0) * 20, 1e-15), phi0 + np.pi],
        ]
        self.popt_str = ["A", "B", "f", "tau", "phi"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "rabi"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def decay(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        if self.plot_type == "2D":
            return FitResult(["A", "B", "tau"], None, None, "decay"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=3)
        self.formula_str = r"$f(x)=Ae^{-x/\tau}+B$"

        def _exp_decay(x, amplitude, offset, decay):
            return amplitude * np.exp(-x / decay) + offset

        self._fit_func = _exp_decay
        if p0 is None:
            amp = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
            offset = np.nanmean(self.data_y_p)
            decay = abs(self.data_x_p[-1] - self.data_x_p[0]) / 2 or 1
            self.p0_list = [[amp, offset, decay], [-amp, offset, decay]]
        else:
            self.p0_list = [list(p0)]
        yrange = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
        decay0 = abs(self.p0_list[0][2]) or 1
        off0 = self.p0_list[0][1]
        self.bounds = [[-4 * yrange, off0 - yrange, decay0 / 10], [4 * yrange, off0 + yrange, decay0 * 10]]
        self.popt_str = ["A", "B", "tau"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "decay"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def center(self, p0=None, is_display: bool = True, is_fit: bool = True, **kwargs):
        if self.plot_type == "1D":
            return FitResult(["A", "B", "R", "x0", "y0"], None, None, "center"), None
        self.data_x_p, self.data_y_p = self._select_fit(min_num=5)
        self.formula_str = r"$f(r)=Ae^{-(r-(x0,y0))^2/R^2}+B$"

        def _center(coord, amplitude, offset, size, x0, y0):
            x, y = np.asarray(coord[0]), np.asarray(coord[1])
            return amplitude * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / size**2) + offset

        self._fit_func = _center
        if p0 is None:
            amp = abs(np.nanmax(self.data_y_p) - np.nanmin(self.data_y_p)) or 1
            offset = np.nanmean(self.data_y_p)
            top = np.argsort(self.data_y_p)[::-1][: min(5, len(self.data_y_p))]
            size = np.hypot(np.ptp(self.data_x_p[0][top]), np.ptp(self.data_x_p[1][top])) or 1
            x0 = float(np.nanmean(self.data_x_p[0][top]))
            y0 = float(np.nanmean(self.data_x_p[1][top]))
            self.p0_list = [[amp, offset, size, x0, y0]]
        else:
            self.p0_list = [list(p0)]
        amp0, off0, size0, *_ = self.p0_list[0]
        self.bounds = [
            [-5 * abs(amp0), off0 - abs(off0) - abs(amp0), abs(size0) / 20, np.nanmin(self.data_x_p[0]), np.nanmin(self.data_x_p[1])],
            [5 * abs(amp0), off0 + abs(off0) + abs(amp0), abs(size0) * 20, np.nanmax(self.data_x_p[0]), np.nanmax(self.data_x_p[1])],
        ]
        self.popt_str = ["A", "B", "R", "x0", "y0"]
        popt, pcov = self._fit_and_draw(is_fit, is_display, kwargs)
        self.fit_func = "center"
        return FitResult(self.popt_str, popt, pcov, self.fit_func), popt

    def clear(self) -> None:
        if self.text is not None:
            self.text.remove()
            self.text = None
        if self.fit is not None:
            for artist in self.fit:
                try:
                    artist.remove()
                except Exception:
                    pass
            self.fit = None
        self._scatter_to_line()
        lines = getattr(self.live_plot, "lines", None) if self.live_plot is not None else self._ax.lines
        for line in lines:
            if hasattr(line, "set_alpha"):
                line.set_alpha(1)
        self.fig.canvas.draw_idle()

    def _line_to_scatter(self) -> None:
        if self.plot_type != "1D" or self._scatter_list:
            return
        ax = self._ax
        line = self._ax.lines[0] if self._ax.lines else None
        if line is None:
            return
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        sc = ax.scatter(x, y, s=20, color=PALETTE["data_scatter"], edgecolors="none")
        self._scatter_list.append(sc)
        line.set_visible(False)

    def _scatter_to_line(self) -> None:
        for sc in self._scatter_list:
            try:
                sc.remove()
            except Exception:
                pass
        self._scatter_list = []
        if self.fig.axes and self._ax.lines:
            self._ax.lines[0].set_visible(True)

    def _update_transform_back(self) -> None:
        transforms: list[Callable[[Any], Any]] = []
        temp_unit = self.unit
        while self.conversion_map is not None and temp_unit != self.unit_original:
            try:
                next_unit, conv_func = self.conversion_map[temp_unit]
            except KeyError:
                break
            transforms.append(conv_func)
            temp_unit = next_unit

        def _identity(x):
            return x

        def _composed(x):
            out = x
            for func in transforms:
                out = func(out)
            return out

        self.transform_back = _composed if transforms else _identity

    def _update_unit(self, transform: Callable[[Any], Any]) -> None:
        ax = self._ax
        for line in ax.lines:
            data_x = np.asarray(line.get_xdata())
            if data_x.size == 2 and np.array_equal(data_x, np.array([0, 1])):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                line.set_xdata(np.where(data_x != 0, transform(data_x), np.inf))
        if ax.lines:
            self.data_x = np.asarray(ax.lines[0].get_xdata()).reshape(-1, 1)
        xlim = ax.get_xlim()
        ax.set_xlim(transform(xlim[0]), transform(xlim[1]))

        if self.area is not None and self.area.range[0] is not None:
            self.area.range[0] = transform(self.area.range[0])
            self.area.range[1] = transform(self.area.range[1])
            try:
                self.area.selector.extents = tuple(self.area.range)
            except Exception:
                pass
        if self.cross is not None and self.cross.xy is not None:
            new_x = transform(self.cross.xy[0])
            self.cross.xy[0] = new_x
            if getattr(self.cross, "vline", None) is not None:
                self.cross.vline.set_xdata([new_x, new_x])
            if getattr(self.cross, "point", None) is not None:
                self.cross.point.set_xdata([new_x])
        if self.fit is not None and self.fit_func in VALID_FIT_FUNCS:
            prev_fit = self.fit_func
            self.clear()
            try:
                getattr(self, prev_fit)(is_display=True)
            except Exception:
                pass

    def change_unit(self) -> None:
        """Cycle wavelength/frequency or time units for 1D plots."""
        if self.plot_type == "2D" or self.conversion_map is None:
            return
        new_unit, conversion_func = self.conversion_map[self.unit]
        ax = self._ax
        old_xlabel = ax.get_xlabel()
        if re.search(r"\((.+)\)$", old_xlabel):
            ax.set_xlabel(re.sub(r"\((.+)\)$", f"({new_unit})", old_xlabel))
        else:
            ax.set_xlabel(f"{old_xlabel} ({new_unit})")
        self.unit = new_unit
        self._update_transform_back()
        self._update_unit(conversion_func)
        self.fig.canvas.draw_idle()

    def change_cmap(self, cmap: str) -> None:
        """Change image colormap and matching 2D selector colors."""
        if self.plot_type != "2D":
            return
        try:
            base_cmap = matplotlib.colormaps[cmap]
        except Exception:
            base_cmap = plt.get_cmap(cmap)
        new_cmap = base_cmap.copy()
        bad_color = getattr(self.live_plot, "bad_color", "white")
        new_cmap.set_bad(bad_color)

        ax0 = self._ax
        if not ax0.images:
            return
        mappable = ax0.images[0]
        mappable.set_cmap(new_cmap)
        cbar = getattr(self.live_plot, "cbar", None)
        if cbar is not None:
            cbar.update_normal(mappable)
        for attr, value in (("line_l", new_cmap(0.0)), ("line_h", new_cmap(0.95))):
            line = getattr(self.live_plot, attr, None)
            if line is not None:
                line.set_color(value)
        self.fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Reopening a saved figure (the read-back counterpart of ``DataFigure.save``).
# ---------------------------------------------------------------------------

#: The view-state keys ``SavedFigure`` restores and that ``plot()`` accepts as DATA options.
#: An override only reaches ``plot()`` when it is in THIS set (a saved layout may carry a
#: knob that a re-interpreted kind does not take, e.g. ``cmap`` on a 1-D plot -- it is simply
#: dropped rather than raising).  Geometry / dpi / typography are NOT here: they are frontend-
#: owned (the sealed-API contract), so a saved figure can never smuggle them back in.
_VIEW_PLOT_KWARGS: tuple[str, ...] = (
    "relim_mode", "fixed_lo", "fixed_hi", "cmap", "bins", "thresholds", "labels",
)

#: ``PanelEditor`` stores its Setting/Edit view knobs under this sub-dict of ``info`` (so the
#: raw payload stays tidy); ``relim`` there is confocal naming for ``plot()``'s ``relim_mode``.
_VIEW_INFO_KEY = "view"


def _view_to_plot_kwargs(view: Mapping[str, Any]) -> dict[str, Any]:
    """Translate a saved ``info['view']`` dict into the DATA kwargs ``plot()`` accepts.

    The stored ``relim`` (confocal naming) maps to ``plot()``'s ``relim_mode``, and ``fixed_lo``/
    ``fixed_hi`` are only forwarded when the mode is actually ``fixed`` (mirroring the live panel
    builder, which omits them otherwise so the plotter keeps its autoscale).  ``unit`` / ``size`` /
    ``repeat_mode`` live in ``info`` (not ``plot()`` kwargs) and are handled by the caller."""
    out: dict[str, Any] = {}
    relim = view.get("relim", view.get("relim_mode"))
    if relim:
        out["relim_mode"] = str(relim)
    if str(relim) == "fixed":
        for k in ("fixed_lo", "fixed_hi"):
            if view.get(k) is not None:
                out[k] = float(view[k])
    cmap = view.get("cmap")
    if cmap:
        out["cmap"] = str(cmap)
    return out


class SavedFigure:
    """A lightweight, hardware-free handle for a ``.npz`` written by :meth:`DataFigure.save`.

    It answers "what was saved" (:meth:`info_summary`), lists the plot kinds the saved data can
    be viewed as (:meth:`compatible_kinds`), and re-renders it (:meth:`plot`) through the SAME
    :func:`~.live.plot` factory the live figure used -- so the saved kind reproduces the original
    figure and any other compatible kind re-interprets the SAME ``data_x`` / ``data_y`` with a
    different plotter.  Build one with :func:`load_figure`."""

    def __init__(self, *, data_x, data_y, info: Mapping[str, Any], path=None):
        self.path = Path(path) if path is not None else None
        self.data_x = np.asarray(data_x)
        self.data_y = np.asarray(data_y)
        self.info = dict(info or {})
        self.view = dict(self.info.get(_VIEW_INFO_KEY) or {})

    # -- provenance accessors (all read from ``info``, with sane defaults for old npz) ---------
    @property
    def kind(self) -> str | None:
        return self.info.get("kind")

    @property
    def figure_recipe(self) -> Mapping[str, Any] | None:
        """The REPLAY RECIPE a structured figure stores so it can be re-rendered FAITHFULLY (not from
        ``data_x`` / ``data_y``, which for a structured figure are only a lossy fallback).  A recipe is
        ``{"kind": <family>, ...}`` -- e.g. a pulse figure stores ``{"kind": "pulse", "pulse_state":
        <PulseTableState.to_dict()>, "include_always_off": ...}``.  ``None`` for an ordinary array figure
        (a scan / hist / site map), whose ``data_x`` / ``data_y`` ARE the faithful source.  The dispatch
        is by ``recipe['kind']``, so a new structured-figure family plugs in without touching this seam."""
        recipe = self.info.get("figure_recipe")
        return recipe if isinstance(recipe, Mapping) and recipe.get("kind") else None

    @property
    def name(self) -> str | None:
        return self.info.get("name")

    @property
    def labels(self) -> list[str] | None:
        labels = self.info.get("labels")
        return list(labels) if labels is not None else None

    @property
    def unit(self) -> str | None:
        return self.info.get("unit") or self.view.get("unit")

    @property
    def shape(self) -> dict[str, tuple[int, ...]]:
        return {"data_x": tuple(self.data_x.shape), "data_y": tuple(self.data_y.shape)}

    @property
    def saved_at(self) -> str | None:
        """The save timestamp, recovered from the ``<name>_<YYYY_MM_DD_HH_MM_SS>`` file name
        (the format :meth:`DataFigure.save` writes) or ``None`` if the name does not carry one."""
        if self.path is None:
            return None
        m = re.search(r"(\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})", self.path.stem)
        return m.group(1) if m else None

    def compatible_kinds(self) -> list[str]:
        """The panel plot kinds this saved data can be viewed as, derived from the data SHAPE and
        the ``PLOT_KINDS`` ``render_family`` (never a hard-coded list): a 2-column ``data_x`` (a 2-D
        scan) offers the ``"2D"`` families (image / site map); a 1-column ``data_x`` offers the
        ``"1D"`` families (line / rolling trace / distribution).  The saved kind is always first so
        it reproduces the original figure.

        A STRUCTURED figure (one carrying a ``figure_recipe``) offers ONLY its recipe kind: its
        ``data_x`` / ``data_y`` are a lossy fallback, so re-interpreting them as a line / hist would
        NOT be a faithful view -- the recipe kind is the one true way to draw it."""
        from .live import PLOT_KINDS

        recipe = self.figure_recipe
        if recipe is not None:
            return [str(recipe.get("kind"))]

        is_2d = self.data_x.ndim == 2 and self.data_x.shape[1] >= 2
        want = "2D" if is_2d else "1D"
        kinds: list[str] = []
        for pk in PLOT_KINDS:
            if not pk.panel:
                continue
            # The site map declares ``render_family="auto"`` (image-only when it has a frame), but it
            # is a 2-D VIEW -- it needs (N, 2) centres as data_x -- so it belongs with the 2-D family.
            fam = "2D" if pk.render_family == "auto" else pk.render_family
            if fam == want:
                kinds.append(pk.key)
        saved = self.kind
        if saved in kinds:                       # saved kind FIRST (reproduces the original figure)
            kinds = [saved] + [k for k in kinds if k != saved]
        elif saved:
            kinds = [saved] + kinds
        return kinds

    def info_summary(self) -> str:
        """A human-readable multi-line answer to "what is in this file": name, source, kind, labels,
        unit, array shapes, points/repeat progress, the saved view state and any applied fit."""
        rows: list[tuple[str, Any]] = [
            ("name", self.name),
            ("source", self.info.get("source")),
            ("kind", self.kind),
            ("labels", self.labels),
            ("unit", self.unit),
            ("data_x shape", tuple(self.data_x.shape)),
            ("data_y shape", tuple(self.data_y.shape)),
            ("points_done", self.info.get("points_done")),
            ("repeat_cur", self.info.get("repeat_cur")),
            ("saved_at", self.saved_at),
        ]
        if self.info.get("size"):
            rows.append(("size", self.info.get("size")))
        if self.view:
            view_bits = ", ".join(f"{k}={v}" for k, v in self.view.items() if v is not None)
            rows.append(("view", view_bits or "(defaults)"))
        fit = self.info.get("fit")
        if isinstance(fit, Mapping) and fit.get("func"):
            pairs = ", ".join(f"{n}={float(v):.5g}"
                              for n, v in zip(fit.get("names", []), fit.get("popt", [])))
            rows.append(("fit", f"{fit['func']}: {pairs}" if pairs else str(fit["func"])))
        width = max((len(k) for k, _ in rows), default=0)
        header = f"SavedFigure({self.path.name})" if self.path is not None else "SavedFigure"
        lines = [header] + [f"  {k.ljust(width)} : {v}" for k, v in rows if v is not None]
        return "\n".join(lines)

    def plot(self, kind: str | None = None, size: str | None = None, **overrides):
        """Re-render this saved figure as a :class:`DataFigure` -- DATA-viewing semantics: the SAME
        ``data_x`` / ``data_y`` are handed to :func:`~.live.plot` under ``kind`` (the saved kind by
        default, which reproduces the original figure; any :meth:`compatible_kinds` value instead
        re-interprets the data with a different plotter).  The saved view state (relim / fixed lo-hi /
        cmap / labels) seeds the plotter and ``overrides`` win over it.  Only DATA kwargs
        (:data:`_VIEW_PLOT_KWARGS`) reach ``plot()`` -- geometry / dpi / typography stay frontend-owned,
        and a saved knob a re-interpreted kind does not accept is dropped, never raised.

        When this figure carries a ``figure_recipe`` (a STRUCTURED figure such as a pulse timeline),
        the reproduction goes through :meth:`_replay_recipe` instead: it re-renders from the recipe's
        own source (e.g. rebuilds the ``PulseTableState`` and re-draws the pulse figure), so the reopened
        figure is FAITHFUL (all channels / analog traces / brackets), not a flattened line off the
        fallback arrays.  ``kind`` / ``size`` / ``overrides`` are ignored for a recipe figure -- the
        recipe is self-describing."""
        recipe = self.figure_recipe
        if recipe is not None:
            return self._replay_recipe(recipe)
        return self._plot_from_arrays(kind, size, **overrides)

    def _replay_recipe(self, recipe: Mapping[str, Any]) -> "DataFigure":
        """Re-render a STRUCTURED figure from its ``figure_recipe`` -- a faithful reproduction from the
        recipe's OWN source, not the fallback ``data_x`` / ``data_y``.  Dispatch is by ``recipe['kind']``,
        so each structured-figure family owns its rebuild here (and a new family adds a branch); an
        unknown recipe kind falls back to the ordinary array ``plot`` so nothing crashes.

        ``pulse`` -> rebuild the ``PulseTableState`` and re-draw the pulse figure through the SAME
        :func:`~.pulse_gui.build_pulse_preview_plot` the editor uses, so every digital channel / analog
        bus trace / repeat bracket comes back exactly as saved.  The rebuild imports ``pulse_gui`` LAZILY
        (only when a pulse recipe is actually reopened) so the ordinary array-figure reload path stays
        free of the pulse-editor / PyQt import."""
        rkind = str(recipe.get("kind") or "")
        if rkind == "pulse":
            return self._replay_pulse(recipe)
        # Unknown structured kind: fall back to the ordinary array reproduction (never crash on reopen).
        return self._plot_from_arrays()

    def _pulse_state_dict(self, recipe: Mapping[str, Any]) -> dict:
        """The ``PulseTableState.to_dict()`` payload to rebuild the pulse figure from -- resolved from ONE
        serialization format (``PulseTableState.to_dict``) with provenance PREFERRED over the recipe copy.

        A fired pulse's provenance already carries the applied state: the sequencer's ``.snapshot()``
        records ``last_payload_json = json.dumps(PulseTableState.to_dict())`` (the exact same format), and
        the rich save folds that under ``info['provenance']['devices']['sequencer']``.  So when the save
        HAS provenance with a sequencer payload, THAT is the reproduction source (single source with the
        device state the run was taken under -- no second copy needed); the recipe's own ``pulse_state``
        is the FALLBACK for a pure preview that was never fired (the standalone editor has no device, so
        the editor's current ``PulseTableState`` is the only truth for what the preview drew)."""
        prov = self.info.get("provenance")
        devices = prov.get("devices") if isinstance(prov, Mapping) else None
        seq = devices.get("sequencer") if isinstance(devices, Mapping) else None
        payload = seq.get("last_payload_json") if isinstance(seq, Mapping) else None
        if payload:
            import json
            try:
                data = json.loads(payload) if isinstance(payload, str) else dict(payload)
                if isinstance(data, Mapping) and "periods" in data:
                    return dict(data)                        # the device-applied state (provenance-sourced)
            except Exception:
                pass
        return dict(recipe.get("pulse_state") or {})         # preview fallback (editor state, never fired)

    def _replay_pulse(self, recipe: Mapping[str, Any]) -> "DataFigure":
        """Rebuild the pulse figure: ``PulseTableState.from_dict`` (provenance-preferred source) ->
        :func:`~.pulse_gui.build_pulse_preview_plot` -> its :class:`DataFigure`.  Both source paths feed
        the ONE renderer, so a fired pulse (rebuilt from its provenance) and a pure preview (rebuilt from
        the recipe) draw through identical code."""
        from Zou_lab_control.neutral_atom.timing.pulse_table import PulseTableState

        state = PulseTableState.from_dict(self._pulse_state_dict(recipe))
        include_always_off = bool(recipe.get("include_always_off", True))
        from .pulse_gui import build_pulse_preview_plot

        plotter, _channels, _repeat = build_pulse_preview_plot(
            state, include_always_off=include_always_off)
        df = plotter.to_data_figure()
        df.info = {**df.info, **{k: v for k, v in self.info.items() if k != "labels"}}
        df.name = self.name or df.name
        return df

    def _plot_from_arrays(self, kind: str | None = None, size: str | None = None, **overrides):
        """The ordinary array-figure reproduction (``data_x`` / ``data_y`` -> :func:`~.live.plot`) -- the
        ONE array path, used by :meth:`plot` for a non-recipe figure AND as the fallback when a recipe's
        kind is unknown (so there is a single array-reproduction implementation, not two copies)."""
        from .live import panel_plot_spec, plot as _plot

        use_kind = kind or self.kind or "auto"
        use_size = size or self.info.get("size", "2x2")
        kwargs = _view_to_plot_kwargs(self.view)
        if self.labels is not None:
            kwargs.setdefault("labels", self.labels)
        kwargs.update(overrides)
        kwargs = {k: v for k, v in kwargs.items() if k in _VIEW_PLOT_KWARGS}
        try:
            spec = panel_plot_spec(use_size)
        except Exception:
            spec = None
        plotter = _plot(self.data_x, self.data_y, kind=use_kind, _spec=spec,
                        display=False, data_figure=True, update=False, **kwargs)
        df = getattr(plotter, "data_figure", None) or plotter.to_data_figure()
        df.info = {**df.info, **{k: v for k, v in self.info.items() if k != "labels"}}
        if self.name:
            df.name = self.name
        return df


def load_figure(path) -> SavedFigure:
    """Load a ``.npz`` saved by :meth:`DataFigure.save` into a :class:`SavedFigure`.

    The read-back counterpart of save: with no hardware or session, reopen an overnight scan /
    a saved panel to inspect what it holds (``.info_summary()``), list how it can be viewed
    (``.compatible_kinds()``) or re-render it (``.plot()`` / ``.plot(kind=...)``).  Robust to old
    payloads that carry only ``data_x`` / ``data_y`` and a minimal ``info``."""
    path = Path(path)
    data = np.load(str(path), allow_pickle=True)
    if "data_x" not in data.files or "data_y" not in data.files:
        raise ValueError(f"{path} is not a DataFigure save (missing data_x/data_y).")
    info = data["info"].item() if "info" in data.files else {}
    if not isinstance(info, Mapping):
        info = {}
    return SavedFigure(data_x=data["data_x"], data_y=data["data_y"], info=info, path=path)


__all__ = ["DataFigure", "FitResult", "SavedFigure", "VALID_FIT_FUNCS", "load_figure"]

