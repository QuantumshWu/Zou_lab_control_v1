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


class DataFigure:
    """Data and post-processing handle for a front-end figure.

    A DataFigure can be created from a ``Live1D``/``Live2DDis``/``HistogramFigure``
    object, or from explicit ``fig``, ``data_x`` and ``data_y`` handles.
    """

    def __init__(
        self,
        live_plot=None,
        *,
        fig: plt.Figure | None = None,
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

        first_ax = self.fig.axes[0]
        self.plot_type = "2D" if first_ax.images else "1D"
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
        self.fig.axes[0].set_xlim(x_min, x_max)
        self.fig.canvas.draw_idle()

    def ylim(self, y_min: float, y_max: float) -> None:
        self.fig.axes[0].set_ylim(y_min, y_max)
        self.fig.canvas.draw_idle()

    def save(
        self,
        path: str | Path = "",
        *,
        extra_info: Mapping[str, Any] | None = None,
        image_ext: str = "png",
    ) -> dict[str, Path]:
        """Save the figure image and a matching ``.npz`` payload."""
        current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        stem = "_".join(p for p in (self.name, current_time) if p)
        path = Path(path)
        if str(path) in ("", "."):
            base = Path(stem)
        elif str(path).endswith(("/", "\\")) or path.is_dir():
            base = path / stem
        elif path.suffix:
            base = path.with_suffix("")
        else:
            base = Path(f"{path}_{stem}")
        base.parent.mkdir(parents=True, exist_ok=True)

        image_path = base.with_suffix(f".{image_ext}")
        data_path = base.with_suffix(".npz")
        info = {
            **self.info,
            **dict(extra_info or {}),
            "labels": self.labels,
            "name": self.name,
            "unit": self.unit_original,
            "points_done": getattr(self.live_plot, "points_done", len(self.data_x)),
            "repeat_cur": getattr(self.live_plot, "repeat_cur", 1),
        }
        self.fig.savefig(image_path, bbox_inches="tight")
        np.savez(data_path, data_x=self.data_x_original, data_y=self.data_y, info=info)
        return {"figure": image_path, "data": data_path}

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
                xlim = self.fig.axes[0].get_xlim()
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
            xl, xh = sorted(self.fig.axes[0].get_xlim())
            yl, yh = sorted(self.fig.axes[0].get_ylim())
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
                self.text = self.fig.axes[0].text(
                    0.5,
                    0.5,
                    result,
                    transform=self.fig.axes[0].transAxes,
                    color="blue",
                    ha="center",
                    va="center",
                    fontsize=matplotlib.rcParams["legend.fontsize"],
                )
            else:
                self.text.set_text(result)
            self._place_text(self.fig.axes[0], self.text)
        elif self.text is not None:
            self.text.remove()
            self.text = None

        lines = getattr(self.live_plot, "lines", None) if self.live_plot is not None else self.fig.axes[0].lines
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
        ax = self.fig.axes[0]
        if self.plot_type == "1D":
            yfit = self._fit_func(self.data_x[:, 0], *popt)
            if self.fit is None:
                self.fit = ax.plot(self.data_x[:, 0], yfit, color="orange", linestyle="-", linewidth=2, alpha=0.5)
            else:
                self.fit[0].set_data(self.data_x[:, 0], yfit)
        else:
            if self.fit is None:
                self.fit = [ax.scatter(popt[-2], popt[-1], color="orange", s=50)]
                circle = matplotlib.patches.Circle(
                    (popt[-2], popt[-1]),
                    radius=abs(popt[-3]),
                    edgecolor="orange",
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
        lines = getattr(self.live_plot, "lines", None) if self.live_plot is not None else self.fig.axes[0].lines
        for line in lines:
            if hasattr(line, "set_alpha"):
                line.set_alpha(1)
        self.fig.canvas.draw_idle()

    def _line_to_scatter(self) -> None:
        if self.plot_type != "1D" or self._scatter_list:
            return
        ax = self.fig.axes[0]
        line = self.fig.axes[0].lines[0] if self.fig.axes[0].lines else None
        if line is None:
            return
        x = np.asarray(line.get_xdata())
        y = np.asarray(line.get_ydata())
        sc = ax.scatter(x, y, s=20, color="lightgrey", edgecolors="none")
        self._scatter_list.append(sc)
        line.set_visible(False)

    def _scatter_to_line(self) -> None:
        for sc in self._scatter_list:
            try:
                sc.remove()
            except Exception:
                pass
        self._scatter_list = []
        if self.fig.axes and self.fig.axes[0].lines:
            self.fig.axes[0].lines[0].set_visible(True)

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
        ax = self.fig.axes[0]
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
        ax = self.fig.axes[0]
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

        ax0 = self.fig.axes[0]
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


__all__ = ["DataFigure", "FitResult", "VALID_FIT_FUNCS"]

