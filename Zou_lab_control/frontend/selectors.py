"""Interactive Matplotlib selectors for notebook front-end figures."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Optional

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np


@dataclass
class PlotState:
    """Small metadata object shared by plots, selectors, and DataFigure."""

    plot_type: str
    x_array: np.ndarray | None = None
    y_array: np.ndarray | None = None
    grid: np.ndarray | None = None
    axdis: plt.Axes | None = None
    cax: plt.Axes | None = None
    extents_square: list[float] | None = None
    bad_color: str = "white"


@dataclass
class InteractionBundle:
    area: Optional["AreaSelector"] = None
    cross: Optional["CrossSelector"] = None
    zoom: Optional["ZoomPan"] = None
    drag: Optional["DragHLine"] = None
    axdis: Optional[plt.Axes] = None
    cax: Optional[plt.Axes] = None


def _format_xy_precision(ax: plt.Axes) -> tuple[int, int]:
    xspan = abs(np.nanmax(ax.get_xlim()) - np.nanmin(ax.get_xlim()))
    yspan = abs(np.nanmax(ax.get_ylim()) - np.nanmin(ax.get_ylim()))
    gap_x = xspan / 1000 if xspan else 0.01
    gap_y = yspan / 1000 if yspan else 0.01
    dx = max(0, -int(np.ceil(np.log10(gap_x))))
    dy = max(0, -int(np.ceil(np.log10(gap_y))))
    return dx, dy


def _default_interaction_color(ax: plt.Axes, fallback: str = "grey"):
    if ax.images:
        return ax.images[0].get_cmap()(0.95)
    return fallback


class AreaSelector:
    """Left-drag rectangular selector with displayed coordinates."""

    def __init__(self, ax: plt.Axes, color=None):
        self.ax = ax
        self.text = None
        self.range = [None, None, None, None]
        self.callback: Optional[Callable[[], None]] = None
        self.color = _default_interaction_color(ax) if color is None else color
        self.selector = RectangleSelector(
            ax,
            self.onselect,
            interactive=True,
            useblit=False,
            button=[1],
            props=dict(alpha=0.8, fill=False, linestyle="-", color=self.color),
            handle_props=dict(
                marker="s",
                markersize=matplotlib.rcParams["legend.fontsize"] / 2,
                markeredgecolor=self.color,
                markerfacecolor="white",
                markeredgewidth=matplotlib.rcParams["lines.linewidth"] / 2,
            ),
        )

    def _call(self) -> None:
        if self.callback is not None:
            self.callback()

    def onselect(self, eclick, erelease) -> None:
        if getattr(self.ax.figure, "_zlc_disable_area_once", False):
            self.ax.figure._zlc_disable_area_once = False
            self.range = [None, None, None, None]
            return
        x1, x2, y1, y2 = self.selector.extents
        if x1 == x2 or y1 == y2:
            self.range = [None, None, None, None]
            if self.text is not None:
                self.text.remove()
                self.text = None
            self.ax.figure.canvas.draw_idle()
            self._call()
            return

        self.range = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        dx, dy = _format_xy_precision(self.ax)
        fmt = f"{{:.{dx}f}}, {{:.{dy}f}}"
        label = f"({fmt.format(x1, y1)})\n({fmt.format(x2, y2)})"

        if self.text is None:
            self.text = self.ax.text(
                0.025,
                0.975,
                label,
                transform=self.ax.transAxes,
                color=self.color,
                ha="left",
                va="top",
                fontsize=matplotlib.rcParams["legend.fontsize"],
            )
        else:
            self.text.set_text(label)
        self.ax.figure.canvas.draw_idle()
        self._call()

    def destroy(self) -> None:
        try:
            self.selector.set_active(False)
            self.selector.disconnect_events()
        except Exception:
            pass
        if self.text is not None:
            try:
                self.text.remove()
            except Exception:
                pass
            self.text = None


class CrossSelector:
    """Right-click crosshair; double right-click clears it."""

    def __init__(self, ax: plt.Axes, color=None):
        self.ax = ax
        self.point = None
        self.xy = None
        self.callback: Optional[Callable[[], None]] = None
        self.last_click_time = None
        self.color = _default_interaction_color(ax) if color is None else color
        self.color_dis = self.color
        if ax.images:
            self.color_dis = ax.images[0].get_cmap()(0.55)
        self._dis_line = None
        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)

    def _call(self) -> None:
        if self.callback is not None:
            self.callback()

    def _image_z(self, x: float, y: float) -> float | None:
        state: PlotState | None = getattr(self.ax.figure, "_zlc_state", None)
        if state is None or state.grid is None or state.x_array is None or state.y_array is None:
            return None
        try:
            ix = int(np.argmin(np.abs(np.asarray(state.x_array) - x)))
            iy = int(np.argmin(np.abs(np.asarray(state.y_array) - y)))
            return float(np.asarray(state.grid)[iy, ix])
        except Exception:
            return None

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or event.button != 3:
            return

        now = time.time()
        is_double = bool(getattr(event, "dblclick", False))
        if self.last_click_time is not None and (now - self.last_click_time) < 0.35:
            is_double = True

        if is_double:
            self.last_click_time = None
            self.remove_point()
            self.ax.figure.canvas.draw_idle()
            self._call()
            return
        self.last_click_time = now

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        dx, dy = _format_xy_precision(self.ax)
        fmt = f"{{:.{dx}f}}, {{:.{dy}f}}"

        zval = self._image_z(x, y) if self.ax.images else None
        z_suffix = ""
        if zval is not None:
            z_suffix = f", {zval:.6g}" if np.isfinite(zval) else ", NaN"
        label = f"({fmt.format(x, y)}{z_suffix})"
        self.xy = [x, y]

        if self.point is None:
            self.vline = self.ax.axvline(x, color=self.color, linestyle="-", alpha=0.8)
            self.hline = self.ax.axhline(y, color=self.color, linestyle="-", alpha=0.8)
            self.text = self.ax.text(
                0.975,
                0.975,
                label,
                transform=self.ax.transAxes,
                color=self.color,
                ha="right",
                va="top",
                fontsize=matplotlib.rcParams["legend.fontsize"],
            )
            (self.point,) = self.ax.plot(x, y, "o", alpha=0.8, color=self.color)
        else:
            self.vline.set_xdata([x, x])
            self.hline.set_ydata([y, y])
            self.point.set_xdata([x])
            self.point.set_ydata([y])
            self.text.set_text(label)

        state: PlotState | None = getattr(self.ax.figure, "_zlc_state", None)
        axdis = getattr(state, "axdis", None)
        if axdis is not None and zval is not None and np.isfinite(zval):
            if self._dis_line is None:
                self._dis_line = axdis.axhline(
                    zval,
                    color=self.color_dis,
                    linewidth=matplotlib.rcParams["legend.fontsize"] / 4,
                    alpha=0.3,
                )
            else:
                self._dis_line.set_ydata([zval, zval])
        elif self._dis_line is not None:
            self._dis_line.remove()
            self._dis_line = None

        self.ax.figure.canvas.draw_idle()
        self._call()

    def remove_point(self) -> None:
        for name in ("vline", "hline", "point", "text", "_dis_line"):
            artist = getattr(self, name, None)
            if artist is not None:
                try:
                    artist.remove()
                except Exception:
                    pass
                setattr(self, name, None)
        self.xy = None

    def destroy(self) -> None:
        try:
            self.ax.figure.canvas.mpl_disconnect(self.cid_press)
        except Exception:
            pass
        self.remove_point()


class ZoomPan:
    """Scroll to zoom, middle drag to pan, middle double-click to reset/zoom area."""

    def __init__(self, ax: plt.Axes, area_selector: AreaSelector | None = None, zoom_scale: float = 1.1):
        self.ax = ax
        self.area_selector = area_selector
        self.zoom_scale = float(zoom_scale)
        self.callback: Optional[Callable[[], None]] = None
        self.dragging = False
        self._press_xy = None
        self._xlim0 = None
        self._ylim0 = None
        self._home_xlim = ax.get_xlim()
        self._home_ylim = ax.get_ylim()
        self.image_type = "2D" if ax.images else "1D"
        state: PlotState | None = getattr(ax.figure, "_zlc_state", None)
        if self.image_type == "2D" and state is not None:
            ax.set_facecolor(state.bad_color)

        self.cid_scroll = ax.figure.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.cid_press = ax.figure.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_motion = ax.figure.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release = ax.figure.canvas.mpl_connect("button_release_event", self.on_release)

    def _call(self) -> None:
        if self.callback is not None:
            self.callback()

    def _set_limits(self, xlim, ylim) -> None:
        self.ax.set_xlim(xlim)
        if self.image_type == "2D":
            self.ax.set_ylim(ylim)

    def on_scroll(self, event) -> None:
        if event.inaxes != self.ax:
            return
        x = event.xdata if event.xdata is not None else float(np.mean(self.ax.get_xlim()))
        y = event.ydata if event.ydata is not None else float(np.mean(self.ax.get_ylim()))
        # Scroll DOWN zooms in (smaller view range), scroll UP zooms out.
        scale = self.zoom_scale if event.button == "up" else 1 / self.zoom_scale
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_xlim = [x - (x - xlim[0]) * scale, x + (xlim[1] - x) * scale]
        new_ylim = [y - (y - ylim[0]) * scale, y + (ylim[1] - y) * scale]
        self._set_limits(new_xlim, new_ylim)
        self.ax.figure.canvas.draw_idle()
        self._call()

    def on_press(self, event) -> None:
        if event.inaxes != self.ax or event.button != 2:
            return
        if event.dblclick:
            if self.area_selector is not None and self.area_selector.range[0] is not None:
                xl, xh, yl, yh = self.area_selector.range
                self.ax.set_xlim(xl, xh)
                if self.image_type == "2D":
                    self.ax.set_ylim(yl, yh)
            else:
                state: PlotState | None = getattr(self.ax.figure, "_zlc_state", None)
                ext = getattr(state, "extents_square", None)
                if self.image_type == "2D" and ext is not None:
                    self.ax.set_xlim(ext[0], ext[1])
                    self.ax.set_ylim(ext[2], ext[3])
                else:
                    self._set_limits(self._home_xlim, self._home_ylim)
            self.ax.figure.canvas.draw_idle()
            self._call()
            return

        self.dragging = True
        self._press_xy = (event.xdata, event.ydata)
        self._xlim0 = self.ax.get_xlim()
        self._ylim0 = self.ax.get_ylim()

    def on_motion(self, event) -> None:
        if not self.dragging or event.inaxes != self.ax or self._press_xy is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - self._press_xy[0]
        dy = event.ydata - self._press_xy[1]
        self.ax.set_xlim(self._xlim0[0] - dx, self._xlim0[1] - dx)
        if self.image_type == "2D":
            self.ax.set_ylim(self._ylim0[0] - dy, self._ylim0[1] - dy)
        self.ax.figure.canvas.draw_idle()
        self._call()

    def on_release(self, event) -> None:
        self.dragging = False
        self._press_xy = None

    def destroy(self) -> None:
        canvas = self.ax.figure.canvas
        for cid in (self.cid_scroll, self.cid_press, self.cid_motion, self.cid_release):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass


class DragHLine:
    """Drag low/high horizontal indicator lines, typically for image clim."""

    def __init__(self, line_l, line_h, callback, ax: plt.Axes):
        self.line_l = line_l
        self.line_h = line_h
        self.callback = callback
        self.ax = ax
        self.dragging = None
        fig = ax.figure
        self.cid_press = fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self.on_release)

    def _near(self, event, line) -> bool:
        if event.inaxes != self.ax or event.ydata is None:
            return False
        y = float(line.get_ydata()[0])
        ylim = self.ax.get_ylim()
        tol = abs(ylim[1] - ylim[0]) * 0.03
        return abs(event.ydata - y) <= tol

    def on_press(self, event) -> None:
        if event.button != 1:
            return
        if self._near(event, self.line_l):
            self.dragging = self.line_l
        elif self._near(event, self.line_h):
            self.dragging = self.line_h

    def on_motion(self, event) -> None:
        if self.dragging is None or event.inaxes != self.ax or event.ydata is None:
            return
        y = float(event.ydata)
        other = self.line_h if self.dragging is self.line_l else self.line_l
        oy = float(other.get_ydata()[0])
        y = min(y, oy) if self.dragging is self.line_l else max(y, oy)
        self.dragging.set_ydata([y, y])
        if self.callback is not None:
            self.callback()
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event) -> None:
        self.dragging = None

    def destroy(self) -> None:
        canvas = self.ax.figure.canvas
        for cid in (self.cid_press, self.cid_motion, self.cid_release):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass


class DragVLine:
    """Drag one vertical line and call back whenever its x-position changes."""

    def __init__(self, line, callback, ax: plt.Axes):
        self.line = line
        self.callback = callback
        self.ax = ax
        self.dragging = False
        fig = ax.figure
        self.cid_press = fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_motion = fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_release = fig.canvas.mpl_connect("button_release_event", self.on_release)

    def _set_area_active(self, active: bool) -> None:
        tools = getattr(self.ax.figure, "_zlc_tools", None)
        area = getattr(tools, "area", None)
        selector = getattr(area, "selector", None)
        if selector is not None:
            try:
                selector.set_active(active)
            except Exception:
                pass

    def _near(self, event) -> bool:
        if event.inaxes != self.ax or event.xdata is None:
            return False
        x = float(self.line.get_xdata()[0])
        xlim = self.ax.get_xlim()
        tol = abs(xlim[1] - xlim[0]) * 0.02
        return abs(event.xdata - x) <= tol

    def on_press(self, event) -> None:
        if event.button == 1 and self._near(event):
            self.dragging = True
            self.ax.figure._zlc_disable_area_once = True
            self._set_area_active(False)

    def on_motion(self, event) -> None:
        if not self.dragging or event.inaxes != self.ax or event.xdata is None:
            return
        x = float(event.xdata)
        self.line.set_xdata([x, x])
        if self.callback is not None:
            self.callback(x)
        self.ax.figure.canvas.draw_idle()

    def on_release(self, event) -> None:
        if self.dragging:
            self._set_area_active(True)
        self.dragging = False

    def destroy(self) -> None:
        canvas = self.ax.figure.canvas
        for cid in (self.cid_press, self.cid_motion, self.cid_release):
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass


def attach_interaction(
    ax: plt.Axes,
    *,
    area: bool = True,
    cross: bool = True,
    zoompan: bool = True,
    drag: DragHLine | None = None,
    axdis: plt.Axes | None = None,
    cax: plt.Axes | None = None,
) -> InteractionBundle:
    """Attach selector tools and keep strong references on the figure."""
    area_sel = AreaSelector(ax) if area else None
    cross_sel = CrossSelector(ax) if cross else None
    zoom_sel = ZoomPan(ax, area_selector=area_sel) if zoompan else None
    tools = InteractionBundle(area=area_sel, cross=cross_sel, zoom=zoom_sel, drag=drag, axdis=axdis, cax=cax)
    ax.figure._zlc_tools = tools
    return tools


__all__ = [
    "AreaSelector",
    "CrossSelector",
    "DragHLine",
    "DragVLine",
    "InteractionBundle",
    "PlotState",
    "ZoomPan",
    "attach_interaction",
]
