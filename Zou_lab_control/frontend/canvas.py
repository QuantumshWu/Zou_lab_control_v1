"""Figure lifecycle and fixed-pixel layout utilities for Jupyter plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

from .style import apply_style

try:
    from IPython import get_ipython
    from IPython.display import display
except Exception:  # pragma: no cover - only absent outside IPython.
    get_ipython = None
    display = None


@dataclass(frozen=True)
class FigureSpec:
    """Fixed logical-pixel layout for a notebook figure."""

    data_px: tuple[int, int] = (480, 360)
    margins_px: tuple[int, int, int, int] = (110, 110, 100, 40)
    dpi: int = 300


_CELL_FIGS: dict[str, list[int]] = {}
_CELL_RUNS: dict[str, str] = {}
_FIG_COUNTER = 0
_CACHED_DPR = 1.0


def _is_widget_backend() -> bool:
    backend = str(matplotlib.get_backend()).lower()
    return "ipympl" in backend or "widget" in backend


def _get_notebook_context() -> tuple[Optional[str], Optional[str]]:
    if get_ipython is None:
        return None, None
    try:
        ip = get_ipython()
        if ip is None or not hasattr(ip, "kernel"):
            return None, None
        parent = ip.kernel.get_parent()
        metadata = parent.get("metadata", {})
        header = parent.get("header", {})
        return metadata.get("cellId"), header.get("msg_id") or parent.get("msg_id")
    except Exception:
        return None, None


def _destroy_frontend_tools(fig: plt.Figure) -> None:
    for attr in ("_zlc_tools", "_npt_tools"):
        tools = getattr(fig, attr, None)
        if tools is None:
            continue
        for name in ("area", "cross", "zoom", "drag"):
            handler = getattr(tools, name, None)
            if handler is not None and hasattr(handler, "destroy"):
                try:
                    handler.destroy()
                except Exception:
                    pass
        setattr(fig, attr, None)


def _close_fig_num(fig_num: int) -> None:
    global _CACHED_DPR
    try:
        if not plt.fignum_exists(fig_num):
            return
        fig = plt.figure(fig_num)
        dpr = getattr(fig.canvas, "device_pixel_ratio", 1)
        if dpr and dpr > 1:
            _CACHED_DPR = float(dpr)
        _destroy_frontend_tools(fig)
        plt.close(fig_num)
    except Exception:
        pass


def close_all() -> None:
    """Close all figures tracked by this front-end package."""
    for nums in list(_CELL_FIGS.values()):
        for fig_num in nums:
            _close_fig_num(fig_num)
    _CELL_FIGS.clear()
    _CELL_RUNS.clear()


def configure_canvas(fig: plt.Figure, *, capture_scroll: bool = True) -> None:
    """Hide ipympl chrome and configure scroll capture when those attrs exist."""
    canvas = getattr(fig, "canvas", None)
    if canvas is None:
        return
    for attr, value in (
        ("toolbar_visible", False),
        ("header_visible", False),
        ("footer_visible", False),
        ("resizable", False),
        ("capture_scroll", capture_scroll),
    ):
        if hasattr(canvas, attr):
            try:
                setattr(canvas, attr, value)
            except Exception:
                pass


def new_figure(*, spec: FigureSpec | None = None, track_cell: bool = True) -> plt.Figure:
    """Create a fresh figure, closing only stale figures from the same cell rerun."""
    global _FIG_COUNTER

    apply_style({"figure.dpi": spec.dpi} if spec is not None else None)
    cell_id, run_id = _get_notebook_context()

    if track_cell and cell_id is not None and run_id is not None:
        old_run = _CELL_RUNS.get(cell_id)
        if old_run is not None and old_run != run_id:
            for fig_num in _CELL_FIGS.pop(cell_id, []):
                _close_fig_num(fig_num)
        _CELL_RUNS[cell_id] = run_id

    _FIG_COUNTER += 1
    with plt.ioff():
        fig = plt.figure(num=_FIG_COUNTER, dpi=(spec.dpi if spec is not None else None))
    configure_canvas(fig)

    if track_cell and cell_id is not None:
        _CELL_FIGS.setdefault(cell_id, []).append(fig.number)
    return fig


def display_figure(fig: plt.Figure) -> None:
    """Display and draw a figure after all artists/layout are configured."""
    global _CACHED_DPR

    is_widget = display is not None and _is_widget_backend()
    if is_widget:
        canvas = fig.canvas
        for num in plt.get_fignums():
            if num == fig.number:
                continue
            try:
                dpr = getattr(plt.figure(num).canvas, "device_pixel_ratio", 1)
                if dpr and dpr > 1:
                    _CACHED_DPR = float(dpr)
                    break
            except Exception:
                pass
        if _CACHED_DPR > 1 and hasattr(canvas, "_set_device_pixel_ratio"):
            try:
                canvas._set_device_pixel_ratio(_CACHED_DPR)
            except Exception:
                pass
        display(canvas)
        for msg in ("send_image_mode", "refresh", "initialized", "draw"):
            try:
                canvas._handle_message(canvas, {"type": msg}, [])
            except Exception:
                pass

    fig.canvas.draw()
    if is_widget:
        try:
            fig.canvas.flush_events()
            fig.canvas._force_full = True
            fig.canvas.draw_idle()
        except Exception:
            pass


def design_dpi(fig: plt.Figure) -> float:
    """Return the logical design dpi rather than a HiDPI-boosted canvas dpi."""
    dpi = getattr(fig, "_original_dpi", None)
    if dpi is None:
        dpi = matplotlib.rcParams["figure.dpi"]
    return float(dpi)


def create_axes_fixed(
    fig: plt.Figure,
    data_px: tuple[int, int] = (480, 360),
    margins_px: tuple[int, int, int, int] = (110, 110, 100, 40),
) -> plt.Axes:
    """Create one axes with a fixed logical-pixel data box and margins."""
    dpi = design_dpi(fig)
    w_in = data_px[0] / dpi
    h_in = data_px[1] / dpi
    L, R, B, T = [m / dpi for m in margins_px]

    fig_w = L + w_in + R
    fig_h = B + h_in + T
    fig.set_size_inches(fig_w, fig_h, forward=True)

    ax = fig.add_axes([0, 0, 1, 1])
    divider = Divider(
        fig,
        (0, 0, 1, 1),
        horizontal=[Size.Fixed(L), Size.Fixed(w_in), Size.Fixed(R)],
        vertical=[Size.Fixed(B), Size.Fixed(h_in), Size.Fixed(T)],
    )
    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    fig._zlc_fixed_box_in = (w_in, h_in)
    fig._zlc_fixed_bounds_frac = (L / fig_w, B / fig_h, w_in / fig_w, h_in / fig_h)
    return ax


def split_axes_horizontally(
    fig: plt.Figure,
    main_ax: plt.Axes,
    widths_rel: Sequence[float],
    pads_rel: Sequence[float],
) -> list[plt.Axes]:
    """Split the fixed data box into columns; the first axes is reused."""
    if not hasattr(fig, "_zlc_fixed_box_in") or not hasattr(fig, "_zlc_fixed_bounds_frac"):
        raise RuntimeError("Call create_axes_fixed(fig, ...) before splitting axes.")
    if len(pads_rel) != len(widths_rel) - 1:
        raise ValueError("pads_rel must have length len(widths_rel)-1.")

    w_in, h_in = fig._zlc_fixed_box_in
    bounds = fig._zlc_fixed_bounds_frac
    horiz: list[Any] = []
    for i, width in enumerate(widths_rel):
        horiz.append(Size.Fixed(float(width) * w_in))
        if i < len(pads_rel):
            horiz.append(Size.Fixed(float(pads_rel[i]) * w_in))

    subdiv = Divider(fig, bounds, horizontal=horiz, vertical=[Size.Fixed(h_in)])
    main_ax.set_axes_locator(subdiv.new_locator(nx=0, ny=0))
    axes = [main_ax]
    for i in range(1, len(widths_rel)):
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axes_locator(subdiv.new_locator(nx=2 * i, ny=0))
        axes.append(ax)
    return axes


def auto_data_size_px(
    ncols: int = 1,
    nrows: int = 1,
    aspect: float | None = None,
    min_w: int = 420,
    max_w: int = 760,
    base_h: int = 360,
) -> tuple[int, int]:
    """Choose a conservative fixed data-box size for notebook plots."""
    ncols = max(1, int(ncols))
    nrows = max(1, int(nrows))
    if aspect is None:
        width = 480 if ncols == 1 else min(max_w, 420 + 80 * (ncols - 1))
        height = base_h if nrows == 1 else min(620, base_h + 90 * (nrows - 1))
        return int(width), int(height)
    width = int(np_clip(base_h * float(aspect), min_w, max_w))
    return width, int(base_h)


def np_clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def save_figure_data(
    fig: plt.Figure,
    *,
    data: dict[str, Any] | None = None,
    path: str | Path = "figure",
    image_ext: str = "png",
    extra: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Save a figure plus an optional NumPy-friendly payload."""
    path = Path(path)
    base = path.with_suffix("") if path.suffix else path
    base.parent.mkdir(parents=True, exist_ok=True)

    image_path = base.with_suffix(f".{image_ext}")
    fig.savefig(image_path, bbox_inches="tight")
    out = {"figure": image_path}

    if data is not None:
        import numpy as np

        payload = dict(data)
        if extra:
            payload["info"] = extra
        data_path = base.with_suffix(".npz")
        np.savez(data_path, **payload)
        out["data"] = data_path
    return out


__all__ = [
    "FigureSpec",
    "auto_data_size_px",
    "close_all",
    "configure_canvas",
    "create_axes_fixed",
    "design_dpi",
    "display_figure",
    "new_figure",
    "save_figure_data",
    "split_axes_horizontally",
]
