"""Figure lifecycle and fixed-pixel layout utilities for Jupyter plots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

from zlc_frontend.render_style import (
    DESIGN_DPI,
    STOCK_DATA_PX,
    STOCK_MARGINS_PX,
    apply_style,
)

try:
    from IPython import get_ipython
    from IPython.display import display
except Exception:  # pragma: no cover - only absent outside IPython.
    get_ipython = None
    display = None


@dataclass(frozen=True)
class FigureSpec:
    """Fixed logical-pixel layout for a notebook figure.

    The defaults ARE the stock confocal single-axes region; they read the geometry
    tokens in ``zlc_frontend.render_style`` so this and the default figure size
    can never drift apart."""

    data_px: tuple[int, int] = STOCK_DATA_PX
    margins_px: tuple[int, int, int, int] = STOCK_MARGINS_PX
    dpi: int = DESIGN_DPI


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
    def _destroy_bundle(bundle) -> None:
        for name in ("area", "cross", "zoom", "drag"):
            handler = getattr(bundle, name, None)
            if handler is not None and hasattr(handler, "destroy"):
                try:
                    handler.destroy()
                except Exception:
                    pass

    for attr in ("_zlc_tools", "_npt_tools"):
        tools = getattr(fig, attr, None)
        if tools is None:
            continue
        _destroy_bundle(tools)
        setattr(fig, attr, None)
    # Multi-axes plots (the site-histogram grid) keep one selector bundle PER
    # cell on the figure; destroy them all so a cell rerun does not leak them.
    grid_tools = getattr(fig, "_zlc_grid_tools", None)
    if grid_tools:
        for bundle in grid_tools:
            _destroy_bundle(bundle)
        fig._zlc_grid_tools = None


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
    # OFF the GUI/main thread (for example a worker rendering an export), build a
    # MANAGER-LESS Agg figure: ``plt.figure`` registers
    # with pyplot's GUI figure manager, which warns ("Starting a Matplotlib GUI outside of
    # the main thread...") and is unsafe off-main-thread.  Such figures are saved to disk,
    # never shown live, so they need no pyplot manager or cell tracking.
    import threading
    if threading.current_thread() is not threading.main_thread():
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(dpi=(spec.dpi if spec is not None else None))
        FigureCanvasAgg(fig)
        configure_canvas(fig)
        return fig
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
    data_px: tuple[int, int] = STOCK_DATA_PX,
    margins_px: tuple[int, int, int, int] = STOCK_MARGINS_PX,
) -> plt.Axes:
    """Create one axes with a fixed logical-pixel data box and margins (the stock confocal
    single-axes geometry -- the SAME render-style tokens FigureSpec reads, never a hand-typed copy)."""
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


def create_axes_grid(
    fig: plt.Figure,
    nrows: int,
    ncols: int,
    *,
    col_gap_px: int,
    row_gap_px: int,
    margins_px: tuple[int, int, int, int],
    data_px: tuple[int, int],
) -> list[plt.Axes]:
    """Create ``nrows*ncols`` fixed-pixel cells in row-major (top-left first) order that FILL a total
    data region of ``data_px`` -- the SAME data region every other panel kind uses
    (``panel_plot_spec(size).data_px``) -- by SUBDIVIDING it into ``ncols`` x ``nrows`` cells with the
    given inter-cell gaps (each cell's size is DERIVED to fill the region after the gaps).  Cells never
    overlap and the figure is sized to fit them plus ``margins_px`` (``(left, right, bottom, top)``, the
    top leaves room for a suptitle), so nothing is cut off.  The whole cell block is recorded as the
    fixed data box (``fig._zlc_fixed_box_in``) EXACTLY as :func:`create_axes_fixed` does, so a grid
    figure's data region equals a single-axes panel's of the same size and a size change truly rescales
    the cells (not just the padding)."""

    nrows, ncols = max(1, int(nrows)), max(1, int(ncols))
    dpi = design_dpi(fig)
    cgap, rgap = col_gap_px / dpi, row_gap_px / dpi
    L, R, B, T = [m / dpi for m in margins_px]
    # The cells + gaps exactly span ``data_px``, so a grid's data box matches a single-axes panel's
    # of the same size.  Each cell's size is DERIVED to fill it (remaining width/height after the
    # gaps, split evenly).
    dw_in, dh_in = data_px[0] / dpi, data_px[1] / dpi
    cw_in = max((dw_in - (ncols - 1) * cgap) / ncols, 1.0 / dpi)
    ch_in = max((dh_in - (nrows - 1) * rgap) / nrows, 1.0 / dpi)

    block_w = ncols * cw_in + (ncols - 1) * cgap
    block_h = nrows * ch_in + (nrows - 1) * rgap
    fig_w = L + block_w + R
    fig_h = B + block_h + T
    fig.set_size_inches(fig_w, fig_h, forward=True)

    horiz: list[Any] = [Size.Fixed(L)]
    for c in range(ncols):
        horiz.append(Size.Fixed(cw_in))
        if c < ncols - 1:
            horiz.append(Size.Fixed(cgap))
    horiz.append(Size.Fixed(R))
    vert: list[Any] = [Size.Fixed(B)]
    for r in range(nrows):
        vert.append(Size.Fixed(ch_in))
        if r < nrows - 1:
            vert.append(Size.Fixed(rgap))
    vert.append(Size.Fixed(T))

    divider = Divider(fig, (0, 0, 1, 1), horizontal=horiz, vertical=vert)
    axes: list[plt.Axes] = []
    for r in range(nrows):  # row 0 = TOP
        for c in range(ncols):
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axes_locator(divider.new_locator(nx=1 + 2 * c, ny=1 + 2 * (nrows - 1 - r)))
            axes.append(ax)
    fig._zlc_grid = (nrows, ncols)
    # Record the WHOLE cell block as the fixed data box (like create_axes_fixed) so the grid's
    # data region equals panel_plot_spec(size).data_px and a focused single cell / contract test
    # can read it back.
    fig._zlc_fixed_box_in = (block_w, block_h)
    fig._zlc_fixed_bounds_frac = (L / fig_w, B / fig_h, block_w / fig_w, block_h / fig_h)
    return axes


def grid_shape_for(n: int, *, max_cols: int = 8, prefer: tuple[int, int] | None = None) -> tuple[int, int]:
    """Choose ``(nrows, ncols)`` for ``n`` panels: an explicit ``prefer`` if it
    fits, else a near-square layout capped at ``max_cols`` columns."""

    n = max(1, int(n))
    if prefer is not None:
        pr, pc = int(prefer[0]), int(prefer[1])
        if pr > 0 and pc > 0 and pr * pc >= n:
            return pr, pc
    import math

    ncols = min(int(max_cols), int(math.ceil(math.sqrt(n))))
    ncols = max(1, ncols)
    nrows = int(math.ceil(n / ncols))
    return nrows, ncols


def fit_grid_shape_for_aspect(
    n: int, cell_aspect: float, region_px: tuple[int, int], *, max_cols: int = 8
) -> tuple[int, int]:
    """``(nrows, ncols)`` that MAXIMISES the displayed size of ``n`` FIXED-ASPECT cells tiled to fill a
    ``region_px`` ``(W, H)`` data region -- for a 2D image grid, whose cells keep their pixel ratio
    (``imshow(aspect='equal')``), so a near-square cell COUNT leaves the letterbox gaps the user sees
    (a square image dropped in a 4:3 cell wastes the sides).

    ``cell_aspect`` is the image WIDTH / HEIGHT.  For a candidate ``(rows, cols)`` the per-cell box is
    ``(W/cols, H/rows)`` and the image, keeping its ratio, renders at height ``min(H/rows, (W/cols)/a)``;
    since every cell is identical the utilised area is ``n * a * height**2``, so the shape with the
    largest per-cell image height fills the panel best.  Ties break toward fewer EMPTY cells, then a
    more-square layout -- so this collapses to the near-square shape when the image ratio already
    matches the region, and only departs from it to pack fixed-aspect images.  ``cols`` is capped at
    ``max_cols`` (the shared column cap); ``rows = ceil(n / cols)``."""

    import math

    n = max(1, int(n))
    a = float(cell_aspect) if cell_aspect and cell_aspect > 0 else 1.0
    W, H = float(region_px[0]), float(region_px[1])
    best_key: tuple | None = None
    best = (1, n)
    for ncols in range(1, min(int(max_cols), n) + 1):
        nrows = int(math.ceil(n / ncols))
        cw, ch = W / ncols, H / nrows
        img_h = min(ch, cw / a)                       # rendered image height, ratio preserved
        key = (img_h, -(nrows * ncols), -abs(nrows - ncols))
        if best_key is None or key > best_key:
            best_key, best = key, (nrows, ncols)
    return best


__all__ = [
    "FigureSpec",
    "close_all",
    "configure_canvas",
    "create_axes_fixed",
    "create_axes_grid",
    "design_dpi",
    "display_figure",
    "grid_shape_for",
    "new_figure",
    "split_axes_horizontally",
]
