"""Decoupled live/static plotting classes for Jupyter experiment front-ends."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import ceil
from typing import Any, Mapping, Sequence

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import Formatter, FuncFormatter, MaxNLocator, ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

from .canvas import FigureSpec, configure_canvas, create_axes_fixed, create_axes_grid, display_figure, grid_shape_for, new_figure, split_axes_horizontally
from .selectors import AreaSelector, CrossSelector, DragHLine, DragVLine, InteractionBundle, PlotState, ZoomPan, attach_interaction
from Zou_lab_control._readout_math import (
    bimodal_jacobian,
    bimodal_model,
    confidence_weighted_fidelity,
    gaussian,
    gaussian_jacobian,
)
from .style import (
    DESIGN_DPI,
    PALETTE,
    SITE_OCCUPANCY_STYLE,
    STOCK_MARGINS_PX,
    apply_style,
    apply_title,
    axis_label_fontsize,
    small_fontsize,
    smaller_fontsize,
    threshold_line_kwargs,
    tick_fontsize,
)
from .ticks import apply_smart_ticks


# Colours/sizes come from the one owned source in style.py (style.PALETTE / the
# fontsize accessors); these module names stay as convenience aliases.
DEFAULT_COLORS = list(PALETTE["series"])
PULSE_COLORS = list(PALETTE["pulse_cycle"])
#: A lone 1-D curve / rolling trace draws in this grey -- EXACTLY confocal's repeat=1 line (matplotlib
#: 'grey' #808080, alpha 1, the global lines.linewidth=1); the colour CYCLES for multi-line plots.
LINE_SINGLE = PALETTE["line_single"]
#: The per-line colour cycle, confocal's ['grey', 'skyblue', ...]: a lone line is the first (grey),
#: extra lines (data dimensions OR ``create`` repeats) take the next colours.
LINE_CYCLE = [LINE_SINGLE, PALETTE.get("bright", "skyblue")] + list(PALETTE["series"])


# --- Repeat-axis reduction (the PLOT's `repeat_mode`) ---------------------------------------------
# Every measurement publishes a RAW block whose LEADING axis is the repeat: ``(repeat, *points_shape,
# *data_shape)`` -- a 1-D scan's ``(repeat, points, dim)``, a 2-D scan's ``(repeat, n0*n1, dim)``, a
# camera's ``(repeat, 1, H, W)``.  It just FILLS each repeat (NaN = not-yet-measured), never combines.
# `repeat_mode` is a PLOT parameter that decides HOW to collapse the repeat axis O0 for display -- so
# the SAME raw data can be shown as a mean, a sum, the latest, a rolling newest, or (1-D) every repeat
# as its own line, without re-running.
#: The PLOT's repeat-combine modes (confocal "Update mode" naming: average / add / replace / roll /
#: create).  ``create`` is 1-D only -- it keeps every repeat as its OWN line (confocal's "create").
REPEAT_MODES = ("average", "add", "replace", "roll", "create")


def _has_repeat_axis(a, core_ndim) -> bool:
    """Whether array ``a``'s LEADING axis is the repeat to collapse.

    STRUCTURE-driven when the producing signal declares its ``core_ndim`` (= len(points_shape) +
    len(data_shape), #H3s-F3): a block carries the repeat axis exactly when ``a.ndim == 1 + core_ndim``
    (and an already-reduced ``a.ndim == core_ndim`` value passes through).  This is the clean fix for
    the muddle a bare ndim heuristic caused -- a clean ``(repeat, n_sites)`` (ndim 2, core_ndim 1) is a
    repeat block, while a static ``(n_sites,)`` value is not.  When ``core_ndim is None`` (an
    undeclared/legacy caller -- a camera/scan signal that does not pass it), fall back to the EXACT
    ndim>=3 rule so those paths behave byte-identically."""
    if core_ndim is not None:
        return a.ndim == 1 + int(core_ndim)
    return a.ndim >= 3


def reduce_repeat(raw, mode: str = "average", *, core_ndim=None):
    """Collapse a raw block over its LEADING (repeat) axis O0 for display.  Works for ANY trailing
    shape -- a 1-D scan's ``(repeat, points, dim)``, a 2-D scan's ``(repeat, n0*n1, dim)``, a camera's
    ``(repeat, 1, H, W)``, a clean occupancy ``(repeat, n_sites)`` -- because the repeat axis is always
    axis 0.  WHETHER axis 0 IS the repeat is decided by the producing signal's declared structure when
    given (``core_ndim``, #H3s-F3): collapse when ``raw.ndim == 1 + core_ndim``, pass through an
    already-reduced ``raw.ndim == core_ndim`` value.  When ``core_ndim is None`` the EXISTING ndim>=3
    fallback is kept EXACTLY (an already-reduced <3-D array passes through, so a plain image is never
    mistaken for a stack) -- so undeclared camera/scan callers are unaffected.

    * ``average`` -> ``nanmean`` over the repeats that HAVE data (the true running mean; magnitude-
      stable regardless of how many repeats completed = a long exposure for a camera) -> drops O0.
    * ``add``     -> ``nansum``  over repeats (accumulating exposure).
    * ``replace`` / ``roll`` -> the LATEST repeat slice that holds data.
    * ``pool``    -> a DISTRIBUTION mode: do NOT reduce the repeat axis -- flatten EVERY repeat-with-data's
      samples into one 1-D set so a histogram bins all shots together.  This is the ONLY repeat mode a
      histogram offers, so the panel needs no per-kind special case: it just calls ``reduce_repeat(mode)``.
    * ``create``  -> 1-D blocks only: keep EVERY repeat-with-data as its own column block
      ``(points, n*dim)`` so the curve draws one line per repeat (confocal's "create"); for a 3-D+
      data block (an image) ``create`` has no meaning and falls back to the mean."""
    a = np.asarray(raw, dtype=float)
    if not _has_repeat_axis(a, core_ndim):              # not a repeat block -> leave as-is
        return a
    has = np.isfinite(a).any(axis=tuple(range(1, a.ndim)))   # which repeat slices hold any data
    idx = np.flatnonzero(has)
    if mode == "add":
        return np.nansum(a, axis=0)
    if mode in ("replace", "roll"):
        return a[idx[-1]] if idx.size else a[0]
    if mode == "pool":                                    # histogram: flatten ALL repeats' samples (no reduce)
        return (a[idx] if idx.size else a[:1]).reshape(-1)
    if mode == "create":
        cols = idx if idx.size else np.array([0])
        if a.ndim == 2:                                       # (R, points) reduced scan -> (points, R) lines
            return a[cols].T
        if a.ndim == 3:                                       # (R, points, dim) scan -> (points, R*dim)
            return np.concatenate([a[r] for r in cols], axis=1)   #   confocal: repeat-major dim-minor columns
        # ndim >= 4: an image block (a camera's (R, 1, H, W)) -- create is ORTHOGONAL to the data axes:
        # flatten each repeat's core to a column so there is ONE trace per repeat (NOT per image row).
        return a[cols].reshape(len(cols), -1).T               # (prod(core), R) = x=pixel index, R lines
    # An all-NaN cell (a not-yet-measured scan point during a LIVE sweep) averages to NaN -- a gap in
    # the curve, BY INTENT.  numpy reports that via two channels: errstate covers the 0/0 it computes,
    # and a separate ``warnings`` message ("Mean of empty slice") that errstate does NOT catch -- so
    # silence exactly that benign message here (the result is already the intended NaN).
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(a, axis=0)


def repeats_with_data(raw, *, core_ndim=None) -> int:
    """How many repeat slices of a raw block currently hold data (drives the plot's ``xN`` label).
    A non-repeat-block (structure-driven when ``core_ndim`` is given, else the ndim>=3 fallback) -> 1."""
    a = np.asarray(raw, dtype=float)
    if not _has_repeat_axis(a, core_ndim):
        return 1
    return int(np.count_nonzero(np.isfinite(a).any(axis=tuple(range(1, a.ndim)))))


# --- Pulse-plot display margins -------------------------------------------
# The pulse timeline draws a viewport deliberately a little WIDER than the data
# extent, so pulses/edges sitting on the boundary (e.g. a first edge at t=0) are
# clearly visible instead of flush against a spine -- the same idea the other
# plot types use when they pad their display limits beyond the raw data range
# (Live1D/Histogram pad y by ~10-20%; Live2DDis pads both axes).  off_lines and
# analog baselines are still drawn out to these limits, so the full-bleed
# baseline look is preserved on purpose.  X uses a fraction of the time span (so
# the margin scales with the data/zoom); Y uses a fixed gap in channel-row units
# (rows are unit-spaced, so a fixed gap reads the same regardless of how many
# channels are shown).
# PULSE PLOT GEOMETRY -- OWNED by frontend, NOT a public knob (private `_`-prefixed
# so the package surface can never set them; see frontend/__init__.py contract).
_PULSE_X_MARGIN_FRAC = 0.04          # per-side x headroom, as a fraction of the time span
_PULSE_X_BRACKET_LABEL_FRAC = 0.05   # extra RIGHT headroom when a repeat x N bracket label is shown
# X AUTO-EXTEND (mirrors the y row-block behaviour): the pulse plot grows one base-width
# block per _PULSE_X_PERIODS_PER_BLOCK periods, capped at _PULSE_X_MAX_WIDTH_FACTOR x the base width.
_PULSE_X_PERIODS_PER_BLOCK = 5       # periods per horizontal width block
_PULSE_X_MAX_WIDTH_FACTOR = 3        # max width as a multiple of the base data width
_PULSE_Y_PAD_BOTTOM = 0.62           # gap below the bottom channel row (row units)
_PULSE_Y_PAD_TOP = 0.38              # gap above the top channel row (row units)
_PULSE_MARGINS_PX = (110, 90, 100, 50)   # owned pulse-plot margins (L, R, B, T)
_PULSE_DPI = DESIGN_DPI                    # the ONE design dpi (never per-call; one source)

# SHARED PLOT GEOMETRY -- OWNED by the frontend, never per-call.  These are the
# few raw numbers the plot layouts used to spell inline; naming them ONCE here
# means a value is written in a single place (and the contract tests DERIVE from
# these rather than re-typing the literal, so they cannot silently go stale).
TITLE_SLOT_PX = 70                         # vertical px a centred plot title needs.
                                           # A titled figure floors its TOP margin
                                           # here (_with_title_margin), and a panel
                                           # ALWAYS reserves it so a panel is ONE size
                                           # whether or not it carries a title -- ONE
                                           # source so PANEL_MARGINS_PX[3] and the
                                           # title-margin floor can never disagree (the
                                           # desync that made panel_display_size
                                           # under-report the card height).
# Horizontal axes splits for the composite plots (fractions of the data width that
# go to image | side-distribution | colorbar, with the inter-band gaps).  0.75 is
# what makes the "2x2" 2D image square (480 * 0.75 = 360).
_DIST_SPLIT = ([0.825, 0.15], [0.025])             # 1D + side distribution (plot | dist)
_IMAGE_SPLIT = ([0.75, 0.1, 0.1], [0.025, 0.025])  # 2D image | side dist | colorbar

# Owned geometry for the per-site histogram grid (site_histogram_grid).  Mirrors
# the pulse-plot many-data policy: a FIXED small per-cell box, COLUMNS capped so
# the figure never runs off-screen sideways, and ROWS that grow (wrap) for large
# N.  Sized so a typical trap grid (e.g. 5x7) lands near one standard plot's
# 700x500 px footprint instead of ~2x it.
_SITE_CELL_PX = (104, 84)                 # (width, height) of one site cell box
_SITE_COL_GAP_PX = 12
_SITE_ROW_GAP_PX = 16
# (L, R, B, T) around the whole grid: B holds the bottom-row tick labels AND the
# outer x-axis label (two text rows, so it is generous); T holds the suptitle.
_SITE_GRID_MARGINS_PX = (54, 20, 62, 42)
_SITE_MAX_COLS = 7                        # column cap (pulse-style width cap); extra sites wrap to rows


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


def site_ring_radius(centers) -> float:
    """The occupancy-ring radius (camera px) for a site map: 30 % of the nearest
    inter-site spacing, with a 1.5 px floor, so the rings scale with the lattice and
    never overlap.  ONE rule shared by the live console site map and the calibration
    report's site map, so the two never draw different-sized rings."""
    c = np.asarray(centers, dtype=float)
    spacing = 6.0
    if c.ndim == 2 and len(c) > 1:
        deltas = np.linalg.norm(np.diff(c[:, :2], axis=0), axis=1)
        deltas = deltas[deltas > 0]
        if deltas.size:
            spacing = float(np.min(deltas))
    return max(1.5, 0.3 * spacing)


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
    s = f"{float(x):.{max(0, length - 1)}g}"
    if "e" not in s and "E" not in s:
        return s                              # small values (865, 0.5, 1500) unchanged
    # Compact the scientific form so big values fit a tight colorbar margin:
    # 1-decimal mantissa, drop trailing zeros, strip the exponent's '+' and
    # leading zeros (e.g. "1.234e+04" 9ch -> "1.2e4" 5ch, "5e+05" -> "5e5").
    mant, exp = f"{float(x):.1e}".split("e")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


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
    return FigureSpec(data_px=spec.data_px, margins_px=(left, right, bottom, max(top, TITLE_SLOT_PX)), dpi=spec.dpi)


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
    #: The DataFigure fitting FAMILY -- "1D" (line/hist fits) or "2D" (image
    #: clim/centroid).  Declared ONCE per plot class (the single-source plot-kind
    #: table ``PLOT_KINDS`` mirrors it) so ``DataFigure`` reads it instead of
    #: re-deriving the family from matplotlib artists.  Default "1D".
    render_family = "1D"

    def __init__(
        self,
        data_x=np.arange(100),
        data_y=None,
        *,
        labels: Sequence[str] = ("X", "Y", "Z"),
        update_time: float = 0.1,
        fig: plt.Figure | None = None,
        relim_mode: str = "normal",
        fixed_lo: float | None = None,
        fixed_hi: float | None = None,
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
        # "fixed" lim mode: the y-axis (1D/monitor) / colour-limit (2D, sites) is pinned to these
        # operator-set bounds and never autoscales (see _mode_target / relim).  Default to a unit
        # span so a fixed panel is usable before the operator types real bounds.
        self.fixed_lo = 0.0 if fixed_lo is None else float(fixed_lo)
        self.fixed_hi = 1.0 if fixed_hi is None else float(fixed_hi)
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
        self.ax = self._create_axes()
        if self.axes is None:
            self.axes = self.ax
        if self.smart_ticks:
            # cap the tick count by the DATA-AREA size so small (dashboard
            # panel) axes never crowd their labels; the caps saturate at the
            # stock 8 for full-size notebook figures.
            data_w, data_h = self.spec.data_px
            # scale the tick caps with the data area RELATIVE TO the stock
            # 480x360 region, so full-size figures keep the stock 8 and a
            # half-height panel gets proportionally fewer, never-crowded ticks
            apply_smart_ticks(
                self.ax,
                max_ticks_x=max(3, min(8, round(8 * int(data_w) / 480))),
                max_ticks_y=max(3, min(8, round(8 * int(data_h) / 360))),
            )
        self.init_core()
        self._apply_title()
        self._install_state()
        if self.interactions:
            self._attach_interactions()
        # Strong self-ref on the figure (confocal's `fig._live_plotter` pattern):
        # the nb live loop / QTimer hold the plotter, but an explicit anchor keeps
        # it alive for the figure's lifetime even if a caller drops its reference,
        # rather than relying only on the transitive `_zlc_tools` -> bound-method ref.
        self.fig._zlc_plotter = self
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

    def _create_axes(self):
        """Create this plot's axes and return the PRIMARY axes.

        Single-axes plots get one fixed-pixel data box.  Multi-axes plots (e.g.
        the site-histogram grid) override this to build their full layout and set
        ``self.axes`` to the list of all axes; ``show()`` then keeps that list."""
        return create_axes_fixed(self.fig, self.spec.data_px, self.spec.margins_px)

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type=self.plot_type)

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def interaction_handles(self) -> list:
        """Every active selector/interaction object on this plot (zoom/pan, area,
        cross, drag).  The reusable-interaction contract: a shown plot returns a
        NON-EMPTY list -- enforced for every plot type by the plot-contract test,
        so a hand-rolled figure that skips selectors cannot ship."""
        return [h for h in (self.zoom, self.area, self.cross, self.drag) if h is not None]

    def _apply_title(self) -> None:
        if self.ax is not None:
            apply_title(self.ax, self.title)

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

    def _mode_target(self, min_y: float, max_y: float) -> tuple[float, float]:
        """The (lo, hi) limits for the current ``relim_mode`` -- the ONE place the
        normal-vs-tight rule lives, shared by the 1D/monitor y-axis (``relim``) and
        the 2D colour-limit (``Live2DDis``).  ``normal`` anchors at 0 (a counts-like
        quantity reads against zero); ``tight`` brackets the data with 10% padding.
        Negative data forces tight (anchoring at 0 would hide it)."""
        if self.relim_mode == "fixed":
            return self.fixed_lo, self.fixed_hi   # user-pinned bounds; ignore the data range
        rng = (max_y - min_y) or (abs(max_y) or 1.0)
        if self.relim_mode == "normal" and min_y >= 0:
            return 0.0, (max_y * 1.2 if max_y else 1.0)
        return min_y - 0.1 * rng, max_y + 0.1 * rng

    def relim(self, values=None, *, force: bool = False) -> bool:
        # "off": the y-limits are pinned by the caller (a dashboard panel with a
        # manual ylim); keep ylim_min/ylim_max frozen so update_core's set_ylim
        # re-applies the same fixed range instead of re-autoscaling each tick.
        # ``force`` bypasses the dead-band: a relim_mode SWITCH must rescale NOW
        # (the band assumes the mode is unchanged, so it would otherwise refuse).
        if self.relim_mode == "off":
            return False
        if self.relim_mode == "fixed":
            # User-pinned (lo, hi): re-apply them and never autoscale (must short-circuit BEFORE
            # the negative-data auto-switch below, which would otherwise clobber "fixed" to tight).
            old = (self.ylim_min, self.ylim_max)
            self.ylim_min, self.ylim_max = self.fixed_lo, self.fixed_hi
            return old != (self.ylim_min, self.ylim_max)
        vals = np.asarray(self.data_y[:, 0] if values is None else values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return False
        max_y = float(np.nanmax(vals))
        min_y = float(np.nanmin(vals))
        if min_y < 0:
            self.relim_mode = "tight"
        old = (self.ylim_min, self.ylim_max)
        lo, hi = self.ylim_min, self.ylim_max
        # Dead-band hysteresis: only rescale when the data leaves a band inside the
        # current view, so a noisy rolling trace does not jitter its y-limits (and
        # re-blit) every single frame.  When the data still sits comfortably in the
        # view, freeze the limits and return False -- the unconditional
        # set_ylim(ylim_min, ylim_max) in update_core is then a no-op.  Rescaling
        # always happens when data would clip, so a point is never hidden.
        if self.relim_mode == "normal":
            if not force and hi > 0 and (0.7 * hi) <= max_y <= hi:
                return False                          # within band -> no rescale
            self.ylim_min, self.ylim_max = self._mode_target(min_y, max_y)
        else:
            span = hi - lo
            clips = (min_y < lo) or (max_y > hi)
            too_empty = span > 0 and (max_y < hi - 0.35 * span or min_y > lo + 0.35 * span)
            if not force and span > 0 and not clips and not too_empty:
                return False                          # within band -> no rescale
            self.ylim_min, self.ylim_max = self._mode_target(min_y, max_y)
        return old != (self.ylim_min, self.ylim_max)

    def apply_relim_now(self) -> None:
        """Force the axis limits to the current relim_mode and redraw immediately.
        Called when the mode toggles (Setting combo) -- a switch must take effect
        now, not wait for the next data frame, and must bypass relim's dead-band.
        1D/monitor: rescale the y-axis.  Live2DDis overrides for the colour limit."""
        self.relim(force=True)
        if self.ax is not None:
            self.ax.set_ylim(self.ylim_min, self.ylim_max)

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
    render_family = "1D"

    def _color_lines(self) -> None:
        """Colour the curve(s) EXACTLY like Confocal-GUIv2: every line solid, ``alpha=1``,
        ``linewidth=1``; the COLOUR cycles (grey, skyblue, ...) by column index.  A lone line is
        grey.  There is NO per-repeat fade and NO dim-vs-create distinction -- the data-dimension
        lines and the ``create`` repeat-lines are styled identically (one line per column), so this
        matches the reference and the single-line look is unchanged."""
        for i, line in enumerate(self.lines):
            line.set_color(LINE_CYCLE[i % len(LINE_CYCLE)])
            line.set_alpha(1.0)
            line.set_linewidth(1.0)

    def init_core(self) -> None:
        self.lines = self.ax.plot(self.data_x[:, 0], self.data_y, alpha=1)
        self._color_lines()
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        # Only pin xlim when there are >=2 distinct x values; a single unique x
        # (e.g. a 1-point first frame) would make low==high and trip a
        # matplotlib "Attempting to set identical low and high xlims" warning --
        # skip it and let matplotlib pick a default until more points arrive.
        if np.unique(self.data_x[:, 0]).size >= 2:
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


class LiveLive(Live1D):
    """Live rolling trace: a 1-D curve that rolls newest-first, with a
    latest-value readout in the top-right corner and NO side distribution.

    This is the BARE rolling-trace plot type.  The distribution variant is a
    SEPARATE plot type, :class:`LiveLiveDis`, which EXTENDS this one by adding a
    right-side histogram + Gaussian-sigma band -- it is not a boolean toggle on
    one class.  Loading-rate-like quantities use this bare type, because a
    right-side histogram + Gaussian-sigma fit carries no physical meaning for a
    running rate.  Geometry stays inside the plot class, so no ``split`` /
    ``distribution`` art kwarg ever crosses the sealed ``plot()`` /
    ``panel_plot()`` surface."""

    plot_type = "live"

    def init_core(self) -> None:
        self.axes = self.ax
        self.lines = self.ax.plot(self.data_x[:, 0], self.data_y, alpha=1)
        self._color_lines()                       # a lone rolling trace -> grey (confocal style)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_xlim(np.nanmin(self.data_x[:, 0]), np.nanmax(self.data_x[:, 0]))
        self.relim()
        self.ax.set_ylim(self.ylim_min, self.ylim_max)
        self.text = None

    def update_core(self) -> None:
        super().update_core()
        # latest-value readout in the top-right corner (inside the axes; the
        # band above the axes belongs to the centred title, which may be long)
        newest = self.data_y[0, 0]
        if np.isfinite(newest):
            label = f"{newest:.6g}"
            if self.text is None:
                self.text = self.ax.text(
                    0.97,
                    0.95,
                    label,
                    transform=self.ax.transAxes,
                    color=PALETTE["readout"],
                    ha="right",
                    va="top",
                    fontsize=small_fontsize(),
                )
            else:
                self.text.set_text(label)

    # _install_state is byte-identical to the inherited Live1D._install_state
    # (same "1D" PlotState with the same x/y arrays), so it is NOT overridden
    # here -- the inherited one is reused (one fewer copy of the same code).


class LiveLiveDis(LiveLive):
    """Live rolling trace PLUS a right-side distribution band (histogram +
    Gaussian-sigma fit).  A SEPARATE plot type that EXTENDS :class:`LiveLive`
    with the side-distribution axes -- the bare rolling trace is its own type,
    this ADDS to it rather than flipping a flag.  Geometry stays inside the plot
    class, so no ``split`` / ``distribution`` art kwarg ever crosses the sealed
    ``plot()`` / ``panel_plot()`` surface."""

    plot_type = "live-distribution"

    def init_core(self) -> None:
        # carve the side band off FIRST so the base draws the trace into the
        # narrowed main axes
        self.ax, self.axdis = split_axes_horizontally(self.fig, self.ax, *_DIST_SPLIT)
        super().init_core()
        self.fit_text = None
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.axdis.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
        self.axdis.tick_params(axis="both", which="both", bottom=False, top=False)
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.n_bins = int(max(3, min(self.points_total // 4, 50)))
        self.n, self.bins = self._hist()
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"])
        self.axdis.add_collection(self.poly)
        self.counts_max = max(10, int(np.nanmax(self.n) + 5 if self.n.size else 10))
        self.axdis.set_xlim(0, self.counts_max)
        (self.gauss_line,) = self.axdis.plot([], [], color=PALETTE["fit_right"], alpha=1)

    def _hist(self):
        vals = self.data_y[: max(self.points_done, 1), 0]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.array([0.0])
        return np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))

    def _update_gauss_fit(self):
        mask = self.n > 0
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        x = centers[mask]
        y = self.n[mask]
        if len(x) < 3 or np.ptp(x) == 0:
            return
        try:
            popt, _ = curve_fit(
                gaussian,
                x,
                y,
                p0=[np.max(y), np.mean(x), max(np.ptp(x) / 4, 1e-12)],
                bounds=([0, np.min(x), max(np.ptp(x) / 100, 1e-12)], [max(np.max(y) * 4, 1), np.max(x), max(np.ptp(x) * 10, 1e-12)]),
                jac=gaussian_jacobian,
            )
        except Exception:
            return
        x_fit = np.linspace(self.ylim_min, self.ylim_max, 100)
        self.gauss_line.set_data(gaussian(x_fit, *popt), x_fit)
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
                color=PALETTE["fit_right"],
                ha="center",
                va="bottom",
                fontsize=small_fontsize(),
            )
        else:
            self.fit_text.set_text(label)

    def update_core(self) -> None:
        super().update_core()           # rolling trace + latest-value readout
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n, self.bins = self._hist()
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly.set_verts(self.verts)
        peak = np.nanmax(self.n)
        counts_max = max(10, int(max(peak + 5, peak * 1.5)))
        self.axdis.set_xlim(0, counts_max)
        self._update_gauss_fit()

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="1D", x_array=self.data_x[:, 0], y_array=self.data_y, axdis=self.axdis)


class Live2DDis(BaseLivePlot):
    """Live 2D image with side distribution, colorbar, and draggable clim."""

    plot_type = "2D"
    render_family = "2D"

    def __init__(self, *args, cmap: str = PALETTE["cmap_scan"], bad_color: str = PALETTE["bad"], square: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_x.shape[1] != 2:
            raise ValueError("Live2DDis requires data_x with shape (N, 2).")
        self.cmap = cmap
        self.bad_color = bad_color
        self.square = square

    def fill_grid(self) -> np.ndarray:
        # vectorised point->grid scatter (the per-point Python loop took ~50 ms
        # per refresh on a camera-frame panel); same semantics as the old loop:
        # searchsorted indices, out-of-range points dropped, and on duplicate
        # indices the LAST point wins (C-order fancy assignment).
        grid = np.full(self.data_shape, np.nan)
        ix = np.searchsorted(self.x_array, self.data_x[:, 0])
        iy = np.searchsorted(self.y_array, self.data_x[:, 1])
        ok = (ix < grid.shape[1]) & (iy < grid.shape[0])
        grid[iy[ok], ix[ok]] = self.data_y[ok, 0]
        return grid

    def init_core(self) -> None:
        self.ax, self.axdis, self.cax = split_axes_horizontally(self.fig, self.ax, *_IMAGE_SPLIT)
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
        # Image coordinates are the source's real space (a camera ROI can put the
        # origin at a 4-5 digit pixel like 1648); at panel size the standard ~7-8
        # ticks of such wide labels crowd into an unreadable blur.  Re-apply the
        # frontend's smart locator/formatter PAIR with a tighter cap (replacing
        # both as a unit -- swapping only the locator would orphan the paired
        # formatter and blank every label) so the square image keeps a few legible
        # ticks on each axis whatever the ROI origin.
        if self.smart_ticks:
            apply_smart_ticks(self.ax, max_ticks_x=4, max_ticks_y=4)
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
        # The colour limit is the 2D analogue of the 1D y-axis, so it obeys the
        # SAME relim_mode via the shared _mode_target: "normal" anchors the colorbar
        # at 0 (counts read against zero), "tight" brackets the data.  (Previously
        # this was hard-coded tight, so the Setting's normal/tight did nothing.)
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        self.image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n_bins = int(max(8, min(max(self.points_total, 1) // 4, 50)))
        self.n, self.bins = np.histogram(vals if vals.size else [0], bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"])
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, max(10, int(np.max(self.n) + 5)))
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
        self.axdis.tick_params(axis="y", which="both", left=True, right=False, labelleft=False, labelright=False)
        self.line_min = self.axdis.axhline(y_min, color=PALETTE["guide"], linewidth=small_fontsize() / 2, alpha=0.3)
        self.line_max = self.axdis.axhline(y_max, color=PALETTE["guide"], linewidth=small_fontsize() / 2, alpha=0.3)
        cmap = self.image.get_cmap()
        self.line_l = self.axdis.axhline(self.ylim_min, color=cmap(0.0), linewidth=small_fontsize() / 2)
        self.line_h = self.axdis.axhline(self.ylim_max, color=cmap(0.95), linewidth=small_fontsize() / 2)
        # colorbar end ticks are set by update_core (cax.set_yticks/labels)
        self.drag = DragHLine(self.line_l, self.line_h, self.update_clim, self.axdis)

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax, drag=self.drag, axdis=self.axdis, cax=self.cax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def update_clim(self) -> None:
        self.image.set_clim(float(self.line_l.get_ydata()[0]), float(self.line_h.get_ydata()[0]))

    def apply_relim_now(self) -> None:
        # 2D analogue of the y-axis rescale: recompute the colour limit for the
        # current relim_mode and re-apply it + the draggable clim lines now, so a
        # normal<->tight switch in Setting visibly re-maps the colorbar.
        vals = self._finite_values()
        if not vals.size:
            return
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        self.image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.line_l.set_ydata([self.ylim_min, self.ylim_min])
        self.line_h.set_ydata([self.ylim_max, self.ylim_max])

    def update_core(self) -> None:
        self.grid = self.fill_grid()
        self.image.set_array(self.grid)
        vals = self._finite_values()
        if vals.size:
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)   # normal=from 0, tight=bracket
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


class LiveSiteMap(BaseLivePlot):
    """Live atom-array site map: a 2D camera frame with one hollow ring per tweezer,
    FAINT for an empty site and BOLD for an occupied one (the Rb87 readout look).

    A site map is a plain 2D image (the camera frame, with its own counts colorbar)
    plus an occupancy OVERLAY -- not a heat-map of per-site scalars.  ``data_y[:, 0]``
    is read as occupancy: a site is "occupied" (bold ``SITE_OCCUPANCY_STYLE`` ring)
    when its value is >= 0.5, else "empty" (faint ring); the rings are always unfilled
    so the underlying frame shows through.  Same array contract as every live plot:
    ``data_x`` is the ``(N, 2)`` site centers in camera pixels (x, y).  The internal
    split matches :class:`Live2DDis` ([0.75, 0.1, 0.1]) so the square image + colorbar
    line up with the 2D panels (the side-distribution band stays empty)."""

    plot_type = "SITES"
    # "auto" = let DataFigure resolve the fitting family from the artists PER FIGURE, not a
    # fixed "1D"/"2D".  A site map's primary axes only carries an image when a background frame
    # is supplied (``image=``); with only occupancy rings + no frame there is no imshow, so the
    # legacy fitting family was "1D".  A static declared family would change that conditional
    # behaviour, so the site map opts OUT of the declared override (behaviour-preserving).
    render_family = "auto"

    def __init__(self, *args, image=None, roi_radius: float = 3.0, cmap: str = PALETTE["cmap_camera"], **kwargs):
        super().__init__(*args, **kwargs)
        if self.data_x.shape[1] != 2:
            raise ValueError("LiveSiteMap requires data_x with shape (N, 2) site centers.")
        self.background = None if image is None else np.asarray(image, dtype=float)
        self.roi_radius = max(0.5, float(roi_radius))
        self.cmap = str(cmap)   # colormap of the camera-FRAME underlay (the rings are fixed-colour)

    # the dashboard refreshes the underlay every shot (set, don't rebuild)
    def set_background(self, image, *, draw: bool = False) -> None:
        if image is None:
            return
        arr = np.asarray(image, dtype=float)
        if self.background is not None and getattr(self, "_bg_image", None) is not None \
                and arr.shape == self.background.shape:
            self.background = arr
            self._bg_image.set_data(arr)
            # The colour limit + side histogram are owned by the distribution band (per the
            # lim-mode), not pinned to this frame's min/max here -- update_core re-clims + refreshes.
            if draw:
                self.draw()
        else:
            self.background = arr               # shape changed: rebuilt by the host

    def _ring_styles(self, values: np.ndarray):
        """Per-site (edge RGBA, linewidth) from occupancy: value >= 0.5 -> occupied
        (bold ring), else empty (faint ring).  Single source = SITE_OCCUPANCY_STYLE."""
        from matplotlib.colors import to_rgba

        occupied = np.asarray(values, dtype=float).reshape(-1) >= 0.5
        empty, occ = SITE_OCCUPANCY_STYLE["empty"], SITE_OCCUPANCY_STYLE["occupied"]
        edge = [to_rgba(occ["color"], occ["alpha"]) if flag else to_rgba(empty["color"], empty["alpha"])
                for flag in occupied]
        widths = [occ["linewidth"] if flag else empty["linewidth"] for flag in occupied]
        return edge, widths

    def init_core(self) -> None:
        from matplotlib.collections import EllipseCollection

        self.ax, self.axdis, self.cax = split_axes_horizontally(self.fig, self.ax, *_IMAGE_SPLIT)
        self.axes = self.ax
        centers = self.data_x[:, :2]
        if self.background is not None:
            h, w = self.background.shape
            extent = [-0.5, w - 0.5, h - 0.5, -0.5]
        else:
            pad = 2.5 * self.roi_radius
            extent = [float(centers[:, 0].min()) - pad, float(centers[:, 0].max()) + pad,
                      float(centers[:, 1].max()) + pad, float(centers[:, 1].min()) - pad]
        self.extent = extent
        if self.background is not None:
            self._bg_image = self.ax.imshow(self.background, cmap=self.cmap, extent=extent, interpolation="none")
        else:
            self._bg_image = None
        diameter = 2.0 * self.roi_radius
        edge, widths = self._ring_styles(self.data_y[:, 0])
        self.sites = EllipseCollection(
            widths=diameter, heights=diameter, angles=0.0, units="xy",
            offsets=centers, transOffset=self.ax.transData,
            facecolors="none", edgecolors=edge, linewidths=widths, zorder=5)
        self.ax.add_collection(self.sites)
        self.lines = [self.sites]
        self.ax.set_anchor("W")
        self.ax.set_aspect("equal", adjustable="box")
        self.extents_square = _square_extent(list(extent))
        self.ax.set_xlim(self.extents_square[0], self.extents_square[1])
        self.ax.set_ylim(self.extents_square[2], self.extents_square[3])
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        # The colorbar reflects the CAMERA FRAME counts (the 2D image); occupancy is a
        # binary ring overlay and carries no scale.  No frame -> no colorbar band.
        if self._bg_image is not None:
            self.cbar = self.fig.colorbar(self._bg_image, cax=self.cax)
            self.cbar.set_label(self.zlabel)
        else:
            self.cbar = None
            self.cax.set_visible(False)
        # Side distribution of the CAMERA-FRAME intensities + draggable colour limit -- the SAME
        # band a 2D image has (#10).  Occupancy stays a binary ring overlay; the histogram is of
        # the underlay frame, and the drag/relim controls the frame clim (so the fixed/normal/tight
        # lim mode applies to the site-map frame too).
        self._init_distribution()

    def _frame_values(self) -> np.ndarray:
        if self.background is None:
            return np.asarray([], dtype=float)
        vals = np.asarray(self.background, dtype=float).ravel()
        return vals[np.isfinite(vals)]

    def _init_distribution(self) -> None:
        vals = self._frame_values()
        if self._bg_image is None or vals.size == 0:
            self.axdis.set_visible(False)        # no frame yet -> no distribution band
            self.poly = None
            return
        y_min, y_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        self._bg_image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n_bins = 40
        self.n, self.bins = np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"])
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, max(10, int(np.max(self.n) + 5)))
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
        self.axdis.tick_params(axis="y", which="both", left=True, right=False, labelleft=False, labelright=False)
        cmap = self._bg_image.get_cmap()
        self.line_l = self.axdis.axhline(self.ylim_min, color=cmap(0.0), linewidth=small_fontsize() / 2)
        self.line_h = self.axdis.axhline(self.ylim_max, color=cmap(0.95), linewidth=small_fontsize() / 2)
        self.drag = DragHLine(self.line_l, self.line_h, self.update_clim, self.axdis)

    def _attach_interactions(self) -> None:
        drag = getattr(self, "drag", None)
        self.tools = attach_interaction(self.ax, drag=drag, axdis=self.axdis, cax=self.cax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def update_clim(self) -> None:
        if self._bg_image is not None and getattr(self, "line_l", None) is not None:
            self._bg_image.set_clim(float(self.line_l.get_ydata()[0]), float(self.line_h.get_ydata()[0]))

    def apply_relim_now(self) -> None:
        vals = self._frame_values()
        if self._bg_image is None or getattr(self, "poly", None) is None or not vals.size:
            return
        y_min, y_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        self._bg_image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.line_l.set_ydata([self.ylim_min, self.ylim_min])
        self.line_h.set_ydata([self.ylim_max, self.ylim_max])

    def _update_distribution(self) -> None:
        # Recompute the frame colour-limit for the current lim-mode (normal=from 0, tight=bracket,
        # fixed=pinned via _mode_target) + refresh the side histogram, like Live2DDis.update_core.
        vals = self._frame_values()
        if getattr(self, "poly", None) is None or not vals.size:
            return
        y_min, y_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        if self._bg_image is not None:
            self._bg_image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.n, self.bins = np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly.set_verts(self.verts)
        self.axdis.set_xlim(0, max(10, int(np.max(self.n) + 5)))
        # keep the draggable clim lines at the (re)computed limits unless the user dragged inside
        if getattr(self, "line_l", None) is not None:
            self.line_l.set_ydata([self.ylim_min, self.ylim_min])
            self.line_h.set_ydata([self.ylim_max, self.ylim_max])

    def update_core(self) -> None:
        edge, widths = self._ring_styles(self.data_y[:, 0])
        self.sites.set_edgecolors(edge)
        self.sites.set_linewidths(widths)
        self._update_distribution()              # refresh the frame-intensity histogram + clim

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="SITES", x_array=self.data_x[:, 0], y_array=self.data_y,
                                        axdis=self.axdis, cax=self.cax)


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


class _PulseTimeFormatter(Formatter):
    """X-tick formatter whose precision FOLLOWS the tick spacing.

    A fixed significant-figure format degenerates when the view is zoomed deep
    into a long timeline: ticks at 8.0001/8.0002/8.0003 ms all rendered as
    "8.000" -- every label identical, the axis unreadable (the reported "all
    ticks become 8ms" failure).  The tick machinery stores the current tick
    locations in ``self.locs`` before formatting, so the needed decimal count
    is derived from the SMALLEST adjacent tick spacing -- adjacent labels are
    then always distinct at any zoom.  Negative (cosmetic headroom) ticks stay
    blanked.
    """

    def __init__(self, time_scale: float):
        super().__init__()
        self.time_scale = float(time_scale)

    def _decimals(self) -> int:
        locs_src = self.locs if self.locs is not None else []   # may be an ndarray: no `or`
        locs = [float(l) / self.time_scale for l in locs_src if float(l) >= 0.0]
        if len(locs) < 2:
            return 0
        spacing = min(abs(b - a) for a, b in zip(locs, locs[1:]) if b != a) if any(
            b != a for a, b in zip(locs, locs[1:])) else 0.0
        if spacing <= 0.0:
            return 0
        if spacing >= 1.0:
            return 0
        return min(9, int(np.ceil(-np.log10(spacing) - 1e-9)))

    def __call__(self, value, pos=None) -> str:
        if value < 0:
            return ""
        v = float(value) / self.time_scale
        decimals = self._decimals()
        if decimals == 0 and v != 0 and abs(v) < 1.0:
            # single-tick / no-spacing fallback: keep the old 4-sig-fig look
            return _float2str_eng(v, length=4)
        return f"{v:.{decimals}f}"


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
    explicit_channels = channels is not None
    if channels is None:
        if hasattr(sequence, "channels"):
            channels = list(getattr(sequence, "channels"))
        else:
            channels = sorted({row["channel"] for row in rows})
    ordered = [str(channel) for channel in (channels or [])]
    if include_always_off:
        return ordered if explicit_channels else (ordered or ["pulse"])
    active = {row["channel"] for row in rows if row["value"] and row["duration"] > 0}
    visible = [channel for channel in ordered if channel in active]
    if not visible and ordered:
        visible = ordered[: max(0, min(int(minimum), len(ordered)))]
    return visible if explicit_channels else (visible or ["pulse"])


# --------------------------------------------------------------------- panels
# Dashboard panel presets.  THE design rule (the confocal rule): every panel
# kind shares ONE plot region and ONE margin set, so panels of any kind line
# up exactly.  "2x2" is the stock frontend plot region (480x360 data px); a
# size "RxC" spans R height-halves x C width-halves of it ("1x2" = the stock
# region at half height).  The 2D kind splits ITS region internally
# ([0.75, 0.1, 0.1] -> image + side distribution + colorbar), which at "2x2"
# makes the image exactly 360x360 -- square, and aligned with the distribution
# and colorbar.  dpi and every font size never change with size.
#
# The geometry below is NOT a public knob: hosts pick a kind and a size preset,
# nothing else -- the visual language is owned here.
PANEL_SIZES = ("1x2", "2x2", "1x4", "2x4", "4x4")
PANEL_UNIT_PX = (180, 240)     # (height, width) of one half-unit of the stock region
PANEL_MARGINS_PX = (STOCK_MARGINS_PX[0], 86, 80, TITLE_SLOT_PX)   # stock margins (L, R, B, T).
                                         # L = STOCK_MARGINS_PX[0] (confocal's left, 110) and
                                         # is the MINIMUM that holds a 4-5 digit y-tick label
                                         # ("1180", a qCMOS ROI pixel) PLUS the rotated y-title:
                                         # at the earlier L=92 the title was pushed ~11 px past
                                         # the figure's left edge and CLIPPED (every panel kind,
                                         # 2D and 1D).  Sharing the constant guarantees a panel
                                         # never clips where the stock plot does not.
                                         # T = TITLE_SLOT_PX (the always-reserved title slot, =
                                         # _with_title_margin's floor, so panel_display_size --
                                         # which reads this top -- matches the real titled figure
                                         # instead of under-reporting the card height).
                                         # R/B stay tightened from the old (110,110,100,70)
                                         # -- they never clipped (R holds the 2D colorbar
                                         # tick labels + z-label, B the x-label+ticks) and
                                         # keep the data area dense (~71% wide).
# Panels are DISPLAYED scaled through the standard high-DPI canvas path
# (qt_canvas.panel_canvas), so on screen their text sits at ~70% of a
# notebook/pulse-preview figure while the figure stays an ordinary 300 dpi
# frontend figure.
PANEL_DISPLAY_SCALE = 0.7


def panel_size_cells(size: str) -> tuple[int, int]:
    """Parse a panel size ("rows x cols" in half-units) against the preset list."""

    key = str(size).strip().lower().replace(" ", "")
    if key not in PANEL_SIZES:
        raise ValueError(f"unknown panel size {size!r}; choose from {', '.join(PANEL_SIZES)}.")
    rows, cols = key.split("x")
    return int(rows), int(cols)


def panel_plot_spec(size: str = "2x2") -> FigureSpec:
    """FigureSpec for a dashboard panel: the stock plot region scaled in
    half-units, with the stock margins -- identical for EVERY panel kind."""

    rows, cols = panel_size_cells(size)
    return FigureSpec(
        data_px=(cols * PANEL_UNIT_PX[1], rows * PANEL_UNIT_PX[0]),
        margins_px=PANEL_MARGINS_PX)


def panel_display_size(size: str = "2x2") -> tuple[int, int]:
    """On-screen (logical px) canvas size of a panel of ``size`` -- what a host
    reserves for the canvas built by ``qt_canvas.panel_canvas``."""

    spec = panel_plot_spec(size)
    left, right, bottom, top = spec.margins_px
    width = spec.data_px[0] + left + right
    height = spec.data_px[1] + bottom + top
    return (round(width * PANEL_DISPLAY_SCALE), round(height * PANEL_DISPLAY_SCALE))


def panel_plot(data_x, data_y=None, *, kind: str, size: str = "2x2", **kwargs):
    """``plot()`` preset for dashboard panels: pick a kind and one of the
    LIMITED ``PANEL_SIZES`` and get a correctly sized, consistently styled live
    plot (no display, no DataFigure -- the host embeds the figure through
    ``qt_canvas.panel_canvas``).  Everything else IS the plot() factory,
    including the 2D square + aligned side-distribution/colorbar design.

    ``kwargs`` are the plot()/plotter DATA options (labels, title, cmap, bins,
    relim_mode, ...) -- the panel GEOMETRY is not configurable from outside
    (the size preset is the only sizing knob; geometry kwargs are rejected)."""

    spec = panel_plot_spec(size)
    return plot(data_x, data_y, kind=kind, _spec=spec, display=False, data_figure=False, **kwargs)


def pulse_plot_spec(
    channel_count: int,
    *,
    data_width_px: int = 520,
    rows_per_block: int = 10,
    block_height_px: int = 360,
    period_count: int | None = None,
    periods_per_block: int | None = None,
    max_width_factor: int | None = None,
) -> FigureSpec:
    """FigureSpec sized so pulse rows stay legible beyond 10 channels.

    The HEIGHT grows one block per ``rows_per_block`` channels, and -- mirroring
    that -- the WIDTH grows one ``data_width_px`` block per ``periods_per_block``
    periods when ``period_count`` is given, capped at ``max_width_factor`` x the
    base width (content-driven legibility knobs).  Margins and dpi are OWNED by
    the frontend visual system (``_PULSE_MARGINS_PX`` / ``_PULSE_DPI``), never
    set per-call."""

    rows_per_block = max(1, int(rows_per_block))
    chunks = max(1, ceil(max(1, int(channel_count)) / rows_per_block))
    width_blocks = 1
    if period_count is not None:
        per_block = _PULSE_X_PERIODS_PER_BLOCK if periods_per_block is None else int(periods_per_block)
        cap = _PULSE_X_MAX_WIDTH_FACTOR if max_width_factor is None else int(max_width_factor)
        if per_block > 0:
            width_blocks = min(max(1, int(cap)), max(1, ceil(max(1, int(period_count)) / per_block)))
    return FigureSpec(
        data_px=(int(data_width_px) * width_blocks, int(block_height_px) * chunks),
        margins_px=_PULSE_MARGINS_PX, dpi=_PULSE_DPI)


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
        default_forever = bool(getattr(state_or_start, "repeat_forever", default_forever))
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


def _pulse_period_starts_ns(periods, *, slots=None, time_step_ns: float | None = None) -> list[float]:
    starts_ns = [0.0]
    for period in periods:
        duration_ns = period.duration_ns(slots=slots, time_step_ns=time_step_ns)
        starts_ns.append(starts_ns[-1] + float(duration_ns))
    return starts_ns


def pulse_repeat_marker(
    state_or_periods=None,
    *,
    repeat_start: int | None = None,
    repeat_end: int | None = None,
    repeat_count: int | None = None,
    slots=None,
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
        default_forever = bool(getattr(state_or_periods, "repeat_forever", default_forever))
        slots = state_or_periods.reference_slots() if hasattr(state_or_periods, "reference_slots") else slots
        time_step_ns = getattr(state_or_periods, "time_step_ns", time_step_ns)
    elif state_or_periods is not None:
        periods = list(state_or_periods)

    if periods is None:
        if total_duration_s is None or not default_forever:
            return None
        return (0.0, float(total_duration_s), "×∞")

    starts_ns = _pulse_period_starts_ns(periods, slots=slots, time_step_ns=time_step_ns)
    if repeat_start is None or repeat_end is None:
        if not default_forever:
            return None
        return (0.0, starts_ns[-1] * 1e-9, "×∞")
    repeat_start = int(repeat_start)
    repeat_end = int(repeat_end)
    if repeat_start < 0 or repeat_end < repeat_start or repeat_end + 1 >= len(starts_ns):
        return None
    repeat_count = 1 if repeat_count is None else int(repeat_count)
    return (starts_ns[repeat_start] * 1e-9, starts_ns[repeat_end + 1] * 1e-9, f"×{repeat_count}")


def pulse_repeat_markers(
    state_or_periods=None,
    *,
    repeat_start: int | None = None,
    repeat_end: int | None = None,
    repeat_count: int | None = None,
    slots=None,
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
        default_forever = bool(getattr(state_or_periods, "repeat_forever", default_forever))
        slots = state_or_periods.reference_slots() if hasattr(state_or_periods, "reference_slots") else slots
        time_step_ns = getattr(state_or_periods, "time_step_ns", time_step_ns)
    elif state_or_periods is not None:
        periods = list(state_or_periods)

    if periods is None:
        if total_duration_s is None or not default_forever:
            return []
        return [(0.0, float(total_duration_s), "×∞")]

    starts_ns = _pulse_period_starts_ns(periods, slots=slots, time_step_ns=time_step_ns)
    total = starts_ns[-1] * 1e-9
    if repeat_start is None or repeat_end is None:
        return [(0.0, total, "×∞")] if default_forever else []

    repeat_start = int(repeat_start)
    repeat_end = int(repeat_end)
    if repeat_start < 0 or repeat_end < repeat_start or repeat_end + 1 >= len(starts_ns):
        return []
    repeat_count = 1 if repeat_count is None else int(repeat_count)
    inner = (starts_ns[repeat_start] * 1e-9, starts_ns[repeat_end + 1] * 1e-9, f"×{repeat_count}")
    if repeat_start == 0 and repeat_end == len(periods) - 1:
        return [inner]
    return [(0.0, total, "×∞"), inner] if default_forever else [inner]


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
        analog_traces: Sequence[Mapping[str, Any]] | None = None,
        auto_height: bool = True,
        period_count: int | None = None,
        labels: Sequence[str] = ("Time (s)", "", "State"),
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
        self.analog_traces = [dict(item) for item in (analog_traces or [])]
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
            # period_count drives the x auto-extend (one width block per
            # _PULSE_X_PERIODS_PER_BLOCK periods, capped at _PULSE_X_MAX_WIDTH_FACTOR).
            kwargs["spec"] = pulse_plot_spec(
                len(self.channels) + len(self.analog_traces), period_count=period_count)
        super().__init__(np.arange(dummy_n, dtype=float), np.zeros((dummy_n, 1), dtype=float), labels=labels, relim_mode="tight", **kwargs)

    @property
    def duration(self) -> float:
        if not self.pulses:
            return 1.0
        return max(float(row["stop"]) for row in self.pulses)

    def _xlimits_for_timeline(self, start_min: float, stop_max: float, *, has_bracket: bool) -> tuple[float, float]:
        """Display x-limits, deliberately a bit wider than ``[start_min, stop_max]``.

        A symmetric margin (a fraction of the time span, see
        ``_PULSE_X_MARGIN_FRAC``) is added on each side and the left edge is
        *never* clamped to 0, so a first edge at ``t=0`` still gets breathing
        room instead of sitting flush on the spine.  Negative-time headroom is
        cosmetic only -- the x-axis formatter blanks tick labels for ``value <
        0``, so no negative tick is ever shown.  When a repeat bracket is
        present, extra room is added on the right for its ``x N`` label.
        ``off_lines`` and the analog baselines are drawn out to these same
        limits, which keeps the intentional full-bleed baseline look.
        """
        span = max(float(stop_max - start_min), 1e-12)
        margin_x = max(span * _PULSE_X_MARGIN_FRAC, 1e-12)
        left_limit = start_min - margin_x
        right_limit = stop_max + margin_x
        if has_bracket:
            right_limit += span * _PULSE_X_BRACKET_LABEL_FRAC
        return left_limit, right_limit

    def init_core(self) -> None:
        self.ax.set_ylabel("")
        color_map = {channel: self.channel_colors[i % len(self.channel_colors)] for i, channel in enumerate(self.channels)}
        row_height = 0.64 if len(self.channels) <= 10 else max(0.42, 6.4 / len(self.channels))
        analog_keys = [f"analog:{index}:{trace.get('name', 'analog')}" for index, trace in enumerate(self.analog_traces)]
        row_keys = list(self.channels) + analog_keys
        n_channels = len(row_keys)
        index_map = {key: n_channels - 1 - i for i, key in enumerate(row_keys)}
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
        self._analog_baseline_y = {}
        self._pulse_row_height = row_height
        for channel in self.channels:
            y = index_map[channel]
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
                    color=PALETTE["pulse_name"],
                    fontsize=smaller_fontsize(1.2, 4.8),
                    clip_on=True,
                    zorder=pulse_zorder + 1,
                )

        self.ax.set_xlim(left_limit, right_limit)
        ylim_top = n_channels - _PULSE_Y_PAD_TOP
        if self.repeat_brackets:
            ylim_top = n_channels + 0.78 + 0.26 * max(0, len(self.repeat_brackets) - 1)
        self.ax.set_ylim(-_PULSE_Y_PAD_BOTTOM, ylim_top)
        self.analog_trace_artists = []
        self.analog_trace_labels = []
        for trace_index, (key, trace) in enumerate(zip(analog_keys, self.analog_traces)):
            y = index_map[key]
            baseline_y = y - baseline_offset
            row_top = baseline_y + row_height
            name = str(trace.get("name", f"analog{trace_index}"))
            self._analog_baseline_y[name] = baseline_y
            # SIGNED DAC values: trace["min"]..trace["max"] span the row (the legacy
            # unsigned form had min=0).  0 = true 0 V; with the signed range the 0 V
            # reference sits MID-ROW and negative values dip below it.
            # CONTRACT: trace builders (pulse_gui._analog_bus_traces, manual figure code)
            # must supply BOTH "min" and "max" for signed scaling; the 0/1 fallbacks below
            # only keep a hand-rolled legacy unsigned trace from crashing.
            v_max = int(trace.get("max", 1))
            v_min = int(trace.get("min", 0))
            v_span = max(1, v_max - v_min)
            zero_frac = min(1.0, max(0.0, (0.0 - v_min) / v_span))
            zero_y = baseline_y + row_height * zero_frac
            self._analog_zero_y = getattr(self, "_analog_zero_y", {})
            self._analog_zero_y[name] = zero_y
            color = self.channel_colors[(len(self.channels) + trace_index) % len(self.channel_colors)]
            starts = np.asarray(trace.get("starts", []), dtype=float)
            values = np.asarray(trace.get("values", []), dtype=float)
            # Dashed, semi-transparent 0 V reference line -- the analog analogue of a
            # digital channel's off line, drawn where the SIGNED value 0 maps (mid-row
            # for a bipolar bus).  Same colour + weight as the value trace, but dashed
            # and more transparent so it reads as a reference.
            self.ax.plot(
                [left_limit, right_limit],
                [zero_y, zero_y],
                color=color,
                linewidth=0.65,
                alpha=0.5,
                linestyle=(0, (4, 3)),
                zorder=pulse_zorder + 1,
            )
            if starts.size >= 2 and values.size >= 1:
                # One continuous step line that reflects the ACTUAL signed DAC value
                # (0 V on the dashed mid-row line, negatives below it), drawn solid on
                # top of the reference.  Same weight + opacity as the digital channel
                # lines (off_lines use linewidth=0.65, alpha=1.0).
                x = starts[: values.size + 1]
                y_values = baseline_y + row_height * np.clip((values[: x.size - 1] - v_min) / v_span, 0.0, 1.0)
                line_x = np.repeat(x, 2)[1:-1]
                line_y = np.repeat(y_values, 2)
                artist = self.ax.plot(
                    line_x,
                    line_y,
                    color=color,
                    linewidth=0.65,
                    alpha=1.0,
                    zorder=pulse_zorder + 2,
                )[0]
                self.analog_trace_artists.append(artist)
            else:
                # No value data: a flat solid line at 0 V (the mid-row reference).
                artist = self.ax.plot(
                    [left_limit, right_limit], [zero_y, zero_y],
                    color=color, linewidth=0.65, alpha=1.0, zorder=pulse_zorder + 2,
                )[0]
                self.analog_trace_artists.append(artist)

        self.ax.set_yticks([index_map[key] for key in row_keys])
        self.ax.set_yticklabels(
            [self.channel_labels.get(channel, channel) for channel in self.channels]
            + [str(trace.get("label") or trace.get("name") or "analog") for trace in self.analog_traces]
        )
        self.ax.tick_params(axis="y", labelsize=max(4.8, matplotlib.rcParams["ytick.labelsize"] - 1.2))
        tick_labels = self.ax.get_yticklabels()
        for tick, channel in zip(tick_labels, self.channels):
            tick.set_color(color_map[channel])
        # Colour the analog-bus row labels to match their trace, exactly like the
        # digital channels above -- otherwise the DAC rows' names stayed default
        # black while every other label was tinted to its line.
        for trace_index, tick in enumerate(tick_labels[len(self.channels):]):
            tick.set_color(self.channel_colors[(len(self.channels) + trace_index) % len(self.channel_colors)])
        if self.repeat_notation and not self.repeat_brackets:
            self.ax.text(
                0.995,
                1.012,
                self.repeat_notation,
                transform=self.ax.transAxes,
                ha="right",
                va="bottom",
                color=PALETTE["pulse_repeat_note"],
                fontsize=smaller_fontsize(1.0, 5.5),
            )
        self._draw_repeat_bracket(n_channels)
        self.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
        self.ax.xaxis.set_major_formatter(_PulseTimeFormatter(self.time_scale))
        self.ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False, pad=2)
        self.ax.set_axisbelow(True)
        self.ax.grid(axis="x", color=PALETTE["pulse_grid"], linewidth=0.35, zorder=0)
        for gridline in self.ax.get_xgridlines():
            gridline.set_zorder(0)
        self.ax.spines[["top", "right"]].set_visible(False)
        self.lines = [*self.off_lines, *self.pulse_artists, *self.analog_trace_artists]

    def update_core(self) -> None:
        pass

    def _draw_repeat_bracket(self, n_channels: int) -> None:
        if not self.repeat_brackets:
            return
        colors = PALETTE["bracket_cycle"]
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
            y_low = -0.42 - 0.13 * outer_depth
            y_high = float(n_channels) - 0.10 + 0.34 * outer_depth
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
                y_high + 0.055,
                label,
                ha="left",
                va="bottom",
                color=color,
                fontfamily="DejaVu Sans",
                alpha=alpha,
                fontsize=smaller_fontsize(0.8, 5.5),
                clip_on=False,
                zorder=9 + index,
            )
            self.repeat_bracket_labels.append(label_artist)
        if self.repeat_bracket_artists:
            self.repeat_bracket_artist = self.repeat_bracket_artists[0]
        if self.repeat_bracket_labels:
            self.repeat_bracket_label = self.repeat_bracket_labels[-1]

    def _attach_interactions(self) -> None:
        # Enable the left-drag area selector on the pulse timeline too (it used to
        # be disabled), so a region can be box-selected like the other plots.
        self.tools = attach_interaction(self.ax)
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
        ylog: bool = False,
        bimodal: bool = True,
        **kwargs,
    ):
        self.values = np.asarray(values, dtype=float).reshape(-1)
        self.bins_arg = bins
        self.ylog = bool(ylog)          # log-scale the COUNT axis (reveals a sparse bright tail)
        # The readout histogram is a dark/bright DISTRIBUTION, so the default IS the two-Gaussian
        # (bimodal) decomposition -- consistently, for EVERY signal -- not auto-chosen per data (which
        # made it flip single/double between signals).  Turn it OFF for a plain single-Gaussian view.
        self._bimodal = bool(bimodal)
        self._fit_separated = False     # whether the bimodal fit cleanly separated (gates the fidelity stat)
        # A threshold (cut line) is only MEANINGFUL when there are two populations to separate (#issue-2):
        # shown when the bimodal fit separated OR a threshold was supplied explicitly (a calibration's
        # per-site cut).  A pure single-Gaussian fit shows the FIT params + out-of-fit fraction instead.
        self._explicit_thr = bool(thresholds)
        self._has_threshold = self._explicit_thr
        self.single_popt = None         # (amp, mu, sigma) of the single-Gaussian fit, for the no-threshold stat
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
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"])
        self.ax.add_collection(self.poly)
        (self.fit_line_left,) = self.ax.plot([], [], color=PALETTE["fit_left"], linewidth=1, alpha=0.8)
        (self.fit_line_right,) = self.ax.plot([], [], color=PALETTE["fit_right"], linewidth=1, alpha=0.8)
        (self.fit_line_total,) = self.ax.plot([], [], color=PALETTE["fit_total"], linewidth=1, alpha=0.35)
        self.bimodal_popt = None
        self.fit_threshold = None
        self._fit_bimodal()
        if not self.thresholds:
            self.thresholds = [self.fit_threshold if self.fit_threshold is not None else float(np.nanmedian(vals))]
        self.threshold_lines = []
        self.threshold_draggers = []
        for threshold in self.thresholds:
            line = self.ax.axvline(threshold, **threshold_line_kwargs())
            line.set_visible(self._has_threshold)        # hidden on a single-Gaussian (no meaningful cut)
            self.threshold_lines.append(line)
            self.threshold_draggers.append(DragVLine(line, self._on_threshold_drag, self.ax))
        self.stats_text = self.ax.text(
            0.975,
            0.975,
            "",
            transform=self.ax.transAxes,
            ha="right",
            va="top",
            color=PALETTE["annotation"],
            fontsize=small_fontsize(),
        )
        self.ax.set_xlim(self.bins[0], self.bins[-1])
        self._apply_count_yscale()
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
        self._apply_count_yscale()
        self._fit_bimodal()
        while len(self.threshold_lines) < len(self.thresholds):
            line = self.ax.axvline(self.thresholds[len(self.threshold_lines)], **threshold_line_kwargs())
            self.threshold_lines.append(line)
            self.threshold_draggers.append(DragVLine(line, self._on_threshold_drag, self.ax))
        for line, threshold in zip(self.threshold_lines, self.thresholds):
            line.set_xdata([threshold, threshold])
            line.set_visible(self._has_threshold)        # show/hide live as the data separates / merges
        self._update_hist_stats()

    def set_thresholds(self, thresholds: Sequence[float]):
        self.thresholds = list(thresholds)
        self._explicit_thr = bool(thresholds)            # an explicit cut shows even on a single Gaussian
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

    def _apply_count_yscale(self) -> None:
        """Set the count (y) axis scale + limits.  Log mode floors at 0.5 so 0-count bars sit BELOW
        the axis (no 0 -> -inf on the filled poly) and a sparse bright tail becomes visible."""
        top = max(1.0, float(np.max(self.n)) * (3.0 if self.ylog else 1.2))
        if self.ylog:
            self.ax.set_yscale("log")
            self.ax.set_ylim(0.5, top)
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(0, top)

    def _fit_bimodal(self) -> None:
        # A robust two-Gaussian fit with a UNIMODAL FALLBACK.  The old median split sat INSIDE the
        # dark blob whenever the bright mode was sparse (rare high occupancy), seeding both Gaussians
        # on dark so curve_fit collapsed to one blob and reported a MISLEADING fidelity.  Fix: split by
        # between-class variance (Otsu) over the samples, seed each side's amplitude from its own bin
        # counts, and -- when the bright mode is too sparse or unseparated -- fit ONE Gaussian and
        # report fit F = N/A (honest), never a fake number.
        vals = self.values[np.isfinite(self.values)]
        self.bimodal_popt = None
        self.fit_threshold = None
        self.single_popt = None
        self._fit_separated = False
        self._has_threshold = self._explicit_thr        # no fit yet -> only an explicit cut shows
        for ln in (self.fit_line_left, self.fit_line_right, self.fit_line_total):
            ln.set_data([], [])
        if vals.size < 6 or np.ptp(vals) == 0:
            return
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        counts = self.n.astype(float)
        span = float(np.ptp(vals)) or 1.0
        x_fit = np.linspace(self.bins[0], self.bins[-1], 400)

        # Otsu split on the sorted samples (between-class variance), NOT the median.
        xs = np.sort(vals)
        csum = np.cumsum(xs)
        k = np.arange(1, xs.size, dtype=float)
        score = k * (xs.size - k) * ((csum[-1] - csum[:-1]) / (xs.size - k) - csum[:-1] / k) ** 2
        si = int(np.argmax(score))
        split = float(0.5 * (xs[si] + xs[si + 1]))
        left, right = vals[vals <= split], vals[vals > split]
        if left.size < 2 or right.size < 2:
            left, right = vals[: vals.size // 2], vals[vals.size // 2:]
        mu0, mu1 = float(np.mean(left)), float(np.mean(right))
        if mu0 > mu1:
            mu0, mu1 = mu1, mu0
        s0 = max(float(np.std(left)), span / 40, 1e-9)
        s1 = max(float(np.std(right)), span / 40, 1e-9)

        def _amp_near(mu):
            return max(float(counts[int(np.clip(np.searchsorted(centers, mu), 0, len(counts) - 1))]), 1.0)

        def _draw_unimodal():
            try:
                p0 = [max(float(np.max(counts)), 1.0), float(np.mean(vals)), max(float(np.std(vals)), span / 40, 1e-9)]
                popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=20000)
                self.fit_line_total.set_data(x_fit, gaussian(x_fit, *popt))   # one curve, fit F = N/A
                self.single_popt = (float(popt[0]), float(popt[1]), float(popt[2]))   # for the no-threshold stat
            except Exception:
                pass

        # A SINGLE-Gaussian outcome -> NO threshold (a cut between two populations is meaningless when
        # there is one): only show a cut if it was supplied EXPLICITLY (a calibration's per-site cut).
        def _single_gaussian():
            _draw_unimodal()
            self.fit_threshold = None
            self._has_threshold = self._explicit_thr

        # SINGLE-Gaussian mode (the toggle is OFF): one curve, no dark/bright split, fit F = N/A.
        if not self._bimodal:
            _single_gaussian()
            return

        # BIMODAL mode (the DEFAULT): always draw the two-Gaussian dark/bright decomposition, so the
        # readout histogram reads consistently for EVERY signal -- never auto-collapsed to one peak by
        # the data.  Only a genuine numerical inability to seed two peaks (a handful of bright samples,
        # or curve_fit failure) falls back to one Gaussian.
        if right.size < max(4, int(0.01 * vals.size)):
            _single_gaussian()
            return

        p0 = [_amp_near(mu0), mu0, s0, _amp_near(mu1), mu1, s1]
        bounds = (
            [0, float(np.min(vals)), span / 200, 0, float(np.min(vals)), span / 200],
            [max(_amp_near(mu0) * 5, 1), float(np.max(vals)), span * 2,
             max(_amp_near(mu1) * 5, 1), float(np.max(vals)), span * 2],
        )
        try:
            popt, _ = curve_fit(bimodal_model, centers, counts, p0=p0, bounds=bounds,
                                 jac=bimodal_jacobian, maxfev=20000)
        except Exception:
            _single_gaussian()
            return
        if popt[1] > popt[4]:
            popt = np.array([popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]], dtype=float)
        # DRAW both peaks always (bimodal mode).  The FIDELITY stat stays honest: it is reported only
        # when the fitted means separate by >~1.5 summed widths (a real readout bimodal); below that the
        # two peaks still draw but ``_fit_fidelity`` returns N/A (never a fake number) -- the same
        # FITTED-separation gate as before, now driving only the stat, not the visual.
        self.bimodal_popt = popt
        fitted_sep = abs(popt[4] - popt[1]) / (abs(popt[2]) + abs(popt[5]) + 1e-12)
        self._fit_separated = fitted_sep >= 1.5
        y0 = gaussian(x_fit, *popt[:3])
        y1 = gaussian(x_fit, *popt[3:])
        self.fit_line_left.set_data(x_fit, y0)
        self.fit_line_right.set_data(x_fit, y1)
        self.fit_line_total.set_data(x_fit, y0 + y1)
        # A threshold is only meaningful when the two peaks actually SEPARATED.  Separated -> the cut at
        # the cross-over; not separated (one effective population) -> NO auto cut (#issue-2): show the fit
        # + out-of-fit fraction instead.  An explicitly-supplied cut (calibration) still shows either way.
        self._has_threshold = self._fit_separated or self._explicit_thr
        lo, hi = float(popt[1]), float(popt[4])
        if self._fit_separated and hi > lo:
            x_mid = np.linspace(lo, hi, 400)
            diff = np.abs(gaussian(x_mid, *popt[:3]) - gaussian(x_mid, *popt[3:]))
            self.fit_threshold = float(x_mid[int(np.nanargmin(diff))])
        else:
            self.fit_threshold = None

    def _fit_fidelity(self, threshold: float) -> float | None:
        # Honest fidelity only when the two-Gaussian fit cleanly SEPARATED (>=1.5 summed widths); the
        # decomposition may be DRAWN for an unseparated distribution (bimodal mode) but its fidelity is
        # meaningless, so report N/A rather than a fake number.
        if self.bimodal_popt is None or not self._fit_separated:
            return None
        amp0, mu0, sigma0, amp1, mu1, sigma1 = self.bimodal_popt
        w0 = abs(amp0 * sigma0)
        w1 = abs(amp1 * sigma1)
        if (w0 + w1) == 0:
            return None
        # The single-shot readout fidelity = the HONEST two-Gaussian overlap about the
        # threshold (the weighted dark-below + bright-above probability of the FITTED
        # populations).  We deliberately use ``raw``, NOT the confidence-DAMPED value: the
        # damping (effective_separation = sep - 2) pulls a cleanly separated distribution
        # toward 0.5 -- a sep~2.5 split read ~0.58, contradicting the visibly separated
        # histogram.  Because this path has a real bimodal FIT, the overlap IS the fidelity
        # (matching the per-site grids' gaussian_fidelity); the damping only guards the
        # threshold-SPLIT estimate (analysis.estimate_threshold_fidelity), which is untouched.
        _fidelity, raw, _sep = confidence_weighted_fidelity(threshold, mu0, sigma0, w0, mu1, sigma1, w1)
        return float(raw)

    def _on_threshold_drag(self, x: float) -> None:
        if not self._has_threshold:
            return                                       # a hidden (single-Gaussian) cut can't be dragged
        if not self.thresholds:
            self.thresholds = [float(x)]
        else:
            self.thresholds[0] = float(x)
        self._update_hist_stats()

    def _out_of_fit_fraction(self) -> float | None:
        """Fraction of the histogram mass NOT explained by the fitted curve (the overlap-coefficient
        complement: ``1 - sum(min(counts, fitted)) / sum(counts)``).  Uses the single-Gaussian fit when
        there is one, else the two-Gaussian sum; the gaussian math is the single-source _readout_math."""
        n = getattr(self, "n", None)
        if n is None or float(np.sum(n)) <= 0 or getattr(self, "bins", None) is None:
            return None
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        if self.single_popt is not None:
            fitted = gaussian(centers, *self.single_popt)
        elif self.bimodal_popt is not None:
            fitted = bimodal_model(centers, *self.bimodal_popt)
        else:
            return None
        overlap = float(np.sum(np.minimum(n.astype(float), np.clip(fitted, 0.0, None))))
        return float(max(0.0, 1.0 - overlap / float(np.sum(n))))

    def _update_hist_stats(self) -> None:
        # WITH a meaningful cut (separated fit or an explicit calibration cut): the threshold readout.
        if self._has_threshold and self.thresholds:
            threshold = float(self.thresholds[0])
            vals = self.values[np.isfinite(self.values)]
            if vals.size:
                left = float(np.mean(vals <= threshold))
                right = 1.0 - left
            else:
                left = right = 0.0
            fidelity = self._fit_fidelity(threshold)
            fidelity_text = "fit F=N/A" if fidelity is None else f"fit F={100 * fidelity:.1f}%"
            fit_threshold = "" if self.fit_threshold is None else f"\nfit cut={self.fit_threshold:.4g}"
            self.stats_text.set_text(
                f"th={threshold:.4g}\n{fidelity_text}\nL/R={100 * left:.1f}%/{100 * right:.1f}%{fit_threshold}"
            )
            return
        # SINGLE-Gaussian (no meaningful cut, #issue-2): show the FIT params + the out-of-fit fraction
        # (how much of the data the single Gaussian does NOT explain), NOT a meaningless threshold.
        out = self._out_of_fit_fraction()
        out_text = "" if out is None else f"\nout-of-fit={100 * out:.1f}%"
        if self.single_popt is not None:
            _amp, mu, sigma = self.single_popt          # ASCII labels: the figure font lacks Greek glyphs
            self.stats_text.set_text(f"gauss mean={mu:.4g}\nsd={abs(sigma):.3g}{out_text}")
        else:
            self.stats_text.set_text(f"single peak (no split){out_text}")

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="hist", x_array=self.bins, y_array=self.n)


def _is_watch_update(update) -> bool:
    if isinstance(update, str):
        return update.lower() in {"watch", "timer", "live", "auto"}
    return bool(update)


@dataclass(frozen=True)
class PlotKind:
    """ONE declarative record per plot kind -- the single source both ``live.plot()``
    and the task_console Add-Panel / PANEL_* lookups READ.

    Adding a plot kind means adding ONE ``PlotKind`` to :data:`PLOT_KINDS`; the
    ``plot()`` dispatch, the kind->class lookup, the panel label, the accepted
    ``value`` format, the starting signal slots and the single-vs-multi-slot rule
    all derive from it -- no parallel dicts to keep in sync.

    Fields
    ------
    key            the canonical kind string (``"1d"``, ``"2d"``, ...).
    cls            the :class:`BaseLivePlot` subclass ``plot()`` instantiates.
    label          the human Add-Panel / panel-title label.
    render_family  the DataFigure fitting family ("1D" / "2D"), mirroring
                   ``cls.render_family``.  The sentinel ``"auto"`` means "resolve
                   per-figure from the artists" (the site map: image-family only
                   when a background frame is supplied) -- DataFigure then keeps
                   its legacy artist heuristic for that kind.
    panel          True if this kind is offered as a console Add-Panel plot
                   (every kind except the notebook-only ``pulse`` diagram).
    input_format   one-line description of the accepted ``value`` shape (shown in
                   the Setting; ``""`` for a non-panel kind).
    input_slots    the STARTING signal slot(s) a fresh panel opens with --
                   ``((label, default_signal, tooltip), ...)``; empty = the
                   universal single ``signal`` slot.
    single_slot    True if the kind takes EXACTLY ONE signal (no +/- slot growing).
    repeat_modes   the repeat-DISPLAY modes that are MEANINGFUL for this kind, in menu order
                   (the first is the default).  A TRACE/IMAGE reduces its repeats
                   (``average``/``add``/``replace``/``roll``, plus ``create`` = one line per repeat
                   for 1-D); a DISTRIBUTION instead chooses how to BIN the repeats' samples
                   (``pool`` = all repeats in one histogram -- the only distribution mode).
                   Empty = a non-repeat kind (no repeat_mode control).  This is what stops a
                   trace verb like ``roll`` being offered on a histogram (where it is meaningless)
                   and stops a histogram silently ignoring the control (#issue-1).
    """

    key: str
    cls: type
    label: str
    render_family: str = "1D"
    panel: bool = True
    input_format: str = ""
    input_slots: tuple[tuple[str, str, str], ...] = ()
    single_slot: bool = False
    repeat_modes: tuple[str, ...] = ()


# The ONE plot-kind table.  ``plot()`` looks the class up here (no if/elif ladder)
# and the task_console derives PANEL_KINDS / PANEL_INPUT_FORMAT / PANEL_INPUT_SLOTS /
# PANEL_SINGLE_SLOT_KINDS from it (no parallel literals).  Order is the Add-Panel
# menu order.  ``monitor`` lists its DEFAULT class (LiveLiveDis, show_dist=True); the
# bare LiveLive variant is the show_dist=False toggle, still inside plot().
# Repeat-display vocabularies (single source).  TRACE/IMAGE modes REDUCE the repeat axis;
# the DISTRIBUTION instead BINS the repeats' samples -- a separate, dedicated vocabulary so a
# trace verb (roll/replace) is never offered on (or silently ignored by) a histogram (#issue-1).
TRACE_REPEAT_MODES: tuple[str, ...] = REPEAT_MODES                                     # full set (create = 1-D)
IMAGE_REPEAT_MODES: tuple[str, ...] = ("average", "add", "replace")                    # a frame: mean/sum/latest
HIST_REPEAT_MODES: tuple[str, ...] = ("pool",)              # distribution: bin ALL repeats' samples together

PLOT_KINDS: tuple[PlotKind, ...] = (
    PlotKind(
        key="2d", cls=Live2DDis, label="2D image", render_family="2D",
        input_format="value must be a 2D array / camera frame (H×W)",
        repeat_modes=IMAGE_REPEAT_MODES,
    ),
    PlotKind(
        key="sites", cls=LiveSiteMap, label="Site map", render_family="auto",
        single_slot=True, repeat_modes=IMAGE_REPEAT_MODES,
        input_format=(
            "value must be a per-site (N,) vector -- one number per tweezer (e.g. occupancy "
            "0/1 or loading rate); signal[0]'s producing node also supplies the ring centres "
            "+ frame underlay"),
        input_slots=(
            # BLANK default (like every other plot) -- a fresh site-map panel must NOT auto-bind
            # to a running "occupied" signal on open; the user picks the occupancy signal in the
            # Setting, and only THEN do the centres + frame underlay auto-resolve from that signal's
            # producing node (_sites_inputs).  A non-blank default here was the "opens already
            # connected" bug.
            ("occupancy", "", "per-site (N,) occupancy vector (signal[0]) -- colours the rings; its "
                              "producing node also supplies the centres + frame underlay"),
        ),
    ),
    PlotKind(
        key="1d", cls=Live1D, label="1D vector", render_family="1D",
        input_format="value must be a 1D vector (N,) or per-site array",
        repeat_modes=TRACE_REPEAT_MODES,
    ),
    PlotKind(
        key="monitor", cls=LiveLiveDis, label="Rolling trace", render_family="1D",
        input_format="value must be a scalar per shot (rolling trace)",
        repeat_modes=TRACE_REPEAT_MODES,
    ),
    PlotKind(
        key="hist", cls=HistogramFigure, label="Distribution", render_family="1D",
        input_format="value must be a 1D sample vector",
        repeat_modes=HIST_REPEAT_MODES,
    ),
    # Notebook-only static timing diagram -- NOT a console Add-Panel kind.
    PlotKind(key="pulse", cls=PulseSequenceFigure, label="Pulse sequence", render_family="1D", panel=False),
)

#: ``key -> PlotKind`` for O(1) dispatch.
PLOT_KIND_BY_KEY: dict[str, PlotKind] = {pk.key: pk for pk in PLOT_KINDS}


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
        # Rolling trace is ONE kind ("monitor"); the side distribution is a show_dist toggle,
        # NOT a separate kind, so these synonyms all collapse to "monitor".
        "live": "monitor",
        "live-dis": "monitor",
        "live-distribution": "monitor",
        "rolling": "monitor",
        "loading-rate": "monitor",
        "site-map": "sites",
        "sitemap": "sites",
        "site": "sites",
    }
    return aliases.get(normalized, normalized)


# Geometry / dpi / NaN-and-palette colours are OWNED by the frontend visual
# system (see the frontend/__init__.py contract): the public factories reject
# any attempt to set them per call.  ``spec`` is the INTERNAL geometry channel
# (panel_plot/pulse spec pass it via the private ``_spec`` argument), so a
# caller can never inject a FigureSpec through the public surface either.
_SEALED_PLOT_KWARGS = ("spec", "data_px", "margins_px", "dpi", "bad_color", "colors")


def _reject_sealed_kwargs(kwargs: dict) -> None:
    leaked = [key for key in _SEALED_PLOT_KWARGS if key in kwargs]
    if leaked:
        raise TypeError(
            "frontend.plot()/panel_plot() do not accept "
            + ", ".join(leaked)
            + ": figure geometry, margins, dpi and NaN/palette colours are owned by "
            "the frontend visual system and are not configurable per call. Pick a panel "
            "size from PANEL_SIZES for sizing; labels/title/cmap/bins/thresholds/relim_mode "
            "are the data options you may pass."
        )


def _as_per_site_list(per_site_values, n_sites: int | None = None):
    """Normalize ``(n_sites, n_samples)`` array OR list-of-1D-arrays to a list."""

    if isinstance(per_site_values, np.ndarray) and per_site_values.ndim == 2:
        out = [np.asarray(row, dtype=float).reshape(-1) for row in per_site_values]
    else:
        out = [np.asarray(v, dtype=float).reshape(-1) for v in per_site_values]
    if n_sites is not None and len(out) != n_sites:
        raise ValueError(f"expected {n_sites} sites, got {len(out)}.")
    return out


class GridCell:
    """Strategy for ONE cell of a :class:`GridPlot` -- subclass per multi-panel
    plot type (``HistogramCell`` now; future ``Image2DCell`` / ``Line1DCell``).

    The GridPlot owns everything generic (layout, focus-zoom, per-cell selectors,
    DataFigure plumbing); the cell only knows how to DRAW one panel and what its
    threshold/DataFigure are.  ``draw(ax, k, detail=False)`` must work at any axes
    size -- the same method draws the small grid cell AND the enlarged focus view,
    which is exactly why the focus-zoom is plot-type-agnostic."""

    n_cells: int = 0

    def prepare(self) -> None:
        """Compute any shared state (e.g. common histogram bin edges) once."""

    def draw(self, ax, k: int, *, detail: bool = False):
        """Draw cell ``k`` into ``ax``.  ``detail=True`` is the enlarged focus view
        (full ticks/labels/title).  Return the cell's draggable threshold line or
        ``None``."""
        raise NotImplementedError

    def threshold_line(self, k: int):
        return None

    def on_threshold_drag(self, k: int, x: float) -> None:
        """Called while a threshold is dragged (update data + any annotation)."""

    def data_figure(self, fig, ax, k: int):
        """The per-cell :class:`DataFigure` (reusable fitting stack)."""
        raise NotImplementedError


class HistogramCell(GridCell):
    """A GridPlot cell that is a per-site count histogram (the distribution grid).

    ``per_site_values`` is ``(n, n_samples)`` or a list of 1D arrays; ``occupied``
    colours dark/bright populations; ``thresholds`` draws a draggable cut;
    ``site_fidelities`` annotates each cell."""

    def __init__(self, per_site_values, *, occupied=None, thresholds=None,
                 site_fidelities=None, bins: int = 36, labels: Sequence[str] = ("Signal", "Shots")):
        self.values = _as_per_site_list(per_site_values)
        self.n_cells = len(self.values)
        if self.n_cells == 0:
            raise ValueError("per_site_values must contain at least one cell.")
        self.occupied = None if occupied is None else _as_per_site_list(occupied, self.n_cells)
        thr = None if thresholds is None else np.asarray(thresholds, dtype=float).reshape(-1)
        if thr is not None and thr.size != self.n_cells:
            raise ValueError("thresholds must have one value per cell.")
        self.thresholds = None if thr is None else [float(t) for t in thr]
        self.site_fidelities = None if site_fidelities is None else np.asarray(site_fidelities, dtype=float).reshape(-1)
        self.bins_arg = int(bins)
        self.labels = tuple(labels)
        self.edges = None
        self.threshold_lines: list = [None] * self.n_cells
        self.tag_texts: list = [None] * self.n_cells
        self.cell_counts: list = [None] * self.n_cells

    def prepare(self) -> None:
        vals = self.values
        pooled = np.concatenate([v[np.isfinite(v)] for v in vals if v.size]) if any(v.size for v in vals) else np.array([0.0, 1.0])
        lo, hi = np.quantile(pooled, [0.002, 0.998]) if pooled.size > 2 else (float(np.min(pooled)), float(np.max(pooled)))
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = float(np.min(pooled)), float(np.max(pooled)) + 1.0
        span = hi - lo
        self.x_lo, self.x_hi = float(lo - 0.04 * span), float(hi + 0.04 * span)
        self.edges = np.linspace(self.x_lo, self.x_hi, self.bins_arg + 1)

    def _tag_text(self, k: int) -> str:
        fids = self.site_fidelities
        return f"s{k}" if fids is None or not np.isfinite(fids[k]) else f"s{k}\n{fids[k] * 100:.0f}%"

    def draw(self, ax, k: int, *, detail: bool = False):
        if self.edges is None:
            self.prepare()
        edges = self.edges
        v = self.values[k]
        v = v[np.isfinite(v)]
        occ = self.occupied
        if occ is not None and occ[k].size == self.values[k].size:
            mask = np.asarray(occ[k], dtype=bool)[np.isfinite(self.values[k])]
            dark, bright = v[~mask], v[mask]
            if dark.size or bright.size:
                ax.hist([dark, bright], bins=edges, stacked=True, color=[PALETTE["dark"], PALETTE["bright"]], edgecolor="none")
        elif v.size:
            ax.hist(v, bins=edges, color=PALETTE["hist_fill"], edgecolor="none")
        self.cell_counts[k] = np.histogram(v, bins=edges)[0].astype(float) if v.size else np.zeros(self.bins_arg)
        line = None
        if self.thresholds is not None and np.isfinite(self.thresholds[k]):
            line = ax.axvline(float(self.thresholds[k]), **threshold_line_kwargs(1.4 if not detail else 1.9))
        ax.set_xlim(self.x_lo, self.x_hi)
        top = ax.get_ylim()[1]
        if detail:
            # Enlarged single-cell view: a PROPER standalone plot -- show the counts
            # axis, both axis labels, and ONE title naming the site (no tiny corner
            # tag, no doubled labels; the grid's outer labels are hidden in focus).
            ax.set_ylim(0, top * 1.18 if top > 0 else 1.0)
            fid = self.site_fidelities
            ftxt = "" if (fid is None or not np.isfinite(fid[k])) else f"   F={fid[k] * 100:.1f}%"
            ax.set_xlabel(self.labels[0], fontsize=axis_label_fontsize())
            ax.set_ylabel(self.labels[1] if len(self.labels) > 1 else "Shots", fontsize=axis_label_fontsize())
            ax.tick_params(labelsize=axis_label_fontsize(), length=2.5)
            apply_title(ax, f"site {k}{ftxt}")
        else:
            # Grid cell (unchanged, byte-identical): corner tag, hidden counts axis.
            ax.set_ylim(0, top * 1.45 if top > 0 else 1.0)   # headroom so the tag clears the bars
            ax.text(0.06, 0.95, self._tag_text(k), transform=ax.transAxes, ha="left", va="top",
                    fontsize=small_fontsize(), color=PALETTE["annotation"], linespacing=0.92)
            ax.tick_params(labelsize=tick_fontsize(), length=2)
            ax.set_yticklabels([])               # counts scale varies per cell; shape is the point
            self.threshold_lines[k] = line       # grid line, kept for per-cell drag
        return line

    def threshold_line(self, k: int):
        return self.threshold_lines[k] if k < len(self.threshold_lines) else None

    def on_threshold_drag(self, k: int, x: float) -> None:
        if self.thresholds is None:
            self.thresholds = [float("nan")] * self.n_cells
        self.thresholds[k] = float(x)
        grid_line = self.threshold_lines[k]
        if grid_line is not None:             # keep the grid line synced with a focus-view drag
            grid_line.set_xdata([x, x])

    def data_figure(self, fig, ax, k: int):
        from .data_figure import DataFigure
        centers = (self.edges[:-1] + self.edges[1:]) / 2
        ylabel = self.labels[1] if len(self.labels) > 1 else "Shots"
        return DataFigure(fig=fig, ax=ax, data_x=centers, data_y=self.cell_counts[k],
                          labels=(self.labels[0], ylabel), name=f"site{k}")

    def classify(self, k: int, values=None) -> np.ndarray:
        vals = self.values[k] if values is None else np.asarray(values, dtype=float)
        if self.thresholds is not None and np.isfinite(self.thresholds[k]):
            thr = self.thresholds[k]
        else:
            finite = vals[np.isfinite(vals)]
            thr = float(np.nanmedian(finite)) if finite.size else 0.0
        return (vals > thr).astype(int)


class ImageCell(GridCell):
    """A GridPlot cell that is a 2D image -- one ``imshow`` per cell.  The readout's
    per-site **PSF weight kernel** uses it, so the operator SEES the real (asymmetric,
    non-Gaussian) atom spot the matched filter weights by.

    ``images`` is a sequence of 2D arrays (one per cell); ALL cells share ONE colour
    scale so the grid is comparable (the art "aligned" rule).  An image cell has no 1D
    cut, so it contributes the zoom/pan/area/cross selectors but no draggable threshold
    line -- exactly the ``Image2DCell`` the :class:`GridCell` docstring reserves."""

    def __init__(self, images, *, labels: Sequence[str] = ("x (px)", "y (px)")):
        self.images = [np.asarray(im, dtype=float) for im in images]
        self.n_cells = len(self.images)
        if self.n_cells == 0:
            raise ValueError("images must contain at least one cell.")
        self.labels = tuple(labels)
        self.vmax = 1.0

    def prepare(self) -> None:
        finite = [im[np.isfinite(im)] for im in self.images if im.size]
        pooled = np.concatenate(finite) if finite else np.array([0.0, 1.0])
        hi = float(np.nanmax(pooled)) if pooled.size else 1.0
        self.vmax = hi if hi > 0 else 1.0   # one shared colour scale -> cells comparable

    def draw(self, ax, k: int, *, detail: bool = False):
        ax.imshow(self.images[k], origin="lower", cmap=PALETTE["cmap_camera"],
                  vmin=0.0, vmax=self.vmax, aspect="equal")
        if detail:
            # Enlarged single-cell view: a proper standalone image (axes + labels + title).
            ax.set_xlabel(self.labels[0], fontsize=axis_label_fontsize())
            ax.set_ylabel(self.labels[1] if len(self.labels) > 1 else "", fontsize=axis_label_fontsize())
            ax.tick_params(labelsize=axis_label_fontsize(), length=2.5)
            apply_title(ax, f"site {k}")
        else:
            # Grid cell: a corner tag, no ticks (the kernel SHAPE is the point).  The tag
            # sits ON the image, so a plain dark label vanishes on the dark corner -- draw
            # it light (the "label on a coloured fill" token) with a dark stroke so it reads
            # on ANY grayscale pixel (dark corner OR a bright lobe that reaches the corner).
            import matplotlib.patheffects as pe
            txt = ax.text(0.06, 0.95, f"s{k}", transform=ax.transAxes, ha="left", va="top",
                          fontsize=small_fontsize(), color=PALETTE["pulse_name"], linespacing=0.92)
            txt.set_path_effects([pe.withStroke(linewidth=1.4, foreground=PALETTE["annotation"])])
            ax.set_xticks([])
            ax.set_yticks([])
        return None

    def data_figure(self, fig, ax, k: int):
        from .data_figure import DataFigure
        # The grid's ``.npz`` (the save contract) carries each site's PSF kernel as a
        # 1D series (square-recoverable); the canonical 2D weights live in calibration.npz.
        flat = self.images[k].reshape(-1)
        return DataFigure(fig=fig, ax=ax, data_x=np.arange(flat.size, dtype=float),
                          data_y=flat, labels=self.labels, name=f"site{k}_psf")


class _GridData:
    """Composite ``DataFigure`` handle for a :class:`GridPlot`: one per-cell
    :class:`DataFigure` (each panel fits with the SAME stack as any plot) + a
    whole-grid ``save``."""

    def __init__(self, grid: "GridPlot"):
        self.grid = grid
        self.fig = grid.fig
        self.cells = [grid.cell_renderer.data_figure(grid.fig, ax, k) for k, ax in enumerate(grid.site_axes)]

    def cell(self, k: int):
        return self.cells[k]

    def save(self, path: str = "", *, extra_info=None, image_ext: str = ".png", **kwargs):
        """Save the whole grid the SAME way DataFigure saves a single plot: ONE png AND
        ONE matching ``.npz`` of the plotted data (here the per-cell raw distributions).
        The path-stem logic + the ``extra_info`` / ``image_ext`` kwargs mirror
        :meth:`DataFigure.save` so the SAME call works on a grid or a single plot (else a
        ``grid.save(path, extra_info=...)`` would crash by forwarding those into ``savefig``)."""
        import time
        from .data_figure import resolve_save_base
        base = resolve_save_base(path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))   # shared path stem (#C4)
        image_path = base.with_suffix(image_ext if str(image_ext).startswith(".") else f".{image_ext}")
        data_path = base.with_suffix(".npz")
        self.fig.savefig(image_path, **kwargs)
        # Pack each cell's DataFigure data (data_x / data_y) into the .npz, one key per
        # cell; the experimenter can reload the per-site distributions identically to a
        # single-plot save.
        bundle: dict[str, np.ndarray] = {}
        for k, c in enumerate(self.cells):
            try:
                bundle[f"cell{k}_data_x"] = np.asarray(c.data_x, dtype=float)
                bundle[f"cell{k}_data_y"] = np.asarray(c.data_y, dtype=float)
            except Exception:
                continue
        bundle["n_cells"] = np.asarray(len(self.cells), dtype=int)
        if extra_info is not None:    # mirror DataFigure.save: caller metadata travels with the data
            try:
                bundle["info"] = np.asarray(extra_info)
            except Exception:
                pass
        np.savez(data_path, **bundle)
        return {"figure": image_path, "data": data_path}


class GridPlot(BaseLivePlot):
    """A general multi-panel plot: N aligned cells, the cell CONTENT pluggable via
    a :class:`GridCell` (``HistogramCell`` now; future ``Image2DCell`` /
    ``Line1DCell``).  A FIRST-CLASS plot (subclasses :class:`BaseLivePlot`), so it
    reuses the frontend layer: per-cell selectors, per-cell :class:`DataFigure`,
    and a **focus-zoom** -- double-click a cell to enlarge it to the whole figure
    (full ticks/labels + that cell's selectors) to see detail, double-click or
    press Esc to return to the grid.  Geometry/gaps/colours/dpi/fonts are owned by
    the frontend -- cells never overlap, nothing is cut off, all cells align."""

    plot_type = "grid"

    def __init__(self, cell: GridCell, *, grid_shape: tuple[int, int] | None = None,
                 labels: Sequence[str] = ("X", "Y"), **kwargs):
        self.cell_renderer = cell
        self.n_cells = int(cell.n_cells)
        if self.n_cells == 0:
            raise ValueError("a GridPlot needs at least one cell.")
        cell.prepare()
        self.nrows, self.ncols = grid_shape_for(self.n_cells, prefer=grid_shape, max_cols=_SITE_MAX_COLS)
        # cells set their own ticks; the base single-axes smart-tick pass would
        # only touch cell 0 and make it differ from the rest.
        super().__init__(np.arange(self.n_cells), labels=labels, smart_ticks=False, **kwargs)
        self.site_axes: list = []
        self._cell_interactions: list = []
        self.focus_ax = None
        self._focused: int | None = None
        self._focus_tools = None

    def _create_axes(self):
        axes = create_axes_grid(
            self.fig, self.nrows, self.ncols,
            cell_px=_SITE_CELL_PX, col_gap_px=_SITE_COL_GAP_PX,
            row_gap_px=_SITE_ROW_GAP_PX, margins_px=_SITE_GRID_MARGINS_PX,
        )
        self.axes = axes
        return axes[0]

    def init_core(self) -> None:
        self.site_axes = []
        for k, ax in enumerate(self.axes):
            if k >= self.n_cells:
                ax.set_visible(False)
                continue
            self.site_axes.append(ax)
            self.cell_renderer.draw(ax, k, detail=False)
            if (k + self.ncols) < self.n_cells:        # hide x tick labels off the bottom row
                ax.set_xticklabels([])

    def update_core(self) -> None:
        # A snapshot plot; a fresh acquisition rebuilds via show().
        pass

    def _apply_title(self) -> None:
        labels = self.labels
        self.fig.text(0.5, 0.012, str(labels[0]), ha="center", va="bottom", fontsize=axis_label_fontsize())
        self.fig.text(0.008, 0.5, str(labels[1]) if len(labels) > 1 else "Shots", ha="left", va="center",
                      rotation="vertical", fontsize=axis_label_fontsize())
        apply_title(self.fig, self.title)

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type=self.plot_type)

    def _attach_interactions(self) -> None:
        # Every cell gets the SAME selector bundle a standalone plot does (area +
        # cross + zoom + draggable threshold), filtered by event.inaxes.
        self._cell_interactions = []
        for k, ax in enumerate(self.site_axes):
            area = AreaSelector(ax)
            cross = CrossSelector(ax)
            zoom = ZoomPan(ax, area_selector=area)
            line = self.cell_renderer.threshold_line(k)
            drag = DragVLine(line, self._make_threshold_cb(k), ax) if line is not None else None
            self._cell_interactions.append(InteractionBundle(area=area, cross=cross, zoom=zoom, drag=drag, axdis=ax))
        if self._cell_interactions:
            self.tools = self._cell_interactions[0]
            self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag
        # Pin to the FIGURE so the selectors outlive the returned handle (a notebook
        # keeps the figure, not necessarily the object) -- otherwise "no selector".
        self.fig._zlc_tools = self.tools
        self.fig._zlc_grid_tools = self._cell_interactions
        # Focus-zoom: double-click a cell to enlarge it; double-click / Esc to return.
        self._click_cid = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._key_cid = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def interaction_handles(self) -> list:
        out: list = []
        for bundle in self._cell_interactions:
            out.extend(h for h in (bundle.zoom, bundle.drag, bundle.area, bundle.cross) if h is not None)
        return out

    def _make_threshold_cb(self, k: int):
        return lambda x: self.cell_renderer.on_threshold_drag(k, float(x))

    # ----------------------------------------------------------- focus-zoom
    def _on_click(self, event) -> None:
        # Only a LEFT double-click toggles focus.  Some backends report the scroll
        # wheel / middle button as button 2/4/5 press pairs with dblclick=True, which
        # otherwise kicked us out of the enlarged view while scrolling.
        if not getattr(event, "dblclick", False) or getattr(event, "button", 1) != 1:
            return
        if self._focused is None:
            if event.inaxes in self.site_axes:
                self.focus(self.site_axes.index(event.inaxes))
        else:
            self.unfocus()

    def _set_grid_areas_active(self, active: bool) -> None:
        """Clear and (de)activate every grid cell's area selector.  Clearing wipes
        the residual selection box the focus double-click leaves on a cell (it would
        otherwise reappear when the grid returns); deactivating stops hidden cells
        reacting while focused."""
        for bundle in self._cell_interactions:
            area = getattr(bundle, "area", None)
            if area is None:
                continue
            try:
                area.clear()
                area.selector.set_active(active)
            except Exception:
                pass

    def _on_key(self, event) -> None:
        if getattr(event, "key", None) == "escape" and self._focused is not None:
            self.unfocus()

    def focus(self, k: int) -> None:
        """Enlarge cell ``k`` to fill the figure (see detail); call again or
        :meth:`unfocus` to return to the grid."""
        if self._focused is not None:
            self.unfocus()
        k = int(k)
        for ax in self.axes:
            ax.set_visible(False)
        # Hide the grid's OUTER labels/title while focused, so the enlarged cell's
        # own xlabel/ylabel/title do not collide / double up with them.
        self._hidden_texts = [t for t in self.fig.texts if t.get_visible()]
        for t in self._hidden_texts:
            t.set_visible(False)
        self._set_grid_areas_active(False)   # no residual box from the focus double-click
        # A clean, comfortably-margined single plot: room on the left for the counts
        # axis + ylabel, on the bottom for the xlabel, on top for the title.
        self.focus_ax = self.fig.add_axes([0.10, 0.13, 0.85, 0.77])
        line = self.cell_renderer.draw(self.focus_ax, k, detail=True)
        area = AreaSelector(self.focus_ax)
        cross = CrossSelector(self.focus_ax)
        zoom = ZoomPan(self.focus_ax, area_selector=area)
        drag = DragVLine(line, self._make_threshold_cb(k), self.focus_ax) if line is not None else None
        self._focus_tools = InteractionBundle(area=area, cross=cross, zoom=zoom, drag=drag, axdis=self.focus_ax)
        self.fig._zlc_tools = self._focus_tools
        self._focused = k
        self.draw()

    def unfocus(self) -> None:
        """Return from the enlarged single-cell view to the grid."""
        if self._focused is None:
            return
        for name in ("area", "cross", "zoom", "drag"):
            handler = getattr(self._focus_tools, name, None)
            if handler is not None and hasattr(handler, "destroy"):
                handler.destroy()
        if self.focus_ax is not None:
            self.fig.delaxes(self.focus_ax)
            self.focus_ax = None
        self._focus_tools = None
        self._focused = None
        for ax in self.site_axes:
            ax.set_visible(True)
        for t in getattr(self, "_hidden_texts", []):    # restore the grid's outer labels/title
            t.set_visible(True)
        self._hidden_texts = []
        self._set_grid_areas_active(True)               # re-arm grid area selectors, residue cleared
        self.fig._zlc_tools = self.tools
        self.draw()

    def to_data_figure(self):
        self.data_figure = _GridData(self)
        return self.data_figure
    # No ``save`` override here: ``BaseLivePlot.save`` -> ``to_data_figure().save`` ->
    # ``_GridData.save`` writes BOTH the png AND a matching ``.npz`` of every cell's data
    # (the DataFigure contract).  The saved png picks up the UNIFIED high save dpi from
    # ``rcParams['savefig.dpi'] = 600``, the same as every single-axes DataFigure.save.


class SiteHistogramGrid(GridPlot):
    """One histogram per site on an aligned grid (a :class:`GridPlot` whose cell is
    a :class:`HistogramCell`).  Kept as a named type for the readout site-grid; for
    other multi-panel plots build ``GridPlot(SomeCell(...))`` directly."""

    plot_type = "site_grid"

    def __init__(self, per_site_values, *, thresholds=None, occupied=None,
                 grid_shape: tuple[int, int] | None = None, bins: int = 36,
                 site_fidelities=None, labels: Sequence[str] = ("Signal", "Shots"), **kwargs):
        cell = HistogramCell(per_site_values, occupied=occupied, thresholds=thresholds,
                             site_fidelities=site_fidelities, bins=bins, labels=labels)
        super().__init__(cell, grid_shape=grid_shape, labels=labels, **kwargs)

    # the readout site-grid's public surface delegates to the histogram cell
    @property
    def n_sites(self) -> int:
        return self.n_cells

    @property
    def site_values(self) -> list:
        return self.cell_renderer.values

    @property
    def thresholds(self):
        return self.cell_renderer.thresholds

    @property
    def site_fidelities(self):
        return self.cell_renderer.site_fidelities

    @property
    def threshold_lines(self) -> list:
        return self.cell_renderer.threshold_lines

    @property
    def cell_edges(self):
        return self.cell_renderer.edges

    @property
    def cell_counts(self) -> list:
        return self.cell_renderer.cell_counts

    def classify(self, k: int, values=None) -> np.ndarray:
        return self.cell_renderer.classify(k, values)


def site_histogram_grid(
    per_site_values,
    *,
    thresholds=None,
    occupied=None,
    grid_shape: tuple[int, int] | None = None,
    bins: int = 36,
    site_fidelities=None,
    labels: Sequence[str] = ("Signal", "Shots"),
    title: str = "",
    display: bool = True,
) -> SiteHistogramGrid:
    """Plot one histogram per site on an aligned, non-overlapping grid (general N).

    Returns a :class:`SiteHistogramGrid` (a :class:`GridPlot` of histogram cells):
    a first-class plot, so it is interactive (per-cell selectors + a draggable
    per-site threshold, and **double-click a cell to enlarge it to see detail**)
    and every cell exposes a :class:`DataFigure` via ``grid.to_data_figure().cell(k)``.
    Geometry/gaps/colours/dpi/fonts are owned by the frontend (cells never overlap,
    nothing is cut off, all cells align)."""

    return SiteHistogramGrid(
        per_site_values,
        thresholds=thresholds,
        occupied=occupied,
        grid_shape=grid_shape,
        bins=bins,
        site_fidelities=site_fidelities,
        labels=labels,
        title=title,
    ).show(display=display)


def site_psf_grid(
    images,
    *,
    labels: Sequence[str] = ("x (px)", "y (px)"),
    title: str = "",
    display: bool = True,
    **kwargs,
):
    """Plot one image per site on an aligned, non-overlapping grid (general N).

    Each cell is a site's PSF weight kernel (an :class:`ImageCell`), so the operator
    SEES the real asymmetric, non-Gaussian spot the readout weights by -- the
    calibration's PSF *shape* output, drawn as a FIRST-CLASS plot rather than
    hand-rolled matplotlib.  Returns a :class:`GridPlot` (of :class:`ImageCell`): it is
    interactive (per-cell zoom + **double-click a cell to enlarge it**) and saves a png
    + matching ``.npz`` of the kernels via the same contract as every plot.  Geometry/
    gaps/colours/dpi/fonts are owned by the frontend (cells never overlap / cut off,
    all cells align)."""

    return GridPlot(
        ImageCell(images, labels=labels), labels=labels, title=title, **kwargs
    ).show(display=display)


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
    _spec: "FigureSpec | None" = None,
    **kwargs,
):
    """Create a static or live notebook plot from the same array contract.

    ``data_x`` is ``(N, coord_dim)`` and ``data_y`` is ``(N, channel_dim)``.
    ``coord_dim == 1`` creates a 1D line plot and ``coord_dim == 2`` creates a
    2D scan image. ``kind="hist"`` treats ``data_x`` as the values array. With
    ``update="watch"``, the returned object starts a frontend timer and refreshes
    from the same shared arrays while acquisition code mutates them.

    Figure geometry/dpi/margins are owned by the frontend, not arguments here
    (``_spec`` is the internal channel used by ``panel_plot``); see the
    package-level design contract in ``frontend/__init__.py``.
    """
    _reject_sealed_kwargs(kwargs)
    if _spec is not None:
        kwargs = {**kwargs, "spec": _spec}
    normalized_kind = _normalize_kind(kind)
    should_watch = _is_watch_update(update)

    # The kind -> class dispatch is a SINGLE lookup in the PLOT_KINDS table (no if/elif
    # ladder).  The handful of kinds with a per-kind input convention keep their tiny
    # special-case (hist reads data_x as the values array; pulse is data_x-only + rejects
    # watch; 2d validates the sealed `square`; monitor's show_dist toggle picks the bare
    # vs side-distribution sibling) -- everything else is ``cls(x, y, labels=..).show()``.
    if normalized_kind == "hist":
        values = data_x if data_y is None else data_y
        labels = tuple(labels or ("Counts", "Shots", "Population"))
        plotter = PLOT_KIND_BY_KEY["hist"].cls(values, labels=labels, **kwargs).show(display=display)
    elif normalized_kind == "pulse":
        if should_watch:
            raise ValueError("pulse plots are static timing diagrams; update='watch' is not supported.")
        labels = tuple(labels or ("Time (s)", "", "State"))
        plotter = PLOT_KIND_BY_KEY["pulse"].cls(data_x, labels=labels, **kwargs).show(display=display)
    else:
        x = _as_data_x(data_x)
        y = _as_data_y(data_y, len(x))
        if normalized_kind == "auto":
            normalized_kind = "2d" if x.shape[1] == 2 else "1d"
        labels = tuple(labels or ("X", "Y", "Z"))
        spec = PLOT_KIND_BY_KEY.get(normalized_kind)
        if spec is None or spec.key in ("hist", "pulse"):
            raise ValueError("kind must be auto, 1d, 2d, monitor, hist, sites, or pulse.")
        cls = spec.cls
        if normalized_kind == "2d":
            if "square" in kwargs:
                square = kwargs.pop("square")
                if square is not True:
                    raise ValueError("frontend.plot 2D figures are always square; call Live2DDis directly for internal non-square experiments.")
            kwargs = {**kwargs, "square": True}
        elif normalized_kind == "monitor":
            # ONE rolling-trace kind; the side distribution is a show_dist toggle (default on).
            # LiveLiveDis (with the side histogram, the table's default class) vs LiveLive (bare)
            # -- both keep their own plot_type string ("live-distribution"/"live") so saved
            # layouts still resolve.
            show_dist = bool(kwargs.pop("show_dist", True))
            cls = spec.cls if show_dist else LiveLive
        plotter = cls(x, y, labels=labels, **kwargs).show(display=display)

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


def load(path, *, kind: str | None = "auto", display: bool = True):
    """Reopen a ``.npz`` saved by :meth:`DataFigure.save` as a static, refittable
    figure -- the read-back counterpart of save.

    The saved payload carries the original ``data_x``/``data_y`` plus an ``info``
    dict (labels, unit, name).  This rebuilds the figure through the SAME
    :func:`plot` renderer (so the reloaded figure looks identical to the live
    one), restores the recorded unit/name, and returns the :class:`DataFigure`
    handle -- so an overnight scan saved to disk can be reopened weeks later to
    re-fit, change units, or re-save without any hardware or session.

    ``kind`` defaults to ``"auto"`` (1d/2d inferred from ``data_x`` shape, the
    usual scan case); pass an explicit kind to reopen a ``hist``/``sites`` save.
    """
    data = np.load(str(path), allow_pickle=True)
    if "data_x" not in data.files or "data_y" not in data.files:
        raise ValueError(f"{path} is not a DataFigure save (missing data_x/data_y).")
    data_x = data["data_x"]
    data_y = data["data_y"]
    info = data["info"].item() if "info" in data.files else {}
    labels = info.get("labels")
    plotter = plot(data_x, data_y, kind=kind, labels=labels, update=False,
                   display=display, data_figure=True)
    df = plotter.data_figure if plotter.data_figure is not None else plotter.to_data_figure()
    # Restore provenance so unit conversion + a re-save round-trip identically.
    df.info = {**df.info, **info}
    if info.get("name"):
        df.name = info["name"]
    if info.get("unit"):
        df.unit = info["unit"]
        df.unit_original = info["unit"]
    return df


__all__ = [
    "BaseLivePlot",
    "PlotKind",
    "PLOT_KINDS",
    "PLOT_KIND_BY_KEY",
    "HistogramFigure",
    "Live1D",
    "Live2DDis",
    "LiveLive",
    "LiveLiveDis",
    "LiveSiteMap",
    "PANEL_DISPLAY_SCALE",
    "PANEL_SIZES",
    "panel_display_size",
    "PulseSequenceFigure",
    "GridPlot",
    "GridCell",
    "HistogramCell",
    "SiteHistogramGrid",
    "panel_plot",
    "panel_plot_spec",
    "panel_size_cells",
    "plot",
    "site_histogram_grid",
    "site_ring_radius",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_marker",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
]
