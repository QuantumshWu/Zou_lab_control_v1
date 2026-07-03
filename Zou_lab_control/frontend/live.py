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
from matplotlib.ticker import FixedLocator, Formatter, FuncFormatter, LogLocator, MaxNLocator, NullLocator, ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

from ._validate import _positive_float
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
    HIST_FILL_ALPHA,
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
# The pulse RENDER lives here (the plot layer owns every plot kind's rendering): building the timeline
# figure from a PulseTableState, folding the analog DAC buses into their own rows and shading the scanned
# regions.  ORANGE / fluent_font_size are the two GUI tokens the scan-region highlight reuses (imported
# here, not re-declared -- single source); qt_fluent does not import live, so there is no cycle.
from .qt_fluent import ORANGE, fluent_font_size


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


def _nanmean_gap_safe(a, axis):
    """``np.nanmean`` that treats an ALL-NaN slice as an intended GAP (result NaN), not an error: a
    not-yet-measured LIVE scan point / an empty grid cell / an unfilled site averages to NaN BY INTENT.
    numpy flags that via TWO channels -- the 0/0 it computes (``errstate``) and a separate ``warnings``
    message ("Mean of empty slice") that ``errstate`` does NOT catch -- so silence exactly that benign
    pair here.  The ONE gap-safe average every repeat-collapse / decimation / site-pool shares, so a
    partly-filled live plot never spams the console with a warning per empty cell."""
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def reduce_repeat(raw, mode: str = "average", *, core_ndim=None, hist=False):
    """Collapse a raw block over its LEADING (repeat) axis O0 for display.  Works for ANY trailing
    shape -- a 1-D scan's ``(repeat, points, dim)``, a 2-D scan's ``(repeat, n0*n1, dim)``, a camera's
    ``(repeat, 1, H, W)``, a clean occupancy ``(repeat, 1, n_sites)`` -- because the repeat axis is always
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
      samples into one 1-D set so a histogram bins all shots together.
    * ``create``  -> keep EVERY repeat-with-data as its own column.  For a TRACE the core's leading axis
      is the x-points, so ``(R, points, dim) -> (points, R*dim)`` draws one line per repeat (confocal's
      "create"); for an image ``(R, 1, H, W)`` each repeat's whole core flattens to a column.

    ``hist=True`` reduces for a DISTRIBUTION (the kind decides this, #iron-law): a histogram's core is
    ALL samples -- it has NO x-axis -- so EVERY non-pool reduction is flattened to one 1-D sample set
    (the dis bins all of it), and ``create`` keeps each repeat's WHOLE flattened core as its own column
    ``(n_samples, R)`` = one histogram per repeat.  This is why a trace and a histogram, whose blocks can
    be the SAME ndim (``(R, points, dim)`` vs ``(R, 1, n_sites)``), need the kind to pick the layout."""
    a = np.asarray(raw, dtype=float)
    if not _has_repeat_axis(a, core_ndim):              # not a repeat block -> leave as-is (a hist with no
        return a.reshape(-1) if hist else a             # repeat axis is just ONE sample set -> flatten)
    has = np.isfinite(a).any(axis=tuple(range(1, a.ndim)))   # which repeat slices hold any data
    idx = np.flatnonzero(has)
    if mode == "create":
        cols = idx if idx.size else np.array([0])
        if hist:                                              # a histogram's core is ALL samples (no x-axis):
            return a[cols].reshape(len(cols), -1).T           #   each repeat's whole core -> a column (n_samples, R)
        if a.ndim == 2:                                       # (R, points) reduced scan -> (points, R) lines
            return a[cols].T
        if a.ndim == 3:                                       # (R, points, dim) scan -> (points, R*dim)
            return np.concatenate([a[r] for r in cols], axis=1)   #   confocal: repeat-major dim-minor columns
        # ndim >= 4: an image block (a camera's (R, 1, H, W)) -- create is ORTHOGONAL to the data axes:
        # flatten each repeat's core to a column so there is ONE trace per repeat (NOT per image row).
        return a[cols].reshape(len(cols), -1).T               # (prod(core), R) = x=pixel index, R lines
    if mode == "pool":                                    # histogram: flatten ALL repeats' samples (no reduce)
        return (a[idx] if idx.size else a[:1]).reshape(-1)
    if mode == "add":
        out = np.nansum(a, axis=0)
        return out.reshape(-1) if hist else out
    if mode in ("replace", "roll"):
        out = a[idx[-1]] if idx.size else a[0]
        return out.reshape(-1) if hist else out
    out = _nanmean_gap_safe(a, 0)      # 'average': an all-NaN (not-yet-measured) cell -> NaN gap, silently
    return out.reshape(-1) if hist else out


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
_PULSE_Y_PAD_BOTTOM = 0.62           # gap below the bottom channel row (row units)
_PULSE_Y_PAD_TOP = 0.38              # gap above the top channel row (row units)
# The pulse plot's LEFT margin: the ONLY margin that differs from a stock panel, because a pulse
# row is labelled by a CHANNEL NAME on the y axis ("DA bus 5", "Trap AOM") -- wider than a scan's
# 4-5 digit tick -- so it needs a touch more left room than PANEL_MARGINS_PX's 110 or the longest
# channel name clips.  R/B/T are the SAME as a stock panel (see panel_margins_px), so a pulse card
# lines up with every other kind; only this one value is pulse-specific (folded into the single
# margin source below, NOT a separate margin tuple).
_PULSE_LEFT_PX = 122

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

# Owned geometry for the per-site histogram grid (site_histogram_grid).  The grid fills the SAME total
# data region every other panel kind uses (``panel_plot_spec(size).data_px``) and SUBDIVIDES it into
# cells with these inter-cell gaps -- so a grid's data box equals a single-axes panel's of the same size
# (create_axes_grid(data_px=...) derives the per-cell size to fill the region), and a size change truly
# rescales the cells, not just the padding.  The gaps scale with the size preset like the cells do.  The
# OUTER margins around the whole grid come from the ONE panel margin source (:func:`panel_margins_px`) --
# NOT a bespoke grid margin tuple -- so a grid's TOTAL figure px (data box + margins) equals a single-axes
# panel's of the same size to the pixel (the same three-region L|data|R / B|data|T layout), and the grid
# thumbnail's padding matches every other kind's card instead of looking visibly different (#5).
_SITE_MAX_COLS = 7                        # column cap (pulse-style width cap); extra sites wrap to rows
#: Hard cap on the number of grid cells: beyond this the per-tick thumbnail update (and the first
#: matplotlib layout) freezes the UI.  ONE source -- the grid factory raises past it and the console
#: greys out facet axes that would exceed it.
MAX_GRID_CELLS = 80
#: Cell-text rule (fixed by the operator): a grid squeezed into a panel size SMALLER than its
#: recommended default (:func:`optimal_grid_size`; "smaller" = EITHER side below the
#: recommendation) uses HALF the standard 2x2 cell text; every other size uses the plot kind's
#: STANDARD style sizes unscaled.  Exactly two tiers -- a binary squeezed-or-not rule, no
#: continuous shrink.
def _cell_font_scale(size: str, recommended: str) -> float:
    rows, cols = panel_size_cells(size)
    rec_rows, rec_cols = panel_size_cells(recommended)
    return 0.5 if (rows < rec_rows or cols < rec_cols) else 1.0


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


def _dist_count_xlim(n) -> int:
    """The side distribution-band count(x)-axis upper limit, from the per-bin counts ``n`` -- the ONE
    rule every band shares (Live2DDis / LiveSiteMap / LiveLiveDis, INIT and UPDATE alike) so the bar
    column never touches the right edge and the five call sites cannot drift into two headroom rules.

    Small headroom (``peak * 1.5``, floored to ``peak + 5`` for tiny peaks, never below 10) keeps the
    tallest bar clear of the axis edge per the no-cutoff/no-overlap layout rule -- an appearance-neutral
    single source, NOT the old no-headroom ``peak + 5`` that left the top bar flush against the frame."""
    arr = np.asarray(n, dtype=float)
    peak = float(np.nanmax(arr)) if arr.size and np.isfinite(arr).any() else 0.0
    return max(10, int(max(peak + 5, peak * 1.5)))


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

    def _build_distribution_band(self, image_artist, values, *, n_bins, guide_minmax=None):
        """Build the side clim-distribution band on ``self.axdis`` (which assumes ``self.ylim_min`` /
        ``self.ylim_max`` and the image clim are already set): the intensity histogram PolyCollection,
        the x-axis ticks, the two draggable clim lines (coloured by the image cmap ends) + their
        DragHLine, and -- only when ``guide_minmax=(y_min, y_max)`` is given -- the faint guide lines
        at the data min/max drawn BEFORE the clim lines.  The ONE place Live2DDis + LiveSiteMap build
        the band, in the SAME artist order, so the poly / guide / clim-line wiring cannot drift (#C1)."""
        self.n_bins = int(n_bins)
        self.n, self.bins = np.histogram(values, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
        self.verts = np.empty((self.n_bins, 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="horizontal")
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"])
        self.axdis.add_collection(self.poly)
        self.axdis.set_xlim(0, _dist_count_xlim(self.n))
        self.axdis.xaxis.set_major_locator(MaxNLocator(nbins=1, prune="lower"))
        self.axdis.xaxis.set_major_formatter(ScalarFormatter())
        self.axdis.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True, labeltop=False)
        self.axdis.tick_params(axis="y", which="both", left=True, right=False, labelleft=False, labelright=False)
        if guide_minmax is not None:
            y_min, y_max = guide_minmax
            self.line_min = self.axdis.axhline(y_min, color=PALETTE["guide"], linewidth=small_fontsize() / 2, alpha=0.3)
            self.line_max = self.axdis.axhline(y_max, color=PALETTE["guide"], linewidth=small_fontsize() / 2, alpha=0.3)
        cmap = image_artist.get_cmap()
        self.line_l = self.axdis.axhline(self.ylim_min, color=cmap(0.0), linewidth=small_fontsize() / 2)
        self.line_h = self.axdis.axhline(self.ylim_max, color=cmap(0.95), linewidth=small_fontsize() / 2)
        self.drag = DragHLine(self.line_l, self.line_h, self.update_clim, self.axdis)

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
        # The SOURCE this plot's data came from (a SignalHub + the producing logic node + the wired
        # signal names), stamped by :meth:`bind_source` when the plot is created FROM live signals (the
        # console panel, a notebook hub-bound plot).  Its presence makes ``.save()`` a RICH save -- it
        # captures ``info['signals']`` (raw hub blocks + roles) + ``info['provenance']`` (device state)
        # through the ONE core capture, identical to the console panel's Save.  ``None`` (an array plot
        # with no producing node) => ``.save()`` degrades to the basic figure+npz save.  The binding is a
        # DATA fact (WHERE the data came from), not art/geometry, so it lives here, off the sealed set.
        self._figure_source = None

    def _infer_points_done(self) -> int:
        finite = np.isfinite(self.data_y[:, 0])
        return int(np.count_nonzero(finite))

    def show(self, *, display: bool = True):
        """Initialize and optionally display the figure."""
        apply_style({"figure.dpi": self.spec.dpi})
        if self.fig is None:
            self.fig = new_figure(spec=self.spec)
        else:
            # Build onto a caller-supplied figure (e.g. the grid's own canvas when a cell is focused into a
            # standalone panel): clear it and lay out this plot's fixed-pixel box, so the plot OWNS the figure
            # exactly as a fresh one would -- no external-axes overlay, just the standard geometry path.
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
            self._initial_draw()
        return self

    def _initial_draw(self) -> None:
        """The undisplayed (``display=False``) build's synchronous first render.  A subclass whose
        real consumers ALWAYS re-render (the grid: the Qt panel canvas draws on construction,
        savefig renders itself) overrides this to skip the wasted rasterisation."""
        self.fig.canvas.draw()

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

    def set_selectors_active(self, active: bool) -> None:
        """Enable/disable every mouse selector on this plot IN PLACE (a DATA-level behaviour
        gate, no art/geometry): the tools stay constructed -- their artists, thresholds and
        selections keep their state -- only their EVENT HANDLING is gated, so flipping the gate
        never rebuilds a figure.  The consumer is the console's Monitor "Selectors" switch: a
        dashboard panel builds WITH its selector layer and parks it inactive (display-only, the
        historical default), and the switch re-arms it live.  Safe no-op on a plot with no tools
        (``interactions=False``, e.g. the grid thumbnails).

        Two tool families, one rule each:

        * ``AreaSelector`` wraps a matplotlib widget with a REAL active gate -- delegate to
          ``selector.set_active`` (the same surface ``DragVLine._set_area_active`` already uses).
        * ``CrossSelector`` / ``ZoomPan`` / ``DragVLine`` / ``DragHLine`` self-manage their
          canvas callbacks under the UNIFORM ``cid_*``/``on_*`` names (selectors.py convention),
          so ONE wiring table disconnects/reconnects exactly the events each tool owns --
          idempotent via the ``_zlc_selectors_off`` marker, so repeated calls never double-connect.

        The enumeration reuses :meth:`interaction_handles` (the grid override contributes every
        per-cell bundle) plus the histogram's ``threshold_draggers`` (its extra draggable lines
        live outside the bundle), so no tool escapes the gate."""
        active = bool(active)
        wiring = (
            ("cid_scroll", "scroll_event", "on_scroll"),
            ("cid_press", "button_press_event", "on_press"),
            ("cid_motion", "motion_notify_event", "on_motion"),
            ("cid_release", "button_release_event", "on_release"),
        )
        handles = list(self.interaction_handles())
        handles.extend(d for d in getattr(self, "threshold_draggers", ()) if d is not None)
        for handle in handles:
            selector = getattr(handle, "selector", None)
            if selector is not None and hasattr(selector, "set_active"):
                selector.set_active(active)
                continue
            if bool(getattr(handle, "_zlc_selectors_off", False)) == (not active):
                continue                     # already in the requested state (idempotent)
            canvas = handle.ax.figure.canvas
            for cid_attr, event_name, cb_attr in wiring:
                cid = getattr(handle, cid_attr, None)
                if cid is None:
                    continue                 # this tool never owned that event
                if active:
                    setattr(handle, cid_attr, canvas.mpl_connect(event_name, getattr(handle, cb_attr)))
                else:
                    canvas.mpl_disconnect(cid)
            handle._zlc_selectors_off = not active

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

    def current_lims(self) -> tuple[float, float]:
        """The (lo, hi) the view shows NOW -- the ``ylim_min``/``ylim_max`` pair EVERY relim path
        maintains (1D/monitor y-axis, 2D/sitemap colour-limit alike).  Read when the Setting flips
        relim to ``fixed``: fixed FREEZES the current view, so its lo/hi seed from here instead of
        a blind 0..1 default that would blank the plot (bars/image all outside the pinned range).
        HistogramFigure overrides (its count axis is set directly on the axes, not via ylim_*)."""
        return float(self.ylim_min), float(self.ylim_max)

    def apply_relim_now(self) -> None:
        """Force the axis limits to the current relim_mode and redraw immediately.
        Called when the mode toggles (Setting combo) -- a switch must take effect
        now, not wait for the next data frame, and must bypass relim's dead-band.
        1D/monitor: rescale the y-axis.  Live2DDis overrides for the colour limit."""
        self.relim(force=True)
        if self.ax is not None:
            lo, hi = self.ylim_min, self.ylim_max
            if self.ax.get_yscale() == "log" and lo <= 0:
                # a log axis rejects a non-positive lower limit -- floor it the way
                # HistogramFigure._apply_count_yscale already does (#log-ylim).
                lo = 0.5 if hi > 0.5 else hi * 0.1
            self.ax.set_ylim(lo, hi)

    def apply_param(self, key: str, value) -> bool:
        """Apply a DISPLAY-ONLY plot knob IN PLACE on the existing axes (no figure/canvas rebuild) and
        return True if handled.  The RELIM FAMILY (``relim`` / ``fixed_lo`` / ``fixed_hi``) is handled
        HERE for every kind -- it maps onto the one relim path every plot already maintains
        (``relim_mode`` + :meth:`apply_relim_now`), so a lim edit from ANY surface (Setting popup,
        Edit tab, a grid's focused cell) lands in place and never falls back to a teardown/rebuild.
        Subclass overrides handle their own knobs and DELEGATE unknown keys back here (super()), so
        the family works uniformly.  Anything else returns False -> the caller may rebuild."""
        if str(key) == "relim":
            if str(value) == "fixed" and getattr(self, "relim_mode", "") != "fixed":
                # fixed FREEZES the current view: seed lo/hi from what the plot shows NOW, never
                # a stale/default 0..1 pair (which empties a counts axis / camera clim).  Seeded
                # HERE so EVERY surface -- Setting, Edit tab, a grid's focused cell -- agrees.
                self.fixed_lo, self.fixed_hi = self.current_lims()
            self.relim_mode = str(value)
            self.apply_relim_now()
            self.draw()
            return True
        if str(key) in ("fixed_lo", "fixed_hi"):
            setattr(self, str(key), float(value))
            if getattr(self, "relim_mode", "") == "fixed":
                self.apply_relim_now()
                self.draw()
            return True
        return False

    def after_plot(self):
        """Create and attach a DataFigure handle."""
        return self.to_data_figure()

    def to_data_figure(self):
        from .data_figure import DataFigure

        self.data_figure = DataFigure(self)
        return self.data_figure

    def bind_source(self, hub, node, *, inputs, resolve_node=None, session=None):
        """Stamp WHERE this plot's data came from so ``.save()`` writes the RICH npz (``info['signals']``
        + ``info['provenance']``), identical to the console panel's Save.

        ``hub`` is the :class:`SignalHub` the data was published to; ``node`` is the producing logic node
        of the panel's first input; ``inputs`` are the wired signal names; ``resolve_node`` (optional)
        maps a signal name -> its producing node for the UPSTREAM provenance walk (a processor panel
        hoists its source measurement's devices); ``session`` is the fallback device-snapshot source when
        no node produced the data.  Returns ``self`` so a create can chain
        ``plot(...).bind_source(hub, node, inputs=...)``.  The capture itself is done LAZILY at save time
        (through the frontend-neutral core), so binding never touches the hub eagerly."""
        self._figure_source = {"hub": hub, "node": node, "inputs": list(inputs or []),
                               "resolve_node": resolve_node, "session": session}
        if self.data_figure is not None:
            self.data_figure._figure_source = self._figure_source
        return self

    def save(self, path: str = "", **kwargs):
        df = self.to_data_figure()
        # Carry the source binding onto the DataFigure so its save is the RICH save (the plotter is the
        # object a notebook holds; the DataFigure is where save lives -- the binding rides across).
        df._figure_source = self._figure_source
        return df.save(path, **kwargs)


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
        # Grow a Line2D per data column that appeared AFTER init (notebook update_point(mode='create')
        # pads data_y with a new column per repeat).  self.lines is frozen at show() time, so without this
        # the new columns would never be drawn -- the SAME on-demand artist pattern HistogramFigure uses
        # (_ensure_overlays).  When no column was added the while-loop is a no-op (byte-identical old path).
        while len(self.lines) < self.data_y.shape[1]:
            (ln,) = self.ax.plot([], [])
            self.lines.append(ln)
        self._color_lines()
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
        self.axdis.set_xlim(0, _dist_count_xlim(self.n))
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
        self.axdis.set_xlim(0, _dist_count_xlim(self.n))
        self._update_gauss_fit()

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type="1D", x_array=self.data_x[:, 0], y_array=self.data_y, axdis=self.axdis)


def _image_axes_px_budget(ax) -> tuple[int, int]:
    """The image axes' size in DESIGN pixels (position fraction x figure design px) -- the
    upper bound of what the screen can show of it, renderer-free.  Used as the display
    decimation budget: an array decimated to >= this budget loses nothing on screen."""
    pos = ax.get_position()
    fig = ax.figure
    w_in, h_in = fig.get_size_inches()
    return (max(1, int(round(pos.width * w_in * DESIGN_DPI))),
            max(1, int(round(pos.height * h_in * DESIGN_DPI))))


def _decimate_image_view(grid, extent, xlim, ylim, budget) -> tuple[np.ndarray, tuple]:
    """Slice the current VIEW out of a full-resolution image and block-mean it down to the
    display budget: ``(array, extent)`` ready for ``imshow``.

    matplotlib happily resamples a full camera frame (1920x1200+) on EVERY draw -- ~90 ms
    per tick, at the wrong layer.  The display layer instead keeps the full array and shows
    an area-averaged view: each displayed pixel is the mean of an integer block (photometric,
    the same filtering a camera viewer's smooth scaling applies -- and it IMPROVES SNR,
    unlike subsampling), recomputed per new frame / view change (a ~2 ms numpy pass).
    Zooming re-slices from the full array, so detail comes back progressively until the
    factor hits 1 (a plain no-copy view).  Arrays already within budget pass through.

    ``extent``/``xlim``/``ylim`` follow imshow's edge conventions and may be inverted on
    either axis (the 2D plots draw y top-to-bottom); the returned extent maps EXACTLY the
    pixels kept.  The block factor is floor-based, so the result is never smaller than the
    budget (never softer than the screen)."""
    grid = np.asarray(grid)
    ny, nx = grid.shape[:2]
    ex0, ex1 = float(extent[0]), float(extent[1])       # column edges: col 0 -> nx
    ey1, ey0 = float(extent[2]), float(extent[3])       # imshow extent = (left, right, bottom, top)

    def _index_window(lo, hi, e0, e1, n):
        # data coords -> half-open index window on an axis whose edge j sits at
        # e0 + j*(e1-e0)/n; handles inverted extents and inverted view limits.
        step = (e1 - e0) / n
        a, b = (lo - e0) / step, (hi - e0) / step
        a, b = (a, b) if a <= b else (b, a)
        return max(0, min(int(np.floor(a)), n - 1)), max(1, min(int(np.ceil(b)), n))

    jx0, jx1 = _index_window(float(xlim[0]), float(xlim[1]), ex0, ex1, nx)
    jy0, jy1 = _index_window(float(ylim[0]), float(ylim[1]), ey0, ey1, ny)
    sub = grid[jy0:jy1, jx0:jx1]
    sy, sx = sub.shape[:2]
    bx, by = (budget, budget) if np.isscalar(budget) else (int(budget[0]), int(budget[1]))
    fx = max(1, sx // max(1, bx))
    fy = max(1, sy // max(1, by))
    if fx == 1 and fy == 1:
        small, ky, kx = sub, sy, sx
    else:
        ky, kx = (sy // fy) * fy, (sx // fx) * fx
        small = sub[:ky, :kx].reshape(ky // fy, fy, kx // fx, fx).mean(axis=(1, 3))
    sxp = (ex1 - ex0) / nx
    syp = (ey1 - ey0) / ny
    small_extent = (ex0 + jx0 * sxp, ex0 + (jx0 + kx) * sxp,
                    ey0 + (jy0 + ky) * syp, ey0 + jy0 * syp)
    return small, small_extent


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

    def _is_regular_raster(self) -> bool:
        """True when the points are a COMPLETE row-major grid -- a camera frame or any dense 2-D
        map: exactly one point per cell, ascending in (y, then x).  Detected ONCE (``data_x`` is
        fixed for the plot's life) so :meth:`fill_grid` reshapes in O(1) per frame instead of an
        O(N) searchsorted scatter -- a full 1920x1200 frame then updates with no per-tick cost,
        which is why the image reaches the plot at NATIVE resolution (no upstream point-count cap)."""
        ny, nx = self.data_shape
        if self.data_x.shape[0] != ny * nx:
            return False
        ix = np.searchsorted(self.x_array, self.data_x[:, 0])
        iy = np.searchsorted(self.y_array, self.data_x[:, 1])
        return bool(np.array_equal(iy * nx + ix, np.arange(ny * nx)))

    def fill_grid(self) -> np.ndarray:
        # A complete regular raster (a camera frame / dense 2-D map) IS the row-major image, so
        # reshape in O(1) -- no per-frame scatter (regularity detected once in init_core).
        if self._regular_raster:
            return self.data_y[:, 0].reshape(self.data_shape)
        # General scatter -- a sparse / unordered scan grid; same semantics as the old loop:
        # searchsorted indices, out-of-range points dropped, LAST point wins on a duplicate.
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
        self._regular_raster = self._is_regular_raster()   # cache once: data_x is fixed for the plot's life
        self.grid = self.fill_grid()
        try:
            cmap = matplotlib.colormaps[self.cmap].copy()
        except Exception:
            cmap = plt.get_cmap(self.cmap).copy()
        cmap.set_bad(self.bad_color)
        dx = 0.5 * (self.x_array[-1] - self.x_array[0]) / len(self.x_array) if len(self.x_array) > 1 else 0.5
        dy = 0.5 * (self.y_array[-1] - self.y_array[0]) / len(self.y_array) if len(self.y_array) > 1 else 0.5
        self.extent = [self.x_array[0] - dx, self.x_array[-1] + dx, self.y_array[-1] + dy, self.y_array[0] - dy]
        # "auto" interpolation: nearest when the grid is smaller than the screen (scan grids keep
        # crisp pixels), filtered when larger (a decimated camera frame never shows aliasing).
        # The budget is read PER REFRESH: the split axes reach their final position only at the
        # first layout pass, so a cached value would freeze the pre-layout (full-figure) size.
        small, small_ext = _decimate_image_view(self.grid, self.extent,
                                                (self.extent[0], self.extent[1]),
                                                (self.extent[2], self.extent[3]),
                                                _image_axes_px_budget(self.ax))
        self.image = self.ax.imshow(small, cmap=cmap, extent=small_ext, interpolation="antialiased")
        self._in_display_refresh = False
        self.ax.callbacks.connect("xlim_changed", lambda _ax: self._refresh_display_image())
        self.ax.callbacks.connect("ylim_changed", lambda _ax: self._refresh_display_image())
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

    #: cap on the pixel count fed to the side-distribution histogram + colour-range min/max.  A
    #: dense image's distribution needs only a REPRESENTATIVE sample -- a strided subset is
    #: statistically identical, and the full 2.3M-pixel isfinite+histogram was the live-2D per-frame
    #: bottleneck (~22 ms), NOT the display decimation (~3 ms).  A scan (< cap) passes through whole.
    _DIST_SAMPLE_CAP = 200_000

    def _distribution_values(self):
        vals = self.data_y[: self.points_done, 0]
        if vals.size > self._DIST_SAMPLE_CAP:
            vals = vals[:: vals.size // self._DIST_SAMPLE_CAP + 1]   # even stride -> unbiased sample
        return vals[np.isfinite(vals)]

    def _init_distribution(self) -> None:
        vals = self._distribution_values()
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
        # shared band (#C1): histogram + faint guide lines (this kind has them) + draggable clim
        # lines; the colorbar end ticks are set by update_core (cax.set_yticks/labels).
        self._build_distribution_band(self.image, vals if vals.size else [0],
                                      n_bins=int(max(8, min(max(self.points_total, 1) // 4, 50))),
                                      guide_minmax=(y_min, y_max))

    def _attach_interactions(self) -> None:
        self.tools = attach_interaction(self.ax, drag=self.drag, axdis=self.axdis, cax=self.cax)
        self.area, self.cross, self.zoom, self.drag = self.tools.area, self.tools.cross, self.tools.zoom, self.tools.drag

    def _refresh_display_image(self) -> None:
        """Re-slice + re-decimate the displayed view from the full-resolution grid -- on every
        new frame and every xlim/ylim change (zooming back INTO the full array is what brings
        the detail back).  Guarded: set_extent inside the refresh must not recurse through the
        lim callbacks that triggered it."""
        if self._in_display_refresh:
            return
        self._in_display_refresh = True
        try:
            small, small_ext = _decimate_image_view(self.grid, self.extent, self.ax.get_xlim(),
                                                    self.ax.get_ylim(), _image_axes_px_budget(self.ax))
            self.image.set_array(small)
            self.image.set_extent(small_ext)
        finally:
            self._in_display_refresh = False

    def update_clim(self) -> None:
        self.image.set_clim(float(self.line_l.get_ydata()[0]), float(self.line_h.get_ydata()[0]))

    def apply_relim_now(self) -> None:
        # 2D analogue of the y-axis rescale: recompute the colour limit for the
        # current relim_mode and re-apply it + the draggable clim lines now, so a
        # normal<->tight switch in Setting visibly re-maps the colorbar.
        vals = self._distribution_values()
        if not vals.size:
            return
        y_min = float(np.nanmin(vals))
        y_max = float(np.nanmax(vals))
        self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)
        self.image.set_clim(self.ylim_min, self.ylim_max)
        self.axdis.set_ylim(self.ylim_min, self.ylim_max)
        self.line_l.set_ydata([self.ylim_min, self.ylim_min])
        self.line_h.set_ydata([self.ylim_max, self.ylim_max])

    def apply_param(self, key: str, value) -> bool:
        """The colormap is DISPLAY-ONLY: swap it on the EXISTING image in place (the colorbar hangs off
        the same artist, so it recolours with it) and redraw -- no figure/canvas rebuild, so a cmap change
        never flashes/resizes the panel (#dis-resize) and a FOCUSED grid cell's enlarged 2d view recolours
        while STAYING zoomed (never bouncing through a rebuild).  Returns True if handled."""
        if str(key) in ("cmap", "colorset"):
            self.image.set_cmap(str(value))
            self.draw()
            return True
        return super().apply_param(key, value)   # the relim family lands in place via the base class

    def update_core(self) -> None:
        self.grid = self.fill_grid()
        self._refresh_display_image()
        vals = self._distribution_values()
        if vals.size:
            y_min = float(np.nanmin(vals))
            y_max = float(np.nanmax(vals))
            self.ylim_min, self.ylim_max = self._mode_target(y_min, y_max)   # normal=from 0, tight=bracket
            self.axdis.set_ylim(self.ylim_min, self.ylim_max)
            self.n, self.bins = np.histogram(vals, bins=self.n_bins, range=(self.ylim_min, self.ylim_max))
            _update_verts(self.bins, self.n, self.verts, mode="horizontal")
            self.poly.set_verts(self.verts)
            self.axdis.set_xlim(0, _dist_count_xlim(self.n))
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
            self._refresh_display_image()
            # The colour limit + side histogram are owned by the distribution band (per the
            # lim-mode), not pinned to this frame's min/max here -- update_core re-clims + refreshes.
            if draw:
                self.draw()
        else:
            self.background = arr               # shape changed: rebuilt by the host

    def _refresh_display_image(self) -> None:
        """Same display refresh as :meth:`Live2DDis._refresh_display_image`, over the sitemap's
        full-resolution camera frame (``self.background``)."""
        if self._bg_image is None or self._in_display_refresh:
            return
        self._in_display_refresh = True
        try:
            small, small_ext = _decimate_image_view(self.background, self.extent, self.ax.get_xlim(),
                                                    self.ax.get_ylim(), _image_axes_px_budget(self.ax))
            self._bg_image.set_array(small)
            self._bg_image.set_extent(small_ext)
        finally:
            self._in_display_refresh = False

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
            # same display policy as Live2DDis: the FULL frame stays in self.background, the
            # artist shows a block-mean view within the axes' pixel budget, re-sliced on zoom
            # (budget read per refresh -- see Live2DDis.init_core).
            small, small_ext = _decimate_image_view(self.background, extent,
                                                    (extent[0], extent[1]), (extent[2], extent[3]),
                                                    _image_axes_px_budget(self.ax))
            self._bg_image = self.ax.imshow(small, cmap=self.cmap, extent=small_ext, interpolation="antialiased")
            self._in_display_refresh = False
            self.ax.callbacks.connect("xlim_changed", lambda _ax: self._refresh_display_image())
            self.ax.callbacks.connect("ylim_changed", lambda _ax: self._refresh_display_image())
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
        # shared band (#C1): the sitemap's frame-intensity band has NO guide lines (guide_minmax=None).
        self._build_distribution_band(self._bg_image, vals, n_bins=40, guide_minmax=None)

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
        self.axdis.set_xlim(0, _dist_count_xlim(self.n))
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
PANEL_SIZES = ("1x2", "2x2", "4x2", "1x4", "2x4", "4x4", "4x8", "8x4", "8x8")
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


def panel_margins_px(kind: str = "default") -> tuple[int, int, int, int]:
    """The ONE panel margin source: ``(left, right, bottom, top)`` px around a panel's data box.

    EVERY panel kind reads its outer margins from HERE -- there is no second margin tuple anywhere
    (a pulse plot used to carry its own ``_PULSE_MARGINS_PX``, which let the two drift).  The default
    is the stock ``PANEL_MARGINS_PX`` (confocal's left, a reserved title slot on top).  A ``pulse``
    panel differs in ONE value only: a wider LEFT margin (``_PULSE_LEFT_PX``) so a row's channel-NAME
    y label ("DA bus 5") is not clipped; R/B/T stay identical to a stock panel, so a pulse card lines
    up with every other kind and its x label / title never clip either."""

    left, right, bottom, top = PANEL_MARGINS_PX
    if str(kind) == "pulse":
        left = _PULSE_LEFT_PX
    return (left, right, bottom, top)


def panel_plot_spec(size: str = "2x2", *, kind: str = "default") -> FigureSpec:
    """FigureSpec for a dashboard panel: the stock plot DATA region scaled in half-units, with the
    ONE margin source (:func:`panel_margins_px`) -- so the data box is IDENTICAL for every panel kind
    at a given ``size`` (it scales with the size preset, never with content), and only the pulse left
    margin differs (channel-name room).  This is the SINGLE geometry source every kind's card reads --
    2D / hist / 1D / monitor / sites AND pulse / grid -- so changing ``size`` truly rescales the data
    region, not just the padding."""

    rows, cols = panel_size_cells(size)
    return FigureSpec(
        data_px=(cols * PANEL_UNIT_PX[1], rows * PANEL_UNIT_PX[0]),
        margins_px=panel_margins_px(kind))


# The per-site grid's TOTAL data region is ``panel_plot_spec(size).data_px`` -- the SAME data region every
# other panel kind uses at that size -- and ``create_axes_grid(data_px=...)`` SUBDIVIDES it into cells (the
# per-cell size is DERIVED to fill the region after the gaps).  So a grid's data box equals a single-axes
# panel's of the same size (part A: grid_data_px == panel_plot_spec(size).data_px), and a size change truly
# rescales the cells, not just the padding.  The gaps scale with the size preset (cols/2 wide, rows/2 tall,
# relative to the 2x2 baseline) so a bigger grid's inter-cell spacing grows with the cells.  The OUTER
# margins are :func:`panel_margins_px` -- the SAME margins every single-axes panel uses -- so the grid's
# TOTAL figure px (data box + margins) EQUALS a same-size single-axes panel's to the pixel, and the grid
# thumbnail's outer padding matches every other kind's card (#5).  ONE size rule, ONE margin source.
def _site_grid_geometry(size: str, recommended: str) -> tuple[tuple[int, int], int, int, tuple[int, int, int, int], float]:
    """The grid's layout numbers for a panel-size preset: total data region (== every other kind's at
    this size), the inter-cell gaps, the outer margins, and the CELL-TEXT scale.

    The text scale compares the panel size against the grid's RECOMMENDED default size with exactly
    TWO tiers (the operator's fixed rule, :func:`_cell_font_scale`): squeezed below the
    recommendation halves the standard 2x2 cell text, everything else keeps the plot kind's
    standard style size.  The gaps then DERIVE from that text -- exactly the band
    the cell title (+ a row's x tick labels) needs, nothing more -- so every spare pixel goes into
    the CELLS (resize fills the area; the only hard constraint is that no cell title is cut)."""
    data_px = panel_plot_spec(size).data_px               # total data region == every other kind's at this size
    font_scale = _cell_font_scale(size, recommended)
    # All grid geometry is in DESIGN px (300 dpi -- style.DESIGN_DPI), so pt -> px is DPI/72.
    # The row corridor carries exactly ONE cell-title line (the edge-label policy strips x tick
    # labels off every non-bottom row, and the bottom row's labels live in the B margin), so its
    # budget is the title's own height (~= its point size, probe-pinned) + a few px of air.
    # Tighter clips the title into the cell above; looser only steals area from the cells.
    px_per_pt = DESIGN_DPI / 72.0
    row_gap = round(small_fontsize() * font_scale * px_per_pt) + 4
    # the column corridor carries NO text at all (the centered-single-tick policy keeps x labels
    # inside their cell -- zero spill, probe-verified -- and y labels live in the L margin): it
    # is a pure visual separator.
    col_gap = max(6, round(10 * font_scale))
    l, r, b, t = panel_margins_px()                       # SAME outer margins as a single-axes panel -> total px matches
    return data_px, col_gap, row_gap, (l, r, b, t), font_scale


# The cell-count THRESHOLDS a grid dimension crosses to earn a double-size half-unit, PER AXIS: an axis
# with UP TO its threshold opens compact (2 half-units = a stock 2x2 side), MORE opens double (4 = a 4x4
# side).  The column threshold is one higher than the row threshold because a cell is WIDER than tall
# (PANEL_UNIT_PX), so more columns fit legibly before the grid needs the double width.  Each axis is
# judged INDEPENDENTLY (a tall 5x3 grid opens 4x2, never forced square).  The ONE source for the "when
# does a grid need a bigger side" rule, shared with the contract test.
_GRID_DOUBLE_ROWS_THRESHOLD = 4    # > 4 rows    -> the double height (4 half-units)
_GRID_DOUBLE_COLS_THRESHOLD = 5    # > 5 columns -> the double width  (4 half-units)


def optimal_grid_size(nrows: int, ncols: int) -> str:
    """The ``PANEL_SIZES`` preset a per-site grid of ``nrows`` x ``ncols`` cells defaults to -- the grid
    counterpart of :func:`optimal_pulse_size`.  EACH axis is judged INDEPENDENTLY against its own
    boundary (:data:`_GRID_DOUBLE_ROWS_THRESHOLD` rows / :data:`_GRID_DOUBLE_COLS_THRESHOLD` columns):
    up to the boundary gets the compact half-unit (2), more gets the double (4) -- rows and columns are
    never coupled, so a tall 5x3 grid opens ``4x2`` (tall), a wide 3x6 opens ``2x4`` (wide), a 5x7 opens
    ``4x4``, and a 4x5 stays the compact ``2x2``.  Every {2,4}^2 combination is a real ``PANEL_SIZES``
    preset, so the result maps directly (no nearest-preset snap).  ONE default-size source: the grid
    factories and the reopen/seed paths all call it, so a saved grid with no recorded ``panel_size``
    and a fresh grid pick the same preset."""
    rows_half = 4 if int(nrows) > _GRID_DOUBLE_ROWS_THRESHOLD else 2
    cols_half = 4 if int(ncols) > _GRID_DOUBLE_COLS_THRESHOLD else 2
    return f"{rows_half}x{cols_half}"


def recommended_grid_size(n_cells: int) -> str:
    """The ``PANEL_SIZES`` preset an ``n_cells``-cell facet grid defaults to, from the CELL COUNT alone.
    Lays the cells out with the SAME rule :class:`GridPlot` uses (:func:`grid_shape_for` + the shared
    ``_SITE_MAX_COLS`` column cap) then maps that shape through the ONE :func:`optimal_grid_size` -- so a
    caller that only knows how many cells it will have (a task's mid-run facet panel, sized BEFORE the
    grid object exists) picks the SAME preset the grid itself would (a few cells -> ``2x2``, never an
    over-large magic size)."""
    nrows, ncols = grid_shape_for(int(n_cells), max_cols=_SITE_MAX_COLS)
    return optimal_grid_size(nrows, ncols)


# The readability floor a pulse timeline needs in the size-preset DATA region: enough px PER ROW that a
# channel name is not squashed, and enough px PER PERIOD that the periods do not blur into one band.
# These REPLACE the old content-driven inches (_PULSE_X_PERIODS_PER_BLOCK / auto-height): instead of
# a bespoke figure size, a busy pulse now picks a BIGGER size PRESET (optimal_pulse_size), and the data
# region rescales with that preset like every other kind.  Owned tokens (ART), never a per-call knob.
_PULSE_ROW_MIN_PX = 26       # min data-region height per pulse row (channel + analog) for a legible name
_PULSE_PERIOD_MIN_PX = 46    # min data-region width per period so periods stay distinct


def optimal_pulse_size(channel_count: int, period_count: int) -> str:
    """The SMALLEST ``PANEL_SIZES`` preset whose size-driven DATA region holds ``channel_count`` rows and
    ``period_count`` periods legibly (>= ``_PULSE_ROW_MIN_PX`` per row, >= ``_PULSE_PERIOD_MIN_PX`` per
    period), else the LARGEST preset.  The ONE source for a pulse preview's / loaded pulse panel's default
    size: a busy pulse (many channels / periods) defaults to a bigger preset so nothing overlaps, and the
    data region scales with that preset -- the size preset CARRIES the content density (the old
    content-driven inches did).  The user can still override the size afterwards.

    NOTE the ceiling: the largest preset is ``4x4`` (960x720 data px), holding ~27 rows / ~20 periods at
    the floor; an EXTREME pulse beyond that still returns ``4x4`` (the biggest available) and scrolls in
    its card -- there is no larger preset to grow into."""

    rows_needed = max(1, int(channel_count))
    periods_needed = max(1, int(period_count))
    # PANEL_SIZES ordered smallest-area first, so the first fit is the smallest sufficient preset.
    by_area = sorted(PANEL_SIZES, key=lambda s: (lambda rc: rc[0] * rc[1])(panel_size_cells(s)))
    best = by_area[-1]                       # largest, the fallback when nothing fits
    for size in by_area:
        data_w, data_h = panel_plot_spec(size, kind="pulse").data_px
        if data_h >= rows_needed * _PULSE_ROW_MIN_PX and data_w >= periods_needed * _PULSE_PERIOD_MIN_PX:
            return size
    return best


def panel_display_size(size: str = "2x2") -> tuple[int, int]:
    """On-screen (logical px) canvas size of a panel of ``size`` -- what a host
    reserves for the canvas built by ``qt_canvas.panel_canvas``."""

    spec = panel_plot_spec(size)
    left, right, bottom, top = spec.margins_px
    width = spec.data_px[0] + left + right
    height = spec.data_px[1] + bottom + top
    return (round(width * PANEL_DISPLAY_SCALE), round(height * PANEL_DISPLAY_SCALE))


def coerce_panel_value(kind, value, *, structure=None, params=None, repeat_mode="last"):
    """Turn a hub signal VALUE (+ its producing node's declared ``structure``) into the array a
    panel of ``kind`` consumes -- the ONE place that owns each plot kind's INPUT contract (2d wants
    an image, hist bins samples, monitor a scalar, sites one value/site).  This lives WITH the plots,
    NOT in task_console: the console only GATHERS the inputs (value, structure, params, repeat mode)
    and calls here, so the wiring layer holds zero per-kind reshape logic.  ``structure`` is the
    producing node's ``{"data_shape", "grid_shape"}`` mapping, or None when unknown (a custom
    expression / raw array) -- then shape is INFERRED from the value, never assumed."""
    params = params or {}
    # A pulse panel's value is a STRUCTURED object (a sequence / PulseTableState) with no array shape.
    if kind == "pulse":
        return value
    arr = np.asarray(value, dtype=float)
    if kind == "grid":
        # A FACET grid consumes the bound block RAW (repeat, points, *data_dim); facet_cells slices it.
        return arr
    # Reshape is decided by the DATA dimensionality (#H3o, NOT a size threshold): data_shape 2-D -> an
    # image; 1-D -> multiple series (lines); grid_shape un-flattens a 2-D scan's points to a map.
    st = structure
    ds = tuple(st["data_shape"]) if st else ()
    gs = tuple(st["grid_shape"]) if st else ()
    if kind == "1d":
        # y-vs-index by default; only an explicit xy=True panel reads (N, 2) as an x-y curve.
        if params.get("xy") and arr.ndim == 2 and arr.shape[1] == 2 and arr.shape[0] >= 1:
            return arr
        a = np.squeeze(arr)
        if len(ds) == 2:
            # 2-D DATA: 'create' arrived as (pixels, repeat) -> one line per repeat; else flatten to one trace.
            if repeat_mode == "create" and a.ndim == 2:
                return a
            flat = a.reshape(-1)
            if flat.size < 1:
                raise ValueError("panel value is empty")
            return flat
        # 1-D DATA / unknown: a 2-D (points, lines) stays 2-D (one line per series); a 1-D value is one line.
        if a.ndim == 2:
            return a
        flat = a.reshape(-1)
        if flat.size < 1:
            raise ValueError("panel value is empty")
        return flat
    if kind == "2d":
        # Image by structure: a 2-D data core IS the image; a 2-D scan un-flattens to grid_shape; an
        # unknown-structure 2-D value is shown as-is.  NATIVE resolution -- the DISPLAY layer (Live2DDis)
        # block-means only for the screen budget, reversibly (zoom re-slices the full array to 1:1).
        a = np.asarray(arr, dtype=float)
        if len(ds) == 2:
            img = np.squeeze(a)
            if img.ndim > 2:
                img = img.reshape(img.shape[0], img.shape[1], -1)[:, :, 0]
        elif gs and int(np.prod(gs)) == int(a.size):
            img = a.reshape(tuple(int(n) for n in gs))
        elif st is None and np.squeeze(a).ndim == 2:
            img = np.squeeze(a)
        else:
            raise ValueError(
                f"2D panel needs an image (a 2-D data_shape or a grid_shape); got value shape "
                f"{arr.shape}, data_shape={ds}, grid_shape={gs}")
        if img.ndim != 2 or min(img.shape) < 2:
            raise ValueError(f"2D panel needs a 2D image value (got shape {arr.shape})")
        return img
    if kind == "hist":
        # Bin SAMPLES: 'create' per-repeat COLUMNS (n_samples, R) stay 2-D (one histogram per repeat);
        # anything else flattens to ONE histogram -- bin exactly what the source gives.
        return arr if (arr.ndim == 2 and arr.shape[1] > 1) else arr.reshape(-1)
    if kind == "monitor":
        flat = arr.reshape(-1)
        if flat.size != 1:
            raise ValueError(f"rolling-trace panel needs a scalar value (got shape {arr.shape})")
        return float(flat[0])
    flat = arr.reshape(-1)
    if flat.size < 1:
        raise ValueError("panel value is empty")
    if kind == "sites":
        # ONE value per site: 'create' concatenates repeats -> collapse back to n_sites (mean over repeats).
        if len(ds) == 1 and int(ds[0]) > 0 and flat.size != int(ds[0]) and flat.size % int(ds[0]) == 0:
            flat = _nanmean_gap_safe(flat.reshape(-1, int(ds[0])), 0)   # unfilled sites -> NaN, silently
        if flat.size > 4096:
            raise ValueError(f"site-map panel needs one value per site (got {flat.size} values)")
    return flat


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


def pulse_plot_spec(size: str = "2x2") -> FigureSpec:
    """FigureSpec for a pulse timeline: the SAME size-driven panel geometry every other kind uses --
    ``panel_plot_spec(size, kind="pulse")``.  The data region scales with the ``size`` PRESET (not with
    channel / period COUNT: a busy pulse instead picks a bigger preset via :func:`optimal_pulse_size`),
    and the margins come from the ONE :func:`panel_margins_px` source (only the pulse LEFT margin differs,
    for channel-name room).  Kept as the pulse geometry entry point named in the sealed-API contract, but
    it is now a thin size-preset wrapper -- there is no per-call content geometry knob left (the old
    ``_PULSE_MARGINS_PX`` / content-driven width blocks are gone; the size preset carries the density)."""

    return panel_plot_spec(size, kind="pulse")


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


# ============================================================ pulse render (state -> figure)
# The pulse plot kind, like every other kind, has its RENDER here in the plot layer -- building the
# timeline FIGURE from a PulseTableState (digital channels + analog DAC-bus rows + repeat brackets +
# scanned-region shading).  ``pulse_gui`` (the editor app) and ``task_console`` / ``data_figure`` all
# CONSUME this; none of them owns the render.  A contract test (test_pulse_render_single_source) pins
# that these live here and that the consumers do not re-implement or import a pulse renderer from the GUI.


def bus_signed_bounds(max_value: int) -> "tuple[int, int]":
    """SIGNED user range of a bus whose full-scale code is ``max_value`` (2^B - 1):
    (-2^(B-1), +2^(B-1)-1).  0 = true 0 V on the offset-binary DAC driver.  Shared by the pulse render
    (bus-row min/max) and the editor's value validation -- one source."""
    half = (int(max_value) + 1) >> 1
    return (-half, half - 1)


def bus_display_label(name: str) -> str:
    """Human label for a DAC bus name (underscores -> spaces).  Shared by the render + the editor."""
    return str(name).replace("_", " ")


# In-module aliases so the render code below reads the same names it used in the editor module.
_bus_signed_bounds = bus_signed_bounds
_bus_display_label = bus_display_label


def _bus_has_signal(state, bus_name: str) -> bool:
    """True if a DAC bus carries a real signal: any edge/ramp entry (even to 0 V -- the user explicitly
    drives it), OR a scanned (slot-referenced) value.  An untouched / all-hold bus rests at 0 V and is
    treated as "off" (hidden when "show off rows" is off)."""
    for slot in state.scan_slots:
        if getattr(slot, "kind", "") == "dac" and slot.dac_bus == bus_name:
            return True
    for entry in state.analog_bus_modes.get(bus_name, []):
        if str(entry.get("mode", "hold")).lower() != "hold":
            return True
    if bus_name not in state.analog_bus_modes:
        for index in range(len(state.periods)):
            try:
                if state.bus_value(index, bus_name) != 0:
                    return True
            except Exception:
                pass
    return False


def analog_bus_traces(state, *, include_always_off: bool = True) -> "tuple[list[dict[str, object]], set[str]]":
    """Fold each DAC bus into ONE signed analog trace row (starts + step values), and return
    ``(traces, folded_members)`` -- the member bit-channels that must NOT leak into the digital plot as
    their own rows.  A pure function of the ``PulseTableState`` (the render layer owns this)."""
    from ..neutral_atom.timing.pulse_table import analog_bus_ticks, _analog_bus_value_at_tick

    buses = state.bus_channels()
    if not buses:
        return [], set()
    starts_steps = [0]
    slots = state.reference_slots()
    for period in state.periods:
        starts_steps.append(starts_steps[-1] + period.duration_steps(slots=slots, time_step_ns=state.time_step_ns))
    traces: list[dict[str, object]] = []
    folded_members: set[str] = set()
    for bus_name, members in buses.items():
        # A recognized DAC bus is ALWAYS folded -- its bit channels must never leak into the digital plot
        # as individual rows (the bug where DA appeared as 10 separate channels).
        folded_members.update(members)
        # "Show off rows" off -> hide idle (all-zero / hold) DAC buses, like an always-off TTL channel.
        if not include_always_off and not _bus_has_signal(state, bus_name):
            continue
        # Resolve scanned (slot-referenced) DAC values to their reference value so the preview shows a
        # concrete trace instead of crashing on int("s2").
        plan = state._resolved_bus_plan(bus_name, slots)
        bus_ticks = analog_bus_ticks(plan, starts_steps)
        lo_signed, hi_signed = _bus_signed_bounds((1 << len(members)) - 1)
        traces.append(
            {
                "name": bus_name,
                "label": _bus_display_label(bus_name),
                "members": list(members),
                # values are SIGNED (0 = true 0 V); min/max bound the signed range so the plotter can
                # place the 0 V baseline mid-row and show negatives.
                "min": lo_signed,
                "max": hi_signed,
                "starts": [tick * state.time_step_ns * 1e-9 for tick in bus_ticks],
                # looping=True: the GUI runs the pulse forever, so the preview shows the STEADY STATE the
                # loop converges to (a looping [ramp V, hold V] reads FLAT V, matching the bench output).
                "values": [
                    _analog_bus_value_at_tick(plan, starts_steps, tick, looping=True)
                    for tick in bus_ticks[:-1]
                ],
            }
        )
    return traces, folded_members


def pulse_drawn_rows(state, *, include_always_off: bool) -> tuple[list[str], list[Mapping[str, Any]]]:
    """The exact rows :func:`build_pulse_preview_plot` DRAWS for ``state`` -- ``(digital_channels,
    analog_traces)`` -- with the SAME folding / clk-hiding / show-off-rows rules the render applies.  The
    ONE source for "how many rows does this pulse draw", so a caller sizing the plot (``optimal_pulse_size``
    for the preview default, the loaded-panel default, ...) counts EXACTLY the rows the figure will have,
    never a re-derived approximation that could drift from the render."""
    sequence = state.to_sequence(expand_repeat=False)
    analog_traces, folded_members = analog_bus_traces(state, include_always_off=include_always_off)
    clk_set = set(getattr(state, "clk_channels", []))
    universe = [c for c in state.channels if c not in folded_members and c not in clk_set]
    channels = [c for c in pulse_plot_channels(sequence, channels=universe,
                                               include_always_off=include_always_off)
                if c not in folded_members]
    return channels, list(analog_traces)


def default_pulse_size(state, *, include_always_off: bool) -> str:
    """The default size preset for ``state`` -- :func:`optimal_pulse_size` of the rows it DRAWS
    (:func:`pulse_drawn_rows`) and its period count.  The ONE default-size source shared by the preview
    dropdown, the Save recipe and the reopened panel, so all three agree on a pulse's natural size."""
    channels, traces = pulse_drawn_rows(state, include_always_off=include_always_off)
    return optimal_pulse_size(len(channels) + len(traces), len(getattr(state, "periods", []) or []))


def build_pulse_preview_plot(state, *, include_always_off: bool, size: str | None = None,
                             interactions: bool = True):
    """Render the pretty pulse timeline for ``state`` -- the SINGLE source of the pulse preview figure.

    A pure function of the ``PulseTableState`` (no editor widgets): it builds the sequence, folds the
    analog DAC buses into their own rows, draws the digital channels + analog traces + repeat brackets,
    and shades the scanned regions.  Returns ``(plotter, channels, repeat_notation)``.

    ``size`` is one of ``PANEL_SIZES`` -- the data region scales with it exactly like every other panel
    kind (a busy pulse defaults to a bigger preset).  ``None`` picks :func:`optimal_pulse_size` for the
    figure's own row / period counts (the single default source shared by the preview + the loaded panel).
    ``interactions=False`` builds a display-only figure with NO selectors (the read-only Monitor card).

    This is the ONE renderer the plot layer owns and every consumer shares: (a) the editor's live preview
    / Save-Figure image (``pulse_gui``), (b) a reopened ``figure_recipe`` (``data_figure.SavedFigure``),
    and (c) a seeded ``kind="pulse"`` console panel (``task_console.PanelCard``) -- so all three draw the
    identical faithful figure (same channels / analog traces / brackets), never a flattened 1-D trace."""
    sequence = state.to_sequence(expand_repeat=False)
    repeat = pulse_repeat_notation(state)
    repeat_brackets = pulse_repeat_markers(state)
    # Channel delays can push edges past the period-table end; the repeat markers are computed from period
    # starts only, so without this the delayed tail renders OUTSIDE the ×∞ loop bracket (reads as
    # unphysical).  Extend the infinite-loop bracket to enclose the whole drawn sequence.
    seq_end = float(getattr(sequence, "duration", 0.0) or 0.0)
    if seq_end > 0.0:
        repeat_brackets = [
            (start, max(stop, seq_end), label) if "∞" in str(label) else (start, stop, label)
            for (start, stop, label) in repeat_brackets
        ]
    analog_traces, folded_members = analog_bus_traces(state, include_always_off=include_always_off)
    clk_set = set(getattr(state, "clk_channels", []))
    digital_channel_universe = [
        channel for channel in state.channels
        if channel not in folded_members and channel not in clk_set   # clk channels aren't engine-driven
    ]
    channels = pulse_plot_channels(
        sequence,
        channels=digital_channel_universe,
        include_always_off=include_always_off,
    )
    # Defensive: a folded DAC bit must never appear as its own digital row.
    channels = [channel for channel in channels if channel not in folded_members]
    # The default size preset carries the content density: a busy pulse (many rows / periods) picks a
    # bigger preset so nothing overlaps, and the data region scales with it (the single default source
    # shared with the loaded panel).  An explicit ``size`` (the preview's size dropdown) overrides it.
    if size is None:
        size = optimal_pulse_size(len(channels) + len(analog_traces), len(state.periods))
    plotter = plot(
        sequence,
        kind="pulse",
        channels=channels,
        include_always_off=True,
        repeat_notation=repeat,
        repeat_brackets=repeat_brackets,
        channel_labels={channel: state.label_for(channel) for channel in channels},
        analog_traces=analog_traces,
        title=state.name,
        show_names=True,
        display=False,
        data_figure=False,
        interactions=interactions,
        size=size,
    )
    bus_rows = [str(trace.get("name")) for trace in analog_traces]
    annotate_pulse_variable_regions(plotter, state, channels, bus_rows=bus_rows)
    return plotter, channels, repeat


def annotate_pulse_variable_regions(plotter, state, channels=None, *, bus_rows=None) -> None:
    """Shade the time spans affected by each scan slot in transparent orange.

    Only ``duration`` and ``dac`` are scannable (a channel delay is a fixed value):
    - a scanned *duration* spans its whole period across all channels;
    - a scanned *DAC* value spans its period on its own analog-bus row.

    Each slot carries its 1-based number exactly once, placed on the row it affects (the DAC's bus row),
    so several scanned DAC buses get distinct, non-overlapping labels instead of piling up."""
    if not hasattr(plotter, "ax"):
        return
    slots = state.reference_slots()
    starts_ns = [0.0]
    for period in state.periods:
        starts_ns.append(starts_ns[-1] + period.duration_ns(slots=slots, time_step_ns=state.time_step_ns))
    # Use the plotter's ACTUAL row geometry (data coordinates) so highlights land exactly on the channels
    # they belong to -- guessing y from a row count drifts and puts delay bands on the wrong channel.
    ax = plotter.ax
    base_y = dict(getattr(plotter, "_pulse_baseline_y", {}) or {})
    analog_y = dict(getattr(plotter, "_analog_baseline_y", {}) or {})
    row_h = float(getattr(plotter, "_pulse_row_height", 0.64) or 0.64)
    if not base_y and not analog_y:
        return
    all_baselines = list(base_y.values()) + list(analog_y.values())
    area_bottom = min(all_baselines)
    area_top = max(all_baselines) + row_h          # top edge of the top channel
    ylim_top = float(ax.get_ylim()[1])
    x_lo, x_hi = ax.get_xlim()
    min_width = max((x_hi - x_lo) * 0.004, 1e-12)

    plotter.variable_region_artists = []
    plotter.variable_region_labels = []

    def add_band(x0: float, x1: float, y0: float, y1: float, alpha: float) -> None:
        if x1 < x0:
            x0, x1 = x1, x0
        if x1 - x0 < min_width:
            x1 = x0 + min_width
        patch = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=ORANGE, edgecolor="none",
                          alpha=alpha, linewidth=0.0, zorder=6, transform=ax.transData)
        ax.add_patch(patch)
        plotter.variable_region_artists.append(patch)

    def add_number(xc: float, yc: float, tag: str, va: str = "center") -> None:
        if not tag:
            return
        # Mimic the bound scan-dot badge: a filled orange circle with a white digit (same look as
        # FluentScanDot), keeping the small font size.
        text = ax.text(xc, yc, tag, transform=ax.transData, ha="center", va=va,
                       color="white", fontsize=max(2.6, float(fluent_font_size()) * 0.28),
                       fontweight="bold", clip_on=False, zorder=12,
                       bbox=dict(boxstyle="circle,pad=0.3", facecolor=ORANGE, edgecolor="none"))
        plotter.variable_region_labels.append(text)

    # Unified tint; the value highlight (DAC) is a touch stronger but same hue.
    BAND_ALPHA = 0.18
    bus_groups = state.bus_channels()
    for slot_index, slot in enumerate(state.scan_slots):
        tag = str(slot_index + 1)
        if slot.kind == "duration":
            pidx = int(slot.target) if slot.target.lstrip("-").isdigit() else -1
            if not (0 <= pidx < len(state.periods)):
                continue
            x0 = starts_ns[pidx] * 1e-9
            x1 = starts_ns[pidx + 1] * 1e-9
            # Band covers exactly the channel rows (top = top channel's top, never above it).  Number
            # sits in the headroom just above the top channel but below the title/bracket.
            add_band(x0, x1, area_bottom, area_top, BAND_ALPHA)
            label_y = min(area_top + row_h * 0.5, ylim_top - row_h * 0.2)
            add_number((x0 + x1) / 2, label_y, tag, va="center")
        elif slot.kind == "dac":
            bus = slot.dac_bus
            pidx = slot.dac_period
            if bus not in analog_y or not (0 <= pidx < len(state.periods)):
                continue
            x0 = starts_ns[pidx] * 1e-9
            x1 = starts_ns[pidx + 1] * 1e-9
            members = bus_groups.get(bus, [])
            lo_v, hi_v = _bus_signed_bounds((1 << max(1, len(members))) - 1)
            span_v = max(1, hi_v - lo_v)
            try:
                value = float(state.analog_bus_value_at_period_start(pidx, bus))
            except Exception:
                value = 0.0
            # Follow the DA LINE: highlight the trace segment at the SIGNED value's height over the
            # scanned period (0 V = mid-row, negatives below).
            vy = analog_y[bus] + row_h * min(1.0, max(0.0, (value - lo_v) / span_v))
            if x1 - x0 < min_width:
                x1 = x0 + min_width
            seg = ax.plot([x0, x1], [vy, vy], color=ORANGE, linewidth=3.0, alpha=0.9,
                          solid_capstyle="butt", zorder=8)[0]
            plotter.variable_region_artists.append(seg)
            # Number centred vertically in the bus row (only *duration* labels sit above the band;
            # delay and DAC labels live inside their row).
            add_number((x0 + x1) / 2, analog_y[bus] + row_h * 0.5, tag, va="center")


class PulseSequenceFigure(BaseLivePlot):
    """Filled-rectangle pulse timeline for sequencer/verilog inspection."""

    plot_type = "pulse"

    def apply_relim_now(self) -> None:
        """A pulse timeline has NO dependent-axis relim: its y axis IS the channel-row layout
        (set once by init_core), not a measured quantity.  The base implementation would relim
        against the dummy zero data_y and squash every channel row to (-0.1, 0.1) -- so the relim
        family's apply (pushed uniformly by the console's _apply_lim_to_plotter) is a no-op here."""

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
        size: str = "2x2",
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
            # The pulse timeline gets the SAME size-driven panel geometry every kind uses: the data
            # region scales with the ``size`` PRESET (the caller picks a bigger preset for a busy pulse
            # via optimal_pulse_size), NOT with channel / period count -- one geometry source, no bespoke
            # content inches.
            kwargs["spec"] = pulse_plot_spec(size)
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


#: The histogram-fit chooser's default -- "double" is the dark/bright readout convention.  ONE
#: source for the standalone dis, the grid's hist cells AND the console's PANEL_PARAMS spec, so the
#: Setting UI's default can never disagree with what the figure actually draws (the "fit says double
#: but the grid shows no fit until toggled" bug).
DEFAULT_HIST_FIT = "double"


def fit_histogram_curves(vals, edges, counts, mode: str):
    """The ONE histogram-fit rule: fit ``mode`` ("none" / "single" / "double") to a binned sample set
    and return the CURVE DATA + threshold -- consumed by :class:`HistogramFigure` (the standalone dis)
    AND by the grid's :class:`HistogramCell` thumbnails, so the two can never fit differently.

    Returns ``None`` when there is nothing to draw (mode "none", too few samples, or a genuinely
    failed fit), else ``{"x", "left", "right", "total", "threshold", "separated", "bimodal_popt",
    "single_popt"}`` (``left``/``right`` are None for a single-Gaussian outcome; ``threshold`` is set
    only when the two fitted peaks separate by >= 1.5 summed widths -- the honest-threshold gate)."""
    vals = np.asarray(vals, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    mode = str(mode)
    if mode == "none" or vals.size < 6 or np.ptp(vals) == 0:
        return None
    edges = np.asarray(edges, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2
    counts = np.asarray(counts, dtype=float)
    span = float(np.ptp(vals)) or 1.0
    x_fit = np.linspace(edges[0], edges[-1], 400)

    def _single():
        try:
            p0 = [max(float(np.max(counts)), 1.0), float(np.mean(vals)),
                  max(float(np.std(vals)), span / 40, 1e-9)]
            popt, _ = curve_fit(gaussian, centers, counts, p0=p0, maxfev=20000)
        except Exception:
            return None
        return {"x": x_fit, "left": None, "right": None, "total": gaussian(x_fit, *popt),
                "threshold": None, "separated": False, "bimodal_popt": None,
                "single_popt": (float(popt[0]), float(popt[1]), float(popt[2]))}

    if mode != "double":
        return _single()                     # "single": one Gaussian, no dark/bright split
    # Otsu split on the sorted samples (between-class variance), NOT the median -- seeds each side.
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

    # DOUBLE mode ALWAYS attempts the two-Gaussian decomposition -- the toggle drives the display
    # DIRECTLY, never a hidden auto-collapse based on how the data looks; only a genuine curve_fit
    # failure falls back to the single Gaussian (the "toggle does nothing" bug).
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
        return _single()
    if popt[1] > popt[4]:
        popt = np.array([popt[3], popt[4], popt[5], popt[0], popt[1], popt[2]], dtype=float)
    # Both peaks always DRAW in double mode; the threshold (and the fidelity stat upstream) is only
    # meaningful when the fitted means separate by >= 1.5 summed widths -- the same gate as before.
    fitted_sep = abs(popt[4] - popt[1]) / (abs(popt[2]) + abs(popt[5]) + 1e-12)
    separated = fitted_sep >= 1.5
    y0 = gaussian(x_fit, *popt[:3])
    y1 = gaussian(x_fit, *popt[3:])
    threshold = None
    lo, hi = float(popt[1]), float(popt[4])
    if separated and hi > lo:
        x_mid = np.linspace(lo, hi, 400)
        diff = np.abs(gaussian(x_mid, *popt[:3]) - gaussian(x_mid, *popt[3:]))
        threshold = float(x_mid[int(np.nanargmin(diff))])
    return {"x": x_fit, "left": y0, "right": y1, "total": y0 + y1, "threshold": threshold,
            "separated": separated, "bimodal_popt": popt, "single_popt": None}


class HistogramFigure(BaseLivePlot):
    """Neutral-atom-friendly histogram with threshold classification tools.

    Fidelity note (#1b): when fed the live ``counts`` signal this POOLS every site's readout into ONE
    histogram and reports the two-Gaussian fidelity about ONE global cut.  That number is EXPECTED to
    read a few tenths of a percent BELOW the readout calibration's reported fidelity -- not a bug, and
    the overlap math is single-sourced (``_readout_math``).  The calibration scores each site against
    its OWN threshold and averages (per-site-centred, ``operations/calibration_report``), so each
    per-site distribution is tight; pooling N sites that each have a different dark/bright mean widens
    both peaks, so a single global cut overlaps more.  The pooled number is the honest "all sites, one
    cut" figure; the calibration's is the per-site-cut figure -- two different (both correct) questions."""

    plot_type = "hist"

    def __init__(
        self,
        values,
        *,
        bins: int | Sequence[float] = 50,
        thresholds: Sequence[float] | None = None,
        labels: Sequence[str] = ("Counts", "Shots", "Population"),
        ylog: bool = False,
        fit: str = DEFAULT_HIST_FIT,    # fit chooser: "none" | "single" | "double" (ONE default source)
        **kwargs,
    ):
        # 'create' repeat mode delivers (n_samples, R): column 0 is the FIRST repeat (the FULL
        # treatment -- fill + fit + threshold + stats), columns 1.. draw the SAME FILLED histogram as a
        # no-repeat dis, just a different LINE_CYCLE colour + alpha so overlaps stay legible (NOT outlines
        # -- a repeat reads like an ordinary distribution, only the fit/threshold/text are the first's).
        self._overlay_polys: list = []
        self._overlay_counts: list = []
        self.values = self._split_columns(values)
        self.bins_arg = bins
        self.ylog = bool(ylog)          # log-scale the COUNT axis (reveals a sparse bright tail)
        # The fit is a DISPLAY chooser (none / single / double) driven by the toggle -- never an
        # auto-decision based on how the data happens to look.  It defaults to the two-Gaussian dark/
        # bright decomposition (the readout convention); "single" gives one Gaussian, "none" no fit.
        self._fit = str(fit)
        self._fit_separated = False     # whether the bimodal fit cleanly separated (gates the fidelity stat)
        # A threshold (cut line) is only MEANINGFUL when there are two populations to separate (#issue-2):
        # shown when the bimodal fit separated OR a threshold was supplied explicitly (a calibration's
        # per-site cut).  A pure single-Gaussian fit shows the FIT params + out-of-fit fraction instead.
        self._explicit_thr = bool(thresholds)
        self._has_threshold = self._explicit_thr
        self.single_popt = None         # (amp, mu, sigma) of the single-Gaussian fit, for the no-threshold stat
        self.thresholds = list(thresholds or [])
        # relim/fixed pins the VALUE (x) axis for a histogram (the count y-axis is always auto, #3);
        # default to "tight" but HONOR a relim_mode the panel passes (Setting/Edit "fixed" + lo/hi).
        kwargs.setdefault("relim_mode", "tight")
        super().__init__(np.arange(len(self.values)), self.values, labels=labels, **kwargs)

    def _split_columns(self, values):
        """Parse the bound value into (first-repeat samples, extra per-repeat columns).  A 2-D
        ``(n_samples, R>1)`` block ('create' mode) keeps each column as its own repeat -- column 0 is the
        primary histogram, columns 1.. become outline overlays.  Anything else is one flat sample set."""
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2 and arr.shape[1] > 1:
            self._extra_cols = [arr[:, j][np.isfinite(arr[:, j])] for j in range(1, arr.shape[1])]
            return np.asarray(arr[:, 0], dtype=float).reshape(-1)
        self._extra_cols = []
        return arr.reshape(-1)

    def _ensure_overlays(self, n_extra: int) -> None:
        """Keep exactly ``n_extra`` FILLED overlay histograms (one per non-first 'create' repeat) -- the
        SAME bar fill a no-repeat dis uses (PolyCollection), only a different LINE_CYCLE colour + alpha so
        the overlapping repeats stay legible.  A repeat reads like an ordinary distribution, not an outline."""
        while len(self._overlay_polys) < n_extra:
            j = len(self._overlay_polys)
            poly = PolyCollection(np.empty((0, 4, 2)),
                                  facecolors=LINE_CYCLE[(j + 1) % len(LINE_CYCLE)], alpha=HIST_FILL_ALPHA)
            self.ax.add_collection(poly)
            self._overlay_polys.append(poly)
        while len(self._overlay_polys) > n_extra:
            self._overlay_polys.pop().remove()

    def _refresh_overlays(self) -> None:
        """Re-bin each extra 'create' repeat on the CURRENT bins and refill its bars; cache the counts so
        the count-axis can include the overlays' peak (else a taller repeat would clip)."""
        self._ensure_overlays(len(self._extra_cols))
        self._overlay_counts = []
        for poly, col in zip(self._overlay_polys, self._extra_cols):
            c = col if col.size else np.array([0.0])
            counts, _ = np.histogram(c, bins=self.bins)
            verts = np.empty((len(counts), 4, 2), dtype=float)
            _update_verts(self.bins, counts, verts, mode="vertical")     # SAME bar geometry as the primary
            poly.set_verts(verts)
            self._overlay_counts.append(counts)

    def init_core(self) -> None:
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        vals = self.values[np.isfinite(self.values)]
        if vals.size == 0:
            vals = np.array([0.0])
        self.n, self.bins = np.histogram(vals, bins=self.bins_arg)
        self.verts = np.empty((len(self.n), 4, 2), dtype=float)
        _update_verts(self.bins, self.n, self.verts, mode="vertical")
        # The PRIMARY (repeat=1) histogram is drawn with the SAME owned bar opacity as the 'create'
        # overlays (style.HIST_FILL_ALPHA) -- a lone dis reads identically to one with overlays, and the
        # threshold/fit lines read through the bars -- never an opaque block beside translucent ones.
        self.poly = PolyCollection(self.verts, facecolors=PALETTE["hist_fill"], alpha=HIST_FILL_ALPHA)
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
            if self.interactions:                        # DRAGGABLE only on the Edit/notebook surface; a
                self.threshold_draggers.append(DragVLine(line, self._on_threshold_drag, self.ax))  # read-only Monitor card (interactions=False) keeps the line but no grab
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
        self._refresh_overlays()         # 'create' overlay histograms (empty unless in create mode)
        self._apply_value_xlim()
        self._apply_count_yscale()
        self._update_hist_stats()

    def update(self, values=None, *, data_y=None, points_done: int | None = None, repeat_cur: int | None = None, draw: bool = True):
        if values is None and data_y is not None:
            values = data_y
        if values is not None:
            self.values = self._split_columns(values)
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
        self._refresh_overlays()
        self._apply_value_xlim()
        self._apply_count_yscale()
        self._fit_bimodal()
        while len(self.threshold_lines) < len(self.thresholds):
            line = self.ax.axvline(self.thresholds[len(self.threshold_lines)], **threshold_line_kwargs())
            self.threshold_lines.append(line)
            if self.interactions:                        # gate the grab, not the line (#dis-threshold-disable)
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

    def _apply_value_xlim(self) -> None:
        """The VALUE (x) axis = the binning COORDINATE: always the natural bin span.  A histogram's
        relim/fixed acts on the COUNT (y) axis -- the measured quantity -- exactly as a 1D plot's
        relim acts on its SIGNAL (dependent) axis, not on its independent scan coordinate.  So the
        value axis simply frames the data's bins; tight/normal/fixed live on the count axis
        (:meth:`_apply_count_yscale`)."""
        self.ax.set_xlim(self.bins[0], self.bins[-1])

    def current_lims(self) -> tuple[float, float]:
        """The count (y) axis range the view shows NOW.  A histogram sets its count axis directly on
        the axes (:meth:`_apply_count_yscale`), not via the base ``ylim_min``/``ylim_max`` pair -- so
        the fixed-mode seed reads the live axis instead (same semantics: fixed freezes this view)."""
        lo, hi = self.ax.get_ylim()
        return float(lo), float(hi)

    def apply_relim_now(self) -> None:
        """A histogram's relim controls the COUNT (y) axis -- the measured quantity, like a 1D plot's
        signal axis, NOT the value/binning axis: re-apply the tight/normal/fixed count range NOW (the
        Setting/Edit relim toggle), bypassing any dead-band."""
        self._apply_count_yscale()
        self.draw()

    def apply_param(self, key: str, value) -> bool:
        """Log count-axis + bimodal-fit toggles are DISPLAY-ONLY: flip the flag then re-fit + redraw
        on the EXISTING axes (``update_core`` + ``draw``, exactly like :meth:`set_thresholds`), with no
        figure/canvas rebuild -- so the dis plot never flashes/resizes when the Setting or Edit form
        toggles them (#dis-resize).  Returns True if handled."""
        if key == "ylog":
            self.ylog = bool(value)
        elif key == "fit":
            self._fit = str(value)
        elif key == "bins":
            self.bins_arg = int(value)               # re-bin in place: update_core refills the SAME
        else:                                        # PolyCollection (set_verts) -- no figure rebuild,
            # not a histogram knob -> the base class (the relim family in place)
            return super().apply_param(key, value)
        self.update_core()
        self.draw()
        return True

    def classify(self, values=None) -> np.ndarray:
        vals = self.values if values is None else np.asarray(values, dtype=float)
        return np.digitize(vals, np.sort(self.thresholds))

    def fractions(self, values=None) -> dict[int, float]:
        states = self.classify(values)
        if len(states) == 0:
            return {}
        return {int(state): float(np.mean(states == state)) for state in np.unique(states)}

    def _apply_count_yscale(self) -> None:
        """Set the count (y) axis scale + limits -- the MEASURED axis a histogram's relim acts on
        (the dependent quantity, like a 1D plot's signal axis).  ``fixed`` pins the operator's lo/hi;
        ``normal``/``tight`` both anchor at 0 (counts are non-negative) and differ only in headroom
        above the peak -- the single-source normal-vs-tight rule (:meth:`_mode_target`), clamped to 0
        for the count axis.  Log mode floors at 0.5 so 0-count bars sit BELOW the axis (no
        0 -> -inf on the filled poly) and a sparse bright tail becomes visible."""
        peak = float(np.max(self.n))
        for counts in self._overlay_counts:           # include the 'create' overlay histograms' peak
            if len(counts):
                peak = max(peak, float(np.max(counts)))
        if getattr(self, "relim_mode", "normal") == "fixed":
            lo, hi = float(getattr(self, "fixed_lo", 0.0)), float(getattr(self, "fixed_hi", 1.0))
            if not hi > lo:                            # invalid pin -> fall back to auto headroom
                lo, hi = 0.0, max(1.0, peak * 1.2)
        else:
            lo, hi = self._mode_target(0.0, max(peak, 1.0))   # ONE tight/normal rule, shared with 1D/2D
            lo = max(0.0, lo)                          # counts never read below 0 (tight's -10% -> 0)
        hi = max(hi, lo + 1.0)
        if self.ylog:
            self.ax.set_yscale("log")
            self.ax.set_ylim(max(0.5, lo), max(hi, peak * 3.0 if peak else 1.0))
        else:
            self.ax.set_yscale("linear")
            self.ax.set_ylim(lo, hi)

    def _fit_bimodal(self) -> None:
        # A robust two-Gaussian fit with a UNIMODAL FALLBACK.  The old median split sat INSIDE the
        # dark blob whenever the bright mode was sparse (rare high occupancy), seeding both Gaussians
        # on dark so curve_fit collapsed to one blob and reported a MISLEADING fidelity.  Fix: split by
        # between-class variance (Otsu) over the samples, seed each side's amplitude from its own bin
        # counts, and -- when the bright mode is too sparse or unseparated -- fit ONE Gaussian and
        # report fit F = N/A (honest), never a fake number.  The MATH lives in the ONE
        # fit_histogram_curves primitive (shared with the grid's HistogramCell thumbnails);
        # this method only lands the result on the figure's artists + stat state.
        self.bimodal_popt = None
        self.fit_threshold = None
        self.single_popt = None
        self._fit_separated = False
        self._has_threshold = self._explicit_thr        # no fit yet -> only an explicit cut shows
        for ln in (self.fit_line_left, self.fit_line_right, self.fit_line_total):
            ln.set_data([], [])
        res = fit_histogram_curves(self.values, self.bins, self.n.astype(float), self._fit)
        if res is None:
            return
        if res["left"] is not None:
            self.fit_line_left.set_data(res["x"], res["left"])
            self.fit_line_right.set_data(res["x"], res["right"])
        self.fit_line_total.set_data(res["x"], res["total"])
        self.bimodal_popt = res["bimodal_popt"]
        self.single_popt = res["single_popt"]
        self._fit_separated = bool(res["separated"])
        self.fit_threshold = res["threshold"]
        self._has_threshold = self._fit_separated or self._explicit_thr

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
    panel          True if this kind is offered in the live console ADD-PANEL
                   dropdown (you add a blank panel of it and wire it to a signal).
                   ``pulse`` is ``panel=False`` -- you do not add a blank pulse
                   panel live -- but it is STILL a full panel kind: a saved pulse
                   figure SEEDS a ``kind="pulse"`` PanelCard (the seed path does not
                   gate on this flag), rendered by PanelCard's ``pulse`` branch.
                   So the flag means "addable live", NOT "can be a panel".
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
# Repeat-display vocabularies (single source).  The BASE verbs (average/add/replace) are GENERIC --
# ``reduce_repeat`` collapses the repeat axis the same way for ANY plot kind -- so every kind offers
# them.  Only two specialisations: ``create`` (one line / one sub-distribution per repeat) is for the
# 1-D families incl. the distribution, but NOT 2d/sites (an image has no per-repeat-line meaning);
# ``pool`` (bin EVERY repeat's samples into ONE histogram) is the distribution's own extra mode.
_BASE_REPEAT_MODES: tuple[str, ...] = ("average", "add", "replace")
TRACE_REPEAT_MODES: tuple[str, ...] = _BASE_REPEAT_MODES + ("create",)                 # 1-D vector: base + per-repeat lines (NO roll)
ROLLING_REPEAT_MODES: tuple[str, ...] = _BASE_REPEAT_MODES + ("roll", "create")        # rolling trace ONLY adds 'roll' (a rolling buffer)
IMAGE_REPEAT_MODES: tuple[str, ...] = _BASE_REPEAT_MODES                               # a frame: mean/sum/latest, no create/roll
HIST_REPEAT_MODES: tuple[str, ...] = ("pool",) + _BASE_REPEAT_MODES + ("create",)      # pool (default) + base + create (one overlaid histogram per repeat)

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
        repeat_modes=ROLLING_REPEAT_MODES,                  # rolling trace is the ONLY kind with 'roll'
    ),
    PlotKind(
        key="hist", cls=HistogramFigure, label="Distribution", render_family="1D",
        input_format="value must be a 1D sample vector",
        repeat_modes=HIST_REPEAT_MODES,
    ),
    # Notebook-only static timing diagram -- NOT a console Add-Panel kind.
    PlotKind(key="pulse", cls=PulseSequenceFigure, label="Pulse sequence", render_family="1D", panel=False),
    # NOTE: the per-site GRID kind (``key="grid"``, ``cls=GridPlot``) is APPENDED to this table right
    # after ``GridPlot`` is defined below (``_append_grid_plot_kind``) -- ``GridPlot`` does not exist yet
    # at this point in the module, so it cannot go in this literal.  It is still a single-source entry.
)

#: ``key -> PlotKind`` for O(1) dispatch.  Rebuilt by ``_append_grid_plot_kind`` once ``grid`` lands.
PLOT_KIND_BY_KEY: dict[str, PlotKind] = {pk.key: pk for pk in PLOT_KINDS}


def kind_for_plotter(plotter) -> str | None:
    """The ``PLOT_KINDS`` key a plotter object BELONGS to -- reverse-looked-up from ``type(plotter)`` in
    the ONE table (never a hand-typed name), so a bare ``plot.save()`` can stamp the correct ``kind`` into
    the saved ``info`` and the reopen (``SavedFigure.kind``) round-trips to the SAME kind that drew it.

    The match walks the plotter's MRO against each ``PlotKind.cls``, so a SUBCLASS resolves to its base
    kind (``SiteHistogramGrid`` / an ``ImageCell`` grid -> the ``GridPlot`` kind ``"grid"``; the
    ``LiveLiveDis`` rolling-trace variant -> ``"monitor"``).  The FIRST matching table entry wins (table
    order), so a concrete class ranks before a base it also subclasses.  ``None`` when the object is not a
    known plotter (a bare externally-built DataFigure has nothing to stamp)."""
    cls = type(plotter)
    for base in cls.__mro__:
        for pk in PLOT_KINDS:
            if pk.cls is base:
                return pk.key
    return None


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


# ------------------------------------------------------------------ facet: the grid as an axis-expander
# A grid is a DIMENSION-EXPANSION tool: pick ONE axis of a measurement block -- the block always obeys
# the shape iron law ``(repeat, points, *data_dim)`` -- and lay that axis out as N aligned cells, each a
# standard ``sub_plot_kind`` panel of the REMAINING axes.  ``facet_cells`` below is the ONE slicing rule
# every surface shares (the console's live grid panel, the notebook ``grid(value, facet=...)`` factory),
# so "which axis becomes the cells and what each cell shows" is defined in exactly one place.

def normalize_facet(facet):
    """Parse a facet spec into the canonical ``(group, index)`` pair -- or ``None`` (no faceting:
    the grid shows a static per-cell recipe, the pre-facet behaviour).

    Accepted spellings: ``"repeat"`` (one cell per repeat), ``"points"``/``"points:i"`` (one cell per
    entry of the i-th POINTS axis -- the scan grid's axes, e.g. the outer axis of a 3-D pulse scan),
    ``"dim"``/``"dim:i"`` (one cell per entry of the i-th DATA axis, e.g. per site of a per-site
    vector), or an explicit ``(group, index)`` tuple.  Anything else raises."""
    if facet is None or facet == "":
        return None
    if isinstance(facet, (tuple, list)) and len(facet) == 2:
        group, index = str(facet[0]), int(facet[1])
    else:
        text = str(facet).strip()
        group, _, idx = text.partition(":")
        group = group.strip()
        index = int(idx) if idx.strip() else 0
    if group not in ("repeat", "points", "dim") or index < 0:
        raise ValueError(
            f"facet must be 'repeat', 'points[:i]' or 'dim[:i]' (or a (group, i) pair); got {facet!r}.")
    return group, index


def _collapse_repeat(sliced, repeat_mode: str):
    """Collapse the leading REPEAT axis of a per-cell slice for an image/line cell -- the same display
    verbs the standalone kinds use (``average`` mean / ``add`` sum / ``replace`` latest).  A histogram
    cell never calls this: a distribution POOLS every repeat's samples by definition."""
    mode = str(repeat_mode)
    if mode == "add":
        return np.nansum(sliced, axis=0)
    if mode == "replace":
        return np.asarray(sliced[-1], dtype=float)
    # 'average' (and the fallback for any other verb): an all-NaN cell -- the up-front EMPTY grid a
    # scanning task publishes before its first point, or a not-yet-filled facet plane -- collapses to
    # NaN (a blank cell) BY INTENT, through the ONE gap-safe mean so it never warns per empty cell.
    return _nanmean_gap_safe(sliced, 0)


def default_sub_plot_kind(facet, *, points_shape=(), data_shape=()) -> str:
    """The natural per-cell kind for a facet slice -- from what each cell has LEFT after the slice
    (the same by-dimensionality rule the console's reshape uses, #H3o): a remaining 2-D points grid
    OR a remaining 2-D data_dim (a camera frame) is an image cell, more than one remaining value is
    a curve, a bare sample set is a distribution.  The ONE auto rule the console's 'auto'
    sub_plot_kind and the notebook factory share."""
    group, index = normalize_facet(facet) or ("", 0)
    pts = tuple(int(n) for n in points_shape) or (1,)
    dim = tuple(int(n) for n in data_shape) or (1,)
    if group == "points":
        pts = tuple(n for i, n in enumerate(pts) if i != index)
    elif group == "dim":
        dim = tuple(n for i, n in enumerate(dim) if i != index)
    if len(pts) == 2:
        return "2d"                            # a remaining 2-D scan grid is an image
    if int(np.prod(pts, dtype=np.int64)) == 1 and len([n for n in dim if n > 1]) == 2:
        return "2d"                            # a remaining 2-D data_dim (a camera frame) is an image
    if int(np.prod(pts, dtype=np.int64)) > 1 or int(np.prod(dim, dtype=np.int64)) > 1:
        return "1d"                            # an ordered remaining axis reads as a curve
    return "hist"                              # nothing left but repeats -> a sample distribution


def facet_cells(value, facet, *, sub_plot_kind: str, points_shape=(), repeat_mode: str = "average"):
    """Slice ONE ``(repeat, points, *data_dim)`` block along the facet axis into the per-cell inputs a
    :class:`GridPlot` of ``sub_plot_kind`` cells displays -- the SINGLE data rule of the grid-as-facet
    design.  Returns ``[cell_0_data, cell_1_data, ...]``.

    ``points_shape`` is the MULTI-D shape of the points axis when the producing scan declared one (the
    node's ``grid_shape`` -- e.g. ``(n_outer, ny, nx)`` for a 3-level scan; the stored block keeps points
    FLAT).  Each cell keeps the iron law's remaining axes and is shaped for its kind:

    * ``hist``  -- the cell's SAMPLES: everything left after the slice, pooled (a distribution pools
      repeats by definition; ``repeat_mode`` is not consulted).
    * ``2d``    -- the cell's FRAME: repeats collapsed by ``repeat_mode``, then the remaining points
      grid (2-D) or the 2-D data_dim is the image.
    * ``1d``    -- the cell's CURVE: repeats collapsed by ``repeat_mode``, remaining values flattened.
    """
    spec = normalize_facet(facet)
    if spec is None:
        raise ValueError("facet_cells needs an explicit facet; None means a recipe (non-faceted) grid.")
    group, index = spec
    a = np.asarray(value, dtype=float)
    if a.ndim < 2:
        raise ValueError(f"a facet grid slices a (repeat, points, *data_dim) block; got shape {a.shape}.")
    n_repeat, n_points = int(a.shape[0]), int(a.shape[1])
    dim_shape = tuple(int(n) for n in a.shape[2:])
    pts_shape = tuple(int(n) for n in (points_shape or (n_points,)))
    if int(np.prod(pts_shape)) != n_points:
        raise ValueError(f"points_shape {pts_shape} does not match the block's points axis ({n_points}).")
    # expose the points axis in its declared multi-D form: (repeat, *pts_shape, *dim_shape)
    a = a.reshape((n_repeat, *pts_shape, *dim_shape))
    if group == "repeat":
        slices = [a[r] for r in range(n_repeat)]                    # each: (*pts, *dim), repeat consumed
        pts_rest, dim_rest, has_repeat = pts_shape, dim_shape, False
    elif group == "points":
        if index >= len(pts_shape):
            raise ValueError(f"facet points:{index} out of range for points_shape {pts_shape}.")
        moved = np.moveaxis(a, 1 + index, 0)                        # (n_i, repeat, *pts_rest, *dim)
        slices = [moved[j] for j in range(moved.shape[0])]
        pts_rest = tuple(n for i, n in enumerate(pts_shape) if i != index)
        dim_rest, has_repeat = dim_shape, True
    else:  # "dim"
        if index >= len(dim_shape):
            raise ValueError(f"facet dim:{index} out of range for data_dim {dim_shape}.")
        moved = np.moveaxis(a, 1 + len(pts_shape) + index, 0)       # (n_i, repeat, *pts, *dim_rest)
        slices = [moved[j] for j in range(moved.shape[0])]
        pts_rest = pts_shape
        dim_rest = tuple(n for i, n in enumerate(dim_shape) if i != index)
        has_repeat = True

    kind = str(sub_plot_kind)
    if kind == "hist":
        return [s.reshape(-1) for s in slices]                      # a distribution pools everything left
    cells = []
    for s in slices:
        core = _collapse_repeat(s, repeat_mode) if has_repeat else np.asarray(s, dtype=float)
        if kind == "2d":
            # the frame is the remaining 2-D points grid (data_dim trivial -- keep declared 1-length
            # scan axes), else whatever 2-D core the slice leaves (a camera frame).  Anything that is
            # not exactly 2-D after dropping 1-length axes raises -- NEVER a silent component pick.
            if len(pts_rest) == 2 and int(np.prod(dim_rest, dtype=np.int64)) == 1:
                core = core.reshape(pts_rest)
            else:
                core = np.squeeze(core)
            if core.ndim != 2:
                raise ValueError(
                    f"a 2d facet cell needs a 2-D frame after slicing; got shape {core.shape} "
                    f"(points left {pts_rest}, data_dim left {dim_rest}).")
            cells.append(core)
        else:  # "1d"
            cells.append(core.reshape(-1))
    return cells


class GridCell:
    """Strategy for ONE cell of a :class:`GridPlot` -- subclass per multi-panel
    plot type (``HistogramCell`` now; future ``Image2DCell`` / ``Line1DCell``).

    The GridPlot owns everything generic (layout, focus-zoom, per-cell selectors,
    DataFigure plumbing); the cell only knows how to DRAW one panel and what its
    threshold/DataFigure are.  ``draw(ax, k)`` draws the small grid THUMBNAIL; the
    ENLARGED focus view is a STANDALONE ``panel_plot`` of the cell's own plot KIND
    (``sub_plot_kind`` + :meth:`focus_data`), never a hand-rolled copy -- which is
    exactly why the focus-zoom is plot-type-agnostic.

    A subclass declares which ``PLOT_KINDS`` key its per-site plot is
    (``sub_plot_kind`` -- ``"hist"`` for a distribution cell, ``"2d"`` for an image
    cell) and supplies that cell's data as ``panel_plot`` kwargs (:meth:`focus_data`).
    :meth:`GridPlot.build_focus_plotter` then dispatches through the ONE
    ``PLOT_KIND_BY_KEY`` table, so a NEW cell family adds ONE ``sub_plot_kind`` +
    ``focus_data`` (never a bespoke figure) -- the SAME plot the standalone panel of
    that kind uses, so its lim / fit / threshold-drag all work on the standard path."""

    n_cells: int = 0
    #: The ``PLOT_KINDS`` key of this cell's PER-SITE plot -- the SINGLE source that (a) selects the cell
    #: class (``GRID_CELL_BY_KIND``), (b) dispatches the focus-zoom to that kind's standalone panel, and
    #: (c) drives which PANEL_PARAMS the grid panel's Setting/Edit UI shows.  Subclasses set it
    #: ("hist" / "2d" / ...).  It is the grid's ``sub_plot_kind`` -- ONE declaration, no hard-coded class.
    sub_plot_kind: str = "hist"
    #: The operator's lim state for every cell thumbnail's DEPENDENT axis (a hist cell's count axis, an
    #: image cell's colour scale) -- the SAME relim family the standalone kinds keep (``relim_mode`` /
    #: ``fixed_lo`` / ``fixed_hi``), fed by ``GridPlot.store_display_param`` exactly like ``bins``/``cmap``
    #: are.  So a lim edit reaches the THUMBNAILS through the one store -> cell-state -> redraw mechanism
    #: every display knob uses -- never "only the enlarged view changed".  ``fixed`` pins every cell to
    #: the operator's lo/hi; any other mode keeps the cell's own auto range (a static thumbnail has no
    #: dead-band autoscaler to re-run).
    relim_mode: str = "tight"
    fixed_lo: float = 0.0
    fixed_hi: float = 1.0
    #: Cell-text scale, set by the GRID from the PANEL SIZE preset (_site_grid_geometry): a grid
    #: squeezed below its RECOMMENDED default size uses HALF the standard small/tick font sizes,
    #: every other size uses them unscaled.  Cells multiply their apply_title size by it; the
    #: grid's tick policy multiplies its label size by it.
    font_scale: float = 1.0

    def thumb_lims(self, auto_lo: float, auto_hi: float) -> tuple[float, float]:
        """The thumbnail's dependent-axis range: the operator's pinned lo/hi in ``fixed`` mode, else the
        cell's own auto range -- the ONE place a cell's draw asks, so every cell family honours fixed."""
        if self.relim_mode == "fixed":
            return float(self.fixed_lo), float(self.fixed_hi)
        return float(auto_lo), float(auto_hi)

    def auto_lims(self) -> tuple[float, float]:
        """The cells' SHARED auto range for the dependent axis -- what the grid 'shows now' as one
        number pair (an image family already keeps ONE colour scale; a distribution family's is the
        tallest cell's count axis).  This is the grid's ``current_lims`` -- the seed when the operator
        flips relim to ``fixed`` without an enlarged cell (fixed freezes what you see, never 0..1)."""
        raise NotImplementedError

    def consume_param(self, key: str, value) -> bool:
        """Land a per-kind DISPLAY knob on this cell family's own state (a hist cell's bins / fit /
        ylog, an image cell's cmap) and return True when the THUMBNAILS are now stale.  The single
        param fan-out: ``GridPlot.store_display_param`` owns only the generic relim family and hands
        every other key HERE -- the cell family, not the grid, declares which of its standalone
        kind's ``PANEL_PARAMS`` it renders (so a new knob is one method here, never a GridPlot edit).
        Unknown keys return False (stored on the grid for the focus seed, thumbnails untouched)."""
        return False

    def prepare(self) -> None:
        """Compute any shared state (e.g. common histogram bin edges) once."""

    def draw(self, ax, k: int):
        """Draw the THUMBNAIL of cell ``k`` into ``ax`` -- the compact grid-cell view (a corner tag, hidden
        axes; the SHAPE is the point).  The ENLARGED focus view is a STANDALONE ``panel_plot`` of this cell's
        :data:`sub_plot_kind`, built from :meth:`focus_data`, NOT a detail flag on this draw.  Return the cell's
        draggable threshold line or ``None``."""
        raise NotImplementedError

    # ---- live (facet) update contract: replace the data, move the EXISTING artists.  A live facet
    # grid refreshes every shot; rebuilding N axes' ticks/text each tick is exactly the cost the grid
    # perf work removed (#perf), so a cell family updates in place and only falls back to a full
    # thumbnail redraw when it says so (update_cell -> False).
    def set_cell_data(self, k: int, data) -> None:
        """Replace cell ``k``'s data (the live facet feed).  Same per-cell payload the constructor
        takes for this family (hist: sample vector; 2d: 2-D frame; 1d: y curve)."""
        raise NotImplementedError

    def update_cell(self, ax, k: int) -> bool:
        """Move cell ``k``'s EXISTING artists to the current data (no axes clear, no new artists --
        the ticks/text built once by :meth:`draw` stay).  Return False when an in-place move cannot
        represent the change (artist structure changed) -- the grid then redraws that thumbnail."""
        raise NotImplementedError

    def focus_update(self, focus_plotter, k: int) -> None:
        """Feed cell ``k``'s current data to its ENLARGED standalone view -- the kind's ordinary
        ``update`` (a HistogramFigure rebins, a Live2DDis re-images), so the zoomed cell stays live
        with zero bespoke code."""
        raise NotImplementedError

    def threshold_line(self, k: int):
        return None

    def on_threshold_drag(self, k: int, x: float) -> None:
        """Called while a threshold is dragged (update data + any annotation)."""

    def data_figure(self, fig, ax, k: int):
        """The per-cell :class:`DataFigure` (reusable fitting stack)."""
        raise NotImplementedError

    def focus_data(self, k: int, *, display_params: Mapping[str, Any] | None = None) -> dict:
        """The ``panel_plot`` args for the STANDALONE enlarged view of cell ``k``: ``{"data_x": ..., "data_y":
        ..., ...kind-specific kwargs...}`` merged into ``panel_plot(kind=self.sub_plot_kind, size=..., **here)``
        by :meth:`GridPlot.build_focus_plotter`.  A distribution cell returns its sample vector + bins / fit /
        threshold; an image cell returns the pixel-coordinate scatter + cmap.  ``display_params`` are the
        grid's live display knobs (bins / fit / ylog / cmap) folded in, so a Setting change reaches the
        enlarged cell.  Because the result is a REAL standalone ``panel_plot`` of ``sub_plot_kind``, the enlarged
        cell carries that kind's full x/y axes, draggable threshold, fit and standard relim -- no bespoke code."""
        raise NotImplementedError


class HistogramCell(GridCell):
    """A GridPlot cell that is a per-site count histogram (the distribution grid).

    ``per_site_values`` is ``(n, n_samples)`` or a list of 1D arrays; ``occupied``
    colours dark/bright populations; ``thresholds`` draws a draggable cut;
    ``site_fidelities`` annotates each cell."""

    sub_plot_kind = "hist"       # each cell is a ``hist`` per-site plot (enlarged -> a standalone HistogramFigure)

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
        # The hist kind's OWN display knobs, rendered on the thumbnails through the SAME primitives
        # the standalone HistogramFigure uses (fit_histogram_curves / set_yscale) -- and the SAME
        # defaults (DEFAULT_HIST_FIT): what the Setting UI shows as the default IS what the grid
        # draws, never a per-surface divergence.
        self.fit = DEFAULT_HIST_FIT
        self.ylog = False
        self.threshold_lines: list = [None] * self.n_cells
        self.tag_texts: list = [None] * self.n_cells
        self.cell_counts: list = [None] * self.n_cells
        self._bar_colls: list = [None] * self.n_cells      # cell k's PolyCollections, for the live in-place move
        self._fit_lines: list = [None] * self.n_cells      # cell k's (left, right, total) fit Line2Ds

    def consume_param(self, key: str, value) -> bool:
        if str(key) == "bins":
            self.bins_arg = int(value)
            self.edges = None
            self.prepare()
            return True
        if str(key) == "fit":
            if self.fit == str(value):
                return False
            self.fit = str(value)
            return True
        if str(key) == "ylog":
            if self.ylog == bool(value):
                return False
            self.ylog = bool(value)
            return True
        return False

    def prepare(self) -> None:
        vals = self.values
        pooled = np.concatenate([v[np.isfinite(v)] for v in vals if v.size]) if any(v.size for v in vals) else np.array([0.0, 1.0])
        lo, hi = np.quantile(pooled, [0.002, 0.998]) if pooled.size > 2 else (float(np.min(pooled)), float(np.max(pooled)))
        if not np.isfinite([lo, hi]).all() or hi <= lo:
            lo, hi = float(np.min(pooled)), float(np.max(pooled)) + 1.0
        span = hi - lo
        self.x_lo, self.x_hi = float(lo - 0.04 * span), float(hi + 0.04 * span)
        self.edges = np.linspace(self.x_lo, self.x_hi, self.bins_arg + 1)
        # ONE shared count-axis top (the tallest cell): the grid is a shared-axes family, so the
        # leftmost column's y labels read for every cell in the row -- per-cell tops would make
        # the edge labels lie.  Cached here; auto_lims and the per-cell ylim consume it.
        top = 0.0
        for v in vals:
            v = v[np.isfinite(v)]
            if v.size:
                top = max(top, float(np.histogram(v, bins=self.edges)[0].max()))
        self.y_top = top

    def _tag_text(self, k: int) -> str:
        """The per-cell TITLE text -- the site index (+ its fidelity when known).  It is placed in the
        cell's small axes title ABOVE the plot (never inside the data area), so the histogram stays clean."""
        fids = self.site_fidelities
        return f"s{k}" if fids is None or not np.isfinite(fids[k]) else f"s{k}  {fids[k] * 100:.0f}%"

    def _apply_fit_lines(self, ax, k: int, counts) -> None:
        """(Re)fit + draw cell ``k``'s fit curves through the ONE primitive the standalone dis uses
        (:func:`fit_histogram_curves`) -- the same three lines in the same colours, moved in place on
        a live tick.  Off ("none") / impossible fits leave the lines empty."""
        lines = self._fit_lines[k]
        if self.fit == "none" and lines is None:
            return                              # never fitted on this cell -> nothing to build/clear
        if lines is None:                       # lazily build the same 3-line set the standalone dis owns
            lines = (ax.plot([], [], color=PALETTE["fit_left"], linewidth=1, alpha=0.8)[0],
                     ax.plot([], [], color=PALETTE["fit_right"], linewidth=1, alpha=0.8)[0],
                     ax.plot([], [], color=PALETTE["fit_total"], linewidth=1, alpha=0.35)[0])
            self._fit_lines[k] = lines
        for ln in lines:
            ln.set_data([], [])
        res = fit_histogram_curves(self.values[k], self.edges, counts, self.fit)
        if res is None:
            return
        if res["left"] is not None:
            lines[0].set_data(res["x"], res["left"])
            lines[1].set_data(res["x"], res["right"])
        lines[2].set_data(res["x"], res["total"])

    def _apply_count_ylim(self, ax, counts) -> None:
        """The thumbnail's count-axis scale + range: the SHARED top (every cell shows the same count
        axis, so the leftmost column's edge labels read for the whole grid) through the operator's
        relim family (``thumb_lims``), with the standalone dis's log rule (floor at 0.5 so 0-count
        bars sit below the axis).  ``set_yscale`` only on an actual scale CHANGE -- it rebuilds the
        axis' locators, and calling it per cell per live tick re-introduced the tick cost (#perf)."""
        del counts                              # the shared top is the range; per-cell counts are not
        top = float(getattr(self, "y_top", 0.0))
        lo, hi = self.thumb_lims(0.0, top * 1.08 if top > 0 else 1.0)
        target = "log" if self.ylog else "linear"
        if ax.get_yscale() != target:
            ax.set_yscale(target)
        if self.ylog:
            ax.set_ylim(max(0.5, lo), max(hi, 1.0))
        else:
            ax.set_ylim(lo, hi)

    def _make_bar_colls(self, ax, k: int):
        """(Re)build cell ``k``'s bar PolyCollections from the current samples on the shared edges --
        the ONE bar construction :meth:`draw` and a re-bin (:meth:`update_cell`) share.  ONE filled
        PolyCollection per population -- the SAME bar geometry the standalone hist builds via
        :func:`_update_verts` (single source), NEVER ``ax.hist`` whose per-bin Rectangles made an
        N-cell grid crawl (#perf).  Returns the total counts (also stored in ``cell_counts``)."""
        edges = self.edges
        v = self.values[k]
        v = v[np.isfinite(v)]
        colls = []

        def _bars(counts, color):
            verts = np.empty((len(counts), 4, 2), dtype=float)
            _update_verts(edges, counts, verts, mode="vertical")
            coll = PolyCollection(verts, facecolors=color, edgecolors="none")
            ax.add_collection(coll)
            colls.append(coll)                 # kept so the live facet feed can move bars in place

        counts_all = np.histogram(v, bins=edges)[0].astype(float) if v.size else np.zeros(len(edges) - 1)
        occ = self.occupied
        if occ is not None and occ[k].size == self.values[k].size and v.size:
            mask = np.asarray(occ[k], dtype=bool)[np.isfinite(self.values[k])]
            dark = v[~mask]
            dark_counts = (np.histogram(dark, bins=edges)[0].astype(float) if dark.size
                           else np.zeros(len(edges) - 1))
            # The stacked dark/bright look with TWO collections: the full bar in the bright colour,
            # the dark portion overdrawn from zero -- per bin the top segment (dark..total) reads bright,
            # exactly the ax.hist(stacked=True) rendering.
            _bars(counts_all, PALETTE["bright"])
            _bars(dark_counts, PALETTE["dark"])
        elif v.size:
            _bars(counts_all, PALETTE["hist_fill"])
        self.cell_counts[k] = counts_all
        self._bar_colls[k] = colls
        return counts_all

    def draw(self, ax, k: int):
        """The compact grid THUMBNAIL of cell ``k`` (corner tag; the SHAPE is the point).  Draws only
        the DATA artists (bars via :meth:`_make_bar_colls`, fit curves, threshold) + the small title;
        the tick policy is the GRID's (:meth:`GridPlot._style_cell_ticks`).  The enlarged focus view
        is a standalone ``hist`` panel (:class:`HistogramFigure`) -- see :meth:`focus_data`."""
        if self.edges is None:
            self.prepare()
        counts_all = self._make_bar_colls(ax, k)
        self._fit_lines[k] = None               # the axes were cleared -> the old fit lines are gone
        self._apply_fit_lines(ax, k, counts_all)
        line = None
        if self.thresholds is not None and np.isfinite(self.thresholds[k]):
            line = ax.axvline(float(self.thresholds[k]), **threshold_line_kwargs(1.4))
        ax.set_xlim(self.x_lo, self.x_hi)
        # collections do not autoscale -- set the count axis from the binned data directly (cheaper too);
        # in ``fixed`` mode the operator's pinned lo/hi win (thumb_lims -- the SAME relim family the
        # enlarged view honours, so a lim edit changes thumbnails AND zoom together, like cmap/bins do)
        self._apply_count_ylim(ax, counts_all)
        # The site index (+ fidelity) is the cell's small TITLE, ABOVE the axes -- NOT text inside the data
        # area (#5) -- through the ONE title mechanism (apply_title), just at small_fontsize().
        apply_title(ax, self._tag_text(k), size=small_fontsize() * self.font_scale, pad=1.5)
        self.threshold_lines[k] = line           # grid line, kept for per-cell drag
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

    def auto_lims(self) -> tuple[float, float]:
        """The tallest cell's count-axis range (same 8% headroom the thumbnail draw applies) --
        the distribution grid's shared 'what you see now' pair (cached by prepare)."""
        if self.edges is None:
            self.prepare()
        top = float(getattr(self, "y_top", 0.0))
        return 0.0, top * 1.08 if top > 0 else 1.0

    def set_cell_data(self, k: int, data) -> None:
        self.values[k] = np.asarray(data, dtype=float).reshape(-1)
        self.edges = None                      # samples moved -> the shared bin edges re-derive (prepare)

    def update_cell(self, ax, k: int) -> bool:
        """Move cell ``k``'s bar collections to the current samples IN PLACE: recompute counts on the
        (re-)prepared shared edges and ``set_verts``; when the artist structure no longer matches (a
        re-bin changed the vert count, a stacked pair, an empty-born cell) rebuild ONLY the bar
        collections through the one constructor (:meth:`_make_bar_colls`) -- the ticks, title,
        threshold and fit lines all stay, so no display knob ever costs an axes clear (#perf)."""
        if self.edges is None:
            self.prepare()
        colls = self._bar_colls[k]
        v = self.values[k]
        v = v[np.isfinite(v)]
        counts = np.histogram(v, bins=self.edges)[0].astype(float) if v.size else np.zeros(len(self.edges) - 1)
        if colls and len(colls) == 1 and len(counts) == len(colls[0].get_paths()):
            verts = np.empty((len(counts), 4, 2), dtype=float)
            _update_verts(self.edges, counts, verts, mode="vertical")
            colls[0].set_verts(verts)
            self.cell_counts[k] = counts
        else:                                  # re-binned / stacked / empty-born -> rebuild bars only
            for coll in (colls or []):
                coll.remove()
            counts = self._make_bar_colls(ax, k)
        self._apply_fit_lines(ax, k, counts)   # the fit curves track the moved samples (one primitive)
        ax.set_xlim(self.x_lo, self.x_hi)      # shared edges track the pooled samples (prepare)
        self._apply_count_ylim(ax, counts)
        return True

    def focus_update(self, focus_plotter, k: int) -> None:
        focus_plotter.update(self.values[k])   # the standalone hist rebins itself (its ordinary update)

    def classify(self, k: int, values=None) -> np.ndarray:
        vals = self.values[k] if values is None else np.asarray(values, dtype=float)
        if self.thresholds is not None and np.isfinite(self.thresholds[k]):
            thr = self.thresholds[k]
        else:
            finite = vals[np.isfinite(vals)]
            thr = float(np.nanmedian(finite)) if finite.size else 0.0
        return (vals > thr).astype(int)

    def focus_data(self, k: int, *, display_params: Mapping[str, Any] | None = None) -> dict:
        """The ``panel_plot(kind="hist")`` args for the STANDALONE enlarged view of distribution cell ``k`` --
        the SAME ``hist`` panel a standalone histogram uses, so the enlarged cell carries the draggable
        threshold that live-updates the fidelity readout, the two-Gaussian fit + fidelity stat, the fit chooser
        and full x/y axes -- reusing the ONE plot-kind renderer (never a hand-rolled copy).  The grid's
        per-cell values / threshold / fidelity seed it, and the grid's live display knobs (``bins`` / ``fit`` /
        ``ylog``) are folded in so a Setting change reaches the enlarged cell.  :meth:`GridPlot.build_focus_plotter`
        merges this into ``panel_plot``; a drag of the enlarged threshold is mirrored back onto the cell by
        :meth:`sync_threshold_from_focus` (so the grid thumbnail + the save recipe pick up the new cut)."""
        params = dict(display_params or {})
        thr = None
        if self.thresholds is not None and np.isfinite(self.thresholds[k]):
            thr = [float(self.thresholds[k])]
        fid = self.site_fidelities
        ftxt = "" if (fid is None or not np.isfinite(fid[k])) else f"   F={fid[k] * 100:.1f}%"
        return {
            "data_x": self.values[k],                    # hist reads data_x as the values array (plot() convention)
            "bins": int(params.get("bins", self.bins_arg)),
            "thresholds": thr,
            "fit": str(params.get("fit", "double")),
            "ylog": bool(params.get("ylog", False)),
            "labels": (self.labels[0], self.labels[1] if len(self.labels) > 1 else "Shots", "Population"),
            "title": f"site {k}{ftxt}",
        }

    def sync_threshold_from_focus(self, k: int, focus_plotter) -> None:
        """After the operator dragged the FOCUS view's threshold, copy the resulting cut back onto cell
        ``k`` (the single source the grid thumbnail + the save recipe read) so an enlarged-view drag is not
        lost when the grid returns.  Called by :meth:`GridPlot.unfocus`; a no-op for a cell / plotter without
        a meaningful threshold."""
        thr = getattr(focus_plotter, "thresholds", None)
        if not thr or not np.isfinite(thr[0]):
            return
        if self.thresholds is None:
            self.thresholds = [float("nan")] * self.n_cells
        self.thresholds[k] = float(thr[0])
        grid_line = self.threshold_lines[k]
        if grid_line is not None:
            grid_line.set_xdata([thr[0], thr[0]])


class ImageCell(GridCell):
    """A GridPlot cell that is a 2D image -- one ``imshow`` per cell.  The readout's
    per-site **PSF weight kernel** uses it, so the operator SEES the real (asymmetric,
    non-Gaussian) atom spot the matched filter weights by.

    ``images`` is a sequence of 2D arrays (one per cell); ALL cells share ONE colour
    scale so the grid is comparable (the art "aligned" rule).  An image cell has no 1D
    cut, so it contributes the zoom/pan/area/cross selectors but no draggable threshold
    line -- exactly the ``Image2DCell`` the :class:`GridCell` docstring reserves."""

    sub_plot_kind = "2d"         # each cell is a ``2d`` per-site plot (enlarged -> a standalone Live2DDis)

    def __init__(self, images, *, labels: Sequence[str] = ("x (px)", "y (px)"), cmap: str | None = None):
        self.images = [np.asarray(im, dtype=float) for im in images]
        self.n_cells = len(self.images)
        if self.n_cells == 0:
            raise ValueError("images must contain at least one cell.")
        self.labels = tuple(labels)
        # The colormap the thumbnails + the enlarged cell draw with -- the grid's ``cmap`` display param
        # (PANEL_PARAMS["2d"]) flows here so a 2d grid's Setting colormap chooser recolours the cells (a
        # hist grid has no cmap).  ``None`` = the camera default (the single PALETTE source).
        self.cmap = str(cmap) if cmap else PALETTE["cmap_camera"]
        self.vmin, self.vmax = 0.0, 1.0
        self._image_artists: list = [None] * self.n_cells   # cell k's AxesImage, for the live in-place move

    def consume_param(self, key: str, value) -> bool:
        if str(key) in ("cmap", "colorset"):
            self.cmap = str(value)
            return True
        return False

    def prepare(self) -> None:
        finite = [im[np.isfinite(im)] for im in self.images if im.size]
        pooled = np.concatenate(finite) if finite else np.array([0.0, 1.0])
        lo = float(np.nanmin(pooled)) if pooled.size else 0.0
        hi = float(np.nanmax(pooled)) if pooled.size else 1.0
        if not hi > lo:
            lo, hi = lo, lo + 1.0
        # ONE shared colour scale spanning the pooled data -> cells comparable AND contrasty: a
        # zero-anchored scale washed out any family whose values sit on a baseline (a PSF kernel
        # starts at ~0 so it renders identically; a camera-count facet keeps its contrast).
        self.vmin, self.vmax = lo, hi

    def auto_lims(self) -> tuple[float, float]:
        """The ONE shared colour scale every cell already draws with -- the image grid's
        'what you see now' pair (exact, since the family keeps a single scale by design)."""
        return float(self.vmin), float(self.vmax)

    def draw(self, ax, k: int):
        """The compact grid THUMBNAIL of kernel cell ``k`` (no ticks; the kernel SHAPE is the point).  The
        site index goes in the cell's small TITLE (above the axes), NOT inside the image (#5).  The enlarged
        focus view is a standalone ``2d`` panel (:class:`Live2DDis`) -- see :meth:`focus_data`."""
        # one shared colour scale keeps cells comparable; in ``fixed`` mode the operator's pinned
        # lo/hi win (thumb_lims -- the SAME relim family the enlarged view honours, like cmap)
        vmin, vmax = self.thumb_lims(self.vmin, self.vmax)
        self._image_artists[k] = ax.imshow(self.images[k], origin="lower", cmap=self.cmap,
                                           vmin=vmin, vmax=vmax, aspect="equal")
        apply_title(ax, f"s{k}", size=small_fontsize() * self.font_scale, pad=1.5)   # ticks are the grid's ONE policy
        return None

    def set_cell_data(self, k: int, data) -> None:
        frame = np.asarray(data, dtype=float)
        if frame.ndim != 2:
            raise ValueError(f"an image cell takes a 2-D frame; got shape {frame.shape}.")
        self.images[k] = frame                 # the shared colour scale re-derives in prepare()

    def update_cell(self, ax, k: int) -> bool:
        """Move cell ``k``'s EXISTING image to the current frame (set_array + the shared/pinned colour
        scale) -- no axes clear, no new artists.  A frame whose SHAPE changed cannot reuse the image's
        pixel extent, so it returns False and the grid redraws that thumbnail once."""
        im = self._image_artists[k]
        if im is None or tuple(im.get_array().shape) != tuple(self.images[k].shape):
            return False
        im.set_array(self.images[k])
        im.set_cmap(self.cmap)                 # a Setting colormap pick lands in place too
        im.set_clim(*self.thumb_lims(self.vmin, self.vmax))
        return True

    def focus_update(self, focus_plotter, k: int) -> None:
        focus_plotter.update(self.images[k].reshape(-1))     # the standalone 2d consumes the flat frame

    def data_figure(self, fig, ax, k: int):
        from .data_figure import DataFigure
        # The grid's ``.npz`` (the save contract) carries each site's PSF kernel as a
        # 1D series (square-recoverable); the canonical 2D weights live in calibration.npz.
        flat = self.images[k].reshape(-1)
        return DataFigure(fig=fig, ax=ax, data_x=np.arange(flat.size, dtype=float),
                          data_y=flat, labels=self.labels, name=f"site{k}_psf")

    def focus_data(self, k: int, *, display_params: Mapping[str, Any] | None = None) -> dict:
        """The ``panel_plot(kind="2d")`` args for the STANDALONE enlarged view of kernel cell ``k`` -- the SAME
        ``2d`` panel a standalone image uses, so the enlarged kernel carries the side clim distribution, the
        colorbar, the draggable clim lines and full x/y pixel axes -- reusing the ONE plot-kind renderer (never
        a hand-rolled copy).  The image's pixel grid is unpacked into the ``(N, 2)`` coordinate scatter + value
        column ``Live2DDis`` consumes (the SAME convention the console's ``2d`` panel uses).  The cmap can be
        overridden by the grid's ``cmap`` / ``colorset`` display param.  :meth:`GridPlot.build_focus_plotter`
        merges this into ``panel_plot``."""
        params = dict(display_params or {})
        im = self.images[k]
        ny, nx = im.shape
        xx, yy = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
        data_x = np.column_stack([xx.ravel(), yy.ravel()])
        cmap = str(params.get("cmap") or params.get("colorset") or self.cmap)
        return {
            "data_x": data_x,
            "data_y": im.ravel(),
            "cmap": cmap,
            "labels": (self.labels[0], self.labels[1] if len(self.labels) > 1 else "y (px)", "weight"),
            "title": f"site {k}",
        }


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
        """Save the whole grid the SAME way DataFigure saves a single plot: ONE png AND ONE matching
        ``.npz`` that ``load_figure`` reopens FAITHFULLY.  A grid is a first-class LOADABLE kind (like the
        pulse figure): the ``.npz`` carries the standard ``data_x`` / ``data_y`` / ``info`` triple with
        ``info['kind'] == 'grid'`` and a ``info['figure_recipe']`` (:func:`grid_recipe_from_cells`) that is
        the SINGLE truth source for reproduction -- :func:`build_grid_figure` rebuilds the exact grid from
        it, so ``na.load_figure(npz).plot()`` and a seeded ``kind="grid"`` console panel both redraw the
        per-site distributions / kernels faithfully (not a flattened line off the fallback arrays).

        The ``data_x`` / ``data_y`` are a VALID no-recipe fallback (each cell's flattened series stacked +
        the cell index) so ``load_figure``'s ``data_x`` / ``data_y`` assertion passes and an older reader
        still gets an array payload -- exactly as the pulse save writes numeric fallback arrays beside its
        recipe.  ``extra_info`` (the caller's metadata) WINS over the auto keys, and its ``kind`` is not
        overridden."""
        import time
        from .data_figure import resolve_save_base
        base = resolve_save_base(path, time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))   # shared path stem (#C4)
        image_path = base.with_suffix(image_ext if str(image_ext).startswith(".") else f".{image_ext}")
        data_path = base.with_suffix(".npz")
        self.fig.savefig(image_path, **kwargs)
        # The faithful reproduction source: the grid's replay recipe (per-cell distributions / kernels +
        # thresholds / fidelities / labels / shape).  Never raises the save -- a cell family with no recipe
        # degrades to the array-only fallback.
        try:
            recipe = grid_recipe_from_cells(self.grid)
        except Exception:
            recipe = None
        # A VALID no-recipe fallback: stack each cell's flat series as data_y columns (padded to a common
        # length so it is a clean 2-D array) and the cell index as data_x, so ``load_figure``'s
        # data_x/data_y assertion passes even for a reader that ignores the recipe.
        columns: list[np.ndarray] = []
        for c in self.cells:
            try:
                columns.append(np.asarray(c.data_y, dtype=float).reshape(-1))
            except Exception:
                columns.append(np.zeros(1, dtype=float))
        width = max((col.size for col in columns), default=1)
        data_y = np.full((len(columns) or 1, width), np.nan, dtype=float)
        for k, col in enumerate(columns):
            data_y[k, : col.size] = col
        data_x = np.arange(data_y.shape[0], dtype=float).reshape(-1, 1)
        info: dict[str, Any] = {"name": str(self.grid.title or "grid")}
        # Only stamp ``kind='grid'`` when a recipe was captured -- the reopen resolves a grid figure by its
        # recipe, so a grid ``kind`` WITHOUT a recipe would send the reopen down ``plot(kind='grid')``
        # (which is rejected, like pulse) and crash.  A recipe-less save therefore has NO ``kind`` and the
        # ``data_x`` / ``data_y`` fallback reopens through shape inference, exactly as the docstring's
        # "degrades to the array-only fallback" promises.
        if recipe is not None:
            info["kind"] = "grid"
            info["figure_recipe"] = recipe
        if extra_info is not None:    # mirror DataFigure.save: caller metadata WINS over the auto keys
            info.update(dict(extra_info))
        np.savez(data_path, data_x=data_x, data_y=data_y, info=info)
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
                 size: str | None = None, labels: Sequence[str] = ("X", "Y"), **kwargs):
        self.cell_renderer = cell
        # The grid's per-site plot kind (``"hist"`` / ``"2d"``) -- the ONE declaration that (a) chose the
        # cell class, (b) dispatches the focus-zoom, and (c) drives the panel's Setting/Edit param UI
        # (PANEL_PARAMS[sub_plot_kind]).  Read straight off the cell so there is a single source.
        self.sub_plot_kind = str(cell.sub_plot_kind)
        self.n_cells = int(cell.n_cells)
        if self.n_cells == 0:
            raise ValueError("a GridPlot needs at least one cell.")
        cell.prepare()
        self.nrows, self.ncols = grid_shape_for(self.n_cells, prefer=grid_shape, max_cols=_SITE_MAX_COLS)
        # The grid's data region scales with the size PRESET like every other kind: the cells / gaps
        # grow via _site_grid_geometry, and a FOCUSED cell fills a data box IDENTICAL to a same-size
        # single-axes panel (panel_plot_spec) so its x label never clips.  ``size=None`` picks the default
        # preset from the grid SHAPE (optimal_grid_size: >=4 cells on an axis -> the double half-unit), the
        # ONE default-size source the reopen / seed paths also use, so a fresh grid and a recorded-size-less
        # reopen agree; an explicit ``size`` (a Setting pick) overrides it.  The recommendation is KEPT
        # (not consumed only by the default): the cell-text rule (_cell_font_scale) compares the current
        # size against it -- a grid squeezed below its recommended preset halves the cell text.
        self._recommended_size = optimal_grid_size(self.nrows, self.ncols)
        self._size = self._recommended_size if size is None else str(size)
        panel_size_cells(self._size)                     # validate against PANEL_SIZES (raises otherwise)
        # cells set their own ticks; the base single-axes smart-tick pass would
        # only touch cell 0 and make it differ from the rest.
        super().__init__(np.arange(self.n_cells), labels=labels, smart_ticks=False, **kwargs)
        self.site_axes: list = []
        self._cell_interactions: list = []
        self._focused: int | None = None
        # The STANDALONE plot-kind figure a focused cell is enlarged into (a HistogramFigure / Live2DDis built
        # by :meth:`build_focus_plotter` onto this grid's own canvas); ``None`` when showing the grid.
        self._focus_plotter: BaseLivePlot | None = None
        # The grid's live DISPLAY knobs (bins / fit / ylog / colorset), applied to a focused cell's full
        # plot-kind figure and re-drawn onto the thumbnails.  Empty = the per-cell defaults.  A saved grid
        # records these (part C) so a reopen restores them.
        self._display_params: dict[str, Any] = {}

    def _create_axes(self):
        # The grid FILLS a total data region == panel_plot_spec(size).data_px (the SAME data box every
        # other panel kind uses at this size) and create_axes_grid SUBDIVIDES it into cells -- so a grid's
        # data region equals a single-axes panel's of the same size (part A), records fig._zlc_fixed_box_in,
        # and a size change truly rescales the cells.
        data_px, col_gap, row_gap, margins, font_scale = _site_grid_geometry(self._size, self._recommended_size)
        axes = create_axes_grid(
            self.fig, self.nrows, self.ncols,
            data_px=data_px, col_gap_px=col_gap,
            row_gap_px=row_gap, margins_px=margins,
        )
        self.axes = axes
        # The PANEL-SIZE-derived cell-text scale (ONE source: _site_grid_geometry) -- consumed by
        # the cell draws (apply_title) and the grid's tick policy.
        self.cell_renderer.font_scale = font_scale
        return axes[0]

    def _style_cell_ticks(self, ax, k: int) -> None:
        """The ONE tick policy for every thumbnail, owned by the GRID (the cell families draw only
        their data artists): the shared-axes convention -- y tick LABELS only on each row's LEFTMOST
        cell, x tick LABELS only on the BOTTOM cell of each column, no ticks anywhere else (which is
        also what keeps the N-cell draw fast, #perf: ticks dominated it).  The cell families share
        their dependent-axis range across cells (pooled clim / shared count top / shared y), so an
        edge label reads for the whole grid.  EVERY (re)draw path funnels back through this method --
        a ylog / fit toggle re-applies it after ``set_yscale`` rebuilt the locators, so the policy
        never silently decays.  Label sizes follow the cell's ``font_scale`` (a panel squeezed
        below the grid's recommended size halves them; every other size uses the standard tick
        size)."""
        scale = float(getattr(self.cell_renderer, "font_scale", 1.0))
        if (k % self.ncols) == 0:               # the row's leftmost cell carries the y labels
            if ax.get_yscale() == "linear":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:                               # log axis: a LogLocator (MaxNLocator misplaces log ticks)
                ax.yaxis.set_major_locator(LogLocator(numticks=3))
                ax.yaxis.set_minor_locator(NullLocator())
            ax.tick_params(axis="y", labelleft=True, labelsize=tick_fontsize() * scale, length=2)
        else:
            ax.set_yticks([])
            if ax.get_yscale() != "linear":     # log installs minor ticks too -- clear them as well
                ax.yaxis.set_minor_locator(NullLocator())
        if (k + self.ncols) >= self.n_cells:    # the column's bottom cell carries the x labels
            # ONE nice tick per cell, nearest the centre: a cell is ~1/6 of a panel wide, so even two
            # standard-size labels collide (with each other or across the gap with the neighbour's).
            lo, hi = ax.get_xlim()
            interior = [t for t in MaxNLocator(nbins=4).tick_values(lo, hi) if lo < t < hi]
            pick = [min(interior, key=lambda t: abs(t - (lo + hi) / 2.0))] if interior else []
            ax.xaxis.set_major_locator(FixedLocator(pick))
            ax.tick_params(axis="x", labelbottom=True, labelsize=tick_fontsize() * scale, length=2)
        else:
            ax.set_xticks([])

    def _initial_draw(self) -> None:
        # Skip the base class's synchronous build-time draw: every real consumer draws again anyway
        # (the Qt panel canvas renders on construction, savefig renders itself), so the ~40% of the
        # N-cell build spent rasterising a figure nobody sees was pure waste (#perf).
        pass

    def init_core(self) -> None:
        self.site_axes = []
        for k, ax in enumerate(self.axes):
            if k >= self.n_cells:
                ax.set_visible(False)
                continue
            self.site_axes.append(ax)
            self.cell_renderer.draw(ax, k)
            self._style_cell_ticks(ax, k)

    def update_core(self) -> None:
        # A recipe (non-faceted) grid is a snapshot; the LIVE facet feed arrives via update_cells.
        pass

    def update_cells(self, per_cell, *, focus=None, draw: bool = False) -> None:
        """The LIVE facet feed: replace every cell's data and move the EXISTING artists (the grid
        counterpart of a standalone kind's ``update``).  ``per_cell`` is the cell-input sequence
        :func:`facet_cells` produces -- one entry per cell, same payload the cell family's
        constructor takes.

        While a cell is ENLARGED only the focus view updates (its kind's ordinary ``update``);
        the thumbnails are redrawn once on unfocus -- the same defer rule every display knob uses.
        The enlarged view is this grid's own (:meth:`focus`, the notebook path) or the HOST's
        ``focus=(plotter, k)`` (the console card enlarges into its own canvas) -- one rule for both.
        Otherwise each thumbnail moves in place (``update_cell``: set_verts / set_array /
        set_data -- never an axes clear, the live-grid perf contract) and only a cell whose artist
        structure changed redraws.  A different CELL COUNT is a structure change the caller owns
        (the console rebuilds the panel); this raises so it is never papered over."""
        cell = self.cell_renderer
        if len(per_cell) != self.n_cells:
            raise ValueError(
                f"update_cells got {len(per_cell)} cells for a {self.n_cells}-cell grid -- a cell-count "
                "change is a rebuild, not an update.")
        for k, data in enumerate(per_cell):
            cell.set_cell_data(k, data)
        cell.prepare()                                   # shared scales (edges / vmax / y range) re-derive
        focus_plotter, focus_k = ((self._focus_plotter, self._focused)
                                  if self._focused is not None else (focus or (None, None)))
        if focus_plotter is not None:
            cell.focus_update(focus_plotter, int(focus_k))
        else:
            for k, ax in enumerate(self.site_axes):
                if not cell.update_cell(ax, k):
                    self._redraw_thumbnail(k, ax)        # artist structure changed -> one full redraw
                else:
                    self._style_cell_ticks(ax, k)        # the centre tick tracks the moved lims
        if draw:
            self.draw()

    def _apply_title(self) -> None:
        labels = self.labels
        self.fig.text(0.5, 0.012, str(labels[0]), ha="center", va="bottom", fontsize=axis_label_fontsize())
        self.fig.text(0.008, 0.5, str(labels[1]) if len(labels) > 1 else "Shots", ha="left", va="center",
                      rotation="vertical", fontsize=axis_label_fontsize())
        apply_title(self.fig, self.title)

    def _install_state(self) -> None:
        self.fig._zlc_state = PlotState(plot_type=self.plot_type)

    def _attach_interactions(self) -> None:
        self._attach_cell_selectors()
        # Focus-zoom: double-click a cell to enlarge it; double-click / Esc to return.  Connected to the
        # CANVAS (which persists across a focus, since the notebook path swaps the figure CONTENT in place),
        # so a double-click on the enlarged view returns to the grid.  Idempotent: only connect once.
        self._connect_focus_zoom()

    def _attach_cell_selectors(self) -> None:
        """Build the per-cell selector bundle (area + cross + zoom + draggable threshold) -- the SAME bundle a
        standalone plot has, filtered by ``event.inaxes``.  Separate from the focus-zoom connection so the grid
        can rebuild its cell selectors on unfocus WITHOUT re-connecting (and double-firing) the canvas-level
        double-click handler."""
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

    def _connect_focus_zoom(self) -> None:
        """Connect the canvas-level double-click / Esc focus-zoom handlers ONCE (idempotent).  The canvas
        persists across a focus (the notebook path clears + rebuilds the figure content on the SAME canvas),
        so these stay live -- a double-click on the enlarged view calls :meth:`unfocus`."""
        if getattr(self, "_click_cid", None) is not None:
            return
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

    def _on_key(self, event) -> None:
        if getattr(event, "key", None) == "escape" and self._focused is not None:
            self.unfocus()

    def build_focus_plotter(self, k: int, *, size: str = "2x2", interactions: bool = True,
                            fig: "plt.Figure | None" = None, **view) -> "BaseLivePlot":
        """Build the STANDALONE plot-kind figure for the enlarged cell ``k`` (part B) -- the ONE reusable
        focus builder both surfaces call.

        The enlarged cell is a REAL ``panel_plot`` of the cell's own KIND, dispatched through the ONE
        ``PLOT_KIND_BY_KEY`` table off :data:`GridCell.sub_plot_kind` (``"hist"`` -> a :class:`HistogramFigure`,
        ``"2d"`` -> a :class:`Live2DDis`) -- never a hard-coded class and never a hand-rolled copy.  So the
        enlarged cell is EXACTLY a standalone panel of that kind: the standard ``size`` geometry (default the
        stock ``2x2``), the full x/y axes, the draggable threshold that live-updates fidelity, the fit chooser
        and the STANDARD relim path (``relim_mode`` / ``fixed_lo`` / ``fixed_hi`` via ``**view``) -- so a lim
        change takes effect and never bounces back to a thumbnail, with no bespoke focus code.

        The cell supplies its data as ``panel_plot`` args (:meth:`GridCell.focus_data`), with the grid's live
        display knobs (bins / fit / ylog / cmap) folded in.  ``fig`` builds it ONTO an existing figure (the
        notebook path swaps it into the grid's own canvas); ``None`` builds a fresh figure (the console path,
        which embeds it in a new canvas)."""
        pk = PLOT_KIND_BY_KEY.get(str(self.cell_renderer.sub_plot_kind))
        if pk is None:                                   # a cell family whose sub_plot_kind is not a real panel kind
            raise ValueError(
                f"grid cell {type(self.cell_renderer).__name__}.sub_plot_kind="
                f"{self.cell_renderer.sub_plot_kind!r} is not a known PLOT_KINDS panel kind")
        # The grid's STORED relim family (apply_param stashed it in _display_params) seeds the enlarged
        # view, so a re-focus (a knob that rebuilt the focus view, the notebook focus(), a re-snapshot)
        # keeps the operator's fixed lim instead of silently reverting to the default autoscale.  An
        # EXPLICIT view kwarg from the caller (the console card's _view_kwargs) still wins (setdefault).
        if "relim" in self._display_params:
            view.setdefault("relim_mode", str(self._display_params["relim"]))
        for lim_key in ("fixed_lo", "fixed_hi"):
            if lim_key in self._display_params:
                view.setdefault(lim_key, float(self._display_params[lim_key]))
        data = dict(self.cell_renderer.focus_data(int(k), display_params=self._display_params))
        data_x = data.pop("data_x")
        data_y = data.pop("data_y", None)
        return panel_plot(data_x, data_y, kind=pk.key, size=size, interactions=interactions,
                          fig=fig, **view, **data)

    def focus(self, k: int) -> None:
        """Enlarge cell ``k`` to a FULL standalone plot-kind figure ON THE GRID'S OWN CANVAS (part B) -- call
        again or :meth:`unfocus` to return to the grid.  This is the NOTEBOOK / bare-figure path; inside a
        TaskConsole the host (``PanelCard``) builds the SAME :meth:`build_focus_plotter` into its own canvas.

        The enlarged cell is NOT a simplified thumbnail: it is a real ``panel_plot`` of the cell's KIND (a
        :class:`HistogramFigure` for a distribution cell, a :class:`Live2DDis` for a kernel cell) built onto
        the grid's figure via :meth:`build_focus_plotter`, so it carries the draggable threshold that
        live-updates the fidelity readout, the fit chooser, full x/y axes and the STANDARD relim path -- a lim
        change sticks and never bounces back.  Since a grid's total figure px equals a same-size single-axes
        panel's (#5), building the ``2x2`` standalone view onto this figure is seamless (no resize jump)."""
        if self._focused is not None:
            self.unfocus()
        k = int(k)
        # Tear down the grid's per-cell selectors + clear the figure, then build the standalone plot-kind view
        # ONTO the same figure (fig=self.fig) so it lives in the grid's own canvas -- a genuine standalone
        # panel_plot, not an axes overlay.  The canvas-level focus-zoom double-click / Esc handlers STAY
        # connected (they live on the persistent canvas), so a double-click on the enlarged view returns to
        # the grid.  The 2x2 preset is the stock single-axes panel size.
        self._teardown_cell_selectors()
        self.fig.clear()
        self._focus_plotter = self.build_focus_plotter(
            k, size="2x2", interactions=self.interactions, fig=self.fig)
        self._focused = k
        self._sync_canvas_design_size()   # the figure is now stock 2x2 -- an embedded canvas follows
        self.draw()

    def unfocus(self) -> None:
        """Return from the enlarged single-cell view to the grid (rebuild the grid onto the same figure)."""
        if self._focused is None:
            return
        # A focus-view threshold drag is copied back onto the cell (the single source the grid thumbnail +
        # save recipe read) so an enlarged-view cut is not lost when the grid returns.
        if self._focus_plotter is not None and hasattr(self.cell_renderer, "sync_threshold_from_focus"):
            self.cell_renderer.sync_threshold_from_focus(self._focused, self._focus_plotter)
        for handler in self.interaction_handles_of(self._focus_plotter):
            if hasattr(handler, "destroy"):
                handler.destroy()
        self._focus_plotter = None
        self._focused = None
        self._rebuild_grid()                            # redraw the grid onto the same figure
        self._sync_canvas_design_size()   # the figure is grid-sized again -- an embedded canvas follows
        self.draw()

    def _sync_canvas_design_size(self) -> None:
        """After a focus/unfocus swapped THIS figure's size (stock 2x2 <-> the grid size), let an
        embedded Qt canvas re-pin its widget to the new design size (duck-typed: the Qt canvas
        exposes ``refresh_design_size``; every other backend follows the figure natively).  Without
        it the pinned widget stretched the 2x2 buffer over the whole grid-sized panel."""
        refresh = getattr(getattr(self.fig, "canvas", None), "refresh_design_size", None)
        if callable(refresh):
            refresh()

    def _teardown_cell_selectors(self) -> None:
        """Destroy the grid's PER-CELL selectors so the cleared figure carries no stale per-axes callbacks
        when the standalone focus view is built onto it.  The canvas-level focus-zoom double-click / Esc
        handlers are LEFT connected (they persist across the focus so the enlarged view can return)."""
        for handler in self.interaction_handles():
            if hasattr(handler, "destroy"):
                try:
                    handler.destroy()
                except Exception:
                    pass
        self._cell_interactions = []
        self.fig._zlc_tools = None
        self.fig._zlc_grid_tools = None

    def _rebuild_grid(self) -> None:
        """Redraw the GRID onto the (cleared) figure -- the return leg of :meth:`unfocus`.  Replays the SAME
        lifecycle steps ``show`` runs for the grid (axes -> cells -> outer labels -> per-cell selectors), so
        the returned grid is identical to a fresh one, with its cells / thumbnails / selectors back.  The
        canvas-level focus-zoom handlers were never disconnected (idempotent :meth:`_connect_focus_zoom`)."""
        self.fig.clear()
        self.ax = self._create_axes()
        self.init_core()
        self._apply_title()
        self._install_state()
        if self.interactions:
            self._attach_interactions()

    @staticmethod
    def interaction_handles_of(plotter) -> list:
        """Every selector object a focus plotter attached (so unfocus can destroy them) -- reuses the
        plotter's own :meth:`BaseLivePlot.interaction_handles` list, never a second enumeration."""
        if plotter is None:
            return []
        try:
            return list(plotter.interaction_handles())
        except Exception:
            return []

    def store_display_param(self, key: str, value) -> bool:
        """Record a DISPLAY knob on the grid WITHOUT drawing: stash it in ``_display_params`` (the save
        recipe + the focus seed read it) and update the cell renderer's own state for a knob that changes
        the THUMBNAILS (``bins`` re-bins a hist grid, ``cmap`` recolours a 2d grid).  Returns True when the
        thumbnails are now STALE (the caller decides when to redraw them -- :meth:`apply_param` redraws at
        once when the grid is showing; the console's focused path defers to unfocus, so a Setting edit on an
        ENLARGED cell never synchronously repaints N invisible cells)."""
        self._display_params[str(key)] = value
        # The relim family updates the CELL state exactly like bins/cmap (GridCell.relim_mode /
        # fixed_lo / fixed_hi -- every thumbnail's draw consumes it via thumb_lims), so a lim edit
        # reaches the thumbnails through the SAME store -> cell-state -> redraw mechanism, never
        # "only the enlarged view changed".  An UNCHANGED value returns False (not dirty): the
        # console re-pushes the whole relim family after every rebuild (_apply_display_params),
        # and a no-change push must not cost an N-cell thumbnail repaint.
        if str(key) == "relim":
            cell = self.cell_renderer
            if cell.relim_mode == str(value):
                return False
            if str(value) == "fixed":
                # Flipping INTO fixed freezes what is showing NOW -- the SAME rule
                # BaseLivePlot.apply_param applies to a standalone panel, landed on the CELL
                # state (and _display_params, so a refocus / save / re-snapshot keeps it): the
                # zoomed cell's own lims when enlarged, else the cells' shared auto range.
                lo, hi = (self._focus_plotter.current_lims() if self._focus_plotter is not None
                          else cell.auto_lims())
                cell.fixed_lo, cell.fixed_hi = float(lo), float(hi)
                self._display_params.setdefault("fixed_lo", float(lo))
                self._display_params.setdefault("fixed_hi", float(hi))
            cell.relim_mode = str(value)
            return True
        if str(key) in ("fixed_lo", "fixed_hi"):
            if getattr(self.cell_renderer, str(key)) == float(value):
                return False
            setattr(self.cell_renderer, str(key), float(value))
            return self.cell_renderer.relim_mode == "fixed"   # thumbnails only move in fixed mode
        # Every other key is the CELL FAMILY's own knob (a hist's bins/fit/ylog, a 2d's cmap) -- the
        # family declares what it renders (GridCell.consume_param), the grid never key-matches per kind.
        return self.cell_renderer.consume_param(str(key), value)

    def current_lims(self) -> tuple[float, float]:
        """What the grid 'shows now' as one (lo, hi) pair: the enlarged cell's own lims when zoomed,
        else the cells' SHARED auto range (:meth:`GridCell.auto_lims`).  Overrides the base class --
        a grid never maintains ``ylim_min``/``ylim_max`` (its update_core is a no-op), so the base
        pair would sit at the constructor's 0..1 forever and a fixed flip would seed garbage."""
        if self._focus_plotter is not None:
            return self._focus_plotter.current_lims()
        if self.cell_renderer.relim_mode == "fixed":
            return float(self.cell_renderer.fixed_lo), float(self.cell_renderer.fixed_hi)
        return self.cell_renderer.auto_lims()

    def apply_param(self, key: str, value) -> bool:
        """A DISPLAY-ONLY grid knob (part B): store it (:meth:`store_display_param`), apply it to the
        FOCUSED cell's full plot-kind figure in place (so a Setting change reaches the enlarged cell --
        ylog / fit / bins / colorset), and for a knob that changes the THUMBNAILS re-draw the grid cells
        too.  The stored params are saved with the grid (part C) so a reopen restores them.  Returns True
        (a grid handles its display knobs itself, so the console never falls back to a full
        teardown/rebuild)."""
        thumb_dirty = self.store_display_param(key, value)
        if thumb_dirty and self._focused is None:        # only redraw thumbnails when not zoomed in
            self._redraw_thumbnails()
        # Apply to the focused STANDALONE figure in place (HistogramFigure / Live2DDis / the base
        # relim family).  A key the enlarged view has NO in-place use for (e.g. ``repeat_mode``) was
        # STORED above and simply does not touch the view -- NEVER an unfocus/refocus rebuild: every
        # per-site-kind knob (bins / fit / ylog / cmap / relim family) applies in place, so a rebuild
        # here could only mean flashing the whole grid for a knob that changes nothing visible (the
        # "adjusting repeat_mode bounces the enlarged cell" bug).  The stored param still seeds the
        # next build_focus_plotter, so nothing is lost.
        if self._focused is not None and self._focus_plotter is not None:
            self._focus_plotter.apply_param(str(key), value)
        self.draw()
        return True

    def _redraw_thumbnail(self, k: int, ax) -> None:
        """Fully re-draw ONE cell thumbnail (axes cleared) -- the fallback when an in-place move cannot
        represent the change.  Mirrors the :meth:`init_core` cell step (draw + the one tick policy)."""
        ax.clear()
        self.cell_renderer.draw(ax, k)
        self._style_cell_ticks(ax, k)

    def _redraw_thumbnails(self) -> None:
        """Refresh every thumbnail after a display knob changed: IN PLACE first (update_cell moves /
        rebuilds only the data artists -- ticks, title and threshold stay), a full axes-clear redraw
        only when the cell family says the change cannot be represented in place.  Clearing all N
        axes re-ran the whole per-cell text/tick build and made every Setting edit cost a rebuild
        (#perf: 35 cells ~420 ms -> in-place ~tens of ms)."""
        for k, ax in enumerate(self.site_axes):
            if not self.cell_renderer.update_cell(ax, k):
                self._redraw_thumbnail(k, ax)
            else:
                # An in-place knob (ylog's set_yscale) may have rebuilt the axis locators --
                # re-assert the ONE tick policy so the edge-label rule never silently decays.
                self._style_cell_ticks(ax, k)

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


def _append_grid_plot_kind() -> None:
    """Register the per-site GRID kind in the ONE ``PLOT_KINDS`` table, now that ``GridPlot`` exists.

    ``grid`` is a first-class LOADABLE panel kind, seeded from a saved grid figure the SAME way ``pulse``
    is (``panel=False`` = not offered in the live Add-Panel dropdown, but a real panel kind on the SEED
    path).  Its cell content (hist / image) is carried in the saved ``figure_recipe`` and rebuilt by
    :func:`build_grid_figure`.  It cannot live in the ``PLOT_KINDS`` literal above because ``GridPlot`` is
    defined AFTER that literal, so it is appended here (still a single-source entry -- ``PLOT_KIND_BY_KEY``
    is rebuilt so the two stay in lock-step)."""
    global PLOT_KINDS, PLOT_KIND_BY_KEY
    if "grid" in PLOT_KIND_BY_KEY:
        return
    PLOT_KINDS = PLOT_KINDS + (
        # panel=True: since the FACET design a grid is a live Add-Panel kind like any other -- bind a
        # measurement block signal and pick the facet axis in Setting (the recipe/snapshot path is the
        # facet="" case, it no longer makes the grid seed-only).
        PlotKind(key="grid", cls=GridPlot, label="Site grid", render_family="1D", panel=True),
    )
    PLOT_KIND_BY_KEY = {pk.key: pk for pk in PLOT_KINDS}


_append_grid_plot_kind()


#: ``sub_plot_kind -> GridCell class`` -- the SINGLE source that turns a grid's declared per-site plot kind
#: into the cell strategy that draws it.  The unified :func:`grid` factory and :func:`build_grid_figure`
#: both look up here (never a hard-coded ``if kind == ...``), so ADDING a per-site kind to a grid is ONE
#: entry here + the cell class (which already declares its ``sub_plot_kind``) -- nothing else to touch.
class LineCell(GridCell):
    """A GridPlot cell that is a 1-D curve -- one thin line per cell.  The facet family for a sliced
    scan whose remaining axes flatten to a curve (e.g. a 3-level scan faceted on TWO axes, or a
    per-repeat trace).  ``per_cell`` is a sequence of 1-D y vectors; ``x`` is the shared coordinate
    (``None`` = the point index).  The enlarged focus view is a standalone ``1d`` panel
    (:class:`Live1D`), so fit / relim / zoom all arrive on the standard path."""

    sub_plot_kind = "1d"         # each cell is a ``1d`` per-site plot (enlarged -> a standalone Live1D)

    def __init__(self, per_cell, *, x=None, labels: Sequence[str] = ("X", "Y")):
        self.ys = [np.asarray(y, dtype=float).reshape(-1) for y in per_cell]
        self.n_cells = len(self.ys)
        if self.n_cells == 0:
            raise ValueError("per_cell must contain at least one curve.")
        self.x = None if x is None else np.asarray(x, dtype=float).reshape(-1)
        self.labels = tuple(labels)
        self._line_artists: list = [None] * self.n_cells    # cell k's Line2D, for the live in-place move

    def _cell_x(self, k: int) -> np.ndarray:
        y = self.ys[k]
        if self.x is not None and self.x.size == y.size:
            return self.x
        return np.arange(y.size, dtype=float)

    def prepare(self) -> None:
        finite = [y[np.isfinite(y)] for y in self.ys if y.size]
        pooled = np.concatenate(finite) if finite else np.array([0.0, 1.0])
        lo = float(np.nanmin(pooled)) if pooled.size else 0.0
        hi = float(np.nanmax(pooled)) if pooled.size else 1.0
        pad = 0.08 * ((hi - lo) or (abs(hi) or 1.0))
        self.y_lo, self.y_hi = lo - pad, hi + pad            # one shared y range -> cells comparable

    def auto_lims(self) -> tuple[float, float]:
        """The ONE shared y range every cell already draws with -- the line grid's
        'what you see now' pair."""
        if not hasattr(self, "y_lo"):
            self.prepare()
        return float(self.y_lo), float(self.y_hi)

    def draw(self, ax, k: int):
        if not hasattr(self, "y_lo"):
            self.prepare()
        x = self._cell_x(k)
        (line,) = ax.plot(x, self.ys[k], color=PALETTE["line_single"])
        self._line_artists[k] = line
        if x.size:
            ax.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0)
        ax.set_ylim(*self.thumb_lims(self.y_lo, self.y_hi))
        apply_title(ax, f"s{k}", size=small_fontsize() * self.font_scale, pad=1.5)   # ticks are the grid's ONE policy
        return None

    def set_cell_data(self, k: int, data) -> None:
        self.ys[k] = np.asarray(data, dtype=float).reshape(-1)
        if hasattr(self, "y_lo"):
            del self.y_lo, self.y_hi           # curves moved -> the shared y range re-derives (prepare)

    def update_cell(self, ax, k: int) -> bool:
        """Move cell ``k``'s EXISTING line to the current curve -- no axes clear, no new artists.
        A curve whose LENGTH changed moves the same Line2D too (set_data replaces both arrays)."""
        if not hasattr(self, "y_lo"):
            self.prepare()
        line = self._line_artists[k]
        if line is None:
            return False
        x = self._cell_x(k)
        line.set_data(x, self.ys[k])
        if x.size:
            ax.set_xlim(float(x[0]), float(x[-1]) if x[-1] > x[0] else float(x[0]) + 1.0)
        ax.set_ylim(*self.thumb_lims(self.y_lo, self.y_hi))
        return True

    def focus_update(self, focus_plotter, k: int) -> None:
        focus_plotter.update(self.ys[k])       # the standalone 1d redraws its single curve

    def data_figure(self, fig, ax, k: int):
        from .data_figure import DataFigure
        ylabel = self.labels[1] if len(self.labels) > 1 else "Y"
        return DataFigure(fig=fig, ax=ax, data_x=self._cell_x(k), data_y=self.ys[k],
                          labels=(self.labels[0], ylabel), name=f"site{k}")

    def focus_data(self, k: int, *, display_params: Mapping[str, Any] | None = None) -> dict:
        """The ``panel_plot(kind="1d")`` args for the STANDALONE enlarged view of curve cell ``k`` --
        the SAME ``1d`` panel a standalone line uses (full axes, fit, relim), reusing the ONE
        plot-kind renderer.  :meth:`GridPlot.build_focus_plotter` merges this into ``panel_plot``."""
        return {
            "data_x": self._cell_x(k),
            "data_y": self.ys[k],
            "labels": (self.labels[0], self.labels[1] if len(self.labels) > 1 else "Y", "Z"),
            "title": f"site {k}",
        }


GRID_CELL_BY_KIND: dict[str, type[GridCell]] = {"hist": HistogramCell, "2d": ImageCell, "1d": LineCell}


def grid(
    per_cell,
    *,
    sub_plot_kind: str = "hist",
    facet=None,
    points_shape=(),
    repeat_mode: str = "average",
    grid_shape: tuple[int, int] | None = None,
    size: str | None = None,
    labels: Sequence[str] | None = None,
    title: str = "",
    display: bool = True,
    fig: "plt.Figure | None" = None,
    interactions: bool = True,
    **cell_kwargs,
):
    """The ONE per-site GRID factory: N aligned cells, each a ``sub_plot_kind`` PER-SITE plot.

    ``sub_plot_kind`` is a :data:`PLOT_KINDS` key (``"hist"`` = one distribution per site, ``"2d"`` = one
    image per site) -- the SINGLE declaration that drives (a) which cell class draws the grid
    (:data:`GRID_CELL_BY_KIND`), (b) the double-click focus-zoom (each cell enlarges to a standalone panel of
    that kind), and (c) which ``PANEL_PARAMS`` the console panel's Setting / Edit UI shows.  The remaining
    ``**cell_kwargs`` are that kind's own cell arguments (a ``hist`` grid takes ``thresholds`` / ``occupied`` /
    ``bins`` / ``site_fidelities``; a ``2d`` grid takes ``cmap``).

    EVERY grid is built through THIS entry -- the calibration report, the notebook helpers
    (:func:`site_histogram_grid` / :func:`site_psf_grid` are one-line ``sub_plot_kind`` presets over it), and
    the reopened-recipe path (:func:`build_grid_figure`) -- so a grid is created the SAME way everywhere:
    pick the per-site kind, hand the per-cell data.  Geometry / gaps / colours / dpi / fonts are frontend-owned
    (cells never overlap / clip, all cells align).  Returns the :class:`GridPlot` plotter (already ``show``-n)."""
    kind = str(sub_plot_kind)
    if kind not in GRID_CELL_BY_KIND:
        raise ValueError(
            f"unknown grid sub_plot_kind {sub_plot_kind!r}; choose from {sorted(GRID_CELL_BY_KIND)}.")
    if facet is not None:
        # the grid as an AXIS-EXPANDER: hand ONE (repeat, points, *data_dim) block + the axis to
        # expand, and the single slicing rule (facet_cells) produces the per-cell inputs
        per_cell = facet_cells(per_cell, facet, sub_plot_kind=kind,
                               points_shape=points_shape, repeat_mode=repeat_mode)
    if len(per_cell) > MAX_GRID_CELLS:
        raise ValueError(
            f"a grid of {len(per_cell)} cells would freeze the UI (limit {MAX_GRID_CELLS}); "
            "pick a shorter facet axis, or collapse that axis via repeat_mode / an expression instead.")
    if labels is None:
        labels = {"hist": ("Signal", "Shots"), "1d": ("X", "Y")}.get(kind, ("x (px)", "y (px)"))
    if kind == "hist":
        # the readout hist grid keeps its named type + property surface (n_sites / thresholds / classify);
        # it builds the HistogramCell itself, so the hist cell kwargs flow straight through.
        return SiteHistogramGrid(
            per_cell, grid_shape=grid_shape, labels=labels, title=title, size=size,
            fig=fig, interactions=interactions, **cell_kwargs).show(display=display)
    cell = GRID_CELL_BY_KIND[kind](per_cell, labels=labels, **cell_kwargs)
    return GridPlot(cell, grid_shape=grid_shape, labels=labels, title=title, size=size,
                    fig=fig, interactions=interactions).show(display=display)


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
    size: str | None = None,
    display: bool = True,
) -> SiteHistogramGrid:
    """Plot one histogram per site on an aligned, non-overlapping grid (general N).

    Returns a :class:`SiteHistogramGrid` (a :class:`GridPlot` of histogram cells):
    a first-class plot, so it is interactive (per-cell selectors + a draggable
    per-site threshold, and **double-click a cell to enlarge it to see detail**)
    and every cell exposes a :class:`DataFigure` via ``grid.to_data_figure().cell(k)``.
    Geometry/gaps/colours/dpi/fonts are owned by the frontend (cells never overlap,
    nothing is cut off, all cells align)."""

    return grid(
        per_site_values, sub_plot_kind="hist", thresholds=thresholds, occupied=occupied,
        grid_shape=grid_shape, bins=bins, site_fidelities=site_fidelities, labels=labels,
        title=title, size=size, display=display)


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

    return grid(images, sub_plot_kind="2d", labels=labels, title=title, display=display, **kwargs)


def grid_recipe_from_cells(grid: "GridPlot") -> dict:
    """The REPLAY RECIPE a :class:`GridPlot` stores so it can be re-rendered FAITHFULLY on reopen -- the
    grid counterpart of a pulse figure's ``figure_recipe``.  A grid's per-cell distributions / kernels
    cannot be recovered from the flat ``data_x`` / ``data_y`` a bare save writes (``data_x`` is just the
    cell index), so the recipe carries everything :func:`build_grid_figure` needs to rebuild the exact
    grid: the ``sub_plot_kind`` (``"hist"`` per-site distribution / ``"2d"`` per-site kernel), the per-cell
    payload, and the shared thresholds / fidelities / labels / title / grid shape.

    Dispatch is by the concrete :class:`GridCell` the grid was built with, so a NEW grid-cell family adds
    a branch here (and a matching one in :func:`build_grid_figure`) without touching either seam's callers.
    Append-only keys -- the ``.npz`` structure is unchanged, so an older reader still reads the file.

    Beyond the per-cell payload the recipe records the VIEW STATE so a reopen looks exactly as it did when
    saved (part C): the ``panel_size`` (the preset the grid was drawn at), the ``focused_cell_index`` (which
    cell, if any, was enlarged), and the live ``display_params`` (bins / fit / ylog / colorset).  ``build_grid_figure``
    restores all three, so ``na.load_figure(grid_npz).plot()`` and the figure viewer reopen a grid at its saved
    size, display knobs and even zoomed into the same cell -- and the ``sub_plot_kind`` drives the reopen
    renderer (a hist grid -> the hist grid, NEVER the site map)."""
    cell = grid.cell_renderer
    recipe: dict[str, Any] = {
        "kind": "grid",
        "labels": [str(x) for x in getattr(cell, "labels", grid.labels)],
        "title": str(grid.title),
        "grid_shape": [int(grid.nrows), int(grid.ncols)],
        "panel_size": str(getattr(grid, "_size", "2x2")),
        "focused_cell_index": (None if getattr(grid, "_focused", None) is None else int(grid._focused)),
        "display_params": dict(getattr(grid, "_display_params", {}) or {}),
    }
    # ``sub_plot_kind`` (the grid's per-site plot kind: "hist" / "2d") is the ONE dispatch key -- it selects
    # the cell class on reopen (GRID_CELL_BY_KIND) exactly as it did on create.  The per-cell payload branch
    # is keyed off the concrete GridCell type (each family stores its own data shape).
    recipe["sub_plot_kind"] = str(cell.sub_plot_kind)
    if isinstance(cell, HistogramCell):
        recipe["per_cell"] = [np.asarray(v, dtype=float).reshape(-1).tolist() for v in cell.values]
        recipe["bins"] = int(cell.bins_arg)
        recipe["thresholds"] = None if cell.thresholds is None else [float(t) for t in cell.thresholds]
        recipe["fidelities"] = (None if cell.site_fidelities is None
                                else [float(f) for f in np.asarray(cell.site_fidelities).reshape(-1)])
        recipe["occupied"] = (None if cell.occupied is None
                              else [np.asarray(o, dtype=float).reshape(-1).tolist() for o in cell.occupied])
    elif isinstance(cell, ImageCell):
        recipe["per_cell"] = [np.asarray(im, dtype=float).tolist() for im in cell.images]
        recipe["cmap"] = str(cell.cmap)
    elif isinstance(cell, LineCell):
        recipe["per_cell"] = [np.asarray(y, dtype=float).reshape(-1).tolist() for y in cell.ys]
        recipe["x"] = None if cell.x is None else np.asarray(cell.x, dtype=float).reshape(-1).tolist()
    else:
        raise TypeError(f"grid cell {type(cell).__name__} has no replay recipe")
    return recipe


def build_grid_figure(recipe: Mapping[str, Any], *, fig: plt.Figure | None = None,
                      interactions: bool = True, size: str | None = None, display: bool = False):
    """Rebuild a per-site GRID figure from its :func:`grid_recipe_from_cells` recipe -- the SINGLE source
    of the grid reproduction, the grid counterpart of :func:`build_pulse_preview_plot`.

    ``recipe['sub_plot_kind']`` selects the family: ``"hist"`` rebuilds the per-site distribution grid
    (:class:`SiteHistogramGrid`, one histogram per site with its threshold line + fidelity annotation);
    ``"2d"`` rebuilds the per-site kernel grid (a :class:`GridPlot` of :class:`ImageCell`, one imshow per
    site).  It is the SAME ``sub_plot_kind`` the unified :func:`grid` factory takes on create.  Every
    consumer -- ``na.load_figure(npz).plot()``, the figure viewer, a seeded ``kind="grid"`` console panel
    -- draws through THIS one builder, so all three reproduce the identical faithful grid (never a
    flattened 1-D line off the fallback arrays).  Returns the :class:`GridPlot` plotter (already ``show``-n
    with ``display``).

    ``size`` is one of ``PANEL_SIZES`` -- the grid's cells scale with it (like every other panel kind), and
    a focused cell fills a data box with the same-size panel's margins.  ``size=None`` opens at the size the
    grid was SAVED at (``recipe['panel_size']``) if recorded, else the shape-driven :func:`optimal_grid_size`
    default -- so a reopen matches how it was saved and a fresh grid picks its natural preset (ONE source).
    ``interactions=False`` builds a display-only grid (the read-only Monitor card) with no selectors.  The
    recipe's saved DISPLAY params (bins / fit / ylog for a hist grid) and any ``focused_cell_index`` are
    re-applied after the build, so a reopened grid looks EXACTLY as it did when saved (part C)."""
    # Saved-DATA migration (not an API compat shim): npz written before the sub_plot_kind rename
    # recorded the family as recipe['cell'] = 'hist'/'image' -- map it so an operator's stored
    # image grid never silently reopens as a histogram of flattened pixels.
    legacy = {"image": "2d", "hist": "hist"}.get(str(recipe.get("cell") or ""))
    sub_plot_kind = str(recipe.get("sub_plot_kind") or legacy or "hist")
    labels = tuple(str(x) for x in (recipe.get("labels") or ("X", "Y")))
    title = str(recipe.get("title") or "")
    grid_shape = recipe.get("grid_shape")
    grid_shape = tuple(int(n) for n in grid_shape) if grid_shape else None
    # size resolution: an explicit arg wins; else the saved panel_size; else the shape-driven default
    # (GridPlot's own size=None path).  ONE rule shared with the pulse reopen's panel_size handling.
    if size is None:
        recorded = recipe.get("panel_size")
        size = str(recorded) if recorded else None
    display_params = dict(recipe.get("display_params") or {})
    focused = recipe.get("focused_cell_index")
    if sub_plot_kind == "2d":
        images = [np.asarray(im, dtype=float) for im in (recipe.get("per_cell") or [])]
        cmap = recipe.get("cmap") or (display_params.get("cmap") or display_params.get("colorset"))
        plotter = GridPlot(ImageCell(images, labels=labels, cmap=cmap), grid_shape=grid_shape, labels=labels,
                           title=title, fig=fig, interactions=interactions, size=size).show(display=display)
    elif sub_plot_kind == "1d":
        curves = [np.asarray(y, dtype=float).reshape(-1) for y in (recipe.get("per_cell") or [])]
        x = recipe.get("x")
        plotter = GridPlot(LineCell(curves, x=None if x is None else np.asarray(x, dtype=float), labels=labels),
                           grid_shape=grid_shape, labels=labels,
                           title=title, fig=fig, interactions=interactions, size=size).show(display=display)
    else:
        # per-site distribution grid (the default)
        per_site = [np.asarray(v, dtype=float).reshape(-1) for v in (recipe.get("per_cell") or [])]
        thresholds = recipe.get("thresholds")
        fidelities = recipe.get("fidelities")
        occupied = recipe.get("occupied")
        plotter = SiteHistogramGrid(
            per_site, thresholds=thresholds, site_fidelities=fidelities,
            occupied=None if occupied is None else [np.asarray(o, dtype=float).reshape(-1) for o in occupied],
            grid_shape=grid_shape, bins=int(recipe.get("bins", 36)), labels=labels, title=title,
            fig=fig, interactions=interactions, size=size).show(display=display)
    for key, value in display_params.items():          # restore saved display knobs (bins / fit / ylog ...)
        plotter.apply_param(key, value)
    if focused is not None and 0 <= int(focused) < plotter.n_cells:
        plotter.focus(int(focused))                    # reopen already zoomed into the cell it was saved on
    return plotter


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
        if spec is None or spec.key in ("hist", "pulse", "grid"):
            # ``grid`` has a per-cell construction (a GridCell, not a data_x/data_y array), so it is built
            # through ``build_grid_figure`` / ``site_histogram_grid`` / ``site_psf_grid``, never this
            # array factory -- the SAME reason ``pulse`` is excluded here.
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
    "ImageCell",
    "GRID_CELL_BY_KIND",
    "grid",
    "kind_for_plotter",
    "build_grid_figure",
    "grid_recipe_from_cells",
    "optimal_grid_size",
    "recommended_grid_size",
    "default_pulse_size",
    "optimal_pulse_size",
    "panel_margins_px",
    "pulse_drawn_rows",
    "panel_plot",
    "panel_plot_spec",
    "panel_size_cells",
    "plot",
    "site_histogram_grid",
    "site_psf_grid",
    "site_ring_radius",
    "pulse_plot_channels",
    "pulse_plot_spec",
    "pulse_repeat_markers",
    "pulse_repeat_notation",
    "build_pulse_preview_plot",
    "analog_bus_traces",
    "annotate_pulse_variable_regions",
    "bus_signed_bounds",
    "bus_display_label",
]
