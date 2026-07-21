"""The task console's Qt x matplotlib zone: the panel card, its board, and the card geometry.

Everything here HOLDS both worlds at once -- a card owns its matplotlib plotter and canvas
inside a Fluent Qt group box; the geometry bridge sizes Qt card chrome from figure pixels.
Neither ``zlc_frontend/qt_widgets`` (may not import matplotlib) nor ``zlc_data`` (no toolkit
at all) may hold that marriage, which is exactly what this transitional zone exists for: it
empties in the render-purification pass, when the worker rasterises off-thread and Qt sees
only pixels.

Every import names a TRUE owner -- nothing in this module touches the legacy tree, so
deleting ``Zou_lab_control`` cannot orphan it.
"""

from __future__ import annotations

import time
from typing import Mapping, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    CARD_PAD,
    CARD_TITLE_PX,
    FluentButton,
    FluentComboBox,
    FluentFloatingEditor,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentTreeComboBox,
    GREEN,
    GREY,
    ORANGE,
    RED,
    fluent_scrollbar_thickness,
    fluent_text_width,
    popup_gap,
    scaled_px,
    signals_blocked as _signals_blocked,
)
from zlc_frontend import board_layout as _layout
from zlc_frontend.form import lenient_float as _safe_float
from zlc_frontend.live_plot.live import (
    coerce_panel_value,
    normalize_facet,
    panel_display_size,
    panel_plot,
    site_ring_radius,
)
from zlc_frontend.panel_params import (
    PANEL_PARAMS,
    panel_display_decls as _panel_display_decls,
    resolved_cmap as _resolved_cmap,
    resolved_param as _resolved_param,
)
from zlc_data.console_records import (
    BLANK_SOURCE as _BLANK_SOURCE,
    DEFAULT_UPDATE_MS,
    PANEL_INPUT_FORMAT,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
    panel_allows_multi_slot,
    panel_input_slots,
    repeat_mode_for_kind as _repeat_mode_for_kind,
    repeat_modes_for_kind as _repeat_modes_for_kind,
)
from zlc_data.curve_fitting import build_fit_request
from zlc_data.panel_size import PANEL_SIZES, panel_size_cells
from zlc_data.param_decl import ParamDecl
from zlc_data.signal_expr import SIGNAL_EXPR_HELP

# qt_widgets submodules are reached as ATTRIBUTES of the one facade binding: their names are
# deliberately absent from the facade __all__, and the package forbids outside deep imports.
AnalysisControls = _qt_widgets.analysis_controls.AnalysisControls
_general_fit_models_for_kind = _qt_widgets.analysis_controls._general_fit_models_for_kind
PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
coerce_short_labels = _qt_widgets.param_widgets.coerce_short_labels
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo

# Mirrors the legacy shell's guard: a matplotlib install without the Qt backend still lets the
# headless documents import; only actually BUILDING a panel canvas needs the backend.
try:
    import matplotlib.pyplot as plt
    from .plot_bridge_canvas import panel_canvas
except Exception:  # pragma: no cover - depends on the local matplotlib install
    panel_canvas = None


#: Per-signal publish counters ({name: version}) so a rolling monitor tells a new sample of
#: its OWN source from an unrelated node's version bump.
SIG_VERSIONS_KEY = "__sig_versions__"

#: Shot-coherent per-signal physical validity masks ({name: (R,P) bool}).
SIG_VALID_KEY = "__sig_valid__"

#: Coordinate frames ({signal_name: [x0, x1, y0, y1]}) from any node whose acquisition source
#: declares a ROI -- a 2D panel puts its axes in real camera pixels.
COORD_FRAMES_KEY = "__coord_frames__"

# A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so
# it shows nothing until the user picks a hub signal in its Setting (signal_combo)
# -- it must NOT auto-bind to any node's signal.  An empty source is the blank
# state; ``refresh`` treats it (and a source that produces None) as "pick a signal"
# rather than an error, so a blank panel sits quietly until wired.

def _card_y_is_view_axis(card) -> bool:
    """Does THIS panel's y axis take a view-window pin -- an image, where x AND y are spatial
    (pixel) coordinates and the value lives on the colour limit?  Reads the LIVE object's own
    ``y_is_view_axis`` flag -- the plot class's for a flat panel, the CELL family's for a grid
    (so a grid of image cells offers the pin and a hist/1d grid does not) -- falling back to the
    kind's plot class in ``PLOT_KIND_BY_KEY`` before a plotter exists.  The ONE resolver every
    Edit surface keys the "y range" row off, mirroring how the plot side gates ``view_ylim`` in
    ``apply_param`` / ``consume_param`` on the same flag: UI and apply can never disagree."""
    from zlc_frontend.live_plot.live import PLOT_KIND_BY_KEY

    plotter = getattr(card, "plotter", None)
    cell = getattr(plotter, "cell_renderer", None)
    if cell is not None:
        return bool(getattr(cell, "y_is_view_axis", False))
    if plotter is not None:
        return bool(getattr(plotter, "y_is_view_axis", False))
    pk = PLOT_KIND_BY_KEY.get(str(getattr(card.config, "kind", "")))
    return bool(getattr(pk.cls, "y_is_view_axis", False)) if pk is not None else False

# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH still scales with the size (``cols // 2`` base-widths so 1x4 is wider than 1x2);
# HEIGHT HUGS the plot -- the card is exactly tall enough for its figure + chrome, with NO blank
# padding below (every size hugs like 1x2, #H3i-3).  ``PanelConfig.col`` is the card's pixel X and
# ``row`` is the card's pixel Y; :func:`pack` is the order-driven TOP-LEFT GRAVITY packer that places
# every card at the first free NW slot in list order.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
# content padding) belongs to the FluentGroupBox COMPONENT (qt_widgets.CARD_PAD / CARD_TITLE_PX,
# the single source); this module only lays cards out.
GRID_UNIT = 8

# The ONE spacing setting (#H3s-F8).  GAP is the UNIFORM clear distance between any two cards on
# every side -- top, bottom, left, right -- AND the board margin from the (0, 0) origin.  It equals
# the HORIZONTAL inter-card gap the user likes: two cards on adjacent base-columns pitched by
# ``_cell_size()[0] + GAP`` sit exactly GAP px apart (and a multi-column card's internal columns are
# joined by the SAME GAP, see ``_card_size``).  Reusing this one existing spacing constant (no new
# public art/geom knob); change this one number to retune all board
# spacing.
GAP = GRID_UNIT

def _board_metrics() -> "_layout.BoardMetrics":
    """The two facts the moved packer cannot derive, read LIVE on every call.

    ``_card_size`` stays here because it is the bridge between the FIGURE size (render
    layer) and the card CHROME (Qt tokens) -- neither of which the packer may import.
    Built fresh rather than cached: card pixels follow the current Qt scale, and a value
    captured once would go stale exactly the way a snapshot shim does.
    """

    return _layout.BoardMetrics(gap=GAP, card_size=_card_size)

def _cell_size() -> tuple[int, int]:
    """The base CARD WIDTH unit = a 1x2 panel's card: the figure (1 row x 2 cols) plus the card
    chrome (L/R border + grey title strip + bottom border).  Every card width is a whole number of
    these base widths joined by ``GAP``, so widths stay on one rhythm (height hugs the plot)."""

    width = panel_display_size("1x2")[0] + 2 * CARD_PAD
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size("1x2")[1] + CARD_PAD
    return (width, height)

def _card_size(size: str) -> tuple[int, int]:
    """Outer card size: WIDTH = ``cols // 2`` base widths joined by ``GAP`` (1x4 is wider than 1x2),
    HEIGHT HUGS the plot (title strip + the size's own figure height + bottom pad, NO blank padding
    -- every size hugs like 1x2).  So width still scales with the size's columns, but height is
    just tall enough for the figure -- the card's bottom edge sits right under the plot."""

    rows, cols = panel_size_cells(size)
    w_units = max(1, cols // 2)
    cw, _ch = _cell_size()
    width = w_units * cw + (w_units - 1) * GAP
    # Height = the SAME chrome the cell uses, around THIS size's figure (not a cell multiple) -> hug.
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size(size)[1] + CARD_PAD
    return (width, height)

def _opaque_white_composite(pm):
    """Composite a (possibly transparent, possibly HiDPI) grabbed pixmap onto an opaque WHITE canvas of
    the SAME size AND devicePixelRatio, so the saved PNG is not see-through AND has no blank margin.

    The dpr match is the crux: on a HiDPI screen ``QWidget.grab`` returns a pixmap whose PHYSICAL size is
    ``logical × dpr`` but whose LOGICAL size is ``logical``.  A plain ``QPixmap(pm.size())`` is dpr=1, so
    ``drawPixmap`` paints the pixmap at its smaller LOGICAL size into the top-left and leaves the rest
    blank -- the giant white margin around the panels the user saw on a scaled display.  Carrying the
    pixmap's dpr makes the canvas's logical size equal the pixmap's, so the composite fills it exactly."""
    canvas = QtGui.QPixmap(pm.size())
    canvas.setDevicePixelRatio(pm.devicePixelRatio())
    canvas.fill(QtGui.QColor("#FFFFFF"))
    painter = QtGui.QPainter(canvas)
    painter.drawPixmap(0, 0, pm)
    painter.end()
    return canvas

def _aabb(cfg) -> tuple[int, int, int, int]:
    """The card's pixel AABB -- see :func:`zlc_frontend.board_layout._aabb`."""
    return _layout._aabb(cfg, _board_metrics())

def _overlaps_with_gap(box: tuple[int, int, int, int], placed) -> bool:
    """See :func:`zlc_frontend.board_layout._overlaps_with_gap`."""
    return _layout._overlaps_with_gap(box, placed, _board_metrics())

def _first_free_slot(cfg, placed, board_w: int) -> tuple[int, int]:
    """See :func:`zlc_frontend.board_layout.first_free_slot`."""
    return _layout.first_free_slot(cfg, placed, board_w, _board_metrics())

def _board_width(configs: Sequence["PanelConfig"]) -> int:
    """See :func:`zlc_frontend.board_layout.board_width`."""
    return _layout.board_width(configs, _board_metrics())

def pack(order: Sequence["PanelConfig"], board_w: int | None = None) -> bool:
    """See :func:`zlc_frontend.board_layout.pack` -- the ONE board packer, now in the target package."""
    return _layout.pack(order, _board_metrics(), board_w)

def drop_index(cfg, others: Sequence["PanelConfig"], board_w: int | None = None) -> int:
    """See :func:`zlc_frontend.board_layout.drop_index`."""
    return _layout.drop_index(cfg, others, _board_metrics(), board_w)

#: relim modes (confocal_gui combo_relim naming) -- the SINGLE source for both the Setting
#: popup combo and the Edit-tab combo, so the two never list different options.
#: "fixed" (#8) pins the y-axis / colour-limit to operator-set ``fixed_lo``/``fixed_hi`` bounds
#: (the lo/hi inputs reveal only in that mode); tight/normal autoscale as before.
_RELIM_MODES = ("tight", "normal", "fixed")

#: The relim mode as a declarative ``ParamDecl`` -- so every panel's relim chooser renders
#: through the SAME _make_param_widget / PARAM_WIDGETS path every other plot param uses (one source,
#: auto-injected into BOTH the Setting popup and the Edit tab, #H3v-4b).  Edits route through
#: ``_set_param`` (which pushes the mode onto the live plotter + reveals the fixed lo/hi row).
_RELIM_PARAM = ParamDecl(
    key="relim", label="relim", kind="choice", default="tight", choices=_RELIM_MODES, display=True,
    tooltip="Relim mode (confocal_gui combo_relim naming):\n"
            "  tight  = autoscale hugs the data\n"
            "  normal = autoscale with the matplotlib default margin\n"
            "  fixed  = pin the y-axis / colour-limit to the lo/hi below")

#: Debounce window (ms) for coalescing a burst of STRUCTURE-knob edits into ONE plotter rebuild.  A
#: fast scroll on a rebuild-class param (history length, colormap, ...) fires many _set_param calls; each
#: restarts this timer, so the (heavy) teardown+rebuild runs ONCE after the burst settles, on the latest
#: values -- removing the per-tick rebuild race.  Short enough that a single deliberate edit still feels
#: instant; long enough that a wheel spin coalesces.
REBUILD_DEBOUNCE_MS = 90

def _unit_df_for(plotter):
    """A :class:`DataFigure` bound to ``plotter``'s figure/axes whose unit is inferred from
    the LIVE x-axis label (NOT ``live_plot=``, which would mask a unit-bearing label like
    'Detuning (GHz)').  ``change_unit`` operates on ``ax.lines``, so cycling rewrites that
    figure's curve in place.  Shared by the Setting popup (the live card) and the Edit-tab
    snapshot so the x-axis unit cycle is ONE implementation."""
    from zlc_frontend.live_plot.plot_figure import DataFigure
    ax = getattr(plotter, "ax", None)
    unit = DataFigure._infer_unit(ax.get_xlabel()) if ax is not None else None
    return DataFigure(fig=plotter.fig, ax=ax, data_x=plotter.data_x, data_y=plotter.data_y,
                      labels=getattr(plotter, "labels", None), unit=unit)

# ====================================================================== panels
_MONITOR_UNSET = object()   # sentinel: a monitor panel that has never rolled yet

class _GridFocus:
    """Parked GRID state while one of its cells is ENLARGED into a standalone plot-kind figure in the
    console.  Holds the grid plotter + its (detached, not closed) canvas so :meth:`PanelCard._unfocus_grid_cell`
    can swap the grid back, the focused cell index (to mirror a threshold drag back onto it), and the mpl
    callback ids on the focus canvas so they are disconnected on return / teardown."""

    def __init__(self, *, grid, grid_canvas, k: int):
        self.grid = grid
        self.grid_canvas = grid_canvas
        self.k = int(k)
        self.click_cid = None
        self.key_cid = None
        #: A Setting edit was persisted onto the (parked) grid while it was zoomed -> its buffer is stale,
        #: so unfocus must re-render.  Left False the common "just look and go back" case keeps the grid's
        #: already-rendered buffer (fit + thumbnails) and unfocus is an instant re-blit, never a re-render.
        self.dirty = False

class PanelCard(FluentGroupBox):
    """One dashboard panel: a TITLED frame (title strip = the panel KIND + the signal-source
    legend, top-left) holding the frontend canvas, and a text
    "Setting" button on the title strip (top-right).  The frame border is the DRAG
    HANDLE (the matplotlib canvas keeps all its own interactions); the card
    spans whole layout slots -- a 2-row card is exactly two
    1-row cards plus the gap."""

    changed = QtCore.pyqtSignal()          # any config edit (console marks dirty)
    layout_changed = QtCore.pyqtSignal()   # size/slot change (console re-arranges)
    dropped = QtCore.pyqtSignal(object)    # drag-release ONLY (console snaps the drop to its nearest anchor)
    update_interval_changed = QtCore.pyqtSignal()  # per-panel refresh rate change (console re-bases the timer)
    remove_requested = QtCore.pyqtSignal(object)
    edit_requested = QtCore.pyqtSignal(object)   # "Edit…" -> open the panel's Edit tab

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None,
                 sources_provider=None, formats_provider=None, axes_provider=None,
                 sites_inputs_provider=None, curve_x_provider=None,
                 structure_provider=None, short_names_provider=None,
                 live_namespace_provider=None, pulse_state_provider=None,
                 grid_recipe_provider=None, render_barrier=None, area_select_sink=None,
                 selection_clear_sink=None, fit_node_sink=None):
        # Titled frame: the title strip carries the panel KIND (top-left) and the
        # Setting button (top-right), so the card is delineated like the rest.
        super().__init__(PANEL_KINDS[config.kind], parent)
        self.config = config
        self.names_provider = names_provider   # callable -> live signal names (Setting combo)
        # callable -> {signal name: [source node labels]}, so the picker can show
        # WHICH measurement/processor each signal comes from (not just bare names).
        self.sources_provider = sources_provider
        # callable -> {signal name: array-format}, so the picker also shows each
        # signal's SHAPE (e.g. occupied -> per-site (N,)).
        self.formats_provider = formats_provider
        # callable -> {signal name: (axis_label, unit)} from the PRODUCING node's
        # SignalSpec, so a plot reads its y-axis label/unit from the measurement that
        # makes the signal (confocal: the measurement owns its labels), not a per-kind
        # hard-coded string.
        self.axes_provider = axes_provider
        # callable -> {full hub signal: SHORT name} (the producing node's prefix stripped), so the
        # picker NEST shows the short name (frame / survival / rate) -- never the full prefixed key
        # nor the verbose SignalSpec axis label.  ONE rule, shared with the Logic tab.
        self.short_names_provider = short_names_provider
        # callable(occupancy_signal) -> (centres_signal, image_signal): the site map takes
        # ONE signal (occupancy) and resolves its centres + frame underlay from the SAME
        # producing node (via that node's spec metadata), so rings + underlay are one shot.
        self.sites_inputs_provider = sites_inputs_provider
        # callable(y_signal) -> companion x_signal: a 1d plot wired to a scan's y curve
        # draws it vs the swept x from the SAME producing node (one signal pick, #3).
        self.curve_x_provider = curve_x_provider
        # callable(signal) -> (points_shape, data_shape) from the producing node's output contract,
        # so the plot auto-reshapes by the DATA dimensionality (1-D data -> multiple lines; 2-D data
        # -> reshape/imshow) instead of guessing from sizes (#H3o).
        self.structure_provider = structure_provider
        # callable() -> the console's CURRENT shared per-tick namespace (a fresh hub snapshot).  An
        # IMMEDIATE re-render (signal switch / colormap / resize) must draw from THIS -- the SAME shot
        # every other panel is on this tick -- NOT the panel's own stale ``_last_namespace`` (a PAST
        # tick).  Otherwise after switching a panel's source it shows an OLDER shot than its siblings,
        # so e.g. a 2D image and the sitemap display different shots (#shot-coherence-on-switch).
        self.live_namespace_provider = live_namespace_provider
        # callable(value_signal) -> (PulseTableState, include_always_off): a PULSE panel resolves its
        # reproduction state off the SAME producing node (the object is carried on the node, not the
        # float-only hub) -- the SAME "aux data from the producing node" wiring the site map uses.
        self.pulse_state_provider = pulse_state_provider
        # callable(value_signal) -> grid recipe dict: a GRID panel resolves its replay recipe off the SAME
        # producing node the SAME way a pulse panel resolves its state (a dict carried on the node, the
        # float-only hub cannot hold it).
        self.grid_recipe_provider = grid_recipe_provider
        # callable() -> waits until the console's render worker is idle.  EVERY GUI-thread path
        # that mutates this card's figure/plotter (a Setting edit, a source apply, the coalesced
        # rebuild, the selectors switch, teardown) must hold this barrier first -- while a batch
        # is in flight the render thread owns the figure (see frontend/render_loop.py).  None
        # (a standalone/test card) makes it a no-op via _wait_render_idle.
        self.render_barrier = render_barrier
        # callable(card, (x_min, x_max, y_min, y_max)) -> the console's ROI-chain sink: a
        # rectangle drawn on this LIVE image panel retargets/creates a RoiProcessor consuming
        # this panel's signal (the "draw a box -> get roi_frame/roi_value signals" gesture).
        self.area_select_sink = area_select_sink
        # callable(card, action) -> the console's ONE selection-teardown sink, symmetric for BOTH
        # analyses (#10): leaving/clearing an analysis STOPS + removes the hub node it created for this
        # card's signal -- a FitProcessor for "fit" (clearing the curve fit / picking a non-fit action),
        # a RoiProcessor for "roi" (switching the ROI action off).  Both have a full create-on-apply /
        # remove-on-clear lifecycle, so neither lingers as an orphan republishing after the operator
        # turned its analysis off.
        self.selection_clear_sink = selection_clear_sink
        # callable(card, request) -> the console's ONE fit-node sink: create OR retarget the hub
        # FitProcessor that publishes a 2-D image fit's parameters as signals (fit_x0/... ), for the
        # non-grid image case.  Its teardown counterpart is ``selection_clear_sink(card, "fit")``.  The
        # ONE mutator :meth:`set_fit_request` calls this, so the overlay + the hub node stay in lockstep.
        self.fit_node_sink = fit_node_sink
        self.plotter = None
        self.canvas = None
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF = the historical display-only Monitor board).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # Last DATA selection made on this panel.  It is plot-independent and
        # serializable; selecting never implies an ROI or a fit.  The explicit
        # ``selection_action`` config decides what a later release does.
        self._active_selection = None
        # GRID focus-zoom state (console path).  A GRID panel is display-only (interactions=False), so its
        # own double-click handler is dormant; THIS card catches the double-click on the grid canvas and
        # swaps ``self.plotter`` / ``self.canvas`` to a STANDALONE plot-kind figure of the clicked cell
        # (``GridPlot.build_focus_plotter`` -- the SAME builder the notebook path uses).  While focused,
        # ``self.plotter`` IS that standalone figure, so a lim / fit / relim edit reaches it through the
        # ORDINARY _set_param / _apply_lim_to_plotter path (no bespoke code) and the live tick is gated OFF
        # so the grid never rebuilds over it (no bounce-back).  ``None`` = showing the grid; else a
        # ``_GridFocus`` holding the parked grid plotter/canvas + the focused cell index + the click/key cids.
        self._grid_focus = None
        # The TOP spacer that centres the stock-size (2x2) enlarged cell vertically inside a larger grid
        # region while focused -- inserted by _focus_grid_cell, removed on unfocus/teardown.
        self._focus_top_stretch = None
        # The cell index to AUTO RE-FOCUS after a teardown+rebuild (size / source / structure-param
        # change) that happened WHILE a grid cell was enlarged: _teardown_plot records the focused index
        # here, _build_plot's grid branch re-focuses it -- so a rebuild-class edit lands the operator back
        # on the SAME enlarged cell (at the new size/params) instead of silently bouncing to the main
        # grid view (#no-focus-bounce).  ``None`` = nothing to restore.  An EXPLICIT unfocus (double-click
        # / Esc) never passes through _teardown_plot, so it never re-focuses.
        self._pending_refocus_k: int | None = None
        self._value_shape: tuple[int, ...] | None = None
        # A STRUCTURE knob (bins / colormap / fit ...) sets this to force the NEXT _render to REBUILD
        # the plotter -- WITHOUT pre-tearing it down.  _build_plot build-then-swaps, so a failed rebuild
        # keeps the OLD figure (the card never goes blank) -- the root fix for the "scroll bins and the
        # figure occasionally vanishes" race, where a pre-null teardown could latch plotter=None.
        self._force_rebuild = False
        # ANTI-RACING rebuild debounce: a STRUCTURE-knob edit that needs a rebuild schedules it on this
        # single-shot timer instead of rebuilding inline.  A fast burst of edits (scrolling the history
        # spin / a colormap combo) RESTARTS the timer, so only ONE rebuild runs after the burst settles
        # -- the latest config.params values.  This is GENERAL (every rebuild-class param, not history
        # only): coalescing the rebuilds removes the race where per-tick teardown+rebuild outran the Qt
        # holder reflow (the figure flickered / grew / vanished).  _build_plot itself is atomic
        # build-then-swap at a fixed canvas size, so even a rebuild that DOES run never blanks the card.
        self._rebuild_timer = QtCore.QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(REBUILD_DEBOUNCE_MS)
        self._rebuild_timer.timeout.connect(self._run_pending_rebuild)
        # The LAST namespace this panel rendered from (a reference to the hub
        # snapshot of that tick).  A display-only change (cmap / relim / source pick)
        # tears the plotter down and normally waits for the NEXT hub tick to rebuild --
        # but a STOPPED measurement freezes hub.version, so _tick's gate never calls
        # refresh again and the panel would stay blank (white).  Re-rendering from this
        # cached namespace makes such a change take effect immediately, stopped or not.
        self._last_namespace: Mapping[str, object] | None = None
        # ``_repeat_cur`` = how many repeats of the measurement's block currently have data (drives
        # the plotter's "xN" ylabel); set by _signal_then_repeat when it reduces the repeat axis.
        self._repeat_cur: int = 1
        # The hub version at this panel's LAST render -- the per-panel multi-rate refresh
        # (see TaskConsole._tick) skips a panel on its beat when nothing new was published
        # since, so a slow panel does not redraw stale data and a fast one only when needed.
        self._render_version = -1
        # Fairness under overload: True when this panel's beat fell on a tick the render
        # worker was busy (or the panel was mid-drag) -- the next idle tick serves it
        # regardless of the beat modulo, so a slow-beat panel can never phase-lock onto
        # busy ticks and starve behind a heavy fast-beat sibling (see TaskConsole._tick).
        self._beat_owed = False
        self._compiled_source = config.source
        # Monitor roll-gate: remembers the per-signal version of this panel's
        # source at the last roll, so an unrelated node's version bump does not
        # append a duplicate point.  `_MONITOR_UNSET` = never rolled yet.
        self._last_monitor_key: object = _MONITOR_UNSET
        self._ref_src: tuple | None = None      # (compiled_source, inputs) the names were derived from
        self._ref_names: frozenset = frozenset()
        # 2D coordinate frame: the ROI the axes were built from, so a ROI that
        # SHIFTS (same shape, new origin) still triggers an axes rebuild.
        self._roi_built: list | None = None
        self._drag_offset: QtCore.QPoint | None = None
        self.setCursor(QtCore.Qt.OpenHandCursor)   # the frame border drags

        holder = QtWidgets.QVBoxLayout(self)
        # The card's content padding is the component-library token CARD_PAD (L/R + bottom); the
        # grey title strip is above (the FluentGroupBox padding-top), and the bottom pad makes the
        # height proportional (see _card_size).  No footer -- the signal source lives in the title.
        holder.setContentsMargins(CARD_PAD, scaled_px(2), CARD_PAD, CARD_PAD)
        holder.setSpacing(0)
        self.canvas_holder = holder
        # The transient status (the Setting tooltip / button colour) and the persistent SIGNAL
        # legend (which node each read comes from) -- the legend goes into the frame TITLE; the
        # per-shot status no longer takes panel space.
        self._status_text = ""
        self._signal_info = ""

        self._build_settings()

        self.setting_button = FluentButton("Setting", color=GREY)
        self.setting_button.setParent(self)
        self.setting_button.setFixedSize(scaled_px(74, minimum=64), scaled_px(26, minimum=22))
        self.setting_button.setToolTip("Panel settings")
        self.setting_button.clicked.connect(self._open_settings)

        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(self) -> None:
        """Card size = whole layout slots (the footer absorbs the slack)."""
        self.setFixedSize(*_card_size(self.config.size))
        self._place_setting_button()

    def _place_setting_button(self) -> None:
        if hasattr(self, "setting_button"):
            # top-right, on the title strip (the title kind sits top-left).
            self.setting_button.move(
                self.width() - self.setting_button.width() - scaled_px(8),
                scaled_px(4))
            self.setting_button.raise_()

    # ------------------------------------------------------------- settings UI
    def _build_settings(self) -> None:
        """Confocal-style VERTICAL settings popup.

        Layout idiom borrowed from ``Confocal_GUIv2/gui/gui_individual.py``'s
        ``BaseLivePlotGUI`` settings page: one section header per group (bold,
        own line), then one control per row underneath the header as a
        ``[fixed-width label | control]`` cell.  Five sections stacked top-to-
        bottom:

        * **Source** -- pick a signal or write an expression + ``Apply``.
        * **Display** -- size + (for image kinds) colormap (the COLORSET
          chooser; there is no separate cbar show/hide toggle), plus each
          kind's declarative ``PANEL_PARAMS`` widget.
        * **Unit** -- ``Unit`` cycle button + current unit label.
        * **Limits** -- ``auto``/``manual`` mode combo + a 3-column mini-grid
          of ``[axis | lo | hi]`` rows (axis labels left, lo/hi headers top).
        * **Panel** -- title edit, then a single row of
          ``Remove | Edit… | <stretch> | Save Fig``.

        Every edit is LIVE-applied -- there is no global ``Apply`` button for
        the popup itself (Source has its own Apply because the expression is
        validated separately).  Lifted helpers ``FluentSectionLabel`` and
        ``FluentSettingRow`` (in ``zlc_frontend.qt_widgets``) own the visual rhythm so
        future settings popups stay identical."""

        # The rounded card is painted by FluentPopup (translucent, frameless),
        # NOT a stylesheet border-radius on an opaque popup -- that left a
        # square white nub past the arc at the corners + a native popup shadow.
        # Painting the card also means no border stylesheet rule can cascade
        # onto child labels/controls.
        popup = FluentPopup(self)
        # The sections (Source / Display / Unit / Limits / Panel) can stack taller than the
        # screen when a panel has many signals -> wrap them in a FluentScrollArea so the popup
        # never exceeds the visible area (it scrolls instead of running off / overflowing the
        # panel).  The scroll viewport is transparent so FluentPopup's painted rounded card
        # still shows through; the height is capped in _open_settings.
        popup_outer = QtWidgets.QVBoxLayout(popup)
        popup_outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._settings_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        # NEVER a horizontal scrollbar -- the popup widens to fit its content instead (the rows
        # are narrow; only the HEIGHT is capped, below).  Vertical scroll only when the sections
        # are taller than the panel they belong to.
        self._settings_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._settings_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        popup_content = QtWidgets.QWidget()
        popup_content.setStyleSheet("background: transparent;")
        self._settings_scroll.setWidget(popup_content)
        popup_outer.addWidget(self._settings_scroll)
        root = QtWidgets.QVBoxLayout(popup_content)
        pad = scaled_px(10)
        # RESERVE the vertical scrollbar's width on the RIGHT so the scrollbar never sits ON TOP
        # of a control (it would otherwise occlude the combo / value boxes when content scrolls).
        # The popup anchors its RIGHT edge at the gear, so this extra width grows the popup
        # LEFTWARD into the empty space there -- exactly "extend the settings frame to the left".
        scrollbar_w = fluent_scrollbar_thickness()   # ONE source for the bar width (not a hand-typed 12)
        root.setContentsMargins(pad, pad, pad + scrollbar_w + scaled_px(4), pad)
        root.setSpacing(scaled_px(10, minimum=6))

        # fixed left-label column, SIZED TO CONTENT so nothing truncates: the longest
        # label is normally "colormap", but a multi-input plot adds slot rows like
        # "signal[0] occupancy" -- measure every label this popup will show and widen the
        # column to the widest (floored at 80 px), so all rows still align and read fully.
        base_slots = panel_input_slots(self.config.kind)
        # EVERY plot kind uses the SAME source MECHANISM: a signal picker + a ``value = ...``
        # expression (the reusable SignalExpr).  Whether a kind can GROW extra slots
        # (+signal / −signal) is data-driven from the per-kind declaration
        # (``panel_allows_multi_slot`` / PANEL_SINGLE_SLOT_KINDS), NOT an inline ``kind == 'sites'``
        # check: a site map takes EXACTLY ONE occupancy signal (its ring centres + frame underlay
        # resolve from signal[0]'s producing node, so a 2nd slot is meaningless) -> single picker,
        # no +/-, but it STILL has the expression box.  The per-kind value SHAPE is enforced in
        # _coerce (PANEL_INPUT_FORMAT): sites = a per-site (N,) vector, 2D = an (H×W) frame, etc.
        self._multi_slot = panel_allows_multi_slot(self.config.kind)
        n_slots = max(1, len(base_slots), len(self.config.inputs)) if self._multi_slot else max(1, len(base_slots))
        slot_labels = [(f"signal[{i}]" if n_slots > 1 else "signal") for i in range(n_slots)]
        slot_tips = [base_slots[i][2] if i < len(base_slots)
                     else f"an added signal slot — read as signal[{i}] in the expression"
                     for i in range(n_slots)]
        fm = self.fontMetrics()
        widest = max((fluent_text_width(fm, t) for t in [*slot_labels, "colormap", "threshold"]), default=0)
        label_w = max(scaled_px(80, minimum=56), widest + scaled_px(10))

        def section_box(title: str) -> QtWidgets.QVBoxLayout:
            """Header label + inner VBox; rows added to the inner VBox stack
            tightly under the header (own-line, vertical, confocal style)."""
            root.addWidget(FluentSectionLabel(title))
            inner = QtWidgets.QVBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(scaled_px(6, minimum=4))
            root.addLayout(inner)
            return inner

        # ---- Source --------------------------------------------------------
        sec = section_box("Source")
        # Tell the experimenter what SHAPE of signal this plot kind expects, so the
        # signal picker / expression below is unambiguous (single source PANEL_INPUT_FORMAT).
        accepts = PANEL_INPUT_FORMAT.get(self.config.kind)
        if accepts:
            accepts_label = FluentLabel(f"accepts {accepts}")
            accepts_label.setWordWrap(True)
            accepts_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            sec.addWidget(accepts_label)
        # ONE signal picker: a combobox of the live hub signals.  Picking one sets the
        # source to ``value = signal`` (the canonical form -- the picked signal IS ``signal``
        # in the expression).  A site map needs only its occupancy signal; its ring centres
        # + frame underlay are resolved from the SAME producing node, never extra slots.
        self.slot_combos: list = []
        for i in range(n_slots):
            combo = FluentTreeComboBox()             # collapsible-tree signal picker (G2)
            combo.setToolTip(slot_tips[i] + (
                "\nSets the source to `value = signal`." if n_slots == 1
                else f"\nRead as signal[{i}] in the expression (e.g. value = signal[0] - signal[1])."))
            combo.activated.connect(lambda _ix, idx=i: self._on_slot_pick(idx))
            self.slot_combos.append(combo)
            sec.addWidget(FluentSettingRow(slot_labels[i], combo, label_width=label_w))
        self.signal_combo = self.slot_combos[0]    # the (first) signal picker
        # Add / remove signal slots (every kind except the site map): so a panel can combine
        # several signals -- e.g. plot the DIFFERENCE of two occupancies with signal[0]-signal[1].
        if self._multi_slot:
            self.add_slot_button = FluentButton("+ signal", color=GREY)
            self.add_slot_button.setToolTip("Add another signal slot (read as signal[i] in the expression).")
            self.add_slot_button.clicked.connect(self._add_signal_slot)
            self.remove_slot_button = FluentButton("− signal", color=GREY)
            self.remove_slot_button.setToolTip("Remove the last signal slot.")
            self.remove_slot_button.clicked.connect(self._remove_signal_slot)
            self.remove_slot_button.setEnabled(n_slots > 1)
            slot_btn_row = QtWidgets.QHBoxLayout()
            slot_btn_row.setContentsMargins(0, 0, 0, 0)
            slot_btn_row.setSpacing(scaled_px(6, minimum=4))
            slot_btn_row.addWidget(self.add_slot_button, 0)
            slot_btn_row.addWidget(self.remove_slot_button, 0)
            slot_btn_row.addStretch(1)
            sec.addLayout(slot_btn_row)

        self.source_edit = FluentLineEdit(self.config.source)
        self.source_edit.setMinimumWidth(scaled_px(280, minimum=220))
        self.source_edit.setStyleSheet(
            self.source_edit.styleSheet() + " QLineEdit { font-family: Consolas, monospace; }")
        self.source_edit.setToolTip(SIGNAL_EXPR_HELP)
        # An inline one-liner is cramped for a real expression; "Edit…" pops a LARGE floating
        # multi-line editor (a modal card, so it does NOT reflow the panel layout) prefilled
        # with the source -- comfortable for typing math across signals -- and writes it back.
        # The "Edit…" wording matches the panel's other Fluent "Edit…" buttons (one house style).
        self.expand_button = FluentButton("Edit…", color=GREY)
        self.expand_button.setFixedWidth(scaled_px(56, minimum=44))
        self.expand_button.setToolTip("Open a large floating editor for this expression")
        self.expand_button.clicked.connect(self._open_expr_editor)
        self.apply_button = FluentButton("Apply", color=GREEN)
        self.apply_button.setFixedWidth(scaled_px(64, minimum=52))
        self.apply_button.clicked.connect(self._apply_source)
        self.source_edit.textChanged.connect(lambda: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        expr_row = QtWidgets.QHBoxLayout()
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(scaled_px(6, minimum=4))
        expr_row.addWidget(self.source_edit, 1)
        expr_row.addWidget(self.expand_button, 0)
        expr_row.addWidget(self.apply_button, 0)
        sec.addLayout(expr_row)

        self.status = FluentLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        sec.addWidget(self.status)

        # ---- Display -------------------------------------------------------
        sec = section_box("Display")
        self.size_combo = FluentComboBox()
        self.size_combo.addItems(list(PANEL_SIZES))
        self.size_combo.setCurrentText(self.config.size)
        self.size_combo.setToolTip("Panel size preset (height x width half-units; 2x2 = the stock plot region)")
        self.size_combo.currentTextChanged.connect(self._on_size)
        sec.addWidget(FluentSettingRow("size", self.size_combo, label_width=label_w))

        # The plot's DISPLAY knobs are DECLARATIVE ParamDecls rendered through the SAME _make_param_widget
        # / PARAM_WIDGETS path everywhere (#H3v-4b): the per-kind colormap / toggles (display=True) for
        # every panel, plus the relim chooser.  Adding a plot display
        # ParamDecl here makes it appear in the Edit tab too with NO hand-wiring (both call
        # _emit_param_rows).  Size, colormap, relim, repeat, unit, and update are all view controls;
        # data production remains on the separate Logic tab.
        self.param_widgets: dict[str, QtWidgets.QWidget] = {}
        # {key: kind} for every declarative Setting control, so refresh_on_show can re-seed each widget
        # from config.params through its kind's PARAM_WIDGETS.write (one source -- no per-key handwiring).
        self._param_kinds: dict[str, str] = {}
        # remember which kind's params this popup baked -- when a grid panel's RESOLVED per-cell
        # kind changes later (facet / sub-plot pick, a signal bind), _sync_settings_param_rows
        # rebuilds the popup so the rows below are never a stale bake of the old kind.
        self._settings_param_kind = self._param_kind()
        display_specs = (
            [s for s in PANEL_PARAMS.get(self._settings_param_kind, ()) if s.display]
            + [_RELIM_PARAM]
        )
        self.param_widgets.update(
            self._emit_param_rows(display_specs, sec.addWidget, self._set_param, label_w))
        self.lim_combo = self.param_widgets.get("relim")     # named back-ref (relim is now declarative)

        # ``repeat_mode`` (the DISPLAY collapse) is the ONLY repeat knob the plot owns: how to
        # collapse a measurement's repeat axis for display (average / add / replace / roll / create).
        # How MANY repeats lives on the MEASUREMENT (``repeat``, 0 = ∞, auto-injected by
        # _acquisition_param_decls -- #H3l), NOT here.  Rendered through the SAME _make_param_widget
        # path as every other param (no hand-placed widget); edits route through _set_param.
        for spec in self._repeat_param_specs():
            widget = self._make_param_widget(spec)
            self.param_widgets[spec.key] = widget
            self._param_kinds[spec.key] = spec.kind    # remember for refresh_on_show re-seed
            sec.addWidget(FluentSettingRow(spec.label, widget, label_width=label_w))

        # fixed lo/hi inputs (#8): ONE bespoke [lo | hi] row (the single special-cased control,
        # shown only when relim == "fixed") -- built by the shared helper the Edit tab also uses.
        self.fixed_lim_row, self.fixed_lo_edit, self.fixed_hi_edit = \
            self._make_fixed_lim_row(self._on_fixed_lim_edited, label_w)
        sec.addWidget(self.fixed_lim_row)

        # FACET chooser (grid panels only): which axis of the bound (R,P,*data_shape)
        # block expands into the cells -- the grid-as-axis-expander declaration.  Options derive
        # from the producing node's declared structure and refresh on every Setting open (the
        # signal-combo rule); "(recipe)" keeps the loaded-figure snapshot behaviour.  Beside it
        # the SUB PLOT chooser picks what each cell draws (auto = derive from what the slice
        # leaves; else an explicit hist / 2d / 1d) -- the params section below always shows the
        # RESOLVED kind's knobs (see _sync_settings_param_rows).
        self.facet_combo = None
        self.sub_kind_combo = None
        if self.config.kind == "grid":
            self.facet_combo = FluentComboBox()
            self.facet_combo.setToolTip(
                "Expand ONE axis of the bound block into the grid cells: each repeat / scan-axis "
                "entry / data-axis entry becomes its own cell ((recipe) = a loaded figure's snapshot)")
            self.facet_combo.activated.connect(self._on_facet_changed)
            sec.addWidget(FluentSettingRow("facet", self.facet_combo, label_width=label_w))
            self._refresh_facet_combo()
            from zlc_frontend.live_plot.live import GRID_CELL_BY_KIND
            self.sub_kind_combo = FluentComboBox()
            self.sub_kind_combo.addItem("auto", "")
            for cell_kind in GRID_CELL_BY_KIND:     # the cell families, ONE source
                self.sub_kind_combo.addItem(cell_kind, cell_kind)
            self.sub_kind_combo.setToolTip(
                "What each cell draws: auto derives it from what the facet slice leaves "
                "(a 2-D frame -> 2d, an ordered axis -> 1d, bare samples -> hist); "
                "an explicit pick overrides")
            self.sub_kind_combo.activated.connect(self._on_sub_kind_changed)
            sec.addWidget(FluentSettingRow("sub plot", self.sub_kind_combo, label_width=label_w))
            self._refresh_sub_kind_combo()

        # unit cycle: a single row [Unit button | <stretch> | current unit text] under the "unit"
        # label (one-control-per-row rhythm) -- the IDENTICAL row the Edit tab builds via the helper.
        unit_row, self.unit_button, self.unit_label = \
            self._make_unit_cycle_row(self._on_unit_cycle, label_w, with_label=True)
        sec.addWidget(unit_row)

        # per-panel display refresh rate (this panel only).  A fixed, harmonic set so
        # panels that share a beat stay frame-coherent (see UPDATE_INTERVALS); a fast rate
        # suits a live-1D alignment monitor.  Changing it re-bases the console timer.
        self.update_combo = FluentComboBox()
        for ms in UPDATE_INTERVALS:
            self.update_combo.addItem(f"{ms} ms", ms)
        idx = self.update_combo.findData(self.config.update_ms)
        self.update_combo.setCurrentIndex(idx if idx >= 0 else self.update_combo.findData(DEFAULT_UPDATE_MS))
        self.update_combo.setToolTip(
            "How often THIS panel redraws.  Every rate shares one base tick, so panels that\n"
            "share a beat refresh on the SAME tick from the same data -- a 2-D frame and its\n"
            "site-map stay shot-coherent.  A fast 100 ms suits a live-1D alignment monitor.")
        self.update_combo.currentIndexChanged.connect(self._on_update_interval)
        sec.addWidget(FluentSettingRow("update", self.update_combo, label_width=label_w))

        # ---- Analysis --------------------------------------------------
        # ONE picker for what a drag-selection DOES, spanning BOTH analyses this section owns: a
        # general CURVE FIT (its whole state = config.params['fit_request'] presence) and an ROI
        # crop (selection_action == 'roi').  The combo is a pure VIEW -- it DERIVES its selected
        # item from that state on every open (_refresh_analysis_controls), so the Setting picker
        # and the Edit picker can never disagree (#8); picking "curve fit" toggles fit_request
        # (never writes selection_action='fit'), "ROI" arms the crop, "none" clears both.  The
        # section is named "Analysis" because it is no longer fit-only nor roi-only (#3 naming).
        self._build_analysis_section(section_box, label_w)

        # ---- Panel ---------------------------------------------------------
        sec = section_box("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.textChanged.connect(self._on_title)
        sec.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))

        # Action row: Remove on the left (destructive, ORANGE) + Edit… to open the
        # panel's Edit tab.  Saving lives ONLY in the Edit tab now (it owns the folder
        # picker + the full DataFigure controls) -- the lightweight Setting popup no
        # longer carries a Save button, so there is one place to save from.
        remove = FluentButton("Remove", color=ORANGE)
        remove.setFixedWidth(scaled_px(72, minimum=58))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        edit_button = FluentButton("Edit…", color=ACCENT)
        edit_button.setFixedWidth(scaled_px(64, minimum=52))
        edit_button.setToolTip("Open this panel's Edit tab: colormap / unit / relim, curve fit, limits, save")
        edit_button.clicked.connect(lambda: (self.settings_popup.hide(), self.edit_requested.emit(self)))
        action_row = QtWidgets.QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(scaled_px(6, minimum=4))
        action_row.addWidget(remove, 0)
        action_row.addWidget(edit_button, 0)
        action_row.addStretch(1)
        sec.addLayout(action_row)

        self.settings_popup = popup
        # Setting-frame height high-water mark (#H3i-2): the popup GROWS to fit its content and
        # GROWS when the panel size grows, but NEVER shrinks back within a session.  Reset here so a
        # REBUILT popup (its content changed, e.g. a +/- signal slot) fits fresh, then grows again.
        self._settings_h_hwm = 0
        # A Qt.Popup auto-closes on the press that lands on the Setting button; record
        # WHEN so the button's release does not immediately re-open it (real toggle).
        self._settings_dismissed_at = 0.0
        popup._on_hidden = self._note_settings_dismissed

    def _note_settings_dismissed(self) -> None:
        self._settings_dismissed_at = time.monotonic()

    def _make_param_widget(self, spec: ParamDecl, *, apply=None) -> QtWidgets.QWidget:
        """One widget per declarative ParamDecl; edits apply INSTANTLY.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback so a functional-param edit re-renders the
        live panel AND its snapshot.  Dispatches through the SAME PARAM_WIDGETS
        registry the measurement form uses (one handler per kind), with the edit
        wired as the ``instant_apply`` path so a plot param applies on each edit."""

        cb = apply if apply is not None else self._set_param
        current = self.config.params.get(spec.key, spec.default)
        ctx = ParamWidgetContext(instant_apply=cb)
        return PARAM_WIDGETS[spec.kind].build(spec, current, ctx)

    def _emit_param_rows(self, specs, add, apply, label_w) -> dict:
        """Render each declarative ParamDecl in ``specs`` as a ``[label | control]`` row through the
        SAME _make_param_widget / PARAM_WIDGETS path the measurement form uses, appending it via the
        ``add`` callback.  Returns ``{key: widget}`` so a caller can keep a named back-reference.  BOTH
        the Setting popup AND the Edit tab call this for a plot's display knobs, so adding a plot
        ParamDecl shows up in both surfaces with NO hand-wiring (#H3v-4b)."""
        out = {}
        for spec in specs:
            widget = self._make_param_widget(spec, apply=apply)
            out[spec.key] = widget
            self._param_kinds[spec.key] = spec.kind        # remember the kind for refresh_on_show re-seed
            add(FluentSettingRow(spec.label, widget, label_width=label_w))
        return out

    def _make_fixed_lim_row(self, apply_cb, label_w):
        """The fixed lo/hi inputs as ONE bespoke ``[lo | hi]`` row (the single display knob kept
        special-cased rather than declarative -- a 2-box combined control PARAM_WIDGETS has no kind for),
        shown only when relim == "fixed".  Built by this ONE helper so the Setting popup and the Edit tab
        get the IDENTICAL row instead of two hand-copied blocks (#H3v-4b).  Returns ``(row, lo, hi)``."""
        lo = FluentLineEdit(str(self.config.params.get("fixed_lo", 0.0)))
        hi = FluentLineEdit(str(self.config.params.get("fixed_hi", 1.0)))
        for ed in (lo, hi):
            ed.setMinimumWidth(scaled_px(64, minimum=52))
            ed.editingFinished.connect(apply_cb)
        lo.setToolTip("Fixed lower limit (used only when relim = fixed)")
        hi.setToolTip("Fixed upper limit (used only when relim = fixed)")
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))
        lay.addWidget(lo, 1)
        lay.addWidget(hi, 1)
        row = FluentSettingRow("lo / hi", host, label_width=label_w)
        row.setVisible(self._relim() == "fixed")
        return row, lo, hi

    def _make_unit_cycle_row(self, on_click, label_w, *, with_label: bool):
        """The x-axis unit-cycle row as ONE bespoke ``[Unit button | <stretch> [| current-unit text]]``
        row.  Built by this ONE helper (like ``_make_fixed_lim_row``) so the Setting popup and the Edit
        tab get the IDENTICAL row -- same button width, same flush-left idiom, same SINGLE tooltip --
        instead of two hand-copied blocks whose tooltips had already drifted apart.  ``with_label=True``
        (Setting) also builds the live current-unit ``self.unit_label`` (refreshed by ``_on_unit_cycle``);
        Edit passes ``with_label=False``.  Returns ``(row, button, label_or_None)``."""
        button = FluentButton("Unit", color=GREY)
        button.setFixedWidth(scaled_px(70, minimum=56))
        button.setToolTip(
            "Cycle the x-axis unit (GHz/nm/MHz or ns/us/ms) where the axis label\n"
            "declares a convertible unit; otherwise a no-op")
        button.clicked.connect(on_click)
        host = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(scaled_px(6, minimum=4))
        lay.addWidget(button, 0)
        lay.addStretch(1)
        label = None
        if with_label:
            label = FluentLabel(self._current_unit_text())
            label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            lay.addWidget(label, 0)
        row = FluentSettingRow("unit", host, label_width=label_w)
        return row, button, label

    def _apply_lim_to_plotter(self) -> None:
        """Push the relim mode + fixed lo/hi onto the LIVE plotter through the ONE in-place entry
        every surface uses -- ``apply_param``'s relim family (BaseLivePlot applies mode + lo/hi and
        re-relims now; a GridPlot stores them and fans out to its THUMBNAILS and any focused cell) --
        never a bare attribute poke, which a grid would silently ignore.  Shared by the declarative
        relim edit (_set_param) and the rebuild-time re-apply (_apply_display_params)."""
        if self.plotter is None:
            return
        # Coalesce: each apply_param ends in BaseLivePlot.draw() (a SYNCHRONOUS draw_idle+flush_events, and
        # for a grid a whole-N-cell thumbnail rebuild), so three back-to-back applies did three full renders
        # of the same final state.  suspend_draws makes those inner draws no-ops; the ONE trailing
        # canvas.draw_idle() below paints the final result once (identical pixels, ~3x fewer renders).
        with self.plotter.suspend_draws():
            self.plotter.apply_param("relim", self._relim())
            self.plotter.apply_param("fixed_lo", float(self.config.params.get("fixed_lo", 0.0)))
            self.plotter.apply_param("fixed_hi", float(self.config.params.get("fixed_hi", 1.0)))
        if self.canvas is not None:
            self.canvas.draw_idle()

    def _kind_repeat_modes(self) -> list:
        """The repeat modes THIS plot kind offers (its ``PLOT_KINDS`` entry, else the image
        default) -- the ONE lookup shared by the param spec and strict value reader."""
        return list(_repeat_modes_for_kind(self.config.kind))

    def _relim(self) -> str:
        """The panel's relim mode, defaulting to ``_RELIM_PARAM.default`` -- the ONE place that
        default lives, so every reader agrees instead of re-typing the 'tight' literal (#A3)."""
        return str(self.config.params.get("relim", _RELIM_PARAM.default))

    def _repeat_param_specs(self) -> tuple:
        """The PLOT's repeat-DISPLAY param (``repeat_mode``), DECLARED so the Setting auto-renders it
        through the same widget path as every other param.  This is the ONLY repeat knob on the plot:
        the COUNT (``repeat``) belongs to the MEASUREMENT (the plot cannot tell a measurement how many
        times to run) -- the plot just chooses how to DISPLAY the repeats the measurement produced.
        Each plot KIND exposes the repeat modes meaningful for it (#issue-1), read from the one
        ``PLOT_KINDS`` table: the BASE verbs (average/add/replace) are GENERIC across EVERY kind (one
        ``reduce_repeat`` collapses the axis the same way); a trace adds ``roll``; a DISTRIBUTION adds
        ``pool`` (bin every repeat together) and defaults to it.  2d/sites omit per-repeat ``create``.
        No kind offers a mode it would silently ignore; the default is the kind's first (canonical) mode."""
        modes = self._kind_repeat_modes()
        return (
            ParamDecl(key="repeat_mode", label="repeat mode", kind="choice", default=modes[0],
                      choices=tuple(modes),
                      tooltip=self._repeat_mode_tooltip()),
        )

    def _repeat_mode_value(self) -> str:
        """Return the kind-valid stored repeat mode or its declared default."""
        return _repeat_mode_for_kind(
            self.config.kind,
            self.config.params.get("repeat_mode", _MISSING_REPEAT_MODE),
        )

    def _bound_is_occupancy(self) -> bool:
        """Whether the bound producer declares site-map companion signals.

        The decision follows producer metadata rather than a processor class or
        panel-kind string, so any current occupancy producer can supply the same
        centres/underlay contract.
        """
        occ = self.config.inputs[0] if self.config.inputs else ""
        if not occ or not callable(self.sites_inputs_provider):
            return False
        try:
            centers_name, _ = self.sites_inputs_provider(occ)
        except Exception:
            return False
        return bool(centers_name)

    def _repeat_mode_tooltip(self) -> str:
        """The repeat-mode tooltip, SPECIALISED for an occupancy signal (#H3s-F5): when the bound
        signal is per-site occupancy, ``average`` is the per-site LOADING PROBABILITY (mean of the N
        shots' 0/1), ``add`` the total loads, ``replace`` the latest shot, ``roll`` the last N --
        the experiment meaning, not a generic array verb.  Generic otherwise, driven off the signal's
        role rather than the panel kind."""
        if self.config.kind == "hist":
            return ("How to combine the measurement's repeats into the distribution:\n"
                    "  pool    = bin EVERY repeat's samples into ONE histogram (all repeats together)\n"
                    "  average = bin the per-point MEAN over the repeats (one histogram)\n"
                    "  add     = bin the per-point SUM over the repeats\n"
                    "  replace = bin only the newest repeat's samples\n"
                    "  create  = ONE filled histogram per repeat overlaid (each a different colour; the "
                    "first repeat also draws the fit/threshold/stats)")
        if self._bound_is_occupancy():
            return ("How to combine the N shots of per-site occupancy for display:\n"
                    "  average = per-site LOADING PROBABILITY = mean of the N shots' 0/1\n"
                    "  add     = total loads per site over the N shots\n"
                    "  replace = the latest shot's 0/1\n"
                    "  roll    = the last N shots (rolling)\n"
                    "  create  = draw EVERY shot as its own line (1-D only)")
        return ("How to combine the measurement's repeats for display (decoupled from\n"
                "the signal):\n"
                "  average = mean over the repeats that have data (noise reduction)\n"
                "  add     = sum over repeats\n"
                "  replace = show the latest repeat\n"
                "  roll    = show the newest repeat (rolling)\n"
                "  create  = draw EVERY repeat as its own line (1-D only)")

    def refresh_on_show(self) -> None:
        """Re-seed every Setting control from ``config.params`` -- the SINGLE source of truth for a
        panel's params -- so the Setting popup shows the CURRENT values whenever it opens, even if they
        were changed elsewhere (the Edit tab writes the same config.params).  Each widget is re-seeded
        through its kind's ``PARAM_WIDGETS.write`` (one entry point, no per-key handwiring), with its
        change signals blocked so re-seeding does not re-fire ``_set_param`` (which would needlessly
        rebuild).  This is the #6 fix: a control is a VIEW of config.params, refreshed on show, never a
        private copy that drifts from the other surface."""
        for key, widget in self.param_widgets.items():
            kind = self._param_kinds.get(key)
            if kind is None or key not in self.config.params:
                continue
            with _signals_blocked(widget):
                try:
                    PARAM_WIDGETS[kind].write(widget, self.config.params[key])
                except (TypeError, ValueError):
                    continue

    def _open_settings(self) -> None:
        # Click-to-open / click-again-to-close TOGGLE.  A Qt.Popup already auto-closes
        # on the mouse PRESS that lands on this button, so by the time the button's
        # release fires the popup is hidden -- naively re-opening it would make the
        # button never close it.  So: if visible, hide; and if it was auto-dismissed
        # within the last moment (this very click closed it), do NOT re-open.
        popup = self.settings_popup
        if popup.isVisible():
            popup.hide()
            return
        if time.monotonic() - self._settings_dismissed_at < 0.25:
            return
        self._sync_settings_param_rows()   # a grid's resolved kind may have changed since the last bake
        popup = self.settings_popup        # (the sync may have swapped in a fresh popup)
        self.refresh_on_show()          # Setting controls are a VIEW of config.params -- refresh on open (#6)
        self._refresh_analysis_controls()   # the Analysis section derives on every open too (#6 result-row fix)
        self._refresh_signal_combo()
        self._refresh_facet_combo()     # facet choices re-derive from the CURRENT node structure
        self._refresh_sub_kind_combo()
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        self._size_settings_popup()                        # height: show-all, grow-not-shrink (#H3i-2)
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        top_y = anchor.y() + popup_gap()   # the ONE below-anchor Fluent popup gap (combo / overflow share it)
        x = anchor.x() - popup.width()
        if avail is not None:
            x = max(avail.left(), min(x, avail.right() - popup.width()))
        popup.move(x, top_y)
        popup.show()
        popup.raise_()

    def _size_settings_popup(self) -> None:
        """Size the Setting frame: EXPAND to show all its content, UNTIL it reaches the PLOT PANEL's
        own bottom edge -- then the FluentScrollArea scrolls the overflow.  So the cap is the panel
        boundary (`panel_bottom - top_y`), NOT the screen: a tall panel gets a tall frame, a short
        panel scrolls, and either way the frame stays within the panel.  GROW with the panel size,
        clamped to the live cap (so shrinking the panel re-clamps the frame to the smaller panel)."""
        popup = getattr(self, "settings_popup", None)
        if popup is None:
            return
        popup.adjustSize()
        screen = QtWidgets.QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen is not None else None
        anchor_y = self.setting_button.mapToGlobal(QtCore.QPoint(0, self.setting_button.height())).y()
        top_y = anchor_y + popup_gap()      # match the same gap the open path (above) uses to place it
        content = self._settings_scroll.widget()
        content_h = (content.sizeHint().height() if content is not None else popup.height()) + 2 * scaled_px(10)
        # The cap is the PLOT PANEL's own bottom edge (the popup opens just below the gear near the
        # panel top and grows DOWN); content past it scrolls.
        panel_bottom = self.mapToGlobal(QtCore.QPoint(0, self.height())).y()
        cap = max(scaled_px(140), panel_bottom - top_y)
        if avail is not None:                              # last-resort: never escape the physical screen
            cap = min(cap, avail.bottom() - top_y)
        want = min(content_h, cap)                         # grow to the content, capped at the panel bottom
        self._settings_h_hwm = max(int(self._settings_h_hwm), int(want))   # grow within a session...
        h = min(self._settings_h_hwm, cap)                 # ...but always re-clamp to the LIVE panel cap
        popup.setMaximumHeight(int(cap))                   # content beyond the panel bottom scrolls
        popup.resize(popup.width(), max(scaled_px(140), int(h)))

    def _fill_slot_combo(self, combo, current: str) -> None:
        """Populate ONE slot combobox with ``(none)`` + every live hub signal GROUPED by its
        producing node (the shared :func:`fill_grouped_signal_combo`): bold non-selectable
        headers, indented signals.  Keeps the slot's current pick selected even when its source
        node is not running yet (so a saved layout's ``signal[1]`` survives a restart)."""
        fill_grouped_signal_combo(
            combo, names=(self.names_provider() if callable(self.names_provider) else []),
            sources=(self.sources_provider() if callable(self.sources_provider) else {}),
            formats=(self.formats_provider() if callable(self.formats_provider) else {}),
            labels=self._signal_short_names_map(),
            current=current, none_label="(none)")

    def _signal_short_names_map(self) -> dict:
        """``{full hub signal: SHORT name}`` (the producing node's prefix stripped) from the
        ``short_names_provider``, so the picker nest leaf is the short name (``frame`` / ``survival`` /
        ``rate``) -- NOT the full prefixed key and NOT the verbose SignalSpec axis label (``camera
        image``).  The picker nest already names the producer node, so the leaf is the short NAME."""
        return coerce_short_labels(self.short_names_provider)

    def _refresh_signal_combo(self) -> None:
        """Refresh the signal picker with the hub's current signals, each labelled with the
        measurement/processor that PRODUCES it (``occupied — occupancy``) so the input is
        filled by origin, not a bare name.  It keeps ``config.inputs[0]`` selected; the source
        expression reads it as ``signal`` (``value = signal``)."""
        for i, combo in enumerate(getattr(self, "slot_combos", [])):
            cur = self.config.inputs[i] if i < len(self.config.inputs) else ""
            self._fill_slot_combo(combo, cur)

    def _on_slot_pick(self, idx: int) -> None:
        """The signal picker changed: write its bare signal name into ``config.inputs[idx]``
        and point the source at ``value = signal`` (a blank pick blanks the panel).  The
        picker IS the "plot this signal" control; the expression box stays the advanced
        override for a custom ``value = ...``.  Then re-apply so the pick takes effect now."""
        name = str(self.slot_combos[idx].currentData() or "")   # bare name (not the labelled text)
        while len(self.config.inputs) <= idx:
            self.config.inputs.append("")
        self.config.inputs[idx] = name
        # Single-slot: the picker IS the "plot this signal" control -> point the source at it
        # (value = signal).  Multi-slot: the expression (value = signal[0] - signal[1]) is
        # user-authored and combines the slots, so a pick just rebinds slot idx -- don't clobber it.
        if len(self.slot_combos) <= 1 and idx == 0:
            self.source_edit.setText("value = signal" if name else _BLANK_SOURCE)
        self._apply_source()                      # picking a slot applies instantly
        self._sync_settings_param_rows()          # a grid's resolved cell kind follows the new bind

    def _add_signal_slot(self) -> None:
        """Add a signal slot (signal[i]) so the panel can combine more signals.  When growing
        from one slot to two, seed the canonical two-signal expression so the panel is usable
        at once; the user can edit it.  Rebuilds the Setting popup so the new row appears."""
        self.config.inputs.append("")
        if len(self.config.inputs) == 2 and self.config.source.strip() in ("value = signal", "", _BLANK_SOURCE.strip()):
            self.config.set_source("value = signal[0] - signal[1]")
            self._compiled_source = self.config.source
        self._rebuild_settings_popup()
        self._apply_source()

    def _remove_signal_slot(self) -> None:
        """Remove the LAST signal slot (never below one).  Back at a single slot, restore the
        canonical ``value = signal`` so the picker drives the plot again."""
        if len(self.config.inputs) <= 1:
            return
        self.config.inputs.pop()
        if len(self.config.inputs) == 1:
            self.config.set_source("value = signal" if self.config.inputs[0] else _BLANK_SOURCE)
            self._compiled_source = self.config.source
        self._rebuild_settings_popup()
        self._apply_source()

    def _rebuild_settings_popup(self, *, reopen: bool = True) -> None:
        """Rebuild the Setting popup (the slot count and the per-kind param rows are fixed at build
        time, so adding/removing a slot or a resolved-kind change rebuilds it).  ``reopen`` shows the
        fresh popup at once (the user was in it); False rebuilds silently for the next open."""
        old = getattr(self, "settings_popup", None)
        if old is not None:
            old.hide()
            old.deleteLater()
        self._build_settings()                    # builds a fresh self.settings_popup + combos
        if reopen:
            self._settings_dismissed_at = 0.0
            self._open_settings()                 # reopen (positioned + height-capped)

    def _sync_settings_param_rows(self) -> None:
        """A grid panel's per-kind Setting rows (bins/fit/ylog for hist cells, the colormap for 2d
        cells, ...) follow the RESOLVED per-cell kind: whenever the resolve changes -- a facet or
        sub-plot pick, a signal bind, a load -- the popup is rebuilt so the rows are the new kind's
        PANEL_PARAMS, never a stale bake of the kind the popup happened to open with."""
        if self.config.kind != "grid":
            return
        if self._param_kind() == getattr(self, "_settings_param_kind", None):
            return
        self._rebuild_settings_popup(reopen=self.settings_popup.isVisible())

    # ------------------------------------------------ basic display toggles
    def _current_unit_text(self) -> str:
        n = int(self.config.params.get("unit_index", 0) or 0)
        return "unit" if n == 0 else f"unit x{n}"

    def _on_unit_cycle(self) -> None:
        """Cycle the x-axis unit ONE step on the live panel and remember the new
        index (reuses DataFigure.change_unit).  Cycling once on the current axis
        (rather than re-applying index-from-original) keeps repeated clicks
        correct; a rebuild re-derives the same state from the stored index.  When
        the axis label carries no convertible unit the cycle is a no-op."""
        if self.plotter is None:
            return
        self._wait_render_idle()       # change_unit rewrites live x data/labels: own the figure first
        df = self._unit_df()
        length = len(df.conversion_map) if getattr(df, "conversion_map", None) else 0
        if length:
            df.change_unit()
            self.config.params["unit_index"] = (
                int(self.config.params.get("unit_index", 0) or 0) + 1) % length
            self.changed.emit()
        if hasattr(self, "unit_label"):
            self.unit_label.setText(self._current_unit_text())
        if self.canvas is not None:
            self.canvas.draw_idle()

    def apply_fixed_lims(self, lo: float, hi: float) -> None:
        """Persist + apply the fixed lo/hi NOW through the ONE in-place entry every surface uses
        (``apply_param``'s relim family): on a ZOOMED grid the pin is ALSO stored on the parked grid
        (thumbnails carry it when the zoom returns); otherwise it lands on the live plotter directly
        (a GridPlot fans it out to its thumbnails itself).  The Setting popup's lo/hi inputs and the
        Edit tab's both route here -- no second hand-copied push path."""
        self._wait_render_idle()       # relim + axis mutation on the live plotter: own the figure first
        self.config.params["fixed_lo"], self.config.params["fixed_hi"] = float(lo), float(hi)
        if self._grid_focus is not None:
            self._set_focused_grid_param("fixed_lo", float(lo))
            self._set_focused_grid_param("fixed_hi", float(hi))
        elif self.plotter is not None:
            with self.plotter.suspend_draws():   # both applies paint the SAME final state -> one render below
                self.plotter.apply_param("fixed_lo", float(lo))
                self.plotter.apply_param("fixed_hi", float(hi))
        if self.canvas is not None:
            self.canvas.draw_idle()
        self.changed.emit()

    def _on_fixed_lim_edited(self) -> None:
        """The Setting popup's fixed lo/hi inputs committed (#8): read + apply via the ONE path."""
        self.apply_fixed_lims(_safe_float(self.fixed_lo_edit.text(), 0.0),
                              _safe_float(self.fixed_hi_edit.text(), 1.0))

    def _on_update_interval(self, _idx: int) -> None:
        """Persist THIS panel's display refresh interval (``config.params["update_ms"]``,
        one of :data:`UPDATE_INTERVALS`) and ask the console to re-base its timer so the new
        rate co-aligns with the others.  No plot rebuild -- only the refresh cadence changes."""
        ms = int(self.update_combo.currentData() or DEFAULT_UPDATE_MS)
        if ms == int(self.config.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS):
            return
        self.config.params["update_ms"] = ms
        self.update_interval_changed.emit()    # console re-bases the shared timer
        self.changed.emit()                    # mark the layout dirty

    # --------------------------------------------- basic display application
    def _param_kind(self) -> str:
        """The plot kind whose :data:`PANEL_PARAMS` drive THIS panel's Setting / Edit param UI.

        For every ordinary kind that is ``config.kind``.  For a GRID panel it is the grid's per-site
        ``sub_plot_kind`` -- so a hist grid shows the ``"hist"`` bins/fit/ylog knobs and a 2d (kernel) grid
        shows the ``"2d"`` colormap chooser, INSTEAD of the one hard-coded hist set a fixed ``"grid"`` entry
        used to give every grid regardless of what its cells actually were (#4).  Resolved from the built grid
        (``plotter.sub_plot_kind``); before the grid is built, peeked off its producing node's recipe."""
        if self.config.kind != "grid":
            return self.config.kind
        sub = getattr(self.plotter, "sub_plot_kind", None)
        if sub:
            return str(sub)
        if self._facet():
            return self._resolved_sub_kind()     # a facet grid's kind derives from the slice, not a recipe
        recipe = self._grid_recipe_or_none()
        if recipe:
            return str(recipe.get("sub_plot_kind") or "hist")
        return "hist"

    def _grid_recipe_or_none(self):
        """The bound signal's grid RECIPE (a loaded saved-figure's reproduction dict) when its
        producing node carries one -- else None.  The ONE probe _param_kind and the facet choices
        share (a "(saved figure)" option only exists when there IS a saved figure to show)."""
        if not (callable(self.grid_recipe_provider) and self.config.inputs and self.config.inputs[0]):
            return None
        try:
            return self.grid_recipe_provider(self.config.inputs[0])
        except Exception:
            return None

    def _grid_recipe_with_params(self, recipe: Mapping[str, object]) -> dict:
        """Fold THIS panel's live grid DISPLAY knobs (the per-site ``sub_plot_kind``'s params -- ``bins`` /
        ``fit`` / ``ylog`` for a hist grid, ``cmap`` for a 2d grid) from ``config.params`` into the producing
        node's grid recipe, so a rebuild redraws the grid with the operator's current Setting choices (not just
        the recipe's saved defaults).  The panel's params WIN over the recipe's stored ``display_params`` (the
        operator's live pick is the source of truth); ``bins`` also overrides the recipe's top-level ``bins`` so
        the thumbnails re-bin.  A copy -- the node's recipe is untouched."""
        out = dict(recipe)
        # The panel's CURRENT title wins over the recipe's saved one -- same "live pick beats stored
        # default" rule as the display knobs below.  build_grid_figure reads recipe['title'], so without
        # this an Edit-tab title edit on a RECIPE grid (a loaded figure) rebuilt the snapshot from the
        # stale saved title and never followed the edit (the facet branch already passes config.title to
        # _build_facet_plotter; this makes the recipe branch agree).  ONE injection point -> both the live
        # card's recipe rebuild and the Edit snapshot rebuild pick up the edited title.
        out["title"] = self.config.title or out.get("title") or ""
        display = dict(out.get("display_params") or {})
        for decl in _panel_display_decls(self.config.kind, self._param_kind()):   # sub-kind knobs + grid title (#5)
            if decl.key in self.config.params:
                display[decl.key] = self.config.params[decl.key]
        # The relim family is display state too (it lives beside PANEL_PARAMS as _RELIM_PARAM + the
        # bespoke lo/hi row): folded in like every other knob, so a rebuilt grid / Edit snapshot
        # replays the operator's lim onto the thumbnails and the focus seed -- never a silent revert.
        for lim_key in ("relim", "fixed_lo", "fixed_hi"):
            if lim_key in self.config.params:
                display[lim_key] = self.config.params[lim_key]
        if "bins" in display:
            out["bins"] = int(display["bins"])          # top-level bins drives the thumbnail binning
        out["display_params"] = display
        return out

    # --------------------------------------------------------------- facet (the grid as an axis-expander)
    def _facet(self) -> str | None:
        """The panel's facet declaration (``config.params["facet"]``): which axis of the bound
        canonical ``(R,P,*data_shape)`` block expands into the grid cells.  A missing value means a
        non-faceted recipe grid; every present value is one canonical facet token."""
        value = self.config.params.get("facet")
        if value is None:
            return None
        normalize_facet(value)
        return str(value)

    def _facet_value_shapes(self) -> tuple[tuple, tuple]:
        """(points multi-D shape, data shape) from the producing node's declared structure (#H3o) --
        the points axis is stored FLAT, its multi-D form lives in ``grid_shape`` when the scan
        declared one.  The ONE source the facet choices, the auto sub-kind and the slicer share."""
        st = self._bound_structure() or {}
        gs = tuple(int(n) for n in (st.get("grid_shape") or ()))
        ps = tuple(int(n) for n in (st.get("points_shape") or ()))
        ds = tuple(int(n) for n in (st.get("data_shape") or ()))
        return (gs or ps), ds

    def _resolved_sub_kind(self) -> str:
        """The facet grid's per-cell kind: the operator's explicit ``sub_plot_kind`` param when set,
        else the ONE auto rule (``default_sub_plot_kind``: what each cell has left after the slice)."""
        from zlc_frontend.live_plot.live import GRID_CELL_BY_KIND, default_sub_plot_kind
        sub = str(self.config.params.get("sub_plot_kind") or "")
        if sub in GRID_CELL_BY_KIND:
            return sub
        points_shape, data_shape = self._facet_value_shapes()
        return default_sub_plot_kind(
            self._facet() or "repeat", points_shape=points_shape, data_shape=data_shape)

    def _facet_cells(self, value):
        """Slice the bound block into the per-cell inputs through the ONE rule (live.facet_cells)."""
        from zlc_frontend.live_plot.live import facet_cells, normalize_facet
        block = np.asarray(value)                 # NATIVE dtype (uint8 camera block stays uint8);
        if block.dtype.kind not in "iubf":        # only non-numeric results normalize to float
            block = np.asarray(block, dtype=float)
        if block.ndim < 2:
            raise ValueError(
                "a facet grid slices the bound signal's canonical (R,P,*data_shape) block; got shape "
                f"{block.shape} -- bind a measurement's block signal (not a scalar).")
        pts, _ = self._facet_value_shapes()
        if int(np.prod(pts, dtype=np.int64) if pts else 0) != int(block.shape[1]):
            pts = ()                              # declared shape does not match this block -> flat points
        cells = facet_cells(block, self._facet(), sub_plot_kind=self._resolved_sub_kind(),
                            points_shape=pts, repeat_mode=self._repeat_mode_value())
        # A LIVE finite block carries only the repeats that HOLD data (it grows 1..ring as shots
        # land), so a facet=repeat grid would see its cell count change every shot -- a full
        # build-then-swap per shot for the WHOLE fill window, and a >MAX_GRID_CELLS repeat would
        # only error at shot ring+1.  Pad to the producer's declared ring with ZERO-memory NaN
        # placeholders (broadcast views -- never a materialised full-size frame), so the grid holds
        # a constant ring cells from the first shot: not-yet-taken cells render blank (NaN), filled
        # cells stream through the in-place update_cells fast path, and the cell-count cap fires
        # immediately.  Only the repeat facet pads: a points/data facet's cell count is a declared
        # shape, not the fill state.
        spec = normalize_facet(self._facet())
        if spec is not None and spec[0] == "repeat" and cells:
            st = self._bound_structure()
            ring = int((st or {}).get("ring", 0) or 0)
            if ring > len(cells):
                blank = np.broadcast_to(np.float32(np.nan), np.shape(cells[0]))
                cells = list(cells) + [blank] * (ring - len(cells))
        return cells

    def _build_facet_plotter(self, value, *, interactions: bool):
        """Build this panel's FACET grid through the ONE factory + replay its persisted per-kind
        display knobs (bins / fit / ylog / cmap) AND the relim family (BaseLivePlot.apply_param owns
        them all).  The ONE builder the live card (display-only) and the Edit-tab snapshot
        (interactive) share -- so the two can never drift."""
        from zlc_frontend.live_plot.live import facet_axis_labels, facet_cell_labels, grid as build_facet_grid
        sub = self._resolved_sub_kind()
        cells = self._facet_cells(value)
        # Per-cell TITLE identifiers (#5): the console pre-slices (facet=None to the factory), so it hands
        # the labels EXPLICITLY, derived from the bound facet + its points shape through the ONE
        # ``facet_cell_labels`` source -- so a repeat / scan / site grid's cells read 'rep k' / a scan
        # coordinate / 's k' instead of a hardcoded site tag, from the same source the notebook path uses.
        points_shape, data_shape = self._facet_value_shapes()
        # The swept axis NAMES + per-cell COORDINATES are metadata of the producing SCAN NODE and the
        # facet spec -- NOT of the value expression -- so they are fetched UNGATED: even a transforming
        # value (``np.log(f)``, ``a-b``) still faceted on scan axis i, so cell k is still scan point k
        # of axis i and its coordinate is known.  (``_bound_structure`` is identity-gated because it
        # drives the value's SHAPE reshape, a DIFFERENT concern; the scan names/coordinates never
        # change under a value transform -- gating them there was the root cause of the ``pt k``
        # fallback.)  ``param_names`` are needed for EVERY facet group -- a repeat / data facet's cells
        # keep the scan axes as their remaining x/y (#6) -- while the per-cell COORDS label only a
        # POINTS facet's cells (``Bz=1.2`` instead of a bare ``pt k``).
        from zlc_frontend.live_plot.live import normalize_facet
        coords = names = None
        spec = normalize_facet(self._facet())
        if self.config.inputs and callable(self.structure_provider):
            try:
                raw = self.structure_provider(self.config.inputs[0])
            except Exception:
                raw = None
            if raw:
                names = raw.get("param_names")
                if spec and spec[0] == "points":
                    all_coords = raw.get("points_coords") or []
                    axis = int(spec[1])
                    if axis < len(all_coords) and len(all_coords[axis]) == len(cells):
                        coords = all_coords[axis]
        cell_labels = facet_cell_labels(self._facet(), len(cells), points_shape=points_shape,
                                        coords=coords, param_names=names)
        plotter = build_facet_grid(
            cells, sub_plot_kind=sub, size=self.config.size, cell_labels=cell_labels,
            # figure-level x/y axis names FOLLOW the facet (#6): the ONE facet_axis_labels rule maps
            # the REMAINING axes to a cell's x/y (scan param names / the value's declared axis
            # label), degrading to the kind's stock defaults when the metadata is unknown.  The
            # console pre-slices (facet=None to the factory), so it hands the labels explicitly --
            # the same derivation the notebook ``grid(value, facet=...)`` path runs inside.
            labels=facet_axis_labels(
                self._facet(), sub, points_shape=points_shape, data_shape=data_shape,
                param_names=names, value_label=self._source_axis_label()),
            display=False, interactions=interactions, title=self.config.title or "")
        # Fold the persisted display knobs in with the draws SUSPENDED.  Each ``apply_param`` otherwise
        # forces a synchronous N-cell repaint, so ~10 keys = up to 10 full grid ``draw()``s per build --
        # the draw-per-mutation anti-pattern, byte-identical to the loop in ``build_grid_figure`` above
        # (which already suspends).  The ONE first render happens via the caller's ``canvas.draw`` (the
        # plotter is built ``display=False``), exactly as the non-facet grid path.
        # A console-driven grid with an active fit is DISPLAY-ONLY: mark it before replaying fit_request
        # so the build's apply_param never solves in place on the build thread (#6b -- the worker node
        # fits per-cell; the next _update_fit_overlays pushes the params and this reconstructs them).
        if callable(self.fit_node_sink) and self.config.params.get("fit_request") \
                and getattr(plotter, "_published_cell_popt", None) is None:
            plotter._published_cell_popt = {}
        with plotter.suspend_draws():
            for key in ([d.key for d in _panel_display_decls("grid", sub)]
                        + ["relim", "fixed_lo", "fixed_hi", "view_xlim", "view_ylim", "fit_request"]):
                if key == "cmap":
                    # Inject the RESOLVED cmap (operator's pick ELSE the sub-kind's declared PANEL_PARAMS
                    # default) through the ONE ``_resolved_cmap`` resolver -- the SAME source the Setting
                    # popup shows.  Without this a grid whose params carry no explicit cmap fell through to
                    # ImageCell's own default (grey), so the live grid drew grey while the Setting said the
                    # default: render == Setting now, one source, no silent divergence.
                    plotter.apply_param("cmap", _resolved_cmap(sub, self.config.params))
                elif key in self.config.params:
                    plotter.apply_param(key, self.config.params[key])
        return plotter

    def _facet_choices(self) -> list[tuple[str | None, str, bool]]:
        """The facet dropdown's ``[(stored value, display text, enabled)]`` -- derived from the
        producing node's declared axis structure, with the axis LENGTH shown so the operator picks
        by meaning ('scan axis 0 (5)').  An axis longer than :data:`MAX_GRID_CELLS` is listed but
        DISABLED (the grid factory refuses it -- the UI would freeze); the "(saved figure)" row
        exists only when the bound node actually carries a saved grid recipe to replay."""
        from zlc_frontend.live_plot.live import MAX_GRID_CELLS
        out = []
        if self._grid_recipe_or_none() is not None:
            out.append((None, "(saved figure)", True))
        out.append(("repeat", "repeat", True))

        def _axis(value, text, n):
            ok = int(n) <= MAX_GRID_CELLS
            out.append((value, text if ok else f"{text} – too many", ok))

        points_shape, data_shape = self._facet_value_shapes()
        if len(points_shape) > 1:
            for i, n in enumerate(points_shape):
                _axis(f"points:{i}", f"scan axis {i} ({n})", n)
        elif points_shape and points_shape[0] > 1:
            _axis("points:0", f"scan axis 0 ({points_shape[0]})", points_shape[0])
        for i, n in enumerate(data_shape):
            if n > 1:
                _axis(f"data:{i}", f"data axis {i} ({n})", n)
        return out

    def _refresh_facet_combo(self) -> None:
        """Refill the facet dropdown from the CURRENT node structure (a Setting open re-derives it,
        like the signal combo) keeping the stored pick selected; over-limit axes come out greyed."""
        combo = getattr(self, "facet_combo", None)
        if combo is None:
            return
        current = self._facet()
        with _signals_blocked(combo):
            combo.clear()
            index = 0
            for j, (value, text, enabled) in enumerate(self._facet_choices()):
                combo.addItem(text, value)
                if not enabled:
                    item = combo.model().item(combo.count() - 1)
                    if item is not None:
                        item.setEnabled(False)
                if value == current:
                    index = j
            combo.setCurrentIndex(index)

    def _on_facet_changed(self, index: int) -> None:
        """The operator picked a facet axis: persist + rebuild.  A facet change is a STRUCTURE change
        (the cell count and even the per-cell kind may differ), so it goes through the ordinary reset
        path -- the generic refocus rule returns to the enlarged cell when it survives the rebuild."""
        raw_value = self.facet_combo.itemData(int(index))
        value = None if raw_value is None else str(raw_value)
        if value is not None:
            normalize_facet(value)
        if value == self._facet():
            return
        self._wait_render_idle()   # the teardown+rebuild below must own the figure (ownership protocol)
        if value is None:
            self.config.params.pop("facet", None)
        else:
            self.config.params["facet"] = value
        self._reset_plot()
        self._render_version = -1
        self._rerender_last()
        self._sync_settings_param_rows()   # the resolved per-cell kind may have changed with the axis
        self.changed.emit()

    def _refresh_sub_kind_combo(self) -> None:
        """Re-select the stored ``sub_plot_kind`` pick ("" = auto) on the Setting's sub-plot chooser."""
        combo = getattr(self, "sub_kind_combo", None)
        if combo is None:
            return
        with _signals_blocked(combo):
            combo.setCurrentIndex(max(0, combo.findData(str(self.config.params.get("sub_plot_kind") or ""))))

    def _on_sub_kind_changed(self, index: int) -> None:
        """The operator picked what each cell draws (auto / hist / 2d / 1d): persist + rebuild -- a
        per-cell kind change is a structure change exactly like a facet change, and the Setting's
        param rows follow the new resolve."""
        value = str(self.sub_kind_combo.itemData(int(index)) or "")
        if value == str(self.config.params.get("sub_plot_kind") or ""):
            return
        self._wait_render_idle()   # the teardown+rebuild below must own the figure (ownership protocol)
        if value:
            self.config.params["sub_plot_kind"] = value
        else:
            self.config.params.pop("sub_plot_kind", None)      # auto: derive from the slice
        self._reset_plot()
        self._render_version = -1
        self._rerender_last()
        self._sync_settings_param_rows()
        self.changed.emit()

    def _apply_display_params(self) -> None:
        """Apply the persisted Setting toggles (relim mode + unit cycle) to
        the current plotter -- called after every rebuild and on each edit.

        The persisted knobs: ``config.params["relim"]`` (confocal naming:
        ``tight`` / ``normal``), ``config.params["unit_index"]`` (the x-axis unit
        cycle count), and the view-window pins ``view_xlim`` / ``view_ylim`` (the
        Edit tab's range rows, #3) -- re-applied here so a rebuild / reopen keeps
        the operator's window instead of snapping back to autoscale.  (The y VALUE
        axis of a 1d panel is relim-owned; ``view_ylim`` exists only on the image
        family, whose ``apply_param`` gates it on ``y_is_view_axis``.)"""
        if self.plotter is None:
            return
        self._apply_lim_to_plotter()                     # relim family via the ONE in-place entry
        self._apply_unit()
        self._apply_view_lims()
        if "fit_request" in self.config.params:
            self.plotter.apply_param("fit_request", self.config.params.get("fit_request"))

    def _apply_view_lims(self) -> None:
        """Re-apply the persisted view-window pins (#3) to the LIVE plotter through the SAME
        ``apply_param`` entry every display knob uses -- a grid stores them and re-asserts them on every
        cell after each tick (:meth:`GridPlot._apply_view_knobs`), a flat panel sets its one axes -- so
        the operator's Apply survives a rebuild / reopen AND sticks across the live autoscale (never the
        old one-shot ``to_data_figure().xlim`` the next redraw wiped).  Absent (the default) leaves the
        plot's own autoscale untouched; a ``view_ylim`` on a non-image kind (a stale recipe) is refused
        by the plot's own ``y_is_view_axis`` gate."""
        if self.plotter is None:
            return
        for key in ("view_xlim", "view_ylim"):
            pin = self.config.params.get(key)
            if not pin:
                continue
            try:
                self.plotter.apply_param(key, (float(pin[0]), float(pin[1])))
            except Exception:
                pass                                      # a stale pin from a re-interpreted kind: ignore

    def _unit_df(self):
        """A DataFigure bound to the live card's figure/axes for the x-axis unit cycle
        (shared impl: :func:`_unit_df_for`)."""
        return _unit_df_for(self.plotter)

    def _unit_cycle_len(self) -> int:
        """Length of the x-axis unit cycle for the CURRENT plotter (0 if none)."""
        if self.plotter is None:
            return 0
        try:
            cmap = getattr(self._unit_df(), "conversion_map", None)
        except Exception:
            return 0
        return len(cmap) if cmap else 0

    def _apply_unit(self) -> bool:
        """Cycle the x-axis unit ``unit_index`` times from its original (reuse
        DataFigure.change_unit).  Returns True if any conversion was applied."""
        if self.plotter is None:
            return False
        index = int(self.config.params.get("unit_index", 0) or 0)
        if index <= 0:
            return False
        try:
            df = self._unit_df()
            if not getattr(df, "conversion_map", None):
                return False
            for _ in range(index % max(1, len(df.conversion_map))):
                df.change_unit()
            return True
        except Exception:
            return False

    def _refresh_title(self) -> None:
        """Compose the grey frame TITLE: the panel KIND + WHERE its signal comes from (the
        legend the console computes), e.g. ``1D vector — value ← Analysis``.  This is
        the ordinary QGroupBox title -- the grey chip, alignment and font are the frame's own."""
        head = PANEL_KINDS[self.config.kind]
        info = " · ".join(p for p in self._signal_info.splitlines() if p.strip())
        self.setTitle(f"{head} — {info}" if info else head)

    def set_signal_info(self, info: str) -> None:
        """Set the signal legend (computed by the console: which node each read comes from).
        Shown in the frame TITLE (the grey strip), replacing the old footer legend."""
        info = str(info or "")
        if info == self._signal_info:
            return
        self._signal_info = info
        self._refresh_title()

    def set_status(self, text: str, *, error: bool) -> None:
        # No per-shot status line in the panel any more (it needed a footer, which broke the
        # height proportion).  The status lives in the Setting popup + the Setting-button
        # tooltip; an error turns the Setting button red.  Restyle only on the ok<->error edge.
        # THREAD-SAFE by deferral: compose() may run on the console's render thread, and Qt
        # widget calls (setText/setStyleSheet) are GUI-thread-only -- park the newest status and
        # let the GUI flush it when the batch is presented (flush_deferred_status).  Keeping the
        # deferral INSIDE the one status funnel means compose never forks per thread.
        if QtCore.QThread.currentThread() is not self.thread():
            self._deferred_status = (str(text), bool(error))
            return
        self._status_text = str(text)
        if hasattr(self, "status"):
            self.status.setText(str(text)[:200])
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        if error is not getattr(self, "_status_error", None):
            self._status_error = bool(error)
            colour = RED if error else GREY
            if hasattr(self, "status"):
                self.status.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
            self.setting_button.set_color(colour)

    def flush_deferred_status(self) -> None:
        """Apply the newest status a render-thread compose parked (GUI thread, present phase)."""
        pending = self.__dict__.pop("_deferred_status", None)
        if pending is not None:
            self.set_status(pending[0], error=pending[1])

    # ------------------------------------------------------------- drag to grid
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt naming
        # only the border frame starts a drag; the canvas consumes its own events
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.pos()
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.raise_()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            new_pos = self.mapToParent(event.pos() - self._drag_offset)
            self.move(max(0, new_pos.x()), max(0, new_pos.y()))
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self._drag_offset is not None:
            self._drag_offset = None
            self.setCursor(QtCore.Qt.OpenHandCursor)
            # Record the raw drop pixel as this card's (col, row); the console then REORDERS the card
            # to the ORDER position nearest that drop (:func:`drop_index` via ``dropped`` -- a drop onto
            # a card's slot displaces it, a drop past the last card appends to the bottom), and
            # :func:`pack` recomputes every pixel top-left from the new order.
            col, row = max(0, self.x()), max(0, self.y())
            if (col, row) != (self.config.col, self.config.row):
                self.config.col, self.config.row = col, row
                self.changed.emit()
            self.dropped.emit(self)         # drag-release ONLY: the console snaps the drop seed
            self.layout_changed.emit()      # re-pack (even when dropped back near the same spot)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        super().resizeEvent(event)
        self._place_setting_button()

    # ------------------------------------------------------------- config edits
    def _on_title(self, text: str) -> None:
        """Update the plot title via the FRONTEND'S sealed title API.

        ``BaseLivePlot.title`` is the single source of truth and
        ``BaseLivePlot._apply_title`` routes through ``style.apply_title``,
        which paints the title at ``title_fontsize()`` with the correct pad.
        Calling ``ax.set_title(text)`` directly bypasses both -- matplotlib
        then uses its OWN ``axes.titlesize`` rcParam (NOT the frontend's
        ``axes.labelsize``-derived size), so the title visibly shrinks /
        grows after every edit.  Always go through the sealed API."""
        self.config.title = str(text)
        if self.plotter is not None and getattr(self.plotter, "ax", None) is not None:
            self._wait_render_idle()   # in-place artist mutation: own the figure first
            self.plotter.title = self.config.title
            self.plotter._apply_title()
            if self.canvas is not None:
                self.canvas.draw_idle()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        self._wait_render_idle()   # the teardown+regrow+rebuild below must own the figure
        self.config.size = str(size)
        # ATOMIC relayout transaction.  A size change is THREE synchronous steps: _reset_plot tears the
        # plotter down (canvas -> None, the holder goes momentarily EMPTY), _apply_fixed_size grows the
        # card to the new preset, _rerender_last rebuilds the canvas.  If ANY event-loop slice runs
        # between the grow and the rebuild -- a closing size-combo popup, matplotlib's own draw() -- the
        # user sees one frame of a BIG EMPTY card before the image lands = the "resize jump" (reproduced:
        # a 994x653 card with only the title strip painted, canvas still None).  Freezing THIS card's
        # repaints for the whole mutation (updates disabled -> the Agg buffer still renders, only the Qt
        # blit defers) makes the paint system composite one clean final frame: card at the new size WITH
        # the new-size canvas in place (verified: zero intermediate paintEvents).  The layout_changed
        # emit -- and so the sibling gravity repack it drives -- runs INSIDE the freeze, so this card's
        # final POSITION + size + image all land together; finally re-enables even if a rebuild raises.
        self.setUpdatesEnabled(False)
        try:
            self._reset_plot()
            self._apply_fixed_size()
            # If the Setting frame is open, GROW it immediately to match the new (taller) panel -- but
            # the high-water mark means a SMALLER size never snaps it shorter (#H3i-2).
            if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                self._size_settings_popup()
            self._rerender_last()   # re-draw at the new size NOW, even if the source is stopped (else the
                                    # torn-down panel stays blank until the next hub tick -- same as _set_param)
            self.changed.emit()
            self.layout_changed.emit()
        finally:
            self.setUpdatesEnabled(True)

    def current_selection(self):
        """Return this panel's selection in the displayed coordinate frame."""

        from zlc_data.plot_region import Selection

        selection = self._active_selection
        if selection is None:
            payload = self.config.params.get("selection")
            if isinstance(payload, Mapping):
                selection = Selection.from_dict(payload)
        if selection is None and self.plotter is not None:
            selection = self.plotter.to_data_figure().selection()
        if selection is None:
            selection = Selection()
        metadata = dict(selection.metadata)
        roi = getattr(self, "_roi_built", None)
        if roi and len(roi) >= 4:
            metadata["origin"] = [float(roi[0]), float(roi[2])]
        scope = selection.scope
        if self._grid_focus is not None:
            scope = (*scope, f"cell:{int(self._grid_focus.k)}")
        return Selection(selection.ranges, frame=selection.frame, scope=scope, metadata=metadata)

    def _selection_coordinates_for_binding(self) -> dict:
        """The plot's per-axis selection COORDINATES (a site map's centres, a 1-D panel's curve x) that
        :func:`live.region_binding` needs to bind an ROI drag to the consumed block's axes -- read from
        the SAME DataFigure adapter the selection itself comes from.  Empty when there is no plotter (a
        2-D / hist ROI needs none: pixel index / sample value carry no extra coordinate)."""
        if self.plotter is None:
            return {}
        try:
            df = self.plotter.to_data_figure()
        except Exception:
            return {}
        return {str(key): np.asarray(value)
                for key, value in dict(getattr(df, "_selection_coordinates", {})).items()}

    # ---- Analysis (curve fit + ROI): ONE state, ONE mutator --------------------------------------
    def _build_analysis_section(self, section_box, label_w) -> None:
        """Build the Setting popup's "Analysis" section through the ONE :class:`AnalysisControls` builder
        (shared VERBATIM with the Edit tab, ``surface='setting'``).  The composite owns the action / model
        / fix-seed / result widgets + their handlers; this only embeds it and aliases the test-keyed
        attribute names so nothing that keys off ``analysis_combo`` / ``fit_model_combo`` / ``fit_fix_seed``
        / ``fit_result_label`` has to change."""
        controls = AnalysisControls(self, surface="setting", label_w=label_w)
        # Kind offers neither a fit nor an ROI -> no Analysis section at all (empty controls discarded).
        if controls.empty:
            controls.deleteLater()
            self._analysis_controls = None
            self.analysis_combo = self.fit_model_combo = self.fit_fix_seed = self.fit_result_label = None
            return
        section_box("Analysis").addWidget(controls)
        self._analysis_controls = controls
        # The test-keyed aliases (single builder, unchanged public names): the card exposes the SAME
        # attributes as before, now backed by the shared composite.
        self.analysis_combo = controls.action_combo
        self.fit_model_combo = controls.model_combo
        self.fit_fix_seed = controls.fix_seed
        self.fit_result_label = controls.result_label

    def _default_fit_model(self) -> str:
        """The model a fresh curve fit uses: the stored request's model FIRST (single source once fit
        is on), else this panel family's first offered model."""
        saved = self.config.params.get("fit_request")
        if isinstance(saved, Mapping) and saved.get("model"):
            return str(saved["model"])
        models = _general_fit_models_for_kind(self._param_kind())
        return models[0].key if models else "gaussian"

    def _build_fit_request_from_widgets(self, model_combo, fix_seed, selection):
        """Build a fresh fit request from a surface's OWN model combo + fix/seed editor + the selection
        (through the ONE :func:`build_fit_request`).  Used when a curve fit is first turned on from the
        Setting popup or the Edit tab -- each surface passes ITS widgets, so neither reads the other."""
        model = (str(model_combo.currentData()) if (model_combo is not None and model_combo.currentData())
                 else self._default_fit_model())
        fixed, initial = fix_seed.values() if fix_seed is not None else ({}, None)
        return build_fit_request(model, selection, fixed=fixed, initial=initial,
                                 coordinate_frame=selection.frame)

    def _retarget_fit_request(self, selection):
        """Rebuild the ACTIVE fit's request onto a NEW selection, preserving its model + fixed/initial
        (the stored request is the single source) -- the drag-to-retarget path."""
        from zlc_data.curve_fitting import FitRequest
        saved = self.config.params.get("fit_request")
        req = FitRequest.from_dict(saved) if isinstance(saved, Mapping) else None
        model = req.model if req is not None else self._default_fit_model()
        fixed = dict(req.fixed) if req is not None else {}
        initial = req.initial if req is not None else None
        return build_fit_request(model, selection, fixed=fixed, initial=initial,
                                 coordinate_frame=selection.frame)

    def set_fit_request(self, request) -> None:
        """The ONE fit mutator.  Its argument's presence is the single 'fit is on' state: it stores the
        request (or None) as ``config.params['fit_request']``, applies it to the live plotter overlay in
        place, re-derives any open Analysis control from that key (Setting + result line), and syncs the
        console's hub FitProcessor node -- create/retarget for a 2-D image fit, remove otherwise.  Every
        fit path (the Analysis combo, an Edit Fit action, a drag retarget) funnels through here, so the
        overlay, the two surfaces, and the hub node can never hold divergent fit state (#8)."""
        payload = request.to_dict() if request is not None else None
        # A console-attached facet grid fits per-cell on its worker node (below) and only DISPLAYS the
        # published params: put it in display-only mode BEFORE apply_param, so the arming _set_param never
        # solves in place on the Qt thread (#6b).  ``fit_node_sink`` set == a console (a notebook grid has
        # none and keeps its in-place solve).
        plotter = self.plotter
        if request is not None and callable(self.fit_node_sink) \
                and hasattr(plotter, "apply_published_cell_fits") \
                and getattr(plotter, "_published_cell_popt", None) is None:
            plotter._published_cell_popt = {}
        self._set_param("fit_request", payload)          # store + apply overlay in place (apply_param)
        self._refresh_analysis_controls()
        if request is None:
            if callable(self.selection_clear_sink):
                self.selection_clear_sink(self, "fit")   # remove the hub FitProcessor this fit created
        elif callable(self.fit_node_sink):
            self.fit_node_sink(self, request)            # create/retarget the hub node (2-D image only)

    def _select_analysis_action(self, action, *, model_combo=None, fix_seed=None) -> None:
        """Apply a chosen post-drag action from EITHER surface: ``fit`` turns the curve fit ON (build +
        apply a request from the given model/fix-seed + the current selection), ``roi`` arms the crop,
        ``none`` clears both.  fit-vs-roi are the TWO independent analyses -- picking one clears the
        other so a single drag has one meaning; leaving ROI stops its node (#10)."""
        action = str(action or "none")
        if action == "fit":
            self.config.params["selection_action"] = "none"     # roi off; the fit lives in fit_request
            self.set_fit_request(
                self._build_fit_request_from_widgets(model_combo, fix_seed, self.current_selection()))
            return
        if self.config.params.get("fit_request"):
            self.set_fit_request(None)                          # leaving a fit removes overlay + node
        old = str(self.config.params.get("selection_action") or "none")
        self.config.params["selection_action"] = action
        if old == "roi" and action != old and callable(self.selection_clear_sink):
            self.selection_clear_sink(self, old)
        self._refresh_analysis_controls()
        self.changed.emit()

    def _refresh_analysis_controls(self) -> None:
        """Re-derive THIS card's Setting Analysis controls from state (fit_request presence +
        selection_action) -- called from :meth:`set_fit_request`, the section build, and Setting open.
        The Edit tab derives its OWN copy the same way, so both are pure views of the one source (#8)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.derive()
        if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
            self._size_settings_popup()

    def _fit_result_text(self, result=None) -> str:
        """The one-line fit-result string -- shared VERBATIM by the Setting AND the Edit result label
        (both surfaces are refreshed from it by :meth:`_set_fit_result_text` / the console's overlay push)."""
        result = result if result is not None else getattr(self.plotter, "_last_fit_result", None)
        if result is None:
            return "not fitted"
        if result.valid:
            quality = result.quality
            return (f"ok · {result.n_points} points · R²={quality.get('r2', float('nan')):.4g} · "
                    f"RMSE={quality.get('rmse', float('nan')):.4g}")
        return f"invalid · {result.status}"

    def _set_fit_result_text(self, result=None) -> None:
        """Push the fit-result string to this card's Setting Analysis result line (change-gated).  The
        Edit tab's copy is refreshed alongside by the console's overlay push (both surfaces, one source)."""
        controls = getattr(self, "_analysis_controls", None)
        if controls is not None:
            controls.set_result(self._fit_result_text(result))

    def _set_param(self, key: str, value) -> None:
        """A declarative parameter edit: store, apply, mark dirty.

        Most params (colormap / bins / toggles …) change the plot's STRUCTURE, so they tear the plotter
        down + re-render.  ``relim`` is the exception: it is a LIVE axis adjustment (the dead-band-aware
        autoscale mode), so it pushes onto the existing plotter WITHOUT a teardown and reveals the fixed
        lo/hi row only in ``fixed`` mode -- the SAME effect the old hand-wired ``_on_relim_mode`` had,
        now reached through this one declarative path (#H3v-4b)."""
        if self.config.params.get(key) == value:
            return
        self._wait_render_idle()
        self.config.params[key] = value
        if key == "relim":
            # Flipping INTO ``fixed`` FREEZES the current view: seed lo/hi from what the plot shows
            # NOW (the plotter's live limits), never the stale/default 0..1 pair -- pinning a counts
            # histogram or a camera image to 0..1 empties it (every bar/pixel outside the range),
            # which reads as "the enlarged plot just died".  The operator then types exact bounds.
            if str(value) == "fixed" and self.plotter is not None \
                    and hasattr(self.plotter, "current_lims"):
                lo, hi = self.plotter.current_lims()
                self.config.params["fixed_lo"], self.config.params["fixed_hi"] = lo, hi
                for edit, val in ((getattr(self, "fixed_lo_edit", None), lo),
                                  (getattr(self, "fixed_hi_edit", None), hi)):
                    if edit is not None:
                        edit.setText(f"{val:g}")     # setText does NOT re-fire editingFinished
            if getattr(self, "fixed_lim_row", None) is not None:        # the Setting popup's lo/hi row
                self.fixed_lim_row.setVisible(str(value) == "fixed")    # (the Edit tab toggles its own)
                # Revealing/hiding the lo/hi row CHANGES the popup's content height.  The Setting popup
                # is a free-floating top-level Qt.Popup whose window size is fixed at open time, so a bare
                # setVisible reflows the rows INSIDE that fixed window -> the layout "jumps".  Re-run the
                # ONE sizing rule (_size_settings_popup: top-left anchored, grow-down, never re-position,
                # high-water so it never snaps shorter) so the window grows DOWNWARD to fit the new row --
                # smooth single-direction expand, not a jump (#fixed-lim-row-jump).
                if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                    self._size_settings_popup()
            # A ZOOMED grid: route through the focused-param path so the mode lands on the enlarged
            # view AND is stored on the parked grid (the thumbnails carry it when the zoom returns);
            # otherwise apply to the live plotter directly (a GridPlot fans out to its cells itself).
            if self._grid_focus is not None:
                self._set_focused_grid_param(key, value)
                if str(value) == "fixed":
                    # land the just-seeded lo/hi on the parked grid too: its own flip-seed could only
                    # use the cells' auto envelope, but the operator froze the ENLARGED view -- the
                    # config values (read from that view above) are the authority.
                    self._set_focused_grid_param("fixed_lo", float(self.config.params.get("fixed_lo", 0.0)))
                    self._set_focused_grid_param("fixed_hi", float(self.config.params.get("fixed_hi", 1.0)))
            else:
                self._apply_lim_to_plotter()
            self.changed.emit()
            return
        # A GRID cell is currently ENLARGED (``_grid_focus`` set): a param edit must apply to the FOCUS view
        # and persist onto the PARKED grid, and NEVER mark a grid rebuild -- a grid rebuild swaps the focus
        # canvas out and bounces back to the thumbnail (the "adjusting a param jumps back to the main grid
        # view instead of staying zoomed" bug, #4).  Route through the dedicated focus-param path.
        if self._grid_focus is not None:
            self._set_focused_grid_param(key, value)
            self.changed.emit()
            return
        # Display-only knobs the plotter applies IN PLACE (e.g. log y-scale / bimodal-fit toggle): like
        # ``relim``, NO teardown -- a teardown + rebuild reflows the Qt holder so the plot visibly
        # resizes/flashes (#dis-resize).  Fall back to the rebuild path only for knobs it doesn't handle.
        if self.plotter is not None and self.plotter.apply_param(key, value):
            self.changed.emit()
            return
        # A STRUCTURE knob the plotter can't apply in place (e.g. a 2D colormap, a new history length):
        # mark a rebuild and schedule it on the DEBOUNCE timer rather than rebuilding inline.  A fast
        # burst of edits (scroll) restarts the timer, so the heavy teardown+rebuild runs ONCE after the
        # burst on the latest config.params -- not once per tick (the race source).  _build_plot itself
        # build-then-swaps at a fixed canvas size, so even the single rebuild never blanks the card.
        self._force_rebuild = True
        self._rebuild_timer.start()   # coalesce; _run_pending_rebuild fires after the burst settles
        self.changed.emit()

    def _run_pending_rebuild(self) -> None:
        """The debounce timer fired: perform the ONE coalesced rebuild for the latest config.params.
        Re-renders from the live/last namespace (so a stopped producer still rebuilds) -- _build_plot
        build-then-swaps atomically, so the card never blanks even if this rebuild raises."""
        if not self._force_rebuild:
            return
        self._wait_render_idle()
        self._rerender_last()

    def _apply_source(self) -> None:
        self._wait_render_idle()
        previous_inputs = tuple(self.config.inputs)
        self.config.set_source(self.source_edit.text())
        if tuple(self.config.inputs) != previous_inputs:
            self._refresh_signal_combo()
        self._compiled_source = self.config.source
        self._reset_plot()      # output shape may change with the expression
        # Force the console's NEXT tick to re-render this panel against a FRESH hub namespace (the
        # version gate honours -1), so re-picking a signal recovers a panel that previously errored
        # on a now-available signal -- never "remove the panel to recover" (#H3w-2).
        self._render_version = -1
        self._rerender_last()   # also re-evaluate on the last GOOD data at once (stopped node)
        self.apply_button.set_dirty(False)
        self.changed.emit()

    def _open_expr_editor(self) -> None:
        """Pop a NON-modal floating editor for the panel's source expression (the shared
        :class:`FluentFloatingEditor`): the panel behind stays VISIBLE and live, and the
        editor's Apply (below the box) writes the text back + re-renders so you watch the plot
        change.  Parented to this panel's TOP-LEVEL window so it shares the same screen scale
        (no DPI-mismatch shrink).  Re-clicking just raises the existing one."""
        existing = getattr(self, "_expr_editor", None)
        if existing is not None:
            existing.raise_(); existing.activateWindow(); return
        editor = FluentFloatingEditor(SIGNAL_EXPR_HELP, self.source_edit.text(), self.window(),
                                      title="Edit panel source expression")

        def _apply(text: str) -> None:
            # the source is one Python line -> collapse any newlines to spaces, write back to
            # the inline field, and apply live (the panel behind updates while the editor stays).
            self.source_edit.setText(" ".join(text.split("\n")).strip())
            self._apply_source()

        editor.applied.connect(_apply)
        editor.destroyed.connect(lambda *_: setattr(self, "_expr_editor", None))
        self._expr_editor = editor
        editor.show()

    def _rerender_last(self) -> None:
        """Re-render from the LAST namespace instead of waiting for the next hub tick.

        A display-only change (colormap / relim / a new source pick) tears the plotter
        down via :meth:`_reset_plot`; normally the next refresh rebuilds it.  But when the
        producing measurement is STOPPED the hub version is frozen, so ``_tick`` never calls
        ``refresh`` again -- the torn-down panel would stay blank (white).  Replaying the
        cached namespace rebuilds the plot at once.  No-op before the first render.

        Prefer the console's CURRENT shared namespace (a fresh hub snapshot = the SAME shot every
        other panel is on this tick) over this panel's own ``_last_namespace`` (a PAST tick): after a
        SIGNAL SWITCH or Stop/Start, replaying the stale cache would draw an OLDER shot than the
        siblings, so a 2D image and the sitemap would show different shots (#shot-coherence-on-switch).
        Fall back to the cached namespace only when no live one is available (or it lacks the new
        source -- e.g. a stopped producer whose frozen value lives only in the cache)."""
        live = None
        if callable(self.live_namespace_provider):
            try:
                live = self.live_namespace_provider()
            except Exception:
                live = None
        if live is not None and all(name in live for name in (self.config.inputs or ()) if name):
            self.refresh(live)
        elif self._last_namespace is not None:
            self.refresh(self._last_namespace)

    # ------------------------------------------------------------- data path
    def compose(self, namespace: dict[str, object], *, offthread: bool = False) -> bool:
        """Evaluate this panel's source against ``namespace`` and render the plot INTO ITS OFFSCREEN
        BUFFER -- phase 1 of the board's two-phase render: nothing is pushed to the screen here; the
        board presents every composed panel together in phase 2 so the whole board only ever shows ONE
        coherent shot.  Every failure lands on the gear/status -- a bad expression in one panel must
        never break the console or its siblings.

        ``offthread=True`` is the render-thread entry: identical logic, but a value that needs a
        STRUCTURAL (re)build -- a Qt-widget operation only the GUI thread may run -- is NOT rendered;
        the call returns False and the GUI thread re-runs the full compose for this panel when the
        batch lands.  In-place updates (the steady live stream) return True fully composed; status
        writes defer themselves inside set_status.  Returns True on the GUI thread always."""

        # A BLANK panel (a freshly added pure view, source not yet wired) sits
        # quietly with a hint -- it is decoupled, so it shows nothing until the
        # user picks a signal in its Setting.  Not an error.
        if not str(self._compiled_source).strip():
            self.set_status("pick a signal in Setting", error=False)
            return True
        try:
            # The three decoupled stages (#H3l): signal (per physical P slice) -> repeat reduce
            # (the measurement's own repeat axis, per the plot's repeat_mode) -> data-axis split.  The
            # measurement OWNS ``repeat``; the plot only chooses how to display it.
            value = self._signal_then_repeat(namespace)
            if offthread and self._needs_structural_build(value, namespace):
                return False
            self._render(value, namespace)
        except Exception as exc:
            # A failed eval (e.g. the bound signal is not in this namespace yet -- a still-WAITING
            # producer) must NOT latch the panel: keep the LAST GOOD namespace (set only on success
            # below), so re-picking a signal / a display change replays valid data, and a fresh hub
            # tick re-renders the moment the signal arrives -- never "remove the panel to recover"
            # (#H3w-2).
            self.set_status(str(exc).splitlines()[0][:160] or type(exc).__name__, error=True)
            return True
        # Remember the LAST SUCCESSFULLY-rendered namespace so a later display-only change (cmap /
        # relim / a source re-pick) can re-render immediately from it even when the producing
        # measurement is stopped (hub version frozen, no fresh tick) -- see _rerender_last.  Set ONLY
        # on success, so an error never overwrites it with a namespace that re-errors on replay.
        self._last_namespace = namespace
        shot = namespace.get("shot")
        self.set_status(f"shot {int(shot)}" if isinstance(shot, (int, float)) else "ok", error=False)
        return True

    def present(self) -> None:
        """Phase 2 of the board's two-phase render: flush this panel's composed buffer to the screen
        (schedule its Qt paint).  The board calls this for every panel it composed THIS tick, so they
        all repaint in ONE frame -- the board never shows a torn mix of shots.

        Always GUI-thread: this is where a hist that grew NEW threshold draggers during its
        worker-thread compose gets those draggers PARKED to the panel's Selectors switch (an mpl
        connect/disconnect the render worker must never do itself, #4-A)."""
        if self.plotter is not None and self.canvas is not None:
            if getattr(self.plotter, "_zlc_draggers_grew", False):
                self.plotter._zlc_draggers_grew = False
                self._apply_selectors_state()
            self.plotter.present()

    def refresh(self, namespace: dict[str, object]) -> None:
        """Compose + present this ONE panel now -- a single-card refresh (a source re-pick, the running
        task's mid-run panel).  The board (TaskConsole tick) instead splits these two phases across ALL
        its panels for cross-panel shot coherence; a lone card just composes then presents itself."""
        self.compose(namespace)
        self.present()

    def _signal_expr(self):
        """This panel's source as the ONE reusable :class:`SignalExpr` (the slot rule + the
        ``value = ...`` contract live there, shared with processors / pulse-scan).  Lazy import
        keeps the frontend module off neutral_atom's import graph (every neutral_atom use here
        is lazy)."""
        from zlc_data.signal_expr import SignalExpr
        return SignalExpr(self.config.inputs, self._compiled_source)

    def _signal_then_repeat(self, namespace: Mapping[str, object]):
        """Evaluate and reduce canonical signal tensors without rank inference."""
        from zlc_frontend.live_plot.live import reduce_repeat, repeats_with_data
        # A pulse panel's ``value`` is a STRUCTURED object (a sequence / PulseTableState), not an array --
        # it has no repeat axis and must NOT be float-coerced.  Read the bound signal (or the ``value =
        # ...`` expression result) as-is and hand it straight to _render / _build_plot, exactly as the
        # array pipeline hands a reduced block to the other kinds -- the kind (not the shape) decides how
        # its own data is consumed (a 2d reshapes to an image, sites uses centres, pulse uses the sequence).
        if self.config.kind == "pulse":
            self._repeat_cur = 1
            return self._signal_expr().evaluate(namespace)
        mode = self._repeat_mode_value()
        structure = self._bound_structure()
        if structure is not None:
            block = self._signal_expr().evaluate(namespace)
            b = self._validate_canonical_block(block, structure, self.config.inputs[0])
            valid = (namespace.get(SIG_VALID_KEY, {}) or {}).get(self.config.inputs[0])
            self._repeat_cur = repeats_with_data(b, valid=valid)
            if self.config.kind == "grid":
                return b
            return reduce_repeat(
                b, mode, valid=valid, hist=(self.config.kind == "hist"))

        block, had_repeat, valid = self._eval_signal_per_slice(namespace)
        b = np.asarray(block)
        if b.dtype.kind not in "iubf":
            b = np.asarray(b, dtype=float)
        if not had_repeat:
            self._repeat_cur = 1
            return b
        self._repeat_cur = repeats_with_data(b, valid=valid)
        return reduce_repeat(b, mode, valid=valid, hist=(self.config.kind == "hist"))

    @staticmethod
    def _validate_canonical_block(value, structure, name="signal") -> np.ndarray:
        from zlc_data.signal_tensor import canonical_physical_shape
        array = np.asarray(value)
        point_shape = tuple(int(n) for n in structure["points_shape"])
        data_shape = tuple(int(n) for n in structure["data_shape"])
        # (points, *data_shape) = the per-slice tail of the ONE canonical (R,P,*data) shape (single source).
        expected_tail = canonical_physical_shape(1, point_shape, data_shape)[1:]
        points = expected_tail[0]
        if array.ndim != 1 + len(expected_tail) or tuple(array.shape[1:]) != expected_tail:
            raise ValueError(
                f"signal {name!r} violates its canonical schema: expected (R,{points},"
                f"{','.join(map(str, data_shape))}), got {array.shape}")
        return array

    def _eval_signal_per_slice(self, namespace: Mapping[str, object]):
        """Evaluate a transformed expression once per declared R slice."""
        expr = self._signal_expr()
        sig = expr.signal_for(namespace)
        slots = sig if isinstance(sig, list) else [sig]
        structured: dict[str, tuple[np.ndarray, Mapping[str, object]]] = {}
        for name, value in zip(self.config.inputs, slots):
            structure = self.structure_provider(name) if callable(self.structure_provider) and name else None
            if structure is not None:
                structured[f"slot:{len(structured)}"] = (
                    self._validate_canonical_block(value, structure, name), structure)
        raw_names = []
        for name in expr.direct_names():
            if name == "signal" or name not in namespace:
                continue
            structure = self.structure_provider(name) if callable(self.structure_provider) else None
            if structure is not None:
                structured[f"raw:{name}"] = (
                    self._validate_canonical_block(namespace[name], structure, name), structure)
                raw_names.append(name)
        if not structured:
            ns = dict(namespace); ns["signal"] = sig
            return expr.exec_in(ns), False, None
        repeats = {array.shape[0] for array, _structure in structured.values()}
        if len(repeats) != 1:
            raise ValueError(f"expression inputs have incompatible repeat sizes {sorted(repeats)}")
        repeat = repeats.pop()
        valid_by_name = namespace.get(SIG_VALID_KEY, {}) or {}
        repeat_valid = np.ones(repeat, dtype=bool)
        for name in (*self.config.inputs, *raw_names):
            mask = valid_by_name.get(name)
            if mask is not None:
                mask = np.asarray(mask, dtype=bool)
                if mask.ndim != 2 or mask.shape[0] != repeat:
                    raise ValueError(f"signal {name!r} validity must be (R,P); got {mask.shape}")
                repeat_valid &= mask.any(axis=1)

        def _run(r):
            ns = dict(namespace)
            for name in raw_names:
                ns[name] = np.asarray(namespace[name])[r]
            sliced_slots = []
            for name, value in zip(self.config.inputs, slots):
                structure = self.structure_provider(name) if callable(self.structure_provider) and name else None
                sliced_slots.append(np.asarray(value)[r] if structure is not None else value)
            ns["signal"] = sliced_slots[0] if len(sliced_slots) == 1 else sliced_slots
            return np.asarray(expr.exec_in(ns), dtype=float)

        return np.stack([_run(r) for r in range(repeat)], axis=0), True, repeat_valid[:, None]

    def _bound_structure(self):
        """The producing node's ``{points_shape, data_shape, grid_shape}`` for this panel's bound
        signal -- authoritative for any IDENTITY source: the canonical ``value = signal`` OR a bare
        ``value = <the one input's name>`` (:func:`is_identity_source`), because naming the signal
        passes it through unchanged so the node's declared structure still describes it.  A transforming
        expression (``value = signal[0]-signal[1]``, ``value = np.log(f)``) rewrites the core shape (#H3o),
        so it returns ``None`` and the reshape falls back to shape inference."""
        from zlc_data.signal_expr import is_identity_source
        if not is_identity_source(self._compiled_source, self.config.inputs):
            return None
        if not (self.config.inputs and callable(self.structure_provider)):
            return None
        try:
            return self.structure_provider(self.config.inputs[0])
        except Exception:
            return None

    def _coerce(self, value):
        # The per-kind reshape (image / lines / samples / one-value-per-site) lives WITH the plots
        # now -- ``live.coerce_panel_value`` owns every plot kind's INPUT contract.  The console only
        # GATHERS the inputs (value + the bound node's structure + params + repeat mode) and dispatches,
        # so the wiring layer holds ZERO per-kind reshape logic (a plot kind's input shape can change
        # without touching task_console).
        return coerce_panel_value(
            self.config.kind, value,
            structure=self._bound_structure(),
            params=self.config.params,
            repeat_mode=self._repeat_mode_value())

    def _sites_aux(self, namespace: Mapping[str, object]):
        """Resolve a site map's centres and underlay from its value producer.

        The producer declares the companion signal names and publishes all three
        atomically, so the user binds one value signal without class-specific wiring.
        """
        occ = self.config.inputs[0] if self.config.inputs else ""
        centers_name, image_name = (None, None)
        if callable(self.sites_inputs_provider) and occ:
            try:
                centers_name, image_name = self.sites_inputs_provider(occ)
            except Exception:
                centers_name, image_name = (None, None)
        centers = namespace.get(centers_name) if centers_name else None
        if centers is None:
            raise ValueError(
                f"site map needs centres declared by `{occ}`'s producing node; "
                "bind a producer that supplies value, centres and frame companions.")
        centers = np.asarray(centers, dtype=float)
        if centers.ndim != 4 or centers.shape[:2] != (1, 1) or centers.shape[-1] != 2:
            raise ValueError(
                "centres signal must be canonical (1,1,N,2) with data_shape=(N,2); "
                f"got {centers.shape}")
        centers = centers[0, 0]
        image = namespace.get(image_name) if image_name else None
        if image is not None:
            image = np.asarray(image, dtype=float)
            if image.ndim != 4 or image.shape[1] != 1:
                raise ValueError(
                    "site underlay must be canonical (R,1,H,W) with data_shape=(H,W); "
                    f"got {image.shape}")
            from zlc_frontend.live_plot.live import reduce_repeat
            # Reduce only the declared repeat axis.  P remains explicit until we
            # select its sole point; no squeeze/rank guess is involved.
            mode = self._repeat_mode_value()
            valid = (namespace.get(SIG_VALID_KEY, {}) or {}).get(image_name)
            image = reduce_repeat(
                image[:, 0], "replace" if mode == "create" else mode, valid=valid)
            image = np.asarray(image).reshape(image.shape[-2:])
        return centers[:, :2], image

    def _co_names(self) -> frozenset:
        """The hub-signal names this panel reads (cached) -- for the monitor roll-gate
        and the 2D coordinate-frame (ROI) lookup.  This is BOTH the identifiers the source
        expression names directly AND the picked input: the default ``value = signal`` form
        references the pseudo ``signal`` (not the real name), so ``config.inputs`` is folded
        in or version-gating would miss the input's updates."""
        key = (self._compiled_source, tuple(self.config.inputs))
        if key != self._ref_src:               # (re)derive on source OR slot change
            self._ref_src = key
            self._ref_names = self._signal_expr().co_names()   # names referenced + picked slots
        return self._ref_names

    def _source_coord_frame(self, namespace: Mapping[str, object] | None):
        """The ROI ([x, w, y, h]) of the camera signal this 2D panel's source
        reads, or None -- so the image axes can be real pixel coordinates."""
        frames = (namespace or {}).get(COORD_FRAMES_KEY)
        if not isinstance(frames, dict) or not frames:
            return None
        for name in self._co_names():
            if name in frames:
                return frames[name]
        return None

    def _monitor_source_key(self, namespace: Mapping[str, object] | None):
        """Version key of the signals this panel's source reads, or None when
        none are detectable.  None => caller rolls every tick (safe fallback)."""
        versions = (namespace or {}).get(SIG_VERSIONS_KEY)
        if not isinstance(versions, dict) or not versions:
            return None
        refs = [n for n in self._co_names() if n in versions]
        if not refs:
            return None
        return tuple(sorted((n, versions[n]) for n in refs))

    def _wait_render_idle(self) -> None:
        """Hold until the console's render worker is idle -- a GUI path must OWN the figure before
        mutating it (frontend/render_loop.py ownership protocol).  No-op on a standalone card and
        free when the worker is idle.  Never call from the render thread itself."""
        if callable(self.render_barrier):
            self.render_barrier()

    def set_selectors_enabled(self, on: bool) -> None:
        """The console header's "Selectors" switch for THIS card: remember the desired state and
        gate the CURRENT plotter now (in place -- no rebuild, no flash).  Every later (re)build /
        focus swap re-applies it through ``_apply_selectors_state``, so a fresh figure always
        inherits the switch."""
        self._wait_render_idle()
        was_on = self._selectors_on
        self._selectors_on = bool(on)
        self._apply_selectors_state()
        # Turning the Selectors switch OFF tears down this panel's analysis whole (#1): the operator
        # has stopped driving it, so its Analysis node must not keep republishing.  An armed FIT
        # clears through the ONE fit mutator (:meth:`set_fit_request(None)`) so the persisted
        # ``fit_request`` state dies WITH the node -- never a zombie "fit on but computing nothing"
        # config the next save would replay (BUG-F).  Anything else (an armed ROI) goes straight
        # through the teardown seam; both funnel into ``_remove_panel_analysis`` (idempotent).
        if was_on and not on:
            if self.config.params.get("fit_request"):
                self.set_fit_request(None)
            elif callable(self.selection_clear_sink):
                self.selection_clear_sink(self, "selectors")

    def _apply_selectors_state(self) -> None:
        """Gate the current plotter's selector layer to the card's switch state (the ONE apply
        point every build / focus / tick path converges on).  Delegates to
        ``BaseLivePlot.set_selectors_active`` -- a safe no-op for a tool-less plot (the grid
        thumbnails) -- and aligns the canvas WHEEL policy with it: selectors ON isolates the
        wheel (in-plot scroll-zoom, the Edit-tab behaviour), OFF returns it to the board scroll.
        Only a plotter that actually HAS tools flips the wheel, so a grid card keeps scrolling
        the board either way."""
        plotter = self.plotter
        if plotter is None or not hasattr(plotter, "set_selectors_active"):
            return
        on = bool(self._selectors_on)
        plotter.set_selectors_active(on)
        canvas = self.canvas
        if canvas is not None and hasattr(canvas, "_zlc_isolate_wheel"):
            canvas._zlc_isolate_wheel = on and bool(plotter.interaction_handles())
        if canvas is not None:
            # Ownership barrier for every mouse path into this canvas (press / double-click /
            # wheel): wait for the render worker's in-flight batch before selector callbacks
            # mutate artists.  Hung here because this is the ONE apply point every build / focus
            # swap converges on, so a fresh or enlarged canvas always carries it (idempotent).
            canvas._zlc_render_barrier = self.render_barrier
        # Every family exposes the same DATA-selection callback.  Its DataFigure
        # adapter determines whether the gesture means x, xy, or value bounds;
        # selecting alone never implies either fit or ROI.
        bundles = list(getattr(getattr(plotter, "fig", None), "_zlc_grid_tools", ()) or ())
        if bundles:
            for cell_index, bundle in enumerate(bundles):
                area = getattr(bundle, "area", None)
                if area is not None:
                    area.callback = (lambda k=cell_index: self._forward_area_select(k)) \
                        if on and callable(self.area_select_sink) else None
        else:
            tools = getattr(getattr(plotter, "fig", None), "_zlc_tools", None)
            area = getattr(tools, "area", None)
            if area is not None:
                area.callback = self._forward_area_select \
                    if on and callable(self.area_select_sink) else None

    def _forward_area_select(self, cell_index: int | None = None) -> None:
        """Persist and forward one plot-independent selection.

        The displayed coordinate frame is retained for fitting.  Conversion to
        local frame indices belongs exclusively to the explicit ROI action.
        """
        bundles = list(getattr(getattr(self.plotter, "fig", None), "_zlc_grid_tools", ()) or ())
        tools = bundles[int(cell_index)] if cell_index is not None and int(cell_index) < len(bundles) \
            else getattr(getattr(self.plotter, "fig", None), "_zlc_tools", None)
        rng = list(getattr(getattr(tools, "area", None), "range", None) or [None] * 4)
        if any(v is None for v in rng):
            return
        data_figure = self.plotter.to_data_figure()
        target = data_figure.cell(int(cell_index)) if cell_index is not None else data_figure
        selection = target.selection()
        metadata = dict(selection.metadata)
        roi = self._roi_built
        if roi and len(roi) >= 4:
            metadata["origin"] = [float(roi[0]), float(roi[2])]
        scope = selection.scope
        if self._grid_focus is not None and not any(
                item == f"cell:{int(self._grid_focus.k)}" for item in scope):
            scope = (*scope, f"cell:{int(self._grid_focus.k)}")
        self._active_selection = type(selection)(
            selection.ranges, frame=selection.frame, scope=scope, metadata=metadata)
        self.config.params["selection"] = self._active_selection.to_dict()
        self.changed.emit()
        self.area_select_sink(self, self._active_selection)

    def _needs_structural_build(
        self,
        value,
        namespace: Mapping[str, object] | None,
        *,
        value_is_coerced: bool = False,
    ) -> bool:
        """Whether rendering ``value`` requires (re)building the plotter -- a Qt-widget operation
        ONLY the GUI thread may run.  False -> a pure in-place artist update (thread-agnostic).
        The ONE dirtiness rule: ``_render`` dispatches on it AND the render thread probes it
        first (``compose(offthread=True)``) so a structural panel is handed back to the board
        instead of touching Qt off-thread.  ``value`` is the raw ``_signal_then_repeat`` result
        (coerced here -- idempotent and zero-copy on the live path)."""
        if self._grid_focus is not None:
            return False                    # an enlarged cell's feed is in-place (or a silent skip)
        kind = self.config.kind
        if kind == "pulse":
            return True                     # a pulse panel rebuilds its structured figure every refresh
        if self.plotter is None or self._force_rebuild:
            return True
        if not value_is_coerced:
            value = self._coerce(value)
        if kind == "grid":
            if self._facet():
                # A cell-count change (the scan restructured) rebuilds through build-then-swap.
                return len(self._facet_cells(value)) != getattr(self.plotter, "n_cells", -1)
            return False                    # a recipe grid is a snapshot: never a per-tick rebuild
        if kind in ("2d", "1d", "sites") and tuple(np.shape(value)) != self._value_shape:
            return True
        if kind == "2d":
            # A ROI that *shifts* without resizing keeps the frame shape, but the
            # image's pixel coordinates moved -- rebuild so the axes track the ROI.
            roi = self._source_coord_frame(namespace)
            return (list(roi) if roi else None) != getattr(self, "_roi_built", None)
        return False

    def _render(self, value, namespace: Mapping[str, object] | None = None) -> None:
        # PERFORMANCE: push data with draw=False and queue ONE draw_idle per
        # panel -- rendering happens in Qt's paint pass (coalesced per frame),
        # never synchronously inside the refresh tick.  hist accepts any sample
        # count (HistogramFigure.update rebins itself), so a growing history
        # must NOT rebuild the whole plot every shot.
        #
        # A ``dis`` bins EXACTLY the array it is bound to -- it does NOT reach into any
        # calibration / processor to transform its input (decoupling: downstream never
        # knows an upstream node's internals).  Bind it to a processor's per-site COUNTS to
        # see the bimodal readout; bind it to a raw frame and it histograms the frame's
        # pixels, honestly.  Whatever the source gives the dis is what the dis bins.
        # A GRID cell is ENLARGED into a standalone plot-kind figure (``_grid_focus`` set): the live tick
        # must NOT rebuild the grid over it, or the enlarged view (and any lim edit on it) would bounce back
        # to the thumbnail.  A LIVE facet grid still feeds the ENLARGED view (update_cells' host-focus
        # channel touches ONLY the focus plotter; the thumbnails redraw once on unfocus); a recipe grid
        # stays a gated snapshot.
        if self._grid_focus is not None:
            if self.config.kind == "grid" and self._facet():
                parked = self._grid_focus
                per_cell = self._facet_cells(self._coerce(value))
                if len(per_cell) == parked.grid.n_cells:
                    parked.grid.update_cells(per_cell, focus=(self.plotter, parked.k))
                    # a hist focus feed can grow NEW threshold draggers inside update_core; the
                    # plotter flags that (``_zlc_draggers_grew``) and the GUI re-parks them to the
                    # switch at present() -- never here (this runs on the render worker; a selector
                    # connect/disconnect off the GUI thread is what drove the drag self-deadlock, #4-A).
                    if self.plotter is not None:
                        self.plotter.compose()
            return
        # A recipe grid is a frozen artifact, not a view of this tick's scalar placeholder.
        # Once built, steady refreshes return before the canonical signal coercer (which correctly
        # rejects the numeric placeholder as grid data).  The FIRST tick and an explicit rebuild must
        # continue into ``_build_plot`` so the recipe carried by the producing node actually becomes a
        # plotter; the placeholder is intentionally ignored by that build branch.
        recipe_grid = self.config.kind == "grid" and not self._facet()
        if recipe_grid and self.plotter is not None and not self._force_rebuild:
            return
        if not recipe_grid:
            value = self._coerce(value)
        kind = self.config.kind
        if self._needs_structural_build(value, namespace, value_is_coerced=True):
            # STRUCTURAL (re)build -- creates the plotter + its Qt canvas, so this branch runs on
            # the GUI thread only (compose(offthread=True) hands such panels back to the board).
            # A pulse panel renders a STRUCTURED figure (a whole timeline) from its sequence
            # object; there is no in-place array ``update`` for it, so it (re)builds the
            # PulseSequenceFigure through _build_plot every refresh -- cheap (a static,
            # rarely-changing recipe) and always faithful.
            self._build_plot(value, namespace)             # build-then-swap: a failed build keeps the old figure
            self._force_rebuild = False                    # cleared ONLY after a clean build (else retry next tick)
            if kind == "grid":
                if self._facet():
                    # Every (re)build is followed by the Setting-rows sync: the build is the ONE
                    # point every kind-changing path converges on (a pick, a load, an expression edit).
                    self._sync_settings_param_rows()
                return
            if kind != "pulse":
                self._value_shape = (1,) if isinstance(value, float) else tuple(np.shape(value))
            if kind == "monitor":
                # The build already plotted this value as the first point; record
                # its source version so the next UNRELATED bump won't duplicate it.
                self._last_monitor_key = self._monitor_source_key(namespace)
            return
        # IN-PLACE update -- pure numpy + matplotlib artist work into the offscreen Agg buffer,
        # thread-agnostic (this is what the console's render thread executes every steady tick).
        if kind == "grid":
            if self._facet():
                # A LIVE facet grid: every tick moves the cells IN PLACE (update_cells -- the grid
                # counterpart of a standalone kind's update).
                self.plotter.update_cells(self._facet_cells(value))
                if self.canvas is not None:
                    self.plotter.compose()
            # a RECIPE grid is a SNAPSHOT built from its recipe: NEVER a per-tick redraw (#3).
            return
        if kind == "2d":
            self.plotter.update(np.asarray(value).ravel(), draw=False)
        elif kind == "monitor":
            # Roll ONE point per new sample of this monitor's own source.  With
            # several nodes, an unrelated node bumps hub.version and refreshes
            # every panel; without this gate the monitor would append a duplicate
            # point each time.  When the source's referenced signals are
            # undetectable (e.g. it reads via history("name")) we fall back to
            # rolling every tick -- never worse than before.
            key = self._monitor_source_key(namespace)
            if key is not None and key == self._last_monitor_key:
                return
            self._last_monitor_key = key
            self.plotter.roll(value, draw=False)
        elif kind == "sites":
            self.plotter.update(value, draw=False)
            if namespace is not None:           # refresh the camera underlay too
                _, image = self._sites_aux(namespace)
                self.plotter.set_background(image)
        elif kind == "1d" and self.config.params.get("xy") and np.ndim(value) == 2:
            # an x-y curve whose point count is unchanged (e.g. a finished scan
            # refreshed again): the x column is fixed at build time, so only the
            # y column updates in place (Live1D plots data_y vs its data_x[:, 0]).
            self.plotter.update(np.asarray(value)[:, 1], draw=False)
        elif kind == "1d":
            # a reduced scan curve refreshed in place: push the (points, ncols) y AND the current
            # repeat count so the "xN" ylabel tracks how many repeats have data.  Colours cycle by
            # column index inside Live1D (confocal-exact), so the panel no longer styles per-line.
            self.plotter.update(value, repeat_cur=int(getattr(self, "_repeat_cur", 1)), draw=False)
        else:  # hist / 1d-vector
            self.plotter.update(value, draw=False)
        # an in-place update can grow NEW selector handles (a hist whose threshold count changed
        # rebuilds its draggers inside update_core).  This runs on the RENDER WORKER, so it must NOT
        # re-park them here (an mpl connect/disconnect off the GUI thread + the canvas.draw() an mpl
        # set_active triggers is the drag self-deadlock, #4-A): the plotter flags the growth
        # (``_zlc_draggers_grew``) and the GUI re-parks at present().
        # COMPOSE this panel's buffer (blit: ~0.6 ms restore chrome bg + redraw data artists); nothing is
        # pushed to the screen here.  The board PRESENTS every composed buffer TOGETHER in phase 2 so the
        # whole board flips ONE coherent shot; a lone single-card refresh() presents right after.  A chrome
        # change full-renders into the buffer inside compose().
        if self.canvas is not None:
            self.plotter.compose()

    def _source_axis_label(self) -> str | None:
        """The y-axis label for this panel's sourced signal, taken from the PRODUCING
        node's :class:`SignalSpec` (``axes_provider``) -- so a plot of ``rate`` is
        labelled "loading rate" by the measurement that makes it, not a per-kind string.
        ``None`` when the source is a free expression or its source node declares no axis."""
        source = (self.config.source or "").strip()
        if not source.startswith("value ="):
            return None
        name = source[len("value ="):].strip()
        # The default source ``value = signal`` plots the picked input: resolve ``signal`` to
        # the real hub-signal name (``config.inputs[0]``) so its producing node's axis label
        # is found.
        if name == "signal":
            name = self.config.inputs[0] if self.config.inputs else ""
        return self._axis_label_for(name)

    def _axis_label_for(self, name: str | None) -> str | None:
        """The axis label a producing node declares for hub signal ``name`` (via
        ``axes_provider`` -> the node's :class:`SignalSpec`), or None."""
        if not name or not callable(self.axes_provider):
            return None
        try:
            axes = dict(self.axes_provider())
        except Exception:
            return None
        entry = axes.get(str(name))
        return entry[0] if entry else None

    def _panel_labels(self, xlabel: str, ylabel: str, zlabel: str = "") -> tuple[str, str, str]:
        """The (x, y, z) axis labels this panel draws.  A REPRODUCED figure seeds the SAVED labels as
        ``xlabel`` / ``ylabel`` / ``zlabel`` panel params (figure_viewer._seed_state, from the ONE
        ``SavedFigure.axis_labels`` source) so a reopened panel draws the SAME axes it was saved with --
        the params are then the single source; a LIVE panel carries no such params and keeps the kind's
        reconstructed defaults passed in here.  A NON-EMPTY override wins (``axis_labels`` only ever seeds
        non-empty labels); an absent OR empty param falls back to the default, so this never blanks a live
        axis (matching the xy-curve branch's own ``params.get('ylabel', '') or label``)."""
        p = self.config.params
        return (str(p.get("xlabel") or xlabel), str(p.get("ylabel") or ylabel), str(p.get("zlabel") or zlabel))

    # ------------------------------------------------------------- plot lifecycle
    def _fixed_lim_kwargs(self) -> dict:
        """``fixed_lo``/``fixed_hi`` kwargs for the plotter, ONLY when the lim mode is
        "fixed" (#8) -- otherwise omitted so the plotter keeps its tight/normal autoscale."""
        if self._relim() != "fixed":
            return {}
        return {"fixed_lo": float(self.config.params.get("fixed_lo", 0.0)),
                "fixed_hi": float(self.config.params.get("fixed_hi", 1.0))}

    def _view_kwargs(self, kind: str) -> dict:
        """The relim / limit view kwargs EVERY relim-capable kind passes to its plotter -- the ONE
        source for BOTH the live build (``_build_plot``) and the Edit snapshot (``PanelEditor.rebuild``),
        so no kind can silently omit relim/fixed.  This is exactly what regressed: the ``sites`` panel
        omitted these in both builders, so a Site map's lim mode / fixed lo-hi never re-rendered (#4).
        For a histogram the relim/fixed pins the VALUE (x) axis (the count y-axis is always auto, #3);
        every other kind pins its value axis (1D y-axis / 2D·sites clim)."""
        return {"relim_mode": self._relim(), **self._fixed_lim_kwargs()}

    def _build_plot(self, value, namespace: Mapping[str, object] | None = None) -> None:
        if panel_canvas is None:
            raise RuntimeError("matplotlib Qt canvas is not available")
        kind = self.config.kind
        size = self.config.size
        label = self.config.title or PANEL_KINDS[kind]
        if kind == "pulse":
            # A pulse panel renders its full timeline through the ONE pulse renderer in the PLOT LAYER
            # (``live.build_pulse_preview_plot`` -- the SAME one the editor + the reopened-recipe path use;
            # NEVER imported from the pulse_gui app -- see test_pulse_render_single_source), so a seeded
            # pulse figure is faithful (every digital channel / analog bus trace / repeat bracket), not a
            # flattened line.  The reproduction STATE (a PulseTableState) is an OBJECT the float-only hub
            # cannot carry, so it is resolved off the SAME producing node via ``pulse_state_provider`` --
            # the SAME "aux data from the producing node" wiring the site map uses for its centres/frame
            # (the hub ``value`` here is only a numeric placeholder).  The figure keeps its own spec-owned
            # geometry (frontend-owned); its own repeat brackets carry ×N, so there is no repeat_mode.
            from zlc_frontend.live_plot.live import build_pulse_preview_plot

            resolved = self.pulse_state_provider(self.config.inputs[0]) \
                if (callable(self.pulse_state_provider) and self.config.inputs) else None
            if resolved is None:
                raise ValueError(
                    "pulse panel needs a pulse figure's producing node -- point it at a loaded pulse "
                    "figure's fig_value signal (it carries the PulseTableState to reproduce).")
            state, node_include_off = resolved
            # The "show off rows" toggle is the panel's own display param (seeded from the saved value);
            # fall back to the node's recorded value when the param is unset -- so toggling re-renders.
            include_off = bool(self.config.params.get("include_always_off", node_include_off))
            # size drives the geometry like every other kind (config.size); interactions=True builds the
            # selector layer, then _apply_selectors_state (end of this build) PARKS it to the header's
            # "Selectors" switch -- OFF (default) keeps the Monitor card display-only exactly as before,
            # ON arms zoom / area / cross in place (the same rule the sites / 2d / hist / 1d branches
            # apply below).  The Edit tab stays the always-interactive surface.
            plotter, _channels, _repeat = build_pulse_preview_plot(
                state, include_always_off=include_off, size=size, interactions=True)
        elif kind == "grid":
            # A grid panel renders its per-site distribution / kernel grid through the ONE grid builder in
            # the PLOT LAYER (``live.build_grid_figure`` -- the SAME one ``na.load_figure(npz).plot()`` and
            # the report use).  Its reproduction RECIPE (a dict) is an OBJECT the float-only hub cannot
            # carry, so it is resolved off the producing node via ``grid_recipe_provider`` -- the SAME "aux
            # data from the producing node" wiring the pulse panel + site map use (the hub ``value`` here is
            # only a numeric placeholder).  Its geometry (cell count-driven size) is frontend-owned.
            # Build every cell's selector bundle; the card disables GridPlot's
            # notebook focus callback below so its own focus host remains the sole
            # double-click owner.
            facet = self._facet()
            if facet:
                # A FACET grid: the bound block IS the data -- slice it along the declared axis
                # (facet_cells, the ONE rule) and build through the ONE shared builder (the Edit-tab
                # snapshot uses the same one); every tick after this first build moves the cells in
                # place (update_cells in _render).
                plotter = self._build_facet_plotter(value, interactions=True)
            else:
                from zlc_frontend.live_plot.live import build_grid_figure

                recipe = self.grid_recipe_provider(self.config.inputs[0]) \
                    if (callable(self.grid_recipe_provider) and self.config.inputs) else None
                if recipe is None:
                    raise ValueError(
                        "grid panel needs a grid figure's producing node (a loaded grid figure's "
                        "fig_value signal) -- or pick a facet in Setting to expand a measurement "
                        "block's axis into cells.")
                recipe = self._grid_recipe_with_params(recipe)   # fold the panel's live display knobs in
                plotter = build_grid_figure(recipe, interactions=True, size=size, display=False)
            click_cid = getattr(plotter, "_click_cid", None)
            if click_cid is not None:
                plotter.fig.canvas.mpl_disconnect(click_cid)
                plotter._click_cid = None
        elif kind == "sites":
            centers, image = self._sites_aux(namespace or {})
            vec = np.asarray(value, dtype=float).reshape(-1)
            if len(vec) != len(centers):
                raise ValueError(
                    f"site-map value has {len(vec)} entries but the centers signal has {len(centers)} sites")
            plotter = panel_plot(
                centers, vec, kind="sites", size=size, interactions=True,
                image=image, roi_radius=site_ring_radius(centers),
                cmap=_resolved_cmap("sites", self.config.params),   # operator pick, else the kind default (ONE resolver)
                **self._view_kwargs("sites"),
                labels=self._panel_labels("Camera x (px)", "Camera y (px)", label),
                title=self.config.title or None)
        elif kind == "2d":
            arr = np.asarray(value, dtype=float)
            ny, nx = arr.shape
            # Coordinate axes ARE the source's pixel space: when the producing node
            # declares a spatial region, its endpoints [x_min, x_max, y_min, y_max]
            # give the axis ORIGIN (index 0 = x_min, index 2 = y_min); the image x/y
            # are the REAL pixels (x_min..x_min+nx), not 0..N -- so the axes match
            # the camera window and a selection maps straight back to a new region.
            roi = self._source_coord_frame(namespace)
            self._roi_built = list(roi) if roi else None
            if roi and len(roi) >= 4:
                x0, y0 = float(roi[0]), float(roi[2])
                xlabel, ylabel = "Camera x (px)", "Camera y (px)"
            else:
                x0, y0 = 0.0, 0.0
                xlabel, ylabel = "X (px)", "Y (px)"
            from zlc_data.raster import RegularRaster
            data_x = RegularRaster((ny, nx), origin=(x0, y0))
            plotter = panel_plot(
                data_x, arr.ravel(), kind="2d", size=size, interactions=True,
                cmap=_resolved_cmap("2d", self.config.params),   # operator pick, else the kind default (ONE resolver)
                **self._view_kwargs("2d"),
                # x / y / colour-bar all via the ONE _panel_labels source: a LIVE 2D panel labels its
                # colour bar with the bound signal's own axis label (a camera frame's "Counts"); a
                # REPRODUCED 2D figure overrides x / y / colour bar with the SAVED labels seeded into the
                # panel params (_seed_state), so the reopened image draws the axes it was saved with.
                labels=self._panel_labels(xlabel, ylabel, self._source_axis_label() or ""),
                title=self.config.title or None)
        elif kind == "monitor":
            # Declared PANEL_PARAMS defaults via the ONE _resolved_param resolver (never a
            # re-typed consume-site literal); the >=20 floor mirrors the decl's lo bound.
            length = max(20, int(_resolved_param(kind, self.config.params, "length")))
            history = np.full(length, np.nan)
            plotter = panel_plot(
                np.arange(length, dtype=float), history, kind="monitor", size=size, interactions=True,
                show_dist=bool(_resolved_param(kind, self.config.params, "show_dist")),
                labels=self._panel_labels("Shots ago", self._source_axis_label() or label),
                **self._view_kwargs("monitor"),
                title=self.config.title or None)
            plotter.roll(float(value), draw=False)
        elif kind == "hist":
            plotter = panel_plot(
                np.asarray(value, dtype=float), kind="hist", size=size, interactions=True,
                bins=int(_resolved_param(kind, self.config.params, "bins")),
                ylog=bool(_resolved_param(kind, self.config.params, "ylog")),
                fit=str(_resolved_param(kind, self.config.params, "fit")),
                **self._view_kwargs("hist"),                # relim/fixed pins the VALUE (x) axis (#3)
                labels=self._panel_labels("Value", "Shots"), title=self.config.title or None)
        else:  # 1d
            arr = np.asarray(value, dtype=float)
            if self.config.params.get("xy") and arr.ndim == 2 and arr.shape[1] == 2:
                # an x-y curve (col0 = x, col1 = y): plot y vs the supplied x
                # (a scanned measurement's result panel uses this -- value =
                # column_stack([x_key, y_key]) grows the curve over the scan).
                data_x, vec = arr[:, 0], arr[:, 1]
                xlabel = str(self.config.params.get("xlabel", "X"))
                ylabel = str(self.config.params.get("ylabel", "")) or label
            else:
                # A reduced scan curve is ``(points, ncols)`` -- KEEP it 2-D so each column draws as
                # its own line (the one-dimensional data_shape, or one line per repeat in ``create`` mode); a
                # plain vector stays 1-D (one line).  npts = the point count either way.
                vec = arr if arr.ndim == 2 else arr.reshape(-1)
                npts = arr.shape[0] if arr.ndim == 2 else vec.size
                ylabel = self._source_axis_label() or label
                # A scan's y curve: draw it vs the companion x signal from the SAME
                # producing node (its axis label/unit come with it) -- so a 1d plot wired
                # to ``temperature_survival`` reads "Trap-off time (s)" on x (#3).  Falls
                # back to a per-site index when there is no companion x.
                x_name = None
                if self.config.inputs and callable(self.curve_x_provider):
                    try:
                        x_name = self.curve_x_provider(self.config.inputs[0])
                    except Exception:
                        x_name = None
                x_vals = (namespace or {}).get(x_name) if x_name else None
                x_arr = None if x_vals is None else np.asarray(x_vals, dtype=float).reshape(-1)
                if x_arr is not None and x_arr.size == npts:
                    data_x, xlabel = x_arr, (self._axis_label_for(x_name) or "X")
                else:
                    data_x, xlabel = np.arange(npts, dtype=float), "Site"
            plotter = panel_plot(
                data_x, vec, kind="1d", size=size, interactions=True,
                labels=self._panel_labels(xlabel, ylabel),
                **self._view_kwargs("1d"),
                title=self.config.title or None)
            # Colours cycle by column index inside Live1D (confocal-exact: grey, skyblue, ...; a lone
            # line is grey, every line alpha=1 / linewidth=1) -- no per-line styling needed here.
            # The plot shows the repeat count as a "xN" ylabel suffix (the panel computed it while
            # reducing the measurement's repeat axis).  Apply it NOW (an in-place update with the same
            # data) so the "xN" shows on the first build too -- not only after the next live tick.
            rc = int(getattr(self, "_repeat_cur", 1))
            plotter.repeat_cur = rc
            if rc != 1:
                plotter.update(plotter.data_y, repeat_cur=rc, draw=False)
        # ATOMIC build-then-swap: the new plotter is now fully built.  Capture the OLD canvas/figure,
        # INSERT the new canvas FIRST, then remove the old -- so the holder is NEVER empty during the
        # swap (no one-frame blank).  A build error above (e.g. the sites length check) returns before
        # here, leaving the previous figure untouched (#4 "图有时消失").  The card is setFixedSize, so the
        # swap cannot reflow its geometry -- the plot never visibly resizes/jumps either (#5 "图变大").
        old_canvas, old_plotter = self.canvas, self.plotter
        self.plotter = plotter
        # Re-apply the persisted Setting toggles (unit + x/y limits + x-window) to the FRESH plotter NOW
        # -- BEFORE wrapping it in the canvas -- so the canvas's construction render draws the FINAL
        # content ONCE.  Applying them AFTER the canvas (as it used to) rendered pre-knob content, then
        # re-rendered: 2+ full re-renders of a heavy 36-cell grid.  Each apply_param itself redraws the
        # whole figure, so suspend the plotter's per-knob draws; the canvas's own render is the single one.
        with self.plotter.suspend_draws():
            self._apply_display_params()
        # Monitor cards default to display-only: the selector layer is built but PARKED
        # inactive (see _apply_selectors_state below, gated by the header's "Selectors"
        # switch) and the wheel scrolls the board (isolate_wheel=False) instead of being
        # swallowed.  Switch ON arms the selectors in place -- Edit-tab behaviour on the
        # live board; _apply_selectors_state also flips the wheel policy to match.
        self.canvas = panel_canvas(self.plotter.fig, isolate_wheel=False)
        # The canvas OWNS its size: EmbeddedFigureCanvas setFixedSize's itself to its DPR-invariant
        # design size at construction (and renders its buffer synchronously), so the host no longer pins
        # setMinimumSize(sizeHint()) on top -- that two-pin + DPR-derived-sizeHint race is what ballooned
        # the figure.  The card is setFixedSize to hold the canvas + the proportional bottom padding the
        # trailing stretch absorbs.
        add_stretch = self.canvas_holder.count() == 0       # first build (no canvas, no stretch yet)
        # canvas pins to the TOP of the content (right below the grey title strip); the trailing
        # stretch is the proportional bottom padding (collapses to ~0 for a 1-row card).
        self.canvas_holder.insertWidget(0, self.canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        if add_stretch:
            self.canvas_holder.addStretch(1)
        # NOW retire the old canvas/figure -- the new one is already in the holder, so there is no blank
        # window between remove and insert (the swap is atomic from the user's view).
        if old_canvas is not None:
            self.canvas_holder.removeWidget(old_canvas)
            old_canvas.setParent(None)
            old_canvas.deleteLater()
        if old_plotter is not None and plt is not None and old_plotter.fig is not None:
            plt.close(old_plotter.fig)
        # park (or arm) the fresh plotter's selector layer to the header's "Selectors" switch --
        # a rebuild must inherit the current state, never come up with live selectors while OFF.
        self._apply_selectors_state()
        # The canvas already rendered the FINAL content at construction (the display knobs were applied to
        # the plotter ABOVE, before it was wrapped).  Re-render only if the selector-state pass dirtied the
        # figure -- otherwise a no-op, not a second full render of the heavy grid.
        self.canvas._zlc_draw_if_needed()
        self._place_setting_button()
        # A GRID panel is display-only, so its own focus-zoom is dormant -- THIS card catches the double-click
        # on the grid canvas and swaps to the clicked cell's standalone plot-kind figure (build_focus_plotter).
        if kind == "grid" and self._grid_focus is None:
            self._connect_grid_focus_click(self.plotter)
            # A rebuild that interrupted an ENLARGED cell (recorded by _teardown_plot) re-focuses the SAME
            # cell on the fresh grid -- a size/structure edit while zoomed stays zoomed (#no-focus-bounce).
            k = self._pending_refocus_k
            self._pending_refocus_k = None
            if k is not None and 0 <= k < int(getattr(self.plotter, "n_cells", 0)):
                self._focus_grid_cell(self.plotter, k)
        else:
            self._pending_refocus_k = None      # a non-grid rebuild has no cell to restore

    def _connect_grid_focus_click(self, grid) -> None:
        """Wire a double-click on the GRID canvas to enlarge the clicked cell into its standalone plot-kind
        figure (and Esc / another double-click to return).  The grid panel is display-only, so it has no
        selectors; this is the ONE handler the console adds for the focus-zoom -- it reuses the grid's own
        ``site_axes`` for hit-testing and :meth:`GridPlot.build_focus_plotter` for the enlarged view."""
        canvas = getattr(grid.fig, "canvas", None)
        if canvas is None:
            return
        self._grid_click_cid = canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)

    def _on_grid_canvas_click(self, event) -> None:
        # Only a LEFT double-click toggles focus (mirror GridPlot._on_click -- some backends emit wheel /
        # middle as dblclick button 2/4/5).
        if not getattr(event, "dblclick", False) or getattr(event, "button", 1) != 1:
            return
        if self._grid_focus is None:
            grid = self.plotter
            axes = list(getattr(grid, "site_axes", []) or [])
            if event.inaxes in axes:
                self._focus_grid_cell(grid, axes.index(event.inaxes))
        else:
            self._unfocus_grid_cell()

    def _on_grid_focus_key(self, event) -> None:
        if getattr(event, "key", None) == "escape" and self._grid_focus is not None:
            self._unfocus_grid_cell()

    def _focus_grid_cell(self, grid, k: int) -> None:
        """Enlarge grid cell ``k`` into its STANDALONE plot-kind figure and SWAP it into this card's canvas,
        parking the grid (plotter + canvas) to swap back on unfocus.  The enlarged view is the STOCK ``2x2``
        panel of the cell's kind -- the SAME size the notebook path (:meth:`GridPlot.focus`) enlarges to,
        never the grid's own (possibly larger) preset -- CENTRED in the card's plot region, with the
        STANDARD relim view kwargs so a subsequent lim / fit / relim edit reaches it through the ordinary
        _set_param / _apply_lim_to_plotter path; the live tick is gated off (``_grid_focus`` set), so it
        never bounces back to the thumbnail."""
        if panel_canvas is None:
            return
        # the enlarged view is a standalone panel of the cell's per-site kind -> its standard relim view
        # kwargs.  interactions=True builds its selector layer; _apply_selectors_state (below) parks it
        # to the header's "Selectors" switch -- OFF (default) keeps the enlarged view display-only as
        # before, ON arms zoom / area / cross / the threshold drag (the same rule every _build_plot
        # branch applies).  The Edit tab stays the always-interactive surface.
        sub_kind = str(getattr(grid, "sub_plot_kind", "hist"))
        focus = grid.build_focus_plotter(
            k, size="2x2", interactions=True, **self._view_kwargs(sub_kind))
        focus_canvas = panel_canvas(focus.fig, isolate_wheel=False)
        # atomic swap-in (mirror _build_plot): insert the focus canvas, remove the grid canvas but do NOT
        # close the grid figure -- it is parked for the return.  A TOP stretch balances the holder's
        # trailing one, so the stock-size enlarged view sits at the CENTRE of the (larger) grid region
        # instead of pinned to the top-left.
        grid_canvas = self.canvas
        self.canvas_holder.insertWidget(0, focus_canvas, alignment=QtCore.Qt.AlignHCenter)
        # SAME stretch factor (1) as the holder's trailing addStretch(1): equal factors split the
        # spare height evenly above/below -- a factor-0 spacer would win nothing against the
        # trailing factor-1 stretch and the enlarged view would pin to the top, not the centre.
        self.canvas_holder.insertStretch(0, 1)
        self._focus_top_stretch = self.canvas_holder.itemAt(0)
        if grid_canvas is not None:
            self.canvas_holder.removeWidget(grid_canvas)
            grid_canvas.setParent(None)
        self.plotter = focus
        self.canvas = focus_canvas
        self._grid_focus = _GridFocus(grid=grid, grid_canvas=grid_canvas, k=int(k))
        self._grid_focus.key_cid = focus_canvas.mpl_connect("key_press_event", self._on_grid_focus_key)
        self._grid_focus.click_cid = focus_canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)
        self._apply_display_params()        # re-apply unit / manual lims to the fresh focus plotter
        self._apply_selectors_state()       # park/arm the enlarged view's selectors to the switch
        focus_canvas.draw()
        self._place_setting_button()

    def _set_focused_grid_param(self, key: str, value) -> None:
        """Apply a param edit while a grid cell is ENLARGED (#4): keep it on the FOCUS view AND persist it onto
        the parked grid, and NEVER rebuild the grid (which would swap the focus canvas out and bounce back to
        the thumbnail).  The focus plotter (``self.plotter``) applies the sub-kind's knob in place; a knob it
        can't apply in place re-focuses the SAME cell -- rebuilding ONLY the enlarged view, staying zoomed."""
        parked = self._grid_focus
        if parked is None:
            return
        # Persist on the parked grid WITHOUT drawing (store_display_param): the stored params travel into
        # the save recipe and re-seed build_focus_plotter; the (invisible) thumbnails are NOT synchronously
        # repainted here -- _unfocus_grid_cell redraws them on return, so a Setting edit while zoomed costs
        # only the enlarged view's own in-place apply, never an N-cell repaint stall.
        try:
            if parked.grid.store_display_param(str(key), value):
                parked.dirty = True         # a thumbnail-affecting knob changed -> unfocus must repaint the grid
        except Exception:
            pass
        # Apply to the enlarged view in place.  A knob it can't apply live re-focuses the same cell ONLY
        # when the knob actually belongs to the per-site kind's own param set (bins/fit/ylog/cmap ...) --
        # an unrelated panel knob (e.g. ``repeat_mode``, which has no effect on a snapshot grid's enlarged
        # cell) is just stored, so it never triggers a visible rebuild/flash of the enlarged view.
        if self.plotter is not None and not self.plotter.apply_param(str(key), value):
            if any(d.key == str(key) for d in _panel_display_decls(self.config.kind, self._param_kind())):
                self._refocus_current_cell()   # rebuild ONLY the enlarged view, staying zoomed
        elif self.canvas is not None:
            self.canvas.draw_idle()

    def _remove_focus_stretch(self) -> None:
        """Drop the TOP spacer that centred the enlarged cell in the grid region (no-op when absent)."""
        stretch = self._focus_top_stretch
        self._focus_top_stretch = None
        if stretch is not None:
            self.canvas_holder.removeItem(stretch)

    def _refocus_current_cell(self) -> None:
        """Rebuild the ENLARGED cell view in place (same cell, current display params) -- used when the focus
        plotter cannot apply a knob live.  Swaps ONLY the focus canvas; the grid stays parked and
        ``_grid_focus`` stays set, so the view never bounces back to the thumbnail."""
        parked = self._grid_focus
        if parked is None or panel_canvas is None:
            return
        grid, k = parked.grid, parked.k
        old_focus, old_canvas = self.plotter, self.canvas
        sub_kind = str(getattr(grid, "sub_plot_kind", "hist"))
        focus = grid.build_focus_plotter(k, size="2x2", interactions=True,
                                         **self._view_kwargs(sub_kind))    # reads grid._display_params
        # (interactions=True + _apply_selectors_state below: the rebuilt enlarged view parks/arms its
        # selector layer to the header's "Selectors" switch, same as _focus_grid_cell)
        focus_canvas = panel_canvas(focus.fig, isolate_wheel=False)
        # replace the old focus canvas IN PLACE (the centring top stretch stays where it is)
        at = self.canvas_holder.indexOf(old_canvas) if old_canvas is not None else 0
        self.canvas_holder.insertWidget(max(0, at), focus_canvas, alignment=QtCore.Qt.AlignHCenter)
        if old_canvas is not None:
            self.canvas_holder.removeWidget(old_canvas)
            old_canvas.setParent(None)
            old_canvas.deleteLater()
        self.plotter = focus
        self.canvas = focus_canvas
        parked.key_cid = focus_canvas.mpl_connect("key_press_event", self._on_grid_focus_key)
        parked.click_cid = focus_canvas.mpl_connect("button_press_event", self._on_grid_canvas_click)
        self._apply_display_params()
        self._apply_selectors_state()
        focus_canvas.draw()
        if old_focus is not None and plt is not None and getattr(old_focus, "fig", None) is not None:
            plt.close(old_focus.fig)
        self._place_setting_button()

    def _unfocus_grid_cell(self) -> None:
        """Return from the enlarged cell to the grid: mirror any threshold drag back onto the grid cell, swap
        the parked grid canvas back in, and close the focus figure."""
        parked = self._grid_focus
        if parked is None:
            return
        grid, grid_canvas, k = parked.grid, parked.grid_canvas, parked.k
        focus, focus_canvas = self.plotter, self.canvas
        # copy an enlarged-view threshold cut back onto the grid cell (the grid thumbnail + save recipe read
        # it), noting whether it ACTUALLY changed -- an unfocus that edited nothing needs no repaint.
        cell = getattr(grid, "cell_renderer", None)
        threshold_changed = False
        if cell is not None and hasattr(cell, "sync_threshold_from_focus"):
            try:
                threshold_changed = bool(cell.sync_threshold_from_focus(k, focus))
            except Exception:
                pass
        # The parked grid kept its FULLY-RENDERED buffer (fit + x-window + thumbnails) untouched while zoomed,
        # so swapping it back shows it AS IT WAS.  Re-render ONLY when the zoom changed something the grid
        # shows -- a Setting edit persisted onto it (parked.dirty) or a threshold dragged in the focus view.
        # Otherwise just re-blit the valid buffer: the fit STAYS (fixing the "fit vanished after zoom-in-and-
        # back" bug the old unconditional redraw caused) AND unfocus is instant, never an N-cell re-render.
        needs_redraw = bool(parked.dirty) or threshold_changed
        self._grid_focus = None
        self._remove_focus_stretch()
        # swap the grid canvas back in FIRST (never a blank holder), then retire the focus canvas + figure
        if grid_canvas is not None:
            self.canvas_holder.insertWidget(0, grid_canvas, alignment=QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop)
        self.plotter = grid
        self.canvas = grid_canvas
        if focus_canvas is not None:
            self.canvas_holder.removeWidget(focus_canvas)
            focus_canvas.setParent(None)
            focus_canvas.deleteLater()
        if focus is not None and plt is not None and focus.fig is not None:
            plt.close(focus.fig)
        if hasattr(grid, "discard_focus_fit"):
            grid.discard_focus_fit()        # release the (now-closed) enlarged cell's fit tracker entry
        if grid_canvas is not None:
            if needs_redraw and hasattr(grid, "_redraw_thumbnails"):
                try:
                    grid._redraw_thumbnails()       # self-contained: re-applies the fit / x-window too
                except Exception:
                    pass
                grid_canvas.draw()
            else:
                grid_canvas.update()                # valid buffer -> re-blit only (fit preserved, instant)
        self._place_setting_button()

    def _reset_plot(self) -> None:
        """Drop the plot so the next refresh rebuilds it (size/params/source changed)."""
        self._teardown_plot()
        self._value_shape = None

    def _teardown_plot(self) -> None:
        # If a grid cell is ENLARGED, the parked grid holds a SECOND figure/canvas -- close it too so a
        # teardown while focused leaks neither figure (self.plotter/self.canvas are the FOCUS view here).
        # Record the focused index so the NEXT _build_plot re-focuses the SAME cell: a rebuild-class edit
        # (size / source / structure param) made while zoomed lands back on the enlarged cell, never
        # silently bouncing to the main grid (#no-focus-bounce).
        parked = self._grid_focus
        self._grid_focus = None
        if parked is not None:
            self._pending_refocus_k = int(parked.k)
            self._remove_focus_stretch()
            if parked.grid_canvas is not None:
                parked.grid_canvas.setParent(None)
                parked.grid_canvas.deleteLater()
            if parked.grid is not None and plt is not None and getattr(parked.grid, "fig", None) is not None:
                plt.close(parked.grid.fig)
        canvas, plotter = self.canvas, self.plotter
        self.canvas = None
        self.plotter = None
        if canvas is not None:
            self.canvas_holder.removeWidget(canvas)
            # setParent(None) BEFORE deleteLater: removeWidget only detaches from the LAYOUT -- the
            # widget stays parented (and painted!) until the deferred delete runs, so the old figure
            # lingered on screen under the replacement (the ghost/overlap the size-change showed).
            canvas.setParent(None)
            canvas.deleteLater()
        if plotter is not None and plt is not None and plotter.fig is not None:
            plt.close(plotter.fig)

    def shutdown(self) -> None:
        self._wait_render_idle()            # the worker must not be composing into this figure
        self._rebuild_timer.stop()          # never fire a coalesced rebuild into a torn-down card
        self._force_rebuild = False
        self._teardown_plot()

class _PanelBoard(QtWidgets.QWidget):
    """Absolute-positioned canvas the cards live on (drag + snap layout)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

    def arrange(self, cards: Sequence[PanelCard]) -> None:
        # ``col``/``row`` ARE the card's pixel top-left (gravity-packed by :func:`pack`); place verbatim
        # and reserve a GAP margin past the lowest-right card so the board scrolls cleanly.
        max_x = max_y = 0
        for card in cards:
            x, y = card.config.col, card.config.row
            card.move(x, y)
            max_x = max(max_x, x + card.width())
            max_y = max(max_y, y + card.height())
        self.setMinimumSize(max_x + GAP, max_y + GAP)
