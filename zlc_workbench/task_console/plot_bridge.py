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
from zlc_frontend.render_style import panel_display_size
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
PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
coerce_short_labels = _qt_widgets.param_widgets.coerce_short_labels
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo

# Mirrors the legacy shell's guard: a matplotlib install without the Qt backend still lets the
# headless documents import; only actually BUILDING a panel canvas needs the backend.


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


# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH still scales with the size (``cols // 2`` base-widths so 1x4 is wider than 1x2);
# HEIGHT HUGS the plot -- the card is exactly tall enough for its figure + chrome, with NO blank
# padding below (every size hugs like 1x2, #H3i-3).  ``PanelConfig.col`` is the card's pixel X and
# ``row`` is the card's pixel Y; :func:`pack` is the order-driven TOP-LEFT GRAVITY packer that places
# every card at the first free NW slot in list order.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
# content padding) belongs to the FluentGroupBox COMPONENT (qt_widgets.CARD_PAD / CARD_TITLE_PX,
# the single source); this module only lays cards out.
#: Which view a panel kind asks its data for.  A kind is the operator's word
#: for what they want to see; a ViewIntent is what the figure layer understands.
def _panel_view_intents():
    from zlc_frontend.figure import ViewIntent

    return {
        "2d": ViewIntent.IMAGE,
        "sites": ViewIntent.IMAGE,
        "1d": ViewIntent.CURVE,
        "monitor": ViewIntent.CURVE,
        "hist": ViewIntent.HISTOGRAM,
        "grid": ViewIntent.IMAGE,
    }

GRID_UNIT = 8

# The ONE spacing setting (#H3s-F8).  GAP is the UNIFORM clear distance between any two cards on
# every side -- top, bottom, left, right -- AND the board margin from the (0, 0) origin.  It equals
# the HORIZONTAL inter-card gap the user likes: two cards on adjacent base-columns pitched by
# ``_cell_size()[0] + GAP`` sit exactly GAP px apart (and a multi-column card's internal columns are
# joined by the SAME GAP, see ``_card_size``).  Reusing this one existing spacing constant (no new
# public art/geom knob); change this one number to retune all board
# spacing.
GAP = GRID_UNIT


def _cell_size() -> tuple[int, int]:
    """The board's base CELL in pixels: the footprint of the narrowest card ("1x2").

    The packer works in cells; this is the one place a cell is converted to pixels.
    Width and height both come from the panel's displayed raster box
    (:func:`~zlc_frontend.render_style.panel_display_size`) plus the card chrome the
    FluentGroupBox component owns, so a card is exactly as tall as its content.
    """

    width = panel_display_size("1x2")[0] + 2 * CARD_PAD
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size("1x2")[1] + CARD_PAD
    return (width, height)


def _card_size(size: str) -> tuple[int, int]:
    """Pixel footprint of a card at a panel-size preset.

    Width snaps to a whole number of base cells (so cards align to the board grid and
    the inter-column joins are the same GAP as between cards); height hugs the panel's
    own displayed height, so no size leaves blank padding under the plot.
    """

    rows, cols = panel_size_cells(size)
    w_units = max(1, cols // 2)
    cw, _ch = _cell_size()
    width = w_units * cw + (w_units - 1) * GAP
    height = scaled_px(CARD_TITLE_PX) + scaled_px(2) + panel_display_size(size)[1] + CARD_PAD
    return (width, height)


def _board_metrics() -> "_layout.BoardMetrics":
    """The two facts the moved packer cannot derive, read LIVE on every call.

    ``_card_size`` stays here because it is the bridge between the FIGURE size (render
    layer) and the card CHROME (Qt tokens) -- neither of which the packer may import.
    Built fresh rather than cached: card pixels follow the current Qt scale, and a value
    captured once would go stale exactly the way a snapshot shim does.
    """

    return _layout.BoardMetrics(gap=GAP, card_size=_card_size)



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
        # The card's display surface: an immutable-bytes raster board (contract 4).
        # The panel's stable identity: the board, its composer and every frame
        # they exchange are keyed on it, so a presented frame can only ever land
        # on the panel it was composed for.
        self.panel_id = f"panel-{id(self):x}"
        self.board = None
        self._pending_frame = None    # composed front awaiting its present pass
        self._composer_obj = None
        self._compose_key = None
        self._last_value = None       # last value drawn, for an immediate re-render
        # Bumped by every display-knob edit.  The renderer reads it to tell a
        # genuinely new display from a repeat of the same one.
        self._display_revision = 0
        self.plotter = None      # no Matplotlib object lives on a card any more
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

    def _build_plot(self) -> None:
        """Give this card its raster surface.

        A panel shows PIXELS the worker produced -- an encoded raster handed to
        :class:`~zlc_frontend.qt_widgets.board.QtImageBoard`, which paints from
        immutable bytes and owns no figure.  The card therefore holds no
        Matplotlib object at all: rendering happens off the GUI thread and
        arrives here already rasterised, which is what keeps a 2.3 MP frame from
        being drawn on the thread that also has to stay responsive.
        """

        from zlc_frontend.qt_widgets.board import QtImageBoard

        if self.board is not None:
            return
        self.board = QtImageBoard(self.panel_id, empty_text="waiting for data")
        self.canvas_holder.addWidget(self.board)

    def _composer(self, value):
        """This panel's composer for the currently bound signal.

        Rebuilt when the panel's kind or its source changes, because a composer
        carries the colour/count window already on screen: a new signal must not
        inherit the previous one's limits, which would show the new data through
        the old one's contrast.
        """

        from zlc_frontend.figure import ViewIntent
        from zlc_frontend.panel_render import PanelComposer

        key = (self.config.kind, str(value.name), str(value.source))
        if self._compose_key != key:
            self._composer_obj = PanelComposer(
                self.panel_id,
                intent=_panel_view_intents().get(self.config.kind, ViewIntent.IMAGE),
                label=str(value.name),
            )
            self._compose_key = key
        return self._composer_obj

    def compose_signal_value(self, value) -> bool:
        """Rasterise one frozen signal value.  Worker-safe; returns whether it drew.

        Phase 1 of the board's two-phase render: this produces immutable bytes
        and touches no Qt, so the console runs it off the GUI thread.  A value
        this panel's kind cannot show says so on the status line rather than
        leaving the last frame up pretending to be current.
        """

        from zlc_frontend.panel_render import PanelProvenance, PanelRenderError

        if value is None or getattr(value, "snapshot", None) is None:
            return False
        try:
            frame = self._composer(value).compose(
                value.snapshot,
                display=self._display_state(),
                provenance=PanelProvenance(
                    value.run_id, value.epoch_id, value.join_digest,
                ),
            )
        except PanelRenderError as error:
            self.set_status(str(error)[:160], error=True)
            return False
        except Exception as error:                 # one bad panel never kills the batch
            self.set_status(f"{type(error).__name__}: {error}"[:160], error=True)
            return False
        self._pending_frame = frame
        self._last_value = value
        self.set_status("ok", error=False)
        return True

    def _display_state(self):
        """The display knobs this panel's kind exposes, as the renderer's own state.

        The stored panel params are the persisted layout; the display state is
        what the rasteriser reads.  Deriving one from the other on every compose
        keeps a saved board and a live board showing the same thing, with no
        second copy of "what the operator chose" to fall out of step.
        """

        from zlc_frontend.curve_display import CurveDisplayState
        from zlc_frontend.histogram_display import HistogramDisplayState
        from zlc_frontend.image_display import ImageColormap, ImageDisplayState
        from zlc_frontend.display_range import RelimMode
        from zlc_frontend.figure import ViewIntent

        params = self.config.params
        mode = RelimMode.NORMAL if str(self._relim()) == "normal" else RelimMode.TIGHT
        fixed = None
        if str(params.get("relim", "")) == "fixed":
            fixed = (float(params.get("fixed_lo", 0.0)),
                     float(params.get("fixed_hi", 1.0)))
        intent = _panel_view_intents().get(self.config.kind, ViewIntent.IMAGE)
        if intent is ViewIntent.CURVE:
            return CurveDisplayState(
                revision=self._display_revision, relim_mode=mode, fixed_y_limits=fixed,
            )
        if intent is ViewIntent.HISTOGRAM:
            return HistogramDisplayState(
                revision=self._display_revision, relim_mode=mode,
                bin_count=int(params.get("bins", 60) or 60),
                fixed_count_limits=fixed,
            )
        return ImageDisplayState(
            revision=self._display_revision,
            relim_mode=mode,
            colormap=ImageColormap(str(params.get("colormap", "gray") or "gray")),
            fixed_color_limits=fixed,
        )

    def present(self) -> None:
        """Flush this card's composed front to the screen.  GUI thread only.

        Phase 2 of the board's two-phase render: the board composes every panel
        of a tick, then presents them together, so the screen never shows a torn
        mix of instants.
        """

        frame = self._pending_frame
        if frame is None:
            return
        self._pending_frame = None
        self._build_plot()
        self.board.present(frame)

    def _build_settings(self) -> None:
        """The Setting popup's shell: the card that holds the sections.

        The sections themselves are rebuilt from the panel's declared params as
        the display state lands (contract 4); this builds the popup, its scroll
        viewport and the geometry bookkeeping the open path reads, so a card is
        constructible and its gear opens.
        """

        popup = FluentPopup(self)
        outer = QtWidgets.QVBoxLayout(popup)
        outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._settings_scroll.viewport().setAutoFillBackground(False)
        content = QtWidgets.QWidget()
        content.setAutoFillBackground(False)
        self._settings_col = QtWidgets.QVBoxLayout(content)
        self._settings_col.setContentsMargins(popup_gap(), popup_gap(), popup_gap(), popup_gap())
        self._settings_col.setSpacing(popup_gap())
        self._settings_scroll.setWidget(content)
        outer.addWidget(self._settings_scroll)
        self.settings_popup = popup
        self._settings_h_hwm = 0

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

    def _facet_value_shapes(self) -> tuple[tuple, tuple]:
        """(points multi-D shape, data shape) from the producing node's declared structure (#H3o) --
        the points axis is stored FLAT, its multi-D form lives in ``grid_shape`` when the scan
        declared one.  The ONE source the facet choices, the auto sub-kind and the slicer share."""
        st = self._bound_structure() or {}
        gs = tuple(int(n) for n in (st.get("grid_shape") or ()))
        ps = tuple(int(n) for n in (st.get("points_shape") or ()))
        ds = tuple(int(n) for n in (st.get("data_shape") or ()))
        return (gs or ps), ds





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

    def compose(self, snapshot, *, offthread: bool = False) -> bool:
        """Rasterise this panel from ONE frozen tick.  Phase 1 of the board's render.

        Every panel of a tick composes from the SAME freeze, so the board cannot
        show a 2-D image of one shot beside a histogram of another.  A panel with
        no bound signal, or one whose signal is not in this freeze yet (a producer
        still waiting), sits quietly with a hint -- it is decoupled, and a missing
        source is a state, not an error.

        ``offthread=True`` is the render-worker entry.  Nothing here touches Qt:
        the product is immutable bytes, which is exactly why the worker may run
        it while the GUI thread stays responsive.
        """

        name = next((item for item in (self.config.inputs or ()) if item), "")
        if not name:
            self.set_status("pick a signal in Setting", error=False)
            return True
        value = None if snapshot is None else snapshot.value(name)
        if value is None:
            self.set_status(f"waiting for {name}", error=False)
            return True
        self.compose_signal_value(value)
        return True

    def refresh(self, snapshot) -> None:
        """Compose + present this ONE card now -- a single-card refresh.

        The board (TaskConsole._tick) instead splits those two phases across ALL
        its panels for cross-panel coherence; a lone card just does both.
        """

        self.compose(snapshot)
        self.present()

    def _signal_expr(self):
        """This panel's source as the ONE reusable :class:`SignalExpr` (the slot rule + the
        ``value = ...`` contract live there, shared with processors / pulse-scan).  Lazy import
        keeps the frontend module off neutral_atom's import graph (every neutral_atom use here
        is lazy)."""
        from zlc_data.signal_expr import SignalExpr
        return SignalExpr(self.config.inputs, self._compiled_source)


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
