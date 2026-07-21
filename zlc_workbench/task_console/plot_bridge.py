"""The console's panel card: its raster surface, its Setting, and the card geometry.

A card owns NO plotting object.  Its surface is a raster board painted from
immutable bytes, and every picture on it was composed on a worker from one
frozen snapshot (:mod:`zlc_frontend.panel_render`).  That is what keeps a
megapixel frame off the thread that also has to stay responsive, and it is why
the display knobs here are stored FACTS (``config.params``) rather than pushes
into a live figure: the knobs are read back on the next compose, so what is
stored and what is drawn cannot drift.

Every import names a TRUE owner -- nothing in this module touches the legacy
tree, so deleting ``Zou_lab_control`` cannot orphan it.
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

#: Sentinel for "this panel has never stored a repeat mode".  A stored ``None``
#: is a CHOICE the operator made; absence is not, and the kind's default only
#: applies to absence.
_MISSING_REPEAT_MODE = object()

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
#: ``_set_param`` (which stores the mode and reveals the fixed lo/hi row).
_RELIM_PARAM = ParamDecl(
    key="relim", label="relim", kind="choice", default="tight", choices=_RELIM_MODES, display=True,
    tooltip="Relim mode (confocal_gui combo_relim naming):\n"
            "  tight  = autoscale hugs the data\n"
            "  normal = autoscale, holding the window until the data leaves it\n"
            "  fixed  = pin the y-axis / colour-limit to the lo/hi below")

def panel_input_slots_is_single(kind: str) -> bool:
    """Whether a plot kind takes EXACTLY one signal, so no slot grows beside it.

    Read from the kind's own declaration (:data:`zlc_data.plot_kind.PLOT_KIND_SPEC_BY_KEY`)
    rather than listed here: a site map states its single-slot nature once, and the Setting
    popup asks.
    """

    from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY

    spec = PLOT_KIND_SPEC_BY_KEY.get(str(kind or ""))
    return bool(spec is not None and spec.single_slot)


# ====================================================================== panels
_MONITOR_UNSET = object()   # sentinel: a monitor panel that has never rolled yet

class PanelCard(FluentGroupBox):
    """One dashboard panel: a TITLED frame (title strip = the panel KIND + the signal-source
    legend, top-left) holding the frontend canvas, and a text
    "Setting" button on the title strip (top-right).  The frame border is the DRAG
    HANDLE (the board keeps all its own pointer interactions); the card
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
                 selection_clear_sink=None, fit_node_sink=None,
                 analysis_actions_provider=None):
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
        # callable() -> the analysis actions this host can actually carry out, as a subset of
        # ``zlc_data.vocabulary.ANALYSIS_ACTIONS``.  The panel asks rather than assumes: the
        # host owns the catalog and therefore knows whether the seam each action needs is
        # present, and an action absent here is one the Setting popup never offers.  None =
        # ask nothing and offer nothing beyond plain selection.
        self.analysis_actions_provider = analysis_actions_provider
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
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF = the historical display-only Monitor board).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # Last DATA selection made on this panel.  It is plot-independent and
        # serializable; selecting never implies an ROI or a fit.  The explicit
        # ``selection_action`` config decides what a later release does.
        self._active_selection = None
        # When the Setting popup last dismissed itself.  The button re-opens it, and a
        # click that DISMISSED the popup would otherwise arrive here as "open" a moment
        # later -- the popup would flicker shut and straight back open.  Zero means it
        # has never been dismissed, which is safely outside any debounce window.
        self._settings_dismissed_at = 0.0
        # {param key: declared kind} for each rendered row, so reopening the Setting
        # re-seeds a control through its OWN kind's writer instead of guessing from
        # the stored value's Python type.
        self._param_kinds: dict[str, str] = {}
        self._value_shape: tuple[int, ...] | None = None
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

        from zlc_frontend.qt_widgets import QtImageBoard

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
                self.panel_id, intent=self.view_intent(), label=str(value.name),
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

    def view_intent(self):
        """Which view this panel's kind asks its data for.

        Public because the Edit tab composes the same panel on its own surface:
        both must ask for the same view, or the snapshot would be a different
        picture of the same data.
        """

        from zlc_frontend.figure import ViewIntent

        return _panel_view_intents().get(self.config.kind, ViewIntent.IMAGE)

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
        # The panel's relim vocabulary IS the renderer's: tight / normal / fixed.
        # Converting rather than re-deciding keeps one set of names, so a mode
        # the renderer grows is a mode the Setting can offer with no mapping to
        # update in between.
        mode = RelimMode(str(self._relim()))
        fixed = None
        if mode is RelimMode.FIXED:
            fixed = (float(params.get("fixed_lo", 0.0)),
                     float(params.get("fixed_hi", 1.0)))
        intent = self.view_intent()
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
        """The Setting popup: the sections the operator tunes this panel through.

        Everything here is a VIEW of ``config.params`` / ``config.inputs``.  No control
        owns state of its own: each writes through the card's one writer and is re-seeded
        from the stored value on open, which is what stops this popup and the Edit tab --
        which renders the same declarations -- from drifting apart.
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

        label_w = scaled_px(96, minimum=72)

        def section_box(title):
            box = FluentGroupBox(title, content)
            layout = QtWidgets.QVBoxLayout(box)
            layout.setContentsMargins(popup_gap(), popup_gap(), popup_gap(), popup_gap())
            layout.setSpacing(scaled_px(4, minimum=2))
            self._settings_col.addWidget(box)
            return layout

        # ---- Source: which signal(s) this panel reads, and the expression over them.
        # The picker IS the "plot this" control; the expression is the advanced override,
        # which is why picking a slot rewrites the canonical expression for a single-slot
        # panel and leaves a hand-authored multi-slot one alone.
        source = section_box("Source")
        self.slot_combos = []
        slots = list(self.config.inputs) or [""]
        for index in range(len(slots)):
            combo = FluentTreeComboBox()
            combo.setToolTip("The signal this slot reads, grouped by the node that produces it.")
            combo.currentIndexChanged.connect(lambda _i, idx=index: self._on_slot_pick(idx))
            self.slot_combos.append(combo)
            name = "signal" if len(slots) == 1 else "signal[%d]" % index
            source.addWidget(FluentSettingRow(name, combo, label_width=label_w))
        if not panel_input_slots_is_single(self.config.kind):
            grow = QtWidgets.QWidget()
            grow_row = QtWidgets.QHBoxLayout(grow)
            grow_row.setContentsMargins(0, 0, 0, 0)
            grow_row.setSpacing(scaled_px(6, minimum=4))
            add_slot = FluentButton("+", color=GREY)
            add_slot.setToolTip("Add a signal slot so the expression can combine more signals")
            add_slot.clicked.connect(self._add_signal_slot)
            drop_slot = FluentButton("-", color=GREY)
            drop_slot.setToolTip("Remove the last signal slot")
            drop_slot.clicked.connect(self._remove_signal_slot)
            grow_row.addWidget(add_slot, 0)
            grow_row.addWidget(drop_slot, 0)
            grow_row.addStretch(1)
            source.addWidget(FluentSettingRow("slots", grow, label_width=label_w))
        self.source_edit = FluentLineEdit(self.config.source)
        self.source_edit.setPlaceholderText(_BLANK_SOURCE)
        self.source_edit.setToolTip("value = <expression over the slots above>.  Click to edit in full.")
        self.source_edit.textChanged.connect(lambda *_: self.apply_button.set_dirty(True))
        self.source_edit.returnPressed.connect(self._apply_source)
        self.source_edit.mouseDoubleClickEvent = lambda _e: self._open_expr_editor()
        source.addWidget(FluentSettingRow("value", self.source_edit, label_width=label_w))
        self.apply_button = FluentButton("Apply", color=ACCENT)
        self.apply_button.setToolTip("Apply the expression (a slot pick applies on its own)")
        self.apply_button.clicked.connect(self._apply_source)
        source.addWidget(self.apply_button)

        # ---- Display: the declared view knobs for this kind, emitted through the SHARED
        # row builder, so a kind that gains a knob shows it in both surfaces with no
        # wiring here.
        display = section_box("Display")
        display_specs = ([spec for spec in _panel_display_decls(self.config.kind, self._param_kind())
                          if spec.display] + [_RELIM_PARAM] + list(self._repeat_param_specs()))
        self.param_widgets = self._emit_param_rows(
            display_specs, display.addWidget, self._set_param, label_w)
        self.fixed_lim_row, self.fixed_lo_edit, self.fixed_hi_edit = self._make_fixed_lim_row(
            self._on_fixed_lim_edited, label_w)
        display.addWidget(self.fixed_lim_row)
        self.fixed_lim_row.setVisible(self._relim() == "fixed")
        unit_row, self.unit_button, self.unit_label = self._make_unit_cycle_row(
            self._on_unit_cycle, label_w, with_label=True)
        display.addWidget(unit_row)

        # ---- Analysis: what a drag on this panel means (the shared composite).
        self._build_analysis_section(section_box, label_w)

        # ---- Panel: the card itself -- its name, its footprint, and how often it redraws.
        panel = section_box("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setToolTip("Rename this panel (also the default save name)")
        self.title_edit.textChanged.connect(self._on_title)
        panel.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))
        self.size_combo = FluentComboBox()
        for preset in PANEL_SIZES:
            self.size_combo.addItem(preset, preset)
        index = self.size_combo.findData(self.config.size)
        if index >= 0:
            self.size_combo.setCurrentIndex(index)
        self.size_combo.setToolTip("Card footprint in layout half-units (rows x columns)")
        self.size_combo.currentIndexChanged.connect(
            lambda _i: self._on_size(str(self.size_combo.currentData() or self.config.size)))
        panel.addWidget(FluentSettingRow("size", self.size_combo, label_width=label_w))
        self.update_combo = FluentComboBox()
        for interval in UPDATE_INTERVALS:
            self.update_combo.addItem("%d ms" % interval, interval)
        index = self.update_combo.findData(
            int(self.config.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS))
        if index >= 0:
            self.update_combo.setCurrentIndex(index)
        self.update_combo.setToolTip(
            "How often THIS panel redraws.  Acquisition is unaffected -- this is a display rate.")
        self.update_combo.currentIndexChanged.connect(self._on_update_interval)
        panel.addWidget(FluentSettingRow("update", self.update_combo, label_width=label_w))

        # The status line lives at the bottom of the popup, where a message about the panel
        # belongs; the card's own title strip stays clean.
        self.status = FluentLabel(self._status_text)
        self.status.setStyleSheet("color: %s; background: transparent; border: none;" % GREY)
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self._settings_col.addWidget(self.status)
        self._settings_col.addStretch(1)

        self._refresh_signal_combo()

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

    def _current_unit_text(self) -> str:
        """The bound signal's declared unit, or a neutral placeholder."""

        value = self._last_value
        unit = "" if value is None else (value.unit or "")
        return unit or "unit"

    def _on_unit_cycle(self) -> None:
        """Refresh the unit readout.

        The unit is the PRODUCER's declared fact (``cell_schema.value_unit`` and
        each axis's own unit), carried on the data and printed by the renderer.
        A panel cannot cycle it: rewriting the unit here would make the picture
        disagree with the schema every other reader trusts.  The row stays as
        the readout of what the bound signal actually declares.
        """

        self.unit_label.setText(self._current_unit_text()) if hasattr(self, "unit_label") else None

    def apply_fixed_lims(self, lo: float, hi: float) -> None:
        """Persist the fixed lo/hi and re-compose with them NOW.

        The Setting popup's inputs and the Edit tab's both route here, so the
        pinned window has ONE writer; the rasteriser reads it back out of the
        display state on the next compose.
        """

        self.config.params["fixed_lo"], self.config.params["fixed_hi"] = float(lo), float(hi)
        self._rerender_now()
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

    def _param_kind(self) -> str:
        """The plot kind whose :data:`PANEL_PARAMS` drive THIS panel's Setting / Edit param UI.

        For every ordinary kind that is ``config.kind``.  A GRID panel's knobs are
        its per-cell kind (``sub_plot_kind``): a histogram grid must offer bins /
        fit, an image grid a colormap.  That choice is a stored panel param, so
        the Setting rows follow it without asking a figure what it turned out to
        be.
        """

        if self.config.kind != "grid":
            return self.config.kind
        sub = self.config.params.get("sub_plot_kind")
        if sub:
            return str(sub)
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
        for decl in _panel_display_decls(self.config.kind, self._param_kind()):   # sub-kind knobs + grid title
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
        self._rerender_now()
        self._sync_settings_param_rows()
        self.changed.emit()

    def _apply_display_params(self) -> None:
        """Re-assert every stored display knob by re-composing this panel.

        There is no second push path: the knobs ARE ``config.params``, the
        composer reads them through :meth:`_display_state`, and re-composing is
        how they take effect.  A panel whose producer has stopped still updates,
        because the last frozen value is what it re-composes from.
        """

        self._rerender_now()

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

    def _on_title(self, text: str) -> None:
        """Rename the panel.  The title is the card's, and the figure's label.

        One string: the frame header shows it and the composer carries it as the
        dataset label, so the picture and the card can never be captioned
        differently.
        """

        self.config.title = str(text)
        self._compose_key = None          # the label is part of the composer's identity
        self._rerender_now()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        self._wait_render_idle()   # the worker must not be composing into the surface being replaced
        self.config.size = str(size)
        # ATOMIC relayout transaction.  A size change is three synchronous steps: drop the surface,
        # grow the card to the new preset, re-compose.  If any event-loop slice runs between the grow
        # and the re-compose -- a closing size-combo popup, a deferred paint -- the operator sees one
        # frame of a BIG EMPTY card before the picture lands, which is what the "resize jump" was.
        # Freezing this card's repaints for the whole mutation makes the paint system composite ONE
        # clean final frame.  The layout_changed emit -- and the sibling repack it drives -- runs
        # INSIDE the freeze, so position, size and picture all land together; the finally re-enables
        # even if the re-compose raises.
        self.setUpdatesEnabled(False)
        try:
            self._reset_plot()
            self._apply_fixed_size()
            # If the Setting frame is open, GROW it immediately to match the new (taller) panel -- but
            # the high-water mark means a SMALLER size never snaps it shorter (#H3i-2).
            if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                self._size_settings_popup()
            self._rerender_now()   # re-draw at the new size NOW, even if the source is stopped (else the
                                    # torn-down panel stays blank until the next hub tick -- same as _set_param)
            self.changed.emit()
            self.layout_changed.emit()
        finally:
            self.setUpdatesEnabled(True)

    def current_selection(self):
        """This panel's stored data selection -- plot-independent and serializable.

        A selection is DATA, not a figure state: it survives a re-compose, a
        source re-pick and a saved layout, and it is what an ROI or a fit is
        later asked to act on.  Selecting never implies either.
        """

        from zlc_data.plot_region import Selection

        selection = self._active_selection
        if selection is None:
            payload = self.config.params.get("selection")
            if isinstance(payload, Mapping):
                selection = Selection.from_dict(payload)
        if selection is None:
            selection = Selection()
        metadata = dict(selection.metadata)
        roi = getattr(self, "_roi_built", None)
        if roi and len(roi) >= 4:
            metadata["origin"] = [float(roi[0]), float(roi[2])]
        return Selection(selection.ranges, frame=selection.frame,
                         scope=selection.scope, metadata=metadata)

    def _selection_coordinates_for_binding(self) -> dict:
        """Per-axis coordinates an ROI binding needs, from the composed front.

        The front's typed payload carries the viewport that maps a pointer
        rectangle back onto the declared axes; reading them from the SAME front
        the operator dragged on is what keeps a box drawn on screen and the
        block it crops in the same coordinate system.  Empty until this card's
        surface exposes a selector (a 2-D / histogram ROI needs no extra
        coordinate: pixel index and sample value carry themselves).
        """

        return {}

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
        """The model a fit starts from when no surface has chosen one yet.

        Taken from the catalog rather than a literal here: the definitions know
        which models exist and what each one renders as, so a panel that gains a
        new admissible model does not need a second list updating to offer it.
        """

        from zlc_data import fit_model_catalog

        catalog = fit_model_catalog()
        if not catalog:
            raise RuntimeError("no fit model is defined")
        return str(catalog[0].model_id)

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
        """The ONE mutator for this panel's fit: store it, then tell the console.

        A fit is a PROCESSOR's job -- it consumes the same signal and publishes
        its parameters -- so this stores what was asked for and hands it to the
        console's fit-node sink.  Clearing removes the node the fit created,
        rather than leaving it republishing after the operator turned it off.
        """

        payload = request.to_dict() if request is not None else None
        self._set_param("fit_request", payload)
        self._refresh_analysis_controls()
        if request is None:
            if callable(self.selection_clear_sink):
                self.selection_clear_sink(self, "fit")
        elif callable(self.fit_node_sink):
            self.fit_node_sink(self, request)

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
        """The one-line fit-result string, shared VERBATIM by Setting and Edit.

        ``result`` comes from the fit processor's published output; there is no
        figure to interrogate for one, because the panel displays a fit rather
        than owning it.
        """

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
        """The ONE writer for a display knob: store it, then re-compose.

        Every surface that edits this panel (the Setting popup, the Edit tab, a
        drag that arms an analysis) comes through here, so a knob has one home
        -- ``config.params``, which is also what the saved layout persists.  The
        composer reads them back through :meth:`_display_state`, so what is
        stored and what is drawn cannot drift.
        """

        if self.config.params.get(key) == value:
            return
        self.config.params[key] = value
        if key == "relim":
            # Flipping INTO ``fixed`` FREEZES the window currently on screen.
            # Seeding from the composed front's own limits (never the default
            # 0..1 pair) is what keeps the picture: pinning a counts histogram
            # or a camera frame to 0..1 empties it, which reads as "the panel
            # just died".  The operator then types exact bounds.
            if str(value) == "fixed":
                shown = self._shown_limits()
                if shown is not None:
                    lo, hi = shown
                    self.config.params["fixed_lo"], self.config.params["fixed_hi"] = lo, hi
                    for edit, val in ((getattr(self, "fixed_lo_edit", None), lo),
                                      (getattr(self, "fixed_hi_edit", None), hi)):
                        if edit is not None:
                            edit.setText(f"{val:g}")   # setText does NOT re-fire editingFinished
            if getattr(self, "fixed_lim_row", None) is not None:        # the Setting popup's lo/hi row
                self.fixed_lim_row.setVisible(str(value) == "fixed")    # (the Edit tab toggles its own)
                # Revealing/hiding the row changes the popup's content height, and the popup is a
                # free-floating Qt.Popup whose window size was fixed at open time -- re-run the ONE
                # sizing rule so the window grows DOWNWARD to fit instead of reflowing inside a
                # fixed frame (which reads as a jump).
                if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                    self._size_settings_popup()
        self._rerender_now()
        self.changed.emit()

    def _shown_limits(self):
        """The value window the last composed front actually used, or None.

        Read off the front rather than recomputed: the whole point of pinning is
        to freeze WHAT IS ON SCREEN, and a freshly derived envelope is a
        different number the moment the data moved.
        """

        payload = getattr(self._pending_frame, "panels", None)
        front = getattr(self.board, "front_frame", None)
        panels = payload or getattr(front, "panels", ())
        for panel in panels or ():
            display = panel.display_payload
            for attr in ("color_limits",):
                limits = getattr(display, attr, None)
                if limits:
                    return (float(limits[0]), float(limits[1]))
            viewport = getattr(display, "viewport", None)
            for attr in ("y_limits", "count_limits"):
                limits = getattr(viewport, attr, None)
                if limits:
                    return (float(limits[0]), float(limits[1]))
        return None

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
        self._rerender_now()   # also re-evaluate on the last GOOD data at once (stopped node)
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

    def _rerender_now(self) -> None:
        """Re-compose + present from the LAST frozen value, now.

        Every display edit lands here.  The display revision advances first so
        the rasteriser can tell a genuinely new display from a repeat of the
        same one -- that distinction is what lets ``normal`` relim hold the
        window it already resolved instead of re-resolving it every edit.

        A stopped producer publishes no new tick, so re-composing the value this
        panel last drew is the only way a display edit can show up at all.
        """

        self._display_revision += 1
        if self._last_value is None:
            return
        if self.compose_signal_value(self._last_value):
            self.present()

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
        """Carry the board header's Selectors switch onto this card's surface.

        The switch is the card's state, not the surface's: it is stored here so
        a surface built (or rebuilt) later comes up matching the switch instead
        of coming up live behind the operator's back.
        """

        board = self.board
        if board is not None and hasattr(board, "set_selectors_enabled"):
            board.set_selectors_enabled(bool(self._selectors_on))

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


    def _reset_plot(self) -> None:
        """Drop the plot so the next refresh rebuilds it (size/params/source changed)."""
        self._teardown_plot()
        self._value_shape = None

    def _teardown_plot(self) -> None:
        """Drop this card's surface, leaving nothing painted behind it.

        ``setParent(None)`` before the deferred delete: removing a widget from
        the layout only detaches it from the LAYOUT -- it stays parented, and
        painted, until the delete actually runs, which is how a replaced surface
        used to linger under its successor.
        """

        board = self.board
        self.board = None
        self._pending_frame = None
        self._composer_obj = None
        self._compose_key = None
        if board is not None:
            self.canvas_holder.removeWidget(board)
            board.setParent(None)
            board.deleteLater()

    def shutdown(self) -> None:
        """Release this card's surface.  The worker must not be composing into it."""

        self._wait_render_idle()
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
