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

import math
from typing import Mapping
from PyQt5 import QtCore, QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    CARD_PAD,
    FluentButton,
    FluentComboBox,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentPopup,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingsPopupAnchor,
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
from zlc_frontend.form import lenient_float as _safe_float
from zlc_frontend.render_style import panel_display_size
from zlc_frontend.panel_params import (
    panel_param_decls as _panel_param_decls,
    resolved_panel_param as _resolved_panel_param,
)
from zlc_data.console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
)
from zlc_data.panel_size import PANEL_SIZES
from zlc_data.param_decl import ParamDecl
from zlc_data.plot_kind import PLOT_KIND_SPEC_BY_KEY

from .panel_board import card_size as _card_size
from .panel_types import (
    HISTOGRAM_CELL_THRESHOLDS_PARAM as _HISTOGRAM_CELL_THRESHOLDS_PARAM,
    HISTOGRAM_THRESHOLDS_PARAM as _HISTOGRAM_THRESHOLDS_PARAM,
    RELIM_PARAM as _RELIM_PARAM,
    VIEW_SPEC_PARAM as _VIEW_SPEC_PARAM,
    grid_view_intents as _grid_view_intents,
    panel_view_intents as _panel_view_intents,
    repeat_mode_label as _repeat_mode_label,
)
from .render_lane import PanelRenderRequest as _PanelRenderRequest

# qt_widgets submodules are reached as ATTRIBUTES of the one facade binding: their names are
# deliberately absent from the facade __all__, and the package forbids outside deep imports.
PARAM_WIDGETS = _qt_widgets.param_widgets.PARAM_WIDGETS
ParamWidgetContext = _qt_widgets.param_widgets.ParamWidgetContext
coerce_short_labels = _qt_widgets.param_widgets.coerce_short_labels
fill_grouped_signal_combo = _qt_widgets.param_widgets.fill_grouped_signal_combo



# A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so
# it shows nothing until the user picks a hub signal in its Setting (signal_combo)
# -- it must NOT auto-bind to any node's signal.  An empty source is the blank
# state; ``refresh`` treats it (and a source that produces None) as "pick a signal"
# rather than an error, so a blank panel sits quietly until wired.


# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH and HEIGHT wrap the exact FigureSpec logical panel size plus Fluent chrome;
# the card is exactly large enough for its figure, with NO stretch or blank
# padding below (every size hugs like 1x2, #H3i-3).  ``PanelConfig.col`` is the card's pixel X and
# ``row`` is the card's pixel Y; :func:`pack` is the order-driven TOP-LEFT GRAVITY packer that places
# every card at the first free NW slot in list order.  The CARD'S FORMAT (rounded corners, shadow, grey title strip,
# content padding) belongs to the FluentGroupBox COMPONENT (qt_widgets.CARD_PAD / CARD_TITLE_PX,
# the single source); this module only lays cards out.
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
    fit_analysis_available_changed = QtCore.pyqtSignal(bool)
    front_presented = QtCore.pyqtSignal()
    rectangle_selected = QtCore.pyqtSignal(object)

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None,
                 sources_provider=None, formats_provider=None,
                 short_names_provider=None, axis_labels_provider=None,
                 render_request=None,
                 fit_analysis_sink=None):
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
        # callable -> {full hub signal: SHORT name} (the producing node's prefix stripped), so the
        # picker NEST shows the short name (frame / survival / rate) -- never the full prefixed key
        # nor the verbose SignalSpec axis label.  ONE rule, shared with the Logic tab.
        self.short_names_provider = short_names_provider
        # callable -> {exact signal key: catalog-authored plot-axis label}.
        # Routing keys identify data; they are never visible plot chrome.
        self.axis_labels_provider = axis_labels_provider
        # callable(card, force=False) -> enqueue one latest-only worker compose.
        # The callback receives no mutable render state: ``render_request`` asks
        # this card to freeze a request first, then the worker owns every
        # PanelComposer/Agg object and Qt only presents its immutable result.
        if not callable(render_request):
            raise TypeError("render_request must be callable")
        self._render_request = render_request
        if fit_analysis_sink is not None and not callable(fit_analysis_sink):
            raise TypeError("fit_analysis_sink must be callable or None")
        self.fit_analysis_sink = fit_analysis_sink
        self._fit_analysis_available = False
        # The card's display surface: an immutable-bytes raster board (contract 4).
        # The panel's stable identity: the board, its composer and every frame
        # they exchange are keyed on it, so a presented frame can only ever land
        # on the panel it was composed for.
        self.panel_id = f"panel-{id(self):x}"
        self.board = None
        self._pending_frame = None    # composed front awaiting its present pass
        self._pending_faceted_result = None
        self._last_value = None       # last value drawn, for an immediate re-render
        # A grid has no renderable front until the operator chooses a named
        # facet.  Keep the latest already-immutable data-plane value so that
        # making that choice can render immediately; this is not a second
        # snapshot, mutable cache, or accepted-front claim.
        self._candidate_value = None
        self._last_document = None
        self._last_figure = None
        self._last_display = None
        self._candidate_schema = None
        self._grid_focus = None
        self._render_request_revision = 0
        self._requested_signature = None
        # Qt paints the card in logical pixels, while the worker raster is
        # authored at the physical-pixel ratio of the screen containing the
        # TaskConsole.  The console owns screen observation and updates this
        # value; a DPR change is therefore a render-key change even when the
        # displayed dataset revision did not advance.
        self._raster_pixel_ratio = 1.0
        self._pending_interaction_origin = None
        # Bumped by every display-knob edit.  The renderer reads it to tell a
        # genuinely new display from a repeat of the same one.
        self._display_revision = 0
        # The board gestures' authored viewport pin (x_view, y_view) in DATA
        # coordinates, or None for home.  Display-only continuity like the
        # colour window -- deliberately NOT in config.params: a saved layout
        # reopens at home, exactly like the pulse preview.
        self._view_pin = None
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF = the historical display-only Monitor board).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # {param key: declared kind} for each rendered row, so reopening the Setting
        # re-seeds a control through its OWN kind's writer instead of guessing from
        # the stored value's Python type.
        self._param_kinds: dict[str, str] = {}
        # The hub version at this panel's LAST render -- the per-panel multi-rate refresh
        # (see TaskConsole._tick) skips a panel on its beat when nothing new was published
        # since, so a slow panel does not redraw stale data and a fast one only when needed.
        self._render_version = -1
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
        self._settings_anchor = FluentSettingsPopupAnchor(
            self.settings_popup,
            self.setting_button,
        )
        self.setting_button.clicked.connect(self._open_settings)

        self.fit_analysis_button = None
        if self.fit_analysis_sink is not None:
            self.fit_analysis_button = self.make_fit_analysis_button(
                parent=self,
                text="Analyze",
            )
            self.fit_analysis_button.setFixedSize(
                scaled_px(78, minimum=66), scaled_px(26, minimum=22)
            )

        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(self) -> None:
        """Card size = whole layout slots (the footer absorbs the slack)."""
        self.setFixedSize(*_card_size(self.config.size))
        self._place_setting_button()

    def set_raster_pixel_ratio(self, ratio: float) -> bool:
        """Set the Qt-owner screen ratio used by the next worker request."""

        ratio = float(ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("raster pixel ratio must be positive and finite")
        if ratio == self._raster_pixel_ratio:
            return False
        self._raster_pixel_ratio = ratio
        return True

    def _place_setting_button(self) -> None:
        if hasattr(self, "setting_button"):
            # top-right, on the title strip (the title kind sits top-left).
            self.setting_button.move(
                self.width() - self.setting_button.width() - scaled_px(8),
                scaled_px(4))
            self.setting_button.raise_()
        fit_button = getattr(self, "fit_analysis_button", None)
        if fit_button is not None:
            fit_button.move(
                self.setting_button.x()
                - fit_button.width()
                - scaled_px(5, minimum=3),
                scaled_px(4),
            )
            fit_button.raise_()

    def make_fit_analysis_button(
        self,
        *,
        parent=None,
        text: str = "Analyze → Fit",
    ) -> FluentButton:
        """Build another view of this card's one FINAL-fit command.

        Title bar, Setting and Edit use this same builder.  The button owns no
        source reference, FitSpec, solver or result; it asks the composition
        root again at click time, so a rerun cannot leave an old artifact
        armed in a widget.
        """

        button = FluentButton(text, parent, color=ORANGE)
        button.setToolTip(
            "Open the shared Fit editor for this panel's current FINAL scan artifact"
        )
        button.setEnabled(self._fit_analysis_available)
        button.clicked.connect(self._open_fit_analysis)
        self.fit_analysis_available_changed.connect(button.setEnabled)
        return button

    def set_fit_analysis_available(self, available: bool) -> None:
        """Project whether this card currently resolves one exact FINAL source."""

        available = bool(available and self.fit_analysis_sink is not None)
        if available == self._fit_analysis_available:
            return
        self._fit_analysis_available = available
        self.fit_analysis_available_changed.emit(available)

    def _open_fit_analysis(self) -> None:
        sink = self.fit_analysis_sink
        if sink is None or not self._fit_analysis_available:
            self.set_status(
                "Fit requires this panel's current FINAL scan artifact",
                error=True,
            )
            return
        try:
            sink(self)
        except Exception as error:
            self.set_status(f"Fit failed to open: {error}", error=True)

    # ------------------------------------------------------------- settings UI

    def _make_param_widget(self, spec: ParamDecl, *, apply=None) -> QtWidgets.QWidget:
        """One widget per declarative ParamDecl with a semantic commit edge.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback.  Choice/toggle activation is already a
        complete user command.  Numeric spins disable keyboard tracking so an
        in-progress token remains local until Qt commits it.  Free text is a
        local draft until ``editingFinished``; no character can start a render.
        """

        cb = apply if apply is not None else self._set_param
        current = self.config.params.get(spec.key, spec.default)
        if spec.kind == "text":
            widget = PARAM_WIDGETS[spec.kind].build(
                spec,
                current,
                ParamWidgetContext(),
            )

            def commit() -> None:
                try:
                    value = PARAM_WIDGETS[spec.kind].read(widget)
                except (TypeError, ValueError):
                    return
                cb(spec.key, value)

            widget.editingFinished.connect(commit)
            return widget
        widget = PARAM_WIDGETS[spec.kind].build(
            spec,
            current,
            ParamWidgetContext(instant_apply=cb),
        )
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.setKeyboardTracking(False)
        return widget

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

    def _relim(self) -> str:
        """The panel's relim mode, defaulting to ``_RELIM_PARAM.default`` -- the ONE place that
        default lives, so every reader agrees instead of re-typing the 'tight' literal (#A3)."""
        return str(self.config.params.get("relim", _RELIM_PARAM.default))

    def refresh_on_show(self) -> None:
        """Re-seed every Setting control from ``config.params`` -- the SINGLE source of truth for a
        panel's params -- so the Setting popup shows the CURRENT values whenever it opens, even if they
        were changed elsewhere (the Edit tab writes the same config.params).  Each widget is re-seeded
        through its kind's ``PARAM_WIDGETS.write`` (one entry point, no per-key handwiring), with its
        change signals blocked so re-seeding does not re-fire ``_set_param`` (which would enqueue a
        duplicate compose).  This is the #6 fix: a control is a VIEW of config.params, refreshed on show, never a
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
        self._refresh_repeat_mode_control()

    def _build_plot(self) -> None:
        """Give this card its raster surface.

        A panel shows PIXELS the worker produced -- an already-coherent
        BoardFrame handed to :class:`~zlc_frontend.qt_widgets.SinglePanelHost`,
        which paints from immutable bytes and owns no figure.  The card
        therefore holds no Matplotlib object at all: rendering happens off the
        GUI thread and arrives here already rasterised.  The host is the ONE
        selector owner (design rule: an interactive window uses the
        QtRasterBoard chain; FrozenRasterView stays a frozen-report presenter), so
        the console card's selector switch and future gesture wiring go
        through the same binding as every other one-panel window.
        """

        from zlc_frontend.qt_widgets import FacetedPanelHost, SinglePanelHost

        if self.board is not None:
            return
        if self.config.kind == "grid":
            self.board = FacetedPanelHost(
                self.panel_id,
                empty_text="choose a facet axis in Setting",
            )
            self.board.focusRequested.connect(self._focus_grid_cell)
            self.board.overviewRequested.connect(self._return_to_grid_overview)
        else:
            self.board = SinglePanelHost(
                self.panel_id, empty_text="waiting for data")
            # A panel card exposes the frontend owner's complete typed gesture
            # without deciding what that rectangle means.  ROI/control-domain
            # routing belongs to the TaskConsole composition layer; keeping
            # that meaning out of this view is what lets every image-like
            # consumer share the same selector.
            self.board.rectangleSelected.connect(self.rectangle_selected.emit)
        # The pulse-preview answer protocol, verbatim: a wheel-zoom / pan /
        # double-middle commit is answered by re-composing THIS card at the
        # candidate's view under the candidate's revision; a clim-rail commit
        # routes through the ONE fixed-limits writer the Setting inputs use.
        self.board.viewCommitted.connect(self._on_view_committed)
        self.board.colorLimitsCommitted.connect(self._on_color_limits_committed)
        self.board.thresholdsCommitted.connect(
            self._on_histogram_thresholds_committed
        )
        # The console switch may have been armed before this card received its
        # first frame.  Replay the card-owned state onto the newly-created host;
        # otherwise the visible switch says ON while this first surface stays
        # inert until the operator toggles it twice.
        self._apply_selectors_state()
        self.canvas_holder.addWidget(self.board)

    def selection_for_rectangle_gesture(self, gesture):
        """Promote this card's exact held rectangle through its selector owner."""

        from zlc_frontend.qt_widgets import SinglePanelHost
        from zlc_frontend.selector import RectangleGesture

        if not isinstance(gesture, RectangleGesture):
            raise TypeError("gesture must be RectangleGesture")
        if gesture.panel_id != self.panel_id:
            raise ValueError("rectangle gesture belongs to another panel card")
        if not isinstance(self.board, SinglePanelHost):
            raise RuntimeError(
                "rectangle selection requires this card's single-panel host"
            )
        return self.board.board.selection_for_rectangle_gesture(gesture)

    def _focus_grid_cell(self, panel_index: int, selection) -> None:
        """Show one exact cell from the currently painted coherent overview."""

        if self.config.kind != "grid":
            return
        from zlc_frontend.panel_render import FacetedPanelFocus

        focus = FacetedPanelFocus(int(panel_index), selection)
        if focus == self._grid_focus:
            return
        self._grid_focus = focus
        self._view_pin = None
        self._request_display_render()

    def _return_to_grid_overview(self) -> None:
        """Return to the same typed grid without replacing its Qt host."""

        if self.config.kind != "grid" or self._grid_focus is None:
            return
        self._grid_focus = None
        self._view_pin = None
        self._request_display_render()

    def freeze_render_request(
        self,
        snapshot,
        frame_key,
        *,
        force: bool = False,
    ) -> _PanelRenderRequest | None:
        """Freeze one worker request without exposing mutable Qt/card state.

        Repeated timer ticks for the same source/display signature are folded
        here, before they can become executor work.  ``force`` is reserved for
        the visible Refresh action and still creates exactly one request.
        """

        name = self.config.signal
        if not name:
            self.set_status("pick a signal in Setting", error=False)
            return None
        value = None if snapshot is None else snapshot.value(name)
        if value is None or getattr(value, "snapshot", None) is None:
            self.set_status(f"waiting for {name}", error=False)
            return None
        return self._freeze_value_render_request(value, frame_key, force=force)

    def freeze_current_view_request(
        self,
        *,
        force: bool = False,
    ) -> _PanelRenderRequest | None:
        """Freeze a pure view edit against the already accepted data front.

        A selector/display/title/size commit is not a data-acquisition boundary.
        It must not advance ``ConsoleDataPlane`` merely because the operator
        moved a control.  A source rebind whose selected name differs from the
        accepted value waits for the next base tick or explicit Refresh.
        """

        name = self.config.signal
        value = (
            self._candidate_value
            if self.config.kind == "grid" and self._candidate_value is not None
            else self._last_value
        )
        if (
            not name
            or value is None
            or str(getattr(value, "name", "")) != str(name)
            or getattr(value, "snapshot", None) is None
        ):
            if name:
                self.set_status(f"waiting for {name}", error=False)
            else:
                self.set_status("pick a signal in Setting", error=False)
            return None
        return self._freeze_value_render_request(
            value,
            self._render_version,
            force=force,
        )

    def _freeze_value_render_request(
        self,
        value,
        frame_key,
        *,
        force: bool,
    ) -> _PanelRenderRequest | None:
        """Freeze one immutable value/display pair for the raster worker."""

        from zlc_frontend.panel_render import PanelProvenance

        schema = value.snapshot.block.schema
        site_map_view = None
        if self.config.kind == "sites":
            from zlc_frontend.site_map_render import (
                CalibrationSiteMapView,
                OccupancyCellView,
                OccupancySummarySiteMapView,
            )

            site_map_view = getattr(value, "presentation", None)
            if not isinstance(
                site_map_view,
                (
                    OccupancyCellView,
                    CalibrationSiteMapView,
                    OccupancySummarySiteMapView,
                ),
            ):
                self.set_status(
                    "Site map needs a committed calibration or an exact "
                    "single-cell occupancy result",
                    error=True,
                )
                return None
            if site_map_view.site_state_input.ref != value.snapshot.ref:
                self.set_status(
                    "Site map presentation belongs to another data revision",
                    error=True,
                )
                return None
        if schema != self._candidate_schema:
            self._candidate_schema = schema
            self._grid_focus = None
            self._last_figure = None
            self._pending_faceted_result = None
            self._refresh_grid_view_controls()
            self._refresh_repeat_mode_control()
        self._candidate_value = value
        view = (
            None
            if self.config.kind == "sites"
            else self._saved_view_spec(schema)
        )
        if self.config.kind == "grid" and view is None:
            self.set_status(
                "choose a named facet axis in Setting",
                error=False,
            )
            return None
        display = self._display_state()
        logical_size = tuple(
            int(value) for value in panel_display_size(self.config.size)
        )
        pixel_ratio = float(self._raster_pixel_ratio)
        size = tuple(
            max(1, QtCore.qRound(value * pixel_ratio))
            for value in logical_size
        )
        source_key = (
            str(self.config.kind),
            str(value.name),
            str(value.source),
            str(self.config.title),
            size,
            pixel_ratio,
            view,
            (
                None
                if site_map_view is None
                else (
                    site_map_view.view_identity,
                    site_map_view.calibration_identity,
                )
            ),
        )
        focus = self._grid_focus if self.config.kind == "grid" else None
        signature = (frame_key, source_key, display, focus)
        if not force and signature == self._requested_signature:
            return None
        self._render_request_revision += 1
        self._requested_signature = signature
        value_label = self._signal_axis_label(str(value.name))
        return _PanelRenderRequest(
            self.panel_id,
            self.config.kind,
            self._render_request_revision,
            signature,
            source_key,
            frame_key,
            value,
            display,
            None if self.config.kind == "sites" else self.view_intent(),
            str(self.config.title or value.name),
            value_label,
            size,
            self.config.size,
            pixel_ratio,
            PanelProvenance(value.run_id, value.epoch_id, value.join_digest),
            view,
            self.config.kind == "grid",
            focus,
        )

    def _signal_axis_label(self, signal_key: str) -> str:
        """Resolve one visible label without exposing a producer routing key."""

        labels = (
            self.axis_labels_provider()
            if callable(self.axis_labels_provider)
            else {}
        )
        authored = str(labels.get(signal_key, "")).strip() if labels else ""
        if authored:
            return authored
        short = str(
            self._signal_short_names_map().get(signal_key, "")
        ).strip()
        if short:
            return short
        return signal_key.rsplit("/", 1)[-1].strip() or "Signal"

    def accept_render_result(
        self,
        request: _PanelRenderRequest,
        *,
        frame=None,
        faceted_result=None,
        document=None,
        error: str | None = None,
    ) -> bool:
        """Accept one worker result on the Qt owner and nothing else."""

        if request.request_revision != self._render_request_revision:
            return False
        self._render_version = request.frame_key
        if error is not None:
            origin, self._pending_interaction_origin = (
                self._pending_interaction_origin,
                None,
            )
            if origin is not None and self.board is not None:
                self.board.discard_pending_interaction(origin)
            self.set_status(error, error=True)
            return True
        if request.faceted:
            from zlc_frontend.panel_render import FacetedPanelResult

            if not isinstance(faceted_result, FacetedPanelResult):
                self.set_status(
                    "render worker returned no complete faceted front",
                    error=True,
                )
                return True
            self._pending_faceted_result = faceted_result
            self._pending_frame = None
            self._last_figure = faceted_result.figure
            document = faceted_result.figure.document
        elif frame is None or (
            self.config.kind != "sites" and document is None
        ):
            self.set_status("render worker returned no complete front", error=True)
            return True
        else:
            self._pending_frame = frame
            self._pending_faceted_result = None
            self._last_figure = None
        self._last_value = request.value
        self._candidate_value = request.value
        self._last_document = document
        self._last_display = request.display
        self._candidate_schema = request.value.snapshot.block.schema
        self._refresh_grid_view_controls()
        self._refresh_repeat_mode_control()
        self.set_status("ok", error=False)
        return True

    def view_intent(self):
        """Which view this panel's kind asks its data for.

        Public because the worker request must carry the exact view requested
        by this card.  The Edit tab copies the accepted front and never derives
        a second view of the same data.
        """

        if self.config.kind == "sites":
            raise ValueError(
                "Site map is an exact composite payload, not a dataset ViewIntent"
            )
        if self.config.kind == "grid":
            raw = self.config.params.get(_VIEW_SPEC_PARAM)
            if raw is None:
                raise ValueError(
                    "grid panel needs a named facet ViewSpec before rendering"
                )
            from zlc_frontend.figure import view_spec_from_tree

            intent = view_spec_from_tree(raw).intent
            if intent not in {item[1] for item in _grid_view_intents()}:
                raise ValueError(
                    f"grid cell intent {intent.value!r} is not supported"
                )
            return intent
        try:
            return _panel_view_intents()[self.config.kind]
        except KeyError as error:
            raise ValueError(
                f"panel kind {self.config.kind!r} has no declared view intent"
            ) from error

    def _display_state(self):
        """The display knobs this panel's kind exposes, as the renderer's own state.

        The stored panel params are the persisted layout; the display state is
        what the rasteriser reads.  Deriving one from the other on every compose
        keeps a saved board and a live board showing the same thing, with no
        second copy of "what the operator chose" to fall out of step.
        """

        from zlc_frontend.curve_display import CurveDisplayState
        from zlc_frontend.histogram_display import (
            FacetedHistogramDisplayState,
            HistogramCountScale,
            HistogramDisplayState,
            HistogramFitMode,
            histogram_cell_thresholds_from_tree,
        )
        from zlc_frontend.image_display import ImageColormap, ImageDisplayState
        from zlc_frontend.meter_display import MeterDisplayState
        from zlc_frontend.display_range import RelimMode
        from zlc_frontend.figure import ViewIntent

        params = self.config.params
        intent = (
            None
            if self.config.kind == "sites"
            else self.view_intent()
        )
        # The panel's relim vocabulary IS the renderer's: tight / normal / fixed.
        # Converting rather than re-deciding keeps one set of names, so a mode
        # the renderer grows is a mode the Setting can offer with no mapping to
        # update in between.
        mode = RelimMode(str(self._relim()))
        fixed = None
        if mode is RelimMode.FIXED:
            fixed = (float(params.get("fixed_lo", 0.0)),
                     float(params.get("fixed_hi", 1.0)))
        pin_x, pin_y = self._view_pin or (None, None)
        if intent is ViewIntent.CURVE:
            return CurveDisplayState(
                revision=self._display_revision, relim_mode=mode, fixed_y_limits=fixed,
                x_view=pin_x,
            )
        if intent is ViewIntent.HISTOGRAM:
            param_kind = "hist" if self.config.kind == "grid" else self.config.kind
            display = HistogramDisplayState(
                revision=self._display_revision, relim_mode=mode,
                count_scale=(
                    HistogramCountScale.LOG
                    if bool(_resolved_panel_param(param_kind, params, "ylog"))
                    else HistogramCountScale.LINEAR
                ),
                bin_count=int(
                    _resolved_panel_param(
                        param_kind,
                        params,
                        "bins",
                    )
                ),
                fit_mode=HistogramFitMode(
                    str(_resolved_panel_param(param_kind, params, "fit"))
                ),
                fixed_count_limits=fixed,
                x_view=pin_x,
                thresholds=(
                    ()
                    if self.config.kind == "grid"
                    else tuple(
                        params.get(_HISTOGRAM_THRESHOLDS_PARAM, ())
                    )
                ),
            )
            if self.config.kind != "grid":
                return display
            raw_thresholds = params.get(
                _HISTOGRAM_CELL_THRESHOLDS_PARAM
            )
            return FacetedHistogramDisplayState(
                display,
                (
                    ()
                    if raw_thresholds is None
                    else histogram_cell_thresholds_from_tree(raw_thresholds)
                ),
            )
        if intent is ViewIntent.METER:
            focus = self._grid_focus
            return MeterDisplayState(
                0 if focus is None else int(focus.panel_index),
                None if focus is None else focus.selection,
                self._display_revision,
            )
        image_param_kind = (
            "2d" if self.config.kind == "grid" else self.config.kind
        )
        return ImageDisplayState(
            revision=self._display_revision,
            relim_mode=mode,
            colormap=ImageColormap(
                str(_resolved_panel_param(image_param_kind, params, "colormap"))
            ),
            fixed_color_limits=fixed,
            x_view=pin_x,
            y_view=pin_y,
        )

    def frozen_data_figure(self):
        """Return the exact typed figure behind the currently displayed panel.

        This is a projection of the already-owned immutable monitor revision,
        not another acquisition and not a GUI snapshot.  The same
        :class:`PanelComposer` supplies the document used to draw the card, so
        saving cannot re-guess axes or plot kind from an array shape.
        """

        if self.config.kind == "sites":
            raise RuntimeError(
                "Site map is an exact composite front, not a single-dataset DataFigure"
            )
        if self._last_figure is not None:
            return self._last_figure

        from zlc_frontend import DataFigure
        from zlc_frontend.figure import ResolvedDataset, ResolvedDatasetMap

        value = self._last_value
        snapshot = None if value is None else getattr(value, "snapshot", None)
        block = None if snapshot is None else getattr(snapshot, "block", None)
        if block is None:
            raise RuntimeError("the panel has no immutable data revision to save")
        document = self._last_document
        if document is None:
            raise RuntimeError("the panel has no accepted render document to save")
        if len(document.datasets) != 1:
            raise RuntimeError("a saved panel must bind exactly one typed dataset")
        dataset_id = document.datasets[0].dataset_id
        return DataFigure(
            document,
            ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        )

    def present(self) -> None:
        """Flush this card's composed front to the screen.  GUI thread only.

        Phase 2 of the board's two-phase render: the board composes every panel
        of a tick, then presents them together, so the screen never shows a torn
        mix of instants.
        """

        frame = self._pending_frame
        faceted = self._pending_faceted_result
        if frame is None and faceted is None:
            return
        self._pending_frame = None
        self._pending_faceted_result = None
        self._build_plot()
        if faceted is not None:
            if faceted.overview_png is not None:
                self.board.present_overview(
                    faceted.overview_png,
                    faceted.regions,
                )
            else:
                self.board.present_frame(faceted.frame)
        else:
            self.board.present_frame(frame)
        self._pending_interaction_origin = None
        self.front_presented.emit()

    def _build_settings(self) -> None:
        """Build the main-UI flat Setting surface over current typed state."""

        popup = FluentPopup(self)
        outer = QtWidgets.QVBoxLayout(popup)
        outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
        self._settings_scroll.setWidgetResizable(True)
        self._settings_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._settings_scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
        )
        self._settings_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff
        )
        self._settings_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        content = QtWidgets.QWidget()
        content.setStyleSheet("background: transparent;")
        self._settings_col = QtWidgets.QVBoxLayout(content)
        pad = scaled_px(10)
        self._settings_col.setContentsMargins(
            pad,
            pad,
            pad + fluent_scrollbar_thickness() + scaled_px(4),
            pad,
        )
        self._settings_col.setSpacing(scaled_px(10, minimum=6))
        self._settings_scroll.setWidget(content)
        outer.addWidget(self._settings_scroll)
        self.settings_popup = popup
        self._settings_h_hwm = 0

        display_specs = [
            spec for spec in _panel_param_decls(self.config.kind) if spec.display
        ] + [_RELIM_PARAM]
        labels = [
            "signal",
            "size",
            "view",
            "facet",
            "repeat",
            "lo / hi",
            "unit",
            "update",
            "title",
            *(spec.label for spec in display_specs),
        ]
        fm = self.fontMetrics()
        widest = max((fluent_text_width(fm, label) for label in labels), default=0)
        label_w = max(scaled_px(80, minimum=56), widest + scaled_px(10))

        def section(title):
            self._settings_col.addWidget(FluentSectionLabel(title))
            layout = QtWidgets.QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(scaled_px(6, minimum=4))
            self._settings_col.addLayout(layout)
            return layout

        # ---- Source: one typed dataset.  Combining producers belongs to a
        # Processor or explicit join, never an independent-latest GUI expression.
        source = section("Source")
        input_format = PLOT_KIND_SPEC_BY_KEY[self.config.kind].input_format
        if input_format:
            accepts = FluentLabel(f"accepts {input_format}")
            accepts.setWordWrap(True)
            accepts.setStyleSheet(
                f"color: {GREY}; background: transparent; border: none;"
            )
            source.addWidget(accepts)
        self.signal_combo = FluentTreeComboBox()
        self.signal_combo.setToolTip(
            "The typed dataset this panel displays, grouped by its producing node."
        )
        self.signal_combo.currentIndexChanged.connect(self._on_signal_pick)
        source.addWidget(
            FluentSettingRow("signal", self.signal_combo, label_width=label_w)
        )
        self.status = FluentLabel(self._status_text)
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self.status.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        source.addWidget(self.status)

        # ---- Display: the declared view knobs for this kind, emitted through the SHARED
        # row builder, so a kind that gains a knob shows it in both surfaces with no
        # wiring here.
        display = section("Display")
        self.size_combo = FluentComboBox()
        for preset in PANEL_SIZES:
            self.size_combo.addItem(preset, preset)
        index = self.size_combo.findData(self.config.size)
        if index >= 0:
            self.size_combo.setCurrentIndex(index)
        self.size_combo.setToolTip("Panel size preset (height × width half-units)")
        self.size_combo.currentIndexChanged.connect(
            lambda _i: self._on_size(
                str(self.size_combo.currentData() or self.config.size)
            )
        )
        display.addWidget(
            FluentSettingRow("size", self.size_combo, label_width=label_w)
        )
        self.param_widgets = self._emit_param_rows(
            display_specs, display.addWidget, self._set_param, label_w)
        self.grid_bins_row, self.grid_bins_widget = self._make_grid_bins_row(
            self._set_param,
            label_w,
        )
        if self.grid_bins_widget is not None:
            self.param_widgets["bins"] = self.grid_bins_widget
        display.addWidget(self.grid_bins_row)
        self.grid_ylog_row, self.grid_ylog_widget = self._make_grid_hist_param_row(
            "ylog",
            self._set_param,
            label_w,
        )
        if self.grid_ylog_widget is not None:
            self.param_widgets["ylog"] = self.grid_ylog_widget
        display.addWidget(self.grid_ylog_row)
        self.grid_colormap_row, self.grid_colormap_widget = (
            self._make_grid_cell_param_row(
                "2d",
                "colormap",
                self._image_intent(),
                self._set_param,
                label_w,
            )
        )
        if self.grid_colormap_widget is not None:
            self.param_widgets["colormap"] = self.grid_colormap_widget
        display.addWidget(self.grid_colormap_row)
        (
            self.grid_intent_row,
            self.grid_intent_combo,
            self.grid_facet_row,
            self.grid_facet_combo,
        ) = self._make_grid_view_rows(label_w)
        display.addWidget(self.grid_intent_row)
        display.addWidget(self.grid_facet_row)
        self.repeat_mode_row, self.repeat_mode_combo = self._make_repeat_mode_row(
            self._commit_repeat_mode,
            label_w,
        )
        display.addWidget(self.repeat_mode_row)
        self.fixed_lim_row, self.fixed_lo_edit, self.fixed_hi_edit = self._make_fixed_lim_row(
            self._on_fixed_lim_edited, label_w)
        display.addWidget(self.fixed_lim_row)
        self.fixed_lim_row.setVisible(self._relim() == "fixed")
        unit_row, self.unit_button, self.unit_label = self._make_unit_cycle_row(
            self._on_unit_cycle, label_w, with_label=True)
        display.addWidget(unit_row)
        self.update_combo = FluentComboBox()
        for interval in UPDATE_INTERVALS:
            self.update_combo.addItem(f"{interval} ms", interval)
        index = self.update_combo.findData(
            int(
                self.config.params.get("update_ms", DEFAULT_UPDATE_MS)
                or DEFAULT_UPDATE_MS
            )
        )
        if index >= 0:
            self.update_combo.setCurrentIndex(index)
        self.update_combo.setToolTip(
            "How often this panel redraws; acquisition is unaffected."
        )
        self.update_combo.currentIndexChanged.connect(self._on_update_interval)
        display.addWidget(
            FluentSettingRow("update", self.update_combo, label_width=label_w)
        )

        # ---- Analysis: what a drag on this panel means (the shared composite).
        self._build_analysis_section(section, label_w)

        # ---- Panel: card identity and the two standard panel actions.
        panel = section("Panel")
        self.title_edit = FluentLineEdit(self.config.title)
        self.title_edit.setPlaceholderText("panel title…")
        self.title_edit.setToolTip("Rename this panel (also the default save name)")
        self.title_edit.editingFinished.connect(self._commit_title)
        panel.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))
        self.remove_button = FluentButton("Remove", color=ORANGE)
        self.remove_button.setFixedWidth(scaled_px(72, minimum=58))
        self.remove_button.clicked.connect(self._remove_from_settings)
        self.edit_button = FluentButton("Edit…", color=ACCENT)
        self.edit_button.setFixedWidth(scaled_px(64, minimum=52))
        self.edit_button.setToolTip("Open this panel's full Edit tab")
        self.edit_button.clicked.connect(self._edit_from_settings)
        actions = QtWidgets.QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(scaled_px(6, minimum=4))
        actions.addWidget(self.remove_button)
        actions.addWidget(self.edit_button)
        actions.addStretch(1)
        panel.addLayout(actions)

        self._settings_col.addStretch(1)

        self._refresh_signal_combo()

    def _remove_from_settings(self) -> None:
        self.settings_popup.hide()
        self.remove_requested.emit(self)

    def _edit_from_settings(self) -> None:
        self.settings_popup.hide()
        self.edit_requested.emit(self)

    def _open_settings(self) -> None:
        self._settings_anchor.toggle(
            self._settings_scroll.widget(),
            prepare=self._prepare_settings_popup,
            present=self._present_settings_popup,
        )

    def _prepare_settings_popup(self) -> None:
        self.refresh_on_show()
        self._refresh_signal_combo()
        self._refresh_grid_view_controls()

    def _present_settings_popup(self) -> None:
        popup = self.settings_popup
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
        node is not running yet, so a saved layout's one typed binding survives a restart."""
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
        """Refresh the one dataset picker without rebuilding the popup."""

        combo = getattr(self, "signal_combo", None)
        if combo is not None:
            self._fill_slot_combo(combo, self.config.signal)

    def _current_schema(self):
        value = self._last_value
        snapshot = None if value is None else getattr(value, "snapshot", None)
        block = None if snapshot is None else getattr(snapshot, "block", None)
        return (
            self._candidate_schema
            if block is None
            else getattr(block, "schema", None)
        )

    def _preferences_from_view(
        self,
        view,
        schema,
        *,
        repeat_mode=None,
        facet_axis_ids=None,
    ):
        """Re-author display preferences without copying a ViewSpec as authority."""

        from zlc_frontend.figure import (
            AxisViewRole,
            ViewPreferences,
        )

        by_role = {
            role: tuple(
                binding.axis_id
                for binding in view.axis_bindings
                if binding.role is role
            )
            for role in (
                AxisViewRole.X,
                AxisViewRole.IMAGE_X,
                AxisViewRole.IMAGE_Y,
                AxisViewRole.BATCH,
                AxisViewRole.FACET,
                AxisViewRole.SAMPLE,
            )
        }
        repeat_id = schema.repeat_axis.axis_id
        facets = (
            tuple(facet_axis_ids)
            if facet_axis_ids is not None
            else tuple(
                axis_id
                for axis_id in by_role[AxisViewRole.FACET]
                if axis_id != repeat_id
            )
        )
        return ViewPreferences(
            repeat_mode=(
                repeat_mode
                if repeat_mode is not None
                else self._repeat_mode_from_view(view, schema)
            ),
            x_axis_id=next(iter(by_role[AxisViewRole.X]), None),
            image_x_axis_id=next(
                iter(by_role[AxisViewRole.IMAGE_X]),
                None,
            ),
            image_y_axis_id=next(
                iter(by_role[AxisViewRole.IMAGE_Y]),
                None,
            ),
            batch_axis_ids=tuple(
                axis_id
                for axis_id in by_role[AxisViewRole.BATCH]
                if axis_id != repeat_id
            ),
            facet_axis_ids=facets,
            sample_axis_ids=tuple(
                axis_id
                for axis_id in by_role[AxisViewRole.SAMPLE]
                if axis_id != repeat_id
            ),
        )

    @staticmethod
    def _display_selection_from_view(view):
        """Merge the persisted display-only terms for a re-suggestion.

        ``suggest_view`` accepts one Selection while ``ViewSpec`` stores a
        normalized tuple.  Rebuilding a facet/repeat binding must therefore
        carry the complete term set across instead of silently resetting ROI
        or named-axis selections.
        """

        from zlc_data import Selection

        terms = tuple(
            term
            for selection in view.display_selections
            for term in selection.terms
        )
        return None if not terms else Selection(terms)

    @staticmethod
    def _facet_axis_ids(view):
        from zlc_frontend.figure import AxisViewRole

        return tuple(
            binding.axis_id
            for binding in view.axis_bindings
            if binding.role is AxisViewRole.FACET
        )

    def _grid_candidate_view(self, intent, facet_axis_id):
        """Resolve one explicit facet choice to the complete named facet tuple."""

        from zlc_frontend.figure import (
            AxisViewRole,
            RepeatViewMode,
            SuggestionStatus,
            ViewPreferences,
            dataset_contract_for,
            suggest_view,
        )

        schema = self._current_schema()
        if schema is None:
            return None
        saved = self._saved_view_spec(schema)
        selection = (
            None
            if saved is None
            else self._display_selection_from_view(saved)
        )
        if saved is not None and saved.intent is intent:
            repeat_mode = self._repeat_mode_from_view(saved, schema)
            # Moving the explicit primary choice away from repeat restores the
            # intent's ordinary repeat default.  Other contract-required FACET
            # axes remain explicit in the resolved ViewSpec and are shown in
            # the control's complete tuple.
            if (
                repeat_mode is RepeatViewMode.FACET
                and facet_axis_id != schema.repeat_axis.axis_id
            ):
                repeat_mode = dataset_contract_for(intent).default_repeat_mode
            preferences = self._preferences_from_view(
                saved,
                schema,
                repeat_mode=(
                    RepeatViewMode.FACET
                    if facet_axis_id == schema.repeat_axis.axis_id
                    else repeat_mode
                ),
                facet_axis_ids=(
                    ()
                    if facet_axis_id == schema.repeat_axis.axis_id
                    else (facet_axis_id,)
                ),
            )
        else:
            preferences = ViewPreferences(
                repeat_mode=(
                    RepeatViewMode.FACET
                    if facet_axis_id == schema.repeat_axis.axis_id
                    else dataset_contract_for(intent).default_repeat_mode
                ),
                facet_axis_ids=(
                    ()
                    if facet_axis_id == schema.repeat_axis.axis_id
                    else (facet_axis_id,)
                ),
            )
        suggestion = suggest_view(
            schema,
            intent,
            selection,
            preferences=preferences,
        )
        if (
            suggestion.status is SuggestionStatus.NEEDS_INPUT
            or suggestion.spec is None
            or suggestion.spec.binding(facet_axis_id).role
            is not AxisViewRole.FACET
        ):
            return None
        return suggestion.spec

    def _grid_facet_choices(self, intent):
        from zlc_frontend.figure import dataset_axes

        schema = self._current_schema()
        if schema is None:
            return ()
        out = []
        for axis in dataset_axes(schema):
            if axis.size <= 1:
                continue
            view = self._grid_candidate_view(intent, axis.axis_id)
            if view is not None:
                out.append((axis, view))
        return tuple(out)

    def _make_grid_view_rows(self, label_w, *, apply=None):
        apply = self._commit_grid_facet if apply is None else apply
        intent_combo = FluentComboBox()
        intent_combo.setToolTip(
            "What every cell displays. This is a typed ViewIntent, not a "
            "shape-derived plot guess."
        )
        facet_combo = FluentComboBox()
        facet_combo.setToolTip(
            "Choose the primary declared facet. Each item shows the complete "
            "named facet_axis_ids tuple persisted in the ViewSpec."
        )
        intent_row = FluentSettingRow(
            "cell view",
            intent_combo,
            label_width=label_w,
        )
        facet_row = FluentSettingRow(
            "facet axes",
            facet_combo,
            label_width=label_w,
        )
        intent_combo.currentIndexChanged.connect(
            lambda _index: self._seed_grid_facet_combo(
                intent_combo,
                facet_combo,
            )
        )

        def commit(index: int) -> None:
            intent = intent_combo.currentData()
            axis_id = facet_combo.itemData(int(index))
            if intent is not None and axis_id is not None:
                apply(intent, axis_id)

        facet_combo.currentIndexChanged.connect(commit)
        visible = self.config.kind == "grid"
        intent_row.setVisible(visible)
        facet_row.setVisible(visible)
        return intent_row, intent_combo, facet_row, facet_combo

    def _make_grid_hist_param_row(self, key: str, apply, label_w):
        """Build one histogram-only Grid knob through the shared param owner."""

        return self._make_grid_cell_param_row(
            "hist",
            key,
            self._histogram_intent(),
            apply,
            label_w,
        )

    def _make_grid_cell_param_row(
        self,
        param_kind: str,
        key: str,
        intent,
        apply,
        label_w,
    ):
        """Build one cell-family Grid knob from its ordinary panel declaration."""

        if self.config.kind != "grid":
            row = FluentSettingRow(str(key), QtWidgets.QWidget(), label_width=label_w)
            row.hide()
            return row, None
        declaration = next(
            item
            for item in _panel_param_decls(str(param_kind))
            if item.key == str(key)
        )
        widget = self._make_param_widget(declaration, apply=apply)
        self._param_kinds[declaration.key] = declaration.kind
        row = FluentSettingRow(declaration.label, widget, label_width=label_w)
        row.setVisible(self._grid_cell_intent() is intent)
        return row, widget

    def _make_grid_bins_row(self, apply, label_w):
        return self._make_grid_hist_param_row("bins", apply, label_w)

    @staticmethod
    def _histogram_intent():
        from zlc_frontend.figure import ViewIntent

        return ViewIntent.HISTOGRAM

    @staticmethod
    def _image_intent():
        from zlc_frontend.figure import ViewIntent

        return ViewIntent.IMAGE

    def _grid_cell_intent(self):
        if self.config.kind != "grid":
            return None
        raw = self.config.params.get(_VIEW_SPEC_PARAM)
        if raw is None:
            return None
        from zlc_frontend.figure import view_spec_from_tree

        intent = view_spec_from_tree(raw).intent
        return (
            intent
            if intent in {item[1] for item in _grid_view_intents()}
            else None
        )

    def _seed_grid_facet_combo(self, intent_combo, facet_combo) -> None:
        if self.config.kind != "grid":
            return
        intent = intent_combo.currentData()
        if intent is None:
            return
        schema = self._current_schema()
        current = self._saved_view_spec(schema) if schema is not None else None
        current_ids = (
            ()
            if current is None or current.intent is not intent
            else tuple(
                binding.axis_id
                for binding in current.axis_bindings
                if binding.role.value == "FACET"
            )
        )
        preferred = next(
            (
                axis_id
                for axis_id in current_ids
                if axis_id != schema.repeat_axis.axis_id
            ),
            next(iter(current_ids), None),
        ) if schema is not None else None
        with _signals_blocked(facet_combo):
            facet_combo.clear()
            from zlc_frontend.figure import dataset_axes

            axes_by_id = {
                axis.axis_id: axis
                for axis in (() if schema is None else dataset_axes(schema))
            }
            for axis, view in self._grid_facet_choices(intent):
                displayed_view = (
                    current
                    if (
                        current is not None
                        and current.intent is intent
                        and axis.axis_id == preferred
                    )
                    else view
                )
                facet_summary = " × ".join(
                    (
                        f"{axes_by_id[axis_id].name} "
                        f"[{axis_id.value}]"
                        if axis_id in axes_by_id
                        else axis_id.value
                    )
                    for axis_id in self._facet_axis_ids(displayed_view)
                )
                facet_combo.addItem(
                    (
                        f"{axis.name} · {axis.role.value} → "
                        f"{facet_summary}"
                    ),
                    axis.axis_id,
                )
            index = facet_combo.findData(preferred)
            facet_combo.setCurrentIndex(index)
        facet_combo.setEnabled(facet_combo.count() > 0)

    def _refresh_grid_view_controls(self) -> None:
        if self.config.kind != "grid":
            return
        intent_combo = getattr(self, "grid_intent_combo", None)
        facet_combo = getattr(self, "grid_facet_combo", None)
        if intent_combo is None or facet_combo is None:
            return
        self._seed_grid_view_controls(intent_combo, facet_combo)
        histogram_grid = self._grid_cell_intent() is self._histogram_intent()
        for row_name in ("grid_bins_row", "grid_ylog_row"):
            row = getattr(self, row_name, None)
            if row is not None:
                row.setVisible(histogram_grid)
        image_row = getattr(self, "grid_colormap_row", None)
        if image_row is not None:
            image_row.setVisible(
                self._grid_cell_intent() is self._image_intent()
            )

    def _seed_grid_view_controls(self, intent_combo, facet_combo) -> None:
        """Seed either Setting or Edit from the same persisted ViewSpec."""

        if self.config.kind != "grid":
            return
        schema = self._current_schema()
        saved = self._saved_view_spec(schema) if schema is not None else None
        selected = (
            saved.intent
            if saved is not None
            else next(iter(_grid_view_intents()))[1]
        )
        with _signals_blocked(intent_combo):
            intent_combo.clear()
            for label, intent in _grid_view_intents():
                intent_combo.addItem(label, intent)
            index = intent_combo.findData(selected)
            intent_combo.setCurrentIndex(max(0, index))
        intent_combo.setEnabled(schema is not None)
        self._seed_grid_facet_combo(intent_combo, facet_combo)

    def _commit_grid_facet(self, intent, axis_id) -> bool:
        from zlc_frontend.figure import view_spec_to_tree

        candidate = self._grid_candidate_view(intent, axis_id)
        if candidate is None:
            self.set_status(
                "that named axis cannot form a complete typed grid view",
                error=True,
            )
            return False
        self._grid_focus = None
        self._view_pin = None
        changed = self._set_params(
            {_VIEW_SPEC_PARAM: view_spec_to_tree(candidate)}
        )
        self._refresh_grid_view_controls()
        self._refresh_repeat_mode_control()
        return changed

    def _saved_view_spec(self, schema):
        """Decode the one current owner-coded presentation value, if authored."""

        if self.config.kind == "sites":
            return None
        raw = self.config.params.get(_VIEW_SPEC_PARAM)
        if raw is None:
            return None
        from zlc_frontend.figure import view_spec_from_tree

        view = view_spec_from_tree(raw)
        if view.schema_fingerprint != schema.fingerprint:
            # A producer schema generation is a real view boundary.  Never
            # reinterpret the old bindings by axis position; discard this
            # presentation-only value and let the Figure owner suggest a new
            # typed default for the new schema.
            self.config.params.pop(_VIEW_SPEC_PARAM, None)
            self.config.params.pop(_HISTOGRAM_CELL_THRESHOLDS_PARAM, None)
            self.config.params.pop(_HISTOGRAM_THRESHOLDS_PARAM, None)
            return None
        allowed_intents = (
            {item[1] for item in _grid_view_intents()}
            if self.config.kind == "grid"
            else {_panel_view_intents()[self.config.kind]}
        )
        if view.intent not in allowed_intents:
            raise ValueError("saved panel view belongs to a different panel kind")
        return view

    def _repeat_modes_for_current_schema(self):
        """Typed repeat policies this one-panel host can actually render.

        FACET stays out of an ordinary SinglePanelHost because it evaluates to
        multiple cells.  It remains available on the real Grid host, where the
        repeat axis itself can be the one explicit facet.  BATCH remains a
        single cell with multiple series.  Rolling retention is a dataset
        concern and has no entry in RepeatViewMode.
        """

        if self.config.kind == "sites":
            return ()
        schema = self._current_schema()
        if schema is None:
            return ()
        from zlc_frontend.figure import RepeatViewMode, dataset_contract_for

        contract = dataset_contract_for(self._repeat_control_intent())
        return tuple(
            mode
            for mode in contract.repeat_modes
            if self.config.kind == "grid" or mode is not RepeatViewMode.FACET
        )

    def _repeat_mode_from_view(self, view, schema):
        from zlc_frontend.figure import (
            AxisViewRole,
            DisplayReductionMethod,
            RepeatViewMode,
        )

        binding = view.binding(schema.repeat_axis.axis_id)
        if binding.role is AxisViewRole.REDUCED:
            return (
                RepeatViewMode.MEAN
                if binding.reduction.method is DisplayReductionMethod.MEAN
                else RepeatViewMode.SUM
            )
        if binding.role is AxisViewRole.BATCH:
            return RepeatViewMode.BATCH
        if binding.role is AxisViewRole.FACET:
            return RepeatViewMode.FACET
        if binding.role is AxisViewRole.SAMPLE:
            return RepeatViewMode.SAMPLE
        if binding.role is AxisViewRole.SELECTED:
            # A one-repeat LATEST suggestion resolves to FixedIndex(0); for a
            # larger axis it is LatestNonempty.  Both are the same authored
            # policy, and no other repeat policy produces SELECTED.
            return RepeatViewMode.LATEST
        raise ValueError(f"unsupported repeat binding {binding.role.value}")

    def _selected_repeat_mode(self, schema):
        from zlc_frontend.figure import dataset_contract_for

        view = self._saved_view_spec(schema)
        if view is None and self._last_document is not None:
            layers = tuple(self._last_document.layers)
            if len(layers) == 1 and layers[0].view.schema_fingerprint == schema.fingerprint:
                view = layers[0].view
        if view is not None:
            return self._repeat_mode_from_view(view, schema)
        return dataset_contract_for(
            self._repeat_control_intent()
        ).default_repeat_mode

    def _repeat_control_intent(self):
        """Intent whose repeat policies the stable controls should display.

        A new Grid has a locally selected cell family before it has a committed
        facet ViewSpec.  Rendering still refuses that incomplete state, but the
        repeat row must be seedable while the operator is choosing the facet.
        """

        if self.config.kind != "grid":
            return self.view_intent()
        intent = self._grid_cell_intent()
        if intent is not None:
            return intent
        combo = getattr(self, "grid_intent_combo", None)
        candidate = None if combo is None else combo.currentData()
        return candidate or _grid_view_intents()[0][1]

    def _seed_repeat_mode_control(self, combo, row) -> None:
        modes = self._repeat_modes_for_current_schema()
        schema = self._current_schema()
        with _signals_blocked(combo):
            combo.clear()
            for mode in modes:
                combo.addItem(_repeat_mode_label(mode), mode)
            if modes and schema is not None:
                selected = self._selected_repeat_mode(schema)
                index = combo.findData(selected)
                combo.setCurrentIndex(max(0, index))
        row.setVisible(bool(modes))
        combo.setEnabled(bool(modes))

    def _make_repeat_mode_row(self, apply, label_w):
        from zlc_frontend.figure import RepeatViewMode

        combo = FluentComboBox()
        combo.setToolTip(
            "How the declared repeat axis is shown. Mean/Sum reduce only the "
            "repeat axis; Latest selects the latest non-empty logical repeat; "
            "Overlay keeps every repeat as a named series; Samples is the "
            "histogram sample binding. Rolling history is controlled by the "
            "rolling dataset, not by this menu."
        )

        def commit(index: int) -> None:
            mode = combo.itemData(int(index))
            if isinstance(mode, RepeatViewMode):
                apply(mode)

        combo.currentIndexChanged.connect(commit)
        row = FluentSettingRow("repeat", combo, label_width=label_w)
        self._seed_repeat_mode_control(combo, row)
        return row, combo

    def _refresh_repeat_mode_control(self) -> None:
        combo = getattr(self, "repeat_mode_combo", None)
        row = getattr(self, "repeat_mode_row", None)
        if combo is not None and row is not None:
            self._seed_repeat_mode_control(combo, row)

    def _commit_repeat_mode(self, mode) -> bool:
        """Resolve one typed preference to the sole persistent ViewSpec."""

        from zlc_frontend.figure import (
            RepeatViewMode,
            SuggestionStatus,
            ViewPreferences,
            suggest_view,
            view_spec_to_tree,
        )

        if not isinstance(mode, RepeatViewMode):
            raise TypeError("repeat mode must be RepeatViewMode")
        if mode not in self._repeat_modes_for_current_schema():
            raise ValueError(f"repeat mode {mode.value} is not renderable by this panel")
        schema = self._current_schema()
        if schema is None:
            return False
        intent = self.view_intent()
        saved = self._saved_view_spec(schema)
        if self.config.kind == "grid":
            if saved is None:
                return False
            if mode is RepeatViewMode.FACET:
                preferences = self._preferences_from_view(
                    saved,
                    schema,
                    repeat_mode=mode,
                    facet_axis_ids=(),
                )
            else:
                from zlc_frontend.figure import AxisViewRole

                facets = tuple(
                    binding.axis_id
                    for binding in saved.axis_bindings
                    if binding.role is AxisViewRole.FACET
                    and binding.axis_id != schema.repeat_axis.axis_id
                )
                if not facets:
                    self.set_status(
                        "choose a non-repeat facet axis before reducing repeats",
                        error=True,
                    )
                    return False
                preferences = self._preferences_from_view(
                    saved,
                    schema,
                    repeat_mode=mode,
                    facet_axis_ids=facets,
                )
        elif saved is not None:
            preferences = self._preferences_from_view(
                saved,
                schema,
                repeat_mode=mode,
            )
        else:
            preferences = ViewPreferences(repeat_mode=mode)
        selection = (
            None
            if saved is None
            else self._display_selection_from_view(saved)
        )
        suggestion = suggest_view(
            schema,
            intent,
            selection,
            preferences=preferences,
        )
        if suggestion.status is SuggestionStatus.NEEDS_INPUT or suggestion.spec is None:
            self.set_status(
                "repeat choice needs an explicit axis selection for this data",
                error=True,
            )
            return False
        if self.config.kind == "grid":
            self._grid_focus = None
        changed = self._set_params(
            {_VIEW_SPEC_PARAM: view_spec_to_tree(suggestion.spec)}
        )
        if changed:
            self._refresh_grid_view_controls()
        return changed

    def _on_signal_pick(self, _index: int) -> None:
        """Commit one card-local dataset binding and request one compose."""

        name = str(self.signal_combo.currentData() or "")
        if name == self.config.signal:
            return
        self.config.signal = name
        self.config.params.pop(_VIEW_SPEC_PARAM, None)
        self.config.params.pop(_HISTOGRAM_CELL_THRESHOLDS_PARAM, None)
        self.config.params.pop(_HISTOGRAM_THRESHOLDS_PARAM, None)
        self._last_value = None
        self._candidate_value = None
        self._last_document = None
        self._last_figure = None
        self._candidate_schema = None
        self._grid_focus = None
        self._pending_faceted_result = None
        self._refresh_repeat_mode_control()
        self._invalidate_render_binding()
        self._render_version = -1
        self._request_display_render()
        self.changed.emit()

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

    def _on_view_committed(self, commit) -> None:
        """CAS and answer one exact-front zoom/pan commit.

        ``SinglePanelHost`` deliberately forwards the whole typed commit.  A
        delayed gesture from an older front therefore cannot rewrite this
        card's current view, and a failed compose releases only that pending
        origin instead of leaving the selector permanently wedged.
        """

        from zlc_frontend.selector import (
            CurveViewportCommit,
            HistogramViewportCommit,
            ImageViewportCommit,
        )

        if not isinstance(
            commit,
            (CurveViewportCommit, HistogramViewportCommit, ImageViewportCommit),
        ):
            raise TypeError("view commit must retain its typed exact origin")
        host = self.board
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if self._display_revision != commit.origin.presentation.panel_revision:
            host.discard_pending_interaction(commit.origin)
            return
        candidate = commit.viewport
        viewport_revision = getattr(candidate, "viewport_revision", None)
        if viewport_revision is not None:      # image-family transform
            views = candidate.optional_coordinate_views_for_normalized_bounds()
            pin = views if any(view is not None for view in views) else None
            revision = int(viewport_revision)
        else:                                  # curve/histogram transform
            span = tuple(float(value) for value in candidate.x_limits)
            home = tuple(float(value) for value in candidate.home_x_limits)
            pin = None if span == home else (span, None)
            revision = int(candidate.display_revision)
        self._view_pin = pin
        self._display_revision = revision
        self._pending_interaction_origin = commit.origin
        # The commit changes only the card-owned display state.  The worker
        # answers it from the already accepted immutable data revision; Qt
        # never composes or waits for that answer.
        self._request_current_render()

    def _on_color_limits_committed(self, commit) -> None:
        """CAS one clim-rail commit into the shared fixed-limits fact."""

        from zlc_frontend.selector import ImageColorLimitsCommit

        if not isinstance(commit, ImageColorLimitsCommit):
            raise TypeError("color-limit commit must retain its typed exact origin")
        host = self.board
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if self._display_revision != commit.origin.presentation.panel_revision:
            host.discard_pending_interaction(commit.origin)
            return

        old_revision = self._display_revision
        lo, hi = (float(value) for value in commit.color_limits)
        self._store_fixed_lims(lo, hi)
        self._display_revision = old_revision + 1
        self._pending_interaction_origin = commit.origin
        self._request_current_render()
        self.changed.emit()

    def _on_histogram_thresholds_committed(self, commit) -> None:
        """CAS one drag step into the exact visible histogram cell."""

        from zlc_frontend.histogram_display import (
            FacetedHistogramDisplayState,
            faceted_histogram_display_with_thresholds,
            histogram_cell_thresholds_to_tree,
            histogram_display_with_thresholds,
        )
        from zlc_frontend.selector import HistogramThresholdCommit

        if not isinstance(commit, HistogramThresholdCommit):
            raise TypeError(
                "threshold commit must retain its typed exact origin"
            )
        host = self.board
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if self._display_revision != commit.origin.presentation.panel_revision:
            host.discard_pending_interaction(commit.origin)
            return

        display = self._display_state()
        if self.config.kind == "grid":
            focus = self._grid_focus
            if (
                focus is None
                or not isinstance(display, FacetedHistogramDisplayState)
            ):
                host.discard_pending_interaction(commit.origin)
                return
            candidate = faceted_histogram_display_with_thresholds(
                display,
                focus.selection,
                commit.thresholds,
            )
            if candidate == display:
                host.discard_pending_interaction(commit.origin)
                return
            self.config.params[_HISTOGRAM_CELL_THRESHOLDS_PARAM] = (
                histogram_cell_thresholds_to_tree(
                    candidate.cell_thresholds
                )
            )
            self._display_revision = candidate.revision
        else:
            candidate = histogram_display_with_thresholds(
                display,
                commit.thresholds,
            )
            if candidate == display:
                host.discard_pending_interaction(commit.origin)
                return
            self.config.params[_HISTOGRAM_THRESHOLDS_PARAM] = list(
                candidate.thresholds
            )
            self._display_revision = candidate.revision
        self._pending_interaction_origin = commit.origin
        self._request_current_render()
        self.changed.emit()

    def _store_fixed_lims(self, lo: float, hi: float) -> None:
        """The sole low-level writer for persisted fixed colour/count limits."""

        self.config.params["relim"] = "fixed"
        self.config.params["fixed_lo"] = float(lo)
        self.config.params["fixed_hi"] = float(hi)

    def apply_fixed_lims(self, lo: float, hi: float) -> None:
        """Persist the fixed lo/hi and re-compose with them NOW.

        The Setting popup's inputs and the Edit tab's both route here, so the
        pinned window has ONE writer; the rasteriser reads it back out of the
        display state on the next compose.
        """

        self._set_params(
            {
                "relim": "fixed",
                "fixed_lo": float(lo),
                "fixed_hi": float(hi),
            }
        )

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
        # Render workers return string-only outcomes.  This funnel is therefore
        # Qt-owner-only; no worker ever reaches a widget or parks a mutable
        # exception on the card.
        if QtCore.QThread.currentThread() is not self.thread():
            raise RuntimeError("panel status is Qt-owner-only")
        self._status_text = str(text)
        if hasattr(self, "status"):
            self.status.setText(str(text))
        self.setting_button.setToolTip(f"Panel settings — {text}" if text else "Panel settings")
        if error is not getattr(self, "_status_error", None):
            self._status_error = bool(error)
            colour = RED if error else GREY
            if hasattr(self, "status"):
                self.status.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
            self.setting_button.set_color(colour)

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

    def _commit_title(self) -> None:
        """Commit the locally edited title once focus/Return finishes it.

        One string: the frame header shows it and the composer carries it as the
        dataset label, so the picture and the card can never be captioned
        differently.
        """

        text = str(self.title_edit.text())
        if text == self.config.title:
            return
        self.config.title = text
        self._invalidate_render_binding()
        self._request_display_render()
        self.changed.emit()

    def _on_size(self, size: str) -> None:
        if str(size) == self.config.size:
            return
        self.config.size = str(size)
        # The accepted front remains installed while the card and its siblings
        # relayout.  Repaints are held only for that synchronous geometry
        # transaction; the worker later swaps in the correctly sized immutable
        # raster without ever clearing/rebuilding this Qt surface.
        self.setUpdatesEnabled(False)
        try:
            self._invalidate_render_binding()
            self._apply_fixed_size()
            # If the Setting frame is open, GROW it immediately to match the new (taller) panel -- but
            # the high-water mark means a SMALLER size never snaps it shorter (#H3i-2).
            if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                self._size_settings_popup()
            self._request_display_render()
            self.changed.emit()
            self.layout_changed.emit()
        finally:
            self.setUpdatesEnabled(True)

    def _build_analysis_section(self, section_box, label_w) -> None:
        """Expose the card's one exact-source Fit entrance in Setting.

        Fit authoring itself lives in the shared DataFigure host.  Repeating its
        model/constraint widgets here would create a second draft and a second
        solver lifecycle, while the old Hub Analysis controls cannot represent
        the current named-axis FitSpec at all.
        """

        if self.fit_analysis_sink is None:
            return
        layout = section_box("Analysis")
        layout.addWidget(self.make_fit_analysis_button())
        note = FluentLabel("Available only for this panel's current FINAL scan artifact")
        note.setWordWrap(True)
        note.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        layout.addWidget(note)

    def _set_param(self, key: str, value) -> bool:
        return self._set_params({str(key): value})

    def _set_params(self, updates: Mapping[str, object]) -> bool:
        """Commit one semantic parameter transaction and request one compose.

        Every surface that edits this panel (the Setting popup, the Edit tab, a
        drag that arms an analysis) comes through here, so a knob has one home
        -- ``config.params``, which is also what the saved layout persists.  The
        composer reads them back through :meth:`_display_state`, so what is
        stored and what is drawn cannot drift.
        """

        missing = object()
        changed = {
            str(key): value
            for key, value in dict(updates).items()
            if self.config.params.get(str(key), missing) != value
        }
        if not changed:
            return False
        self.config.params.update(changed)
        if "relim" in changed:
            value = changed["relim"]
            # Flipping INTO ``fixed`` FREEZES the window currently on screen.
            # Seeding from the composed front's own limits (never the default
            # 0..1 pair) is what keeps the picture: pinning a counts histogram
            # or a camera frame to 0..1 empties it, which reads as "the panel
            # just died".  The operator then types exact bounds.
            if str(value) == "fixed" and not {
                "fixed_lo", "fixed_hi"
            }.issubset(changed):
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
        self._request_display_render()
        self.changed.emit()
        return True

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

    def _request_display_render(self) -> None:
        """Commit one display revision and enqueue one latest-only compose.

        This method never rasterises and never waits.  The console freezes the
        latest immutable source value and coalesces requests while its one
        render worker is busy.
        """

        self._display_revision += 1
        self._request_current_render()

    def _request_current_render(self, *, force: bool = False) -> None:
        self._render_request(self, force=bool(force))

    def set_selectors_enabled(self, on: bool) -> None:
        """The console header's "Selectors" switch for THIS card: remember the desired state and
        gate the CURRENT plotter now (in place -- no rebuild, no flash).  Every later rebind /
        focus swap re-applies it through ``_apply_selectors_state``, so a fresh figure always
        inherits the switch."""
        self._selectors_on = bool(on)
        self._apply_selectors_state()

    def _apply_selectors_state(self) -> None:
        """Carry the board header's Selectors switch onto this card's surface.

        The switch is the card's state, not the surface's: it is stored here so
        a surface bound later comes up matching the switch instead
        of coming up live behind the operator's back.
        """

        board = self.board
        if board is not None and hasattr(board, "set_selectors_enabled"):
            board.set_selectors_enabled(bool(self._selectors_on))

    # ------------------------------------------------------------- plot lifecycle
    def _invalidate_render_binding(self) -> None:
        """Invalidate only the worker request identity, never the Qt surface.

        Source/title/size changes cause the worker to replace its composer from
        the next request's frozen ``source_key``.  The accepted front stays in
        place until that replacement is ready, so an edit cannot flash an empty
        card or rebuild a widget subtree.
        """

        self._pending_frame = None
        self._pending_faceted_result = None
        self._render_request_revision += 1
        self._requested_signature = None

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
        self._pending_faceted_result = None
        self._candidate_value = None
        self._candidate_schema = None
        self._grid_focus = None
        self._last_figure = None
        self._last_document = None
        self._last_display = None
        self._requested_signature = None
        if board is not None:
            self.canvas_holder.removeWidget(board)
            board.setParent(None)
            board.deleteLater()

    def shutdown(self) -> None:
        """Release this card's Qt surface.

        Worker requests contain no card/QWidget reference and are rejected by
        panel identity after removal, so teardown never waits on raster work.
        """

        self._teardown_plot()
