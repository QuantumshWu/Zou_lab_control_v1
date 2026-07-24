"""The console's panel card: its raster surface, its Setting, and the card geometry.

A card owns NO plotting object.  Its surface is a raster board painted from
immutable bytes, and every picture on it was composed on a worker from one
frozen snapshot (:mod:`zlc_frontend.panel_render`).  That is what keeps a
megapixel frame off the thread that also has to stay responsive, and it is why
the display knobs here are stored FACTS (``config.params``) rather than pushes
into a live figure: the knobs are read back on the next compose, so what is
stored and what is drawn cannot drift.

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
    FitAuthoringPane,
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
    popup_gap,
    scaled_px,
    setting_label_width as _setting_label_width,
    signals_blocked as _signals_blocked,
)
from zlc_frontend.form import (
    FormFieldProps,
    choice_value_from_tree,
    choice_value_to_tree,
    lenient_float as _safe_float,
)
from zlc_frontend.render_style import panel_display_size
from zlc_frontend.panel_params import (
    panel_param_decls as _panel_param_decls,
    resolved_panel_param as _resolved_panel_param,
)
from zlc_frontend import (
    HISTOGRAM_CELL_THRESHOLDS_PARAM as _HISTOGRAM_CELL_THRESHOLDS_PARAM,
    HISTOGRAM_THRESHOLDS_PARAM as _HISTOGRAM_THRESHOLDS_PARAM,
    RELIM_PARAM as _RELIM_PARAM,
    VIEW_SPEC_PARAM as _VIEW_SPEC_PARAM,
    grid_view_intents as _grid_view_intents,
    panel_view_intents as _panel_view_intents,
    repeat_mode_label as _repeat_mode_label,
)
from .console_records import (
    DEFAULT_UPDATE_MS,
    PANEL_KINDS,
    PanelConfig,
    UPDATE_INTERVALS,
)
from zlc_frontend.panel_size import PANEL_SIZES
from zlc_frontend.plot_kind import PLOT_KIND_SPEC_BY_KEY

from .panel_board import card_size as _card_size
from .render_lane import PanelRenderRequest as _PanelRenderRequest


_FIT_SPEC_PARAM = "figure_fit_spec"

# qt_widgets submodules are reached as ATTRIBUTES of the one facade binding: their names are
# deliberately absent from the facade __all__, and the package forbids outside deep imports.
FORM_WIDGET_HANDLERS = _qt_widgets.FORM_WIDGET_HANDLERS
coerce_short_labels = _qt_widgets.coerce_short_labels
fill_grouped_signal_combo = _qt_widgets.fill_grouped_signal_combo


def _decode_panel_view(config: PanelConfig):
    """Decode and validate the one persisted frontend-owned ViewSpec."""

    raw = config.params.get(_VIEW_SPEC_PARAM)
    if raw is None:
        return None
    if config.kind == "sites":
        raise ValueError("Site-map panels cannot persist a generic ViewSpec")
    from zlc_frontend.figure import grid_facet_axis, view_spec_from_tree

    view = view_spec_from_tree(raw)
    allowed = (
        {item[1] for item in _grid_view_intents()}
        if config.kind == "grid"
        else {_panel_view_intents()[config.kind]}
    )
    if view.intent not in allowed:
        raise ValueError("saved panel view belongs to another panel kind")
    if config.kind == "grid":
        grid_facet_axis(view)
    return view



# A fresh plot panel is BLANK: a pure view is fully decoupled from acquisition, so
# it shows nothing until the user picks a declared signal in its Setting
# -- it must NOT auto-bind to any node's signal.  An empty source is the blank
# state; ``refresh`` treats it (and a source that produces None) as "pick a signal"
# rather than an error, so a blank panel sits quietly until wired.


# Board layout (raw px).  The board is a pure PIXEL plane of card AABBs -- there is NO column
# grid.  WIDTH and HEIGHT wrap the exact FigureSpec logical panel size plus Fluent chrome;
# the card is exactly large enough for its figure, with NO stretch or blank
# padding below (every size hugs its content).  ``PanelConfig.col`` is the card's pixel X and
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
    front_presented = QtCore.pyqtSignal()
    selectors_enabled_changed = QtCore.pyqtSignal(bool)
    figure_outputs_changed = QtCore.pyqtSignal()
    fit_requested = QtCore.pyqtSignal()
    fit_cancel_requested = QtCore.pyqtSignal()

    @staticmethod
    def validate_config(config: PanelConfig):
        """Validate every owner-coded value consumed during card construction.

        This is intentionally QWidget-free so a whole TaskConsole layout can be
        checked before the currently running console is stopped.
        """

        if not isinstance(config, PanelConfig):
            raise TypeError("panel card config must be PanelConfig")
        config.update_ms
        fit_spec = None
        raw_fit_spec = config.params.get(_FIT_SPEC_PARAM)
        if raw_fit_spec is not None:
            from zlc_data import fit_spec_from_tree

            fit_spec = fit_spec_from_tree(raw_fit_spec)
        _decode_panel_view(config)
        return fit_spec

    def __init__(self, config: PanelConfig, parent=None, *, names_provider=None,
                 sources_provider=None, formats_provider=None,
                 short_names_provider=None, axis_labels_provider=None,
                 render_request=None):
        fit_spec = self.validate_config(config)
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
        # callable -> {stable signal key: SHORT name}, so the
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
        # Figure-owned derived signals.  Area is an authoritative named-axis
        # Selection promoted from a completed gesture; Cross is a completed
        # right-click coordinate.  Neither is a Measurement parameter and
        # neither opens another window.
        self._area_commit = None
        self._cross_commit = None
        # Figure Fit is one card-owned request/result state with two editable
        # views (Setting and Edit).  Neither view owns a solver or opens a
        # second window.  Results are exact-source values and therefore cannot
        # survive a source revision change as a visible overlay.
        self._fit_panes: list[FitAuthoringPane] = []
        self._fit_syncing_panes = False
        self._fit_options = ()
        self._fit_options_identity = None
        self._fit_draft = None
        self._fit_draft_context = None
        self._fit_active_spec = fit_spec
        self._fit_result = None
        self._fit_result_identity = None
        self._fit_request_revision = 0
        self._fit_pending_source_ref = None
        # The card's display surface is an immutable-bytes raster board.
        # The panel's stable identity: the board, its composer and every frame
        # they exchange are keyed on it, so a presented frame can only ever land
        # on the panel it was composed for.
        self.panel_id = str(config.panel_id)
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
        self._last_display = None
        # Worker completion and visible presentation are different facts.
        # Promote this group only after the Qt board accepts its matching
        # immutable frame in ``present()``.
        self._pending_figure = None
        self._pending_display = None
        self._pending_size_name = None
        self._pending_pixel_ratio = None
        self._pending_run_id = None
        self._pending_title = None
        self._pending_value_label = None
        self._pending_value = None
        self._presented_figure = None
        self._presented_display = None
        self._presented_size_name = None
        self._presented_pixel_ratio = None
        self._presented_run_id = None
        self._presented_title = None
        self._presented_value_label = None
        self._presented_value = None
        self._candidate_schema = None
        self._grid_focus = None
        self._render_request_revision = 0
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None
        # Qt paints the card in logical pixels, while the worker raster is
        # authored at the physical-pixel ratio of the screen containing the
        # TaskConsole.  The console owns screen observation and updates this
        # value; a DPR change is therefore a render-key change even when the
        # displayed dataset revision did not advance.
        self._raster_pixel_ratio = 1.0
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        # The newest display revision authored by a wheel/pan/rail gesture.
        # An older worker front may still be useful while a button is held,
        # but it cannot settle this intent.  Pulse Preview uses the identical
        # revision-owned answer rule.
        self._pending_interaction_revision = None
        # Bumped by every display-knob edit.  The renderer reads it to tell a
        # genuinely new display from a repeat of the same one.
        self._display_revision = 0
        # The board gestures' authored viewport pin (x_view, y_view) in DATA
        # coordinates, or None for home.  Display-only continuity like the
        # colour window -- deliberately NOT in config.params: a saved layout
        # reopens at home, exactly like the pulse preview.
        self._view_pin = None
        # The console header's "Selectors" switch state for THIS card (set via
        # ``set_selectors_enabled``; default OFF keeps Monitor display-only).
        # Every plotter (re)build parks its selector layer to this flag (``_apply_selectors_state``),
        # so a fresh figure always inherits the switch instead of coming up live.
        self._selectors_on = False
        # {param key: declared kind} for each rendered row, so reopening the Setting
        # re-seeds a control through its OWN kind's writer instead of guessing from
        # the stored value's Python type.
        self._param_fields: dict[str, FormFieldProps] = {}
        # The source-frame key at this panel's last render -- the per-panel multi-rate refresh
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

        self._apply_fixed_size()
        self.set_status("waiting for data…", error=False)

    # ------------------------------------------------------------- geometry
    def _apply_fixed_size(
        self,
        size_name: str | None = None,
        *,
        sync_board: bool = True,
    ) -> None:
        """Apply one size fact to both Fluent chrome and the plot host."""

        resolved_size = self.config.size if size_name is None else str(size_name)
        logical_size = tuple(
            int(value) for value in panel_display_size(resolved_size)
        )
        if sync_board and self.board is not None:
            self.board.set_logical_size(logical_size)
        self.setFixedSize(*_card_size(resolved_size))
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

    # ------------------------------------------------------------- Figure Fit
    def _fit_capable_kind(self) -> bool:
        """Whether this panel can expose the named-axis Figure Fit editor."""

        return self.config.kind in {"1d", "monitor", "2d", "grid"}

    def make_fit_authoring_pane(
        self,
        parent=None,
        *,
        label_width: int | None = None,
    ) -> FitAuthoringPane:
        """Build another editable view of this card's one Fit state."""

        if not self._fit_capable_kind():
            raise ValueError("this panel kind does not support Figure Fit")
        pane = FitAuthoringPane(parent, label_width=label_width)
        pane.fitRequested.connect(
            lambda _revision, spec, owner=pane: self._accept_fit_request(
                owner,
                spec,
            )
        )
        pane.fitRequestRejected.connect(
            lambda diagnostic: self.set_status(
                f"Fit request invalid: {diagnostic}",
                error=True,
            )
        )
        pane.clearRequested.connect(self.clear_fit)
        pane.editorChanged.connect(
            lambda _revision, owner=pane: self._fit_editor_changed(owner)
        )
        self._fit_panes.append(pane)
        if (
            self._fit_options_identity is None
            and self._presented_figure is not None
            and self._presented_value is not None
        ):
            self._sync_fit_authoring_from_presented(prepare_authoring=True)
            # The shared sync reconciles every registered pane, including the
            # one just appended above.  Do not perform the same model/draft
            # reconciliation a second time below.  If no identity could be
            # prepared, the ordinary retained-options path still leaves this
            # new pane in an explicit prepare state.
            if self._fit_options_identity is not None:
                return pane
        if self._fit_options:
            selected = (
                None
                if self._fit_active_spec is None
                else self._fit_active_spec.model_id
            )
            if selected not in {item.spec.model_id for item in self._fit_options}:
                selected = None
            pane.reconcile_options(
                self._fit_options,
                selected_model=selected,
            )
            if self._fit_draft is not None:
                pane.set_draft_state(self._fit_draft)
            else:
                self._fit_draft = pane.draft_state
            pane.set_busy(None, draft_ready=self._fit_result is not None)
        else:
            pane.set_busy("prepare", draft_ready=False)
        return pane

    def release_fit_authoring_pane(self, pane: FitAuthoringPane) -> None:
        """Forget one Edit-surface view without touching the shared Fit state."""

        if pane in self._fit_panes:
            self._fit_panes.remove(pane)

    def _fit_editor_changed(self, owner: FitAuthoringPane) -> None:
        """Mirror one local draft into the other stable Fit surface."""

        if self._fit_syncing_panes or owner not in self._fit_panes:
            return
        try:
            draft = owner.draft_state
        except (TypeError, ValueError, RuntimeError):
            return
        if draft is None:
            return
        self._fit_draft = draft
        self._fit_syncing_panes = True
        try:
            for pane in tuple(self._fit_panes):
                if pane is owner or not pane.fit_models:
                    continue
                pane.set_draft_state(draft, notify=False)
        finally:
            self._fit_syncing_panes = False

    def _current_fit_selection(self):
        from zlc_data import Selection

        value = self._presented_value
        commit = self._area_commit
        if (
            commit is not None
            and isinstance(commit.selection, Selection)
            and self._figure_commit_matches_value(commit, value)
        ):
            return commit.selection
        return None

    def _fit_selection_changed(self) -> None:
        """Reprepare the shared Fit draft when its Figure Area changes.

        A completed selector gesture is still only a Figure-owned candidate.
        It may prefill both Fit panes, but it cannot replace the submitted
        ``FitSpec`` or start a solver run until the operator presses ``Fit``.
        """

        self._fit_options_identity = None
        # Setting owns a permanent Fit pane even while its popup is closed.
        # Rebinding every model against a multi-megapixel Area solely to update
        # that hidden pane put transform/digest work in the mouse-release
        # handler.  Keep the Figure selection now; opening Setting or building
        # Edit catches the shared draft up through the same owner below.
        if (
            self._presented_figure is not None
            and self._presented_value is not None
            and any(pane.isVisible() for pane in self._fit_panes)
        ):
            self._sync_fit_authoring_from_presented(prepare_authoring=True)

    def _clear_fit_result(self, *, notify: bool) -> None:
        changed = self._fit_result is not None
        self._fit_result = None
        self._fit_result_identity = None
        if changed and notify:
            self.figure_outputs_changed.emit()

    def _sync_fit_authoring_from_presented(
        self,
        *,
        prepare_authoring: bool,
    ) -> None:
        """Advance a submitted Fit and optionally prepare its visible editor.

        A new camera front must keep an explicitly submitted Fit running, but
        it must not prepare every catalog model merely because the permanent
        Setting pane exists while hidden.  Callers therefore state whether an
        authoring surface is actually being presented.  Result/source
        correctness and submitted-Fit scheduling remain unconditional; only
        reversible model/draft preparation is lazy.
        """

        if not isinstance(prepare_authoring, bool):
            raise TypeError("prepare_authoring must be bool")

        if not self._fit_capable_kind():
            return
        figure = self._presented_figure
        source = self._presented_value
        snapshot = None if source is None else getattr(source, "snapshot", None)
        if figure is None or snapshot is None:
            return

        # A result is meaningful only for the exact currently painted revision.
        if self._fit_result is not None and self._fit_result.source_ref != snapshot.ref:
            self._clear_fit_result(notify=True)

        # Schema incompatibility invalidates a submitted request regardless of
        # editor visibility.  This is cheap source correctness, not authoring
        # preparation, and therefore remains on every accepted-front path.
        seed = self._fit_active_spec
        if (
            seed is not None
            and seed.input_schema_fingerprint != snapshot.ref.schema_fingerprint
        ):
            seed = None
            self._fit_active_spec = None
            self._fit_draft = None
            self._fit_draft_context = None
            self._fit_options_identity = None
            self._fit_pending_source_ref = None
            self._commit_persisted_params(remove=(_FIT_SPEC_PARAM,))
            self.fit_cancel_requested.emit()
            self._clear_fit_result(notify=True)

        # Focus is renderer state only: the presented DataFigure remains the
        # complete Grid authority.  A hidden editor still does no preparation;
        # when Setting/Edit is actually shown it may prepare against that full
        # Figure without replacing batch axes with the selected cell.
        if (
            self.config.kind == "grid"
            and self._grid_focus is not None
            and not prepare_authoring
        ):
            self._queue_active_fit_for_presented()
            return
        if not prepare_authoring:
            self._queue_active_fit_for_presented()
            return
        selection = self._current_fit_selection()
        from zlc_storage import canonical_digest
        from zlc_data import selection_to_tree

        selection_identity = (
            None
            if selection is None
            else canonical_digest(selection_to_tree(selection))
        )
        identity = (
            snapshot.ref.schema_fingerprint,
            figure.document.document_id,
            figure.document.revision,
            figure.document.layers[0].view,
            selection_identity,
        )
        if identity != self._fit_options_identity:
            from zlc_frontend import prepare_fit_authoring_options

            try:
                options = prepare_fit_authoring_options(
                    figure,
                    selection,
                    seed_spec=seed,
                )
            except (TypeError, ValueError, RuntimeError) as error:
                self._fit_options = ()
                self._fit_options_identity = identity
                self._fit_draft = None
                self._fit_draft_context = None
                for pane in tuple(self._fit_panes):
                    if pane.fit_models:
                        pane.clear_options()
                    pane.set_busy("prepare", draft_ready=False)
                self.set_status(f"Fit unavailable: {error}", error=True)
                return

            selected_model = None if seed is None else seed.model_id
            models = {option.spec.model_id for option in options}
            if selected_model not in models:
                selected_model = None
                if self._fit_active_spec is not None:
                    self._fit_active_spec = None
                    self._commit_persisted_params(remove=(_FIT_SPEC_PARAM,))
                    self._clear_fit_result(notify=True)
            self._fit_options = options
            self._fit_options_identity = identity
            from zlc_frontend import reconcile_fit_authoring_draft

            draft_context = tuple(
                (
                    option.spec.model_id,
                    option.spec.input_schema_fingerprint,
                    option.spec.fit_axis_ids,
                    option.parameter_names,
                )
                for option in options
            )
            previous_draft = (
                self._fit_draft
                if self._fit_draft_context == draft_context
                else None
            )
            self._fit_draft = reconcile_fit_authoring_draft(
                options,
                previous_draft,
                selected_model=selected_model,
            )
            self._fit_draft_context = draft_context
            for pane in tuple(self._fit_panes):
                pane.reconcile_options(
                    options,
                    selected_model=selected_model,
                )
                pane.set_draft_state(self._fit_draft)
                pane.set_busy(None, draft_ready=self._fit_result is not None)

        self._queue_active_fit_for_presented()

    def _accept_fit_request(self, owner: FitAuthoringPane, spec) -> None:
        """Promote one visible pane draft into the card's active FitSpec."""

        from zlc_data import FitSpec, fit_spec_to_tree

        if owner not in self._fit_panes or not isinstance(spec, FitSpec):
            return
        draft = owner.draft_state
        if draft is None:
            return
        option = next(
            (
                item
                for item in self._fit_options
                if item.spec.model_id == spec.model_id
            ),
            None,
        )
        if option is None or (
            spec.input_schema_fingerprint != option.spec.input_schema_fingerprint
            or spec.committed_transform != option.spec.committed_transform
            or spec.fit_axis_ids != option.spec.fit_axis_ids
            or spec.batch_axis_ids != option.spec.batch_axis_ids
            or spec.numeric_policy != option.spec.numeric_policy
        ):
            self.set_status("Fit request differs from the prepared Figure", error=True)
            return
        self._fit_request_revision += 1
        self._fit_pending_source_ref = None
        self.fit_cancel_requested.emit()
        self._fit_draft = draft
        self._fit_active_spec = spec
        self._commit_persisted_params(
            {_FIT_SPEC_PARAM: fit_spec_to_tree(spec)}
        )
        self._clear_fit_result(notify=True)
        self._queue_active_fit_for_presented()

    def _queue_active_fit_for_presented(self) -> None:
        spec = self._fit_active_spec
        source = self._presented_value
        snapshot = None if source is None else getattr(source, "snapshot", None)
        if spec is None or snapshot is None:
            return
        if spec.input_schema_fingerprint != snapshot.ref.schema_fingerprint:
            return
        if self._fit_result is not None and self._fit_result.source_ref == snapshot.ref:
            return
        if self._fit_pending_source_ref == snapshot.ref:
            return
        for pane in tuple(self._fit_panes):
            pane.set_busy("fit", draft_ready=False)
        self.fit_requested.emit()

    def freeze_fit_request(self):
        """Freeze the active Fit against the exact currently painted source."""

        spec = self._fit_active_spec
        source = self._presented_value
        snapshot = None if source is None else getattr(source, "snapshot", None)
        if spec is None or snapshot is None:
            return None
        if spec.input_schema_fingerprint != snapshot.ref.schema_fingerprint:
            return None
        from .panel_fit import PanelFitRequest

        self._fit_request_revision += 1
        request = PanelFitRequest(
            self.panel_id,
            self._fit_request_revision,
            source,
            spec,
        )
        self._fit_pending_source_ref = snapshot.ref
        return request

    def accept_fit_completion(self, request, result, error: str | None) -> bool:
        """Admit a solver result only for the exact still-visible source."""

        from .panel_fit import PanelFitRequest

        if not isinstance(request, PanelFitRequest):
            return False
        source = self._presented_value
        snapshot = None if source is None else getattr(source, "snapshot", None)
        if (
            request.panel_id != self.panel_id
            or request.request_revision != self._fit_request_revision
            or snapshot is None
            or request.source.snapshot.ref != snapshot.ref
            or self._fit_active_spec != request.spec
        ):
            return False
        self._fit_pending_source_ref = None
        if error is not None:
            for pane in tuple(self._fit_panes):
                pane.set_busy(None, draft_ready=False)
            if not request.cancelled.is_set():
                self.set_status(f"Fit failed: {error}", error=True)
            return True
        from hashlib import sha256
        from zlc_data import FitResultBatch, encode_fit_result_batch

        if not isinstance(result, FitResultBatch) or result.source_ref != snapshot.ref:
            self.set_status("Fit worker returned another source revision", error=True)
            return True
        self._fit_result = result
        self._fit_result_identity = "draft-fit:" + sha256(
            encode_fit_result_batch(result)
        ).hexdigest()
        for pane in tuple(self._fit_panes):
            pane.set_busy(None, draft_ready=True)
        converged = sum(status.value == "CONVERGED" for status in result.statuses)
        self.set_status(
            f"fit {result.spec.model_id}: {converged}/{len(result.statuses)} converged",
            error=False,
        )
        self.figure_outputs_changed.emit()
        self._request_current_render()
        return True

    def clear_fit(self) -> None:
        """Remove the active request, overlay, and every fit.* output."""

        had_state = self._fit_active_spec is not None or self._fit_result is not None
        self._fit_request_revision += 1
        self._fit_pending_source_ref = None
        self.fit_cancel_requested.emit()
        self._fit_active_spec = None
        self._commit_persisted_params(remove=(_FIT_SPEC_PARAM,))
        self._clear_fit_result(notify=True)
        for pane in tuple(self._fit_panes):
            pane.set_busy(None, draft_ready=False)
        if had_state:
            self._request_current_render()
    # ------------------------------------------------------------- settings UI

    def _make_param_widget(self, spec: FormFieldProps, *, apply=None) -> QtWidgets.QWidget:
        """One widget per declarative form field with a semantic commit edge.

        ``apply`` overrides where the edit goes (default ``self._set_param``); the
        Edit tab passes its own callback.  Choice/toggle activation is already a
        complete user command.  Numeric spins disable keyboard tracking so an
        in-progress token remains local until Qt commits it.  Free text is a
        local draft until ``editingFinished``; no character can start a render.
        """

        cb = apply if apply is not None else self._set_param
        current = self.config.params.get(spec.key, spec.default)
        if spec.kind == "choice" and spec.key in self.config.params:
            current = choice_value_from_tree(spec, current)
        handler = FORM_WIDGET_HANDLERS[spec.kind]
        holder = {}

        def apply_current() -> None:
            widget = holder.get("widget")
            if widget is None:
                return
            try:
                value = handler.read(spec, widget)
            except (TypeError, ValueError) as exc:
                self.set_status(
                    f"Invalid {spec.label}: {exc}",
                    error=True,
                )
                return
            cb(spec.key, value)

        if spec.kind == "text":
            widget = handler.build(spec, current, lambda: None)
            holder["widget"] = widget
            widget.editingFinished.connect(apply_current)
            return widget
        widget = handler.build(spec, current, apply_current)
        holder["widget"] = widget
        if isinstance(widget, QtWidgets.QAbstractSpinBox):
            widget.setKeyboardTracking(False)
        return widget

    def _emit_param_rows(self, specs, add, apply, label_w, *, parent) -> dict:
        """Render each declarative field in ``specs`` as a ``[label | control]`` row through the
        same frontend form-handler path the measurement form uses, appending it via the
        ``add`` callback.  Returns ``{key: widget}`` so a caller can keep a named back-reference.  BOTH
        the Setting popup AND the Edit tab call this for a plot's display knobs, so adding a plot
        field shows up in both surfaces with no hand-wiring."""
        out = {}
        for spec in specs:
            widget = self._make_param_widget(spec, apply=apply)
            out[spec.key] = widget
            self._param_fields[spec.key] = spec
            add(
                FluentSettingRow(
                    spec.label,
                    widget,
                    label_width=label_w,
                    parent=parent,
                )
            )
        return out

    def _make_fixed_lim_row(self, apply_cb, label_w, *, parent):
        """The fixed lo/hi inputs as ONE bespoke ``[lo | hi]`` row (the single display knob kept
        special-cased rather than declarative -- it is not a scalar form field),
        shown only when relim == "fixed".  Built by this ONE helper so the Setting popup and the Edit tab
        get the identical row instead of two hand-copied blocks.  Returns ``(row, lo, hi)``."""
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
        row = FluentSettingRow(
            "lo / hi",
            host,
            label_width=label_w,
            parent=parent,
        )
        row.setVisible(self._relim() == "fixed")
        return row, lo, hi

    def _make_unit_readout_row(self, label_w, *, parent):
        """Build the shared read-only producer-unit row for Setting and Edit."""

        label = FluentLabel(self._current_unit_text())
        label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        label.setToolTip(
            "Declared by the producer schema. Display panels do not rewrite units."
        )
        row = FluentSettingRow(
            "unit",
            label,
            label_width=label_w,
            parent=parent,
        )
        return row, label

    def _relim(self) -> str:
        """The panel's relim mode, defaulting to ``_RELIM_PARAM.default`` -- the ONE place that
        default lives, so every reader agrees instead of re-typing the 'tight' literal."""
        return str(self.config.params.get("relim", _RELIM_PARAM.default))

    def refresh_on_show(self) -> None:
        """Re-seed every Setting control from ``config.params`` -- the SINGLE source of truth for a
        panel's params -- so the Setting popup shows the CURRENT values whenever it opens, even if they
        were changed elsewhere (the Edit tab writes the same config.params).  Each widget is re-seeded
        through its form handler's ``write`` (one entry point, no per-key handwiring), with its
        change signals blocked so re-seeding does not re-fire ``_set_param`` (which would enqueue a
        duplicate compose).  A control is a view of config.params, refreshed on show, never a
        private copy that drifts from the other surface."""
        for key, widget in self.form_widgets.items():
            field = self._param_fields.get(key)
            if field is None or key not in self.config.params:
                continue
            with _signals_blocked(widget):
                FORM_WIDGET_HANDLERS[field.kind].write(
                    field,
                    widget,
                    (
                        choice_value_from_tree(
                            field,
                            self.config.params[key],
                        )
                        if field.kind == "choice"
                        else self.config.params[key]
                    ),
                )
        self._refresh_repeat_mode_control()
        self._refresh_unit_readout()

    def _build_plot(self) -> None:
        """Give this card its raster surface.

        A panel shows PIXELS the worker produced -- an already-coherent
        BoardFrame handed to :class:`~zlc_frontend.qt_widgets.SinglePanelHost`,
        which paints from immutable bytes and owns no figure.  The card
        therefore holds no Matplotlib object at all: rendering happens off the
        GUI thread and arrives here already rasterised.  The host is the ONE
        selector owner (design rule: an interactive window uses the
        QtRasterBoard chain; FrozenRasterView stays a frozen-report presenter), so
        the console card's selector switch and completed gestures go through
        the same binding as every other one-panel window.
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
            self.board.rangeSelected.connect(self._accept_area_range)
            self.board.rectangleSelected.connect(
                self._accept_area_rectangle
            )
            self.board.crossSelected.connect(self._accept_cross)
        else:
            self.board = SinglePanelHost(
                self.panel_id, empty_text="waiting for data")
            # A panel card exposes the frontend owner's complete typed gesture
            # without deciding what that rectangle means.  ROI/control-domain
            # routing belongs to the TaskConsole composition layer; keeping
            # that meaning out of this view is what lets every image-like
            # consumer share the same selector.
            self.board.rangeSelected.connect(self._accept_area_range)
            self.board.rectangleSelected.connect(
                self._accept_area_rectangle
            )
            self.board.crossSelected.connect(self._accept_cross)
        self._apply_fixed_size()
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

        from zlc_frontend.selector import RectangleGesture

        if not isinstance(gesture, RectangleGesture):
            raise TypeError("gesture must be RectangleGesture")
        if gesture.panel_id != self.panel_id:
            raise ValueError("rectangle gesture belongs to another panel card")
        if self.board is None:
            raise RuntimeError(
                "rectangle selection requires this card's panel host"
            )
        return self.board.selection_for_rectangle_gesture(gesture)

    def _set_area_commit(self, commit) -> None:
        from zlc_frontend.figure_outputs import FigureAreaCommit

        if commit is not None and not isinstance(commit, FigureAreaCommit):
            raise TypeError("Area output requires FigureAreaCommit or None")
        changed = commit != self._area_commit
        self._area_commit = commit
        if changed:
            self.figure_outputs_changed.emit()
            self._fit_selection_changed()

    def _clear_figure_outputs(self, *, notify: bool) -> None:
        changed = self._area_commit is not None or self._cross_commit is not None
        self._area_commit = None
        self._cross_commit = None
        if changed and notify:
            self.figure_outputs_changed.emit()
        if changed:
            self._fit_selection_changed()

    def _accept_area_range(self, gesture, *, host=None) -> None:
        """Promote a completed curve span into the Figure's Area output."""

        from zlc_frontend.selector import (
            CurveRangeGesture,
            HistogramRangeGesture,
        )

        if not isinstance(
            gesture,
            (CurveRangeGesture, HistogramRangeGesture),
        ):
            return
        if gesture.x_span is None:
            self._set_area_commit(None)
            return
        board = self.board if host is None else host
        if board is None:
            return
        try:
            commit = board.area_commit_for_range_gesture(gesture)
        except RuntimeError:
            self._set_area_commit(None)
            return
        self._set_area_commit(commit)

    def _accept_area_rectangle(self, gesture, *, host=None) -> None:
        """Promote a completed image rectangle into the Figure's Area output."""

        from zlc_frontend.selector import RectangleGesture

        if not isinstance(gesture, RectangleGesture):
            return
        if gesture.normalized_bounds is None:
            self._set_area_commit(None)
            return
        board = self.board if host is None else host
        if board is None:
            return
        try:
            commit = board.area_commit_for_rectangle_gesture(gesture)
        except RuntimeError:
            # A stale origin is ineligible.  SiteMap rectangles resolve through
            # the painted background's named spatial axes here, then the Figure
            # output owner selects site centres from that exact joined view.
            # They never become a Measurement ROI.
            self._set_area_commit(None)
            return
        self._set_area_commit(commit)

    def _accept_cross(self, gesture, *, host=None) -> None:
        """Accept one completed Cross click; pointer motion never reaches here."""

        from zlc_frontend.figure_outputs import FigureCrossCommit
        from zlc_frontend.selector import CrossGesture

        if not isinstance(gesture, CrossGesture):
            return
        board = self.board if host is None else host
        if board is None or board.visible_interaction_origin() != gesture.origin:
            return
        if gesture.point is None:
            changed = self._cross_commit is not None
            self._cross_commit = None
            if changed:
                self.figure_outputs_changed.emit()
            return
        try:
            commit = board.cross_commit_for_gesture(gesture)
        except (RuntimeError, TypeError, ValueError):
            return
        if not isinstance(commit, FigureCrossCommit):
            raise TypeError("panel host returned another Cross contract")
        changed = commit != self._cross_commit
        self._cross_commit = commit
        if changed:
            self.figure_outputs_changed.emit()

    def accept_range_from(self, host, gesture) -> None:
        """Accept one selector gesture from another view of this exact front."""

        self._accept_area_range(gesture, host=host)

    def accept_rectangle_from(self, host, gesture) -> None:
        """Accept one rectangle from another host without duplicating mapping."""

        self._accept_area_rectangle(gesture, host=host)

    def accept_cross_from(self, host, gesture) -> None:
        """Accept a Cross click from another host of this exact front."""

        self._accept_cross(gesture, host=host)

    @staticmethod
    def _figure_commit_matches_value(commit, value) -> bool:
        if commit is None or value is None:
            return False
        snapshot = getattr(value, "snapshot", None)
        if snapshot is None:
            return False
        from zlc_frontend.figure_outputs import source_identity_matches_snapshot

        return source_identity_matches_snapshot(commit.source_identity, snapshot)

    def frozen_figure_output_state(self):
        """Return Figure intents against the exact immutable visible data value.

        The returned value is merely the already-owned data-plane value
        promoted in :meth:`present`; this method performs no acquisition and
        creates no GUI/data snapshot.
        """

        value = self._presented_value
        area_commit = (
            self._area_commit
            if self._figure_commit_matches_value(self._area_commit, value)
            else None
        )
        cross_commit = (
            self._cross_commit
            if self._figure_commit_matches_value(self._cross_commit, value)
            else None
        )
        fit_result = self._fit_result
        if (
            fit_result is not None
            and (
                value is None
                or fit_result.source_ref
                != getattr(getattr(value, "snapshot", None), "ref", None)
            )
        ):
            fit_result = None
        return value, area_commit, cross_commit, fit_result

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
        if self._pending_interaction_origin is not None:
            # A held pointer gesture edits the exact data front the operator
            # can still see.  A newer live-camera completion may already be in
            # ``_last_value`` while its raster is queued or while the held
            # front deliberately remains painted.  Advancing to that value
            # here would splice two input identities into one gesture.
            value = self._presented_value
        else:
            value = (
                self._candidate_value
                if self.config.kind == "grid"
                and self._candidate_value is not None
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
            from zlc_frontend.site_map_render import SiteMapView

            site_map_view = getattr(value, "presentation", None)
            if not isinstance(site_map_view, SiteMapView.__args__):
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
            self._reconcile_persisted_view_for_schema(schema)
            self._candidate_schema = schema
            self._grid_focus = None
            self._pending_faceted_result = None
            self._pending_figure = None
            self._pending_display = None
            self._pending_size_name = None
            self._pending_pixel_ratio = None
            self._pending_run_id = None
            self._pending_title = None
            self._pending_value_label = None
            self._pending_value = None
            self._refresh_grid_view_controls()
            self._refresh_repeat_mode_control()
        self._candidate_value = value
        view = (
            None
            if self.config.kind == "sites"
            else self._saved_view_spec(schema)
        )
        if self.config.kind == "grid" and view is None:
            view = self._default_grid_view(schema)
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
        rolling_distribution = (
            self.config.kind == "monitor"
            and bool(
                _resolved_panel_param(
                    "monitor",
                    self.config.params,
                    "show_dist",
                )
            )
        )
        source_key = (
            str(self.config.kind),
            str(value.name),
            str(value.source),
            str(self.config.title),
            size,
            pixel_ratio,
            view,
            rolling_distribution,
            (
                None
                if site_map_view is None
                else (
                    # Composer ownership is structural.  A live occupancy
                    # cell gets a new view_identity for every exact Camera
                    # revision; putting that value here recreated the Agg
                    # renderer on every shot and reset its BoardFrame sequence
                    # to one.  The per-shot identity already belongs to
                    # frame_key/signature/provenance below.  Presentation kind
                    # plus the admitted calibration identify the stable site
                    # geometry a SiteMapComposer may safely reuse.
                    site_map_view.presentation_kind,
                    site_map_view.calibration_identity,
                )
            ),
        )
        fit_result = self._fit_result
        fit_result_identity = self._fit_result_identity
        if (
            fit_result is None
            or fit_result.source_ref != value.snapshot.ref
        ):
            fit_result = None
            fit_result_identity = None
        focus = self._grid_focus if self.config.kind == "grid" else None
        signature = (
            frame_key,
            source_key,
            display,
            focus,
            fit_result_identity,
        )
        if not force and signature == self._requested_signature:
            return None
        self._render_request_revision += 1
        self._requested_signature = signature
        self._latest_requested_source_ref = value.snapshot.ref
        self._latest_requested_source_key = source_key
        self._latest_requested_display_revision = int(display.revision)
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
            fit_result,
            fit_result_identity,
            rolling_distribution=rolling_distribution,
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
        figure=None,
        error: str | None = None,
    ) -> bool:
        """Accept any useful completed front from the current source generation.

        Request revisions only order queued work; they are not presentation
        authority.  A completed immutable source revision remains a real front
        while a later same-generation request is waiting in the capacity-one
        lane, so show it unless an equal/newer front is already pending or
        painted.  Structural rebinds and generation replacement still reject
        a superseded result.
        """

        source_ref = request.value.snapshot.ref
        latest_ref = self._latest_requested_source_ref
        if (
            latest_ref is None
            or request.source_key != self._latest_requested_source_key
            or source_ref.block_id != latest_ref.block_id
            or source_ref.stream_generation != latest_ref.stream_generation
            or source_ref.schema_fingerprint != latest_ref.schema_fingerprint
        ):
            return False
        if error is not None and (
            source_ref.revision != latest_ref.revision
            or int(request.display.revision)
            != self._latest_requested_display_revision
        ):
            return False
        for value, display in (
            (self._pending_value, self._pending_display),
            (self._presented_value, self._presented_display),
        ):
            existing_ref = getattr(getattr(value, "snapshot", None), "ref", None)
            if (
                existing_ref is None
                or existing_ref.stream_generation != source_ref.stream_generation
            ):
                continue
            if existing_ref.revision > source_ref.revision:
                return False
            if (
                existing_ref.revision == source_ref.revision
                and display is not None
                and int(display.revision) >= int(request.display.revision)
            ):
                return False
        self._render_version = request.frame_key
        if error is not None:
            self._settle_pending_interaction_through(
                request.display.revision,
                failed=True,
            )
            self.set_status(error, error=True)
            return True
        if request.faceted:
            from zlc_frontend.panel_render import FacetedPanelResult

            if not isinstance(faceted_result, FacetedPanelResult):
                self._settle_pending_interaction_through(
                    request.display.revision,
                    failed=True,
                )
                self.set_status(
                    "render worker returned no complete faceted front",
                    error=True,
                )
                return True
            self._pending_faceted_result = faceted_result
            self._pending_frame = None
            pending_figure = faceted_result.figure
            if figure is not pending_figure:
                self._settle_pending_interaction_through(
                    request.display.revision,
                    failed=True,
                )
                self.set_status(
                    "faceted worker result lost its exact DataFigure",
                    error=True,
                )
                return True
            pending_display = request.display
            if faceted_result.focus is not None:
                from zlc_frontend.histogram_display import (
                    FacetedHistogramDisplayState,
                )

                if isinstance(pending_display, FacetedHistogramDisplayState):
                    pending_display = pending_display.display_for(
                        faceted_result.focus.selection
                    )
        elif frame is None or (
            self.config.kind != "sites" and figure is None
        ):
            self._settle_pending_interaction_through(
                request.display.revision,
                failed=True,
            )
            self.set_status("render worker returned no complete front", error=True)
            return True
        else:
            self._pending_frame = frame
            self._pending_faceted_result = None
            pending_figure = None
            if self.config.kind != "sites":
                from zlc_frontend import DataFigure

                if not isinstance(figure, DataFigure):
                    self._settle_pending_interaction_through(
                        request.display.revision,
                        failed=True,
                    )
                    self.set_status(
                        "render worker returned no exact DataFigure",
                        error=True,
                    )
                    return True
                pending_figure = figure
            pending_display = request.display
        self._last_value = request.value
        candidate_ref = getattr(
            getattr(self._candidate_value, "snapshot", None),
            "ref",
            None,
        )
        if (
            candidate_ref is None
            or candidate_ref.stream_generation != source_ref.stream_generation
            or candidate_ref.revision <= source_ref.revision
        ):
            self._candidate_value = request.value
        self._last_document = (
            None if pending_figure is None else pending_figure.document
        )
        self._last_display = request.display
        self._pending_figure = pending_figure
        self._pending_display = pending_display
        self._pending_size_name = str(request.size_name)
        self._pending_pixel_ratio = float(request.pixel_ratio)
        self._pending_run_id = getattr(request.value, "run_id", None)
        self._pending_title = str(request.label)
        self._pending_value_label = str(request.value_label)
        self._pending_value = request.value
        self._candidate_schema = request.value.snapshot.block.schema
        self._refresh_grid_view_controls()
        self._refresh_repeat_mode_control()
        self.set_status("ok", error=False)
        return True

    def _settle_pending_interaction_through(
        self,
        presentation_revision: int,
        *,
        failed: bool,
    ) -> None:
        """Settle only the display intent reached by this worker answer.

        Live data and intermediate viewport answers may be presented while a
        newer pointer motion is already queued.  They remain valid fronts, but
        cannot release the board's newest pending interaction.
        """

        pending_revision = self._pending_interaction_revision
        if (
            pending_revision is None
            or int(presentation_revision) < pending_revision
        ):
            return
        origin = self._pending_interaction_origin
        host = self._pending_interaction_host
        if failed and origin is not None and host is not None:
            host.discard_pending_interaction(origin)
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        self._pending_interaction_revision = None

    def _continues_pending_interaction(self, host, origin) -> bool:
        """Whether ``origin`` is a newer front of this host's same gesture.

        An intermediate worker answer advances the held front's sequence and
        presentation revision.  Exact origin equality would therefore reject
        the next motion of the same drag.  Host identity plus monotonic exact
        presentation lineage admits that advance while preventing the Edit
        tab's second host from taking over another host's pending command.
        """

        pending = self._pending_interaction_origin
        if pending is None or host is not self._pending_interaction_host:
            return False
        return (
            origin.panel_id == pending.panel_id
            and origin.board_id == pending.board_id
            and origin.layout_generation == pending.layout_generation
            and origin.source_identity == pending.source_identity
            and origin.input_identity == pending.input_identity
            and origin.sequence >= pending.sequence
            and origin.presentation.panel_id
            == pending.presentation.panel_id
            and origin.presentation.document_id
            == pending.presentation.document_id
            and origin.presentation.document_revision
            >= pending.presentation.document_revision
            and origin.presentation.selection_revision
            == pending.presentation.selection_revision
            and origin.presentation.panel_revision
            >= pending.presentation.panel_revision
        )

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
            count_scale = _resolved_panel_param(
                param_kind, params, "count_scale"
            )
            fit_mode = _resolved_panel_param(param_kind, params, "fit_mode")
            if not isinstance(count_scale, HistogramCountScale):
                raise TypeError("histogram count_scale lost its typed form value")
            if not isinstance(fit_mode, HistogramFitMode):
                raise TypeError("histogram fit_mode lost its typed form value")
            display = HistogramDisplayState(
                revision=self._display_revision, relim_mode=mode,
                count_scale=count_scale,
                bin_count=int(
                    _resolved_panel_param(
                        param_kind,
                        params,
                        "bin_count",
                    )
                ),
                fit_mode=fit_mode,
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
        colormap = _resolved_panel_param(
            image_param_kind, params, "colormap"
        )
        if not isinstance(colormap, ImageColormap):
            raise TypeError("image colormap lost its typed form value")
        return ImageDisplayState(
            revision=self._display_revision,
            relim_mode=mode,
            colormap=colormap,
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
        figure = self._presented_figure
        if figure is None:
            raise RuntimeError("the panel has no presented typed figure to save")

        # Grid overview is an encoded image whose exact DataFigure is promoted
        # in the same Qt present transaction.  Focused and ordinary panels also
        # carry an EvaluatedInput, so verify that visible provenance before
        # exposing their figure to Fit or Save.
        board = self.board
        if board is not None and getattr(board, "front_frame", None) is None:
            if not bool(getattr(board, "showing_overview", False)):
                raise RuntimeError("front replacement in progress")
            return figure
        visible_input = self._visible_evaluated_input()
        entries = tuple(figure.datasets.entries)
        if (
            len(entries) != 1
            or entries[0].dataset_id != visible_input.dataset_id
            or entries[0].snapshot.ref != visible_input.ref
        ):
            raise RuntimeError("front replacement in progress")
        return figure

    def frozen_display_state(self):
        """Return the display state promoted with the visible immutable front."""

        if self._presented_display is None:
            raise RuntimeError("the panel has no presented display state")
        return self._presented_display

    def frozen_panel_geometry(self) -> tuple[str, float]:
        """Return the size preset and DPR promoted with the visible front."""

        size_name = self._presented_size_name
        pixel_ratio = self._presented_pixel_ratio
        if size_name is None or pixel_ratio is None:
            raise RuntimeError("the panel has no presented plot geometry")
        return str(size_name), float(pixel_ratio)

    def frozen_panel_labels(self) -> tuple[str, str]:
        """Return the exact title and value label painted on the visible front."""

        title = self._presented_title
        value_label = self._presented_value_label
        if title is None or value_label is None:
            raise RuntimeError("the panel has no presented plot labels")
        return str(title), str(value_label)

    def frozen_render_payload(self):
        """Return the exact typed payload painted by the current focused front."""

        board = self.board
        front = None if board is None else board.front_frame
        if front is None:
            return None
        if len(front.panels) != 1:
            raise RuntimeError("front replacement in progress")
        return front.panels[0].display_payload

    def visible_run_id(self):
        """Return the producer RunId promoted with the visible immutable front.

        A FINAL result and its projected dataset become available before the
        worker necessarily presents that revision.  Fit must not use the
        new artifact while the operator is still looking at the previous
        front.  The value therefore crosses the same Qt present boundary as
        its figure/display/geometry rather than being read from the latest
        worker completion.
        """

        return self._presented_run_id

    def _visible_evaluated_input(self):
        """Read the exact typed input painted by the one current board front."""

        board = self.board
        front = None if board is None else board.front_frame
        if front is None or len(front.panels) != 1:
            raise RuntimeError("front replacement in progress")
        visible_input = getattr(
            front.panels[0].display_payload,
            "evaluated_input",
            None,
        )
        if visible_input is None:
            raise RuntimeError("front replacement in progress")
        return visible_input

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
        pending_figure = self._pending_figure
        pending_display = self._pending_display
        pending_size_name = self._pending_size_name
        pending_pixel_ratio = self._pending_pixel_ratio
        pending_run_id = self._pending_run_id
        pending_title = self._pending_title
        pending_value_label = self._pending_value_label
        pending_value = self._pending_value
        self._pending_figure = None
        self._pending_display = None
        self._pending_size_name = None
        self._pending_pixel_ratio = None
        self._pending_run_id = None
        self._pending_title = None
        self._pending_value_label = None
        self._pending_value = None
        geometry_changes = (
            self._presented_size_name is not None
            and pending_size_name is not None
            and pending_size_name != self._presented_size_name
        )
        if geometry_changes:
            self.setUpdatesEnabled(False)
        try:
            self._build_plot()
            logical_size = tuple(
                int(value)
                for value in panel_display_size(pending_size_name)
            )
            if faceted is not None:
                if faceted.overview_png is not None:
                    self.board.present_overview(
                        faceted.overview_png,
                        faceted.regions,
                        logical_size=logical_size,
                    )
                else:
                    self.board.present_frame(
                        faceted.frame,
                        logical_size=logical_size,
                    )
            else:
                self.board.present_frame(frame, logical_size=logical_size)
            self._presented_figure = pending_figure
            self._presented_display = pending_display
            self._presented_size_name = pending_size_name
            self._presented_pixel_ratio = pending_pixel_ratio
            self._presented_run_id = pending_run_id
            self._presented_title = pending_title
            self._presented_value_label = pending_value_label
            self._presented_value = pending_value
            self._refresh_unit_readout()
            self._settle_pending_interaction_through(
                pending_display.revision,
                failed=False,
            )
            if geometry_changes:
                # Promote raster and geometry as one visible fact.  Until this
                # exact-size answer arrived, the accepted raster stayed at its authored
                # authored extent and was never stretched into the new size.
                self._apply_fixed_size(
                    pending_size_name,
                    sync_board=False,
                )
                if (
                    getattr(self, "settings_popup", None) is not None
                    and self.settings_popup.isVisible()
                ):
                    self._size_settings_popup()
                self.layout_changed.emit()
        finally:
            if geometry_changes:
                self.setUpdatesEnabled(True)
        self._sync_fit_authoring_from_presented(
            prepare_authoring=any(pane.isVisible() for pane in self._fit_panes)
        )
        self.front_presented.emit()
        if (
            self._area_commit is not None
            or self._cross_commit is not None
            or self._fit_result is not None
        ):
            self.figure_outputs_changed.emit()

    def setting_label_width(self, _metrics) -> int:
        """One label column for Setting and Edit, independent of live text."""

        labels = {
            "signal",
            "size",
            "sub plot",
            "facet",
            "repeat",
            "lo / hi",
            "unit",
            "update",
            "title",
            "x range",
            "y range",
            "colour range",
            "path",
            "name",
            "file",
        }
        labels.update(
            spec.label for spec in _panel_param_decls(self.config.kind)
        )
        if self.config.kind == "grid":
            labels.update(
                spec.label
                for kind in ("hist", "2d")
                for spec in _panel_param_decls(kind)
            )
        labels.add(_RELIM_PARAM.label)
        # Delegate the actual typography/scale rule to the component library;
        # TaskConsole contributes only the product's field inventory.
        return _setting_label_width(labels, minimum=80)

    def _build_settings(self) -> None:
        """Build the main-UI flat Setting surface over current typed state."""

        popup = FluentPopup(self)
        outer = QtWidgets.QVBoxLayout(popup)
        outer.setContentsMargins(0, 0, 0, 0)
        self._settings_scroll = FluentScrollArea()
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
        self._settings_scroll.set_width_bounded_widget(content)
        outer.addWidget(self._settings_scroll)
        self.settings_popup = popup
        self._settings_h_hwm = 0

        display_specs = list(_panel_param_decls(self.config.kind)) + [_RELIM_PARAM]
        label_w = self.setting_label_width(self.fontMetrics())
        # The Setting surface owns one stable width.  Descendant combo/path/error
        # size hints are content, not a request to widen the outer popup/window.
        popup.setFixedWidth(
            label_w
            + scaled_px(360, minimum=320)
            + 2 * pad
            + fluent_scrollbar_thickness()
            + scaled_px(4)
        )

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
        self.form_widgets = self._emit_param_rows(
            display_specs,
            display.addWidget,
            self._set_param,
            label_w,
            parent=content,
        )
        self._mount_grid_row_inventory(
            self,
            display.addWidget,
            view_apply=self._commit_grid_facet,
            param_apply=self._set_param,
            label_w=label_w,
            form_widgets=self.form_widgets,
            parent=content,
        )
        self.repeat_mode_row, self.repeat_mode_combo = self._make_repeat_mode_row(
            self._commit_repeat_mode,
            label_w,
            parent=content,
        )
        display.addWidget(self.repeat_mode_row)
        self.fixed_lim_row, self.fixed_lo_edit, self.fixed_hi_edit = self._make_fixed_lim_row(
            self._on_fixed_lim_edited,
            label_w,
            parent=content,
        )
        display.addWidget(self.fixed_lim_row)
        self.fixed_lim_row.setVisible(self._relim() == "fixed")
        unit_row, self.unit_label = self._make_unit_readout_row(
            label_w,
            parent=content,
        )
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

        # ---- Fit: one Figure-owned editor.  The Edit tab builds a
        # second view through the same factory; both reconcile against the
        # card's one request/result state and neither opens a DataFigure window.
        self.fit_authoring_pane = None
        if self._fit_capable_kind():
            fit_section = section("Fit")
            self.fit_authoring_pane = self.make_fit_authoring_pane(
                popup,
                label_width=label_w,
            )
            fit_section.addWidget(self.fit_authoring_pane)

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
        if self._fit_options_identity is None:
            self._sync_fit_authoring_from_presented(prepare_authoring=True)

    def _present_settings_popup(self) -> None:
        popup = self.settings_popup
        anchor = self.setting_button.mapToGlobal(
            QtCore.QPoint(self.setting_button.width(), self.setting_button.height()))
        self._size_settings_popup()                        # height: show-all, grow-not-shrink
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
        """Populate one slot combobox with ``(none)`` + every declared signal grouped by its
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
        """``{stable signal key: SHORT name}`` from the
        ``short_names_provider``, so the picker nest leaf is the short name (``frame`` / ``survival`` /
        ``rate``) -- NOT the full prefixed key and NOT the verbose SignalSpec axis label (``camera
        image``).  The picker nest already names the producer node, so the leaf is the short NAME."""
        return coerce_short_labels(self.short_names_provider)

    def _refresh_signal_combo(self) -> None:
        """Refresh the one dataset picker without rebuilding the popup."""

        combo = getattr(self, "signal_combo", None)
        if combo is not None:
            self._fill_slot_combo(combo, self.config.signal)

    def reconcile_visible_signal_metadata(self) -> bool:
        """Refresh one open Setting picker's leaf metadata in place.

        Signal topology is owned by the console's explicit add/remove/start/stop
        paths.  A newly published value or a schema change is not topology, so it
        may only change the existing leaves' shape/readiness text.
        """

        popup = getattr(self, "settings_popup", None)
        combo = getattr(self, "signal_combo", None)
        if (
            popup is None
            or combo is None
            or not popup.isVisible()
        ):
            return False
        groups = _qt_widgets.signal_tree_groups(
            self.names_provider() if callable(self.names_provider) else [],
            self.sources_provider() if callable(self.sources_provider) else {},
            self.formats_provider() if callable(self.formats_provider) else {},
            self._signal_short_names_map(),
        )
        return combo.reconcile_signal_tree_metadata(groups)

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

    def _resolved_grid_view(self, intent, facet_axis_id, *, repeat_mode=None):
        """Supply card state to the frontend's pure one-facet resolver."""

        schema = self._current_schema()
        if schema is None:
            return None
        from zlc_frontend.figure import resolve_grid_view

        saved = self._saved_view_spec(schema)
        suggestion = resolve_grid_view(
            schema,
            intent,
            facet_axis_id,
            current_view=saved,
            repeat_mode=repeat_mode,
        )
        return suggestion.spec

    def _default_grid_view(self, schema):
        """Resolve the stable schema-derived default without authoring state."""

        from zlc_frontend.figure import suggest_default_grid_view

        suggestion = suggest_default_grid_view(schema)
        return suggestion.spec

    def _grid_facet_choices(self, intent):
        from zlc_frontend.figure import grid_facet_axes

        schema = self._current_schema()
        if schema is None:
            return ()
        saved = self._saved_view_spec(schema)
        return grid_facet_axes(
            schema,
            intent,
            current_view=saved,
        )

    def _make_grid_view_rows(self, label_w, *, apply, parent):
        intent_combo = FluentComboBox()
        intent_combo.setToolTip(
            "What each Grid cell draws. The selected family is resolved "
            "against the dataset's declared axes."
        )
        facet_combo = FluentComboBox()
        facet_combo.setToolTip(
            "Expand exactly one declared dataset axis into Grid cells."
        )
        intent_row = FluentSettingRow(
            "sub plot",
            intent_combo,
            label_width=label_w,
            parent=parent,
        )
        facet_row = FluentSettingRow(
            "facet",
            facet_combo,
            label_width=label_w,
            parent=parent,
        )
        def commit_facet(index: int) -> None:
            intent = intent_combo.currentData()
            axis_id = facet_combo.itemData(int(index))
            if intent is not None and axis_id is not None:
                apply(intent, axis_id)

        def commit_intent(_index: int) -> None:
            # Cell intent and facet are independent controls.  Changing what a
            # cell draws starts from the already-authored AxisId, not from the
            # transient combo contents.  The persisted typed ViewSpec is the
            # sole state owner; the combo is only its projection.
            intent = intent_combo.currentData()
            if intent is None:
                return
            schema = self._current_schema()
            current = self._saved_view_spec(schema) if schema is not None else None
            axis_id = None
            if current is not None:
                from zlc_frontend.figure import grid_facet_axis

                axis_id = grid_facet_axis(current)
            legal = {axis.axis_id for axis in self._grid_facet_choices(intent)}
            if axis_id in legal:
                apply(intent, axis_id)
                return
            # No authored facet exists (or it is not legal for this intent).
            # Present the legal named axes and wait for an explicit choice.
            self._seed_grid_facet_combo(intent_combo, facet_combo)

        intent_combo.currentIndexChanged.connect(commit_intent)
        facet_combo.currentIndexChanged.connect(commit_facet)
        visible = self.config.kind == "grid"
        intent_row.setVisible(visible)
        facet_row.setVisible(visible)
        return intent_row, intent_combo, facet_row, facet_combo

    def _grid_row_inventory(self, *, view_apply, param_apply, label_w, parent):
        """Build the one ordered Grid field inventory used by both surfaces."""

        (
            intent_row,
            intent_combo,
            facet_row,
            facet_combo,
        ) = self._make_grid_view_rows(
            label_w,
            apply=view_apply,
            parent=parent,
        )
        bins_row, bins_widget = self._make_grid_bins_row(
            param_apply,
            label_w,
            parent=parent,
        )
        count_scale_row, count_scale_widget = self._make_grid_hist_param_row(
            "count_scale",
            param_apply,
            label_w,
            parent=parent,
        )
        colormap_row, colormap_widget = self._make_grid_cell_param_row(
            "2d",
            "colormap",
            self._image_intent(),
            param_apply,
            label_w,
            parent=parent,
        )
        # Attribute names are explicit here so neither host can reorder or
        # silently omit a field while still claiming to share the same UI.
        return (
            ("grid_facet_row", "grid_facet_combo", None, facet_row, facet_combo),
            ("grid_intent_row", "grid_intent_combo", None, intent_row, intent_combo),
            (
                "grid_bin_count_row",
                "grid_bin_count_widget",
                "bin_count",
                bins_row,
                bins_widget,
            ),
            (
                "grid_count_scale_row",
                "grid_count_scale_widget",
                "count_scale",
                count_scale_row,
                count_scale_widget,
            ),
            (
                "grid_colormap_row",
                "grid_colormap_widget",
                "colormap",
                colormap_row,
                colormap_widget,
            ),
        )

    def _mount_grid_row_inventory(
        self,
        owner,
        add,
        *,
        view_apply,
        param_apply,
        label_w,
        form_widgets,
        parent,
    ) -> None:
        """Mount the shared Grid rows without a second Setting/Edit schema."""

        if self.config.kind != "grid":
            return
        for row_name, widget_name, param_key, row, widget in (
            self._grid_row_inventory(
                view_apply=view_apply,
                param_apply=param_apply,
                label_w=label_w,
                parent=parent,
            )
        ):
            setattr(owner, row_name, row)
            setattr(owner, widget_name, widget)
            if param_key is not None and widget is not None:
                form_widgets[param_key] = widget
            add(row)

    def _make_grid_hist_param_row(self, key: str, apply, label_w, *, parent):
        """Build one histogram-only Grid knob through the shared param owner."""

        return self._make_grid_cell_param_row(
            "hist",
            key,
            self._histogram_intent(),
            apply,
            label_w,
            parent=parent,
        )

    def _make_grid_cell_param_row(
        self,
        param_kind: str,
        key: str,
        intent,
        apply,
        label_w,
        *,
        parent,
    ):
        """Build one cell-family Grid knob from its ordinary panel declaration."""

        declaration = next(
            item
            for item in _panel_param_decls(str(param_kind))
            if item.key == str(key)
        )
        widget = self._make_param_widget(declaration, apply=apply)
        self._param_fields[declaration.key] = declaration
        row = FluentSettingRow(
            declaration.label,
            widget,
            label_width=label_w,
            parent=parent,
        )
        row.setVisible(self._grid_cell_intent() is intent)
        return row, widget

    def _make_grid_bins_row(self, apply, label_w, *, parent):
        return self._make_grid_hist_param_row(
            "bin_count",
            apply,
            label_w,
            parent=parent,
        )

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
        preferred = None
        if current is not None:
            from zlc_frontend.figure import grid_facet_axis

            preferred = grid_facet_axis(current)
        with _signals_blocked(facet_combo):
            facet_combo.clear()
            choices = self._grid_facet_choices(intent)
            duplicate_names = {
                axis.name
                for axis in choices
                if sum(item.name == axis.name for item in choices) > 1
            }
            for axis in choices:
                axis_size = f" ({axis.size})"
                label = (
                    f"{axis.name}{axis_size} [{axis.axis_id.value}]"
                    if axis.name in duplicate_names
                    else f"{axis.name}{axis_size}"
                )
                facet_combo.addItem(label, axis.axis_id)
            # ``AxisId`` is a Python value object.  Qt's QVariant-backed
            # ``findData`` may compare two equal instances by identity after a
            # serialize/decode round trip, so use the type's equality contract
            # explicitly.
            index = next(
                (
                    item_index
                    for item_index in range(facet_combo.count())
                    if facet_combo.itemData(item_index) == preferred
                ),
                -1,
            )
            facet_combo.setCurrentIndex(index)
        facet_combo.setEnabled(facet_combo.count() > 0)

    def _refresh_grid_control_surface(self, owner) -> None:
        """Reconcile one Setting/Edit projection from the sole stored ViewSpec."""

        if self.config.kind != "grid":
            return
        intent_combo = getattr(owner, "grid_intent_combo", None)
        facet_combo = getattr(owner, "grid_facet_combo", None)
        if intent_combo is None or facet_combo is None:
            return
        self._seed_grid_view_controls(intent_combo, facet_combo)
        intent = self._grid_cell_intent()
        visibility_changed = False
        for row_name in ("grid_bin_count_row", "grid_count_scale_row"):
            row = getattr(owner, row_name, None)
            if row is not None:
                visible = intent is self._histogram_intent()
                if row.isHidden() == visible:
                    row.setVisible(visible)
                    visibility_changed = True
        image_row = getattr(owner, "grid_colormap_row", None)
        if image_row is not None:
            visible = intent is self._image_intent()
            if image_row.isHidden() == visible:
                image_row.setVisible(visible)
                visibility_changed = True
        if visibility_changed:
            owner.updateGeometry()
            # Setting is a top-level popup whose height was fixed when it was
            # presented; Edit is ordinary scroll content and reflows itself.
            # Re-measure the former exactly once after the shared inventory has
            # reconciled, rather than letting each row callback resize chrome.
            popup = getattr(self, "settings_popup", None)
            if owner is self and popup is not None and popup.isVisible():
                self._size_settings_popup()

    def _refresh_grid_view_controls(self) -> None:
        self._refresh_grid_control_surface(self)

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

        candidate = self._resolved_grid_view(intent, axis_id)
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
        """Purely read the current owner-coded presentation value, if valid."""

        if self.config.kind == "sites":
            return None
        view = _decode_panel_view(self.config)
        if view is None:
            return None
        if view.schema_fingerprint != schema.fingerprint:
            # Reconciliation belongs to the explicit schema-change boundary,
            # never to this read.  A stale value is simply not readable here.
            return None
        return view

    def _reconcile_persisted_view_for_schema(self, schema) -> None:
        """Retire prior-generation view state at the schema commit boundary."""

        if self.config.kind == "sites":
            return
        view = _decode_panel_view(self.config)
        if view is None or view.schema_fingerprint == schema.fingerprint:
            return
        self._commit_persisted_params(
            remove=(
                _VIEW_SPEC_PARAM,
                _HISTOGRAM_CELL_THRESHOLDS_PARAM,
                _HISTOGRAM_THRESHOLDS_PARAM,
            )
        )

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
        if self.config.kind != "grid":
            return tuple(
                mode
                for mode in contract.repeat_modes
                if mode is not RepeatViewMode.FACET
            )
        view = self._saved_view_spec(schema)
        if view is None:
            return tuple(
                mode
                for mode in contract.repeat_modes
                if mode is not RepeatViewMode.FACET
            )
        from zlc_frontend.figure import grid_facet_axis

        return (
            (RepeatViewMode.FACET,)
            if grid_facet_axis(view) == schema.repeat_axis.axis_id
            else tuple(
                mode
                for mode in contract.repeat_modes
                if mode is not RepeatViewMode.FACET
            )
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
        combo.setEnabled(len(modes) > 1)

    def _make_repeat_mode_row(self, apply, label_w, *, parent):
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
        row = FluentSettingRow(
            "repeat",
            combo,
            label_width=label_w,
            parent=parent,
        )
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
            from zlc_frontend.figure import grid_facet_axis, resolve_grid_view

            suggestion = resolve_grid_view(
                schema,
                intent,
                grid_facet_axis(saved),
                current_view=saved,
                repeat_mode=mode,
            )
        elif saved is not None:
            preferences = self._preferences_from_view(
                saved,
                schema,
                repeat_mode=mode,
            )
        else:
            preferences = ViewPreferences(repeat_mode=mode)
        if self.config.kind != "grid":
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
        self._commit_persisted_params(
            remove=(
                _VIEW_SPEC_PARAM,
                _HISTOGRAM_CELL_THRESHOLDS_PARAM,
                _HISTOGRAM_THRESHOLDS_PARAM,
            ),
            emit_changed=False,
        )
        self._last_value = None
        self._candidate_value = None
        self._last_document = None
        self._clear_figure_outputs(notify=True)
        self._candidate_schema = None
        self._grid_focus = None
        self._pending_faceted_result = None
        self._refresh_repeat_mode_control()
        self._invalidate_render_binding()
        self._render_version = -1
        self._request_display_render()
        self.changed.emit()

    def _current_unit_text(self) -> str:
        """The visible signal's producer-declared unit."""

        value = self._presented_value or self._last_value
        if value is None:
            return "—"
        return value.unit or "dimensionless"

    def _refresh_unit_readout(self) -> None:
        label = getattr(self, "unit_label", None)
        if label is not None:
            label.setText(self._current_unit_text())

    def _on_view_committed(self, commit) -> None:
        self.accept_view_commit_from(self.board, commit)

    def accept_view_commit_from(self, host, commit) -> None:
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
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
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
        pending_origin = self._pending_interaction_origin
        if pending_origin is None:
            if self._display_revision != commit.origin.presentation.panel_revision:
                host.discard_pending_interaction(commit.origin)
                return
        elif not self._continues_pending_interaction(host, commit.origin):
            host.discard_pending_interaction(commit.origin)
            return
        if revision <= self._display_revision:
            host.discard_pending_interaction(commit.origin)
            return
        self._view_pin = pin
        self._display_revision = revision
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = revision
        # The commit changes only the card-owned display state.  The worker
        # answers it from the already accepted immutable data revision; Qt
        # never composes or waits for that answer.
        self._request_current_render()

    def _on_color_limits_committed(self, commit) -> None:
        self.accept_color_limits_from(self.board, commit)

    def accept_color_limits_from(self, host, commit) -> None:
        """CAS one clim-rail commit into the shared fixed-limits fact."""

        from zlc_frontend.selector import ImageColorLimitsCommit

        if not isinstance(commit, ImageColorLimitsCommit):
            raise TypeError("color-limit commit must retain its typed exact origin")
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if (
            (
                self._pending_interaction_origin is None
                and self._display_revision
                != commit.origin.presentation.panel_revision
            )
            or (
                self._pending_interaction_origin is not None
                and not self._continues_pending_interaction(
                    host,
                    commit.origin,
                )
            )
        ):
            host.discard_pending_interaction(commit.origin)
            return

        old_revision = self._display_revision
        lo, hi = (float(value) for value in commit.color_limits)
        self._store_fixed_lims(lo, hi)
        self._display_revision = old_revision + 1
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = self._display_revision
        self._request_current_render()

    def _on_histogram_thresholds_committed(self, commit) -> None:
        self.accept_thresholds_from(self.board, commit)

    def accept_thresholds_from(self, host, commit) -> None:
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
        if host is None:
            return
        if commit.origin != host.visible_interaction_origin():
            host.discard_pending_interaction(commit.origin)
            return
        if (
            (
                self._pending_interaction_origin is None
                and self._display_revision
                != commit.origin.presentation.panel_revision
            )
            or (
                self._pending_interaction_origin is not None
                and not self._continues_pending_interaction(
                    host,
                    commit.origin,
                )
            )
        ):
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
            self._commit_persisted_params(
                {
                    _HISTOGRAM_CELL_THRESHOLDS_PARAM:
                        histogram_cell_thresholds_to_tree(
                            candidate.cell_thresholds
                        )
                }
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
            self._commit_persisted_params(
                {
                    _HISTOGRAM_THRESHOLDS_PARAM: list(candidate.thresholds)
                }
            )
            self._display_revision = candidate.revision
        self._pending_interaction_origin = commit.origin
        self._pending_interaction_host = host
        self._pending_interaction_revision = self._display_revision
        self._request_current_render()

    def _store_fixed_lims(self, lo: float, hi: float) -> None:
        """The sole low-level writer for persisted fixed colour/count limits."""

        self._commit_persisted_params(
            {
                "relim": "fixed",
                "fixed_lo": float(lo),
                "fixed_hi": float(hi),
            }
        )

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
        """Commit the Setting popup's fixed lo/hi through the one apply path."""
        self.apply_fixed_lims(_safe_float(self.fixed_lo_edit.text(), 0.0),
                              _safe_float(self.fixed_hi_edit.text(), 1.0))

    def _on_update_interval(self, _idx: int) -> None:
        """Persist THIS panel's display refresh interval (``config.params["update_ms"]``,
        one of :data:`UPDATE_INTERVALS`) and ask the console to re-base its timer so the new
        rate co-aligns with the others.  No plot rebuild -- only the refresh cadence changes."""
        ms = int(self.update_combo.currentData() or DEFAULT_UPDATE_MS)
        if ms == int(self.config.params.get("update_ms", DEFAULT_UPDATE_MS) or DEFAULT_UPDATE_MS):
            return
        self._commit_persisted_params({"update_ms": ms})
        self.update_interval_changed.emit()    # console re-bases the shared timer

    def _refresh_title(self) -> None:
        """Compose the grey frame TITLE: the panel KIND + WHERE its signal comes from (the
        legend the console computes), e.g. ``1D vector — value ← Fit``.  This is
        the ordinary QGroupBox title -- the grey chip, alignment and font are the frame's own."""
        head = PANEL_KINDS[self.config.kind]
        info = " · ".join(p for p in self._signal_info.splitlines() if p.strip())
        self.setTitle(f"{head} — {info}" if info else head)

    def set_signal_info(self, info: str) -> None:
        """Set the signal legend (computed by the console: which node each read comes from).
        Shown in the frame title (the grey strip)."""
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
        self._invalidate_render_binding()
        if self._presented_size_name is None:
            # With no painted front there is nothing to stretch.  Keep initial
            # empty-card geometry responsive while waiting for first data.
            self._apply_fixed_size()
            if (
                getattr(self, "settings_popup", None) is not None
                and self.settings_popup.isVisible()
            ):
                self._size_settings_popup()
            self.layout_changed.emit()
        # Once a front exists, geometry is presented only with the matching
        # worker raster in ``present``.  The accepted surface therefore remains
        # exactly unchanged while this latest-only request is in flight.
        self._request_display_render()
        self.changed.emit()

    def _set_param(self, key: str, value) -> bool:
        return self._set_params({str(key): value})

    def _commit_persisted_params(
        self,
        updates: Mapping[str, object] | None = None,
        *,
        remove=(),
        request_render: bool = False,
        emit_changed: bool = True,
    ) -> frozenset[str]:
        """The only writer for persisted panel parameters.

        Callers supply one complete semantic transaction.  This method owns
        choice encoding, removals, equality checks, and the two observable
        effects of a persisted edit (optional render request and dirty signal).
        Pure readers must never call it.
        """

        if not isinstance(request_render, bool) or not isinstance(emit_changed, bool):
            raise TypeError("persisted parameter effects must be bool")
        incoming = {} if updates is None else dict(updates)
        removals = tuple(str(key) for key in remove)
        overlap = set(incoming).intersection(removals)
        if overlap:
            raise ValueError(
                "persisted parameter transaction both updates and removes "
                f"{tuple(sorted(overlap))}"
            )
        missing = object()
        encoded = {}
        for raw_key, value in incoming.items():
            key = str(raw_key)
            field = self._param_fields.get(key)
            encoded[key] = (
                choice_value_to_tree(field, value)
                if field is not None and field.kind == "choice"
                else value
            )
        changed = {
            key for key, value in encoded.items()
            if self.config.params.get(key, missing) != value
        }
        changed.update(key for key in removals if key in self.config.params)
        if not changed:
            return frozenset()
        for key in removals:
            self.config.params.pop(key, None)
        self.config.params.update(encoded)
        if request_render:
            self._request_display_render()
        if emit_changed:
            self.changed.emit()
        return frozenset(changed)

    def _set_params(self, updates: Mapping[str, object]) -> bool:
        """Commit one semantic parameter transaction and request one compose.

        Every surface that edits this panel (the Setting popup and the Edit tab)
        comes through here, so a display knob has one home
        -- ``config.params``, which is also what the saved layout persists.  The
        composer reads them back through :meth:`_display_state`, so what is
        stored and what is drawn cannot drift.
        """

        effective = {str(key): value for key, value in dict(updates).items()}
        if (
            str(effective.get("relim", "")) == "fixed"
            and not {"fixed_lo", "fixed_hi"}.intersection(effective)
        ):
            # Flipping INTO ``fixed`` FREEZES the window currently on screen.
            # Seeding from the composed front's own limits (never the default
            # 0..1 pair) is what keeps the picture: pinning a counts histogram
            # or a camera frame to 0..1 empties it, which reads as "the panel
            # just died".  The operator then types exact bounds.
            shown = self._shown_limits()
            if shown is not None:
                lo, hi = shown
                effective["fixed_lo"], effective["fixed_hi"] = lo, hi
        changed = self._commit_persisted_params(
            effective,
            request_render=True,
        )
        if not changed:
            return False
        if "relim" in changed:
            value = self.config.params["relim"]
            if str(value) == "fixed":
                for edit, key in (
                    (getattr(self, "fixed_lo_edit", None), "fixed_lo"),
                    (getattr(self, "fixed_hi_edit", None), "fixed_hi"),
                ):
                    if edit is not None and key in self.config.params:
                        edit.setText(
                            f"{float(self.config.params[key]):g}"
                        )  # setText does not re-fire editingFinished
            if getattr(self, "fixed_lim_row", None) is not None:        # the Setting popup's lo/hi row
                self.fixed_lim_row.setVisible(str(value) == "fixed")    # (the Edit tab toggles its own)
                # Revealing/hiding the row changes the popup's content height, and the popup is a
                # free-floating Qt.Popup whose window size was fixed at open time -- re-run the ONE
                # sizing rule so the window grows DOWNWARD to fit instead of reflowing inside a
                # fixed frame (which reads as a jump).
                if getattr(self, "settings_popup", None) is not None and self.settings_popup.isVisible():
                    self._size_settings_popup()
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
        enabled = bool(on)
        changed = enabled != self._selectors_on
        self._selectors_on = enabled
        self._apply_selectors_state()
        if changed:
            self.selectors_enabled_changed.emit(enabled)

    @property
    def selectors_enabled(self) -> bool:
        """The one console-authored selector switch state for this panel."""

        return bool(self._selectors_on)

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
        self._pending_figure = None
        self._pending_display = None
        self._pending_size_name = None
        self._pending_pixel_ratio = None
        self._pending_run_id = None
        self._pending_title = None
        self._pending_value_label = None
        self._pending_value = None
        self._render_request_revision += 1
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None

    def _teardown_plot(self) -> None:
        """Drop this card's surface, leaving nothing painted behind it.

        Hide synchronously, then keep the host parented until deferred deletion.
        Reparenting a QWidget to ``None`` would temporarily make it a top-level
        window, which is never a legal Figure lifecycle state.
        """

        board = self.board
        if (
            self._pending_interaction_origin is not None
            and self._pending_interaction_host is not None
        ):
            self._pending_interaction_host.discard_pending_interaction(
                self._pending_interaction_origin
            )
        self._pending_interaction_origin = None
        self._pending_interaction_host = None
        self._pending_interaction_revision = None
        self.board = None
        self._pending_frame = None
        self._pending_faceted_result = None
        self._pending_figure = None
        self._pending_display = None
        self._pending_size_name = None
        self._pending_pixel_ratio = None
        self._pending_run_id = None
        self._pending_title = None
        self._pending_value_label = None
        self._pending_value = None
        self._candidate_value = None
        self._candidate_schema = None
        self._grid_focus = None
        self._last_document = None
        self._last_display = None
        self._presented_figure = None
        self._presented_display = None
        self._presented_size_name = None
        self._presented_pixel_ratio = None
        self._presented_run_id = None
        self._presented_title = None
        self._presented_value_label = None
        self._presented_value = None
        self._clear_figure_outputs(notify=False)
        self._requested_signature = None
        self._latest_requested_source_ref = None
        self._latest_requested_source_key = None
        self._latest_requested_display_revision = None
        if board is not None:
            self.canvas_holder.removeWidget(board)
            board.hide()
            board.deleteLater()

    def shutdown(self) -> None:
        """Release this card's Qt surface.

        Worker requests contain no card/QWidget reference and are rejected by
        panel identity after removal, so teardown never waits on raster work.
        """

        self._fit_request_revision += 1
        self._fit_pending_source_ref = None
        self.fit_cancel_requested.emit()
        self._fit_panes.clear()
        self._fit_draft = None
        self._fit_draft_context = None
        self._fit_active_spec = None
        self._fit_result = None
        self._fit_result_identity = None
        self._teardown_plot()
