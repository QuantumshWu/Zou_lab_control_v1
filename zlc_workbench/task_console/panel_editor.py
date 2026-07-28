"""The panel Edit tab: a snapshot of the card plus its full parameter surface.

The editor freezes the card's accepted immutable input into its stable Figure
host.  View changes are composed by the TaskConsole's one worker lane against
that same frozen input; only explicit Refresh replaces it.  A calibrated
SiteMap, which is a two-input physical join rather than a DataFigure, follows
the same surface route.  The editor never evaluates data or owns a Matplotlib
composer.

"""

from __future__ import annotations

from pathlib import Path
import time

from PyQt5 import QtWidgets

import zlc_frontend.qt_widgets as _qt_widgets
from zlc_frontend.qt_widgets import (
    ACCENT,
    FluentButton,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentPathEdit,
    FluentReadoutEdit,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentSwitch,
    FigureSurfaceContext,
    FigureSurfaceHost,
    ViewSpecEditor,
    GREY,
    scaled_px,
    signals_blocked as _signals_blocked,
)
from zlc_frontend.form import choice_value_from_tree
from zlc_frontend.panel_params import panel_param_decls as _panel_param_decls
from zlc_frontend import RELIM_PARAM as _RELIM_PARAM
from zlc_frontend.render_style import panel_display_size
from zlc_storage.paths import display_path, user_output_path


FORM_WIDGET_HANDLERS = _qt_widgets.FORM_WIDGET_HANDLERS

#: Containers the Save row offers.  A container is a DATA-layer choice (which file
#: the same picture lands in), never an art knob: geometry, dpi and typography are
#: identical whichever is picked.
SAVE_IMAGE_FORMATS = ("png", "jpg")



def _front_qimage(frame):
    """The presented front as one Qt image, or None.

    Reads the panel's own raster rather than re-rendering: what gets written is
    the exact front the operator was looking at, palette and colour window
    included.
    """

    panels = tuple(getattr(frame, "panels", ()) or ())
    if not panels:
        return None
    return _qt_widgets.owned_qimage_for_raster(panels[0].raster)


class PanelEditor(QtWidgets.QWidget):
    """One panel's processing tab (confocal per-plot control card).

    Opened from a panel's ``Edit...`` button as its OWN closable tab, so several
    panels can be edited side by side.  Holds a FROZEN snapshot of that panel's
    current data plus a second projection of the same Figure-owned controls:

      * another editable view of the panel's Figure-owned Fit request; the
        same card state also backs Setting and no second window is opened;
      * manual x/y limits + Save Fig;

    Acquisition/Processor parameters remain on their one Logic-node form.  A
    plot tab never embeds a second copy or restarts its producer implicitly.

    The whole page lives in a scroll area, so the snapshot never pushes the
    fit/limits row off-screen."""


    def __init__(
        self,
        card: "PanelCard",
        console: "TaskConsole",
        parent: QtWidgets.QWidget,
    ):
        if parent is None:
            raise TypeError("PanelEditor requires its tab-stack parent")
        super().__init__(parent)
        self.card = card
        self.console = console
        self.setStyleSheet("background: transparent;")
        self._board = None
        self.render_surface_id = f"{card.panel_id}::edit::{id(self):x}"
        self._snapshot_value = None
        self._snapshot_publication = None
        self._snapshot_figure = None
        self._snapshot_display = None
        self._snapshot_contract = None
        self._render_request_revision = 0
        self._pending_render_result = None
        self._presented_render_request_revision = 0
        card.selectors_enabled_changed.connect(
            self._sync_snapshot_selectors
        )
        card.fit_presentation_changed.connect(
            self._request_snapshot_render
        )
        # A plot panel's Edit never carries a measurement form or Start/Stop: a plot is a
        # pure VIEW, and the node that produces its data lives on the Logic tab.
        self.meas_panel = None
        self._node = None                       # the node that produces this panel's data
        self.xmin = self.xmax = self.ymin = self.ymax = None
        self.clo = self.chi = None
        self._fit_pane = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._scroll = FluentScrollArea()
        outer.addWidget(self._scroll)
        # Build the complete Edit page off-tree, then attach it atomically.
        # Setting-row helpers deliberately return ordinary QWidget values;
        # inserting them one by one into an already-visible scroll viewport can
        # deliver a top-level Show event in the short interval before Qt reparents
        # them.  That was the visible "small boxes" flash on Edit.  The model and
        # handlers are shared with Setting, but each surface owns a stable widget
        # tree from birth to teardown.
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        col = QtWidgets.QVBoxLayout(page)
        margin = scaled_px(10, minimum=6)
        col.setContentsMargins(margin, margin, margin, margin)
        col.setSpacing(scaled_px(6, minimum=4))

        def section(text):
            col.addWidget(FluentSectionLabel(text))

        def inline(*widgets, trailing=None):
            host = QtWidgets.QWidget()
            row = QtWidgets.QHBoxLayout(host)
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(scaled_px(6, minimum=4))
            for widget in widgets:
                row.addWidget(widget, 0)
            row.addStretch(1)
            if trailing is not None:
                row.addWidget(trailing, 0)
            return host

        label_w = card.setting_label_width(self.fontMetrics())

        # ---- Panel: rename here as well as in the Setting popup; both go through the
        # card's one title handler, so the two surfaces stay views of one string.
        section("Panel")
        self.title_edit = FluentLineEdit(card.config.title)
        self.title_edit.setPlaceholderText("panel title...")
        self.title_edit.setToolTip("Rename this panel (also the default save name).")
        self.title_edit.editingFinished.connect(self._commit_title)
        col.addWidget(FluentSettingRow("title", self.title_edit, label_width=label_w))

        # ---- Parameters: the plot's own functional params, auto-discovered from the
        # kind's declarations, so a kind that gains a knob shows it with no wiring here.
        # ---- Display: the same view knobs the Setting popup renders, through the card's
        # SHARED row emitter and writing the SAME config.params through the card's one
        # writer -- the two surfaces are views of one state and cannot drift.
        self.ed_cmap = self.ed_relim = self.ed_unit_label = self.ed_fixed_row = None
        self.ed_fixed_lo = self.ed_fixed_hi = None
        self.ed_params = {}
        section("Display")
        display_specs = list(_panel_param_decls(card.config.kind)) + [_RELIM_PARAM]
        self.ed_params = card._emit_param_rows(
            display_specs,
            col.addWidget,
            self._edit_param,
            label_w,
            parent=page,
        )
        card._mount_grid_row_inventory(
            self,
            col.addWidget,
            view_apply=self._edit_grid_facet,
            param_apply=self._edit_param,
            label_w=label_w,
            form_widgets=self.ed_params,
            parent=page,
        )
        self.view_spec_editor = ViewSpecEditor(
            label_width=label_w,
            parent=page,
        )
        self.view_spec_editor.viewChanged.connect(self._edit_view_spec)
        col.addWidget(self.view_spec_editor)
        self.repeat_mode_row, self.repeat_mode_combo = card._make_repeat_mode_row(
            self._edit_repeat_mode,
            label_w,
            parent=page,
        )
        col.addWidget(self.repeat_mode_row)
        self.ed_cmap = self.ed_params.get("colormap")
        self.ed_relim = self.ed_params.get("relim")
        # An image's VALUE axis is its colour limit, pinned by the "colour range" row in
        # Limits; a second fixed lo/hi here would put two inputs on one source.
        if card.config.kind not in ("2d", "sites"):
            self.ed_fixed_row, self.ed_fixed_lo, self.ed_fixed_hi = card._make_fixed_lim_row(
                self._edit_fixed_lim,
                label_w,
                parent=page,
            )
            col.addWidget(self.ed_fixed_row)
            # The row stays PERMANENTLY in the layout; only its inputs enable in fixed
            # mode.  A visibility toggle above the snapshot reflowed everything below it
            # by the row's height on every relim change -- the reported Edit-tab jump.
            self._sync_fixed_lim_enabled(card._relim())
        unit_row, self.ed_unit_label = card._make_unit_readout_row(
            label_w,
            parent=page,
        )
        col.addWidget(unit_row)

        # ---- Processing: the frozen snapshot and its Refresh.
        section("Processing")
        head = QtWidgets.QHBoxLayout()
        head.addWidget(FluentLabel("frozen snapshot of current data"), 1)
        self.refresh_button = FluentButton("Refresh", color=GREY)
        self.refresh_button.setToolTip("Re-snapshot the panel's current data")
        self.refresh_button.clicked.connect(self.refresh_snapshot)
        head.addWidget(self.refresh_button)
        col.addLayout(head)
        self.canvas_holder = QtWidgets.QVBoxLayout()
        self.canvas_holder.setContentsMargins(0, 0, 0, 0)
        col.addLayout(self.canvas_holder)

        if card._fit_capable_kind():
            section("Fit")
            self._fit_pane = card.make_fit_authoring_pane(
                page,
                label_width=label_w,
                context_provider=self._fit_context,
            )
            col.addWidget(self._fit_pane)

        # ---- Limits: the view-window pins.  A box holds the STORED pin (empty =
        # autoscale) and is re-seeded only on build / show / Clear, so the refresh tick
        # can never clobber typing; the live window shows as the grey placeholder.
        section("Limits")
        self.xmin = FluentLineEdit("")
        self.xmax = FluentLineEdit("")
        if card.config.kind in ("2d", "sites"):
            # An image's x AND y are pixel coordinates: pinning both is what makes a crop
            # real.  A curve's y is owned by the relim family instead, so it gets no row.
            self.ymin = FluentLineEdit("")
            self.ymax = FluentLineEdit("")
            self.clo = FluentLineEdit("")
            self.chi = FluentLineEdit("")
        boxes = (self.xmin, self.xmax) + ((self.ymin, self.ymax) if self.ymin is not None else ())
        for widget in boxes:
            widget.setFixedWidth(scaled_px(88, minimum=68))
            widget.returnPressed.connect(self.apply_limits)
        apply_button = FluentButton("Apply lim", color=ACCENT)
        apply_button.clicked.connect(self.apply_limits)
        clear_button = FluentButton("Clear", color=GREY)
        clear_button.clicked.connect(self.clear_limits)
        lim_row = inline(self.xmin, self.xmax, trailing=apply_button)
        lim_row.layout().addWidget(clear_button, 0)
        col.addWidget(FluentSettingRow("x range", lim_row, label_width=label_w))
        if self.ymin is not None:
            col.addWidget(FluentSettingRow("y range", inline(self.ymin, self.ymax),
                                           label_width=label_w))
        if self.clo is not None:
            for widget in (self.clo, self.chi):
                widget.setFixedWidth(scaled_px(88, minimum=68))
                widget.returnPressed.connect(self.apply_clim)
            clim_apply = FluentButton("Apply", color=ACCENT)
            clim_apply.clicked.connect(self.apply_clim)
            clim_auto = FluentButton("Auto", color=GREY)
            clim_auto.clicked.connect(self.clear_clim)
            clim_row = inline(self.clo, self.chi, trailing=clim_apply)
            clim_row.layout().addWidget(clim_auto, 0)
            col.addWidget(FluentSettingRow("colour range", clim_row, label_width=label_w))

        # ---- Save: the figure this panel is showing.  Only the picture is written here:
        # the DATA behind it is already owned by the run's repository, and a second copy
        # written by the GUI would be a second answer to what was measured.
        section("Save")
        default_output_dir = user_output_path("figures", "task-console")
        self.save_dir_edit = FluentPathEdit(
            self.console._last_save_dir or str(default_output_dir),
            mode="dir",
            caption="Choose where to save",
            base_dir=str(default_output_dir),
        )
        self.save_dir_edit.setToolTip(
            "Where to save (folder, or a full path base).  Remembered across saves this "
            "session.  With auto-name OFF this is the exact output path.")
        col.addWidget(FluentSettingRow("path", self.save_dir_edit, label_width=label_w))
        self.save_autoname = FluentSwitch("auto-name (type + time)   ")
        self.save_autoname.setChecked(True)
        self.save_autoname.setToolTip(
            "ON: append _<plot-kind>_<timestamp> to the path (unique files).  "
            "OFF: write the path verbatim (you set the exact name; overwrites).")
        self.save_format_combo = FluentComboBox()
        self.save_format_combo.addItems(list(SAVE_IMAGE_FORMATS))
        self.save_format_combo.setCurrentText(SAVE_IMAGE_FORMATS[0])
        self.save_format_combo.setFixedWidth(scaled_px(72, minimum=56))
        self.save_format_combo.setToolTip("Image container for the saved figure.")
        self.save_button = FluentButton("Save Fig", color=ACCENT)
        self.save_button.setToolTip("Save exactly the picture this snapshot is showing.")
        col.addWidget(FluentSettingRow(
            "name", inline(self.save_autoname, self.save_format_combo, trailing=self.save_button),
            label_width=label_w))
        # A read-only-but-copyable field, not a wrapping label: a long absolute path has
        # nothing to wrap on, so a label would drag the whole page wider.
        self.save_preview = FluentReadoutEdit("")
        self.save_preview.setToolTip("The exact file that will be written -- select to copy.")
        self.save_preview.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        col.addWidget(FluentSettingRow("file", self.save_preview, label_width=label_w))
        self.save_dir_edit.changed.connect(lambda *_: self._update_save_preview())
        self.save_autoname.toggled.connect(lambda *_: self._update_save_preview())
        self.save_format_combo.currentTextChanged.connect(lambda *_: self._update_save_preview())
        self.save_button.clicked.connect(self.save)
        self._update_save_preview()

        self.status = FluentLabel("")
        self.status.setStyleSheet("color: %s; background: transparent; border: none;" % GREY)
        # A status line must never drive the page WIDTH: a label's size hint tracks its
        # text, so a long message would balloon the column into a horizontal scroll.
        self.status.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        col.addWidget(self.status)
        col.addStretch(1)

        # Publish the fully constructed subtree in one ownership transition.
        self._scroll.set_width_bounded_widget(page)

        # The host is permanent Edit chrome, even before the source has data.
        # TaskConsole supplies the tab stack as parent before construction, so
        # this child can never transiently become a native top-level window and
        # never has to be created after an already-visible page (which leaves a
        # newly inserted QWidget explicitly hidden on Qt).
        self._ensure_snapshot_surface()

    def refresh_snapshot(self) -> None:
        """Copy the card's immutable front into one stable shared host.

        Edit is another view of the same Figure, not a nested DataFigureWindow.
        The TaskConsole render lane remains the sole composer owner; this host
        receives an exact accepted frame and forwards explicit gestures back to
        the card-owned display state.
        """

        card = self.card
        source_host = None if card is None else card.board
        front = None if source_host is None else source_host.front_frame
        overview_artifact = (
            None
            if source_host is None
            else getattr(source_host, "overview_artifact", None)
        )
        if front is None and overview_artifact is None:
            self.status.setText("open the panel with data first")
            return
        try:
            value = card.frozen_render_value()
            publication = card.frozen_render_publication()
            contract = card.frozen_plot_panel_contract()
            size_name = contract.size_name
            if card.config.kind == "sites":
                figure = None
                display = card.frozen_display_state()
            else:
                figure = card.frozen_data_figure()
                display = card.frozen_display_state()
        except Exception as error:
            self.status.setText("%s: %s" % (type(error).__name__, error))
            return

        board = self._ensure_snapshot_surface()
        logical_size = tuple(
            int(value) for value in panel_display_size(size_name)
        )
        if board.faceted and overview_artifact is not None:
            board.present_overview(
                overview_artifact,
                context=FigureSurfaceContext.for_figure(
                    figure,
                    display=display,
                    contract=contract,
                ),
            )
        elif front is not None:
            selector_figure = figure
            live_context = None if source_host is None else source_host.context
            if live_context is not None:
                selector_figure = live_context.selector_figure
            board.present_frame(
                front,
                context=FigureSurfaceContext.for_frame(
                    front,
                    figure=figure,
                    display=display,
                    contract=contract,
                    selector_figure=(
                        selector_figure
                        if selector_figure is not figure
                        else None
                    ),
                ),
                logical_size=logical_size,
            )
        else:
            self.status.setText("the panel has no complete focused frame")
            return
        board.set_selectors_enabled(card.selectors_enabled)
        # Explicit Refresh is the only data-replacement edge for Edit.  Bump
        # the surface-local order even though this exact front was copied from
        # the live card, so any older worker answer still in flight is stale.
        self._render_request_revision += 1
        self._presented_render_request_revision = self._render_request_revision
        self._pending_render_result = None
        self._snapshot_value = value
        # Edit copies the exact publication already promoted with the painted
        # live front.  Looking it up again in an advancing data plane would
        # splice a newer camera transaction into this frozen surface.
        self._snapshot_publication = publication
        self._snapshot_figure = figure
        self._snapshot_display = display
        self._snapshot_contract = contract
        self._refresh_unit_readout()
        if self._fit_pane is not None:
            card.refresh_fit_authoring_pane(self._fit_pane)
        card._refresh_view_spec_control_surface(self)
        self.refresh_limit_hints()
        self.status.setText("")

    def _fit_context(self):
        """The exact frozen Figure/source this Edit surface is displaying."""

        if self._snapshot_value is None or self._snapshot_figure is None:
            return None
        front = None if self._board is None else self._board.front_frame
        payload = (
            None
            if front is None or len(front.panels) != 1
            else front.panels[0].display_payload
        )
        return (
            self._snapshot_value,
            self._snapshot_figure,
            self._snapshot_publication,
            payload,
        )

    def freeze_current_view_request(self, *, axis_labels=None, short_labels=None):
        """Freeze one display answer against the explicit Refresh input."""

        if self.card is None or self._snapshot_value is None:
            return None
        self._render_request_revision += 1
        try:
            return self.card.freeze_surface_request(
                self._snapshot_value,
                surface_id=self.render_surface_id,
                request_revision=self._render_request_revision,
                frame_key=(
                    "edit-snapshot",
                    self._snapshot_value.snapshot.ref,
                ),
                axis_labels=axis_labels,
                short_labels=short_labels,
                publication=self._snapshot_publication,
            )
        except (KeyError, LookupError, RuntimeError, TypeError, ValueError) as error:
            self.status.setText(f"{type(error).__name__}: {error}")
            return None

    def invalidate_raster_surface(self) -> None:
        """Make a prior-screen front ineligible without rebuilding this tab."""

        self._pending_render_result = None
        if self._board is not None:
            self._board.clear()

    def accept_render_result(
        self,
        request,
        *,
        frame=None,
        faceted_result=None,
        figure=None,
        error: str | None = None,
    ) -> bool:
        """Admit a worker answer only for this frozen Edit input/surface."""

        value = self._snapshot_value
        if (
            value is None
            or self._snapshot_publication is None
            or request.render_surface_id != self.render_surface_id
            or request.value is not value
            or self._snapshot_publication.value(value.name) is not value
            or request.contract.pixel_ratio != self.card.raster_pixel_ratio
        ):
            return False
        pending_revision = (
            None
            if self._pending_render_result is None
            else int(self._pending_render_result[0].request_revision)
        )
        floor = max(
            int(self._presented_render_request_revision),
            -1 if pending_revision is None else pending_revision,
        )
        if int(request.request_revision) <= floor:
            return False
        if error is not None:
            if int(request.request_revision) != self._render_request_revision:
                return False
            self.card._settle_pending_interaction_through(
                request.display.revision,
                failed=True,
                answer_host=self._board,
            )
            self.status.setText(str(error))
            return True
        if request.contract.faceted:
            from zlc_frontend.panel_render import FacetedPanelResult

            if not isinstance(faceted_result, FacetedPanelResult):
                self.status.setText("render worker returned no faceted front")
                return False
            if figure is not faceted_result.figure:
                self.status.setText("faceted render lost its exact DataFigure")
                return False
        elif frame is None or (
            self.card.config.kind != "sites" and figure is None
        ):
            self.status.setText("render worker returned no complete front")
            return False
        self._pending_render_result = (
            request,
            frame,
            faceted_result,
            figure,
        )
        return True

    def _has_staged_render(self, request) -> bool:
        pending = self._pending_render_result
        return bool(pending is not None and pending[0] is request)

    def _discard_staged_render(self, request) -> None:
        if self._has_staged_render(request):
            self._pending_render_result = None

    def present_render_result(self) -> None:
        """Present one accepted frozen-input answer on the Qt owner."""

        pending = self._pending_render_result
        if pending is None:
            return
        self._pending_render_result = None
        request, frame, faceted, figure = pending
        board = self._ensure_snapshot_surface()
        logical_size = tuple(
            int(value) for value in panel_display_size(request.contract.size_name)
        )
        selector_figure = None
        if faceted is not None:
            if faceted.focus is not None:
                intent = request.contract.intent
                if figure is None or intent is None:
                    raise RuntimeError(
                        "focused Edit front lost its typed Figure context"
                    )
                selector_figure = figure.focused_typed_panel(
                    faceted.focus.panel_index,
                    expected_address=faceted.focus.address,
                    expected_intent=intent,
                )
            board.present_faceted(
                faceted,
                context=(
                    FigureSurfaceContext.for_figure(
                        figure,
                        display=request.display,
                        contract=request.contract,
                    )
                    if faceted.overview is not None
                    else FigureSurfaceContext.for_frame(
                        faceted.frame,
                        figure=figure,
                        display=request.display,
                        contract=request.contract,
                        selector_figure=selector_figure,
                    )
                ),
                logical_size=logical_size,
            )
        else:
            board.present_frame(
                frame,
                context=FigureSurfaceContext.for_frame(
                    frame,
                    figure=figure,
                    display=request.display,
                    contract=request.contract,
                ),
                logical_size=logical_size,
            )
        self._snapshot_figure = figure
        self._snapshot_display = request.display
        self._snapshot_contract = request.contract
        self._presented_render_request_revision = int(
            request.request_revision
        )
        self.card._settle_pending_interaction_through(
            request.display.revision,
            failed=False,
            answer_host=board,
        )
        self.refresh_limit_hints()
        self.status.setText("")

    def _ensure_snapshot_surface(self):
        """Construct the one host type this panel needs, at most once."""

        wanted_faceted = self.card.config.kind == "grid"
        board = self._board
        if isinstance(board, FigureSurfaceHost) and board.faceted == wanted_faceted:
            return board
        self._retire_snapshot_surface(board)
        board = FigureSurfaceHost(
            self.card.panel_id,
            faceted=wanted_faceted,
            empty_text="no snapshot yet",
            output_authority=self.card._figure_output_authority,
            parent=self,
        )
        if wanted_faceted:
            board.focusRequested.connect(self._forward_grid_focus)
            board.overviewRequested.connect(self._forward_grid_overview)
        board.viewCommitted.connect(self._forward_view_commit)
        board.colorLimitsCommitted.connect(self._forward_color_limits)
        board.thresholdsCommitted.connect(self._forward_thresholds)
        board.interactionStarted.connect(
            lambda origin, host=board: self.card._begin_pointer_interaction(
                host,
                origin,
                value=self._snapshot_value,
                publication=self._snapshot_publication,
                surface_id=self.render_surface_id,
            )
        )
        board.interactionFinished.connect(
            lambda host=board: self.card._finish_pointer_interaction(host)
        )
        self._board = board
        self.canvas_holder.addWidget(board)
        return board

    def _retire_snapshot_surface(self, board) -> None:
        if board is None:
            return
        self.canvas_holder.removeWidget(board)
        board.hide()
        shutdown = getattr(board, "shutdown", None)
        if callable(shutdown):
            shutdown()
        # Keep the host parented until deferred deletion.  Reparenting a QWidget
        # to None makes it a transient top-level and can flash a second figure.
        board.deleteLater()

    def _sync_snapshot_selectors(self, enabled: bool) -> None:
        if self._board is not None:
            self._board.set_selectors_enabled(bool(enabled))

    def _request_snapshot_render(self) -> None:
        """Recompose presentation while retaining the explicit Refresh value."""

        if self.card is not None and self._board is not None:
            self.console._request_card_render(
                self.card,
                surface=self._board,
            )

    def _forward_view_commit(self, commit) -> None:
        self.card.accept_view_commit_from(self._board, commit)

    def _forward_color_limits(self, commit) -> None:
        self.card.accept_color_limits_from(self._board, commit)

    def _forward_thresholds(self, commit) -> None:
        self.card.accept_thresholds_from(self._board, commit)

    def _forward_grid_focus(self, panel_index: int, address) -> None:
        self.card._focus_grid_cell(panel_index, address)
        self._request_snapshot_render()

    def _forward_grid_overview(self) -> None:
        self.card._return_to_grid_overview()
        self._request_snapshot_render()

    def teardown(self) -> None:
        """Release this tab's frozen Qt presentation surface."""

        self._retire_snapshot_surface(self._board)
        self._board = None
        if self.card is not None and self._fit_pane is not None:
            self.card.release_fit_authoring_pane(self._fit_pane)
        self._fit_pane = None
        self._snapshot_publication = None
        if self.card is not None:
            try:
                self.card.fit_presentation_changed.disconnect(
                    self._request_snapshot_render
                )
            except (TypeError, RuntimeError):
                pass
            try:
                self.card.selectors_enabled_changed.disconnect(
                    self._sync_snapshot_selectors
                )
            except (TypeError, RuntimeError):
                pass

    def _edit_param(self, key: str, value) -> None:
        """Commit one authored parameter to the card's sole render request."""
        changed = False
        if self.card is not None:
            changed = self.card._set_param(key, value)
        if key == "relim" and getattr(self, "ed_fixed_row", None) is not None:
            self._sync_fixed_lim_enabled(str(value))   # enable lo/hi in fixed WITHOUT moving the page
            if str(value) == "fixed" and self.card is not None:
                # mirror the freeze-current-view seed _set_param just wrote (config.params is the
                # one source) into THIS tab's lo/hi inputs -- setText does not re-fire editingFinished
                for edit, pkey in ((self.ed_fixed_lo, "fixed_lo"), (self.ed_fixed_hi, "fixed_hi")):
                    if edit is not None and pkey in self.card.config.params:
                        with _signals_blocked(edit):
                            edit.setValue(float(self.card.config.params[pkey]))
        if key == "relim":
            # a 2D image has no Display fixed row (its clim lives in the Limits colour-range row):
            # re-seed THOSE boxes here so picking relim in the chooser fills/empties them to match the
            # pin -- runs even when ed_fixed_row is None, unlike the colour-range block above.
            self._seed_clim_boxes()
        if changed:
            self._request_snapshot_render()

    def _edit_repeat_mode(self, mode) -> None:
        if self.card is None:
            return
        if self.card._commit_repeat_mode(mode):
            self.card._refresh_view_spec_control_surface(self)
            self._request_snapshot_render()

    def _edit_grid_facet(self, intent, axis_id) -> bool:
        if self.card is None:
            return False
        changed = self.card._commit_grid_facet(intent, axis_id)
        if changed:
            self.card._refresh_grid_control_surface(self)
            self.card._refresh_view_spec_control_surface(self)
            self._request_snapshot_render()
        return changed

    def _edit_view_spec(self, view) -> None:
        if self.card is None:
            return
        if self.card._commit_view_spec(view):
            self.card._refresh_view_spec_control_surface(self)
            self._request_snapshot_render()

    def _sync_fixed_lim_enabled(self, relim: str) -> None:
        """The Edit tab's fixed lo/hi row is ALWAYS in the layout -- only its INPUTS enable when
        ``relim == "fixed"``.  A ``setVisible`` toggle on a row that sits above the snapshot canvas in
        the shared scroll page reflowed the unit row + the whole canvas down by the row's height on
        every relim change (the reported Edit-tab jump); greying the inputs in place leaves every
        widget put.  (The Setting popup keeps its reveal-and-grow-down: it is a compact floating card
        with nothing beneath the row, so growing downward moves nothing.)"""
        if self.ed_fixed_row is not None:
            # ALWAYS shown here (the shared _make_fixed_lim_row hides it by default for the Setting
            # popup's reveal-on-fixed) -- so the footprint is constant and the canvas never moves.
            self.ed_fixed_row.setVisible(True)
        fixed = str(relim) == "fixed"
        for w in (self.ed_fixed_lo, self.ed_fixed_hi):
            if w is not None:
                w.setEnabled(fixed)

    def _edit_fixed_lim(self) -> None:
        """Commit the Edit tab's fixed lo/hi to the live card through its one
        ``apply_fixed_lims`` path (config.params + local display commit), then let the next accepted
        front update this tab.  It never re-reads acquisition merely because a limit changed."""
        if self.card is None:
            return
        lo = float(self.ed_fixed_lo.value())
        hi = float(self.ed_fixed_hi.value())
        self.card.apply_fixed_lims(lo, hi)
        self._request_snapshot_render()

    def _refresh_unit_readout(self) -> None:
        if self.ed_unit_label is None:
            return
        value = self._snapshot_value
        unit = None if value is None else getattr(value, "unit", None)
        self.ed_unit_label.setText(
            "—" if value is None else (unit or "dimensionless")
        )

    def _commit_title(self) -> None:
        """Commit the Edit tab's local title draft once."""
        if self.card is None:
            return
        text = str(self.title_edit.text())
        if text == self.card.config.title:
            return
        edit = getattr(self.card, "title_edit", None)
        if edit is not None:
            with _signals_blocked(edit):
                edit.setText(text)
        self.card._commit_title()
        self._request_snapshot_render()
        self._update_save_preview()               # default save name follows the title

    def refresh_on_show(self) -> None:
        """Re-seed this Figure projection from the card's sole authored state."""
        card = self.card
        self._refresh_display_params()
        if card is not None:
            self._refresh_unit_readout()
            if self._fit_pane is not None:
                card.refresh_fit_authoring_pane(self._fit_pane)
            card._refresh_grid_control_surface(self)
            card._seed_repeat_mode_control(
                self.repeat_mode_combo,
                self.repeat_mode_row,
            )
            card._refresh_view_spec_control_surface(self)
        self._seed_limit_boxes()        # re-seed the pin from config.params (may have changed in Setting)
        self.refresh_limit_hints()

    def _refresh_display_params(self) -> None:
        """Re-seed the Edit tab's display-knob controls (``ed_params``) from the live card's
        ``config.params`` -- the SINGLE source of truth -- so switching back to this tab shows the
        CURRENT values even when they were changed in the Setting popup (which writes the same
        config.params).  Each control is re-seeded through its frontend form handler (the card
        records the kinds while building both surfaces' rows), signals blocked so re-seeding does not
        re-fire ``_edit_param``.  Like ``PanelCard.refresh_on_show``, both surfaces are a
        VIEW of config.params, refreshed on show, never private copies that drift."""
        if self.card is None:
            return
        fields = getattr(self.card, "_param_fields", {})
        params = self.card.config.params
        for key, widget in self.ed_params.items():
            field = fields.get(key)
            if field is None or key not in params:
                continue
            with _signals_blocked(widget):
                FORM_WIDGET_HANDLERS[field.kind].write(
                    field,
                    widget,
                    (
                        choice_value_from_tree(field, params[key])
                        if field.kind == "choice"
                        else params[key]
                    ),
                )

    def _limit_axes(self):
        """The view-window rows this editor built, as ``(param key, lo box, hi box)``.

        One list drives seeding, applying and clearing, so a kind that has no y
        row simply contributes no y triple instead of every reader repeating the
        same "does this kind have y?" test.
        """

        rows = [("view_xlim", self.xmin, self.xmax)]
        if self.ymin is not None:
            rows.append(("view_ylim", self.ymin, self.ymax))
        return tuple(rows)

    def _visible_snapshot_pixels(self):
        """Return the immutable front or overview artifact actually on screen."""

        board = self._board
        return (
            None if board is None else getattr(board, "front_frame", None),
            None if board is None else getattr(board, "overview_artifact", None),
        )

    def _front_view_bounds(self):
        """The window the composed front is showing, as (xlo, xhi, ylo, yhi).

        Read off the front's own viewport: it is the transform that maps this
        picture's pixels back onto the declared axes, so the hint describes the
        picture rather than a range derived beside it.
        """

        front, _overview_artifact = self._visible_snapshot_pixels()
        panels = tuple(getattr(front, "panels", ()) or ())
        if not panels:
            return None
        payload = panels[0].display_payload
        viewport = getattr(getattr(payload, "background", payload), "viewport", None)
        axes = tuple(getattr(viewport, "axes", ()) or ())
        bounds = tuple(getattr(viewport, "visible_bounds", ()) or ())
        if len(axes) != 2 or len(bounds) != 4:
            return None
        y_axis, x_axis = axes
        x_size = float(getattr(x_axis, "size", 0) or 0)
        y_size = float(getattr(y_axis, "size", 0) or 0)
        left, top, right, bottom = (float(value) for value in bounds)
        return (left * x_size, right * x_size, top * y_size, bottom * y_size)

    def _seed_limit_boxes(self) -> None:
        """Put the STORED x-window pin (``view_xlim`` in ``config.params``) into the boxes.  The boxes
        EDIT THE PIN, never the live autoscaled range, so their text never wanders on its own: empty
        boxes mean 'no pin' (autoscale), a value means 'pinned there'.  Called on build / tab-show /
        after Clear -- NEVER on the refresh tick.  The live range is shown separately as a
        non-destructive grey hint (:meth:`refresh_limit_hints`)."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        for key, lo_box, hi_box in self._limit_axes():
            pin = self.card.config.params.get(key) if self.card is not None else None
            lo = hi = ""
            if pin is not None:
                try:
                    lo, hi = f"{float(pin[0]):.6g}", f"{float(pin[1]):.6g}"
                except (TypeError, ValueError, IndexError):
                    lo = hi = ""
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(lo); hi_box.setText(hi)
        self._seed_clim_boxes()          # keep the 2D colour-range boxes in step with the clim pin

    def refresh_limit_hints(self) -> None:
        """Refresh ONLY the grey PLACEHOLDER of the x-range boxes to the panel's current x-window -- a
        non-destructive live reference.  Qt shows a placeholder ONLY while the box is empty, so this can
        never overwrite a pinned value or the operator's in-progress typing.  The
        tick updates only this hint, so the boxes stay a live view of the x-window
        while remaining freely editable."""
        if self.xmin is None:
            return                          # no Limits controls on this editor instance
        # Colour-range live hint: show the current clim as the grey placeholder so an empty
        # box (= Auto) still tells the operator what range the image is using -- the clim counterpart of
        # the x-window hint below.  Non-destructive: Qt draws a placeholder only while the box is empty,
        # so a pinned/typed value is never overwritten.
        if getattr(self, "clo", None) is not None and self.card is not None:
            shown = self.card._shown_limits()      # None until something has been composed
            if shown is not None:
                self.clo.setPlaceholderText(f"{shown[0]:.6g}")
                self.chi.setPlaceholderText(f"{shown[1]:.6g}")
        bounds = self._front_view_bounds()
        if bounds is None:
            return
        xlo, xhi, ylo, yhi = bounds
        self.xmin.setPlaceholderText(f"{xlo:.6g}"); self.xmax.setPlaceholderText(f"{xhi:.6g}")
        if self.ymin is not None:           # y hint only where y is a view axis (an image family)
            self.ymin.setPlaceholderText(f"{ylo:.6g}"); self.ymax.setPlaceholderText(f"{yhi:.6g}")


    def apply_limits(self) -> None:
        """Apply the typed view-window pins -- PER AXIS: a filled pair pins that axis, an empty pair
        releases it (so 'clear the y boxes + Apply' un-pins y while keeping x).  Each pin is an ORDINARY
        display knob (``view_xlim``/``view_ylim``) applied through the SAME ``_edit_param`` entry as
        bins / relim -> the live card + accepted front + ``config.params`` + save."""
        if self.xmin is None:
            return
        applied, cleared, updates = [], [], {}
        for key, lo_box, hi_box in self._limit_axes():
            lo_text, hi_text = lo_box.text().strip(), hi_box.text().strip()
            if not lo_text and not hi_text:
                updates[key] = None                  # empty pair = release THIS axis's pin
                cleared.append(key[5])               # 'x' / 'y'
                continue
            try:
                lo, hi = float(lo_text), float(hi_text)
            except ValueError as exc:
                self.status.setText(f"bad limits: {exc}")
                return
            updates[key] = (lo, hi)
            applied.append(key[5])
        if self.card is not None:
            if self.card._set_params(updates):
                self._request_snapshot_render()
        parts = ([f"{'/'.join(applied)} range applied"] if applied else []) + \
                ([f"{'/'.join(cleared)} range cleared"] if cleared else [])
        self.status.setText("; ".join(parts) + " (all subplots)")

    def clear_limits(self) -> None:
        """Release EVERY view-window pin -> autoscale.  ``None`` is the SAME stored display knob
        ``apply_limits`` writes, routed through the ONE ``_edit_param`` entry: BaseLivePlot.apply_param /
        GridCell.consume_param already read ``None`` as 'no pin', so the live card, the snapshot, and the
        save all drop the pin.  EMPTIES the boxes (empty = no pin) so the grey placeholder hint takes
        over showing the now-auto range -- the Limits counterpart of :meth:`clear_fit`."""
        if self.xmin is None:
            return
        updates = {}
        for key, lo_box, hi_box in self._limit_axes():
            updates[key] = None
            with _signals_blocked(lo_box, hi_box):
                lo_box.setText(""); hi_box.setText("")
        if self.card is not None:
            if self.card._set_params(updates):
                self._request_snapshot_render()
        self.refresh_limit_hints()
        self.status.setText("view range cleared (auto)")

    def _sync_relim_combo(self, value: str) -> None:
        """Reflect a PROGRAMMATIC relim change (the colour-range Apply/Auto) in the Display relim combo
        WITHOUT re-firing its handler, so the chooser and the colour-range row always agree on whether
        the clim is pinned.  No-op when the combo is absent (a non-image Edit that has no relim row)."""
        combo = self.ed_params.get("relim") if getattr(self, "ed_params", None) else None
        if combo is None:
            return
        idx = combo.findText(value)
        if idx >= 0:
            with _signals_blocked(combo):
                combo.setCurrentIndex(idx)

    def apply_clim(self) -> None:
        """Pin a 2D panel's colour limit to the typed lo/hi.  Routes through the ONE clim source the
        relim family owns (relim="fixed" + the card's ``apply_fixed_lims``), so the live card (every cell,
        re-asserted each tick), this tab's snapshot, ``config.params`` and Save all move together -- there
        is no second hand-copied clim path.  Both boxes empty = Auto (release the pin)."""
        if self.clo is None:
            return
        if not self.clo.text().strip() and not self.chi.text().strip():
            self.clear_clim()
            return
        try:
            lo, hi = float(self.clo.text()), float(self.chi.text())
        except ValueError as exc:
            self.status.setText(f"bad colour range: {exc}")
            return
        if self.card is not None:
            self.card.apply_fixed_lims(lo, hi)
            self._request_snapshot_render()
        self._sync_relim_combo("fixed")
        self._seed_clim_boxes()
        self.status.setText("colour range applied")

    def clear_clim(self) -> None:
        """Release the colour-limit pin back to the autoscaled clim (relim="normal") -- the image
        counterpart of :meth:`clear_limits`.  Empties the boxes so the grey placeholder hint takes over
        showing the now-auto clim."""
        if self.clo is None:
            return
        self._edit_param("relim", "normal")
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(""); self.chi.setText("")
        self._sync_relim_combo("normal")
        self.refresh_limit_hints()
        self.status.setText("colour range cleared (auto)")

    def _seed_clim_boxes(self) -> None:
        """Put the STORED clim pin (``fixed_lo/hi``, in force only while relim=="fixed") into the
        colour-range boxes.  Like :meth:`_seed_limit_boxes` for x: the boxes edit the PIN, so empty = Auto
        and a value = pinned; re-seeded on build / tab-show / relim change, NEVER on the tick."""
        if getattr(self, "clo", None) is None:
            return
        p = self.card.config.params if self.card is not None else {}
        lo = hi = ""
        if str(p.get("relim")) == "fixed":
            lo, hi = (
                f"{float(p.get('fixed_lo')):.6g}",
                f"{float(p.get('fixed_hi')):.6g}",
            )
        with _signals_blocked(self.clo, self.chi):
            self.clo.setText(lo); self.chi.setText(hi)

    def _save_stem(self, timestamp: str | None) -> Path:
        """The output file stem (no extension) the Save section resolves to.

        ``path`` (the picker) is the base.  With auto-name ON the file is
        ``<base-or-base/title>_<plot-kind>_<timestamp>`` (unique); a ``timestamp`` of None
        yields the literal ``<time>`` placeholder for the read-only preview.  With auto-name
        OFF the base is used VERBATIM (its extension stripped), so the operator sets the exact
        name.  ONE resolver, shared by :meth:`save` and :meth:`_update_save_preview`."""
        title = (self.card.config.title or self.card.config.kind).strip() or "panel"
        kind = self.card.config.kind
        text = self.save_dir_edit.text().strip() if hasattr(self, "save_dir_edit") else ""
        base = Path(text) if text else Path(
            self.console._last_save_dir
            or user_output_path("figures", "task-console")
        )
        # a bare folder (blank default / trailing sep / an existing dir) -> the file is
        # <folder>/<title>; otherwise the path already names the file stem.
        if not text or str(base).endswith(("/", "\\")) or (base.is_dir() and not base.suffix):
            base = base / title
        base = base.with_suffix("")               # we always append our own .png / .npz
        if hasattr(self, "save_autoname") and self.save_autoname.isChecked():
            return base.parent / f"{base.name}_{kind}_{timestamp or '<time>'}"   # unique
        return base                                # verbatim (operator-set name; overwrites)

    def _save_image_ext(self) -> str:
        """The image container the next Save writes (``png`` / ``pdf`` / ``jpg``).

        ONE reader of the format picker, shared by :meth:`save` and :meth:`_update_save_preview`, so the
        previewed suffix and the file actually written never disagree.  Falls back to the first
        :data:`SAVE_IMAGE_FORMATS` entry when the picker is absent (e.g. a non-plot editor) or empty."""
        combo = getattr(self, "save_format_combo", None)
        ext = combo.currentText().strip().lower() if combo is not None else ""
        return ext or SAVE_IMAGE_FORMATS[0]

    def _update_save_preview(self) -> None:
        """Show the actual file (full path) the next Save writes -- not just the folder."""
        if not hasattr(self, "save_preview"):
            return
        archive_suffix = "" if self.card.config.kind == "sites" else " + .npz"
        self.save_preview.setText(
            f"{display_path(str(self._save_stem(None)))}."
            f"{self._save_image_ext()}{archive_suffix}")

    def save(self) -> None:
        """Write the exact visible pixels and their exact typed data revision."""

        front, overview_artifact = self._visible_snapshot_pixels()
        if front is None and overview_artifact is None:
            self.status.setText("no snapshot to save")
            return
        try:
            stem = self._save_stem(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
            stem.parent.mkdir(parents=True, exist_ok=True)
            target = stem.with_suffix(".%s" % self._save_image_ext())
            image = (
                _front_qimage(front)
                if front is not None
                else _qt_widgets.owned_qimage_for_raster(
                    overview_artifact.raster
                )
            )
            if image is None or not image.save(str(target)):
                raise RuntimeError("Qt refused to write %s" % target.name)
            if self.card.config.kind == "sites":
                self.console._last_save_dir = str(stem.parent)
                self._update_save_preview()
                self.status.setText("saved %s" % target.name)
                self.status.setToolTip(str(stem.parent))
                return
            figure = self._snapshot_figure
            display = self._snapshot_display
            contract = self._snapshot_contract
            if figure is None or display is None or contract is None:
                raise RuntimeError("the editor has no exact typed figure to save")
            from zlc_frontend import FigurePresentationContract
            from zlc_workbench.data_figure.archive_repository import save_figure_archive

            presentation = FigurePresentationContract.from_plot_panel(
                contract,
                display,
            )

            save_figure_archive(
                figure,
                stem.with_suffix(".npz"),
                presentation=presentation,
            )
            self.console._last_save_dir = str(stem.parent)
            self._update_save_preview()
            self.status.setText("saved %s + %s" % (target.name, stem.with_suffix(".npz").name))
            self.status.setToolTip(str(stem.parent))
        except Exception as error:
            self.status.setText("save failed: %s" % error)

