"""Qt DataFigure surface and lifecycle owner."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
from pathlib import Path
import threading
import time

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    AxisId,
    CoordinateRangeSelection,
    FitCancelled,
    FitDeadlineExceeded,
    FitResultBatch,
    FitSpec,
    IndexRangeSelection,
    Selection,
)
from zlc_frontend import (
    CurvePanelPayload,
    DataFigure,
    FitAuthoringDraft,
    FitAuthoringOption,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterDisplayState,
    MeterPanelPayload,
    reconcile_fit_authoring_draft,
)
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.display_range import RelimMode
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.figure import AxisViewRole, ViewIntent
from zlc_frontend.histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
    histogram_display_with_thresholds,
)
from zlc_frontend.image_display import ImageDisplayState, image_display_for_viewport
from zlc_frontend.qt_widgets import (
    FitAuthoringPane,
    FluentButton,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentSwitch,
    GREY,
    GREEN,
    ORANGE,
    QtRasterBoard,
    FrozenRasterView,
    runtime_range_placeholders,
    FluentSettingsPopupAnchor,
    sync_revisioned_form_editors,
)
from zlc_frontend.selector import (
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
    HistogramInteractionIntent,
    HistogramRangeGesture,
    HistogramThresholdCommit,
    HistogramViewportCommit,
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
    RectangleGesture,
)
from zlc_neutral_atom.artifacts import FitResultArtifactRef
from zlc_workbench.fit import FitDraftAuthority, FitDraftResult
from zlc_storage import nonnegative_integer
from zlc_workbench.frozen_raster import FrozenRasterWindow
from zlc_workbench.window_runtime import cancel_export_commits, error_summary

from .projection import (
    _FitSaveReceipt,
    _TYPED_PANEL_ID,
    _FitOverlayRequest,
    _FitWorkbenchBindings,
    _GridFocusRequest,
    _TypedDisplayState,
    _TypedFigureFront,
    _TypedGridOverview,
    _build_typed_front_contract,
    _default_typed_state,
    _same_exact_data_owners,
    _same_fit_overlay_request,
    _state_intent,
    _typed_form_spec,
    _typed_form_values,
    _typed_state_from_form,
    _typed_state_with_x_view,
    _validate_rendered_authored_payload,
)
from .render_lane import (
    _FIT_WORK_EXECUTOR,
    _execute_fit_draft,
    _export_encoded_png,
    _export_typed_png,
    _prepare_fit_options,
)


class DataFigureWindow(FrozenRasterWindow):
    """Frozen generic viewer with one closed IMAGE/CURVE/HISTOGRAM/METER front."""

    initialReady = QtCore.pyqtSignal()
    initialFailed = QtCore.pyqtSignal(str)
    fitSaved = QtCore.pyqtSignal(object)

    def __init__(
        self,
        initial_loader,
        typed_renderer,
        fit_overlay_renderer=None,
        *,
        fit_bindings: _FitWorkbenchBindings | None = None,
        typed_front_committed=None,
        initial_display: _TypedDisplayState | None = None,
        embedded: bool = False,
        logical_panel_size: tuple[int, int] | None = None,
    ) -> None:
        if not callable(initial_loader) or not callable(typed_renderer):
            raise TypeError("figure worker callables must be callable")
        if fit_bindings is not None and not isinstance(
            fit_bindings,
            _FitWorkbenchBindings,
        ):
            raise TypeError("fit_bindings must be _FitWorkbenchBindings or None")
        if (fit_bindings is None) != (fit_overlay_renderer is None):
            raise ValueError("Fit bindings and overlay renderer must be supplied together")
        if fit_overlay_renderer is not None and not callable(fit_overlay_renderer):
            raise TypeError("fit_overlay_renderer must be callable or None")
        if typed_front_committed is not None and not callable(typed_front_committed):
            raise TypeError("typed_front_committed must be callable or None")
        if initial_display is not None:
            _state_intent(initial_display)
        if not isinstance(embedded, bool):
            raise TypeError("embedded must be bool")
        if logical_panel_size is not None:
            logical_panel_size = tuple(logical_panel_size)
            if (
                len(logical_panel_size) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in logical_panel_size
                )
            ):
                raise ValueError(
                    "logical_panel_size must contain two positive integers"
                )
        self._typed_renderer = typed_renderer
        self._fit_overlay_renderer = fit_overlay_renderer
        self._fit_bindings = fit_bindings
        self._typed_front_committed = typed_front_committed
        self._initial_display = initial_display
        self._embedded = embedded
        self._logical_panel_size = logical_panel_size
        self._view_family: str | None = None
        self._display: _TypedDisplayState | None = None
        self._typed_contract: (
            tuple[tuple[object, ...], object] | None
        ) = None
        self._typed_pages_admitted = False
        self._typed_ui_faulted = False
        self._initial_outcome: str | None = None
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: _TypedDisplayState | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._pending_editor: FluentRevisionedFormEditor | None = None
        self._pending_editor_revision: int | None = None
        self._completion_handoff_active = False
        self._deferred_typed_retry: tuple[object, ...] | None = None
        self._edit_display: FluentRevisionedFormEditor | None = None
        self._setting_display: FluentRevisionedFormEditor | None = None
        self._export_commit_lock = threading.Lock()
        self._visible_transient_fit_result_owner: FitResultBatch | None = None
        self._fit_axis_ids: tuple[AxisId, ...] = ()
        self._fit_axis_roles: tuple[tuple[AxisId, AxisViewRole], ...] = ()
        self._visible_fit_result_identity: str | None = None
        self._grid_overview: _TypedGridOverview | None = None
        self._visible_figure: DataFigure | None = None
        self._grid_focus_pending: _GridFocusRequest | None = None
        self._discard_grid_focus_sequence: int | None = None

        self._fit_future: Future | None = None
        self._fit_job_kind: str | None = None
        self._fit_job_revision: int | None = None
        self._fit_editor_revision = 0
        self._fit_prepare_pending = False
        self._fit_overlay_desired: _FitOverlayRequest | None = None
        self._fit_overlay_pending: _FitOverlayRequest | None = None
        self._fit_overlay_inflight: _FitOverlayRequest | None = None
        self._fit_selection_candidate: Selection | None = None
        self._fit_initial_selection_consumed = False
        self._fit_auto_open_consumed = False
        self._fit_options: dict[str, FitAuthoringOption] = {}
        self._fit_authoring_draft: FitAuthoringDraft | None = None
        self._fit_cancelled: threading.Event | None = None
        self._fit_draft: FitDraftResult | None = None
        self._fit_draft_summary: str | None = None
        self._fit_save_inflight: FitDraftResult | None = None
        self._deferred_fit_reload: tuple[_FitSaveReceipt, int] | None = None
        self._saved_fit_receipt: _FitSaveReceipt | None = None
        self._saved_fit_reference: FitResultArtifactRef | None = None
        self._fit_save_path: Path | None = (
            None
            if fit_bindings is None
            or fit_bindings.initial_save_path is None
            else Path(fit_bindings.initial_save_path)
        )
        self._close_deferred_during_fit_save = False
        self._fit_pane: FitAuthoringPane | None = None
        self._fit_save_button: FluentButton | None = None
        self._fit_authority: FitDraftAuthority | None = None
        if fit_bindings is not None:
            def execute_for_authority(spec, cancel_check, deadline):
                execution = fit_bindings.execute(
                    spec,
                    cancel_check,
                    deadline,
                )
                return execution, fit_bindings.result(execution)

            self._fit_authority = FitDraftAuthority(
                execute_for_authority,
                lambda execution, destination, display: fit_bindings.save(
                    execution,
                    destination,
                    display,
                ),
            )

        super().__init__(
            None,
            window_title="Data Figure",
            mode_text="FROZEN DATA FIGURE · INTERACTIVE",
            loading_summary="Resolving immutable input…",
            object_prefix="figureViewer",
            subject="figure",
        )
        if embedded:
            self.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding,
                QtWidgets.QSizePolicy.Expanding,
            )
            self._layout.setContentsMargins(0, 0, 0, 0)
            self._close_button.hide()

        self._typed_page = QtWidgets.QWidget(self._tabs)
        self._typed_page.hide()
        page_layout = QtWidgets.QVBoxLayout(self._typed_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            (_TYPED_PANEL_ID,),
            self._typed_page,
            columns=1,
            empty_text="Resolving exact typed figure…",
        )
        self._board_widget.setObjectName("figureViewerTypedBoard")
        if logical_panel_size is None:
            self._board_widget.setMinimumSize(480, 320)
        else:
            self._board_widget.setFixedSize(*logical_panel_size)
        page_layout.addWidget(self._board_widget, 1)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("figureViewerTypedSettingsPopup")
        self._settings_popup_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._interaction_switch = FluentSwitch("Interact", self)
        self._interaction_switch.setObjectName("figureViewerTypedInteractSwitch")
        self._interaction_switch.setChecked(True)
        self._settings_button = FluentButton("Setting…", self, color=GREY)
        self._settings_button.setObjectName("figureViewerTypedSettingButton")
        self._export_button = FluentButton("Export PNG…", self, color=ORANGE)
        self._export_button.setObjectName("figureViewerTypedExportButton")
        self._fit_button = FluentButton("Fit", self, color=ORANGE)
        self._fit_button.setObjectName("figureViewerFitButton")
        self._overview_button = FluentButton("Overview", self, color=GREY)
        self._overview_button.setObjectName("figureViewerGridOverviewButton")
        self._controls.insertWidget(0, self._interaction_switch)
        self._controls.insertWidget(1, self._settings_button)
        self._controls.insertWidget(2, self._fit_button)
        self._controls.insertWidget(3, self._overview_button)
        self._controls.insertWidget(4, self._export_button)
        for widget in (
            self._interaction_switch,
            self._settings_button,
            self._fit_button,
            self._overview_button,
            self._export_button,
        ):
            widget.hide()

        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._settings_button,
        )
        self._settings_button.clicked.connect(self._open_display_settings)
        self._export_button.clicked.connect(self._choose_export)
        self._fit_button.clicked.connect(self._open_fit_pane)
        self._overview_button.clicked.connect(self._show_grid_overview)
        self._interaction_switch.toggled.connect(self._toggle_interaction)
        if fit_bindings is not None:
            pane = FitAuthoringPane(self._tabs)
            pane.setObjectName("figureViewerFitAuthoring")
            pane.fitRequested.connect(self._start_fit)
            pane.fitRequestRejected.connect(self._reject_fit_request)
            pane.clearRequested.connect(self._clear_fit)
            pane.editorChanged.connect(self._fit_editor_changed)
            save_button = FluentButton("Save Fit", pane, color=GREEN)
            save_button.setObjectName("figureViewerSaveFitButton")
            save_button.clicked.connect(self._save_fit)
            save_button.setEnabled(False)
            pane.layout().addWidget(save_button)
            pane.hide()
            self._fit_pane = pane
            self._fit_save_button = save_button
        self._set_typed_controls_enabled(False)
        if not self._submit_future(
            initial_loader,
            self._request_revision,
            self._cancelled,
        ):
            self._active_kind = None
            failure = self._diagnostic.text() or "initial figure work was not submitted"
            QtCore.QTimer.singleShot(
                0,
                lambda detail=failure: self._emit_initial_failed(detail),
            )

    def _emit_initial_ready(self) -> None:
        """Publish the one-time boundary after an actual front is admitted."""

        if self._initial_outcome is not None:
            return
        self._initial_outcome = "ready"
        self.initialReady.emit()

    def _emit_initial_failed(self, detail: str) -> None:
        """Publish the one-time boundary when no initial front was admitted."""

        if self._initial_outcome is not None:
            return
        self._initial_outcome = "failed"
        self.initialFailed.emit(str(detail))

    def _present_grid_overview(self, overview: _TypedGridOverview) -> None:
        if not isinstance(overview, _TypedGridOverview):
            raise TypeError("overview must be _TypedGridOverview")
        if not self._present_bundle(overview.bundle):
            raise RuntimeError("Qt rejected the immutable typed grid overview")
        if len(self._boards) != 1 or not isinstance(self._boards[0], FrozenRasterView):
            raise RuntimeError("typed grid overview did not admit one encoded board")
        board = self._boards[0]
        tab_host = self._tab_host_for_board(board)
        if self._logical_panel_size is not None:
            board.setFixedSize(*self._logical_panel_size)
        board.normalizedDoubleClicked.connect(self._focus_grid_region)
        self._grid_overview = overview
        self._visible_figure = overview.figure
        self._view_family = f"{overview.intent.value.lower()}-overview"
        self._display = None
        self._typed_contract = None
        self._tabs.setCurrentWidget(tab_host)
        self._tabs.tabBar().setVisible(False)
        self._mode.setText(f"EXACT {overview.intent.value} GRID · DISPLAY ONLY")
        self._status.setText("READY")
        self._summary.setText(overview.bundle.summary)
        self._diagnostic.setText("")
        for widget in (
            self._interaction_switch,
            self._settings_button,
            self._fit_button,
            self._overview_button,
        ):
            widget.hide()
        self._export_button.show()
        self._set_typed_controls_enabled(True)

    @QtCore.pyqtSlot(float, float)
    def _focus_grid_region(self, x: float, y: float) -> None:
        overview = self._grid_overview
        if (
            overview is None
            or self._view_family != f"{overview.intent.value.lower()}-overview"
            or self._future is not None
            or self._closing
        ):
            return
        hits = tuple(
            (index, region)
            for index, region in enumerate(overview.regions)
            if region.contains(x, y)
        )
        if not hits:
            return
        if len(hits) != 1:
            self._diagnostic.setText(
                "Grid focus rejected an ambiguous panel-boundary hit."
            )
            return
        panel_index, region = hits[0]
        if region.focus_selection is None:
            raise RuntimeError("typed grid region lost its exact selection")
        grid_display = overview.display_state
        if isinstance(grid_display, FacetedHistogramDisplayState):
            display = grid_display.display_for(region.focus_selection)
        elif isinstance(grid_display, MeterDisplayState):
            display = MeterDisplayState(
                panel_index,
                region.focus_selection,
                grid_display.revision,
            )
        elif grid_display is None:
            display = (
                MeterDisplayState(panel_index, region.focus_selection)
                if overview.intent is ViewIntent.METER
                else _default_typed_state(overview.intent)
            )
        else:
            display = grid_display
        request = _GridFocusRequest(
            panel_index,
            region.focus_selection,
            display,
            overview.histogram_home_x_limits,
        )
        self._request_revision += 1
        self._grid_focus_pending = request
        self._discard_grid_focus_sequence = None
        self._active_kind = "grid_focus"
        self._status.setText(f"RENDERING {overview.intent.value}")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        renderer = self._typed_renderer
        if renderer is None or not self._submit_future(
            renderer,
            None,
            None,
            request,
            None,
            None,
            None,
            self._request_revision,
            self._cancelled,
        ):
            self._grid_focus_pending = None
            self._active_kind = None
            self._set_typed_controls_enabled(True)

    def _show_grid_overview(self) -> None:
        overview = self._grid_overview
        if overview is None or self._closing:
            return
        if self._active_kind in ("grid_focus", "typed") and self._future is not None:
            self._discard_grid_focus_sequence = self._request_revision
            self._future.cancel()
            self._status.setText("RETURNING TO OVERVIEW")
            return
        if self._future is not None or self._view_family != overview.intent.value.lower():
            return
        if len(self._boards) != 1 or not self._boards[0].has_front:
            raise RuntimeError("cached typed grid overview front is unavailable")
        self._board_widget.clear()
        self._display = None
        self._typed_contract = None
        self._visible_fit_result_identity = None
        self._visible_transient_fit_result_owner = None
        self._fit_axis_ids = ()
        self._fit_axis_roles = ()
        self._fit_overlay_desired = None
        self._visible_figure = overview.figure
        self._view_family = f"{overview.intent.value.lower()}-overview"
        overview_board = self._boards[0]
        overview_host = self._tab_host_for_board(overview_board)
        if self._tabs.indexOf(overview_host) < 0:
            self._tabs.insertTab(0, overview_host, "Overview")
        self._tabs.setCurrentWidget(overview_host)
        self._tabs.tabBar().setVisible(False)
        self._mode.setText(f"EXACT {overview.intent.value} GRID · DISPLAY ONLY")
        self._status.setText("READY")
        self._summary.setText(overview.bundle.summary)
        self._diagnostic.setText("")
        self._overview_button.hide()
        self._export_button.show()
        self._set_typed_controls_enabled(True)

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Escape and self._grid_overview is not None:
            self._show_grid_overview()
            event.accept()
            return
        super().keyPressEvent(event)

    @property
    def raster_ready(self) -> bool:
        if self._view_family in ("image", "curve", "histogram", "meter"):
            display = self._display
            payload = self._visible_typed_payload()
            visible_revision = (
                None
                if payload is None
                else (
                    payload.viewport.viewport_revision
                    if isinstance(payload, ImagePanelPayload)
                    else payload.display_revision
                    if isinstance(payload, MeterPanelPayload)
                    else payload.viewport.display_revision
                )
            )
            return bool(
                display is not None
                and self._board_widget.front_frame is not None
                and self._pending_state is None
                and payload is not None
                and visible_revision == display.revision
            )
        return super().raster_ready

    @property
    def worker_idle(self) -> bool:
        return bool(
            self._future is None
            and self._fit_future is None
            and self._fit_overlay_pending is None
            and not self._fit_prepare_pending
            and self._deferred_typed_retry is None
            and self._deferred_fit_reload is None
            and not self._completion_handoff_active
        )

    @property
    def draft_ready(self) -> bool:
        return self._fit_draft is not None

    @property
    def saved_reference(self) -> FitResultArtifactRef | None:
        return self._saved_fit_reference

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

    def _visible_typed_payload(self) -> _TypedPanelPayload | None:
        if self._view_family == "image":
            payload = self._board_widget.visible_image_payload(_TYPED_PANEL_ID)
        elif self._view_family == "curve":
            payload = self._board_widget.visible_curve_payload(_TYPED_PANEL_ID)
        elif self._view_family == "histogram":
            payload = self._board_widget.visible_histogram_payload(_TYPED_PANEL_ID)
        elif self._view_family == "meter":
            payload = self._board_widget.visible_meter_payload(_TYPED_PANEL_ID)
        else:
            return None
        if payload is not None:
            return payload
        # A valid front is admitted before optional interaction controls.  If
        # their construction fails there is deliberately no binding, but the
        # exact current raster/payload remains visible and ready.
        frame = self._board_widget.front_frame
        if frame is None or len(frame.panels) != 1:
            return None
        candidate = frame.panels[0].display_payload
        expected_type = (
            ImagePanelPayload
            if self._view_family == "image"
            else (
                CurvePanelPayload
                if self._view_family == "curve"
                else HistogramPanelPayload
                if self._view_family == "histogram"
                else MeterPanelPayload
            )
        )
        return candidate if isinstance(candidate, expected_type) else None

    def _visible_typed_origin(self) -> PanelInteractionOrigin | None:
        if self._view_family == "image":
            return self._board_widget.visible_image_origin(_TYPED_PANEL_ID)
        if self._view_family == "curve":
            return self._board_widget.visible_curve_origin(_TYPED_PANEL_ID)
        if self._view_family == "histogram":
            return self._board_widget.visible_histogram_origin(_TYPED_PANEL_ID)
        return None

    def _visible_value_limits(self) -> tuple[float, float] | None:
        payload = self._visible_typed_payload()
        if isinstance(payload, ImagePanelPayload):
            return payload.color_limits
        if isinstance(payload, CurvePanelPayload):
            return payload.viewport.y_limits
        if isinstance(payload, HistogramPanelPayload):
            return payload.viewport.count_limits
        return None

    def _runtime_placeholders(self):
        display = self._display
        if isinstance(display, ImageDisplayState):
            payload = self._visible_typed_payload()
            if not isinstance(payload, ImagePanelPayload):
                return {}
            x_view, y_view = (
                payload.viewport.optional_coordinate_views_for_normalized_bounds()
            )
            placeholders: dict[str, str] = {}
            for limits, low, high in (
                (x_view, "x_min", "x_max"),
                (y_view, "y_min", "y_max"),
                (payload.color_limits, "color_min", "color_max"),
            ):
                resolved = runtime_range_placeholders(limits, low, high)
                if resolved is not None:
                    placeholders.update(resolved)
            return placeholders
        if isinstance(display, CurveDisplayState):
            fields = ("y_min", "y_max")
        elif isinstance(display, HistogramDisplayState):
            fields = ("count_min", "count_max")
        else:
            return {}
        return runtime_range_placeholders(self._visible_value_limits(), *fields)

    def _ensure_typed_controls(self, state: _TypedDisplayState) -> None:
        if self._edit_display is not None or self._setting_display is not None:
            if (
                self._display is not None
                and _state_intent(self._display) is not _state_intent(state)
            ):
                raise RuntimeError("typed window cannot change family")
            return
        if isinstance(state, ImageDisplayState):
            runtime_fields = (
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "color_min",
                "color_max",
            )
            subject = "image display"
            bind = None
        elif isinstance(state, CurveDisplayState):
            runtime_fields = ("y_min", "y_max")
            subject = "curve display"
            bind = self._board_widget.bind_curve_interaction
        else:
            runtime_fields = ("count_min", "count_max")
            subject = "histogram display"
            bind = self._board_widget.bind_histogram_interaction
        spec = _typed_form_spec(state)
        edit = None
        setting = None
        try:
            edit = FluentRevisionedFormEditor(
                spec,
                subject,
                runtime_placeholder_fields=runtime_fields,
                parent=self._tabs,
            )
            setting = FluentRevisionedFormEditor(
                spec,
                subject,
                runtime_placeholder_fields=runtime_fields,
                parent=self._settings_popup,
            )
            edit.setObjectName("figureViewerTypedEditEditor")
            setting.setObjectName("figureViewerTypedSettingEditor")
            edit.hide()
            edit.applyRequested.connect(
                lambda revision, values: self._apply_display_form(
                    edit,
                    revision,
                    values,
                )
            )
            setting.applyRequested.connect(
                lambda revision, values: self._apply_display_form(
                    setting,
                    revision,
                    values,
                )
            )
            edit.cancelRequested.connect(lambda: self._reload_editor(edit))
            setting.cancelRequested.connect(lambda: self._reload_editor(setting))
            self._settings_popup_layout.addWidget(setting)
            if isinstance(state, ImageDisplayState):
                payload = self._visible_typed_payload()
                if not isinstance(payload, ImagePanelPayload):
                    raise RuntimeError("IMAGE controls require one exact payload")
                self._board_widget.bind_rectangle_selector(
                    _TYPED_PANEL_ID,
                    payload.viewport,
                    self._accept_image_rectangle,
                    enabled=self._interaction_switch.isChecked(),
                    interaction_callback=self._accept_image_interaction,
                )
            else:
                assert bind is not None
                bind(
                    _TYPED_PANEL_ID,
                    self._accept_numeric_interaction,
                    enabled=self._interaction_switch.isChecked(),
                )
        except BaseException:
            if setting is not None:
                self._settings_popup_layout.removeWidget(setting)
                setting.hide()
                setting.deleteLater()
            if edit is not None:
                edit.hide()
                edit.deleteLater()
            raise
        self._edit_display = edit
        self._setting_display = setting

    def _editors(self) -> tuple[FluentRevisionedFormEditor, FluentRevisionedFormEditor]:
        if self._edit_display is None or self._setting_display is None:
            raise RuntimeError("typed controls are not admitted")
        return self._edit_display, self._setting_display

    def _sync_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        display = self._display
        if display is None:
            raise RuntimeError("typed display state is not admitted")
        sync_revisioned_form_editors(
            self._editors(),
            revision=display.revision,
            semantic_identity=display,
            values=_typed_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _sync_committed_typed_controls(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        """Finish ancillary Qt state without rolling back an admitted front."""

        try:
            self._sync_editors(
                accepted_editor=accepted_editor,
                accepted_base_revision=accepted_base_revision,
            )
            self._set_typed_controls_enabled(True)
        except BaseException as error:
            self._typed_ui_faulted = True
            try:
                self._set_typed_controls_enabled(False)
            except BaseException:
                pass
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _open_display_settings(self) -> None:
        setting = self._setting_display
        if setting is None:
            return
        self._settings_anchor.toggle(
            setting,
            prepare=lambda: self._reload_editor(setting),
        )

    def _reload_editor(self, editor: FluentRevisionedFormEditor) -> None:
        if editor not in self._editors():
            raise ValueError("typed editor does not belong to this window")
        display = self._display
        if display is None:
            raise RuntimeError("typed display state is not admitted")
        editor.load(
            revision=display.revision,
            semantic_identity=display,
            values=_typed_form_values(display),
            runtime_placeholders=self._runtime_placeholders(),
        )

    def _set_typed_controls_enabled(self, enabled: bool) -> None:
        overview = self._grid_overview
        overview_family = (
            None
            if overview is None
            else f"{overview.intent.value.lower()}-overview"
        )
        if self._view_family == overview_family:
            ready = bool(enabled and not self._typed_ui_faulted)
            self._board_widget.set_interaction_readiness(
                image=False,
                curve=False,
                histogram=False,
            )
            self._settings_button.setEnabled(False)
            self._fit_button.setEnabled(False)
            self._interaction_switch.setEnabled(False)
            self._overview_button.setEnabled(False)
            self._export_button.setEnabled(
                ready
                and self._bundle is not None
                and len(self._boards) == 1
                and self._boards[0].has_front
            )
            return
        if self._view_family == "meter":
            ready = bool(enabled and not self._typed_ui_faulted)
            self._board_widget.set_interaction_readiness(
                image=False,
                curve=False,
                histogram=False,
            )
            self._settings_button.setEnabled(False)
            self._fit_button.setEnabled(False)
            self._interaction_switch.setEnabled(False)
            self._overview_button.setEnabled(
                ready and overview is not None and self._future is None
            )
            self._export_button.setEnabled(
                ready and self._board_widget.front_frame is not None
            )
            return
        active = bool(
            enabled
            and not self._typed_ui_faulted
            and self._view_family in ("image", "curve", "histogram")
        )
        self._board_widget.set_interaction_readiness(
            image=active and self._view_family == "image",
            curve=active and self._view_family == "curve",
            histogram=active and self._view_family == "histogram",
        )
        self._settings_button.setEnabled(active)
        self._export_button.setEnabled(
            active and self._board_widget.front_frame is not None
        )
        self._fit_button.setEnabled(
            active
            and self._fit_bindings is not None
            and self._grid_overview is None
            and self._view_family in ("curve", "image")
        )
        self._interaction_switch.setEnabled(active)
        self._overview_button.setEnabled(
            active
            and overview is not None
            and self._view_family == overview.intent.value.lower()
            and self._future is None
        )
        for editor in (self._edit_display, self._setting_display):
            if editor is not None:
                editor.setEnabled(active)

    def _toggle_interaction(self, enabled: bool) -> None:
        if self._view_family not in ("image", "curve", "histogram"):
            return
        try:
            self._board_widget.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))

    def _fit_pane_is_open(self) -> bool:
        pane = self._fit_pane
        return pane is not None and self._tabs.indexOf(pane) >= 0

    def _fit_available_for_intent(self, intent: ViewIntent) -> bool:
        """Keep display-faceted grid focus outside authority Fit preparation."""

        return bool(
            self._fit_bindings is not None
            and self._grid_overview is None
            and intent in (ViewIntent.CURVE, ViewIntent.IMAGE)
        )

    def _fit_authoring_busy_kind(self) -> str | None:
        kind = self._fit_job_kind
        if kind in ("prepare", "fit", "save"):
            return kind
        if kind == "reload_saved":
            return "render"
        if (
            self._future is not None
            or self._fit_overlay_pending is not None
            or self._fit_overlay_inflight is not None
            or self._deferred_fit_reload is not None
        ):
            return "render"
        if self._fit_prepare_pending:
            return "prepare"
        return None

    def _sync_fit_authoring_busy(self) -> None:
        pane = self._fit_pane
        if pane is not None and not self._closing:
            busy = self._fit_authoring_busy_kind()
            pane.set_busy(
                busy,
                draft_ready=self._fit_draft is not None,
            )
            if self._fit_save_button is not None:
                self._fit_save_button.setEnabled(
                    busy is None and self._fit_draft is not None
                )

    def _open_fit_pane(self) -> None:
        pane = self._fit_pane
        intent = (
            ViewIntent.CURVE
            if self._view_family == "curve"
            else ViewIntent.IMAGE
            if self._view_family == "image"
            else None
        )
        if (
            pane is None
            or self._closing
            or intent is None
            or not self._fit_available_for_intent(intent)
        ):
            return
        if self._tabs.indexOf(pane) < 0:
            self._tabs.addTab(pane, "Fit")
            pane.show()
        self._tabs.setCurrentWidget(pane)
        self._start_fit_prepare()

    def _submit_fit_future(self, kind: str, function, *args) -> bool:
        if self._fit_future is not None:
            raise RuntimeError("Fit worker already has active work")
        try:
            future = _FIT_WORK_EXECUTOR.submit(function, *args)
        except BaseException as error:
            self._status.setText("FIT SUBMISSION FAILED")
            self._diagnostic.setText(error_summary(error))
            return False
        self._fit_job_kind = kind
        self._fit_future = future
        future.add_done_callback(lambda _done: self._wake.request_owner_wake())
        return True

    def _start_fit_prepare(self) -> None:
        bindings = self._fit_bindings
        pane = self._fit_pane
        if (
            bindings is None
            or pane is None
            or self._closing
            or not self._fit_axis_ids
            or not self._fit_pane_is_open()
        ):
            return
        if self._completion_handoff_active:
            self._fit_prepare_pending = True
            return
        if self._fit_future is not None:
            self._fit_prepare_pending = True
            return
        self._fit_prepare_pending = False
        pane.set_busy("prepare", draft_ready=self._fit_draft is not None)
        self._status.setText("PREPARING FIT")
        self._diagnostic.setText("")
        self._fit_job_revision = self._fit_editor_revision
        if not self._submit_fit_future(
            "prepare",
            _prepare_fit_options,
            bindings.prepare,
            self._fit_axis_ids,
            self._fit_axis_roles,
            self._fit_selection_candidate,
            bindings.allow_prepared_transform,
        ):
            self._fit_job_revision = None
            pane.set_busy(None, draft_ready=self._fit_draft is not None)

    def _discard_fit_draft(self) -> None:
        draft, self._fit_draft = self._fit_draft, None
        self._fit_draft_summary = None
        authority = self._fit_authority
        if draft is not None and authority is not None:
            authority.discard(draft)

    def _capture_fit_authoring_draft(self) -> None:
        """Retain the pane's reversible model/args text before option refresh."""

        pane = self._fit_pane
        if pane is None:
            return
        draft = pane.draft_state
        if draft is not None:
            self._fit_authoring_draft = draft

    def _advance_fit_editor(self, *, prepare: bool) -> None:
        self._capture_fit_authoring_draft()
        self._fit_editor_revision += 1
        # A saved-result reload belongs to the exact editor revision that
        # requested the save; never occupy the lane with stale authority.
        self._deferred_fit_reload = None
        self._discard_fit_draft()
        if self._fit_job_kind == "fit" and self._fit_cancelled is not None:
            self._fit_cancelled.set()
        if prepare:
            self._fit_options = {}
            pane = self._fit_pane
            if pane is not None:
                pane.clear_options()
            self._summary.setText("")
            self._fit_prepare_pending = True
            self._wake.request_owner_wake()
        # Editing model/args or accepting a selector candidate only changes the
        # local authoring draft.  Keep the last submitted overlay painted until
        # Fit or Clear is explicitly pressed; merely typing must never enqueue
        # a raster job.  The old execution is no longer saveable against the
        # changed draft, so reconcile the buttons synchronously.
        self._sync_fit_authoring_busy()

    def _fit_editor_changed(self, _pane_revision: int) -> None:
        if self._closing:
            return
        self._advance_fit_editor(prepare=False)
        self._status.setText("FIT DRAFT CHANGED")
        self._diagnostic.setText(
            "The visible model or constraints changed; press Fit to submit the "
            "new authoritative draft."
        )

    def _reject_fit_request(self, diagnostic: str) -> None:
        self._status.setText("FIT REQUEST INVALID")
        self._diagnostic.setText(str(diagnostic))

    def _start_fit(self, _pane_revision: int, spec: FitSpec) -> None:
        pane = self._fit_pane
        authority = self._fit_authority
        bindings = self._fit_bindings
        if (
            pane is None
            or authority is None
            or bindings is None
            or self._closing
            or self._fit_future is not None
            or self._deferred_fit_reload is not None
        ):
            return
        try:
            current = pane.current_option()
            if not isinstance(spec, FitSpec):
                raise TypeError("Fit pane emitted another request type")
            if (
                spec.model_id != current.spec.model_id
                or spec.input_schema_fingerprint
                != current.spec.input_schema_fingerprint
                or spec.committed_transform != current.spec.committed_transform
                or spec.fit_axis_ids != current.spec.fit_axis_ids
                or spec.batch_axis_ids != current.spec.batch_axis_ids
                or spec.numeric_policy != current.spec.numeric_policy
            ):
                raise ValueError("Fit request differs from the prepared authority draft")
        except BaseException as error:
            self._status.setText("FIT REQUEST INVALID")
            self._diagnostic.setText(error_summary(error))
            return

        self._discard_fit_draft()
        self._fit_cancelled = threading.Event()
        self._fit_job_revision = self._fit_editor_revision
        deadline = time.monotonic() + bindings.timeout_seconds
        pane.set_busy("fit", draft_ready=False)
        self._status.setText("FITTING")
        self._summary.setText(pane.axis_summary_text)
        self._diagnostic.setText("")
        self._queue_fit_overlay(None, None)
        if not self._submit_fit_future(
            "fit",
            _execute_fit_draft,
            authority,
            spec,
            deadline,
            self._cancelled,
            self._fit_cancelled,
        ):
            self._fit_job_revision = None
            self._fit_cancelled = None
            pane.set_busy(None, draft_ready=False)

    def _save_fit(self) -> None:
        pane = self._fit_pane
        authority = self._fit_authority
        bindings = self._fit_bindings
        draft = self._fit_draft
        if (
            pane is None
            or authority is None
            or bindings is None
            or draft is None
            or self._fit_future is not None
            or self._closing
        ):
            return
        destination = None
        if bindings.save_requires_path:
            destination = self._fit_save_path
            if destination is None:
                selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save fitted DataFigure archive",
                    "",
                    "DataFigure archive (*.npz)",
                )
                if not selected:
                    return
                destination = Path(selected)
                if not destination.suffix:
                    destination = destination.with_suffix(".npz")
        self._fit_save_inflight = draft
        self._fit_job_revision = self._fit_editor_revision
        pane.set_busy("save", draft_ready=True)
        if self._fit_save_button is not None:
            self._fit_save_button.setEnabled(False)
        self._status.setText("SAVING FIT")
        self._summary.setText(
            "Saving and reopening the fitted DataFigure archive…"
            if bindings.save_requires_path
            else "Publishing and reopening the exact Fit artifact…"
        )
        self._diagnostic.setText("")
        if not self._submit_fit_future(
            "save",
            authority.save,
            draft,
            destination,
            self._display,
        ):
            self._fit_save_inflight = None
            self._fit_job_revision = None
            pane.set_busy(None, draft_ready=True)
            if self._fit_save_button is not None:
                self._fit_save_button.setEnabled(True)

    def _clear_fit(self) -> None:
        if self._closing:
            return
        fitting = self._fit_future is not None and self._fit_job_kind == "fit"
        if self._fit_future is not None and not fitting:
            return
        self._fit_editor_revision += 1
        if fitting and self._fit_cancelled is not None:
            self._fit_cancelled.set()
        self._discard_fit_draft()
        self._queue_fit_overlay(None, None)
        pane = self._fit_pane
        if pane is not None:
            pane.set_busy("fit" if fitting else None, draft_ready=False)
        if self._fit_save_button is not None:
            self._fit_save_button.setEnabled(False)
        self._status.setText("CLEARING FIT" if fitting else "FIT CLEARED")
        self._summary.setText("Source view preserved; selection remains a draft candidate")
        self._diagnostic.setText("")

    def _queue_fit_overlay(
        self,
        result: FitResultBatch | None,
        result_identity: str | None,
    ) -> None:
        if self._fit_overlay_renderer is None or self._display is None:
            return
        request = _FitOverlayRequest(
            self._fit_editor_revision,
            result,
            result_identity,
        )
        self._fit_overlay_desired = request
        if (
            self._fit_overlay_inflight is None
            and self._fit_overlay_pending is None
            and self._visible_fit_result_identity == result_identity
        ):
            return
        self._fit_overlay_pending = request
        self._start_pending_fit_overlay()

    def _start_pending_fit_overlay(self) -> None:
        request = self._fit_overlay_pending
        display = self._display
        renderer = self._fit_overlay_renderer
        if (
            request is None
            or display is None
            or renderer is None
            or self._future is not None
            or self._closing
            or self._completion_handoff_active
        ):
            return
        candidate = replace(display, revision=display.revision + 1)
        self._request_revision += 1
        self._active_kind = "fit_overlay"
        self._pending_state = candidate
        self._fit_overlay_inflight = request
        self._fit_overlay_pending = None
        self._status.setText("RENDERING FIT OVERLAY")
        self._set_typed_controls_enabled(False)
        previous_scale = (
            display.count_scale
            if isinstance(display, HistogramDisplayState)
            else None
        )
        if not self._submit_future(
            renderer,
            request.result,
            request.result_identity,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._cancelled,
        ):
            self._fit_overlay_inflight = None
            self._pending_state = None
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        else:
            self._sync_fit_authoring_busy()

    @staticmethod
    def _curve_span_for_selection(
        selection: Selection,
        payload: CurvePanelPayload,
    ) -> tuple[float, float]:
        axis = payload.viewport.x_axis
        matches = tuple(term for term in selection.terms if term.axis_id == axis.axis_id)
        if len(matches) != 1:
            raise ValueError("curve Fit selection does not name the displayed x axis")
        term = matches[0]
        if isinstance(term, CoordinateRangeSelection):
            return float(term.lower), float(term.upper)
        if isinstance(term, IndexRangeSelection):
            coordinates = axis.coordinates
            if term.stop > len(coordinates):
                raise IndexError("curve Fit index range exceeds displayed coordinates")
            low = float(coordinates[term.start])
            high = float(coordinates[term.stop - 1])
            return (min(low, high), max(low, high))
        raise ValueError("curve Fit selection must preserve a non-empty range")

    def _reapply_fit_candidate(self) -> None:
        selection = self._fit_selection_candidate
        if selection is None:
            return
        origin = self._visible_typed_origin()
        payload = self._visible_typed_payload()
        if origin is None or payload is None:
            return
        if isinstance(payload, CurvePanelPayload):
            self._board_widget.set_curve_range_candidate(
                self._curve_span_for_selection(selection, payload),
                panel_id=_TYPED_PANEL_ID,
            )
        elif isinstance(payload, ImagePanelPayload):
            self._board_widget.set_selector_applied_selection(
                selection,
                panel_id=_TYPED_PANEL_ID,
            )

    def _accept_fit_selection_candidate(
        self,
        selection: Selection | None,
    ) -> None:
        intent = (
            ViewIntent.CURVE
            if self._view_family == "curve"
            else ViewIntent.IMAGE
            if self._view_family == "image"
            else None
        )
        if intent is None or not self._fit_available_for_intent(intent):
            return
        self._fit_selection_candidate = selection
        self._advance_fit_editor(prepare=self._fit_pane_is_open())
        self._status.setText("FIT SELECTION READY" if selection is not None else "FIT SELECTION CLEARED")
        self._diagnostic.setText(
            "Press Fit to submit this named-axis selection as authority."
            if selection is not None
            else "Fit will use the full named dataset; current zoom is not authority."
        )

    def _apply_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        if editor not in self._editors():
            raise ValueError("typed editor does not belong to this window")
        try:
            display = self._display
            if display is None:
                raise RuntimeError("typed display state is not admitted")
            if self._future is not None or self._closing:
                raise RuntimeError("typed display work is already active")
            if base_revision != display.revision:
                raise RuntimeError(
                    f"typed draft r{base_revision} is stale; "
                    f"current revision is r{display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("typed display form must emit one exact mapping")
            candidate = _typed_state_from_form(
                display,
                values,
                current_value_limits=self._visible_value_limits(),
            )
            self._start_typed_render(
                candidate,
                editor=editor,
                editor_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Typed display edit rejected: {error_summary(error)}"
            )

    def _accept_image_rectangle(self, gesture: RectangleGesture) -> None:
        display = self._display
        origin = self._visible_typed_origin()
        if not isinstance(display, ImageDisplayState) or origin is None:
            raise RuntimeError("IMAGE rectangle has no current exact front")
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("IMAGE rectangle must be RectangleGesture")
        if (
            gesture.panel_id != _TYPED_PANEL_ID
            or (
                gesture.board_id,
                gesture.layout_generation,
                gesture.sequence,
                gesture.source_identity,
                gesture.viewport_revision,
            )
            != (
                origin.board_id,
                origin.layout_generation,
                origin.sequence,
                origin.source_identity,
                display.revision,
            )
        ):
            raise RuntimeError("IMAGE rectangle origin is stale")
        selection = None
        if (
            gesture.normalized_bounds is not None
            and self._fit_available_for_intent(ViewIntent.IMAGE)
        ):
            # Resolve authority while QtRasterBoard still holds the exact front
            # on which this gesture was completed.  Painting the candidate first
            # would release that proof and make a later conversion racy.
            selection = self._board_widget.selection_for_rectangle_gesture(gesture)
        self._board_widget.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=_TYPED_PANEL_ID,
        )
        if gesture.normalized_bounds is None:
            if self._fit_available_for_intent(ViewIntent.IMAGE):
                self._accept_fit_selection_candidate(None)
            else:
                self._diagnostic.setText("")
            return
        if selection is not None:
            self._accept_fit_selection_candidate(selection)
            return
        left, top, right, bottom = gesture.normalized_bounds
        self._diagnostic.setText(
            "DISPLAY ONLY rectangle "
            f"({left:.6g}, {top:.6g})..({right:.6g}, {bottom:.6g})"
        )

    def _accept_image_interaction(self, command: ImageInteractionCommit) -> None:
        display = self._display
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("unknown IMAGE interaction command")
        if not isinstance(display, ImageDisplayState):
            raise RuntimeError("IMAGE interaction belongs to another family")
        origin = command.origin
        if (
            origin.panel_id != _TYPED_PANEL_ID
            or self._visible_typed_origin() != origin
            or origin.presentation.panel_revision != display.revision
        ):
            raise RuntimeError("IMAGE interaction origin is stale")
        if isinstance(command, ImageViewportCommit):
            candidate = image_display_for_viewport(display, command.viewport)
        else:
            candidate = replace(
                display,
                revision=display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
        self._start_typed_render(candidate, origin=origin)

    def _accept_numeric_interaction(
        self,
        command: CurveInteractionIntent | HistogramInteractionIntent,
    ) -> None:
        display = self._display
        is_curve = isinstance(command, (CurveViewportCommit, CurveRangeGesture))
        is_histogram = isinstance(
            command,
            (
                HistogramViewportCommit,
                HistogramRangeGesture,
                HistogramThresholdCommit,
            ),
        )
        if not (is_curve or is_histogram):
            raise TypeError("unknown numeric interaction command")
        if (
            display is None
            or is_curve != isinstance(display, CurveDisplayState)
            or is_histogram != isinstance(display, HistogramDisplayState)
        ):
            raise RuntimeError("numeric interaction belongs to another family")
        origin = command.origin
        if (
            origin.panel_id != _TYPED_PANEL_ID
            or self._visible_typed_origin() != origin
            or origin.presentation.panel_revision != display.revision
        ):
            raise RuntimeError("numeric interaction origin is stale")
        if isinstance(command, (CurveRangeGesture, HistogramRangeGesture)):
            selection = None
            if (
                is_curve
                and self._fit_available_for_intent(ViewIntent.CURVE)
                and command.x_span is not None
            ):
                # As with IMAGE, the exact held origin must be consumed before
                # set_curve_range_candidate finalizes the display-only gesture.
                selection = self._board_widget.selection_for_curve_range_gesture(
                    command
                )
            setter = (
                self._board_widget.set_curve_range_candidate
                if is_curve
                else self._board_widget.set_histogram_range_candidate
            )
            setter(command.x_span, panel_id=_TYPED_PANEL_ID)
            if is_curve and self._fit_available_for_intent(ViewIntent.CURVE):
                self._accept_fit_selection_candidate(selection)
                return
            self._diagnostic.setText(
                ""
                if command.x_span is None
                else (
                    "DISPLAY ONLY x span "
                    f"{command.x_span[0]:.6g}..{command.x_span[1]:.6g}"
                )
            )
            return
        if isinstance(command, HistogramThresholdCommit):
            # A live threshold drag step (the reference's per-motion
            # DragVLine callback): author the new cut set and re-render.
            self._start_typed_render(
                histogram_display_with_thresholds(display, command.thresholds),
                origin=origin,
            )
            return
        if command.viewport.display_revision != display.revision + 1:
            raise RuntimeError("numeric viewport commit must advance once")
        self._start_typed_render(
            _typed_state_with_x_view(display, command.viewport.x_limits),
            origin=origin,
        )

    def _start_typed_render(
        self,
        candidate: _TypedDisplayState,
        *,
        editor: FluentRevisionedFormEditor | None = None,
        editor_revision: int | None = None,
        origin: PanelInteractionOrigin | None = None,
    ) -> None:
        if self._completion_handoff_active:
            self._deferred_typed_retry = (
                candidate,
                editor,
                editor_revision,
                origin,
            )
            return
        display = self._display
        payload = self._visible_typed_payload()
        if display is None or payload is None:
            raise RuntimeError("typed figure is not ready")
        if self._future is not None or self._closing:
            raise RuntimeError("typed render is already active")
        if _state_intent(candidate) is not _state_intent(display):
            raise TypeError("candidate belongs to another typed family")
        if candidate == display:
            if origin is not None:
                raise ValueError("typed interaction cannot commit a no-op")
            self._sync_editors(
                accepted_editor=editor,
                accepted_base_revision=editor_revision,
            )
            return
        if candidate.revision != display.revision + 1:
            raise ValueError("typed display revision must advance once")
        self._request_revision += 1
        self._active_kind = "typed"
        self._pending_state = candidate
        self._pending_origin = origin
        self._pending_editor = editor
        self._pending_editor_revision = editor_revision
        self._status.setText(f"RENDERING {self._view_family.upper()}")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        previous_scale = (
            display.count_scale
            if isinstance(display, HistogramDisplayState)
            else None
        )
        overlay = self._fit_overlay_desired
        overlay_result = None if overlay is None else overlay.result
        overlay_identity = None if overlay is None else overlay.result_identity
        self._fit_overlay_inflight = overlay
        submitted = self._submit_future(
            self._typed_renderer,
            overlay_result,
            overlay_identity,
            candidate,
            self._visible_value_limits(),
            display.relim_mode,
            previous_scale,
            self._request_revision,
            self._cancelled,
        )
        if not submitted:
            self._fit_overlay_inflight = None
            self._discard_pending_typed()
        else:
            self._sync_fit_authoring_busy()

    def _discard_pending_typed(self) -> None:
        origin = self._pending_origin
        family = self._view_family
        self._fit_overlay_inflight = None
        self._pending_state = None
        self._pending_origin = None
        self._pending_editor = None
        self._pending_editor_revision = None
        self._active_kind = None
        cleanup_errors = []
        if origin is not None:
            try:
                discard = {
                    "image": self._board_widget.discard_pending_image_interaction,
                    "curve": self._board_widget.discard_pending_curve_interaction,
                    "histogram": (
                        self._board_widget.discard_pending_histogram_interaction
                    ),
                }.get(family)
                if discard is None:
                    raise RuntimeError("pending interaction has no typed family")
                discard(origin)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if family in ("image", "curve", "histogram"):
            try:
                self._sync_editors()
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                cleanup_errors.append(error_summary(error))
        if cleanup_errors:
            existing = self._diagnostic.text()
            suffix = "cleanup: " + " | ".join(cleanup_errors)
            self._diagnostic.setText(suffix if not existing else f"{existing} | {suffix}")

    @staticmethod
    def _validate_authored_front(
        front: _TypedFigureFront,
        expected_state: _TypedDisplayState,
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        # The worker validates the semantic payload.  Qt repeats only compact
        # display fields and immutable object identities; it never rescans a
        # coordinate vector or trusts a token detached from the current frame.
        if (
            front.state != expected_state
            or front.intent is not _state_intent(expected_state)
        ):
            raise ValueError("typed worker returned conflicting authored state")
        payload = front.frame.panels[0].display_payload
        assert isinstance(
            payload,
            (
                ImagePanelPayload,
                CurvePanelPayload,
                HistogramPanelPayload,
                MeterPanelPayload,
            ),
        )
        _validate_rendered_authored_payload(
            payload,
            expected_state,
            front.fit_result_identity,
        )
        current = _build_typed_front_contract(front.intent, front.frame)
        frozen_identity, frozen_data = front.data_contract
        identity, exact_data = current
        if identity != frozen_identity:
            raise ValueError("typed worker changed frozen source provenance")
        if not _same_exact_data_owners(exact_data, frozen_data):
            raise ValueError("typed worker changed frozen evaluated data")
        return current

    def _present_typed_front(
        self,
        front: _TypedFigureFront,
        *,
        expected_state: _TypedDisplayState,
        request_revision: int,
    ) -> None:
        request_revision = nonnegative_integer(
            request_revision,
            "typed request revision",
        )
        if front.frame.sequence != request_revision:
            raise ValueError("typed worker returned another request sequence")
        contract = self._validate_authored_front(front, expected_state)
        expected_contract = self._typed_contract
        if expected_contract is not None:
            expected_identity, expected_data = expected_contract
            identity, exact_data = contract
            if identity[0] != expected_identity[0]:
                raise ValueError("typed worker changed frozen source provenance")
            if not _same_exact_data_owners(exact_data, expected_data):
                raise ValueError("typed worker changed frozen evaluated data")

        self._board_widget.present(front.frame)
        # The admitted board front is the transaction boundary.  Commit the
        # exact Figure/authored state/contract before cache release or any
        # optional Qt chrome work.
        if expected_contract is None:
            self._typed_contract = contract
        self._display = expected_state
        self._visible_figure = front.figure
        self._view_family = front.intent.value.lower()
        self._fit_axis_ids = front.fit_axis_ids
        self._fit_axis_roles = front.axis_roles
        self._visible_fit_result_identity = front.fit_result_identity
        self._visible_transient_fit_result_owner = (
            front.transient_fit_result_owner
        )
        if self._typed_front_committed is not None:
            self._typed_front_committed(
                front.release_initial_canonical_on_commit
            )
        if front.intent is not ViewIntent.METER and self._fit_overlay_desired is None:
            self._fit_overlay_desired = _FitOverlayRequest(
                self._fit_editor_revision,
                None,
                front.fit_result_identity,
            )
        # Page/chrome and controls are ancillary to the already-admitted
        # immutable data front.  Their faults can disable UI, never roll it back.
        try:
            if self._grid_overview is not None and len(self._boards) == 1:
                overview_index = self._tabs.indexOf(
                    self._tab_host_for_board(self._boards[0])
                )
                if overview_index >= 0:
                    self._tabs.removeTab(overview_index)
            if not self._typed_pages_admitted:
                if self._grid_overview is None:
                    self._retire_tab_pages()
                self._tabs.addTab(self._typed_page, front.intent.value.title())
                self._tabs.tabBar().setVisible(False)
                self._typed_page.show()
                self._typed_pages_admitted = True
            self._tabs.setCurrentWidget(self._typed_page)
            fit_capable = self._fit_available_for_intent(front.intent)
            self._mode.setText(
                f"EXACT {front.intent.value} · DISPLAY ONLY"
                if front.intent is ViewIntent.METER
                else f"EXACT {front.intent.value} · INTERACTIVE"
                + ("" if fit_capable else " · DISPLAY ONLY")
            )
            self._status.setText("READY")
            self._summary.setText(front.summary)
            self._diagnostic.setText("")
        except BaseException as error:
            self._typed_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        if front.intent is ViewIntent.METER:
            for widget in (
                self._interaction_switch,
                self._settings_button,
                self._fit_button,
            ):
                widget.hide()
            self._export_button.show()
            self._overview_button.setVisible(self._grid_overview is not None)
            self._tabs.tabBar().setVisible(False)
            self._set_typed_controls_enabled(True)
            return
        try:
            self._ensure_typed_controls(expected_state)
            edit, _setting = self._editors()
            if self._tabs.indexOf(edit) < 0:
                self._tabs.addTab(edit, "Edit")
            self._tabs.tabBar().setVisible(True)
            edit.show()
            for widget in (
                self._interaction_switch,
                self._settings_button,
                self._export_button,
            ):
                widget.show()
            self._overview_button.setVisible(self._grid_overview is not None)
            if self._fit_available_for_intent(front.intent):
                self._fit_button.show()
                if not self._fit_initial_selection_consumed:
                    initial = self._fit_bindings.initial_selection
                    self._fit_initial_selection_consumed = True
                    origin = self._visible_typed_origin()
                    if initial is not None and origin is not None:
                        self._fit_selection_candidate = initial
                self._reapply_fit_candidate()
                if (
                    self._fit_bindings.open_fit
                    and not self._fit_auto_open_consumed
                ):
                    self._fit_auto_open_consumed = True
                    self._open_fit_pane()
            else:
                self._fit_button.hide()
                if (
                    self._grid_overview is not None
                    and self._fit_bindings is not None
                    and front.intent is ViewIntent.CURVE
                ):
                    self._diagnostic.setText(
                        "Fit unavailable: grid focus is a display projection; "
                        "Fit authority requires an axis-complete source view."
                    )
        except BaseException as error:
            self._typed_ui_faulted = True
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("FIGURE CANCELLED")
                if kind == "typed":
                    return_to_overview = (
                        self._discard_grid_focus_sequence == self._request_revision
                    )
                    self._discard_grid_focus_sequence = None
                    self._discard_pending_typed()
                    if return_to_overview:
                        self._show_grid_overview()
                elif kind == "grid_focus":
                    self._grid_focus_pending = None
                    self._discard_grid_focus_sequence = None
                    self._active_kind = None
                    self._status.setText("READY")
                    self._diagnostic.setText("")
                    self._set_typed_controls_enabled(True)
                elif kind == "initial":
                    self._active_kind = None
                    self._emit_initial_failed("initial figure render was cancelled")
                else:
                    self._active_kind = None
        except BaseException as error:
            if not self._closing:
                self._reject_completed_work(kind, error)
        else:
            if self._closing:
                return
            try:
                self._accept_completed_work(kind, result)
            except BaseException as error:
                self._reject_completed_work(kind, error)

    def _accept_finished_fit_future(self, future: Future) -> None:
        kind, self._fit_job_kind = self._fit_job_kind, None
        job_revision, self._fit_job_revision = self._fit_job_revision, None
        fit_cancelled, self._fit_cancelled = self._fit_cancelled, None
        save_inflight, self._fit_save_inflight = self._fit_save_inflight, None
        close_after_save = bool(
            kind == "save" and self._close_deferred_during_fit_save
        )
        if kind == "save":
            self._close_deferred_during_fit_save = False
        pane = self._fit_pane
        completed_draft: FitDraftResult | None = None
        completed_summary: str | None = None
        try:
            result = future.result()
            if kind == "prepare":
                options = (
                    () if job_revision != self._fit_editor_revision else tuple(result)
                )
            elif kind == "fit":
                if (
                    not isinstance(result, tuple)
                    or len(result) != 2
                    or not isinstance(result[0], FitDraftResult)
                    or not isinstance(result[1], str)
                ):
                    raise TypeError("Fit worker returned another draft type")
                (
                    completed_draft,
                    completed_summary,
                ) = result
                if fit_cancelled is not None and fit_cancelled.is_set():
                    raise CancelledError()
            elif kind == "save":
                if not isinstance(result, _FitSaveReceipt):
                    raise TypeError("Fit save returned another receipt type")
                receipt = result
            elif kind == "reload_saved":
                if not isinstance(result, FitResultBatch):
                    raise TypeError("saved Fit reload returned another result type")
                reloaded_result = result
            else:
                raise RuntimeError(f"unknown Fit worker completion {kind!r}")
        except (CancelledError, FitCancelled):
            if completed_draft is not None and self._fit_authority is not None:
                self._fit_authority.discard(completed_draft)
            if not self._closing:
                self._status.setText("FIT CANCELLED")
                self._diagnostic.setText("")
        except FitDeadlineExceeded as error:
            if not self._closing:
                self._status.setText("FIT DEADLINE EXCEEDED")
                self._diagnostic.setText(error_summary(error))
        except BaseException as error:
            if completed_draft is not None and self._fit_authority is not None:
                self._fit_authority.discard(completed_draft)
            if not self._closing:
                label = {
                    "prepare": "FIT PREPARATION FAILED",
                    "fit": "FIT FAILED",
                    "save": "FIT SAVE FAILED",
                    "reload_saved": "FIT SAVED · REOPEN FAILED",
                }.get(kind, "FIT WORK FAILED")
                self._status.setText(label)
                self._diagnostic.setText(error_summary(error))
                if kind == "save" and save_inflight is not None:
                    if job_revision == self._fit_editor_revision:
                        self._fit_draft = save_inflight
                    else:
                        if self._fit_authority is not None:
                            self._fit_authority.discard(save_inflight)
                        self._status.setText("FIT SAVE FAILED · EDITOR CHANGED")
        else:
            if self._closing:
                if completed_draft is not None and self._fit_authority is not None:
                    self._fit_authority.discard(completed_draft)
            elif kind == "prepare":
                if job_revision != self._fit_editor_revision:
                    self._fit_prepare_pending = True
                else:
                    assert pane is not None
                    model_ids = {option.spec.model_id for option in options}
                    previous = self._fit_authoring_draft
                    preferred = (
                        previous.selected_model_id
                        if previous is not None
                        and previous.selected_model_id in model_ids
                        else pane.model_combo.currentData()
                    )
                    if preferred not in model_ids:
                        preferred = (
                            None
                            if self._fit_bindings is None
                            else self._fit_bindings.selected_model
                        )
                    if preferred not in model_ids:
                        preferred = None
                    authoring = reconcile_fit_authoring_draft(
                        options,
                        previous,
                        selected_model=preferred,
                    )
                    pane.install_options(
                        options,
                        selected_model=authoring.selected_model_id,
                    )
                    pane.set_draft_state(authoring)
                    self._fit_authoring_draft = authoring
                    self._fit_options = {
                        option.spec.model_id: option for option in options
                    }
                    self._status.setText("FIT READY")
                    self._summary.setText(pane.axis_summary_text)
                    self._diagnostic.setText(
                        "Fit submits the displayed named axes; zoom is presentation only."
                    )
            elif kind == "fit":
                assert completed_draft is not None
                assert completed_summary is not None
                if job_revision != self._fit_editor_revision:
                    assert self._fit_authority is not None
                    self._fit_authority.discard(completed_draft)
                    self._status.setText("STALE FIT DISCARDED")
                else:
                    self._fit_draft = completed_draft
                    self._fit_draft_summary = completed_summary
                    identity = (
                        f"draft-fit:r{job_revision}:g{completed_draft.generation}"
                    )
                    self._queue_fit_overlay(
                        completed_draft.result,
                        identity,
                    )
                    self._status.setText("DRAFT FIT READY")
                    self._summary.setText(completed_summary)
                    self._diagnostic.setText("")
            elif kind == "save":
                # The persistence receipt is accepted before any later overlay
                # rendering.  Nothing after this assignment may erase it.
                self._saved_fit_receipt = receipt
                artifact_reference = receipt.artifact_reference
                if artifact_reference is not None and not isinstance(
                    artifact_reference,
                    FitResultArtifactRef,
                ):
                    raise TypeError(
                        "Fit save receipt carries another artifact reference type"
                    )
                self._saved_fit_reference = artifact_reference
                saved_path = getattr(receipt.handle, "path", None)
                if saved_path is not None:
                    self._fit_save_path = Path(saved_path)
                self._fit_draft = None
                self._fit_draft_summary = None
                self.fitSaved.emit(receipt.handle)
                if job_revision != self._fit_editor_revision:
                    self._status.setText("FIT SAVED · EDITOR CHANGED")
                elif receipt.reloaded_result is not None and not close_after_save:
                    self._queue_fit_overlay(
                        receipt.reloaded_result,
                        receipt.identity,
                    )
                    self._status.setText("FIT SAVED")
                else:
                    self._status.setText("FIT SAVED · REOPENING")
                self._summary.setText(receipt.summary)
                bindings = self._fit_bindings
                if close_after_save:
                    self._diagnostic.setText(
                        "Saved Fit identity accepted; completing the deferred close."
                    )
                elif job_revision != self._fit_editor_revision:
                    self._diagnostic.setText(
                        "Fit is saved; stale editor authority was not reloaded."
                    )
                elif receipt.reloaded_result is not None and not close_after_save:
                    self._diagnostic.setText("")
                elif bindings is None:
                    self._diagnostic.setText(
                        "Fit is saved; no reload capability is available."
                    )
                else:
                    self._deferred_fit_reload = (receipt, job_revision)
                    if pane is not None:
                        pane.set_busy("render", draft_ready=False)
            elif kind == "reload_saved":
                receipt = self._saved_fit_receipt
                if receipt is None:
                    raise RuntimeError("saved Fit reload lost its persistence receipt")
                if job_revision == self._fit_editor_revision:
                    self._queue_fit_overlay(
                        reloaded_result,
                        receipt.identity,
                    )
                    self._status.setText("FIT SAVED")
                    self._diagnostic.setText("")
                else:
                    self._status.setText("FIT SAVED · EDITOR CHANGED")
                    self._diagnostic.setText(
                        "Saved artifact preserved; its overlay was not applied to a newer draft."
                    )
        if close_after_save and not self._closing:
            self.shutdown()
            return
        if pane is not None and not self._closing:
            self._sync_fit_authoring_busy()

    def _accept_completed_work(self, kind: str | None, result: object) -> None:
        if kind == "initial":
            if isinstance(result, EncodedRasterDocument):
                self._view_family = "encoded"
                self._set_typed_controls_enabled(False)
                self._mode.setText("FROZEN DATA FIGURE · DISPLAY ONLY")
                if not self._present_bundle(result):
                    raise RuntimeError(
                        self._diagnostic.text()
                        or "initial encoded figure could not be presented"
                    )
            elif isinstance(result, _TypedFigureFront):
                initial_display = self._initial_display
                if initial_display is None:
                    initial_display = _default_typed_state(result.intent)
                self._present_typed_front(
                    result,
                    expected_state=initial_display,
                    request_revision=self._request_revision,
                )
                if (
                    result.intent is not ViewIntent.METER
                    and not self._typed_ui_faulted
                ):
                    self._sync_committed_typed_controls()
            elif isinstance(result, _TypedGridOverview):
                self._present_grid_overview(result)
            else:
                raise TypeError("initial figure worker returned another result")
            self._active_kind = None
            self._emit_initial_ready()
            return
        if kind == "grid_focus":
            if not isinstance(result, _TypedFigureFront):
                raise TypeError("typed grid focus worker returned another result")
            pending = self._grid_focus_pending
            if pending is None:
                raise RuntimeError("typed grid focus completed without a pending panel")
            discarded = self._discard_grid_focus_sequence == self._request_revision
            self._grid_focus_pending = None
            self._discard_grid_focus_sequence = None
            self._active_kind = None
            if discarded:
                self._status.setText("READY")
                self._diagnostic.setText("")
                self._set_typed_controls_enabled(True)
                return
            self._present_typed_front(
                result,
                expected_state=pending.display,
                request_revision=self._request_revision,
            )
            if (
                result.intent is not ViewIntent.METER
                and not self._typed_ui_faulted
            ):
                self._sync_committed_typed_controls()
            return
        if kind == "typed":
            if not isinstance(result, _TypedFigureFront):
                raise TypeError("typed worker returned another result")
            pending = self._pending_state
            editor = self._pending_editor
            editor_revision = self._pending_editor_revision
            if pending is None:
                raise RuntimeError("typed worker completed without pending state")
            if self._discard_grid_focus_sequence == self._request_revision:
                self._discard_grid_focus_sequence = None
                self._fit_overlay_inflight = None
                self._discard_pending_typed()
                self._show_grid_overview()
                return
            rendered_overlay = self._fit_overlay_inflight
            self._fit_overlay_inflight = None
            if not _same_fit_overlay_request(
                rendered_overlay,
                self._fit_overlay_desired,
            ):
                if _same_fit_overlay_request(
                    self._fit_overlay_pending,
                    self._fit_overlay_desired,
                ):
                    self._fit_overlay_pending = None
                editor = self._pending_editor
                editor_revision = self._pending_editor_revision
                origin = self._pending_origin
                self._active_kind = None
                self._start_typed_render(
                    pending,
                    editor=editor,
                    editor_revision=editor_revision,
                    origin=origin,
                )
                return
            expected_overlay_identity = (
                None
                if rendered_overlay is None
                else rendered_overlay.result_identity
            )
            if result.fit_result_identity != expected_overlay_identity:
                raise ValueError(
                    "typed worker returned another Fit result identity"
                )
            self._present_typed_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            self._pending_state = None
            self._pending_origin = None
            self._pending_editor = None
            self._pending_editor_revision = None
            self._active_kind = None
            if not self._typed_ui_faulted:
                self._sync_committed_typed_controls(
                    accepted_editor=editor,
                    accepted_base_revision=editor_revision,
                )
            return
        if kind == "fit_overlay":
            if not isinstance(result, _TypedFigureFront):
                raise TypeError("Fit overlay worker returned another result")
            pending = self._pending_state
            request = self._fit_overlay_inflight
            if pending is None or request is None:
                raise RuntimeError("Fit overlay completed without an admitted request")
            self._pending_state = None
            self._fit_overlay_inflight = None
            self._active_kind = None
            if not _same_fit_overlay_request(
                request,
                self._fit_overlay_desired,
            ):
                self._set_typed_controls_enabled(True)
                return
            if result.fit_result_identity != request.result_identity:
                raise ValueError("Fit overlay worker returned another result identity")
            self._present_typed_front(
                result,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            if self._fit_job_kind == "fit":
                self._status.setText("FITTING")
            elif self._fit_draft is not None:
                self._status.setText("DRAFT FIT READY")
                if self._fit_draft_summary is None:
                    raise RuntimeError("Fit draft summary was not returned by its worker")
                self._summary.setText(self._fit_draft_summary)
            elif self._saved_fit_receipt is not None and request.result is not None:
                self._status.setText("FIT SAVED")
            else:
                self._status.setText("READY")
            self._set_typed_controls_enabled(True)
            return
        if kind == "export":
            if not isinstance(result, tuple) or len(result) != 2:
                raise TypeError("typed export returned another result")
            revision, destination = result
            if revision != self._request_revision:
                raise ValueError("typed export revision is stale")
            self._active_kind = None
            self._status.setText("READY")
            self._diagnostic.setText(f"Exported {destination}")
            try:
                self._set_typed_controls_enabled(True)
            except BaseException as error:
                self._typed_ui_faulted = True
                self._status.setText("TYPED CONTROLS FAILED")
                self._diagnostic.setText(
                    f"Exported {destination} | {error_summary(error)}"
                )
            return
        raise RuntimeError("figure window completed unknown work")

    def _reject_completed_work(
        self,
        kind: str | None,
        error: BaseException,
    ) -> None:
        if kind == "typed":
            return_to_overview = (
                self._discard_grid_focus_sequence == self._request_revision
            )
            self._discard_grid_focus_sequence = None
            family = (self._view_family or "typed").upper()
            self._status.setText(f"{family} DISPLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._fit_overlay_inflight = None
            self._discard_pending_typed()
            if return_to_overview:
                self._show_grid_overview()
        elif kind == "grid_focus":
            discarded = self._discard_grid_focus_sequence == self._request_revision
            intent = (
                None
                if self._grid_focus_pending is None
                else _state_intent(self._grid_focus_pending.display)
            )
            self._grid_focus_pending = None
            self._discard_grid_focus_sequence = None
            self._active_kind = None
            label = "GRID" if intent is None else intent.value
            self._status.setText("READY" if discarded else f"{label} FOCUS FAILED")
            self._diagnostic.setText("" if discarded else error_summary(error))
            self._set_typed_controls_enabled(True)
        elif kind == "fit_overlay":
            self._status.setText("FIT OVERLAY FAILED")
            self._diagnostic.setText(error_summary(error))
            self._pending_state = None
            self._fit_overlay_inflight = None
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        elif kind == "export":
            self._status.setText("TYPED EXPORT FAILED")
            self._diagnostic.setText(error_summary(error))
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        elif kind == "initial":
            self._status.setText("FIGURE FAILED")
            self._summary.setText("No raster was admitted")
            detail = error_summary(error)
            self._diagnostic.setText(detail)
            self._active_kind = None
            self._emit_initial_failed(detail)
        else:
            self._status.setText("FIGURE FAILED")
            self._summary.setText("No raster was admitted")
            self._diagnostic.setText(error_summary(error))
            self._active_kind = None

    def _choose_export(self) -> None:
        overview = self._grid_overview
        overview_family = (
            None
            if overview is None
            else f"{overview.intent.value.lower()}-overview"
        )
        overview_ready = bool(
            self._view_family == overview_family
            and self._bundle is not None
            and len(self._bundle.pages) == 1
            and len(self._boards) == 1
            and self._boards[0].has_front
        )
        typed_ready = self._board_widget.front_frame is not None
        if (
            self._future is not None
            or self._closing
            or (
                self._view_family not in ("image", "curve", "histogram", "meter")
                and self._view_family != overview_family
            )
            or not (overview_ready or typed_ready)
        ):
            return
        family = self._view_family
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export current {family} view",
            f"{family}.png",
            "PNG image (*.png)",
        )
        if path:
            destination = Path(path)
            if destination.suffix.lower() != ".png":
                destination = destination.with_suffix(".png")
            self._start_export(destination)

    def _start_export(self, destination: Path) -> None:
        frame = self._board_widget.front_frame
        overview = self._grid_overview
        overview_family = (
            None
            if overview is None
            else f"{overview.intent.value.lower()}-overview"
        )
        overview_payload = (
            self._bundle.pages[0].png_bytes
            if self._view_family == overview_family
            and overview is not None
            and self._bundle is not None
            and len(self._bundle.pages) == 1
            else None
        )
        if (
            self._future is not None
            or self._closing
            or (frame is None and overview_payload is None)
        ):
            return
        self._request_revision += 1
        self._active_kind = "export"
        self._status.setText(f"EXPORTING {self._view_family.upper()}")
        self._diagnostic.setText("")
        self._set_typed_controls_enabled(False)
        if overview_payload is not None:
            if not self._submit_future(
                _export_encoded_png,
                overview_payload,
                Path(destination),
                self._request_revision,
                self._cancelled,
                self._export_commit_lock,
            ):
                self._active_kind = None
                self._set_typed_controls_enabled(True)
            return
        display = self._display
        if display is None:
            self._active_kind = None
            self._set_typed_controls_enabled(True)
            return
        if not self._submit_future(
            _export_typed_png,
            frame,
            display,
            Path(destination),
            self._request_revision,
            self._cancelled,
            self._export_commit_lock,
        ):
            self._active_kind = None
            self._set_typed_controls_enabled(True)
        else:
            self._sync_fit_authoring_busy()

    def _clear_bundle(self) -> None:
        super()._clear_bundle()
        self._board_widget.clear()
        self._grid_overview = None
        self._visible_figure = None
        self._grid_focus_pending = None
        self._discard_grid_focus_sequence = None

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        # Raster owns the currently visible front, so accept it first.  Fit may
        # then enqueue the newest overlay against that accepted viewport in the
        # same Qt-owner turn without either future touching a QWidget.
        consumed_completion = False
        self._completion_handoff_active = True
        try:
            raster_future = self._future
            if raster_future is not None and raster_future.done():
                self._future = None
                self._accept_finished_future(raster_future)
                consumed_completion = True
            fit_future = self._fit_future
            if fit_future is not None and fit_future.done():
                self._fit_future = None
                self._accept_finished_fit_future(fit_future)
                consumed_completion = True
        finally:
            self._completion_handoff_active = False
        if consumed_completion:
            # The completed Future/traceback can retain a whole rejected front
            # until this callback unwinds.  Resume only on a fresh queued turn.
            self._wake.request_owner_wake()
        elif not self._closing:
            retry = self._deferred_typed_retry
            if retry is not None and self._future is None:
                self._deferred_typed_retry = None
                candidate, editor, editor_revision, origin = retry
                self._start_typed_render(
                    candidate,
                    editor=editor,
                    editor_revision=editor_revision,
                    origin=origin,
                )
            if self._future is None:
                self._start_pending_fit_overlay()
            deferred_reload = self._deferred_fit_reload
            if deferred_reload is not None and self._fit_future is None:
                receipt, revision = deferred_reload
                self._deferred_fit_reload = None
                bindings = self._fit_bindings
                if (
                    revision == self._fit_editor_revision
                    and bindings is not None
                ):
                    self._fit_job_revision = revision
                    if not self._submit_fit_future(
                        "reload_saved",
                        bindings.reload,
                        receipt.handle,
                    ):
                        self._fit_job_revision = None
            if self._fit_prepare_pending and self._fit_future is None:
                self._start_fit_prepare()
            self._sync_fit_authoring_busy()
        self._finish_close_if_ready()

    def _finish_close_if_ready(self) -> None:
        if self._fit_future is not None:
            return
        if self._closing and self._future is None and not self._closed:
            self._fit_draft = None
            self._fit_draft_summary = None
            self._fit_save_inflight = None
            self._saved_fit_receipt = None
            self._fit_selection_candidate = None
            self._fit_options.clear()
            pane = self._fit_pane
            if pane is not None:
                pane.clear_options()
            self._fit_authority = None
            self._typed_renderer = None
            self._typed_front_committed = None
            self._typed_contract = None
            self._fit_overlay_renderer = None
            self._fit_bindings = None
            self._fit_save_path = None
            self._fit_overlay_pending = None
            self._fit_overlay_inflight = None
            self._fit_overlay_desired = None
            self._deferred_fit_reload = None
            self._deferred_typed_retry = None
            self._visible_transient_fit_result_owner = None
        super()._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        if self._fit_job_kind == "save" and self._fit_future is not None:
            self._close_deferred_during_fit_save = True
            self._status.setText("FIT SAVE IN PROGRESS · CLOSE DEFERRED")
            self._diagnostic.setText(
                "The saved Fit identity will be accepted before close can continue."
            )
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        authority = self._fit_authority
        if authority is not None:
            authority.close()
        if self._fit_cancelled is not None:
            self._fit_cancelled.set()
        self._fit_prepare_pending = False
        self._fit_overlay_pending = None
        super().shutdown()
        fit_future = self._fit_future
        if fit_future is not None:
            fit_future.cancel()


__all__ = ["DataFigureWindow"]
