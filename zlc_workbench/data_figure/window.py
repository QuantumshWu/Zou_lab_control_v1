"""Qt DataFigure surface and lifecycle owner."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
from pathlib import Path
import threading
import time

from PyQt5 import QtCore, QtWidgets

from zlc_data import (
    FitCancelled,
    FitDeadlineExceeded,
    FitResultBatch,
    FitSpec,
    Selection,
)
from zlc_frontend import (
    DataFigure,
    FitAuthoringDraft,
    FitAuthoringOption,
    reconcile_fit_authoring_draft,
)
from zlc_frontend.curve_display import CurveDisplayState
from zlc_frontend.curve_display import curve_display_with_x_view
from zlc_frontend.meter_display import MeterDisplayState
from zlc_frontend.render import (
    BoardFrame,
    CurvePanelPayload,
    HistogramPanelPayload,
    ImagePanelPayload,
    MeterPanelPayload,
)
from zlc_frontend.data_figure_presentation import (
    DATA_FIGURE_PANEL_ID,
    DataFigureDisplayState,
    DataFigureGridFocusRequest,
    DataFigureGridOverview,
    DataFigurePanelPayload,
    data_figure_frame_contract,
    data_figure_summary,
    default_data_figure_display_state,
    same_exact_data_owners,
    validate_rendered_data_figure_payload,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.figure import ViewIntent
from zlc_frontend.histogram_display import (
    FacetedHistogramDisplayState,
    HistogramDisplayState,
    histogram_display_with_thresholds,
    histogram_display_with_x_view,
)
from zlc_frontend.image_display import ImageDisplayState, image_display_for_viewport
from zlc_frontend.qt_widgets import (
    FitAuthoringPane,
    FluentButton,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentSwitch,
    FrozenRasterWindow,
    FigureSurfaceContext,
    FigureSurfaceHost,
    GREY,
    GREEN,
    ORANGE,
    RasterPixelRatioObserver,
    FrozenRasterView,
    runtime_range_placeholders,
    FluentSettingsPopupAnchor,
    error_summary,
    sync_revisioned_form_editors,
)
from zlc_frontend.plot_layout import PanelSurfaceGeometry, panel_surface_geometry
from zlc_frontend.panel_params import (
    panel_display_form_spec,
    panel_display_form_values,
    panel_display_state_from_form,
    panel_display_state_intent,
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
from zlc_storage import nonnegative_integer
from zlc_workbench.window_runtime import (
    cancel_export_commits,
    submit_compute,
)

from .fit_contract import (
    FitSaveReceipt,
    FitWorkbenchBindings,
)
from .worker_jobs import (
    DataFigureSurfaceResult,
    _execute_snapshot_fit,
    _execute_surface_work,
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
        *,
        fit_bindings: FitWorkbenchBindings | None = None,
        worker_release=None,
        initial_display: DataFigureDisplayState | None = None,
        embedded: bool = False,
        surface_size_name: str,
        output_root: Path,
    ) -> None:
        if not callable(initial_loader) or not callable(typed_renderer):
            raise TypeError("figure worker callables must be callable")
        if fit_bindings is not None and not isinstance(
            fit_bindings,
            FitWorkbenchBindings,
        ):
            raise TypeError("fit_bindings must be FitWorkbenchBindings or None")
        if initial_display is not None:
            panel_display_state_intent(initial_display)
        if not isinstance(embedded, bool):
            raise TypeError("embedded must be bool")
        initial_surface = panel_surface_geometry(surface_size_name)
        self._typed_renderer = typed_renderer
        self._fit_bindings = fit_bindings
        self._initial_display = initial_display
        self._embedded = embedded
        self._surface_size_name = initial_surface.size_name
        self._output_root = Path(output_root).expanduser()
        if not self._output_root.is_absolute():
            raise ValueError("DataFigure output_root must be absolute")
        self._output_root = self._output_root.resolve()
        self._surface_geometry = initial_surface
        self._surface_revision = 0
        self._surface_job: tuple[str, object, tuple[object, ...], int] | None = None
        self._surface_retry: tuple[str, object, tuple[object, ...]] | None = None
        self._surface_refresh_pending = False
        self._initial_loader = initial_loader
        self._view_family: str | None = None
        self._display: DataFigureDisplayState | None = None
        self._typed_contract: (
            tuple[tuple[object, ...], object] | None
        ) = None
        self._typed_pages_admitted = False
        self._typed_ui_faulted = False
        self._initial_outcome: str | None = None
        self._request_revision = 0
        self._active_kind: str | None = "initial"
        self._pending_state: DataFigureDisplayState | None = None
        self._pending_origin: PanelInteractionOrigin | None = None
        self._pending_editor: FluentRevisionedFormEditor | None = None
        self._pending_editor_revision: int | None = None
        self._completion_handoff_active = False
        self._deferred_typed_retry: tuple[object, ...] | None = None
        self._edit_display: FluentRevisionedFormEditor | None = None
        self._setting_display: FluentRevisionedFormEditor | None = None
        self._export_commit_lock = threading.Lock()
        self._grid_overview: DataFigureGridOverview | None = None
        self._visible_figure: DataFigure | None = None
        self._grid_focus_pending: DataFigureGridFocusRequest | None = None
        self._discard_grid_focus_sequence: int | None = None

        self._fit_future: Future | None = None
        self._fit_job_kind: str | None = None
        self._fit_job_revision: int | None = None
        self._fit_editor_revision = 0
        self._fit_prepare_pending = False
        self._fit_selection_candidate: Selection | None = None
        self._fit_initial_selection_consumed = False
        self._fit_auto_open_consumed = False
        self._fit_options: dict[str, FitAuthoringOption] = {}
        self._fit_authoring_draft: FitAuthoringDraft | None = None
        self._fit_cancelled: threading.Event | None = None
        self._fit_execution: object | None = None
        self._fit_result: FitResultBatch | None = None
        self._fit_result_summary: str | None = None
        self._fit_save_inflight: tuple[object, FitResultBatch, str] | None = None
        self._fit_invocation = 0
        self._deferred_fit_reload: tuple[FitSaveReceipt, int] | None = None
        self._saved_fit_receipt: FitSaveReceipt | None = None
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

        super().__init__(
            None,
            window_title="Data Figure",
            mode_text="FROZEN DATA FIGURE · INTERACTIVE",
            loading_summary="Resolving immutable input…",
            object_prefix="figureViewer",
            subject="figure",
            worker_release=worker_release,
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
        self._surface_host = FigureSurfaceHost(
            DATA_FIGURE_PANEL_ID,
            empty_text="Resolving exact typed figure…",
            parent=self._typed_page,
        )
        # Retain the named low-level event target for Qt input tooling.  The
        # window does not bind or present through it; FigureSurfaceHost is the
        # sole interaction/presentation owner.
        self._surface_host.board.setObjectName("figureViewerTypedBoard")
        self._surface_host.set_logical_size(initial_surface.logical_size)
        page_layout.addWidget(self._surface_host, 1)
        self._surface_host.rectangleSelected.connect(
            self._accept_image_rectangle
        )
        self._surface_host.rangeSelected.connect(
            self._accept_numeric_interaction
        )
        self._surface_host.viewCommitted.connect(
            self._accept_surface_view_commit
        )
        self._surface_host.colorLimitsCommitted.connect(
            self._accept_image_interaction
        )
        self._surface_host.thresholdsCommitted.connect(
            self._accept_numeric_interaction
        )
        self._surface_host.interactionRejected.connect(
            self._diagnostic.setText
        )

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
        self._surface_observer = RasterPixelRatioObserver(
            self,
            self._surface_pixel_ratio_changed,
        )
        self._surface_pixel_ratio_changed(self._surface_observer.current_ratio)
        self._surface_observer.refresh(force=True)
        if not self._submit_surface_future(
            "initial",
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

    def _submit_surface_future(self, kind: str, function, *args) -> bool:
        """Freeze the current surface geometry into one worker request."""

        if not isinstance(kind, str) or not kind:
            raise ValueError("surface work kind must be non-empty text")
        frozen_args = tuple(args)
        revision = self._surface_revision
        if not self._submit_future(
            _execute_surface_work,
            function,
            frozen_args,
            self._surface_geometry,
            revision,
        ):
            return False
        self._surface_job = (kind, function, frozen_args, revision)
        return True

    def _surface_pixel_ratio_changed(self, pixel_ratio: float) -> None:
        """Invalidate an obsolete raster before requesting its replacement."""

        geometry = panel_surface_geometry(
            self._surface_size_name,
            pixel_ratio=pixel_ratio,
        )
        self._surface_host.set_logical_size(geometry.logical_size)
        if geometry == self._surface_geometry:
            return
        self._surface_geometry = geometry
        self._surface_revision += 1

        # Unsupported encoded artifacts are native-pixel frozen pages inside a
        # scroll host.  They are not interactive named panel surfaces.
        if self._view_family == "encoded":
            return
        if self._initial_outcome is not None:
            self._status.setText("UPDATING DISPLAY")
        self._set_typed_controls_enabled(False)

        surface_job = self._surface_job
        if surface_job is not None:
            kind, function, args, _revision = surface_job
            self._surface_retry = (kind, function, args)
            return
        if self._future is not None:
            # Export is not surface-bound; finish its atomic file commit first.
            self._surface_refresh_pending = True
            return
        if self._initial_outcome is not None:
            self._queue_surface_refresh()

    def _queue_surface_refresh(self) -> None:
        """Re-render the accepted semantic view for the current named surface."""

        if self._closing or self._future is not None:
            self._surface_refresh_pending = True
            return
        self._surface_refresh_pending = False
        family = self._view_family
        if family == "encoded" or family is None:
            return
        self._request_revision += 1
        if family.endswith("-overview"):
            self._active_kind = "surface_overview"
            self._status.setText("UPDATING GRID DISPLAY")
            if not self._submit_surface_future(
                "surface_overview",
                self._initial_loader,
                self._request_revision,
                self._cancelled,
            ):
                self._active_kind = None
            return

        display = self._display
        if display is None:
            raise RuntimeError("typed surface refresh lost its display state")
        self._pending_state = display
        self._active_kind = "surface"
        self._status.setText(f"UPDATING {family.upper()} DISPLAY")
        if not self._submit_surface_future(
            "surface",
            self._typed_renderer,
            display,
            self._request_revision,
            self._cancelled,
        ):
            self._pending_state = None
            self._active_kind = None

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

    def _present_grid_overview(self, overview: DataFigureGridOverview) -> None:
        if not isinstance(overview, DataFigureGridOverview):
            raise TypeError("overview must be DataFigureGridOverview")
        if not self._present_bundle(overview.bundle):
            raise RuntimeError("Qt rejected the immutable typed grid overview")
        if len(self._boards) != 1 or not isinstance(self._boards[0], FrozenRasterView):
            raise RuntimeError("typed grid overview did not admit one encoded board")
        board = self._boards[0]
        tab_host = self._tab_host_for_board(board)
        board.setFixedSize(*self._surface_geometry.logical_size)
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
        grid_display = overview.display_state
        if isinstance(grid_display, FacetedHistogramDisplayState):
            display = grid_display.display_for(region.focus_address)
        elif isinstance(grid_display, MeterDisplayState):
            display = MeterDisplayState(
                panel_index,
                region.focus_address,
                grid_display.revision,
            )
        elif grid_display is None:
            display = (
                MeterDisplayState(panel_index, region.focus_address)
                if overview.intent is ViewIntent.METER
                else default_data_figure_display_state(overview.intent)
            )
        else:
            display = grid_display
        request = DataFigureGridFocusRequest(
            panel_index,
            region.focus_address,
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
        if renderer is None or not self._submit_surface_future(
            "grid_focus",
            renderer,
            request,
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
        self._surface_host.clear()
        self._display = None
        self._typed_contract = None
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
                and self._surface_host.front_frame is not None
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
            and self._surface_job is None
            and self._surface_retry is None
            and not self._surface_refresh_pending
            and not self._fit_prepare_pending
            and self._deferred_typed_retry is None
            and self._deferred_fit_reload is None
            and not self._completion_handoff_active
        )

    @property
    def draft_ready(self) -> bool:
        return self._fit_execution is not None and self._fit_result is not None

    @property
    def saved_reference(self) -> FitResultArtifactRef | None:
        return self._saved_fit_reference

    @property
    def fit_models(self) -> tuple[str, ...]:
        return tuple(self._fit_options)

    def _visible_typed_payload(self) -> DataFigurePanelPayload | None:
        frame = self._surface_host.front_frame
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
        return self._surface_host.visible_interaction_origin()

    def _visible_value_limits(self) -> tuple[float, float] | None:
        payload = self._visible_typed_payload()
        if isinstance(payload, ImagePanelPayload):
            return payload.color_limits
        if isinstance(payload, CurvePanelPayload):
            return payload.viewport.y_limits
        if isinstance(payload, HistogramPanelPayload):
            return payload.viewport.count_limits
        return None

    def _display_plot_kind(self):
        context = self._surface_host.context
        if context is None:
            raise RuntimeError("typed display has no admitted Figure context")
        return context.contract.figure.kind

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

    def _ensure_typed_controls(self, state: DataFigureDisplayState) -> None:
        if self._edit_display is not None or self._setting_display is not None:
            if (
                self._display is not None
                and panel_display_state_intent(self._display)
                is not panel_display_state_intent(state)
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
        elif isinstance(state, CurveDisplayState):
            runtime_fields = ("y_min", "y_max")
            subject = "curve display"
        else:
            runtime_fields = ("count_min", "count_max")
            subject = "histogram display"
        spec = panel_display_form_spec(self._display_plot_kind())
        if spec is None:
            raise ValueError("current Figure kind has no authored display form")
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
        self._surface_host.set_selectors_enabled(
            self._interaction_switch.isChecked()
        )

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
            values=panel_display_form_values(self._display_plot_kind(), display),
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
            values=panel_display_form_values(self._display_plot_kind(), display),
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
            self._surface_host.set_interaction_ready(False)
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
            self._surface_host.set_interaction_ready(False)
            self._settings_button.setEnabled(False)
            self._fit_button.setEnabled(False)
            self._interaction_switch.setEnabled(False)
            self._overview_button.setEnabled(
                ready and overview is not None and self._future is None
            )
            self._export_button.setEnabled(
                ready and self._surface_host.front_frame is not None
            )
            return
        active = bool(
            enabled
            and not self._typed_ui_faulted
            and self._view_family in ("image", "curve", "histogram")
        )
        self._surface_host.set_interaction_ready(active)
        self._settings_button.setEnabled(active)
        self._export_button.setEnabled(
            active and self._surface_host.front_frame is not None
        )
        self._fit_button.setEnabled(
            active
            and self._fit_bindings is not None
            and self._view_family in ("curve", "image", "histogram")
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
            self._surface_host.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            self._diagnostic.setText(error_summary(error))

    def _fit_pane_is_open(self) -> bool:
        pane = self._fit_pane
        return pane is not None and self._tabs.indexOf(pane) >= 0

    def _fit_available_for_intent(self, intent: ViewIntent) -> bool:
        return bool(
            self._fit_bindings is not None
            and intent in (ViewIntent.CURVE, ViewIntent.IMAGE, ViewIntent.HISTOGRAM)
        )

    def _fit_authoring_busy_kind(self) -> str | None:
        kind = self._fit_job_kind
        if kind in ("prepare", "fit", "save"):
            return kind
        if kind == "reload_saved":
            return "render"
        if (
            self._future is not None
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
                draft_ready=self.draft_ready,
            )
            if self._fit_save_button is not None:
                self._fit_save_button.setEnabled(
                    busy is None and self.draft_ready
                )

    def _open_fit_pane(self) -> None:
        pane = self._fit_pane
        intent = (
            ViewIntent.CURVE
            if self._view_family == "curve"
            else ViewIntent.IMAGE
            if self._view_family == "image"
            else ViewIntent.HISTOGRAM
            if self._view_family == "histogram"
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
            future = submit_compute(function, *args)
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
        pane.set_busy("prepare", draft_ready=self.draft_ready)
        self._status.setText("PREPARING FIT")
        self._diagnostic.setText("")
        self._fit_job_revision = self._fit_editor_revision
        visible_figure = self._visible_figure
        visible_payload = self._visible_typed_payload()
        if visible_figure is None or visible_payload is None:
            pane.set_busy(None, draft_ready=self.draft_ready)
            self._status.setText("FIT SOURCE NOT READY")
            return
        histogram_projection = (
            visible_payload.bin_projection
            if isinstance(visible_payload, HistogramPanelPayload)
            else None
        )
        if not self._submit_fit_future(
            "prepare",
            _prepare_fit_options,
            bindings.prepare,
            visible_figure,
            self._fit_selection_candidate,
            histogram_projection,
        ):
            self._fit_job_revision = None
            pane.set_busy(None, draft_ready=self.draft_ready)

    def _discard_snapshot_fit(self) -> None:
        self._fit_execution = None
        self._fit_result = None
        self._fit_result_summary = None

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
        self._discard_snapshot_fit()
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
        bindings = self._fit_bindings
        if (
            pane is None
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
                or spec.committed_transform != current.spec.committed_transform
                or spec.independent_sources != current.spec.independent_sources
                or spec.batch_sources != current.spec.batch_sources
                or spec.numeric_policy != current.spec.numeric_policy
            ):
                raise ValueError("Fit request differs from the prepared authority draft")
        except BaseException as error:
            self._status.setText("FIT REQUEST INVALID")
            self._diagnostic.setText(error_summary(error))
            return

        visible_figure = self._visible_figure
        source_frame = self._surface_host.front_frame
        if visible_figure is None or source_frame is None:
            self._status.setText("FIT SOURCE NOT READY")
            return
        self._discard_snapshot_fit()
        self._fit_cancelled = threading.Event()
        self._fit_job_revision = self._fit_editor_revision
        self._fit_invocation += 1
        result_identity = (
            f"snapshot-fit:r{self._fit_editor_revision}:i{self._fit_invocation}"
        )
        deadline = time.monotonic() + bindings.timeout_seconds
        pane.set_busy("fit", draft_ready=False)
        self._status.setText("FITTING")
        self._summary.setText(pane.axis_summary_text)
        self._diagnostic.setText("")
        self._surface_host.clear_fit_overlays()
        if not self._submit_fit_future(
            "fit",
            _execute_snapshot_fit,
            bindings.execute,
            bindings.result,
            visible_figure,
            source_frame,
            result_identity,
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
        bindings = self._fit_bindings
        execution = self._fit_execution
        result = self._fit_result
        summary = self._fit_result_summary
        if (
            pane is None
            or bindings is None
            or execution is None
            or result is None
            or summary is None
            or self._fit_future is not None
            or self._closing
        ):
            return
        destination = None
        if bindings.save_requires_path:
            destination = self._fit_save_path
            if destination is None:
                output_dir = self._output_root / "figures" / "data-figure"
                output_dir.mkdir(parents=True, exist_ok=True)
                selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    "Save fitted DataFigure archive",
                    str(output_dir / "fitted_figure.npz"),
                    "DataFigure archive (*.npz)",
                )
                if not selected:
                    return
                destination = Path(selected)
                if not destination.suffix:
                    destination = destination.with_suffix(".npz")
        self._fit_save_inflight = (execution, result, summary)
        self._discard_snapshot_fit()
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
            bindings.save,
            execution,
            destination,
            self._display,
        ):
            self._fit_save_inflight = None
            self._fit_execution = execution
            self._fit_result = result
            self._fit_result_summary = summary
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
        self._discard_snapshot_fit()
        self._surface_host.clear_fit_overlays()
        pane = self._fit_pane
        if pane is not None:
            pane.set_busy("fit" if fitting else None, draft_ready=False)
        if self._fit_save_button is not None:
            self._fit_save_button.setEnabled(False)
        self._status.setText("CLEARING FIT" if fitting else "FIT CLEARED")
        self._summary.setText("Source view preserved; selection remains a draft candidate")
        self._diagnostic.setText("")

    def _reapply_fit_candidate(self) -> None:
        selection = self._fit_selection_candidate
        if selection is None:
            return
        origin = self._visible_typed_origin()
        if origin is None or self._visible_typed_payload() is None:
            return
        self._surface_host.set_area_selection_candidate(selection)

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
            candidate, _rolling_distribution = panel_display_state_from_form(
                self._display_plot_kind(),
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
            gesture.panel_id != DATA_FIGURE_PANEL_ID
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
            commit = self._surface_host.area_commit
            if commit is not None and isinstance(commit.selection, Selection):
                selection = commit.selection
        self._surface_host.set_rectangle_candidate(gesture.normalized_bounds)
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
            origin.panel_id != DATA_FIGURE_PANEL_ID
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

    def _accept_surface_view_commit(self, command) -> None:
        """Route the unified host's typed viewport commit by its contract."""

        if isinstance(command, ImageViewportCommit):
            self._accept_image_interaction(command)
            return
        self._accept_numeric_interaction(command)

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
            origin.panel_id != DATA_FIGURE_PANEL_ID
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
                commit = self._surface_host.area_commit
                if commit is not None and isinstance(commit.selection, Selection):
                    selection = commit.selection
            self._surface_host.set_range_candidate(command.x_span)
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
        candidate = (
            curve_display_with_x_view(display, command.viewport.x_limits)
            if isinstance(display, CurveDisplayState)
            else histogram_display_with_x_view(display, command.viewport.x_limits)
            if isinstance(display, HistogramDisplayState)
            else None
        )
        if candidate is None:
            raise TypeError("numeric viewport requires CURVE or HISTOGRAM state")
        self._start_typed_render(candidate, origin=origin)

    def _start_typed_render(
        self,
        candidate: DataFigureDisplayState,
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
        if panel_display_state_intent(candidate) is not panel_display_state_intent(display):
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
        submitted = self._submit_surface_future(
            "typed",
            self._typed_renderer,
            candidate,
            self._request_revision,
            self._cancelled,
        )
        if not submitted:
            self._discard_pending_typed()
        else:
            self._sync_fit_authoring_busy()

    def _discard_pending_typed(self) -> None:
        origin = self._pending_origin
        family = self._view_family
        self._pending_state = None
        self._pending_origin = None
        self._pending_editor = None
        self._pending_editor_revision = None
        self._active_kind = None
        cleanup_errors = []
        if origin is not None:
            try:
                if family not in ("image", "curve", "histogram"):
                    raise RuntimeError("pending interaction has no typed family")
                self._surface_host.discard_pending_interaction(origin)
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
    def _validate_authored_frame(
        frame: BoardFrame,
        context: FigureSurfaceContext,
        expected_state: DataFigureDisplayState,
    ) -> tuple[tuple[object, ...], tuple[object, ...]]:
        # The worker validates the semantic payload.  Qt repeats only compact
        # display fields and immutable object identities; it never rescans a
        # coordinate vector or trusts a token detached from the current frame.
        if context.display != expected_state:
            raise ValueError("typed worker returned conflicting authored state")
        if context.figure is None:
            raise ValueError("typed worker returned no exact DataFigure")
        intent = panel_display_state_intent(expected_state)
        payload = frame.panels[0].display_payload
        assert isinstance(
            payload,
            (
                ImagePanelPayload,
                CurvePanelPayload,
                HistogramPanelPayload,
                MeterPanelPayload,
            ),
        )
        validate_rendered_data_figure_payload(
            payload,
            expected_state,
            None,
        )
        return data_figure_frame_contract(intent, frame)

    @staticmethod
    def _typed_completion(
        result: object,
    ) -> tuple[BoardFrame, FigureSurfaceContext, object | None]:
        if (
            isinstance(result, tuple)
            and len(result) == 3
            and isinstance(result[0], BoardFrame)
            and isinstance(result[1], FigureSurfaceContext)
        ):
            return result
        raise TypeError("typed Figure worker returned another completion")

    def _install_initial_fit_overlays(
        self,
        frame: BoardFrame,
        context: FigureSurfaceContext,
        overlays: object | None,
    ) -> None:
        if overlays is None:
            return
        if context.figure is None:
            raise RuntimeError("Fit overlay has no exact source Figure")
        status = self._surface_host.install_fit_overlays(
            context.figure,
            frame,
            overlays,
        )
        if status != "CURRENT":
            raise RuntimeError(
                "saved Fit overlay was not current with its admitted base"
            )

    def _present_typed_front(
        self,
        frame: BoardFrame,
        context: FigureSurfaceContext,
        *,
        expected_state: DataFigureDisplayState,
        request_revision: int,
    ) -> None:
        request_revision = nonnegative_integer(
            request_revision,
            "typed request revision",
        )
        if frame.sequence != request_revision:
            raise ValueError("typed worker returned another request sequence")
        contract = context.contract
        figure = context.figure
        if figure is None:
            raise ValueError("typed worker returned no exact DataFigure")
        if contract.pixel_size != self._surface_geometry.raster_size:
            raise ValueError("typed worker returned another raster surface geometry")
        if figure.has_fit_overlays:
            raise ValueError("DataFigure base renderer returned a composed Fit result")
        frame_contract = self._validate_authored_frame(
            frame,
            context,
            expected_state,
        )
        expected_contract = self._typed_contract
        if expected_contract is not None:
            expected_identity, expected_data = expected_contract
            identity, exact_data = frame_contract
            if identity[0] != expected_identity[0]:
                raise ValueError("typed worker changed frozen source provenance")
            if not same_exact_data_owners(exact_data, expected_data):
                raise ValueError("typed worker changed frozen evaluated data")

        self._surface_host.present_frame(
            frame,
            context=context,
            logical_size=contract.logical_size,
        )
        # The admitted board front is the transaction boundary.  Commit the
        # exact Figure/authored state/contract before cache release or any
        # optional Qt chrome work.
        if expected_contract is None:
            self._typed_contract = frame_contract
        self._display = expected_state
        self._visible_figure = figure
        intent = panel_display_state_intent(expected_state)
        self._view_family = intent.value.lower()
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
                self._tabs.addTab(self._typed_page, intent.value.title())
                self._tabs.tabBar().setVisible(False)
                self._typed_page.show()
                self._typed_pages_admitted = True
            self._tabs.setCurrentWidget(self._typed_page)
            fit_capable = self._fit_available_for_intent(intent)
            self._mode.setText(
                f"EXACT {intent.value} · DISPLAY ONLY"
                if intent is ViewIntent.METER
                else f"EXACT {intent.value} · INTERACTIVE"
                + ("" if fit_capable else " · DISPLAY ONLY")
            )
            self._status.setText("READY")
            self._summary.setText(data_figure_summary(figure))
            self._diagnostic.setText("")
        except BaseException as error:
            self._typed_ui_faulted = True
            self._surface_host.unbind_interaction()
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        if intent is ViewIntent.METER:
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
            if self._fit_available_for_intent(intent):
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
                    and intent is ViewIntent.CURVE
                ):
                    self._diagnostic.setText(
                        "Fit unavailable: grid focus is a display projection; "
                        "Fit authority requires an axis-complete source view."
                    )
        except BaseException as error:
            self._typed_ui_faulted = True
            self._surface_host.unbind_interaction()
            self._set_typed_controls_enabled(False)
            self._status.setText("TYPED CONTROLS FAILED")
            self._diagnostic.setText(error_summary(error))

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        surface_job = self._surface_job
        surface_stale = bool(
            surface_job is not None
            and surface_job[3] != self._surface_revision
        )
        try:
            result = future.result()
        except CancelledError:
            self._surface_job = None
            if surface_stale or self._surface_retry is not None:
                return
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
            self._surface_job = None
            if surface_stale or self._surface_retry is not None:
                return
            if not self._closing:
                self._reject_completed_work(kind, error)
        else:
            if surface_job is not None:
                self._surface_job = None
                if not isinstance(result, DataFigureSurfaceResult):
                    self._reject_completed_work(
                        kind,
                        TypeError("surface worker returned another result envelope"),
                    )
                    return
                if result.surface_revision != surface_job[3]:
                    self._reject_completed_work(
                        kind,
                        ValueError("surface worker returned another surface revision"),
                    )
                    return
                if surface_stale:
                    if self._surface_retry is None:
                        self._surface_retry = (
                            surface_job[0],
                            surface_job[1],
                            surface_job[2],
                        )
                    return
                result = result.payload
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
        completed_execution: object | None = None
        completed_result: FitResultBatch | None = None
        completed_summary: str | None = None
        completed_figure: DataFigure | None = None
        completed_source_frame: object | None = None
        completed_overlays: object | None = None
        try:
            result = future.result()
            if kind == "prepare":
                options = (
                    () if job_revision != self._fit_editor_revision else tuple(result)
                )
            elif kind == "fit":
                if (
                    not isinstance(result, tuple)
                    or len(result) != 6
                    or result[0] is None
                    or not isinstance(result[1], FitResultBatch)
                    or not isinstance(result[2], str)
                    or not isinstance(result[3], DataFigure)
                    or result[4] is None
                    or result[5] is None
                ):
                    raise TypeError("Fit worker returned another snapshot result")
                (
                    completed_execution,
                    completed_result,
                    completed_summary,
                    completed_figure,
                    completed_source_frame,
                    completed_overlays,
                ) = result
                if fit_cancelled is not None and fit_cancelled.is_set():
                    raise CancelledError()
            elif kind == "save":
                if not isinstance(result, FitSaveReceipt):
                    raise TypeError("Fit save returned another receipt type")
                receipt = result
            elif kind == "reload_saved":
                if not isinstance(result, FitResultBatch):
                    raise TypeError("saved Fit reload returned another result type")
                reloaded_result = result
            else:
                raise RuntimeError(f"unknown Fit worker completion {kind!r}")
        except (CancelledError, FitCancelled):
            if not self._closing:
                self._status.setText("FIT CANCELLED")
                self._diagnostic.setText("")
        except FitDeadlineExceeded as error:
            if not self._closing:
                self._status.setText("FIT DEADLINE EXCEEDED")
                self._diagnostic.setText(error_summary(error))
        except BaseException as error:
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
                        (
                            self._fit_execution,
                            self._fit_result,
                            self._fit_result_summary,
                        ) = save_inflight
                    else:
                        self._status.setText("FIT SAVE FAILED · EDITOR CHANGED")
        else:
            if self._closing:
                pass
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
                assert completed_execution is not None
                assert completed_result is not None
                assert completed_summary is not None
                assert completed_figure is not None
                assert completed_source_frame is not None
                assert completed_overlays is not None
                if job_revision != self._fit_editor_revision:
                    self._status.setText("STALE FIT DISCARDED")
                else:
                    self._fit_execution = completed_execution
                    self._fit_result = completed_result
                    self._fit_result_summary = completed_summary
                    overlay_status = self._surface_host.install_fit_overlays(
                        completed_figure,
                        completed_source_frame,
                        completed_overlays,
                    )
                    self._status.setText("DRAFT FIT READY")
                    self._summary.setText(completed_summary)
                    self._diagnostic.setText(
                        ""
                        if overlay_status == "CURRENT"
                        else f"Fit overlay {overlay_status.lower()} for the current view"
                    )
            elif kind == "save":
                # Persistence authority is independent of the already-painted
                # vector layer.  Nothing after this assignment may erase it.
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
                self._discard_snapshot_fit()
                self.fitSaved.emit(receipt.handle)
                if job_revision != self._fit_editor_revision:
                    self._status.setText("FIT SAVED · EDITOR CHANGED")
                elif receipt.reloaded_result is not None and not close_after_save:
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
            elif (
                isinstance(result, tuple)
                and len(result) == 3
                and isinstance(result[0], BoardFrame)
                and isinstance(result[1], FigureSurfaceContext)
            ):
                frame, context, overlays = self._typed_completion(result)
                self._present_typed_front(
                    frame,
                    context,
                    expected_state=context.display,
                    request_revision=self._request_revision,
                )
                self._install_initial_fit_overlays(frame, context, overlays)
                if (
                    panel_display_state_intent(context.display) is not ViewIntent.METER
                    and not self._typed_ui_faulted
                ):
                    self._sync_committed_typed_controls()
            elif isinstance(result, DataFigureGridOverview):
                self._present_grid_overview(result)
            else:
                raise TypeError("initial figure worker returned another result")
            self._active_kind = None
            self._emit_initial_ready()
            return
        if kind == "surface_overview":
            if not isinstance(result, DataFigureGridOverview):
                raise TypeError("grid surface worker returned another result")
            self._present_grid_overview(result)
            self._active_kind = None
            return
        if kind == "grid_focus":
            frame, context, overlays = self._typed_completion(result)
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
                frame,
                context,
                expected_state=pending.display,
                request_revision=self._request_revision,
            )
            self._install_initial_fit_overlays(frame, context, overlays)
            if (
                panel_display_state_intent(context.display) is not ViewIntent.METER
                and not self._typed_ui_faulted
            ):
                self._sync_committed_typed_controls()
            return
        if kind == "typed":
            frame, context, overlays = self._typed_completion(result)
            if overlays is not None:
                raise ValueError("ordinary display commit returned Fit overlays")
            pending = self._pending_state
            editor = self._pending_editor
            editor_revision = self._pending_editor_revision
            if pending is None:
                raise RuntimeError("typed worker completed without pending state")
            if self._discard_grid_focus_sequence == self._request_revision:
                self._discard_grid_focus_sequence = None
                self._discard_pending_typed()
                self._show_grid_overview()
                return
            self._present_typed_front(
                frame,
                context,
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
        if kind == "surface":
            frame, context, overlays = self._typed_completion(result)
            if overlays is not None:
                raise ValueError("surface resize returned Fit overlays")
            pending = self._pending_state
            if pending is None:
                raise RuntimeError("surface worker completed without display state")
            self._present_typed_front(
                frame,
                context,
                expected_state=pending,
                request_revision=self._request_revision,
            )
            self._pending_state = None
            self._active_kind = None
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
            self._discard_pending_typed()
            if return_to_overview:
                self._show_grid_overview()
        elif kind == "grid_focus":
            discarded = self._discard_grid_focus_sequence == self._request_revision
            intent = (
                None
                if self._grid_focus_pending is None
                else panel_display_state_intent(self._grid_focus_pending.display)
            )
            self._grid_focus_pending = None
            self._discard_grid_focus_sequence = None
            self._active_kind = None
            label = "GRID" if intent is None else intent.value
            self._status.setText("READY" if discarded else f"{label} FOCUS FAILED")
            self._diagnostic.setText("" if discarded else error_summary(error))
            self._set_typed_controls_enabled(True)
        elif kind == "surface":
            self._status.setText("SURFACE UPDATE FAILED")
            self._diagnostic.setText(error_summary(error))
            self._pending_state = None
            self._active_kind = None
        elif kind == "surface_overview":
            self._status.setText("GRID SURFACE UPDATE FAILED")
            self._diagnostic.setText(error_summary(error))
            self._active_kind = None
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
        typed_ready = self._surface_host.front_frame is not None
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
        output_dir = self._output_root / "figures" / "data-figure"
        output_dir.mkdir(parents=True, exist_ok=True)
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            f"Export current {family} view",
            str(output_dir / f"{family}.png"),
            "PNG image (*.png)",
        )
        if path:
            destination = Path(path)
            if destination.suffix.lower() != ".png":
                destination = destination.with_suffix(".png")
            self._start_export(destination)

    def _start_export(self, destination: Path) -> None:
        frame = self._surface_host.front_frame
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
        self._surface_host.clear()
        self._grid_overview = None
        self._visible_figure = None
        self._grid_focus_pending = None
        self._discard_grid_focus_sequence = None

    def _drain_owner_completions(self) -> None:
        # Raster owns the visible front, so accept it before a Fit completion;
        # the latter installs only already-materialized vector primitives.
        consumed_completion = False
        self._completion_handoff_active = True
        try:
            raster_future = self._future
            if raster_future is not None and raster_future.done():
                self._future = None
                if not self._consume_worker_release_future(raster_future):
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
            surface_retry = self._surface_retry
            if surface_retry is not None and self._future is None:
                self._surface_retry = None
                kind, function, args = surface_retry
                self._active_kind = kind
                if not self._submit_surface_future(kind, function, *args):
                    self._reject_completed_work(
                        kind,
                        RuntimeError("surface retry was not submitted"),
                    )
            if (
                self._surface_refresh_pending
                and self._future is None
                and self._surface_retry is None
            ):
                self._queue_surface_refresh()
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

    def _finish_close_if_ready(self) -> None:
        if self._fit_future is not None:
            return
        if self._closing and self._future is None and not self.closed:
            self._discard_snapshot_fit()
            self._fit_save_inflight = None
            self._saved_fit_receipt = None
            self._fit_selection_candidate = None
            self._fit_options.clear()
            pane = self._fit_pane
            if pane is not None:
                pane.clear_options()
            self._initial_loader = None
            self._typed_renderer = None
            self._typed_contract = None
            self._surface_job = None
            self._surface_retry = None
            self._fit_bindings = None
            self._fit_save_path = None
            self._deferred_fit_reload = None
            self._deferred_typed_retry = None
        super()._finish_close_if_ready()

    def shutdown(self) -> None:
        if self._closing or self.closed:
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
        if self._fit_cancelled is not None:
            self._fit_cancelled.set()
        self._fit_prepare_pending = False
        self._surface_observer.detach()
        super().shutdown()
        fit_future = self._fit_future
        if fit_future is not None:
            fit_future.cancel()


__all__ = ["DataFigureWindow"]
