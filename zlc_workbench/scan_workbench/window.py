"""Thin Qt shell for one current typed pulse-scan request."""

from __future__ import annotations


from PyQt5 import QtCore, QtWidgets

from Zou_lab_control.notebook.facade import (
    Experiment,
    OccupancyScanRequest,
    ScanRequest,
)
from zlc_frontend.qt_widgets import (
    ElidedLabel,
    FluentButton,
    FluentLabel,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentSwitch,
    FluentTabWidget,
    GREEN,
    GREY,
    ORANGE,
    FrozenRasterView,
    QtOwnerWake,
    QtRasterBoard,
    release_window,
    runtime_range_placeholders,
    scaled_px,
    FluentSettingsPopupAnchor,
    sync_revisioned_form_editors,
)
from zlc_frontend.curve_display import (
    CurveDisplayState,
    curve_display_form_spec,
    curve_display_form_values,
    curve_display_from_form,
    curve_display_with_x_view,
)
from zlc_frontend.selector import (
    CurveInteractionIntent,
    CurveRangeGesture,
    CurveViewportCommit,
    PanelInteractionOrigin,
)
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_neutral_atom.scan.contracts import AutonomousScanSlotProgram
from zlc_workbench.progressive_scan import ScanDisplayIntent
from zlc_workbench.scan import (
    FinalScanPresentation,
    ScanPanelController,
    ScanPanelRuntimeUpdate,
    ScanPanelViewModel,
)

from .application import _FrozenScanApplication


_SCAN_CURVE_PANEL_ID = "scan-curve"


class ScanWorkbenchWindow(QtWidgets.QWidget):
    """Progressive occupancy/final scan panel; rendering never blocks Qt."""

    finalReferenceChanged = QtCore.pyqtSignal(object)
    stateChanged = QtCore.pyqtSignal()

    def __init__(
        self,
        experiment: Experiment,
        request: ScanRequest | OccupancyScanRequest,
        *,
        display_intent: ScanDisplayIntent = ScanDisplayIntent(),
    ) -> None:
        super().__init__()
        if not isinstance(experiment, Experiment):
            raise TypeError("experiment must be Experiment")
        if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
            raise TypeError("request must be a current scan request")

        self.setWindowTitle("Pulse Scan")
        self._allow_close = False
        self._shown_presentation: FinalScanPresentation | None = None
        self._rejected_presentation: FinalScanPresentation | None = None
        self._curve_display = CurveDisplayState()
        self._pending_curve_interaction_origin: PanelInteractionOrigin | None = None
        self._curve_range_candidate: tuple[
            PanelInteractionOrigin,
            tuple[float, float],
        ] | None = None
        self._curve_binding_active = False
        self._curve_readiness: bool | None = None
        self._curve_editor_sync_signature: object = None
        self._local_display_diagnostic = ""
        self._reported_final_reference = None

        occupancy = isinstance(request, OccupancyScanRequest)
        progressive = occupancy and isinstance(
            request.program,
            AutonomousScanSlotProgram,
        )
        self._progressive_requested = progressive
        self._final_mode_text = (
            "OCCUPANCY · CANONICAL FINAL-ONLY"
            if occupancy
            else "DIRECT CAMERA · CANONICAL FINAL-ONLY"
        )
        self._mode = FluentLabel(
            (
                "PROVISIONAL OCCUPANCY CURVE → CANONICAL FINAL"
                if progressive
                else self._final_mode_text
            ),
            self,
        )
        self._mode.setObjectName("scanMode")
        self._status = FluentLabel("IDLE · FINAL-ONLY", self)
        self._status.setObjectName("scanStatus")
        self._artifact = ElidedLabel("Artifact: —", self)
        self._artifact.setObjectName("scanArtifact")
        self._projection = FluentLabel("Display: waiting for FINAL artifact", self)
        self._projection.setObjectName("projectionSummary")
        self._projection.setWordWrap(True)
        self._raster = FrozenRasterView(
            "scan-final",
            self,
            empty_text="No FINAL result",
        )
        self._raster.setObjectName("scanRaster")
        self._raster.setMinimumSize(
            scaled_px(320, minimum=240),
            scaled_px(240, minimum=160),
        )
        self._provisional_board = QtRasterBoard(
            (_SCAN_CURVE_PANEL_ID,),
            self,
            columns=1,
            empty_text="Waiting for progressive scan data",
        )
        self._provisional_board.setObjectName("scanProvisionalBoard")
        self._provisional_board.setMinimumSize(
            scaled_px(320, minimum=240),
            scaled_px(240, minimum=160),
        )
        self._display_stack = QtWidgets.QStackedWidget(self)
        self._display_stack.addWidget(self._provisional_board)
        self._display_stack.addWidget(self._raster)
        self._display_stack.setCurrentWidget(self._raster)
        self._diagnostics = FluentLabel("", self)
        self._diagnostics.setObjectName("scanDiagnostics")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._start = FluentButton("Run Scan", self, color=GREEN)
        self._start.setObjectName("startScanButton")
        self._stop = FluentButton("Stop", self, color=ORANGE)
        self._stop.setObjectName("stopScanButton")
        self._selector_switch = FluentSwitch("Selector", self)
        self._selector_switch.setObjectName("scanSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("scanDisplaySettingButton")
        self._setting_button.setEnabled(False)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._selector_switch)
        controls.addWidget(self._setting_button)
        controls.addWidget(self._start)
        controls.addWidget(self._stop)
        controls.addStretch(1)

        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName("scanTabs")
        self._live_page = QtWidgets.QWidget(self._tabs)
        self._live_page.setObjectName("scanLiveTab")
        live_layout = QtWidgets.QVBoxLayout(self._live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.addWidget(self._display_stack, 1)
        self._edit_curve_display = FluentRevisionedFormEditor(
            curve_display_form_spec(),
            "curve display",
            runtime_placeholder_fields=("y_min", "y_max"),
            parent=self._tabs,
        )
        self._edit_curve_display.setObjectName("scanCurveEditEditor")
        self._tabs.addTab(self._live_page, "Live")
        self._tabs.addTab(self._edit_curve_display, "Edit")

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("scanDisplaySettingsPopup")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        settings_layout.setContentsMargins(*(scaled_px(10, minimum=8),) * 4)
        self._setting_curve_display = FluentRevisionedFormEditor(
            curve_display_form_spec(),
            "curve display",
            runtime_placeholder_fields=("y_min", "y_max"),
            parent=self._settings_popup,
        )
        self._setting_curve_display.setObjectName("scanCurveSettingEditor")
        settings_layout.addWidget(self._setting_curve_display)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._setting_button,
        )
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._mode)
        layout.addWidget(self._status)
        layout.addWidget(self._artifact)
        layout.addWidget(self._projection)
        layout.addWidget(self._tabs, 1)
        layout.addWidget(self._diagnostics)
        layout.addLayout(controls)

        self._wake = QtOwnerWake(self)
        self._experiment = experiment
        self._application = _FrozenScanApplication(
            experiment,
            request,
            display_intent,
        )
        self._controller = ScanPanelController(
            self._application,
            self._wake.request_owner_wake,
            preview_presenter=self._provisional_board,
        )
        if self._controller.progressive_curve_display != self._curve_display:
            raise RuntimeError("scan curve display owners disagree at construction")
        self._wake.bind(self._owner_cycle)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._runtime_tick)
        self._reported_runtime_state: tuple[bool, bool] | None = None
        self._start.clicked.connect(self._start_scan)
        self._stop.clicked.connect(self._stop_scan)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._edit_curve_display.applyRequested.connect(
            lambda revision, values: self._apply_curve_display_form(
                self._edit_curve_display,
                revision,
                values,
            )
        )
        self._setting_curve_display.applyRequested.connect(
            lambda revision, values: self._apply_curve_display_form(
                self._setting_curve_display,
                revision,
                values,
            )
        )
        self._edit_curve_display.cancelRequested.connect(
            lambda: self._reload_curve_display_editor(self._edit_curve_display)
        )
        self._setting_curve_display.cancelRequested.connect(
            lambda: self._reload_curve_display_editor(
                self._setting_curve_display
            )
        )
        self._sync_curve_display_editors()
        self._apply_model(self._controller.view_model)

    @property
    def final_reference(self) -> ScanArtifactRef | None:
        return self._controller.view_model.artifact_ref

    @property
    def worker_idle(self) -> bool:
        return self._controller.worker_idle

    @property
    def can_reconfigure(self) -> bool:
        return self._controller.can_reconfigure

    @property
    def closed(self) -> bool:
        return self._controller.closed

    @staticmethod
    def _set_text_if_changed(
        widget: QtWidgets.QLabel | QtWidgets.QAbstractButton,
        text: str,
    ) -> None:
        if widget.text() != text:
            widget.setText(text)

    @staticmethod
    def _set_enabled_if_changed(widget: QtWidgets.QWidget, enabled: bool) -> None:
        enabled = bool(enabled)
        if widget.isEnabled() != enabled:
            widget.setEnabled(enabled)

    @staticmethod
    def _set_tooltip_if_changed(widget: QtWidgets.QWidget, tooltip: str) -> None:
        if widget.toolTip() != tooltip:
            widget.setToolTip(tooltip)

    def _visible_curve_matches_current_state(self) -> bool:
        origin = self._provisional_board.visible_curve_origin()
        return (
            origin is not None
            and origin.panel_id == _SCAN_CURVE_PANEL_ID
            and origin.presentation.panel_revision == self._curve_display.revision
        )

    def _visible_curve_y_limits(self) -> tuple[float, float] | None:
        if not self._visible_curve_matches_current_state():
            return None
        payload = self._provisional_board.visible_curve_payload()
        return None if payload is None else payload.viewport.y_limits

    def _reload_curve_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        editors = (self._edit_curve_display, self._setting_curve_display)
        if editor not in editors:
            raise ValueError("curve display editor does not belong to this window")
        editor.load(
            revision=self._curve_display.revision,
            semantic_identity=self._curve_display,
            values=curve_display_form_values(self._curve_display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_curve_y_limits(),
                "y_min",
                "y_max",
            ),
        )

    def _sync_curve_display_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        replace_owner: bool = False,
    ) -> None:
        y_limits = self._visible_curve_y_limits()
        sync_revisioned_form_editors(
            (self._edit_curve_display, self._setting_curve_display),
            revision=self._curve_display.revision,
            semantic_identity=self._curve_display,
            values=curve_display_form_values(self._curve_display),
            runtime_placeholders=runtime_range_placeholders(
                y_limits,
                "y_min",
                "y_max",
            ),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
            replace_owner=replace_owner,
        )
        self._curve_editor_sync_signature = (
            self._curve_display.revision,
            self._visible_curve_matches_current_state(),
            y_limits,
        )

    def _apply_curve_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        if editor not in (
            self._edit_curve_display,
            self._setting_curve_display,
        ):
            raise ValueError("curve display editor does not belong to this window")
        try:
            if not self._visible_curve_matches_current_state():
                raise RuntimeError(
                    "curve display cannot change until its current front is visible"
                )
            if base_revision != self._curve_display.revision:
                raise RuntimeError(
                    f"curve display edit base r{base_revision} is stale; "
                    f"current revision is r{self._curve_display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("curve display form must emit one exact mapping")
            candidate = curve_display_from_form(
                self._curve_display,
                values,
                current_y_limits=self._visible_curve_y_limits(),
            )
            self._commit_curve_display(
                candidate,
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except Exception as error:
            self._record_display_failure(
                f"Curve display edit rejected: {type(error).__name__}: {error}"
            )

    def _commit_curve_display(
        self,
        state: CurveDisplayState,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        interaction_origin: PanelInteractionOrigin | None = None,
    ) -> None:
        current = self._curve_display
        if not isinstance(state, CurveDisplayState):
            raise TypeError("state must be CurveDisplayState")
        changed = state != current
        if changed and state.revision != current.revision + 1:
            raise ValueError("curve display commit must advance exactly once")
        if not changed and state.revision != current.revision:
            raise ValueError("curve display no-op changed revision")
        if accepted_editor is not None:
            if accepted_base_revision != current.revision:
                raise ValueError("accepted curve editor base differs from current revision")
            if accepted_editor.base_revision != accepted_base_revision:
                raise ValueError("accepted curve editor no longer owns its emitted base")
        if interaction_origin is not None and not changed:
            raise ValueError("curve interaction cannot commit a semantic no-op")
        if changed:
            self._controller.reconfigure_progressive_curve_display(state)
            self._curve_display = state
            self._pending_curve_interaction_origin = interaction_origin
            self._clear_curve_range_candidate()
        self._local_display_diagnostic = ""
        self._refresh_diagnostics(self._controller.view_model)
        self._sync_curve_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        self._update_curve_controls(self._controller.view_model)

    def _accept_curve_interaction(
        self,
        command: CurveInteractionIntent,
    ) -> None:
        if not isinstance(command, (CurveViewportCommit, CurveRangeGesture)):
            raise TypeError("curve interaction callback received an unknown command")
        origin = command.origin
        if origin.panel_id != _SCAN_CURVE_PANEL_ID:
            raise RuntimeError("curve interaction belongs to another panel")
        if not self._visible_curve_matches_current_state():
            raise RuntimeError("curve interaction provenance is stale")
        if self._provisional_board.visible_curve_origin() != origin:
            raise RuntimeError("curve interaction origin is no longer painted")
        if origin.presentation.panel_revision != self._curve_display.revision:
            raise RuntimeError("curve interaction origin is stale")
        if isinstance(command, CurveRangeGesture):
            self._curve_range_candidate = (
                None if command.x_span is None else (origin, command.x_span)
            )
            self._provisional_board.set_curve_range_candidate(command.x_span)
            self._update_projection_label(self._controller.view_model)
            return
        if command.viewport.display_revision != self._curve_display.revision + 1:
            raise RuntimeError("curve viewport interaction must advance once")
        self._commit_curve_display(
            curve_display_with_x_view(
                self._curve_display,
                command.viewport.x_limits,
            ),
            interaction_origin=origin,
        )

    def _set_selector_enabled(self, enabled: bool) -> None:
        try:
            self._provisional_board.set_selectors_enabled(bool(enabled))
        except Exception as error:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            self._provisional_board.set_selectors_enabled(False)
            self._record_display_failure(
                f"Curve selector rejected: {type(error).__name__}: {error}"
            )

    def _ensure_curve_binding(self, required: bool) -> None:
        if required == self._curve_binding_active:
            return
        if required:
            self._provisional_board.bind_curve_interaction(
                _SCAN_CURVE_PANEL_ID,
                self._accept_curve_interaction,
                enabled=False,
            )
        else:
            self._provisional_board.unbind_curve_interaction()
            self._pending_curve_interaction_origin = None
            self._clear_curve_range_candidate()
        self._curve_binding_active = required
        self._curve_readiness = None

    def _clear_curve_range_candidate(self) -> None:
        self._curve_range_candidate = None
        self._provisional_board.set_curve_range_candidate(None)

    def _update_curve_controls(self, model: ScanPanelViewModel) -> None:
        binding_required = model.preview_attached and model.progressive_interactive
        self._ensure_curve_binding(binding_required)
        fault = (
            self._provisional_board.curve_selector_fault
            if self._curve_binding_active
            else None
        )
        interaction_unavailable = not binding_required or fault is not None
        if interaction_unavailable and self._settings_popup.isVisible():
            self._settings_popup.hide()
        if interaction_unavailable and self._selector_switch.isChecked():
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
        ready = (
            binding_required
            and fault is None
            and self._visible_curve_matches_current_state()
        )
        if ready:
            pending = self._pending_curve_interaction_origin
            visible = self._provisional_board.visible_curve_origin()
            if (
                pending is not None
                and visible is not None
                and visible.presentation.panel_revision
                > pending.presentation.panel_revision
            ):
                self._pending_curve_interaction_origin = None
            y_limits = self._visible_curve_y_limits()
            editor_signature = (
                self._curve_display.revision,
                True,
                y_limits,
            )
            if editor_signature != self._curve_editor_sync_signature:
                self._sync_curve_display_editors()
        if self._curve_binding_active and ready != self._curve_readiness:
            self._provisional_board.set_interaction_readiness(
                image=False,
                curve=ready,
            )
            self._curve_readiness = ready
        selector_enabled = ready and self._selector_switch.isChecked()
        if self._provisional_board.selectors_enabled != selector_enabled:
            self._provisional_board.set_selectors_enabled(selector_enabled)
        self._set_enabled_if_changed(self._selector_switch, ready)
        self._set_enabled_if_changed(self._setting_button, ready)
        self._set_enabled_if_changed(self._edit_curve_display, ready)
        self._set_enabled_if_changed(self._setting_curve_display, ready)
        if fault is not None:
            unavailable = (
                "Curve interaction disabled after callback failure: "
                f"{type(fault).__name__}: {fault}"
            )
        else:
            unavailable = (
                model.interaction_unavailable_reason
                if model.preview_attached and not model.progressive_interactive
                else None
            )
        tooltip = "" if unavailable is None else unavailable
        self._set_tooltip_if_changed(self._selector_switch, tooltip)
        self._set_tooltip_if_changed(self._setting_button, tooltip)
        self._set_tooltip_if_changed(self._edit_curve_display, tooltip)
        self._set_tooltip_if_changed(self._setting_curve_display, tooltip)
        self._refresh_diagnostics(model)

    def _open_display_settings(self) -> None:
        self._settings_anchor.toggle(
            self._setting_curve_display,
            prepare=lambda: self._reload_curve_display_editor(
                self._setting_curve_display
            ),
        )

    def _record_display_failure(self, message: str) -> None:
        self._local_display_diagnostic = str(message).strip()
        self._refresh_diagnostics(self._controller.view_model)

    def _refresh_diagnostics(self, model: ScanPanelViewModel) -> None:
        fault = (
            self._provisional_board.curve_selector_fault
            if self._curve_binding_active
            else None
        )
        interaction_fault = (
            ""
            if fault is None
            else (
                "Curve interaction disabled after callback failure: "
                f"{type(fault).__name__}: {fault}"
            )
        )
        parts = tuple(
            part
            for part in (
                model.diagnostic,
                self._local_display_diagnostic,
                interaction_fault,
            )
            if part
        )
        self._set_text_if_changed(self._diagnostics, "\n".join(parts))

    def reconfigure(
        self,
        request: ScanRequest | OccupancyScanRequest,
        *,
        display_intent: ScanDisplayIntent = ScanDisplayIntent(),
    ) -> None:
        """Apply one validated immutable request while the panel is fully idle."""

        if not isinstance(request, (ScanRequest, OccupancyScanRequest)):
            raise TypeError("request must be a current scan request")
        replacement = _FrozenScanApplication(
            self._experiment,
            request,
            display_intent,
        )
        self._controller.reconfigure(replacement)
        self._application = replacement
        self._curve_display = self._controller.progressive_curve_display
        occupancy = isinstance(request, OccupancyScanRequest)
        self._progressive_requested = occupancy and isinstance(
            request.program,
            AutonomousScanSlotProgram,
        )
        self._final_mode_text = (
            "OCCUPANCY · CANONICAL FINAL-ONLY"
            if occupancy
            else "DIRECT CAMERA · CANONICAL FINAL-ONLY"
        )
        self._shown_presentation = None
        self._rejected_presentation = None
        self._raster.clear()
        self._local_display_diagnostic = ""
        self._pending_curve_interaction_origin = None
        self._clear_curve_range_candidate()
        self._selector_switch.setChecked(False)
        self._settings_popup.hide()
        self._sync_curve_display_editors(replace_owner=True)
        self._apply_model(self._controller.view_model)

    def shutdown(self) -> None:
        """Begin the same nonblocking close path used by the standalone window."""

        self._settings_popup.hide()
        self._selector_switch.setChecked(False)
        self._controller.close()
        self._owner_cycle()

    def _start_scan(self) -> None:
        try:
            self._selector_switch.setChecked(False)
            self._pending_curve_interaction_origin = None
            self._clear_curve_range_candidate()
            self._settings_popup.hide()
            self._local_display_diagnostic = ""
            self._rejected_presentation = None
            self._controller.start()
        except BaseException as error:
            self._record_display_failure(
                f"Start failed: {type(error).__name__}: {error}"
            )
        self._owner_cycle()

    def _stop_scan(self) -> None:
        try:
            self._controller.stop()
        except BaseException as error:
            self._record_display_failure(
                f"Stop failed: {type(error).__name__}: {error}"
            )
        self._owner_cycle()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        model = self._controller.owner_cycle()
        self._apply_model(model)
        self._sync_runtime_timer()
        self._finish_close_if_needed(model)

    @QtCore.pyqtSlot()
    def _runtime_tick(self) -> None:
        update = self._controller.poll_runtime_change()
        if update is not None:
            if update.terminal_boundary:
                model = self._controller.view_model
                self._apply_model(model)
                self._finish_close_if_needed(model)
            else:
                self._apply_runtime_update(update)
        self._sync_runtime_timer()

    def _sync_runtime_timer(self) -> None:
        if self._controller.needs_periodic_poll:
            if not self._timer.isActive():
                self._timer.start()
        else:
            self._timer.stop()

    def _finish_close_if_needed(self, model: ScanPanelViewModel) -> None:
        if model.closed and not self._allow_close:
            self._timer.stop()
            self._wake.detach()
            self._allow_close = True
            QtCore.QTimer.singleShot(0, self.close)

    def _apply_runtime_update(self, update: ScanPanelRuntimeUpdate) -> None:
        if update.generation != self._controller.view_model.generation:
            return
        self._set_text_if_changed(self._status, update.status)
        self._set_enabled_if_changed(self._stop, update.can_stop)

    def _apply_model(self, model: ScanPanelViewModel) -> None:
        runtime_state = (model.closed, model.can_start and model.worker_idle)
        if runtime_state != self._reported_runtime_state:
            self._reported_runtime_state = runtime_state
            self.stateChanged.emit()
        if model.artifact_ref != self._reported_final_reference:
            self._reported_final_reference = model.artifact_ref
            self.finalReferenceChanged.emit(model.artifact_ref)
        progressive_mode = self._progressive_requested and model.artifact_ref is None and (
            model.can_start
            or model.status.startswith("PREPARING")
            or not model.final_only
        )
        self._set_text_if_changed(
            self._mode,
            (
                "PROVISIONAL OCCUPANCY CURVE → CANONICAL FINAL"
                if progressive_mode
                else self._final_mode_text
            ),
        )
        self._set_text_if_changed(self._status, model.status)
        self._set_text_if_changed(
            self._artifact,
            (
                "Artifact: —"
                if model.artifact_ref is None
                else f"Artifact: {model.artifact_ref.target_ref}"
            ),
        )
        self._set_enabled_if_changed(self._start, model.can_start)
        self._set_enabled_if_changed(self._stop, model.can_stop)
        display_widget = (
            self._provisional_board
            if model.display_phase == "PROVISIONAL"
            else self._raster
        )
        if self._display_stack.currentWidget() is not display_widget:
            self._display_stack.setCurrentWidget(display_widget)
        self._update_projection_label(model)
        presentation = model.presentation
        if presentation is None:
            if model.artifact_ref is None:
                if (
                    self._shown_presentation is not None
                    or self._rejected_presentation is not None
                ):
                    self._shown_presentation = None
                    self._rejected_presentation = None
                    self._raster.clear()
        elif (
            presentation is not self._shown_presentation
            and presentation is not self._rejected_presentation
        ):
            try:
                self._raster.present_encoded(
                    presentation.png_bytes,
                    image_format="PNG",
                )
            except BaseException as error:
                self._rejected_presentation = presentation
                self._record_display_failure(
                    f"Qt rejected the worker-produced PNG raster: {error}"
                )
            else:
                self._shown_presentation = presentation
                self._rejected_presentation = None
                self._local_display_diagnostic = ""
        self._update_curve_controls(model)

    def _update_projection_label(self, model: ScanPanelViewModel) -> None:
        if model.projection_summary is not None:
            prefix = (
                "Display (PROVISIONAL): "
                if model.display_phase == "PROVISIONAL"
                else "Display: "
            )
            text = prefix + model.projection_summary
            candidate = self._curve_range_candidate
            if candidate is not None and model.display_phase == "PROVISIONAL":
                text += (
                    " · DISPLAY ONLY range="
                    f"{candidate[1][0]:.6g}..{candidate[1][1]:.6g}"
                )
        elif model.artifact_ref is not None:
            text = "Display: FINAL artifact retained; raster unavailable"
        else:
            text = (
                "Display: FINAL-only; waiting for canonical artifact"
                if model.final_only and not model.can_start
                else "Display: waiting for scan preparation"
            )
        self._set_text_if_changed(self._projection, text)

    def closeEvent(self, event) -> None:
        if self._allow_close:
            release_window(self)
            event.accept()
            return
        event.ignore()
        self._settings_popup.hide()
        self._selector_switch.setChecked(False)
        self._controller.close()
        self._owner_cycle()
