"""Qt saved-fit grid explorer and interaction lifecycle owner."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
from pathlib import Path
import threading

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_data import Selection
from zlc_frontend import (
    BoardFrame,
    DataFigure,
    FigurePanelRegion,
    FitGridCellSummary,
    FitGridModel,
    FitGridPage,
    ImageDisplayState,
    ImagePanelPayload,
    RadialGaussianImageFitPanel,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.encoded_raster import EncodedRasterDocument
from zlc_frontend.image_display import (
    image_display_for_viewport,
    image_display_form_spec,
    image_display_form_values,
    image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.qt_widgets import (
    AxisLayoutNavigator,
    FluentButton,
    FluentLabel,
    FluentPopup,
    FluentRevisionedFormEditor,
    FluentScrollArea,
    FluentSwitch,
    GREY,
    ORANGE,
    FrozenRasterView,
    QtRasterBoard,
    runtime_range_placeholders,
    FluentSettingsPopupAnchor,
    signals_blocked,
    sync_revisioned_form_editors,
)
from zlc_frontend.selector import (
    ImageColorLimitsCommit,
    ImageInteractionCommit,
    ImageViewportCommit,
    PanelInteractionOrigin,
    RectangleGesture,
)
from zlc_neutral_atom.fit_reference import FitResultArtifactRef
from zlc_workbench.frozen_raster import FrozenRasterWindow
from zlc_workbench.window_runtime import cancel_export_commits, error_summary

from .projection import _fit_grid_join_identity, _fit_panel_id, _grid_columns
from .render_lane import (
    _export_grid_view,
    _export_typed_grid_view,
    _load_grid_view,
    _reframe_existing_image_panels,
    _rerasterize_grid_view,
)

class SavedFitGridWindow(FrozenRasterWindow):
    """Browse one exact saved ``FitResultBatch`` without ever re-solving it."""

    def __init__(
        self,
        view_loader,
        refit_opener,
        reference: FitResultArtifactRef,
    ) -> None:
        if not callable(view_loader):
            raise TypeError("saved-fit view_loader must be callable")
        if not callable(refit_opener):
            raise TypeError("saved-fit refit_opener must be callable")
        if not isinstance(reference, FitResultArtifactRef):
            raise TypeError("reference must be FitResultArtifactRef")
        self._view_loader = view_loader
        self._refit_opener = refit_opener
        self._reference = reference
        self._model: FitGridModel | None = None
        self._navigator: AxisLayoutNavigator | None = None
        self._page: FitGridPage | None = None
        self._page_panels: tuple[RadialGaussianImageFitPanel, ...] = ()
        self._page_frame: BoardFrame | None = None
        self._page_color_limits: tuple[float, float] | None = None
        self._page_encoded_bundle: EncodedRasterDocument | None = None
        self._page_regions: tuple[FigurePanelRegion, ...] = ()
        self._current_panels: tuple[RadialGaussianImageFitPanel, ...] = ()
        self._current_frame: BoardFrame | None = None
        self._current_encoded_bundle: EncodedRasterDocument | None = None
        self._regions: tuple[FigurePanelRegion, ...] = ()
        self._view_family: str | None = None
        self._current_selection: Selection | None = None
        self._showing_page = True
        self._requested_selection: Selection | None = None
        self._request_revision = 0
        self._active_revision = 0
        self._active_kind: str | None = "page"
        self._layout_generation = -1
        self._display = ImageDisplayState()
        self._current_color_limits: tuple[float, float] | None = None
        self._previous_relim_mode: RelimMode | None = None
        self._pending_image_interaction_origin: PanelInteractionOrigin | None = None
        self._display_rollback: tuple[
            ImageDisplayState,
            tuple[float, float] | None,
            RelimMode | None,
        ] | None = None
        self._area_candidates: dict[
            str,
            tuple[float, float, float, float],
        ] = {}
        self._bound_panel_ids: set[str] = set()
        self._export_commit_lock = threading.Lock()

        super().__init__(
            None,
            window_title="Saved Fit Grid",
            mode_text="EXACT SAVED FIT · GRID EXPLORER · EXPLICIT REFIT",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="savedFitGrid",
            subject="SAVED FIT GRID",
        )

        self._retire_tab_pages()
        self._live_page = QtWidgets.QWidget(self._tabs)
        live_layout = QtWidgets.QVBoxLayout(self._live_page)
        live_layout.setContentsMargins(0, 0, 0, 0)
        self._board_widget = QtRasterBoard(
            ("saved-fit-loading",),
            self._live_page,
            columns=1,
            empty_text="Resolving exact saved fit…",
        )
        self._board_widget.setObjectName("savedFitGridBoard")
        self._board_widget.setMinimumSize(480, 320)
        live_layout.addWidget(self._board_widget, 1)
        self._encoded_board = FrozenRasterView(
            "saved-fit-generic",
            self._live_page,
            empty_text="Saved fit raster unavailable",
        )
        self._encoded_board.setObjectName("savedFitGridEncodedBoard")
        self._encoded_board.setMinimumSize(480, 320)
        self._encoded_scroll = FluentScrollArea(self._live_page)
        self._encoded_scroll.setObjectName("savedFitGridEncodedScroll")
        self._encoded_scroll.setWidgetResizable(False)
        self._encoded_scroll.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self._encoded_scroll.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAsNeeded
        )
        self._encoded_scroll.setWidget(self._encoded_board)
        self._encoded_scroll.hide()
        live_layout.addWidget(self._encoded_scroll, 1)
        self._edit_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._tabs,
        )
        self._edit_image_display.setObjectName("savedFitGridImageDisplayEditEditor")
        self._tabs.addTab(self._live_page, "Grid")
        self._tabs.addTab(self._edit_image_display, "Edit")
        self._tabs.tabBar().setVisible(True)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("savedFitGridDisplaySettingsPopup")
        settings_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._setting_image_display = FluentRevisionedFormEditor(
            image_display_form_spec(),
            "image display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._settings_popup,
        )
        self._setting_image_display.setObjectName(
            "savedFitGridImageDisplaySettingEditor"
        )
        settings_layout.addWidget(self._setting_image_display)

        self._previous_page_button = FluentButton(
            "Previous page",
            self,
            color=GREY,
        )
        self._previous_page_button.setObjectName("savedFitGridPreviousPage")
        self._overview_button = FluentButton("Overview", self, color=GREY)
        self._overview_button.setObjectName("savedFitGridOverview")
        self._next_page_button = FluentButton("Next page", self, color=GREY)
        self._next_page_button.setObjectName("savedFitGridNextPage")
        self._selector_switch = FluentSwitch("Selectors", self)
        self._selector_switch.setObjectName("savedFitGridSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._selector_switch.setEnabled(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("savedFitGridDisplaySettingButton")
        self._setting_button.setEnabled(False)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._setting_button,
        )
        self._fit_button = FluentButton(
            "Fit / Refit",
            self,
            color=ORANGE,
        )
        self._fit_button.setObjectName("savedFitGridFitButton")
        self._fit_button.setEnabled(False)
        self._export_button = FluentButton("Export image…", self, color=ORANGE)
        self._export_button.setObjectName("savedFitGridExport")
        actions = QtWidgets.QHBoxLayout()
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
        ):
            button.setEnabled(False)
            actions.addWidget(button)
        actions.addWidget(self._selector_switch)
        actions.addWidget(self._setting_button)
        actions.addWidget(self._fit_button)
        actions.addWidget(self._export_button)
        self._export_button.setEnabled(False)
        actions.addStretch(1)

        self._navigation_host = QtWidgets.QWidget(self)
        host_layout = QtWidgets.QVBoxLayout(self._navigation_host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(6)
        host_layout.addLayout(actions)
        self._cell_detail = FluentLabel("", self._navigation_host)
        self._cell_detail.setObjectName("savedFitGridCellDetail")
        self._cell_detail.setWordWrap(True)
        self._cell_detail.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        host_layout.addWidget(self._cell_detail)
        self._layout.insertWidget(3, self._navigation_host)

        self._previous_page_button.clicked.connect(
            lambda: self._move_page(-1)
        )
        self._overview_button.clicked.connect(self._show_page)
        self._next_page_button.clicked.connect(lambda: self._move_page(1))
        self._export_button.clicked.connect(self._choose_export)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._fit_button.clicked.connect(self._open_refit)
        self._board_widget.imagePanelLeftDoubleClicked.connect(
            self._focus_panel_id
        )
        self._encoded_board.normalizedDoubleClicked.connect(self._focus_at)
        self._edit_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._edit_image_display,
                revision,
                values,
            )
        )
        self._setting_image_display.applyRequested.connect(
            lambda revision, values: self._apply_image_display_form(
                self._setting_image_display,
                revision,
                values,
            )
        )
        self._edit_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._edit_image_display)
        )
        self._setting_image_display.cancelRequested.connect(
            lambda: self._reload_image_display_editor(self._setting_image_display)
        )
        self._sync_image_display_editors()
        self._submit_view(
            "page",
            page_address=None,
            cell_selection=None,
            layout_generation=0,
        )

    @property
    def raster_ready(self) -> bool:
        if self._view_family == "encoded":
            return (
                self._current_encoded_bundle is not None
                and self._encoded_board.has_front
            )
        return (
            self._view_family == "typed-image"
            and self._current_frame is not None
            and self._board_widget.has_front
            and self._board_widget.front_frame is self._current_frame
        )

    def _open_display_settings(self) -> None:
        self._settings_anchor.toggle(
            self._setting_image_display,
            prepare=lambda: self._reload_image_display_editor(
                self._setting_image_display
            ),
        )

    def _install_model(self, model: FitGridModel) -> None:
        if self._model is not None:
            raise RuntimeError("saved-fit compact metadata was installed twice")
        self._model = model
        if not model.axes:
            return
        navigator = AxisLayoutNavigator(
            model.axes,
            model.layout,
            object_prefix="savedFitGrid",
            action_text="Focus exact cell",
            parent=self._navigation_host,
        )
        navigator.candidateChanged.connect(self._candidate_changed)
        navigator.activated.connect(self._focus_indices)
        self._navigator = navigator
        self._navigation_host.layout().insertWidget(1, navigator)
        self._candidate_changed()

    def _frame_matches_display_revision(
        self,
        frame: BoardFrame | None = None,
    ) -> bool:
        candidate = self._current_frame if frame is None else frame
        if candidate is None:
            return False
        if candidate is self._current_frame and (
            not self._board_widget.has_front
            or self._board_widget.front_frame is not candidate
        ):
            return False
        for panel in candidate.panels:
            presentations = {
                item.panel_id: item
                for item in panel.coherence_stamp.presentations
            }
            presentation = presentations.get(panel.panel_id)
            if (
                not isinstance(panel.display_payload, ImagePanelPayload)
                or panel.display_payload.viewport.viewport_revision
                != self._display.revision
                or presentation is None
                or presentation.panel_revision != self._display.revision
            ):
                return False
        return True

    def _visible_image_color_limits(self) -> tuple[float, float] | None:
        if not self._frame_matches_display_revision():
            return None
        frame = self._current_frame
        assert frame is not None
        limits = {
            panel.display_payload.color_limits
            for panel in frame.panels
            if isinstance(panel.display_payload, ImagePanelPayload)
        }
        if len(limits) != 1:
            raise RuntimeError("saved-fit grid front escaped its shared color scale")
        return next(iter(limits))

    def _reload_image_display_editor(
        self,
        editor: FluentRevisionedFormEditor,
    ) -> None:
        if editor not in (
            self._edit_image_display,
            self._setting_image_display,
        ):
            raise ValueError("image display editor does not belong to this window")
        editor.load(
            revision=self._display.revision,
            semantic_identity=self._display,
            values=image_display_form_values(self._display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
        )

    def _sync_image_display_editors(
        self,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
    ) -> None:
        sync_revisioned_form_editors(
            (self._edit_image_display, self._setting_image_display),
            revision=self._display.revision,
            semantic_identity=self._display,
            values=image_display_form_values(self._display),
            runtime_placeholders=runtime_range_placeholders(
                self._visible_image_color_limits(),
                "color_min",
                "color_max",
            ),
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )

    def _apply_image_display_form(
        self,
        editor: FluentRevisionedFormEditor,
        base_revision: int,
        values: object,
    ) -> None:
        try:
            if editor not in (
                self._edit_image_display,
                self._setting_image_display,
            ):
                raise ValueError("image display editor does not belong to this window")
            if self._closing or self._future is not None:
                raise RuntimeError("saved-fit display already has active work")
            if not self._frame_matches_display_revision():
                raise RuntimeError("saved-fit display has no current exact front")
            if base_revision != self._display.revision:
                raise RuntimeError(
                    f"image display edit base r{base_revision} is stale; "
                    f"current revision is r{self._display.revision}"
                )
            if not isinstance(values, dict):
                raise TypeError("image display form must emit one exact mapping")
            candidate = image_display_from_form(
                self._display,
                values,
                current_color_limits=self._visible_image_color_limits(),
            )
            self._commit_image_display(
                candidate,
                accepted_editor=editor,
                accepted_base_revision=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(
                f"Image display edit rejected: {error_summary(error)}"
            )

    def _visible_origin_is_current(
        self,
        origin: PanelInteractionOrigin,
    ) -> bool:
        return (
            isinstance(origin, PanelInteractionOrigin)
            and self._frame_matches_display_revision()
            and origin.panel_id in self._bound_panel_ids
            and self._board_widget.visible_image_origin(origin.panel_id) == origin
            and origin.presentation.panel_revision == self._display.revision
        )

    def _accept_rectangle_gesture(self, gesture: RectangleGesture) -> None:
        if not isinstance(gesture, RectangleGesture):
            raise TypeError("saved-fit area callback requires RectangleGesture")
        origin = self._board_widget.visible_image_origin(gesture.panel_id)
        if origin is None or not self._visible_origin_is_current(origin):
            raise RuntimeError("saved-fit area origin is stale")
        if (
            gesture.board_id,
            gesture.layout_generation,
            gesture.sequence,
            gesture.source_identity,
            gesture.viewport_revision,
        ) != (
            origin.board_id,
            origin.layout_generation,
            origin.sequence,
            origin.source_identity,
            self._display.revision,
        ):
            raise RuntimeError("saved-fit area differs from its painted origin")
        if gesture.normalized_bounds is None:
            self._area_candidates.pop(gesture.panel_id, None)
        else:
            self._area_candidates[gesture.panel_id] = gesture.normalized_bounds
        self._board_widget.set_image_rectangle_candidate(
            gesture.normalized_bounds,
            panel_id=gesture.panel_id,
        )
        if gesture.normalized_bounds is None:
            self._diagnostic.setText(
                f"{gesture.panel_id}: DISPLAY ONLY area cleared"
            )
            return
        left, top, right, bottom = gesture.normalized_bounds
        self._diagnostic.setText(
            f"{gesture.panel_id}: DISPLAY ONLY area "
            f"({left:.4f}, {top:.4f})..({right:.4f}, {bottom:.4f})"
        )

    def _accept_image_interaction(
        self,
        command: ImageInteractionCommit,
    ) -> None:
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("saved-fit image callback received an unknown command")
        origin = command.origin
        if not self._visible_origin_is_current(origin):
            raise RuntimeError("saved-fit image interaction origin is stale")
        if isinstance(command, ImageViewportCommit):
            if command.viewport.viewport_revision != self._display.revision + 1:
                raise RuntimeError("saved-fit viewport must advance exactly once")
            candidate = image_display_for_viewport(
                self._display,
                command.viewport,
            )
        else:
            candidate = replace(
                self._display,
                revision=self._display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
        self._commit_image_display(candidate, interaction_origin=origin)

    def _commit_image_display(
        self,
        state: ImageDisplayState,
        *,
        accepted_editor: FluentRevisionedFormEditor | None = None,
        accepted_base_revision: int | None = None,
        interaction_origin: PanelInteractionOrigin | None = None,
    ) -> None:
        current = self._display
        if not isinstance(state, ImageDisplayState):
            raise TypeError("state must be ImageDisplayState")
        changed = state != current
        if state.revision != current.revision + int(changed):
            raise ValueError("saved-fit display commit has an invalid revision")
        if accepted_editor is not None and (
            accepted_base_revision != current.revision
            or accepted_editor.base_revision != accepted_base_revision
        ):
            raise ValueError("accepted display editor no longer owns its base")
        if interaction_origin is not None and not changed:
            raise ValueError("image interaction cannot commit a semantic no-op")
        if not changed:
            self._sync_image_display_editors(
                accepted_editor=accepted_editor,
                accepted_base_revision=accepted_base_revision,
            )
            return
        if self._future is not None or self._display_rollback is not None:
            raise RuntimeError("saved-fit display already has active work")
        targets = (*self._page_panels, *self._current_panels)
        seen = set()
        for panel in targets:
            identity = id(panel)
            if identity in seen:
                continue
            seen.add(identity)
            image_viewport_for_display_state(state, panel.home_viewport)
        rollback = (
            current,
            self._current_color_limits,
            self._previous_relim_mode,
        )
        self._display = state
        self._display_rollback = rollback
        self._pending_image_interaction_origin = interaction_origin
        self._sync_image_display_editors(
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base_revision,
        )
        self._status.setText(f"RENDERING SAVED FIT DISPLAY r{state.revision}")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        kind = "display-page" if self._showing_page else "display-focus"
        if not self._submit_grid_reraster(
            kind,
            self._current_panels,
            layout_generation=self._layout_generation,
            previous_relim_mode=current.relim_mode,
            baseline_color_limits=self._current_color_limits,
        ):
            self._restore_display_after_failure()
            self._set_controls_enabled(True)

    def _restore_display_after_failure(self) -> None:
        rollback = self._display_rollback
        if rollback is not None:
            self._display, self._current_color_limits, self._previous_relim_mode = (
                rollback
            )
        origin = self._pending_image_interaction_origin
        if origin is not None:
            self._board_widget.discard_pending_image_interaction(origin)
        self._pending_image_interaction_origin = None
        self._display_rollback = None
        self._sync_image_display_editors()

    def _sync_panel_bindings(self, frame: BoardFrame) -> None:
        intent = self._selector_switch.isChecked()
        for panel_id in tuple(self._bound_panel_ids):
            self._board_widget.unbind_rectangle_selector(panel_id)
        self._bound_panel_ids.clear()
        for panel in frame.panels:
            payload = panel.display_payload
            if not isinstance(payload, ImagePanelPayload):
                raise TypeError("saved-fit board admitted a non-IMAGE panel")
            self._board_widget.bind_rectangle_selector(
                panel.panel_id,
                payload.viewport,
                self._accept_rectangle_gesture,
                enabled=intent,
                interaction_callback=self._accept_image_interaction,
            )
            self._bound_panel_ids.add(panel.panel_id)
            bounds = self._area_candidates.get(panel.panel_id)
            if bounds is not None:
                self._board_widget.set_image_rectangle_candidate(
                    bounds,
                    panel_id=panel.panel_id,
                )

    def _present_typed_frame(
        self,
        frame: BoardFrame,
        panels: tuple[RadialGaussianImageFitPanel, ...],
        *,
        color_limits: tuple[float, float],
    ) -> None:
        if self._view_family not in (None, "typed-image"):
            raise RuntimeError("saved-fit view family changed within one exact session")
        panel_ids = tuple(panel.panel_id for panel in frame.panels)
        if panel_ids != tuple(_fit_panel_id(panel) for panel in panels):
            raise ValueError("saved-fit frame order differs from projected panels")
        front = self._board_widget.front_frame
        needs_stage = tuple(self._board_widget.panel_ids) != panel_ids or (
            front is not None
            and (front.board_id, front.layout_generation)
            != (frame.board_id, frame.layout_generation)
        )
        if needs_stage:
            self._board_widget.stage_layout(
                panel_ids,
                board_id=frame.board_id,
                layout_generation=frame.layout_generation,
                columns=_grid_columns(len(panel_ids)),
            )
        self._board_widget.present(frame)
        self._view_family = "typed-image"
        self._encoded_scroll.hide()
        self._board_widget.show()
        self._current_panels = tuple(panels)
        self._current_frame = frame
        self._current_color_limits = color_limits
        self._layout_generation = frame.layout_generation
        self._sync_panel_bindings(frame)
        self._pending_image_interaction_origin = None
        self._sync_image_display_editors()

    def _present_encoded_bundle(
        self,
        bundle: EncodedRasterDocument,
    ) -> None:
        if self._view_family not in (None, "encoded"):
            raise RuntimeError("saved-fit view family changed within one exact session")
        if not isinstance(bundle, EncodedRasterDocument) or len(bundle.pages) != 1:
            raise TypeError("generic saved-fit view requires one encoded page")
        self._encoded_board.present_encoded(
            bundle.pages[0].png_bytes,
            image_format="PNG",
        )
        self._encoded_board.adjustSize()
        self._view_family = "encoded"
        self._board_widget.hide()
        self._encoded_scroll.show()
        self._current_encoded_bundle = bundle

    def _set_selector_enabled(self, enabled: bool) -> None:
        try:
            if enabled and not self._frame_matches_display_revision():
                raise RuntimeError("selector has no current exact IMAGE front")
            self._board_widget.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            self._board_widget.set_selectors_enabled(False)
            self._diagnostic.setText(
                f"Selector rejected: {error_summary(error)}"
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        page = self._page
        focused = False
        if enabled and not self._showing_page and self._model is not None:
            try:
                self._model.resolve_selection(self._current_selection)
            except (TypeError, ValueError):
                pass
            else:
                focused = True
        self._fit_button.setEnabled(focused)
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(enabled)
        self._previous_page_button.setEnabled(
            enabled and page is not None and page.previous_address is not None
        )
        self._next_page_button.setEnabled(
            enabled and page is not None and page.next_address is not None
        )
        if self._view_family == "encoded":
            self._overview_button.setEnabled(
                enabled
                and not self._showing_page
                and self._page_encoded_bundle is not None
            )
            self._selector_switch.setEnabled(False)
            self._setting_button.setEnabled(False)
            self._edit_image_display.setEnabled(False)
            self._setting_image_display.setEnabled(False)
            self._export_button.setEnabled(
                enabled and self._current_encoded_bundle is not None
            )
            if self._board_widget.selectors_enabled:
                self._board_widget.set_selectors_enabled(False)
            return
        self._overview_button.setEnabled(
            enabled
            and not self._showing_page
            and self._page_frame is not None
        )
        front_ready = enabled and self._frame_matches_display_revision()
        healthy = False
        for panel_id in self._bound_panel_ids:
            ready = (
                front_ready
                and self._board_widget.image_selector_fault(panel_id) is None
            )
            self._board_widget.set_image_interaction_readiness(panel_id, ready)
            healthy = healthy or ready
        self._selector_switch.setEnabled(healthy)
        intended = self._selector_switch.isChecked() and healthy
        if self._board_widget.selectors_enabled != intended:
            self._board_widget.set_selectors_enabled(intended)
        self._setting_button.setEnabled(front_ready)
        self._edit_image_display.setEnabled(front_ready)
        self._setting_image_display.setEnabled(front_ready)
        self._export_button.setEnabled(front_ready)

    def _open_refit(self) -> None:
        if (
            self._future is not None
            or self._closing
            or self._showing_page
            or self._model is None
            or not self._fit_button.isEnabled()
        ):
            return
        selection = self._current_selection
        try:
            self._model.resolve_selection(selection)
            self._refit_opener(self._reference, selection)
        except BaseException as error:
            self._status.setText("FIT/REFIT OPEN FAILED")
            self._summary.setText("The exact saved artifact remains unchanged")
            self._diagnostic.setText(error_summary(error))

    def _candidate_changed(self) -> None:
        navigator = self._navigator
        model = self._model
        if navigator is None or model is None:
            return
        indices = navigator.indices
        if indices is None:
            self._cell_detail.setText(
                "Choose every non-singleton batch axis to inspect one exact saved fit cell."
            )
            return
        try:
            selection = model.selection_for_indices(indices)
            storage, _multi, address = model.resolve_selection(selection)
        except BaseException as error:
            self._cell_detail.setText(error_summary(error))
            return
        self._cell_detail.setText(
            f"{address}\nstorage row {storage} · activate to load stored fit facts"
        )

    def _submit_view(
        self,
        kind: str,
        *,
        page_address: tuple[int, ...] | None,
        cell_selection: Selection | None,
        layout_generation: int,
    ) -> None:
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = kind
        self._requested_selection = cell_selection
        if not self._submit_future(
            _load_grid_view,
            self._view_loader,
            self._reference,
            page_address,
            cell_selection,
            self._active_revision,
            self._model is None,
            self._display,
            self._current_color_limits,
            self._previous_relim_mode,
            layout_generation,
            self._cancelled,
        ):
            self._active_kind = None
            self._set_controls_enabled(True)

    def _submit_grid_reraster(
        self,
        kind: str,
        panels: tuple[RadialGaussianImageFitPanel, ...],
        *,
        layout_generation: int,
        previous_relim_mode: RelimMode | None,
        baseline_color_limits: tuple[float, float] | None,
    ) -> bool:
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = kind
        submitted = self._submit_future(
            _rerasterize_grid_view,
            tuple(panels),
            self._display,
            baseline_color_limits,
            previous_relim_mode,
            layout_generation,
            self._active_revision,
            self._cancelled,
        )
        if not submitted:
            self._active_kind = None
        return submitted

    def _start_page(self, address: tuple[int, ...]) -> None:
        if self._future is not None or self._closing:
            return
        self._status.setText("BUILDING FIT GRID PAGE")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "page",
            page_address=tuple(address),
            cell_selection=None,
            layout_generation=self._layout_generation + 1,
        )

    def _move_page(self, direction: int) -> None:
        if direction not in (-1, 1):
            raise ValueError("fit grid page direction must be -1 or 1")
        page = self._page
        if page is None:
            return
        address = (
            page.previous_address if direction < 0 else page.next_address
        )
        if address is not None:
            self._start_page(address)

    def _focus_panel_id(self, panel_id: str) -> None:
        if self._future is not None or self._closing:
            return
        if not self._showing_page:
            self._show_page()
            return
        panel = next(
            (
                candidate
                for candidate in self._page_panels
                if _fit_panel_id(candidate) == panel_id
            ),
            None,
        )
        if panel is None:
            self._diagnostic.setText("Ignored stale saved-fit panel activation")
            return
        if panel.fit_storage_index is None:
            self._status.setText("FIT CELL NOT PRESENT")
            self._cell_detail.setText(
                "This logical gallery position is a sparse-layout hole; "
                "no neighbouring stored fit row was substituted."
            )
            return
        self._start_focus(panel.selection)

    def _focus_at(self, x: float, y: float) -> None:
        if (
            self._future is not None
            or self._closing
            or self._view_family != "encoded"
        ):
            return
        if not self._showing_page:
            self._show_page()
            return
        for region in self._regions:
            if not region.contains(x, y):
                continue
            if region.fit_storage_index is None:
                self._status.setText("FIT CELL NOT PRESENT")
                self._cell_detail.setText(
                    "This logical gallery position is a sparse-layout hole; "
                    "no neighbouring stored fit row was substituted."
                )
                return
            self._start_focus(region.fit_selection)
            return

    def _focus_indices(self, indices: object) -> None:
        model = self._model
        if model is None:
            return
        try:
            selection = model.selection_for_indices(tuple(indices))
            if selection is None:
                return
            self._start_focus(selection)
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))

    def _start_focus(self, selection: Selection | None) -> None:
        if self._future is not None or self._closing:
            return
        model = self._model
        if model is None or (
            self._view_family == "typed-image" and self._page_frame is None
        ) or (
            self._view_family == "encoded"
            and self._page_encoded_bundle is None
        ):
            return
        try:
            model.resolve_selection(selection)
            cached = (
                next(
                    (
                        panel
                        for panel in self._page_panels
                        if panel.selection == selection
                        and panel.fit_storage_index is not None
                    ),
                    None,
                )
                if self._view_family == "typed-image"
                else None
            )
            if cached is not None:
                self._start_cached_focus(cached)
                return
        except BaseException as error:
            self._status.setText("FIT CELL INVALID")
            self._diagnostic.setText(error_summary(error))
            return
        self._status.setText("BUILDING FIT CELL")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        self._submit_view(
            "focus",
            page_address=None,
            cell_selection=selection,
            layout_generation=self._layout_generation + 1,
        )

    def _start_cached_focus(
        self,
        panel: RadialGaussianImageFitPanel,
    ) -> None:
        page_frame = self._page_frame
        model = self._model
        if page_frame is None or model is None:
            return
        panel_id = _fit_panel_id(panel)
        if self._frame_matches_display_revision(page_frame):
            self._request_revision += 1
            frame = _reframe_existing_image_panels(
                page_frame,
                (panel,),
                layout_generation=self._layout_generation + 1,
                sequence=self._request_revision,
            )
            page_color_limits = self._page_color_limits
            payload = frame.panels[0].display_payload
            if (
                page_color_limits is None
                or not isinstance(payload, ImagePanelPayload)
                or payload.color_limits != page_color_limits
            ):
                self._diagnostic.setText(
                    "Cached fit cell differs from its exact page colour scale"
                )
                return
            try:
                self._present_typed_frame(
                    frame,
                    (panel,),
                    color_limits=page_color_limits,
                )
            except BaseException as error:
                self._status.setText("FIT CELL INVALID")
                self._diagnostic.setText(error_summary(error))
                return
            self._showing_page = False
            self._current_selection = panel.selection
            self._requested_selection = panel.selection
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(model.summary)
            self._cell_detail.setText(panel.summary)
            self._diagnostic.setText("")
            if self._navigator is not None:
                _storage, multi, _label = model.resolve_selection(panel.selection)
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
            self._set_controls_enabled(True)
            return
        self._requested_selection = panel.selection
        self._status.setText("RENDERING CACHED FIT CELL")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if not self._submit_grid_reraster(
            "cached-focus",
            (panel,),
            layout_generation=self._layout_generation + 1,
            previous_relim_mode=self._previous_relim_mode,
            baseline_color_limits=self._page_color_limits,
        ):
            self._set_controls_enabled(True)

    def _show_page(self) -> None:
        model = self._model
        page = self._page
        if self._view_family == "encoded":
            bundle = self._page_encoded_bundle
            if (
                self._future is not None
                or model is None
                or page is None
                or bundle is None
                or self._showing_page
            ):
                return
            try:
                self._present_encoded_bundle(bundle)
            except BaseException as error:
                self._status.setText("DISPLAY FAILED")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
                return
            self._regions = self._page_regions
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
            self._diagnostic.setText("")
            self._set_controls_enabled(True)
            return
        frame = self._page_frame
        if (
            self._future is not None
            or model is None
            or page is None
            or frame is None
            or self._showing_page
        ):
            return
        if self._frame_matches_display_revision(frame):
            self._request_revision += 1
            restored = _reframe_existing_image_panels(
                frame,
                self._page_panels,
                layout_generation=self._layout_generation + 1,
                sequence=self._request_revision,
            )
            try:
                self._present_typed_frame(
                    restored,
                    self._page_panels,
                    color_limits=self._page_color_limits or (0.0, 1.0),
                )
            except BaseException as error:
                self._status.setText("DISPLAY FAILED")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
                return
            self._page_frame = restored
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
            self._diagnostic.setText("")
            self._set_controls_enabled(True)
            return
        self._requested_selection = None
        self._status.setText("RENDERING FIT GRID PAGE")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if not self._submit_grid_reraster(
            "return-page-display",
            self._page_panels,
            layout_generation=self._layout_generation + 1,
            previous_relim_mode=self._previous_relim_mode,
            baseline_color_limits=self._page_color_limits,
        ):
            self._set_controls_enabled(True)

    def _choose_export(self) -> None:
        if (
            self._future is not None
            or self._closing
            or (
                self._current_frame is None
                and self._current_encoded_bundle is None
            )
        ):
            return
        path, _selected = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export saved fit view",
            "saved_fit_grid.png",
            "Images (*.png *.pdf *.svg *.jpg *.jpeg)",
        )
        if path:
            self._start_export(Path(path))

    def _start_export(self, destination: Path) -> None:
        if self._future is not None or self._closing:
            return
        page = self._page
        model = self._model
        try:
            if model is None:
                raise RuntimeError("saved-fit export has no compact model")
            typed_export = self._view_family == "typed-image"
            if typed_export:
                frame = self._current_frame
                panels = self._current_panels
                color_limits = self._current_color_limits
                if (
                    frame is None
                    or not panels
                    or color_limits is None
                    or not self._frame_matches_display_revision(frame)
                ):
                    raise RuntimeError("saved-fit typed export has no exact current front")
                panel_ids = tuple(_fit_panel_id(panel) for panel in panels)
                if panel_ids != tuple(panel.panel_id for panel in frame.panels):
                    raise ValueError("saved-fit export panel order differs from its front")
                _artifact, _inputs, expected_join = _fit_grid_join_identity(
                    panels,
                    panel_ids,
                )
                painted_join = frame.panels[0].coherence_stamp.join_key_digest
                if expected_join != painted_join:
                    raise ValueError("saved-fit export join differs from its front")
                for projected, rendered in zip(panels, frame.panels, strict=True):
                    payload = rendered.display_payload
                    if (
                        not isinstance(payload, ImagePanelPayload)
                        or payload.image is not projected.image
                        or payload.evaluated_input != projected.evaluated_input
                        or payload.fit_overlay != projected.fit_overlay
                        or payload.color_limits != color_limits
                    ):
                        raise ValueError(
                            "saved-fit export projection differs from its painted front"
                        )
        except BaseException as error:
            self._status.setText("FIT VIEW EXPORT FAILED")
            self._diagnostic.setText(error_summary(error))
            return
        self._request_revision += 1
        self._active_revision = self._request_revision
        self._active_kind = "export"
        self._status.setText("EXPORTING FIT VIEW")
        self._diagnostic.setText("")
        self._set_controls_enabled(False)
        if typed_export:
            submitted = self._submit_future(
                _export_typed_grid_view,
                panels,
                self._display,
                color_limits,
                _grid_columns(len(panels)),
                expected_join,
                Path(destination),
                self._active_revision,
                self._cancelled,
                self._export_commit_lock,
            )
        else:
            submitted = self._submit_future(
                _export_grid_view,
                self._view_loader,
                self._reference,
                model.identity,
                (
                    None
                    if self._current_selection is not None or page is None
                    else page.address
                ),
                self._current_selection,
                Path(destination),
                self._active_revision,
                self._cancelled,
                self._export_commit_lock,
            )
        if not submitted:
            self._active_kind = None
            self._set_controls_enabled(True)

    def _accept_view_result(self, result, kind: str, revision: int) -> None:
        (
            result_revision,
            model,
            model_identity,
            page,
            cell_summary,
            resolved_selection,
            summary,
            projection,
        ) = result
        if result_revision != revision or revision != self._request_revision:
            self._set_controls_enabled(True)
            return
        if self._model is None:
            if not isinstance(model, FitGridModel):
                raise TypeError("initial saved-fit page omitted compact metadata")
            if model.identity != model_identity:
                raise ValueError("saved-fit worker metadata identity changed")
            self._install_model(model)
        else:
            if (
                model is not None
                or model_identity != self._model.identity
            ):
                raise ValueError("saved-fit metadata changed during one exact-ref session")
        current_model = self._model
        assert current_model is not None
        if not isinstance(projection, tuple) or not projection:
            raise TypeError("saved-fit worker omitted its presentation family")
        family = projection[0]
        if family not in ("typed-image", "encoded"):
            raise ValueError("saved-fit worker returned an unknown presentation family")
        if self._view_family is not None and family != self._view_family:
            raise ValueError("saved-fit presentation family changed during one session")
        if kind == "page":
            if not isinstance(page, FitGridPage) or cell_summary is not None:
                raise TypeError("saved-fit page result is invalid")
            if family == "typed-image":
                if len(projection) != 4:
                    raise TypeError("typed saved-fit projection has invalid fields")
                _tag, panels, frame, color_limits = projection
                next_panels = tuple(panels)
                self._present_typed_frame(
                    frame,
                    next_panels,
                    color_limits=color_limits,
                )
                self._page_panels = next_panels
                self._page_frame = frame
                self._page_color_limits = color_limits
                self._current_frame = self._page_frame
            else:
                if len(projection) != 3:
                    raise TypeError("encoded saved-fit projection has invalid fields")
                _tag, bundle, regions = projection
                self._present_encoded_bundle(bundle)
                self._page_encoded_bundle = bundle
                self._page_regions = tuple(regions)
                self._regions = self._page_regions
            self._page = page
            self._current_selection = None
            self._showing_page = True
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(summary)
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
        elif kind == "focus":
            if page is not None or not isinstance(cell_summary, FitGridCellSummary):
                raise TypeError("saved-fit focus result is invalid")
            if family == "typed-image":
                if len(projection) != 4:
                    raise TypeError("typed saved-fit projection has invalid fields")
                _tag, panels, frame, color_limits = projection
                self._present_typed_frame(
                    frame,
                    tuple(panels),
                    color_limits=color_limits,
                )
            else:
                if len(projection) != 3:
                    raise TypeError("encoded saved-fit projection has invalid fields")
                _tag, bundle, regions = projection
                self._present_encoded_bundle(bundle)
                self._regions = tuple(regions)
            self._current_selection = resolved_selection
            self._showing_page = False
            if self._navigator is not None:
                _storage, multi, _label = current_model.resolve_selection(
                    resolved_selection
                )
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(summary)
            self._cell_detail.setText(cell_summary.text)
        else:
            raise RuntimeError("unknown saved-fit view result")
        self._requested_selection = (
            None if kind == "page" else resolved_selection
        )
        self._previous_relim_mode = self._display.relim_mode
        self._diagnostic.setText("")
        self._set_controls_enabled(True)

    def _accept_reraster_result(
        self,
        result,
        kind: str,
        revision: int,
    ) -> None:
        (
            result_revision,
            panels,
            display,
            frame,
            color_limits,
        ) = result
        if result_revision != revision or revision != self._request_revision:
            self._set_controls_enabled(True)
            return
        if display != self._display:
            raise ValueError("saved-fit reraster returned another display revision")
        model = self._model
        page = self._page
        if model is None:
            raise RuntimeError("saved-fit reraster has no compact model")
        if kind in ("display-page", "return-page-display"):
            if page is None:
                raise RuntimeError("saved-fit page reraster has no page metadata")
            candidate_panels = tuple(panels)
            self._present_typed_frame(
                frame,
                candidate_panels,
                color_limits=color_limits,
            )
            self._page_panels = candidate_panels
            self._page_frame = frame
            self._page_color_limits = color_limits
            self._current_frame = self._page_frame
            self._showing_page = True
            self._current_selection = None
            self._requested_selection = None
            self._status.setText("SAVED FIT GRID READY")
            self._summary.setText(f"{model.summary} · {page.label}")
            self._cell_detail.setText(
                "Double-click a present panel or choose an exact batch cell below."
            )
        elif kind in ("display-focus", "cached-focus"):
            focused = tuple(panels)
            if len(focused) != 1:
                raise ValueError("saved-fit focus reraster must contain one panel")
            panel = focused[0]
            if kind == "display-focus":
                selection = self._current_selection
            else:
                selection = self._requested_selection
            if panel.selection != selection:
                raise ValueError("saved-fit focus reraster changed exact selection")
            self._present_typed_frame(
                frame,
                focused,
                color_limits=color_limits,
            )
            self._showing_page = False
            self._current_selection = selection
            self._status.setText("FIT CELL FOCUSED")
            self._summary.setText(model.summary)
            self._cell_detail.setText(panel.summary)
            if self._navigator is not None:
                _storage, multi, _label = model.resolve_selection(selection)
                with signals_blocked(self._navigator):
                    self._navigator.set_indices(multi)
        else:
            raise RuntimeError("unknown saved-fit reraster result")
        self._previous_relim_mode = self._display.relim_mode
        self._display_rollback = None
        self._pending_image_interaction_origin = None
        self._diagnostic.setText("")
        self._set_controls_enabled(True)

    def _accept_finished_future(self, future: Future) -> None:
        kind = self._active_kind
        revision = self._active_revision
        self._active_kind = None
        try:
            result = future.result()
        except CancelledError:
            if not self._closing:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID CANCELLED")
                self._set_controls_enabled(True)
        except BaseException as error:
            if not self._closing:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)
        else:
            if self._closing:
                return
            try:
                if kind in ("page", "focus"):
                    self._accept_view_result(result, kind, revision)
                elif kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._accept_reraster_result(result, kind, revision)
                elif kind == "export":
                    result_revision, destination = result
                    if (
                        result_revision == revision == self._request_revision
                    ):
                        self._status.setText("FIT VIEW EXPORTED")
                        self._diagnostic.setText(str(destination))
                    self._set_controls_enabled(True)
                else:
                    raise RuntimeError("unknown saved-fit worker result")
            except BaseException as error:
                if kind in (
                    "display-page",
                    "display-focus",
                    "return-page-display",
                    "cached-focus",
                ):
                    self._restore_display_after_failure()
                self._status.setText("SAVED FIT GRID FAILED")
                self._summary.setText("The exact saved artifact remains unchanged")
                self._diagnostic.setText(error_summary(error))
                self._set_controls_enabled(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_Escape and not self._showing_page:
            self._show_page()
            event.accept()
            return
        super().keyPressEvent(event)

    def shutdown(self) -> None:
        if self._closing or self._closed:
            return
        cancel_export_commits(
            cancelled=self._cancelled,
            commit_lock=self._export_commit_lock,
        )
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(False)
        for button in (
            self._previous_page_button,
            self._overview_button,
            self._next_page_button,
            self._selector_switch,
            self._setting_button,
            self._fit_button,
            self._export_button,
        ):
            button.setEnabled(False)
        self._settings_popup.hide()
        super().shutdown()

    def _forget_presented_pages(self) -> None:
        """Drop every page and focus display fact this window still holds."""

        self._page = None
        self._page_panels = ()
        self._page_frame = None
        self._page_color_limits = None
        self._page_encoded_bundle = None
        self._page_regions = ()
        self._current_panels = ()
        self._current_frame = None
        self._current_encoded_bundle = None
        self._regions = ()

    def _clear_bundle(self) -> None:
        self._bundle = None
        self._boards = ()
        board = getattr(self, "_board_widget", None)
        if board is not None:
            origin = self._pending_image_interaction_origin
            if origin is not None:
                board.discard_pending_image_interaction(origin)
            for panel_id in tuple(self._bound_panel_ids):
                board.unbind_rectangle_selector(panel_id)
            self._bound_panel_ids.clear()
            board.clear()
        encoded_board = getattr(self, "_encoded_board", None)
        if encoded_board is not None:
            encoded_board.clear()
        self._forget_presented_pages()
        self._current_selection = None
        self._requested_selection = None
        self._showing_page = True
        self._pending_image_interaction_origin = None
        self._display_rollback = None
        self._area_candidates.clear()

    def _finish_close_if_ready(self) -> None:
        if self._closing and self._future is None and not self._closed:
            self._view_loader = None
            self._refit_opener = None
            self._model = None
            self._navigator = None
            self._forget_presented_pages()
        super()._finish_close_if_ready()


__all__ = ["SavedFitGridWindow"]
