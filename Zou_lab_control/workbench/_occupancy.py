"""Interactive Workbench for one exact committed occupancy cell."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace
import threading

from PyQt5 import QtCore, QtWidgets

from zlc_data import Selection
from zlc_frontend.display_range import RelimMode
from zlc_frontend.image_display import (
    ImageDisplayState, image_display_for_viewport, image_display_form_spec,
    image_display_form_values, image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.image_raster import rasterize_image_indexed8
from zlc_frontend.occupancy_render import OccupancyCellNavigation, OccupancyCellView
from zlc_frontend.qt_widgets import (
    AxisLayoutNavigator, FluentButton, FluentLabel, FluentPopup, FluentSwitch,
    FluentRevisionedFormEditor, FluentTabWidget, GREY, QtOwnerWake,
    QtRasterBoard, RectangleGesture,
    FluentSettingsPopupAnchor, runtime_range_placeholders, signals_blocked,
    sync_revisioned_form_editors,
)
from zlc_frontend.render import (
    BoardFrame, CoherenceStamp, ImagePanelPayload, PanelFrame,
    PanelPresentationIdentity, SITE_MAP_JOIN_SCHEMA_DIGEST, SiteMapPanelPayload,
    SourceIdentity,
)
from zlc_frontend.render_style import indexed_colormap
from zlc_frontend.selector import (
    ImageColorLimitsCommit, ImageInteractionCommit, ImageViewportCommit,
)
from zlc_neutral_atom.readout.occupancy_reference import OccupancyArtifactRef

from ._window_runtime import (
    RASTER_WORK_EXECUTOR,
    error_summary,
    open_workbench_window,
    release_window,
)


_PANEL_ID = "sites"
_BOARD_ID = "occupancy-cell"
def _cancel_point(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _load_navigation(loader, reference, cancelled):
    _cancel_point(cancelled)
    result = loader(reference)
    if not isinstance(result, OccupancyCellNavigation):
        raise TypeError("navigation loader must return OccupancyCellNavigation")
    if result.artifact_identity != reference.target_ref:
        raise ValueError("occupancy navigation names a different artifact")
    _cancel_point(cancelled)
    return result


def _build_front(
    view, display, color_limits, previous_relim, cell_revision, sequence,
    cancelled,
):
    if not isinstance(view, OccupancyCellView):
        raise TypeError("cell loader must return OccupancyCellView")
    _cancel_point(cancelled)
    viewport = image_viewport_for_display_state(display, view.home_viewport)
    raster, data_range, histogram, effective_limits = rasterize_image_indexed8(
        view.background, display, current_color_limits=color_limits,
        previous_relim_mode=previous_relim,
    )
    _cancel_point(cancelled)
    background = ImagePanelPayload(
        view.background, view.background_input, viewport, data_range, histogram,
        indexed_colormap(display.colormap.value), effective_limits,
    )
    payload = SiteMapPanelPayload(
        background, view.occupancy_input, view.site_axis, view.coordinate_frame,
        view.centers_xy, view.occupied, view.site_validity,
        view.calibration_identity, view.cell_identity,
    )
    presentation = PanelPresentationIdentity(
        _PANEL_ID, f"exact-occupancy-cell:{view.cell_identity}", 0,
        cell_revision, display.revision,
    )
    stamp = CoherenceStamp(
        view.run_id,
        view.provenance_epoch_id,
        "ExactOccupancyCell",
        SITE_MAP_JOIN_SCHEMA_DIGEST,
        payload.join_key_digest, (view.background_input, view.occupancy_input),
        (presentation,),
    )
    ref = view.occupancy_input.ref
    source = SourceIdentity(
        view.occupancy_input.dataset_id, ref.block_id, ref.stream_generation,
        ref.schema_fingerprint,
    )
    panel = PanelFrame(
        _PANEL_ID, "exact-occupancy-cell", source, stamp, raster, payload,
    )
    _cancel_point(cancelled)
    return BoardFrame(_BOARD_ID, 0, sequence, (panel,))


def _cell_job(
    loader, reference, selection, navigation, loaded_view, display, color_limits,
    previous_relim, cell_revision, sequence, cancelled,
):
    _cancel_point(cancelled)
    if loaded_view is None:
        loaded_view = loader(
            reference, selection,
            expected_navigation=navigation,
        )
    repeat, _storage, logical, _label = navigation.resolve_selection(selection)
    expected_selection = navigation.selection_for_indices(repeat, logical)
    if (
        not isinstance(loaded_view, OccupancyCellView)
        or loaded_view.cell_selection != expected_selection
    ):
        raise ValueError("cell loader returned a different exact selection")
    frame = _build_front(
        loaded_view, display, color_limits, previous_relim, cell_revision,
        sequence, cancelled,
    )
    return (
        navigation.identity, selection, cell_revision, display.revision,
        loaded_view, frame, display.relim_mode,
    )


def _label(parent, name, text, *, wrap=False):
    result = FluentLabel(text, parent)
    result.setObjectName(name)
    result.setWordWrap(wrap)
    return result


class OccupancyCellWindow(QtWidgets.QWidget):
    """Navigate exact cells and interact with one worker-rasterized SiteMap."""

    def __init__(
        self, navigation_loader, cell_loader, reference, *, selection,
    ) -> None:
        super().__init__()
        if not callable(navigation_loader) or not callable(cell_loader):
            raise TypeError("occupancy loaders must be callable")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if selection is not None and not isinstance(selection, Selection):
            raise TypeError("selection must be Selection or None")
        self._cell_loader = cell_loader
        self._reference, self._initial_selection = reference, selection
        self._navigation = self._navigator = None
        self._display = ImageDisplayState()
        self._loaded_view = self._loaded_selection = None
        self._requested_selection = self._presented_selection = None
        self._presented_key = self._presented_relim = None
        self._rectangle_candidate = None
        self._request_revision = self._sequence = 0
        self._future: Future | None = None
        self._active_kind, self._active_key = None, None
        self._pending_cell, self._render_requested = None, False
        self._cancelled = threading.Event()
        self._closing = self._closed = self._allow_close = False
        self._board_bound = False

        self.setWindowTitle("Occupancy Cell")
        self._mode = _label(
            self, "occupancyCellMode",
            "INTERACTIVE EXACT OCCUPANCY CELL · SAME-SHOT FRAME + SITES",
        )
        self._status = _label(self, "occupancyCellStatus", "READING FINAL CELL INDEX")
        self._summary = _label(
            self, "occupancyCellSummary", f"Resolving {reference.target_ref}…",
            wrap=True,
        )
        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName("occupancyCellTabs")
        page = QtWidgets.QWidget(self._tabs)
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._board = QtRasterBoard(
            (_PANEL_ID,), page, columns=1, empty_text="No exact occupancy cell",
        )
        self._board.setObjectName("occupancyCellBoard")
        self._board.setMinimumSize(320, 240)
        page_layout.addWidget(self._board, 1)
        spec = image_display_form_spec()
        self._edit_display = FluentRevisionedFormEditor(
            spec, "occupancy display",
            runtime_placeholder_fields=("color_min", "color_max"), parent=self._tabs,
        )
        self._edit_display.setObjectName("occupancyCellEditEditor")
        self._tabs.addTab(page, "Sites")
        self._tabs.addTab(self._edit_display, "Edit")
        self._diagnostic = _label(self, "occupancyCellDiagnostic", "", wrap=True)
        self._diagnostic.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._selector_switch = FluentSwitch("Selector", self)
        self._selector_switch.setObjectName("occupancyCellSelectorSwitch")
        self._selector_switch.setChecked(False)
        self._setting_button = FluentButton("Setting…", self, color=GREY)
        self._setting_button.setObjectName("occupancyCellDisplaySettingButton")
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName("closeOccupancyCellButton")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self._selector_switch)
        buttons.addWidget(self._setting_button)
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)
        self._layout = QtWidgets.QVBoxLayout(self)
        for widget in (self._mode, self._status, self._summary):
            self._layout.addWidget(widget)
        self._layout.addWidget(self._tabs, 1)
        self._layout.addWidget(self._diagnostic)
        self._layout.addLayout(buttons)

        self._settings_popup = FluentPopup(self)
        self._settings_popup.setObjectName("occupancyCellDisplaySettingsPopup")
        popup_layout = QtWidgets.QVBoxLayout(self._settings_popup)
        self._setting_display = FluentRevisionedFormEditor(
            spec, "occupancy display",
            runtime_placeholder_fields=("color_min", "color_max"),
            parent=self._settings_popup,
        )
        self._setting_display.setObjectName("occupancyCellSettingEditor")
        popup_layout.addWidget(self._setting_display)
        self._settings_anchor = FluentSettingsPopupAnchor(
            self._settings_popup,
            self._setting_button,
        )

        self._wake = QtOwnerWake(self)
        self._wake.bind(self._owner_cycle)
        self._close_button.clicked.connect(self.shutdown)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        for editor in (self._edit_display, self._setting_display):
            editor.applyRequested.connect(
                lambda revision, values, owner=editor:
                self._apply_display_form(owner, revision, values)
            )
            editor.cancelRequested.connect(
                lambda owner=editor: self._reload_editor(owner)
            )
        self._sync_editors()
        self._update_controls()
        self._submit(
            "navigation", 0, _load_navigation, navigation_loader, reference,
            self._cancelled,
        )

    @property
    def worker_idle(self):
        return self._future is None and self._pending_cell is None and not self._render_requested

    @property
    def navigation_ready(self):
        return self._navigation is not None

    @property
    def raster_ready(self):
        return self._front_is_current()

    @property
    def closed(self):
        return self._closed

    def _submit(self, kind, key, function, *args):
        if self._future is not None:
            raise RuntimeError("occupancy Workbench already has active worker work")
        try:
            self._future = RASTER_WORK_EXECUTOR.submit(function, *args)
        except BaseException as error:
            self._fail(error)
            return
        self._active_kind, self._active_key = kind, key
        self._future.add_done_callback(lambda _done: self._wake.request_owner_wake())

    def _build_navigator(self):
        navigation = self._navigation
        navigator = AxisLayoutNavigator(
            navigation.axes, navigation.cell_layout, object_prefix="occupancyCell",
            action_text="Load exact cell", parent=self,
        )
        navigator.candidateChanged.connect(self._refresh_candidate)
        navigator.activated.connect(self._activate_indices)
        navigator.action_button.setObjectName("occupancyCellLoad")
        self._navigator = navigator
        self._layout.insertWidget(3, navigator)

    def _control_selection(self):
        if self._navigation is None or self._navigator is None:
            return None
        indices = self._navigator.indices
        if indices is None:
            return None
        return self._navigation.selection_for_indices(indices[0], tuple(indices[1:]))

    def _set_controls(self, selection):
        repeat, _storage, logical, _label_text = self._navigation.resolve_selection(selection)
        with signals_blocked(self._navigator):
            self._navigator.set_indices((repeat, *logical))

    def _refresh_candidate(self):
        if self._closing:
            return
        try:
            selection = self._control_selection()
        except BaseException as error:
            self._status.setText("POINT NOT ACQUIRED")
            self._diagnostic.setText(error_summary(error))
            return
        if selection is None:
            self._status.setText("NEEDS CELL SELECTION")
            self._summary.setText("Choose every non-singleton repeat / point index")
        elif selection != self._presented_selection:
            _r, _p, _logical, label = self._navigation.resolve_selection(selection)
            self._status.setText("EXACT CELL READY TO LOAD")
            self._summary.setText(label)
        self._diagnostic.setText("")

    def _activate_indices(self, indices):
        indices = tuple(indices)
        self._queue_cell(
            self._navigation.selection_for_indices(indices[0], tuple(indices[1:]))
        )

    def _queue_cell(self, selection):
        if self._closing:
            return
        repeat, storage, logical, label = self._navigation.resolve_selection(selection)
        selection = self._navigation.selection_for_indices(repeat, logical)
        self._request_revision += 1
        self._requested_selection = selection
        self._pending_cell = (self._request_revision, selection)
        self._render_requested = False
        self._loaded_view = self._loaded_selection = None
        self._set_controls(selection)
        self._discard_front()
        self._status.setText("BUILDING OCCUPANCY CELL")
        self._summary.setText(
            f"{label} | physical point row {storage} | request {self._request_revision}"
        )
        self._diagnostic.setText("")
        self._start_next()

    def _start_next(self):
        if self._closing or self._future is not None:
            return
        load = self._pending_cell is not None
        if load:
            revision, selection = self._pending_cell
            self._pending_cell = None
            view, color_limits, previous_relim = None, None, None
        elif self._render_requested:
            revision, selection = self._request_revision, self._loaded_selection
            view = self._loaded_view
            if view is None or selection != self._requested_selection:
                return
            self._render_requested = False
            color_limits, previous_relim = self._painted_limits(), self._presented_relim
        else:
            return
        self._sequence += 1
        display = self._display
        self._submit(
            "cell", (revision, selection), _cell_job,
            self._cell_loader if load else None, self._reference, selection,
            self._navigation, view, display, color_limits, previous_relim,
            revision, self._sequence, self._cancelled,
        )

    def _accept_navigation(self, navigation):
        self._navigation = navigation
        self._build_navigator()
        initial, self._initial_selection = self._initial_selection, None
        if initial is None and all(axis.size == 1 for axis in navigation.axes):
            initial = navigation.selection_at_linear(0)
        if initial is None:
            self._refresh_candidate()
            return
        try:
            repeat, _storage, logical, _label_text = navigation.resolve_selection(initial)
            self._queue_cell(navigation.selection_for_indices(repeat, logical))
        except BaseException as error:
            self._status.setText("INITIAL CELL SELECTION INVALID")
            self._diagnostic.setText(error_summary(error))

    def _accept_cell(self, result):
        nav_id, selection, revision, display_revision, view, frame, relim = result
        if (
            self._navigation.identity != nav_id or revision != self._request_revision
            or selection != self._requested_selection
        ):
            return
        self._loaded_view = view
        self._loaded_selection = selection
        if display_revision == self._display.revision:
            self._present(revision, selection, display_revision, frame, relim)
        else:
            self._render_requested = True

    def _present(self, revision, selection, display_revision, frame, relim):
        self._board.present(frame)
        payload = frame.panels[0].display_payload
        if not isinstance(payload, SiteMapPanelPayload):
            raise TypeError("occupancy worker returned a non-SiteMap payload")
        if not self._board_bound:
            self._board.bind_rectangle_selector(
                _PANEL_ID, payload.background.viewport, self._accept_rectangle,
                enabled=False, interaction_callback=self._accept_interaction,
            )
            self._board_bound = True
        self._presented_selection = selection
        self._presented_key, self._presented_relim = (revision, display_revision), relim
        self._status.setText("READY")
        self._diagnostic.setText("")
        self._refresh_summary()
        self._sync_editors()
        self._update_controls()

    def _front_is_current(self):
        if (
            not self._board.has_front
            or self._presented_key != (self._request_revision, self._display.revision)
            or self._presented_selection != self._requested_selection
        ):
            return False
        origin = self._board.visible_image_origin()
        return origin is not None and (
            origin.presentation.selection_revision == self._request_revision
            and origin.presentation.panel_revision == self._display.revision
        )

    def _painted_limits(self):
        payload = self._board.visible_image_payload() if self._board_bound else None
        return None if payload is None else payload.color_limits

    def _current_limits(self):
        return self._painted_limits() if self._front_is_current() else None

    def _refresh_summary(self):
        view = self._loaded_view
        if view is None:
            return
        valid = int(view.site_validity.sum())
        occupied = int((view.occupied & view.site_validity).sum())
        text = (
            f"{view.summary}\noccupied={occupied}/{valid} valid sites | "
            f"invalid={view.site_axis.size - valid}"
        )
        if self._rectangle_candidate is not None:
            lo_x, lo_y, hi_x, hi_y = self._rectangle_candidate
            text += (
                f"\nDISPLAY ONLY rectangle=({lo_x:.4f}, {lo_y:.4f}).."
                f"({hi_x:.4f}, {hi_y:.4f})"
            )
        self._summary.setText(text)

    def _accept_rectangle(self, gesture):
        frame = self._board.front_frame
        if not isinstance(gesture, RectangleGesture) or not self._front_is_current():
            raise RuntimeError("site-map rectangle origin is stale")
        if frame is None or (
            gesture.board_id, gesture.layout_generation, gesture.sequence,
            gesture.source_identity,
        ) != (
            frame.board_id, frame.layout_generation, frame.sequence,
            frame.panels[0].source_identity,
        ):
            raise RuntimeError("site-map rectangle origin is stale")
        self._rectangle_candidate = gesture.normalized_bounds
        self._board.set_image_rectangle_candidate(gesture.normalized_bounds)
        self._refresh_summary()

    def _accept_interaction(self, command: ImageInteractionCommit):
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("unknown image interaction")
        if not self._front_is_current() or command.origin != self._board.visible_image_origin():
            raise RuntimeError("site-map interaction origin is stale")
        if isinstance(command, ImageViewportCommit):
            candidate = image_display_for_viewport(self._display, command.viewport)
        else:
            candidate = replace(
                self._display, revision=self._display.revision + 1,
                relim_mode=RelimMode.FIXED,
                fixed_color_limits=command.color_limits,
            )
        self._commit_display(candidate, interaction=True)

    def _editor_projection(self):
        return (
            image_display_form_values(self._display),
            runtime_range_placeholders(
                self._current_limits(), "color_min", "color_max",
            ),
        )

    def _reload_editor(self, editor):
        if editor not in (self._edit_display, self._setting_display):
            raise ValueError("display editor does not belong to this window")
        values, placeholders = self._editor_projection()
        editor.load(
            revision=self._display.revision, semantic_identity=self._display,
            values=values, runtime_placeholders=placeholders,
        )

    def _sync_editors(self, *, accepted_editor=None, accepted_base=None):
        values, placeholders = self._editor_projection()
        sync_revisioned_form_editors(
            (self._edit_display, self._setting_display),
            revision=self._display.revision, semantic_identity=self._display,
            values=values, runtime_placeholders=placeholders,
            accepted_editor=accepted_editor,
            accepted_base_revision=accepted_base,
        )

    def _apply_display_form(self, editor, base_revision, values):
        try:
            if editor not in (self._edit_display, self._setting_display):
                raise ValueError("display editor does not belong to this window")
            if not self._front_is_current() or base_revision != self._display.revision:
                raise RuntimeError("display edit is stale")
            if not isinstance(values, dict):
                raise TypeError("display form must emit one exact mapping")
            candidate = image_display_from_form(
                self._display, values, current_color_limits=self._current_limits(),
            )
            self._commit_display(
                candidate, accepted_editor=editor, accepted_base=base_revision,
            )
        except BaseException as error:
            self._diagnostic.setText(f"Display edit rejected: {error_summary(error)}")

    def _commit_display(
        self, state, *, accepted_editor=None, accepted_base=None, interaction=False,
    ):
        changed = state != self._display
        expected = self._display.revision + int(changed)
        if not isinstance(state, ImageDisplayState) or state.revision != expected:
            raise ValueError("display commit has an invalid revision")
        if accepted_editor is not None and (
            accepted_base != self._display.revision
            or accepted_editor.base_revision != accepted_base
        ):
            raise ValueError("accepted editor base differs from current revision")
        if interaction and not changed:
            raise ValueError("display interaction cannot commit a no-op")
        if changed:
            image_viewport_for_display_state(state, self._loaded_view.home_viewport)
            self._display, self._rectangle_candidate = state, None
            self._render_requested = True
            self._status.setText(f"RENDERING DISPLAY r{state.revision}")
            self._diagnostic.setText("")
            self._board.set_interaction_readiness(image=False, curve=False)
        self._sync_editors(accepted_editor=accepted_editor, accepted_base=accepted_base)
        self._update_controls()
        self._start_next()

    def _open_display_settings(self):
        self._settings_anchor.toggle(
            self._setting_display,
            prepare=lambda: self._reload_editor(self._setting_display),
        )

    def _update_controls(self):
        fault = self._board.selector_fault if self._board_bound else None
        ready = not self._closing and fault is None and self._front_is_current()
        for widget in (self._setting_button, self._edit_display, self._setting_display):
            widget.setEnabled(ready)
        self._selector_switch.setEnabled(ready)
        self._board.set_interaction_readiness(image=ready, curve=False)
        if fault is not None:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            if self._board.selectors_enabled:
                self._board.set_selectors_enabled(False)
            self._diagnostic.setText(f"Selector failed closed: {error_summary(fault)}")

    def _set_selector_enabled(self, enabled):
        try:
            if enabled and (
                not self._front_is_current() or self._board.selector_fault is not None
            ):
                raise RuntimeError("selector has no healthy current exact front")
            self._board.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            self._board.set_selectors_enabled(False)
            self._diagnostic.setText(f"Selector rejected: {error_summary(error)}")

    def _discard_front(self):
        blocker = QtCore.QSignalBlocker(self._selector_switch)
        self._selector_switch.setChecked(False)
        del blocker
        if self._board_bound:
            self._board.unbind_rectangle_selector()
            self._board_bound = False
        self._board.clear()
        self._presented_selection = self._presented_key = self._presented_relim = None
        self._rectangle_candidate = None
        self._update_controls()

    def _fail(self, error):
        self._discard_front()
        self._status.setText("OCCUPANCY CELL FAILED")
        self._summary.setText("No exact-cell SiteMap front was admitted")
        self._diagnostic.setText(error_summary(error))

    def _job_current(self, kind, key):
        return self._navigation is None if kind == "navigation" else (
            isinstance(key, tuple) and key == (
                self._request_revision, self._requested_selection,
            )
        )

    @QtCore.pyqtSlot()
    def _owner_cycle(self):
        future = self._future
        if future is not None and future.done():
            kind, key = self._active_kind, self._active_key
            self._future = self._active_kind = self._active_key = None
            try:
                result = future.result()
                if not self._closing:
                    self._accept_navigation(result) if kind == "navigation" else self._accept_cell(result)
            except CancelledError:
                if not self._closing and self._job_current(kind, key):
                    self._status.setText("OCCUPANCY CELL CANCELLED")
            except BaseException as error:
                if not self._closing and self._job_current(kind, key):
                    self._fail(error)
            if not self._closing:
                self._start_next()
        if not self._closing:
            self._update_controls()
        self._finish_close()

    def shutdown(self):
        if self._closing or self._closed:
            return
        self._closing = True
        self._pending_cell, self._render_requested = None, False
        self._cancelled.set()
        self._settings_popup.hide()
        self._close_button.setEnabled(False)
        if self._navigator is not None:
            self._navigator.set_interaction_enabled(False)
        self._discard_front()
        self._status.setText("CLOSING")
        if self._future is not None:
            self._future.cancel()
        self._finish_close()

    def _finish_close(self):
        if not self._closing or self._future is not None or self._closed:
            return
        self._cell_loader = self._navigation = None
        self._navigator = self._loaded_view = None
        self._wake.detach()
        self._closed = self._allow_close = True
        QtCore.QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        if self._allow_close:
            release_window(self)
            event.accept()
        else:
            event.ignore()
            self.shutdown()


def open_occupancy_cell_workbench(
    navigation_loader, cell_loader, reference, *, selection=None,
):
    return open_workbench_window(
        lambda: OccupancyCellWindow(
            navigation_loader,
            cell_loader,
            reference,
            selection=selection,
        )
    )


__all__ = ["OccupancyCellWindow", "open_occupancy_cell_workbench"]
