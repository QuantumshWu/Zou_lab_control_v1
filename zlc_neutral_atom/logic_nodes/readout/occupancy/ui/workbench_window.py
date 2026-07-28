"""Interactive Workbench for one exact committed occupancy cell."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
from dataclasses import replace

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.display_range import RelimMode
from zlc_frontend.image_display import (
    ImageDisplayState, image_display_for_viewport, image_display_form_spec,
    image_display_form_values, image_display_from_form,
    image_viewport_for_display_state,
)
from zlc_frontend.plot_layout import panel_surface_geometry
from zlc_frontend.qt_widgets import (
    FluentButton, FluentFormGrid, FluentLabel, FluentPopup, FluentSpinBox,
    FluentSwitch,
    FluentRevisionedFormEditor, FluentTabWidget, GREY,
    RasterPixelRatioObserver, RectangleGesture, SinglePanelHost,
    FluentSettingsPopupAnchor, runtime_range_placeholders, signals_blocked,
    sync_revisioned_form_editors,
)
from zlc_frontend.render import SiteMapPanelPayload
from zlc_frontend.site_map_render import SiteMapComposer
from zlc_frontend.selector import (
    ImageColorLimitsCommit, ImageInteractionCommit, ImageViewportCommit,
)
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import OccupancyArtifactRef
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from zlc_workbench.window_runtime import (
    RASTER_WORK_EXECUTOR,
    SerialWorkerWindow,
    error_summary,
)

from .workbench_jobs import _BOARD_ID, _PANEL_ID, _cell_job, _load_navigation


def _label(parent, name, text, *, wrap=False):
    result = FluentLabel(text, parent)
    result.setObjectName(name)
    result.setWordWrap(wrap)
    return result


def _cell_label(navigation, repeat_index, point_ordinal, logical_point) -> str:
    repeat_axis = navigation.occupancy_schema.repeat_axis
    repeat_coordinate = repeat_axis.coordinate_at(repeat_index)
    repeat_unit = "" if repeat_axis.unit is None else f" {repeat_axis.unit}"
    labels = [
        f"{repeat_axis.name}={repeat_coordinate}{repeat_unit} "
        f"[index {repeat_index}]",
        f"point row={point_ordinal}",
    ]
    for column in navigation.occupancy_schema.point_table.columns:
        value = column.values[point_ordinal]
        unit = "" if column.unit is None else f" {column.unit}"
        labels.append(f"{column.name}={value}{unit}")
    if navigation.occupancy_schema.grid_topology is not None:
        labels.append(f"grid cell={tuple(logical_point)}")
    return " | ".join(labels)


class OccupancyCellWindow(SerialWorkerWindow):
    """Navigate exact cells and interact with one worker-rasterized SiteMap."""

    def __init__(
        self, navigation_loader, cell_loader, reference, *, address,
    ) -> None:
        super().__init__(executor=RASTER_WORK_EXECUTOR)
        if not callable(navigation_loader) or not callable(cell_loader):
            raise TypeError("occupancy loaders must be callable")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if address is not None and not isinstance(address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress or None")
        self._cell_loader = cell_loader
        self._reference, self._initial_address = reference, address
        self._navigation = self._navigator = None
        self._repeat_control = self._point_control = None
        self._previous_cell = self._load_cell_button = self._next_cell = None
        self._display = ImageDisplayState()
        self._loaded_view = self._loaded_address = None
        self._requested_address = self._presented_address = None
        self._presented_key = None
        self._rectangle_candidate = None
        self._request_revision = 0
        self._surface_revision = 0
        self._active_kind, self._active_key = None, None
        self._pending_cell, self._render_requested = None, False
        self._surface_pixel_ratio_observer = RasterPixelRatioObserver(
            self,
            self._apply_surface_pixel_ratio,
        )
        pixel_ratio = self._surface_pixel_ratio_observer.current_ratio
        # This object contains only authored geometry until its first worker
        # call.  Every Agg mutation, including final close, stays on the sole
        # raster-work thread.
        self._surface_geometry = panel_surface_geometry(
            "2x2",
            pixel_ratio=pixel_ratio,
        )
        self._logical_size = self._surface_geometry.logical_size
        self._composer = SiteMapComposer(
            _PANEL_ID,
            board_id=_BOARD_ID,
            surface_geometry=self._surface_geometry,
            title="Site map",
            value_label="Counts",
        )
        self._set_worker_release(self._composer.close)

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
        self._panel_host = SinglePanelHost(
            _PANEL_ID,
            group=_PANEL_ID,
            empty_text="No exact occupancy cell",
            parent=page,
        )
        self._panel_host.setObjectName("occupancyCellBoard")
        self._panel_host.setMinimumSize(320, 240)
        page_layout.addWidget(self._panel_host, 1)
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

        self._close_button.clicked.connect(self.shutdown)
        self._setting_button.clicked.connect(self._open_display_settings)
        self._selector_switch.toggled.connect(self._set_selector_enabled)
        self._panel_host.rectangleSelected.connect(self._accept_rectangle)
        self._panel_host.viewCommitted.connect(self._accept_interaction)
        self._panel_host.colorLimitsCommitted.connect(self._accept_interaction)
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

    def _apply_surface_pixel_ratio(self, ratio: float) -> None:
        """Author a new raster surface only after Qt binds the real screen."""

        if self._closing:
            return
        geometry = panel_surface_geometry(
            "2x2",
            pixel_ratio=ratio,
        )
        if geometry == self._surface_geometry:
            return
        self._surface_geometry = geometry
        self._logical_size = geometry.logical_size
        self._surface_revision += 1
        if self._panel_host.has_front:
            self._discard_front()
        if (
            self._loaded_view is not None
            and self._loaded_address == self._requested_address
        ):
            self._render_requested = True
            self._status.setText(
                f"RENDERING DISPLAY SURFACE r{self._surface_revision}"
            )
        self._start_next()

    def _submit(self, kind, key, function, *args):
        if not self._submit_future(function, *args):
            return False
        self._active_kind, self._active_key = kind, key
        return True

    def _worker_submit_failed(self, error: BaseException) -> None:
        if self._closing:
            self._status.setText("CLOSE FAILED")
            self._diagnostic.setText(error_summary(error))
        else:
            self._fail(error)

    def _worker_release_failed(self, error: BaseException) -> None:
        self._status.setText("CLOSE FAILED")
        self._diagnostic.setText(error_summary(error))

    def _build_navigator(self):
        navigation = self._navigation
        if max(navigation.repeat_count, navigation.point_count) > (1 << 31):
            raise OverflowError("occupancy cell navigator exceeds Qt integer range")
        navigator = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(navigator)
        outer.setContentsMargins(0, 0, 0, 0)
        form = FluentFormGrid(navigator)
        form.setObjectName("occupancyCellAxisForm")
        self._repeat_control = FluentSpinBox(form)
        self._point_control = FluentSpinBox(form)
        for control, count, name in (
            (self._repeat_control, navigation.repeat_count, "occupancyCellRepeat"),
            (self._point_control, navigation.point_count, "occupancyCellPointRow"),
        ):
            control.setObjectName(name)
            if count == 1:
                control.setRange(0, 0)
                control.setValue(0)
            else:
                control.setRange(-1, count - 1)
                control.setSpecialValueText("Select…")
                control.setValue(-1)
            control.valueChanged.connect(self._refresh_candidate)
        form.add_row("Repeat index", self._repeat_control)
        form.add_row("Point row", self._point_control)
        outer.addWidget(form)
        self._previous_cell = FluentButton("Previous", navigator, color=GREY)
        self._load_cell_button = FluentButton("Load exact cell", navigator)
        self._next_cell = FluentButton("Next", navigator, color=GREY)
        buttons = QtWidgets.QHBoxLayout()
        for button, name in (
            (self._previous_cell, "occupancyCellPrevious"),
            (self._load_cell_button, "occupancyCellLoad"),
            (self._next_cell, "occupancyCellNext"),
        ):
            button.setObjectName(name)
            buttons.addWidget(button)
        buttons.addStretch(1)
        outer.addLayout(buttons)
        self._previous_cell.clicked.connect(lambda: self._move_cell(-1))
        self._load_cell_button.clicked.connect(self._activate_address)
        self._next_cell.clicked.connect(lambda: self._move_cell(1))
        self._navigator = navigator
        self._layout.insertWidget(3, navigator)
        self._refresh_candidate()

    def _control_address(self):
        if self._navigation is None or self._navigator is None:
            return None
        indices = (self._repeat_control.value(), self._point_control.value())
        if any(index < 0 for index in indices):
            return None
        return DatasetCellAddress(*indices)

    def _set_controls(self, address):
        repeat, point_ordinal, _logical = self._navigation.resolve_address(address)
        with signals_blocked(self._repeat_control, self._point_control):
            self._repeat_control.setValue(repeat)
            self._point_control.setValue(point_ordinal)
        self._refresh_navigation_controls()

    def _refresh_candidate(self):
        if self._closing:
            return
        try:
            address = self._control_address()
        except BaseException as error:
            self._status.setText("POINT NOT ACQUIRED")
            self._diagnostic.setText(error_summary(error))
            return
        if address is None:
            self._status.setText("NEEDS CELL SELECTION")
            self._summary.setText("Choose every non-singleton repeat / point index")
        elif address != self._presented_address:
            repeat, point_ordinal, logical = self._navigation.resolve_address(address)
            label = _cell_label(
                self._navigation,
                repeat,
                point_ordinal,
                logical,
            )
            self._status.setText("EXACT CELL READY TO LOAD")
            self._summary.setText(label)
        self._diagnostic.setText("")
        self._refresh_navigation_controls()

    def _activate_address(self):
        address = self._control_address()
        if address is not None:
            self._queue_cell(address)

    def _refresh_navigation_controls(self):
        if self._navigator is None:
            return
        address = self._control_address()
        linear = None if address is None else self._navigation.linear_index(address)
        enabled = not self._closing
        self._repeat_control.setEnabled(enabled and self._navigation.repeat_count > 1)
        self._point_control.setEnabled(enabled and self._navigation.point_count > 1)
        self._load_cell_button.setEnabled(enabled and address is not None)
        self._previous_cell.setEnabled(enabled and linear is not None and linear > 0)
        self._next_cell.setEnabled(
            enabled
            and linear is not None
            and linear + 1 < self._navigation.linear_cell_count
        )

    def _move_cell(self, delta):
        address = self._control_address()
        if delta not in (-1, 1) or address is None:
            return
        target = self._navigation.linear_index(address) + delta
        if not 0 <= target < self._navigation.linear_cell_count:
            return
        address = self._navigation.address_at_linear(target)
        self._queue_cell(address)

    def _queue_cell(self, address):
        if self._closing:
            return
        repeat, point_ordinal, logical = self._navigation.resolve_address(address)
        label = _cell_label(
            self._navigation,
            repeat,
            point_ordinal,
            logical,
        )
        address = DatasetCellAddress(repeat, point_ordinal)
        self._request_revision += 1
        self._requested_address = address
        self._pending_cell = (self._request_revision, address)
        self._render_requested = False
        self._loaded_view = self._loaded_address = None
        self._set_controls(address)
        self._discard_front()
        self._status.setText("BUILDING OCCUPANCY CELL")
        self._summary.setText(
            f"{label} | request {self._request_revision}"
        )
        self._diagnostic.setText("")
        self._start_next()

    def _start_next(self):
        if self._closing or self._future is not None:
            return
        load = self._pending_cell is not None
        if load:
            revision, address = self._pending_cell
            self._pending_cell = None
            view = None
        elif self._render_requested:
            revision, address = self._request_revision, self._loaded_address
            view = self._loaded_view
            if view is None or address != self._requested_address:
                return
            self._render_requested = False
        else:
            return
        display = self._display
        surface_revision = self._surface_revision
        self._submit(
            "cell", (revision, address, surface_revision), _cell_job,
            self._cell_loader if load else None, self._reference, address,
            self._navigation, view, display, self._composer,
            revision, self._surface_geometry, surface_revision,
            self._cancelled,
        )

    def _accept_navigation(self, navigation):
        self._navigation = navigation
        self._build_navigator()
        initial, self._initial_address = self._initial_address, None
        if initial is None and navigation.linear_cell_count == 1:
            initial = navigation.address_at_linear(0)
        if initial is None:
            self._refresh_candidate()
            return
        try:
            navigation.resolve_address(initial)
            self._queue_cell(initial)
        except BaseException as error:
            self._status.setText("INITIAL CELL ADDRESS INVALID")
            self._diagnostic.setText(error_summary(error))

    def _accept_cell(self, result):
        (
            nav_id, address, revision, display_revision, surface_revision,
            view, frame,
        ) = result
        if (
            self._navigation.identity != nav_id or revision != self._request_revision
            or address != self._requested_address
        ):
            return
        self._loaded_view = view
        self._loaded_address = address
        if (
            display_revision == self._display.revision
            and surface_revision == self._surface_revision
        ):
            self._present(
                revision,
                address,
                display_revision,
                surface_revision,
                frame,
            )
        else:
            self._render_requested = True

    def _present(
        self, revision, address, display_revision, surface_revision, frame,
    ):
        self._panel_host.present_frame(frame, logical_size=self._logical_size)
        payload = frame.panels[0].display_payload
        if not isinstance(payload, SiteMapPanelPayload):
            raise TypeError("occupancy worker returned a non-SiteMap payload")
        self._presented_address = address
        self._presented_key = (revision, display_revision, surface_revision)
        self._status.setText("READY")
        self._diagnostic.setText("")
        self._refresh_summary()
        self._sync_editors()
        self._update_controls()

    def _front_is_current(self):
        if (
            not self._panel_host.has_front
            or self._presented_key != (
                self._request_revision,
                self._display.revision,
                self._surface_revision,
            )
            or self._presented_address != self._requested_address
        ):
            return False
        origin = self._panel_host.visible_interaction_origin()
        return origin is not None and (
            origin.presentation.selection_revision == self._request_revision
            and origin.presentation.panel_revision == self._display.revision
            and origin.presentation.document_revision == self._surface_revision
        )

    def _painted_limits(self):
        frame = self._panel_host.front_frame
        if frame is None:
            return None
        payload = frame.panels[0].display_payload
        if not isinstance(payload, SiteMapPanelPayload):
            raise TypeError("occupancy front is not a SiteMap payload")
        return payload.background.color_limits

    def _current_limits(self):
        return self._painted_limits() if self._front_is_current() else None

    def _refresh_summary(self):
        view = self._loaded_view
        if view is None:
            return
        text = (
            f"{view.summary}\n{view.site_count_summary}"
        )
        if self._rectangle_candidate is not None:
            lo_x, lo_y, hi_x, hi_y = self._rectangle_candidate
            text += (
                f"\nDISPLAY ONLY rectangle=({lo_x:.4f}, {lo_y:.4f}).."
                f"({hi_x:.4f}, {hi_y:.4f})"
            )
        self._summary.setText(text)

    def _accept_rectangle(self, gesture):
        frame = self._panel_host.front_frame
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
        self._panel_host.set_rectangle_candidate(gesture.normalized_bounds)
        self._refresh_summary()

    def _accept_interaction(self, command: ImageInteractionCommit):
        if not isinstance(command, (ImageViewportCommit, ImageColorLimitsCommit)):
            raise TypeError("unknown image interaction")
        if (
            not self._front_is_current()
            or command.origin != self._panel_host.visible_interaction_origin()
        ):
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
            self._panel_host.set_interaction_ready(False)
        self._sync_editors(accepted_editor=accepted_editor, accepted_base=accepted_base)
        self._update_controls()
        self._start_next()

    def _open_display_settings(self):
        self._settings_anchor.toggle(
            self._setting_display,
            prepare=lambda: self._reload_editor(self._setting_display),
        )

    def _update_controls(self):
        fault = self._panel_host.selector_fault
        ready = not self._closing and fault is None and self._front_is_current()
        for widget in (self._setting_button, self._edit_display, self._setting_display):
            widget.setEnabled(ready)
        self._selector_switch.setEnabled(ready)
        self._panel_host.set_interaction_ready(ready)
        if fault is not None:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            if self._panel_host.selectors_enabled:
                self._panel_host.set_selectors_enabled(False)
            self._diagnostic.setText(f"Selector failed closed: {error_summary(fault)}")

    def _set_selector_enabled(self, enabled):
        try:
            if enabled and (
                not self._front_is_current()
                or self._panel_host.selector_fault is not None
            ):
                raise RuntimeError("selector has no healthy current exact front")
            self._panel_host.set_selectors_enabled(bool(enabled))
        except BaseException as error:
            blocker = QtCore.QSignalBlocker(self._selector_switch)
            self._selector_switch.setChecked(False)
            del blocker
            self._panel_host.set_selectors_enabled(False)
            self._diagnostic.setText(f"Selector rejected: {error_summary(error)}")

    def _discard_front(self):
        blocker = QtCore.QSignalBlocker(self._selector_switch)
        self._selector_switch.setChecked(False)
        del blocker
        self._panel_host.clear()
        self._presented_address = self._presented_key = None
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
                self._request_revision,
                self._requested_address,
                self._surface_revision,
            )
        )

    def _accept_finished_future(self, future: Future) -> None:
        kind, key = self._active_kind, self._active_key
        self._active_kind = self._active_key = None
        try:
            result = future.result()
            if not self._closing:
                if kind == "navigation":
                    self._accept_navigation(result)
                else:
                    self._accept_cell(result)
        except CancelledError:
            if not self._closing and self._job_current(kind, key):
                self._status.setText("OCCUPANCY CELL CANCELLED")
        except BaseException as error:
            if self._closing:
                self._status.setText("CLOSE FAILED")
                self._diagnostic.setText(error_summary(error))
            elif self._job_current(kind, key):
                self._fail(error)

    def _after_worker_completion(self) -> None:
        if not self._closing:
            self._start_next()
            self._update_controls()

    def _before_worker_shutdown(self) -> None:
        self._surface_pixel_ratio_observer.detach()
        self._pending_cell, self._render_requested = None, False
        self._settings_popup.hide()
        self._close_button.setEnabled(False)
        if self._navigator is not None:
            self._navigator.setEnabled(False)
        self._discard_front()
        self._status.setText("CLOSING")
        self._cell_loader = self._navigation = None
        self._navigator = self._loaded_view = None
        self._composer = None
