"""Exact Occupancy-cell navigator using the generic Figure surface."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future

from PyQt5 import QtWidgets

from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import (
    PlotPanelContract,
    plot_panel_display_state,
)
from zlc_frontend.qt_widgets import (
    FigureSurfaceCompletion,
    FigureSurfaceContext,
    FigureSurfaceHost,
    FigureSurfaceLane,
    FigureSurfaceRenderRequest,
    FluentButton,
    FluentFormGrid,
    FluentLabel,
    FluentSpinBox,
    FluentSwitch,
    GREY,
    RASTER_WORK_EXECUTOR,
    SerialWorkerWindow,
    error_summary,
    signals_blocked,
)
from zlc_frontend.site_map_view import SiteMapView
from zlc_neutral_atom.logic_nodes.readout.occupancy.reference import (
    OccupancyArtifactRef,
)
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from .workbench_jobs import _load_cell_figure, _load_navigation


_PANEL_ID = "sites"


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
    """Navigate exact cells while frontend owns every Figure concern."""

    def __init__(
        self,
        navigation_loader,
        cell_loader,
        reference,
        *,
        address,
    ) -> None:
        super().__init__(executor=RASTER_WORK_EXECUTOR)
        if not callable(navigation_loader) or not callable(cell_loader):
            raise TypeError("occupancy loaders must be callable")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if address is not None and not isinstance(address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress or None")

        self._cell_loader = cell_loader
        self._reference = reference
        self._initial_address = address
        self._navigation = self._navigator = None
        self._cell_selection_switch = None
        self._repeat_control = self._point_control = None
        self._previous_cell = self._load_cell_button = self._next_cell = None
        self._requested_address = self._shown_address = None
        self._pending_cell = None
        self._request_revision = 0
        self._active_kind = self._active_key = None

        self.setWindowTitle("Occupancy Cell")
        self._mode = _label(
            self,
            "occupancyCellMode",
            "EXACT OCCUPANCY CELL · SAME-SHOT FRAME + SITES",
        )
        self._status = _label(
            self,
            "occupancyCellStatus",
            "READING FINAL CELL INDEX",
        )
        self._summary = _label(
            self,
            "occupancyCellSummary",
            f"Resolving {reference.target_ref}…",
            wrap=True,
        )
        self._surface_host = FigureSurfaceHost(
            _PANEL_ID,
            empty_text="No exact occupancy cell",
            parent=self,
        )
        self._surface_host.setObjectName("occupancyCellFigureSurface")
        self._surface_host.setMinimumSize(320, 240)
        self._surface_host.set_selectors_enabled(False)
        self._surface_host.set_interaction_ready(False)
        self._surface_host.interactionRejected.connect(self._show_diagnostic)
        self._diagnostic = _label(
            self,
            "occupancyCellDiagnostic",
            "",
            wrap=True,
        )
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName("closeOccupancyCellButton")

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._close_button)
        self._layout = QtWidgets.QVBoxLayout(self)
        for widget in (self._mode, self._status, self._summary):
            self._layout.addWidget(widget)
        self._layout.addWidget(self._surface_host, 1)
        self._layout.addWidget(self._diagnostic)
        self._layout.addLayout(buttons)

        self._surface_lane = FigureSurfaceLane(
            self,
            accept_completion=self._accept_render_completion,
            request_shutdown_wake=self._wake.request_owner_wake,
            submit_compute=RASTER_WORK_EXECUTOR.submit,
        )
        self._close_button.clicked.connect(self.shutdown)
        self._submit(
            "navigation",
            0,
            _load_navigation,
            navigation_loader,
            reference,
            self._cancelled,
        )

    @property
    def worker_idle(self):
        return (
            self._future is None
            and self._pending_cell is None
            and self._surface_lane.idle
        )

    @property
    def navigation_ready(self):
        return self._navigation is not None

    @property
    def raster_ready(self):
        return (
            self._surface_host.has_front
            and self._shown_address == self._requested_address
        )

    def _show_diagnostic(self, detail) -> None:
        self._diagnostic.setText(str(detail))

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

    def _build_navigator(self):
        navigation = self._navigation
        if max(navigation.repeat_count, navigation.point_count) > (1 << 31):
            raise OverflowError("occupancy cell navigator exceeds Qt integer range")
        navigator = QtWidgets.QWidget(self)
        outer = QtWidgets.QVBoxLayout(navigator)
        outer.setContentsMargins(0, 0, 0, 0)
        form = FluentFormGrid(navigator)
        form.setObjectName("occupancyCellAxisForm")
        self._cell_selection_switch = FluentSwitch("Use exact cell", form)
        self._cell_selection_switch.setObjectName("occupancyCellSelection")
        self._cell_selection_switch.setChecked(navigation.linear_cell_count == 1)
        self._cell_selection_switch.toggled.connect(self._refresh_candidate)
        self._repeat_control = FluentSpinBox(form)
        self._point_control = FluentSpinBox(form)
        for control, count, name in (
            (self._repeat_control, navigation.repeat_count, "occupancyCellRepeat"),
            (self._point_control, navigation.point_count, "occupancyCellPointRow"),
        ):
            control.setObjectName(name)
            control.setRange(0, count - 1)
            control.setValue(0)
            control.valueChanged.connect(self._refresh_candidate)
        form.add_row("Cell selection", self._cell_selection_switch)
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
        if (
            self._navigation is None
            or self._navigator is None
            or self._cell_selection_switch is None
            or not self._cell_selection_switch.isChecked()
        ):
            return None
        return DatasetCellAddress(
            self._repeat_control.value(),
            self._point_control.value(),
        )

    def _set_controls(self, address):
        repeat, point_ordinal, _logical = self._navigation.resolve_address(address)
        with signals_blocked(
            self._cell_selection_switch,
            self._repeat_control,
            self._point_control,
        ):
            self._cell_selection_switch.setChecked(True)
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
            self._summary.setText(
                "Enable exact cell selection to choose a repeat / point row"
            )
        elif address != self._shown_address:
            repeat, point_ordinal, logical = self._navigation.resolve_address(address)
            self._status.setText("EXACT CELL READY TO LOAD")
            self._summary.setText(
                _cell_label(
                    self._navigation,
                    repeat,
                    point_ordinal,
                    logical,
                )
            )
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
        selection_enabled = self._cell_selection_switch.isChecked()
        self._cell_selection_switch.setEnabled(
            enabled and self._navigation.linear_cell_count > 1
        )
        self._repeat_control.setEnabled(
            enabled and selection_enabled and self._navigation.repeat_count > 1
        )
        self._point_control.setEnabled(
            enabled and selection_enabled and self._navigation.point_count > 1
        )
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
        if 0 <= target < self._navigation.linear_cell_count:
            self._queue_cell(self._navigation.address_at_linear(target))

    def _queue_cell(self, address):
        if self._closing:
            return
        repeat, point_ordinal, logical = self._navigation.resolve_address(address)
        address = DatasetCellAddress(repeat, point_ordinal)
        self._request_revision += 1
        self._requested_address = address
        self._pending_cell = (self._request_revision, address)
        self._shown_address = None
        self._set_controls(address)
        self._surface_host.clear()
        self._status.setText("LOADING OCCUPANCY CELL")
        self._summary.setText(
            f"{_cell_label(self._navigation, repeat, point_ordinal, logical)} | "
            f"request {self._request_revision}"
        )
        self._diagnostic.setText("")
        self._start_next()

    def _start_next(self):
        if self._closing or self._future is not None or self._pending_cell is None:
            return
        revision, address = self._pending_cell
        self._pending_cell = None
        self._submit(
            "cell",
            (revision, address),
            _load_cell_figure,
            self._cell_loader,
            self._reference,
            address,
            self._navigation,
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

    def _accept_loaded_cell(self, result):
        navigation_id, address, figure, source = result
        if (
            self._navigation.identity != navigation_id
            or address != self._requested_address
        ):
            return
        view = source.site_map
        if not isinstance(view, SiteMapView) or figure.kind is not PlotKind.SITE_MAP:
            raise TypeError("exact Occupancy loader returned another Figure kind")
        contract = PlotPanelContract(
            _PANEL_ID,
            figure,
            size_name="2x2",
            pixel_ratio=float(self.devicePixelRatioF()),
        )
        display = plot_panel_display_state(contract, {}, revision=0)
        source_key = (contract.session_identity, source.session_identity)
        frame_key = (
            navigation_id,
            address,
            view.background_input.ref,
            view.site_state_input.ref,
            view.cell_selection,
            id(view.centers_xy),
        )
        request = FigureSurfaceRenderRequest(
            _PANEL_ID,
            self._request_revision,
            (frame_key, source_key, display),
            source_key,
            frame_key,
            address,
            contract,
            source,
            display,
            None,
        )
        self._status.setText("RENDERING OCCUPANCY CELL")
        self._summary.setText(f"{view.summary}\n{view.site_count_summary}")
        self._surface_lane.enqueue((request,))

    def _accept_render_completion(self, completion: object) -> set[str]:
        reset: set[str] = set()
        if isinstance(completion, str):
            self._fail(RuntimeError(completion))
            return {_PANEL_ID}
        if not isinstance(completion, FigureSurfaceCompletion):
            raise TypeError("Figure surface lane returned another completion")
        if completion.selector_outputs:
            self._diagnostic.setText(
                "Occupancy navigator received unexpected selector output"
            )
        for request, frame, faceted, figure, error in completion.renders:
            surface_id = request.render_surface_id
            if (
                request.panel_id != _PANEL_ID
                or request.request_revision != self._request_revision
                or request.value != self._requested_address
            ):
                reset.add(surface_id)
                continue
            try:
                if error is not None:
                    raise RuntimeError(error)
                if frame is None or faceted is not None or figure is not None:
                    raise TypeError("SiteMap lane returned another front kind")
                self._surface_host.present_frame(
                    frame,
                    context=FigureSurfaceContext.for_frame(
                        frame,
                        figure=None,
                        display=request.display,
                        contract=request.contract,
                    ),
                    logical_size=request.contract.logical_size,
                )
            except BaseException as render_error:
                reset.add(surface_id)
                self._fail(render_error)
                continue
            self._shown_address = request.value
            view = request.source.site_map
            self._status.setText("READY")
            self._summary.setText(f"{view.summary}\n{view.site_count_summary}")
            self._diagnostic.setText("")
            self._refresh_navigation_controls()
        return reset

    def _fail(self, error):
        self._shown_address = None
        if not self._closing:
            self._surface_host.clear()
        self._status.setText("OCCUPANCY CELL FAILED")
        self._summary.setText("No exact-cell SiteMap front was admitted")
        self._diagnostic.setText(error_summary(error))

    def _job_current(self, kind, key):
        return self._navigation is None if kind == "navigation" else (
            isinstance(key, tuple)
            and key == (self._request_revision, self._requested_address)
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
                    self._accept_loaded_cell(result)
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
            self._refresh_navigation_controls()

    def shutdown(self) -> None:
        self._surface_lane.shutdown()
        super().shutdown()

    def _finish_close_if_ready(self) -> None:
        if self._closing and not self._surface_lane.shutdown():
            return
        super()._finish_close_if_ready()

    def _before_worker_shutdown(self) -> None:
        self._pending_cell = None
        self._close_button.setEnabled(False)
        if self._navigator is not None:
            self._navigator.setEnabled(False)
        self._surface_host.close_surface()
        self._status.setText("CLOSING")
        self._cell_loader = self._navigation = None
        self._navigator = None


__all__ = ["OccupancyCellWindow"]
