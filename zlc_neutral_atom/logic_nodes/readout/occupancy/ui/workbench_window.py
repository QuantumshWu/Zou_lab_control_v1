"""Exact Occupancy-cell navigation over the sole :mod:`zlc_plot` surface."""

from __future__ import annotations

from concurrent.futures import CancelledError, Future
import threading

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentFormGrid,
    FluentLabel,
    FluentScrollArea,
    FluentSpinBox,
    FluentSwitch,
    GREY,
    SerialWorkerWindow,
    error_summary,
    signals_blocked,
)
from zlc_plot import Qt5PlotWidget, RasterPlotHost
from zlc_neutral_atom.runtime.dataset import DatasetCellAddress

from ..cell import ExactOccupancyCellSource, OccupancyCellDomain
from ..reference import OccupancyArtifactRef
from .plot import occupancy_cell_host, occupancy_cell_summary


def _cancel_point(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def _load_navigation(loader, reference, cancelled) -> OccupancyCellDomain:
    _cancel_point(cancelled)
    result = loader(reference)
    if not isinstance(result, OccupancyCellDomain):
        raise TypeError("navigation loader must return OccupancyCellDomain")
    if result.artifact_identity != reference.target_ref:
        raise ValueError("occupancy navigation names another artifact")
    _cancel_point(cancelled)
    return result


def _load_cell(loader, reference, address, navigation, cancelled):
    _cancel_point(cancelled)
    result = loader(
        reference,
        address,
        expected_navigation=navigation,
    )
    if not isinstance(result, ExactOccupancyCellSource):
        raise TypeError("cell loader must return ExactOccupancyCellSource")
    if result.domain.identity != navigation.identity or result.address != address:
        raise ValueError("cell loader returned another exact address")
    _cancel_point(cancelled)
    return result


def _label(parent, name, text, *, wrap=False):
    result = FluentLabel(text, parent)
    result.setObjectName(name)
    result.setWordWrap(wrap)
    return result


def _cell_label(navigation, address) -> str:
    repeat_index, point_ordinal, logical_point = navigation.resolve_address(address)
    repeat_axis = navigation.occupancy_schema.repeat_axis
    repeat_unit = "" if repeat_axis.unit is None else f" {repeat_axis.unit}"
    labels = [
        f"{repeat_axis.name}={repeat_axis.coordinate_at(repeat_index)}"
        f"{repeat_unit} [index {repeat_index}]",
        f"point row={point_ordinal}",
    ]
    for column in navigation.occupancy_schema.point_table.columns:
        unit = "" if column.unit is None else f" {column.unit}"
        labels.append(f"{column.name}={column.values[point_ordinal]}{unit}")
    if navigation.occupancy_schema.grid_topology is not None:
        labels.append(f"grid cell={logical_point}")
    return " | ".join(labels)


class OccupancyCellWindow(SerialWorkerWindow):
    """Navigate exact cells while zlc_plot owns the complete Figure surface."""

    def __init__(
        self,
        navigation_loader,
        cell_loader,
        reference,
        *,
        address,
    ) -> None:
        super().__init__()
        if not callable(navigation_loader) or not callable(cell_loader):
            raise TypeError("occupancy loaders must be callable")
        if not isinstance(reference, OccupancyArtifactRef):
            raise TypeError("reference must be OccupancyArtifactRef")
        if address is not None and not isinstance(address, DatasetCellAddress):
            raise TypeError("address must be DatasetCellAddress or None")

        self._cell_loader = cell_loader
        self._reference = reference
        self._initial_address = address
        self._navigation: OccupancyCellDomain | None = None
        self._navigator = None
        self._cell_selection_switch = None
        self._repeat_control = self._point_control = None
        self._previous_cell = self._load_cell_button = self._next_cell = None
        self._requested_address = self._shown_address = None
        self._pending_cell = None
        self._request_revision = 0
        self._active_kind = self._active_key = None
        self._plot_host: RasterPlotHost | None = None
        self._plot_widget: Qt5PlotWidget | None = None
        self._retired_hosts: list[RasterPlotHost] = []
        self._set_worker_release(self._release_plot_hosts)

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
        self._surface = FluentScrollArea(self)
        self._surface.setObjectName("occupancyCellPlotSurface")
        self._surface.setWidgetResizable(False)
        self._surface.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._surface.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._surface.setMinimumSize(320, 240)
        self._placeholder = FluentLabel("No exact occupancy cell", self._surface)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setMinimumSize(320, 240)
        self._surface.setWidget(self._placeholder)
        self._diagnostic = _label(
            self,
            "occupancyCellDiagnostic",
            "",
            wrap=True,
        )
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName("closeOccupancyCellButton")

        controls = QtWidgets.QHBoxLayout()
        controls.addStretch(1)
        controls.addWidget(self._close_button)
        self._layout = QtWidgets.QVBoxLayout(self)
        for widget in (self._mode, self._status, self._summary):
            self._layout.addWidget(widget)
        self._layout.addWidget(self._surface, 1)
        self._layout.addWidget(self._diagnostic)
        self._layout.addLayout(controls)
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
        return super().worker_idle and self._pending_cell is None

    @property
    def navigation_ready(self):
        return self._navigation is not None

    @property
    def raster_ready(self):
        return (
            self._plot_widget is not None
            and self._plot_widget.presented_front is not None
            and self._shown_address == self._requested_address
        )

    def _submit(self, kind, key, function, *args):
        if not self._submit_future(function, *args):
            return False
        self._active_kind, self._active_key = kind, key
        return True

    def _worker_submit_failed(self, error: BaseException) -> None:
        if self._closing:
            self._status.setText("CLOSE FAILED")
        else:
            self._fail(error)

    def _worker_release_failed(self, error: BaseException) -> None:
        self._status.setText("CLOSE FAILED")
        self._diagnostic.setText(error_summary(error))

    def _build_navigator(self):
        navigation = self._navigation
        assert navigation is not None
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
            self._status.setText("EXACT CELL READY TO LOAD")
            self._summary.setText(_cell_label(self._navigation, address))
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
        selected = self._cell_selection_switch.isChecked()
        self._cell_selection_switch.setEnabled(
            enabled and self._navigation.linear_cell_count > 1
        )
        self._repeat_control.setEnabled(
            enabled and selected and self._navigation.repeat_count > 1
        )
        self._point_control.setEnabled(
            enabled and selected and self._navigation.point_count > 1
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
        self._navigation.resolve_address(address)
        address = DatasetCellAddress(address.repeat_index, address.point_ordinal)
        self._request_revision += 1
        self._requested_address = address
        self._pending_cell = (self._request_revision, address)
        self._set_controls(address)
        self._status.setText("LOADING OCCUPANCY CELL")
        self._summary.setText(_cell_label(self._navigation, address))
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
            _load_cell,
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

    def _retire_plot(self) -> None:
        widget, self._plot_widget = self._plot_widget, None
        host, self._plot_host = self._plot_host, None
        if widget is not None:
            widget.close_adapter()
            widget.hide()
            widget.deleteLater()
        if host is not None and not host.close(timeout=0.0):
            self._retired_hosts.append(host)

    def _install_cell(self, source: ExactOccupancyCellSource) -> None:
        host = occupancy_cell_host(source)
        try:
            widget = Qt5PlotWidget(host, self._surface)
            widget.setObjectName("occupancyCellPlot")
            widget.errorOccurred.connect(self._diagnostic.setText)
        except BaseException:
            host.close()
            raise
        self._retire_plot()
        previous = self._surface.widget()
        if previous is not None and previous is not widget:
            previous.hide()
            previous.deleteLater()
        self._surface.setWidget(widget)
        self._plot_host = host
        self._plot_widget = widget
        self._shown_address = source.address
        self._status.setText("READY")
        self._summary.setText(occupancy_cell_summary(source))
        self._diagnostic.setText("")
        self._refresh_navigation_controls()

    def _fail(self, error):
        self._shown_address = None
        self._status.setText("OCCUPANCY CELL FAILED")
        self._summary.setText("No exact Occupancy cell was admitted")
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
                elif self._job_current(kind, key):
                    self._install_cell(result)
        except CancelledError:
            if not self._closing and self._job_current(kind, key):
                self._status.setText("OCCUPANCY CELL CANCELLED")
        except BaseException as error:
            if not self._closing and self._job_current(kind, key):
                self._fail(error)

    def _after_worker_completion(self) -> None:
        if not self._closing:
            self._start_next()
            self._refresh_navigation_controls()

    def _release_plot_hosts(self) -> None:
        hosts, self._retired_hosts = tuple(self._retired_hosts), []
        for host in hosts:
            host.close()

    def _before_worker_shutdown(self) -> None:
        self._pending_cell = None
        self._close_button.setEnabled(False)
        if self._navigator is not None:
            self._navigator.setEnabled(False)
        self._retire_plot()
        self._status.setText("CLOSING")
        self._cell_loader = self._navigation = None
        self._navigator = None


__all__ = ["OccupancyCellWindow"]
