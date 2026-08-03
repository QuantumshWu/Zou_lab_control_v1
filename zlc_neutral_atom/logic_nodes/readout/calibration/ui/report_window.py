"""Calibration workflow windows over the shared :mod:`zlc_plot` surface."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import CancelledError, Future
import threading

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentLabel,
    FluentScrollArea,
    FluentTabWidget,
    GREY,
    SerialWorkerWindow,
    error_summary,
)
from zlc_neutral_atom.dataset_output import FinalDatasetOutput
from zlc_plot import Qt5PlotWidget, RasterPlotHost

from ..reference import CalibrationArtifactRef
from .plot_report import calibration_plot_hosts, calibration_report_outputs


def _require_not_cancelled(cancelled: threading.Event) -> None:
    if cancelled.is_set():
        raise CancelledError()


def load_calibration_report_outputs(
    loader: Callable,
    reference: CalibrationArtifactRef,
    cancelled: threading.Event,
) -> tuple[dict[str, FinalDatasetOutput], str]:
    """Load domain facts and materialize only the declared FINAL outputs."""

    if not callable(loader):
        raise TypeError("loader must be callable")
    if not isinstance(reference, CalibrationArtifactRef):
        raise TypeError("reference must be CalibrationArtifactRef")
    _require_not_cancelled(cancelled)
    computation = loader(reference)
    _require_not_cancelled(cancelled)
    outputs, summary = calibration_report_outputs(computation, reference)
    _require_not_cancelled(cancelled)
    return outputs, summary


class CalibrationReportSurfaceWindow(SerialWorkerWindow):
    """Thin Calibration shell around the leaf's shared-plot adapter."""

    _plotFutureReady = QtCore.pyqtSignal(object)

    def __init__(
        self,
        *,
        window_title: str,
        mode_text: str,
        loading_summary: str,
        object_prefix: str,
        subject: str,
    ) -> None:
        super().__init__()
        self._prefix = str(object_prefix)
        self._subject = str(subject).upper()
        self._report_outputs: dict[str, FinalDatasetOutput] | None = None
        self._plot_hosts: dict[str, RasterPlotHost] = {}
        self._plot_widgets: dict[str, Qt5PlotWidget] = {}
        self._plot_futures: set[Future] = set()

        self.setWindowTitle(str(window_title))
        self._mode = FluentLabel(str(mode_text), self)
        self._mode.setObjectName(f"{self._prefix}Mode")
        self._status = FluentLabel(f"BUILDING {self._subject}", self)
        self._status.setObjectName(f"{self._prefix}Status")
        self._summary = FluentLabel(str(loading_summary), self)
        self._summary.setObjectName(f"{self._prefix}Summary")
        self._summary.setWordWrap(True)
        self._tabs = FluentTabWidget(self)
        self._tabs.setObjectName(f"{self._prefix}Tabs")
        self._placeholder = FluentLabel(
            f"Building {self._subject.lower()}…",
            self._tabs,
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setMinimumSize(320, 240)
        self._tabs.addTab(self._placeholder, "Loading")
        self._diagnostic = FluentLabel("", self)
        self._diagnostic.setObjectName(f"{self._prefix}Diagnostic")
        self._diagnostic.setWordWrap(True)
        self._diagnostic.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._close_button = FluentButton("Close", self, color=GREY)
        self._close_button.setObjectName(f"close{self._subject.title()}Button")

        self._controls = QtWidgets.QHBoxLayout()
        self._controls.addStretch(1)
        self._controls.addWidget(self._close_button)
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.addWidget(self._mode)
        self._layout.addWidget(self._status)
        self._layout.addWidget(self._summary)
        self._layout.addWidget(self._tabs, 1)
        self._layout.addWidget(self._diagnostic)
        self._layout.addLayout(self._controls)
        self._close_button.clicked.connect(self.shutdown)
        self._plotFutureReady.connect(
            self._accept_plot_future,
            type=QtCore.Qt.QueuedConnection,
        )

    @property
    def report_outputs(self) -> Mapping[str, FinalDatasetOutput] | None:
        return self._report_outputs

    @property
    def raster_ready(self) -> bool:
        return not self._plot_futures and bool(self._plot_widgets) and all(
            widget.presented_front is not None
            for widget in self._plot_widgets.values()
        )

    def _track_plot_future(self, future: object) -> None:
        if not isinstance(future, Future):
            raise TypeError("Calibration plot operation must return Future")
        self._plot_futures.add(future)

        def completed(done: Future) -> None:
            try:
                self._plotFutureReady.emit(done)
            except RuntimeError:
                return

        future.add_done_callback(completed)

    @QtCore.pyqtSlot(object)
    def _accept_plot_future(self, future: object) -> None:
        if not isinstance(future, Future):
            return
        tracked = future in self._plot_futures
        self._plot_futures.discard(future)
        if not tracked or self._closing:
            try:
                future.result()
            except BaseException:
                pass
            return
        try:
            future.result()
        except CancelledError:
            self._status.setText(f"{self._subject} DISPLAY CANCELLED")
        except BaseException as error:
            self._status.setText(f"{self._subject} DISPLAY FAILED")
            self._diagnostic.setText(error_summary(error))

    def _worker_submit_failed(self, error: BaseException) -> None:
        self._status.setText(f"{self._subject} FAILED")
        self._diagnostic.setText(error_summary(error))

    def _worker_release_failed(self, error: BaseException) -> None:
        self._status.setText("CLOSE FAILED")
        self._diagnostic.setText(error_summary(error))

    def _retire_plot_pages(self) -> None:
        self._plot_futures.clear()
        widgets, self._plot_widgets = self._plot_widgets, {}
        hosts, self._plot_hosts = self._plot_hosts, {}
        for widget in widgets.values():
            try:
                widget.close_adapter()
            except RuntimeError:
                pass
        while self._tabs.count():
            page = self._tabs.widget(0)
            self._tabs.removeTab(0)
            page.hide()
            page.deleteLater()
        for host in hosts.values():
            host.close()
        self._placeholder = None

    @QtCore.pyqtSlot(str)
    def _plot_error(self, message: str) -> None:
        self._diagnostic.setText(str(message))

    def _install_report_outputs(
        self,
        outputs: Mapping[str, FinalDatasetOutput],
        summary: str,
    ) -> None:
        values = dict(outputs)
        if any(not isinstance(value, FinalDatasetOutput) for value in values.values()):
            raise TypeError("report outputs must contain FinalDatasetOutput values")
        if not isinstance(summary, str):
            raise TypeError("report summary must be str")
        self._retire_plot_pages()
        entries, operations = calibration_plot_hosts(values)
        if not isinstance(entries, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, tuple)
            or len(value) != 2
            or not isinstance(value[0], str)
            or not isinstance(value[1], RasterPlotHost)
            for key, value in entries.items()
        ):
            raise TypeError("plot host factory returned another contract")
        self._plot_hosts = {
            key: host for key, (_title, host) in entries.items()
        }
        try:
            for key, (title, host) in entries.items():
                scroll = FluentScrollArea(self._tabs)
                scroll.setWidgetResizable(False)
                scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
                widget = Qt5PlotWidget(host, scroll)
                widget.setObjectName(f"{self._prefix}Plot_{key}")
                widget.errorOccurred.connect(self._plot_error)
                scroll.setWidget(widget)
                self._tabs.addTab(scroll, title)
                self._plot_widgets[key] = widget
            for operation in operations:
                self._track_plot_future(operation)
        except BaseException:
            self._retire_plot_pages()
            raise
        self._report_outputs = values
        self._tabs.tabBar().setVisible(len(entries) > 1)
        self._summary.setText(summary)
        self._diagnostic.setText("")

    def _discard_report_outputs(self) -> None:
        self._report_outputs = None
        self._retire_plot_pages()

    def _before_worker_shutdown(self) -> None:
        self._status.setText("CLOSING")
        self._close_button.setEnabled(False)
        self._report_outputs = None
        self._retire_plot_pages()


class CalibrationReportWindow(CalibrationReportSurfaceWindow):
    """Load one FINAL calibration and display its declared output plots."""

    def __init__(
        self,
        computation_loader,
        reference: CalibrationArtifactRef,
    ) -> None:
        if not callable(computation_loader):
            raise TypeError("computation_loader must be callable")
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        super().__init__(
            window_title="Calibration Report",
            mode_text="FROZEN CALIBRATION REPORT · DISPLAY ONLY",
            loading_summary=f"Resolving {reference.target_ref}…",
            object_prefix="calibrationReport",
            subject="report",
        )
        self._submit_future(
            load_calibration_report_outputs,
            computation_loader,
            reference,
            self._cancelled,
        )

    def _accept_finished_future(self, future: Future) -> None:
        try:
            outputs, summary = future.result()
        except CancelledError:
            if not self._closing:
                self._status.setText("REPORT CANCELLED")
        except BaseException as error:
            if not self._closing:
                self._status.setText("REPORT FAILED")
                self._summary.setText("No report was admitted")
                self._diagnostic.setText(error_summary(error))
        else:
            if not self._closing:
                try:
                    self._install_report_outputs(outputs, summary)
                except BaseException as error:
                    self._status.setText("REPORT DISPLAY FAILED")
                    self._diagnostic.setText(error_summary(error))
                else:
                    self._status.setText("READY")


__all__ = [
    "CalibrationReportSurfaceWindow",
    "CalibrationReportWindow",
    "load_calibration_report_outputs",
]
