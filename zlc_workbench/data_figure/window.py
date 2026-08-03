"""Standalone Qt shell around one worker-owned ``zlc_plot`` surface."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import CancelledError, Future
from pathlib import Path
from weakref import ref

from PyQt5 import QtCore, QtWidgets

from zlc_data import OwnedSnapshot
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentComboBox,
    FluentFrame,
    FluentLabel,
    FluentPlotFitPanel,
    FluentPlotParameterPanel,
    FluentPopup,
    FluentScrollArea,
    FluentSettingRow,
    FluentSettingsPopupAnchor,
    FluentStatusStrip,
    FluentSwitch,
    GREY,
    ORANGE,
    window_pad,
)
from zlc_plot import (
    DEFAULTS,
    DisplayDescription,
    FacetFitBatchResult,
    FitResult,
    PlotSpec,
    PlotSessionConfig,
    Qt5PlotWidget,
    RasterOperation,
    RasterPlotHost,
)
from zlc_plot.primitives import ImageFrame, PlotInput
from zlc_workbench.window_runtime import submit_compute

from .archive_io import FigureArchive, LoadedFigureArchive, save_figure_archive


class DataFigureWindow(QtWidgets.QWidget):
    """Frozen-data Figure whose plot, Fit, selector and export share one host."""

    initialReady = QtCore.pyqtSignal()
    initialFailed = QtCore.pyqtSignal(str)
    archiveSaved = QtCore.pyqtSignal(object)
    _futureReady = QtCore.pyqtSignal(object)

    def __init__(
        self,
        snapshot: PlotInput,
        spec: PlotSpec,
        *,
        output_root: Path,
        size: str | None = None,
        parameters: Mapping[str, object] | None = None,
        archive_path: str | Path | None = None,
        metadata: Mapping[str, object] | None = None,
        open_fit: bool = False,
        embedded: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        if not isinstance(snapshot, (OwnedSnapshot, ImageFrame)):
            raise TypeError("snapshot must be a zlc_plot PlotInput")
        if not isinstance(open_fit, bool) or not isinstance(embedded, bool):
            raise TypeError("open_fit and embedded must be bool")
        super().__init__(parent)
        self._snapshot = snapshot
        self._archive_enabled = isinstance(snapshot, OwnedSnapshot)
        self._output_root = Path(output_root).expanduser()
        if not self._output_root.is_absolute():
            raise ValueError("DataFigure output_root must be absolute")
        self._output_root = self._output_root.resolve()
        if not self._output_root.is_dir():
            raise FileNotFoundError(f"DataFigure output_root does not exist: {self._output_root}")
        selected_size = DEFAULTS.layout.validate_preset(
            DEFAULTS.layout.default_preset if size is None else size
        )
        self._metadata = {} if metadata is None else dict(metadata)
        self._archive_path = self._resolve_archive_path(archive_path)
        self._open_fit_when_ready = open_fit
        self._embedded = embedded
        self._closing = False
        self._closed = False
        self._initial_outcome: str | None = None
        self._pending: dict[Future, tuple[str, object]] = {}
        self._host = RasterPlotHost.from_plot(
            snapshot,
            spec,
            size=selected_size,
            parameters=parameters,
        )
        self._plot_widget: QtWidgets.QWidget | None = None
        self._parameter_panel: FluentPlotParameterPanel | None = None
        self._fit_panel: FluentPlotFitPanel | None = None
        self._settings_popup: FluentPopup | None = None
        self._fit_popup: FluentPopup | None = None
        self._shutdown_timer = QtCore.QTimer(self)
        self._shutdown_timer.setInterval(25)
        self._shutdown_timer.timeout.connect(self._poll_shutdown)

        self.setObjectName("dataFigurePane")
        self.setStyleSheet("background: transparent;")
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0 if embedded else window_pad(1), window_pad(1), 0 if embedded else window_pad(1), window_pad(1))
        root.setSpacing(window_pad(0.5))

        controls = FluentFrame(self, bordered=False)
        control_layout = QtWidgets.QHBoxLayout(controls)
        control_layout.setContentsMargins(window_pad(0.5), 0, window_pad(0.5), 0)
        control_layout.setSpacing(window_pad(0.5))
        self.interaction_switch = FluentSwitch("Interact", controls)
        self.interaction_switch.setObjectName("dataFigureInteract")
        self.interaction_switch.setChecked(True)
        self.setting_button = FluentButton("Setting…", controls, color=GREY)
        self.setting_button.setObjectName("dataFigureSetting")
        self.fit_button = FluentButton("Fit…", controls, color=ORANGE)
        self.fit_button.setObjectName("dataFigureFit")
        self.export_button = FluentButton("Export PNG…", controls, color=ORANGE)
        self.export_button.setObjectName("dataFigureExport")
        self.save_button = FluentButton("Save Figure…", controls, color=GREY)
        self.save_button.setObjectName("dataFigureSave")
        control_layout.addWidget(self.interaction_switch)
        control_layout.addWidget(self.setting_button)
        control_layout.addWidget(self.fit_button)
        control_layout.addStretch(1)
        control_layout.addWidget(self.export_button)
        control_layout.addWidget(self.save_button)
        root.addWidget(controls, 0)

        self._surface = FluentFrame(self)
        self._surface.setObjectName("dataFigureSurface")
        self._surface_layout = QtWidgets.QVBoxLayout(self._surface)
        self._surface_layout.setContentsMargins(0, 0, 0, 0)
        self._surface_layout.setSpacing(0)
        self._placeholder = FluentLabel("Preparing figure…", self._surface)
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._surface_layout.addWidget(self._placeholder, 1)
        root.addWidget(self._surface, 1)

        self.status = FluentStatusStrip(self)
        self.status.setObjectName("dataFigureStatus")
        self.status.show_message("Preparing figure…", severity="task")
        root.addWidget(self.status, 0)

        for control in (
            self.interaction_switch,
            self.setting_button,
            self.fit_button,
            self.export_button,
            self.save_button,
        ):
            control.setEnabled(False)
        self.interaction_switch.toggled.connect(self._set_interaction)
        self.setting_button.clicked.connect(self._toggle_settings)
        self.fit_button.clicked.connect(self._toggle_fit)
        self.export_button.clicked.connect(self._choose_export)
        self.save_button.clicked.connect(self._choose_archive)
        if not self._archive_enabled:
            self.save_button.hide()
        self._futureReady.connect(self._accept_future, QtCore.Qt.QueuedConnection)
        self._track(self._host.describe_display(), "initial", self._accept_initial)

    @property
    def host(self) -> RasterPlotHost:
        return self._host

    @property
    def plot_widget(self) -> QtWidgets.QWidget | None:
        return self._plot_widget

    @property
    def worker_idle(self) -> bool:
        return not self._pending

    @property
    def closed(self) -> bool:
        return self._closed

    def _resolve_archive_path(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None
        selected = Path(path).expanduser()
        if not selected.is_absolute():
            selected = self._output_root / selected
        return selected.resolve()

    def _track(self, future: object, action: str, handler) -> None:
        if not isinstance(future, Future):
            raise TypeError(f"{action} operation must return Future")
        self._pending[future] = (action, handler)
        pane_ref = ref(self)

        def completed(done: Future) -> None:
            pane = pane_ref()
            if pane is None:
                return
            try:
                pane._futureReady.emit(done)
            except RuntimeError:
                # The Qt object can disappear after a non-cancellable archive
                # write finishes; no UI owner remains to consume the result.
                return

        future.add_done_callback(completed)

    @QtCore.pyqtSlot(object)
    def _accept_future(self, future: object) -> None:
        if not isinstance(future, Future):
            return
        entry = self._pending.pop(future, None)
        if entry is None:
            return
        action, handler = entry
        if self._closing:
            try:
                future.result()
            except BaseException:
                pass
            return
        try:
            result = future.result()
            handler(result)
        except CancelledError:
            self.status.show_message(f"{action} cancelled", severity="warning")
        except BaseException as error:
            detail = f"{type(error).__name__}: {error}"
            self.status.show_message(detail, severity="error")
            if action == "initial" and self._initial_outcome is None:
                self._initial_outcome = "failed"
                self.initialFailed.emit(detail)

    def _accept_initial(self, operation: object) -> None:
        if not isinstance(operation, RasterOperation) or not isinstance(
            operation.value, DisplayDescription
        ):
            raise TypeError("initial plot description has another type")
        widget = Qt5PlotWidget(self._host, self._surface)
        if not isinstance(widget, QtWidgets.QWidget):
            raise TypeError("Qt5PlotWidget did not return QWidget")
        widget.setObjectName("dataFigurePlot")
        widget.errorOccurred.connect(
            lambda detail: self.status.show_message(detail, severity="error")
        )
        self._surface_layout.removeWidget(self._placeholder)
        self._placeholder.hide()
        self._placeholder.deleteLater()
        self._surface_layout.addWidget(widget, 0, QtCore.Qt.AlignCenter)
        self._plot_widget = widget
        self._build_settings(operation.value)
        self._build_fit()
        for control in (
            self.interaction_switch,
            self.setting_button,
            self.fit_button,
            self.export_button,
            self.save_button,
        ):
            control.setEnabled(True)
        self.status.show_message("Figure ready")
        self._initial_outcome = "ready"
        self.initialReady.emit()
        if self._open_fit_when_ready:
            QtCore.QTimer.singleShot(0, self._toggle_fit)

    def _build_settings(self, description: DisplayDescription) -> None:
        popup = FluentPopup(self)
        popup.setObjectName("dataFigureSettingsPopup")
        content = QtWidgets.QWidget(popup)
        layout = QtWidgets.QVBoxLayout(content)
        layout.setContentsMargins(window_pad(1), window_pad(1), window_pad(1), window_pad(1))
        layout.setSpacing(window_pad(0.5))
        size_combo = FluentComboBox(content)
        size_combo.setObjectName("dataFigureSize")
        for name in description.size_choices:
            size_combo.addItem(name, name)
        size_combo.setCurrentIndex(size_combo.findData(description.size))
        layout.addWidget(FluentSettingRow("Size", size_combo, parent=content))
        panel = FluentPlotParameterPanel(self._host, content)
        panel.setObjectName("dataFigureParameters")
        panel.parameterRejected.connect(self._plot_parameter_rejected)
        scroll = FluentScrollArea(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setWidget(panel)
        layout.addWidget(scroll, 1)
        popup_layout = QtWidgets.QVBoxLayout(popup)
        popup_layout.setContentsMargins(0, 0, 0, 0)
        popup_layout.addWidget(content)
        self._settings_popup = popup
        self._settings_content = content
        self._settings_anchor = FluentSettingsPopupAnchor(popup, self.setting_button)
        self._size_combo = size_combo
        self._parameter_panel = panel
        size_combo.currentIndexChanged.connect(self._size_changed)

    def _build_fit(self) -> None:
        popup = FluentPopup(self)
        popup.setObjectName("dataFigureFitPopup")
        panel = FluentPlotFitPanel(self._host, False, popup)
        panel.setObjectName("dataFigureFitPanel")
        panel.fitAccepted.connect(self._fit_accepted)
        panel.fitRejected.connect(self._fit_rejected)
        layout = QtWidgets.QVBoxLayout(popup)
        layout.setContentsMargins(window_pad(1), window_pad(1), window_pad(1), window_pad(1))
        layout.addWidget(panel)
        self._fit_popup = popup
        self._fit_panel = panel
        self._fit_anchor = FluentSettingsPopupAnchor(popup, self.fit_button)

    @QtCore.pyqtSlot(bool)
    def _set_interaction(self, enabled: bool) -> None:
        widget = self._plot_widget
        if widget is not None:
            widget.set_interaction_enabled(bool(enabled))

    @QtCore.pyqtSlot()
    def _toggle_settings(self) -> None:
        if self._settings_popup is None:
            return
        self._parameter_panel.reconcile()
        self._settings_anchor.toggle(
            self._settings_content,
            minimum_width=390,
            minimum_height=360,
        )

    @QtCore.pyqtSlot()
    def _toggle_fit(self) -> None:
        if self._fit_popup is not None and self._fit_panel is not None:
            self._fit_anchor.toggle(
                self._fit_panel,
                minimum_width=390,
                minimum_height=180,
            )

    @QtCore.pyqtSlot(int)
    def _size_changed(self, _index: int) -> None:
        if self._closing:
            return
        selected = self._size_combo.currentData()
        if not isinstance(selected, str):
            return
        self.status.show_message(f"Resizing to {selected}…", severity="task")
        self._track(
            self._host.set_size(selected),
            "resize",
            lambda _operation: self.status.show_message(f"Size {selected}"),
        )

    @QtCore.pyqtSlot(str, object)
    def _plot_parameter_rejected(self, name: str, error: object) -> None:
        self.status.show_message(
            f"{name}: {type(error).__name__}: {error}",
            severity="error",
        )

    @QtCore.pyqtSlot(object)
    def _fit_accepted(self, result: object) -> None:
        if isinstance(result, FacetFitBatchResult):
            succeeded = sum(
                item is not None and item.success for item in result.results
            )
            self.status.show_message(
                f"Fit complete · {succeeded}/{len(result.results)} facets"
            )
            return
        if not isinstance(result, FitResult):
            self.status.show_message(
                "Fit returned another result type",
                severity="error",
            )
            return
        values = ", ".join(
            f"{name}={value:.6g}"
            for name, value in zip(result.model.parameter_names, result.parameter_values)
        )
        self.status.show_message(f"Fit complete · {values}")

    @QtCore.pyqtSlot(str, object)
    def _fit_rejected(self, action: str, error: object) -> None:
        self.status.show_message(
            f"{action}: {type(error).__name__}: {error}",
            severity="error",
        )

    @QtCore.pyqtSlot()
    def _choose_export(self) -> None:
        selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Figure PNG",
            str(self._output_root / "figure.png"),
            "PNG image (*.png)",
        )
        if selected:
            self.export_png(selected)

    def export_png(self, path: str | Path) -> None:
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = self._output_root / target
        if target.suffix.lower() != ".png":
            target = target.with_suffix(".png")
        target = target.resolve()
        self.status.show_message(f"Exporting {target}…", severity="task")
        self._track(
            self._host.save(target),
            "export",
            lambda _operation: self.status.show_message(f"Exported {target}"),
        )

    @QtCore.pyqtSlot()
    def _choose_archive(self) -> None:
        initial = self._archive_path or (self._output_root / "figure.npz")
        selected, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Figure",
            str(initial),
            "ZLC Figure (*.npz)",
        )
        if selected:
            self.save_archive(selected)

    def save_archive(self, path: str | Path) -> None:
        if not self._archive_enabled or not isinstance(self._snapshot, OwnedSnapshot):
            self.status.show_message(
                "This plot input has no standalone Dataset archive",
                severity="error",
            )
            return
        target = self._resolve_archive_path(path)
        if target is None:  # pragma: no cover - path is required by this method
            raise TypeError("archive path must not be None")
        self.status.show_message(f"Saving {target}…", severity="task")

        def configuration_ready(operation: object) -> None:
            if not isinstance(operation, RasterOperation) or not isinstance(
                operation.value, PlotSessionConfig
            ):
                raise TypeError("plot configuration has another type")
            configuration = operation.value
            archive = FigureArchive(
                self._snapshot,
                configuration.spec,
                configuration.size,
                configuration.parameters,
                self._metadata,
            )
            self._track(
                submit_compute(save_figure_archive, archive, target),
                "archive save",
                lambda saved: self._archive_saved(saved, archive),
            )

        self._track(self._host.configuration(), "archive configuration", configuration_ready)

    def _archive_saved(self, path: object, archive: FigureArchive) -> None:
        if not isinstance(path, Path):
            raise TypeError("archive save returned another path type")
        self._archive_path = path.resolve()
        loaded = LoadedFigureArchive(self._archive_path, archive)
        self.status.show_message(f"Saved {self._archive_path}")
        self.archiveSaved.emit(loaded)

    def teardown(self) -> bool:
        if self._closed:
            return True
        if not self._closing:
            self._closing = True
            for future in tuple(self._pending):
                future.cancel()
            self._pending.clear()
            for popup in (self._settings_popup, self._fit_popup):
                if popup is not None:
                    popup.hide()
            widget = self._plot_widget
            if widget is not None:
                widget.close_adapter()
        self._closed = self._host.close(timeout=0.0)
        if not self._closed and not self._shutdown_timer.isActive():
            self._shutdown_timer.start()
        return self._closed

    @QtCore.pyqtSlot()
    def _poll_shutdown(self) -> None:
        if self._closed or not self._closing:
            self._shutdown_timer.stop()
            return
        self._closed = self._host.close(timeout=0.0)
        if not self._closed:
            return
        self._shutdown_timer.stop()
        if not self._embedded:
            window = getattr(self, "_zlc_window", None)
            if window is not None:
                QtCore.QTimer.singleShot(0, window.close)


__all__ = ["DataFigureWindow"]
