"""Thin Qt shell for one autonomous SCAN_SLOT request."""

from __future__ import annotations

from PyQt5 import QtCore, QtGui, QtWidgets

from Zou_lab_control.notebook.facade import (
    Experiment,
    OccupancyScanRequest,
    ScanRequest,
    _prepare_occupancy_scan_for_workbench,
)
from zlc_frontend.qt_board import QtImageBoard, QtOwnerWake
from zlc_frontend.figure import ViewIntent
from zlc_neutral_atom.scan.reference import ScanArtifactRef
from zlc_storage import canonical_digest
from zlc_workbench.progressive_scan import (
    ProgressiveScanSpec,
    ScanDisplayIntent,
    build_occupancy_progressive_spec,
)
from zlc_workbench.scan import (
    FinalScanPresentation,
    PreparedScanPanelRun,
    ScanPanelController,
    ScanPanelViewModel,
)


class _FrozenScanApplication:
    """Composition-owned bridge from a frozen public request to the controller."""

    __slots__ = (
        "_experiment",
        "_request",
        "_display_intent",
        "_final_selection",
        "_final_preferences",
    )

    def __init__(
        self,
        experiment: Experiment,
        request: ScanRequest | OccupancyScanRequest,
        display_intent: ScanDisplayIntent = ScanDisplayIntent(),
    ) -> None:
        if not isinstance(display_intent, ScanDisplayIntent):
            raise TypeError("display_intent must be ScanDisplayIntent")
        if isinstance(request, ScanRequest) and display_intent != ScanDisplayIntent():
            raise ValueError("direct-camera scan has no site display setting")
        self._experiment = experiment
        self._request = request
        self._display_intent = display_intent
        self._final_selection = None
        self._final_preferences = None

    def prepare(self):
        if isinstance(self._request, ScanRequest):
            def start_direct(preview):
                if preview is not None:
                    raise ValueError(
                        "direct camera scan has no progressive counts port"
                    )
                return self._experiment.start_scan(self._request)

            return PreparedScanPanelRun(None, start_direct)
        command = _prepare_occupancy_scan_for_workbench(
            self._experiment,
            self._request,
        )
        identity = canonical_digest(
            {
                "owner": "Zou_lab_control.workbench.occupancy-scan",
                "pulse_document": self._request.pulse_document.fingerprint,
                "source_schema": command.source_schema.fingerprint,
                "output_contract": command.output_contract.fingerprint,
            }
        )[:20]
        progressive = build_occupancy_progressive_spec(
            command.source_schema,
            command.output_contract,
            identity=identity,
            display_intent=self._display_intent,
        )
        self._final_selection = progressive.display_selection
        self._final_preferences = progressive.display_preferences
        def start_occupancy(preview):
            if preview is not None and preview.spec != progressive.preview_spec:
                raise ValueError(
                    "prepared progressive preview budget changed before start"
                )
            return command.start(preview)

        return PreparedScanPanelRun(progressive, start_occupancy)

    def project_final(
        self,
        source_ref: ScanArtifactRef,
        *,
        memory_limit_bytes: int,
    ) -> FinalScanPresentation:
        figure_options = {"memory_limit_bytes": memory_limit_bytes}
        if isinstance(self._request, OccupancyScanRequest):
            if self._final_preferences is None:
                raise RuntimeError("occupancy display was not prepared")
            figure_options.update(
                intent=ViewIntent.CURVE,
                selection=self._final_selection,
                preferences=self._final_preferences,
            )
        figure = self._experiment.figure(source_ref, **figure_options)
        layer = figure.document.layers[0]
        bindings = " · ".join(
            f"{binding.axis_id.value}={binding.role.value.lower()}"
            for binding in layer.view.axis_bindings
        )
        summary = layer.view.intent.value.lower()
        if bindings:
            summary = f"{summary} · {bindings}"
        if layer.view.display_selections:
            summary += f" · selections={len(layer.view.display_selections)}"
        return FinalScanPresentation(
            source_ref,
            figure.to_png_bytes(memory_limit_bytes=memory_limit_bytes),
            summary,
        )


class ScanWorkbenchWindow(QtWidgets.QWidget):
    """Progressive occupancy/final scan panel; rendering never blocks Qt."""

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

        self.setWindowTitle("Autonomous Scan")
        self.resize(900, 680)
        self._allow_close = False
        self._shown_presentation: FinalScanPresentation | None = None
        self._source_pixmap: QtGui.QPixmap | None = None

        progressive = isinstance(request, OccupancyScanRequest)
        self._progressive_requested = progressive
        self._final_mode_text = (
            "OCCUPANCY · CANONICAL FINAL-ONLY"
            if progressive
            else "DIRECT CAMERA · CANONICAL FINAL-ONLY"
        )
        self._mode = QtWidgets.QLabel(
            (
                "PROVISIONAL OCCUPANCY CURVE → CANONICAL FINAL"
                if progressive
                else self._final_mode_text
            ),
            self,
        )
        self._mode.setObjectName("scanMode")
        self._status = QtWidgets.QLabel("IDLE · FINAL-ONLY", self)
        self._status.setObjectName("scanStatus")
        self._artifact = QtWidgets.QLabel("Artifact: —", self)
        self._artifact.setObjectName("scanArtifact")
        self._artifact.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._projection = QtWidgets.QLabel("Display: waiting for FINAL artifact", self)
        self._projection.setObjectName("projectionSummary")
        self._projection.setWordWrap(True)
        self._raster = QtWidgets.QLabel("No FINAL result", self)
        self._raster.setObjectName("scanRaster")
        self._raster.setAlignment(QtCore.Qt.AlignCenter)
        self._raster.setMinimumSize(320, 240)
        self._raster.setStyleSheet("background: #111; color: #bbb;")
        self._provisional_board = QtImageBoard("scan-curve", self)
        self._provisional_board.setObjectName("scanProvisionalBoard")
        self._display_stack = QtWidgets.QStackedWidget(self)
        self._display_stack.addWidget(self._provisional_board)
        self._display_stack.addWidget(self._raster)
        self._display_stack.setCurrentWidget(self._raster)
        self._diagnostics = QtWidgets.QLabel("", self)
        self._diagnostics.setObjectName("scanDiagnostics")
        self._diagnostics.setWordWrap(True)
        self._diagnostics.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self._start = QtWidgets.QPushButton("Run Scan", self)
        self._start.setObjectName("startScanButton")
        self._stop = QtWidgets.QPushButton("Stop", self)
        self._stop.setObjectName("stopScanButton")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._start)
        controls.addWidget(self._stop)
        controls.addStretch(1)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._mode)
        layout.addWidget(self._status)
        layout.addWidget(self._artifact)
        layout.addWidget(self._projection)
        layout.addWidget(self._display_stack, 1)
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
        self._wake.bind(self._owner_cycle)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(40)
        self._timer.timeout.connect(self._owner_cycle)
        self._timer.start()
        self._start.clicked.connect(self._start_scan)
        self._stop.clicked.connect(self._stop_scan)
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
        self._progressive_requested = isinstance(request, OccupancyScanRequest)
        self._final_mode_text = (
            "OCCUPANCY · CANONICAL FINAL-ONLY"
            if self._progressive_requested
            else "DIRECT CAMERA · CANONICAL FINAL-ONLY"
        )
        self._shown_presentation = None
        self._source_pixmap = None
        self._raster.clear()
        self._raster.setText("No FINAL result")
        self._apply_model(self._controller.view_model)

    def shutdown(self) -> None:
        """Begin the same nonblocking close path used by the standalone window."""

        self._controller.close()
        self._owner_cycle()

    def _start_scan(self) -> None:
        try:
            self._controller.start()
        except BaseException as error:
            self._diagnostics.setText(
                f"Start failed: {type(error).__name__}: {error}"
            )
        self._owner_cycle()

    def _stop_scan(self) -> None:
        try:
            self._controller.stop()
        except BaseException as error:
            self._diagnostics.setText(
                f"Stop failed: {type(error).__name__}: {error}"
            )
        self._owner_cycle()

    @QtCore.pyqtSlot()
    def _owner_cycle(self) -> None:
        model = self._controller.owner_cycle()
        self._apply_model(model)
        if model.closed and not self._allow_close:
            self._timer.stop()
            self._wake.detach()
            self._allow_close = True
            QtCore.QTimer.singleShot(0, self.close)

    def _apply_model(self, model: ScanPanelViewModel) -> None:
        progressive_mode = self._progressive_requested and (
            model.generation == 0 or not model.final_only
        )
        self._mode.setText(
            (
                "PROVISIONAL OCCUPANCY CURVE → CANONICAL FINAL"
                if progressive_mode
                else self._final_mode_text
            )
        )
        self._status.setText(model.status)
        self._artifact.setText(
            "Artifact: —"
            if model.artifact_ref is None
            else f"Artifact: {model.artifact_ref.target_ref}"
        )
        self._diagnostics.setText(model.diagnostic or "")
        self._start.setEnabled(model.can_start)
        self._stop.setEnabled(model.can_stop)
        if model.display_phase == "PROVISIONAL":
            self._display_stack.setCurrentWidget(self._provisional_board)
        else:
            self._display_stack.setCurrentWidget(self._raster)
        if model.projection_summary is not None:
            prefix = (
                "Display (PROVISIONAL): "
                if model.display_phase == "PROVISIONAL"
                else "Display: "
            )
            self._projection.setText(prefix + model.projection_summary)
        presentation = model.presentation
        if presentation is None:
            if model.artifact_ref is None:
                if self._shown_presentation is not None:
                    self._shown_presentation = None
                    self._source_pixmap = None
                    self._raster.clear()
                    self._raster.setText("No FINAL result")
                if model.projection_summary is None:
                    self._projection.setText("Display: waiting for scan preparation")
            else:
                self._projection.setText("Display: FINAL artifact retained; raster unavailable")
            return
        self._projection.setText(f"Display: {presentation.projection_summary}")
        if presentation is self._shown_presentation:
            return
        pixmap = QtGui.QPixmap()
        if not pixmap.loadFromData(presentation.png_bytes, "PNG"):
            self._diagnostics.setText("Qt rejected the worker-produced PNG raster")
            return
        self._shown_presentation = presentation
        self._source_pixmap = pixmap
        self._scale_pixmap()

    def _scale_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        target = self._raster.size()
        if (
            target.width() > self._source_pixmap.width()
            or target.height() > self._source_pixmap.height()
        ):
            target = QtCore.QSize(
                min(target.width(), self._source_pixmap.width()),
                min(target.height(), self._source_pixmap.height()),
            )
        self._raster.setPixmap(
            self._source_pixmap.scaled(
                target,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scale_pixmap()

    def closeEvent(self, event) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self._controller.close()
        self._owner_cycle()


def open_scan_workbench(
    experiment: Experiment,
    request: ScanRequest | OccupancyScanRequest,
) -> ScanWorkbenchWindow:
    application = QtWidgets.QApplication.instance()
    owns_application = application is None
    if application is None:
        application = QtWidgets.QApplication([])
    if QtCore.QThread.currentThread() != application.thread():
        raise RuntimeError("scan Workbench must be opened on the Qt GUI thread")
    window = ScanWorkbenchWindow(experiment, request)
    if owns_application:
        window._application_owner = application
    window.show()
    return window


__all__ = ["ScanWorkbenchWindow", "open_scan_workbench"]
