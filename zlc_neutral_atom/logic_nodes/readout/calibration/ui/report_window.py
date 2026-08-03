"""Interactive Calibration report composed from the shared DataFigure surface."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import FluentLabel, FluentScrollArea
from zlc_plot import ImageFrame, ImagePointOverlay, PointStatus
from zlc_workbench.data_figure.archive_io import load_figure_archive
from zlc_workbench.data_figure.window import DataFigureWindow

from ..reference import CalibrationArtifactRef


def _page_source(archive):
    metadata = dict(archive.archive.metadata)
    overlay = metadata.get("overlay")
    if not isinstance(overlay, dict):
        return archive.archive.snapshot
    statuses = overlay.get("statuses")
    source = ImagePointOverlay(
        int(overlay.get("revision", archive.archive.snapshot.ref.revision.value)),
        overlay.get("coordinates", ()),
        None if overlay.get("point_ids") is None else tuple(overlay["point_ids"]),
        None if overlay.get("labels") is None else tuple(overlay["labels"]),
        None
        if statuses is None
        else tuple(PointStatus(value) for value in statuses),
    )
    return ImageFrame(archive.archive.snapshot, source)


class CalibrationReportWindow(QtWidgets.QWidget):
    """One report window whose pages are ordinary interactive DataFigures.

    PNGs remain explicit exports.  The visible report never paints a PNG and
    therefore inherits selector, Fit, zoom, size, Divider and style behavior
    from the same ``zlc_plot``/DataFigure path as TaskConsole and FigureViewer.
    """

    def __init__(
        self,
        report_root: str | Path,
        reference: CalibrationArtifactRef,
    ) -> None:
        super().__init__()
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        root = Path(report_root).expanduser().resolve()
        archives = tuple(sorted(root.glob("*.npz")))
        if not archives:
            raise FileNotFoundError(
                f"Calibration report has no interactive page archives: {root}"
            )

        self.setWindowTitle("Calibration Report")
        summary = FluentLabel(
            f"FINAL {reference.record_path} · report: {root}",
            self,
        )
        summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        summary.setWordWrap(True)

        tabs = QtWidgets.QTabWidget(self)
        tabs.setObjectName("calibrationReportPages")
        self._panes: list[DataFigureWindow] = []
        for archive_path in archives:
            loaded = load_figure_archive(archive_path)
            page = loaded.archive.metadata.get("calibration_page", archive_path.stem)
            pane = DataFigureWindow(
                _page_source(loaded),
                loaded.archive.spec,
                output_root=root,
                size=loaded.archive.size,
                parameters=loaded.archive.parameters,
                metadata=loaded.archive.metadata,
                embedded=True,
                parent=tabs,
            )
            pane.initialReady.connect(
                lambda _=None, current=pane, data=loaded.archive.metadata: self._configure_page(
                    current, data
                ),
                QtCore.Qt.QueuedConnection,
            )
            tabs.addTab(pane, str(page).title())
            self._panes.append(pane)

        scroll = FluentScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setWidget(tabs)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(scroll, 1)
        self.resize(1120, 820)

    @staticmethod
    def _configure_page(pane: DataFigureWindow, metadata: object) -> None:
        if not isinstance(metadata, Mapping) or metadata.get("calibration_page") != "distribution":
            return
        host = pane.host
        if host is None:
            return
        thresholds = metadata.get("facet_thresholds")
        if isinstance(thresholds, list):
            host.set_facet_thresholds(
                tuple(None if value is None else float(value) for value in thresholds),
                display=False,
            )
        host.fit("bimodal_gaussian", live=False, fit_all_facets=True)

    def teardown(self) -> bool:
        closed = True
        for pane in self._panes:
            closed = bool(pane.teardown()) and closed
        return closed

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt naming
        if self.teardown():
            event.accept()
        else:
            event.ignore()


__all__ = ["CalibrationReportWindow"]
