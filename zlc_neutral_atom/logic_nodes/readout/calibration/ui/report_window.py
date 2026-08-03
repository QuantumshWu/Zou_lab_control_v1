"""Lazy viewer for the report pages exported through :mod:`zlc_plot`."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.qt_widgets import FluentLabel, FluentScrollArea

from ..reference import CalibrationArtifactRef


class CalibrationReportWindow(QtWidgets.QWidget):
    """Display the exact PNG pages produced by the shared plot stack."""

    def __init__(
        self,
        report_root: str | Path,
        reference: CalibrationArtifactRef,
    ) -> None:
        super().__init__()
        if not isinstance(reference, CalibrationArtifactRef):
            raise TypeError("reference must be CalibrationArtifactRef")
        root = Path(report_root).expanduser().resolve()
        pages = tuple(sorted(root.glob("*.png")))
        if not pages:
            raise FileNotFoundError(f"Calibration report has no PNG pages: {root}")

        self.setWindowTitle("Calibration Report")
        summary = FluentLabel(
            f"FINAL {reference.record_path} · shared zlc_plot report",
            self,
        )
        summary.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        body = QtWidgets.QWidget(self)
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(12)
        for path in pages:
            pixmap = QtGui.QPixmap(str(path))
            if pixmap.isNull():
                raise ValueError(f"Calibration report page is unreadable: {path}")
            label = QtWidgets.QLabel(body)
            label.setPixmap(pixmap)
            label.setFixedSize(pixmap.size())
            body_layout.addWidget(label)
        body_layout.addStretch(1)
        scroll = FluentScrollArea(self)
        scroll.setWidgetResizable(False)
        scroll.setWidget(body)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(summary)
        layout.addWidget(scroll, 1)
        self.resize(980, 760)


__all__ = ["CalibrationReportWindow"]
