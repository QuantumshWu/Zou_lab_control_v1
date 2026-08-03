"""One generic TaskConsole Logic-node Edit page."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.qt_widgets import (
    FluentScrollArea,
    FluentSectionLabel,
    FormRuntimeContext,
    scaled_px,
)
from zlc_neutral_atom.installation import DeviceCatalogView
from zlc_neutral_atom.logic_node import LogicNodeDescriptor

from .logic_node_parameter_panel import LogicNodeParameterPanel

__all__ = ["LogicNodeEditor"]


class LogicNodeEditor(QtWidgets.QWidget):
    start_requested = QtCore.pyqtSignal()
    stop_requested = QtCore.pyqtSignal()
    draft_changed = QtCore.pyqtSignal()

    def __init__(
        self,
        *,
        title: str,
        descriptor: LogicNodeDescriptor,
        authored,
        inputs,
        device_catalog: DeviceCatalogView,
        project_root,
        pulses_root,
        runtime: FormRuntimeContext,
        parent: QtWidgets.QWidget,
    ):
        if parent is None:
            raise TypeError("LogicNodeEditor requires its tab-stack parent")
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = FluentScrollArea()
        outer.addWidget(scroll)
        page = QtWidgets.QWidget()
        page.setStyleSheet("background: transparent;")
        scroll.set_width_bounded_widget(page)
        col = QtWidgets.QVBoxLayout(page)
        margin = scaled_px(10, minimum=6)
        col.setContentsMargins(margin, margin, margin, margin)
        col.setSpacing(scaled_px(6, minimum=4))
        col.addWidget(FluentSectionLabel(str(title)))
        self.form = LogicNodeParameterPanel(
            descriptor,
            device_catalog=device_catalog,
            project_root=project_root,
            pulses_root=pulses_root,
            runtime=runtime,
            parent=page,
        )
        self.form.seed_binding(authored or {}, inputs or {})
        self.form.start_requested.connect(self.start_requested.emit)
        self.form.stop_requested.connect(self.stop_requested.emit)
        self.form.draft_changed.connect(self.draft_changed.emit)
        col.addWidget(self.form)
        col.addStretch(1)

    def collect_binding(self):
        return self.form.collect_binding()

    def set_running(self, running: bool) -> None:
        self.form.set_running(running)

    def set_status(self, text: str, *, error: bool) -> None:
        self.form.set_status(text, error=error)

    def teardown(self) -> None:
        pass

    def refresh_on_show(self) -> None:
        self.form.refresh_on_show()
