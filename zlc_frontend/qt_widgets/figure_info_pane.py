"""Pure-Qt File/Info pane used by the formal saved FigureViewer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import Enum
from pprint import pformat

from PyQt5 import QtCore, QtWidgets

from .flow_graph_view import FlowGraphView
from .fluent import (
    FluentCodeEdit,
    FluentFrame,
    FluentPathEdit,
    FluentReadoutMultiline,
    FluentScrollArea,
    FluentSectionLabel,
    FluentSettingRow,
    FluentStatusStrip,
    FluentTabWidget,
    scaled_px,
    setting_label_width,
    window_pad,
)
from .style import CARD_PAD


class FigureInfoPane(QtWidgets.QWidget):
    """The established File header and five-tab read-only Info surface.

    ``pathCommitted`` is deliberately narrower than text editing: Browse and
    ``editingFinished`` emit it, while ordinary keystrokes remain a local draft.
    """

    pathCommitted = QtCore.pyqtSignal(str)

    def __init__(self, *, label_names: Iterable[str], parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")

        self._label_width = setting_label_width(tuple(label_names))
        self.setFixedWidth(
            self._label_width + scaled_px(320, minimum=240)
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(window_pad(1), 0, window_pad(1), 0)
        layout.setSpacing(window_pad(0.5))

        header_frame = FluentFrame(bordered=False)
        header_frame.setFixedHeight(scaled_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(
            scaled_px(12),
            scaled_px(6),
            scaled_px(12),
            scaled_px(6),
        )
        header.setSpacing(scaled_px(8, minimum=5))
        header.addWidget(FluentSectionLabel("File"))

        self.path_edit = FluentPathEdit(
            "",
            mode="file",
            caption="Open a saved figure (image or .npz)",
            file_filter="Saved figures (*.png *.jpg *.jpeg *.npz);;All files (*)",
        )
        self.path_edit.setToolTip(
            "Choose a current saved Figure .npz, or its same-stem PNG/JPEG image."
        )
        self.path_edit.selected.connect(self.pathCommitted.emit)
        self.path_edit.edit.editingFinished.connect(self._commit_path_draft)
        header.addWidget(self.path_edit, 1)
        layout.addWidget(header_frame)

        self.info_tabs = FluentTabWidget()
        self.info_tabs.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.plot_layout = self._add_rows_tab("Plot")
        self.measurement_layout = self._add_rows_tab("Measurement")
        self.device_layout = self._add_rows_tab("Device")
        self.flow_view = self._add_flow_tab("Flow")
        self.raw_info = self._add_raw_tab("Raw")
        layout.addWidget(self.info_tabs, 1)

        self.status = FluentStatusStrip()
        self.status.show_message("Open a current saved Figure (.npz).")
        layout.addWidget(self.status)

    @QtCore.pyqtSlot()
    def _commit_path_draft(self) -> None:
        self.pathCommitted.emit(self.path_edit.text())

    def _add_rows_tab(self, title: str) -> QtWidgets.QVBoxLayout:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(scaled_px(3, minimum=2))
        layout.setAlignment(QtCore.Qt.AlignTop)
        scroll.setWidget(body)
        self.info_tabs.add_permanent_tab(scroll, title)
        return layout

    def _add_flow_tab(self, title: str) -> FlowGraphView:
        scroll = FluentScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        view = FlowGraphView()
        view.setToolTip(
            "How this figure's data was produced: raw data / a measurement "
            "signal / through which processor(s). The tree branches upward."
        )
        scroll.setWidget(view)
        self.info_tabs.add_permanent_tab(scroll, title)
        return view

    def _add_raw_tab(self, title: str) -> FluentCodeEdit:
        body = QtWidgets.QWidget()
        body.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(body)
        margin = scaled_px(CARD_PAD, minimum=6)
        layout.setContentsMargins(margin, margin, margin, margin)
        raw = FluentCodeEdit("", read_only=True)
        raw.setToolTip(
            "The complete typed archive description; source array bytes are "
            "represented by schema."
        )
        layout.addWidget(raw)
        self.info_tabs.add_permanent_tab(body, title)
        return raw

    def replace_info(
        self,
        *,
        plot_rows,
        measurement_rows,
        device_rows,
        flow_graph,
        raw_text: str,
    ) -> None:
        """Replace the Info generation after a new pane has been admitted."""

        for layout in (
            self.plot_layout,
            self.measurement_layout,
            self.device_layout,
        ):
            self._clear_layout(layout)
        self._fill_rows(self.plot_layout, plot_rows)
        self._fill_rows(self.measurement_layout, measurement_rows)
        self._fill_rows(self.device_layout, device_rows)
        self.flow_view.set_graph(flow_graph)
        self.raw_info.setPlainText(raw_text)

    @staticmethod
    def _clear_layout(layout: QtWidgets.QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _fill_rows(self, layout, rows) -> None:
        for key, value in rows:
            field = FluentReadoutMultiline(self._readout_text(value))
            field.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored,
                QtWidgets.QSizePolicy.Fixed,
            )
            layout.addWidget(
                FluentSettingRow(
                    str(key),
                    field,
                    label_width=self._label_width,
                )
            )

    @staticmethod
    def _readout_text(value: object) -> str:
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, Mapping):
            return pformat(dict(value), sort_dicts=False, width=80)
        if isinstance(value, (tuple, list)):
            return ", ".join(
                str(item.value if isinstance(item, Enum) else item)
                for item in value
            )
        return str(value)


__all__ = ["FigureInfoPane"]
