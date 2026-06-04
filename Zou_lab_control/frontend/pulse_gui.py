"""Confocal-style PyQt pulse editor for neutral-atom ``PulseSequence``.

The GUI is a front-end only.  It edits ``PulseTableState`` and calls an
optional existing sequencer/experiment; it does not introduce a separate
hardware-control layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import os

from PyQt5 import QtCore, QtGui, QtWidgets

from Zou_lab_control.neutral_atom.timing.pulse_table import PulsePeriod, PulseTableState, default_pulse_name
from .live import plot as frontend_plot, pulse_plot_channels, pulse_repeat_markers, pulse_repeat_notation
from .qt_fluent import (
    ACCENT,
    BG,
    FONT,
    GREEN,
    GREY,
    ORANGE,
    RED,
    YELLOW,
    FloatOrXLineEdit,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentSwitch,
    FluentTabWidget,
    FluentWindow,
    ensure_qt_app,
    fluent_font_size,
    fluent_scrollbar_stylesheet,
    fluent_text_width,
    fluent_widget_stylesheet,
    format_compact_number,
    scaled_px,
    set_fluent_scale,
)

try:  # Matplotlib is already a frontend dependency, but keep import errors tidy.
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - depends on the local desktop environment.
    FigureCanvas = None
    plt = None


TIME_UNITS = ["ns", "us", "ms", "s", "str (ns)"]
UNIT_TO_NS = {"ns": 1.0, "us": 1_000.0, "ms": 1_000_000.0, "s": 1_000_000_000.0, "str (ns)": 1.0}
ROW_HEIGHT = 30
CHANNEL_LABEL_WIDTH = 100
TIME_UNIT_WIDTH = 60
HIDE_BUTTON_WIDTH = 26
PANEL_TOP_HEIGHT = 110
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 118
DEFAULT_WINDOW_RATIO = 0.90
SUMMARY_DEBOUNCE_MS = 90
PREVIEW_DEBOUNCE_MS = 160
PULSE_FILES_ENV = "ZLC_PULSE_DIR"


def _px(value: int | float, *, minimum: int = 1) -> int:
    return scaled_px(value, minimum=minimum)


def _font_metrics() -> QtGui.QFontMetrics:
    return QtGui.QFontMetrics(QtGui.QFont(FONT, fluent_font_size()))


def _text_width(text: str) -> int:
    return fluent_text_width(_font_metrics(), text)


def _row_height() -> int:
    return _px(ROW_HEIGHT, minimum=22)


def _row_spacing() -> int:
    return _px(CHANNEL_ROW_SPACING, minimum=3)


def _channel_label_width() -> int:
    return _px(CHANNEL_LABEL_WIDTH, minimum=84)


def _channel_name_edit_width() -> int:
    return _px(108, minimum=88)


def _delay_edit_width() -> int:
    return _px(76, minimum=62)


def _time_unit_width() -> int:
    return _px(TIME_UNIT_WIDTH, minimum=62)


def _hide_button_width() -> int:
    return _px(HIDE_BUTTON_WIDTH, minimum=22)


def _panel_top_height() -> int:
    return _px(PANEL_TOP_HEIGHT, minimum=120)


def _shadow_pad() -> int:
    return _px(5, minimum=4)


def _panel_width(title: str, content_width: int) -> int:
    return content_width


def _period_card_width() -> int:
    return _px(PERIOD_CARD_WIDTH, minimum=112)


def _period_top_label_width() -> int:
    return _px(56, minimum=46)


def _default_pulse_name() -> str:
    return default_pulse_name()


def _pulse_files_dir() -> Path:
    configured = os.environ.get(PULSE_FILES_ENV, "").strip()
    directory = Path(configured).expanduser() if configured else Path(__file__).resolve().parents[2] / "pulses"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _safe_file_stem(name: str) -> str:
    out = []
    for char in str(name or "").strip():
        if char.isalnum() or char in ("-", "_"):
            out.append(char)
        elif char.isspace():
            out.append("_")
    return "".join(out).strip("_") or _default_pulse_name()


def _period_duration_text(period: PulsePeriod) -> str:
    return str(period.duration)


def _period_control_width(card_width: int) -> int:
    return max(_px(76, minimum=68), card_width - 2 * _px(7) - _px(4))


def _unit_resolution(step_ns: float, unit: str) -> float:
    factor = UNIT_TO_NS.get(unit or "ns", 1.0)
    if factor <= 0:
        return float(step_ns)
    return float(step_ns) / factor


def _summary_time_text(value_ns: float) -> str:
    value_ns = float(value_ns)
    units = (("s", 1_000_000_000.0), ("ms", 1_000_000.0), ("us", 1_000.0), ("ns", 1.0))
    for unit, factor in units:
        if abs(value_ns) >= factor or unit == "ns":
            return f"{format_compact_number(value_ns / factor, digits=6)} {unit}"
    return f"{format_compact_number(value_ns, digits=6)} ns"


def _set_fixed_height(widget: QtWidgets.QWidget, height: int | None = None) -> QtWidgets.QWidget:
    widget.setFixedHeight(_row_height() if height is None else height)
    return widget


def _channel_row_height(channel_count: int) -> int:
    return _px(26 if channel_count > 16 else ROW_HEIGHT, minimum=22)


class PulseStateUIManager(QtCore.QObject):
    class RunState:
        INIT = "INIT"
        PREPARED = "PREPARED"
        RUNNING = "RUNNING"
        STOP = "STOP"
        SAFE = "SAFE"
        ERROR = "ERROR"
        UNSYNCED = "UNSYNCED"

    class FileState:
        UNTITLED = "UNTITLED"
        SAVE = "SAVE"
        LOAD = "LOAD"
        UNSAVED = "UNSAVED"

    def __init__(
        self,
        *,
        status_dot: FluentStatusDot,
        label: FluentLabel,
        save_button: FluentButton,
        title_callback=None,
    ):
        super().__init__()
        self.status_dot = status_dot
        self.label = label
        self.save_button = save_button
        self.title_callback = title_callback
        self.address_str = ""
        self.pulse_name = "pulse"
        self._runstate = self.RunState.INIT
        self._filestate = self.FileState.UNTITLED
        self._update()

    @property
    def runstate(self):
        return self._runstate

    @runstate.setter
    def runstate(self, value):
        self._runstate = value
        self._update()

    @property
    def filestate(self):
        return self._filestate

    @filestate.setter
    def filestate(self, value):
        self._filestate = value
        self._update()

    def _update(self) -> None:
        colors = {
            self.RunState.INIT: GREY,
            self.RunState.PREPARED: YELLOW,
            self.RunState.RUNNING: GREEN,
            self.RunState.STOP: GREEN,
            self.RunState.SAFE: RED,
            self.RunState.ERROR: RED,
            self.RunState.UNSYNCED: ORANGE,
        }
        self.status_dot.set_color(colors.get(self._runstate, GREY))

        local = self.address_str.replace("\\", "/").split("/")[-1] if self.address_str else ""
        pulse_name = self.pulse_name.strip() or "pulse"
        if self._filestate == self.FileState.SAVE:
            status, star = "saved", ""
        elif self._filestate == self.FileState.LOAD:
            status, star = "loaded", ""
        elif self._filestate == self.FileState.UNSAVED:
            status, star = "unsaved", "*"
        else:
            status, star = "new", "*"
        if local:
            text = f"PulseGUI - {pulse_name} ({status}: {local}){star}"
        else:
            text = f"PulseGUI - {pulse_name} ({status}){star}"
        self.label.setText(text)
        self.save_button.setText(f"Save\nPulse{star if star else ''}")
        self.save_button.set_color(YELLOW if star else ACCENT)
        if self.title_callback is not None:
            self.title_callback(f"{pulse_name} - PulseGUI{star}")

    def set_pulse_name(self, name: str) -> None:
        self.pulse_name = str(name or "pulse")
        self._update()


class PeriodCard(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(
        self,
        index: int,
        period: PulsePeriod,
        *,
        total_periods: int = 1,
        channels: Sequence[str],
        labels: dict[str, str],
        hidden_states: dict[str, int] | None = None,
        compact: bool = False,
        time_step_ns: float = 1.0,
        parent=None,
    ):
        super().__init__("", parent)
        self.channels = list(channels)
        self.checks: dict[str, FluentCheckBox] = {}
        self.hidden_states = {str(k): int(v) for k, v in dict(hidden_states or {}).items()}
        self.compact = bool(compact)
        self.time_step_ns = float(time_step_ns)
        self.set_period_position(index, total_periods)

        width = _period_card_width()
        control_width = _period_control_width(width)
        self.setMinimumWidth(width)
        self.setMaximumWidth(width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(7), _px(7), _px(7), _px(7))
        layout.setSpacing(_row_spacing())

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        duration_label = FluentLabel("Duration")
        duration_label.setAlignment(QtCore.Qt.AlignCenter)
        duration_label.setToolTip("Duration")
        top_layout.addWidget(_set_fixed_height(duration_label))

        self.duration_edit = FloatOrXLineEdit(_period_duration_text(period))
        self.duration_edit.set_allow_any(False)
        self.duration_edit.setToolTip("Duration")
        self.duration_edit.setFixedWidth(control_width)
        top_layout.addWidget(_set_fixed_height(self.duration_edit))

        self.unit_combo = FluentComboBox()
        self.unit_combo.addItems(TIME_UNITS)
        self.unit_combo.setCurrentText("str (ns)" if "x" in str(period.duration).lower() else period.unit)
        self.unit_combo.setToolTip("Duration unit")
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(_set_fixed_height(self.unit_combo))
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addSpacing(_px(1, minimum=0))

        row_height = _channel_row_height(len(self.channels))
        for offset, channel in enumerate(self.channels):
            checkbox = FluentCheckBox(labels.get(channel) or channel)
            checkbox.setChecked(bool(period.states[offset]))
            checkbox.setToolTip(channel)
            checkbox.setFixedHeight(row_height)
            checkbox.toggled.connect(self.changed)
            self.checks[channel] = checkbox
            layout.addWidget(checkbox)
        layout.addStretch()

        self.duration_edit.textChanged.connect(self._handle_duration_text)
        self.duration_edit.textChanged.connect(self.changed)
        self.unit_combo.currentTextChanged.connect(self._handle_unit)
        self.unit_combo.currentTextChanged.connect(self.changed)
        self._handle_duration_text(self.duration_edit.text())

    def set_period_position(self, index: int, total: int) -> None:
        self.setTitle(f"Period {int(index) + 1}/{max(1, int(total))}")

    def _handle_duration_text(self, text: str) -> None:
        if "x" in text.lower():
            was_blocked = self.unit_combo.blockSignals(True)
            self.unit_combo.setCurrentText("str (ns)")
            self.unit_combo.blockSignals(was_blocked)
            self.unit_combo.setEnabled(False)
        else:
            self.unit_combo.setEnabled(True)
        self._handle_unit(self.unit_combo.currentText())

    def _handle_unit(self, unit: str) -> None:
        self.duration_edit.set_resolution(_unit_resolution(self.time_step_ns, unit))

    def to_period(self, *, full_channels: Sequence[str], x_ns: float, time_step_ns: float) -> PulsePeriod:
        states = []
        for channel in full_channels:
            if channel in self.checks:
                states.append(1 if self.checks[channel].isChecked() else 0)
            else:
                states.append(1 if self.hidden_states.get(channel, 0) else 0)
        period = PulsePeriod(
            self.duration_edit.text().strip(),
            tuple(states),
            unit=self.unit_combo.currentText(),
            name="",
        )
        period.duration_ns(x_ns=x_ns, time_step_ns=time_step_ns)
        return period

    def set_channel_display_labels(self, labels: Mapping[str, str]) -> None:
        for channel, checkbox in self.checks.items():
            checkbox.setText(str(labels.get(channel) or channel))


class _DragItem:
    def __init__(self, widget: QtWidgets.QWidget, item_type: str):
        self.widget = widget
        self.item_type = item_type


class PulseDragContainer(QtWidgets.QWidget):
    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.items: list[_DragItem] = []
        self.drag_start_pos = None
        self.dragging_index = None
        self.layout_main = QtWidgets.QHBoxLayout(self)
        pad = _shadow_pad()
        self.layout_main.setContentsMargins(pad, pad, pad, pad)
        self.layout_main.setSpacing(_px(5, minimum=3))
        self.layout_main.setAlignment(QtCore.Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.insert_indicator = QtWidgets.QFrame()
        self.insert_indicator.setFrameShape(QtWidgets.QFrame.VLine)
        self.insert_indicator.setStyleSheet(f"background: {ACCENT}; min-width: {_px(3)}px;")
        self.insert_indicator.hide()

    def add_item(self, widget: QtWidgets.QWidget, item_type: str) -> None:
        self.items.append(_DragItem(widget, item_type))
        self.layout_main.addWidget(widget)

    def insert_item(self, index: int, widget: QtWidgets.QWidget, item_type: str) -> None:
        self.items.insert(max(0, min(index, len(self.items))), _DragItem(widget, item_type))
        self.refresh_layout()

    def refresh_layout(self) -> None:
        while self.layout_main.count():
            item = self.layout_main.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        for item in self.items:
            self.layout_main.addWidget(item.widget)
        self.layout_main.addWidget(self.insert_indicator)
        self.insert_indicator.hide()
        self.update_period_titles()
        self.changed.emit()

    def pulse_cards(self) -> list[PeriodCard]:
        return [item.widget for item in self.items if item.item_type == "pulse"]

    def update_period_titles(self) -> None:
        cards = self.pulse_cards()
        total = len(cards)
        for index, card in enumerate(cards):
            card.set_period_position(index, total)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.dragging_index = self._index_at(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_start_pos is None or self.dragging_index is None:
            return super().mouseMoveEvent(event)
        if (event.buttons() & QtCore.Qt.LeftButton) and (event.pos() - self.drag_start_pos).manhattanLength() > QtWidgets.QApplication.startDragDistance():
            self._start_drag(self.dragging_index)
            self.drag_start_pos = None
            self.dragging_index = None
        super().mouseMoveEvent(event)

    def _start_drag(self, index: int) -> None:
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData("application/x-zlc-pulse-card", str(index).encode("utf-8"))
        drag.setMimeData(mime)
        widget = self.items[index].widget
        old_style = widget.styleSheet()
        widget.setStyleSheet(old_style + "; border: 2px solid #808080;")
        drag.exec_(QtCore.Qt.MoveAction)
        widget.setStyleSheet(old_style)
        self.update_period_titles()

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
            self._show_insert_indicator(self._insert_pos(event.pos()))
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        if not event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            return super().dropEvent(event)
        old_index = int(bytes(event.mimeData().data("application/x-zlc-pulse-card")).decode("utf-8"))
        insert_pos = self._insert_pos(event.pos())
        new_items = self.items[:]
        dragged = new_items.pop(old_index)
        if insert_pos > old_index:
            insert_pos -= 1
        new_items.insert(insert_pos, dragged)
        if not self._bracket_ok(new_items):
            event.ignore()
            self.insert_indicator.hide()
            return
        self.items = new_items
        self.refresh_layout()
        self.insert_indicator.hide()
        event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.insert_indicator.hide()
        super().dragLeaveEvent(event)

    def _index_at(self, pos) -> int | None:
        for index, item in enumerate(self.items):
            if item.widget.geometry().contains(pos):
                return index
        return None

    def _insert_pos(self, pos) -> int:
        x = pos.x()
        for index, item in enumerate(self.items):
            geo = item.widget.geometry()
            if x < geo.x() + geo.width() // 2:
                return index
        return len(self.items)

    def _show_insert_indicator(self, index: int) -> None:
        self.layout_main.removeWidget(self.insert_indicator)
        self.layout_main.insertWidget(index, self.insert_indicator)
        self.insert_indicator.show()

    def _bracket_ok(self, items: list[_DragItem]) -> bool:
        start = next((i for i, item in enumerate(items) if item.item_type == "bracket_start"), None)
        end = next((i for i, item in enumerate(items) if item.item_type == "bracket_end"), None)
        if start is None or end is None:
            return True
        return end >= start + 3


class RepeatBracket(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, kind: str, repeat_count: int = 2, parent=None):
        super().__init__("Start" if kind == "start" else "End", parent)
        self.kind = kind
        self.setFixedWidth(_px(78, minimum=60))
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(7), _px(7), _px(7), _px(7))
        layout.setSpacing(_px(6, minimum=4))
        label = FluentLabel("Start\nrepeat" if kind == "start" else "End\nrepeat")
        label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(label)
        self.repeat_spin = None
        if kind == "end":
            self.repeat_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
            self.repeat_spin.setRange(1, 999)
            self.repeat_spin.setValue(repeat_count)
            self.repeat_spin.setFixedHeight(_row_height())
            self.repeat_spin.valueChanged.connect(self.changed)
            layout.addWidget(self.repeat_spin)
        layout.addStretch()


class ChannelNamesPanel(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, state: PulseTableState, parent=None):
        super().__init__("Channel Names and Duration", parent)
        self.state = state
        self.label_edits: dict[str, FluentLineEdit] = {}
        label_w = _channel_label_width()
        edit_w = _channel_name_edit_width()
        panel_w = _panel_width("Channel Names and Duration", label_w + edit_w + _px(5) + _px(20))
        self.setMinimumWidth(panel_w)
        self.setMaximumWidth(panel_w)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(8), _px(8), _px(8), _px(8))
        layout.setSpacing(_row_spacing())

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.name_edit = FluentLineEdit(state.name)
        self.name_edit.setPlaceholderText("pulse name")
        self.name_edit.textChanged.connect(self.changed)
        self._add_labeled_widget(top_layout, "Name:", self.name_edit)

        self.total_label = FluentLineEdit("")
        self.total_label.setEnabled(False)
        self._add_labeled_widget(top_layout, "Total:", self.total_label)
        self.periods_label = FluentLineEdit("")
        self.periods_label.setEnabled(False)
        self._add_labeled_widget(top_layout, "Periods:", self.periods_label)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(state.visible_channels))
        for channel in state.visible_channels:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_px(5, minimum=3))
            hardware = FluentLabel(channel)
            hardware.setToolTip(channel)
            hardware.setAlignment(QtCore.Qt.AlignCenter)
            hardware.setFixedSize(label_w, row_height)
            edit = FluentLineEdit(state.channel_labels.get(channel, ""))
            edit.setPlaceholderText("display name")
            edit.setToolTip(f"Display name for {channel}")
            edit.setFixedSize(edit_w, row_height)
            edit.textChanged.connect(self.changed)
            self.label_edits[channel] = edit
            row.addWidget(hardware)
            row.addWidget(edit, 1)
            layout.addLayout(row)
        layout.addStretch()

    def _add_labeled_widget(self, layout: QtWidgets.QVBoxLayout, label_text: str, widget: QtWidgets.QWidget) -> None:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_px(5, minimum=3))
        label = FluentLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFixedSize(_channel_label_width(), _row_height())
        row.addWidget(label)
        row.addWidget(widget, 1)
        layout.addLayout(row)

    def read_values(self, state: PulseTableState) -> None:
        for channel in state.visible_channels:
            edit = self.label_edits.get(channel)
            if edit is None:
                continue
            label = edit.text().strip()
            if label and label != channel:
                state.channel_labels[channel] = label
            else:
                state.channel_labels.pop(channel, None)


class ChannelPanel(FluentGroupBox):
    changed = QtCore.pyqtSignal()
    clearRequested = QtCore.pyqtSignal(str)

    def __init__(self, state: PulseTableState, parent=None):
        super().__init__("Delay and X", parent)
        self.state = state
        self.delay_edits: dict[str, FloatOrXLineEdit] = {}
        self.delay_units: dict[str, FluentComboBox] = {}
        self.channel_labels: dict[str, FluentLabel] = {}
        label_w = _channel_label_width()
        delay_w = _delay_edit_width()
        unit_w = _time_unit_width()
        hide_w = _hide_button_width()
        content_w = label_w + delay_w + unit_w + hide_w + _px(5) * 3 + _px(20)
        panel_w = _panel_width("Delay and X", content_w)
        self.setMinimumWidth(panel_w)
        self.setMaximumWidth(panel_w)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(8), _px(8), _px(8), _px(8))
        layout.setSpacing(_row_spacing())

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.x_edit = FloatOrXLineEdit(format_compact_number(state.x_ns))
        self.x_edit.set_resolution(state.time_step_ns)
        self.x_edit.textChanged.connect(self.changed)
        self._add_labeled_widget(top_layout, "x (ns):", self.x_edit)

        self.step_edit = FluentLineEdit(format_compact_number(state.time_step_ns))
        self.step_edit.set_resolution(1e-12)
        self.step_edit.textChanged.connect(self.changed)
        self._add_labeled_widget(top_layout, "Step:", self.step_edit)

        self.total_label = FluentLineEdit("")
        self.total_label.setEnabled(False)
        self._add_labeled_widget(top_layout, "Visible:", self.total_label)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(state.visible_channels))
        for channel in state.visible_channels:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_px(5, minimum=3))
            label = FluentLabel(state.label_for(channel))
            label.setToolTip(channel)
            label.setAlignment(QtCore.Qt.AlignCenter)
            label.setFixedSize(label_w, row_height)
            self.channel_labels[channel] = label

            delay_edit = FloatOrXLineEdit(str(state.delays.get(channel, 0)))
            delay_edit.setFixedSize(delay_w, row_height)
            delay_edit.textChanged.connect(lambda text, ch=channel: self._handle_delay_text(ch, text))
            delay_edit.textChanged.connect(self.changed)
            unit = FluentComboBox()
            unit.addItems(TIME_UNITS)
            unit.setCurrentText(state.delay_units.get(channel, "ns"))
            unit.setFixedSize(unit_w, row_height)
            unit.currentTextChanged.connect(lambda unit_text, ch=channel: self._handle_delay_unit(ch, unit_text))
            unit.currentTextChanged.connect(self.changed)
            clear_btn = FluentButton("X", color=ORANGE)
            clear_btn.setFixedSize(hide_w, row_height)
            clear_btn.setToolTip("Set this channel fully off.")
            clear_btn.clicked.connect(lambda _=False, ch=channel: self.clearRequested.emit(ch))

            self.delay_edits[channel] = delay_edit
            self.delay_units[channel] = unit
            row.addWidget(label)
            row.addWidget(delay_edit, 1)
            row.addWidget(unit)
            row.addWidget(clear_btn)
            layout.addLayout(row)
            self._handle_delay_text(channel, delay_edit.text())
            self._handle_delay_unit(channel, unit.currentText())
        layout.addStretch()

    def _add_labeled_widget(self, layout: QtWidgets.QVBoxLayout, label_text: str, widget: QtWidgets.QWidget) -> None:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_px(5, minimum=3))
        label = FluentLabel(label_text)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setFixedSize(_channel_label_width(), _row_height())
        row.addWidget(label)
        row.addWidget(widget, 1)
        layout.addLayout(row)

    def _handle_delay_text(self, channel: str, text: str) -> None:
        combo = self.delay_units.get(channel)
        edit = self.delay_edits.get(channel)
        if combo is None or edit is None:
            return
        if "x" in text.lower():
            was_blocked = combo.blockSignals(True)
            combo.setCurrentText("str (ns)")
            combo.blockSignals(was_blocked)
            combo.setEnabled(False)
        else:
            combo.setEnabled(True)
        self._handle_delay_unit(channel, combo.currentText())

    def _handle_delay_unit(self, channel: str, unit: str) -> None:
        edit = self.delay_edits.get(channel)
        if edit is not None:
            edit.set_resolution(_unit_resolution(self.state.time_step_ns, unit))

    def read_values(self, state: PulseTableState) -> None:
        for channel in state.visible_channels:
            if channel in self.delay_edits:
                raw = self.delay_edits[channel].text().strip() or 0
                state.delays[channel] = raw
                state.delay_units[channel] = self.delay_units[channel].currentText()

    def set_channel_display_labels(self, labels: Mapping[str, str]) -> None:
        for channel, label in self.channel_labels.items():
            label.setText(str(labels.get(channel) or channel))


class PulseSequenceEditor(QtWidgets.QWidget):
    """Confocal-style period-card editor for ``PulseTableState``."""

    def __init__(
        self,
        state: PulseTableState | None = None,
        *,
        channels: Sequence[str] | None = None,
        sequencer=None,
        experiment=None,
        channel_labels: Mapping[str, str] | None = None,
        scale: float | None = None,
        window_ratio: float = DEFAULT_WINDOW_RATIO,
        parent=None,
    ):
        app = ensure_qt_app()
        super().__init__(parent)
        self.ui_scale = self._resolve_scale(scale, app=app)
        set_fluent_scale(self.ui_scale)
        self.window_ratio = max(0.45, min(1.0, float(window_ratio)))
        if state is None:
            if channels is None and sequencer is not None and hasattr(sequencer, "channels"):
                channels = sequencer.channels
            if channels is None and experiment is not None and hasattr(experiment, "devices"):
                sequencer = getattr(experiment.devices, "sequencer", sequencer)
                channels = getattr(sequencer, "channels", channels)
            channels = list(channels or ["trap", "cooling", "probe", "qcm_trigger"])
            labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
            state = PulseTableState(
                channels=channels,
                visible_channels=channels[: min(4, len(channels))],
                time_step_ns=self._clock_step_ns(sequencer) or 1.0,
                channel_labels=labels,
            )
        if state is not None and channels is not None and list(state.channels) != list(channels):
            state = state.aligned_to_channels(channels)
        if state is not None and channel_labels:
            for key, value in dict(channel_labels).items():
                channel = str(key)
                label = str(value)
                if channel in state.channels and channel not in state.channel_labels and label and label != channel:
                    state.channel_labels[channel] = label
            state.validate()
        self.state = state
        self.sequencer = sequencer or (getattr(getattr(experiment, "devices", None), "sequencer", None) if experiment is not None else None)
        self.last_program = None
        self.bracket_exists = False
        self.address_str = ""
        self._last_save_state = None
        self._last_load_state = None
        self._building = False
        self._preview_dirty = True
        self._preview_plot = None
        self._preview_canvas = None
        self._left_panels_collapsed = False
        self._summary_timer = QtCore.QTimer(self)
        self._summary_timer.setSingleShot(True)
        self._summary_timer.setInterval(SUMMARY_DEBOUNCE_MS)
        self._summary_timer.timeout.connect(self._update_summary)
        self._preview_timer = QtCore.QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self.refresh_preview)
        self._build_ui()
        self.load_state(self.state)

    def _build_ui(self) -> None:
        self.setWindowTitle("PulseGUI@Zou lab")
        self.setFixedSize(self._target_editor_size())
        self.setStyleSheet(fluent_widget_stylesheet())

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(_px(18), _px(8), _px(18), _px(8))
        root.setSpacing(_px(8, minimum=5))

        header_frame = FluentFrame()
        header_frame.setFixedHeight(_px(48, minimum=38))
        header = QtWidgets.QHBoxLayout(header_frame)
        header.setContentsMargins(_px(12), _px(6), _px(12), _px(6))
        header.setSpacing(_px(8, minimum=5))
        self.status_dot = FluentStatusDot(size=16)
        self.label_name = FluentLabel("PulseGUI - Untitled*")
        self.label_name.setMinimumWidth(_px(260, minimum=180))
        self.label_name.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
        self.summary = FluentLineEdit("")
        self.summary.setEnabled(False)
        header.addWidget(self.status_dot)
        header.addWidget(self.label_name)
        header.addWidget(self.summary, 1)
        root.addWidget(header_frame)

        self.tabs = FluentTabWidget()
        self.edit_tab = QtWidgets.QWidget()
        self.edit_tab.setStyleSheet("background: transparent;")
        edit_layout = QtWidgets.QVBoxLayout(self.edit_tab)
        tab_margin = _px(8, minimum=5)
        edit_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        edit_layout.setSpacing(_px(8, minimum=5))

        self.dataset_scroll = FluentScrollArea()
        self.dataset_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.dataset_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.dataset_body = QtWidgets.QWidget()
        dataset = QtWidgets.QHBoxLayout(self.dataset_body)
        shadow_pad = _shadow_pad()
        dataset.setContentsMargins(0, 0, 0, 0)
        dataset.setSpacing(0)
        self.names_panel_holder = QtWidgets.QWidget()
        self.names_panel_layout = QtWidgets.QVBoxLayout(self.names_panel_holder)
        self.names_panel_layout.setContentsMargins(shadow_pad, shadow_pad, shadow_pad, shadow_pad)
        self.names_panel_layout.setSpacing(0)
        dataset.addWidget(self.names_panel_holder)

        self.channel_panel_holder = QtWidgets.QWidget()
        self.channel_panel_layout = QtWidgets.QVBoxLayout(self.channel_panel_holder)
        self.channel_panel_layout.setContentsMargins(shadow_pad, shadow_pad, shadow_pad, shadow_pad)
        self.channel_panel_layout.setSpacing(0)
        dataset.addWidget(self.channel_panel_holder)

        self.left_panel_stub = FluentFrame()
        self.left_panel_stub.setFixedWidth(_px(82, minimum=68))
        stub_layout = QtWidgets.QVBoxLayout(self.left_panel_stub)
        stub_layout.setContentsMargins(_px(6), _px(8), _px(6), _px(8))
        stub_layout.setSpacing(_px(6, minimum=4))
        stub_label = FluentLabel("Name\nDelay")
        stub_label.setAlignment(QtCore.Qt.AlignCenter)
        stub_layout.addWidget(stub_label)
        self.stub_show_button = FluentButton("Show", color=ACCENT)
        self.stub_show_button.setFixedHeight(_row_height())
        self.stub_show_button.clicked.connect(self.show_left_panels)
        stub_layout.addWidget(self.stub_show_button)
        stub_layout.addStretch()
        self.left_panel_stub.hide()
        dataset.addWidget(self.left_panel_stub)

        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.drag_container = PulseDragContainer()
        self.drag_container.changed.connect(self._mark_dirty)
        self.scroll.setWidget(self.drag_container)
        dataset.addWidget(self.scroll, 1)
        self.dataset_scroll.setWidget(self.dataset_body)
        edit_layout.addWidget(self.dataset_scroll, 1)

        self.timeline_hbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.timeline_hbar.setStyleSheet(fluent_scrollbar_stylesheet("QScrollBar"))
        self.timeline_hbar.setFixedHeight(_px(12, minimum=10))
        self.timeline_hbar.hide()
        self.timeline_hbar_spacer = QtWidgets.QWidget()
        hbar_row = QtWidgets.QHBoxLayout()
        hbar_row.setContentsMargins(0, 0, 0, 0)
        hbar_row.setSpacing(0)
        hbar_row.addWidget(self.timeline_hbar_spacer)
        hbar_row.addWidget(self.timeline_hbar, 1)
        edit_layout.addLayout(hbar_row)
        inner_hbar = self.scroll.horizontalScrollBar()
        inner_hbar.rangeChanged.connect(self._sync_timeline_scrollbar)
        inner_hbar.valueChanged.connect(self.timeline_hbar.setValue)
        self.timeline_hbar.valueChanged.connect(inner_hbar.setValue)

        self.button_frame = FluentFrame(shadow=False)
        bottom = QtWidgets.QHBoxLayout(self.button_frame)
        bottom.setContentsMargins(shadow_pad, shadow_pad, shadow_pad, shadow_pad)
        bottom.setSpacing(_px(6, minimum=4))

        button_area = QtWidgets.QWidget()
        button_layout = QtWidgets.QGridLayout(button_area)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(_px(8, minimum=5))

        self.safe_button = self._control_button("Stop\nPulse", self.safe_state, RED)
        self.fire_button = self._control_button("On\nPulse", self.fire, GREEN)
        self.remove_button = self._control_button("Remove\nColumn", self.remove_period, ORANGE)
        self.add_button = self._control_button("Add\nColumn", self.add_period, ACCENT)
        self.bracket_button = self._control_button("Add\nBracket", self.toggle_bracket, YELLOW)
        self.save_button = self._control_button("Save to\nfile*", self.save_to_file, YELLOW)
        self.load_button = self._control_button("Load\nfrom\nfile", self.load_from_file, ORANGE)
        self.collapse_button = self._control_button("Collapse\nLeft", self.toggle_left_panels, GREY)
        button_positions = [
            (self.safe_button, 0, 0),
            (self.fire_button, 0, 1),
            (self.remove_button, 0, 2),
            (self.add_button, 0, 3),
            (self.bracket_button, 0, 4),
            (self.collapse_button, 0, 5),
            (self.save_button, 1, 0),
            (self.load_button, 1, 1),
        ]
        for button, row, col in button_positions:
            button_layout.addWidget(button, row, col)
        for col in range(6):
            button_layout.setColumnStretch(col, 1)
        bottom.addWidget(button_area, 1)

        self.channel_view = FluentGroupBox("Channel View")
        panel_margin = _px(8)
        panel_gap = _px(6, minimum=4)
        channel_control_w = _px(124, minimum=98)
        channel_control_h = _px(46, minimum=38)
        panel_width = panel_margin * 2 + panel_gap + channel_control_w * 2
        self.channel_view.setFixedWidth(panel_width)
        view_layout = QtWidgets.QGridLayout(self.channel_view)
        view_layout.setContentsMargins(panel_margin, panel_margin, panel_margin, panel_margin)
        view_layout.setHorizontalSpacing(panel_gap)
        view_layout.setVerticalSpacing(_px(8, minimum=5))
        self.add_channel_combo = FluentComboBox()
        self.add_channel_combo.setFixedSize(channel_control_w, channel_control_h)
        self.add_channel_button = FluentButton("Add\nChannel", color=ACCENT)
        self.add_channel_button.setFixedSize(channel_control_w, channel_control_h)
        self.add_channel_button.clicked.connect(self.add_selected_channel)
        self.hide_off_button = FluentButton("Hide\nOff", color=ORANGE)
        self.hide_off_button.setFixedSize(channel_control_w, channel_control_h)
        self.hide_off_button.clicked.connect(self.hide_off_channels)
        self.show_all_button = FluentButton("Show\nAll", color=ACCENT)
        self.show_all_button.setFixedSize(channel_control_w, channel_control_h)
        self.show_all_button.clicked.connect(self.show_all_channels)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.visible_label.setFixedHeight(_row_height())
        view_layout.addWidget(self.add_channel_combo, 0, 0)
        view_layout.addWidget(self.add_channel_button, 0, 1)
        view_layout.addWidget(self.hide_off_button, 1, 0)
        view_layout.addWidget(self.show_all_button, 1, 1)
        view_layout.addWidget(self.visible_label, 2, 0, 1, 2)
        view_layout.setColumnMinimumWidth(0, channel_control_w)
        view_layout.setColumnMinimumWidth(1, channel_control_w)
        bottom.addWidget(self.channel_view)
        edit_layout.addWidget(self.button_frame)
        self.tabs.addTab(self.edit_tab, "Edit")

        self.preview_tab = QtWidgets.QWidget()
        self.preview_tab.setStyleSheet("background: transparent;")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_tab)
        preview_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        preview_layout.setSpacing(_px(8, minimum=5))

        preview_controls = FluentFrame()
        preview_controls.setFixedHeight(_px(48, minimum=38))
        preview_row = QtWidgets.QHBoxLayout(preview_controls)
        preview_row.setContentsMargins(_px(12), _px(6), _px(12), _px(6))
        preview_row.setSpacing(_px(10, minimum=6))
        self.preview_include_off = FluentSwitch("Show always-off")
        self.preview_include_off.toggled.connect(self._request_preview_refresh)
        self.preview_save_figure_button = FluentButton("Save Figure", color=ACCENT)
        self.preview_save_figure_button.setFixedSize(_px(124, minimum=108), _px(34, minimum=28))
        self.preview_save_figure_button.clicked.connect(self.save_figure)
        self.preview_status = FluentLineEdit("")
        self.preview_status.setEnabled(False)
        preview_row.addWidget(self.preview_include_off)
        preview_row.addWidget(self.preview_status, 1)
        preview_row.addWidget(self.preview_save_figure_button)
        preview_layout.addWidget(preview_controls)

        self.preview_scroll = FluentScrollArea()
        self.preview_scroll.setWidgetResizable(False)
        self.preview_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.preview_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.preview_body = QtWidgets.QWidget()
        self.preview_body_layout = QtWidgets.QVBoxLayout(self.preview_body)
        self.preview_body_layout.setContentsMargins(_px(8), _px(8), _px(8), _px(8))
        self.preview_body_layout.setSpacing(0)
        self.preview_placeholder = FluentLabel("Open Preview to render the pulse plot.")
        self.preview_placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_body_layout.addWidget(self.preview_placeholder)
        self.preview_scroll.setWidget(self.preview_body)
        preview_layout.addWidget(self.preview_scroll, 1)
        self.tabs.addTab(self.preview_tab, "Preview")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        root.addWidget(self.tabs, 1)

        self.stateui_manager = PulseStateUIManager(
            status_dot=self.status_dot,
            label=self.label_name,
            save_button=self.save_button,
            title_callback=self._set_gui_title,
        )

    def _control_button(self, text: str, slot, color: str) -> FluentButton:
        button = FluentButton(text, color=color)
        button.setFixedSize(_px(136, minimum=104), _px(66, minimum=50))
        button.clicked.connect(slot)
        return button

    def toggle_left_panels(self) -> None:
        if self._left_panels_collapsed:
            self.show_left_panels()
        else:
            self.hide_left_panels()

    def hide_left_panels(self) -> None:
        self._left_panels_collapsed = True
        self.names_panel_holder.hide()
        self.channel_panel_holder.hide()
        self.left_panel_stub.show()
        self.collapse_button.setText("Show\nLeft")
        self._sync_dataset_geometry()

    def show_left_panels(self) -> None:
        self._left_panels_collapsed = False
        self.left_panel_stub.hide()
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.collapse_button.setText("Collapse\nLeft")
        self._sync_dataset_geometry()

    def _target_editor_size(self) -> QtCore.QSize:
        app = QtWidgets.QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QtCore.QSize(_px(1280, minimum=960), _px(760, minimum=620))
        available = screen.availableGeometry()
        titlebar_allowance = _px(36, minimum=28)
        margin_w = _px(40, minimum=28)
        margin_h = _px(48, minimum=32)
        max_w = max(360, available.width() - margin_w)
        max_h = max(320, available.height() - titlebar_allowance - margin_h)
        min_w = min(_px(980, minimum=820), max_w)
        min_h = min(_px(640, minimum=560), max_h)
        desired_w = min(max_w, int(available.width() * self.window_ratio))
        desired_h = min(max_h, int(available.height() * self.window_ratio) - titlebar_allowance)
        return QtCore.QSize(
            max(min_w, desired_w),
            max(min_h, desired_h),
        )

    def _set_gui_title(self, title: str) -> None:
        self.setWindowTitle(title)
        window = self.window()
        if window is not self:
            window.setWindowTitle(title)
            title_bar = getattr(window, "titleBar", None)
            if title_bar is not None and hasattr(title_bar, "setTitle"):
                try:
                    title_bar.setTitle(title)
                except Exception:
                    pass

    def load_state(self, state: PulseTableState) -> None:
        self._building = True
        self.state = state
        self._rebuild_channel_panels()
        self._rebuild_periods()
        self._refresh_hidden_combo()
        self._sync_dataset_geometry()
        self._building = False
        self._preview_dirty = True
        self._update_summary()

    def _clear_layout(self, layout: QtWidgets.QLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                self._clear_layout(child_layout)

    def _rebuild_channel_panels(self) -> None:
        self._clear_layout(self.names_panel_layout)
        self._clear_layout(self.channel_panel_layout)

        self.names_panel = ChannelNamesPanel(self.state)
        self.names_panel.changed.connect(self._handle_names_changed)
        self.names_panel_layout.addWidget(self.names_panel)
        self.name_edit = self.names_panel.name_edit

        self.channel_panel = ChannelPanel(self.state)
        self.channel_panel.changed.connect(self._mark_dirty)
        self.channel_panel.clearRequested.connect(self.clear_channel)
        self.channel_panel_layout.addWidget(self.channel_panel)

    def _rebuild_periods(self) -> None:
        while self.drag_container.layout_main.count():
            item = self.drag_container.layout_main.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.drag_container.insert_indicator:
                widget.deleteLater()
        self.drag_container.items = []

        visible_indices = [self.state.channel_index(channel) for channel in self.state.visible_channels]
        labels = {channel: self.state.label_for(channel) for channel in self.state.visible_channels}
        compact = len(self.state.visible_channels) > 16
        total_periods = len(self.state.periods)
        for index, period in enumerate(self.state.periods):
            visible_period = PulsePeriod(
                period.duration,
                tuple(period.states[i] for i in visible_indices),
                unit=period.unit,
                name=period.name,
            )
            hidden_states = {
                channel: period.states[self.state.channel_index(channel)]
                for channel in self.state.channels
                if channel not in self.state.visible_channels
            }
            card = PeriodCard(
                index,
                visible_period,
                total_periods=total_periods,
                channels=self.state.visible_channels,
                labels=labels,
                hidden_states=hidden_states,
                compact=compact,
                time_step_ns=self.state.time_step_ns,
            )
            card.changed.connect(self._mark_dirty)
            self.drag_container.add_item(card, "pulse")
        if self.state.repeat_start is not None and self.state.repeat_end is not None and self.state.repeat_count > 1:
            start = RepeatBracket("start")
            end = RepeatBracket("end", self.state.repeat_count)
            start.changed.connect(self._mark_dirty)
            end.changed.connect(self._mark_dirty)
            self.drag_container.insert_item(self.state.repeat_start, start, "bracket_start")
            self.drag_container.insert_item(self.state.repeat_end + 2, end, "bracket_end")
            self.bracket_exists = True
            self.bracket_button.setText("Delete\nBracket")
        else:
            self.bracket_exists = False
            self.bracket_button.setText("Add\nBracket")
        self.drag_container.refresh_layout()
        self._sync_dataset_geometry()

    def _sync_dataset_geometry(self) -> None:
        if not all(hasattr(self, name) for name in ("names_panel", "channel_panel", "drag_container", "scroll")):
            return
        def vertical_margins(layout: QtWidgets.QLayout | None) -> int:
            if layout is None:
                return 0
            margins = layout.contentsMargins()
            return margins.top() + margins.bottom()

        self.names_panel.adjustSize()
        self.channel_panel.adjustSize()
        self.drag_container.adjustSize()
        card_hints = [item.widget.sizeHint().height() for item in self.drag_container.items]
        drag_height = (max(card_hints) if card_hints else 0) + vertical_margins(self.drag_container.layout_main)
        names_height = 0 if self.names_panel_holder.isHidden() else self.names_panel.sizeHint().height() + vertical_margins(self.names_panel_layout)
        channel_height = 0 if self.channel_panel_holder.isHidden() else self.channel_panel.sizeHint().height() + vertical_margins(self.channel_panel_layout)
        stub_height = self.left_panel_stub.sizeHint().height() if hasattr(self, "left_panel_stub") and not self.left_panel_stub.isHidden() else 0
        content_height = max(
            names_height,
            channel_height,
            stub_height,
            drag_height,
        )
        content_height += _px(2, minimum=1)
        for widget in (self.names_panel_holder, self.channel_panel_holder, self.left_panel_stub, self.scroll, self.drag_container):
            widget.setMinimumHeight(content_height)
        self.dataset_body.setMinimumHeight(content_height + vertical_margins(self.dataset_body.layout()))
        container_width = self._drag_container_width()
        self.drag_container.setFixedSize(container_width, content_height)
        self._sync_timeline_scrollbar()

    def _display_labels_from_name_panel(self) -> dict[str, str]:
        labels: dict[str, str] = {}
        if not hasattr(self, "names_panel"):
            return {channel: self.state.label_for(channel) for channel in self.state.visible_channels}
        for channel in self.state.visible_channels:
            edit = self.names_panel.label_edits.get(channel)
            text = edit.text().strip() if edit is not None else self.state.channel_labels.get(channel, "")
            labels[channel] = text if text and text != channel else channel
        return labels

    def _refresh_visible_display_labels(self) -> None:
        labels = self._display_labels_from_name_panel()
        if hasattr(self, "channel_panel"):
            self.channel_panel.set_channel_display_labels(labels)
        if hasattr(self, "drag_container"):
            for card in self.drag_container.pulse_cards():
                card.set_channel_display_labels(labels)

    def _handle_names_changed(self) -> None:
        if not self._building:
            self._refresh_visible_display_labels()
        self._mark_dirty()
        self._activate_layout_tree()
        QtCore.QTimer.singleShot(0, self._activate_layout_tree)
        QtCore.QTimer.singleShot(0, self._sync_timeline_scrollbar)

    def _sync_timeline_scrollbar(self, *_args) -> None:
        if not hasattr(self, "timeline_hbar"):
            return
        source = self.scroll.horizontalScrollBar()
        self.timeline_hbar.blockSignals(True)
        self.timeline_hbar.setRange(source.minimum(), source.maximum())
        self.timeline_hbar.setPageStep(source.pageStep())
        self.timeline_hbar.setSingleStep(max(1, source.singleStep()))
        self.timeline_hbar.setValue(source.value())
        self.timeline_hbar.blockSignals(False)
        self.timeline_hbar.setVisible(source.maximum() > source.minimum())
        if hasattr(self, "timeline_hbar_spacer"):
            left_width = 0
            for widget in (self.names_panel_holder, self.channel_panel_holder, getattr(self, "left_panel_stub", None)):
                if widget is None or widget.isHidden():
                    continue
                width = widget.width() or widget.sizeHint().width()
                left_width += width
            body_layout = self.dataset_body.layout()
            if body_layout is not None:
                margins = body_layout.contentsMargins()
                left_width += margins.left()
            self.timeline_hbar_spacer.setFixedWidth(max(0, left_width))

    def _drag_container_width(self) -> int:
        widths: list[int] = []
        for item in self.drag_container.items:
            widget = item.widget
            max_width = widget.maximumWidth()
            if 0 < max_width < QtWidgets.QWIDGETSIZE_MAX:
                width = max_width
            else:
                width = widget.sizeHint().width()
            widths.append(max(width, widget.minimumWidth(), widget.width()))
        if not widths:
            return 0
        spacing = max(0, self.drag_container.layout_main.spacing())
        margins = self.drag_container.layout_main.contentsMargins()
        return sum(widths) + spacing * (len(widths) - 1) + margins.left() + margins.right()

    def _activate_layout_tree(self) -> None:
        for widget in (self, self.dataset_body, self.drag_container):
            layout = widget.layout()
            if layout is not None:
                layout.activate()
            widget.updateGeometry()
            widget.update()
        for widget in (self.dataset_scroll, self.scroll, self.scroll.viewport()):
            widget.updateGeometry()
            widget.update()
        if hasattr(self, "timeline_hbar"):
            self._sync_timeline_scrollbar()
        window = self.window()
        if window is not self:
            layout = window.layout()
            if layout is not None:
                layout.activate()
            window.updateGeometry()
            window.update()

    def read_state(self) -> PulseTableState:
        x_ns = float(self.channel_panel.x_edit.text() or 0)
        time_step_ns = float(self.channel_panel.step_edit.text() or self.state.time_step_ns)
        state = PulseTableState(
            channels=self.state.channels,
            visible_channels=self.state.visible_channels,
            periods=[card.to_period(full_channels=self.state.channels, x_ns=x_ns, time_step_ns=time_step_ns) for card in self.drag_container.pulse_cards()],
            name=self.name_edit.text().strip() or self.state.name or _default_pulse_name(),
            x_ns=x_ns,
            time_step_ns=time_step_ns,
            channel_labels=dict(self.state.channel_labels),
            delays=dict(self.state.delays),
            delay_units=dict(self.state.delay_units),
            repeat_forever=bool(self.state.repeat_forever),
        )
        self.names_panel.read_values(state)
        self.channel_panel.read_values(state)
        start, end, repeat = self._read_bracket()
        state.repeat_start = start
        state.repeat_end = end
        state.repeat_count = repeat
        state.validate()
        self.state = state
        return state

    def _read_bracket(self):
        start = None
        end = None
        repeat = 1
        pulse_seen = 0
        for item in self.drag_container.items:
            if item.item_type == "pulse":
                pulse_seen += 1
            elif item.item_type == "bracket_start":
                start = pulse_seen
            elif item.item_type == "bracket_end":
                end = pulse_seen - 1
                spin = getattr(item.widget, "repeat_spin", None)
                repeat = int(spin.value()) if spin is not None else 2
        if start is None or end is None:
            return None, None, 1
        return start, end, repeat

    def add_period(self) -> None:
        state = self.read_state()
        state.periods.append(PulsePeriod(1_000, tuple(0 for _ in state.channels), unit="ns", name=""))
        state.validate()
        self.load_state(state)

    def remove_period(self) -> None:
        state = self.read_state()
        if len(state.periods) > 1:
            state.periods.pop()
        if state.repeat_start is not None and state.repeat_end is not None:
            state.repeat_end = min(state.repeat_end, len(state.periods) - 1)
            if state.repeat_end < state.repeat_start:
                state.repeat_start = state.repeat_end = None
                state.repeat_count = 1
        state.validate()
        self.load_state(state)

    def toggle_bracket(self) -> None:
        if self.bracket_exists:
            self.state = self.read_state()
            self.state.repeat_start = self.state.repeat_end = None
            self.state.repeat_count = 1
            self.load_state(self.state)
            return
        if len(self.drag_container.pulse_cards()) < 2:
            self._message("Repeat needs at least two periods.")
            return
        start = RepeatBracket("start")
        end = RepeatBracket("end", 2)
        start.changed.connect(self._mark_dirty)
        end.changed.connect(self._mark_dirty)
        self.drag_container.insert_item(0, start, "bracket_start")
        self.drag_container.insert_item(len(self.drag_container.items), end, "bracket_end")
        self.bracket_exists = True
        self.bracket_button.setText("Delete\nBracket")
        self._sync_dataset_geometry()
        self._mark_dirty()

    def hide_channel(self, channel: str) -> None:
        try:
            state = self.read_state()
            if self._channel_has_period_on(state, channel):
                self._message(f"Channel {channel!r} has an on period. Clear its pulses before hiding it.")
                return
            state.visible_channels = [item for item in state.visible_channels if item != channel]
            state.validate()
        except Exception as exc:
            self._message(str(exc))
            return
        self.load_state(state)

    def clear_channel(self, channel: str) -> None:
        try:
            state = self.read_state()
            state.clear_channel(channel)
        except Exception as exc:
            self._message(str(exc))
            return
        self.load_state(state)

    def hide_off_channels(self) -> None:
        state = self.read_state()
        off_channels = {channel for channel in state.channels if not self._channel_has_period_on(state, channel)}
        keepers = [channel for channel in state.visible_channels if channel not in off_channels]
        min_visible = min(4, len(state.channels))
        for channel in state.channels:
            if len(keepers) >= min_visible:
                break
            if channel not in keepers:
                keepers.append(channel)
        if not keepers:
            keepers = list(state.channels[:min_visible])
        state.visible_channels = keepers
        state.validate()
        self.load_state(state)

    @staticmethod
    def _channel_has_period_on(state: PulseTableState, channel: str) -> bool:
        index = state.channel_index(channel)
        return any(int(period.states[index]) for period in state.periods)

    def show_all_channels(self) -> None:
        state = self.read_state()
        state.visible_channels = list(state.channels)
        state.validate()
        self.load_state(state)

    def add_selected_channel(self) -> None:
        text = self.add_channel_combo.currentText()
        if not text:
            return
        channel = text.split("  ", 1)[0]
        state = self.read_state()
        state.show_channel(channel)
        self.load_state(state)

    def _refresh_hidden_combo(self) -> None:
        self.add_channel_combo.clear()
        visible = set(self.state.visible_channels)
        for channel in self.state.channels:
            if channel not in visible:
                label = self.state.label_for(channel)
                self.add_channel_combo.addItem(f"{channel}  ({label})" if label != channel else channel)
        has_hidden = self.add_channel_combo.count() > 0
        self.add_channel_combo.setEnabled(has_hidden)
        self.add_channel_button.setEnabled(has_hidden)

    def _prepare_to_device(self):
        state = self.read_state()
        clock_step_ns = self._clock_step_ns(self.sequencer)
        if self.sequencer is None:
            if clock_step_ns is not None:
                state.to_sequence(time_step_ns=clock_step_ns)
            else:
                state.to_sequence()
            return None
        return self.sequencer.prepare(state)

    def prepare(self) -> None:
        try:
            self.last_program = self._prepare_to_device()
            if self.last_program is None:
                self._message("No sequencer attached. Sequence validated only.")
            self.stateui_manager.runstate = PulseStateUIManager.RunState.PREPARED
        except Exception as exc:
            self.stateui_manager.runstate = PulseStateUIManager.RunState.ERROR
            self._message(str(exc))

    def fire(self) -> None:
        try:
            self.last_program = self._prepare_to_device()
            if self.sequencer is None:
                self._message("No sequencer attached. Sequence validated only.")
                self.stateui_manager.runstate = PulseStateUIManager.RunState.PREPARED
                return
            self.sequencer.fire()
            self.stateui_manager.runstate = PulseStateUIManager.RunState.RUNNING
        except Exception as exc:
            self.stateui_manager.runstate = PulseStateUIManager.RunState.ERROR
            self._message(str(exc))

    def safe_state(self) -> None:
        try:
            if self.sequencer is not None:
                if hasattr(self.sequencer, "set_safe_state"):
                    self.sequencer.set_safe_state()
                elif hasattr(self.sequencer, "abort"):
                    self.sequencer.abort()
            self.stateui_manager.runstate = PulseStateUIManager.RunState.SAFE
        except Exception as exc:
            self.stateui_manager.runstate = PulseStateUIManager.RunState.ERROR
            self._message(str(exc))

    def save_to_file(self) -> None:
        try:
            state = self.read_state()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save pulse",
                str(self._default_save_path(state)),
                "ZLC pulse (*.json)",
            )
            if path:
                path_obj = Path(path)
                if path_obj.suffix == "":
                    path_obj = path_obj.with_suffix(".json")
                state.save(path_obj)
                self.address_str = str(path_obj)
                self._last_save_state = state.to_dict()
                self._last_load_state = None
                self.stateui_manager.address_str = str(path_obj)
                self.stateui_manager.filestate = PulseStateUIManager.FileState.SAVE
                if hasattr(self, "preview_status"):
                    self.preview_status.setText(f"Saved pulse: {path_obj.name}")
        except Exception as exc:
            self._message(str(exc))

    def load_from_file(self) -> None:
        try:
            start = str(Path(self.address_str).parent if self.address_str else _pulse_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load pulse", start, "ZLC pulse (*.json)")
            if path:
                state = PulseTableState.load(path)
                self.address_str = path
                self._last_load_state = state.to_dict()
                self._last_save_state = None
                self.stateui_manager.address_str = path
                self.stateui_manager.filestate = PulseStateUIManager.FileState.LOAD
                self.load_state(state)
        except Exception as exc:
            self._message(str(exc))

    def _mark_dirty(self) -> None:
        if self._building:
            return
        self._preview_dirty = True
        self._summary_timer.start()
        if hasattr(self, "tabs") and self.tabs.currentWidget() is getattr(self, "preview_tab", None):
            self._preview_timer.start()

    def _update_summary(self) -> None:
        try:
            state = self.read_state() if hasattr(self, "channel_panel") and hasattr(self, "drag_container") else self.state
            sequence = state.to_sequence()
            hidden = state.hidden_active_channels()
            total_ns = state.total_duration_ns()
            parts = [
                f"{len(state.visible_channels)}/{len(state.channels)} visible",
                f"{len(state.periods)} periods",
                f"step {state.time_step_ns:g} ns",
                f"{total_ns:.3g} ns",
                f"{len(sequence.pulses)} pulses",
                "repeat ∞" if state.repeat_forever else "single",
            ]
            if hidden:
                parts.append(f"hidden active: {', '.join(hidden)}")
            boundary_active = state.repeat_forever_boundary_active_channels()
            if boundary_active:
                labels = [state.label_for(channel) for channel in boundary_active[:4]]
                suffix = "" if len(boundary_active) <= 4 else f", +{len(boundary_active) - 4}"
                parts.append(f"table restart high every {_summary_time_text(total_ns)}: {', '.join(labels)}{suffix}")
            self.summary.setText(" | ".join(parts))
            if hasattr(self, "names_panel"):
                self.names_panel.total_label.setText(f"{total_ns:.9g} ns")
                self.names_panel.periods_label.setText(f"{len(state.periods)}")
            if hasattr(self, "channel_panel"):
                self.channel_panel.total_label.setText(f"{len(state.visible_channels)}/{len(state.channels)}")
            if hasattr(self, "visible_label"):
                self.visible_label.setText(f"Visible: {len(state.visible_channels)}/{len(state.channels)}  Hidden: {len(state.channels) - len(state.visible_channels)}")
            self._update_file_state(state)
        except Exception as exc:
            self.summary.setText(str(exc))
            if hasattr(self, "stateui_manager"):
                self.stateui_manager.filestate = PulseStateUIManager.FileState.UNSAVED

    def _update_file_state(self, state: PulseTableState) -> None:
        if not hasattr(self, "stateui_manager"):
            return
        current = state.to_dict()
        self.stateui_manager.pulse_name = state.name
        self.stateui_manager.address_str = self.address_str
        if not self.address_str:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.UNTITLED
        elif self._last_save_state == current:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.SAVE
        elif self._last_load_state == current:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.LOAD
        else:
            self.stateui_manager.filestate = PulseStateUIManager.FileState.UNSAVED

    def _default_save_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}.json"

    def _default_figure_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}.png"

    def _create_preview_plot(self, state: PulseTableState, *, include_always_off: bool):
        sequence = state.to_sequence(expand_repeat=False)
        repeat = pulse_repeat_notation(state)
        repeat_brackets = pulse_repeat_markers(state)
        channels = pulse_plot_channels(
            sequence,
            channels=state.channels,
            include_always_off=include_always_off,
        )
        plotter = frontend_plot(
            sequence,
            kind="pulse",
            channels=channels,
            include_always_off=True,
            repeat_notation=repeat,
            repeat_brackets=repeat_brackets,
            channel_labels={channel: state.label_for(channel) for channel in channels},
            title=state.name,
            show_names=True,
            display=False,
            data_figure=False,
        )
        return plotter, channels, repeat

    def save_figure(self) -> None:
        try:
            state = self.read_state()
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save pulse figure",
                str(self._default_figure_path(state)),
                "Pulse figure (*.png)",
            )
            if not path:
                return
            image_path = Path(path)
            if image_path.suffix == "":
                image_path = image_path.with_suffix(".png")
            self._save_preview_image(state, image_path)
            self.preview_status.setText(f"Saved figure: {image_path.name}")
        except Exception as exc:
            self._message(str(exc))

    def _save_preview_image(self, state: PulseTableState, image_path: Path) -> Path:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        include_always_off = bool(getattr(self, "preview_include_off", None) and self.preview_include_off.isChecked())
        plotter, _channels, _repeat = self._create_preview_plot(state, include_always_off=include_always_off)
        plotter.fig.savefig(image_path, bbox_inches="tight")
        if plt is not None:
            plt.close(plotter.fig)
        return image_path

    def _on_tab_changed(self, _index: int) -> None:
        if self.tabs.currentWidget() is self.preview_tab:
            self.refresh_preview()

    def _request_preview_refresh(self, *_args) -> None:
        self._preview_dirty = True
        if hasattr(self, "tabs") and self.tabs.currentWidget() is getattr(self, "preview_tab", None):
            self._preview_timer.start()

    def refresh_preview(self) -> None:
        if FigureCanvas is None:
            self.preview_status.setText("Matplotlib Qt canvas is not available.")
            return
        try:
            state = self.read_state()
            include_always_off = self.preview_include_off.isChecked()
            plotter, channels, repeat = self._create_preview_plot(state, include_always_off=include_always_off)
            self._replace_preview_canvas(plotter)
            repeat_part = f" | {repeat}" if repeat else ""
            mode = "all channels" if include_always_off else "active channels"
            self.preview_status.setText(f"{len(channels)}/{len(state.channels)} plotted ({mode}){repeat_part}")
            self._preview_dirty = False
            self._update_file_state(state)
        except Exception as exc:
            self.preview_status.setText(str(exc))

    def _replace_preview_canvas(self, plotter) -> None:
        while self.preview_body_layout.count():
            item = self.preview_body_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        if self._preview_plot is not None and plt is not None:
            try:
                plt.close(self._preview_plot.fig)
            except Exception:
                pass
        canvas = FigureCanvas(plotter.fig)
        canvas.draw()
        hint = canvas.sizeHint()
        canvas.setFixedSize(hint)
        self.preview_body_layout.addWidget(canvas)
        margins = self.preview_body_layout.contentsMargins()
        self.preview_body.setFixedSize(
            hint.width() + margins.left() + margins.right(),
            hint.height() + margins.top() + margins.bottom(),
        )
        self._preview_plot = plotter
        self._preview_canvas = canvas

    def _message(self, text: str) -> None:
        QtWidgets.QMessageBox.warning(self, "Pulse", text)

    @staticmethod
    def _settle_qt_events(ms: int = 1000) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.processEvents()
        if int(ms) > 0:
            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(int(ms), loop.quit)
            loop.exec_()
        app.processEvents()

    def grab_screenshot(self, path: str | Path, *, settle_ms: int = 1000) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._settle_qt_events(settle_ms)
        self.grab().save(str(path))
        return path

    def to_sequence(self):
        """Return the current editor state as a ``PulseSequence``."""

        return self.read_state().to_sequence()

    @staticmethod
    def _clock_step_ns(sequencer) -> float | None:
        clock_hz = getattr(sequencer, "clock_hz", None)
        if clock_hz is None:
            return None
        clock_hz = float(clock_hz)
        if clock_hz <= 0:
            return None
        return 1e9 / clock_hz

    @staticmethod
    def _resolve_scale(scale: float | None, *, app: QtWidgets.QApplication) -> float:
        if scale is not None:
            return set_fluent_scale(scale)
        screen = app.primaryScreen()
        if screen is None:
            return set_fluent_scale(1.0)
        available = screen.availableGeometry()
        target_w = 1280
        target_h = 790
        margin_w = 48
        margin_h = 88
        auto = min(
            1.0,
            max(0.1, (available.width() - margin_w) / target_w),
            max(0.1, (available.height() - margin_h) / target_h),
        )
        return set_fluent_scale(auto)


def show_pulse_gui(
    *,
    state: PulseTableState | None = None,
    channels: Sequence[str] | None = None,
    sequencer=None,
    experiment=None,
    channel_labels: Mapping[str, str] | None = None,
    scale: float | None = None,
    window_ratio: float = DEFAULT_WINDOW_RATIO,
) -> PulseSequenceEditor:
    app = ensure_qt_app()
    editor = PulseSequenceEditor(
        state=state,
        channels=channels,
        sequencer=sequencer,
        experiment=experiment,
        channel_labels=channel_labels,
        scale=scale,
        window_ratio=window_ratio,
    )
    window = FluentWindow(widget=editor, title="PulseGUI@Zou lab", hide_on_close=False)
    editor._set_gui_title(editor.windowTitle())
    window.adjustSize()
    window.setFixedSize(window.size())
    _center_window_on_primary_screen(window, app)
    window.show()
    editor._zlc_window = window
    if not hasattr(app, "_zlc_pulse_windows"):
        app._zlc_pulse_windows = []
    app._zlc_pulse_windows.extend([window, editor])
    return editor


def _center_window_on_primary_screen(window: QtWidgets.QWidget, app: QtWidgets.QApplication) -> None:
    screen = app.primaryScreen()
    if screen is None:
        return
    available = screen.availableGeometry()
    frame = window.frameGeometry()
    frame.moveCenter(available.center())
    window.move(frame.topLeft())


__all__ = ["PulseSequenceEditor", "show_pulse_gui", "ensure_qt_app"]
