"""Confocal-style PyQt pulse editor for neutral-atom ``PulseSequence``.

The GUI is a front-end only.  It edits ``PulseTableState`` and calls an
optional existing sequencer/experiment; it does not introduce a separate
hardware-control layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import os
import re

from PyQt5 import QtCore, QtGui, QtWidgets

from Zou_lab_control.neutral_atom.timing.pulse_table import (
    PulsePeriod,
    PulseTableState,
    ScanSlot,
    default_pulse_name,
    load_scan_table,
    slot_var,
)
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
    ElidedLabel,
    FluentButton,
    FluentCheckBox,
    FluentComboBox,
    FluentDoubleSpinBox,
    FluentFormGrid,
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLabeledField,
    FluentLineEdit,
    FluentScanDot,
    FluentScanLineEdit,
    FluentScrollArea,
    FluentStatusDot,
    FluentSwitch,
    FluentTabWidget,
    FluentWindow,
    Metrics,
    ensure_qt_app,
    fluent_font_size,
    fluent_scrollbar_stylesheet,
    fluent_text_width,
    fluent_widget_stylesheet,
    format_compact_number,
    mark_scan_field,
    measure_text_width,
    scaled_px,
    set_fluent_scale,
    align_to_resolution,
    batched_updates,
    signals_blocked as _signals_blocked,
)

_SLOT_RE = re.compile(r"^s(\d+)$")

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
PANEL_TOP_HEIGHT = 152
CHANNEL_ROW_SPACING = 4
PERIOD_CARD_WIDTH = 146
DEFAULT_WINDOW_RATIO = 0.90
DEFAULT_HARDWARE_CLOCK_HZ = 50_000_000.0
DEFAULT_TIME_STEP_NS = 1_000_000_000.0 / DEFAULT_HARDWARE_CLOCK_HZ
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
    return _px(PANEL_TOP_HEIGHT, minimum=138)


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


def _is_slot_expr(text: object) -> bool:
    """True when a field value is a bare scan-slot reference like ``s0``."""

    return bool(_SLOT_RE.fullmatch(str(text or "").strip()))


def _slot_index_of_expr(text: object) -> int | None:
    match = _SLOT_RE.fullmatch(str(text or "").strip())
    return int(match.group(1)) if match else None


def _scan_slot_label(state: PulseTableState, index: int) -> str:
    """Human description of scan slot ``index`` for GUI lists/tooltips."""

    slot = state.scan_slots[index]
    if slot.kind == "duration":
        try:
            return f"Period {int(slot.target) + 1} duration"
        except ValueError:
            return f"Period {slot.target} duration"
    if slot.kind == "delay":
        return f"{state.label_for(slot.target)} delay"
    if slot.kind == "dac":
        return f"{slot.dac_bus} (period {slot.dac_period + 1})"
    return slot.target


def _default_scan_code(n_slots: int) -> str:
    n_slots = max(1, int(n_slots))
    if n_slots == 1:
        build = "scan_table = points.reshape(-1, 1)"
    else:
        columns = ", ".join(["points"] + ["np.zeros_like(points)"] * (n_slots - 1))
        build = f"scan_table = np.column_stack([{columns}])"
    return (
        "import numpy as np\n\n"
        f"# {n_slots} bound slot(s). Build an (N_points x {n_slots}) array.\n"
        "# Row = one scan point; column j = slot sj, in the slot's display unit.\n"
        "points = np.linspace(1000, 10000, 11)\n"
        f"{build}\n"
    )


def _normalize_bus_value_text(text: str, *, max_value: int) -> str:
    try:
        value = int(round(float(str(text or "0").strip())))
    except Exception as exc:
        raise ValueError(f"analog bus value must be an integer 0..{int(max_value)}.") from exc
    value = max(0, min(int(max_value), value))
    return str(value)


def _unit_resolution(step_ns: float, unit: str) -> float:
    factor = UNIT_TO_NS.get(unit or "ns", 1.0)
    if factor <= 0:
        return float(step_ns)
    return float(step_ns) / factor


def _bus_display_label(name: str) -> str:
    return str(name).replace("_", " ")


def _bus_key(name: str) -> str:
    return f"bus:{name}"


def _bus_mode_title(mode: str) -> str:
    mode = str(mode or "hold").strip().lower()
    return {"edge": "Edge", "ramp": "Ramp", "hold": "Hold"}.get(mode, "Hold")


def _bus_mode_value(title: str) -> str:
    title = str(title or "Hold").strip().lower()
    if title.startswith("ram"):
        return "ramp"
    if title.startswith("edg"):
        return "edge"
    return "hold"


def _is_bus_key(key: str) -> bool:
    return str(key).startswith("bus:")


def _display_rows(state: PulseTableState) -> list[dict[str, object]]:
    buses = state.bus_channels()
    member_to_bus = {channel: bus for bus, members in buses.items() for channel in members}
    rows: list[dict[str, object]] = []
    emitted: set[str] = set()
    visible = set(state.visible_channels)
    for channel in state.visible_channels:
        bus = member_to_bus.get(channel)
        if bus is not None:
            if bus in emitted:
                continue
            members = buses[bus]
            if any(member in visible for member in members):
                rows.append({"kind": "bus", "key": _bus_key(bus), "name": bus, "channels": members, "label": bus})
                emitted.add(bus)
            continue
        rows.append({"kind": "channel", "key": channel, "name": channel, "channels": [channel], "label": state.label_for(channel)})
    return rows


def _display_row_label(row: Mapping[str, object], labels: Mapping[str, str] | None = None) -> str:
    if row.get("kind") == "bus":
        return str((labels or {}).get(str(row["key"])) or row.get("label") or row.get("name"))
    key = str(row["key"])
    return str((labels or {}).get(key) or row.get("label") or key)


def _bus_value_from_states(state: PulseTableState, period: PulsePeriod, bus_name: str) -> int:
    value = 0
    for bit, channel in enumerate(state.bus_channels()[bus_name]):
        if int(period.states[state.channel_index(channel)]):
            value |= 1 << bit
    return value


def _analog_bus_value_at_tick(plan: Sequence[Mapping[str, object]], starts: Sequence[int], tick: int) -> int:
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        value = entry.get("value")
        if mode in {"edge", "ramp"} and value is not None:
            anchors.append((index, int(starts[index]), mode, int(value)))
    if not anchors:
        return 0
    tick = int(tick)
    if tick < anchors[0][1]:
        return 0
    previous = anchors[0]
    for anchor in anchors[1:]:
        if tick < anchor[1]:
            if anchor[2] == "ramp" and anchor[1] > previous[1]:
                fraction = (tick - previous[1]) / (anchor[1] - previous[1])
                return int(round(previous[3] + (anchor[3] - previous[3]) * fraction))
            return int(previous[3])
        previous = anchor
    return int(previous[3])


def _analog_bus_ticks(plan: Sequence[Mapping[str, object]], starts: Sequence[int]) -> list[int]:
    ticks = {int(starts[index]) for index in range(max(0, len(starts) - 1))}
    anchors: list[tuple[int, int, str, int]] = []
    for index, entry in enumerate(plan):
        mode = str(entry.get("mode", "hold")).lower()
        value = entry.get("value")
        if mode in {"edge", "ramp"} and value is not None:
            anchors.append((index, int(starts[index]), mode, int(value)))
    previous = anchors[0] if anchors else None
    for anchor in anchors[1:]:
        ticks.add(anchor[1])
        if previous is not None and anchor[2] == "ramp":
            span = anchor[1] - previous[1]
            steps = abs(anchor[3] - previous[3])
            if span > 0 and steps > 0:
                last_tick = previous[1]
                for step in range(1, steps + 1):
                    tick = int(round(previous[1] + span * (step / steps)))
                    tick = max(previous[1], min(anchor[1], tick))
                    if tick <= last_tick and last_tick < anchor[1]:
                        tick = last_tick + 1
                    if tick <= anchor[1]:
                        ticks.add(tick)
                        last_tick = tick
        previous = anchor
    ticks.add(int(starts[-1]))
    return sorted(ticks)


def _analog_bus_traces(state: PulseTableState) -> tuple[list[dict[str, object]], set[str]]:
    buses = state.bus_channels()
    if not buses:
        return [], set()
    starts_steps = [0]
    slots = state.reference_slots()
    for period in state.periods:
        starts_steps.append(starts_steps[-1] + period.duration_steps(slots=slots, time_step_ns=state.time_step_ns))
    visible = set(state.visible_channels)
    traces: list[dict[str, object]] = []
    folded_members: set[str] = set()
    for bus_name, members in buses.items():
        # A recognized DAC bus is ALWAYS shown as one folded analog row -- its bit
        # channels must never leak into the digital plot as individual rows
        # (that was the bug where DA appeared as 10 separate channels).  So fold
        # every member here; only the *tracing* (drawing the row) is gated on the
        # bus being visible or carrying a non-zero / scanned value.
        folded_members.update(members)
        active = bus_name in state.analog_bus_modes or any(
            state.bus_value(index, bus_name) != 0 for index in range(len(state.periods))
        )
        if not any(member in visible for member in members) and not active:
            continue
        # Resolve scanned (slot-referenced) DAC values to their reference value so
        # the preview shows a concrete trace instead of crashing on int("s2").
        plan = state._resolved_bus_plan(bus_name, slots)
        bus_ticks = _analog_bus_ticks(plan, starts_steps)
        traces.append(
            {
                "name": bus_name,
                "label": _bus_display_label(bus_name),
                "members": list(members),
                "max": (1 << len(members)) - 1,
                "starts": [tick * state.time_step_ns * 1e-9 for tick in bus_ticks],
                "values": [
                    _analog_bus_value_at_tick(plan, starts_steps, tick)
                    for tick in bus_ticks[:-1]
                ],
            }
        )
    return traces, folded_members


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


def _set_form_label_geometry(label: FluentLabel) -> FluentLabel:
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setFixedSize(_channel_label_width(), _row_height())
    return label


def _form_control_cell(widget: QtWidgets.QWidget) -> QtWidgets.QWidget:
    widget.setFixedHeight(_row_height())
    cell = QtWidgets.QWidget()
    cell.setStyleSheet("background: transparent;")
    layout = QtWidgets.QHBoxLayout(cell)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    fixed_width = widget.minimumWidth() > 0 and widget.maximumWidth() == widget.minimumWidth()
    if fixed_width:
        layout.addWidget(widget, 0, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        layout.addStretch(1)
    else:
        layout.addWidget(widget, 1)
    return cell


def _channel_row_height(channel_count: int) -> int:
    return _px(26 if channel_count > 16 else ROW_HEIGHT, minimum=22)


def _bar_title(text: str) -> FluentLabel:
    """Small bold section header used inside the compact bottom control bar."""

    label = FluentLabel(text)
    label.setStyleSheet(
        f'QLabel {{ color: {GREY}; font: 600 {max(8, fluent_font_size() - 2)}pt "{FONT}"; background: transparent; }}'
    )
    return label


def _elide_text(text: object, width: int) -> str:
    """Right-elide ``text`` to fit ``width`` px at the current GUI font."""

    return _font_metrics().elidedText(str(text), QtCore.Qt.ElideRight, max(8, int(width)))


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
        # Keep the button label fixed-width ("Save", never "Save*"): the dirty
        # state is already shown by the button COLOUR (yellow) and the title-bar
        # star, and a changing label made the button visibly narrow after load.
        self.save_button.setText("Save")
        self.save_button.set_color(YELLOW if star else ACCENT)
        if self.title_callback is not None:
            self.title_callback(f"{pulse_name} - PulseGUI{star}")

    def set_pulse_name(self, name: str) -> None:
        self.pulse_name = str(name or "pulse")
        self._update()


class PeriodCard(FluentGroupBox):
    changed = QtCore.pyqtSignal()
    busScanRequested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        index: int,
        period: PulsePeriod,
        *,
        total_periods: int = 1,
        channels: Sequence[str],
        labels: dict[str, str],
        hidden_states: dict[str, int] | None = None,
        rows: Sequence[Mapping[str, object]] | None = None,
        state: PulseTableState | None = None,
        compact: bool = False,
        time_step_ns: float = 1.0,
        parent=None,
    ):
        super().__init__("", parent)
        self.channels = list(channels)
        self.rows = list(rows or [{"kind": "channel", "key": channel, "name": channel, "channels": [channel], "label": labels.get(channel) or channel} for channel in channels])
        self.state_ref = state
        self.checks: dict[str, FluentCheckBox] = {}
        self.bus_mode_combos: dict[str, FluentComboBox] = {}
        self.bus_value_edits: dict[str, FluentLineEdit] = {}
        self.bus_dots: dict[str, FluentScanDot] = {}
        self.bus_max_values: dict[str, int] = {}
        self.bus_members: dict[str, list[str]] = {}
        self.hidden_states = {str(k): int(v) for k, v in dict(hidden_states or {}).items()}
        self.compact = bool(compact)
        self.time_step_ns = float(time_step_ns)
        self.set_period_position(index, total_periods)

        width = _period_card_width()
        control_width = _period_control_width(width)
        self.check_full_labels: dict[str, str] = {}
        self._checkbox_text_width = control_width - _px(24)
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

        scanned = _is_slot_expr(period.duration)
        self.duration_edit = FluentScanLineEdit(_period_duration_text(period), tooltip="Duration value; click the dot to scan it")
        self.duration_edit.setFixedWidth(control_width)
        self.duration_dot = self.duration_edit.dot
        top_layout.addWidget(_set_fixed_height(self.duration_edit))

        self.unit_combo = FluentComboBox()
        self.unit_combo.addItems(TIME_UNITS)
        self.unit_combo.setCurrentText("str (ns)" if scanned else period.unit)
        self.unit_combo.setToolTip("Duration unit")
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(_set_fixed_height(self.unit_combo))
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addSpacing(_px(1, minimum=0))

        if scanned:
            self.duration_edit.set_scan_bound(True, _slot_index_of_expr(period.duration) + 1)
            self.unit_combo.setEnabled(False)

        row_height = _channel_row_height(len(self.rows))
        full_state = self.state_ref
        for offset, row in enumerate(self.rows):
            if row.get("kind") == "bus" and full_state is not None:
                bus_name = str(row["name"])
                members = [str(channel) for channel in row.get("channels", [])]
                plan = full_state.analog_bus_plan(bus_name)
                entry = dict(plan[index]) if index < len(plan) else {"mode": "hold", "value": None}
                mode = str(entry.get("mode", "hold")).lower()
                raw_value = entry.get("value")
                bound = _is_slot_expr(raw_value)
                max_value = (1 << max(1, len(members))) - 1
                if bound:
                    value_display = str(raw_value)
                elif raw_value is None:
                    value_display = str(_bus_value_from_states(full_state, period, bus_name))
                else:
                    value_display = str(max(0, min(max_value, int(raw_value))))
                # The DAC row uses the SAME height as every other channel row so
                # the period card stays aligned, row-for-row, with the Names and
                # Delay panels (which render the bus row at row_height too).  The
                # combo / value field are given setFixedHeight(row_height) below,
                # which overrides their 30 px minimumHeight; their content only
                # needs ~24 px, so nothing clips even at the compressed 26 px.
                bus_row_height = row_height
                row_widget = QtWidgets.QWidget()
                row_widget.setStyleSheet("background: transparent;")
                row_widget.setFixedHeight(bus_row_height)
                row_layout = QtWidgets.QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(_px(3, minimum=2))
                combo = FluentComboBox()
                combo.addItems(["Edge", "Ramp", "Hold"])
                combo.setCurrentText(_bus_mode_title(mode))
                # "Edge"/"Ramp"/"Hold" + dropdown arrow.  Wide enough for "Ramp"
                # (the widest, due to the 'm') even under a wide substitute font,
                # while still leaving the value field room for the widest code (1023).
                combo.setFixedSize(_px(66, minimum=60), bus_row_height)
                combo.setToolTip(f"{bus_name}: output mode")
                value_edit = FluentScanLineEdit(value_display, tooltip=f"{bus_name}: integer 0..{max_value}; click the dot to scan it")
                value_edit.setFixedHeight(bus_row_height)
                # Room for the widest code (e.g. 1023) plus the embedded scan dot.
                value_edit.setMinimumWidth(_px(62, minimum=54))
                # NB: use set_editable (read-only + muted) NOT setEnabled --
                # disabling the field would also disable the embedded scan dot,
                # so you could never click it again to unbind the slot.
                value_edit.set_editable(mode != "hold")
                value_edit.editingFinished.connect(lambda edit=value_edit, limit=max_value: self._normalize_bus_value_edit(edit, limit))
                value_edit.textChanged.connect(self.changed)
                value_edit.scanClicked.connect(lambda b=bus_name: self.busScanRequested.emit(b))
                combo.currentTextChanged.connect(lambda text, edit=value_edit: edit.set_editable(_bus_mode_value(text) != "hold"))
                combo.currentTextChanged.connect(self.changed)
                self.bus_mode_combos[bus_name] = combo
                self.bus_value_edits[bus_name] = value_edit
                self.bus_max_values[bus_name] = max_value
                self.bus_members[bus_name] = members
                self.bus_dots[bus_name] = value_edit.dot
                row_layout.addWidget(combo)
                row_layout.addWidget(value_edit, 1)
                layout.addWidget(row_widget)
                if bound:
                    slot_index = full_state.slot_index_for("dac", f"{bus_name}@{index}")
                    value_edit.set_scan_bound(True, None if slot_index is None else slot_index + 1)
                    combo.setEnabled(False)
                continue
            channel = str(row["key"])
            source_index = self.channels.index(channel) if channel in self.channels else offset
            full_label = str(labels.get(channel) or channel)
            checkbox = FluentCheckBox(_elide_text(full_label, self._checkbox_text_width))
            checkbox.setChecked(bool(period.states[source_index]))
            checkbox.setToolTip(f"{channel} / {full_label}" if full_label != channel else channel)
            checkbox.setFixedHeight(row_height)
            checkbox.toggled.connect(self.changed)
            self.checks[channel] = checkbox
            self.check_full_labels[channel] = full_label
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
        if _is_slot_expr(text):
            was_blocked = self.unit_combo.blockSignals(True)
            self.unit_combo.setCurrentText("str (ns)")
            self.unit_combo.blockSignals(was_blocked)
            self.unit_combo.setEnabled(False)
        else:
            self.unit_combo.setEnabled(True)
        self._handle_unit(self.unit_combo.currentText())

    def _handle_unit(self, unit: str) -> None:
        self.duration_edit.set_resolution(_unit_resolution(self.time_step_ns, unit))

    def to_period(self, *, full_channels: Sequence[str], time_step_ns: float, slots: Mapping[str, float] | None = None) -> PulsePeriod:
        states = []
        for channel in full_channels:
            if channel in self.checks:
                states.append(1 if self.checks[channel].isChecked() else 0)
            else:
                states.append(1 if self.hidden_states.get(channel, 0) else 0)
        channel_index = {channel: index for index, channel in enumerate(full_channels)}
        for bus_name in self.bus_value_edits:
            members = self.bus_members.get(bus_name, [])
            mode_combo = self.bus_mode_combos.get(bus_name)
            mode = _bus_mode_value(mode_combo.currentText()) if mode_combo is not None else "edge"
            if mode == "hold":
                continue
            value_edit = self.bus_value_edits[bus_name]
            if _is_slot_expr(value_edit.text()):
                continue  # scanned DAC value; underlying bits stay as previewed
            value_text = _normalize_bus_value_text(value_edit.text(), max_value=self.bus_max_values.get(bus_name, 0))
            if value_edit.text() != value_text:
                value_edit.setText(value_text)
            value = int(value_text)
            for bit, channel in enumerate(members):
                if channel in channel_index:
                    states[channel_index[channel]] = 1 if (value >> bit) & 1 else 0
        period = PulsePeriod(
            self.duration_edit.text().strip(),
            tuple(states),
            unit=self.unit_combo.currentText(),
            name="",
        )
        period.duration_ns(slots=slots, time_step_ns=time_step_ns)
        return period

    def set_channel_display_labels(self, labels: Mapping[str, str]) -> None:
        for channel, checkbox in self.checks.items():
            full = str(labels.get(channel) or channel)
            self.check_full_labels[channel] = full
            checkbox.setText(_elide_text(full, self._checkbox_text_width))
            checkbox.setToolTip(f"{channel} / {full}" if full != channel else channel)
        for bus_name, edit in self.bus_value_edits.items():
            edit.setToolTip(f"{labels.get(_bus_key(bus_name), bus_name)}: integer value, 0..{self.bus_max_values.get(bus_name, 0)}")

    def bus_modes(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for bus_name, combo in self.bus_mode_combos.items():
            mode = _bus_mode_value(combo.currentText())
            value = None
            if mode != "hold":
                edit = self.bus_value_edits[bus_name]
                if _is_slot_expr(edit.text()):
                    value = edit.text().strip()  # scanned DAC value -> keep slot reference
                else:
                    value_text = _normalize_bus_value_text(edit.text(), max_value=self.bus_max_values.get(bus_name, 0))
                    if edit.text() != value_text:
                        edit.setText(value_text)
                    value = int(value_text)
            out[bus_name] = {"mode": mode, "value": value}
        return out

    def _normalize_bus_value_edit(self, edit: FluentLineEdit, max_value: int) -> None:
        try:
            edit.setText(_normalize_bus_value_text(edit.text(), max_value=max_value))
        except Exception:
            edit.setText("0")


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
        super().__init__("", parent)
        self.kind = kind
        self.setFixedWidth(_px(78, minimum=60))
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(7), _px(7), _px(7), _px(7))
        layout.setSpacing(_px(6, minimum=4))
        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))
        label = FluentLabel("Repeat")
        label.setAlignment(QtCore.Qt.AlignCenter)
        top_layout.addWidget(_set_fixed_height(label))
        self.repeat_spin = None
        if kind == "end":
            self.repeat_spin = FluentDoubleSpinBox(length=5, allow_minus=False)
            self.repeat_spin.setRange(1, 999)
            self.repeat_spin.setValue(repeat_count)
            self.repeat_spin.setFixedHeight(_row_height())
            self.repeat_spin.valueChanged.connect(self.changed)
            top_layout.addWidget(self.repeat_spin)
        else:
            spacer = QtWidgets.QWidget()
            spacer.setStyleSheet("background: transparent;")
            spacer.setFixedHeight(_row_height())
            top_layout.addWidget(spacer)
        unit_spacer = QtWidgets.QWidget()
        unit_spacer.setStyleSheet("background: transparent;")
        unit_spacer.setFixedHeight(_row_height())
        top_layout.addWidget(unit_spacer)
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addStretch()


class ChannelNamesPanel(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, state: PulseTableState, *, raw_labels: Mapping[str, str] | None = None, parent=None):
        super().__init__("Channel Names", parent)
        self.state = state
        self.raw_labels = {str(channel): str(label) for channel, label in dict(raw_labels or {}).items()}
        self.label_edits: dict[str, FluentLineEdit] = {}
        self.raw_label_widgets: dict[str, FluentLabel] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        self.rows = _display_rows(state)
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
        self.top_labels["name"] = self._add_labeled_widget(top_layout, "Name:", self.name_edit)

        self.total_label = FluentLineEdit("")
        self.total_label.setEnabled(False)
        self.top_labels["total"] = self._add_labeled_widget(top_layout, "Total:", self.total_label)
        self.periods_label = FluentLineEdit("")
        self.periods_label.setEnabled(False)
        self.top_labels["periods"] = self._add_labeled_widget(top_layout, "Periods:", self.periods_label)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.top_labels["visible"] = self._add_labeled_widget(top_layout, "Visible:", self.visible_label)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(self.rows))
        for row_info in self.rows:
            key = str(row_info["key"])
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(_px(5, minimum=3))
            hardware_text = str(row_info["name"]) if row_info.get("kind") == "bus" else self.raw_labels.get(key, key)
            hardware = FluentLabel(hardware_text)
            if row_info.get("kind") == "bus":
                hardware.setToolTip(", ".join(str(item) for item in row_info.get("channels", [key])))
            else:
                semantic = state.channel_labels.get(key, key)
                hardware.setToolTip(f"{key} / {semantic}")
            hardware.setAlignment(QtCore.Qt.AlignCenter)
            hardware.setFixedSize(label_w, row_height)
            self.raw_label_widgets[key] = hardware
            if row_info.get("kind") == "bus":
                edit = FluentLineEdit(_bus_display_label(str(row_info["name"])))
                edit.setEnabled(False)
                edit.setToolTip("Analog bus inferred from XDC/JSON labels or analog_buses config.")
            else:
                edit = FluentLineEdit(state.channel_labels.get(key, ""))
                edit.setPlaceholderText("display name")
                edit.setToolTip(f"Display name for {key}")
            edit.setFixedSize(edit_w, row_height)
            edit.textChanged.connect(self.changed)
            self.label_edits[key] = edit
            row.addWidget(hardware)
            row.addWidget(edit, 1)
            layout.addLayout(row)
        layout.addStretch()

    def _add_labeled_widget(self, layout: QtWidgets.QVBoxLayout, label_text: str, widget: QtWidgets.QWidget) -> FluentLabel:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_px(5, minimum=3))
        label = FluentLabel(label_text)
        _set_form_label_geometry(label)
        row.addWidget(label)
        row.addWidget(_form_control_cell(widget), 1)
        layout.addLayout(row)
        return label

    def read_values(self, state: PulseTableState) -> None:
        for row_info in self.rows:
            if row_info.get("kind") == "bus":
                continue
            channel = str(row_info["key"])
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
    delayScanRequested = QtCore.pyqtSignal(str)
    loadScanRequested = QtCore.pyqtSignal()
    editScanRequested = QtCore.pyqtSignal()

    def __init__(self, state: PulseTableState, parent=None):
        super().__init__("Delay / Scan", parent)
        self.state = state
        self.delay_edits: dict[str, FluentLineEdit] = {}
        self.delay_units: dict[str, FluentComboBox] = {}
        self.delay_dots: dict[str, FluentScanDot] = {}
        self.channel_labels: dict[str, ElidedLabel] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        self.rows = _display_rows(state)
        label_w = _channel_label_width()
        delay_w = _px(70, minimum=60)
        unit_w = _time_unit_width()
        hide_w = _hide_button_width()
        gap = _px(4, minimum=3)
        content_w = label_w + delay_w + unit_w + hide_w + gap * 3 + _px(16)
        self.setMinimumWidth(content_w)
        self.setMaximumWidth(content_w)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(_px(8), _px(8), _px(8), _px(8))
        layout.setSpacing(_row_spacing())

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.step_edit = FluentLineEdit(format_compact_number(state.time_step_ns))
        self.step_edit.set_resolution(1e-12)
        self.step_edit.textChanged.connect(self._handle_step_text)
        self.step_edit.textChanged.connect(self.changed)
        self.top_labels["step"] = self._add_labeled_widget(top_layout, "Step:", self.step_edit)

        self.scan_summary = FluentLineEdit("")
        self.scan_summary.setEnabled(False)
        self.scan_summary.setToolTip("Active scan slots and uploaded scan points. Click a dot to bind a field.")
        self.top_labels["scan"] = self._add_labeled_widget(top_layout, "Scan:", self.scan_summary)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(_px(5, minimum=3))
        self.load_button = FluentButton("Load Array", color=ACCENT)
        self.load_button.setFixedHeight(_row_height())
        self.load_button.setToolTip("Load a scan-table file (.npy/.csv/.txt): one row per scan point, one column per slot.")
        self.load_button.clicked.connect(self.loadScanRequested)
        self.edit_button = FluentButton("Scan Tab", color=GREY)
        self.edit_button.setFixedHeight(_row_height())
        self.edit_button.setToolTip("Open the Scan tab to write code that builds the scan table.")
        self.edit_button.clicked.connect(self.editScanRequested)
        # Expanding policy so the two buttons fill the row in EQUAL halves and line
        # up (FluentButton defaults to a fixed width = its own text, which made
        # "Load Array" and "Scan Tab" different widths and look misaligned).
        for _scan_btn in (self.load_button, self.edit_button):
            _scan_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(self.load_button, 1)
        btn_row.addWidget(self.edit_button, 1)
        top_layout.addLayout(btn_row)
        top_layout.addStretch()
        layout.addWidget(top)

        row_height = _channel_row_height(len(self.rows))
        for row_info in self.rows:
            key = str(row_info["key"])
            members = [str(channel) for channel in row_info.get("channels", [key])]
            is_bus = row_info.get("kind") == "bus"
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(gap)
            label = ElidedLabel(_display_row_label(row_info))
            label.setToolTip(", ".join(members))
            label.setFixedSize(label_w, row_height)
            self.channel_labels[key] = label

            if is_bus:
                member_values = [state.delays.get(channel, 0) for channel in members]
                member_units = [state.delay_units.get(channel, "ns") for channel in members]
                delay_value = member_values[0] if member_values and all(value == member_values[0] for value in member_values) else 0
                delay_unit = member_units[0] if member_units and all(unit == member_units[0] for unit in member_units) else "ns"
            else:
                delay_value = state.delays.get(key, 0)
                delay_unit = state.delay_units.get(key, "ns")
            bound = _is_slot_expr(delay_value)
            delay_edit = FluentScanLineEdit(str(delay_value), tooltip="Delay value; click the dot to scan it")
            delay_edit.setFixedSize(delay_w, row_height)
            delay_edit.textChanged.connect(lambda text, ch=key: self._handle_delay_text(ch, text))
            delay_edit.textChanged.connect(self.changed)
            if is_bus:
                delay_edit.dot.setEnabled(False)
                delay_edit.dot.setToolTip("Bind individual channel delays, not a bus, to a scan slot.")
            else:
                delay_edit.scanClicked.connect(lambda ch=key: self.delayScanRequested.emit(ch))
            unit = FluentComboBox()
            unit.addItems(TIME_UNITS)
            unit.setCurrentText("str (ns)" if bound else delay_unit)
            unit.setFixedSize(unit_w, row_height)
            unit.currentTextChanged.connect(lambda unit_text, ch=key: self._handle_delay_unit(ch, unit_text))
            unit.currentTextChanged.connect(self.changed)
            clear_btn = FluentButton("X", color=ORANGE)
            clear_btn.setFixedSize(hide_w, row_height)
            clear_btn.setToolTip("Set this row fully off.")
            clear_btn.clicked.connect(lambda _=False, ch=key: self.clearRequested.emit(ch))

            self.delay_edits[key] = delay_edit
            self.delay_units[key] = unit
            self.delay_dots[key] = delay_edit.dot
            row.addWidget(label)
            row.addWidget(delay_edit, 1)
            row.addWidget(unit)
            row.addWidget(clear_btn)
            layout.addLayout(row)
            if bound and not is_bus:
                slot_index = state.slot_index_for("delay", key)
                delay_edit.set_scan_bound(True, None if slot_index is None else slot_index + 1)
                unit.setEnabled(False)
            else:
                self._handle_delay_text(key, delay_edit.text())
                self._handle_delay_unit(key, unit.currentText())
        layout.addStretch()
        self.set_scan_summary()

    def set_scan_summary(self) -> None:
        n_slots = len(self.state.scan_slots)
        n_points = len(self.state.scan_table)
        if n_slots == 0:
            text = "no scan slots"
        else:
            text = f"{n_slots} slot{'s' if n_slots != 1 else ''} · {n_points} pt{'s' if n_points != 1 else ''}"
        self.scan_summary.setText(text)

    def _handle_step_text(self, _text: str) -> None:
        for channel, combo in self.delay_units.items():
            self._handle_delay_unit(channel, combo.currentText())

    def _add_labeled_widget(self, layout: QtWidgets.QVBoxLayout, label_text: str, widget: QtWidgets.QWidget) -> FluentLabel:
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_px(5, minimum=3))
        label = FluentLabel(label_text)
        _set_form_label_geometry(label)
        row.addWidget(label)
        row.addWidget(_form_control_cell(widget), 1)
        layout.addLayout(row)
        return label

    def _handle_delay_text(self, channel: str, text: str) -> None:
        combo = self.delay_units.get(channel)
        edit = self.delay_edits.get(channel)
        if combo is None or edit is None:
            return
        if _is_slot_expr(text):
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
        for row_info in self.rows:
            key = str(row_info["key"])
            if key not in self.delay_edits:
                continue
            raw = self.delay_edits[key].text().strip() or 0
            unit = self.delay_units[key].currentText()
            for channel in row_info.get("channels", [key]):
                channel = str(channel)
                state.delays[channel] = raw
                state.delay_units[channel] = unit

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
        channel_pins: Mapping[str, str] | None = None,
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
            channels = list(channels or [f"ch{index:02d}" for index in range(62)])
            labels = {str(k): str(v) for k, v in dict(channel_labels or {}).items()}
            state = PulseTableState(
                channels=channels,
                visible_channels=channels[: min(4, len(channels))],
                time_step_ns=self._clock_step_ns(sequencer) or DEFAULT_TIME_STEP_NS,
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
        self.channel_pins = {str(channel): str(pin) for channel, pin in dict(channel_pins or {}).items()}
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

        # --- Bottom control bar: two titled Fluent cards (Control / Channels),
        # using the same group-box-with-title style as the other panels for
        # visual consistency.  Kept compact (single-line buttons, tight 2x4 grid,
        # small margins) so the name/delay/period area keeps its vertical room. ---
        self.button_frame = FluentFrame(shadow=False)
        self.button_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        bar = QtWidgets.QHBoxLayout(self.button_frame)
        # Inset the Control / Channels group boxes by the shadow pad on every side
        # so their drop shadows have room to render inside the frame instead of
        # being clipped flush against its left / right / bottom edges.
        _sp = _shadow_pad()
        bar.setContentsMargins(_sp, _sp + _px(2), _sp, _sp)
        bar.setSpacing(_px(10, minimum=8))
        cb_h = _px(30, minimum=26)

        control_area = FluentGroupBox("Control")
        control_col = QtWidgets.QVBoxLayout(control_area)
        control_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        control_col.setSpacing(_px(4, minimum=3))
        button_layout = QtWidgets.QGridLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(_px(6, minimum=4))
        self.safe_button = self._control_button("Stop Pulse", self.safe_state, RED)
        self.fire_button = self._control_button("On Pulse", self.fire, GREEN)
        self.remove_button = self._control_button("Remove", self.remove_period, ORANGE)
        self.add_button = self._control_button("Add Period", self.add_period, ACCENT)
        self.bracket_button = self._control_button("Add Bracket", self.toggle_bracket, YELLOW)
        self.save_button = self._control_button("Save", self.save_to_file, YELLOW)
        self.load_button = self._control_button("Load", self.load_from_file, ORANGE)
        self.collapse_button = self._control_button("Collapse", self.toggle_left_panels, GREY)
        for button in (
            self.safe_button, self.fire_button, self.remove_button, self.add_button,
            self.bracket_button, self.collapse_button, self.save_button, self.load_button,
        ):
            button.setFixedHeight(cb_h)
            button.setMinimumWidth(_px(74, minimum=62))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        for index, button in enumerate((
            self.safe_button, self.fire_button, self.remove_button, self.add_button,
            self.bracket_button, self.collapse_button, self.save_button, self.load_button,
        )):
            button_layout.addWidget(button, index // 4, index % 4)
        for col in range(4):
            button_layout.setColumnStretch(col, 1)
        # Centre the buttons in the card so the (taller, height-matched) Control
        # box does not leave the grid floating with an uneven gap.
        control_col.addStretch(1)
        control_col.addLayout(button_layout)
        control_col.addStretch(1)
        bar.addWidget(control_area, 1)

        view_area = FluentGroupBox("Channels")
        view_area.setFixedWidth(_px(286, minimum=246))
        view_col = QtWidgets.QVBoxLayout(view_area)
        view_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        view_col.setSpacing(_px(4, minimum=3))
        self.add_channel_combo = FluentComboBox()
        self.add_channel_combo.setFixedHeight(cb_h)
        self.add_channel_combo.setToolTip("Pick a hidden channel or DAC bus to show.")
        view_col.addWidget(self.add_channel_combo)
        view_btn_row = QtWidgets.QHBoxLayout()
        view_btn_row.setContentsMargins(0, 0, 0, 0)
        view_btn_row.setSpacing(_px(6, minimum=4))
        self.add_channel_button = FluentButton("Add", color=ACCENT)
        self.add_channel_button.setToolTip("Add the selected hidden channel/bus to the table.")
        self.add_channel_button.clicked.connect(self.add_selected_channel)
        self.hide_off_button = FluentButton("Hide Off", color=ORANGE)
        self.hide_off_button.setToolTip("Hide channels that are off in every period.")
        self.hide_off_button.clicked.connect(self.hide_off_channels)
        self.show_all_button = FluentButton("Show All", color=ACCENT)
        self.show_all_button.setToolTip("Show every hardware channel.")
        self.show_all_button.clicked.connect(self.show_all_channels)
        for button in (self.add_channel_button, self.hide_off_button, self.show_all_button):
            button.setFixedHeight(cb_h)
            button.setMinimumWidth(_px(56, minimum=48))
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            view_btn_row.addWidget(button, 1)
        view_col.addLayout(view_btn_row)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.visible_label.setFixedHeight(_row_height())
        view_col.addWidget(self.visible_label)
        view_col.addStretch(1)
        bar.addWidget(view_area)

        self.button_frame.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum)
        edit_layout.addWidget(self.button_frame)
        self.tabs.addTab(self.edit_tab, "Edit")

        self.preview_tab = QtWidgets.QWidget()
        self.preview_tab.setStyleSheet("background: transparent;")
        preview_layout = QtWidgets.QVBoxLayout(self.preview_tab)
        preview_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        preview_layout.setSpacing(_px(8, minimum=5))

        preview_controls = FluentFrame()
        preview_controls.setFixedHeight(_px(48, minimum=40))
        preview_row = QtWidgets.QHBoxLayout(preview_controls)
        preview_row.setContentsMargins(_px(12), _px(6), _px(12), _px(6))
        preview_row.setSpacing(_px(10, minimum=6))
        preview_control_h = _px(32, minimum=28)
        self.preview_include_off = FluentSwitch("Show off rows")
        # Wide enough for the toggle plus the full "Show off rows" label even with
        # a wider substitute font (offscreen screenshots), so it never clips.
        self.preview_include_off.setFixedSize(_px(198, minimum=178), preview_control_h)
        self.preview_include_off.setToolTip("Show channels that are always off in the preview.")
        self.preview_include_off.toggled.connect(self._request_preview_refresh)
        self.preview_save_figure_button = FluentButton("Save Figure", color=ACCENT)
        self.preview_save_figure_button.setFixedSize(_px(124, minimum=108), preview_control_h)
        self.preview_save_figure_button.clicked.connect(self.save_figure)
        self.preview_status = FluentLineEdit("")
        self.preview_status.setEnabled(False)
        self.preview_status.setFixedHeight(preview_control_h)
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
        self.scan_tab = self._build_scan_tab()
        self.tabs.addTab(self.scan_tab, "Scan")
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
        self.collapse_button.setText("Show Left")
        self._sync_dataset_geometry()

    def show_left_panels(self) -> None:
        self._left_panels_collapsed = False
        self.left_panel_stub.hide()
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.collapse_button.setText("Collapse")
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
        # Suspend painting while we tear down and rebuild every channel panel and
        # period card (up to 5 periods x 62 channels = hundreds of widgets).  Each
        # addWidget on a *visible* tree would otherwise trigger an immediate
        # relayout + repaint; deferring to a single repaint at the end is the
        # dominant speed-up for "Show All".
        with batched_updates(self):
            self._rebuild_channel_panels()
            self._rebuild_periods()  # ends with _sync_dataset_geometry()
            self._refresh_hidden_combo()
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

        self.names_panel = ChannelNamesPanel(self.state, raw_labels=self.channel_pins)
        self.names_panel.changed.connect(self._handle_names_changed)
        self.names_panel_layout.addWidget(self.names_panel)
        self.name_edit = self.names_panel.name_edit

        self.channel_panel = ChannelPanel(self.state)
        self.channel_panel.changed.connect(self._mark_dirty)
        self.channel_panel.clearRequested.connect(self.clear_channel)
        self.channel_panel.delayScanRequested.connect(self._toggle_delay_scan)
        self.channel_panel.loadScanRequested.connect(self._load_scan_file)
        self.channel_panel.editScanRequested.connect(self._open_scan_tab)
        self.channel_panel_layout.addWidget(self.channel_panel)

    def _rebuild_periods(self) -> None:
        while self.drag_container.layout_main.count():
            item = self.drag_container.layout_main.takeAt(0)
            widget = item.widget()
            if widget is not None and widget is not self.drag_container.insert_indicator:
                widget.deleteLater()
        self.drag_container.items = []

        labels = self._display_labels_from_name_panel() if hasattr(self, "names_panel") else {
            row["key"]: _display_row_label(row) for row in _display_rows(self.state)
        }
        rows = _display_rows(self.state)
        compact = len(rows) > 16
        total_periods = len(self.state.periods)
        for index, period in enumerate(self.state.periods):
            hidden_states = {
                channel: period.states[self.state.channel_index(channel)]
                for channel in self.state.channels
            }
            card = PeriodCard(
                index,
                period,
                total_periods=total_periods,
                channels=self.state.channels,
                labels=labels,
                hidden_states=hidden_states,
                rows=rows,
                state=self.state,
                compact=compact,
                time_step_ns=self.state.time_step_ns,
            )
            card.changed.connect(self._mark_dirty)
            card.duration_dot.clicked.connect(lambda _checked=False, c=card: self._toggle_duration_scan(c))
            card.busScanRequested.connect(lambda bus_name, c=card: self._toggle_dac_scan(c, bus_name))
            self.drag_container.add_item(card, "pulse")
        if self.state.repeat_start is not None and self.state.repeat_end is not None and self.state.repeat_count > 1:
            start = RepeatBracket("start")
            end = RepeatBracket("end", self.state.repeat_count)
            start.changed.connect(self._mark_dirty)
            end.changed.connect(self._mark_dirty)
            self.drag_container.insert_item(self.state.repeat_start, start, "bracket_start")
            self.drag_container.insert_item(self.state.repeat_end + 2, end, "bracket_end")
            self.bracket_exists = True
            self.bracket_button.setText("Del Bracket")
        else:
            self.bracket_exists = False
            self.bracket_button.setText("Add Bracket")
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

        # sizeHint() below already triggers the layouts' size computation, so the
        # explicit adjustSize() resizes were redundant work (each forces a full
        # relayout of a 62-row panel).  The panels' real height is set via
        # setMinimumHeight / setFixedSize at the end of this method anyway.
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
        rows = getattr(getattr(self, "names_panel", None), "rows", _display_rows(self.state))
        if not hasattr(self, "names_panel"):
            return {str(row["key"]): _display_row_label(row) for row in rows}
        for row in rows:
            key = str(row["key"])
            if row.get("kind") == "bus":
                labels[key] = _display_row_label(row)
                continue
            edit = self.names_panel.label_edits.get(key)
            text = edit.text().strip() if edit is not None else self.state.channel_labels.get(key, "")
            labels[key] = text if text and text != key else key
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
        time_step_ns = float(self.channel_panel.step_edit.text() or self.state.time_step_ns)
        slots = self.state.reference_slots()
        cards = self.drag_container.pulse_cards()
        periods = [
            card.to_period(full_channels=self.state.channels, time_step_ns=time_step_ns, slots=slots)
            for card in cards
        ]
        scan_slots = self._reconcile_scan_slots(periods)
        n_slots = len(scan_slots)
        scan_table = [list(row)[:n_slots] + [0.0] * max(0, n_slots - len(row)) for row in self.state.scan_table]
        analog_bus_modes: dict[str, list[dict[str, object]]] = {}
        for card in cards:
            for bus_name, entry in card.bus_modes().items():
                analog_bus_modes.setdefault(bus_name, []).append(dict(entry))
        state = PulseTableState(
            channels=self.state.channels,
            visible_channels=self.state.visible_channels,
            periods=periods,
            name=self.name_edit.text().strip() or self.state.name or _default_pulse_name(),
            scan_slots=scan_slots,
            scan_table=scan_table,
            time_step_ns=time_step_ns,
            channel_labels=dict(self.state.channel_labels),
            analog_buses=dict(self.state.analog_buses),
            analog_bus_modes=analog_bus_modes or dict(self.state.analog_bus_modes),
            delays=dict(self.state.delays),
            delay_units=dict(self.state.delay_units),
            repeat_forever=bool(self.state.repeat_forever),
        )
        self.names_panel.read_values(state)
        self.channel_panel.read_values(state)
        state.apply_analog_bus_modes_to_period_states()
        start, end, repeat = self._read_bracket()
        state.repeat_start = start
        state.repeat_end = end
        state.repeat_count = repeat
        state.validate()
        self.state = state
        return state

    def _reconcile_scan_slots(self, periods: Sequence[PulsePeriod]) -> list[ScanSlot]:
        """Carry scan slots through an edit, realigning ``duration`` targets.

        Slots are owned by ``self.state`` (created/removed only by the scan
        dots).  Here we only re-point ``duration`` slots at the period that now
        holds their ``s{i}`` expression, since drag-reordering can move it.
        """

        var_to_period: dict[int, int] = {}
        for period_index, period in enumerate(periods):
            slot_index = _slot_index_of_expr(period.duration)
            if slot_index is not None:
                var_to_period[slot_index] = period_index
        out: list[ScanSlot] = []
        for index, slot in enumerate(self.state.scan_slots):
            if slot.kind == "duration":
                period_index = var_to_period.get(index)
                target = str(period_index) if period_index is not None else slot.target
                out.append(ScanSlot("duration", target, slot.label, slot.unit, slot.nominal))
            else:
                out.append(slot)
        return out

    def _remember_scan_column(self, state: PulseTableState, kind: str, target: str, slot_index: int) -> None:
        """Stash a field's scan-table column before it is unbound.

        So that toggling a scan dot OFF and back ON restores the values the user
        typed, instead of resetting the column to the field's nominal.
        """

        cache = getattr(self, "_scan_col_cache", None)
        if cache is None:
            cache = self._scan_col_cache = {}
        cache[(kind, str(target))] = [
            float(row[slot_index]) for row in state.scan_table if slot_index < len(row)
        ]

    def _restore_scan_column(self, state: PulseTableState, kind: str, target: str, slot_index: int) -> None:
        cache = getattr(self, "_scan_col_cache", None)
        values = (cache or {}).get((kind, str(target)))
        if not values:
            return
        for row_index, row in enumerate(state.scan_table):
            if slot_index < len(row) and row_index < len(values):
                row[slot_index] = float(values[row_index])

    def _remember_field_state(self, state: PulseTableState, kind: str, target: str) -> None:
        """Snapshot a field's full pre-bind state (mode/value/unit).

        Binding rewrites the field to ``s{i}`` and a plain ``unbind`` only knows
        how to reset it to a hard default (duration -> 1000 ns, delay -> 0, DAC ->
        hold).  Stashing the original here lets us put the field back EXACTLY as
        it was -- e.g. a DAC that was "edge / 500" returns to "edge / 500", not
        "hold".
        """

        cache = getattr(self, "_field_state_cache", None)
        if cache is None:
            cache = self._field_state_cache = {}
        key = (kind, str(target))
        try:
            if kind == "duration":
                period = state.periods[int(target)]
                cache[key] = ("duration", period.duration, period.unit)
            elif kind == "delay":
                cache[key] = ("delay", state.delays.get(target), state.delay_units.get(target, "ns"))
            elif kind == "dac":
                bus, _, period_str = str(target).rpartition("@")
                period_index = int(period_str)
                plan = state.analog_bus_plan(bus)
                entry = dict(plan[period_index]) if period_index < len(plan) else {"mode": "hold", "value": None}
                cache[key] = ("dac", bus, period_index, entry)
        except Exception:
            cache.pop(key, None)

    def _restore_field_state(self, state: PulseTableState, kind: str, target: str) -> None:
        cache = getattr(self, "_field_state_cache", None)
        saved = (cache or {}).get((kind, str(target)))
        if not saved:
            return
        try:
            if saved[0] == "duration":
                _, duration, unit = saved
                period = state.periods[int(target)]
                state.periods[int(target)] = PulsePeriod(duration, period.states, unit=unit, name=period.name)
            elif saved[0] == "delay":
                _, value, unit = saved
                state.delays[target] = value
                state.delay_units[target] = unit
            elif saved[0] == "dac":
                _, bus, period_index, entry = saved
                plan = state.analog_bus_plan(bus)
                if period_index < len(plan):
                    plan[period_index] = dict(entry)
                    state.analog_bus_modes[bus] = plan
                    state.apply_analog_bus_modes_to_period_states()
            state.validate()
        except Exception:
            pass

    def _apply_scan_state_in_place(self, state: PulseTableState) -> bool:
        """Refresh scan-binding visuals on the EXISTING widgets, no rebuild.

        A scan-dot toggle never changes the period/channel structure -- only
        which fields are bound and their slot numbers.  Re-deriving each field's
        display in place (instead of destroying + recreating hundreds of
        widgets) turns a ~400 ms "Show All" toggle into a few milliseconds.
        Returns ``False`` without touching anything if the live widget tree no
        longer matches the state, so the caller can fall back to ``load_state``.
        """

        try:
            cards = self.drag_container.pulse_cards()
            if len(cards) != len(state.periods) or not hasattr(self, "channel_panel"):
                return False
            buses = state.bus_channels()
            for pidx, card in enumerate(cards):
                period = state.periods[pidx]
                # --- duration ---
                scanned = _is_slot_expr(period.duration)
                edit, combo = card.duration_edit, card.unit_combo
                with _signals_blocked(edit, combo):
                    edit.set_scan_bound(False)
                    edit.setText(_period_duration_text(period))
                    combo.setCurrentText("str (ns)" if scanned else period.unit)
                    combo.setEnabled(not scanned)
                    if scanned:
                        idx = _slot_index_of_expr(period.duration)
                        edit.set_scan_bound(True, None if idx is None else idx + 1)
                # --- bus value fields ---
                for bus, value_edit in getattr(card, "bus_value_edits", {}).items():
                    plan = state.analog_bus_plan(bus)
                    entry = dict(plan[pidx]) if pidx < len(plan) else {"mode": "hold", "value": None}
                    mode = str(entry.get("mode", "hold")).lower()
                    raw = entry.get("value")
                    bound = _is_slot_expr(raw)
                    members = buses.get(bus, [])
                    max_value = (1 << max(1, len(members))) - 1
                    if bound:
                        disp = str(raw)
                    elif raw is None:
                        disp = str(_bus_value_from_states(state, period, bus))
                    else:
                        disp = str(max(0, min(max_value, int(raw))))
                    mode_combo = card.bus_mode_combos[bus]
                    with _signals_blocked(value_edit, mode_combo):
                        value_edit.set_scan_bound(False)
                        value_edit.setText(disp)
                        value_edit.set_editable(mode != "hold")
                        mode_combo.setCurrentText(_bus_mode_title(mode))
                        mode_combo.setEnabled(not bound)
                        if bound:
                            si = state.slot_index_for("dac", f"{bus}@{pidx}")
                            value_edit.set_scan_bound(True, None if si is None else si + 1)
            # --- channel panel: delay fields ---
            panel = self.channel_panel
            for key, edit in panel.delay_edits.items():
                is_bus = str(key).startswith("bus:")
                if is_bus:
                    members = buses.get(str(key).split(":", 1)[1], [])
                    vals = [state.delays.get(c, 0) for c in members]
                    units = [state.delay_units.get(c, "ns") for c in members]
                    dval = vals[0] if vals and all(v == vals[0] for v in vals) else 0
                    dunit = units[0] if units and all(u == units[0] for u in units) else "ns"
                else:
                    dval = state.delays.get(key, 0)
                    dunit = state.delay_units.get(key, "ns")
                bound = _is_slot_expr(dval)
                ucombo = panel.delay_units.get(key)
                with _signals_blocked(edit, ucombo):
                    edit.set_scan_bound(False)
                    edit.setText(str(dval))
                    if ucombo is not None:
                        ucombo.setCurrentText("str (ns)" if bound else dunit)
                    if bound and not is_bus:
                        si = state.slot_index_for("delay", key)
                        edit.set_scan_bound(True, None if si is None else si + 1)
                        if ucombo is not None:
                            ucombo.setEnabled(False)
                    elif ucombo is not None:
                        ucombo.setEnabled(True)
            panel.state = state
            panel.set_scan_summary()
            return True
        except Exception:
            return False

    def _toggle_scan_binding(self, state: PulseTableState, kind: str, target: str, *, bind, label: str = "", unit: str = "ns") -> None:
        slot_index = state.slot_index_for(kind, target)
        if slot_index is None:
            self._remember_field_state(state, kind, target)  # capture BEFORE binding overwrites it
            new_index = bind()
            self._restore_scan_column(state, kind, target, new_index)
        else:
            self._remember_scan_column(state, kind, target, slot_index)
            state.unbind_slot(slot_index)
            self._restore_field_state(state, kind, target)  # put the field back exactly as it was
        # Fast path: a scan toggle keeps the structure, so update the existing
        # widgets in place (milliseconds) instead of a full rebuild (~400 ms with
        # all channels shown).  Fall back to load_state if anything looks off.
        self.state = state
        if self._apply_scan_state_in_place(state):
            self._preview_dirty = True
            self._update_summary()
            if hasattr(self, "scan_tab"):
                self._refresh_scan_tab()
        else:
            self.load_state(state)

    def _toggle_duration_scan(self, card: "PeriodCard") -> None:
        try:
            state = self.read_state()
            index = self.drag_container.pulse_cards().index(card)
            unit = card.unit_combo.currentText()
            self._toggle_scan_binding(
                state, "duration", str(index),
                bind=lambda: state.bind_field("duration", str(index), unit="ns" if unit == "str (ns)" else unit),
            )
        except Exception as exc:
            self._message(str(exc))

    def _toggle_delay_scan(self, channel: str) -> None:
        try:
            state = self.read_state()
            unit = state.delay_units.get(channel, "ns")
            self._toggle_scan_binding(
                state, "delay", channel,
                bind=lambda: state.bind_field("delay", channel, unit="ns" if unit == "str (ns)" else unit),
            )
        except Exception as exc:
            self._message(str(exc))

    def _toggle_dac_scan(self, card: "PeriodCard", bus_name: str) -> None:
        try:
            state = self.read_state()
            index = self.drag_container.pulse_cards().index(card)
            target = f"{bus_name}@{index}"
            self._toggle_scan_binding(
                state, "dac", target,
                bind=lambda: state.bind_field("dac", target, unit="value", label=bus_name),
            )
        except Exception as exc:
            self._message(str(exc))

    def _load_scan_file(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_slots:
                self._message("Bind at least one field to a scan slot (click a dot) before loading an array.")
                return
            start = str(Path(self.address_str).parent if self.address_str else _pulse_files_dir())
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load scan array", start, "Scan array (*.npy *.csv *.txt *.json)")
            if not path:
                return
            state.set_scan_table(load_scan_table(path))
            self.load_state(state)
            if hasattr(self, "preview_status"):
                self.preview_status.setText(f"Loaded {len(state.scan_table)} scan points from {Path(path).name}")
        except Exception as exc:
            self._message(str(exc))

    def _open_scan_tab(self) -> None:
        if hasattr(self, "tabs") and hasattr(self, "scan_tab"):
            self.tabs.setCurrentWidget(self.scan_tab)
            self._refresh_scan_tab()

    def _build_scan_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        tab.setStyleSheet("background: transparent;")
        layout = QtWidgets.QVBoxLayout(tab)
        margin = _px(8, minimum=5)
        layout.setContentsMargins(margin, margin, margin, margin)
        layout.setSpacing(_px(8, minimum=5))

        info = FluentFrame()
        info.setMinimumHeight(_px(64, minimum=52))
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(_px(12), _px(8), _px(12), _px(8))
        self.scan_slots_label = FluentLabel("")
        self.scan_slots_label.setWordWrap(True)
        info_layout.addWidget(self.scan_slots_label)
        layout.addWidget(info)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(_px(8, minimum=5))

        editor_box = FluentGroupBox("Generate the scan table (Python)")
        editor_layout = QtWidgets.QVBoxLayout(editor_box)
        editor_layout.setContentsMargins(_px(8), _px(28, minimum=24), _px(8), _px(8))
        editor_layout.setSpacing(_px(6, minimum=4))
        self.scan_code = QtWidgets.QPlainTextEdit()
        self.scan_code.setStyleSheet(
            f'QPlainTextEdit {{ background: white; color: {ACCENT and "#323130"}; '
            f'border: 1px solid #A19F9D; border-radius: {scaled_px(4)}px; '
            f'font: {fluent_font_size()}pt "Consolas", "Courier New", monospace; padding: {_px(4)}px; }}'
        )
        editor_layout.addWidget(self.scan_code, 1)
        code_buttons = QtWidgets.QHBoxLayout()
        code_buttons.setSpacing(_px(6, minimum=4))
        run_btn = FluentButton("Run", color=GREEN)
        run_btn.setFixedHeight(_row_height())
        run_btn.setToolTip("Run the code; assign an N_points x N_slots array to 'scan_table'.")
        run_btn.clicked.connect(self._run_scan_code)
        load_btn = FluentButton("Load File", color=ACCENT)
        load_btn.setFixedHeight(_row_height())
        load_btn.clicked.connect(self._load_scan_file)
        save_btn = FluentButton("Save Array", color=YELLOW)
        save_btn.setFixedHeight(_row_height())
        save_btn.clicked.connect(self._save_scan_array)
        for button in (run_btn, load_btn, save_btn):
            button.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            code_buttons.addWidget(button, 1)
        editor_layout.addLayout(code_buttons)
        body.addWidget(editor_box, 3)

        preview_box = FluentGroupBox("Scan table")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)
        preview_layout.setContentsMargins(_px(8), _px(28, minimum=24), _px(8), _px(8))
        self.scan_table_view = QtWidgets.QPlainTextEdit()
        self.scan_table_view.setReadOnly(True)
        # White with a light border to match the code editor on the left -- the
        # old grey (BG) fill read as an out-of-place grey side panel.
        self.scan_table_view.setStyleSheet(
            f'QPlainTextEdit {{ background: white; color: #323130; border: 1px solid #A19F9D; '
            f'border-radius: {scaled_px(4)}px; font: {fluent_font_size()}pt "Consolas", "Courier New", monospace; '
            f'padding: {_px(4)}px; }}'
        )
        preview_layout.setSpacing(_px(6, minimum=4))
        preview_layout.addWidget(self.scan_table_view, 1)
        # Mirror the left column's button-row footprint so the two boxes share an
        # identical bottom edge.  Without this the table view hangs ~one row
        # lower than the code editor, and its grey border reads as a stray
        # "extra grey edge" protruding past the left box.
        scan_table_footer = QtWidgets.QWidget()
        scan_table_footer.setFixedHeight(_row_height())
        scan_table_footer.setStyleSheet("background: transparent;")
        preview_layout.addWidget(scan_table_footer)
        body.addWidget(preview_box, 2)
        layout.addLayout(body, 1)

        self._scan_code_initialized = False
        return tab

    def _refresh_scan_tab(self) -> None:
        if not hasattr(self, "scan_slots_label"):
            return
        try:
            state = self.read_state()
        except Exception:
            state = self.state
        if state.scan_slots:
            lines = ["Columns of the scan table (one row = one scan point):"]
            for index, slot in enumerate(state.scan_slots):
                lines.append(
                    f"  s{index}: {_scan_slot_label(state, index)}  [{slot.unit}]  (nominal {format_compact_number(slot.nominal)})"
                )
            self.scan_slots_label.setText("\n".join(lines))
        else:
            self.scan_slots_label.setText(
                "No scan slots bound yet. In the Edit tab, click the dot next to any duration, delay, or DAC value to scan it."
            )
        rows = state.scan_table
        if rows:
            header = "   ".join(f"s{i}" for i in range(len(state.scan_slots)))
            shown = ["   ".join(format_compact_number(value) for value in row) for row in rows[:40]]
            footer = f"\n... {len(rows)} points total" if len(rows) > 40 else f"\n{len(rows)} point(s)"
            self.scan_table_view.setPlainText(header + "\n" + "\n".join(shown) + footer)
        else:
            self.scan_table_view.setPlainText("(empty — Run code or Load File)")
        if not self._scan_code_initialized and not self.scan_code.toPlainText().strip():
            self.scan_code.setPlainText(_default_scan_code(max(1, len(state.scan_slots))))
            self._scan_code_initialized = True

    def _run_scan_code(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_slots:
                self._message("Bind at least one field to a scan slot first (click a dot in the Edit tab).")
                return
            import numpy as np
            import math as _math

            namespace = {"np": np, "numpy": np, "math": _math, "n_slots": len(state.scan_slots)}
            exec(self.scan_code.toPlainText(), namespace)  # noqa: S102 - local experiment tool
            table = namespace.get("scan_table")
            if table is None:
                self._message("Assign an N_points x N_slots array to a 'scan_table' variable.")
                return
            array = np.atleast_2d(np.asarray(table, dtype=float))
            state.set_scan_table([[float(value) for value in row] for row in array])
            self.load_state(state)
            self._open_scan_tab()
        except Exception as exc:
            self._message(f"Scan code error: {exc}")

    def _save_scan_array(self) -> None:
        try:
            state = self.read_state()
            if not state.scan_table:
                self._message("No scan table to save yet. Run code or load a file first.")
                return
            import numpy as np

            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save scan array", str(self._default_scan_path(state)), "Scan array (*.npy *.csv)"
            )
            if not path:
                return
            target = Path(path)
            if target.suffix == "":
                target = target.with_suffix(".npy")
            array = np.asarray(state.scan_table, dtype=float)
            if target.suffix.lower() == ".csv":
                np.savetxt(target, array, delimiter=",")
            else:
                np.save(target, array)
            if hasattr(self, "preview_status"):
                self.preview_status.setText(f"Saved scan array: {target.name}")
        except Exception as exc:
            self._message(str(exc))

    def _default_scan_path(self, state: PulseTableState) -> Path:
        directory = Path(self.address_str).parent if self.address_str else _pulse_files_dir()
        return directory / f"{_safe_file_stem(state.name)}_scan.npy"

    @staticmethod
    def _resize_analog_bus_modes(state: PulseTableState) -> None:
        target = len(state.periods)
        for bus_name in list(state.analog_bus_modes):
            entries = [dict(entry) for entry in state.analog_bus_modes.get(bus_name, [])]
            if len(entries) < target:
                entries.extend({"mode": "hold", "value": None} for _ in range(target - len(entries)))
            elif len(entries) > target:
                entries = entries[:target]
            state.analog_bus_modes[bus_name] = entries

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
        self._resize_analog_bus_modes(state)
        state.apply_analog_bus_modes_to_period_states()
        state.validate()
        self.load_state(state)

    def remove_period(self) -> None:
        state = self.read_state()
        if len(state.periods) > 1:
            last_index = len(state.periods) - 1
            for slot_index in reversed(range(len(state.scan_slots))):
                slot = state.scan_slots[slot_index]
                drop = (slot.kind == "duration" and slot.target == str(last_index)) or (
                    slot.kind == "dac" and slot.dac_period == last_index
                )
                if drop:
                    state.unbind_slot(slot_index)
            state.periods.pop()
            self._resize_analog_bus_modes(state)
            state.apply_analog_bus_modes_to_period_states()
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
        self.bracket_button.setText("Del Bracket")
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
            if _is_bus_key(channel):
                bus = channel.split(":", 1)[1]
                for member in state.bus_channels().get(bus, []):
                    state.clear_channel(member)
            else:
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
        data = self.add_channel_combo.currentData()
        text = str(data if data is not None else self.add_channel_combo.currentText())
        if not text:
            return
        channel = text.split("  ", 1)[0]
        state = self.read_state()
        if _is_bus_key(channel):
            bus = channel.split(":", 1)[1]
            for member in state.bus_channels().get(bus, []):
                state.show_channel(member)
        else:
            state.show_channel(channel)
        self.load_state(state)

    def _refresh_hidden_combo(self) -> None:
        self.add_channel_combo.clear()
        visible = set(self.state.visible_channels)
        bus_members = set()
        for bus, members in self.state.bus_channels().items():
            bus_members.update(members)
            if not any(member in visible for member in members):
                self.add_channel_combo.addItem(f"{_bus_display_label(bus)}  ({len(members)} pins)", _bus_key(bus))
        for channel in self.state.channels:
            if channel in bus_members:
                continue
            if channel not in visible:
                label = self.state.label_for(channel)
                raw = self.channel_pins.get(channel, channel)
                display = f"{raw}  ({label})" if label != channel else raw
                self.add_channel_combo.addItem(display, channel)
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
                # Bundle next to the pulse: (1) the pulse itself, (2) the preview
                # figure, and (3) the scan -- both its raw data (.npy) and the
                # compiled, FPGA-ready scan program (.json).  Per-artifact failures
                # are reported (not silently swallowed) so a partial save is visible.
                saved: list[str] = []
                failed: list[str] = []
                state.save(path_obj)
                saved.append(path_obj.name)
                try:
                    figure_path = path_obj.with_suffix(".png")
                    self._save_preview_image(state, figure_path)
                    saved.append(figure_path.name)
                except Exception as exc:
                    failed.append(f"preview ({exc})")
                if state.scan_table:
                    try:
                        import numpy as np

                        scan_path = path_obj.with_name(path_obj.stem + "_scan.npy")
                        np.save(scan_path, np.asarray(state.scan_table, dtype=float))
                        saved.append(scan_path.name)
                    except Exception as exc:
                        failed.append(f"scan data ({exc})")
                if state.scan_slots:
                    try:
                        import json

                        program = state.compile_scan(clock_hz=1e9 / float(state.time_step_ns))
                        program_path = path_obj.with_name(path_obj.stem + "_program.json")
                        program_path.write_text(json.dumps(program.to_dict(), indent=2), encoding="utf-8")
                        saved.append(program_path.name)
                    except Exception as exc:
                        failed.append(f"scan program ({exc})")
                self.address_str = str(path_obj)
                self._last_save_state = state.to_dict()
                self._last_load_state = None
                self.stateui_manager.address_str = str(path_obj)
                self.stateui_manager.filestate = PulseStateUIManager.FileState.SAVE
                if hasattr(self, "preview_status"):
                    message = f"Saved: {', '.join(saved)}"
                    if failed:
                        message += "  |  skipped: " + "; ".join(failed)
                    self.preview_status.setText(message)
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
            state = (
                self.read_state()
                if hasattr(self, "channel_panel") and hasattr(self, "drag_container")
                else self.state
            )
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
            if state.scan_slots:
                parts.append(f"scan {len(state.scan_slots)} slots × {len(state.scan_table)} pts")
            if hasattr(self, "channel_panel"):
                self.channel_panel.state = state
                self.channel_panel.set_scan_summary()
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
                self.names_panel.visible_label.setText(f"{len(state.visible_channels)}/{len(state.channels)}")
            if hasattr(self, "visible_label"):
                self.visible_label.setText(f"Visible {len(state.visible_channels)}/{len(state.channels)} | Hidden {len(state.channels) - len(state.visible_channels)}")
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
        # Channel delays can push edges past the period-table end; the repeat
        # markers are computed from period starts only, so without this the
        # delayed tail renders OUTSIDE the ×∞ loop bracket (reads as unphysical).
        # Extend the infinite-loop bracket to enclose the whole drawn sequence.
        seq_end = float(getattr(sequence, "duration", 0.0) or 0.0)
        if seq_end > 0.0:
            repeat_brackets = [
                (start, max(stop, seq_end), label) if "∞" in str(label) else (start, stop, label)
                for (start, stop, label) in repeat_brackets
            ]
        analog_traces, folded_members = _analog_bus_traces(state)
        digital_channel_universe = [channel for channel in state.channels if channel not in folded_members]
        channels = pulse_plot_channels(
            sequence,
            channels=digital_channel_universe,
            include_always_off=include_always_off,
        )
        # Defensive: a folded DAC bit must never appear as its own digital row.
        channels = [channel for channel in channels if channel not in folded_members]
        plotter = frontend_plot(
            sequence,
            kind="pulse",
            channels=channels,
            include_always_off=True,
            repeat_notation=repeat,
            repeat_brackets=repeat_brackets,
            channel_labels={channel: state.label_for(channel) for channel in channels},
            analog_traces=analog_traces,
            title=state.name,
            show_names=True,
            display=False,
            data_figure=False,
        )
        bus_rows = [str(trace.get("name")) for trace in analog_traces]
        self._annotate_variable_regions(plotter, state, channels, bus_rows=bus_rows)
        return plotter, channels, repeat

    def _annotate_variable_regions(
        self,
        plotter,
        state: PulseTableState,
        channels: Sequence[str] | None = None,
        *,
        bus_rows: Sequence[str] | None = None,
    ) -> None:
        """Shade the time spans affected by each scan slot in transparent orange.

        - a scanned *duration* spans its whole period across all channels;
        - a scanned *delay* spans only the lead-in of its channel's pulses;
        - a scanned *DAC* value spans its period on its own analog-bus row.

        Each slot carries its 1-based number exactly once, placed on the row it
        affects (delay on its channel, DAC on its bus), so several scanned DAC
        buses get distinct, non-overlapping labels instead of piling up.
        """

        if not hasattr(plotter, "ax"):
            return
        slots = state.reference_slots()
        starts_ns = [0.0]
        for period in state.periods:
            starts_ns.append(starts_ns[-1] + period.duration_ns(slots=slots, time_step_ns=state.time_step_ns))
        # Use the plotter's ACTUAL row geometry (data coordinates) so highlights
        # land exactly on the channels they belong to -- guessing y from a row
        # count drifts and put delay bands on the wrong channel.
        ax = plotter.ax
        base_y = dict(getattr(plotter, "_pulse_baseline_y", {}) or {})
        analog_y = dict(getattr(plotter, "_analog_baseline_y", {}) or {})
        row_h = float(getattr(plotter, "_pulse_row_height", 0.64) or 0.64)
        if not base_y and not analog_y:
            return
        all_baselines = list(base_y.values()) + list(analog_y.values())
        area_bottom = min(all_baselines)
        area_top = max(all_baselines) + row_h          # top edge of the top channel
        ylim_top = float(ax.get_ylim()[1])
        x_lo, x_hi = ax.get_xlim()
        min_width = max((x_hi - x_lo) * 0.004, 1e-12)

        from matplotlib.patches import Rectangle
        plotter.variable_region_artists = []
        plotter.variable_region_labels = []

        def add_band(x0: float, x1: float, y0: float, y1: float, alpha: float) -> None:
            if x1 < x0:
                x0, x1 = x1, x0
            if x1 - x0 < min_width:
                x1 = x0 + min_width
            patch = Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=ORANGE, edgecolor="none",
                              alpha=alpha, linewidth=0.0, zorder=6, transform=ax.transData)
            ax.add_patch(patch)
            plotter.variable_region_artists.append(patch)

        def add_number(xc: float, yc: float, tag: str, va: str = "center") -> None:
            if not tag:
                return
            # Mimic the bound scan-dot badge: a filled orange circle with a white
            # digit (same look as FluentScanDot), keeping the small font size.
            text = ax.text(xc, yc, tag, transform=ax.transData, ha="center", va=va,
                           color="white", fontsize=max(2.6, float(fluent_font_size()) * 0.28),
                           fontweight="bold", clip_on=False, zorder=12,
                           bbox=dict(boxstyle="circle,pad=0.3", facecolor=ORANGE, edgecolor="none"))
            plotter.variable_region_labels.append(text)

        # Unified tint; the value highlight (DAC) is a touch stronger but same hue.
        BAND_ALPHA = 0.18
        bus_groups = state.bus_channels()
        for slot_index, slot in enumerate(state.scan_slots):
            tag = str(slot_index + 1)
            if slot.kind == "duration":
                pidx = int(slot.target) if slot.target.lstrip("-").isdigit() else -1
                if not (0 <= pidx < len(state.periods)):
                    continue
                x0 = starts_ns[pidx] * 1e-9
                x1 = starts_ns[pidx + 1] * 1e-9
                # Band covers exactly the channel rows (top = top channel's top,
                # never above it).  Number sits in the headroom just above the top
                # channel but below the title/bracket.
                add_band(x0, x1, area_bottom, area_top, BAND_ALPHA)
                label_y = min(area_top + row_h * 0.5, ylim_top - row_h * 0.2)
                add_number((x0 + x1) / 2, label_y, tag, va="center")
            elif slot.kind == "delay":
                channel = slot.target
                if channel not in base_y:
                    continue
                try:
                    delay_ns = state.delay_ns(channel, slots=slots, time_step_ns=state.time_step_ns)
                except Exception:
                    continue
                channel_idx = state.channel_index(channel)
                y0 = base_y[channel]
                y1 = y0 + row_h
                active_start: float | None = None
                spans: list[tuple[float, float]] = []
                for period_index, period in enumerate(state.periods):
                    on = int(period.states[channel_idx])
                    if on and active_start is None:
                        active_start = starts_ns[period_index]
                    elif not on and active_start is not None:
                        spans.append((active_start, active_start + delay_ns))
                        active_start = None
                if active_start is not None:
                    spans.append((active_start, active_start + delay_ns))
                # Shade each lead-in on THIS channel's row; label once.
                for span_idx, (span_start, span_stop) in enumerate(spans):
                    add_band(span_start * 1e-9, span_stop * 1e-9, y0, y1, BAND_ALPHA + 0.10)
                    if span_idx == 0:
                        add_number((span_start + span_stop) / 2 * 1e-9, (y0 + y1) / 2, tag, va="center")
            elif slot.kind == "dac":
                bus = slot.dac_bus
                pidx = slot.dac_period
                if bus not in analog_y or not (0 <= pidx < len(state.periods)):
                    continue
                x0 = starts_ns[pidx] * 1e-9
                x1 = starts_ns[pidx + 1] * 1e-9
                members = bus_groups.get(bus, [])
                max_v = max(1, (1 << len(members)) - 1)
                try:
                    value = float(state.analog_bus_value_at_period_start(pidx, bus))
                except Exception:
                    value = 0.0
                # Follow the DA LINE: highlight the trace segment at the value's
                # height over the scanned period (not a full-row block).
                vy = analog_y[bus] + row_h * min(1.0, max(0.0, value / max_v))
                if x1 - x0 < min_width:
                    x1 = x0 + min_width
                seg = ax.plot([x0, x1], [vy, vy], color=ORANGE, linewidth=3.0, alpha=0.9,
                              solid_capstyle="butt", zorder=8)[0]
                plotter.variable_region_artists.append(seg)
                # Number centred vertically in the bus row (only *duration* labels
                # sit above the band; delay and DAC labels live inside their row).
                add_number((x0 + x1) / 2, analog_y[bus] + row_h * 0.5, tag, va="center")

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
        current = self.tabs.currentWidget()
        if current is self.preview_tab:
            self.refresh_preview()
        elif current is getattr(self, "scan_tab", None):
            self._refresh_scan_tab()

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
        if os.environ.get("QT_QPA_PLATFORM", "").lower() == "offscreen":
            if hasattr(self, "summary"):
                self.summary.setText(str(text))
            if hasattr(self, "preview_status"):
                self.preview_status.setText(str(text))
            return
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
    channel_pins: Mapping[str, str] | None = None,
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
        channel_pins=channel_pins,
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
