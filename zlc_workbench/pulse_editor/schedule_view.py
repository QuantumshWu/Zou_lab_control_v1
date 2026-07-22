"""Pure-Qt, current-document schedule surface for the Pulse Workbench.

This module is deliberately a view, not an authoring model.  It projects one
immutable :class:`zlc_pulse.PulseDocument`, keeps only stable ids and derived
presentation facts, and emits semantic intents for the Workbench controller.
It never reconstructs a document from widget contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from PyQt5 import QtCore, QtGui, QtWidgets

from zlc_frontend.qt_widgets import (
    ACCENT,
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
    FluentFrame,
    FluentGroupBox,
    FluentLabel,
    FluentLineEdit,
    FluentScanLineEdit,
    FluentScrollArea,
    FluentSwitch,
    fluent_scrollbar_stylesheet,
    fluent_scrollbar_thickness,
    format_compact_number,
    signals_blocked,
)
from zlc_pulse import (
    FIELD_DAC,
    FIELD_DELAY,
    FIELD_DURATION,
    PORT_CLOCK,
    PORT_DAC,
    PORT_DIGITAL,
    TIME_UNIT_TO_NS,
    PulseDocument,
    PulseFieldRef,
    PulseTargetManifest,
    count_authored_digital_pulses,
    pulse_target_manifest_from_lanes,
)

from ._layout import (
    add_labeled_widget as _add_labeled_widget,
    bus_mode_combo_width as _bus_mode_combo_width,
    card_gutter as _card_gutter,
    channel_label_width as _channel_label_width,
    channel_name_edit_width as _channel_name_edit_width,
    channel_row_height as _channel_row_height,
    hide_button_width as _hide_button_width,
    panel_top_height as _panel_top_height,
    period_card_width as _period_card_width,
    period_control_width as _period_control_width,
    px as _px,
    row_height as _row_height,
    row_region_vmetrics as _row_region_vmetrics,
    set_fixed_height as _set_fixed_height,
    time_unit_width as _time_unit_width,
)
from .repeat_presentation import pulse_repeat_presentation


DURATION_UNITS = ("ns", "us", "ms", "s")
DELAY_UNITS = ("ns", "us", "ms", "s")


def _bus_mode_title(mode: str) -> str:
    return {"edge": "Edge", "ramp": "Ramp", "hold": "Hold"}.get(
        str(mode).strip().lower(), "Hold"
    )


def _bus_mode_value(title: str) -> str:
    lowered = str(title).strip().lower()
    if lowered.startswith("ram"):
        return "ramp"
    if lowered.startswith("edg"):
        return "edge"
    return "hold"


def _format_clock_text(time_step_ns: float) -> str:
    step = float(time_step_ns)
    if step <= 0:
        return "—"
    return (
        f"{format_compact_number(1000.0 / step)} MHz · "
        f"{format_compact_number(step)} ns"
    )


def _summary_time_text(value_ns: float) -> str:
    for unit in ("s", "ms", "us", "ns"):
        factor = float(TIME_UNIT_TO_NS[unit])
        if abs(value_ns) >= factor or unit == "ns":
            return f"{format_compact_number(value_ns / factor, digits=6)} {unit}"
    return f"{format_compact_number(value_ns, digits=6)} ns"


def _number(text: object) -> int | float:
    value = float(str(text).strip())
    return int(value) if value.is_integer() else value


def _expanded_pulse_count(document: PulseDocument) -> int:
    """Project the pulse-owned logical count without deployment preflight."""

    return count_authored_digital_pulses(document)


def _repeat_summary_text(document: PulseDocument) -> str:
    """Return the frozen three-state wording from the shared UI policy."""

    period_ids = tuple(period.period_id for period in document.periods)
    repeat = document.repeat
    repeat_spec = (
        None
        if repeat is None
        else (
            period_ids.index(repeat.start_period_id),
            period_ids.index(repeat.end_period_id),
            repeat.count,
        )
    )
    summary, _spans = pulse_repeat_presentation(len(period_ids), repeat_spec)
    return summary


def _restart_high_labels(document: PulseDocument) -> tuple[str, ...]:
    """Labels of digital outputs high when a partial inner loop restarts."""

    repeat = document.repeat
    if repeat is None:
        return ()
    period_ids = tuple(period.period_id for period in document.periods)
    start = period_ids.index(repeat.start_period_id)
    end = period_ids.index(repeat.end_period_id)
    if start == 0 and end == len(period_ids) - 1:
        return ()
    digital_by_lane = {
        port.lanes[0]: port
        for port in document.target.ports
        if port.kind == PORT_DIGITAL
    }
    return tuple(
        digital_by_lane[lane].label
        for lane, high in zip(
            document.target.raw_lanes,
            document.periods[0].states,
            strict=True,
        )
        if high and lane in digital_by_lane
    )


@dataclass(frozen=True)
class _PortRow:
    """Ephemeral logical-port presentation; never a codec or domain authority."""

    key: str
    kind: str
    label: str
    width: int
    signed_range: tuple[int, int] | None
    endpoints: tuple[str, ...]


@dataclass(frozen=True)
class _Binding:
    kind: str
    position: int
    parameter_id: str


@dataclass(frozen=True)
class _FocusSnapshot:
    token: tuple[str, ...]
    cursor_position: int | None = None
    selection_start: int = -1
    selection_length: int = 0


@dataclass(frozen=True)
class _InteractionSnapshot:
    selected_period_id: str | None
    selected_gap: int | None
    horizontal_scroll: int
    vertical_scroll: int
    focus: _FocusSnapshot | None


def _port_rows(
    document: PulseDocument,
    manifest: PulseTargetManifest,
    *,
    visible_only: bool,
    visible_ports: tuple[str, ...],
) -> tuple[_PortRow, ...]:
    if document.target.abi_fingerprint != manifest.target.abi_fingerprint:
        raise ValueError("schedule manifest target differs from PulseDocument")
    visible = set(visible_ports)
    rows = []
    for exposed in manifest.ports:
        port = document.target.by_key[exposed.port_key]
        if port.kind == PORT_CLOCK or (visible_only and port.key not in visible):
            continue
        rows.append(
            _PortRow(
                key=port.key,
                kind=port.kind,
                label=port.label,
                width=port.width,
                signed_range=port.signed_range,
                endpoints=exposed.endpoints,
            )
        )
    return tuple(rows)


def _field_bindings(document: PulseDocument) -> dict[PulseFieldRef, _Binding]:
    result = {
        parameter.field: _Binding("scan", index, parameter.parameter_id)
        for index, parameter in enumerate(document.scan_parameters)
    }
    result.update(
        {
            parameter.field: _Binding("api", index, parameter.parameter_id)
            for index, parameter in enumerate(document.api_parameters)
        }
    )
    return result


def _digital_values(document: PulseDocument) -> dict[tuple[str, str], bool]:
    """Project validated storage cells onto logical digital port ids.

    The returned mapping is presentation-only: rows and emitted identities come
    exclusively from logical ``PulsePortSpec.key`` values.
    """

    positions = {lane: index for index, lane in enumerate(document.target.raw_lanes)}
    result: dict[tuple[str, str], bool] = {}
    for port in document.target.ports:
        if port.kind != PORT_DIGITAL:
            continue
        position = positions[port.lanes[0]]
        for period in document.periods:
            result[(period.period_id, port.key)] = bool(period.states[position])
    return result


def _analog_values(
    document: PulseDocument,
) -> tuple[dict[tuple[str, str], tuple[str, int | None]], dict[str, bool]]:
    carried = {
        port.key: 0 for port in document.target.ports if port.kind == PORT_DAC
    }
    values: dict[tuple[str, str], tuple[str, int | None]] = {}
    active = {key: False for key in carried}
    for period in document.periods:
        actions = {step.port: step for step in period.analog_steps}
        for port in carried:
            action = actions.get(port)
            if action is None:
                values[(period.period_id, port)] = ("hold", carried[port])
            else:
                carried[port] = action.value
                active[port] = True
                values[(period.period_id, port)] = (action.mode, action.value)
    return values, active


def _set_duration_units(combo: FluentComboBox, binding: _Binding | None, unit: str) -> None:
    scanned = binding is not None and binding.kind == "scan"
    desired = tuple(DURATION_UNITS) + (("str (ns)",) if scanned else ())
    current = tuple(combo.itemText(index) for index in range(combo.count()))
    if current != desired:
        combo.clear()
        combo.addItems(list(desired))
    selected = "str (ns)" if scanned else unit
    if combo.currentText() != selected:
        combo.setCurrentText(selected)
    combo.setEnabled(not scanned)


def _set_widget_text(widget, value: object) -> None:
    """Set textual state only when authority actually changed it."""

    text = str(value)
    if widget.text() != text:
        widget.setText(text)


def _reconcile_widget_order(
    layout: QtWidgets.QBoxLayout,
    previous: Sequence[str],
    desired: Sequence[str],
    widgets: Mapping[str, QtWidgets.QWidget],
    *,
    offset: int,
) -> None:
    """Insert or move only rows whose stable-key position actually changed."""

    desired = tuple(desired)
    desired_set = set(desired)
    current = [
        key
        for key in previous
        if key in desired_set
        and key in widgets
        and layout.indexOf(widgets[key]) >= 0
    ]
    for index, key in enumerate(desired):
        if index < len(current) and current[index] == key:
            continue
        widget = widgets[key]
        if key in current:
            current.remove(key)
            layout.removeWidget(widget)
        layout.insertWidget(offset + index, widget)
        current.insert(index, key)


class PeriodCard(FluentGroupBox):
    """One formal period card keyed by immutable ``period_id``."""

    periodNameEdited = QtCore.pyqtSignal(str, str)
    durationEdited = QtCore.pyqtSignal(str, object, str)
    digitalEdited = QtCore.pyqtSignal(str, str, bool)
    analogEdited = QtCore.pyqtSignal(str, str, str, object)
    bindingCycleRequested = QtCore.pyqtSignal(object)

    def __init__(
        self,
        index: int,
        period,
        *,
        total_periods: int,
        rows: Sequence[_PortRow],
        digital_values: Mapping[str, bool],
        analog_values: Mapping[str, tuple[str, int | None]],
        bindings: Mapping[PulseFieldRef, _Binding],
        parent=None,
    ) -> None:
        super().__init__("", parent)
        self.period_id = period.period_id
        self._rows: tuple[_PortRow, ...] = ()
        self.checks: dict[str, FluentCheckBox] = {}
        self.bus_mode_combos: dict[str, FluentComboBox] = {}
        self.bus_value_edits: dict[str, FluentScanLineEdit] = {}
        self.port_rows: dict[str, QtWidgets.QWidget] = {}
        self._projection_state: tuple[object, ...] | None = None

        card_width = _period_card_width()
        self.setMinimumWidth(card_width)
        self.setMaximumWidth(card_width)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        control_width = _period_control_width(card_width)

        column = QtWidgets.QVBoxLayout(self)
        self._column = column
        row_top, row_gap = _row_region_vmetrics()
        card_pad = _px(7)
        column.setContentsMargins(card_pad, card_pad, card_pad, card_pad)
        column.setSpacing(row_gap)
        self.set_period_position(index, total_periods)

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

        duration_ref = PulseFieldRef(FIELD_DURATION, self.period_id)
        duration_binding = bindings.get(duration_ref)
        duration_text = (
            f"s{duration_binding.position}"
            if duration_binding is not None and duration_binding.kind == "scan"
            else str(period.duration)
        )
        self.duration_edit = FluentScanLineEdit(
            duration_text,
            tooltip=(
                "Duration value; click the dot to cycle scan (sN) -> API (aN) -> off"
            ),
        )
        self.duration_edit.setFixedWidth(control_width)
        self.duration_edit.setToolTip("How long this period lasts, in the unit below")
        self.duration_dot = self.duration_edit.dot
        if duration_binding is not None:
            if duration_binding.kind == "scan":
                self.duration_edit.set_scan_bound(True, duration_binding.position + 1)
            else:
                self.duration_edit.set_api_bound(True, duration_binding.position + 1)
            self.duration_edit.setToolTip(
                f"{self.duration_edit.toolTip()}\nParameter: {duration_binding.parameter_id}"
            )
        top_layout.addWidget(_set_fixed_height(self.duration_edit))

        self.unit_combo = FluentComboBox()
        _set_duration_units(self.unit_combo, duration_binding, period.unit)
        self.unit_combo.setFixedWidth(control_width)
        top_layout.addWidget(_set_fixed_height(self.unit_combo))

        self.name_edit = FluentLineEdit(str(period.name or ""))
        self.name_edit.setPlaceholderText("name")
        self.name_edit.setFixedWidth(control_width)
        self.name_edit.setToolTip(
            "This period's name (shown in the preview and the summary)"
        )
        top_layout.addWidget(_set_fixed_height(self.name_edit))
        top_layout.addStretch()
        column.addWidget(top)
        column.addSpacing(max(0, row_top - card_pad))

        self.duration_edit.set_numeric_validator("float", bottom=0.0)
        if duration_binding is not None and duration_binding.kind == "scan":
            self.duration_edit.setValidator(None)
        self.name_edit.editingFinished.connect(self._commit_name)
        self.duration_dot.clicked.connect(
            lambda _checked=False, ref=duration_ref: self.bindingCycleRequested.emit(ref)
        )
        self.duration_edit.editingFinished.connect(self._commit_duration)
        self.unit_combo.currentTextChanged.connect(lambda _text: self._commit_duration())

        column.addStretch(1)
        self.reconcile(
            index,
            period,
            total_periods=total_periods,
            rows=rows,
            digital_values=digital_values,
            analog_values=analog_values,
            bindings=bindings,
        )

    def _create_port_row(
        self,
        row: _PortRow,
        *,
        digital_value: bool,
        analog_value: tuple[str, int | None],
        binding: _Binding | None,
    ) -> QtWidgets.QWidget:
        row_height = _channel_row_height()
        if row.kind == PORT_DAC:
            widget = self._build_bus_row(
                row,
                analog_value,
                binding,
                row_height,
            )
        else:
            check = FluentCheckBox(row.label)
            check.setToolTip(f"{row.label}: high for this whole period")
            check.setFixedHeight(row_height)
            check.setChecked(bool(digital_value))
            check.toggled.connect(
                lambda checked, key=row.key: self.digitalEdited.emit(
                    self.period_id, key, bool(checked)
                )
            )
            self.checks[row.key] = check
            widget = check
        self.port_rows[row.key] = widget
        return widget

    def _remove_port_row(self, key: str) -> None:
        widget = self.port_rows.pop(key)
        self._column.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()
        self.checks.pop(key, None)
        self.bus_mode_combos.pop(key, None)
        self.bus_value_edits.pop(key, None)

    def _reconcile_port_rows(
        self,
        rows: Sequence[_PortRow],
        *,
        digital_values: Mapping[str, bool],
        analog_values: Mapping[str, tuple[str, int | None]],
        bindings: Mapping[PulseFieldRef, _Binding],
    ) -> None:
        rows = tuple(rows)
        desired = tuple(row.key for row in rows)
        desired_by_key = {row.key: row for row in rows}
        if len(desired) != len(desired_by_key):
            raise ValueError("period rows must have unique keys")
        previous_kind = {row.key: row.kind for row in self._rows}
        topology_changed = desired != tuple(row.key for row in self._rows)
        for key in tuple(self.port_rows):
            replacement = desired_by_key.get(key)
            if replacement is None or replacement.kind != previous_kind.get(key):
                self._remove_port_row(key)
                topology_changed = True
        for row in rows:
            if row.key in self.port_rows:
                continue
            self._create_port_row(
                row,
                digital_value=bool(digital_values.get(row.key, False)),
                analog_value=analog_values.get(row.key, ("hold", 0)),
                binding=bindings.get(
                    PulseFieldRef(FIELD_DAC, self.period_id, row.key)
                ),
            )
            topology_changed = True
        if topology_changed:
            _reconcile_widget_order(
                self._column,
                tuple(row.key for row in self._rows),
                desired,
                self.port_rows,
                offset=2,
            )
        self._rows = rows

    def _build_bus_row(
        self,
        row_info: _PortRow,
        value: tuple[str, int | None],
        binding: _Binding | None,
        row_height: int,
    ) -> QtWidgets.QWidget:
        mode, number = value
        lo, hi = row_info.signed_range or (0, 0)
        row = QtWidgets.QWidget()
        row.setStyleSheet("background: transparent;")
        row.setFixedHeight(row_height)
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(_px(7, minimum=5))

        combo = FluentComboBox()
        combo.addItems(["Edge", "Ramp", "Hold"])
        combo.setCurrentText(_bus_mode_title(mode))
        combo.setToolTip(f"{row_info.key}: output mode")
        combo.setFixedSize(_bus_mode_combo_width(), row_height)
        self.bus_mode_combos[row_info.key] = combo
        layout.addWidget(combo, 0)

        text = (
            f"s{binding.position}"
            if binding is not None and binding.kind == "scan"
            else str(0 if number is None else number)
        )
        edit = FluentScanLineEdit(
            text,
            tooltip=(
                f"{row_info.label}: signed integer {lo}..{hi} (0 = 0 V); "
                "click the dot to cycle scan (sN) -> API (aN) -> off"
            ),
        )
        edit.set_numeric_validator("int", bottom=lo, top=hi)
        edit.setFixedHeight(row_height)
        edit.set_editable(mode != "hold")
        field = PulseFieldRef(FIELD_DAC, self.period_id, row_info.key)
        edit.scanClicked.connect(
            lambda key=row_info.key, ref=field: self.bindingCycleRequested.emit(ref)
        )
        if binding is not None:
            if binding.kind == "scan":
                edit.set_scan_bound(True, binding.position + 1)
                edit.setValidator(None)
                combo.setEnabled(False)
            else:
                edit.set_api_bound(True, binding.position + 1)
            edit.setToolTip(f"{edit.toolTip()}\nParameter: {binding.parameter_id}")
        self.bus_value_edits[row_info.key] = edit
        layout.addWidget(edit, 1)

        combo.currentTextChanged.connect(
            lambda _title, key=row_info.key: self._commit_analog(key)
        )
        edit.editingFinished.connect(
            lambda key=row_info.key: self._commit_analog(key)
        )
        return row

    def _commit_duration(self) -> None:
        if not self.unit_combo.isEnabled():
            return
        try:
            value = _number(self.duration_edit.text())
        except (TypeError, ValueError):
            return
        unit = self.unit_combo.currentText()
        if value == self._committed_duration and unit == self._committed_unit:
            return
        self.durationEdited.emit(self.period_id, value, unit)

    def _commit_name(self) -> None:
        name = self.name_edit.text()
        if name == self._committed_name:
            return
        self.periodNameEdited.emit(self.period_id, name)

    def _commit_analog(self, port: str) -> None:
        combo = self.bus_mode_combos[port]
        edit = self.bus_value_edits[port]
        mode = _bus_mode_value(combo.currentText())
        edit.set_editable(mode != "hold")
        if mode == "hold":
            self.analogEdited.emit(self.period_id, port, mode, None)
            return
        try:
            value = int(float(edit.text()))
        except (TypeError, ValueError):
            return
        self.analogEdited.emit(self.period_id, port, mode, value)

    def set_period_position(self, index: int, total: int) -> None:
        self.setTitle(f"Period {int(index) + 1}/{max(1, int(total))}")

    def set_port_label(self, port: str, label: str) -> None:
        check = self.checks.get(port)
        if check is not None:
            check.setText(label)
            check.setToolTip(f"{label}: high for this whole period")

    def reconcile(
        self,
        index: int,
        period,
        *,
        total_periods: int,
        rows: Sequence[_PortRow],
        digital_values: Mapping[str, bool],
        analog_values: Mapping[str, tuple[str, int | None]],
        bindings: Mapping[PulseFieldRef, _Binding],
    ) -> None:
        """Project scalar changes onto this stable card without recreating it."""

        if period.period_id != self.period_id:
            raise ValueError("cannot reconcile a PeriodCard to another period_id")
        projection_state = (
            int(index),
            int(total_periods),
            period,
            tuple(rows),
            tuple((row.key, bool(digital_values.get(row.key, False))) for row in rows),
            tuple(
                (row.key, analog_values.get(row.key, ("hold", 0))) for row in rows
            ),
            tuple(
                (field, binding)
                for field, binding in bindings.items()
                if field.period_id == self.period_id
            ),
        )
        if projection_state == self._projection_state:
            return
        self._reconcile_port_rows(
            rows,
            digital_values=digital_values,
            analog_values=analog_values,
            bindings=bindings,
        )
        self.set_period_position(index, total_periods)

        duration_ref = PulseFieldRef(FIELD_DURATION, self.period_id)
        duration_binding = bindings.get(duration_ref)
        duration_text = (
            f"s{duration_binding.position}"
            if duration_binding is not None and duration_binding.kind == "scan"
            else str(period.duration)
        )
        with signals_blocked(self.duration_edit, self.unit_combo, self.name_edit):
            self.duration_edit.set_scan_bound(False, None)
            self.duration_edit.set_api_bound(False, None)
            self.duration_edit.set_numeric_validator("float", bottom=0.0)
            _set_widget_text(self.duration_edit, duration_text)
            base_tooltip = "How long this period lasts, in the unit below"
            self.duration_edit.setToolTip(base_tooltip)
            if duration_binding is not None:
                if duration_binding.kind == "scan":
                    self.duration_edit.set_scan_bound(
                        True, duration_binding.position + 1
                    )
                    self.duration_edit.setValidator(None)
                else:
                    self.duration_edit.set_api_bound(
                        True, duration_binding.position + 1
                    )
                self.duration_edit.setToolTip(
                    f"{base_tooltip}\nParameter: {duration_binding.parameter_id}"
                )
            _set_duration_units(self.unit_combo, duration_binding, period.unit)
            _set_widget_text(self.name_edit, period.name or "")
        self._committed_duration = period.duration
        self._committed_unit = period.unit
        self._committed_name = str(period.name or "")

        rows_by_key = {row.key: row for row in rows}
        for key, check in self.checks.items():
            row = rows_by_key[key]
            with signals_blocked(check):
                _set_widget_text(check, row.label)
                check.setToolTip(f"{row.label}: high for this whole period")
                check.setChecked(bool(digital_values.get(key, False)))
        for key, combo in self.bus_mode_combos.items():
            row = rows_by_key[key]
            mode, number = analog_values.get(key, ("hold", 0))
            binding = bindings.get(PulseFieldRef(FIELD_DAC, self.period_id, key))
            edit = self.bus_value_edits[key]
            lo, hi = row.signed_range or (0, 0)
            text = (
                f"s{binding.position}"
                if binding is not None and binding.kind == "scan"
                else str(0 if number is None else number)
            )
            with signals_blocked(combo, edit):
                combo.setEnabled(True)
                mode_title = _bus_mode_title(mode)
                if combo.currentText() != mode_title:
                    combo.setCurrentText(mode_title)
                edit.set_scan_bound(False, None)
                edit.set_api_bound(False, None)
                edit.set_numeric_validator("int", bottom=lo, top=hi)
                _set_widget_text(edit, text)
                edit.set_editable(mode != "hold")
                tooltip = (
                    f"{row.label}: signed integer {lo}..{hi} (0 = 0 V); "
                    "click the dot to cycle scan (sN) -> API (aN) -> off"
                )
                if binding is not None:
                    if binding.kind == "scan":
                        edit.set_scan_bound(True, binding.position + 1)
                        edit.setValidator(None)
                        combo.setEnabled(False)
                    else:
                        edit.set_api_bound(True, binding.position + 1)
                    tooltip = f"{tooltip}\nParameter: {binding.parameter_id}"
                edit.setToolTip(tooltip)
        self._projection_state = projection_state

    def set_visible_ports(self, ports: frozenset[str]) -> None:
        for key, widget in self.port_rows.items():
            widget.setVisible(key in ports)


class ChannelNamesPanel(FluentGroupBox):
    """Operator channel catalog projected from the active target manifest."""

    documentNameEdited = QtCore.pyqtSignal(str)
    portLabelEdited = QtCore.pyqtSignal(str, str)

    def __init__(
        self,
        document_name: str,
        rows: Sequence[_PortRow],
        *,
        total_text: str,
        period_count: int,
        visible_text: str,
        parent=None,
    ) -> None:
        super().__init__("Port Catalog", parent)
        self.rows: tuple[_PortRow, ...] = ()
        self.port_labels: dict[str, FluentLineEdit] = {}
        self.hardware_labels: dict[str, FluentLabel] = {}
        self.row_widgets: dict[str, QtWidgets.QWidget] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        label_w = _channel_label_width()
        edit_w = _channel_name_edit_width()
        panel_w = label_w + edit_w + _px(5) + _px(20)
        self.setMinimumWidth(panel_w)
        self.setMaximumWidth(panel_w)

        layout = QtWidgets.QVBoxLayout(self)
        self._column = layout
        row_top, row_gap = _row_region_vmetrics()
        layout.setContentsMargins(_px(8), row_top, _px(8), _px(8))
        layout.setSpacing(row_gap)

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.name_edit = FluentLineEdit(document_name)
        self.name_edit.setPlaceholderText("pulse name")
        self.name_edit.editingFinished.connect(
            lambda: self.documentNameEdited.emit(self.name_edit.text())
        )
        self.top_labels["name"] = _add_labeled_widget(
            top_layout, "Name:", self.name_edit
        )

        self.total_label = FluentLineEdit(total_text)
        self.total_label.setEnabled(False)
        self.top_labels["total"] = _add_labeled_widget(
            top_layout, "Total:", self.total_label
        )
        self.periods_label = FluentLineEdit(str(period_count))
        self.periods_label.setEnabled(False)
        self.top_labels["periods"] = _add_labeled_widget(
            top_layout, "Periods:", self.periods_label
        )
        self.visible_label = FluentLineEdit(visible_text)
        self.visible_label.setEnabled(False)
        self.top_labels["visible"] = _add_labeled_widget(
            top_layout, "Visible:", self.visible_label
        )
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addStretch()
        self._label_width = label_w
        self._edit_width = edit_w
        self.reconcile_rows(rows)

    def _create_row(self, row_info: _PortRow) -> QtWidgets.QWidget:
        row_height = _channel_row_height()
        row_widget = QtWidgets.QWidget()
        row_widget.setStyleSheet("background: transparent;")
        row_widget.setFixedHeight(row_height)
        row = QtWidgets.QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(_px(5, minimum=3))
        hardware = FluentLabel("")
        hardware.setAlignment(QtCore.Qt.AlignCenter)
        hardware.setFixedSize(self._label_width, row_height)
        label = FluentLineEdit("")
        label.setToolTip(
            "The channel's display NAME (editable). Renaming changes only the "
            "human label; the PulseTarget topology and ABI stay fixed."
        )
        label.setFixedWidth(self._edit_width)
        label.setFixedHeight(row_height)
        label.editingFinished.connect(
            lambda key=row_info.key, editor=label: self.portLabelEdited.emit(
                key, editor.text()
            )
        )
        self.hardware_labels[row_info.key] = hardware
        self.port_labels[row_info.key] = label
        self.row_widgets[row_info.key] = row_widget
        row.addWidget(hardware)
        row.addWidget(label, 1)
        return row_widget

    def reconcile_rows(self, rows: Sequence[_PortRow]) -> None:
        rows = tuple(rows)
        if rows == self.rows:
            return
        desired = tuple(row.key for row in rows)
        if len(desired) != len(set(desired)):
            raise ValueError("channel rows must have unique keys")
        topology_changed = desired != tuple(row.key for row in self.rows)
        desired_set = set(desired)
        for key in tuple(self.row_widgets):
            if key in desired_set:
                continue
            widget = self.row_widgets.pop(key)
            self._column.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            self.hardware_labels.pop(key)
            self.port_labels.pop(key)
        for row in rows:
            if row.key not in self.row_widgets:
                self._create_row(row)
                topology_changed = True
        if topology_changed:
            _reconcile_widget_order(
                self._column,
                tuple(row.key for row in self.rows),
                desired,
                self.row_widgets,
                offset=1,
            )
        self.rows = rows
        for row in rows:
            endpoint_text = (
                row.endpoints[0]
                if len(row.endpoints) == 1
                else f"{row.endpoints[0]}…{row.endpoints[-1]}"
            )
            hardware = self.hardware_labels[row.key]
            _set_widget_text(hardware, endpoint_text)
            hardware.setToolTip(
                f"endpoint {row.endpoints[0]}"
                if row.kind == PORT_DIGITAL
                else f"{row.width}-bit DAC endpoints: {', '.join(row.endpoints)}"
            )
            with signals_blocked(self.port_labels[row.key]):
                _set_widget_text(self.port_labels[row.key], row.label)

    def reconcile_header(
        self,
        *,
        document_name: str,
        total_text: str,
        period_count: int,
        visible_text: str,
    ) -> None:
        with signals_blocked(
            self.name_edit,
            self.total_label,
            self.periods_label,
            self.visible_label,
        ):
            _set_widget_text(self.name_edit, document_name)
            _set_widget_text(self.total_label, total_text)
            _set_widget_text(self.periods_label, period_count)
            _set_widget_text(self.visible_label, visible_text)

    def set_visible_ports(self, ports: frozenset[str]) -> None:
        for key, widget in self.row_widgets.items():
            widget.setVisible(key in ports)


class ChannelPanel(FluentGroupBox):
    """Formal Delay / Scan column; every edit is emitted as a current intent."""

    delayEdited = QtCore.pyqtSignal(str, object, str)
    bindingCycleRequested = QtCore.pyqtSignal(object)
    clearPortRequested = QtCore.pyqtSignal(str)
    scanArrayLoadRequested = QtCore.pyqtSignal()
    scanSourceEdited = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        rows: Sequence[_PortRow],
        *,
        time_step_ns: float,
        delays: Mapping[str, tuple[int | float, str]],
        bindings: Mapping[PulseFieldRef, _Binding],
        scan_parameter_count: int,
        scan_point_count: int,
        parent=None,
    ) -> None:
        super().__init__("Delay / Scan", parent)
        self.rows: tuple[_PortRow, ...] = ()
        self.delay_edits: dict[str, FluentScanLineEdit] = {}
        self.delay_units: dict[str, FluentComboBox] = {}
        self.clear_buttons: dict[str, FluentButton] = {}
        self.channel_labels: dict[str, ElidedLabel] = {}
        self.row_widgets: dict[str, QtWidgets.QWidget] = {}
        self.top_labels: dict[str, FluentLabel] = {}
        label_w = _channel_label_width()
        delay_w = _px(70, minimum=60)
        unit_w = _time_unit_width()
        hide_w = _hide_button_width()
        gap = _px(4, minimum=3)
        content_w = label_w + delay_w + unit_w + hide_w + gap * 3 + _px(16)
        self.setMinimumWidth(content_w)
        self.setMaximumWidth(content_w)

        layout = QtWidgets.QVBoxLayout(self)
        self._column = layout
        row_top, row_gap = _row_region_vmetrics()
        layout.setContentsMargins(_px(8), row_top, _px(8), _px(8))
        layout.setSpacing(row_gap)

        top = QtWidgets.QWidget()
        top.setStyleSheet("background: transparent;")
        top.setFixedHeight(_panel_top_height())
        top_layout = QtWidgets.QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(_px(6, minimum=4))

        self.step_display = FluentLineEdit(_format_clock_text(time_step_ns))
        self.step_display.setEnabled(False)
        self.step_display.setToolTip(
            "FPGA clock (fixed by hardware). One tick = 1 / clock; all times snap "
            "to a whole tick."
        )
        self.top_labels["step"] = _add_labeled_widget(
            top_layout, "Clock:", self.step_display
        )

        self.scan_summary = FluentLineEdit("")
        self.scan_summary.setEnabled(False)
        self.scan_summary.setToolTip(
            "Active scan slots and uploaded scan points. Click a dot to bind a field."
        )
        self.top_labels["scan"] = _add_labeled_widget(
            top_layout, "Scan:", self.scan_summary
        )
        self.set_scan_summary(scan_parameter_count, scan_point_count)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(_px(5, minimum=3))
        self.load_button = FluentButton("Load Array", color=ACCENT)
        self.load_button.setFixedHeight(_row_height())
        self.load_button.setToolTip(
            "Load a scan-table file (.npy/.csv/.txt/.json): one row per scan point, "
            "one column per slot."
        )
        self.load_button.clicked.connect(self.scanArrayLoadRequested)
        self.scan_file_label = ElidedLabel("(no file)", mode=QtCore.Qt.ElideLeft)
        self.scan_file_label.setFixedHeight(_row_height())
        self.scan_file_label.setToolTip(
            "The scan-table file currently loaded (used when the toggle below is on)."
        )
        btn_row.addWidget(self.load_button)
        btn_row.addWidget(self.scan_file_label, 1)
        top_layout.addLayout(btn_row)

        self.scan_source_toggle = FluentSwitch("Use loaded file")
        self.scan_source_toggle.setFixedHeight(_row_height())
        self.scan_source_toggle.setToolTip(
            "Off: use the scan table generated in the Scan tab (Run).\n"
            "On: use the array loaded with Load Array."
        )
        self.scan_source_toggle.toggled.connect(self.scanSourceEdited)
        top_layout.addWidget(self.scan_source_toggle)
        top_layout.addStretch()
        layout.addWidget(top)
        layout.addStretch()
        self._label_width = label_w
        self._delay_width = delay_w
        self._unit_width = unit_w
        self._hide_width = hide_w
        self._row_gap = gap
        self._projection_state: tuple[object, ...] | None = None
        self.reconcile_rows(
            rows,
            time_step_ns=time_step_ns,
            delays=delays,
            bindings=bindings,
            scan_parameter_count=scan_parameter_count,
            scan_point_count=scan_point_count,
        )

    def _create_row(self, row_info: _PortRow) -> QtWidgets.QWidget:
        row_height = _channel_row_height()
        row_widget = QtWidgets.QWidget()
        row_widget.setStyleSheet("background: transparent;")
        row_widget.setFixedHeight(row_height)
        row = QtWidgets.QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(self._row_gap)
        label = ElidedLabel("")
        label.setFixedSize(self._label_width, row_height)
        delay_edit = FluentScanLineEdit("")
        field = PulseFieldRef(FIELD_DELAY, None, row_info.key)
        delay_edit.scanClicked.connect(
            lambda key=row_info.key, ref=field: self.bindingCycleRequested.emit(ref)
        )
        delay_edit.setFixedSize(self._delay_width, row_height)
        delay_edit.editingFinished.connect(
            lambda key=row_info.key: self._commit_delay(key)
        )
        unit = FluentComboBox()
        unit.addItems(DELAY_UNITS)
        unit.setFixedSize(self._unit_width, row_height)
        unit.currentTextChanged.connect(
            lambda _text, key=row_info.key: self._commit_delay(key)
        )
        clear_btn = FluentButton("X", color=ORANGE)
        clear_btn.setFixedSize(self._hide_width, row_height)
        clear_btn.setToolTip("Set this row fully off.")
        clear_btn.clicked.connect(
            lambda _checked=False, key=row_info.key: self.clearPortRequested.emit(key)
        )
        self.channel_labels[row_info.key] = label
        self.delay_edits[row_info.key] = delay_edit
        self.delay_units[row_info.key] = unit
        self.clear_buttons[row_info.key] = clear_btn
        self.row_widgets[row_info.key] = row_widget
        row.addWidget(label)
        row.addWidget(delay_edit, 1)
        row.addWidget(unit)
        row.addWidget(clear_btn)
        return row_widget

    def reconcile_rows(
        self,
        rows: Sequence[_PortRow],
        *,
        time_step_ns: float,
        delays: Mapping[str, tuple[int | float, str]],
        bindings: Mapping[PulseFieldRef, _Binding],
        scan_parameter_count: int,
        scan_point_count: int,
    ) -> None:
        rows = tuple(rows)
        projection_state = (
            rows,
            float(time_step_ns),
            tuple((row.key, delays.get(row.key, (0, "ns"))) for row in rows),
            tuple(
                (
                    row.key,
                    bindings.get(PulseFieldRef(FIELD_DELAY, None, row.key)),
                )
                for row in rows
            ),
            int(scan_parameter_count),
            int(scan_point_count),
        )
        if projection_state == self._projection_state:
            return
        desired = tuple(row.key for row in rows)
        if len(desired) != len(set(desired)):
            raise ValueError("channel rows must have unique keys")
        topology_changed = desired != tuple(row.key for row in self.rows)
        desired_set = set(desired)
        for key in tuple(self.row_widgets):
            if key in desired_set:
                continue
            widget = self.row_widgets.pop(key)
            self._column.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            self.channel_labels.pop(key)
            self.delay_edits.pop(key)
            self.delay_units.pop(key)
            self.clear_buttons.pop(key)
        for row in rows:
            if row.key not in self.row_widgets:
                self._create_row(row)
                topology_changed = True
        if topology_changed:
            _reconcile_widget_order(
                self._column,
                tuple(row.key for row in self.rows),
                desired,
                self.row_widgets,
                offset=1,
            )
        self.rows = rows
        with signals_blocked(self.step_display, self.scan_summary):
            _set_widget_text(self.step_display, _format_clock_text(time_step_ns))
            self.set_scan_summary(scan_parameter_count, scan_point_count)
        for row in rows:
            key = row.key
            label = self.channel_labels[key]
            _set_widget_text(label, row.label)
            label.setToolTip(key)
            value, unit_text = delays.get(key, (0, "ns"))
            edit = self.delay_edits[key]
            unit = self.delay_units[key]
            binding = bindings.get(PulseFieldRef(FIELD_DELAY, None, key))
            with signals_blocked(edit, unit):
                edit.set_scan_bound(False, None)
                edit.set_api_bound(False, None)
                edit.set_numeric_validator("float")
                _set_widget_text(edit, value)
                if binding is not None:
                    edit.set_api_bound(True, binding.position + 1)
                    edit.setToolTip(f"API parameter: {binding.parameter_id}")
                elif row.kind == PORT_DAC:
                    edit.setToolTip(
                        "Physical DAC-bus output delay (may be negative): the whole "
                        "bus value shifts by d, out[t] = in[t-d]."
                    )
                else:
                    edit.setToolTip(
                        "Physical per-channel output delay (may be negative): the "
                        "whole channel waveform shifts by d, out[t] = in[t-d]."
                    )
                selected_unit = unit_text if unit_text in DELAY_UNITS else "ns"
                if unit.currentText() != selected_unit:
                    unit.setCurrentText(selected_unit)
        self._projection_state = projection_state

    def _commit_delay(self, port: str) -> None:
        edit = self.delay_edits[port]
        try:
            value = _number(edit.text())
        except (TypeError, ValueError):
            return
        semantic_value = None if float(value) == 0.0 else value
        self.delayEdited.emit(port, semantic_value, self.delay_units[port].currentText())

    def set_scan_summary(self, slots: int, points: int) -> None:
        if slots == 0:
            text = "no scan slots"
        else:
            text = f"{slots} slot{'s' if slots != 1 else ''} · {points} pt{'s' if points != 1 else ''}"
        _set_widget_text(self.scan_summary, text)

    def set_scan_source(self, *, use_loaded: bool, path: str) -> None:
        with signals_blocked(self.scan_source_toggle):
            checked = bool(use_loaded)
            if self.scan_source_toggle.isChecked() != checked:
                self.scan_source_toggle.setChecked(checked)
        display_path = str(path) if path else "(no file)"
        _set_widget_text(self.scan_file_label, display_path)

    def set_port_label(self, port: str, label: str) -> None:
        widget = self.channel_labels.get(port)
        if widget is not None:
            widget.setText(label)

    def set_visible_ports(self, ports: frozenset[str]) -> None:
        for key, widget in self.row_widgets.items():
            widget.setVisible(key in ports)


class RepeatBracket(FluentGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, kind: str, repeat_count: int = 2, parent=None) -> None:
        super().__init__("", parent)
        self.kind = kind
        self._committed_count = int(repeat_count)
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
            self.repeat_spin.editingFinished.connect(self._commit_count)
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

    def _commit_count(self) -> None:
        if self.repeat_spin is None:
            return
        if int(self.repeat_spin.value()) == self._committed_count:
            return
        self.changed.emit()

    def set_repeat_count(self, count: int) -> None:
        if self.repeat_spin is None:
            return
        normalized = int(count)
        with signals_blocked(self.repeat_spin):
            if int(self.repeat_spin.value()) != normalized:
                self.repeat_spin.setValue(normalized)
        self._committed_count = normalized


@dataclass(frozen=True)
class _DragItem:
    widget: QtWidgets.QWidget
    item_type: str
    period_id: str | None = None


class PulseDragContainer(QtWidgets.QWidget):
    """Formal drag surface; proposed moves are emitted, never applied locally."""

    periodClicked = QtCore.pyqtSignal(str)
    gapClicked = QtCore.pyqtSignal(int)
    moveRequested = QtCore.pyqtSignal(str, object)
    repeatMoveRequested = QtCore.pyqtSignal(object, object, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.items: list[_DragItem] = []
        self.drag_start_pos = None
        self.dragging_index = None
        self.layout_main = QtWidgets.QHBoxLayout(self)
        pad = _card_gutter()
        self.layout_main.setContentsMargins(pad, pad, pad, pad)
        self.layout_main.setSpacing(_px(5, minimum=3))
        self.layout_main.setAlignment(QtCore.Qt.AlignLeft)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        self.insert_indicator = QtWidgets.QFrame(self)
        self.insert_indicator.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.insert_indicator.setStyleSheet(f"background: {ACCENT};")
        self.insert_indicator.setFixedWidth(_px(3))
        self.insert_indicator.hide()
        self._selected_period_id: str | None = None
        self._selected_gap: int | None = None

    def reconcile_items(self, desired: Sequence[_DragItem]) -> None:
        """Insert, remove, or move only the structural items that changed."""

        desired = list(desired)
        current_identity = tuple(
            (id(item.widget), item.item_type, item.period_id) for item in self.items
        )
        desired_identity = tuple(
            (id(item.widget), item.item_type, item.period_id) for item in desired
        )
        if desired_identity == current_identity:
            return
        desired_ids = {id(item.widget) for item in desired}
        if len(desired_ids) != len(desired):
            raise ValueError("drag items must have unique widgets")

        current = list(self.items)
        for item in tuple(current):
            if id(item.widget) in desired_ids:
                continue
            self.layout_main.removeWidget(item.widget)
            current.remove(item)

        for index, wanted in enumerate(desired):
            current_index = next(
                (
                    position
                    for position, item in enumerate(current)
                    if item.widget is wanted.widget
                ),
                None,
            )
            if current_index is None:
                self.layout_main.insertWidget(index, wanted.widget)
                current.insert(index, wanted)
                continue
            current[current_index] = wanted
            if current_index == index:
                continue
            self.layout_main.removeWidget(wanted.widget)
            self.layout_main.insertWidget(index, wanted.widget)
            current.insert(index, current.pop(current_index))

        self.items = desired
        self.insert_indicator.hide()
        self.update_period_titles()

    def pulse_cards(self) -> list[PeriodCard]:
        return [
            item.widget
            for item in self.items
            if item.item_type == "pulse" and isinstance(item.widget, PeriodCard)
        ]

    def period_ids(self) -> tuple[str, ...]:
        return tuple(
            item.period_id
            for item in self.items
            if item.item_type == "pulse" and item.period_id is not None
        )

    def update_period_titles(self) -> None:
        cards = self.pulse_cards()
        for index, card in enumerate(cards):
            card.set_period_position(index, len(cards))

    def mousePressEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_start_pos = event.pos()
            self.dragging_index = self._index_at(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.drag_start_pos is None or self.dragging_index is None:
            return super().mouseMoveEvent(event)
        if (
            event.buttons() & QtCore.Qt.LeftButton
            and (event.pos() - self.drag_start_pos).manhattanLength()
            > QtWidgets.QApplication.startDragDistance()
        ):
            self._start_drag(self.dragging_index)
            self.drag_start_pos = None
            self.dragging_index = None
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == QtCore.Qt.LeftButton and self.drag_start_pos is not None:
            index = self._index_at(event.pos())
            if index is not None and self.items[index].item_type == "pulse":
                period_id = self.items[index].period_id
                if period_id is not None:
                    self.periodClicked.emit(period_id)
            else:
                self.gapClicked.emit(
                    self._period_pos_of_insert(self._insert_pos(event.pos()))
                )
        self.drag_start_pos = None
        self.dragging_index = None
        super().mouseReleaseEvent(event)

    def _period_pos_of_insert(self, items_pos: int) -> int:
        return sum(
            1 for item in self.items[:items_pos] if item.item_type == "pulse"
        )

    def _start_drag(self, index: int) -> None:
        drag = QtGui.QDrag(self)
        mime = QtCore.QMimeData()
        mime.setData("application/x-zlc-pulse-card", str(index).encode("utf-8"))
        drag.setMimeData(mime)
        widget = self.items[index].widget
        pixmap = widget.grab()
        if pixmap.height() > _px(180):
            pixmap = pixmap.scaledToHeight(_px(180), QtCore.Qt.SmoothTransformation)
        ghost = QtGui.QPixmap(pixmap.size())
        ghost.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(ghost)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        drag.setPixmap(ghost)
        drag.setHotSpot(QtCore.QPoint(ghost.width() // 2, _px(16)))
        if hasattr(widget, "set_outline"):
            widget.set_outline("#808080")
        drag.exec_(QtCore.Qt.MoveAction)
        if hasattr(widget, "set_outline"):
            widget.set_outline(None)
        self.insert_indicator.hide()

    def _ancestor_scroll_area(self) -> QtWidgets.QAbstractScrollArea | None:
        widget = self.parentWidget()
        while widget is not None and not isinstance(
            widget, QtWidgets.QAbstractScrollArea
        ):
            widget = widget.parentWidget()
        return widget

    def _autoscroll_during_drag(self, pos) -> None:
        area = self._ancestor_scroll_area()
        if area is None:
            return
        viewport = area.viewport()
        viewport_pos = viewport.mapFromGlobal(self.mapToGlobal(pos))
        margin = _px(56, minimum=40)
        bar = area.horizontalScrollBar()
        if viewport_pos.x() > viewport.width() - margin:
            bar.setValue(bar.value() + _px(28, minimum=20))
        elif viewport_pos.x() < margin:
            bar.setValue(bar.value() - _px(28, minimum=20))

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            event.acceptProposedAction()
            self._show_insert_indicator(self._insert_pos(event.pos()))
            self._autoscroll_during_drag(event.pos())
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        if not event.mimeData().hasFormat("application/x-zlc-pulse-card"):
            return super().dropEvent(event)
        old_index = int(
            bytes(event.mimeData().data("application/x-zlc-pulse-card")).decode(
                "utf-8"
            )
        )
        insert_pos = self._insert_pos(event.pos())
        proposed = self.items[:]
        dragged = proposed.pop(old_index)
        if insert_pos > old_index:
            insert_pos -= 1
        proposed.insert(insert_pos, dragged)
        if not self._bracket_ok(proposed):
            event.ignore()
            self.insert_indicator.hide()
            return
        event.acceptProposedAction()
        self.insert_indicator.hide()
        if dragged.item_type == "pulse" and dragged.period_id is not None:
            before = self._next_period_after(proposed, insert_pos)
            if tuple(self._period_ids(proposed)) != self.period_ids():
                self.moveRequested.emit(dragged.period_id, before)
            return
        start, end, count = self._repeat_from_items(proposed)
        if start is not None and end is not None:
            self.repeatMoveRequested.emit(start, end, count)

    def dragLeaveEvent(self, event) -> None:
        self.insert_indicator.hide()
        super().dragLeaveEvent(event)

    @staticmethod
    def _period_ids(items: Sequence[_DragItem]) -> list[str]:
        return [
            item.period_id
            for item in items
            if item.item_type == "pulse" and item.period_id is not None
        ]

    @staticmethod
    def _next_period_after(items: Sequence[_DragItem], position: int) -> str | None:
        for item in items[position + 1 :]:
            if item.item_type == "pulse":
                return item.period_id
        return None

    @staticmethod
    def _repeat_from_items(
        items: Sequence[_DragItem],
    ) -> tuple[str | None, str | None, int]:
        start_pos = next(
            (i for i, item in enumerate(items) if item.item_type == "bracket_start"),
            None,
        )
        end_pos = next(
            (i for i, item in enumerate(items) if item.item_type == "bracket_end"),
            None,
        )
        if start_pos is None or end_pos is None:
            return None, None, 2
        inside = [
            item
            for item in items[start_pos + 1 : end_pos]
            if item.item_type == "pulse" and item.period_id is not None
        ]
        if not inside:
            return None, None, 2
        end_widget = items[end_pos].widget
        spin = getattr(end_widget, "repeat_spin", None)
        count = int(spin.value()) if spin is not None else 2
        return inside[0].period_id, inside[-1].period_id, count

    def _index_at(self, pos) -> int | None:
        for index, item in enumerate(self.items):
            if item.widget.geometry().contains(pos):
                return index
        return None

    def _insert_pos(self, pos) -> int:
        x = pos.x()
        for index, item in enumerate(self.items):
            geometry = item.widget.geometry()
            if x < geometry.x() + geometry.width() // 2:
                return index
        return len(self.items)

    def _indicator_x_for_items_pos(self, items_pos: int) -> int:
        spacing = self.layout_main.spacing()
        if not self.items:
            return _card_gutter()
        if items_pos >= len(self.items):
            geometry = self.items[-1].widget.geometry()
            return geometry.x() + geometry.width() + max(1, spacing // 2) - _px(1)
        geometry = self.items[items_pos].widget.geometry()
        return max(0, geometry.x() - max(1, spacing // 2) - _px(1))

    def _show_insert_indicator(self, items_pos: int) -> None:
        pad = _card_gutter()
        self.insert_indicator.setGeometry(
            self._indicator_x_for_items_pos(items_pos),
            pad,
            self.insert_indicator.width(),
            max(_px(40), self.height() - 2 * pad),
        )
        self.insert_indicator.show()
        self.insert_indicator.raise_()

    def show_selection(
        self, *, period_id: str | None = None, gap: int | None = None
    ) -> None:
        self._selected_period_id = period_id
        self._selected_gap = gap
        for item in self.items:
            if item.item_type == "pulse" and hasattr(item.widget, "set_outline"):
                item.widget.set_outline(
                    ACCENT if period_id is not None and item.period_id == period_id else None
                )
        if gap is not None:
            self._show_insert_indicator(self._items_pos_of_period_gap(gap))
        else:
            self.insert_indicator.hide()

    def _items_pos_of_period_gap(self, period_pos: int) -> int:
        seen = 0
        for index, item in enumerate(self.items):
            if item.item_type == "pulse":
                if seen == period_pos:
                    return index
                seen += 1
        return len(self.items)

    @staticmethod
    def _bracket_ok(items: Sequence[_DragItem]) -> bool:
        start = next(
            (i for i, item in enumerate(items) if item.item_type == "bracket_start"),
            None,
        )
        end = next(
            (i for i, item in enumerate(items) if item.item_type == "bracket_end"),
            None,
        )
        if start is None or end is None:
            return True
        return end >= start + 3


class PulseScheduleView(QtWidgets.QWidget):
    """The final pure-Qt Edit page for one immutable current PulseDocument."""

    documentNameEdited = QtCore.pyqtSignal(str)
    portLabelEdited = QtCore.pyqtSignal(str, str)
    periodNameEdited = QtCore.pyqtSignal(str, str)
    durationEdited = QtCore.pyqtSignal(str, object, str)
    digitalEdited = QtCore.pyqtSignal(str, str, bool)
    analogEdited = QtCore.pyqtSignal(str, str, str, object)
    delayEdited = QtCore.pyqtSignal(str, object, str)
    bindingCycleRequested = QtCore.pyqtSignal(object)
    insertPeriodRequested = QtCore.pyqtSignal(object)
    movePeriodRequested = QtCore.pyqtSignal(str, object)
    removePeriodRequested = QtCore.pyqtSignal(str)
    repeatEdited = QtCore.pyqtSignal(object, object, int)
    visiblePortsEdited = QtCore.pyqtSignal(object)
    clearPortRequested = QtCore.pyqtSignal(str)

    clearAllRequested = QtCore.pyqtSignal()
    runRequested = QtCore.pyqtSignal()
    stopRequested = QtCore.pyqtSignal()
    syncRequested = QtCore.pyqtSignal()
    saveRequested = QtCore.pyqtSignal()
    loadRequested = QtCore.pyqtSignal()
    connectionRequested = QtCore.pyqtSignal(str, str)
    scanArrayLoadRequested = QtCore.pyqtSignal()
    scanSourceEdited = QtCore.pyqtSignal(bool)
    leftPanelsCollapsed = QtCore.pyqtSignal(bool)
    feedbackRequested = QtCore.pyqtSignal(str)

    def __init__(
        self,
        document: PulseDocument,
        manifest: PulseTargetManifest | None = None,
        parent=None,
        *,
        display_visible_ports: tuple[str, ...] | None = None,
        document_generation: int = 0,
        revision: int = 0,
    ) -> None:
        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if manifest is None:
            manifest = pulse_target_manifest_from_lanes(document.target)
        elif not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest or None")
        super().__init__(parent)
        self._document_generation = -1
        self._revision = -1
        self._document_identity = 0
        self._manifest_fingerprint = ""
        self._period_ids: tuple[str, ...] = ()
        self._visible_ports: tuple[str, ...] = ()
        self._programmable_ports: tuple[str, ...] = ()
        self._all_rows: tuple[_PortRow, ...] = ()
        self._active_ports: frozenset[str] = frozenset()
        self._repeat: tuple[str, str, int] | None = None
        self._selected_period_id: str | None = None
        self._selected_gap: int | None = None
        self._left_panels_collapsed = False
        self._scan_source_loaded = False
        self._scan_source_path = ""
        self._build_ui()
        self.set_document(
            document,
            manifest,
            display_visible_ports=display_visible_ports,
            document_generation=document_generation,
            revision=revision,
        )

    @property
    def document_generation(self) -> int:
        return self._document_generation

    @property
    def revision(self) -> int:
        return self._revision

    def _build_ui(self) -> None:
        self.setStyleSheet("background: transparent;")
        edit_layout = QtWidgets.QVBoxLayout(self)
        tab_margin = _px(8, minimum=5)
        edit_layout.setContentsMargins(tab_margin, tab_margin, tab_margin, tab_margin)
        edit_layout.setSpacing(_px(8, minimum=5))

        self.dataset_scroll = FluentScrollArea()
        self.dataset_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # The channel form changes height when ports are shown/hidden.  Reserve
        # the Fluent scrollbar gutter permanently so the first overflow never
        # steals width and shifts every aligned column in the Edit surface.
        self.dataset_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.dataset_body = QtWidgets.QWidget()
        dataset = QtWidgets.QHBoxLayout(self.dataset_body)
        gutter = _card_gutter()
        dataset.setContentsMargins(0, 0, 0, 0)
        dataset.setSpacing(0)

        self.names_panel_holder = QtWidgets.QWidget()
        self.names_panel_layout = QtWidgets.QVBoxLayout(self.names_panel_holder)
        self.names_panel_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        self.names_panel_layout.setSpacing(0)
        dataset.addWidget(self.names_panel_holder)

        self.channel_panel_holder = QtWidgets.QWidget()
        self.channel_panel_layout = QtWidgets.QVBoxLayout(self.channel_panel_holder)
        self.channel_panel_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        self.channel_panel_layout.setSpacing(0)
        dataset.addWidget(self.channel_panel_holder)

        self.left_panel_stub_holder = QtWidgets.QWidget()
        stub_holder_layout = QtWidgets.QVBoxLayout(self.left_panel_stub_holder)
        stub_holder_layout.setContentsMargins(gutter, gutter, gutter, gutter)
        stub_holder_layout.setSpacing(0)
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
        stub_holder_layout.addWidget(self.left_panel_stub)
        self.left_panel_stub_holder.hide()
        dataset.addWidget(self.left_panel_stub_holder)

        self.scroll = FluentScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.drag_container = PulseDragContainer()
        self.drag_container.periodClicked.connect(self._on_period_card_clicked)
        self.drag_container.gapClicked.connect(self._on_period_gap_clicked)
        self.drag_container.moveRequested.connect(self.movePeriodRequested)
        self.drag_container.repeatMoveRequested.connect(self.repeatEdited)
        self.scroll.setWidget(self.drag_container)
        dataset.addWidget(self.scroll, 1)
        self.dataset_scroll.setWidget(self.dataset_body)
        edit_layout.addWidget(self.dataset_scroll, 1)

        self.timeline_hbar = QtWidgets.QScrollBar(QtCore.Qt.Horizontal)
        self.timeline_hbar.setStyleSheet(fluent_scrollbar_stylesheet("QScrollBar"))
        self.timeline_hbar.setFixedHeight(fluent_scrollbar_thickness())
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

        self.button_frame = FluentFrame(bordered=False)
        self.button_frame.setObjectName("zlcPulseButtonBar")
        self.button_frame.setStyleSheet(
            "QFrame#zlcPulseButtonBar { background: transparent; border: none; }"
        )
        bar = QtWidgets.QHBoxLayout(self.button_frame)
        spacing = _card_gutter()
        bar.setContentsMargins(spacing, spacing + _px(2), spacing, spacing)
        bar.setSpacing(_px(10, minimum=8))
        control_height = _px(30, minimum=26)

        control_area = FluentGroupBox("Control")
        control_col = QtWidgets.QVBoxLayout(control_area)
        control_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        control_col.setSpacing(_px(4, minimum=3))
        button_layout = QtWidgets.QGridLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(_px(6, minimum=4))
        self.safe_button = self._control_button("Stop Pulse", self.stopRequested, RED)
        self.fire_button = self._control_button("On Pulse*", self.runRequested, GREEN)
        self.fire_button.setToolTip(
            "Apply the editor's pulse to the device and run it.\n"
            "* = pressing would apply something new (edits not yet applied,\n"
            "or the device is not currently running)."
        )
        self.remove_button = self._control_button(
            "Remove", self._request_remove_period, ORANGE
        )
        self.add_button = self._control_button(
            "Add Period", self._request_add_period, ACCENT
        )
        self.bracket_button = self._control_button(
            "Add Bracket", self._request_toggle_bracket, ACCENT
        )
        self.save_button = self._control_button("Save*", self.saveRequested, YELLOW)
        self.load_button = self._control_button("Load", self.loadRequested, ORANGE)
        self.sync_button = self._control_button("Sync", self.syncRequested, ORANGE)
        self.sync_button.setToolTip(
            "Load the pulse currently applied on the sequencer into the editor\n"
            "(use after changing the device from a notebook / raw API)."
        )
        self.collapse_button = self._control_button(
            "Collapse", self.toggle_left_panels, GREY
        )
        control_buttons = (
            self.fire_button,
            self.safe_button,
            self.sync_button,
            self.add_button,
            self.remove_button,
            self.bracket_button,
            self.save_button,
            self.load_button,
            self.collapse_button,
        )
        for button in control_buttons:
            button.setFixedHeight(control_height)
            button.setMinimumWidth(_px(74, minimum=62))
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
        for index, button in enumerate(control_buttons):
            button_layout.addWidget(button, index // 3, index % 3)
        for column in range(3):
            button_layout.setColumnStretch(column, 1)
        control_col.addStretch(1)
        control_col.addLayout(button_layout)
        control_col.addStretch(1)
        bar.addWidget(control_area, 1)

        conn_area = FluentGroupBox("Connection")
        conn_area.setFixedWidth(_px(252, minimum=212))
        conn_col = QtWidgets.QVBoxLayout(conn_area)
        conn_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        conn_col.setSpacing(_px(4, minimum=3))
        self.conn_target_combo = FluentComboBox()
        self.conn_target_combo.setFixedHeight(control_height)
        self.conn_target_combo.addItem("Virtual (sim)", "virtual")
        self.conn_target_combo.addItem("Remote server", "remote")
        self.conn_target_combo.addItem("Offline (edit only)", "offline")
        self.conn_target_combo.setToolTip(
            "Virtual: a local in-memory sequencer (edit + fire in simulation).\n"
            "Remote: connect to a running FPGA sequencer server (host:port).\n"
            "Offline: edit only, no backend calls."
        )
        self.conn_target_combo.currentIndexChanged.connect(
            self._on_conn_target_changed
        )
        conn_col.addWidget(self.conn_target_combo)
        conn_row = QtWidgets.QHBoxLayout()
        conn_row.setContentsMargins(0, 0, 0, 0)
        conn_row.setSpacing(_px(6, minimum=4))
        self.conn_addr_edit = FluentLineEdit("127.0.0.1:18861")
        self.conn_addr_edit.setFixedHeight(control_height)
        self.conn_addr_edit.setToolTip("Remote sequencer server address as host:port.")
        self.conn_connect_button = FluentButton("Connect", color=ACCENT)
        self.conn_connect_button.setFixedHeight(control_height)
        self.conn_connect_button.setMinimumWidth(_px(64, minimum=54))
        self.conn_connect_button.setToolTip("Apply the selected connection.")
        self.conn_connect_button.clicked.connect(self._request_connection)
        conn_row.addWidget(self.conn_addr_edit, 1)
        conn_row.addWidget(self.conn_connect_button)
        conn_col.addLayout(conn_row)
        self.conn_status = FluentLineEdit("")
        self.conn_status.setEnabled(False)
        self.conn_status.setFixedHeight(_row_height())
        conn_col.addWidget(self.conn_status)
        conn_col.addStretch(1)
        bar.addWidget(conn_area)

        view_area = FluentGroupBox("Ports")
        view_area.setFixedWidth(_px(286, minimum=246))
        view_col = QtWidgets.QVBoxLayout(view_area)
        view_col.setContentsMargins(_px(8), _px(2), _px(8), _px(6))
        view_col.setSpacing(_px(4, minimum=3))
        self.add_channel_combo = FluentComboBox()
        self.add_channel_combo.setFixedHeight(control_height)
        self.add_channel_combo.setToolTip("Pick a hidden sequencer port to show.")
        view_col.addWidget(self.add_channel_combo)
        view_btn_row = QtWidgets.QHBoxLayout()
        view_btn_row.setContentsMargins(0, 0, 0, 0)
        view_btn_row.setSpacing(_px(6, minimum=4))
        self.add_channel_button = FluentButton("Add", color=ACCENT)
        self.add_channel_button.setToolTip("Add the selected hidden port to the table.")
        self.add_channel_button.clicked.connect(self._request_add_port)
        self.hide_off_button = FluentButton("Hide Off", color=ORANGE)
        self.hide_off_button.setToolTip("Hide ports that are inactive in every period.")
        self.hide_off_button.clicked.connect(self._request_hide_off_ports)
        self.show_all_button = FluentButton("Show All", color=ACCENT)
        self.show_all_button.setToolTip("Show every programmable sequencer port.")
        self.show_all_button.clicked.connect(self._request_show_all_ports)
        for button in (
            self.add_channel_button,
            self.hide_off_button,
            self.show_all_button,
        ):
            button.setFixedHeight(control_height)
            button.setMinimumWidth(_px(56, minimum=48))
            button.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
            view_btn_row.addWidget(button, 1)
        view_col.addLayout(view_btn_row)
        self.visible_label = FluentLineEdit("")
        self.visible_label.setEnabled(False)
        self.visible_label.setFixedHeight(_row_height())
        view_col.addWidget(self.visible_label)
        view_col.addStretch(1)
        bar.addWidget(view_area)

        self.button_frame.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Maximum
        )
        edit_layout.addWidget(self.button_frame)
        self._on_conn_target_changed()

    @staticmethod
    def _control_button(text: str, slot, color: str) -> FluentButton:
        button = FluentButton(text, color=color)
        if hasattr(slot, "emit"):
            button.clicked.connect(lambda _checked=False, signal=slot: signal.emit())
        else:
            button.clicked.connect(slot)
        return button

    def _dynamic_focus_targets(self) -> dict[tuple[str, ...], QtWidgets.QWidget]:
        """Map rebuilt controls to stable semantic identities.

        Only controls owned by the document projection appear here.  Header and
        footer controls are not rebuilt and therefore keep their native Qt focus.
        """

        targets: dict[tuple[str, ...], QtWidgets.QWidget] = {}

        def add(token: tuple[str, ...], widget: QtWidgets.QWidget | None) -> None:
            if widget is not None:
                targets[token] = widget

        names = getattr(self, "names_panel", None)
        if names is not None:
            add(("document-name",), names.name_edit)
            for port, widget in names.port_labels.items():
                add(("port-label", port), widget)

        channels = getattr(self, "channel_panel", None)
        if channels is not None:
            add(("scan-load",), channels.load_button)
            add(("scan-source",), channels.scan_source_toggle)
            for port, widget in channels.delay_edits.items():
                add(("delay", port), widget)
                add(("delay-dot", port), getattr(widget, "dot", None))
            for port, widget in channels.delay_units.items():
                add(("delay-unit", port), widget)
            for port, widget in channels.clear_buttons.items():
                add(("clear-port", port), widget)

        drag_container = getattr(self, "drag_container", None)
        if drag_container is None:
            return targets
        for card in drag_container.pulse_cards():
            period_id = card.period_id
            add(("duration", period_id), card.duration_edit)
            add(("duration-dot", period_id), card.duration_dot)
            add(("duration-unit", period_id), card.unit_combo)
            add(("period-name", period_id), card.name_edit)
            for port, widget in card.checks.items():
                add(("digital", period_id, port), widget)
            for port, widget in card.bus_mode_combos.items():
                add(("dac-mode", period_id, port), widget)
            for port, widget in card.bus_value_edits.items():
                add(("dac-value", period_id, port), widget)
                add(("dac-dot", period_id, port), getattr(widget, "dot", None))
        for item in drag_container.items:
            if item.item_type == "bracket_end":
                add(("repeat-count",), getattr(item.widget, "repeat_spin", None))
        return targets

    def _capture_interaction(self) -> _InteractionSnapshot:
        focus_widget = QtWidgets.QApplication.focusWidget()
        focus_snapshot = None
        if focus_widget is not None:
            targets = self._dynamic_focus_targets()
            target = next(
                (
                    (token, widget)
                    for token, widget in targets.items()
                    if widget is focus_widget
                ),
                None,
            )
            if target is None:
                target = next(
                    (
                        (token, widget)
                        for token, widget in targets.items()
                        if widget.isAncestorOf(focus_widget)
                    ),
                    None,
                )
            if target is not None:
                token, widget = target
                if isinstance(widget, QtWidgets.QLineEdit):
                    selection_start = widget.selectionStart()
                    focus_snapshot = _FocusSnapshot(
                        token,
                        widget.cursorPosition(),
                        selection_start,
                        (
                            len(widget.selectedText())
                            if selection_start >= 0
                            else 0
                        ),
                    )
                else:
                    focus_snapshot = _FocusSnapshot(token)
        return _InteractionSnapshot(
            self._selected_period_id,
            self._selected_gap,
            self.scroll.horizontalScrollBar().value(),
            self.dataset_scroll.verticalScrollBar().value(),
            focus_snapshot,
        )

    def _restore_interaction(
        self,
        snapshot: _InteractionSnapshot,
        *,
        version: tuple[int, int],
    ) -> None:
        if version != (self._document_generation, self._revision):
            return
        if snapshot.focus is not None:
            widget = self._dynamic_focus_targets().get(snapshot.focus.token)
            if widget is not None and widget.isEnabled():
                widget.setFocus(QtCore.Qt.OtherFocusReason)
                if isinstance(widget, QtWidgets.QLineEdit):
                    cursor = snapshot.focus.cursor_position
                    if cursor is not None:
                        widget.setCursorPosition(min(cursor, len(widget.text())))
                    if snapshot.focus.selection_start >= 0:
                        start = min(
                            snapshot.focus.selection_start,
                            len(widget.text()),
                        )
                        widget.setSelection(
                            start,
                            min(
                                snapshot.focus.selection_length,
                                max(0, len(widget.text()) - start),
                            ),
                        )
        # A structural add/remove/reorder may ask a scroll area to reveal the
        # focused child.  Restore the saved bars last so the operator keeps the
        # same viewport; scalar reconciliation normally retains the child too.
        self.scroll.horizontalScrollBar().setValue(snapshot.horizontal_scroll)
        self.dataset_scroll.verticalScrollBar().setValue(snapshot.vertical_scroll)

    def _connect_period_card(self, card: PeriodCard) -> None:
        card.periodNameEdited.connect(self.periodNameEdited)
        card.durationEdited.connect(self.durationEdited)
        card.digitalEdited.connect(self.digitalEdited)
        card.analogEdited.connect(self.analogEdited)
        card.bindingCycleRequested.connect(self.bindingCycleRequested)

    def _reconcile_period_items(
        self,
        document: PulseDocument,
        rows: Sequence[_PortRow],
        *,
        digital_values: Mapping[tuple[str, str], bool],
        analog_values: Mapping[tuple[str, str], tuple[str, int | None]],
        bindings: Mapping[PulseFieldRef, _Binding],
    ) -> None:
        """Incrementally add/remove/move cards and repeat markers by stable id."""

        existing_cards = {
            card.period_id: card for card in self.drag_container.pulse_cards()
        }
        cards: list[PeriodCard] = []
        total = len(document.periods)
        for index, period in enumerate(document.periods):
            card = existing_cards.pop(period.period_id, None)
            if card is None:
                card = PeriodCard(
                    index,
                    period,
                    total_periods=total,
                    rows=rows,
                    digital_values={
                        row.key: digital_values.get(
                            (period.period_id, row.key), False
                        )
                        for row in rows
                        if row.kind == PORT_DIGITAL
                    },
                    analog_values={
                        row.key: analog_values.get(
                            (period.period_id, row.key), ("hold", 0)
                        )
                        for row in rows
                        if row.kind == PORT_DAC
                    },
                    bindings=bindings,
                )
                self._connect_period_card(card)
            else:
                card.reconcile(
                    index,
                    period,
                    total_periods=total,
                    rows=rows,
                    digital_values={
                        row.key: digital_values.get(
                            (period.period_id, row.key), False
                        )
                        for row in rows
                        if row.kind == PORT_DIGITAL
                    },
                    analog_values={
                        row.key: analog_values.get(
                            (period.period_id, row.key), ("hold", 0)
                        )
                        for row in rows
                        if row.kind == PORT_DAC
                    },
                    bindings=bindings,
                )
            cards.append(card)

        existing_start = next(
            (
                item.widget
                for item in self.drag_container.items
                if item.item_type == "bracket_start"
            ),
            None,
        )
        existing_end = next(
            (
                item.widget
                for item in self.drag_container.items
                if item.item_type == "bracket_end"
            ),
            None,
        )
        desired_items = [
            _DragItem(card, "pulse", card.period_id) for card in cards
        ]
        repeat = document.repeat
        if repeat is not None:
            if existing_start is None:
                existing_start = RepeatBracket("start")
            if existing_end is None:
                existing_end = RepeatBracket("end", repeat.count)
                existing_end.changed.connect(self._request_repeat_count)
            existing_end.set_repeat_count(repeat.count)
            period_ids = tuple(period.period_id for period in document.periods)
            start_index = period_ids.index(repeat.start_period_id)
            end_index = period_ids.index(repeat.end_period_id)
            desired_items.insert(
                start_index,
                _DragItem(existing_start, "bracket_start"),
            )
            desired_items.insert(
                end_index + 2,
                _DragItem(existing_end, "bracket_end"),
            )
            self.bracket_button.setText("Del Bracket")
        else:
            self.bracket_button.setText("Add Bracket")

        desired_widget_ids = {id(item.widget) for item in desired_items}
        for item in self.drag_container.items:
            if id(item.widget) in desired_widget_ids:
                continue
            self.drag_container.layout_main.removeWidget(item.widget)
            item.widget.setParent(None)
            item.widget.deleteLater()
        self.drag_container.reconcile_items(desired_items)

    def set_document(
        self,
        document: PulseDocument,
        manifest: PulseTargetManifest | None = None,
        *,
        display_visible_ports: tuple[str, ...] | None = None,
        document_generation: int,
        revision: int,
    ) -> bool:
        """Project a frozen current document; stale generation/revision pairs are ignored.

        No document reference is retained.  Widgets receive scalar values and the
        view retains only ids/labels needed to formulate later intents.
        """

        if not isinstance(document, PulseDocument):
            raise TypeError("document must be PulseDocument")
        if manifest is None:
            manifest = pulse_target_manifest_from_lanes(document.target)
        elif not isinstance(manifest, PulseTargetManifest):
            raise TypeError("manifest must be PulseTargetManifest or None")
        if display_visible_ports is None:
            display_visible_ports = document.visible_ports
        else:
            display_visible_ports = tuple(str(key) for key in display_visible_ports)
        unknown_display = set(display_visible_ports) - set(
            manifest.available_port_keys
        )
        if unknown_display:
            raise ValueError(
                "display_visible_ports names ports outside the target manifest: "
                f"{sorted(unknown_display)}"
            )
        if (
            isinstance(document_generation, bool)
            or not isinstance(document_generation, int)
            or document_generation < 0
        ):
            raise ValueError("document_generation must be a non-negative integer")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        document_identity = id(document)
        manifest_fingerprint = manifest.fingerprint
        incoming_version = (document_generation, revision)
        current_version = (self._document_generation, self._revision)
        if incoming_version < current_version:
            return False
        if incoming_version == current_version:
            if document_identity != self._document_identity:
                raise ValueError("one editor revision cannot identify two PulseDocuments")
            if (
                manifest_fingerprint == self._manifest_fingerprint
                and display_visible_ports == self._visible_ports
            ):
                return False

        rows = _port_rows(
            document,
            manifest,
            visible_only=True,
            visible_ports=display_visible_ports,
        )
        all_rows = _port_rows(
            document,
            manifest,
            visible_only=False,
            visible_ports=display_visible_ports,
        )
        bindings = _field_bindings(document)
        digital_values = _digital_values(document)
        analog_values, analog_active = _analog_values(document)
        delays = {item.port: (item.value, item.unit) for item in document.delays}
        period_ids = tuple(period.period_id for period in document.periods)
        programmable = tuple(row.key for row in all_rows)
        visible = tuple(row.key for row in rows)

        active = {
            port
            for (period_id, port), high in digital_values.items()
            if high and period_id in period_ids
        }
        active.update(port for port, is_active in analog_active.items() if is_active)

        base_ns_by_id = {
            period.period_id: float(period.duration) * TIME_UNIT_TO_NS[period.unit]
            for period in document.periods
        }
        total_ns = sum(base_ns_by_id.values())
        pulse_count = _expanded_pulse_count(document)
        repeat = None
        if document.repeat is not None:
            repeat = (
                document.repeat.start_period_id,
                document.repeat.end_period_id,
                document.repeat.count,
            )
            start = period_ids.index(document.repeat.start_period_id)
            end = period_ids.index(document.repeat.end_period_id)
            inner_ids = period_ids[start : end + 1]
            total_ns += (document.repeat.count - 1) * sum(
                base_ns_by_id[item] for item in inner_ids
            )
        row_structure = tuple((row.key, row.kind) for row in all_rows)
        previous_row_structure = tuple(
            (row.key, row.kind) for row in self._all_rows
        )
        repeat_bounds = None if repeat is None else repeat[:2]
        previous_repeat_bounds = (
            None if self._repeat is None else self._repeat[:2]
        )
        structure_changed = bool(
            period_ids != self._period_ids
            or row_structure != previous_row_structure
            or repeat_bounds != previous_repeat_bounds
        )
        visible_changed = visible != self._visible_ports
        layout_changed = structure_changed or visible_changed
        interaction = self._capture_interaction() if layout_changed else None
        repeat_text = _repeat_summary_text(document)
        visible_set = set(visible)
        hidden_active = tuple(
            row.label
            for row in all_rows
            if row.key in active and row.key not in visible_set
        )
        boundary_active = _restart_high_labels(document)

        point_count = 0 if document.scan_table is None else len(document.scan_table.rows)
        summary_parts = [
            f"{len(visible)}/{len(programmable)} ports visible",
            f"{len(period_ids)} periods",
            f"step {document.time_step_ns:g} ns",
            f"{total_ns:.3g} ns",
            f"{pulse_count} pulses",
            repeat_text,
        ]
        if document.scan_parameters:
            summary_parts.append(
                f"scan {len(document.scan_parameters)} slots × {point_count} pts"
            )
        if hidden_active:
            summary_parts.append(f"hidden active: {', '.join(hidden_active)}")
        if boundary_active:
            labels = boundary_active[:4]
            suffix = (
                ""
                if len(boundary_active) <= 4
                else f", +{len(boundary_active) - 4}"
            )
            summary_parts.append(
                "table restart high every "
                f"{_summary_time_text(total_ns)}: {', '.join(labels)}{suffix}"
            )

        if layout_changed:
            self.setUpdatesEnabled(False)
        try:
            if not hasattr(self, "names_panel"):
                self.names_panel = ChannelNamesPanel(
                    document.name,
                    all_rows,
                    total_text=_summary_time_text(total_ns),
                    period_count=len(period_ids),
                    visible_text=f"{len(visible)}/{len(programmable)}",
                )
                self.names_panel.documentNameEdited.connect(self.documentNameEdited)
                self.names_panel.portLabelEdited.connect(self._port_label_edited)
                self.names_panel_layout.addWidget(self.names_panel)
                self.name_edit = self.names_panel.name_edit

                self.channel_panel = ChannelPanel(
                    all_rows,
                    time_step_ns=document.time_step_ns,
                    delays=delays,
                    bindings=bindings,
                    scan_parameter_count=len(document.scan_parameters),
                    scan_point_count=point_count,
                )
                self.channel_panel.delayEdited.connect(self.delayEdited)
                self.channel_panel.bindingCycleRequested.connect(
                    self.bindingCycleRequested
                )
                self.channel_panel.clearPortRequested.connect(self.clearPortRequested)
                self.channel_panel.scanArrayLoadRequested.connect(
                    self.scanArrayLoadRequested
                )
                self.channel_panel.scanSourceEdited.connect(self.scanSourceEdited)
                self.channel_panel_layout.addWidget(self.channel_panel)
            else:
                self.names_panel.reconcile_rows(all_rows)
                self.channel_panel.reconcile_rows(
                    all_rows,
                    time_step_ns=document.time_step_ns,
                    delays=delays,
                    bindings=bindings,
                    scan_parameter_count=len(document.scan_parameters),
                    scan_point_count=point_count,
                )

            self.names_panel.reconcile_header(
                document_name=document.name,
                total_text=_summary_time_text(total_ns),
                period_count=len(period_ids),
                visible_text=f"{len(visible)}/{len(programmable)}",
            )
            self.names_panel.total_label.setToolTip(
                f"{total_ns:.9g} ns total (one frame)"
            )
            self.channel_panel.set_scan_source(
                use_loaded=self._scan_source_loaded, path=self._scan_source_path
            )
            self._reconcile_period_items(
                document,
                all_rows,
                digital_values=digital_values,
                analog_values=analog_values,
                bindings=bindings,
            )

            self._period_ids = period_ids
            self._programmable_ports = programmable
            self._all_rows = tuple(all_rows)
            self._visible_ports = visible
            self._active_ports = frozenset(active)
            self._repeat = repeat
            selected_period_id = (
                self._selected_period_id
                if interaction is None
                else interaction.selected_period_id
            )
            selected_gap = (
                self._selected_gap if interaction is None else interaction.selected_gap
            )
            self._selected_period_id = (
                selected_period_id if selected_period_id in period_ids else None
            )
            self._selected_gap = (
                selected_gap
                if self._selected_period_id is None
                and selected_gap is not None
                and 0 <= selected_gap <= len(period_ids)
                else None
            )
            if layout_changed:
                self.drag_container.show_selection(
                    period_id=self._selected_period_id,
                    gap=self._selected_gap,
                )
            if row_structure != previous_row_structure or visible_changed:
                self._apply_visible_port_projection(frozenset(visible))
            self.remove_button.setEnabled(len(period_ids) > 1)
            self._refresh_hidden_combo(all_rows)
            self.visible_label.setText(
                f"Visible {len(visible)}/{len(programmable)} ports | "
                f"Hidden {len(programmable) - len(visible)}"
            )
            self._summary_text = " | ".join(summary_parts)
            self._document_generation = document_generation
            self._revision = revision
            self._document_identity = document_identity
            self._manifest_fingerprint = manifest_fingerprint
            if layout_changed:
                self._sync_dataset_geometry()
            if interaction is not None:
                self._restore_interaction(interaction, version=incoming_version)
                QtCore.QTimer.singleShot(
                    0,
                    lambda saved=interaction, version=incoming_version: self._restore_interaction(
                        saved,
                        version=version,
                    ),
                )
        finally:
            if layout_changed:
                self.setUpdatesEnabled(True)
                self.update()
        return True

    def _apply_visible_port_projection(self, visible: frozenset[str]) -> None:
        """Reveal existing aligned rows without rebuilding any editor widget."""

        self.names_panel.set_visible_ports(visible)
        self.channel_panel.set_visible_ports(visible)
        for card in self.drag_container.pulse_cards():
            card.set_visible_ports(visible)
        for widget in (
            self.names_panel,
            self.channel_panel,
            *self.drag_container.pulse_cards(),
        ):
            layout = widget.layout()
            if layout is not None:
                layout.invalidate()
                layout.activate()
            widget.updateGeometry()

    def period_cards(self) -> tuple[PeriodCard, ...]:
        return tuple(self.drag_container.pulse_cards())

    def summary_text(self) -> str:
        return self._summary_text

    def set_scan_source(self, *, use_loaded: bool, path: str) -> None:
        self._scan_source_loaded = bool(use_loaded)
        self._scan_source_path = str(path)
        if hasattr(self, "channel_panel"):
            self.channel_panel.set_scan_source(
                use_loaded=self._scan_source_loaded, path=self._scan_source_path
            )

    def set_scan_workspace_busy(self, busy: bool) -> None:
        """Mirror worker admission without changing the frozen scan controls."""

        if hasattr(self, "channel_panel"):
            self.channel_panel.load_button.setEnabled(not bool(busy))
            self.channel_panel.scan_source_toggle.setEnabled(not bool(busy))

    def set_connection_state(
        self,
        mode: str,
        *,
        endpoint: str = "",
        status: str = "",
    ) -> None:
        index = self.conn_target_combo.findData(str(mode))
        if index < 0:
            raise ValueError("connection mode must be virtual, remote, or offline")
        with signals_blocked(self.conn_target_combo):
            self.conn_target_combo.setCurrentIndex(index)
        if endpoint:
            with signals_blocked(self.conn_addr_edit):
                self.conn_addr_edit.setText(str(endpoint))
        display_status = str(status).strip()
        if not display_status:
            if mode == "remote" and endpoint:
                display_status = str(endpoint)
            else:
                display_status = self.conn_target_combo.currentText()
        self.conn_status.setText(display_status)
        self._on_conn_target_changed()

    def set_control_state(
        self,
        *,
        running: bool,
        synchronized: bool,
        file_dirty: bool,
        file_busy: bool = False,
        run_busy: bool = False,
    ) -> None:
        # Busy rejection belongs to the controller.  The formal Edit page keeps
        # this control row visually stable instead of greying individual buttons.
        _ = file_busy, run_busy
        self.fire_button.setText(
            "On Pulse" if running and synchronized else "On Pulse*"
        )
        self.save_button.setText("Save*" if file_dirty else "Save")
        self.save_button.set_color(YELLOW if file_dirty else ACCENT)

    def _port_label_edited(self, port: str, label: str) -> None:
        visible_label = str(label).strip() or str(port)
        if hasattr(self, "channel_panel"):
            self.channel_panel.set_port_label(port, visible_label)
        for card in self.drag_container.pulse_cards():
            card.set_port_label(port, visible_label)
        self.portLabelEdited.emit(port, str(label))

    def _on_period_card_clicked(self, period_id: str) -> None:
        self._selected_period_id = (
            None if self._selected_period_id == period_id else period_id
        )
        self._selected_gap = None
        self.drag_container.show_selection(
            period_id=self._selected_period_id, gap=None
        )

    def _on_period_gap_clicked(self, position: int) -> None:
        self._selected_gap = None if self._selected_gap == position else position
        self._selected_period_id = None
        self.drag_container.show_selection(
            period_id=None, gap=self._selected_gap
        )

    def _request_add_period(self) -> None:
        before = None
        if self._selected_gap is not None:
            if 0 <= self._selected_gap < len(self._period_ids):
                before = self._period_ids[self._selected_gap]
        elif self._selected_period_id is not None:
            position = self._period_ids.index(self._selected_period_id) + 1
            if position < len(self._period_ids):
                before = self._period_ids[position]
        self.insertPeriodRequested.emit(before)

    def _request_remove_period(self) -> None:
        if len(self._period_ids) <= 1:
            return
        if self._selected_period_id is not None:
            target = self._selected_period_id
        elif self._selected_gap is not None:
            position = min(self._selected_gap, len(self._period_ids) - 1)
            target = self._period_ids[position]
        else:
            target = self._period_ids[-1]
        self.removePeriodRequested.emit(target)

    def _request_toggle_bracket(self) -> None:
        if self._repeat is not None:
            self.repeatEdited.emit(None, None, 0)
            return
        if len(self._period_ids) < 2:
            self.feedbackRequested.emit("Repeat needs at least two periods.")
            return
        self.repeatEdited.emit(self._period_ids[0], self._period_ids[-1], 2)

    def _request_repeat_count(self) -> None:
        if self._repeat is None:
            return
        count = self._repeat[2]
        for item in self.drag_container.items:
            if item.item_type != "bracket_end":
                continue
            spin = getattr(item.widget, "repeat_spin", None)
            if spin is not None:
                count = int(spin.value())
            break
        self.repeatEdited.emit(self._repeat[0], self._repeat[1], count)

    def _request_add_port(self) -> None:
        data = self.add_channel_combo.currentData()
        port = str(data if data is not None else "")
        if not port:
            return
        requested = set(self._visible_ports)
        requested.add(port)
        self.visiblePortsEdited.emit(
            tuple(item for item in self._programmable_ports if item in requested)
        )

    def _request_hide_off_ports(self) -> None:
        keepers = [
            port for port in self._visible_ports if port in self._active_ports
        ]
        minimum = min(4, len(self._programmable_ports))
        for port in self._programmable_ports:
            if len(keepers) >= minimum:
                break
            if port not in keepers:
                keepers.append(port)
        keep = set(keepers)
        self.visiblePortsEdited.emit(
            tuple(port for port in self._programmable_ports if port in keep)
        )

    def _request_show_all_ports(self) -> None:
        self.visiblePortsEdited.emit(self._programmable_ports)

    def _refresh_hidden_combo(self, rows: Sequence[_PortRow]) -> None:
        visible = set(self._visible_ports)
        desired: list[tuple[str, str]] = []
        for row in rows:
            if row.key in visible:
                continue
            display = (
                f"{row.label}  ({row.width} pins)"
                if row.kind == PORT_DAC
                else row.label
            )
            desired.append((row.key, display))

        selected = self.add_channel_combo.currentData()
        with signals_blocked(self.add_channel_combo):
            for index, (key, display) in enumerate(desired):
                if (
                    index < self.add_channel_combo.count()
                    and self.add_channel_combo.itemData(index) == key
                ):
                    if self.add_channel_combo.itemText(index) != display:
                        self.add_channel_combo.setItemText(index, display)
                    continue
                previous = self.add_channel_combo.findData(key)
                if previous >= 0:
                    self.add_channel_combo.removeItem(previous)
                self.add_channel_combo.insertItem(index, display, key)
            while self.add_channel_combo.count() > len(desired):
                self.add_channel_combo.removeItem(
                    self.add_channel_combo.count() - 1
                )
            if selected is not None:
                selected_index = self.add_channel_combo.findData(selected)
                if selected_index >= 0:
                    self.add_channel_combo.setCurrentIndex(selected_index)
        has_hidden = self.add_channel_combo.count() > 0
        self.add_channel_combo.setEnabled(has_hidden)
        self.add_channel_button.setEnabled(has_hidden)

    def _on_conn_target_changed(self) -> None:
        self.conn_addr_edit.setEnabled(
            self.conn_target_combo.currentData() == "remote"
        )

    def _request_connection(self) -> None:
        mode = str(self.conn_target_combo.currentData())
        endpoint = self.conn_addr_edit.text().strip() if mode == "remote" else ""
        self.connectionRequested.emit(mode, endpoint)

    def toggle_left_panels(self) -> None:
        if self._left_panels_collapsed:
            self.show_left_panels()
        else:
            self.hide_left_panels()

    def hide_left_panels(self) -> None:
        self._left_panels_collapsed = True
        self.names_panel_holder.hide()
        self.channel_panel_holder.hide()
        self.left_panel_stub_holder.show()
        self.collapse_button.setText("Show Left")
        self._sync_dataset_geometry()
        self.leftPanelsCollapsed.emit(True)

    def show_left_panels(self) -> None:
        self._left_panels_collapsed = False
        self.left_panel_stub_holder.hide()
        self.names_panel_holder.show()
        self.channel_panel_holder.show()
        self.collapse_button.setText("Collapse")
        self._sync_dataset_geometry()
        self.leftPanelsCollapsed.emit(False)

    def _drag_container_width(self) -> int:
        widths = []
        for item in self.drag_container.items:
            widget = item.widget
            maximum = widget.maximumWidth()
            width = (
                maximum
                if 0 < maximum < QtWidgets.QWIDGETSIZE_MAX
                else widget.sizeHint().width()
            )
            widths.append(max(width, widget.minimumWidth(), widget.width()))
        if not widths:
            return 0
        spacing = max(0, self.drag_container.layout_main.spacing())
        margins = self.drag_container.layout_main.contentsMargins()
        return (
            sum(widths)
            + spacing * (len(widths) - 1)
            + margins.left()
            + margins.right()
        )

    def _sync_dataset_geometry(self) -> None:
        if not hasattr(self, "names_panel"):
            return

        def vertical_margins(layout: QtWidgets.QLayout | None) -> int:
            if layout is None:
                return 0
            margins = layout.contentsMargins()
            return margins.top() + margins.bottom()

        card_hints = [
            item.widget.sizeHint().height() for item in self.drag_container.items
        ]
        drag_height = (
            (max(card_hints) if card_hints else 0)
            + vertical_margins(self.drag_container.layout_main)
        )
        names_height = (
            0
            if self.names_panel_holder.isHidden()
            else self.names_panel.sizeHint().height()
            + vertical_margins(self.names_panel_layout)
        )
        channel_height = (
            0
            if self.channel_panel_holder.isHidden()
            else self.channel_panel.sizeHint().height()
            + vertical_margins(self.channel_panel_layout)
        )
        stub_height = (
            0
            if self.left_panel_stub_holder.isHidden()
            else self.left_panel_stub.sizeHint().height()
        )
        content_height = max(
            names_height, channel_height, stub_height, drag_height
        ) + _px(2, minimum=1)
        for widget in (
            self.names_panel_holder,
            self.channel_panel_holder,
            self.left_panel_stub_holder,
            self.scroll,
            self.drag_container,
        ):
            widget.setMinimumHeight(content_height)
        self.dataset_body.setMinimumHeight(
            content_height + vertical_margins(self.dataset_body.layout())
        )
        self.drag_container.setFixedSize(
            self._drag_container_width(), content_height
        )
        self._sync_timeline_scrollbar()

    def _sync_timeline_scrollbar(self, *_args) -> None:
        source = self.scroll.horizontalScrollBar()
        with signals_blocked(self.timeline_hbar):
            self.timeline_hbar.setRange(source.minimum(), source.maximum())
            self.timeline_hbar.setPageStep(source.pageStep())
            self.timeline_hbar.setSingleStep(max(1, source.singleStep()))
            self.timeline_hbar.setValue(source.value())
        self.timeline_hbar.setVisible(source.maximum() > source.minimum())
        left_width = 0
        for widget in (
            self.names_panel_holder,
            self.channel_panel_holder,
            self.left_panel_stub_holder,
        ):
            if widget.isHidden():
                continue
            left_width += widget.width() or widget.sizeHint().width()
        body_layout = self.dataset_body.layout()
        if body_layout is not None:
            left_width += body_layout.contentsMargins().left()
        self.timeline_hbar_spacer.setFixedWidth(max(0, left_width))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._sync_timeline_scrollbar)


__all__ = [
    "ChannelNamesPanel",
    "ChannelPanel",
    "PeriodCard",
    "PulseDragContainer",
    "PulseScheduleView",
    "RepeatBracket",
]
