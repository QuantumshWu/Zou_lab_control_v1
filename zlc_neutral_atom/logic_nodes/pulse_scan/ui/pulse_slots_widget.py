"""The PulseScan-owned pulse-template slot editor.

The template path is a document boundary, but the controls inside this widget are
not disposable projections of that document.  API and program-column rows are
owned by their stable ``(program_id, slot_id)`` keys and reconciled in place.  The
program editor and sweep selector are constructed once and live for the lifetime
of the widget, so a slot insertion or move cannot steal a code cursor/selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_pulse.scan_template import (
    SWEEP_API_SLOT,
    SWEEP_SCAN_SLOT,
    scan_table_template,
)
from zlc_frontend.qt_widgets import (
    FluentButton,
    FluentCodeEdit,
    FluentComboBox,
    FluentLabel,
    FluentLineEdit,
    FluentSectionLabel,
    FluentSettingRow,
    scaled_px,
    setting_label_width,
    signals_blocked,
    GREY,
)

__all__ = ["PulseSlotsWidget"]


def slot_label(kind: str, target: str, *, base_1: bool = True) -> str:
    """Present one already-bound pulse field without owning pulse state."""

    target = str(target)
    offset = 1 if base_1 else 0
    if kind == "duration":
        try:
            return f"Period {int(target) + offset} duration"
        except ValueError:
            return f"Period {target} duration"
    if kind == "dac":
        bus, separator, period = target.partition("@")
        if separator:
            try:
                return f"{bus} (Period {int(period) + offset})"
            except ValueError:
                return f"{bus} (Period {period})"
        return f"{bus} DAC"
    if kind == "delay":
        return f"{target} delay"
    return target


def _is_number(value) -> bool:
    """True when ``value`` is a finite numeric field, excluding booleans."""

    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass
class _ApiRow:
    slot_id: str
    host: FluentSettingRow
    edit: FluentLineEdit
    baseline: str
    dirty: bool = False


class PulseSlotsWidget(QtWidgets.QWidget):
    """Structured editor for the two PulseScan execution strategies.

    ``reconcile`` consumes one committed template description.  Rows that retain
    the same program and slot identities retain their Qt objects; changed row
    metadata is written in place, order changes only move the existing row in its
    layout, and only removed rows are destroyed.  The code editor is never part of
    that lifecycle.

    A saved task override is deferred until the matching ``program_id`` arrives,
    because a workspace can be loaded before its template rows are known.
    """

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._scan_slot_kind = SWEEP_SCAN_SLOT
        self._api_slot_kind = SWEEP_API_SLOT
        self._program_id = ""
        self._sweep_kind = ""
        self._available = {SWEEP_SCAN_SLOT: False, SWEEP_API_SLOT: False}
        self._program_buffers = {SWEEP_SCAN_SLOT: "", SWEEP_API_SLOT: ""}
        self._program_initialized = {SWEEP_SCAN_SLOT: False, SWEEP_API_SLOT: False}
        self._program_baselines = {SWEEP_SCAN_SLOT: "", SWEEP_API_SLOT: ""}
        self._program_dirty = {SWEEP_SCAN_SLOT: False, SWEEP_API_SLOT: False}
        self._specs: dict[str, list] = {SWEEP_SCAN_SLOT: [], SWEEP_API_SLOT: []}

        self._api_rows: dict[tuple[str, str], _ApiRow] = {}
        self._api_order: list[tuple[str, str]] = []
        self._column_rows: dict[str, dict[tuple[str, str], FluentLabel]] = {
            SWEEP_SCAN_SLOT: {},
            SWEEP_API_SLOT: {},
        }
        self._column_order: dict[str, list[tuple[str, str]]] = {
            SWEEP_SCAN_SLOT: [],
            SWEEP_API_SLOT: [],
        }

        self._pending_program_id = ""
        self._pending_api: dict[str, str] = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""

        self._box = QtWidgets.QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(scaled_px(6, minimum=4))
        self._build_api_surface()
        self._build_selector_surface()
        self._build_program_surface()
        self._present_program()

    # ----------------------------------------------------------- stable surfaces
    def _build_api_surface(self) -> None:
        self._api_box = QtWidgets.QVBoxLayout()
        self._api_box.setContentsMargins(0, 0, 0, 0)
        self._api_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._api_box)
        self._api_header = FluentSectionLabel("API parameters")
        self._api_empty = FluentLabel("(this template has no API parameter)", self)
        self._api_empty.setWordWrap(True)
        self._api_empty.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self._api_box.addWidget(self._api_header)
        self._api_box.addWidget(self._api_empty)

    def _build_selector_surface(self) -> None:
        self._selector_box = QtWidgets.QVBoxLayout()
        self._selector_box.setContentsMargins(0, 0, 0, 0)
        self._selector_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._selector_box)

        combo = FluentComboBox()
        combo.addItem("Scan slots (hardware table)", self._scan_slot_kind)
        combo.addItem("API slots (one pulse per point)", self._api_slot_kind)
        combo.setToolTip(
            "Scan slots upload one complete FPGA table. API slots resolve and submit one "
            "finite pulse per program row."
        )
        combo.currentIndexChanged.connect(self._on_sweep_changed)
        self._sweep_combo = combo
        self._selector_row = FluentSettingRow(
            "Sweep",
            combo,
            label_width=setting_label_width(["Sweep"], minimum=72),
        )
        self._selector_box.addWidget(self._selector_row)

    def _build_program_surface(self) -> None:
        self._program_box = QtWidgets.QVBoxLayout()
        self._program_box.setContentsMargins(0, 0, 0, 0)
        self._program_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._program_box)

        self._program_empty = FluentLabel(
            "(bind at least one scan slot or API slot in the Pulse GUI)", self
        )
        self._program_empty.setWordWrap(True)
        self._program_empty.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self._program_title = FluentSectionLabel("")
        self._columns_intro = FluentLabel(
            "Columns of scan_table (one row = one point, columns advance in lockstep):",
            self,
        )
        self._columns_intro.setWordWrap(True)
        self._columns_intro.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )

        self._column_hosts: dict[str, QtWidgets.QWidget] = {}
        self._column_boxes: dict[str, QtWidgets.QVBoxLayout] = {}
        for kind in (self._scan_slot_kind, self._api_slot_kind):
            host = QtWidgets.QWidget(self)
            host.setStyleSheet("background: transparent;")
            box = QtWidgets.QVBoxLayout(host)
            box.setContentsMargins(0, 0, 0, 0)
            box.setSpacing(scaled_px(2, minimum=1))
            self._column_hosts[kind] = host
            self._column_boxes[kind] = box

        self._template_host = QtWidgets.QWidget(self)
        self._template_host.setStyleSheet("background: transparent;")
        btn_row = QtWidgets.QHBoxLayout(self._template_host)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        btn_row.addWidget(FluentLabel("template:", self._template_host))
        for template in ("column_stack", "grid"):
            button = FluentButton(template, self._template_host, color=GREY)
            button.clicked.connect(
                lambda *_args, value=template: self._insert_template(value)
            )
            btn_row.addWidget(button, 0)
        btn_row.addStretch(1)

        # This editor is deliberately constructed exactly once.  Slot reconcile,
        # sweep changes, empty templates, and program generation changes only update
        # its text/visibility and can never replace its QObject identity.
        self._program_code = FluentCodeEdit("", self)
        self._program_code.setMinimumHeight(scaled_px(120, minimum=90))
        self._program_code.setToolTip(
            "Python assigning an (N_points x n_columns) array to scan_table. Values use each "
            "selected slot's native unit."
        )
        self._program_code.textChanged.connect(self._on_program_edited)

        self._program_box.addWidget(self._program_empty)
        self._program_box.addWidget(self._program_title)
        self._program_box.addWidget(self._columns_intro)
        self._program_box.addWidget(self._column_hosts[self._scan_slot_kind])
        self._program_box.addWidget(self._column_hosts[self._api_slot_kind])
        self._program_box.addWidget(self._template_host)
        self._program_box.addWidget(self._program_code)

    # --------------------------------------------------------------- reconcile
    @staticmethod
    def _normal_api_rows(rows) -> list[tuple[str, str, str, str, str, object]]:
        result = []
        seen: set[str] = set()
        for raw in rows:
            try:
                slot_id, coordinate, kind, target, unit, current = raw
            except (TypeError, ValueError) as exc:
                raise ValueError("an API row must have six fields") from exc
            slot_id = str(slot_id)
            if not slot_id or slot_id in seen:
                raise ValueError(f"API slot_id must be non-empty and unique, got {slot_id!r}")
            seen.add(slot_id)
            result.append(
                (slot_id, str(coordinate), str(kind), str(target), str(unit), current)
            )
        return result

    @staticmethod
    def _normal_scan_rows(rows) -> list[tuple[str, str, str, str, str]]:
        result = []
        seen: set[str] = set()
        for raw in rows:
            try:
                slot_id, kind, target, unit, stored_label = raw
            except (TypeError, ValueError) as exc:
                raise ValueError("a scan row must have five fields") from exc
            slot_id = str(slot_id)
            if not slot_id or slot_id in seen:
                raise ValueError(f"scan slot_id must be non-empty and unique, got {slot_id!r}")
            seen.add(slot_id)
            result.append(
                (slot_id, str(kind), str(target), str(unit), str(stored_label or ""))
            )
        return result

    def reconcile(
        self,
        api_rows,
        scan_rows,
        *,
        api_columns=(),
        scan_columns=(),
        hardware_program: str = "",
        program_id: str = "",
    ) -> None:
        """Apply one committed template description without rebuilding this tree."""

        api = self._normal_api_rows(api_rows)
        scan = self._normal_scan_rows(scan_rows)
        program_id = str(program_id or "")
        same_program = bool(program_id and program_id == self._program_id)
        restore_saved = bool(program_id and program_id == self._pending_program_id)

        self._stash_program()
        if not same_program:
            self._remove_all_slot_rows()
            self._program_buffers = {self._scan_slot_kind: "", self._api_slot_kind: ""}
            self._program_initialized = {
                self._scan_slot_kind: False,
                self._api_slot_kind: False,
            }
            self._program_baselines = {self._scan_slot_kind: "", self._api_slot_kind: ""}
            self._program_dirty = {self._scan_slot_kind: False, self._api_slot_kind: False}

        self._program_id = program_id
        self._specs[self._api_slot_kind] = list(api_columns)
        self._specs[self._scan_slot_kind] = list(scan_columns)
        scan_source = str(hardware_program or "")
        if scan and not scan_source.strip():
            scan_source = scan_table_template("column_stack", self._specs[self._scan_slot_kind])
        self._accept_program_source(self._scan_slot_kind, scan_source, available=bool(scan))
        api_source = scan_table_template("column_stack", self._specs[self._api_slot_kind]) \
            if api else ""
        self._accept_program_source(self._api_slot_kind, api_source, available=bool(api))

        self._reconcile_api_rows(api, restore_saved=restore_saved)
        api_legend = [
            (slot_id, coordinate, slot_label(kind, target), unit)
            for slot_id, coordinate, kind, target, unit, _current in api
        ]
        scan_legend = []
        for slot_id, kind, target, unit, stored_label in scan:
            display = stored_label or slot_label(kind, target)
            display_unit = "ns ticks" if kind == "duration" else (
                "integer code (LSB)" if kind == "dac" else unit
            )
            scan_legend.append((slot_id, slot_id, display, display_unit))
        self._reconcile_column_rows(self._api_slot_kind, api_legend)
        self._reconcile_column_rows(self._scan_slot_kind, scan_legend)

        self._available = {
            self._scan_slot_kind: bool(scan),
            self._api_slot_kind: bool(api),
        }
        default_kind = self._scan_slot_kind if scan else (
            self._api_slot_kind if api else ""
        )
        if restore_saved and self._available.get(self._pending_sweep_kind, False):
            self._sweep_kind = self._pending_sweep_kind
            self._program_buffers[self._sweep_kind] = self._pending_program
            self._program_initialized[self._sweep_kind] = True
            self._program_dirty[self._sweep_kind] = True
        elif not same_program or not self._available.get(self._sweep_kind, False):
            self._sweep_kind = default_kind

        self._ensure_program_buffer(self._sweep_kind)
        self._present_program()
        self._pending_program_id = ""
        self._pending_api = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self.changed.emit()

    def _remove_all_slot_rows(self) -> None:
        for row in self._api_rows.values():
            self._api_box.removeWidget(row.host)
            row.host.hide()
            row.host.deleteLater()
        self._api_rows.clear()
        self._api_order.clear()
        for kind, rows in self._column_rows.items():
            box = self._column_boxes[kind]
            for host in rows.values():
                box.removeWidget(host)
                host.hide()
                host.deleteLater()
            rows.clear()
            self._column_order[kind].clear()

    def _reconcile_api_rows(self, rows, *, restore_saved: bool) -> None:
        wanted = [(self._program_id, slot_id) for slot_id, *_rest in rows]
        wanted_set = set(wanted)
        for key in tuple(self._api_rows):
            if key in wanted_set:
                continue
            row = self._api_rows.pop(key)
            self._api_box.removeWidget(row.host)
            row.host.hide()
            row.host.deleteLater()

        labels = [slot_label(kind, target) for _sid, _coord, kind, target, _unit, _cur in rows]
        label_width = setting_label_width(labels or [""], minimum=72)
        for slot_id, _coordinate, kind, target, unit, current in rows:
            key = (self._program_id, slot_id)
            label = slot_label(kind, target)
            baseline = f"{float(current):g}"
            row = self._api_rows.get(key)
            if row is None:
                seed = self._pending_api[slot_id] if (
                    restore_saved and slot_id in self._pending_api
                ) else baseline
                edit = FluentLineEdit(seed, self)
                edit.setMinimumWidth(scaled_px(120, minimum=96))
                host = FluentSettingRow(label, edit, label_width=label_width)
                row = _ApiRow(
                    slot_id=slot_id,
                    host=host,
                    edit=edit,
                    baseline=baseline,
                )
                self._api_rows[key] = row
                edit.textEdited.connect(
                    lambda _text, row_key=key: self._on_api_edited(row_key)
                )
            else:
                row.host.set_label(label, width=label_width)
                if restore_saved and slot_id in self._pending_api:
                    self._write_api_text(row, self._pending_api[slot_id])
                elif not row.dirty and row.baseline != baseline:
                    self._write_api_text(row, baseline)
                row.baseline = baseline
            row.edit.setPlaceholderText(unit)
            row.edit.setToolTip(
                f"Resting value for {label} ({unit}). In an API-slot sweep, the program "
                "overrides this handle once per row."
            )

        for offset, key in enumerate(wanted, start=1):
            host = self._api_rows[key].host
            if self._api_box.indexOf(host) != offset:
                self._api_box.removeWidget(host)
                self._api_box.insertWidget(offset, host)
        self._api_box.removeWidget(self._api_empty)
        self._api_box.addWidget(self._api_empty)
        self._api_empty.setVisible(not wanted)
        self._api_order = wanted

    @staticmethod
    def _write_api_text(row: _ApiRow, text: str) -> None:
        if row.edit.text() != str(text):
            with signals_blocked(row.edit):
                row.edit.setText(str(text))
        row.dirty = False

    def _on_api_edited(self, key: tuple[str, str]) -> None:
        row = self._api_rows.get(key)
        if row is None:
            return
        row.dirty = True
        self.changed.emit()

    def _reconcile_column_rows(self, kind: str, rows) -> None:
        current = self._column_rows[kind]
        box = self._column_boxes[kind]
        wanted = [(self._program_id, slot_id) for slot_id, *_rest in rows]
        wanted_set = set(wanted)
        for key in tuple(current):
            if key in wanted_set:
                continue
            host = current.pop(key)
            box.removeWidget(host)
            host.hide()
            host.deleteLater()

        for slot_id, coordinate, display, unit in rows:
            key = (self._program_id, slot_id)
            text = f"{coordinate}: {display}  [{unit}]"
            host = current.get(key)
            if host is None:
                host = FluentLabel(text, self._column_hosts[kind])
                host.setStyleSheet(
                    f"color: {GREY}; background: transparent; border: none;"
                )
                current[key] = host
            elif host.text() != text:
                host.setText(text)

        for offset, key in enumerate(wanted):
            host = current[key]
            if box.indexOf(host) != offset:
                box.removeWidget(host)
                box.insertWidget(offset, host)
        self._column_order[kind] = wanted

    # ------------------------------------------------------------ program draft
    def _accept_program_source(self, kind: str, source: str, *, available: bool) -> None:
        """Update one clean buffer from its source without overwriting a local draft."""

        if not available:
            return
        source = str(source)
        if not self._program_initialized[kind]:
            self._program_buffers[kind] = source
            self._program_initialized[kind] = True
        elif not self._program_dirty[kind] and self._program_baselines[kind] != source:
            self._program_buffers[kind] = source
        self._program_baselines[kind] = source

    def _ensure_program_buffer(self, kind: str) -> None:
        if not kind or self._program_initialized.get(kind, False):
            return
        source = scan_table_template(kind="column_stack", columns=self._specs[kind])
        self._program_buffers[kind] = source
        self._program_baselines[kind] = source
        self._program_initialized[kind] = True

    def _stash_program(self) -> None:
        if self._sweep_kind:
            self._program_buffers[self._sweep_kind] = self._program_code.toPlainText()

    def _present_program(self) -> None:
        available = bool(self._sweep_kind and self._available.get(self._sweep_kind, False))
        for index in range(self._sweep_combo.count()):
            kind = str(self._sweep_combo.itemData(index) or "")
            item = self._sweep_combo.model().item(index)
            if item is not None:
                item.setEnabled(bool(self._available.get(kind, False)))
        with signals_blocked(self._sweep_combo):
            index = self._sweep_combo.findData(self._sweep_kind)
            self._sweep_combo.setCurrentIndex(index if index >= 0 else 0)

        self._program_empty.setVisible(not available)
        self._program_title.setVisible(available)
        self._columns_intro.setVisible(available)
        self._template_host.setVisible(available)
        self._program_code.setVisible(available)
        for kind, host in self._column_hosts.items():
            host.setVisible(available and kind == self._sweep_kind)
        if not available:
            return

        title = "Hardware scan-slot program" if self._sweep_kind == self._scan_slot_kind else (
            "API-slot sweep program"
        )
        if self._program_title.text() != title:
            self._program_title.setText(title)
        text = self._program_buffers[self._sweep_kind]
        if self._program_code.toPlainText() != text:
            with signals_blocked(self._program_code):
                self._program_code.setPlainText(text)

    def _on_sweep_changed(self, *_args) -> None:
        kind = str(self._sweep_combo.currentData() or "")
        if kind == self._sweep_kind or not self._available.get(kind, False):
            return
        self._stash_program()
        self._sweep_kind = kind
        self._ensure_program_buffer(kind)
        self._present_program()
        self.changed.emit()

    def _on_program_edited(self) -> None:
        if not self._sweep_kind:
            return
        self._program_buffers[self._sweep_kind] = self._program_code.toPlainText()
        self._program_initialized[self._sweep_kind] = True
        self._program_dirty[self._sweep_kind] = True
        self.changed.emit()

    def _insert_template(self, template: str) -> None:
        if not self._sweep_kind or not self._available.get(self._sweep_kind, False):
            return
        self._program_code.setPlainText(
            scan_table_template(template, self._specs[self._sweep_kind])
        )

    # ------------------------------------------------------------------- value
    def values_dict(self) -> dict:
        """Return the sole structured PulseScan form value."""

        api: dict[str, float] = {}
        for key in self._api_order:
            row = self._api_rows[key]
            text = row.edit.text().strip()
            if not text:
                continue
            try:
                value = float(text)
            except ValueError:
                raise ValueError(
                    f"API slot {row.slot_id!r} must be numeric"
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"API slot {row.slot_id!r} must be finite"
                )
            api[row.slot_id] = value
        program = self._program_code.toPlainText() if self._sweep_kind else ""
        return {
            "program_id": self._program_id,
            "api": api,
            "sweep_kind": self._sweep_kind,
            "program": program,
        }

    def seed_value(self, value) -> None:
        """Queue a saved override for the next matching committed program."""

        if not isinstance(value, Mapping):
            raise TypeError("pulse_slots must be a mapping")
        self._pending_api = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self._pending_program_id = ""
        if not value:
            return
        required = {"program_id", "api", "sweep_kind", "program"}
        if set(value) != required:
            raise ValueError(
                "pulse_slots must contain exactly "
                f"{tuple(sorted(required))}"
            )
        program_id = value["program_id"]
        sweep_kind = value["sweep_kind"]
        program = value["program"]
        api = value["api"]
        if not isinstance(program_id, str):
            raise TypeError("pulse_slots program_id must be str")
        if not isinstance(sweep_kind, str):
            raise TypeError("pulse_slots sweep_kind must be str")
        if sweep_kind not in {"", self._scan_slot_kind, self._api_slot_kind}:
            raise ValueError("pulse_slots sweep_kind is unknown")
        if not isinstance(program, str):
            raise TypeError("pulse_slots program must be str")
        if not isinstance(api, Mapping):
            raise TypeError("pulse_slots api must be a mapping")
        for name, item in api.items():
            if not isinstance(name, str) or not name or name.strip() != name:
                raise ValueError("pulse_slots API names must be canonical text")
            if not _is_number(item):
                raise ValueError(
                    f"pulse_slots API value for {name!r} must be finite numeric"
                )
            self._pending_api[name] = f"{float(item):g}"
        self._pending_sweep_kind = sweep_kind
        self._pending_program = program
        self._pending_program_id = program_id
