"""The pulse-template slot editor: API parameter fields plus the sweep/program selector.

The composite the ``pulse_slots`` ParamDecl kind renders.  Everything it reads had
already migrated - the sweep-kind vocabulary and the scan-table template into
``zlc_data``, the slot label into ``zlc_data.shape_text``, the derived column specs
arriving through the ``PulseTemplateRows`` port - so the widget was the last piece
still sitting in the legacy shell, and the only reason ``ParamWidgetContext`` still
carried a factory to reach back for it.  With this move that field is gone and the
context needs no callback into the console at all.

Seeding is deliberately DEFERRED: ``seed_value`` remembers a payload against its
``program_id`` and the next matching ``rebuild`` restores it.  A saved workspace is
loaded before the template rows are known, so applying eagerly would write values
into fields that do not exist yet.
"""

from __future__ import annotations

from typing import Mapping

from PyQt5 import QtCore, QtWidgets

from zlc_data.shape_text import slot_label
from zlc_data.scan_template import scan_table_template
from zlc_data.vocabulary import SWEEP_API_SLOT, SWEEP_SCAN_SLOT

from .fluent import (
    FluentButton, FluentCodeEdit, FluentComboBox, FluentLabel, FluentLineEdit,
    FluentSectionLabel, FluentSettingRow, scaled_px, setting_label_width)
from .style import GREY

__all__ = ["PulseSlotsWidget"]


def _is_number(v) -> bool:
    """True when ``v`` can be read as a finite float (a saved numeric param), else False."""
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


class PulseSlotsWidget(QtWidgets.QWidget):
    """Structured editor for the two PulseScan execution strategies.

    A scan-slot sweep uploads one complete FPGA table; an API-slot sweep submits one finite pulse
    per row.  The selector changes the meaning and columns of a single program editor.  Each
    strategy keeps its own in-memory buffer because those column spaces are not interchangeable.
    Selecting a pulse template seeds the scan-slot buffer from that template's persisted program;
    the API-slot buffer is generated from its API fields.

    A saved task override is tied to ``program_id``.  It is restored only when that exact template
    is selected; values named ``a1`` or a code buffer can therefore never leak from one template
    into another template whose opaque internal slot happens to have the same index."""

    changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._api_widgets: dict[str, QtWidgets.QWidget] = {}
        from zlc_data.vocabulary import SWEEP_API_SLOT, SWEEP_SCAN_SLOT
        self._scan_slot_kind = SWEEP_SCAN_SLOT
        self._api_slot_kind = SWEEP_API_SLOT
        self._program_code = None
        self._sweep_combo = None
        self._sweep_kind = ""
        self._program_buffers = {SWEEP_SCAN_SLOT: "", SWEEP_API_SLOT: ""}
        self._columns: dict[str, list] = {SWEEP_SCAN_SLOT: [], SWEEP_API_SLOT: []}
        self._specs: dict[str, list] = {SWEEP_SCAN_SLOT: [], SWEEP_API_SLOT: []}
        self._available = {SWEEP_SCAN_SLOT: False, SWEEP_API_SLOT: False}
        self._program_id = ""
        self._pending_program_id = ""
        self._pending_api: dict[str, str] = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self._box = QtWidgets.QVBoxLayout(self)
        self._box.setContentsMargins(0, 0, 0, 0)
        self._box.setSpacing(scaled_px(6, minimum=4))
        self._api_box = QtWidgets.QVBoxLayout()
        self._api_box.setContentsMargins(0, 0, 0, 0)
        self._api_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._api_box)
        self._selector_box = QtWidgets.QVBoxLayout()
        self._selector_box.setContentsMargins(0, 0, 0, 0)
        self._selector_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._selector_box)
        self._program_box = QtWidgets.QVBoxLayout()
        self._program_box.setContentsMargins(0, 0, 0, 0)
        self._program_box.setSpacing(scaled_px(6, minimum=4))
        self._box.addLayout(self._program_box)

    @staticmethod
    def _drop_layout(layout) -> None:
        """Tear down every child widget + nested layout under ``layout`` (rebuilt from scratch)."""
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None); w.deleteLater()
            child = item.layout()
            if child is not None:
                PulseSlotsWidget._drop_layout(child)

    def rebuild(self, api_rows, scan_rows, *, api_columns=(), scan_columns=(),
                hardware_program: str = "", program_id: str = "") -> None:
        """Rebuild from one pulse template.

        ``api_rows`` entries are ``(handle, coordinate, kind, target, unit, current)``;
        ``scan_rows`` entries are ``(coordinate, kind, target, unit, label)``.
        ``*_columns`` are the matching ``ScanColumnSpec`` per slot, already derived
        by the domain -- their per-kind default sweep needs the bus signed range and
        the clock tick, which this layer may not reach.
        """

        program_id = str(program_id or "")
        same_program = bool(program_id and program_id == self._program_id)
        restore_saved = bool(program_id and program_id == self._pending_program_id)

        remembered_api = {}
        if same_program:
            remembered_api = {name: widget.text().strip()
                              for name, widget in self._api_widgets.items()}
            self._stash_program()

        self._drop_layout(self._api_box)
        self._drop_layout(self._selector_box)
        self._drop_layout(self._program_box)
        self._api_widgets = {}
        self._program_code = None
        self._sweep_combo = None
        self._program_id = program_id

        self._api_box.addWidget(FluentSectionLabel("API parameters"))
        if api_rows:
            labels = [slot_label(kind, target)
                      for _handle, _coord, kind, target, _unit, _current in api_rows]
            label_width = setting_label_width(labels, minimum=72)
            for handle, _coordinate, kind, target, unit, current in api_rows:
                label = slot_label(kind, target)
                seed = (self._pending_api.get(handle) if restore_saved else None)
                if seed is None and same_program:
                    seed = remembered_api.get(handle)
                if seed is None:
                    seed = f"{float(current):g}"
                edit = FluentLineEdit(seed, self)
                edit.setMinimumWidth(scaled_px(120, minimum=96))
                edit.setPlaceholderText(str(unit))
                edit.setToolTip(
                    f"Resting value for {label} ({unit}).  In an API-slot sweep, the program "
                    "overrides this handle once per row.")
                edit.textChanged.connect(self.changed)
                self._api_box.addWidget(FluentSettingRow(label, edit, label_width=label_width))
                self._api_widgets[str(handle)] = edit
        else:
            note = FluentLabel("(this template has no API parameter)", self)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._api_box.addWidget(note)

        self._columns[self._api_slot_kind] = [
            (coordinate, slot_label(kind, target), str(unit or ""))
            for _handle, coordinate, kind, target, unit, _current in api_rows
        ]
        self._specs[self._api_slot_kind] = list(api_columns)
        self._columns[self._scan_slot_kind] = []
        self._specs[self._scan_slot_kind] = list(scan_columns)
        for coordinate, kind, target, unit, stored_label in scan_rows:
            display = stored_label or slot_label(kind, target)
            display_unit = "ns ticks" if kind == "duration" else (
                "integer code (LSB)" if kind == "dac" else str(unit or ""))
            self._columns[self._scan_slot_kind].append((coordinate, display, display_unit))

        self._available = {
            self._scan_slot_kind: bool(scan_rows),
            self._api_slot_kind: bool(api_rows),
        }
        if not same_program:
            self._program_buffers = {
                self._scan_slot_kind: str(hardware_program or ""),
                self._api_slot_kind: "",
            }
        elif not self._program_buffers[self._scan_slot_kind].strip():
            self._program_buffers[self._scan_slot_kind] = str(hardware_program or "")

        default_kind = self._scan_slot_kind if scan_rows else (
            self._api_slot_kind if api_rows else "")
        if restore_saved and self._available.get(self._pending_sweep_kind, False):
            self._sweep_kind = self._pending_sweep_kind
            self._program_buffers[self._sweep_kind] = self._pending_program
        elif not same_program or not self._available.get(self._sweep_kind, False):
            self._sweep_kind = default_kind

        self._build_sweep_selector()
        self._render_program()
        self._pending_program_id = ""
        self._pending_api = {}
        self._pending_sweep_kind = ""
        self._pending_program = ""
        self.changed.emit()

    def _build_sweep_selector(self) -> None:
        combo = FluentComboBox()
        choices = (
            ("Scan slots (hardware table)", self._scan_slot_kind),
            ("API slots (one pulse per point)", self._api_slot_kind),
        )
        for label, kind in choices:
            combo.addItem(label, kind)
            item = combo.model().item(combo.count() - 1)
            if item is not None:
                item.setEnabled(bool(self._available[kind]))
        index = combo.findData(self._sweep_kind)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.currentIndexChanged.connect(self._on_sweep_changed)
        combo.setToolTip(
            "Scan slots upload one complete FPGA table. API slots resolve and submit one finite "
            "pulse per program row.")
        self._sweep_combo = combo
        self._selector_box.addWidget(FluentSettingRow(
            "Sweep", combo, label_width=setting_label_width(["Sweep"], minimum=72)))

    def _on_sweep_changed(self, *_args) -> None:
        if self._sweep_combo is None:
            return
        kind = str(self._sweep_combo.currentData() or "")
        if kind == self._sweep_kind or not self._available.get(kind, False):
            return
        self._stash_program()
        self._sweep_kind = kind
        self._render_program()
        self.changed.emit()

    def _stash_program(self) -> None:
        if self._program_code is not None and self._sweep_kind:
            self._program_buffers[self._sweep_kind] = self._program_code.toPlainText()

    def _render_program(self) -> None:
        self._drop_layout(self._program_box)
        self._program_code = None
        if not self._sweep_kind or not self._available.get(self._sweep_kind, False):
            note = FluentLabel(
                "(bind at least one scan slot or API slot in the Pulse GUI)", self)
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
            self._program_box.addWidget(note)
            return

        title = "Hardware scan-slot program" if self._sweep_kind == self._scan_slot_kind \
            else "API-slot sweep program"
        self._program_box.addWidget(FluentSectionLabel(title))
        columns = self._columns[self._sweep_kind]
        legend = ["Columns of scan_table (one row = one point, columns advance in lockstep):"]
        legend.extend(f"  {name}: {display}  [{unit}]" for name, display, unit in columns)
        legend_label = FluentLabel("\n".join(legend), self)
        legend_label.setWordWrap(True)
        legend_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self._program_box.addWidget(legend_label)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        btn_row.addWidget(FluentLabel("template:", self))
        for template in ("column_stack", "grid"):
            button = FluentButton(template, color=GREY)
            button.clicked.connect(lambda *_a, value=template: self._insert_template(value))
            btn_row.addWidget(button, 0)
        btn_row.addStretch(1)
        self._program_box.addLayout(btn_row)

        from zlc_data.scan_template import scan_table_template
        seed = str(self._program_buffers[self._sweep_kind] or "").strip()
        if not seed:
            seed = scan_table_template("column_stack", self._specs[self._sweep_kind])
        self._program_buffers[self._sweep_kind] = seed
        editor = FluentCodeEdit(seed)
        editor.setMinimumHeight(scaled_px(120, minimum=90))
        editor.setToolTip(
            "Python assigning an (N_points x n_columns) array to scan_table. Values use each "
            "selected slot's native unit.")
        editor.textChanged.connect(self.changed)
        self._program_box.addWidget(editor)
        self._program_code = editor

    def _insert_template(self, template: str) -> None:
        from zlc_data.scan_template import scan_table_template
        if self._program_code is not None and self._sweep_kind:
            self._program_code.setPlainText(
                scan_table_template(template, self._specs[self._sweep_kind]))

    def values_dict(self) -> dict:
        """Return the sole structured PulseScan form value."""

        api: dict[str, float] = {}
        for name, widget in self._api_widgets.items():
            text = widget.text().strip()
            if not text:
                continue
            try:
                api[name] = float(text)
            except ValueError:
                continue
        program = self._program_code.toPlainText() if self._program_code is not None else ""
        return {
            "program_id": self._program_id,
            "api": api,
            "sweep_kind": self._sweep_kind,
            "program": program,
        }

    def seed_value(self, value) -> None:
        """Queue a saved override, applied only to the exact matching pulse program."""

        if not isinstance(value, Mapping):
            return
        for name, item in dict(value.get("api") or {}).items():
            self._pending_api[str(name)] = f"{float(item):g}" if _is_number(item) else str(item)
        self._pending_sweep_kind = str(value.get("sweep_kind") or "")
        self._pending_program = str(value.get("program") or "")
        self._pending_program_id = str(value.get("program_id") or "")
