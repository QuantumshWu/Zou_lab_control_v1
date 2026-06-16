"""Auto-generated parameter form: render a producer's declared parameters as fluent
controls, with NO per-producer hand-written UI.

It consumes a list of ``ParamField`` records (``name`` / ``kind`` / ``default`` /
``unit`` / ``choices`` / ``target``) by DUCK TYPING -- it never imports the neutral-
atom ``params`` module, so the frontend stays decoupled.  One row per field; the
widget per ``kind`` is: float/int/str -> line edit, bool -> checkbox, choice ->
combo, signal -> signal-name combo, array/pulse_scan -> a scan-range line edit
(``start:stop:step`` or a comma list).  ``values()`` reads them back into a plain
``{name: value}`` dict an API call / measurement build can consume directly.

Sealed: the form takes ParamField records (+ an optional signal-name list for signal
pickers); it never takes ``data_px`` / ``margins_px`` / ``spec`` / ``dpi``.
"""

from __future__ import annotations

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .qt_fluent import (
    FluentCheckBox, FluentComboBox, FluentLabeledField, FluentLineEdit, scaled_px)


def parse_scan_text(text):
    """Parse a scan-array field: ``"start:stop:step"`` -> ``np.arange`` (inclusive of
    the endpoint within half a step), or a comma list ``"1, 2, 3"`` -> that array.
    Empty -> None."""
    text = str(text).strip()
    if not text or text.lower() == "none":
        return None
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        if len(parts) == 3:
            start, stop, step = parts
            if step == 0:
                return np.asarray([start], dtype=float)
            return np.arange(start, stop + step / 2.0, step, dtype=float)
        if len(parts) == 2:
            return np.asarray(parts, dtype=float)
    return np.asarray([float(p) for p in text.replace(",", " ").split()], dtype=float)


def _scan_text(value) -> str:
    if value is None:
        return ""
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size >= 3 and np.allclose(np.diff(arr), arr[1] - arr[0]):
        return f"{arr[0]:g}:{arr[-1]:g}:{arr[1] - arr[0]:g}"   # uniform -> compact range
    return ", ".join(f"{v:g}" for v in arr)


class AutoForm(QtWidgets.QWidget):
    """A vertical stack of ``[label : control]`` rows built from ParamField records."""

    changed = QtCore.pyqtSignal()

    def __init__(self, fields, *, current=None, signal_names=(), parent=None, label_width=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._fields = list(fields)
        self._signal_names = list(signal_names)
        self._widgets: dict[str, tuple[str, QtWidgets.QWidget]] = {}
        current = dict(current or {})

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(scaled_px(6, minimum=4))
        for field in self._fields:
            value = current.get(field.name, field.default)
            widget = self._build_widget(field, value)
            unit = getattr(field, "unit", "") or ""
            suffix = None
            if unit:
                from .qt_fluent import FluentLabel
                suffix = FluentLabel(unit)
            label = getattr(field, "label", "") or field.name
            row = FluentLabeledField(label, widget, label_width=label_width, suffix=suffix)
            layout.addWidget(row)
            self._widgets[field.name] = (field.kind, widget)
        layout.addStretch(1)

    # ------------------------------------------------------------- build widgets
    def _build_widget(self, field, value) -> QtWidgets.QWidget:
        kind = field.kind
        if kind == "bool":
            box = FluentCheckBox("")
            box.setChecked(bool(value))
            box.stateChanged.connect(lambda *_: self.changed.emit())
            return box
        if kind == "choice":
            combo = FluentComboBox()
            combo.addItems([str(c) for c in getattr(field, "choices", ())])
            if value is not None:
                combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(lambda *_: self.changed.emit())
            return combo
        if kind == "signal":
            combo = FluentComboBox()
            combo.setEditable(True)                       # a signal name OR an expression
            combo.addItems([str(s) for s in self._signal_names])
            combo.setCurrentText("" if value is None else str(value))
            combo.currentTextChanged.connect(lambda *_: self.changed.emit())
            return combo
        edit = FluentLineEdit("")
        if kind in ("array", "pulse_scan"):
            edit.setText(_scan_text(value))
            if kind == "pulse_scan" and getattr(field, "target", ""):
                edit.setPlaceholderText(f"{field.target}  start:stop:step")
            else:
                edit.setPlaceholderText("start:stop:step  or  1, 2, 3")
        else:
            edit.setText("" if value is None else str(value))
        edit.editingFinished.connect(self.changed.emit)
        return edit

    # --------------------------------------------------------------- read values
    def values(self) -> dict:
        """Read the form back into ``{name: value}`` (types matched to each field)."""
        out: dict[str, object] = {}
        for field in self._fields:
            kind, widget = self._widgets[field.name]
            out[field.name] = self._read_widget(kind, widget)
        return out

    @staticmethod
    def _read_widget(kind, widget):
        if kind == "bool":
            return bool(widget.isChecked())
        if kind in ("choice", "signal"):
            return widget.currentText()
        text = widget.text().strip()
        if kind in ("array", "pulse_scan"):
            return parse_scan_text(text)
        if kind == "int":
            try:
                return int(round(float(text)))
            except (TypeError, ValueError):
                return None
        if kind == "float":
            try:
                return float(text)
            except (TypeError, ValueError):
                return None
        return text

    def set_values(self, mapping) -> None:
        for name, value in dict(mapping).items():
            entry = self._widgets.get(name)
            if entry is None:
                continue
            kind, widget = entry
            if kind == "bool":
                widget.setChecked(bool(value))
            elif kind in ("choice", "signal"):
                widget.setCurrentText("" if value is None else str(value))
            elif kind in ("array", "pulse_scan"):
                widget.setText(value if isinstance(value, str) else _scan_text(value))
            else:
                widget.setText("" if value is None else str(value))


__all__ = ["AutoForm", "parse_scan_text"]
