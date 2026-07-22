"""The multi-slot signal picker plus a ``value = ...`` expression, as one widget.

This is the composite the ``signal_expr`` ParamDecl kind renders.  Every primitive it
uses - the grouped picker, the editable-combo reader, the floating editor, the seed
rule, the help text - already lived in the target packages; only the composite itself
was still in the legacy shell, which is why ``ParamWidgetContext`` had to carry a
``signal_expr_factory`` to reach back for it.  With the widget here the handler builds
it directly and that inversion is gone.

Qt lives here legitimately: this module is inside ``qt_widgets``, the one place the
placement axiom lets PyQt5 be imported, and it pulls in no Matplotlib.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data.signal_expr import SIGNAL_EXPR_HELP, SignalExpr, seed_source_for_slots

from .fluent import (
    FluentButton, FluentFloatingEditor, FluentLineEdit, FluentSectionLabel,
    FluentSettingRow, FluentTreeComboBox, scaled_px, setting_label_width)
from .style import GREY
from .signal_picker import coerce_short_labels, fill_grouped_signal_combo, read_editable_combo

__all__ = ["SignalExprWidget"]


class SignalExprWidget(QtWidgets.QWidget):
    """A multi-slot hub-signal picker + a ``value = ...`` expression -- the SAME source control a
    plot panel uses, packaged as a measurement/processor param (ParamDecl kind ``signal_expr``).

    Pick one or more live hub signals (read as ``signal`` / ``signal[i]``) and combine them in a
    one-line expression.  So ANY "source" field -- a processor's frame, a pulse-scan's y -- can
    subscribe to several running nodes' signals, never just one bare name.  Output schema:
    ``{"inputs": [name, ...], "source": "value = ..."}`` (the :class:`SignalExpr` value).  Reuses
    the SAME primitives as the panel Setting (``fill_grouped_signal_combo`` / ``read_editable_combo``
    / the ``FluentFloatingEditor``) + the shared seed rule, so the logic is single-source."""

    changed = QtCore.pyqtSignal()

    def __init__(self, *, signals_provider=None, sources_provider=None, formats_provider=None,
                 labels_provider=None, title: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet("background: transparent;")
        self._signals_provider = signals_provider
        self._sources_provider = sources_provider
        self._formats_provider = formats_provider
        # The SHORT-name map (``short_names_provider``: {full hub name -> short name}), passed as the
        # picker's ``labels`` so a leaf shows "frame_0" -- NOT the prefix-stripped "0/1/2" the
        # _common_token_prefix fallback yields without it.  This is what makes THIS source picker render
        # IDENTICALLY to the plot-panel Setting picker (same fill_grouped_signal_combo + same labels).
        self._labels_provider = labels_provider
        self._inputs: list[str] = ["frame_0"]
        self._editor = None
        # ONE label-column width for this widget's rows (signal / signal[i] / value), via the SAME
        # setting_label_width rule every form uses -- so it aligns + follows the one logic.
        self._label_w = setting_label_width(["signal[0]", "value"], minimum=64)
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(scaled_px(4, minimum=3))
        # Spans the FULL form width; its title is a FluentSectionLabel header (the ONE section
        # style -- same as the Setting popup sections), with the signal slots + value rows stacking
        # flush beneath (no indent: section-vs-row hierarchy is weight+colour, never indentation).
        if title:
            root.addWidget(FluentSectionLabel(title))
        self._slot_box = QtWidgets.QVBoxLayout()
        self._slot_box.setContentsMargins(0, 0, 0, 0)
        self._slot_box.setSpacing(scaled_px(4, minimum=3))
        root.addLayout(self._slot_box)
        self.slot_combos: list = []
        self._slot_rows: list[FluentSettingRow] = []
        # +/- buttons sit UNDER the label column (an empty label), so they line up with the
        # control column of the rows above instead of floating full-width.
        btn_inner = QtWidgets.QWidget()
        btn_row = QtWidgets.QHBoxLayout(btn_inner)
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(scaled_px(6, minimum=4))
        self._add_btn = FluentButton("+ signal", color=GREY)
        self._add_btn.setToolTip("Add another signal slot (read as signal[i] in the expression).")
        self._add_btn.clicked.connect(self._add_slot)
        self._rm_btn = FluentButton("− signal", color=GREY)
        self._rm_btn.setToolTip("Remove the last signal slot.")
        self._rm_btn.clicked.connect(self._remove_slot)
        btn_row.addWidget(self._add_btn, 0)
        btn_row.addWidget(self._rm_btn, 0)
        btn_row.addStretch(1)
        # +/- buttons under a blank label so they line up with the signal-slot control column
        root.addWidget(FluentSettingRow("", btn_inner, label_width=self._label_w))
        self._source_edit = FluentLineEdit("value = signal")
        self._source_edit.setMinimumWidth(scaled_px(160, minimum=120))
        self._source_edit.setToolTip(SIGNAL_EXPR_HELP)
        self._source_edit.textChanged.connect(self.changed)
        self._edit_btn = FluentButton("Edit…", color=GREY)
        self._edit_btn.setFixedWidth(scaled_px(56, minimum=44))
        self._edit_btn.setToolTip("Open a large floating editor for this expression")
        self._edit_btn.clicked.connect(self._open_editor)
        expr_inner = QtWidgets.QWidget()
        expr_row = QtWidgets.QHBoxLayout(expr_inner)
        expr_row.setContentsMargins(0, 0, 0, 0)
        expr_row.setSpacing(scaled_px(6, minimum=4))
        expr_row.addWidget(self._source_edit, 1)
        expr_row.addWidget(self._edit_btn, 0)
        # the expression on its OWN labelled "value" row, aligned to the same label column as the
        # signal slots above -- so the whole source control reads as one tidy grid (#4)
        root.addWidget(FluentSettingRow("value", expr_inner, label_width=self._label_w))
        self._reconcile_slots()

    def _names(self) -> list:
        try:
            return [str(n) for n in self._signals_provider()] if callable(self._signals_provider) else []
        except Exception:
            return []

    def _sources(self) -> dict:
        try:
            return self._sources_provider() if callable(self._sources_provider) else {}
        except Exception:
            return {}

    def _formats(self) -> dict:
        try:
            return self._formats_provider() if callable(self._formats_provider) else {}
        except Exception:
            return {}

    def _labels(self) -> dict:
        return coerce_short_labels(self._labels_provider)

    def _reconcile_slots(self) -> None:
        """Keyed-by-index slot delta; retained pickers never get recreated."""

        n = max(1, len(self._inputs))
        while len(self.slot_combos) > n:
            row = self._slot_rows.pop()
            self.slot_combos.pop()
            self._slot_box.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        while len(self.slot_combos) < n:
            i = len(self.slot_combos)
            combo = FluentTreeComboBox()             # collapsible-tree signal picker (G2)
            combo.activated.connect(lambda *_a, idx=i: self._on_pick(idx))
            self.slot_combos.append(combo)
            row = FluentSettingRow("", combo, label_width=self._label_w)
            self._slot_rows.append(row)
            self._slot_box.addWidget(row)
        for i, (combo, row) in enumerate(
            zip(self.slot_combos, self._slot_rows, strict=True)
        ):
            row.set_label(
                f"signal[{i}]" if n > 1 else "signal",
                width=self._label_w,
            )
            current = self._inputs[i] if i < len(self._inputs) else ""
            fill_grouped_signal_combo(
                combo,
                names=self._names(),
                sources=self._sources(),
                formats=self._formats(),
                labels=self._labels(),
                current=current,
            )
        self._rm_btn.setEnabled(n > 1)

    def _collect_inputs(self) -> None:
        if self.slot_combos:
            self._inputs = [read_editable_combo(c) for c in self.slot_combos]

    def _on_pick(self, idx: int) -> None:
        self._collect_inputs()
        # Single slot: the picker IS "read this signal" -> point the source at it (value = signal).
        # Multi-slot: the expression is user-authored across slots, so a pick just rebinds slot idx.
        if len(self.slot_combos) <= 1 and idx == 0:
            self._source_edit.blockSignals(True)
            self._source_edit.setText("value = signal")
            self._source_edit.blockSignals(False)
        self.changed.emit()

    def _add_slot(self) -> None:
        from zlc_data.signal_expr import seed_source_for_slots
        self._collect_inputs()
        self._inputs.append("")
        self._source_edit.blockSignals(True)
        self._source_edit.setText(seed_source_for_slots(len(self._inputs), self._source_edit.text()))
        self._source_edit.blockSignals(False)
        self._reconcile_slots()
        self.changed.emit()

    def _remove_slot(self) -> None:
        if len(self._inputs) <= 1:
            return
        from zlc_data.signal_expr import seed_source_for_slots
        self._collect_inputs()
        self._inputs.pop()
        self._source_edit.blockSignals(True)
        self._source_edit.setText(seed_source_for_slots(len(self._inputs), self._source_edit.text()))
        self._source_edit.blockSignals(False)
        self._reconcile_slots()
        self.changed.emit()

    def _open_editor(self) -> None:
        if self._editor is not None:
            self._editor.raise_(); self._editor.activateWindow(); return
        self._editor = FluentFloatingEditor(SIGNAL_EXPR_HELP, self._source_edit.text(),
                                            self.window(), title="Edit signal expression")
        self._editor.applied.connect(lambda text: self._source_edit.setText(" ".join(text.split("\n")).strip()))
        self._editor.destroyed.connect(self._clear_editor)
        self._editor.show()

    def _clear_editor(self, *_a) -> None:
        self._editor = None

    def rebuild_combos(self) -> None:
        """Refill every slot combo from the providers (a tab re-show: a signal published since
        the form opened becomes pickable), keeping the current pick."""
        for combo in self.slot_combos:
            cur = read_editable_combo(combo)
            fill_grouped_signal_combo(combo, names=self._names(), sources=self._sources(),
                                      formats=self._formats(), labels=self._labels(), current=cur)

    def values_dict(self) -> dict:
        """The current ``{"inputs": [...], "source": "value = ..."}`` (the signal_expr value)."""
        self._collect_inputs()
        inputs = [str(n) for n in self._inputs if str(n).strip()]
        return {"inputs": inputs, "source": self._source_edit.text().strip() or "value = signal"}

    def set_value(self, value) -> None:
        """Seed from a ``{"inputs", "source"}`` dict (a default / saved value)."""
        from zlc_data.signal_expr import SignalExpr
        expr = SignalExpr.from_value(value)
        self._inputs = list(expr.inputs) or ["frame_0"]
        self._source_edit.blockSignals(True)
        self._source_edit.setText(expr.source)
        self._source_edit.blockSignals(False)
        self._reconcile_slots()
