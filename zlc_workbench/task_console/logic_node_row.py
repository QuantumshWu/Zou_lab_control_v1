"""The Logic tab's row card: one node, its state dot, publishes, and four buttons.

Displays a :class:`.console_records.LogicNodeConfig` and emits what the operator
asked for; it owns no lifecycle itself, which is why it can live here while the
console keeps the start/stop/edit/remove decisions.

Qt is legitimate here because this is TaskConsole application chrome; it pulls
in no Matplotlib and owns no runtime lifecycle.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from .console_records import LogicNodeConfig

from zlc_frontend.qt_widgets import (
    ElidedLabel,
    FluentButton,
    FluentFrame,
    FluentLabel,
    PublishedSignalsLegend,
    FluentStatusDot,
    scaled_px,
)
from zlc_frontend.qt_widgets import ACCENT, GREEN, GREY, ORANGE, RED

__all__ = ["LogicNodeRow"]


class LogicNodeRow(FluentFrame):
    """One LOGIC NODE's CARD on the Logic tab: a status dot + name + (kind) + status on
    the top line with Start / Stop / Edit / Remove, and a second line listing the
    signals it PUBLISHES with their schema-derived physical tensor dimensions
    (``R × P × (*data_shape)``).  The dot follows the run state (grey=stopped / green=running /
    red=error), confocal's tab-icon colour map applied to a card.  Start / Stop act
    here directly; the full param form is in the node's Edit tab
    (:class:`LogicNodeEditor`)."""

    edit_requested = QtCore.pyqtSignal(object)     # "Edit" -> open the node's Edit tab
    remove_requested = QtCore.pyqtSignal(object)
    start_requested = QtCore.pyqtSignal(object)    # "Start" -> build + run the node
    stop_requested = QtCore.pyqtSignal(object)     # "Stop"  -> stop the node

    # confocal gui_combine colour map (INIT=grey / RUNNING=green / STOP/ERROR=red).
    STATE_COLORS = {"stopped": GREY, "running": GREEN, "error": RED}

    def __init__(self, node: LogicNodeConfig, kind: str, parent=None):
        super().__init__(parent)
        if kind not in {"measurement", "processor", "task"}:
            raise ValueError("LogicNodeRow kind must come from a descriptor")
        self.node = node
        self.kind = kind
        self._task_takeover = False
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8))
        outer.setSpacing(scaled_px(4, minimum=3))
        # --- top line: status + name + (kind) + Start / Stop / Edit / Remove --------
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(scaled_px(10, minimum=6))
        self.dot = FluentStatusDot(size=14)
        self.dot.set_color(GREY)
        self.name_label = ElidedLabel(node.title)
        self.name_label.setMinimumWidth(0)
        self.name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.kind_label = FluentLabel(f"({kind})")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status_label = ElidedLabel("stopped")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.status_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.start_button = FluentButton("Start", color=GREEN)
        self.start_button.setFixedWidth(scaled_px(60, minimum=48))
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self))
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.stop_button.setFixedWidth(scaled_px(56, minimum=46))
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(self))
        self.stop_button.setEnabled(False)
        self.edit_button = FluentButton("Edit", color=ACCENT)
        self.edit_button.setFixedWidth(scaled_px(56, minimum=46))
        self.edit_button.clicked.connect(lambda: self.edit_requested.emit(self))
        self.remove_button = FluentButton("Remove", color=GREY)
        self.remove_button.setFixedWidth(scaled_px(82, minimum=66))
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(self.dot, 0)
        top.addWidget(self.name_label, 1)
        top.addWidget(self.kind_label, 0)
        top.addWidget(self.status_label, 2)
        for b in (self.start_button, self.stop_button, self.edit_button, self.remove_button):
            top.addWidget(b, 0)
        outer.addLayout(top)
        # --- published-signals legend: one signal per line (name | shape | meaning) ---
        # Monospace so the name/shape columns ALIGN down the rows (a readable table, not a
        # run-on line).
        self.publishes_label = PublishedSignalsLegend()
        outer.addWidget(self.publishes_label)

    def set_state(self, state: str, *, status: str = "") -> None:
        """Reflect the node's run state on the dot + status text + Start/Stop enable.

        Change-gated (cache the last ``(state, status)``): a steady tick calls this on every
        running row, so re-setText / re-setStyleSheet / re-setEnabled an UNCHANGED row every tick
        is pure churn, like the ``set_publishes`` text gate."""
        key = (state, status)
        if key == getattr(self, "_state_key", None):
            return
        self._state_key = key
        self.dot.set_color(self.STATE_COLORS.get(state, GREY))
        self.status_label.setText(status or state)
        colour = RED if state == "error" else GREY
        self.status_label.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
        running = state == "running"
        if not self._task_takeover:
            self.start_button.setEnabled(not running)
            self.stop_button.setEnabled(running)

    def set_task_takeover(self, active: bool) -> None:
        """Project the console's single active-task command gate onto this row.

        The row remains the owner of its controls; TaskConsole only supplies the
        application admission state.  No second Stop action is left enabled while
        a Task owns the console.  Restoring the gate derives button state from the
        cached lifecycle state instead of inventing a second lifecycle registry.
        """

        active = bool(active)
        if active == self._task_takeover:
            return
        self._task_takeover = active
        if active:
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self.edit_button.setEnabled(False)
            self.remove_button.setEnabled(False)
            if self.kind == "task":
                self.stop_button.setVisible(False)
            return
        self.stop_button.setVisible(True)
        state = getattr(self, "_state_key", ("stopped", ""))[0]
        running = state == "running"
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.edit_button.setEnabled(True)
        self.remove_button.setEnabled(True)

    def set_publishes(self, rows) -> None:
        """Show the node's outputs as a SHORT table -- ONE signal per line, ``name`` + ``shape`` only::

            publishes:
              occupied   1 × 1 × (35)
              rate       1 × 1 × (1)

        The per-signal MEANING goes in the label's tooltip (hover), NOT inline -- so the card never
        grows wider than its column and the Logic list needs no horizontal scroll.  ``rows`` is
        ``[(name, shape, description)]`` (dimensions AUTO-EXTRACTED from the authoritative signal schema; meanings
        from the node's ``output_specs``); a pending shape (``—``) just means no value yet."""
        self.publishes_label.set_rows(rows)
