"""The Logic tab's row card: one node, its state dot, publishes, and four buttons.

Displays a :class:`zlc_data.logic_node.LogicNodeConfig` and emits what the operator
asked for; it owns no lifecycle itself, which is why it can live here while the
console keeps the start/stop/edit/remove decisions.

Qt is legitimate here - this module is inside ``qt_widgets``, the one place the
placement axiom lets PyQt5 be imported - and it pulls in no Matplotlib.
"""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data.logic_node import LogicNodeConfig

from .fluent import FluentButton, FluentFrame, FluentLabel, FluentStatusDot, scaled_px
from .style import ACCENT, GREEN, GREY, ORANGE, RED

__all__ = ["LogicNodeRow"]


class LogicNodeRow(FluentFrame):
    """One LOGIC NODE's CARD on the Logic tab: a status dot + name + (kind) + status on
    the top line with Start / Stop / Edit / Remove, and a second line listing the
    signals it PUBLISHES with their array shape (``occupied [per-site (N,)], rate
    [scalar]``).  The dot follows the run state (grey=stopped / green=running /
    red=error), confocal's tab-icon colour map applied to a card.  Start / Stop act
    here directly; the full param form is in the node's Edit tab
    (:class:`LogicNodeEditor`)."""

    edit_requested = QtCore.pyqtSignal(object)     # "Edit" -> open the node's Edit tab
    remove_requested = QtCore.pyqtSignal(object)
    start_requested = QtCore.pyqtSignal(object)    # "Start" -> build + run the node
    stop_requested = QtCore.pyqtSignal(object)     # "Stop"  -> stop the node

    # confocal gui_combine colour map (INIT=grey / RUNNING=green / STOP/ERROR=red).
    STATE_COLORS = {"stopped": GREY, "running": GREEN, "error": RED}

    def __init__(self, node: LogicNodeConfig, parent=None):
        super().__init__(parent)
        self.node = node
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8))
        outer.setSpacing(scaled_px(4, minimum=3))
        # --- top line: status + name + (kind) + Start / Stop / Edit / Remove --------
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(scaled_px(10, minimum=6))
        self.dot = FluentStatusDot(size=14)
        self.dot.set_color(GREY)
        self.name_label = FluentLabel(node.title)
        self.kind_label = FluentLabel(f"({node.kind})")
        self.kind_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.status_label = FluentLabel("stopped")
        self.status_label.setStyleSheet(f"color: {GREY}; background: transparent; border: none;")
        self.start_button = FluentButton("Start", color=GREEN)
        self.start_button.setFixedWidth(scaled_px(60, minimum=48))
        self.start_button.clicked.connect(lambda: self.start_requested.emit(self))
        self.stop_button = FluentButton("Stop", color=ORANGE)
        self.stop_button.setFixedWidth(scaled_px(56, minimum=46))
        self.stop_button.clicked.connect(lambda: self.stop_requested.emit(self))
        self.stop_button.setEnabled(False)
        edit_button = FluentButton("Edit", color=ACCENT)
        edit_button.setFixedWidth(scaled_px(56, minimum=46))
        edit_button.clicked.connect(lambda: self.edit_requested.emit(self))
        remove = FluentButton("Remove", color=GREY)
        remove.setFixedWidth(scaled_px(82, minimum=66))
        remove.clicked.connect(lambda: self.remove_requested.emit(self))
        top.addWidget(self.dot, 0)
        top.addWidget(self.name_label, 0)
        top.addWidget(self.kind_label, 0)
        top.addWidget(self.status_label, 1)
        for b in (self.start_button, self.stop_button, edit_button, remove):
            top.addWidget(b, 0)
        outer.addLayout(top)
        # --- published-signals legend: one signal per line (name | shape | meaning) ---
        # Monospace so the name/shape columns ALIGN down the rows (a readable table, not a
        # run-on line).
        self.publishes_label = FluentLabel("")
        # WRAP, never extend the row horizontally: a logic-node card lives in a vertical list with NO
        # horizontal scroll (#2).  The publishes legend is name + shape only (short, fits) -- the longer
        # per-signal meaning lives in the tooltip, so nothing forces the card wider than the column.
        self.publishes_label.setWordWrap(True)
        self.publishes_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none; "
            "font-family: Consolas, 'DejaVu Sans Mono', monospace;")
        outer.addWidget(self.publishes_label)

    def set_state(self, state: str, *, status: str = "") -> None:
        """Reflect the node's run state on the dot + status text + Start/Stop enable.

        Change-gated (cache the last ``(state, status)``): a steady tick calls this on every
        running row, so re-setText / re-setStyleSheet / re-setEnabled an UNCHANGED row every tick
        is pure churn (#4-E) -- like the ``set_publishes`` text gate."""
        key = (state, status)
        if key == getattr(self, "_state_key", None):
            return
        self._state_key = key
        self.dot.set_color(self.STATE_COLORS.get(state, GREY))
        self.status_label.setText(status or state)
        colour = RED if state == "error" else GREY
        self.status_label.setStyleSheet(f"color: {colour}; background: transparent; border: none;")
        running = state == "running"
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def set_publishes(self, rows) -> None:
        """Show the node's outputs as a SHORT table -- ONE signal per line, ``name`` + ``shape`` only::

            publishes:
              occupied   (35,)
              rate       scalar

        The per-signal MEANING goes in the label's tooltip (hover), NOT inline -- so the card never
        grows wider than its column and the Logic list needs no horizontal scroll (#2).  ``rows`` is
        ``[(name, shape, description)]`` (shapes AUTO-EXTRACTED via ``shape_text.describe_shape``; meanings
        from the node's ``output_specs``); a pending shape (``—``) just means no value yet."""
        rows = list(rows)
        if rows:
            name_w = max(len(str(n)) for n, _, _ in rows)
            shape_w = max(len(str(s)) for _, s, _ in rows)
            lines = [f"  {str(n):<{name_w}}  {str(s):<{shape_w}}".rstrip() for n, s, _ in rows]
            text = "publishes:\n" + "\n".join(lines)
            tip = "\n".join(f"{n} {s} — {d}" for n, s, d in rows if d)   # meanings on hover, off the card
        else:
            text, tip = "publishes: (nothing on the hub)", ""
        if text != self.publishes_label.text():       # skip churn: shapes refresh each tick
            self.publishes_label.setText(text)
            self.publishes_label.setToolTip(tip)
