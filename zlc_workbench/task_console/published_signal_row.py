"""A passive typed-signal publisher row for the TaskConsole Logic page."""

from __future__ import annotations

from PyQt5 import QtWidgets

from zlc_frontend.qt_widgets import (
    ElidedLabel,
    FluentFrame,
    FluentLabel,
    GREY,
    PublishedSignalsLegend,
    scaled_px,
)

__all__ = ["PublishedSignalRow"]


class PublishedSignalRow(FluentFrame):
    """Display outputs from a publisher which has no LogicNode lifecycle.

    The row deliberately has no Start/Stop/Edit controls.  It shares the exact
    signal-inventory renderer used by :class:`LogicNodeRow`, while its owner
    supplies only the publisher label, kind, state, and declared outputs.
    """

    def __init__(self, *, title: str, kind: str, state: str, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Preferred,
        )
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(
            scaled_px(12), scaled_px(8), scaled_px(12), scaled_px(8)
        )
        outer.setSpacing(scaled_px(4, minimum=3))

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(scaled_px(10, minimum=6))
        self.name_label = ElidedLabel("")
        self.name_label.setMinimumWidth(0)
        self.name_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.kind_label = FluentLabel("")
        self.kind_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        self.status_label = ElidedLabel("")
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        self.status_label.setStyleSheet(
            f"color: {GREY}; background: transparent; border: none;"
        )
        top.addWidget(self.name_label, 1)
        top.addWidget(self.kind_label, 0)
        top.addWidget(self.status_label, 2)
        outer.addLayout(top)

        self.publishes_label = PublishedSignalsLegend()
        outer.addWidget(self.publishes_label)
        self.set_identity(title=title, kind=kind, state=state)

    def set_identity(self, *, title: str, kind: str, state: str) -> None:
        identity = (str(title), str(kind), str(state))
        if identity == getattr(self, "_identity", None):
            return
        self._identity = identity
        self.name_label.setText(identity[0])
        self.kind_label.setText(f"({identity[1]})")
        self.status_label.setText(identity[2])

    def set_publishes(self, rows) -> None:
        self.publishes_label.set_rows(rows)
