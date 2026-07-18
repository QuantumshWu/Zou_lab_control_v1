"""Reusable named-axis navigation over one immutable ``AxisLayout``."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data import AxisLayout, AxisSpec
from zlc_storage import canonical_text

from .fluent import (
    FluentButton,
    FluentFormGrid,
    FluentLabel,
    FluentSpinBox,
    GREY,
    signals_blocked,
)


class AxisLayoutNavigator(QtWidgets.QWidget):
    """Edit one exact logical coordinate and move in physical storage order.

    The widget owns only Qt state.  It does not construct a domain Selection,
    guess an axis role, or fill sparse holes.  Hosts translate the emitted
    all-axis index tuple into their own typed display request.
    """

    candidateChanged = QtCore.pyqtSignal()
    activated = QtCore.pyqtSignal(object)

    def __init__(
        self,
        axes: tuple[AxisSpec, ...],
        layout: AxisLayout,
        *,
        object_prefix: str,
        action_text: str,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        prepared_axes = tuple(axes)
        if any(not isinstance(axis, AxisSpec) for axis in prepared_axes):
            raise TypeError("axes must contain AxisSpec values")
        if len({axis.axis_id for axis in prepared_axes}) != len(prepared_axes):
            raise ValueError("navigator axes must have unique AxisId values")
        if not isinstance(layout, AxisLayout):
            raise TypeError("layout must be AxisLayout")
        if layout.logical_shape != tuple(axis.size for axis in prepared_axes):
            raise ValueError("navigator layout shape differs from axes")
        self._axes = prepared_axes
        self._axis_layout = layout
        self._prefix = canonical_text(object_prefix, "object_prefix")
        self._interaction_enabled = True

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        form = FluentFormGrid(self)
        form.setObjectName(f"{self._prefix}AxisForm")
        controls = []
        for position, axis in enumerate(prepared_axes):
            spin = FluentSpinBox(form)
            spin.setObjectName(f"{self._prefix}Axis_{position}")
            spin.setProperty("axisId", axis.axis_id.value)
            coordinate = FluentLabel("", form)
            coordinate.setObjectName(f"{self._prefix}Coordinate_{position}")
            if axis.size == 1:
                spin.setRange(0, 0)
                spin.setValue(0)
                spin.setEnabled(False)
            else:
                spin.setRange(-1, axis.size - 1)
                spin.setSpecialValueText("Select…")
                spin.setValue(-1)
            form.add_row(axis.name, spin, coordinate)
            controls.append((axis, spin, coordinate))
        self._controls = tuple(controls)
        outer.addWidget(form)

        self.previous_button = FluentButton("Previous", self, color=GREY)
        self.previous_button.setObjectName(f"{self._prefix}Previous")
        self.action_button = FluentButton(
            canonical_text(action_text, "action_text"),
            self,
        )
        self.action_button.setObjectName(f"{self._prefix}Action")
        self.next_button = FluentButton("Next", self, color=GREY)
        self.next_button.setObjectName(f"{self._prefix}Next")
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.previous_button)
        buttons.addWidget(self.action_button)
        buttons.addWidget(self.next_button)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        for _axis, spin, _label in self._controls:
            spin.valueChanged.connect(self._changed)
        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.action_button.clicked.connect(self._activate)
        self.next_button.clicked.connect(lambda: self._move(1))
        self._refresh()

    @property
    def indices(self) -> tuple[int, ...] | None:
        values = tuple(spin.value() for _axis, spin, _label in self._controls)
        return None if any(value < 0 for value in values) else values

    @property
    def storage_index(self) -> int | None:
        indices = self.indices
        if indices is None:
            return None
        try:
            return self._axis_layout.storage_index(indices)
        except KeyError:
            return None

    def set_indices(self, indices: tuple[int, ...]) -> None:
        multi = tuple(indices)
        self._axis_layout.storage_index(multi)
        spins = tuple(spin for _axis, spin, _label in self._controls)
        with signals_blocked(*spins):
            for spin, index in zip(spins, multi, strict=True):
                spin.setValue(index)
        self._refresh()
        self.candidateChanged.emit()

    def set_storage_index(self, storage_index: int) -> None:
        self.set_indices(self._axis_layout.multi_index(storage_index))

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        self._refresh()

    def _changed(self, _value: int) -> None:
        self._refresh()
        self.candidateChanged.emit()

    def _refresh(self) -> None:
        for axis, spin, label in self._controls:
            index = spin.value()
            if index < 0:
                label.setText("not selected")
            else:
                unit = "" if axis.unit is None else f" {axis.unit}"
                label.setText(f"{axis.coordinate_at(index)}{unit} · index {index}")
            spin.setEnabled(self._interaction_enabled and axis.size > 1)

        indices = self.indices
        storage = None
        if indices is not None:
            try:
                storage = self._axis_layout.storage_index(indices)
            except KeyError:
                pass
        valid = self._interaction_enabled and storage is not None
        self.action_button.setEnabled(valid)
        self.previous_button.setEnabled(valid and storage > 0)
        self.next_button.setEnabled(
            valid and storage + 1 < self._axis_layout.storage_size
        )

    def _activate(self) -> None:
        indices = self.indices
        if self._interaction_enabled and indices is not None and self.storage_index is not None:
            self.activated.emit(indices)

    def _move(self, delta: int) -> None:
        if delta not in (-1, 1):
            raise ValueError("navigator delta must be -1 or 1")
        storage = self.storage_index
        if not self._interaction_enabled or storage is None:
            return
        target = storage + delta
        if not 0 <= target < self._axis_layout.storage_size:
            return
        self.set_storage_index(target)
        indices = self.indices
        assert indices is not None
        self.activated.emit(indices)


__all__ = ["AxisLayoutNavigator"]
