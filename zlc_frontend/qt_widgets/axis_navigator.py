"""Reusable named-axis navigation over one immutable ``AxisLayout``."""

from __future__ import annotations

from PyQt5 import QtCore, QtWidgets

from zlc_data import AxisLayout, AxisSpec
from zlc_storage import canonical_text

from ..fit_projection import coordinate_label

from .fluent import (
    FluentButton,
    FluentFormGrid,
    FluentLabel,
    FluentLineEdit,
    FluentSpinBox,
    GREY,
    signals_blocked,
)


_QT_INT_MAXIMUM = (1 << 31) - 1


class _ExactIndexEdit(FluentLineEdit):
    """A text editor whose committed value remains an exact Python integer."""

    valueChanged = QtCore.pyqtSignal(object)

    def __init__(self, maximum: int, parent: QtWidgets.QWidget) -> None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 0:
            raise ValueError("exact index maximum must be a non-negative integer")
        super().__init__("", parent)
        self._maximum = maximum
        self._committed_value: int | None = None
        # QLineEdit defaults to 32767 characters.  That silent presentation
        # truncation contradicts this control's exact-index contract; use the
        # Qt property type's real limit instead of inventing an application cap.
        self.setMaxLength(_QT_INT_MAXIMUM)
        self.setPlaceholderText("Select…")
        self.textEdited.connect(self._edited)
        self.editingFinished.connect(self._canonicalize_display)

    @staticmethod
    def _parse(text: str) -> int | None:
        if not text:
            return None
        if text.startswith(("0x", "0X")):
            digits = text[2:]
            if not digits or any(character not in "0123456789abcdefABCDEF" for character in digits):
                return None
            return int(digits, 16)
        if not text.isascii() or not text.isdecimal():
            return None
        try:
            return int(text, 10)
        except ValueError:
            return None

    def _edited(self, text: str) -> None:
        candidate = self._parse(text)
        self._committed_value = (
            candidate
            if candidate is not None and candidate <= self._maximum
            else None
        )
        self.valueChanged.emit(self._committed_value)

    def _canonicalize_display(self) -> None:
        value = self._committed_value
        if value is None:
            return
        self.setText(coordinate_label(value))

    def value(self) -> int | None:
        return self._committed_value

    def setValue(self, value: int) -> None:  # noqa: N802 - QSpinBox-compatible seam
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= self._maximum
        ):
            raise ValueError("exact index value is outside its axis")
        self._committed_value = value
        self.setText(coordinate_label(value))
        self.valueChanged.emit(value)


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
        prepared_axes = tuple(axes)
        if any(not isinstance(axis, AxisSpec) for axis in prepared_axes):
            raise TypeError("axes must contain AxisSpec values")
        if len({axis.axis_id for axis in prepared_axes}) != len(prepared_axes):
            raise ValueError("navigator axes must have unique AxisId values")
        if not isinstance(layout, AxisLayout):
            raise TypeError("layout must be AxisLayout")
        if layout.logical_shape != tuple(axis.size for axis in prepared_axes):
            raise ValueError("navigator layout shape differs from axes")
        super().__init__(parent)
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
            if axis.size - 1 <= _QT_INT_MAXIMUM:
                control = FluentSpinBox(form)
            else:
                control = _ExactIndexEdit(axis.size - 1, form)
            control.setObjectName(f"{self._prefix}Axis_{position}")
            control.setProperty("axisId", axis.axis_id.value)
            coordinate = FluentLabel("", form)
            coordinate.setObjectName(f"{self._prefix}Coordinate_{position}")
            if axis.size == 1:
                assert isinstance(control, FluentSpinBox)
                control.setRange(0, 0)
                control.setValue(0)
                control.setEnabled(False)
            elif isinstance(control, FluentSpinBox):
                control.setRange(-1, axis.size - 1)
                control.setSpecialValueText("Select…")
                control.setValue(-1)
            else:
                control.setText("")
            form.add_row(coordinate_label(axis.name), control, coordinate)
            controls.append((axis, control, coordinate))
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

        for _axis, control, _label in self._controls:
            control.valueChanged.connect(self._changed)
        self.previous_button.clicked.connect(lambda: self._move(-1))
        self.action_button.clicked.connect(self._activate)
        self.next_button.clicked.connect(lambda: self._move(1))
        self._refresh()

    @property
    def indices(self) -> tuple[int, ...] | None:
        values = tuple(
            control.value()
            for _axis, control, _label in self._controls
        )
        return (
            None
            if any(value is None or value < 0 for value in values)
            else tuple(int(value) for value in values)
        )

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
        controls = tuple(control for _axis, control, _label in self._controls)
        with signals_blocked(*controls):
            for control, index in zip(controls, multi, strict=True):
                control.setValue(index)
        self._refresh()
        self.candidateChanged.emit()

    def set_storage_index(self, storage_index: int) -> None:
        self.set_indices(self._axis_layout.multi_index(storage_index))

    def set_interaction_enabled(self, enabled: bool) -> None:
        self._interaction_enabled = bool(enabled)
        self._refresh()

    def _changed(self, _value: object) -> None:
        self._refresh()
        self.candidateChanged.emit()

    def _refresh(self) -> None:
        for axis, control, label in self._controls:
            index = control.value()
            if index is None or index < 0:
                label.setText("not selected")
            else:
                unit = (
                    ""
                    if axis.unit is None
                    else f" {coordinate_label(axis.unit)}"
                )
                label.setText(
                    f"{coordinate_label(axis.coordinate_at(index))}{unit}"
                    f" · index {coordinate_label(index)}"
                )
            control.setEnabled(self._interaction_enabled and axis.size > 1)

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
