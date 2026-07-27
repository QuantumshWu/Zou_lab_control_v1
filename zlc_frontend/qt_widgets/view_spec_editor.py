"""Qt projection of the frontend-owned typed :class:`ViewSpec`."""

from __future__ import annotations

from dataclasses import replace

from PyQt5 import QtCore, QtWidgets

from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    FixedIndex,
    ViewSpec,
    dataset_axes,
    validate_view_spec,
)

from .fluent import FluentComboBox, FluentSettingRow, signals_blocked

__all__ = ["ViewSpecEditor"]


def _coordinate_text(axis, index: int) -> str:
    coordinate = axis.coordinate_at(int(index))
    text = str(coordinate)
    if axis.unit:
        text = f"{text} {axis.unit}"
    return f"{int(index)}: {text}"


class ViewSpecEditor(QtWidgets.QWidget):
    """Edit only the named axis selections an operator can actually change.

    X/Y, reduction, batch, facet, and automatic-latest roles are resolved
    presentation state, not controls.  Showing those as disabled combo boxes
    made an internal ``ViewSpec`` look like a broken form (for example
    ``Reduce``, ``ROI X`` and ``ROI Y`` rows that could not be edited).  Repeat
    and Grid facet have their own explicit controls; this editor owns only a
    finite ``FixedIndex`` choice on a ``SELECTED``/``SLIDER`` axis.

    The owner supplies the exact effective ``ViewSpec`` produced by frontend
    policy.  Rows are reconciled by AxisId: ordinary data revisions update
    nothing, while a true schema change adds, removes, or reorders only the
    affected editable rows.
    """

    viewChanged = QtCore.pyqtSignal(object)

    def __init__(self, *, label_width: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self._label_width = label_width
        self._schema = None
        self._view = None
        self._rows: dict[object, tuple[FluentSettingRow, FluentComboBox]] = {}
        self._updating = False
        self._layout = QtWidgets.QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self.hide()

    def reconcile(self, schema, view: ViewSpec | None) -> None:
        if schema is None or view is None:
            self._schema = None
            self._view = None
            for row, _combo in self._rows.values():
                row.hide()
            self.hide()
            return
        if not isinstance(view, ViewSpec):
            raise TypeError("view editor requires ViewSpec or None")
        validate_view_spec(schema, view)
        axes = tuple(
            axis
            for axis in dataset_axes(schema)
            if (
                (binding := view.binding(axis.axis_id)).role
                in (AxisViewRole.SLIDER, AxisViewRole.SELECTED)
                and isinstance(binding.selector, FixedIndex)
                and axis.size > 1
            )
        )
        axis_ids = {axis.axis_id for axis in axes}
        self._updating = True
        try:
            for axis_id in tuple(self._rows):
                if axis_id in axis_ids:
                    continue
                row, _combo = self._rows.pop(axis_id)
                self._layout.removeWidget(row)
                row.hide()
                row.deleteLater()
            for position, axis in enumerate(axes):
                pair = self._rows.get(axis.axis_id)
                if pair is None:
                    combo = FluentComboBox()
                    row = FluentSettingRow(
                        axis.name,
                        combo,
                        label_width=self._label_width,
                        parent=self,
                    )
                    combo.currentIndexChanged.connect(
                        lambda index, axis_id=axis.axis_id: self._commit_index(
                            axis_id,
                            index,
                        )
                    )
                    self._rows[axis.axis_id] = (row, combo)
                else:
                    row, combo = pair
                    row.set_label(axis.name, width=self._label_width)
                self._layout.insertWidget(position, row)
                self._seed_combo(axis, view.binding(axis.axis_id), combo)
                row.setToolTip(
                    f"Choose which {axis.name} coordinate this panel displays."
                )
                row.show()
        finally:
            self._updating = False
        self._schema = schema
        self._view = view
        self.setVisible(bool(axes))

    def _seed_combo(self, axis, binding: AxisViewBinding, combo) -> None:
        if (
            binding.role not in (AxisViewRole.SLIDER, AxisViewRole.SELECTED)
            or not isinstance(binding.selector, FixedIndex)
            or axis.size <= 1
        ):
            raise ValueError("view editor received a non-editable axis binding")
        with signals_blocked(combo):
            combo.clear()
            for index in range(axis.size):
                combo.addItem(_coordinate_text(axis, index), index)
            combo.setCurrentIndex(int(binding.selector.index))
            combo.setEnabled(True)

    def _commit_index(self, axis_id, combo_index: int) -> None:
        if self._updating or self._schema is None or self._view is None:
            return
        pair = self._rows.get(axis_id)
        if pair is None:
            return
        _row, combo = pair
        selected = combo.itemData(int(combo_index))
        if isinstance(selected, bool) or not isinstance(selected, int):
            return
        binding = self._view.binding(axis_id)
        if binding.role not in (AxisViewRole.SLIDER, AxisViewRole.SELECTED):
            return
        if isinstance(binding.selector, FixedIndex) and binding.selector.index == selected:
            return
        replacement = replace(binding, selector=FixedIndex(selected))
        bindings = tuple(
            replacement if item.axis_id == axis_id else item
            for item in self._view.axis_bindings
        )
        candidate = replace(self._view, axis_bindings=bindings)
        validate_view_spec(self._schema, candidate)
        self._view = candidate
        self.viewChanged.emit(candidate)
