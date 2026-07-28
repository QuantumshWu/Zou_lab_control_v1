"""Qt projection of the frontend-owned typed :class:`ViewSpec`."""

from __future__ import annotations

from dataclasses import replace

from PyQt5 import QtCore, QtWidgets

from zlc_data import AxisSourceRef
from zlc_frontend.figure import (
    AxisViewRole,
    FixedIndex,
    ViewSpec,
    evaluate_axis,
    validate_view_spec,
)
from zlc_frontend.figure.contract import _resolve_selected_point_ordinals

from .fluent import FluentComboBox, FluentSettingRow, signals_blocked

__all__ = ["ViewSpecEditor"]


def _coordinate_text(axis, position: int) -> str:
    coordinate = axis.coordinates[int(position)]
    source_index = axis.indices[int(position)]
    text = str(coordinate)
    if axis.unit:
        text = f"{text} {axis.unit}"
    return f"{int(source_index)}: {text}"


def _grid_choice_text(schema, sources, indices) -> str:
    topology = schema.grid_topology
    assert topology is not None
    labels = []
    for source, index in zip(sources, indices, strict=True):
        assert source.axis_id is not None
        position = topology.dimension_ids.index(source.axis_id)
        column = schema.point_table.column(source.axis_id)
        text = str(topology.coordinate_domains[position][index])
        if column.unit:
            text = f"{text} {column.unit}"
        labels.append(f"{column.name}={text}")
    return ", ".join(labels)


class ViewSpecEditor(QtWidgets.QWidget):
    """Edit only the named axis selections an operator can actually change.

    X/Y, reduction, batch, facet, and automatic-latest roles are resolved
    presentation state, not controls.  Showing those as disabled combo boxes
    made an internal ``ViewSpec`` look like a broken form (for example
    ``Reduce``, ``ROI X`` and ``ROI Y`` rows that could not be edited).  Repeat
    and Grid facet have their own explicit controls; this editor owns only a
    finite ``FixedIndex`` choice on a ``SELECTED`` source.

    The owner supplies the exact effective ``ViewSpec`` produced by frontend
    policy.  Rows are reconciled by AxisSourceRef: ordinary data revisions update
    nothing, while a true schema change adds, removes, or reorders only the
    affected editable rows.
    """

    viewChanged = QtCore.pyqtSignal(object)

    def __init__(self, *, label_width: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self._label_width = label_width
        self._schema = None
        self._view = None
        self._rows: dict[
            tuple[AxisSourceRef, ...],
            tuple[FluentSettingRow, FluentComboBox],
        ] = {}
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
        editable = []
        grid_bindings = tuple(
            binding
            for binding in view.source_bindings
            if binding.source.kind == AxisSourceRef.GRID_DIMENSION
            and binding.role is AxisViewRole.SELECTED
            and isinstance(binding.selector, FixedIndex)
        )
        if grid_bindings:
            topology = schema.grid_topology
            if topology is None:
                raise ValueError("selected Grid source is absent from GridTopology")
            selected = {binding.source: binding for binding in grid_bindings}
            grid_sources = tuple(
                AxisSourceRef.grid_dimension(dimension_id)
                for dimension_id in topology.dimension_ids
                if AxisSourceRef.grid_dimension(dimension_id) in selected
            )
            base_ordinals = _resolve_selected_point_ordinals(
                schema,
                view,
                ignore_selected_sources=grid_sources,
            )
            positions = tuple(
                topology.dimension_ids.index(source.axis_id)
                for source in grid_sources
            )
            choices = tuple(
                dict.fromkeys(
                    tuple(
                        topology.row_to_cell[ordinal][position]
                        for position in positions
                    )
                    for ordinal in base_ordinals
                )
            )
            current = tuple(
                selected[source].selector.index for source in grid_sources
            )
            if current not in choices:
                raise ValueError("current Grid selection has no physical row")
            if len(choices) > 1:
                label = (
                    schema.point_table.column(grid_sources[0].axis_id).name
                    if len(grid_sources) == 1
                    else "Grid cell"
                )
                editable.append(
                    (
                        grid_sources,
                        label,
                        tuple(
                            (_grid_choice_text(schema, grid_sources, choice), choice)
                            for choice in choices
                        ),
                        current,
                        "Choose one physical Grid cell; sparse holes are not offered.",
                    )
                )
        for binding in view.source_bindings:
            if (
                binding.role is not AxisViewRole.SELECTED
                or not isinstance(binding.selector, FixedIndex)
            ):
                continue
            source = binding.source
            if source.kind != AxisSourceRef.TENSOR:
                continue
            axes = (schema.repeat_axis, *schema.cell_schema.data_axes)
            declared = next(
                axis for axis in axes if axis.axis_id == source.axis_id
            )
            size = declared.size
            if size > 1:
                axis = evaluate_axis(schema, source, tuple(range(size)))
                editable.append(
                    (
                        (source,),
                        axis.name,
                        tuple(
                            (_coordinate_text(axis, position), (index,))
                            for position, index in enumerate(axis.indices)
                        ),
                        (binding.selector.index,),
                        f"Choose which {axis.name} coordinate this panel displays.",
                    )
                )
        row_keys = {sources for sources, _label, _choices, _current, _tip in editable}
        self._updating = True
        try:
            for sources in tuple(self._rows):
                if sources in row_keys:
                    continue
                row, _combo = self._rows.pop(sources)
                self._layout.removeWidget(row)
                row.hide()
                row.deleteLater()
            for position, (sources, label, choices, current, tooltip) in enumerate(editable):
                pair = self._rows.get(sources)
                if pair is None:
                    combo = FluentComboBox()
                    row = FluentSettingRow(
                        label,
                        combo,
                        label_width=self._label_width,
                        parent=self,
                    )
                    combo.currentIndexChanged.connect(
                        lambda index, sources=sources: self._commit_choice(
                            sources,
                            index,
                        )
                    )
                    self._rows[sources] = (row, combo)
                else:
                    row, combo = pair
                    row.set_label(label, width=self._label_width)
                self._layout.insertWidget(position, row)
                self._seed_combo(choices, current, combo)
                row.setToolTip(tooltip)
                row.show()
        finally:
            self._updating = False
        self._schema = schema
        self._view = view
        self.setVisible(bool(editable))

    @staticmethod
    def _seed_combo(choices, current, combo) -> None:
        if len(choices) <= 1:
            raise ValueError("view editor received a non-editable choice set")
        with signals_blocked(combo):
            combo.clear()
            for label, value in choices:
                combo.addItem(label, value)
            position = next(
                (
                    index
                    for index in range(combo.count())
                    if combo.itemData(index) == current
                ),
                -1,
            )
            if position < 0:
                raise ValueError("current selection is absent from its physical choices")
            combo.setCurrentIndex(position)
            combo.setEnabled(True)

    def _commit_choice(
        self,
        sources: tuple[AxisSourceRef, ...],
        combo_index: int,
    ) -> None:
        if self._updating or self._schema is None or self._view is None:
            return
        pair = self._rows.get(sources)
        if pair is None:
            return
        _row, combo = pair
        selected = combo.itemData(int(combo_index))
        if (
            not isinstance(selected, tuple)
            or len(selected) != len(sources)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in selected
            )
        ):
            return
        current = tuple(
            self._view.binding(source).selector.index for source in sources
        )
        if current == selected:
            return
        replacements = {
            source: replace(
                self._view.binding(source),
                selector=FixedIndex(index),
            )
            for source, index in zip(sources, selected, strict=True)
        }
        bindings = tuple(
            replacements.get(item.source, item)
            for item in self._view.source_bindings
        )
        candidate = replace(self._view, source_bindings=bindings)
        validate_view_spec(self._schema, candidate)
        self._view = candidate
        self.viewChanged.emit(candidate)
