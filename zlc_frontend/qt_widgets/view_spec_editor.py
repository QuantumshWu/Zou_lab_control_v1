"""Qt authoring surface for the frontend-owned typed :class:`ViewSpec`."""

from __future__ import annotations

from dataclasses import replace

from PyQt5 import QtCore, QtWidgets

from zlc_data import AxisSourceRef
from zlc_frontend.figure import (
    GRID_INTENTS,
    AxisViewRole,
    FixedIndex,
    SourceViewBinding,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
    evaluate_axis,
    grid_facet_source,
    grid_facet_sources,
    resolve_grid_view,
    validate_view_spec,
)
from zlc_frontend.figure.contract import (
    _dataset_sources,
    _resolve_selected_point_ordinals,
    _source_cardinality,
    _source_name,
)
from zlc_frontend.figure.suggest import (
    _repeat_authoring_candidates,
    _view_authoring_candidates,
    _view_preferences_from_spec,
)
from zlc_frontend.form import FormChoice, FormFieldProps, FormSpec

from .form import FluentParameterForm

__all__ = ["ViewSpecEditor"]


_GRID_INTENT_LABELS = {
    ViewIntent.CURVE: "1d",
    ViewIntent.HISTOGRAM: "hist",
    ViewIntent.IMAGE: "2d",
}
_DISPLAY_ROLE_LABELS = {
    AxisViewRole.X: "x",
    AxisViewRole.IMAGE_X: "image x",
    AxisViewRole.IMAGE_Y: "image y",
}
def _choice_field(key, label, choices, current, description=""):
    choices = tuple(choices)
    unavailable = not choices
    return FormFieldProps(
        key,
        "choice",
        label,
        default=None if unavailable else current,
        choices=tuple(FormChoice(text, value) for text, value in choices),
        description=description,
        required=unavailable,
        unavailable_reason=(
            f"No declared {label} is available for this Dataset."
            if unavailable
            else ""
        ),
    )


def _coordinate_text(axis, position: int) -> str:
    coordinate = axis.coordinates[int(position)]
    source_index = axis.indices[int(position)]
    text = f"{coordinate} {axis.unit}" if axis.unit else str(coordinate)
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
        labels.append(f"{column.name}={text} {column.unit}" if column.unit else f"{column.name}={text}")
    return ", ".join(labels)


def _source_key(source: AxisSourceRef) -> str:
    axis = "" if source.axis_id is None else source.axis_id.value
    return f"{source.kind}:{axis}"


def _repeat_text(binding: SourceViewBinding) -> str:
    if binding.role is AxisViewRole.REDUCED:
        return binding.reduction.method.value.title()
    if binding.role is AxisViewRole.SELECTED:
        return "Index" if isinstance(binding.selector, FixedIndex) else "Latest"
    return "Overlay" if binding.role is AxisViewRole.BATCH else "Pool as samples"


class ViewSpecEditor(QtWidgets.QWidget):
    """Expose resolver-owned typed view choices through the keyed Fluent form."""

    viewChanged = QtCore.pyqtSignal(object)

    def __init__(self, *, label_width: int | None = None, parent=None) -> None:
        super().__init__(parent)
        self._label_width = label_width
        self._schema = None
        self._view: ViewSpec | None = None
        self._intent: ViewIntent | None = None
        self._faceted = False
        self._preferences = ViewPreferences()
        self._controls_dirty = True
        self._form: FluentParameterForm | None = None
        self._selection_fields: dict[str, tuple[AxisSourceRef, ...]] = {}
        self._role_fields = {}
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._layout = layout
        self.hide()

    def reconcile(
        self,
        schema,
        view: ViewSpec | None,
        *,
        intent: ViewIntent | None = None,
        faceted: bool = False,
    ) -> None:
        if schema is None:
            self._schema = self._view = self._intent = None
            self._faceted = False
            self._preferences = ViewPreferences()
            self._controls_dirty = False
            if self._form is not None:
                self._form.hide()
            self.hide()
            return
        if view is not None:
            if not isinstance(view, ViewSpec):
                raise TypeError("view editor requires ViewSpec or None")
            validate_view_spec(schema, view)
            if intent is not None and intent is not view.intent:
                raise ValueError("view and requested intent disagree")
            intent = view.intent
            faceted = any(binding.role is AxisViewRole.FACET
                          for binding in view.source_bindings)
        elif intent is not None and not isinstance(intent, ViewIntent):
            raise TypeError("intent must be ViewIntent or None")
        if not isinstance(faceted, bool):
            raise TypeError("faceted must be bool")
        if view is None and intent is None and not faceted:
            raise ValueError("an unresolved non-Grid view requires its intent")
        same_schema = (
            self._schema is not None
            and self._schema.fingerprint == schema.fingerprint
        )
        if (same_schema and self._view == view and self._intent is intent
                and self._faceted is faceted and not self._controls_dirty):
            return
        same_draft = (
            same_schema
            and self._view is None
            and self._intent is intent
            and self._faceted is faceted
        )
        self._schema, self._view = schema, view
        self._intent, self._faceted = intent, faceted
        if view is not None:
            self._preferences = _view_preferences_from_spec(schema, view)
        elif not same_draft:
            self._preferences = ViewPreferences()

        fields, values, selections = self._project_form()
        self._selection_fields = selections
        self._controls_dirty = False
        if not fields:
            if self._form is not None:
                self._form.hide()
            self.hide()
            return
        spec = FormSpec(tuple(fields))
        if self._form is None:
            self._form = FluentParameterForm(
                spec,
                values,
                self,
                label_width=self._label_width,
            )
            self._form.changed.connect(self._commit_field)
            self._layout.addWidget(self._form)
        else:
            self._form.reconcile(spec, values)
        self._form.show()
        self.show()

    def _project_form(self):
        assert self._schema is not None
        fields, values = [], {}

        def add(field, value):
            fields.append(field)
            values[field.key] = value

        if self._faceted:
            facet = None if self._view is None else grid_facet_source(self._view)
            if self._intent is not None:
                choices = self._grid_facet_choices(self._intent)
                add(
                    _choice_field("grid.facet", "facet", choices, facet,
                                  "Expand exactly one declared source into Grid cells."),
                    facet,
                )
            intents = tuple(
                (_GRID_INTENT_LABELS[candidate], candidate)
                for candidate in GRID_INTENTS
                if grid_facet_sources(self._schema, candidate, current_view=self._view)
            )
            add(
                _choice_field("grid.intent", "sub plot", intents, self._intent,
                              "What each Grid cell draws under the same typed ViewSpec."),
                self._intent,
            )

        if self._view is not None:
            repeat = AxisSourceRef.tensor(self._schema.repeat_axis.axis_id)
            bindings = _repeat_authoring_candidates(self._schema, self._view)
            choices = tuple((_repeat_text(binding), binding) for binding in bindings)
            if len(choices) > 1:
                current = self._view.binding(repeat)
                add(_choice_field("repeat", "repeat", choices, current), current)

        self._role_fields = {}
        rows = () if self._intent is None else _view_authoring_candidates(
            self._schema,
            self._intent,
            current_view=self._view,
            preferences=self._preferences,
            faceted=self._faceted,
        )
        for role, current, candidates in rows:
            key = f"role.{role.value}"
            label = _DISPLAY_ROLE_LABELS.get(role, role.value.lower())
            self._role_fields[key] = {
                value: (preferences, candidate)
                for value, preferences, candidate in candidates
            }
            labelled = tuple(
                (self._role_value_text(value), value)
                for value, _preferences, _candidate in candidates
            )
            add(_choice_field(key, label, labelled, current,
                              f"Choose the declared source used as {role.value}."), current)

        selections = {}
        for sources, label, choices, current, tooltip in self._selection_projection():
            key = "selected." + "|".join(_source_key(source) for source in sources)
            selections[key] = sources
            add(_choice_field(key, label, choices, current, tooltip), current)
        return fields, values, selections

    def _role_value_text(self, value) -> str:
        if not value:
            return "None"
        sources = value if isinstance(value, tuple) else (value,)
        all_sources = _dataset_sources(self._schema)
        names = tuple(_source_name(self._schema, source) for source in all_sources)
        labels = []
        for source in sources:
            name = _source_name(self._schema, source)
            labels.append(f"{name} [{_source_key(source)}]"
                          if names.count(name) > 1 else name)
        return " + ".join(labels)

    def _selection_projection(self):
        if self._schema is None or self._view is None:
            return ()
        editable = []
        grid_bindings = tuple(
            binding
            for binding in self._view.source_bindings
            if binding.source.kind == AxisSourceRef.GRID_DIMENSION
            and binding.role is AxisViewRole.SELECTED
            and isinstance(binding.selector, FixedIndex)
        )
        if grid_bindings:
            topology = self._schema.grid_topology
            if topology is None:
                raise ValueError("selected Grid source is absent from GridTopology")
            selected = {binding.source: binding for binding in grid_bindings}
            sources = tuple(
                AxisSourceRef.grid_dimension(axis_id)
                for axis_id in topology.dimension_ids
                if AxisSourceRef.grid_dimension(axis_id) in selected
            )
            ordinals = _resolve_selected_point_ordinals(
                self._schema,
                self._view,
                ignore_selected_sources=sources,
            )
            positions = tuple(topology.dimension_ids.index(source.axis_id) for source in sources)
            choices = tuple(
                dict.fromkeys(
                    tuple(topology.row_to_cell[ordinal][position] for position in positions)
                    for ordinal in ordinals
                )
            )
            current = tuple(selected[source].selector.index for source in sources)
            if current not in choices:
                raise ValueError("current Grid selection has no physical row")
            if len(choices) > 1:
                label = (
                    self._schema.point_table.column(sources[0].axis_id).name
                    if len(sources) == 1
                    else "Grid cell"
                )
                editable.append(
                    (
                        sources,
                        label,
                        tuple((_grid_choice_text(self._schema, sources, item), item) for item in choices),
                        current,
                        "Choose one physical Grid cell; sparse holes are not offered.",
                    )
                )
        axes = (self._schema.repeat_axis, *self._schema.cell_schema.data_axes)
        by_id = {axis.axis_id: axis for axis in axes}
        for binding in self._view.source_bindings:
            if (
                binding.source.kind != AxisSourceRef.TENSOR
                or binding.role is not AxisViewRole.SELECTED
                or not isinstance(binding.selector, FixedIndex)
            ):
                continue
            declared = by_id[binding.source.axis_id]
            if declared.size <= 1:
                continue
            axis = evaluate_axis(self._schema, binding.source, tuple(range(declared.size)))
            editable.append(
                (
                    (binding.source,),
                    axis.name,
                    tuple(
                        (_coordinate_text(axis, position), (index,))
                        for position, index in enumerate(axis.indices)
                    ),
                    (binding.selector.index,),
                    f"Choose which {axis.name} coordinate this panel displays.",
                )
            )
        return tuple(editable)

    @staticmethod
    def _replace_binding(view: ViewSpec, binding: SourceViewBinding) -> ViewSpec:
        return replace(
            view,
            source_bindings=tuple(
                binding if item.source == binding.source else item
                for item in view.source_bindings
            ),
        )

    def _grid_facet_choices(self, intent: ViewIntent):
        assert self._schema is not None
        sources = grid_facet_sources(self._schema, intent, current_view=self._view)
        names = tuple(_source_name(self._schema, source) for source in sources)
        return tuple(
            (
                f"{name} ({_source_cardinality(self._schema, source)})"
                + (f" [{_source_key(source)}]" if names.count(name) > 1 else ""),
                source,
            )
            for source, name in zip(sources, names, strict=True)
        )

    def _commit_field(self, key: str) -> None:
        if self._form is None or self._schema is None:
            return
        value = self._form.read_value(key)
        if key == "grid.intent":
            self._commit_grid_intent(value)
        elif key == "grid.facet":
            self._commit_grid_facet(value)
        elif key == "repeat":
            self._commit_repeat(value)
        elif key in self._role_fields:
            self._commit_role(key, value)
        elif key in self._selection_fields:
            self._commit_selection(self._selection_fields[key], value)

    def _commit_role(self, key, value) -> None:
        commit = self._role_fields.get(key, {}).get(value)
        if commit is None:
            return
        preferences, candidate = commit
        self._preferences = preferences
        if candidate is None:
            if self._view is None:
                self._controls_dirty = True
                self.reconcile(
                    self._schema,
                    None,
                    intent=self._intent,
                    faceted=self._faceted,
                )
            return
        self._emit(candidate)

    def _commit_grid_intent(self, intent) -> None:
        if not isinstance(intent, ViewIntent) or intent not in GRID_INTENTS:
            return
        current = None if self._view is None else grid_facet_source(self._view)
        legal = tuple(value for _label, value in self._grid_facet_choices(intent))
        preferred = current if current in legal else legal[0] if len(legal) == 1 else None
        self._intent = intent
        if preferred is not None:
            self._commit_grid_view(intent, preferred)
            return
        self._view = None
        self._controls_dirty = True
        self.reconcile(self._schema, None, intent=intent, faceted=True)

    def _commit_grid_facet(self, source) -> None:
        if self._intent in GRID_INTENTS and isinstance(source, AxisSourceRef):
            self._commit_grid_view(self._intent, source)

    def _commit_grid_view(self, intent: ViewIntent, source: AxisSourceRef) -> None:
        suggestion = resolve_grid_view(
            self._schema,
            intent,
            source,
            current_view=self._view,
        )
        if suggestion.spec is None:
            raise ValueError("selected Grid cell/facet pair has no complete view")
        self._emit(suggestion.spec)

    def _commit_selection(self, sources, selected) -> None:
        if self._view is None or not isinstance(selected, tuple):
            return
        if len(selected) != len(sources) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in selected
        ):
            return
        replacements = {
            source: replace(self._view.binding(source), selector=FixedIndex(index))
            for source, index in zip(sources, selected, strict=True)
        }
        candidate = replace(
            self._view,
            source_bindings=tuple(
                replacements.get(binding.source, binding)
                for binding in self._view.source_bindings
            ),
        )
        validate_view_spec(self._schema, candidate)
        self._emit(candidate)

    def _commit_repeat(self, binding) -> None:
        if self._view is None or not isinstance(binding, SourceViewBinding):
            return
        candidate = self._replace_binding(self._view, binding)
        validate_view_spec(self._schema, candidate)
        self._emit(candidate)

    def _emit(self, candidate: ViewSpec) -> None:
        if candidate == self._view:
            return
        self._view = candidate
        self._intent = candidate.intent
        self._preferences = _view_preferences_from_spec(self._schema, candidate)
        self._controls_dirty = True
        self.viewChanged.emit(candidate)
