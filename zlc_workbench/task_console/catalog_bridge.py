"""Generic projection records consumed by the TaskConsole shell.

Concrete Logic-node schemas, output declarations, forms and presentation
adapters are assembled outside this package and supplied as a closed tuple.
This module contains no Definition catalog and no Camera/Calibration/Occupancy/
MOT/PulseScan dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from zlc_neutral_atom.catalog import (
    DefinitionKey,
    MeasurementDefinition,
    ProcessorDefinition,
    TaskDefinition,
    definition_key_from_tree,
    definition_key_to_tree,
)
from zlc_neutral_atom.input_spec import require_input_specs
from zlc_neutral_atom.logic_node_declaration import LogicNodeDeclaration
from .input_binding import freeze_input_selections


__all__ = [
    "ConsoleCatalogView",
    "ConsoleNodeSpec",
]


@dataclass(frozen=True, slots=True)
class ConsoleNodeSpec:
    """Ephemeral UI projection of one owner-declared Logic-node capability.

    The record is built by a concrete composition attachment.  It is not a
    persistent Definition and its callbacks are never serialized.  Ordinary
    forms use the shared Fluent form renderer; the one earned structured form
    (PulseScan) supplies ``editor_factory`` from its presentation attachment.
    """

    declaration: LogicNodeDeclaration
    form: object
    input_fields: tuple[object, ...] = ()
    editor_factory: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, LogicNodeDeclaration):
            raise TypeError("declaration must be LogicNodeDeclaration")

        form_keys = tuple(getattr(self.form, "keys", ()))
        form_fields = tuple(getattr(self.form, "fields", ()))
        default_values = getattr(self.form, "default_values", None)
        if not form_keys or not form_fields or not callable(default_values):
            raise TypeError(
                "form must expose fields, keys, and default_values()"
            )
        field_keys = tuple(field.key for field in form_fields)
        if len(set(form_keys)) != len(form_keys) or not set(field_keys) <= set(form_keys):
            raise ValueError("form fields are not a unique subset of its keys")
        owner_keys = tuple(self.declaration.authoring_schema.keys)
        if not set(owner_keys) <= set(form_keys):
            raise ValueError("Workbench form omits an owner authoring field")

        inputs = require_input_specs(self.declaration.input_specs)
        fields = tuple(self.input_fields)
        expected_input_keys = tuple(key for spec in inputs for key in spec.field_keys)
        actual_input_keys = tuple(field.key for field in fields)
        if actual_input_keys != expected_input_keys:
            raise ValueError(
                "projected input fields must exactly follow owner input specs"
            )
        if set(form_keys) & set(actual_input_keys):
            raise ValueError("owner form and input field keys overlap")
        object.__setattr__(self, "input_fields", fields)
        if self.editor_factory is not None and not callable(self.editor_factory):
            raise TypeError("editor_factory must be callable or None")
        if self.editor_factory is None and field_keys != form_keys:
            raise ValueError("structured form keys require an editor_factory")
        if self.editor_factory is None and form_keys != owner_keys:
            raise ValueError("ordinary form keys must exactly match owner authoring fields")

    @property
    def definition(self):
        return self.declaration.definition

    @property
    def title(self) -> str:
        return self.definition.title

    @property
    def description(self) -> str:
        return self.declaration.description

    @property
    def input_specs(self) -> tuple:
        return self.declaration.input_specs

    @property
    def artifact_outputs(self) -> tuple:
        return self.declaration.artifact_outputs

    @property
    def default_views(self) -> tuple:
        return self.declaration.default_views

    @property
    def editor_fields(self) -> tuple:
        return tuple(self.form.fields) + tuple(self.input_fields)

    @property
    def editor_keys(self) -> tuple[str, ...]:
        return tuple(self.form.keys) + tuple(field.key for field in self.input_fields)

    def editor_default_values(self) -> dict[str, object]:
        return {
            **self.form.default_values(),
            **{field.key: field.default for field in self.input_fields},
        }

    def build_request(self, values: Mapping[str, object]) -> object:
        """Freeze only the exact Workbench form leaves through their owner."""

        if not isinstance(values, Mapping):
            raise TypeError("console request values must be a mapping")
        editor_keys = self.editor_keys
        unknown = set(values) - set(editor_keys)
        if unknown:
            raise ValueError(
                "console request values contain unknown fields: "
                f"{tuple(sorted(map(str, unknown)))}"
            )
        form_keys = tuple(self.form.keys)
        # ``base_dir`` is a file-dialog hint only.  Authored project-relative
        # paths remain relative until the domain owner resolves them against
        # its injected WorkspacePaths root during prepare.
        authored = {key: values[key] for key in form_keys if key in values}
        return self.declaration.build_request(
            authored
        )

    def freeze_input_selections(self, values: Mapping[str, object]):
        input_values = {
            key: values[key]
            for field in self.input_fields
            if (key := field.key) in values
        }
        return freeze_input_selections(
            self.input_specs,
            input_values,
        )

    def input_spec_for_signal_field(self, field_key: str):
        matches = tuple(
            spec
            for spec in self.input_specs
            if field_key == getattr(spec, "key", None)
            or field_key == getattr(spec, "producer_key", None)
        )
        if len(matches) > 1:
            raise RuntimeError("one signal field matches multiple input specs")
        return matches[0] if matches else None

    @property
    def name(self) -> str:
        return self.title

    @property
    def key(self) -> DefinitionKey:
        return self.definition.key

    @property
    def kind(self) -> str:
        return {
            TaskDefinition: "task",
            MeasurementDefinition: "measurement",
            ProcessorDefinition: "processor",
        }[type(self.definition)]

    @property
    def definition_tree(self) -> dict[str, object]:
        return definition_key_to_tree(self.key)

    def outputs_for(self, request: object) -> tuple:
        """Delegate the exact owner presentation vocabulary unchanged."""

        return self.declaration.outputs_for(request)


class ConsoleCatalogView:
    """Closed read-only view over explicitly supplied capability specs."""

    def __init__(self, specs) -> None:
        values = tuple(specs)
        if not values:
            raise ValueError("TaskConsole requires at least one capability")
        if any(not isinstance(spec, ConsoleNodeSpec) for spec in values):
            raise TypeError("catalog values must be ConsoleNodeSpec")
        by_key: dict[DefinitionKey, ConsoleNodeSpec] = {}
        for spec in values:
            if spec.key in by_key:
                raise ValueError(f"duplicate console DefinitionKey {spec.key}")
            by_key[spec.key] = spec
        self._specs = values
        self._by_key = MappingProxyType(by_key)

    def specs(self, kind: str | None = None) -> tuple[ConsoleNodeSpec, ...]:
        if kind is None:
            return self._specs
        return tuple(spec for spec in self._specs if spec.kind == kind)

    def spec_for_definition(
        self,
        tree: Mapping[str, object],
    ) -> ConsoleNodeSpec | None:
        return self._by_key.get(definition_key_from_tree(tree))

    def spec_for_key(self, key: DefinitionKey) -> ConsoleNodeSpec | None:
        if not isinstance(key, DefinitionKey):
            return None
        return self._by_key.get(key)
