"""Generic projection records consumed by the TaskConsole shell.

Concrete Logic-node schemas, output declarations, forms and presentation
adapters are assembled outside this package and supplied as a closed tuple.
This module contains no Definition catalog and no Camera/Calibration/Occupancy/
MOT/PulseScan dispatch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from zlc_neutral_atom.catalog import (
    DefinitionKey,
    MeasurementDefinition,
    ProcessorDefinition,
    TaskDefinition,
    definition_key_from_tree,
    definition_key_to_tree,
)
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import require_input_specs
from zlc_storage import canonical_text
from zlc_workbench.input_binding import freeze_input_selections


__all__ = [
    "ConsoleCatalogView",
    "ConsoleDefaultPanel",
    "ConsoleNodeSpec",
    "ConsoleSignalDecl",
]


@dataclass(frozen=True)
class ConsoleSignalDecl:
    """Presentation metadata paired with one owner-declared output."""

    declaration: DatasetOutputDeclaration
    short: str
    axis_label: str
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        for name in ("short", "axis_label"):
            canonical_text(getattr(self, name), f"console signal {name}")
        if not isinstance(self.description, str):
            raise TypeError("console signal description must be str")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


def _owner_request_builder(
    callback: Callable[[Mapping[str, object]], object],
    owner_keys: tuple[str, ...],
    editor_keys: tuple[str, ...],
) -> Callable[[Mapping[str, object]], object]:
    """Restrict an ephemeral editor mapping to its owner's authored fields."""

    editor_key_set = frozenset(editor_keys)

    def build(values: Mapping[str, object]) -> object:
        if not isinstance(values, Mapping):
            raise TypeError("console request values must be a mapping")
        unknown = set(values) - editor_key_set
        if unknown:
            raise ValueError(
                "console request values contain unknown fields: "
                f"{tuple(sorted(map(str, unknown)))}"
            )
        return callback(
            {key: values[key] for key in owner_keys if key in values}
        )

    return build


@dataclass(frozen=True)
class ConsoleDefaultPanel:
    """One initial view over an output admitted by the owning capability."""

    output_name: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        output_name = canonical_text(self.output_name, "default panel output name")
        kind = canonical_text(self.kind, "default panel kind")
        if not isinstance(self.params, Mapping):
            raise TypeError("default panel params must be a mapping")
        object.__setattr__(self, "output_name", output_name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True)
class ConsoleNodeSpec:
    """Ephemeral UI projection of one owner-declared Logic-node capability.

    The record is built by a concrete composition attachment.  It is not a
    persistent Definition and its callbacks are never serialized.  Ordinary
    forms use the shared Fluent form renderer; the one earned structured form
    (PulseScan) supplies ``editor_factory`` from its presentation attachment.
    """

    definition: TaskDefinition | MeasurementDefinition | ProcessorDefinition
    title: str
    description: str
    form: object
    declared_outputs: tuple[ConsoleSignalDecl, ...]
    build_request: Callable[[Mapping[str, object]], object]
    input_specs: tuple[object, ...] = ()
    input_fields: tuple[object, ...] = ()
    default_panels: tuple[ConsoleDefaultPanel, ...] = ()
    request_output_declarations: (
        Callable[[object], tuple[DatasetOutputDeclaration, ...]] | None
    ) = None
    request_output_axis_label: str | None = None
    request_output_description: str = ""
    editor_factory: Callable[..., object] | None = None

    def __post_init__(self) -> None:
        if type(self.definition) not in (
            TaskDefinition,
            MeasurementDefinition,
            ProcessorDefinition,
        ):
            raise TypeError("definition must be a Task/Measurement/Processor Definition")
        canonical_text(self.title, "console node title")
        if not isinstance(self.description, str):
            raise TypeError("console node description must be str")

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

        inputs = require_input_specs(self.input_specs)
        fields = tuple(self.input_fields)
        expected_input_keys = tuple(key for spec in inputs for key in spec.field_keys)
        actual_input_keys = tuple(field.key for field in fields)
        if actual_input_keys != expected_input_keys:
            raise ValueError(
                "projected input fields must exactly follow owner input specs"
            )
        if set(form_keys) & set(actual_input_keys):
            raise ValueError("owner form and input field keys overlap")
        if not callable(self.build_request):
            raise TypeError("build_request must be callable")
        object.__setattr__(self, "input_specs", inputs)
        object.__setattr__(self, "input_fields", fields)
        object.__setattr__(
            self,
            "build_request",
            _owner_request_builder(
                self.build_request,
                form_keys,
                form_keys + actual_input_keys,
            ),
        )

        outputs = tuple(self.declared_outputs)
        if any(not isinstance(output, ConsoleSignalDecl) for output in outputs):
            raise TypeError("declared_outputs must contain ConsoleSignalDecl")
        dynamic = self.request_output_declarations
        if dynamic is not None and not callable(dynamic):
            raise TypeError("request_output_declarations must be callable or None")
        if dynamic is not None and outputs:
            raise ValueError("static and request-owned outputs are mutually exclusive")
        if dynamic is None and self.request_output_axis_label is not None:
            raise ValueError("static outputs cannot declare a dynamic axis label")
        if dynamic is not None:
            canonical_text(
                self.request_output_axis_label,
                "request output axis label",
            )
        if not isinstance(self.request_output_description, str):
            raise TypeError("request_output_description must be str")
        if self.editor_factory is not None and not callable(self.editor_factory):
            raise TypeError("editor_factory must be callable or None")
        if self.editor_factory is None and field_keys != form_keys:
            raise ValueError("structured form keys require an editor_factory")
        panels = tuple(self.default_panels)
        if any(not isinstance(panel, ConsoleDefaultPanel) for panel in panels):
            raise TypeError("default_panels must contain ConsoleDefaultPanel")
        if dynamic is None:
            names = {output.name for output in outputs}
            if any(panel.output_name not in names for panel in panels):
                raise ValueError("default panel names an undeclared output")
        object.__setattr__(self, "declared_outputs", outputs)
        object.__setattr__(self, "default_panels", panels)

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

    def freeze_input_selections(self, values: Mapping[str, object]):
        return freeze_input_selections(
            self.input_specs,
            {
                key: values[key]
                for field in self.input_fields
                if (key := field.key) in values
            },
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

    def outputs_for(self, request: object) -> tuple[ConsoleSignalDecl, ...]:
        """Return the exact vocabulary frozen by the owner request."""

        projector = self.request_output_declarations
        if projector is None:
            outputs = self.declared_outputs
        else:
            declarations = tuple(projector(request))
            if any(
                not isinstance(value, DatasetOutputDeclaration)
                for value in declarations
            ):
                raise TypeError(
                    "request output owner returned another declaration type"
                )
            outputs = tuple(
                ConsoleSignalDecl(
                    declaration,
                    declaration.name,
                    self.request_output_axis_label,
                    self.request_output_description,
                )
                for declaration in declarations
            )
        names = tuple(output.name for output in outputs)
        if len(set(names)) != len(names):
            raise ValueError("console output names must be unique")
        return tuple(outputs)


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
