"""Process-local declarations shared by Logic-node composition surfaces.

The domain owner declares stable Definition, authoring, input, output and
request-binding facts here.  A Workbench may project those facts into its own
widgets and lifecycle host, but no GUI type or runtime registry crosses this
boundary.  Callback-bearing declarations are intentionally not persistence
artifacts; persisted identity remains the DefinitionKey and owner codecs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from zlc_neutral_atom.authoring import AuthoringChoice, AuthoringSchema
from zlc_neutral_atom.catalog import (
    MeasurementDefinition,
    ProcessorDefinition,
    TaskDefinition,
)
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import (
    ArtifactInputSpec,
    NodeInputSpec,
    require_input_specs,
)
from zlc_neutral_atom.node_input import BoundNodeInputs
from zlc_storage import canonical_text


Definition = TaskDefinition | MeasurementDefinition | ProcessorDefinition


@dataclass(frozen=True, slots=True)
class DynamicChoicePresentation:
    """Resolved choices for one owner-declared dynamic authoring field."""

    field_key: str
    choices: tuple[AuthoringChoice, ...]
    default: object = None
    unavailable_reason: str = ""

    def __post_init__(self) -> None:
        canonical_text(self.field_key, "dynamic choice field key")
        choices = tuple(self.choices)
        if any(not isinstance(value, AuthoringChoice) for value in choices):
            raise TypeError("choices must contain AuthoringChoice values")
        if not isinstance(self.unavailable_reason, str):
            raise TypeError("unavailable_reason must be str")
        object.__setattr__(self, "choices", choices)


@dataclass(frozen=True, slots=True)
class PathPresentationHint:
    """Headless file-dialog semantics for one declared path field."""

    field_key: str
    mode: str = "file"
    file_filter: str = "All files (*)"
    base_dir: str = ""

    def __post_init__(self) -> None:
        canonical_text(self.field_key, "path presentation field key")
        if self.mode not in ("file", "dir"):
            raise ValueError("path presentation mode must be file or dir")
        if not isinstance(self.file_filter, str) or not isinstance(self.base_dir, str):
            raise TypeError("path presentation strings must be str")


@dataclass(frozen=True, slots=True)
class OutputPresentation:
    """Human-facing labels paired with one domain-owned Dataset output."""

    declaration: DatasetOutputDeclaration
    short: str
    axis_label: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        canonical_text(self.short, "output presentation short label")
        canonical_text(self.axis_label, "output presentation axis label")
        if not isinstance(self.description, str):
            raise TypeError("output presentation description must be str")


@dataclass(frozen=True, slots=True)
class DefaultOutputView:
    """Optional node-owned initial view hint consumed by generic frontends."""

    output_name: str
    kind: str
    params: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        canonical_text(self.output_name, "default output view name")
        canonical_text(self.kind, "default output view kind")
        if not isinstance(self.params, Mapping):
            raise TypeError("default output view params must be a mapping")
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class LogicNodeDeclaration:
    """One domain-owned, process-local Logic-node capability declaration."""

    definition: Definition
    description: str
    authoring_schema: AuthoringSchema
    input_specs: tuple[NodeInputSpec, ...]
    outputs: tuple[OutputPresentation, ...]
    build_request: Callable[[Mapping[str, object]], object]
    bind_request: Callable[[object, BoundNodeInputs], object]
    request_output_declarations: (
        Callable[[object], tuple[DatasetOutputDeclaration, ...]] | None
    ) = None
    request_output_axis_label: str | None = None
    request_output_description: str = ""
    default_views: tuple[DefaultOutputView, ...] = ()
    path_presentations: tuple[PathPresentationHint, ...] = ()
    input_path_presentations: tuple[PathPresentationHint, ...] = ()
    resolve_dynamic_choices: (
        Callable[[object], tuple[DynamicChoicePresentation, ...]] | None
    ) = None

    def __post_init__(self) -> None:
        if type(self.definition) not in (
            TaskDefinition,
            MeasurementDefinition,
            ProcessorDefinition,
        ):
            raise TypeError(
                "definition must be a Task/Measurement/Processor Definition"
            )
        if not isinstance(self.description, str):
            raise TypeError("Logic-node description must be str")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        object.__setattr__(
            self,
            "input_specs",
            require_input_specs(self.input_specs),
        )
        outputs = tuple(self.outputs)
        if any(not isinstance(value, OutputPresentation) for value in outputs):
            raise TypeError("outputs must contain OutputPresentation values")
        names = tuple(value.declaration.name for value in outputs)
        if len(names) != len(set(names)):
            raise ValueError("Logic-node output names must be unique")
        object.__setattr__(self, "outputs", outputs)
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
        views = tuple(self.default_views)
        if any(not isinstance(value, DefaultOutputView) for value in views):
            raise TypeError("default_views must contain DefaultOutputView values")
        if dynamic is not None and views:
            raise ValueError("request-owned outputs cannot declare static default views")
        if dynamic is None and any(view.output_name not in names for view in views):
            raise ValueError("default view names an undeclared output")
        object.__setattr__(self, "default_views", views)
        authoring_fields = {
            field.key: field for field in self.authoring_schema.fields
        }
        authoring_keys = set(authoring_fields)
        path_presentations = tuple(self.path_presentations)
        input_path_presentations = tuple(self.input_path_presentations)
        if any(
            not isinstance(value, PathPresentationHint)
            for value in (*path_presentations, *input_path_presentations)
        ):
            raise TypeError("path presentations must contain PathPresentationHint")
        if any(value.field_key not in authoring_keys for value in path_presentations):
            raise ValueError("path presentation names an undeclared authoring field")
        if any(
            authoring_fields[value.field_key].kind != "path"
            for value in path_presentations
        ):
            raise ValueError("path presentation requires an authoring path field")
        inputs_by_key = {value.key: value for value in self.input_specs}
        input_keys = set(inputs_by_key)
        if any(
            value.field_key not in input_keys for value in input_path_presentations
        ):
            raise ValueError("input path presentation names an undeclared input")
        if any(
            not isinstance(inputs_by_key[value.field_key], ArtifactInputSpec)
            for value in input_path_presentations
        ):
            raise ValueError("input path presentation requires an Artifact input")
        if len({value.field_key for value in path_presentations}) != len(
            path_presentations
        ):
            raise ValueError("authoring path presentation fields must be unique")
        if len({value.field_key for value in input_path_presentations}) != len(
            input_path_presentations
        ):
            raise ValueError("input path presentation fields must be unique")
        object.__setattr__(self, "path_presentations", path_presentations)
        object.__setattr__(
            self,
            "input_path_presentations",
            input_path_presentations,
        )
        resolver = self.resolve_dynamic_choices
        dynamic_keys = {
            field.key for field in self.authoring_schema.fields if field.dynamic_choices
        }
        if resolver is not None and not callable(resolver):
            raise TypeError("resolve_dynamic_choices must be callable or None")
        if bool(dynamic_keys) != (resolver is not None):
            raise ValueError(
                "dynamic authoring fields require exactly one owner resolver"
            )
        if not callable(self.build_request):
            raise TypeError("build_request must be callable")
        if not callable(self.bind_request):
            raise TypeError("bind_request must be callable")


__all__ = [
    "DefaultOutputView",
    "DynamicChoicePresentation",
    "LogicNodeDeclaration",
    "OutputPresentation",
    "PathPresentationHint",
]
