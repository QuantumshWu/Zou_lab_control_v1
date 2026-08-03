"""Single discovered declaration for every built-in Logic node.

The descriptor is inert product metadata plus one domain operation seam.  It
does not construct a public API, host a lifecycle, resolve Workbench rows, or
persist callback-bearing objects.  Those are all projections of this one
declaration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import cache
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import ContextManager, Protocol

from zlc_neutral_atom.artifact_output import ArtifactOutputDeclaration
from zlc_neutral_atom.authoring import AuthoringSchema
from zlc_neutral_atom.catalog import LogicNodeDefinition
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import NodeInputSpec, require_input_specs
from zlc_neutral_atom.installation import (
    DeviceCatalogView,
    ReadoutApparatusFacts,
)
from zlc_neutral_atom.processing.signal_plane import SignalDataPlane
from zlc_neutral_atom.runtime.run import RunHandle, RunPlan
from zlc_plot.kinds import PlotKind
from zlc_plot.specs import PlotSpec


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} cannot contain surrounding whitespace")
    return value


@dataclass(frozen=True, slots=True)
class DatasetOutputSpec:
    """One typed Dataset output and its stable human labels."""

    declaration: DatasetOutputDeclaration
    label: str
    axis_label: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, DatasetOutputDeclaration):
            raise TypeError("declaration must be DatasetOutputDeclaration")
        _require_text(self.label, "Dataset output label")
        if not isinstance(self.axis_label, str) or not isinstance(
            self.description, str
        ):
            raise TypeError("Dataset output labels must be strings")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


@dataclass(frozen=True, slots=True)
class ArtifactOutputSpec:
    """One typed FINAL artifact output and its stable human label."""

    declaration: ArtifactOutputDeclaration
    label: str
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, ArtifactOutputDeclaration):
            raise TypeError("declaration must be ArtifactOutputDeclaration")
        _require_text(self.label, "Artifact output label")
        if not isinstance(self.description, str):
            raise TypeError("Artifact output description must be str")

    @property
    def name(self) -> str:
        return self.declaration.name

    @property
    def contract_id(self) -> str:
        return self.declaration.contract_id


@dataclass(frozen=True, slots=True)
class TaskPreview:
    """A Task's optional temporary preview; it never implies auto-save."""

    output_name: str
    plot: PlotKind | PlotSpec

    def __post_init__(self) -> None:
        _require_text(self.output_name, "Task preview output name")
        if not isinstance(self.plot, (PlotKind, PlotSpec)):
            raise TypeError("Task preview plot must be PlotKind or PlotSpec")


@dataclass(frozen=True, slots=True)
class UiContribution:
    """One lazy leaf-local UI entry without importing Qt headlessly."""

    purpose: str
    module: str
    symbol: str

    def __post_init__(self) -> None:
        for value, field in (
            (self.purpose, "UI purpose"),
            (self.module, "UI module"),
            (self.symbol, "UI symbol"),
        ):
            _require_text(value, field)
        if not self.purpose.isidentifier() or self.purpose.startswith("_"):
            raise ValueError("UI purpose must be a public identifier")
        if not self.symbol.isidentifier() or self.symbol.startswith("_"):
            raise ValueError("UI symbol must be a public identifier")


@dataclass(frozen=True, slots=True)
class SelectionParameterPatch:
    """A leaf-declared device-parameter draft derived from a physical Selection.

    The patch is deliberately not a hardware command and does not contain a
    Measurement field.  A producer leaf may describe which installed device
    instance and typed installation fields should be prefilled; the Device
    Manager owns staging, validation, and the explicit Apply boundary.
    """

    target_instance_id: str
    values: Mapping[str, object]
    description: str = ""

    def __post_init__(self) -> None:
        target = _require_text(self.target_instance_id, "patch target instance id")
        object.__setattr__(self, "target_instance_id", target)
        values = dict(self.values)
        if not values:
            raise ValueError("selection parameter patch cannot be empty")
        if any(
            not isinstance(key, str) or not key.strip() or key != key.strip()
            for key in values
        ):
            raise ValueError("selection parameter patch keys must be canonical text")
        object.__setattr__(self, "values", MappingProxyType(values))
        if not isinstance(self.description, str):
            raise TypeError("selection parameter patch description must be text")


class LogicNodeApplicationContext(Protocol):
    """Typed application surface supplied once by the composition root.

    Leaf callbacks type against this one neutral owner without importing the
    public facade. Implementations must resolve only descriptor-declared
    fields; the context is not a string-keyed service-locator fact pool.
    """

    @property
    def project_root(self) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    @property
    def pulses_root(self) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    @property
    def tasks_root(self) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    @property
    def runs_root(self) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    @property
    def figures_root(self) -> Path:  # pragma: no cover - interface only
        raise NotImplementedError

    @property
    def device_catalog(self) -> DeviceCatalogView:  # pragma: no cover
        raise NotImplementedError

    @property
    def apparatus(self) -> tuple[ReadoutApparatusFacts, ...]:  # pragma: no cover
        raise NotImplementedError

    @property
    def signal_plane(self) -> SignalDataPlane:  # pragma: no cover
        raise NotImplementedError

    def input(self, spec: NodeInputSpec) -> object:  # pragma: no cover
        raise NotImplementedError

    def device(
        self,
        field_key: str,
        capability: str | None = None,
    ) -> object:  # pragma: no cover
        """Return one declared capability of a selected device instance.

        A single authored instance field may require several capabilities, as
        Camera finite and monitor acquisition do.  The caller may omit
        ``capability`` only when the field declares exactly one.
        """

        raise NotImplementedError

    def optional_device(
        self,
        field_key: str,
        capability: str,
    ) -> object | None:  # pragma: no cover
        """Return one descriptor-declared optional capability when present."""

        raise NotImplementedError

    def start_run(self, plan: RunPlan) -> RunHandle:  # pragma: no cover
        raise NotImplementedError

    def wait_run(self, handle: RunHandle) -> object:  # pragma: no cover
        raise NotImplementedError

    def operation_guard(self) -> ContextManager[None]:  # pragma: no cover
        raise NotImplementedError

    def default_artifact(
        self,
        contract_id: str,
    ) -> object | None:  # pragma: no cover
        raise NotImplementedError

    def remember_artifact(
        self,
        contract_id: str,
        reference: object,
    ) -> None:  # pragma: no cover
        raise NotImplementedError

    def open_ui(
        self,
        contribution: UiContribution,
        *args,
        **kwargs,
    ) -> object:  # pragma: no cover
        raise NotImplementedError


# The callback performs the leaf's one real bind/execute operation against the
# exact typed context above.  It must not return another public
# Intent/Bound/Prepared/API lifecycle layer.
BindExecute = Callable[[object, LogicNodeApplicationContext], object]


@dataclass(frozen=True, slots=True)
class LogicNodeDescriptor:
    """The sole discovered owner of one user-visible Logic node."""

    api_name: str
    definition: LogicNodeDefinition
    description: str
    authoring_schema: AuthoringSchema
    input_specs: tuple[NodeInputSpec, ...]
    outputs: tuple[DatasetOutputSpec, ...]
    build_request: Callable[[Mapping[str, object]], object]
    bind_execute: BindExecute
    artifact_outputs: tuple[ArtifactOutputSpec, ...] = ()
    device_requirements: tuple[tuple[str, tuple[str, ...]], ...] = ()
    optional_device_capabilities: tuple[
        tuple[str, tuple[str, ...]], ...
    ] = ()
    resolve_outputs: (
        Callable[[object], tuple[DatasetOutputSpec, ...]] | None
    ) = None
    task_previews: tuple[TaskPreview, ...] = ()
    ui_contributions: tuple[UiContribution, ...] = ()
    selection_parameter_patch: (
        Callable[[object, object, object], SelectionParameterPatch | None] | None
    ) = None
    operations: Mapping[str, Callable[..., object]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.api_name, "Logic-node API name")
        if not self.api_name.isidentifier() or self.api_name.startswith("_"):
            raise ValueError("Logic-node API name must be a public identifier")
        if not isinstance(self.definition, LogicNodeDefinition):
            raise TypeError("definition must be LogicNodeDefinition")
        if not isinstance(self.description, str):
            raise TypeError("Logic-node description must be str")
        if not isinstance(self.authoring_schema, AuthoringSchema):
            raise TypeError("authoring_schema must be AuthoringSchema")
        object.__setattr__(self, "input_specs", require_input_specs(self.input_specs))

        outputs = self._checked_outputs(self.outputs)
        artifacts = tuple(self.artifact_outputs)
        if any(not isinstance(value, ArtifactOutputSpec) for value in artifacts):
            raise TypeError("artifact_outputs must contain ArtifactOutputSpec")
        if len({value.name for value in artifacts}) != len(artifacts):
            raise ValueError("Artifact output names must be unique")
        if {value.name for value in outputs} & {value.name for value in artifacts}:
            raise ValueError("Dataset and Artifact output names cannot overlap")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "artifact_outputs", artifacts)

        if not callable(self.build_request):
            raise TypeError("build_request must be callable")
        if not callable(self.bind_execute):
            raise TypeError("bind_execute must be callable")
        if self.resolve_outputs is not None and not callable(self.resolve_outputs):
            raise TypeError("resolve_outputs must be callable or None")
        if self.resolve_outputs is not None and outputs:
            raise ValueError("static and request-dependent outputs are exclusive")
        if self.selection_parameter_patch is not None and not callable(
            self.selection_parameter_patch
        ):
            raise TypeError("selection_parameter_patch must be callable or None")

        requirements = tuple(self.device_requirements)
        normalized_requirements: list[tuple[str, tuple[str, ...]]] = []
        for field_key, capabilities in requirements:
            values = tuple(capabilities)
            if not values:
                raise ValueError("device requirement capabilities cannot be empty")
            checked_capabilities = tuple(
                _require_text(value, "capability") for value in values
            )
            if len(checked_capabilities) != len(set(checked_capabilities)):
                raise ValueError("device requirement capabilities must be unique")
            normalized_requirements.append(
                (_require_text(field_key, "device field key"), checked_capabilities)
            )
        keys = tuple(key for key, _capability in normalized_requirements)
        if len(keys) != len(set(keys)):
            raise ValueError("device requirement field keys must be unique")
        undeclared = set(keys).difference(self.authoring_schema.keys)
        if undeclared:
            raise ValueError(
                "device requirements reference undeclared fields: "
                f"{tuple(sorted(undeclared))}"
            )
        object.__setattr__(
            self, "device_requirements", tuple(normalized_requirements)
        )

        optional_requirements = tuple(self.optional_device_capabilities)
        normalized_optional: list[tuple[str, tuple[str, ...]]] = []
        required_by_key = dict(normalized_requirements)
        for field_key, capabilities in optional_requirements:
            checked_key = _require_text(field_key, "optional device field key")
            if checked_key not in required_by_key:
                raise ValueError(
                    "optional device capabilities require an existing device field"
                )
            values = tuple(capabilities)
            if not values:
                raise ValueError("optional device capabilities cannot be empty")
            checked_capabilities = tuple(
                _require_text(value, "optional capability") for value in values
            )
            if len(checked_capabilities) != len(set(checked_capabilities)):
                raise ValueError("optional device capabilities must be unique")
            overlap = set(checked_capabilities).intersection(
                required_by_key[checked_key]
            )
            if overlap:
                raise ValueError(
                    "required and optional device capabilities cannot overlap"
                )
            normalized_optional.append((checked_key, checked_capabilities))
        optional_keys = tuple(key for key, _values in normalized_optional)
        if len(optional_keys) != len(set(optional_keys)):
            raise ValueError("optional device field keys must be unique")
        object.__setattr__(
            self,
            "optional_device_capabilities",
            tuple(normalized_optional),
        )

        previews = tuple(self.task_previews)
        if any(not isinstance(value, TaskPreview) for value in previews):
            raise TypeError("task_previews must contain TaskPreview")
        if previews and self.definition.kind != "task":
            raise ValueError("only Tasks may declare automatic previews")
        if self.resolve_outputs is None:
            output_names = {value.name for value in outputs}
            if any(value.output_name not in output_names for value in previews):
                raise ValueError("Task preview names an undeclared Dataset output")
        object.__setattr__(self, "task_previews", previews)

        contributions = tuple(self.ui_contributions)
        if any(not isinstance(value, UiContribution) for value in contributions):
            raise TypeError("ui_contributions must contain UiContribution")
        purposes = tuple(value.purpose for value in contributions)
        if len(purposes) != len(set(purposes)):
            raise ValueError("UI purposes must be unique within one leaf")
        object.__setattr__(self, "ui_contributions", contributions)

        reserved_operations = {
            "build",
            "current_artifact",
            "open_ui",
            "run",
            "start",
            "stop",
        }
        operations: dict[str, Callable[..., object]] = {}
        for name, operation in dict(self.operations).items():
            checked_name = _require_text(name, "Logic-node operation name")
            if not checked_name.isidentifier() or checked_name.startswith("_"):
                raise ValueError("Logic-node operation names must be public identifiers")
            if checked_name in reserved_operations:
                raise ValueError(
                    f"Logic-node operation {checked_name!r} conflicts with NodeApi"
                )
            if not callable(operation):
                raise TypeError("Logic-node operations must be callable")
            operations[checked_name] = operation
        object.__setattr__(self, "operations", MappingProxyType(operations))

    @staticmethod
    def _checked_outputs(values) -> tuple[DatasetOutputSpec, ...]:
        outputs = tuple(values)
        if any(not isinstance(value, DatasetOutputSpec) for value in outputs):
            raise TypeError("outputs must contain DatasetOutputSpec")
        names = tuple(value.name for value in outputs)
        if len(names) != len(set(names)):
            raise ValueError("Dataset output names must be unique")
        return outputs

    def outputs_for(self, request: object) -> tuple[DatasetOutputSpec, ...]:
        resolver = self.resolve_outputs
        outputs = self.outputs if resolver is None else self._checked_outputs(
            resolver(request)
        )
        artifact_names = {value.name for value in self.artifact_outputs}
        if {value.name for value in outputs} & artifact_names:
            raise ValueError("Dataset and Artifact output names cannot overlap")
        preview_names = {value.output_name for value in self.task_previews}
        if preview_names.difference(value.name for value in outputs):
            raise ValueError("Task preview is absent from resolved outputs")
        return tuple(outputs)


@cache
def discover_logic_nodes() -> tuple[LogicNodeDescriptor, ...]:
    """Discover checked-in leaf descriptors from one fixed namespace."""

    namespace = import_module("zlc_neutral_atom.logic_nodes")
    prefix = namespace.__name__ + "."
    module_names: set[str] = set()
    for root_value in namespace.__path__:
        root = Path(root_value)
        for path in root.rglob("logic_node.py"):
            relative = path.relative_to(root).with_suffix("")
            if "ui" not in relative.parts:
                module_names.add(prefix + ".".join(relative.parts))
    if not module_names:
        raise RuntimeError("no Logic-node descriptors were discovered")

    descriptors: list[LogicNodeDescriptor] = []
    for module_name in sorted(module_names):
        module = import_module(module_name)
        descriptor = getattr(module, "LOGIC_NODE", None)
        if not isinstance(descriptor, LogicNodeDescriptor):
            raise TypeError(
                f"{module_name} must export one LogicNodeDescriptor as LOGIC_NODE"
            )
        expected_owner = module_name.removesuffix(".logic_node")
        if descriptor.definition.key.owner_package != expected_owner:
            raise ValueError(
                f"{module_name} Definition owner must be {expected_owner!r}"
            )
        expected_ui_prefix = expected_owner + ".ui."
        for contribution in descriptor.ui_contributions:
            if not contribution.module.startswith(expected_ui_prefix):
                raise ValueError(
                    f"{module_name} UI contribution {contribution.purpose!r} "
                    f"must be owned below {expected_ui_prefix!r}"
                )
        descriptors.append(descriptor)

    api_names = tuple(value.api_name for value in descriptors)
    if len(api_names) != len(set(api_names)):
        raise ValueError("Logic-node API names must be unique")
    keys = tuple(value.definition.key for value in descriptors)
    if len(keys) != len(set(keys)):
        raise ValueError("Logic-node Definition keys must be unique")
    return tuple(sorted(descriptors, key=lambda value: value.api_name))


__all__ = [
    "ArtifactOutputSpec",
    "DatasetOutputSpec",
    "LogicNodeDescriptor",
    "LogicNodeApplicationContext",
    "SelectionParameterPatch",
    "TaskPreview",
    "UiContribution",
    "discover_logic_nodes",
]
