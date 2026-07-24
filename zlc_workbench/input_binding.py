"""Generic TaskConsole projection and frozen selections for typed node inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from zlc_data import DataTransformSpec
from zlc_frontend.form import FormChoice, FormFieldProps
from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.dataset_output import DatasetOutputDeclaration
from zlc_neutral_atom.input_spec import (
    ArtifactInputSpec,
    DatasetInputSpec,
    NodeInputSpec,
    require_input_specs,
)
from zlc_neutral_atom.node_input import (
    BoundArtifactInput,
    BoundDatasetInput,
    BoundNodeInputs,
)
from zlc_storage import canonical_text

from .form_projection import PathPresentation


@dataclass(frozen=True, slots=True)
class DatasetInputSelection:
    spec: DatasetInputSpec
    signal_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DatasetInputSpec):
            raise TypeError("spec must be DatasetInputSpec")
        canonical_text(self.signal_key, f"{self.spec.key} signal key")


@dataclass(frozen=True, slots=True)
class ArtifactInputSelection:
    spec: ArtifactInputSpec
    producer_signal_key: str | None = None
    reference_path: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ArtifactInputSpec):
            raise TypeError("spec must be ArtifactInputSpec")
        if (self.producer_signal_key is None) == (self.reference_path is None):
            raise ValueError(
                "artifact input requires exactly one producer or saved reference"
            )
        if self.producer_signal_key is not None:
            canonical_text(
                self.producer_signal_key,
                f"{self.spec.key} producer signal key",
            )
        if self.reference_path is not None:
            canonical_text(self.reference_path, f"{self.spec.key} reference path")
            if not self.spec.allow_saved_reference:
                raise ValueError("artifact input does not allow saved references")


NodeInputSelection = DatasetInputSelection | ArtifactInputSelection


@dataclass(frozen=True, slots=True)
class ConsoleProducerBinding:
    """One exact output resolved against one row in this TaskConsole."""

    signal_key: str
    producer_label: str
    definition_key: DefinitionKey
    output: DatasetOutputDeclaration
    request: object
    run_node: object | None
    final_result_resolved: bool
    final_result: object | None

    def __post_init__(self) -> None:
        canonical_text(self.signal_key, "signal_key")
        canonical_text(self.producer_label, "producer_label")
        if not isinstance(self.definition_key, DefinitionKey):
            raise TypeError("definition_key must be DefinitionKey")
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("output must be DatasetOutputDeclaration")
        if type(self.final_result_resolved) is not bool:
            raise TypeError("final_result_resolved must be bool")
        if not self.final_result_resolved and self.final_result is not None:
            raise ValueError("an unresolved producer cannot expose a FINAL result")

    @property
    def running(self) -> bool:
        return bool(
            self.run_node is not None
            and getattr(self.run_node, "running", False)
        )


@dataclass(frozen=True, slots=True)
class ResolvedDatasetInput:
    selection: DatasetInputSelection
    producer: ConsoleProducerBinding
    transform_spec: DataTransformSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, DatasetInputSelection):
            raise TypeError("selection must be DatasetInputSelection")
        if not isinstance(self.producer, ConsoleProducerBinding):
            raise TypeError("producer must be ConsoleProducerBinding")
        if (
            self.producer.output.contract_id
            not in self.selection.spec.accepted_output_contract_ids
        ):
            raise ValueError("Dataset producer has an unaccepted output contract")
        if self.transform_spec is not None:
            if not isinstance(self.transform_spec, DataTransformSpec):
                raise TypeError("transform_spec must be DataTransformSpec or None")
            if not self.transform_spec.operations:
                raise ValueError("an empty transform_spec must be None")


@dataclass(frozen=True, slots=True)
class ResolvedArtifactInput:
    selection: ArtifactInputSelection
    producer: ConsoleProducerBinding | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.selection, ArtifactInputSelection):
            raise TypeError("selection must be ArtifactInputSelection")
        if (self.producer is None) != (self.selection.producer_signal_key is None):
            raise ValueError("resolved artifact producer differs from its selection")
        if self.producer is not None:
            if not isinstance(self.producer, ConsoleProducerBinding):
                raise TypeError("producer must be ConsoleProducerBinding or None")
            if (
                self.producer.output.contract_id
                not in self.selection.spec.accepted_output_contract_ids
            ):
                raise ValueError("Artifact producer has an unaccepted output contract")


ResolvedNodeInput = ResolvedDatasetInput | ResolvedArtifactInput


def bind_resolved_node_inputs(
    values: Mapping[str, ResolvedNodeInput],
    *,
    resolve_artifact_reference: Callable[[ResolvedArtifactInput], object],
) -> BoundNodeInputs:
    """Strip Workbench routing identity before invoking a logic-node binder.

    Dataset selections retain only owner-declared physical facts.  Artifact
    loading/final-result admission is supplied by the application composition
    root because it owns repositories; the receiving logic node still validates
    the returned concrete reference type.
    """

    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping")
    if not callable(resolve_artifact_reference):
        raise TypeError("resolve_artifact_reference must be callable")
    bound = {}
    for key, value in values.items():
        if isinstance(value, ResolvedDatasetInput):
            producer = value.producer
            bound[key] = BoundDatasetInput(
                value.selection.spec,
                producer.definition_key,
                producer.request,
                producer.output,
                value.transform_spec,
            )
            continue
        if not isinstance(value, ResolvedArtifactInput):
            raise TypeError("resolved inputs contain another value type")
        bound[key] = BoundArtifactInput(
            value.selection.spec,
            resolve_artifact_reference(value),
        )
    return BoundNodeInputs(bound)


def project_input_fields(
    specs,
    *,
    path_presentations: Mapping[str, PathPresentation] | None = None,
) -> tuple[FormFieldProps, ...]:
    """Mechanically project current Dataset/Artifact input declarations."""

    declared = require_input_specs(specs)
    paths = dict(path_presentations or {})
    saved_keys = {
        spec.key
        for spec in declared
        if isinstance(spec, ArtifactInputSpec) and spec.allow_saved_reference
    }
    if not set(paths) <= saved_keys:
        raise ValueError("input path presentation names no saved Artifact input")
    if any(not isinstance(value, PathPresentation) for value in paths.values()):
        raise TypeError("input path presentations must be PathPresentation values")
    fields: list[FormFieldProps] = []
    for spec in declared:
        if isinstance(spec, DatasetInputSpec):
            fields.append(
                FormFieldProps(
                    spec.key,
                    "signal",
                    spec.label,
                    required=True,
                    description=spec.description,
                )
            )
            continue
        if not spec.allow_saved_reference:
            fields.append(
                FormFieldProps(
                    spec.producer_key,
                    "signal",
                    spec.label,
                    required=True,
                    description=spec.description,
                )
            )
            continue
        fields.append(
            FormFieldProps(
                spec.source_key,
                "choice",
                f"{spec.label} source",
                default="producer",
                required=True,
                choices=(
                    FormChoice("Task output", "producer"),
                    FormChoice(f"Saved {spec.label.lower()}", "saved"),
                ),
                description=spec.description,
            )
        )
        fields.append(
            FormFieldProps(
                spec.producer_key,
                "signal",
                f"{spec.label} task",
                description=(
                    f"FINAL {spec.label.lower()} output; used when "
                    f"{spec.label} source is Task output"
                ),
            )
        )
        path_key = spec.reference_path_key
        assert path_key is not None
        path = paths.get(spec.key, PathPresentation())
        fields.append(
            FormFieldProps(
                path_key,
                "path",
                f"Saved {spec.label.lower()}",
                default=spec.default_reference_path,
                path_mode=path.mode,
                file_filter=path.file_filter,
                base_dir=path.base_dir,
                description=(
                    f"Exact saved {spec.label.lower()} pointer; used when "
                    f"{spec.label} source is Saved {spec.label.lower()}"
                ),
            )
        )
    return tuple(fields)


def freeze_input_selections(
    specs,
    values: Mapping[str, object],
) -> dict[str, NodeInputSelection]:
    """Freeze exact input keys without resolving another node or artifact."""

    declared = require_input_specs(specs)
    if not isinstance(values, Mapping):
        raise TypeError("input values must be a mapping")
    expected = {key for spec in declared for key in spec.field_keys}
    unknown = set(values) - expected
    if unknown:
        raise ValueError(
            "input values contain unknown fields: "
            f"{tuple(sorted(map(str, unknown)))}"
        )
    frozen: dict[str, NodeInputSelection] = {}
    for spec in declared:
        if isinstance(spec, DatasetInputSpec):
            value = values.get(spec.key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"select {spec.label}")
            frozen[spec.key] = DatasetInputSelection(spec, value.strip())
            continue
        if not spec.allow_saved_reference:
            value = values.get(spec.producer_key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"select {spec.label}")
            frozen[spec.key] = ArtifactInputSelection(
                spec,
                producer_signal_key=value.strip(),
            )
            continue
        source = values.get(spec.source_key, "producer")
        if source == "producer":
            value = values.get(spec.producer_key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"select a {spec.label} Task output")
            frozen[spec.key] = ArtifactInputSelection(
                spec,
                producer_signal_key=value.strip(),
            )
        elif source == "saved":
            path_key = spec.reference_path_key
            assert path_key is not None
            value = values.get(path_key, spec.default_reference_path)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"select an explicit saved {spec.label.lower()}")
            frozen[spec.key] = ArtifactInputSelection(
                spec,
                reference_path=str(Path(value).expanduser().resolve()),
            )
        else:
            raise ValueError(f"unknown {spec.label} source {source!r}")
    return frozen


__all__ = [
    "ArtifactInputSelection",
    "ConsoleProducerBinding",
    "DatasetInputSelection",
    "NodeInputSelection",
    "ResolvedArtifactInput",
    "ResolvedDatasetInput",
    "ResolvedNodeInput",
    "bind_resolved_node_inputs",
    "freeze_input_selections",
    "project_input_fields",
]
