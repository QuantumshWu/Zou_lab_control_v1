"""Framework values passed from a composition input resolver to a logic node.

These values contain the physical facts a logic-node owner needs to bind its
typed request.  They deliberately exclude GUI row ids, widgets, live slots,
Run objects, repositories and resolver callbacks.  A Workbench may resolve a
user selection into these values; only the receiving logic node interprets
their request/reference types.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from zlc_data import DataTransformSpec

from .catalog import DefinitionKey
from .dataset_output import DatasetOutputDeclaration
from .input_spec import ArtifactInputSpec, DatasetInputSpec


@dataclass(frozen=True, slots=True)
class BoundDatasetInput:
    """One exact producer output, stripped of composition/runtime identity."""

    spec: DatasetInputSpec
    producer_definition: DefinitionKey
    producer_request: object
    output: DatasetOutputDeclaration
    transform_spec: DataTransformSpec | None = None
    output_binding: object | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.spec, DatasetInputSpec):
            raise TypeError("spec must be DatasetInputSpec")
        if not isinstance(self.producer_definition, DefinitionKey):
            raise TypeError("producer_definition must be DefinitionKey")
        if not isinstance(self.output, DatasetOutputDeclaration):
            raise TypeError("output must be DatasetOutputDeclaration")
        if not self.spec.accepts(self.output.contract_id):
            raise ValueError("Dataset output contract is not accepted by this input")
        if self.transform_spec is not None:
            if not isinstance(self.transform_spec, DataTransformSpec):
                raise TypeError("transform_spec must be DataTransformSpec or None")
            if not self.transform_spec.operations:
                raise ValueError("an empty transform_spec must be None")


@dataclass(frozen=True, slots=True)
class BoundArtifactInput:
    """One admitted typed artifact reference for an owner-declared input."""

    spec: ArtifactInputSpec
    reference: object

    def __post_init__(self) -> None:
        if not isinstance(self.spec, ArtifactInputSpec):
            raise TypeError("spec must be ArtifactInputSpec")
        if self.reference is None:
            raise ValueError("artifact input reference cannot be None")


BoundNodeInput = BoundDatasetInput | BoundArtifactInput


class BoundNodeInputs:
    """Closed mapping whose keys and value specs must agree exactly."""

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, BoundNodeInput]) -> None:
        if not isinstance(values, Mapping):
            raise TypeError("values must be a mapping")
        frozen = dict(values)
        for key, value in frozen.items():
            if not isinstance(value, (BoundDatasetInput, BoundArtifactInput)):
                raise TypeError("node inputs must be bound Dataset/Artifact inputs")
            if key != value.spec.key:
                raise ValueError("node input key differs from its owner spec")
        self._values = MappingProxyType(frozen)

    @property
    def values(self) -> Mapping[str, BoundNodeInput]:
        return self._values

    def dataset(self, spec: DatasetInputSpec) -> BoundDatasetInput:
        if not isinstance(spec, DatasetInputSpec):
            raise TypeError("spec must be DatasetInputSpec")
        value = self._values.get(spec.key)
        if not isinstance(value, BoundDatasetInput) or value.spec != spec:
            raise ValueError(f"Dataset input {spec.key!r} is absent or mismatched")
        return value

    def artifact(self, spec: ArtifactInputSpec) -> BoundArtifactInput:
        if not isinstance(spec, ArtifactInputSpec):
            raise TypeError("spec must be ArtifactInputSpec")
        value = self._values.get(spec.key)
        if not isinstance(value, BoundArtifactInput) or value.spec != spec:
            raise ValueError(f"Artifact input {spec.key!r} is absent or mismatched")
        return value


def bind_no_node_inputs(request: object, inputs: BoundNodeInputs) -> object:
    """Return a request only when its owner declared no cross-node inputs."""

    if not isinstance(inputs, BoundNodeInputs):
        raise TypeError("inputs must be BoundNodeInputs")
    if inputs.values:
        raise ValueError("this Logic node declares no cross-node inputs")
    return request


__all__ = [
    "BoundArtifactInput",
    "BoundDatasetInput",
    "BoundNodeInput",
    "BoundNodeInputs",
    "bind_no_node_inputs",
]
