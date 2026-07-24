"""Closed declarative input contracts for current Logic-node consumers.

These values describe only the two cross-node dependency kinds earned by the
current product.  They contain no resolver, callback, GUI metadata, registry,
plugin hook, or Definition dispatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import canonical_text

def _contract_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    result = tuple(values)
    if not result:
        raise ValueError("an input must accept at least one output contract")
    for value in result:
        canonical_text(value, "output contract id")
    if len(set(result)) != len(result):
        raise ValueError("accepted output contract ids must be unique")
    return result


@dataclass(frozen=True, slots=True)
class DatasetInputSpec:
    """One exact Dataset producer selection."""

    key: str
    label: str
    accepted_output_contract_ids: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        canonical_text(self.key, "Dataset input key")
        canonical_text(self.label, "Dataset input label")
        if not isinstance(self.description, str):
            raise TypeError("Dataset input description must be str")
        object.__setattr__(
            self,
            "accepted_output_contract_ids",
            _contract_ids(self.accepted_output_contract_ids),
        )

    @property
    def field_keys(self) -> tuple[str, ...]:
        return (self.key,)


@dataclass(frozen=True, slots=True)
class ArtifactInputSpec:
    """One exact artifact producer or explicitly selected saved pointer."""

    key: str
    label: str
    output_contract_id: str
    description: str = ""
    allow_saved_reference: bool = False
    default_reference_path: str | None = None

    def __post_init__(self) -> None:
        canonical_text(self.key, "Artifact input key")
        canonical_text(self.label, "Artifact input label")
        canonical_text(self.output_contract_id, "artifact output contract id")
        if not isinstance(self.description, str):
            raise TypeError("Artifact input description must be str")
        if type(self.allow_saved_reference) is not bool:
            raise TypeError("allow_saved_reference must be bool")
        if self.allow_saved_reference:
            if not isinstance(self.default_reference_path, str):
                raise TypeError("saved artifact input needs a default reference path")
            canonical_text(
                self.default_reference_path,
                "default artifact reference path",
            )
        elif self.default_reference_path is not None:
            raise ValueError(
                "producer-only artifact input cannot declare saved defaults"
            )

    @property
    def accepted_output_contract_ids(self) -> tuple[str, ...]:
        return (self.output_contract_id,)

    @property
    def reference_schema_id(self) -> str:
        """The artifact ref schema is the single accepted output contract."""

        return self.output_contract_id

    @property
    def source_key(self) -> str:
        return f"{self.key}_source"

    @property
    def producer_key(self) -> str:
        return self.key if not self.allow_saved_reference else f"{self.key}_signal"

    @property
    def reference_path_key(self) -> str | None:
        return f"{self.key}_path" if self.allow_saved_reference else None

    @property
    def field_keys(self) -> tuple[str, ...]:
        if not self.allow_saved_reference:
            return (self.producer_key,)
        path_key = self.reference_path_key
        assert path_key is not None
        return (self.source_key, self.producer_key, path_key)


NodeInputSpec = DatasetInputSpec | ArtifactInputSpec


def require_input_specs(values) -> tuple[NodeInputSpec, ...]:
    specs = tuple(values)
    if any(not isinstance(value, (DatasetInputSpec, ArtifactInputSpec)) for value in specs):
        raise TypeError("input specs must contain DatasetInputSpec/ArtifactInputSpec")
    keys = tuple(key for spec in specs for key in spec.field_keys)
    if len(set(keys)) != len(keys):
        raise ValueError("input field keys must be unique")
    logical = tuple(spec.key for spec in specs)
    if len(set(logical)) != len(logical):
        raise ValueError("input keys must be unique")
    return specs


__all__ = [
    "ArtifactInputSpec",
    "DatasetInputSpec",
    "NodeInputSpec",
    "require_input_specs",
]
