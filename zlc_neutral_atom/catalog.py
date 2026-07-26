"""Stable identities and closed metadata for user-visible capabilities."""

from __future__ import annotations

from dataclasses import dataclass

from zlc_storage import (
    canonical_text as _canonical_text,
    exact_mapping,
)


DEFINITION_KEY_SCHEMA = "zlc_neutral_atom.DefinitionKey"


@dataclass(frozen=True, order=True)
class DefinitionKey:
    owner_package: str
    stable_definition_id: str

    def __post_init__(self) -> None:
        _canonical_text(self.owner_package, "owner_package")
        _canonical_text(self.stable_definition_id, "stable_definition_id")

    def __str__(self) -> str:
        return f"{self.owner_package}:{self.stable_definition_id}"


def definition_key_to_tree(value: DefinitionKey) -> dict[str, object]:
    """Encode the Definition owner's one current key schema."""

    if not isinstance(value, DefinitionKey):
        raise TypeError("value must be DefinitionKey")
    return {
        "schema": DEFINITION_KEY_SCHEMA,
        "owner_package": value.owner_package,
        "stable_definition_id": value.stable_definition_id,
    }


def definition_key_from_tree(tree: object) -> DefinitionKey:
    """Decode only the Definition owner's current key schema."""

    data = exact_mapping(
        tree,
        {"schema", "owner_package", "stable_definition_id"},
        DEFINITION_KEY_SCHEMA,
    )
    value = DefinitionKey(
        data["owner_package"],
        data["stable_definition_id"],
    )
    if definition_key_to_tree(value) != tree:
        raise ValueError("DefinitionKey tree is typed but non-canonical")
    return value


@dataclass(frozen=True)
class TaskDefinition:
    """Stable catalog metadata for one task kind, never one bound run."""

    key: DefinitionKey
    title: str
    request_schema_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        _canonical_text(self.title, "title")
        _canonical_text(self.request_schema_id, "request_schema_id")


@dataclass(frozen=True)
class MeasurementDefinition:
    """Stable catalog metadata; output schema belongs to the bound contract."""

    key: DefinitionKey
    title: str
    request_schema_id: str
    binding_schema_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        for field in ("title", "request_schema_id", "binding_schema_id"):
            _canonical_text(getattr(self, field), field)


@dataclass(frozen=True)
class ProcessorDefinition:
    """Stable catalog metadata with no callable or binding-generation facts."""

    key: DefinitionKey
    title: str
    config_schema_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        _canonical_text(self.title, "title")
        _canonical_text(self.config_schema_id, "config_schema_id")


__all__ = [
    "DEFINITION_KEY_SCHEMA",
    "DefinitionKey",
    "MeasurementDefinition",
    "ProcessorDefinition",
    "TaskDefinition",
    "definition_key_from_tree",
    "definition_key_to_tree",
]
