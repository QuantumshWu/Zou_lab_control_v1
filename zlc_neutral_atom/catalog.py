"""Stable identities for the discovered Logic-node catalog."""

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


@dataclass(frozen=True, slots=True)
class LogicNodeDefinition:
    """The one catalog identity shared by Tasks, Measurements and Processors.

    Request and binding schemas are owned by the discovered descriptor and its
    domain request.  Keeping schema ids here used to force three almost
    identical Definition classes and made every consumer recover ``kind`` by
    inspecting a Python class.
    """

    key: DefinitionKey
    title: str
    kind: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        _canonical_text(self.title, "title")
        if self.kind not in {"task", "measurement", "processor"}:
            raise ValueError(
                "Logic-node kind must be 'task', 'measurement', or 'processor'"
            )


__all__ = [
    "DEFINITION_KEY_SCHEMA",
    "DefinitionKey",
    "LogicNodeDefinition",
    "definition_key_from_tree",
    "definition_key_to_tree",
]
