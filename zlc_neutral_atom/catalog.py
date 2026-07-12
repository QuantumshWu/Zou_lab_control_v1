"""Explicit immutable capability catalog; composition is ordinary imports only."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import math
from numbers import Integral
from types import MappingProxyType
from typing import Mapping, TypeVar

import numpy as np


DefinitionT = TypeVar("DefinitionT")


def is_declarative_value(value: object, active: set[int] | None = None) -> bool:
    if value is None or type(value) in (bool, int, str, bytes):
        return True
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, np.dtype):
        return True
    if isinstance(value, Enum):
        return is_declarative_value(value.value, active)
    identity = id(value)
    active = set() if active is None else active
    if identity in active:
        return False
    if isinstance(value, (tuple, frozenset)):
        active.add(identity)
        result = all(is_declarative_value(item, active) for item in value)
        active.remove(identity)
        return result
    if isinstance(value, MappingProxyType):
        active.add(identity)
        result = all(
            is_declarative_value(key, active) and is_declarative_value(item, active)
            for key, item in value.items()
        )
        active.remove(identity)
        return result
    if is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if not parameters or not parameters.frozen:
            return False
        active.add(identity)
        result = all(
            is_declarative_value(getattr(value, field.name), active)
            for field in fields(value)
        )
        active.remove(identity)
        return result
    return False


def _canonical_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


@dataclass(frozen=True, order=True)
class DefinitionKey:
    owner_package: str
    stable_definition_id: str
    schema_version: int

    def __post_init__(self) -> None:
        _canonical_text(self.owner_package, "owner_package")
        _canonical_text(self.stable_definition_id, "stable_definition_id")
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be a positive integer")
        object.__setattr__(self, "schema_version", int(self.schema_version))

    @property
    def logical_id(self) -> tuple[str, str]:
        return (self.owner_package, self.stable_definition_id)

    def __str__(self) -> str:
        return (
            f"{self.owner_package}:{self.stable_definition_id}"
            f"@{self.schema_version}"
        )


class DefinitionCatalog:
    """A frozen heterogeneous tuple indexed by unique DefinitionKey.

    There is intentionally no register, override, package scan, entry point, or
    mutable discovery API.  Each bounded context exports an explicit tuple and the
    composition root calls :meth:`compose` with those tuples.
    """

    __slots__ = ("_definitions", "_by_key")

    def __init__(self, definitions: tuple[object, ...]) -> None:
        if not isinstance(definitions, tuple):
            raise TypeError("definitions must be an explicit immutable tuple")
        by_key: dict[DefinitionKey, object] = {}
        logical_ids: dict[tuple[str, str], DefinitionKey] = {}
        for definition in definitions:
            key = getattr(definition, "key", None)
            if not isinstance(key, DefinitionKey):
                raise TypeError("every definition must expose a DefinitionKey as .key")
            parameters = getattr(type(definition), "__dataclass_params__", None)
            if not is_dataclass(definition) or not parameters or not parameters.frozen:
                raise TypeError("catalog definitions must be frozen dataclass values")
            if not is_declarative_value(definition):
                raise TypeError(
                    "catalog definition fields must be recursively declarative data"
                )
            if key in by_key:
                raise ValueError(f"duplicate DefinitionKey {key}")
            previous = logical_ids.get(key.logical_id)
            if previous is not None:
                raise ValueError(
                    f"definition logical id {key.logical_id!r} appears as both "
                    f"schema versions {previous.schema_version} and {key.schema_version}"
                )
            by_key[key] = definition
            logical_ids[key.logical_id] = key
        self._definitions = definitions
        self._by_key: Mapping[DefinitionKey, object] = MappingProxyType(by_key)

    @classmethod
    def compose(cls, *definition_groups: tuple[object, ...]) -> "DefinitionCatalog":
        for group in definition_groups:
            if not isinstance(group, tuple):
                raise TypeError("each definition group must be an explicit tuple")
        return cls(tuple(item for group in definition_groups for item in group))

    @property
    def definitions(self) -> tuple[object, ...]:
        return self._definitions

    @property
    def by_key(self) -> Mapping[DefinitionKey, object]:
        return self._by_key

    def resolve(
        self,
        key: DefinitionKey,
        expected_type: type[DefinitionT] | None = None,
    ) -> DefinitionT | object:
        if not isinstance(key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        try:
            definition = self._by_key[key]
        except KeyError as exc:
            raise KeyError(f"DefinitionKey {key} is absent from the explicit catalog") from exc
        if expected_type is not None and not isinstance(definition, expected_type):
            raise TypeError(
                f"DefinitionKey {key} resolves to {type(definition).__name__}, "
                f"not {expected_type.__name__}"
            )
        return definition

    def __len__(self) -> int:
        return len(self._definitions)

    def __iter__(self):
        return iter(self._definitions)


__all__ = ["DefinitionCatalog", "DefinitionKey", "is_declarative_value"]
