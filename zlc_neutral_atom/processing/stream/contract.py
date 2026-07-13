"""Declarative processor catalog values and one explicitly bound operator."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import inspect
import math
from types import MappingProxyType
from typing import Callable

import numpy as np

from zlc_storage import canonical_digest

from zlc_neutral_atom.catalog import DefinitionKey, is_declarative_value
from zlc_neutral_atom.runtime.streams import (
    ArtifactInputRef,
    JoinKeyContract,
    PayloadContract,
    StreamId,
)


def _digest(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field} must be canonical non-empty text")
    return value


def _tree(value: object) -> object:
    """Canonical tree for already-validated declarative binding data."""

    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bytes):
        return {"$type": "bytes", "hex": value.hex()}
    if isinstance(value, np.dtype):
        return {"$type": "numpy.dtype", "value": value.str}
    if isinstance(value, Enum):
        return {
            "$type": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _tree(value.value),
        }
    if isinstance(value, tuple):
        return {"$type": "tuple", "items": [_tree(item) for item in value]}
    if isinstance(value, frozenset):
        items = tuple(_tree(item) for item in value)
        return {
            "$type": "frozenset",
            "items": sorted(
                items,
                key=lambda item: canonical_digest({"item": item}),
            ),
        }
    if isinstance(value, MappingProxyType):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("processor config mappings require canonical string keys")
            _text(key, "processor config mapping key")
        return {
            "$type": "mapping",
            "entries": [
                [key, _tree(item)]
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            ],
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "$type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _tree(getattr(value, field.name))] for field in fields(value)
            ],
        }
    raise TypeError(f"unsupported declarative binding value {type(value).__name__}")


class StreamJoinPolicy(str, Enum):
    EXACT_KEY = "EXACT_KEY"


class StreamCardinality(str, Enum):
    ONE_TO_ONE = "ONE_TO_ONE"


class JoinKeyTransform(str, Enum):
    PASS_THROUGH = "PASS_THROUGH"


@dataclass(frozen=True)
class StreamProcessorDefinition:
    """Catalog-safe declaration; it deliberately contains no Python callable."""

    key: DefinitionKey
    title: str
    config_schema_id: str
    input_payload_contract_fingerprint: str
    output_payload_contract_fingerprint: str
    join_key_contract_fingerprint: str
    join_policy: StreamJoinPolicy = StreamJoinPolicy.EXACT_KEY
    cardinality: StreamCardinality = StreamCardinality.ONE_TO_ONE
    join_key_transform: JoinKeyTransform = JoinKeyTransform.PASS_THROUGH
    operator_deadline_seconds: float = 1.0
    terminal_wait_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.key, DefinitionKey):
            raise TypeError("key must be DefinitionKey")
        _text(self.title, "title")
        _text(self.config_schema_id, "config_schema_id")
        for field in (
            "input_payload_contract_fingerprint",
            "output_payload_contract_fingerprint",
            "join_key_contract_fingerprint",
        ):
            _digest(getattr(self, field), field)
        if self.join_policy is not StreamJoinPolicy.EXACT_KEY:
            raise ValueError("baseline supports only EXACT_KEY")
        if self.cardinality is not StreamCardinality.ONE_TO_ONE:
            raise ValueError("baseline supports only ONE_TO_ONE")
        if self.join_key_transform is not JoinKeyTransform.PASS_THROUGH:
            raise ValueError("baseline supports only PASS_THROUGH")
        for field in ("operator_deadline_seconds", "terminal_wait_seconds"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{field} must be finite and positive")
            object.__setattr__(self, field, float(value))


@dataclass(frozen=True)
class BoundStreamProcessor:
    """Runtime binding of one definition to one trusted synchronous operator.

    Python cannot safely preempt arbitrary in-process code.  The composition root
    therefore admits only reviewed top-level functions.  The worker measures every
    invocation and rejects a result returned after the declared deadline; an operator
    which never returns remains a violated trust boundary, not a cancellable task.
    """

    definition: StreamProcessorDefinition
    config: object
    input_payload_contract: PayloadContract
    output_payload_contract: PayloadContract
    join_key_contract: JoinKeyContract
    output_stream_id: StreamId
    output_source_id: str
    operator: Callable[[object, object], object]
    artifact_inputs: tuple[ArtifactInputRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.definition, StreamProcessorDefinition):
            raise TypeError("definition must be StreamProcessorDefinition")
        if not is_declarative_value(self.config):
            raise TypeError("processor config must be recursively declarative frozen data")
        _tree(self.config)
        for name, contract, expected in (
            (
                "input_payload_contract",
                self.input_payload_contract,
                self.definition.input_payload_contract_fingerprint,
            ),
            (
                "output_payload_contract",
                self.output_payload_contract,
                self.definition.output_payload_contract_fingerprint,
            ),
        ):
            if getattr(contract, "fingerprint", None) != expected:
                raise ValueError(f"{name} fingerprint differs from definition")
            for member in ("snapshot", "validate", "retained_nbytes"):
                if not callable(getattr(contract, member, None)):
                    raise TypeError(f"{name}.{member} must be callable")
        if (
            getattr(self.join_key_contract, "fingerprint", None)
            != self.definition.join_key_contract_fingerprint
        ):
            raise ValueError("join_key_contract fingerprint differs from definition")
        for member in ("snapshot", "validate"):
            if not callable(getattr(self.join_key_contract, member, None)):
                raise TypeError(f"join_key_contract.{member} must be callable")
        if not isinstance(self.output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        _text(self.output_source_id, "output_source_id")
        inputs = tuple(self.artifact_inputs)
        if any(not isinstance(item, ArtifactInputRef) for item in inputs):
            raise TypeError("artifact_inputs must contain ArtifactInputRef values")
        object.__setattr__(self, "artifact_inputs", inputs)
        operator = self.operator
        if (
            not inspect.isfunction(operator)
            or inspect.iscoroutinefunction(operator)
            or inspect.isgeneratorfunction(operator)
            or operator.__closure__ is not None
            or operator.__name__ == "<lambda>"
            or "<locals>" in operator.__qualname__
        ):
            raise TypeError("operator must be one importable top-level Python function")
        parameters = tuple(inspect.signature(operator).parameters.values())
        if (
            len(parameters) != 2
            or any(
                parameter.kind
                not in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                for parameter in parameters
            )
            or any(parameter.default is not inspect.Parameter.empty for parameter in parameters)
        ):
            raise TypeError("operator must accept exactly payload and frozen config")

    @property
    def fingerprint(self) -> str:
        definition = self.definition
        return canonical_digest(
            {
                "contract": "zlc_neutral_atom.BoundStreamProcessor/v1",
                "definition_key": str(definition.key),
                "definition": _tree(definition),
                "config": _tree(self.config),
                "output_stream_id": self.output_stream_id.value,
                "output_source_id": self.output_source_id,
                "operator": f"{self.operator.__module__}.{self.operator.__qualname__}",
                "artifact_inputs": [item.content_digest for item in self.artifact_inputs],
            }
        )


__all__ = [
    "BoundStreamProcessor",
    "JoinKeyTransform",
    "StreamCardinality",
    "StreamJoinPolicy",
    "StreamProcessorDefinition",
]
