"""Declarative processor catalog values and one explicitly bound operator."""

from __future__ import annotations

import hashlib
import inspect
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable

import numpy as np

from zlc_storage import canonical_digest

from zlc_neutral_atom.catalog import DefinitionKey
from zlc_neutral_atom.runtime.streams import (
    ArtifactInputRef,
    JoinKeyContract,
    PayloadContract,
    ProcessorStageProvenance,
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


def _canonical_dtype_tree(value: np.dtype) -> dict[str, object]:
    """Return one collision-resistant tree for a supported scalar dtype."""

    dtype = np.dtype(value)
    if (
        dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.itemsize == 0
        or dtype.kind not in "biufc"
        or dtype != dtype.newbyteorder("<")
        or (dtype.kind == "b" and dtype.itemsize != 1)
        or (dtype.kind in "iu" and dtype.itemsize not in (1, 2, 4, 8))
        or (dtype.kind == "f" and dtype.itemsize not in (2, 4, 8))
        or (dtype.kind == "c" and dtype.itemsize not in (8, 16))
    ):
        raise TypeError(
            "processor config dtypes must be canonical little-endian scalar numeric dtypes"
        )
    return {
        "$type": "numpy.dtype",
        "str": dtype.str,
        "kind": dtype.kind,
        "itemsize": int(dtype.itemsize),
        "byteorder": "|" if dtype.byteorder == "|" else "<",
    }


def _tree(
    value: object,
    memo: dict[int, object] | None = None,
    active: set[int] | None = None,
) -> object:
    """Canonical tree for already-validated declarative binding data."""

    memo = {} if memo is None else memo
    active = set() if active is None else active
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bytes):
        return {
            "$type": "bytes-content",
            "nbytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.dtype):
        return _canonical_dtype_tree(value)
    identity = id(value)
    if identity in active:
        raise TypeError("processor config contains a recursive cycle")
    if identity in memo:
        return memo[identity]
    active.add(identity)
    if isinstance(value, np.ndarray):
        if not _is_canonical_immutable_array(value):
            raise TypeError(
                "processor config arrays must be canonical immutable ndarrays"
            )
        result = {
            "$type": "numpy.ndarray-content",
            "dtype": _canonical_dtype_tree(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
        }
    elif isinstance(value, Enum):
        result = {
            "$type": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _tree(value.value, memo, active),
        }
    elif isinstance(value, tuple):
        result = {
            "$type": "tuple",
            "items": [_tree(item, memo, active) for item in value],
        }
    elif isinstance(value, frozenset):
        items = tuple(_tree(item, memo, active) for item in value)
        result = {
            "$type": "frozenset",
            "items": sorted(
                items,
                key=lambda item: canonical_digest({"item": item}),
            ),
        }
    elif isinstance(value, MappingProxyType):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("processor config mappings require canonical string keys")
            _text(key, "processor config mapping key")
        result = {
            "$type": "mapping",
            "entries": [
                [key, _tree(item, memo, active)]
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            ],
        }
    elif is_dataclass(value) and not isinstance(value, type):
        result = {
            "$type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _tree(getattr(value, field.name), memo, active)]
                for field in fields(value)
            ],
        }
    else:
        active.remove(identity)
        raise TypeError(f"unsupported declarative binding value {type(value).__name__}")
    active.remove(identity)
    memo[identity] = result
    return result


def _has_bytes_backing(value: np.ndarray) -> bool:
    """Reject arrays whose read-only flag can be reversed by a mutable owner."""

    current: object = value
    seen: set[int] = set()
    while isinstance(current, np.ndarray):
        identity = id(current)
        if identity in seen or current.flags.writeable or current.flags.owndata:
            return False
        seen.add(identity)
        current = current.base
    if isinstance(current, bytes):
        return True
    return isinstance(current, memoryview) and current.readonly and isinstance(
        current.obj,
        bytes,
    )


def _is_canonical_immutable_array(value: object) -> bool:
    if type(value) is not np.ndarray:
        return False
    dtype = value.dtype
    try:
        _canonical_dtype_tree(dtype)
    except TypeError:
        return False
    return bool(value.ndim > 0 and value.flags.c_contiguous and _has_bytes_backing(value))


def _snapshot_binding_value(
    value: object,
    memo: dict[int, object] | None = None,
    active: set[int] | None = None,
    preserve: dict[int, object] | None = None,
) -> object:
    """Owner-rebuild binding data while preserving explicit generation identities."""

    preserve = {} if preserve is None else preserve
    if id(value) in preserve:
        return preserve[id(value)]
    if value is None or type(value) in (bool, int, float, str, bytes):
        return value
    if isinstance(value, (np.dtype, np.ndarray)):
        return value
    if isinstance(value, Enum):
        snapshot = _snapshot_binding_value(value.value, memo, active, preserve)
        if snapshot is not value.value:
            raise TypeError("processor config Enum values must be intrinsically immutable")
        return value
    memo = {} if memo is None else memo
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise TypeError("processor config contains a recursive cycle")
    if identity in memo:
        return memo[identity]
    active.add(identity)
    if isinstance(value, tuple):
        items = tuple(
            _snapshot_binding_value(item, memo, active, preserve) for item in value
        )
        result = items
    elif isinstance(value, frozenset):
        result = frozenset(
            _snapshot_binding_value(item, memo, active, preserve) for item in value
        )
    elif isinstance(value, MappingProxyType):
        owned = {
            key: _snapshot_binding_value(item, memo, active, preserve)
            for key, item in value.items()
        }
        result = MappingProxyType(owned)
    elif is_dataclass(value) and not isinstance(value, type):
        updates: dict[str, object] = {}
        for field in fields(value):
            if field.init:
                updates[field.name] = _snapshot_binding_value(
                    getattr(value, field.name),
                    memo,
                    active,
                    preserve,
                )
        try:
            result = replace(value, **updates)
        except (TypeError, ValueError) as error:
            active.remove(identity)
            raise TypeError(
                f"cannot reconstruct processor config {type(value).__name__}"
            ) from error
    else:
        active.remove(identity)
        raise TypeError(f"unsupported declarative binding value {type(value).__name__}")
    active.remove(identity)
    memo[identity] = result
    return result


def _generation_owner_graph(*roots: object) -> dict[int, object]:
    """Collect identity-owned declarative descendants that configs may reference."""

    preserved: dict[int, object] = {}
    pending = list(roots)
    while pending:
        value = pending.pop()
        if value is None or type(value) in (bool, int, float, str, bytes):
            continue
        identity = id(value)
        if identity in preserved:
            continue
        preserved[identity] = value
        if isinstance(value, Enum):
            pending.append(value.value)
        elif isinstance(value, (tuple, frozenset)):
            pending.extend(value)
        elif isinstance(value, MappingProxyType):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, item.name) for item in fields(value))
    return preserved


_PROCESSOR_BINDING_MAX_DEPTH = 32
# Calibration permits up to 100,000 explicitly labelled sites; the binding tree
# must admit that declared default while still bounding arbitrary recursive data.
_PROCESSOR_BINDING_MAX_NODES = 131072
_PROCESSOR_BINDING_MAX_ARRAYS = 64
_PROCESSOR_BINDING_MAX_ARRAY_RANK = 8
_PROCESSOR_BINDING_MAX_SINGLE_ARRAY_BYTES = 192 * 1024 * 1024
_PROCESSOR_BINDING_MAX_TOTAL_ARRAY_BYTES = 256 * 1024 * 1024
_PROCESSOR_BINDING_MAX_SINGLE_TEXT_BYTES = 1024 * 1024
_PROCESSOR_BINDING_MAX_TOTAL_TEXT_BYTES = 16 * 1024 * 1024
_PROCESSOR_BINDING_MAX_EXPANDED_NODES = 262_144
_PROCESSOR_BINDING_MAX_CANONICAL_BYTES = 32 * 1024 * 1024
_PROCESSOR_BINDING_MAX_INTEGER_BITS = 4096
_PROCESSOR_BINDING_MAX_TOTAL_INTEGER_BYTES = 4 * 1024 * 1024


@dataclass
class _ProcessorBindingBudget:
    nodes: int = 0
    arrays: int = 0
    array_bytes: int = 0
    text_bytes: int = 0
    expanded_nodes: int = 0
    canonical_bytes: int = 0
    integer_bytes: int = 0

    def admit_node(self, depth: int) -> bool:
        self.nodes += 1
        return (
            depth <= _PROCESSOR_BINDING_MAX_DEPTH
            and self.nodes <= _PROCESSOR_BINDING_MAX_NODES
        )

    def admit_array(self, value: np.ndarray) -> bool:
        self.arrays += 1
        self.array_bytes += int(value.nbytes)
        return (
            value.ndim <= _PROCESSOR_BINDING_MAX_ARRAY_RANK
            and self.arrays <= _PROCESSOR_BINDING_MAX_ARRAYS
            and value.nbytes <= _PROCESSOR_BINDING_MAX_SINGLE_ARRAY_BYTES
            and self.array_bytes <= _PROCESSOR_BINDING_MAX_TOTAL_ARRAY_BYTES
        )

    def admit_text(self, value: str | bytes) -> bool:
        size = len(value.encode("utf-8")) if isinstance(value, str) else len(value)
        self.text_bytes += size
        return (
            size <= _PROCESSOR_BINDING_MAX_SINGLE_TEXT_BYTES
            and self.text_bytes <= _PROCESSOR_BINDING_MAX_TOTAL_TEXT_BYTES
        )

    def admit_integer(self, value: int) -> bool:
        bits = abs(value).bit_length()
        # Upper-bound decimal text without materializing an attacker-sized int.
        encoded_bytes = 1 if bits == 0 else 2 + (bits * 30103) // 100000
        self.integer_bytes += encoded_bytes
        return (
            bits <= _PROCESSOR_BINDING_MAX_INTEGER_BITS
            and self.integer_bytes <= _PROCESSOR_BINDING_MAX_TOTAL_INTEGER_BYTES
        )

    def admit_expansion(self, nodes: int, canonical_bytes: int) -> bool:
        self.expanded_nodes += nodes
        self.canonical_bytes += canonical_bytes
        return (
            self.expanded_nodes <= _PROCESSOR_BINDING_MAX_EXPANDED_NODES
            and self.canonical_bytes <= _PROCESSOR_BINDING_MAX_CANONICAL_BYTES
        )


def _is_processor_binding_value(
    value: object,
    active: set[int] | None = None,
    memo: dict[int, tuple[int, int]] | None = None,
    budget: _ProcessorBindingBudget | None = None,
    depth: int = 0,
) -> bool:
    """Binding-only extension of catalog data with content-addressed arrays.

    Catalog definitions remain small declarative values.  Only an already-bound
    processor may retain numeric calibration arrays, and only when their backing
    storage is intrinsically immutable rather than a reversible ``writeable=False``
    view over mutable memory.
    """

    budget = _ProcessorBindingBudget() if budget is None else budget
    if value is None or type(value) is bool:
        return budget.admit_node(depth) and budget.admit_expansion(1, 8)
    if type(value) is int:
        return (
            budget.admit_node(depth)
            and budget.admit_integer(value)
            and budget.admit_expansion(
                1,
                8 + max(1, (abs(value).bit_length() * 30103) // 100000),
            )
        )
    if type(value) in (str, bytes):
        canonical_bytes = (
            len(value.encode("utf-8")) + 16
            if isinstance(value, str)
            else 128
        )
        return (
            budget.admit_node(depth)
            and budget.admit_text(value)
            and budget.admit_expansion(1, canonical_bytes)
        )
    if type(value) is float:
        return (
            budget.admit_node(depth)
            and math.isfinite(value)
            and budget.admit_expansion(1, 32)
        )
    if isinstance(value, np.dtype):
        if not budget.admit_node(depth):
            return False
        try:
            _canonical_dtype_tree(value)
        except TypeError:
            return False
        return budget.admit_expansion(1, 128)
    identity = id(value)
    active = set() if active is None else active
    memo = {} if memo is None else memo
    if identity in active:
        return False
    if identity in memo:
        return budget.admit_expansion(*memo[identity])
    if not budget.admit_node(depth):
        return False
    expanded_nodes_before = budget.expanded_nodes
    canonical_bytes_before = budget.canonical_bytes
    if isinstance(value, np.ndarray):
        local_bytes = 192 + 16 * value.ndim
    elif isinstance(value, Enum):
        local_bytes = 64 + len(type(value).__module__) + len(type(value).__qualname__)
    elif isinstance(value, (tuple, frozenset)):
        local_bytes = 32 + 8 * len(value)
    elif isinstance(value, MappingProxyType):
        local_bytes = 48 + 16 * len(value)
    elif is_dataclass(value) and not isinstance(value, type):
        local_bytes = (
            64
            + len(type(value).__module__)
            + len(type(value).__qualname__)
            + sum(len(item.name) + 8 for item in fields(value))
        )
    else:
        return False
    if not budget.admit_expansion(1, local_bytes):
        return False
    if isinstance(value, np.ndarray):
        result = _is_canonical_immutable_array(value) and budget.admit_array(value)
    elif isinstance(value, Enum):
        active.add(identity)
        result = _is_processor_binding_value(
            value.value,
            active,
            memo,
            budget,
            depth + 1,
        )
        active.remove(identity)
    elif isinstance(value, (tuple, frozenset)):
        active.add(identity)
        result = all(
            _is_processor_binding_value(item, active, memo, budget, depth + 1)
            for item in value
        )
        active.remove(identity)
    elif isinstance(value, MappingProxyType):
        active.add(identity)
        result = all(
            _is_processor_binding_value(key, active, memo, budget, depth + 1)
            and _is_processor_binding_value(item, active, memo, budget, depth + 1)
            for key, item in value.items()
        )
        active.remove(identity)
    elif is_dataclass(value) and not isinstance(value, type):
        parameters = getattr(type(value), "__dataclass_params__", None)
        if not parameters or not parameters.frozen:
            return False
        active.add(identity)
        result = all(
            _is_processor_binding_value(
                getattr(value, field.name),
                active,
                memo,
                budget,
                depth + 1,
            )
            for field in fields(value)
        )
        active.remove(identity)
    else:
        result = False
    if result:
        memo[identity] = (
            budget.expanded_nodes - expanded_nodes_before,
            budget.canonical_bytes - canonical_bytes_before,
        )
    return result


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

    Declarative definition/config/artifact values are reconstructed into this
    binding's ownership.  Payload and join-key contracts remain deliberately shared
    generation owners because identity is part of their runtime contract; the
    composition root must not expose reflective mutation of those owners.
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
        definition = _snapshot_binding_value(self.definition)
        if not isinstance(definition, StreamProcessorDefinition):
            raise TypeError("definition snapshot changed its declared type")
        _tree(definition)
        object.__setattr__(self, "definition", definition)
        source_output_stream_id = self.output_stream_id
        if not isinstance(source_output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        output_stream_id = StreamId(source_output_stream_id.value)
        object.__setattr__(self, "output_stream_id", output_stream_id)
        _text(self.output_source_id, "output_source_id")
        if not _is_processor_binding_value(self.config):
            raise TypeError(
                "processor config must be recursively frozen binding data"
            )
        generation_owners = _generation_owner_graph(
            self.input_payload_contract,
            self.output_payload_contract,
            self.join_key_contract,
            output_stream_id,
        )
        generation_owners[id(source_output_stream_id)] = output_stream_id
        config = _snapshot_binding_value(
            self.config,
            preserve=generation_owners,
        )
        if not _is_processor_binding_value(config):
            raise TypeError("processor config snapshot violated its binding contract")
        _tree(config)
        object.__setattr__(self, "config", config)
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
        if not isinstance(self.artifact_inputs, tuple):
            raise TypeError("artifact_inputs must be an immutable tuple")
        inputs = tuple(item for item in self.artifact_inputs)
        if any(not isinstance(item, ArtifactInputRef) for item in inputs):
            raise TypeError("artifact_inputs must contain ArtifactInputRef values")
        try:
            provenance = ProcessorStageProvenance(
                "0" * 64,
                inputs,
            )
        except ValueError as error:
            if str(error) == "processor stage repeats an artifact input":
                raise ValueError(
                    "artifact_inputs must not repeat the same typed reference"
                ) from error
            raise
        object.__setattr__(
            self,
            "artifact_inputs",
            provenance.direct_artifact_inputs,
        )
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
                "artifact_inputs": [item.fingerprint for item in self.artifact_inputs],
            }
        )


__all__ = [
    "BoundStreamProcessor",
    "JoinKeyTransform",
    "StreamCardinality",
    "StreamJoinPolicy",
    "StreamProcessorDefinition",
]
