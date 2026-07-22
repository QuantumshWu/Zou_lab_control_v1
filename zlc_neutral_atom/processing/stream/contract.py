"""Declarative processor catalog values and one explicitly bound operator."""

from __future__ import annotations

import hashlib
import inspect
import math
import sys
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Callable

import numpy as np

from zlc_storage import (
    canonical_digest,
    canonical_text as _text,
    sha256_text as _digest,
)

from zlc_neutral_atom.catalog import (
    StreamProcessorDefinition as _StreamProcessorDefinition,
)
from zlc_neutral_atom.runtime.streams import (
    ArtifactInputRef,
    JoinKeyContract,
    PayloadContract,
    ProcessorStageProvenance,
    StreamId,
)


def _canonical_dtype(value: np.dtype) -> np.dtype:
    """Admit one supported scalar dtype at the binding boundary."""

    dtype = np.dtype(value)
    canonical = dtype.newbyteorder("<")
    if (
        dtype.hasobject
        or dtype.metadata is not None
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.itemsize == 0
        or dtype.kind not in "biufc"
        or dtype != canonical
        or (dtype.kind == "b" and dtype.itemsize != 1)
        or (dtype.kind in "iu" and dtype.itemsize not in (1, 2, 4, 8))
        or (dtype.kind == "f" and dtype.itemsize not in (2, 4, 8))
        or (dtype.kind == "c" and dtype.itemsize not in (8, 16))
    ):
        raise TypeError(
            "processor config dtypes must be canonical little-endian scalar numeric dtypes"
        )
    return canonical


def _dtype_tree(dtype: np.dtype) -> dict[str, object]:
    """Project an already-admitted dtype without owning its validation again."""

    return {
        "$type": "numpy.dtype",
        "str": dtype.str,
        "kind": dtype.kind,
        "itemsize": int(dtype.itemsize),
        "byteorder": "|" if dtype.byteorder == "|" else "<",
    }


def _resolve_module_member(module_name: str, qualname: str) -> object | None:
    resolved: object = sys.modules.get(module_name)
    if resolved is None or "<locals>" in qualname.split("."):
        return None
    for name in qualname.split("."):
        try:
            resolved = vars(resolved)[name]
        except (KeyError, TypeError):
            return None
    return resolved


def _admit_stable_binding_type(value: object) -> None:
    """Require the fingerprint type tag to resolve to this exact class."""

    cls = type(value)
    if _resolve_module_member(cls.__module__, cls.__qualname__) is not cls:
        raise TypeError("processor config types must be stable module-owned classes")


def _admit_declared_dataclass_storage(
    value: object,
    dataclass_fields: tuple[object, ...],
) -> None:
    """Reject instance state that the canonical dataclass tree cannot see."""

    field_names = {item.name for item in dataclass_fields}
    namespace = getattr(value, "__dict__", None)
    if namespace is not None and set(namespace) - field_names:
        raise TypeError("processor config dataclasses may only store declared fields")
    permitted_slots = field_names | {"__dict__", "__weakref__"}
    for owner in type(value).__mro__:
        slots = vars(owner).get("__slots__", ())
        slots = (slots,) if isinstance(slots, str) else tuple(slots)
        if set(slots) - permitted_slots:
            raise TypeError("processor config dataclasses may only store declared fields")


def _tree(
    value: object,
    memo: dict[int, tuple[object, object]] | None = None,
) -> object:
    """Canonical projection of data already admitted by the binding owner."""

    memo = {} if memo is None else memo
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, bytes):
        return {
            "$type": "bytes-content",
            "nbytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.dtype):
        return _dtype_tree(value)
    identity = id(value)
    memoized = memo.get(identity)
    if memoized is not None and memoized[0] is value:
        return memoized[1]
    if isinstance(value, np.ndarray):
        result = {
            "$type": "numpy.ndarray-content",
            "dtype": _dtype_tree(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(memoryview(value).cast("B")).hexdigest(),
        }
    elif isinstance(value, Enum):
        result = {
            "$type": "enum",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _tree(value.value, memo),
        }
    elif isinstance(value, tuple):
        result = {
            "$type": "tuple",
            "items": [_tree(item, memo) for item in value],
        }
    elif isinstance(value, frozenset):
        items = tuple(_tree(item, memo) for item in value)
        result = {
            "$type": "frozenset",
            "items": sorted(
                items,
                key=lambda item: canonical_digest({"item": item}),
            ),
        }
    elif isinstance(value, MappingProxyType):
        result = {
            "$type": "mapping",
            "entries": [
                [key, _tree(item, memo)]
                for key, item in sorted(value.items(), key=lambda pair: pair[0])
            ],
        }
    elif is_dataclass(value) and not isinstance(value, type):
        result = {
            "$type": "dataclass",
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _tree(getattr(value, field.name), memo)]
                for field in fields(value)
            ],
        }
    else:
        raise TypeError(f"unsupported declarative binding value {type(value).__name__}")
    memo[identity] = (value, result)
    return result


def _snapshot_immutable_array(value: object) -> np.ndarray:
    """Own an ndarray header directly over its terminal immutable bytes."""

    message = "processor config arrays must be canonical immutable ndarrays"
    if type(value) is not np.ndarray:
        raise TypeError(message)
    try:
        dtype = _canonical_dtype(value.dtype)
    except TypeError as error:
        raise TypeError(message) from error
    if value.ndim == 0 or not value.flags.c_contiguous:
        raise TypeError(message)
    stride = dtype.itemsize
    reversed_strides: list[int] = []
    for size in reversed(value.shape):
        reversed_strides.append(stride)
        stride *= max(size, 1)
    canonical_strides = tuple(reversed(reversed_strides))
    if value.strides != canonical_strides:
        raise TypeError(message)
    current: object = value
    seen: set[int] = set()
    while isinstance(current, np.ndarray):
        identity = id(current)
        if identity in seen or current.flags.writeable or current.flags.owndata:
            raise TypeError(message)
        seen.add(identity)
        current = current.base
    if type(current) is bytes:
        backing = current
    elif (
        isinstance(current, memoryview)
        and current.readonly
        and type(current.obj) is bytes
    ):
        backing = current.obj
    else:
        raise TypeError(message)
    root = np.frombuffer(backing, dtype=np.uint8)
    if root.nbytes != value.nbytes:
        raise TypeError(message)
    if value.nbytes:
        if int(value.__array_interface__["data"][0]) != int(
            root.__array_interface__["data"][0]
        ):
            raise TypeError(message)
    return np.ndarray(
        value.shape,
        dtype=dtype,
        buffer=backing,
        offset=0,
        strides=canonical_strides,
    )


def _snapshot_binding_value(
    value: object,
    memo: dict[int, tuple[object, object]] | None = None,
    active: set[int] | None = None,
    preserve: dict[int, object] | None = None,
) -> object:
    """Admit and owner-rebuild one config graph at its sole boundary."""

    preserve = {} if preserve is None else preserve
    if value is None or type(value) in (bool, int, str, bytes):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise TypeError("processor config floats must be finite")
        return value
    memo = {} if memo is None else memo
    active = set() if active is None else active
    identity = id(value)
    memoized = memo.get(identity)
    if memoized is not None and memoized[0] is value:
        return memoized[1]
    dataclass_value = is_dataclass(value) and not isinstance(value, type)
    if dataclass_value and type(value).__bases__ != (object,):
        raise TypeError("processor config dataclasses require plain dataclass storage")
    if isinstance(value, np.dtype):
        result = _canonical_dtype(value)
    elif isinstance(value, np.ndarray):
        result = _snapshot_immutable_array(value)
    else:
        if identity in active:
            raise TypeError("processor config contains a recursive cycle")
        active.add(identity)
        try:
            if isinstance(value, Enum):
                _admit_stable_binding_type(value)
                snapshot = _snapshot_binding_value(value.value, memo, active, preserve)
                if snapshot is not value.value:
                    raise TypeError(
                        "processor config Enum values must be intrinsically immutable"
                    )
                result = value
            elif isinstance(value, tuple):
                result = tuple(
                    _snapshot_binding_value(item, memo, active, preserve)
                    for item in value
                )
            elif isinstance(value, frozenset):
                result = frozenset(
                    _snapshot_binding_value(item, memo, active, preserve)
                    for item in value
                )
            elif isinstance(value, MappingProxyType):
                for key in value:
                    if type(key) is not str:
                        raise TypeError(
                            "processor config mappings require canonical string keys"
                        )
                    _text(key, "processor config mapping key")
                result = MappingProxyType(
                    {
                        key: _snapshot_binding_value(
                            value[key],
                            memo,
                            active,
                            preserve,
                        )
                        for key in sorted(value)
                    }
                )
            elif dataclass_value:
                _admit_stable_binding_type(value)
                namespace = vars(type(value))
                parameters = namespace.get("__dataclass_params__")
                declared_fields = namespace.get("__dataclass_fields__")
                if (
                    parameters is None
                    or declared_fields is None
                    or not parameters.frozen
                ):
                    raise TypeError("processor config dataclasses must be frozen")
                items = fields(value)
                _admit_declared_dataclass_storage(value, items)
                updates: dict[str, object] = {}
                for item in items:
                    admitted = _snapshot_binding_value(
                        getattr(value, item.name),
                        memo,
                        active,
                        preserve,
                    )
                    if item.init:
                        updates[item.name] = admitted
                try:
                    result = replace(value, **updates)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"cannot reconstruct processor config {type(value).__name__}"
                    ) from error
                if type(result) is not type(value):
                    raise TypeError(
                        f"cannot reconstruct processor config {type(value).__name__}"
                    )
                result_identity = id(result)
                active.add(result_identity)
                try:
                    result_items = fields(result)
                    _admit_declared_dataclass_storage(result, result_items)
                    for item in result_items:
                        admitted = _snapshot_binding_value(
                            getattr(result, item.name),
                            memo,
                            active,
                            preserve,
                        )
                        object.__setattr__(result, item.name, admitted)
                finally:
                    active.remove(result_identity)
            else:
                raise TypeError(
                    f"unsupported declarative binding value {type(value).__name__}"
                )
        finally:
            active.remove(identity)
    preserved = preserve.get(identity)
    if preserved is not None:
        result = preserved
    memo[identity] = (value, result)
    memo.setdefault(id(result), (result, result))
    return result


def _generation_owner_graph(*roots: object) -> dict[int, object]:
    """Collect identity-owned declarative descendants that configs may reference."""

    preserved: dict[int, object] = {}
    seen: set[int] = set()
    pending = list(roots)
    while pending:
        value = pending.pop()
        if value is None or type(value) in (bool, int, float, str, bytes):
            continue
        identity = id(value)
        if identity in seen:
            continue
        seen.add(identity)
        if isinstance(value, Enum):
            pending.append(value.value)
        elif isinstance(value, (tuple, frozenset)):
            pending.extend(value)
        elif isinstance(value, MappingProxyType):
            pending.extend(value.keys())
            pending.extend(value.values())
        elif is_dataclass(value) and not isinstance(value, type):
            preserved[identity] = value
            pending.extend(getattr(value, item.name) for item in fields(value))
    return preserved


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

    definition: _StreamProcessorDefinition
    config: object
    input_payload_contract: PayloadContract
    output_payload_contract: PayloadContract
    join_key_contract: JoinKeyContract
    output_stream_id: StreamId
    output_source_id: str
    operator: Callable[[object, object], object]
    operator_deadline_seconds: float
    terminal_wait_seconds: float
    artifact_inputs: tuple[ArtifactInputRef, ...] = ()
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.definition, _StreamProcessorDefinition):
            raise TypeError("definition must be StreamProcessorDefinition")
        definition = _snapshot_binding_value(self.definition)
        if not isinstance(definition, _StreamProcessorDefinition):
            raise TypeError("definition snapshot changed its declared type")
        definition_tree = _tree(definition)
        object.__setattr__(self, "definition", definition)
        source_output_stream_id = self.output_stream_id
        if not isinstance(source_output_stream_id, StreamId):
            raise TypeError("output_stream_id must be StreamId")
        output_stream_id = StreamId(source_output_stream_id.value)
        object.__setattr__(self, "output_stream_id", output_stream_id)
        _text(self.output_source_id, "output_source_id")
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
        config_tree = _tree(config)
        object.__setattr__(self, "config", config)
        for field_name in (
            "operator_deadline_seconds",
            "terminal_wait_seconds",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{field_name} must be finite and positive")
            object.__setattr__(self, field_name, float(value))
        for name, contract in (
            ("input_payload_contract", self.input_payload_contract),
            ("output_payload_contract", self.output_payload_contract),
        ):
            _digest(getattr(contract, "fingerprint", None), f"{name}.fingerprint")
            for member in ("snapshot", "digest"):
                if not callable(getattr(contract, member, None)):
                    raise TypeError(f"{name}.{member} must be callable")
        _digest(
            getattr(self.join_key_contract, "fingerprint", None),
            "join_key_contract.fingerprint",
        )
        if not callable(getattr(self.join_key_contract, "snapshot", None)):
            raise TypeError("join_key_contract.snapshot must be callable")
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
            or _resolve_module_member(operator.__module__, operator.__qualname__)
            is not operator
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
        object.__setattr__(
            self,
            "_fingerprint",
            canonical_digest(
                {
                    "contract": "zlc_neutral_atom.BoundStreamProcessor",
                    "definition_key": str(definition.key),
                    "definition": definition_tree,
                    "config": config_tree,
                    "input_payload_contract_fingerprint": (
                        self.input_payload_contract.fingerprint
                    ),
                    "output_payload_contract_fingerprint": (
                        self.output_payload_contract.fingerprint
                    ),
                    "join_key_contract_fingerprint": (
                        self.join_key_contract.fingerprint
                    ),
                    "output_stream_id": output_stream_id.value,
                    "output_source_id": self.output_source_id,
                    "operator": f"{operator.__module__}.{operator.__qualname__}",
                    "operator_deadline_seconds": self.operator_deadline_seconds,
                    "terminal_wait_seconds": self.terminal_wait_seconds,
                    "artifact_inputs": [
                        item.fingerprint for item in self.artifact_inputs
                    ],
                }
            ),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint


__all__ = [
    "BoundStreamProcessor",
]
