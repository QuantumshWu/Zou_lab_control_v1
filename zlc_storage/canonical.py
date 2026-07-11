"""The single canonical primitive encoder used by every ZLC artifact owner.

The format is deliberately small: domain owners first project their frozen value
objects to primitive trees, then this module turns those trees into stable bytes.
It knows nothing about artifact kinds, repositories, axes, figures, pulses, or runs.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


CANONICAL_MEDIA_TYPE = "application/vnd.zlc.canonical-v1+json"
_FRAME = b"ZLC-CANONICAL-1\n"


class CanonicalEncodingError(ValueError):
    """Raised when a value is outside the closed canonical primitive model."""


def sha256_digest(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 digest of an already encoded byte sequence."""

    return hashlib.sha256(bytes(data)).hexdigest()


def canonical_digest(value: Any) -> str:
    """Return the SHA-256 digest of :func:`encode` for ``value``."""

    return sha256_digest(encode(value))


def encode(value: Any) -> bytes:
    """Encode a canonical primitive tree into deterministic framed UTF-8 bytes.

    Supported leaves are ``None``, bool, int, float, str, bytes-like values and
    non-object NumPy arrays. Lists and tuples share the canonical list meaning;
    mappings require string keys. Arrays are normalized to C-contiguous,
    little-endian storage so host byte order and input strides cannot change an
    artifact digest.
    """

    tagged = _encode_value(value, path="$")
    payload = json.dumps(
        tagged,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return _FRAME + payload


def decode(data: bytes | bytearray | memoryview) -> Any:
    """Decode canonical bytes and reject alternate/non-canonical spellings."""

    raw = bytes(data)
    if not raw.startswith(_FRAME):
        raise CanonicalEncodingError("missing canonical v1 frame")
    try:
        tagged = json.loads(raw[len(_FRAME) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalEncodingError("invalid canonical UTF-8/JSON payload") from exc
    value = _decode_value(tagged, path="$")
    if encode(value) != raw:
        raise CanonicalEncodingError("payload is valid but not in canonical form")
    return value


def _encode_value(value: Any, *, path: str) -> list[Any]:
    if value is None:
        return ["null"]
    if isinstance(value, (bool, np.bool_)):
        return ["bool", bool(value)]
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return ["int", str(int(value))]
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise CanonicalEncodingError(f"{path}: floating scalars wider than float64 are forbidden")
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            bits = "nan"
        elif math.isinf(number):
            bits = "+inf" if number > 0 else "-inf"
        else:
            bits = struct.pack(">d", number).hex()
        return ["float64", bits]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = base64.b64encode(bytes(value)).decode("ascii")
        return ["bytes", payload]
    if isinstance(value, np.ndarray):
        return _encode_array(value, path=path)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalEncodingError(f"{path}: mapping keys must be strings")
        items: list[list[Any]] = []
        for key in sorted(value):
            items.append([key, _encode_value(value[key], path=f"{path}.{key}")])
        return ["map", items]
    if isinstance(value, Sequence):
        return [
            "list",
            [_encode_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)],
        ]
    raise CanonicalEncodingError(
        f"{path}: unsupported canonical value type {type(value).__name__}"
    )


def _encode_array(value: np.ndarray, *, path: str) -> list[Any]:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.fields is not None:
        raise CanonicalEncodingError(f"{path}: object and structured ndarrays are forbidden")
    dtype = array.dtype.newbyteorder("<")
    normalized = np.array(array, dtype=dtype, order="C", copy=True).reshape(array.shape)
    if dtype.kind == "f":
        canonical_nan = np.array(float("nan"), dtype=dtype)
        np.copyto(normalized, canonical_nan, where=np.isnan(normalized))
    elif dtype.kind == "c":
        component_dtype = np.dtype("<f4" if dtype.itemsize == 8 else "<f8")
        components = normalized.view(component_dtype)
        canonical_nan = np.array(float("nan"), dtype=component_dtype)
        np.copyto(components, canonical_nan, where=np.isnan(components))
    payload = base64.b64encode(normalized.tobytes(order="C")).decode("ascii")
    return [
        "ndarray",
        [dtype.str, [str(int(size)) for size in normalized.shape], payload],
    ]


def _decode_value(tagged: Any, *, path: str) -> Any:
    if not isinstance(tagged, list) or not tagged or not isinstance(tagged[0], str):
        raise CanonicalEncodingError(f"{path}: expected a tagged canonical value")
    tag = tagged[0]
    if tag == "null":
        _require_arity(tagged, 1, path)
        return None
    if tag == "bool":
        _require_arity(tagged, 2, path)
        if type(tagged[1]) is not bool:
            raise CanonicalEncodingError(f"{path}: bool payload must be JSON bool")
        return tagged[1]
    if tag == "int":
        _require_arity(tagged, 2, path)
        text = tagged[1]
        if not isinstance(text, str) or str(int(text)) != text:
            raise CanonicalEncodingError(f"{path}: integer is not canonical decimal")
        return int(text)
    if tag == "float64":
        _require_arity(tagged, 2, path)
        bits = tagged[1]
        if bits == "nan":
            return float("nan")
        if bits == "+inf":
            return float("inf")
        if bits == "-inf":
            return float("-inf")
        if not isinstance(bits, str) or len(bits) != 16:
            raise CanonicalEncodingError(f"{path}: invalid float64 encoding")
        try:
            return struct.unpack(">d", bytes.fromhex(bits))[0]
        except (ValueError, struct.error) as exc:
            raise CanonicalEncodingError(f"{path}: invalid float64 bits") from exc
    if tag == "str":
        _require_arity(tagged, 2, path)
        if not isinstance(tagged[1], str):
            raise CanonicalEncodingError(f"{path}: str payload must be text")
        return tagged[1]
    if tag == "bytes":
        _require_arity(tagged, 2, path)
        try:
            return base64.b64decode(tagged[1], validate=True)
        except (TypeError, ValueError) as exc:
            raise CanonicalEncodingError(f"{path}: invalid base64 bytes") from exc
    if tag == "list":
        _require_arity(tagged, 2, path)
        if not isinstance(tagged[1], list):
            raise CanonicalEncodingError(f"{path}: list payload must be a JSON list")
        return [_decode_value(item, path=f"{path}[{index}]") for index, item in enumerate(tagged[1])]
    if tag == "map":
        _require_arity(tagged, 2, path)
        return _decode_map(tagged[1], path=path)
    if tag == "ndarray":
        _require_arity(tagged, 2, path)
        return _decode_array(tagged[1], path=path)
    raise CanonicalEncodingError(f"{path}: unknown canonical tag {tag!r}")


def _decode_map(payload: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(payload, list):
        raise CanonicalEncodingError(f"{path}: map payload must be a JSON list")
    result: dict[str, Any] = {}
    previous: str | None = None
    for index, pair in enumerate(payload):
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
            raise CanonicalEncodingError(f"{path}: invalid map entry {index}")
        key = pair[0]
        if previous is not None and key <= previous:
            raise CanonicalEncodingError(f"{path}: map keys must be unique and sorted")
        result[key] = _decode_value(pair[1], path=f"{path}.{key}")
        previous = key
    return result


def _decode_array(payload: Any, *, path: str) -> np.ndarray:
    if not isinstance(payload, list) or len(payload) != 3:
        raise CanonicalEncodingError(f"{path}: invalid ndarray payload")
    dtype_text, shape_text, encoded = payload
    try:
        dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"{path}: invalid ndarray dtype") from exc
    if dtype.hasobject or dtype.fields is not None or dtype != dtype.newbyteorder("<"):
        raise CanonicalEncodingError(f"{path}: ndarray dtype is not canonical little-endian")
    if not isinstance(shape_text, list):
        raise CanonicalEncodingError(f"{path}: ndarray shape must be a list")
    try:
        shape = tuple(int(size) for size in shape_text)
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"{path}: invalid ndarray shape") from exc
    if any(str(size) != text or size < 0 for size, text in zip(shape, shape_text)):
        raise CanonicalEncodingError(f"{path}: ndarray shape is not canonical")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"{path}: invalid ndarray base64") from exc
    expected = math.prod(shape) * dtype.itemsize
    if len(raw) != expected:
        raise CanonicalEncodingError(
            f"{path}: ndarray byte length {len(raw)} does not match expected {expected}"
        )
    array = np.frombuffer(raw, dtype=dtype).reshape(shape, order="C").copy()
    array.setflags(write=False)
    return array


def _require_arity(tagged: list[Any], arity: int, path: str) -> None:
    if len(tagged) != arity:
        raise CanonicalEncodingError(f"{path}: tag {tagged[0]!r} expects {arity - 1} payload fields")
