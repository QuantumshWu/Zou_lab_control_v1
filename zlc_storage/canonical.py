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
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Callable, TypeAlias

import numpy as np


_FRAME = b"ZLC-CANONICAL-1\n"


class CanonicalEncodingError(ValueError):
    """Raised when a value is outside the closed canonical primitive model."""


def canonical_text(
    value: object,
    field: str,
    *,
    empty: bool = False,
) -> str:
    """Validate the one canonical text invariant used by all value owners."""

    if not isinstance(value, str) or value.strip() != value or (not empty and not value):
        qualifier = "canonical text" if empty else "canonical non-empty text"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def normalized_text(value: object, field: str) -> str:
    """Normalize one non-empty human/external text input before binding it."""

    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} cannot be empty")
    return normalized


def sha256_text(value: object, field: str, *, optional: bool = False) -> str | None:
    """Validate a lowercase SHA-256 text value without recomputing its content."""

    if optional and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        suffix = " or None" if optional else ""
        raise ValueError(f"{field} must be a lowercase SHA-256 digest{suffix}")
    return value


def integer(
    value: object,
    field: str,
    *,
    optional: bool = False,
    minimum: int | None = None,
    nonnegative: bool = False,
) -> int | None:
    """Validate an integer once and normalize ``numbers.Integral`` to ``int``."""

    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        suffix = " or None" if optional else ""
        raise TypeError(f"{field} must be an integer{suffix}")
    normalized = int(value)
    lower_bound = 0 if nonnegative and minimum is None else minimum
    if lower_bound is not None and normalized < lower_bound:
        raise ValueError(f"{field} must be at least {lower_bound}")
    return normalized


def nonnegative_integer(value: object, field: str) -> int:
    result = integer(value, field, minimum=0)
    assert result is not None
    return result


def positive_integer(value: object, field: str) -> int:
    result = integer(value, field, minimum=1)
    assert result is not None
    return result


def finite_real(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    positive: bool = False,
    normalize_zero: bool = True,
) -> float:
    """Validate a finite real number under one shared numeric boundary rule."""

    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{field} must be positive")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return 0.0 if normalize_zero and result == 0.0 else result


def nonnegative_real(value: object, field: str) -> float:
    return finite_real(value, field, minimum=0.0)


def positive_real(value: object, field: str) -> float:
    return finite_real(value, field, positive=True)


def exact_mapping(
    value: object,
    fields: set[str] | frozenset[str],
    format_name: str,
    *,
    discriminator: str | None = "schema",
) -> dict[str, Any]:
    """Admit one exact owner mapping and, when present, its discriminator."""

    expected = set(fields)
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{format_name} must contain exactly {sorted(expected)}")
    if discriminator is not None and value.get(discriminator) != format_name:
        raise ValueError(
            f"expected {discriminator} {format_name!r}, "
            f"got {value.get(discriminator)!r}"
        )
    return value


@dataclass(frozen=True)
class CanonicalListEvent:
    path: tuple[str | int, ...]
    length: int


@dataclass(frozen=True)
class CanonicalArrayEvent:
    path: tuple[str | int, ...]
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def _probe_numpy_max_ndim() -> int:
    """Return this NumPy build's actual ndarray rank limit without private APIs."""

    def supported(rank: int) -> bool:
        try:
            np.empty((0,) * rank, dtype=np.uint8)
        except (TypeError, ValueError, OverflowError):
            return False
        return True

    lower = 0
    upper = 1
    while upper <= 1024 and supported(upper):
        lower = upper
        upper *= 2
    if upper > 1024:
        # NumPy has historically used a small fixed maximum (32 or 64).  This
        # guard keeps an unexpected implementation from making import unbounded.
        return lower
    while lower + 1 < upper:
        middle = (lower + upper) // 2
        if supported(middle):
            lower = middle
        else:
            upper = middle
    return lower


_NUMPY_MAX_NDIM = _probe_numpy_max_ndim()
_NUMPY_MAX_INDEX = int(np.iinfo(np.intp).max)


CanonicalStructureEvent: TypeAlias = CanonicalListEvent | CanonicalArrayEvent
CanonicalStructureAdmission: TypeAlias = Callable[
    [tuple[CanonicalStructureEvent, ...]],
    None,
]


def sha256_digest(data: bytes | bytearray | memoryview) -> str:
    """Return the lowercase SHA-256 digest of an already encoded byte sequence."""

    return hashlib.sha256(data).hexdigest()


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


def decode(
    data: bytes | bytearray | memoryview,
    *,
    admit_structure: CanonicalStructureAdmission | None = None,
) -> Any:
    """Decode canonical bytes, optionally admitting structure before arrays materialize.

    The admission hook receives a complete immutable list/ndarray inventory
    after framed JSON and every tagged node have been validated, but before any
    ndarray base64 is decoded or NumPy array is allocated.  Domain owners use
    it only for exact wire/schema facts such as forbidden arrays, declared
    shapes and fixed chunk cardinality.  The second pass remains the sole value
    materializer.
    """

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("canonical payload must be bytes-like")
    raw = bytes(data)
    if not raw.startswith(_FRAME):
        raise CanonicalEncodingError("missing canonical v1 frame")
    try:
        tagged = json.loads(raw[len(_FRAME) :].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise CanonicalEncodingError("invalid canonical UTF-8/JSON payload") from exc
    try:
        canonical_json = json.dumps(
            tagged,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, RecursionError) as exc:
        raise CanonicalEncodingError("invalid canonical JSON structure") from exc
    if _FRAME + canonical_json != raw:
        raise CanonicalEncodingError("payload is valid but not in canonical form")
    if admit_structure is not None and not callable(admit_structure):
        raise TypeError("admit_structure must be callable or None")
    events: list[CanonicalStructureEvent] = []
    try:
        _inspect_value(tagged, path="$", event_path=(), events=events)
    except RecursionError as exc:
        raise CanonicalEncodingError("canonical structure exceeds recursion limit") from exc
    if admit_structure is not None:
        admit_structure(tuple(events))
    try:
        value = _decode_value(tagged, path="$")
        rebuilt = encode(value)
    except RecursionError as exc:
        raise CanonicalEncodingError("canonical structure exceeds recursion limit") from exc
    if rebuilt != raw:
        raise CanonicalEncodingError("payload is valid but not in canonical form")
    return value


_BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_BASE64_INDEX = {character: index for index, character in enumerate(_BASE64_ALPHABET)}


def _canonical_base64_nbytes(encoded: Any, *, path: str) -> int:
    if not isinstance(encoded, str):
        raise CanonicalEncodingError(f"{path}: base64 payload must be text")
    if len(encoded) % 4:
        raise CanonicalEncodingError(f"{path}: invalid base64 length")
    padding = len(encoded) - len(encoded.rstrip("="))
    if padding > 2 or "=" in encoded[: len(encoded) - padding]:
        raise CanonicalEncodingError(f"{path}: invalid base64 padding")
    body = encoded[: len(encoded) - padding] if padding else encoded
    if any(character not in _BASE64_INDEX for character in body):
        raise CanonicalEncodingError(f"{path}: invalid base64 alphabet")
    if padding == 2 and (not body or _BASE64_INDEX[body[-1]] & 0x0F):
        raise CanonicalEncodingError(f"{path}: non-canonical base64 pad bits")
    if padding == 1 and (not body or _BASE64_INDEX[body[-1]] & 0x03):
        raise CanonicalEncodingError(f"{path}: non-canonical base64 pad bits")
    return (len(encoded) // 4) * 3 - padding


def _inspect_value(
    tagged: Any,
    *,
    path: str,
    event_path: tuple[str | int, ...],
    events: list[CanonicalStructureEvent],
) -> None:
    if not isinstance(tagged, list) or not tagged or not isinstance(tagged[0], str):
        raise CanonicalEncodingError(f"{path}: expected a tagged canonical value")
    tag = tagged[0]
    if tag == "null":
        _require_arity(tagged, 1, path)
        return
    if tag == "bool":
        _require_arity(tagged, 2, path)
        if type(tagged[1]) is not bool:
            raise CanonicalEncodingError(f"{path}: bool payload must be JSON bool")
        return
    if tag == "int":
        _require_arity(tagged, 2, path)
        text = tagged[1]
        try:
            canonical = isinstance(text, str) and str(int(text)) == text
        except ValueError:
            canonical = False
        if not canonical:
            raise CanonicalEncodingError(f"{path}: integer is not canonical decimal")
        return
    if tag == "float64":
        _require_arity(tagged, 2, path)
        bits = tagged[1]
        if bits in {"nan", "+inf", "-inf"}:
            return
        if not isinstance(bits, str) or len(bits) != 16:
            raise CanonicalEncodingError(f"{path}: invalid float64 encoding")
        try:
            struct.unpack(">d", bytes.fromhex(bits))[0]
        except (ValueError, struct.error) as exc:
            raise CanonicalEncodingError(f"{path}: invalid float64 bits") from exc
        return
    if tag == "str":
        _require_arity(tagged, 2, path)
        if not isinstance(tagged[1], str):
            raise CanonicalEncodingError(f"{path}: str payload must be text")
        return
    if tag == "bytes":
        _require_arity(tagged, 2, path)
        _canonical_base64_nbytes(tagged[1], path=path)
        return
    if tag == "list":
        _require_arity(tagged, 2, path)
        payload = tagged[1]
        if not isinstance(payload, list):
            raise CanonicalEncodingError(f"{path}: list payload must be a JSON list")
        events.append(CanonicalListEvent(event_path, len(payload)))
        for index, item in enumerate(payload):
            _inspect_value(
                item,
                path=f"{path}[{index}]",
                event_path=event_path + (index,),
                events=events,
            )
        return
    if tag == "map":
        _require_arity(tagged, 2, path)
        _inspect_map(
            tagged[1],
            path=path,
            event_path=event_path,
            events=events,
        )
        return
    if tag == "ndarray":
        _require_arity(tagged, 2, path)
        event = _inspect_array(tagged[1], path=path, event_path=event_path)
        events.append(event)
        return
    raise CanonicalEncodingError(f"{path}: unknown canonical tag {tag!r}")


def _inspect_map(
    payload: Any,
    *,
    path: str,
    event_path: tuple[str | int, ...],
    events: list[CanonicalStructureEvent],
) -> None:
    if not isinstance(payload, list):
        raise CanonicalEncodingError(f"{path}: map payload must be a JSON list")
    previous: str | None = None
    for index, pair in enumerate(payload):
        if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
            raise CanonicalEncodingError(f"{path}: invalid map entry {index}")
        key = pair[0]
        if previous is not None and key <= previous:
            raise CanonicalEncodingError(f"{path}: map keys must be unique and sorted")
        _inspect_value(
            pair[1],
            path=f"{path}.{key}",
            event_path=event_path + (key,),
            events=events,
        )
        previous = key


def _inspect_array(
    payload: Any,
    *,
    path: str,
    event_path: tuple[str | int, ...],
) -> CanonicalArrayEvent:
    if not isinstance(payload, list) or len(payload) != 3:
        raise CanonicalEncodingError(f"{path}: invalid ndarray payload")
    dtype_text, shape_text, encoded = payload
    try:
        dtype = np.dtype(dtype_text)
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"{path}: invalid ndarray dtype") from exc
    if (
        not isinstance(dtype_text, str)
        or dtype_text != dtype.str
        or dtype.itemsize == 0
        or dtype.hasobject
        or dtype.fields is not None
        or dtype != dtype.newbyteorder("<")
    ):
        raise CanonicalEncodingError(f"{path}: ndarray dtype is not canonical little-endian")
    if not isinstance(shape_text, list):
        raise CanonicalEncodingError(f"{path}: ndarray shape must be a list")
    try:
        shape = tuple(int(size) for size in shape_text)
    except (TypeError, ValueError) as exc:
        raise CanonicalEncodingError(f"{path}: invalid ndarray shape") from exc
    if any(
        not isinstance(text, str) or str(size) != text or size < 0
        for size, text in zip(shape, shape_text)
    ):
        raise CanonicalEncodingError(f"{path}: ndarray shape is not canonical")
    if len(shape) > _NUMPY_MAX_NDIM:
        raise CanonicalEncodingError(
            f"{path}: ndarray rank {len(shape)} exceeds NumPy limit {_NUMPY_MAX_NDIM}"
        )
    if any(size > _NUMPY_MAX_INDEX for size in shape):
        raise CanonicalEncodingError(
            f"{path}: ndarray dimension exceeds NumPy index range"
        )
    # A zero-sized dimension makes the mathematical element count zero, but
    # NumPy still computes intermediate extents/strides for the other axes.
    # Check those products explicitly so shapes such as (2**40, 2**40, 0)
    # cannot reach admission and fail only during the later reshape.
    logical_extent = dtype.itemsize
    if logical_extent > _NUMPY_MAX_INDEX:
        raise CanonicalEncodingError(
            f"{path}: ndarray dtype extent exceeds NumPy index range"
        )
    for size in shape:
        factor = max(1, size)
        if logical_extent > _NUMPY_MAX_INDEX // factor:
            raise CanonicalEncodingError(
                f"{path}: ndarray logical extent exceeds NumPy index range"
            )
        logical_extent *= factor
    element_count = math.prod(shape)
    if element_count > _NUMPY_MAX_INDEX:
        raise CanonicalEncodingError(
            f"{path}: ndarray element count exceeds NumPy index range"
        )
    expected = element_count * dtype.itemsize
    encoded_nbytes = _canonical_base64_nbytes(encoded, path=path)
    if encoded_nbytes != expected:
        raise CanonicalEncodingError(
            f"{path}: ndarray byte length {encoded_nbytes} does not match expected {expected}"
        )
    return CanonicalArrayEvent(event_path, shape, dtype.str, expected)


def _encode_value(
    value: Any,
    *,
    path: str,
) -> list[Any]:
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
            items.append(
                [
                    key,
                    _encode_value(
                        value[key],
                        path=f"{path}.{key}",
                    ),
                ]
            )
        return ["map", items]
    if isinstance(value, Sequence):
        return [
            "list",
            [
                _encode_value(
                    item,
                    path=f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ],
        ]
    raise CanonicalEncodingError(
        f"{path}: unsupported canonical value type {type(value).__name__}"
    )


def _encode_array(
    value: np.ndarray,
    *,
    path: str,
) -> list[Any]:
    array = np.asarray(value)
    if array.dtype.itemsize == 0 or array.dtype.hasobject or array.dtype.fields is not None:
        raise CanonicalEncodingError(
            f"{path}: zero-itemsize, object, and structured ndarrays are forbidden"
        )
    dtype = array.dtype.newbyteorder("<")
    normalized = np.array(array, dtype=dtype, order="C", copy=True).reshape(array.shape)
    if dtype.kind == "f":
        canonical_nan = np.array(float("nan"), dtype=dtype)
        np.copyto(normalized, canonical_nan, where=np.isnan(normalized))
    elif dtype.kind == "c":
        component_dtype = np.dtype("<f4" if dtype.itemsize == 8 else "<f8")
        # NumPy 2 rejects an itemsize-changing ``view`` directly on a 0-D
        # array.  Flattening only the view (not the stored value) gives scalar
        # and N-D complex arrays the same component-wise NaN normalization.
        components = normalized.reshape(-1).view(component_dtype)
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
    if (
        not isinstance(dtype_text, str)
        or dtype_text != dtype.str
        or dtype.itemsize == 0
        or dtype.hasobject
        or dtype.fields is not None
        or dtype != dtype.newbyteorder("<")
    ):
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
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(shape, order="C")
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalEncodingError(
            f"{path}: ndarray shape cannot be materialized by NumPy"
        ) from exc
    array.setflags(write=False)
    return array


def _require_arity(tagged: list[Any], arity: int, path: str) -> None:
    if len(tagged) != arity:
        raise CanonicalEncodingError(f"{path}: tag {tagged[0]!r} expects {arity - 1} payload fields")
