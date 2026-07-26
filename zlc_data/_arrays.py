"""Internal ndarray ownership helpers for immutable data values."""

from __future__ import annotations

import numpy as np


_IMMUTABLE_FALSE = np.frombuffer(b"\x00", dtype=np.bool_)
_IMMUTABLE_TRUE = np.frombuffer(b"\x01", dtype=np.bool_)


def canonical_dtype(dtype) -> np.dtype:
    """Return the platform-independent little-endian spelling of ``dtype``."""

    result = np.dtype(dtype)
    if result.hasobject or result.fields is not None:
        raise TypeError("object and structured dtypes are not supported")
    if result.kind not in "biufc":
        raise TypeError(
            f"data values require bool or numeric dtype, got {result}"
        )
    if result.kind in "iu" and result.itemsize not in (1, 2, 4, 8):
        raise TypeError(f"unsupported integer dtype width: {result}")
    if result.kind == "f" and result.itemsize not in (2, 4, 8):
        raise TypeError(f"unsupported real dtype width: {result}")
    if result.kind == "c" and result.itemsize not in (8, 16):
        raise TypeError(f"unsupported complex dtype width: {result}")
    return result.newbyteorder("<")


def is_intrinsically_immutable_array(value: object) -> bool:
    """Return whether an ndarray view ultimately rests on immutable ``bytes``.

    ``flags.writeable = False`` is not sufficient: an owning ndarray, or a view
    over one, can be made writable again.  By contrast, slices, transposes and
    broadcasts whose complete base chain ends in ``bytes`` cannot acquire a
    writable buffer.  Recognising that ownership fact lets every downstream
    value object retain the exact view instead of copying the same camera frame
    at every semantic boundary.
    """

    if type(value) is not np.ndarray or value.dtype.hasobject or value.dtype.fields:
        return False
    current: object = value
    seen: set[int] = set()
    while isinstance(current, np.ndarray):
        identity = id(current)
        if (
            identity in seen
            or current.flags.writeable
            or current.flags.owndata
        ):
            return False
        seen.add(identity)
        current = current.base
    if isinstance(current, bytes):
        return True
    while isinstance(current, memoryview):
        identity = id(current)
        if identity in seen or not current.readonly:
            return False
        seen.add(identity)
        current = current.obj
    return isinstance(current, bytes)


def immutable_array(values, *, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Return an intrinsically immutable ndarray, copying only mutable input.

    A mutable producer/builder buffer crosses this boundary through one compact
    ``bytes`` snapshot.  An already bytes-backed immutable slice, transpose or
    broadcast is returned unchanged: it cannot be made writable and therefore
    needs no second ownership copy.  Strides are part of the retained view;
    canonical persistence owners may materialise contiguous bytes when they
    actually serialize it.
    """

    source = np.asarray(values)
    source_dtype = canonical_dtype(source.dtype)
    if source_dtype != dtype:
        raise TypeError(f"values dtype {source.dtype} does not match schema dtype {dtype}")
    if source.shape != shape:
        raise ValueError(f"values shape {source.shape} does not match expected {shape}")
    if source.dtype == dtype and is_intrinsically_immutable_array(source):
        return source
    normalized = source.astype(dtype, copy=False)
    result = np.frombuffer(normalized.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def immutable_bool_broadcast(value: bool, shape: tuple[int, ...]) -> np.ndarray:
    """Return a zero-allocation uniform boolean plane backed by one byte."""

    if not isinstance(value, (bool, np.bool_)):
        raise TypeError("immutable boolean broadcast value must be bool")
    normalized_shape = tuple(shape)
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in normalized_shape
    ):
        raise ValueError("immutable boolean broadcast shape must be nonnegative integers")
    scalar = _IMMUTABLE_TRUE if bool(value) else _IMMUTABLE_FALSE
    return np.broadcast_to(scalar.reshape((1,) * len(normalized_shape)), normalized_shape)


def immutable_bool_array(values, *, shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype != np.dtype(bool):
        raise TypeError(f"validity mask dtype must be bool, got {source.dtype}")
    return immutable_array(source, dtype=np.dtype(bool), shape=shape)
