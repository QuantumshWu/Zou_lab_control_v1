"""Internal ndarray ownership helpers for immutable data values."""

from __future__ import annotations

import numpy as np


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


def immutable_array(values, *, dtype: np.dtype, shape: tuple[int, ...]) -> np.ndarray:
    """Copy values into an intrinsically read-only, C-contiguous ndarray.

    A normal owning ndarray can have ``writeable`` re-enabled by a consumer.  This
    representation is backed by immutable ``bytes``, so the published value cannot
    be made writable and cannot alias a producer or builder buffer.
    """

    source = np.asarray(values)
    source_dtype = canonical_dtype(source.dtype)
    if source_dtype != dtype:
        raise TypeError(f"values dtype {source.dtype} does not match schema dtype {dtype}")
    if source.shape != shape:
        raise ValueError(f"values shape {source.shape} does not match expected {shape}")
    normalized = np.ascontiguousarray(source.astype(dtype, copy=False)).reshape(shape)
    result = np.frombuffer(normalized.tobytes(order="C"), dtype=dtype).reshape(shape)
    result.setflags(write=False)
    return result


def immutable_bool_array(values, *, shape: tuple[int, ...]) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype != np.dtype(bool):
        raise TypeError(f"validity mask dtype must be bool, got {source.dtype}")
    return immutable_array(source, dtype=np.dtype(bool), shape=shape)
