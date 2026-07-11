"""Contracts for the one canonical primitive byte representation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from zlc_storage.canonical import CanonicalEncodingError, canonical_digest, decode, encode


def test_mapping_order_and_sequence_container_do_not_change_bytes():
    left = {"unicode": "原子", "items": (1, True, None), "z": -0.0}
    right = {"z": -0.0, "items": [1, True, None], "unicode": "原子"}

    assert encode(left) == encode(right)
    assert canonical_digest(left) == canonical_digest(right)


@pytest.mark.parametrize("value", [0.0, -0.0, float("inf"), float("-inf"), float("nan")])
def test_float_edges_round_trip_canonically(value):
    restored = decode(encode(value))
    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value
        assert math.copysign(1.0, restored) == math.copysign(1.0, value)


def test_nan_payloads_have_one_canonical_spelling():
    ordinary = np.float64(float("nan"))
    alternate = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0]

    assert encode(ordinary) == encode(alternate)


def test_ndarray_normalizes_byte_order_and_memory_order():
    native_c = np.arange(12, dtype=np.int16).reshape(3, 4)
    big_endian_f = np.asfortranarray(native_c.astype(">i2"))

    assert encode(native_c) == encode(big_endian_f)
    restored = decode(encode(big_endian_f))
    np.testing.assert_array_equal(restored, native_c)
    assert restored.flags.c_contiguous
    assert not restored.flags.writeable


def test_scalar_and_zero_length_arrays_keep_shape_and_dtype():
    for array in (np.array(7, dtype=np.uint16), np.empty((2, 0, 3), dtype=np.float32)):
        restored = decode(encode(array))
        assert restored.shape == array.shape
        assert restored.dtype == array.dtype.newbyteorder("<")
        np.testing.assert_array_equal(restored, array)


def test_object_arrays_and_non_string_map_keys_are_rejected():
    with pytest.raises(CanonicalEncodingError, match="object"):
        encode(np.array([{"unsafe": "pickle-like"}], dtype=object))
    with pytest.raises(CanonicalEncodingError, match="keys must be strings"):
        encode({1: "not a canonical map"})


def test_decoder_rejects_unframed_and_noncanonical_json():
    with pytest.raises(CanonicalEncodingError, match="frame"):
        decode(b'[["null"]]')

    canonical = encode({"a": 1})
    padded = canonical.replace(b'[["a",', b'[[ "a",', 1)
    with pytest.raises(CanonicalEncodingError, match="not in canonical form"):
        decode(padded)
