"""Contracts for the one canonical primitive byte representation."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from zlc_storage.canonical import (
    CanonicalArrayEvent,
    CanonicalEncodingError,
    CanonicalListEvent,
    canonical_text,
    canonical_digest,
    decode,
    encode,
    exact_mapping,
    finite_real,
    integer,
    normalized_text,
    positive_integer,
    sha256_text,
)


def test_canonical_scalar_validators_are_the_single_primitive_boundary():
    digest = "ab" * 32

    assert canonical_text("owner-id", "owner") == "owner-id"
    assert canonical_text("", "optional label", empty=True) == ""
    assert sha256_text(digest, "digest") == digest
    assert sha256_text(None, "digest", optional=True) is None
    assert integer(np.int64(3), "count") == 3
    assert positive_integer(2, "count") == 2
    assert finite_real(np.float64(-0.0), "value") == 0.0
    assert normalized_text("  user stop  ", "reason") == "user stop"

    with pytest.raises(ValueError):
        canonical_text(" padded ", "owner")
    with pytest.raises(ValueError):
        sha256_text(digest.upper(), "digest")
    with pytest.raises(TypeError):
        integer(True, "count")
    with pytest.raises(ValueError):
        positive_integer(0, "count")
    with pytest.raises(ValueError):
        finite_real(float("nan"), "value")
    with pytest.raises(ValueError):
        normalized_text("  ", "reason")


def test_exact_mapping_owns_field_and_discriminator_admission():
    value = {"schema": "owner-format", "payload": 3}
    assert exact_mapping(value, {"schema", "payload"}, "owner-format") is value

    with pytest.raises(ValueError, match="exactly"):
        exact_mapping({**value, "extra": 4}, {"schema", "payload"}, "owner-format")
    with pytest.raises(ValueError, match="expected schema"):
        exact_mapping(value, {"schema", "payload"}, "other-format")

    nested = {"name": "center", "fixed": None}
    assert exact_mapping(
        nested,
        {"name", "fixed"},
        "nested constraint",
        discriminator=None,
    ) is nested
    with pytest.raises(ValueError, match="exactly"):
        exact_mapping(
            {**nested, "extra": 1},
            {"name", "fixed"},
            "nested constraint",
            discriminator=None,
        )


def test_mapping_order_and_sequence_container_do_not_change_bytes():
    left = {"unicode": "原子", "items": (1, True, None), "z": -0.0}
    right = {"z": -0.0, "items": [1, True, None], "unicode": "原子"}

    assert encode(left) == encode(right)
    assert canonical_digest(left) == canonical_digest(right)


def test_mixed_primitive_tree_matches_frozen_canonical_bytes():
    value = {
        "items": [
            None,
            True,
            -7,
            -0.0,
            float("inf"),
            float("-inf"),
            float("nan"),
            "原子",
        ],
        "bytes": b"\x00\xff",
        "array": np.array([[float("nan"), 2.0]], dtype=">f8"),
    }
    expected = (
        b"ZLC-CANONICAL-1\n"
        b'['
        b'"map",[["array",["ndarray",["<f8",["1","2"],'
        b'"AAAAAAAA+H8AAAAAAAAAQA=="]]],["bytes",["bytes","AP8="]],'
        b'["items",["list",[["null"],["bool",true],["int","-7"],'
        b'["float64","8000000000000000"],["float64","+inf"],'
        b'["float64","-inf"],["float64","nan"],'
        b'["str","\xe5\x8e\x9f\xe5\xad\x90"]]]]]'
        b']'
    )

    assert encode(value) == expected
    assert canonical_digest(value) == (
        "9128f6c452f37d70295eae6f9de5d9beb1842a1e6958c3ce43ef35e36871c66e"
    )


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


def test_ndarray_nan_payloads_have_one_canonical_spelling():
    ordinary = np.array([float("nan"), 2.0], dtype=np.float64)
    alternate = np.array([0x7FF8000000000001, 0x4000000000000000], dtype=np.uint64).view(np.float64)
    assert encode(ordinary) == encode(alternate)

    complex_ordinary = np.array([complex(float("nan"), float("nan"))], dtype=np.complex128)
    complex_alternate = np.array(
        [0x7FF8000000000001, 0x7FF8000000000002], dtype=np.uint64
    ).view(np.complex128)
    assert encode(complex_ordinary) == encode(complex_alternate)


@pytest.mark.parametrize(
    ("complex_dtype", "integer_dtype", "payloads"),
    [
        ("<c8", "<u4", [0x7FC00001, 0x7FC00002]),
        ("<c16", "<u8", [0x7FF8000000000001, 0x7FF8000000000002]),
    ],
)
def test_zero_dimensional_complex_nan_round_trips_canonically(
    complex_dtype,
    integer_dtype,
    payloads,
):
    ordinary = np.array(complex(float("nan"), float("nan")), dtype=complex_dtype)
    alternate = np.array(payloads, dtype=integer_dtype).view(complex_dtype).reshape(())

    assert encode(ordinary) == encode(alternate)
    restored = decode(encode(alternate))
    assert restored.shape == ()
    assert restored.dtype == np.dtype(complex_dtype)
    assert np.isnan(restored.real)
    assert np.isnan(restored.imag)


def test_floating_scalar_wider_than_float64_is_not_silently_narrowed():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("this platform's longdouble is float64")
    with pytest.raises(CanonicalEncodingError, match="wider than float64"):
        encode(np.longdouble("1.0000000000000000001"))


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


def test_structure_admission_receives_complete_named_list_and_array_inventory():
    payload = encode(
        {
            "models": [
                {"kernel": np.ones((2, 3), dtype="<f8")},
                {"kernel": np.ones((1, 4), dtype="<f4")},
            ]
        }
    )
    observed = None

    def admit(events):
        nonlocal observed
        observed = events

    restored = decode(payload, admit_structure=admit)
    assert len(restored["models"]) == 2
    assert observed == (
        CanonicalListEvent(("models",), 2),
        CanonicalArrayEvent(("models", 0, "kernel"), (2, 3), "<f8", 48),
        CanonicalArrayEvent(("models", 1, "kernel"), (1, 4), "<f4", 16),
    )


def test_structure_admission_rejects_before_any_ndarray_materialization(monkeypatch):
    payload = encode(
        {
            "sites": [
                np.arange(4, dtype="<u2"),
                np.arange(8, dtype="<u2"),
            ]
        }
    )
    materializations = 0

    import zlc_storage.canonical as canonical

    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    monkeypatch.setattr(canonical, "_decode_array", counted)

    def reject(events):
        sites = next(
            event
            for event in events
            if isinstance(event, CanonicalListEvent) and event.path == ("sites",)
        )
        if sites.length > 1:
            raise RuntimeError("site cardinality rejected")

    with pytest.raises(RuntimeError, match="site cardinality"):
        decode(payload, admit_structure=reject)
    assert materializations == 0


def test_structure_paths_do_not_confuse_dotted_keys_with_nested_maps():
    observed = []

    def admit(events):
        observed.extend(event.path for event in events if isinstance(event, CanonicalArrayEvent))

    decode(
        encode(
            {
                "a.b": np.ones(1, dtype="<u1"),
                "a": {"b": np.ones(1, dtype="<u1")},
            }
        ),
        admit_structure=admit,
    )
    assert observed == [("a", "b"), ("a.b",)]


def test_noncanonical_dtype_spelling_rejects_before_materialization(monkeypatch):
    payload = encode(np.arange(2, dtype="<i4")).replace(b'"<i4"', b'"int32"')
    materializations = 0

    import zlc_storage.canonical as canonical

    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    monkeypatch.setattr(canonical, "_decode_array", counted)
    with pytest.raises(CanonicalEncodingError, match="dtype"):
        decode(payload)
    assert materializations == 0


def test_zero_itemsize_ndarray_is_outside_canonical_model():
    with pytest.raises(CanonicalEncodingError, match="zero-itemsize"):
        encode(np.empty(1, dtype="V0"))


def test_unsupported_ndarray_rank_rejects_before_callback_or_materialization(monkeypatch):
    import zlc_storage.canonical as canonical

    tagged = ["ndarray", ["|u1", ["0"] * (canonical._NUMPY_MAX_NDIM + 1), ""]]
    payload = b"ZLC-CANONICAL-1\n" + json.dumps(
        tagged,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    callback_calls = 0
    materializations = 0
    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    def admit(_events):
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(canonical, "_decode_array", counted)
    with pytest.raises(CanonicalEncodingError, match="rank"):
        decode(payload, admit_structure=admit)
    assert callback_calls == 0
    assert materializations == 0


@pytest.mark.parametrize(
    ("dtype_text", "shape_text"),
    [
        ("|u1", [str(2**40), str(2**40), "0"]),
        ("|V8", [str(2**60), "0"]),
    ],
)
def test_zero_sized_shape_with_stride_overflow_rejects_before_callback(
    monkeypatch,
    dtype_text,
    shape_text,
):
    import zlc_storage.canonical as canonical

    tagged = ["ndarray", [dtype_text, shape_text, ""]]
    payload = b"ZLC-CANONICAL-1\n" + json.dumps(
        tagged,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    callback_calls = 0
    materializations = 0
    original = canonical._decode_array

    def counted(payload, *, path):
        nonlocal materializations
        materializations += 1
        return original(payload, path=path)

    def admit(_events):
        nonlocal callback_calls
        callback_calls += 1

    monkeypatch.setattr(canonical, "_decode_array", counted)
    with pytest.raises(CanonicalEncodingError, match="logical extent"):
        decode(payload, admit_structure=admit)
    assert callback_calls == 0
    assert materializations == 0
