"""Core invariants for zlc_data's named multidimensional values."""

from __future__ import annotations

import sys
import subprocess

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    PointLayout,
    StreamGenerationId,
    TypedCodecError,
    VALID,
    ValidityContract,
    Value,
    ValueSchema,
    decode_dataset_schema,
    decode_data_block,
    decode_value,
    encode_data_block,
    encode_dataset_schema,
    encode_value,
    expand_value_validity,
)


def axis(name: str, role, size: int) -> AxisSpec:
    return AxisSpec(AxisId(name), name, role, size, tuple(range(size)))


def image_schema(*, component_validity: bool = False) -> ValueSchema:
    y = axis("camera.image.y", SPATIAL_Y, 3)
    x = axis("camera.image.x", SPATIAL_X, 4)
    contract = ValidityContract.components(y.axis_id, x.axis_id) if component_validity else ValidityContract.value()
    return ValueSchema((y, x), contract, np.dtype(np.uint16), value_unit="count")


def dataset_schema(*, explicit: bool = False, component_validity: bool = False) -> DatasetSchema:
    repeat = axis("capture.repeat", REPEAT, 2)
    detuning = axis("scan.detuning", SCAN_POINT, 3)
    layout = (
        PointLayout.explicit((3,), ((2,), (0,)))
        if explicit
        else PointLayout.rect_c((3,))
    )
    return DatasetSchema(repeat, (detuning,), layout, image_schema(component_validity=component_validity))


def test_scalar_is_rank_zero_not_length_one_axis():
    scalar_schema = ValueSchema((), ValidityContract.value(), np.dtype(np.float64), "count")
    value = Value(np.array(2.5), VALID, scalar_schema)

    assert value.values.shape == ()
    with pytest.raises(ValueError, match="shape"):
        Value(np.array([2.5]), VALID, scalar_schema)


@pytest.mark.parametrize("layout", [PointLayout.rect_c((2, 3)), PointLayout.rect_f((2, 3))])
def test_rectangular_point_layout_round_trips_every_index(layout):
    for storage_index in range(layout.storage_size):
        multi = layout.multi_index(storage_index)
        assert layout.storage_index(multi) == storage_index


def test_explicit_layout_preserves_sparse_order_and_rejects_duplicates():
    layout = PointLayout.explicit((4, 5), ((3, 1), (0, 4), (2, 2)))
    assert [layout.multi_index(index) for index in range(3)] == [(3, 1), (0, 4), (2, 2)]
    assert layout.storage_index((0, 4)) == 1
    with pytest.raises(KeyError):
        layout.storage_index((1, 1))
    with pytest.raises(ValueError, match="duplicate"):
        PointLayout.explicit((2,), ((0,), (0,)))


def test_explicit_layout_reverse_lookup_is_preindexed():
    mapping = tuple((index,) for index in range(5000))
    layout = PointLayout.explicit((5000,), mapping)
    assert all(layout.storage_index((index,)) == index for index in range(5000))


def test_no_point_axes_have_one_storage_cell():
    layout = PointLayout.rect_c(())
    assert layout.storage_size == 1
    assert layout.multi_index(0) == ()
    assert layout.storage_index(()) == 0


def test_dataset_rejects_duplicate_axis_identity_across_axis_families():
    repeat = axis("same", REPEAT, 1)
    point = axis("same", SCAN_POINT, 1)
    with pytest.raises(ValueError, match="unique"):
        DatasetSchema(repeat, (point,), PointLayout.rect_c((1,)), image_schema())


def test_component_validity_is_named_and_cannot_broadcast_by_trailing_position():
    schema = image_schema(component_validity=True)
    y, x = schema.data_axes
    valid_x = ComponentValidity((x.axis_id,), np.ones((4,), dtype=bool))
    Value(np.zeros((3, 4), dtype=np.uint16), valid_x, schema)

    wrong_order = ComponentValidity((x.axis_id, y.axis_id), np.ones((4, 3), dtype=bool))
    with pytest.raises(ValueError, match="order"):
        Value(np.zeros((3, 4), dtype=np.uint16), wrong_order, schema)

    wrong_shape = ComponentValidity((x.axis_id,), np.ones((3,), dtype=bool))
    with pytest.raises(ValueError, match="shape"):
        Value(np.zeros((3, 4), dtype=np.uint16), wrong_shape, schema)


def test_equal_sized_component_axes_broadcast_by_identity_not_shape():
    site = axis("readout.site", SITE, 2)
    component = axis("readout.component", SPATIAL_X, 2)
    schema = ValueSchema(
        (site, component),
        ValidityContract.components(site.axis_id, component.axis_id),
        np.dtype(np.float64),
    )
    mask = np.array([True, False])
    site_valid = expand_value_validity(ComponentValidity((site.axis_id,), mask), schema)
    component_valid = expand_value_validity(
        ComponentValidity((component.axis_id,), mask), schema
    )
    np.testing.assert_array_equal(site_valid, [[True, True], [False, False]])
    np.testing.assert_array_equal(component_valid, [[True, False], [True, False]])


def test_dataset_component_validity_includes_repeat_and_physical_point_axes():
    schema = dataset_schema(component_validity=True)
    site_like_x = schema.cell_schema.data_axes[1]
    validity = ComponentValidity(
        (site_like_x.axis_id,),
        np.ones((2, 3, 4), dtype=bool),
    )
    block = DataBlock(
        BlockId("capture-1"),
        DatasetRevision(0),
        np.zeros(schema.physical_shape, dtype=np.uint16),
        validity,
        schema,
    )
    assert block.validity.mask.shape == (2, 3, 4)


def test_datablock_owns_intrinsically_immutable_bytes():
    schema = dataset_schema()
    source = np.arange(np.prod(schema.physical_shape), dtype=np.uint16).reshape(schema.physical_shape)
    block = DataBlock(
        BlockId("capture-immutable"),
        DatasetRevision(7),
        source,
        CellValidity(np.ones((2, 3), dtype=bool)),
        schema,
    )
    before = block.values.copy()
    source[...] = 0
    np.testing.assert_array_equal(block.values, before)
    with pytest.raises(ValueError):
        block.values.setflags(write=True)
    with pytest.raises(ValueError):
        block.validity.mask.setflags(write=True)


def test_dataset_ref_carries_complete_identity():
    schema = dataset_schema(explicit=True)
    block = DataBlock(
        BlockId("sparse-capture"),
        DatasetRevision(3),
        np.zeros(schema.physical_shape, dtype=np.uint16),
        VALID,
        schema,
    )
    ref = block.ref(StreamGenerationId("camera-generation-8"))
    assert ref.block_id == block.block_id
    assert ref.revision == block.revision
    assert ref.schema_fingerprint == schema.fingerprint


def test_schema_codec_round_trip_and_fingerprint_cover_layout_and_validity():
    schema = dataset_schema(explicit=True, component_validity=True)
    restored = decode_dataset_schema(encode_dataset_schema(schema))
    assert restored == schema
    assert restored.fingerprint == schema.fingerprint

    dense = dataset_schema(explicit=False, component_validity=True)
    assert dense.fingerprint != schema.fingerprint
    value_only = dataset_schema(explicit=True, component_validity=False)
    assert value_only.fingerprint != schema.fingerprint


def test_value_and_datablock_codecs_are_strict_owned_round_trips():
    value_schema = image_schema(component_validity=True)
    x = value_schema.data_axes[1]
    value = Value(
        np.arange(12, dtype=np.uint16).reshape(3, 4),
        ComponentValidity((x.axis_id,), np.array([True, False, True, False])),
        value_schema,
    )
    restored_value = decode_value(encode_value(value))
    assert restored_value.schema == value.schema
    np.testing.assert_array_equal(restored_value.validity.mask, value.validity.mask)
    assert not restored_value.values.flags.writeable

    schema = dataset_schema()
    block = DataBlock(
        BlockId("codec-roundtrip"),
        DatasetRevision(4),
        np.arange(np.prod(schema.physical_shape), dtype=np.uint16).reshape(schema.physical_shape),
        CellValidity(np.ones((2, 3), dtype=bool)),
        schema,
    )
    restored_block = decode_data_block(encode_data_block(block))
    assert restored_block.block_id == block.block_id
    assert restored_block.revision == block.revision
    assert restored_block.schema == block.schema
    np.testing.assert_array_equal(restored_block.values, block.values)


def test_invalid_storage_bytes_do_not_change_logical_content_encoding():
    schema = DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("point", SCAN_POINT, 2),),
        PointLayout.rect_c((2,)),
        ValueSchema((), ValidityContract.value(), np.dtype(np.int16)),
    )
    validity = CellValidity(np.array([[True, False]]))
    left = DataBlock(
        BlockId("same"), DatasetRevision(1), np.array([[7, 123]], dtype=np.int16), validity, schema
    )
    right = DataBlock(
        BlockId("same"), DatasetRevision(1), np.array([[7, 999]], dtype=np.int16), validity, schema
    )
    assert encode_data_block(left) == encode_data_block(right)
    restored = decode_data_block(encode_data_block(left))
    np.testing.assert_array_equal(restored.values, [[7, 0]])


def test_schema_fingerprint_normalizes_dtype_endianness():
    little = ValueSchema((), ValidityContract.value(), np.dtype("<i2"))
    big = ValueSchema((), ValidityContract.value(), np.dtype(">i2"))
    assert little.fingerprint == big.fingerprint


def test_value_schema_rejects_non_numeric_payload_dtypes():
    with pytest.raises(TypeError, match="numeric"):
        ValueSchema((), ValidityContract.value(), np.dtype("U4"))


def test_axis_coordinates_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        AxisSpec(AxisId("bad"), "bad", SCAN_POINT, 1, (float("nan"),))


def test_typed_decoders_reject_alternate_primitive_spellings():
    from zlc_storage.canonical import decode, encode

    schema = dataset_schema()
    schema_tree = decode(encode_dataset_schema(schema))
    schema_tree["cell_schema"]["dtype"] = "uint16"
    with pytest.raises(TypedCodecError, match="non-canonical typed"):
        decode_dataset_schema(encode(schema_tree))

    block = DataBlock(
        BlockId("typed-canonical"),
        DatasetRevision(0),
        np.zeros(schema.physical_shape, dtype=np.uint16),
        CellValidity(np.ones((2, 3), dtype=bool)),
        schema,
    )
    block_tree = decode(encode_data_block(block))
    block_tree["validity"]["mask"] = block_tree["validity"]["mask"].tolist()
    with pytest.raises(ValueError, match="ndarray"):
        decode_data_block(encode(block_tree))

    invalid_schema = DatasetSchema(
        axis("repeat", REPEAT, 1),
        (axis("point", SCAN_POINT, 2),),
        PointLayout.rect_c((2,)),
        ValueSchema((), ValidityContract.value(), np.dtype(np.int16)),
    )
    invalid_block = DataBlock(
        BlockId("invalid-bytes"),
        DatasetRevision(0),
        np.array([[4, 99]], dtype=np.int16),
        CellValidity(np.array([[True, False]])),
        invalid_schema,
    )
    noncanonical_tree = decode(encode_data_block(invalid_block))
    noncanonical_tree["values"].setflags(write=True)
    noncanonical_tree["values"][0, 1] = 123
    with pytest.raises(TypedCodecError, match="non-canonical typed"):
        decode_data_block(encode(noncanonical_tree))


def test_import_is_headless_and_does_not_pull_legacy_domain():
    code = """
import sys
import zlc_data
for forbidden in ('matplotlib', 'PyQt5', 'Zou_lab_control'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
