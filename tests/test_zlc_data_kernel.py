"""Core invariants for zlc_data's named multidimensional values."""

from __future__ import annotations

import sys
import subprocess
import tracemalloc
from fractions import Fraction

import numpy as np
import pytest

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisLayout,
    AxisSpec,
    BlockId,
    CellValidity,
    ComponentValidity,
    DataBlock,
    DataPatch,
    DatasetRevision,
    DatasetSchema,
    INVALID,
    PointLayout,
    StreamGenerationId,
    VALID,
    ValidityContract,
    Value,
    ValuePayloadContract,
    ValueSchema,
    canonical_value_array,
    dataset_schema_from_tree,
    dataset_schema_to_tree,
    expand_value_validity,
    expand_component_validity,
)


def test_data_patch_is_atomic_immutable_and_revision_linear():
    schema = image_schema()
    patch = DataPatch(
        block_id=BlockId("capture"),
        base_revision=DatasetRevision(2),
        result_revision=DatasetRevision(3),
        target_cells=((0, 1), (0, 2)),
        values=np.arange(24, dtype=np.uint16).reshape(2, 3, 4),
        validity_patch=(VALID, INVALID),
        schema_fingerprint=schema.fingerprint,
    )
    assert patch.target_cells == ((0, 1), (0, 2))
    assert not patch.values.flags.writeable
    with pytest.raises(ValueError, match="immediately follow"):
        DataPatch(
            block_id=patch.block_id,
            base_revision=DatasetRevision(2),
            result_revision=DatasetRevision(4),
            target_cells=((0, 1),),
            values=patch.values[:1],
            validity_patch=(VALID,),
            schema_fingerprint=schema.fingerprint,
        )
    with pytest.raises(ValueError, match="unique"):
        DataPatch(
            block_id=patch.block_id,
            base_revision=DatasetRevision(2),
            result_revision=DatasetRevision(3),
            target_cells=((0, 1), (0, 1)),
            values=patch.values,
            validity_patch=(VALID, VALID),
            schema_fingerprint=schema.fingerprint,
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
    source = np.arange(np.prod(schema.physical_shape), dtype=np.uint16).reshape(
        schema.physical_shape
    )[..., ::-1]
    assert not source.flags.c_contiguous
    expected = source.copy()
    block = DataBlock(
        BlockId("capture-immutable"),
        DatasetRevision(7),
        source,
        CellValidity(np.ones((2, 3), dtype=bool)),
        schema,
    )
    np.testing.assert_array_equal(block.values, expected)
    before = block.values.copy()
    source[...] = 0
    np.testing.assert_array_equal(block.values, before)
    with pytest.raises(ValueError):
        block.values.setflags(write=True)
    with pytest.raises(ValueError):
        block.validity.mask.setflags(write=True)


def test_dataset_ref_carries_complete_identity():
    schema = dataset_schema(explicit=True)
    assert schema.cell_layout is schema.cell_layout
    assert schema.cell_layout.factors is not None
    assert schema.cell_layout.factors[-1] is schema.point_layout
    restored = dataset_schema_from_tree(dataset_schema_to_tree(schema))
    assert restored.cell_layout == schema.cell_layout
    assert restored.cell_layout.factors is not None
    assert restored.cell_layout.factors[-1] is restored.point_layout
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


def test_point_layout_specialization_has_generic_structural_identity():
    point = PointLayout.explicit((4,), ((3,), (1,)))
    generic = AxisLayout.explicit((4,), ((3,), (1,)))

    assert point == generic
    assert generic == point
    assert hash(point) == hash(generic)


def test_dataset_schema_tree_matches_the_independent_current_grammar():
    schema = dataset_schema()
    literal = {
        "schema": "zlc_data.DatasetSchema",
        "repeat_axis": {
            "schema": "zlc_data.AxisSpec",
            "axis_id": "capture.repeat",
            "name": "capture.repeat",
            "role": "repeat",
            "size": 2,
            "coordinates": [0, 1],
            "unit": None,
            "coordinate_frame": None,
            "index_origin": 0,
        },
        "point_axes": [
            {
                "schema": "zlc_data.AxisSpec",
                "axis_id": "scan.detuning",
                "name": "scan.detuning",
                "role": "scan-point",
                "size": 3,
                "coordinates": [0, 1, 2],
                "unit": None,
                "coordinate_frame": None,
                "index_origin": 0,
            }
        ],
        "point_layout": {
            "logical_shape": [3],
            "mode": "RECT_C",
            "storage_size": 3,
            "storage_to_multi": None,
        },
        "cell_schema": {
            "schema": "zlc_data.ValueSchema",
            "data_axes": [
                {
                    "schema": "zlc_data.AxisSpec",
                    "axis_id": "camera.image.y",
                    "name": "camera.image.y",
                    "role": "spatial-y",
                    "size": 3,
                    "coordinates": [0, 1, 2],
                    "unit": None,
                    "coordinate_frame": None,
                    "index_origin": 0,
                },
                {
                    "schema": "zlc_data.AxisSpec",
                    "axis_id": "camera.image.x",
                    "name": "camera.image.x",
                    "role": "spatial-x",
                    "size": 4,
                    "coordinates": [0, 1, 2, 3],
                    "unit": None,
                    "coordinate_frame": None,
                    "index_origin": 0,
                },
            ],
            "validity_contract": {"mode": "VALUE", "component_axis_ids": []},
            "dtype": "<u2",
            "value_unit": "count",
        },
    }

    assert dataset_schema_to_tree(schema) == literal
    restored = dataset_schema_from_tree(literal)
    assert restored == schema
    assert restored.fingerprint == schema.fingerprint

    malformed = dict(literal, revision=1)
    with pytest.raises(ValueError, match="exactly"):
        dataset_schema_from_tree(malformed)


def test_schema_fingerprint_covers_layout_and_component_validity():
    sparse = dataset_schema(explicit=True, component_validity=True)
    dense = dataset_schema(explicit=False, component_validity=True)
    value_only = dataset_schema(explicit=True, component_validity=False)
    assert dense.fingerprint != sparse.fingerprint
    assert value_only.fingerprint != sparse.fingerprint


def test_value_payload_digest_binds_valid_content_and_normalizes_invalid_fillers():
    schema = image_schema(component_validity=True)
    x = schema.data_axes[1]
    validity = ComponentValidity(
        (x.axis_id,),
        np.array([True, False, True, False]),
    )
    left_values = np.arange(12, dtype=np.uint16).reshape(3, 4)
    right_values = np.array(left_values, copy=True)
    right_values[:, 1] = 500
    right_values[:, 3] = 700
    contract = ValuePayloadContract(schema)
    canonical_valid = canonical_value_array(left_values, VALID, schema)
    assert canonical_valid is not None
    assert np.shares_memory(canonical_valid, left_values)
    left = Value(left_values, validity, schema)
    right = Value(right_values, validity, schema)

    assert contract.digest(left) == contract.digest(right)
    assert contract.digest(left) == contract.digest_content(left.values, validity)
    canonical_mask = expand_component_validity(validity, schema)
    canonical_validity = ComponentValidity(
        schema.validity_contract.component_axis_ids,
        canonical_mask,
    )
    assert contract.digest(left) == contract.digest_content(
        left_values,
        canonical_validity,
    )

    changed_valid = np.array(right_values, copy=True)
    changed_valid[0, 0] += 1
    assert (
        contract.digest_content(changed_valid, validity)
        != contract.digest_content(left_values, validity)
    )

    value_schema = image_schema(component_validity=False)
    value_contract = ValuePayloadContract(value_schema)
    assert canonical_value_array(left_values, INVALID, value_schema) is None
    assert value_contract.digest_content(left_values, INVALID) == value_contract.digest_content(
        right_values,
        INVALID,
    )

    all_valid = np.ones(schema.data_shape, dtype=bool)
    all_invalid = np.zeros(schema.data_shape, dtype=bool)
    assert contract.digest_content(left_values, VALID) == contract.digest_content(
        left_values,
        ComponentValidity(schema.validity_contract.component_axis_ids, all_valid),
    )
    assert contract.digest_content(left_values, INVALID) == contract.digest_content(
        right_values,
        ComponentValidity(schema.validity_contract.component_axis_ids, all_invalid),
    )


def test_component_canonicalization_does_not_allocate_an_inverse_frame_mask():
    y = axis("large.camera.y", SPATIAL_Y, 512)
    x = axis("large.camera.x", SPATIAL_X, 512)
    schema = ValueSchema(
        (y, x),
        ValidityContract.components(y.axis_id, x.axis_id),
        np.dtype(np.uint16),
        "count",
    )
    values = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    mask = np.ones(schema.data_shape, dtype=bool)
    mask[0, 0] = False
    validity = ComponentValidity((y.axis_id, x.axis_id), mask)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    tracemalloc.clear_traces()
    canonical = canonical_value_array(values, validity, schema)
    _current, peak = tracemalloc.get_traced_memory()
    if not was_tracing:
        tracemalloc.stop()

    assert canonical is not None
    assert canonical[0, 0] == 0
    np.testing.assert_array_equal(canonical[1:], values[1:])
    # One canonical value frame is necessary.  A dense ``~validity`` mask
    # would add 256 KiB and make this independent peak check fail.
    assert peak <= values.nbytes + (128 << 10)


def test_schema_fingerprint_normalizes_dtype_endianness():
    little = ValueSchema((), ValidityContract.value(), np.dtype("<i2"))
    big = ValueSchema((), ValidityContract.value(), np.dtype(">i2"))
    assert little.fingerprint == big.fingerprint


def test_immutable_schema_fingerprints_are_computed_once(monkeypatch):
    schema = dataset_schema()
    dataset_fingerprint = schema.fingerprint
    value_fingerprint = schema.cell_schema.fingerprint
    import zlc_data.codec as codec

    def forbidden(*_args, **_kwargs):
        raise AssertionError("immutable schema fingerprint was recomputed")

    monkeypatch.setattr(codec, "dataset_schema_fingerprint", forbidden)
    monkeypatch.setattr(codec, "value_schema_fingerprint", forbidden)
    assert schema.fingerprint == dataset_fingerprint
    assert schema.cell_schema.fingerprint == value_fingerprint


def test_value_schema_rejects_non_numeric_payload_dtypes():
    with pytest.raises(TypeError, match="numeric"):
        ValueSchema((), ValidityContract.value(), np.dtype("U4"))


def test_axis_coordinates_reject_nonfinite_values():
    with pytest.raises(ValueError, match="finite"):
        AxisSpec(AxisId("bad"), "bad", SCAN_POINT, 1, (float("nan"),))


def test_numeric_coordinates_have_one_python_and_fingerprint_identity():
    negative_zero = AxisSpec(
        AxisId("scan.coordinate"),
        "coordinate",
        SCAN_POINT,
        2,
        (-0.0, np.float64(1.0)),
    )
    integers = AxisSpec(
        AxisId("scan.coordinate"),
        "coordinate",
        SCAN_POINT,
        2,
        (0, 1),
    )
    assert negative_zero == integers
    assert negative_zero.coordinates == (0, 1)
    assert all(type(value) is int for value in negative_zero.coordinates)

    repeat = axis("repeat", REPEAT, 1)
    left = DatasetSchema(repeat, (negative_zero,), PointLayout.rect_c((2,)), image_schema())
    right = DatasetSchema(repeat, (integers,), PointLayout.rect_c((2,)), image_schema())
    assert left == right
    assert left.fingerprint == right.fingerprint

    with pytest.raises(TypeError, match="boolean"):
        AxisSpec(AxisId("bool"), "bool", SCAN_POINT, 1, (True,))
    with pytest.raises(TypeError, match="scalar"):
        AxisSpec(AxisId("fraction"), "fraction", SCAN_POINT, 1, (Fraction(1, 2),))


def test_layout_constructor_normalizes_equal_physical_mappings():
    assert PointLayout.rect_f((3,)) == PointLayout.rect_c((3,))
    assert PointLayout.rect_f((1, 3, 1)) == PointLayout.rect_c((1, 3, 1))

    c_mapping = tuple(np.ndindex(2, 3))
    explicit_c = PointLayout.explicit((2, 3), c_mapping)
    assert explicit_c == PointLayout.rect_c((2, 3))

    f_mapping = tuple(
        tuple(int(index) for index in np.unravel_index(row, (2, 3), order="F"))
        for row in range(6)
    )
    explicit_f = PointLayout.explicit((2, 3), f_mapping)
    assert explicit_f == PointLayout.rect_f((2, 3))

    repeat = axis("repeat", REPEAT, 1)
    points = (axis("row", SCAN_POINT, 2), axis("column", SCAN_POINT, 3))
    left = DatasetSchema(repeat, points, explicit_c, image_schema())
    right = DatasetSchema(repeat, points, PointLayout.rect_c((2, 3)), image_schema())
    assert left.fingerprint == right.fingerprint

    with_singleton = AxisLayout.product(
        AxisLayout.rect_c((1,)),
        AxisLayout.rect_f((2, 3)),
    )
    assert with_singleton == AxisLayout.rect_f((1, 2, 3))

    empty_product = AxisLayout.product(
        AxisLayout.explicit((4,), ()),
        AxisLayout.rect_c((3,)),
    )
    assert empty_product == AxisLayout.explicit((4, 3), ())


def test_repeat_role_has_exactly_one_structural_owner():
    repeat = axis("repeat", REPEAT, 1)
    counterfeit_point = axis("counterfeit.point", REPEAT, 2)
    with pytest.raises(ValueError, match="only"):
        DatasetSchema(
            repeat,
            (counterfeit_point,),
            PointLayout.rect_c((2,)),
            image_schema(),
        )

    counterfeit_data = axis("counterfeit.data", REPEAT, 2)
    with pytest.raises(ValueError, match="only"):
        DatasetSchema(
            repeat,
            (),
            PointLayout.rect_c(()),
            ValueSchema(
                (counterfeit_data,),
                ValidityContract.value(),
                np.dtype(np.float64),
            ),
        )


def test_invalid_digest_still_validates_shape_and_dtype():
    schema = image_schema(component_validity=False)
    contract = ValuePayloadContract(schema)
    with pytest.raises(ValueError, match="shape"):
        contract.digest_content(np.zeros((1,), dtype=np.uint16), INVALID)
    with pytest.raises(TypeError, match="dtype"):
        contract.digest_content(np.zeros(schema.data_shape, dtype=np.float32), INVALID)


def test_value_canonicalization_accepts_endian_equivalent_input():
    schema = ValueSchema((), ValidityContract.value(), np.dtype("<i2"))
    source = np.array(513, dtype=">i2")
    canonical = canonical_value_array(source, VALID, schema)
    assert canonical is not None
    assert canonical.dtype == np.dtype("<i2")
    assert canonical.item() == 513


def test_big_endian_complex_nan_payloads_have_one_content_identity():
    sample = axis("sample", SITE, 1)
    schema = ValueSchema((sample,), ValidityContract.value(), np.dtype("<c8"))
    first = np.zeros(1, dtype=">c8")
    second = np.zeros(1, dtype=">c8")
    first.view(">u4")[:] = (0x7FC00001, 0x3F800000)
    second.view(">u4")[:] = (0x7FA12345, 0x3F800000)

    first_canonical = canonical_value_array(first, VALID, schema)
    second_canonical = canonical_value_array(second, VALID, schema)
    assert first_canonical is not None and second_canonical is not None
    assert first_canonical.dtype == np.dtype("<c8")
    assert first_canonical.tobytes() == second_canonical.tobytes()
    contract = ValuePayloadContract(schema)
    assert contract.digest_content(first, VALID) == contract.digest_content(second, VALID)


def test_retained_byte_bound_uses_unbounded_integer_arithmetic():
    height = AxisSpec(AxisId("huge.y"), "y", SPATIAL_Y, 2**32)
    width = AxisSpec(AxisId("huge.x"), "x", SPATIAL_X, 2**32 + 1)
    schema = ValueSchema(
        (height, width),
        ValidityContract.components(height.axis_id, width.axis_id),
        np.dtype(np.uint16),
    )
    expected_elements = (2**32) * (2**32 + 1)
    assert ValuePayloadContract(schema).max_retained_nbytes == expected_elements * 3


def test_import_is_headless_and_does_not_pull_legacy_domain():
    code = """
import sys
import zlc_data
for forbidden in ('matplotlib', 'PyQt5', 'Zou_lab_control'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
