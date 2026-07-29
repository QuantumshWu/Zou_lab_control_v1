"""Core invariants for zlc_data's named multidimensional values."""

from __future__ import annotations

import sys
import subprocess
from fractions import Fraction

import numpy as np
import pytest

from zlc_data._arrays import immutable_array, is_intrinsically_immutable_array
from zlc_data.axis import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSpec,
)
from zlc_data.codec import dataset_schema_from_tree, dataset_schema_to_tree
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.validity import (
    INVALID,
    VALID,
    CellValidity,
    ComponentValidity,
    DatasetComponentValidity,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    StreamGenerationId,
    Value,
    ValuePayloadContract,
    canonical_value_array,
    expand_component_validity,
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
    point_values = (2, 0) if explicit else (0, 1, 2)
    point_table = PointTable(
        len(point_values),
        (
            PointColumn(
                detuning.axis_id,
                detuning.name,
                detuning.role,
                PointColumn.NUMERIC,
                point_values,
            ),
        ),
    )
    topology = GridTopology(
        (detuning.axis_id,),
        ((0, 1, 2),),
        tuple((value,) for value in point_values),
    )
    return DatasetSchema(
        repeat,
        point_table,
        topology,
        image_schema(component_validity=component_validity),
    )


def test_scalar_has_the_canonical_length_one_carrier_axis():
    scalar_schema = ValueSchema.scalar(np.dtype(np.float64), "count")
    value = Value(np.array([2.5]), VALID, scalar_schema)

    assert scalar_schema.is_scalar
    assert value.values.shape == (1,)
    with pytest.raises(ValueError, match="shape"):
        Value(np.array(2.5), VALID, scalar_schema)


def test_intrinsically_immutable_strided_views_cross_value_and_dataset_without_copy():
    mutable = np.arange(12, dtype=np.uint16).reshape(3, 4)
    frozen = immutable_array(
        mutable,
        dtype=np.dtype("<u2"),
        shape=mutable.shape,
    )
    transposed = frozen.T
    y = axis("camera.transposed.y", SPATIAL_Y, 4)
    x = axis("camera.transposed.x", SPATIAL_X, 3)
    value_schema = ValueSchema(
        (y, x),
        ValidityContract.value(),
        np.dtype("<u2"),
        "count",
    )
    value = Value(transposed, VALID, value_schema)
    assert np.shares_memory(value.values, transposed)
    assert value.values.strides == transposed.strides
    assert is_intrinsically_immutable_array(value.values)

    repeat = axis("camera.transposed.repeat", REPEAT, 1)
    schema = DatasetSchema(repeat, PointTable(1), None, value_schema)
    block = DataBlock(
        BlockId("immutable-transpose"),
        DatasetRevision(0),
        value.values.reshape(schema.physical_shape),
        VALID,
        schema,
    )
    assert np.shares_memory(block.values, value.values)
    assert is_intrinsically_immutable_array(block.values)

    mutable[:] = 0
    assert np.any(block.values != 0)
    with pytest.raises(ValueError):
        block.values.setflags(write=True)


def test_dataset_rejects_duplicate_axis_identity_across_axis_families():
    repeat = axis("same", REPEAT, 1)
    point = axis("same", SCAN_POINT, 1)
    point_table = PointTable(
        1,
        (
            PointColumn(
                point.axis_id,
                point.name,
                point.role,
                PointColumn.NUMERIC,
                (0,),
            ),
        ),
    )
    with pytest.raises(ValueError, match="unique"):
        DatasetSchema(repeat, point_table, None, image_schema())


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


def test_dataset_component_validity_includes_repeat_and_physical_point_rows():
    schema = dataset_schema(component_validity=True)
    site_like_x = schema.cell_schema.data_axes[1]
    validity = DatasetComponentValidity(
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
    restored = dataset_schema_from_tree(dataset_schema_to_tree(schema))
    assert restored.point_table == schema.point_table
    assert restored.grid_topology == schema.grid_topology
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
        "point_table": {
            "schema": "zlc_data.PointTable",
            "row_count": 3,
            "columns": [
                {
                    "schema": "zlc_data.PointColumn",
                    "coordinate_id": "scan.detuning",
                    "name": "scan.detuning",
                    "role": "scan-point",
                    "value_kind": "NUMERIC",
                    "values": [0, 1, 2],
                    "unit": None,
                    "coordinate_frame": None,
                }
            ],
        },
        "grid_topology": {
            "schema": "zlc_data.GridTopology",
            "dimension_ids": ["scan.detuning"],
            "coordinate_domains": [[0, 1, 2]],
            "row_to_cell": [[0], [1], [2]],
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


def test_schema_fingerprint_covers_point_rows_topology_and_component_validity():
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


def test_schema_fingerprint_normalizes_dtype_endianness():
    little = ValueSchema.scalar(np.dtype("<i2"))
    big = ValueSchema.scalar(np.dtype(">i2"))
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
        ValueSchema.scalar(np.dtype("U4"))


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
    left_column = PointColumn(
        negative_zero.axis_id,
        negative_zero.name,
        negative_zero.role,
        PointColumn.NUMERIC,
        negative_zero.coordinates,
    )
    right_column = PointColumn(
        integers.axis_id,
        integers.name,
        integers.role,
        PointColumn.NUMERIC,
        integers.coordinates,
    )
    left = DatasetSchema(repeat, PointTable(2, (left_column,)), None, image_schema())
    right = DatasetSchema(repeat, PointTable(2, (right_column,)), None, image_schema())
    assert left == right
    assert left.fingerprint == right.fingerprint

    with pytest.raises(TypeError, match="finite number"):
        AxisSpec(AxisId("bool"), "bool", SCAN_POINT, 1, (True,))
    fractional = AxisSpec(
        AxisId("fraction"), "fraction", SCAN_POINT, 1, (Fraction(1, 2),)
    )
    assert fractional.coordinates == (0.5,)


def test_repeat_role_has_exactly_one_structural_owner():
    repeat = axis("repeat", REPEAT, 1)
    with pytest.raises(ValueError, match="point-domain"):
        PointColumn(
            AxisId("counterfeit.point"),
            "counterfeit.point",
            REPEAT,
            PointColumn.NUMERIC,
            (0, 1),
        )

    counterfeit_data = axis("counterfeit.data", REPEAT, 2)
    with pytest.raises(ValueError, match="only"):
        DatasetSchema(
            repeat,
            PointTable(1),
            None,
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
    schema = ValueSchema.scalar(np.dtype("<i2"))
    source = np.array([513], dtype=">i2")
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


def test_import_is_headless_and_does_not_pull_legacy_domain():
    code = """
import sys
import zlc_data
for forbidden in ('matplotlib', 'PyQt5', 'Zou_lab_control'):
    assert forbidden not in sys.modules, (forbidden, sorted(sys.modules))
"""
    subprocess.run([sys.executable, "-c", code], check=True)
