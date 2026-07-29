"""Adversarial contracts for named authoritative transforms."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data.axis import (
    REPEAT,
    SCALAR_AXIS,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    AxisId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
)
from zlc_data.numeric import (
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
)
from zlc_data.schema import (
    DatasetSchema,
    GridTopology,
    PointColumn,
    PointTable,
    ValueSchema,
)
from zlc_data.selection import (
    IndexRangeSelection,
    Selection,
    selection_from_tree,
    selection_to_tree,
)
from zlc_data.transform import (
    DataTransformSpec,
    MissingPolicy,
    ReductionMethod,
    ReductionSpec,
    TransformedData,
    ValidityPolicy,
    apply_transform,
    commit_transform,
    resolve_transformed_schema,
)
from zlc_data.transform_codec import (
    committed_transform_from_tree,
    committed_transform_to_tree,
)
from zlc_data.validity import (
    VALID,
    CellValidity,
    DatasetComponentValidity,
    ValidityContract,
)
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)


def axis(
    name: str,
    role,
    size: int,
    *,
    coordinates=None,
    frame: CoordinateFrameId | None = None,
) -> AxisSpec:
    if coordinates is None:
        coordinates = tuple(range(size))
    return AxisSpec(AxisId(name), name, role, size, tuple(coordinates), None, frame)


def point_table(
    name: str,
    role,
    values,
    *,
    unit: str | None = None,
    frame: CoordinateFrameId | None = None,
) -> PointTable:
    values = tuple(values)
    return PointTable(
        len(values),
        (
            PointColumn(
                AxisId(name),
                name,
                role,
                PointColumn.NUMERIC,
                values,
                unit,
                frame,
            ),
        ),
    )


def block_for(
    *,
    repeat: int,
    points: PointTable,
    data_axes: tuple[AxisSpec, ...] = (),
    grid_topology: GridTopology | None = None,
    values,
    validity=VALID,
    component_axes: tuple[AxisId, ...] = (),
) -> DataBlock:
    contract = (
        ValidityContract.components(*component_axes)
        if component_axes
        else ValidityContract.value()
    )
    array = np.asarray(values)
    cell_schema = (
        ValueSchema(data_axes, contract, array.dtype)
        if data_axes
        else ValueSchema.scalar(array.dtype)
    )
    if not data_axes:
        array = array[..., np.newaxis]
    schema = DatasetSchema(
        axis("repeat", REPEAT, repeat),
        points,
        grid_topology,
        cell_schema,
    )
    return DataBlock(
        BlockId("source"), DatasetRevision(3), array, validity, schema
    )


def apply_spec(
    block: DataBlock,
    spec: DataTransformSpec,
    *,
    point_ordinals: tuple[int, ...] | None = None,
) -> TransformedData:
    """Exercise the committed authority path rather than a preview shortcut."""

    committed = commit_transform(
        block.schema,
        spec,
        point_ordinals=point_ordinals,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("transform-test-generation")), block
    )
    return apply_transform(snapshot, committed)


def test_selection_is_canonical_as_an_axis_constraint_set():
    x = AxisId("x")
    y = AxisId("y")
    left = Selection(
        (
            IndexRangeSelection(y, 1, 3),
            IndexRangeSelection(x, 0, 2),
        )
    )
    right = Selection(
        (
            IndexRangeSelection(x, 0, 2),
            IndexRangeSelection(y, 1, 3),
        )
    )
    assert left == right

    literal = {
        "schema": "zlc_data.Selection",
        "terms": [
            {"kind": "INDEX_RANGE", "axis_id": "x", "start": 0, "stop": 2},
            {"kind": "INDEX_RANGE", "axis_id": "y", "start": 1, "stop": 3},
        ],
    }
    assert selection_to_tree(left) == literal
    assert selection_from_tree(literal) == left

    frame = CoordinateFrameId("camera-pixel")
    integer_bounds = Selection.coordinate_range(x, 1, 2, coordinate_frame=frame)
    float_bounds = Selection.coordinate_range(x, 1.0, 2.0, coordinate_frame=frame)
    assert integer_bounds == float_bounds
    assert selection_to_tree(integer_bounds) == selection_to_tree(float_bounds)


def test_repeated_index_ranges_preserve_absolute_implicit_axis_indices():
    sample = AxisSpec(
        AxisId("sample"), "sample", SPATIAL_X, 8, unit="pixel"
    )
    block = block_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(sample,),
        values=np.arange(sample.size, dtype=np.int16).reshape(1, 1, -1),
    )
    spec = DataTransformSpec(
        (
            Selection.index_range(sample.axis_id, 2, 7),
            Selection.index_range(sample.axis_id, 1, 4),
        )
    )

    result = apply_spec(block, spec)
    selected_axis = result.schema.cell_schema.axis(sample.axis_id)
    assert selected_axis.coordinates is None
    assert selected_axis.index_origin == 3
    assert selected_axis.unit == "pixel"
    assert tuple(selected_axis.coordinate_at(index) for index in range(3)) == (3, 4, 5)
    np.testing.assert_array_equal(result.values, [[[3, 4, 5]]])


def test_exact_point_ordinals_preserve_authored_grid_rows():
    a_id = AxisId("a")
    b_id = AxisId("b")
    row_to_cell = tuple((a, b) for b in range(3) for a in range(2))
    points = PointTable(
        6,
        (
            PointColumn(
                a_id,
                "a",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(cell[0] for cell in row_to_cell),
            ),
            PointColumn(
                b_id,
                "b",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(cell[1] for cell in row_to_cell),
            ),
        ),
    )
    topology = GridTopology((a_id, b_id), ((0, 1), (0, 1, 2)), row_to_cell)
    values = np.arange(12, dtype=np.int16).reshape(2, 6)
    block = block_for(
        repeat=2,
        points=points,
        grid_topology=topology,
        values=values,
    )

    result = apply_spec(
        block,
        DataTransformSpec(),
        point_ordinals=(2, 3),
    )

    assert result.schema.point_table.column(a_id).values == (0, 1)
    assert result.schema.point_table.column(b_id).values == (1, 1)
    assert result.schema.grid_topology is not None
    assert result.schema.grid_topology.row_to_cell == ((0, 1), (1, 1))
    np.testing.assert_array_equal(result.values, values[:, 2:4, np.newaxis])


def test_data_axis_selection_preserves_every_other_data_axis():
    site = axis("site", SITE, 3)
    component = axis("component", SPATIAL_X, 2)
    block = block_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(site, component),
        values=np.arange(6, dtype=np.int16).reshape(1, 1, 3, 2),
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(site.axis_id, 1, 3),)),
    )
    assert tuple(axis.axis_id for axis in result.schema.cell_schema.data_axes) == (
        site.axis_id,
        component.axis_id,
    )
    assert result.schema.cell_schema.data_shape == (2, 2)
    np.testing.assert_array_equal(result.values[0, 0], [[2, 3], [4, 5]])


def test_transform_keeps_repeat_points_and_trailing_data_axes_distinct():
    points = point_table("point", SCAN_POINT, range(3))
    site = axis("site", SITE, 2)
    component = axis("component", SPATIAL_X, 2)
    source = np.arange(2 * 3 * 2 * 2, dtype=np.int16).reshape(2, 3, 2, 2)
    block = block_for(
        repeat=2,
        points=points,
        data_axes=(site, component),
        values=source,
    )

    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(site.axis_id, 1, 2),)),
    )

    assert block.values.shape == (2, 3, 2, 2)
    assert result.schema.repeat_axis.size == 2
    assert result.schema.point_table == points
    assert result.schema.cell_schema.data_shape == (1, 2)
    assert result.values.shape == (2, 3, 1, 2)
    np.testing.assert_array_equal(result.values, source[:, :, 1:2, :])


def test_sparse_grid_missing_policy_never_turns_holes_into_invalid_rows():
    a_id = AxisId("a")
    b_id = AxisId("b")
    mapping = ((0, 0), (1, 0), (0, 1))
    points = PointTable(
        3,
        (
            PointColumn(
                a_id,
                "a",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(cell[0] for cell in mapping),
            ),
            PointColumn(
                b_id,
                "b",
                SCAN_POINT,
                PointColumn.NUMERIC,
                tuple(cell[1] for cell in mapping),
            ),
        ),
    )
    block = block_for(
        repeat=1,
        points=points,
        grid_topology=GridTopology(
            (a_id, b_id),
            ((0, 1), (0, 1)),
            mapping,
        ),
        values=np.array([[2.0, 4.0, 10.0]]),
    )
    required = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.grid_dimension(a_id),),
                    ReductionMethod.MEAN,
                    MissingPolicy.REQUIRE_ALL,
                    ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert required.schema.point_table.row_count == 1
    assert required.schema.grid_topology is not None
    assert required.schema.grid_topology.row_to_cell == ((0,),)
    np.testing.assert_allclose(required.values, [[[3.0]]])
    np.testing.assert_array_equal(required.expanded_validity(), [[[True]]])

    omitted = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.grid_dimension(a_id),),
                    ReductionMethod.MEAN,
                    MissingPolicy.OMIT_MISSING,
                    ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert omitted.schema.point_table.row_count == 2
    np.testing.assert_allclose(omitted.values, [[[3.0], [10.0]]])
    np.testing.assert_array_equal(omitted.expanded_validity(), [[[True], [True]]])


def test_component_validity_is_reduced_per_named_component():
    site = axis("site", SITE, 2)
    validity = DatasetComponentValidity(
        (site.axis_id,),
        np.array([[[True, False]], [[True, True]]]),
    )
    block = block_for(
        repeat=2,
        points=PointTable(1),
        data_axes=(site,),
        values=np.array([[[1.0, 100.0]], [[3.0, 200.0]]]),
        validity=validity,
        component_axes=(site.axis_id,),
    )
    result = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.tensor(block.schema.repeat_axis.axis_id),),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    np.testing.assert_allclose(result.values, [[[2.0, 200.0]]])
    np.testing.assert_array_equal(result.expanded_validity(), [[[True, True]]])


def test_mixed_repeat_and_data_mean_uses_one_valid_contributor_count():
    site = axis("site", SITE, 2)
    validity = DatasetComponentValidity(
        (site.axis_id,), np.array([[[True, True]], [[True, False]]])
    )
    block = block_for(
        repeat=2,
        points=PointTable(1),
        data_axes=(site,),
        values=np.array([[[1.0, 100.0]], [[3.0, 1000.0]]]),
        validity=validity,
        component_axes=(site.axis_id,),
    )
    result = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (
                        AxisSourceRef.tensor(block.schema.repeat_axis.axis_id),
                        AxisSourceRef.tensor(site.axis_id),
                    ),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert result.schema.cell_schema.data_axes == (SCALAR_AXIS,)
    np.testing.assert_allclose(result.values, [[[(1.0 + 100.0 + 3.0) / 3.0]]])


def test_require_all_validity_marks_present_output_invalid_not_missing():
    block = block_for(
        repeat=2,
        points=PointTable(1),
        values=np.array([[1.0], [9.0]]),
        validity=CellValidity(np.array([[True], [False]])),
    )
    result = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.tensor(block.schema.repeat_axis.axis_id),),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.REQUIRE_ALL,
                ),
            )
        ),
    )
    assert result.schema.point_table.row_count == 1
    np.testing.assert_array_equal(result.expanded_validity(), [[[False]]])
    np.testing.assert_array_equal(result.values, [[[0.0]]])


def test_committed_transform_tree_carries_exact_rows_and_effective_schema():
    block = block_for(
        repeat=2,
        points=PointTable(1),
        values=np.array([[1], [2]], dtype=np.int16),
    )
    committed = commit_transform(
        block.schema,
        DataTransformSpec(
            (Selection.index(block.schema.repeat_axis.axis_id, 0),)
        ),
    )
    tree = committed_transform_to_tree(committed)

    assert set(tree) == {
        "schema",
        "source_schema_fingerprint",
        "exact_point_ordinals",
        "spec",
        "effective_output_schema",
    }
    assert tree["exact_point_ordinals"] == [0]
    assert tree["spec"]["operations"][0]["kind"] == "SELECT"
    assert committed_transform_from_tree(tree) == committed


def test_commit_is_schema_bound_and_authoritative():
    block = block_for(
        repeat=1,
        points=point_table("point", SCAN_POINT, (10, 20)),
        values=np.array([[1, 2]], dtype=np.int16),
    )
    committed = commit_transform(
        block.schema,
        DataTransformSpec(),
        point_ordinals=(0,),
    )
    assert (
        resolve_transformed_schema(block.schema, committed).fingerprint
        == committed.output_schema_fingerprint
    )
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("generation-1")), block)
    authoritative = apply_transform(snapshot, committed)
    np.testing.assert_array_equal(authoritative.values, [[[1]]])
    assert authoritative.schema.point_table.row_count == 1
    assert authoritative.source_ref == snapshot.ref
    assert authoritative.transform == committed


def test_huge_point_coordinates_subset_only_present_rows():
    logical_size = 100_000_000
    points = point_table(
        "huge",
        SCAN_POINT,
        (3, logical_size - 1),
        unit="shot",
    )
    block = block_for(
        repeat=2,
        points=points,
        values=np.array([[1, 2], [3, 4]], dtype=np.int16),
    )
    result = apply_spec(
        block,
        DataTransformSpec(),
        point_ordinals=(1,),
    )

    assert result.schema.point_table.row_count == 1
    selected = result.schema.point_table.column(AxisId("huge"))
    assert selected.values == (logical_size - 1,)
    assert selected.unit == "shot"
    assert result.values.nbytes == 4
    np.testing.assert_array_equal(result.values, [[[2]], [[4]]])


def test_coordinate_selection_requires_exact_frame():
    camera = CoordinateFrameId("camera")
    world = CoordinateFrameId("world")
    x = axis("x", SPATIAL_X, 3, coordinates=(0.0, 1.0, 2.0), frame=camera)
    block = block_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(x,),
        values=np.array([[[5, 6, 7]]], dtype=np.int16),
    )
    spec = DataTransformSpec(
        (Selection.coordinate_range(x.axis_id, 0, 1, coordinate_frame=world),)
    )
    with pytest.raises(ValueError, match="frame mismatch"):
        commit_transform(block.schema, spec)


def test_reduction_sources_are_canonical_and_min_max_respect_validity():
    site = axis("site", SITE, 3)
    block = block_for(
        repeat=2,
        points=PointTable(1),
        data_axes=(site,),
        values=np.array([[[2.0, 99.0, 7.0]], [[4.0, 8.0, 6.0]]]),
        validity=DatasetComponentValidity(
            (site.axis_id,), np.array([[[True, False, True]], [[True, True, True]]])
        ),
        component_axes=(site.axis_id,),
    )
    repeat_source = AxisSourceRef.tensor(block.schema.repeat_axis.axis_id)
    site_source = AxisSourceRef.tensor(site.axis_id)
    left = ReductionSpec(
        (site_source, repeat_source),
        ReductionMethod.MIN,
        validity_policy=ValidityPolicy.OMIT_INVALID,
    )
    right = ReductionSpec(
        (repeat_source, site_source),
        ReductionMethod.MIN,
        validity_policy=ValidityPolicy.OMIT_INVALID,
    )
    assert commit_transform(block.schema, DataTransformSpec((left,))).spec == (
        commit_transform(block.schema, DataTransformSpec((right,))).spec
    )
    minimum = apply_spec(block, DataTransformSpec((left,)))
    maximum = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (repeat_source, site_source),
                    ReductionMethod.MAX,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    np.testing.assert_array_equal(minimum.values, [[[2.0]]])
    np.testing.assert_array_equal(maximum.values, [[[8.0]]])


@pytest.mark.parametrize("method", tuple(ReductionMethod))
def test_data_only_reduce_preserves_repeat_and_point_rows(method):
    d0 = axis("d0", SITE, 2)
    d1 = axis("d1", SPATIAL_X, 3)
    values = np.arange(2 * 3 * 2 * 3, dtype=np.float64).reshape(2, 3, 2, 3)
    block = block_for(
        repeat=2,
        points=PointTable(3),
        data_axes=(d0, d1),
        values=values,
    )
    result = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.tensor(d0.axis_id),),
                    method,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    expected = {
        ReductionMethod.MEAN: np.mean,
        ReductionMethod.SUM: np.sum,
        ReductionMethod.MIN: np.min,
        ReductionMethod.MAX: np.max,
    }[method](values, axis=2)
    assert result.values.shape == (2, 3, 3)
    np.testing.assert_allclose(result.values, expected)


def test_integer_sum_is_exact_and_overflow_fails_closed():
    component = axis("component", SITE, 2)

    def reduce_values(values):
        block = block_for(
            repeat=1,
            points=PointTable(1),
            data_axes=(component,),
            values=np.asarray(values).reshape(1, 1, 2),
        )
        return apply_spec(
            block,
            DataTransformSpec(
                (
                    ReductionSpec(
                        (AxisSourceRef.tensor(component.axis_id),),
                        ReductionMethod.SUM,
                    ),
                )
            ),
        )

    maximum = np.iinfo(np.int64).max
    np.testing.assert_array_equal(
        reduce_values(np.array([maximum, -maximum])).values,
        [[[0]]],
    )
    unsigned_maximum = np.iinfo(np.uint64).max
    np.testing.assert_array_equal(
        reduce_values(np.array([unsigned_maximum, 0], dtype=np.uint64)).values,
        [[[unsigned_maximum]]],
    )
    with pytest.raises(OverflowError):
        reduce_values(np.array([maximum, 1], dtype=np.int64))
    floating = reduce_values(np.array([40_000, 40_000], dtype=np.float16))
    assert floating.schema.cell_schema.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(floating.values, [[[80_000.0]]])


def test_min_count_is_distinct_from_missing_and_invalid():
    block = block_for(
        repeat=3,
        points=PointTable(1),
        values=np.array([[1.0], [3.0], [100.0]]),
        validity=CellValidity(np.array([[True], [True], [False]])),
    )

    def reduction(count: int) -> DataTransformSpec:
        return DataTransformSpec(
            (
                ReductionSpec(
                    (AxisSourceRef.tensor(block.schema.repeat_axis.axis_id),),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.MIN_COUNT,
                    minimum_valid_count=count,
                ),
            )
        )

    valid = apply_spec(block, reduction(2))
    invalid = apply_spec(block, reduction(3))
    np.testing.assert_array_equal(valid.values, [[[2.0]]])
    np.testing.assert_array_equal(valid.expanded_validity(), [[[True]]])
    np.testing.assert_array_equal(invalid.expanded_validity(), [[[False]]])
    with pytest.raises(ValueError, match="maximum"):
        commit_transform(block.schema, reduction(4))


def test_large_integral_coordinate_is_selected_without_float_rounding():
    value = 2**53 + 1
    frame = CoordinateFrameId("counter")
    counter = axis(
        "counter",
        SPATIAL_X,
        2,
        coordinates=(value - 1, value),
        frame=frame,
    )
    block = block_for(
        repeat=1,
        points=PointTable(1),
        data_axes=(counter,),
        values=np.array([[[11, 22]]], dtype=np.int16),
    )
    selection = Selection.coordinate_range(
        counter.axis_id, value, value, coordinate_frame=frame
    )
    result = apply_spec(block, DataTransformSpec((selection,)))
    np.testing.assert_array_equal(result.values, [[[22]]])


def test_validity_stays_named_and_compact_for_large_images():
    y = axis("y", SPATIAL_X, 100)
    x = axis("x", SITE, 100)
    block = block_for(
        repeat=2,
        points=PointTable(1),
        data_axes=(y, x),
        values=np.zeros((2, 1, 100, 100), dtype=np.uint16),
        validity=CellValidity(np.array([[True], [False]])),
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(y.axis_id, 0, 50),)),
    )
    assert isinstance(result.validity, CellValidity)
    assert result.validity.mask.shape == (2, 1)
    assert result.validity.mask.nbytes == 2
    assert result.expanded_validity().shape == result.values.shape


def test_authority_api_rejects_raw_blocks_and_raw_specs():
    block = block_for(
        repeat=1,
        points=PointTable(1),
        values=np.array([[1]], dtype=np.int16),
    )
    spec = DataTransformSpec(
        (Selection.index(block.schema.repeat_axis.axis_id, 0),)
    )
    committed = commit_transform(block.schema, spec)
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("authority-generation")), block)
    with pytest.raises(TypeError, match="OwnedSnapshot"):
        apply_transform(block, committed)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CommittedTransform"):
        apply_transform(snapshot, spec)  # type: ignore[arg-type]


def test_dense_schema_commit_derives_the_expected_output_schema():
    repeat = axis("repeat", REPEAT, 20)
    schema = DatasetSchema(
        repeat,
        PointTable(50_000),
        None,
        ValueSchema.scalar(np.dtype(np.float64)),
    )
    spec = DataTransformSpec(
        (
            ReductionSpec(
                (AxisSourceRef.tensor(repeat.axis_id),),
                ReductionMethod.MEAN,
            ),
        )
    )
    committed = commit_transform(schema, spec)
    assert committed.output_schema_fingerprint
    transformed = resolve_transformed_schema(schema, committed)
    assert transformed.repeat_axis.size == 1
    assert transformed.point_table == schema.point_table
    assert transformed.physical_shape == (1, 50_000, 1)


def test_shared_reduction_numeric_policy_rejects_wraparound_and_unsafe_dtype():
    assert canonical_mean_dtype(np.dtype(np.int16)) == np.dtype("<f8")
    assert canonical_sum_dtype(np.dtype(np.uint16)) == np.dtype("<u8")
    np.testing.assert_array_equal(
        checked_numeric_sum(np.array([1, 2, 3], dtype=np.int16), (0,)),
        np.array(6, dtype=np.int64),
    )
    with pytest.raises(OverflowError, match="integer SUM"):
        checked_numeric_sum(
            np.array([np.iinfo(np.int64).max, 1], dtype=np.int64),
            (0,),
        )
    with pytest.raises(TypeError, match="canonical"):
        checked_numeric_sum(
            np.array([1, 2], dtype=np.int16),
            (0,),
            output_dtype=np.dtype(np.int16),
        )
