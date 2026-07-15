"""Adversarial contracts for sparse, named authoritative transforms."""

from __future__ import annotations

import numpy as np
import pytest
import time
import tracemalloc

from zlc_data import (
    REPEAT,
    SCAN_POINT,
    SITE,
    SPATIAL_X,
    AxisId,
    AxisLayout,
    AxisLayoutMode,
    AxisSpec,
    BlockId,
    CellValidity,
    CommittedTransform,
    ComponentValidity,
    CoordinateFrameId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetSchema,
    IndexRangeSelection,
    MissingPolicy,
    OwnedSnapshot,
    PointLayout,
    ReductionMethod,
    ReductionSpec,
    RowComponentValidity,
    Selection,
    StreamGenerationId,
    TransformedData,
    VALID,
    ValidityContract,
    ValidityPolicy,
    ValueSchema,
    apply_transform,
    axis_layout_from_tree,
    axis_layout_to_tree,
    commit_transform,
    committed_transform_from_tree,
    committed_transform_to_tree,
    resolve_transformed_schema,
    selection_from_tree,
    selection_to_tree,
)
from zlc_data.numeric import (
    canonical_mean_dtype,
    canonical_sum_dtype,
    checked_numeric_sum,
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


def block_for(
    *,
    repeat: int,
    point_axes: tuple[AxisSpec, ...],
    layout: PointLayout,
    data_axes: tuple[AxisSpec, ...] = (),
    values,
    validity=VALID,
    component_axes: tuple[AxisId, ...] = (),
) -> DataBlock:
    contract = (
        ValidityContract.components(*component_axes)
        if component_axes
        else ValidityContract.value()
    )
    schema = DatasetSchema(
        axis("repeat", REPEAT, repeat),
        point_axes,
        layout,
        ValueSchema(data_axes, contract, np.asarray(values).dtype),
    )
    return DataBlock(
        BlockId("source"), DatasetRevision(3), np.asarray(values), validity, schema
    )


def apply_spec(block: DataBlock, spec: DataTransformSpec) -> TransformedData:
    """Exercise the real authority path; tests never get a preview shortcut."""

    committed = commit_transform(block.schema, spec)
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


def test_repeated_index_ranges_preserve_absolute_coordinate_less_axis_indices():
    point = AxisSpec(
        AxisId("point"), "point", SCAN_POINT, 8, unit="pixel"
    )
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((point.size,)),
        values=np.arange(point.size, dtype=np.int16).reshape(1, -1),
    )
    spec = DataTransformSpec(
        (
            Selection((IndexRangeSelection(point.axis_id, 2, 7),)),
            Selection((IndexRangeSelection(point.axis_id, 1, 4),)),
        )
    )

    result = apply_spec(block, spec)
    selected_axis = result.schema.axis(point.axis_id)
    assert selected_axis.coordinates is None
    assert selected_axis.index_origin == 3
    assert selected_axis.unit == "pixel"
    assert tuple(selected_axis.coordinate_at(index) for index in range(3)) == (3, 4, 5)
    np.testing.assert_array_equal(result.values, (3, 4, 5))


def test_empty_axis_layout_is_only_for_derived_sparse_results():
    empty = AxisLayout.explicit((4,), ())
    assert empty.storage_size == 0
    with pytest.raises(ValueError, match="positive"):
        PointLayout.explicit((4,), ())
    assert AxisLayout.from_mapping((2, 2), ((0, 0), (0, 1), (1, 0), (1, 1))).mode \
        is AxisLayoutMode.RECT_C
    assert AxisLayout.from_mapping((2, 2), ((0, 0), (1, 0), (0, 1), (1, 1))).mode \
        is AxisLayoutMode.RECT_F


def test_rect_f_multi_point_mapping_is_recovered_before_selecting():
    a = axis("a", SCAN_POINT, 2)
    b = axis("b", SCAN_POINT, 3)
    values = np.arange(12, dtype=np.int16).reshape(2, 6)
    block = block_for(
        repeat=2,
        point_axes=(a, b),
        layout=PointLayout.rect_f((2, 3)),
        values=values,
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index(b.axis_id, 1),)),
    )

    assert tuple(item.axis_id for item in result.schema.cell_axes) == (
        AxisId("repeat"),
        a.axis_id,
    )
    assert [
        result.schema.cell_layout.multi_index(index)
        for index in range(result.schema.cell_layout.storage_size)
    ] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    np.testing.assert_array_equal(result.values, [2, 3, 8, 9])


def test_selecting_a_logically_valid_sparse_hole_returns_no_physical_row():
    point = axis("point", SCAN_POINT, 4)
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.explicit((4,), ((0,), (3,))),
        values=np.array([[10, 40]], dtype=np.int16),
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index(point.axis_id, 2),)),
    )
    assert result.schema.cell_axes == (block.schema.repeat_axis,)
    assert result.schema.cell_layout.storage_size == 0
    assert result.values.shape == (0,)
    assert result.expanded_validity().shape == (0,)


def test_data_axis_selection_preserves_every_other_data_axis():
    site = axis("site", SITE, 3)
    component = axis("component", SPATIAL_X, 2)
    point = axis("point", SCAN_POINT, 1)
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        data_axes=(site, component),
        values=np.arange(6, dtype=np.int16).reshape(1, 1, 3, 2),
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(site.axis_id, 1, 3),)),
    )
    assert tuple(axis.axis_id for axis in result.schema.data_axes) == (
        site.axis_id,
        component.axis_id,
    )
    assert result.schema.data_shape == (2, 2)
    np.testing.assert_array_equal(result.values[0], [[2, 3], [4, 5]])


def test_transform_keeps_repeat_point_and_every_trailing_data_axis_distinct():
    point = axis("point", SCAN_POINT, 3)
    site = axis("site", SITE, 2)
    component = axis("component", SPATIAL_X, 2)
    source = np.arange(2 * 3 * 2 * 2, dtype=np.int16).reshape(2, 3, 2, 2)
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((3,)),
        data_axes=(site, component),
        values=source,
    )

    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(site.axis_id, 1, 2),)),
    )

    assert block.values.shape == (2, 3, 2, 2)
    assert tuple(item.axis_id for item in result.schema.cell_axes) == (
        AxisId("repeat"),
        point.axis_id,
    )
    assert result.schema.cell_layout.logical_shape == (2, 3)
    assert tuple(item.axis_id for item in result.schema.data_axes) == (
        site.axis_id,
        component.axis_id,
    )
    assert result.schema.data_shape == (1, 2)
    assert result.values.shape == (6, 1, 2)
    np.testing.assert_array_equal(result.values, source[:, :, 1:2, :].reshape(6, 1, 2))


def test_sparse_missing_policy_never_turns_holes_into_invalid_rows():
    a = axis("a", SCAN_POINT, 2)
    b = axis("b", SCAN_POINT, 2)
    block = block_for(
        repeat=1,
        point_axes=(a, b),
        layout=PointLayout.explicit((2, 2), ((0, 0), (1, 0), (0, 1))),
        values=np.array([[2.0, 4.0, 10.0]]),
    )
    required = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (a.axis_id,),
                    ReductionMethod.MEAN,
                    MissingPolicy.REQUIRE_ALL,
                    ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert required.schema.cell_layout.storage_size == 1
    assert required.schema.cell_layout.multi_index(0) == (0, 0)
    np.testing.assert_allclose(required.values, [3.0])
    np.testing.assert_array_equal(required.expanded_validity(), [True])

    omitted = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (a.axis_id,),
                    ReductionMethod.MEAN,
                    MissingPolicy.OMIT_MISSING,
                    ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert omitted.schema.cell_layout.storage_size == 2
    np.testing.assert_allclose(omitted.values, [3.0, 10.0])
    np.testing.assert_array_equal(omitted.expanded_validity(), [True, True])


def test_component_validity_is_reduced_per_named_component():
    point = axis("point", SCAN_POINT, 1)
    site = axis("site", SITE, 2)
    validity = ComponentValidity(
        (site.axis_id,),
        np.array([[[True, False]], [[True, True]]]),
    )
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
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
                    (block.schema.repeat_axis.axis_id,),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    np.testing.assert_allclose(result.values, [[2.0, 200.0]])
    np.testing.assert_array_equal(result.expanded_validity(), [[True, True]])


def test_mixed_cell_and_data_mean_uses_one_valid_contributor_count():
    point = axis("point", SCAN_POINT, 1)
    site = axis("site", SITE, 2)
    validity = ComponentValidity(
        (site.axis_id,), np.array([[[True, True]], [[True, False]]])
    )
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
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
                    (block.schema.repeat_axis.axis_id, site.axis_id),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    assert result.schema.data_axes == ()
    np.testing.assert_allclose(result.values, [(1.0 + 100.0 + 3.0) / 3.0])


def test_require_all_validity_marks_present_output_invalid_not_missing():
    point = axis("point", SCAN_POINT, 1)
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        values=np.array([[1.0], [9.0]]),
        validity=CellValidity(np.array([[True], [False]])),
    )
    result = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (block.schema.repeat_axis.axis_id,),
                    ReductionMethod.MEAN,
                    validity_policy=ValidityPolicy.REQUIRE_ALL,
                ),
            )
        ),
    )
    assert result.schema.cell_layout.storage_size == 1
    np.testing.assert_array_equal(result.expanded_validity(), [False])
    np.testing.assert_array_equal(result.values, [0.0])


def test_committed_transform_tree_has_one_hand_written_current_shape():
    spec = DataTransformSpec((Selection.index(AxisId("point"), 0),))
    committed = CommittedTransform("a" * 64, spec, "b" * 64)
    literal = {
        "schema": "zlc_data.CommittedTransform",
        "input_schema_fingerprint": "a" * 64,
        "spec": {
            "schema": "zlc_data.DataTransformSpec",
            "operations": [
                {
                    "kind": "SELECT",
                    "selection": {
                        "schema": "zlc_data.Selection",
                        "terms": [
                            {"kind": "INDEX", "axis_id": "point", "index": 0}
                        ],
                    },
                }
            ],
        },
        "output_schema_fingerprint": "b" * 64,
    }
    assert committed_transform_to_tree(committed) == literal
    assert committed_transform_from_tree(literal) == committed


def test_commit_is_schema_bound_nonempty_and_authoritative():
    point = axis("point", SCAN_POINT, 2)
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((2,)),
        values=np.array([[1, 2]], dtype=np.int16),
    )
    with pytest.raises(ValueError, match="identity"):
        commit_transform(block.schema, DataTransformSpec(()))

    spec = DataTransformSpec((Selection.index(point.axis_id, 0),))
    committed = commit_transform(block.schema, spec)
    assert resolve_transformed_schema(block.schema, committed).fingerprint \
        == committed.output_schema_fingerprint
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("generation-1")), block)
    authoritative = apply_transform(snapshot, committed)
    np.testing.assert_array_equal(authoritative.values, [1])
    assert authoritative.source_ref == snapshot.ref
    assert authoritative.transform == committed


def test_huge_sparse_range_cost_tracks_present_rows_not_logical_length():
    logical_size = 100_000_000
    huge = AxisSpec(
        AxisId("huge"), "huge", SCAN_POINT, logical_size, unit="shot"
    )
    block = block_for(
        repeat=2,
        point_axes=(huge,),
        layout=PointLayout.explicit(
            (logical_size,), ((3,), (logical_size - 1,))
        ),
        values=np.array([[1, 2], [3, 4]], dtype=np.int16),
    )
    spec = DataTransformSpec(
        (Selection.index_range(huge.axis_id, 1, logical_size - 1),)
    )
    tracemalloc.start()
    started = time.perf_counter()
    result = apply_spec(
        block,
        spec,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert result.schema.cell_layout.storage_size == 2
    assert result.values.nbytes == 4
    selected_axis = result.schema.axis(huge.axis_id)
    assert selected_axis.coordinates is None
    assert selected_axis.index_origin == 1
    assert selected_axis.unit == "shot"
    assert [
        result.schema.cell_layout.multi_index(index)
        for index in range(result.schema.cell_layout.storage_size)
    ] == [(0, 2), (1, 2)]
    assert peak < 2_000_000
    assert elapsed < 5.0


def test_coordinate_selection_requires_exact_frame():
    camera = CoordinateFrameId("camera")
    world = CoordinateFrameId("world")
    point = axis("x", SCAN_POINT, 3, coordinates=(0.0, 1.0, 2.0), frame=camera)
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((3,)),
        values=np.array([[5, 6, 7]], dtype=np.int16),
    )
    spec = DataTransformSpec(
        (Selection.coordinate_range(point.axis_id, 0, 1, coordinate_frame=world),)
    )
    with pytest.raises(ValueError, match="frame mismatch"):
        commit_transform(block.schema, spec)


def test_reduction_axes_are_a_canonical_set_and_min_max_respect_validity():
    point = axis("point", SCAN_POINT, 1)
    site = axis("site", SITE, 3)
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        data_axes=(site,),
        values=np.array([[[2.0, 99.0, 7.0]], [[4.0, 8.0, 6.0]]]),
        validity=ComponentValidity(
            (site.axis_id,), np.array([[[True, False, True]], [[True, True, True]]])
        ),
        component_axes=(site.axis_id,),
    )
    repeat_id = block.schema.repeat_axis.axis_id
    left = ReductionSpec(
        (site.axis_id, repeat_id),
        ReductionMethod.MIN,
        validity_policy=ValidityPolicy.OMIT_INVALID,
    )
    right = ReductionSpec(
        (repeat_id, site.axis_id),
        ReductionMethod.MIN,
        validity_policy=ValidityPolicy.OMIT_INVALID,
    )
    assert left == right
    minimum = apply_spec(block, DataTransformSpec((left,)))
    maximum = apply_spec(
        block,
        DataTransformSpec(
            (
                ReductionSpec(
                    (repeat_id, site.axis_id),
                    ReductionMethod.MAX,
                    validity_policy=ValidityPolicy.OMIT_INVALID,
                ),
            )
        ),
    )
    np.testing.assert_array_equal(minimum.values, [2.0])
    np.testing.assert_array_equal(maximum.values, [8.0])


@pytest.mark.parametrize("method", tuple(ReductionMethod))
@pytest.mark.parametrize(
    "layout",
    (
        PointLayout.rect_c((2, 2)),
        PointLayout.rect_f((2, 2)),
        PointLayout.explicit((2, 2), ((1, 0), (0, 1), (0, 0))),
    ),
)
def test_data_only_reduce_drops_the_artificial_row_axis(method, layout):
    p0 = axis("p0", SCAN_POINT, 2)
    p1 = axis("p1", SCAN_POINT, 2)
    d0 = axis("d0", SITE, 2)
    d1 = axis("d1", SPATIAL_X, 3)
    values = np.arange(2 * layout.storage_size * 2 * 3, dtype=np.float64).reshape(
        2, layout.storage_size, 2, 3
    )
    block = block_for(
        repeat=2,
        point_axes=(p0, p1),
        layout=layout,
        data_axes=(d0, d1),
        values=values,
    )
    spec = DataTransformSpec(
        (
            ReductionSpec(
                (d0.axis_id,),
                method,
                validity_policy=ValidityPolicy.OMIT_INVALID,
            ),
        )
    )
    result = apply_spec(block, spec)
    physical = values.reshape((-1, 2, 3))
    expected = {
        ReductionMethod.MEAN: np.mean,
        ReductionMethod.SUM: np.sum,
        ReductionMethod.MIN: np.min,
        ReductionMethod.MAX: np.max,
    }[method](physical, axis=1)
    assert result.values.shape == (2 * layout.storage_size, 3)
    np.testing.assert_allclose(result.values, expected)


def test_integer_sum_is_exact_and_overflow_fails_closed():
    point = axis("point", SCAN_POINT, 1)
    component = axis("component", SITE, 2)

    def reduce_values(values):
        block = block_for(
            repeat=1,
            point_axes=(point,),
            layout=PointLayout.rect_c((1,)),
            data_axes=(component,),
            values=np.asarray(values).reshape(1, 1, 2),
        )
        return apply_spec(
            block,
            DataTransformSpec(
                (
                    ReductionSpec(
                        (component.axis_id,), ReductionMethod.SUM
                    ),
                )
            ),
        )

    maximum = np.iinfo(np.int64).max
    np.testing.assert_array_equal(reduce_values(np.array([maximum, -maximum])).values, [0])
    unsigned_maximum = np.iinfo(np.uint64).max
    np.testing.assert_array_equal(
        reduce_values(np.array([unsigned_maximum, 0], dtype=np.uint64)).values,
        [unsigned_maximum],
    )
    with pytest.raises(OverflowError):
        reduce_values(np.array([maximum, 1], dtype=np.int64))
    floating = reduce_values(np.array([40_000, 40_000], dtype=np.float16))
    assert floating.schema.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(floating.values, [80_000.0])


def test_min_count_is_distinct_from_missing_and_invalid():
    point = axis("point", SCAN_POINT, 1)
    block = block_for(
        repeat=3,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        values=np.array([[1.0], [3.0], [100.0]]),
        validity=CellValidity(np.array([[True], [True], [False]])),
    )
    reduction = lambda count: DataTransformSpec(
        (
            ReductionSpec(
                (block.schema.repeat_axis.axis_id,),
                ReductionMethod.MEAN,
                validity_policy=ValidityPolicy.MIN_COUNT,
                minimum_valid_count=count,
            ),
        )
    )
    valid = apply_spec(block, reduction(2))
    invalid = apply_spec(block, reduction(3))
    np.testing.assert_array_equal(valid.values, [2.0])
    np.testing.assert_array_equal(valid.expanded_validity(), [True])
    np.testing.assert_array_equal(invalid.expanded_validity(), [False])
    with pytest.raises(ValueError, match="maximum contributor"):
        commit_transform(block.schema, reduction(4))


def test_large_integral_coordinate_is_selected_without_float_rounding():
    value = 2**53 + 1
    frame = CoordinateFrameId("counter")
    point = axis(
        "counter",
        SCAN_POINT,
        2,
        coordinates=(value - 1, value),
        frame=frame,
    )
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((2,)),
        values=np.array([[11, 22]], dtype=np.int16),
    )
    selection = Selection.coordinate_range(
        point.axis_id, value, value, coordinate_frame=frame
    )
    result = apply_spec(block, DataTransformSpec((selection,)))
    np.testing.assert_array_equal(result.values, [22])


def test_validity_stays_named_and_compact_for_large_images():
    point = axis("point", SCAN_POINT, 1)
    y = axis("y", SPATIAL_X, 100)
    x = axis("x", SITE, 100)
    block = block_for(
        repeat=2,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        data_axes=(y, x),
        values=np.zeros((2, 1, 100, 100), dtype=np.uint16),
        validity=CellValidity(np.array([[True], [False]])),
    )
    result = apply_spec(
        block,
        DataTransformSpec((Selection.index_range(point.axis_id, 0, 1),)),
    )
    assert isinstance(result.validity, RowComponentValidity)
    assert result.validity.axis_ids == ()
    assert result.validity.mask.shape == (2,)
    assert result.validity.mask.nbytes == 2
    assert result.expanded_validity().shape == result.values.shape


def test_authority_api_rejects_raw_blocks_and_raw_specs():
    point = axis("point", SCAN_POINT, 1)
    block = block_for(
        repeat=1,
        point_axes=(point,),
        layout=PointLayout.rect_c((1,)),
        values=np.array([[1]], dtype=np.int16),
    )
    spec = DataTransformSpec((Selection.index(point.axis_id, 0),))
    committed = commit_transform(block.schema, spec)
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("authority-generation")), block)
    with pytest.raises(TypeError, match="OwnedSnapshot"):
        apply_transform(block, committed)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CommittedTransform"):
        apply_transform(snapshot, spec)  # type: ignore[arg-type]


def test_dense_schema_commit_does_not_materialize_repeat_times_points():
    repeat = axis("repeat", REPEAT, 20)
    point = AxisSpec(AxisId("point"), "point", SCAN_POINT, 50_000)
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((50_000,)),
        ValueSchema((), ValidityContract.value(), np.dtype(np.float64)),
    )
    spec = DataTransformSpec(
        (
            ReductionSpec((repeat.axis_id,), ReductionMethod.MEAN),
        )
    )
    tracemalloc.start()
    started = time.perf_counter()
    committed = commit_transform(schema, spec)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert committed.output_schema_fingerprint
    assert elapsed < 1.0
    assert peak < 5_000_000


def test_product_layout_codec_has_one_canonical_factorization():
    sparse = AxisLayout.explicit((4,), ((1,), (3,)))
    layout = AxisLayout.product(AxisLayout.rect_c((2,)), sparse)
    restored = axis_layout_from_tree(axis_layout_to_tree(layout))
    assert restored == layout
    assert [restored.multi_index(index) for index in range(restored.storage_size)] == [
        (0, 1),
        (0, 3),
        (1, 1),
        (1, 3),
    ]
    assert AxisLayout.product(AxisLayout.rect_c((2,)), AxisLayout.rect_c((3,))).mode \
        is AxisLayoutMode.RECT_C
    noncanonical = {
        "logical_shape": [2, 3],
        "mode": "PRODUCT",
        "storage_size": 6,
        "factors": [
            axis_layout_to_tree(AxisLayout.rect_c((2,))),
            axis_layout_to_tree(AxisLayout.rect_c((3,))),
        ],
    }
    with pytest.raises(ValueError, match="non-canonical"):
        axis_layout_from_tree(noncanonical)
    with pytest.raises(ValueError, match="adjacent RECT_C"):
        AxisLayout(
            (2, 3),
            AxisLayoutMode.PRODUCT,
            6,
            None,
            (AxisLayout.rect_c((2,)), AxisLayout.rect_c((3,))),
        )
    np.testing.assert_array_equal(layout.axis_indices(0), [0, 0, 1, 1])
    np.testing.assert_array_equal(layout.axis_indices(1), [1, 3, 1, 3])
    assert not layout.axis_indices(0).flags.writeable


def test_factored_f_order_reduction_has_a_vectorized_performance_guard():
    repeat = axis("repeat", REPEAT, 10)
    p0 = AxisSpec(AxisId("p0"), "p0", SCAN_POINT, 100)
    p1 = AxisSpec(AxisId("p1"), "p1", SCAN_POINT, 50)
    schema = DatasetSchema(
        repeat,
        (p0, p1),
        PointLayout.rect_f((100, 50)),
        ValueSchema((), ValidityContract.value(), np.dtype(np.uint8)),
    )
    block = DataBlock(
        BlockId("factored"),
        DatasetRevision(0),
        np.ones(schema.physical_shape, dtype=np.uint8),
        VALID,
        schema,
    )
    started = time.perf_counter()
    result = apply_spec(
        block,
        DataTransformSpec(
            (ReductionSpec((p0.axis_id,), ReductionMethod.SUM),)
        ),
    )
    elapsed = time.perf_counter() - started
    assert result.values.shape == (10 * 50,)
    np.testing.assert_array_equal(result.values, 100)
    assert elapsed < 1.0


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
