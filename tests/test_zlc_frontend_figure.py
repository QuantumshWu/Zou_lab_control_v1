"""Headless contracts for current source-aware frontend Figures."""

from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import pytest

from zlc_data.axis import (
    COMPONENT,
    REPEAT,
    SCAN_POINT,
    SCALAR_AXIS,
    SITE,
    SPATIAL_X,
    SPATIAL_Y,
    AxisId,
    AxisSourceRef,
    AxisSpec,
    CoordinateFrameId,
)
from zlc_data.schema import DatasetSchema, GridTopology, PointColumn, PointTable, ValueSchema
from zlc_data.validity import CellValidity, DatasetComponentValidity, VALID, ValidityContract
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_frontend.figure.codec import (
    decode_figure_document,
    encode_figure_document,
    figure_document_to_tree,
    view_spec_from_tree,
    view_spec_to_tree,
)
from zlc_frontend.figure.contract import validate_view_spec
from zlc_frontend.figure.evaluate import (
    FigureEvaluationCancelled,
    FigureEvaluationDeadlineExceeded,
    FigureEvaluator,
    ResolvedDataset,
    ResolvedDatasetMap,
)
from zlc_frontend.figure.grid import resolve_grid_view
from zlc_frontend.figure.model import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    DisplayReduction,
    DisplayReductionMethod,
    EvaluatedAxis,
    EvaluatedCurve,
    EvaluatedHistogram,
    EvaluatedImage,
    EvaluatedMeter,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    LatestNonempty,
    SourceViewBinding,
    SuggestionStatus,
    ViewIntent,
    ViewPreferences,
    ViewSpec,
)
from zlc_frontend.figure.suggest import suggest_view
from zlc_storage.canonical import encode


def axis(name, role, size, *, coordinates=None, unit=None, frame=None):
    return AxisSpec(
        AxisId(name),
        name,
        role,
        size,
        tuple(range(size)) if coordinates is None else tuple(coordinates),
        unit,
        frame,
    )


def make_block(
    values,
    *,
    repeat_axis,
    point_table,
    grid_topology=None,
    data_axes=(),
    validity=VALID,
    component_axes=(),
    value_unit=None,
    revision=4,
):
    contract = (
        ValidityContract.components(*component_axes)
        if component_axes
        else ValidityContract.value()
    )
    array = np.asarray(values)
    cell_schema = (
        ValueSchema(tuple(data_axes), contract, array.dtype, value_unit)
        if data_axes
        else ValueSchema.scalar(array.dtype, value_unit)
    )
    if not data_axes:
        array = array[..., np.newaxis]
    schema = DatasetSchema(repeat_axis, point_table, grid_topology, cell_schema)
    return DataBlock(
        BlockId("block-a"),
        DatasetRevision(revision),
        array,
        validity,
        schema,
    )


def resolved(block):
    snapshot = OwnedSnapshot(block.ref(StreamGenerationId("generation-a")), block)
    dataset_id = DatasetId("dataset-a")
    return dataset_id, ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),))


def document(block, view):
    dataset_id, datasets = resolved(block)
    doc = FigureDocument(
        "figure-a",
        3,
        (DatasetDescriptor(dataset_id, "signal", block.schema.fingerprint),),
        (FigureLayer("layer-a", dataset_id, view),),
    )
    return doc, datasets


def evaluated_data(block, view):
    doc, datasets = document(block, view)
    return FigureEvaluator().evaluate(doc, datasets)


def only_series(evaluated):
    return evaluated.layers[0].cells[0].series[0]


def test_pulse_is_document_fed_and_cannot_enter_the_dataset_figure_path():
    repeat = axis("repeat", REPEAT, 1)
    schema = make_block(
        np.zeros((1, 2)),
        repeat_axis=repeat,
        point_table=PointTable(2),
    ).schema

    with pytest.raises(ValueError, match="document-fed"):
        ViewSpec(schema.fingerprint, ViewIntent.PULSE, ())
    with pytest.raises(ValueError, match="document-fed"):
        suggest_view(schema, ViewIntent.PULSE)


def test_suggest_curve_is_source_total_role_driven_and_deterministic():
    repeat = axis("repeat", REPEAT, 3)
    detuning = axis("detuning", SCAN_POINT, 5, unit="MHz")
    site = axis("site", SITE, 2)
    block = make_block(
        np.zeros((3, 5, 2)),
        repeat_axis=repeat,
        point_table=PointTable(
            detuning.size,
            (
                PointColumn(
                    detuning.axis_id,
                    detuning.name,
                    detuning.role,
                    PointColumn.NUMERIC,
                    detuning.coordinates,
                    detuning.unit,
                ),
            ),
        ),
        data_axes=(site,),
    )

    first = suggest_view(block.schema, ViewIntent.CURVE)
    second = suggest_view(block.schema, ViewIntent.CURVE)
    assert first == second and first.status is SuggestionStatus.RESOLVED
    assert first.spec is not None
    roles = {binding.source: binding.role for binding in first.spec.source_bindings}
    assert roles == {
        AxisSourceRef.point_coordinate(detuning.axis_id): AxisViewRole.X,
        AxisSourceRef.tensor(repeat.axis_id): AxisViewRole.REDUCED,
        AxisSourceRef.tensor(site.axis_id): AxisViewRole.BATCH,
    }
    validate_view_spec(block.schema, first.spec)


def test_curve_defaults_to_authored_rows_when_the_same_axis_has_grid_metadata():
    repeat = axis("repeat", REPEAT, 1)
    time_axis = axis(
        "trap_off_time",
        SCAN_POINT,
        2,
        coordinates=(0.0, 0.001),
        unit="s",
    )
    table = PointTable(
        2,
        (
            PointColumn(
                time_axis.axis_id,
                "Trap-off time",
                time_axis.role,
                PointColumn.NUMERIC,
                time_axis.coordinates,
                time_axis.unit,
            ),
        ),
    )
    block = make_block(
        np.asarray(((0.8, 0.2),)),
        repeat_axis=repeat,
        point_table=table,
        grid_topology=GridTopology(
            (time_axis.axis_id,),
            (time_axis.coordinates,),
            ((0,), (1,)),
        ),
    )

    suggestion = suggest_view(block.schema, ViewIntent.CURVE)

    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec is not None
    assert suggestion.spec.binding(
        AxisSourceRef.point_coordinate(time_axis.axis_id)
    ).role is AxisViewRole.X
    assert all(
        binding.source.kind != AxisSourceRef.GRID_DIMENSION
        for binding in suggestion.spec.source_bindings
    )
    validate_view_spec(block.schema, suggestion.spec)


def test_ambiguous_point_coordinates_require_one_explicit_x_source():
    repeat = axis("repeat", REPEAT, 1)
    first = axis("a", SCAN_POINT, 2)
    second = axis("b", SCAN_POINT, 3)
    schema = make_block(
        np.zeros((1, 6)),
        repeat_axis=repeat,
        point_table=PointTable(
            6,
            (
                PointColumn(
                    first.axis_id,
                    first.name,
                    first.role,
                    PointColumn.NUMERIC,
                    (0, 0, 0, 1, 1, 1),
                ),
                PointColumn(
                    second.axis_id,
                    second.name,
                    second.role,
                    PointColumn.NUMERIC,
                    (0, 1, 2, 0, 1, 2),
                ),
            ),
        ),
    ).schema

    ambiguous = suggest_view(schema, ViewIntent.CURVE)
    assert ambiguous.status is SuggestionStatus.NEEDS_INPUT
    assert ambiguous.reasons[0].code == "AMBIGUOUS_DISPLAY_SOURCE"

    resolved_view = suggest_view(
        schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(
            x_source=AxisSourceRef.point_coordinate(first.axis_id),
        ),
    )
    assert resolved_view.status is SuggestionStatus.RESOLVED
    assert resolved_view.spec is not None
    assert resolved_view.spec.binding(
        AxisSourceRef.point_coordinate(first.axis_id)
    ).role is AxisViewRole.X


def test_exact_point_ordinals_select_the_visible_image_frame():
    repeat = axis("rolling-repeat", REPEAT, 1)
    y = axis("rolling-y", SPATIAL_Y, 8)
    x = axis("rolling-x", SPATIAL_X, 8)
    values = np.empty((1, 8, 8, 8), dtype=np.uint16)
    for point_index in range(8):
        values[0, point_index] = point_index
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(8),
        data_axes=(y, x),
    )

    suggestion = suggest_view(block.schema, ViewIntent.IMAGE, point_ordinals=(7,))
    assert suggestion.status is SuggestionStatus.RESOLVED
    assert suggestion.spec is not None and suggestion.spec.point_ordinals == (7,)
    image = only_series(evaluated_data(block, suggestion.spec)).data
    assert isinstance(image, EvaluatedImage)
    np.testing.assert_array_equal(image.values, np.full((8, 8), 7))


def test_view_and_document_codecs_are_strict_and_canonical():
    fingerprint = "0" * 64
    repeat_source = AxisSourceRef.tensor(AxisId("repeat"))
    view = ViewSpec(
        fingerprint,
        ViewIntent.CURVE,
        (
            SourceViewBinding(AxisSourceRef.point_ordinal(), AxisViewRole.X),
            SourceViewBinding(
                repeat_source,
                AxisViewRole.SELECTED,
                selector=LatestNonempty(),
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(AxisId("site")),
                AxisViewRole.BATCH,
            ),
        ),
        (0, 2),
    )
    assert view_spec_from_tree(view_spec_to_tree(view)) == view

    dataset_id = DatasetId("dataset-a")
    doc = FigureDocument(
        "figure-codec",
        7,
        (DatasetDescriptor(dataset_id, "signal", fingerprint),),
        (FigureLayer("layer-a", dataset_id, view),),
    )
    assert decode_figure_document(encode_figure_document(doc)) == doc

    extra = deepcopy(view_spec_to_tree(view))
    extra["authority_seed"] = "forbidden"
    with pytest.raises(ValueError, match="exactly"):
        view_spec_from_tree(extra)
    malformed = figure_document_to_tree(doc)
    malformed["schema"] = "zlc_frontend.UnknownFigure"
    with pytest.raises(ValueError, match="expected schema"):
        decode_figure_document(encode(malformed))


def test_explicit_f_order_grid_recovers_image_before_repeat_mean():
    repeat = axis("repeat", REPEAT, 2)
    x = axis("grid-x", SPATIAL_X, 2)
    y = axis("grid-y", SPATIAL_Y, 3)
    mapping = ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2))
    table = PointTable(
        len(mapping),
        (
            PointColumn(
                x.axis_id,
                x.name,
                x.role,
                PointColumn.NUMERIC,
                tuple(x.coordinates[ix] for ix, _iy in mapping),
            ),
            PointColumn(
                y.axis_id,
                y.name,
                y.role,
                PointColumn.NUMERIC,
                tuple(y.coordinates[iy] for _ix, iy in mapping),
            ),
        ),
    )
    topology = GridTopology(
        (x.axis_id, y.axis_id),
        (x.coordinates, y.coordinates),
        mapping,
    )
    values = np.empty((2, len(mapping)), dtype=float)
    for r in range(2):
        for p, (ix, iy) in enumerate(mapping):
            values[r, p] = 100 * r + 10 * iy + ix
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=table,
        grid_topology=topology,
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.IMAGE,
        preferences=ViewPreferences(
            image_x_source=AxisSourceRef.grid_dimension(x.axis_id),
            image_y_source=AxisSourceRef.grid_dimension(y.axis_id),
        ),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    image = only_series(evaluated_data(block, suggestion.spec)).data
    np.testing.assert_allclose(image.values, [[50, 51], [60, 61], [70, 71]])
    np.testing.assert_array_equal(image.validity, np.ones((3, 2), dtype=bool))


def test_sparse_grid_preserves_missing_cell_as_invalid():
    repeat = axis("repeat", REPEAT, 1)
    x = axis("x", SPATIAL_X, 2)
    y = axis("y", SPATIAL_Y, 2)
    mapping = ((0, 0), (1, 0), (0, 1))
    table = PointTable(
        3,
        (
            PointColumn(
                x.axis_id,
                x.name,
                x.role,
                PointColumn.NUMERIC,
                (0, 1, 0),
            ),
            PointColumn(
                y.axis_id,
                y.name,
                y.role,
                PointColumn.NUMERIC,
                (0, 0, 1),
            ),
        ),
    )
    block = make_block(
        np.array([[1.0, 2.0, 3.0]]),
        repeat_axis=repeat,
        point_table=table,
        grid_topology=GridTopology(
            (x.axis_id, y.axis_id),
            (x.coordinates, y.coordinates),
            mapping,
        ),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.IMAGE,
        preferences=ViewPreferences(
            image_x_source=AxisSourceRef.grid_dimension(x.axis_id),
            image_y_source=AxisSourceRef.grid_dimension(y.axis_id),
        ),
    )
    image = only_series(evaluated_data(block, suggestion.spec)).data
    np.testing.assert_array_equal(image.values, [[1, 2], [3, 0]])
    np.testing.assert_array_equal(image.validity, [[True, True], [True, False]])


def test_image_evaluator_preserves_frames_units_and_axis_orientation():
    frame = CoordinateFrameId("camera-fidelity")
    repeat = axis("fidelity-repeat", REPEAT, 1)
    y = axis(
        "fidelity-y",
        SPATIAL_Y,
        2,
        coordinates=(20, 22),
        unit="pixel",
        frame=frame,
    )
    x = axis(
        "fidelity-x",
        SPATIAL_X,
        3,
        coordinates=(10, 12, 14),
        unit="pixel",
        frame=frame,
    )
    expected = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)

    for data_axes in ((y, x), (x, y)):
        physical = expected if data_axes == (y, x) else expected.T
        block = make_block(
            physical.reshape((1, 1, *physical.shape)),
            repeat_axis=repeat,
            point_table=PointTable(1),
            data_axes=data_axes,
            value_unit="photoelectron",
        )
        suggestion = suggest_view(block.schema, ViewIntent.IMAGE)
        assert suggestion.status is SuggestionStatus.RESOLVED
        image = only_series(evaluated_data(block, suggestion.spec)).data
        assert image.x_axis.source == AxisSourceRef.tensor(x.axis_id)
        assert image.y_axis.source == AxisSourceRef.tensor(y.axis_id)
        assert image.x_axis.coordinate_frame == frame
        assert image.y_axis.coordinate_frame == frame
        assert image.value_unit == "photoelectron"
        np.testing.assert_array_equal(image.values, expected)


def test_latest_nonempty_uses_one_global_repeat():
    repeat = axis("repeat", REPEAT, 2)
    block = make_block(
        np.array([[10.0, 11.0], [20.0, 21.0]]),
        repeat_axis=repeat,
        point_table=PointTable(2),
        validity=CellValidity(np.array([[True, True], [False, True]])),
    )
    repeat_source = AxisSourceRef.tensor(repeat.axis_id)
    suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(
            repeat_binding=SourceViewBinding(
                repeat_source,
                AxisViewRole.SELECTED,
                selector=LatestNonempty(),
            ),
        ),
    )
    evaluated = evaluated_data(block, suggestion.spec)
    resolution = next(
        item for item in evaluated.layers[0].resolutions if item.source == repeat_source
    )
    assert resolution.selector == "LATEST_NONEMPTY" and resolution.index == 1
    curve = only_series(evaluated).data
    np.testing.assert_array_equal(curve.values, [20, 21])
    np.testing.assert_array_equal(curve.validity, [False, True])


def test_histogram_samples_preserve_sources_and_invalid_drop_count():
    repeat = axis("repeat", REPEAT, 2)
    site = axis("site", SITE, 2, coordinates=("A", "B"))
    values = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(1),
        data_axes=(site,),
        validity=DatasetComponentValidity(
            (site.axis_id,),
            np.array([[[True, False]], [[True, True]]]),
        ),
        component_axes=(site.axis_id,),
    )
    site_source = AxisSourceRef.tensor(site.axis_id)
    suggestion = suggest_view(
        block.schema,
        ViewIntent.HISTOGRAM,
        preferences=ViewPreferences(sample_sources=(site_source,)),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    histogram = only_series(evaluated_data(block, suggestion.spec)).data
    assert isinstance(histogram, EvaluatedHistogram)
    np.testing.assert_array_equal(histogram.samples, [1, 3, 4])
    assert set(histogram.sample_sources) == {
        AxisSourceRef.tensor(repeat.axis_id),
        site_source,
    }
    assert histogram.dropped_count == 1


def test_curve_batch_and_repeat_sum_preserve_named_series():
    repeat = axis("repeat", REPEAT, 2)
    site = axis("site", SITE, 2)
    values = np.arange(12, dtype=float).reshape(2, 3, 2)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(3),
        data_axes=(site,),
    )
    repeat_source = AxisSourceRef.tensor(repeat.axis_id)
    suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(
            repeat_binding=SourceViewBinding(
                repeat_source,
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.SUM),
            ),
        ),
    )
    cell = evaluated_data(block, suggestion.spec).layers[0].cells[0]
    assert len(cell.series) == 2
    for site_index, series in enumerate(cell.series):
        assert series.batch_address[0].source == AxisSourceRef.tensor(site.axis_id)
        assert series.batch_address[0].index == site_index
        np.testing.assert_array_equal(
            series.data.values,
            values[:, :, site_index].sum(axis=0),
        )


def test_display_sum_overflow_is_fail_closed():
    repeat = axis("repeat", REPEAT, 2)
    block = make_block(
        np.array([[np.iinfo(np.int64).max], [1]], dtype=np.int64),
        repeat_axis=repeat,
        point_table=PointTable(1),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.METER,
        preferences=ViewPreferences(
            repeat_binding=SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.SUM),
            ),
        ),
    )
    with pytest.raises(OverflowError, match="integer SUM"):
        evaluated_data(block, suggestion.spec)


def test_evaluation_cancels_and_honours_deadline():
    repeat = axis("repeat", REPEAT, 1)
    block = make_block(
        np.arange(3.0).reshape(1, 3),
        repeat_axis=repeat,
        point_table=PointTable(3),
    )
    view = suggest_view(block.schema, ViewIntent.CURVE).spec
    doc, datasets = document(block, view)
    with pytest.raises(FigureEvaluationCancelled):
        FigureEvaluator().evaluate(doc, datasets, cancel_requested=lambda: True)
    with pytest.raises(FigureEvaluationDeadlineExceeded):
        FigureEvaluator().evaluate(
            doc,
            datasets,
            monotonic_deadline=time.monotonic() - 1.0,
        )


def test_real_36_by_32_faceted_curve_has_bounded_latency():
    repeat = axis("repeat", REPEAT, 4)
    site = axis("site", SITE, 32)
    component = axis("component", COMPONENT, 36)
    values = np.arange(4 * 100 * 32 * 36, dtype=np.float32).reshape(4, 100, 32, 36)
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(100),
        data_axes=(site, component),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.CURVE,
        preferences=ViewPreferences(
            batch_sources=(AxisSourceRef.tensor(site.axis_id),),
            facet_sources=(AxisSourceRef.tensor(component.axis_id),),
        ),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    started = time.perf_counter()
    evaluated = evaluated_data(block, suggestion.spec)
    elapsed = time.perf_counter() - started
    assert len(evaluated.layers[0].cells) == 36
    assert sum(len(cell.series) for cell in evaluated.layers[0].cells) == 36 * 32
    np.testing.assert_allclose(
        evaluated.layers[0].cells[0].series[0].data.values,
        values[:, :, 0, 0].mean(axis=0, dtype=np.float64),
    )
    np.testing.assert_allclose(
        evaluated.layers[0].cells[-1].series[-1].data.values,
        values[:, :, -1, -1].mean(axis=0, dtype=np.float64),
    )
    assert elapsed < 5.0


def test_public_evaluator_preserves_one_full_qcmos_frame():
    repeat = axis("camera-repeat", REPEAT, 1)
    y = AxisSpec(AxisId("camera-y"), "camera y", SPATIAL_Y, 2304)
    x = AxisSpec(AxisId("camera-x"), "camera x", SPATIAL_X, 2304)
    values = np.zeros((1, 1, 2304, 2304), dtype=np.uint16)
    values[0, 0, 0, 0] = 11
    values[0, 0, -1, -1] = 22
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(1),
        data_axes=(y, x),
    )
    view = suggest_view(block.schema, ViewIntent.IMAGE).spec
    doc, datasets = document(block, view)

    started = time.perf_counter()
    evaluated = FigureEvaluator().evaluate(doc, datasets)
    elapsed = time.perf_counter() - started

    series = only_series(evaluated)
    image = series.data
    assert isinstance(image, EvaluatedImage)
    assert image.values.shape == (2304, 2304)
    assert image.values.dtype == np.dtype("<u2")
    assert (image.values[0, 0], image.values[-1, -1]) == (11, 22)
    assert image.validity.dtype == np.dtype(bool) and bool(np.all(image.validity))
    assert series.reductions[0].sources == (AxisSourceRef.tensor(repeat.axis_id),)
    assert not image.values.flags.writeable and not image.validity.flags.writeable
    assert elapsed < 1.0


def test_single_repeat_image_selects_component_and_grid_facets_all_components():
    repeat = axis("component-repeat", REPEAT, 1)
    component = axis("component", COMPONENT, 2)
    y = axis("component-y", SPATIAL_Y, 2)
    x = axis("component-x", SPATIAL_X, 3)
    values = np.arange(1, 13, dtype=np.uint16).reshape(1, 1, 2, 2, 3)
    valid = np.ones(values.shape, dtype=bool)
    valid[0, 0, 1, 1, 2] = False
    block = make_block(
        values,
        repeat_axis=repeat,
        point_table=PointTable(1),
        data_axes=(component, y, x),
        validity=DatasetComponentValidity(
            (component.axis_id, y.axis_id, x.axis_id),
            valid,
        ),
        component_axes=(component.axis_id, y.axis_id, x.axis_id),
    )
    component_source = AxisSourceRef.tensor(component.axis_id)
    ordinary = suggest_view(block.schema, ViewIntent.IMAGE)
    assert ordinary.status is SuggestionStatus.RESOLVED
    assert ordinary.spec.binding(component_source).selector == FixedIndex(0)
    ordinary_image = only_series(evaluated_data(block, ordinary.spec)).data
    np.testing.assert_array_equal(ordinary_image.values, values[0, 0, 0])

    grid = resolve_grid_view(block.schema, ViewIntent.IMAGE, component_source)
    assert grid.status is SuggestionStatus.RESOLVED
    evaluated = evaluated_data(block, grid.spec)
    assert len(evaluated.layers[0].cells) == 2
    for component_index, cell in enumerate(evaluated.layers[0].cells):
        assert cell.facet_address[0].source == component_source
        assert cell.facet_address[0].index == component_index
        image = cell.series[0].data
        np.testing.assert_array_equal(image.values, values[0, 0, component_index])
        np.testing.assert_array_equal(image.validity, valid[0, 0, component_index])


def test_multi_repeat_integer_mean_remains_fractional_float64():
    repeat = axis("integer-repeat", REPEAT, 2)
    y = axis("integer-y", SPATIAL_Y, 1)
    x = axis("integer-x", SPATIAL_X, 2)
    block = make_block(
        np.array([[[[0, 2]]], [[[1, 3]]]], dtype=np.uint16),
        repeat_axis=repeat,
        point_table=PointTable(1),
        data_axes=(y, x),
    )
    series = only_series(
        evaluated_data(block, suggest_view(block.schema, ViewIntent.IMAGE).spec)
    )
    assert series.data.values.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(series.data.values, [[0.5, 2.5]])
    assert series.reductions[0].minimum_contributors == 2
    assert series.reductions[0].maximum_contributors == 2


def test_evaluated_validity_is_bool_and_complex_views_fail_closed():
    x_out = EvaluatedAxis(
        AxisSourceRef.tensor(AxisId("dto-x")),
        "x",
        SPATIAL_X,
        None,
        (0,),
        (0,),
    )
    y_out = EvaluatedAxis(
        AxisSourceRef.tensor(AxisId("dto-y")),
        "y",
        SPATIAL_Y,
        None,
        (0,),
        (0,),
    )
    with pytest.raises(TypeError, match="validity dtype must be bool"):
        EvaluatedImage(
            x_out,
            y_out,
            np.zeros((1, 1), dtype=np.float64),
            np.ones((1, 1), dtype=np.uint8),
        )
    with pytest.raises(TypeError, match="validity dtype must be bool"):
        EvaluatedCurve(
            x_out,
            None,
            np.zeros((1,), dtype=np.float64),
            np.ones((1,), dtype=np.uint8),
        )

    repeat = axis("complex-repeat", REPEAT, 1)
    y = axis("complex-y", SPATIAL_Y, 1)
    x = axis("complex-x", SPATIAL_X, 1)
    complex_block = make_block(
        np.ones((1, 1, 1, 1), dtype=np.complex128),
        repeat_axis=repeat,
        point_table=PointTable(1),
        data_axes=(y, x),
    )
    suggestion = suggest_view(complex_block.schema, ViewIntent.IMAGE)
    assert suggestion.status is SuggestionStatus.NEEDS_INPUT
    assert suggestion.reasons[0].code == "CONTRACT_REJECTED"
    assert "complex-value projection" in suggestion.reasons[0].message


def test_ragged_sparse_grid_reduction_uses_only_present_cells():
    repeat = axis("repeat", REPEAT, 1)
    x = axis("x", SCAN_POINT, 257)
    reduced = axis("reduced", SPATIAL_X, 4096)
    mapping = tuple((0, index) for index in range(4096)) + tuple(
        (index, 0) for index in range(1, 257)
    )
    table = PointTable(
        len(mapping),
        (
            PointColumn(
                x.axis_id,
                x.name,
                x.role,
                PointColumn.NUMERIC,
                tuple(x.coordinates[ix] for ix, _reduced in mapping),
            ),
            PointColumn(
                reduced.axis_id,
                reduced.name,
                reduced.role,
                PointColumn.NUMERIC,
                tuple(reduced.coordinates[iy] for _x, iy in mapping),
            ),
        ),
    )
    block = make_block(
        np.concatenate(
            (
                np.arange(4096, dtype=np.float64),
                np.arange(1, 257, dtype=np.float64) + 10_000,
            )
        ).reshape(1, -1),
        repeat_axis=repeat,
        point_table=table,
        grid_topology=GridTopology(
            (x.axis_id, reduced.axis_id),
            (x.coordinates, reduced.coordinates),
            mapping,
        ),
    )
    view = ViewSpec(
        block.schema.fingerprint,
        ViewIntent.CURVE,
        (
            SourceViewBinding(
                AxisSourceRef.grid_dimension(x.axis_id),
                AxisViewRole.X,
            ),
            SourceViewBinding(
                AxisSourceRef.grid_dimension(reduced.axis_id),
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.REDUCED,
                reduction=DisplayReduction(DisplayReductionMethod.MEAN),
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(SCALAR_AXIS.axis_id),
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
        ),
    )
    validate_view_spec(block.schema, view)
    curve = only_series(evaluated_data(block, view)).data
    assert curve.values[0] == pytest.approx(np.arange(4096).mean())
    np.testing.assert_array_equal(curve.values[1:], np.arange(1, 257) + 10_000)
    np.testing.assert_array_equal(curve.validity, np.ones(257, dtype=bool))


def test_meter_latest_with_point_coordinate_facets_yields_typed_scalars():
    repeat = axis("repeat", REPEAT, 2)
    point = axis("point", SCAN_POINT, 2, coordinates=(10, 20))
    block = make_block(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        repeat_axis=repeat,
        point_table=PointTable(
            2,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    point.coordinates,
                ),
            ),
        ),
    )
    suggestion = suggest_view(
        block.schema,
        ViewIntent.METER,
        preferences=ViewPreferences(
            facet_sources=(AxisSourceRef.point_coordinate(point.axis_id),),
        ),
    )
    assert suggestion.status is SuggestionStatus.RESOLVED
    evaluated = evaluated_data(block, suggestion.spec)
    assert len(evaluated.layers[0].cells) == 2
    meters = [cell.series[0].data for cell in evaluated.layers[0].cells]
    assert all(isinstance(meter, EvaluatedMeter) for meter in meters)
    assert [meter.value for meter in meters] == [3.0, 4.0]


def test_evaluated_meter_rejects_non_numeric_values():
    assert EvaluatedMeter(np.float32(1.25), True).value == pytest.approx(1.25)
    assert EvaluatedMeter(np.bool_(True), True).value is True
    with pytest.raises(TypeError, match="numeric or boolean scalar"):
        EvaluatedMeter("1.25", True)
    with pytest.raises(TypeError, match="numeric or boolean scalar"):
        EvaluatedMeter([1.25], True)
