"""Focused headless contracts for the interactive histogram core."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data import (
    COMPONENT,
    REPEAT,
    SCAN_POINT,
    SITE,
    AxisId,
    AxisSpec,
    BlockId,
    ComponentValidity,
    DataBlock,
    DatasetRevision,
    DatasetSchema,
    OwnedSnapshot,
    PointLayout,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure import (
    AxisViewBinding,
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedHistogram,
    EvaluatedSeries,
    FigureDocument,
    FigureEvaluator,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import (
    HistogramBinProjection,
    HistogramCountScale,
    HistogramDisplayState,
    HistogramViewportTransform,
    histogram_count_limits,
    histogram_display_form_values,
    histogram_display_from_form,
    histogram_display_with_x_view,
    histogram_home_x_limits,
    histogram_projection_home_x_limits,
)
from zlc_frontend.matplotlib_render import SinglePanelAggRenderer
from zlc_frontend.render import HistogramPanelPayload


def test_histogram_form_freezes_exact_front_and_validates_log_and_bins() -> None:
    base = HistogramDisplayState()
    values = histogram_display_form_values(base)
    assert histogram_display_from_form(base, values) is base

    values["relim_mode"] = RelimMode.FIXED
    values["count_min"] = -999.0
    values["count_max"] = 999.0
    fixed = histogram_display_from_form(
        base,
        values,
        current_count_limits=(0.0, 14.0),
    )
    assert fixed.revision == 1
    assert fixed.fixed_count_limits == (0.0, 14.0)
    assert histogram_display_from_form(
        fixed,
        histogram_display_form_values(fixed),
        current_count_limits=(0.0, 100.0),
    ) is fixed

    with pytest.raises(ValueError, match="between 5 and 500"):
        HistogramDisplayState(bin_count=4)
    with pytest.raises(TypeError, match="integer"):
        HistogramDisplayState(bin_count=True)
    with pytest.raises(ValueError, match="positive for log"):
        HistogramDisplayState(
            relim_mode=RelimMode.FIXED,
            count_scale=HistogramCountScale.LOG,
            fixed_count_limits=(0.0, 10.0),
        )

    log_values = histogram_display_form_values(base)
    log_values["relim_mode"] = RelimMode.FIXED
    log_values["count_scale"] = HistogramCountScale.LOG
    with pytest.raises(ValueError, match="positive for log"):
        histogram_display_from_form(
            base,
            log_values,
            current_count_limits=(0.0, 10.0),
        )
    leave_fixed = histogram_display_form_values(fixed)
    leave_fixed["relim_mode"] = RelimMode.TIGHT
    leave_fixed["count_scale"] = HistogramCountScale.LOG
    log_auto = histogram_display_from_form(fixed, leave_fixed)
    assert log_auto.relim_mode is RelimMode.TIGHT
    assert log_auto.count_scale is HistogramCountScale.LOG

    pinned = histogram_display_with_x_view(base, (-2.0, 3.0))
    assert pinned.revision == 1 and pinned.x_view == (-2.0, 3.0)
    assert histogram_display_with_x_view(pinned, (-2.0, 3.0)) is pinned


def test_histogram_viewport_maps_linear_and_log_counts_exactly() -> None:
    linear = HistogramViewportTransform(
        4,
        (0.1, 0.2, 0.9, 0.8),
        (0.0, 10.0),
        (0.0, 20.0),
        (-5.0, 15.0),
        HistogramCountScale.LINEAR,
        RelimMode.TIGHT,
        False,
        5,
    )
    assert linear.widget_normalized_to_data(0.5, 0.5) == pytest.approx(
        (5.0, 10.0)
    )
    assert linear.data_to_widget_normalized(5.0, 10.0) == pytest.approx(
        (0.5, 0.5)
    )
    assert linear.zoomed_x_limits(5.0, 0.5) == (2.5, 7.5)
    assert linear.panned_x_limits(0.3, 0.5) == pytest.approx((-2.5, 7.5))
    assert linear.selection_x_span(0.7, 0.3) == pytest.approx((2.5, 7.5))

    log = HistogramViewportTransform(
        5,
        (0.0, 0.0, 1.0, 1.0),
        (-1.0, 1.0),
        (0.5, 50.0),
        (-1.0, 1.0),
        HistogramCountScale.LOG,
        RelimMode.TIGHT,
        True,
        5,
    )
    geometric_middle = np.sqrt(0.5 * 50.0)
    assert log.widget_normalized_to_data(0.5, 0.5) == pytest.approx(
        (0.0, geometric_middle)
    )
    assert log.data_to_widget_normalized(0.0, geometric_middle) == pytest.approx(
        (0.5, 0.5)
    )
    with pytest.raises(ValueError, match="positive for log"):
        HistogramViewportTransform(
            0,
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 1.0),
            (0.0, 10.0),
            (0.0, 1.0),
            HistogramCountScale.LOG,
            RelimMode.TIGHT,
            True,
            5,
        )
    with pytest.raises(ValueError, match="positive for log"):
        log.data_to_widget_normalized(0.0, 0.0)


def test_histogram_count_relim_is_count_specific_and_hysteretic() -> None:
    tight = HistogramDisplayState(relim_mode=RelimMode.TIGHT)
    normal = HistogramDisplayState(relim_mode=RelimMode.NORMAL)
    log = HistogramDisplayState(count_scale=HistogramCountScale.LOG)
    assert histogram_count_limits(tight, 10) == (0.0, 11.0)
    assert histogram_count_limits(normal, 10) == (0.0, 12.0)
    assert histogram_count_limits(log, 10) == (0.5, 30.0)
    assert histogram_count_limits(tight, 0) == (0.0, 1.0)
    assert histogram_count_limits(log, 0) == (0.5, 1.0)

    assert histogram_count_limits(
        tight,
        11,
        current_count_limits=(0.0, 20.0),
        previous_relim_mode=RelimMode.TIGHT,
        previous_count_scale=HistogramCountScale.LINEAR,
    ) == (0.0, 20.0)
    assert histogram_count_limits(
        tight,
        10,
        current_count_limits=(0.0, 20.0),
        previous_relim_mode=RelimMode.TIGHT,
        previous_count_scale=HistogramCountScale.LINEAR,
    ) == (0.0, 11.0)
    assert histogram_count_limits(
        tight,
        10,
        current_count_limits=(0.0, 20.0),
        previous_relim_mode=RelimMode.NORMAL,
        previous_count_scale=HistogramCountScale.LINEAR,
    ) == (0.0, 11.0)
    fixed = HistogramDisplayState(
        relim_mode=RelimMode.FIXED,
        fixed_count_limits=(2.0, 4.0),
    )
    assert histogram_count_limits(fixed, 1_000_000) == (2.0, 4.0)


def test_shared_histogram_binning_is_common_lossless_and_immutable() -> None:
    first = np.asarray((-10.0, -5.0, 0.0))
    second = np.asarray((10.0, 20.0))
    projection = HistogramBinProjection((first, second), bins=5)
    counts, edges = projection.bin_counts, projection.bin_edges
    assert len(counts) == 2
    assert counts[0].sum() == len(first)
    assert counts[1].sum() == len(second)
    assert edges[0] <= first.min() and edges[-1] >= second.max()
    assert histogram_home_x_limits(edges) == (edges[0], edges[-1])
    assert histogram_projection_home_x_limits((first, second), bins=5) == (
        edges[0],
        edges[-1],
    )
    assert not counts[0].flags.writeable and not edges.flags.writeable
    with pytest.raises(ValueError):
        counts[0][0] = 99
    with pytest.raises(ValueError):
        edges[0] = 99.0

    bool_projection = HistogramBinProjection(
        (
            np.asarray((False, True, True)),
            np.asarray((True,)),
        ),
        bins=500,
    )
    bool_counts, bool_edges = (
        bool_projection.bin_counts,
        bool_projection.bin_edges,
    )
    np.testing.assert_array_equal(bool_edges, (-0.5, 0.5, 1.5))
    np.testing.assert_array_equal(bool_counts[0], (1, 2))
    np.testing.assert_array_equal(bool_counts[1], (0, 1))

    empty_projection = HistogramBinProjection(
        (np.asarray((), dtype=np.float64),), bins=5
    )
    empty_counts = empty_projection.bin_counts[0]
    empty_edges = empty_projection.bin_edges
    np.testing.assert_array_equal(empty_counts, np.zeros(5, dtype=np.int64))
    assert histogram_home_x_limits(empty_edges) == (0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        HistogramBinProjection((np.asarray((1.0, np.nan)),), bins=5)
    with pytest.raises(ValueError, match="one-dimensional"):
        HistogramBinProjection((np.zeros((2, 2)),), bins=5)

    fixed_projection = HistogramBinProjection(
        (first, second),
        bins=5,
        value_range=(-20.0, 30.0),
    )
    np.testing.assert_array_equal(
        fixed_projection.bin_edges,
        np.linspace(-20.0, 30.0, 6),
    )
    assert tuple(int(counts.sum()) for counts in fixed_projection.bin_counts) == (
        len(first),
        len(second),
    )
    assert histogram_projection_home_x_limits(
        (first, second),
        bins=5,
        value_range=(-20.0, 30.0),
    ) == (-20.0, 30.0)
    with pytest.raises(ValueError, match="did not retain every sample"):
        HistogramBinProjection(
            (first, second),
            bins=5,
            value_range=(-6.0, 30.0),
        )
    with pytest.raises(ValueError, match="false/true bins"):
        HistogramBinProjection(
            (np.asarray((False, True)),),
            bins=5,
            value_range=(0.0, 1.0),
        )


def test_histogram_evaluator_preserves_unit_coordinates_and_component_validity() -> None:
    repeat = AxisSpec(AxisId("repeat"), "Repeat", REPEAT, 2)
    point = AxisSpec(AxisId("point"), "Point", SCAN_POINT, 1)
    site = AxisSpec(
        AxisId("site"),
        "Site",
        SITE,
        2,
        coordinates=("A", "B"),
    )
    channel = AxisSpec(
        AxisId("channel"),
        "Channel",
        COMPONENT,
        2,
        coordinates=("x", "y"),
    )
    values = np.asarray(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[5.0, 6.0], [7.0, 8.0]]],
        ]
    )
    valid_mask = np.ones_like(values, dtype=bool)
    valid_mask[0, 0, 1, 1] = False
    validity = ComponentValidity(
        (site.axis_id, channel.axis_id),
        valid_mask,
    )
    schema = DatasetSchema(
        repeat,
        (point,),
        PointLayout.rect_c((1,)),
        ValueSchema(
            (site, channel),
            ValidityContract.components(site.axis_id, channel.axis_id),
            values.dtype,
            value_unit="photoelectron",
        ),
    )
    block = DataBlock(
        BlockId("histogram-block"),
        DatasetRevision(4),
        values,
        validity,
        schema,
    )
    dataset_id = DatasetId("histogram-dataset")
    view = ViewSpec(
        schema.fingerprint,
        ViewIntent.HISTOGRAM,
        (
            AxisViewBinding(repeat.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(
                point.axis_id,
                AxisViewRole.SELECTED,
                selector=FixedIndex(0),
            ),
            AxisViewBinding(site.axis_id, AxisViewRole.SAMPLE),
            AxisViewBinding(channel.axis_id, AxisViewRole.SAMPLE),
        ),
    )
    document = FigureDocument(
        "histogram-document",
        1,
        (DatasetDescriptor(dataset_id, "Counts", schema.fingerprint),),
        (FigureLayer("histogram-layer", dataset_id, view),),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("histogram-generation")),
        block,
    )
    evaluated = FigureEvaluator().evaluate(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )
    histogram = evaluated.layers[0].cells[0].series[0].data
    assert isinstance(histogram, EvaluatedHistogram)
    assert histogram.value_unit == "photoelectron"
    np.testing.assert_array_equal(
        histogram.samples,
        (1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0),
    )
    coordinates = {
        item.axis_id: item.coordinates for item in histogram.sample_coordinates
    }
    assert coordinates[repeat.axis_id] == (0, 0, 0, 1, 1, 1, 1)
    assert coordinates[site.axis_id] == ("A", "A", "B", "A", "A", "B", "B")
    assert coordinates[channel.axis_id] == ("x", "y", "x", "x", "y", "x", "y")
    assert histogram.dropped_count == 1

    renderer = SinglePanelAggRenderer(document, width=420, height=280)
    try:
        raster, payload = renderer.render_interactive_histogram(
            evaluated,
            HistogramDisplayState(),
            current_count_limits=None,
            previous_relim_mode=None,
            previous_count_scale=None,
        )
        assert raster.pixels
        assert isinstance(payload, HistogramPanelPayload)
        assert payload.evaluated_input == evaluated.inputs[0]
        assert payload.value_unit == "photoelectron"
        assert payload.series[0].data.dropped_count == 1
        assert int(payload.bin_counts[0].sum()) == len(histogram.samples)
        assert payload.viewport.display_revision == 0
        assert payload.viewport.home_x_limits == (
            float(payload.bin_edges[0]),
            float(payload.bin_edges[-1]),
        )
        assert not payload.bin_counts[0].flags.writeable
        assert not payload.bin_edges.flags.writeable
        assert payload.bin_projection.series_samples[0] is histogram.samples
        other_histogram = EvaluatedHistogram(
            np.array(histogram.samples, copy=True),
            histogram.sample_coordinates,
            histogram.dropped_count,
            histogram.value_unit,
        )
        original_series = evaluated.layers[0].cells[0].series[0]
        with pytest.raises(ValueError, match="exact samples"):
            HistogramPanelPayload(
                payload.evaluated_input,
                payload.viewport,
                (
                    EvaluatedSeries(
                        original_series.batch_address,
                        other_histogram,
                        original_series.reductions,
                    ),
                ),
                payload.series_labels,
                payload.bin_projection,
            )
        _log_raster, log_payload = renderer.render_interactive_histogram(
            evaluated,
            HistogramDisplayState(
                revision=1,
                count_scale=HistogramCountScale.LOG,
            ),
            current_count_limits=payload.viewport.count_limits,
            previous_relim_mode=RelimMode.TIGHT,
            previous_count_scale=HistogramCountScale.LINEAR,
        )
        assert log_payload.viewport.count_scale is HistogramCountScale.LOG
        assert log_payload.viewport.count_limits[0] > 0.0
        assert renderer._axis.get_yscale() == "log"
    finally:
        renderer.close()
