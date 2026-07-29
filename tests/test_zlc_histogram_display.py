"""Focused headless contracts for the interactive histogram core."""

from __future__ import annotations

import numpy as np
import pytest

from zlc_data.axis import (
    COMPONENT,
    REPEAT,
    SITE,
    AxisId,
    AxisSourceRef,
    AxisSpec,
)
from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
from zlc_data.validity import DatasetComponentValidity, ValidityContract
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_frontend.data_figure import DataFigure
from zlc_frontend.data_figure_render import render_data_figure_front
from zlc_frontend.display_range import RelimMode
from zlc_frontend.figure.evaluate import ResolvedDataset, ResolvedDatasetMap
from zlc_frontend.figure.model import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    EvaluatedHistogram,
    FigureDocument,
    FigureLayer,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import (
    HistogramBinProjection,
    HistogramDisplayState,
    HistogramViewportTransform,
    histogram_count_limits,
    histogram_display_form_values,
    histogram_display_from_form,
    histogram_display_with_thresholds,
    histogram_display_with_x_view,
    histogram_home_x_limits,
    histogram_projection_home_x_limits,
)
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import FigureIntent, PlotPanelContract
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

    with pytest.raises(ValueError, match="at least 5"):
        HistogramDisplayState(bin_count=4)
    with pytest.raises(TypeError, match="integer"):
        HistogramDisplayState(bin_count=True)
    with pytest.raises(ValueError, match="positive for log"):
        HistogramDisplayState(
            relim_mode=RelimMode.FIXED,
            log_count_axis=True,
            fixed_count_limits=(0.0, 10.0),
        )

    log_values = histogram_display_form_values(base)
    log_values["relim_mode"] = RelimMode.FIXED
    log_values["log_count_axis"] = True
    with pytest.raises(ValueError, match="positive for log"):
        histogram_display_from_form(
            base,
            log_values,
            current_count_limits=(0.0, 10.0),
        )
    leave_fixed = histogram_display_form_values(fixed)
    leave_fixed["relim_mode"] = RelimMode.TIGHT
    leave_fixed["log_count_axis"] = True
    log_auto = histogram_display_from_form(fixed, leave_fixed)
    assert log_auto.relim_mode is RelimMode.TIGHT
    assert log_auto.log_count_axis is True

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
        False,
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
        True,
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
            True,
            RelimMode.TIGHT,
            True,
            5,
        )
    with pytest.raises(ValueError, match="positive for log"):
        log.data_to_widget_normalized(0.0, 0.0)


def test_histogram_count_relim_is_count_specific_and_hysteretic() -> None:
    tight = HistogramDisplayState(relim_mode=RelimMode.TIGHT)
    normal = HistogramDisplayState(relim_mode=RelimMode.NORMAL)
    log = HistogramDisplayState(log_count_axis=True)
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
        previous_log_count_axis=False,
    ) == (0.0, 20.0)
    assert histogram_count_limits(
        tight,
        10,
        current_count_limits=(0.0, 20.0),
        previous_relim_mode=RelimMode.TIGHT,
        previous_log_count_axis=False,
    ) == (0.0, 11.0)
    assert histogram_count_limits(
        tight,
        10,
        current_count_limits=(0.0, 20.0),
        previous_relim_mode=RelimMode.NORMAL,
        previous_log_count_axis=False,
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


def _histogram_figure() -> DataFigure:
    """One minimal evaluated HISTOGRAM panel shared by the display tests."""

    repeat = AxisSpec(AxisId("repeat"), "Repeat", REPEAT, 2)
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
    validity = DatasetComponentValidity(
        (site.axis_id, channel.axis_id),
        valid_mask,
    )
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
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
            SourceViewBinding(
                AxisSourceRef.tensor(repeat.axis_id),
                AxisViewRole.SAMPLE,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(site.axis_id),
                AxisViewRole.SAMPLE,
            ),
            SourceViewBinding(
                AxisSourceRef.tensor(channel.axis_id),
                AxisViewRole.SAMPLE,
            ),
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
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def test_histogram_evaluator_preserves_unit_coordinates_and_component_validity() -> None:
    figure = _histogram_figure()
    evaluated = figure.evaluated
    histogram = evaluated.layers[0].cells[0].series[0].data
    assert isinstance(histogram, EvaluatedHistogram)
    assert histogram.value_unit == "photoelectron"
    np.testing.assert_array_equal(
        histogram.samples,
        (1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0),
    )
    assert set(histogram.sample_sources) == {
        AxisSourceRef.tensor(AxisId("repeat")),
        AxisSourceRef.tensor(AxisId("site")),
        AxisSourceRef.tensor(AxisId("channel")),
    }
    assert histogram.dropped_count == 1

    view = figure.document.layers[0].view
    contract = PlotPanelContract(
        "histogram-panel",
        FigureIntent(PlotKind.HISTOGRAM, "Histogram", "Counts", view=view),
    )
    rendered = render_data_figure_front(
        figure,
        HistogramDisplayState(),
        contract=contract,
        sequence=1,
    )
    assert rendered.frame is not None
    payload = rendered.frame.panels[0].display_payload
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

    log_rendered = render_data_figure_front(
        figure,
        HistogramDisplayState(
            revision=1,
            log_count_axis=True,
        ),
        contract=contract,
        sequence=2,
    )
    assert log_rendered.frame is not None
    log_payload = log_rendered.frame.panels[0].display_payload
    assert isinstance(log_payload, HistogramPanelPayload)
    assert log_payload.viewport.log_count_axis is True
    assert log_payload.viewport.count_limits[0] > 0.0


def test_histogram_display_thresholds_state_semantics() -> None:
    """thresholds ride the display state: the drag helper advances the
    revision only on real change and the display form carries them along."""

    base = HistogramDisplayState()
    dragged = histogram_display_with_thresholds(base, (1.5,))
    assert dragged.thresholds == (1.5,)
    assert dragged.revision == base.revision + 1
    assert histogram_display_with_thresholds(dragged, (1.5,)) is dragged
    with pytest.raises(ValueError):
        HistogramDisplayState(thresholds=(float("nan"),))
