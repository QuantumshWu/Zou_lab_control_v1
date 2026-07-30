"""Focused contracts for Distribution's automatic display-only analysis."""

from __future__ import annotations

import numpy as np

import zlc_frontend._mpl_histogram as histogram_module
from zlc_data.axis import (
    REPEAT,
    SCALAR_AXIS,
    SITE,
    AxisId,
    AxisSourceRef,
    AxisSpec,
)
from zlc_data.fit_contract import FitBatchStatus
from zlc_data.schema import DatasetSchema, PointTable, ValueSchema
from zlc_data.validity import DatasetComponentValidity, VALID, ValidityContract
from zlc_data.value import (
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    OwnedSnapshot,
    StreamGenerationId,
)
from zlc_frontend.data_figure import DataFigure
from zlc_frontend.data_figure_render import (
    render_data_figure_front,
    render_data_figure_grid_overview,
)
from zlc_frontend._mpl_histogram import _update_histogram_presentation
from zlc_frontend.figure.evaluate import ResolvedDataset, ResolvedDatasetMap
from zlc_frontend.figure.model import (
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.histogram_display import HistogramDisplayState
from zlc_frontend.plot_kind import PlotKind
from zlc_frontend.plot_panel import FigureIntent, PlotPanelContract
from zlc_frontend.render import HistogramFitOverlay, HistogramPanelPayload
from zlc_frontend.render_style import PALETTE, render_style_context


def _axes():
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4.2, 2.8), dpi=100)
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot(1, 1, 1)


def _separated_counts():
    edges = np.linspace(-6.0, 7.0, 81, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    left = 1100.0 * np.exp(-((centers + 2.0) ** 2) / (2.0 * 0.55**2))
    right = 720.0 * np.exp(-((centers - 2.4) ** 2) / (2.0 * 0.72**2))
    return edges, left + right


def _present(axis, state, counts_group, edges, **kwargs):
    with render_style_context():
        return _update_histogram_presentation(
            axis,
            state,
            counts_group,
            edges,
            **kwargs,
        )


def test_distribution_automatically_draws_bimodal_components_and_threshold() -> None:
    figure, axis = _axes()
    edges, counts = _separated_counts()

    fit_artists, threshold_artists, _stats, thresholds = (
        _present(
            axis,
            HistogramDisplayState(bin_count=len(counts)),
            (counts,),
            edges,
            show_stats=True,
            threshold_linewidth=1.9,
        )
    )

    assert len(fit_artists) == 3
    assert tuple(artist.get_color() for artist in fit_artists) == (
        PALETTE["fit_left"],
        PALETTE["fit_right"],
        PALETTE["fit_total"],
    )
    assert all(
        np.asarray(artist.get_xdata()).size == len(counts)
        for artist in fit_artists
    )
    assert len(threshold_artists) == len(thresholds) == 1
    assert -2.0 < thresholds[0] < 2.4
    np.testing.assert_allclose(
        threshold_artists[0].get_xdata(),
        (thresholds[0], thresholds[0]),
    )
    figure.clear()


def test_distribution_failure_leaves_no_invented_fit_or_threshold() -> None:
    figure, axis = _axes()
    edges = np.linspace(0.0, 1.0, 61, dtype=np.float64)
    counts = np.zeros(60, dtype=np.float64)
    counts[20] = 3.0

    fit_artists, threshold_artists, stats, thresholds = (
        _present(
            axis,
            HistogramDisplayState(),
            (counts,),
            edges,
            show_stats=True,
            threshold_linewidth=1.9,
        )
    )

    assert all(np.asarray(artist.get_xdata()).size == 0 for artist in fit_artists)
    assert threshold_artists == ()
    assert thresholds == ()
    assert stats is not None and stats.get_text() == ""
    figure.clear()


def test_multi_series_distribution_is_draw_only_and_never_autofits(monkeypatch) -> None:
    figure, axis = _axes()
    edges, counts = _separated_counts()

    def reject_arbitrary_series(*_args, **_kwargs):
        raise AssertionError("multi-series histogram selected one series to fit")

    monkeypatch.setattr(
        histogram_module,
        "analyze_bimodal_distribution",
        reject_arbitrary_series,
    )
    assert (
        histogram_module._histogram_analysis_cache(
            None,
            (counts, counts * 0.5),
            edges,
        )
        is None
    )

    fit_artists, threshold_artists, stats, thresholds = _present(
        axis,
        HistogramDisplayState(bin_count=len(counts)),
        (counts, counts * 0.5),
        edges,
        show_stats=True,
        threshold_linewidth=1.9,
    )

    assert len(fit_artists) == 6
    assert all(np.asarray(artist.get_xdata()).size == 0 for artist in fit_artists)
    assert threshold_artists == ()
    assert thresholds == ()
    assert stats is not None and stats.get_text() == ""
    figure.clear()


def test_formal_histogram_fit_overrides_the_automatic_display_analysis() -> None:
    figure, axis = _axes()
    edges, counts = _separated_counts()
    coordinates = np.linspace(-5.0, 6.0, 17, dtype=np.float64)
    components = (
        np.full(coordinates.shape, 3.0),
        np.full(coordinates.shape, 5.0),
        np.full(coordinates.shape, 8.0),
    )
    overlay = HistogramFitOverlay(
        DatasetRevisionRef(
            BlockId("formal-distribution"),
            StreamGenerationId("formal-distribution-generation"),
            "0" * 64,
            DatasetRevision(1),
        ),
        "formal-fit",
        (),
        0,
        FitBatchStatus.CONVERGED,
        "",
        coordinates,
        components,
    )

    fit_artists, threshold_artists, _stats, thresholds = (
        _present(
            axis,
            HistogramDisplayState(bin_count=len(counts)),
            (counts,),
            edges,
            fit_overlays=(overlay,),
            show_stats=True,
            threshold_linewidth=1.9,
        )
    )

    for artist, expected in zip(fit_artists, components, strict=True):
        np.testing.assert_array_equal(artist.get_xdata(), coordinates)
        np.testing.assert_array_equal(artist.get_ydata(), expected)
    assert threshold_artists == ()
    assert thresholds == ()
    figure.clear()


def _distribution_panel() -> DataFigure:
    rng = np.random.default_rng(880)
    repeat = AxisSpec(AxisId("distribution-panel.repeat"), "Repeat", REPEAT, 420)
    values = np.empty((repeat.size, 1, 1), dtype=np.float64)
    values[:210, 0, 0] = rng.normal(-2.0, 0.45, 210)
    values[210:, 0, 0] = rng.normal(2.1, 0.60, 210)
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema.scalar(values.dtype, "count"),
    )
    block = DataBlock(
        BlockId("distribution-panel"),
        DatasetRevision(1),
        values,
        VALID,
        schema,
    )
    dataset_id = DatasetId("distribution-panel")
    document = FigureDocument(
        "distribution-panel-document",
        1,
        (DatasetDescriptor(dataset_id, "Distribution", schema.fingerprint),),
        (
            FigureLayer(
                "distribution-panel-layer",
                dataset_id,
                ViewSpec(
                    schema.fingerprint,
                    ViewIntent.HISTOGRAM,
                    (
                        SourceViewBinding(
                            AxisSourceRef.tensor(repeat.axis_id),
                            AxisViewRole.SAMPLE,
                        ),
                        SourceViewBinding(
                            AxisSourceRef.tensor(SCALAR_AXIS.axis_id),
                            AxisViewRole.SELECTED,
                            selector=FixedIndex(0),
                        ),
                    ),
                ),
            ),
        ),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("distribution-panel-generation")),
        block,
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def test_automatic_cut_is_drawn_only_and_publishes_no_formal_fit() -> None:
    data_figure = _distribution_panel()
    display = HistogramDisplayState()
    view = data_figure.document.layers[0].view
    contract = PlotPanelContract(
        "distribution-panel",
        FigureIntent(
            PlotKind.HISTOGRAM,
            "Distribution",
            "Counts",
            view=view,
        ),
    )

    rendered = render_data_figure_front(
        data_figure,
        display,
        contract=contract,
        sequence=1,
    )
    assert rendered.frame is not None
    payload = rendered.frame.panels[0].display_payload
    assert isinstance(payload, HistogramPanelPayload)
    assert payload.fit_overlays == ()
    assert len(payload.thresholds) == 1
    assert display.thresholds == ()


def _distribution_grid() -> DataFigure:
    rng = np.random.default_rng(771)
    repeat = AxisSpec(AxisId("distribution.repeat"), "Repeat", REPEAT, 420)
    site = AxisSpec(AxisId("distribution.site"), "Site", SITE, 2, ("A", "B"))
    values = np.empty((repeat.size, 1, site.size), dtype=np.float64)
    for index, shift in enumerate((0.0, 0.7)):
        values[:210, 0, index] = rng.normal(-2.0 + shift, 0.45, 210)
        values[210:, 0, index] = rng.normal(2.1 + shift, 0.60, 210)
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("distribution-grid"),
        DatasetRevision(1),
        values,
        DatasetComponentValidity(
            (site.axis_id,),
            np.ones(values.shape, dtype=np.bool_),
        ),
        schema,
    )
    dataset_id = DatasetId("distribution-grid")
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
                AxisViewRole.FACET,
            ),
        ),
    )
    document = FigureDocument(
        "distribution-grid-document",
        1,
        (DatasetDescriptor(dataset_id, "Distribution", schema.fingerprint),),
        (FigureLayer("distribution-grid-layer", dataset_id, view),),
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("distribution-grid-generation")),
        block,
    )
    return DataFigure(
        document,
        ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
    )


def test_distribution_grid_reuses_the_same_automatic_analysis_path() -> None:
    figure = _distribution_grid()
    grid_contract = PlotPanelContract(
        "distribution-grid",
        FigureIntent(
            PlotKind.GRID,
            "Distribution",
            "Counts",
            view=figure.document.layers[0].view,
        ),
        size_name="2x2",
    )
    overview = render_data_figure_grid_overview(
        figure,
        contract=grid_contract,
        display_state=HistogramDisplayState(),
    )
    assert len(overview.regions) == 2
    for index, region in enumerate(overview.regions):
        focused = overview.figure.focused_typed_panel(
            index,
            expected_address=region.focus_address,
            expected_intent=ViewIntent.HISTOGRAM,
        )
        view = focused.document.layers[0].view
        rendered = render_data_figure_front(
            focused,
            HistogramDisplayState(),
            contract=PlotPanelContract(
                "distribution-grid",
                FigureIntent(
                    PlotKind.HISTOGRAM,
                    "Distribution",
                    "Counts",
                    view=view,
                ),
                size_name="2x2",
            ),
            sequence=index + 1,
            histogram_projection_value_range=overview.histogram_home_x_limits,
        )
        assert rendered.frame is not None
        payload = rendered.frame.panels[0].display_payload
        assert isinstance(payload, HistogramPanelPayload)
        assert payload.fit_overlays == ()
        assert len(payload.thresholds) == 1
