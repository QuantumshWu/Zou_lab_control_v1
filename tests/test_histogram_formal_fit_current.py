"""Current contracts for zlc_plot-owned Histogram and Distribution Fit."""

from __future__ import annotations

import numpy as np

from zlc_data import (
    REPEAT,
    SITE,
    VALID,
    AxisId,
    AxisSpec,
    BlockId,
    DataBlock,
    DatasetRevision,
    DatasetRevisionRef,
    DatasetSchema,
    OwnedSnapshot,
    PointTable,
    StreamGenerationId,
    ValidityContract,
    ValueSchema,
)
from zlc_plot import (
    AxisRef,
    FacetFitBatchResult,
    FacetGridPlot,
    HistogramPlot,
    PlotKind,
    PlotSession,
    default_plot_spec,
)


def _histogram_snapshot(
    values: np.ndarray,
    *,
    revision: int = 7,
) -> OwnedSnapshot:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("test histogram values must have shape (repeat, site)")
    repeat = AxisSpec(
        AxisId("histogram.repeat"),
        "repeat",
        REPEAT,
        array.shape[0],
    )
    site = AxisSpec(
        AxisId("histogram.site"),
        "readout site",
        SITE,
        array.shape[1],
        tuple(range(array.shape[1])),
    )
    schema = DatasetSchema(
        repeat,
        PointTable(1),
        None,
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    block = DataBlock(
        BlockId(f"histogram-{revision}"),
        DatasetRevision(revision),
        array[:, None, :],
        VALID,
        schema,
    )
    return OwnedSnapshot(
        block.ref(StreamGenerationId("histogram-generation")),
        block,
    )


def _bimodal_sites(site_count: int = 3) -> OwnedSnapshot:
    rng = np.random.default_rng(20260802)
    repeat_count = 640
    values = np.empty((repeat_count, site_count), dtype=np.float64)
    for site in range(site_count):
        values[:, site] = np.concatenate((
            rng.normal(-2.0 - 0.1 * site, 0.34, repeat_count // 2),
            rng.normal(2.1 + 0.15 * site, 0.48, repeat_count // 2),
        ))
    return _histogram_snapshot(values)


def test_histogram_default_pools_only_repeat_and_indexes_real_data_axes() -> None:
    snapshot = _bimodal_sites(2)

    spec = default_plot_spec(snapshot.block.schema, PlotKind.HISTOGRAM)

    assert spec == HistogramPlot(samples=(AxisRef.repeat(),))
    session = PlotSession(snapshot, spec, parameters={"bin_count": 48})
    try:
        assert "index__data__histogram.site" in session.parameter_schema
        repeat_count = snapshot.block.schema.repeat_axis.size
        assert int(np.sum(session._payload.counts)) == repeat_count
        session.set_parameter("index__data__histogram.site", 1)
        assert int(np.sum(session._payload.counts)) == repeat_count
    finally:
        session.close()


def test_distribution_fit_draws_components_and_suggested_threshold() -> None:
    snapshot = _bimodal_sites(1)
    session = PlotSession(
        snapshot,
        HistogramPlot(samples=(AxisRef.repeat(),)),
        parameters={"bin_count": 56},
    )
    events = []
    session.subscribe_fit(events.append)
    try:
        result = session.fit("bimodal_gaussian")

        assert result.success
        assert result.fitted_values is not None
        assert result.residuals is not None
        assert result.selected_indices is not None
        assert result.observation_count == result.fitted_values.size
        assert len(events) == 1
        assert events[0].source_revisions == (snapshot.ref.revision.value,)
        assert session.last_fit is result
        assert session.rgba().shape == (*session.surface_plan.raster_size[::-1], 4)
        visible_lines = [
            artist
            for artist in session._renderer._fit_artists
            if hasattr(artist, "get_xdata") and artist.get_visible()
        ]
        # Two components, their total, and the display-only suggested threshold.
        assert len(visible_lines) == 4
    finally:
        session.close()


def test_all_facet_fit_uses_one_engine_and_publishes_physical_parameters() -> None:
    snapshot = _bimodal_sites(3)
    spec = FacetGridPlot(
        AxisRef.data("histogram.site"),
        HistogramPlot(samples=(AxisRef.repeat(),)),
    )
    session = PlotSession(
        snapshot,
        spec,
        size="4x4",
        parameters={"bin_count": 48},
    )
    events = []
    session.subscribe_fit(events.append)
    session.set_facet_thresholds((0.0, 0.1, 0.2), display=False)
    try:
        result = session.fit("bimodal_gaussian", fit_all_facets=True)

        assert isinstance(result, FacetFitBatchResult)
        assert bool(np.all(result.success))
        assert session.last_facet_fit is result
        assert len(events) == 1
        assert [axis.get_title() for axis in session._renderer._axes["facet_cell"]] == [
            "0",
            "1",
            "2",
        ]
        assert not [
            artist
            for artist in session._renderer._fit_artists
            if hasattr(artist, "get_text") and artist.get_visible()
        ]
        assert not any(
            label.get_visible()
            for _line, label in session._renderer._facet_threshold_artists.values()
        )

        session.focus_facet(1)
        visible_axes = [
            axis for axis in session._renderer._axes["facet_cell"] if axis.get_visible()
        ]
        assert len(visible_axes) == 1
        assert visible_axes[0].get_title() == "readout site=1"
        assert any(
            artist.get_visible()
            for artist in session._renderer._fit_artists
            if hasattr(artist, "get_text")
        )
        assert [
            index
            for index, (_line, label) in session._renderer._facet_threshold_artists.items()
            if label.get_visible()
        ] == [1]

        def reference_for(name: str, schema: DatasetSchema) -> DatasetRevisionRef:
            return DatasetRevisionRef(
                BlockId(f"fit-{name}"),
                StreamGenerationId(f"fit-{name}-generation"),
                schema.fingerprint,
                snapshot.ref.revision,
            )

        parameters = events[0].materialize_parameters(reference_for)
        assert set(parameters) == set(result.model.parameter_names)
        for parameter in parameters.values():
            assert parameter.block.values.shape == (1, 1, 3)
            assert parameter.block.schema.cell_schema.data_axes[0].axis_id == AxisId(
                "histogram.site"
            )
            assert parameter.ref.revision == snapshot.ref.revision
    finally:
        session.close()
