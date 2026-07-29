"""Focused contracts for the single formal Histogram Fit path."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from zlc_data import (  # noqa: E402
    REPEAT,
    SCAN_POINT,
    COMPONENT,
    SITE,
    AxisId,
    AxisSourceRef,
    AxisSpec,
    BlockId,
    DataBlock,
    DataTransformSpec,
    DatasetRevision,
    DatasetComponentValidity,
    DatasetSchema,
    HistogramSpec,
    IndexSelection,
    OwnedSnapshot,
    PointColumn,
    PointTable,
    SCALAR_AXIS,
    Selection,
    StreamGenerationId,
    VALID,
    ValidityContract,
    ValueSchema,
)
from zlc_data.fit_problem import bind_fit  # noqa: E402
from zlc_data.transform import apply_transform, commit_transform  # noqa: E402
from zlc_data.transform_codec import (  # noqa: E402
    data_transform_spec_from_tree,
    data_transform_spec_to_tree,
)
from zlc_frontend.data_figure import DataFigure  # noqa: E402
from zlc_frontend.figure import (  # noqa: E402
    AxisViewRole,
    DatasetDescriptor,
    DatasetId,
    FigureDocument,
    FigureLayer,
    FixedIndex,
    ResolvedDataset,
    ResolvedDatasetMap,
    SourceViewBinding,
    ViewIntent,
    ViewSpec,
)
from zlc_frontend.fit_editor import prepare_fit_authoring_options  # noqa: E402
from zlc_frontend.render import HistogramPanelPayload  # noqa: E402
from zlc_frontend.data_figure_render import render_data_figure_front  # noqa: E402
from zlc_frontend.data_figure_presentation import DATA_FIGURE_PANEL_ID  # noqa: E402
from zlc_frontend.histogram_display import HistogramDisplayState  # noqa: E402
from zlc_frontend.plot_kind import PlotKind  # noqa: E402
from zlc_frontend.plot_panel import (  # noqa: E402
    FigureIntent,
    PlotPanelContract,
)
from zlc_frontend.qt_widgets import ensure_qt_app  # noqa: E402
from zlc_workbench.data_figure.app import create_data_figure_pane  # noqa: E402
from zlc_workbench.data_figure.worker_jobs import _prepare_fit_options  # noqa: E402


def _figure(selected_point: int = 0):
    rng = np.random.default_rng(90210)
    values = np.concatenate(
        (rng.normal(-2.0, 0.45, 350), rng.normal(2.2, 0.65, 420))
    )
    # Both points have the same extrema and therefore the same bin edges.  The
    # reversed second point still has a different named selection authority.
    physical = np.stack((values, values[::-1]), axis=1)[..., np.newaxis]
    repeat = AxisSpec(AxisId("formal-hist.repeat"), "Repeat", REPEAT, len(values))
    point = AxisSpec(AxisId("formal-hist.point"), "Point", SCAN_POINT, 2)
    schema = DatasetSchema(
        repeat,
        PointTable(
            point.size,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    tuple(range(point.size)),
                ),
            ),
        ),
        None,
        ValueSchema.scalar(np.dtype("<f8"), "count"),
    )
    block = DataBlock(
        BlockId("formal-hist-block"),
        DatasetRevision(1),
        physical,
        VALID,
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("formal-hist-generation")),
        block,
    )
    dataset_id = DatasetId("formal-hist-dataset")
    view = ViewSpec(
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
        point_ordinals=(selected_point,),
    )
    document = FigureDocument(
        f"formal-hist-document-{selected_point}",
        1,
        (DatasetDescriptor(dataset_id, "samples", schema.fingerprint),),
        (FigureLayer("histogram", dataset_id, view),),
    )
    return (
        DataFigure(
            document,
            ResolvedDatasetMap((ResolvedDataset(dataset_id, snapshot),)),
        ),
        snapshot,
    )


def _front(figure, *, sequence=1, display=None):
    intent = FigureIntent(
        PlotKind.HISTOGRAM,
        "Histogram",
        "Counts",
        view=figure.document.layers[0].view,
    )
    return render_data_figure_front(
        figure,
        HistogramDisplayState(bin_count=60) if display is None else display,
        contract=PlotPanelContract(DATA_FIGURE_PANEL_ID, intent, size_name="2x2"),
        sequence=sequence,
    )


def test_histogram_transform_is_one_terminal_operation() -> None:
    histogram = HistogramSpec(
        (AxisSourceRef.tensor(AxisId("sample")),),
        (0.0, 1.0),
    )
    selection = Selection((IndexSelection(AxisId("point"), 0),))
    with pytest.raises(ValueError, match="terminal"):
        DataTransformSpec((histogram, selection))
    with pytest.raises(ValueError, match="at most one"):
        DataTransformSpec((histogram, histogram))
    spec = DataTransformSpec((histogram,))
    tree = data_transform_spec_to_tree(spec)
    assert set(tree["operations"][0]) == {"kind", "sources", "bin_edges"}
    assert data_transform_spec_from_tree(tree) == spec


def test_histogram_transform_groups_multi_cell_and_multi_data_axes_once() -> None:
    """Repeat+component samples preserve point/site batches exactly."""

    repeat = AxisSpec(AxisId("grouped.repeat"), "repeat", REPEAT, 3)
    point = AxisSpec(AxisId("grouped.point"), "point", SCAN_POINT, 2)
    sample = AxisSpec(AxisId("grouped.sample"), "sample", COMPONENT, 2)
    site = AxisSpec(AxisId("grouped.site"), "site", SITE, 2)
    schema = DatasetSchema(
        repeat,
        PointTable(
            point.size,
            (
                PointColumn(
                    point.axis_id,
                    point.name,
                    point.role,
                    PointColumn.NUMERIC,
                    tuple(range(point.size)),
                ),
            ),
        ),
        None,
        ValueSchema(
            (sample, site),
            ValidityContract.components(sample.axis_id, site.axis_id),
            np.dtype("<f8"),
            "count",
        ),
    )
    values = np.empty((repeat.size, point.size, sample.size, site.size))
    for r in range(repeat.size):
        for p in range(point.size):
            for q in range(sample.size):
                for s in range(site.size):
                    values[r, p, q, s] = 10 * p + 2 * s + ((r + q) % 2)
    validity = np.ones(values.shape, dtype=np.bool_)
    values[0, 0, 0, 0] = 999.0
    validity[0, 0, 0, 0] = False
    validity[:, 1, :, 1] = False
    block = DataBlock(
        BlockId("grouped-histogram-block"),
        DatasetRevision(1),
        values,
        DatasetComponentValidity(
            (sample.axis_id, site.axis_id),
            validity,
        ),
        schema,
    )
    snapshot = OwnedSnapshot(
        block.ref(StreamGenerationId("grouped-histogram-generation")),
        block,
    )
    edges = (-0.5, 0.5, 1.5, 2.5, 3.5, 10.5, 11.5, 12.5, 13.5)
    transform = DataTransformSpec(
        (
            HistogramSpec(
                (
                    AxisSourceRef.tensor(repeat.axis_id),
                    AxisSourceRef.tensor(sample.axis_id),
                ),
                edges,
            ),
        )
    )
    result = apply_transform(snapshot, commit_transform(schema, transform))

    assert result.schema.point_table == schema.point_table
    assert result.schema.cell_schema.data_axes[:-1] == (site,)
    assert result.values.shape == (1, point.size, site.size, len(edges) - 1)
    expanded_validity = result.expanded_validity()
    for p in range(point.size):
        for s in range(site.size):
            valid_values = values[:, p, :, s][validity[:, p, :, s]]
            expected = np.histogram(valid_values, bins=edges)[0]
            np.testing.assert_array_equal(result.values[0, p, s], expected)
            assert np.all(expanded_validity[0, p, s]) == bool(valid_values.size)


def test_formal_histogram_fit_uses_full_samples_and_rejects_changed_view() -> None:
    figure, snapshot = _figure(0)
    source = _front(figure)
    payload = source.frame.panels[0].display_payload
    assert isinstance(payload, HistogramPanelPayload)
    options = prepare_fit_authoring_options(
        figure,
        None,
        histogram_projection=payload.bin_projection,
    )
    assert tuple(option.spec.model_id for option in options) == (
        "bimodal_gaussian",
        "histogram_gaussian",
    )
    result = bind_fit(options[0].spec, snapshot.block.schema).run(snapshot)
    fitted_panel = figure.materialize_transient_fit_overlays(
        result,
        source.frame,
        result_identity="formal-hist-fit",
    )
    assert fitted_panel is not None
    fitted_payload = fitted_panel.display_payload
    assert isinstance(fitted_payload, HistogramPanelPayload)
    assert len(fitted_payload.fit_overlays) == 1
    assert len(fitted_payload.fit_overlays[0].component_predictions) == 3
    assert source.frame.panels[0].raster is fitted_panel.raster

    zoomed = _front(
        figure,
        sequence=3,
        display=HistogramDisplayState(bin_count=60, x_view=(-1.0, 1.0)),
    )
    zoomed_payload = zoomed.frame.panels[0].display_payload
    assert isinstance(zoomed_payload, HistogramPanelPayload)
    assert np.array_equal(payload.bin_edges, zoomed_payload.bin_edges)

    changed, _same_snapshot = _figure(1)
    changed_front = _front(changed, sequence=4)
    changed_payload = changed_front.frame.panels[0].display_payload
    assert isinstance(changed_payload, HistogramPanelPayload)
    assert np.array_equal(payload.bin_edges, changed_payload.bin_edges)
    with pytest.raises(ValueError, match="authority differs"):
        changed.materialize_transient_fit_overlays(
            result,
            changed_front.frame,
            result_identity="formal-hist-fit",
        )


def test_standalone_data_figure_uses_the_same_histogram_fit_context() -> None:
    application = ensure_qt_app()
    figure, _snapshot = _figure(0)
    display = HistogramDisplayState(bin_count=60)
    figure_intent = FigureIntent(
        PlotKind.HISTOGRAM,
        "Histogram",
        "Counts",
        view=figure.document.layers[0].view,
    )
    window = create_data_figure_pane(
        figure,
        figure_intent,
        initial_display=display,
        local_fit=True,
        open_fit=True,
    )

    def until(predicate, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            application.processEvents()
            if predicate():
                return
            time.sleep(0.005)
        raise AssertionError("Qt condition did not become true")

    try:
        until(lambda: window.worker_idle and bool(window.fit_models))
        assert window._view_family == "histogram"
        assert window.fit_models[:2] == (
            "bimodal_gaussian",
            "histogram_gaussian",
        )
        before = window._surface_host.grab().toImage()
        pane = window._fit_pane
        assert pane is not None and pane.fit_button.isEnabled()
        pane.fit_button.click()
        until(lambda: window.worker_idle and window.draft_ready)
        application.processEvents()
        after = window._surface_host.grab().toImage()
        assert before != after
    finally:
        window.shutdown()
        until(lambda: window.closed)


def test_artifact_binding_default_admits_the_exact_histogram_transform() -> None:
    """The ordinary artifact binding need not opt into arbitrary transforms."""

    figure, _snapshot = _figure(0)
    front = _front(figure)
    payload = front.frame.panels[0].display_payload
    assert isinstance(payload, HistogramPanelPayload)

    def prepare(exact_figure, selection, projection):
        return prepare_fit_authoring_options(
            exact_figure,
            selection,
            histogram_projection=projection,
        )

    options = _prepare_fit_options(
        prepare,
        figure,
        None,
        payload.bin_projection,
    )
    assert options[0].spec.model_id == "bimodal_gaussian"

    changed, _same_snapshot = _figure(1)
    changed_front = _front(changed, sequence=2)
    changed_payload = changed_front.frame.panels[0].display_payload
    assert isinstance(changed_payload, HistogramPanelPayload)

    def prepare_other_selection(_exact_figure, selection, _projection):
        return prepare_fit_authoring_options(
            changed,
            selection,
            histogram_projection=changed_payload.bin_projection,
        )

    with pytest.raises(ValueError, match="exact visible Figure authority"):
        _prepare_fit_options(
            prepare_other_selection,
            figure,
            None,
            payload.bin_projection,
        )
