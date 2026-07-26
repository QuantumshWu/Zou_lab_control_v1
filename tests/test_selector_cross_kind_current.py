"""Cross-kind Figure selector publication on exact immutable data fronts."""

from __future__ import annotations

import numpy as np
import pytest


def _source(*, revision: int):
    from zlc_data import (
        REPEAT,
        SCAN_POINT,
        SITE,
        AxisId,
        AxisSpec,
        BlockId,
        DataBlock,
        DatasetComponentValidity,
        DatasetRevision,
        DatasetSchema,
        OwnedSnapshot,
        PointLayout,
        StreamGenerationId,
        ValidityContract,
        ValueSchema,
    )

    repeat = AxisSpec(AxisId("selector.repeat"), "repeat", REPEAT, 4, range(4))
    scan = AxisSpec(
        AxisId("selector.detuning"),
        "detuning",
        SCAN_POINT,
        5,
        (-2.0, -1.0, 0.0, 1.0, 2.0),
        "MHz",
    )
    site = AxisSpec(
        AxisId("selector.site"),
        "site",
        SITE,
        3,
        ("left", "middle", "right"),
    )
    repeat_offset = np.arange(4, dtype=np.float64)[:, None, None] * 10.0
    scan_offset = np.arange(5, dtype=np.float64)[None, :, None]
    site_offset = np.arange(3, dtype=np.float64)[None, None, :] * 100.0
    values = repeat_offset + scan_offset + site_offset + float(revision)
    valid = np.ones(values.shape, dtype=np.bool_)
    valid[2, 2, 1] = False
    schema = DatasetSchema(
        repeat,
        (scan,),
        PointLayout.rect_c((scan.size,)),
        ValueSchema(
            (site,),
            ValidityContract.components(site.axis_id),
            values.dtype,
            "count",
        ),
    )
    block = DataBlock(
        BlockId("selector-cross-kind"),
        DatasetRevision(revision),
        values,
        DatasetComponentValidity((site.axis_id,), valid),
        schema,
    )
    return (
        OwnedSnapshot(
            block.ref(StreamGenerationId("selector-cross-kind-generation")),
            block,
        ),
        scan,
        site,
    )


def _composed_front(intent, *, revision: int, faceted: bool):
    from zlc_frontend import (
        CurveDisplayState,
        FacetedHistogramDisplayState,
        HistogramDisplayState,
    )
    from zlc_frontend.figure import ViewPreferences, suggest_view
    from zlc_frontend.panel_render import (
        FacetedPanelFocus,
        PanelComposer,
        PanelProvenance,
    )

    snapshot, scan, site = _source(revision=revision)
    preferences = ViewPreferences(
        facet_axis_ids=((site.axis_id,) if faceted else ()),
        batch_axis_ids=(() if faceted else (site.axis_id,)),
    )
    suggestion = suggest_view(snapshot.block.schema, intent, preferences=preferences)
    assert suggestion.spec is not None
    composer = PanelComposer(
        "selector-cross-kind",
        intent=intent,
        view=suggestion.spec,
    )
    provenance = PanelProvenance("run", "epoch", "a" * 64)
    display = (
        CurveDisplayState()
        if intent.value == "CURVE"
        else HistogramDisplayState()
    )
    if not faceted:
        frame, figure = composer.compose_with_figure(
            snapshot,
            display=display,
            provenance=provenance,
        )
        return composer, snapshot, scan, site, frame, figure

    faceted_display = (
        display
        if intent.value == "CURVE"
        else FacetedHistogramDisplayState(display)
    )
    overview = composer.compose_faceted(
        snapshot,
        display=faceted_display,
        provenance=provenance,
    )
    assert overview.overview is not None
    region = overview.overview.regions[1]
    assert region.focus_selection is not None
    focus = FacetedPanelFocus(1, region.focus_selection)
    focused = composer.compose_faceted(
        snapshot,
        display=faceted_display,
        provenance=provenance,
        focus=focus,
    )
    assert focused.frame is not None
    focused_figure = focused.figure.focused_typed_panel(
        focus.panel_index,
        expected_selection=focus.selection,
        expected_intent=intent,
    )
    return composer, snapshot, scan, site, focused.frame, focused_figure


@pytest.mark.parametrize("intent_name", ("CURVE", "HISTOGRAM"))
@pytest.mark.parametrize("faceted", (False, True), ids=("ordinary", "focused-grid"))
def test_area_and_cross_publish_the_exact_visible_data_context(
    intent_name: str,
    faceted: bool,
) -> None:
    from zlc_data import (
        CoordinateRangeSelection,
        DataTransformSpec,
        IndexRangeSelection,
        Selection,
    )
    from zlc_frontend import FigureSource
    from zlc_frontend.figure import ViewIntent
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        CROSS_DATA_OUTPUT,
        HistogramValueRangeSelection,
        bind_area_data_commit,
        bind_cross_data_commit,
        materialize_area_outputs,
        materialize_cross_outputs,
    )
    from zlc_frontend.render import CurvePanelPayload, HistogramPanelPayload

    intent = ViewIntent[intent_name]
    composer, snapshot, scan, site, frame, figure = _composed_front(
        intent,
        revision=7,
        faceted=faceted,
    )
    try:
        panel = frame.panels[0]
        payload = panel.display_payload
        if isinstance(payload, CurvePanelPayload):
            x_axis = payload.series[0].data.x_axis
            x_position = 2
            series = payload.series[0]
            cross_point = (
                float(x_axis.coordinates[x_position]),
                float(series.data.values[x_position]),
            )
            area_selection = Selection(
                (
                    CoordinateRangeSelection(
                        scan.axis_id,
                        -1.0,
                        1.0,
                        scan.coordinate_frame,
                    ),
                )
            )
            expected_cross = float(series.data.values[x_position])
        else:
            assert isinstance(payload, HistogramPanelPayload)
            counts = np.asarray(payload.bin_counts[0])
            bin_index = int(np.argmax(counts))
            edges = np.asarray(payload.bin_edges)
            cross_point = (
                float((edges[bin_index] + edges[bin_index + 1]) / 2.0),
                float(counts[bin_index]),
            )
            area_selection = HistogramValueRangeSelection(100.0, 140.0)
            expected_cross = int(counts[bin_index])

        area_commit = bind_area_data_commit(
            panel.source_identity,
            area_selection,
            figure,
        )
        cross_commit = bind_cross_data_commit(
            panel.source_identity,
            cross_point,
            figure,
            payload,
        )
        source = FigureSource(snapshot, source_contract_id="tests.selector-source")
        area = materialize_area_outputs(source, area_commit.selection)[AREA_DATA_OUTPUT]
        cross = materialize_cross_outputs(source, cross_commit)[CROSS_DATA_OUTPUT]
    finally:
        composer.close()

    assert area.source_ref == snapshot.ref
    assert cross.source_ref == snapshot.ref
    assert area.snapshot.ref.revision == snapshot.ref.revision
    assert cross.snapshot.ref.revision == snapshot.ref.revision
    assert cross.snapshot.block.values.shape == (1, 1, 1)
    assert cross.snapshot.block.values[0, 0, 0] == pytest.approx(expected_cross)
    assert cross.presentation.contract_id == "zlc_frontend.figure.cross-data"
    assert area.snapshot.block.schema.repeat_axis.size == 4
    assert area.snapshot.block.schema.point_layout.storage_size == (
        3 if intent is ViewIntent.CURVE else 5
    )
    if intent is ViewIntent.HISTOGRAM:
        assert area.source_transform is None
    else:
        assert area.source_transform == DataTransformSpec(
            (area_commit.selection,)
        )

    if faceted:
        if isinstance(area_commit.selection, Selection):
            site_terms = tuple(
                term
                for term in area_commit.selection.terms
                if term.axis_id == site.axis_id
            )
        else:
            context = area_commit.selection.source_selection
            assert context is not None
            site_terms = tuple(
                term for term in context.terms if term.axis_id == site.axis_id
            )
        assert site_terms == (IndexRangeSelection(site.axis_id, 1, 2),)
        assert area.snapshot.block.schema.cell_schema.data_shape == (1,)
        assert area.snapshot.block.schema.cell_schema.data_axes[0].coordinates == (
            "middle",
        )
    else:
        assert area.snapshot.block.schema.cell_schema.data_shape == (3,)


@pytest.mark.parametrize("intent_name", ("CURVE", "HISTOGRAM"))
def test_focused_selector_outputs_advance_only_with_their_exact_source_revision(
    intent_name: str,
) -> None:
    from zlc_data import CoordinateRangeSelection, Selection
    from zlc_frontend import FigureSource
    from zlc_frontend.figure import ViewIntent
    from zlc_frontend.figure_outputs import (
        AREA_DATA_OUTPUT,
        CROSS_DATA_OUTPUT,
        FigureOutputRequest,
        FigureOutputSession,
        HistogramValueRangeSelection,
        bind_area_data_commit,
        bind_cross_data_commit,
    )
    from zlc_frontend.render import CurvePanelPayload, HistogramPanelPayload

    intent = ViewIntent[intent_name]
    composer, snapshot_7, scan, _site, frame, figure = _composed_front(
        intent,
        revision=7,
        faceted=True,
    )
    try:
        panel = frame.panels[0]
        payload = panel.display_payload
        if isinstance(payload, CurvePanelPayload):
            x_axis = payload.series[0].data.x_axis
            position = 1
            point = (
                float(x_axis.coordinates[position]),
                float(payload.series[0].data.values[position]),
            )
            selection = Selection(
                (
                    CoordinateRangeSelection(
                        scan.axis_id,
                        -1.0,
                        1.0,
                        scan.coordinate_frame,
                    ),
                )
            )
        else:
            assert isinstance(payload, HistogramPanelPayload)
            edges = np.asarray(payload.bin_edges)
            position = int(np.argmax(np.asarray(payload.bin_counts[0])))
            point = (
                float((edges[position] + edges[position + 1]) / 2.0),
                float(payload.bin_counts[0][position]),
            )
            selection = HistogramValueRangeSelection(100.0, 140.0)
        area = bind_area_data_commit(panel.source_identity, selection, figure)
        cross = bind_cross_data_commit(
            panel.source_identity,
            point,
            figure,
            payload,
        )
    finally:
        composer.close()

    session = FigureOutputSession()
    try:
        first = session.evaluate(
            FigureOutputRequest(
                FigureSource(
                    snapshot_7,
                    source_contract_id="tests.selector-source",
                ),
                area=area,
                cross=cross,
            )
        )
        snapshot_8, _scan, _site = _source(revision=8)
        second = session.evaluate(
            FigureOutputRequest(
                FigureSource(
                    snapshot_8,
                    source_contract_id="tests.selector-source",
                ),
                area=area,
                cross=cross,
            )
        )
    finally:
        session.close()

    for name in (AREA_DATA_OUTPUT, CROSS_DATA_OUTPUT):
        assert first.outputs[name].source_ref == snapshot_7.ref
        assert first.outputs[name].snapshot.ref.revision.value == 7
        assert second.outputs[name].source_ref == snapshot_8.ref
        assert second.outputs[name].snapshot.ref.revision.value == 8
        assert second.outputs[name].snapshot.ref.revision != first.outputs[name].snapshot.ref.revision
